import json
import random
from pathlib import Path

import numpy as np

from .hmm import (
    BEAT_COUNTS,
    STATES,
    SIGNATURE_BEAT_COUNTS,
    get_beat_family,
    build_transition_matrix
)


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
ANNOTATION_DIR = PROJECT_DIR / "Data" / "uncommon" / "parsed_beats"
OUTPUT_PATH = Path(__file__).resolve().parent / "uncommon_model.json"


def load_tracks(annotation_dir):
    tracks = []

    for path in sorted(Path(annotation_dir).glob("*.json")):
        with open(path, encoding="utf-8") as file:
            data = json.load(file)

        signatures = []
        beat_counts = []

        for measure in data["measures"]:
            signature = str(measure["time_sig_num"]) + "/" + str(measure["time_sig_den"])
            signatures.append(signature)
            beat_counts.append(SIGNATURE_BEAT_COUNTS[signature])

        if len(beat_counts) > 0:
            tracks.append({
                "track_id": data["youtube_id"],
                "signatures": signatures,
                "beat_counts": beat_counts,
            })

    return tracks


def count_annotations(tracks):
    signatures = list(SIGNATURE_BEAT_COUNTS)
    transition_counts = np.zeros((len(BEAT_COUNTS), len(BEAT_COUNTS)), dtype=int)
    signature_transition_counts = np.zeros((len(signatures), len(signatures)), dtype=int)
    measure_counts = np.zeros(len(BEAT_COUNTS), dtype=int)
    signature_measure_counts = np.zeros(len(signatures), dtype=int)
    first_measure_counts = np.zeros(len(BEAT_COUNTS), dtype=int)

    for track in tracks:
        counts = track["beat_counts"]
        first_measure_counts[BEAT_COUNTS.index(counts[0])] += 1

        for measure_index in range(len(counts)):
            source = BEAT_COUNTS.index(counts[measure_index])
            source_signature = signatures.index(track["signatures"][measure_index])
            measure_counts[source] += 1
            signature_measure_counts[source_signature] += 1

            if measure_index + 1 < len(counts):
                destination = BEAT_COUNTS.index(counts[measure_index + 1])
                destination_signature = signatures.index(track["signatures"][measure_index + 1])
                transition_counts[source, destination] += 1
                signature_transition_counts[source_signature, destination_signature] += 1

    return {
        "transition_counts": transition_counts,
        "signature_transition_counts": signature_transition_counts,
        "measure_counts": measure_counts,
        "signature_measure_counts": signature_measure_counts,
        "first_measure_counts": first_measure_counts,
    }


def estimate_transition_probs(transition_counts, smoothing=1.0):
    families = []
    for beat_count in BEAT_COUNTS:
        family = get_beat_family(beat_count)
        if family not in families:
            families.append(family)

    family_counts = np.zeros((len(families), len(families)), dtype=int)

    for source in range(len(BEAT_COUNTS)):
        source_family = families.index(get_beat_family(BEAT_COUNTS[source]))
        for destination in range(len(BEAT_COUNTS)):
            destination_family = families.index(get_beat_family(BEAT_COUNTS[destination]))
            family_counts[source_family, destination_family] += transition_counts[source, destination]

    family_probs = np.zeros(family_counts.shape)
    for source_family in range(len(families)):
        weights = family_counts[source_family] + smoothing
        family_probs[source_family] = weights / np.sum(weights)

    transition_probs = np.zeros(transition_counts.shape)

    for source in range(len(BEAT_COUNTS)):
        source_family = families.index(get_beat_family(BEAT_COUNTS[source]))
        source_members = []
        for index in range(len(BEAT_COUNTS)):
            if get_beat_family(BEAT_COUNTS[index]) == families[source_family]:
                source_members.append(index)

        for destination_family in range(len(families)):
            destinations = []
            for index in range(len(BEAT_COUNTS)):
                if get_beat_family(BEAT_COUNTS[index]) == families[destination_family]:
                    destinations.append(index)

            prior = np.zeros(len(destinations))

            if source_family == destination_family:
                same_count = 0
                for index in source_members:
                    same_count += transition_counts[index, index]

                total = family_counts[source_family, source_family]
                p_same = (same_count + smoothing) / (total + 2 * smoothing)

                for index in range(len(destinations)):
                    if len(destinations) == 1:
                        prior[index] = 1
                    elif destinations[index] == source:
                        prior[index] = p_same
                    else:
                        prior[index] = (1 - p_same) / (len(destinations) - 1)
            else:
                for index in range(len(destinations)):
                    prior[index] = smoothing
                    for source_member in source_members:
                        prior[index] += transition_counts[source_member, destinations[index]]
                prior = prior / np.sum(prior)

            weights = np.zeros(len(destinations))
            for index in range(len(destinations)):
                weights[index] = transition_counts[source, destinations[index]] + smoothing * prior[index]
            weights = weights / np.sum(weights)

            for index in range(len(destinations)):
                transition_probs[source, destinations[index]] = family_probs[source_family, destination_family] * weights[index]

    return transition_probs, families, family_counts, family_probs


def estimate_initial_probs(first_measure_counts, smoothing=1.0):
    initial_probs = np.full(len(STATES), smoothing, dtype=float)

    for state_index in range(len(STATES)):
        beat_count, position = STATES[state_index]
        if position == 1:
            count_index = BEAT_COUNTS.index(beat_count)
            initial_probs[state_index] += first_measure_counts[count_index]

    return initial_probs / np.sum(initial_probs)


def run_training(annotation_dir=ANNOTATION_DIR, output_path=OUTPUT_PATH, test_fraction=0.2, seed=42, smoothing=1.0):
    if smoothing <= 0:
        print("Smoothing must be positive")
        return None

    tracks = load_tracks(annotation_dir)
    if len(tracks) < 2:
        print("At least two tracks are needed to have a train et test set.")
        return None

    if test_fraction <= 0 or test_fraction >= 1:
        print("The proportion of the test set must be between 0 and 1 excluded")
        return None

    random_generator = random.Random(seed)
    random_generator.shuffle(tracks)
    n_test = int(round(len(tracks) * test_fraction))
    if n_test < 1:
        n_test = 1
    if n_test >= len(tracks):
        n_test = len(tracks) - 1
    test_tracks = tracks[:n_test]
    train_tracks = tracks[n_test:]

    seen_signatures = []
    for track in train_tracks:
        for signature in track["signatures"]:
            if signature not in seen_signatures:
                seen_signatures.append(signature)

    remaining_test_tracks = []
    for track in test_tracks:
        has_new_signature = False
        for signature in track["signatures"]:
            if signature not in seen_signatures:
                has_new_signature = True

        if has_new_signature:
            train_tracks.append(track)
            for signature in track["signatures"]:
                if signature not in seen_signatures:
                    seen_signatures.append(signature)
        else:
            remaining_test_tracks.append(track)
    test_tracks = remaining_test_tracks

    if len(test_tracks) == 0:
        print("No test track is left after putting the rare signature tracks in the training set.")
        return None

    counts = count_annotations(train_tracks)
    transition_probs, families, family_counts, family_probs = estimate_transition_probs(counts["transition_counts"], smoothing)
    transition_matrix = build_transition_matrix(transition_probs)
    initial_probs = estimate_initial_probs(counts["first_measure_counts"], smoothing)

    first_measure_probs = counts["first_measure_counts"] + smoothing
    first_measure_probs = first_measure_probs / np.sum(first_measure_probs)

    model = {
        "dataset": "uncommon",
        "signature_beat_counts": SIGNATURE_BEAT_COUNTS,
        "beat_counts": BEAT_COUNTS,
        "states": STATES,
        "families": families,
        "seed": seed,
        "test_fraction": test_fraction,
        "smoothing": smoothing,
        "train_track_ids": [],
        "test_track_ids": [],
        "signatures": list(SIGNATURE_BEAT_COUNTS),
        "signature_measure_counts": counts["signature_measure_counts"].tolist(),
        "signature_transition_counts": counts["signature_transition_counts"].tolist(),
        "measure_counts": counts["measure_counts"].tolist(),
        "transition_counts": counts["transition_counts"].tolist(),
        "family_transition_counts": family_counts.tolist(),
        "family_transition_probs": family_probs.tolist(),
        "first_measure_counts": counts["first_measure_counts"].tolist(),
        "first_measure_probs": first_measure_probs.tolist(),
        "beat_transition_probs": transition_probs.tolist(),
        "transition_matrix": transition_matrix.tolist(),
        "initial_probs": initial_probs.tolist(),
    }

    for track in train_tracks:
        model["train_track_ids"].append(track["track_id"])
    for track in test_tracks:
        model["test_track_ids"].append(track["track_id"])

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    print("Morceaux d'entrainement :", len(train_tracks))
    print("Morceaux reserves au test :", len(test_tracks))
    print("Mesures d'entrainement :", int(np.sum(counts["measure_counts"])))
    print("Frontieres comptees :", int(np.sum(counts["transition_counts"])))
    print("Etats :", len(STATES))
    print("Modele enregistre :", output_path)
    return model







def train_hmm_from_ids(train_ids, annotation_dir=ANNOTATION_DIR,
                       smoothing=1.0, a=0, p_stay=0.9):

    # Load annotations
    all_tracks = load_tracks(annotation_dir)

    # Select only the tracks belonging to the training set
    train_ids = {str(track_id) for track_id in train_ids}

    train_tracks = [
        track for track in all_tracks
        if str(track["track_id"]) in train_ids
    ]

    found_ids = {str(track["track_id"]) for track in train_tracks}
    missing_ids = train_ids - found_ids

    if missing_ids:
        raise ValueError(
            f"Missing HMM annotations for {len(missing_ids)} tracks: "
            f"{sorted(missing_ids)}"
        )

    if not train_tracks:
        raise ValueError("No training tracks found for HMM.")

    # Count transitions using training annotations only
    counts = count_annotations(train_tracks)

    transition_probs, _, _, _ = estimate_transition_probs(
        counts["transition_counts"],
        smoothing=smoothing
    )

    # Estimate HMM parameters
    A = build_transition_matrix(transition_probs, add_uniform=a, p_stay=p_stay)

    pi = estimate_initial_probs(
        counts["first_measure_counts"],
        smoothing=smoothing
    )

    return A, pi




if __name__ == "__main__":
    run_training()
