import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from get_data_frame import get_full_dataframe
from features import build_feature_dictionnary
from classifier import*

SEED = 42
TOLERANCE = 0.10

INCLUDE_HARMONIX_TRAIN = False
EVALUATE_HARMONIX = False

INCLUDE_SVM_MODEL = True

N_HARMONIX_TRAIN = 20  # Number of Harmonix tracks to put in the training


# Load datasets

df = get_full_dataframe(
    include_harmonix=False,
    include_uncommon=True
)

uncommon_df = df[df["dataset"] == "uncommon_meter"].copy()
harmonix_df = df[df["dataset"] == "harmonix"].copy()


# Train / test split

# Uncommon: 80% train, 20% test
train_df, test_df = train_test_split(
    uncommon_df,
    test_size=0.20,
    random_state=SEED
)

# Harmonix: independent training and test subsets
harmonix_train = harmonix_df.sample(
    n=min(N_HARMONIX_TRAIN, len(harmonix_df)),
    random_state=SEED
)

harmonix_test = harmonix_df.drop(harmonix_train.index)

if INCLUDE_HARMONIX_TRAIN:
    train_df = pd.concat(
        [train_df, harmonix_train],
        ignore_index=True
    )

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
harmonix_test = harmonix_test.reset_index(drop=True)

print(f"Train: {len(train_df)} tracks")
print(f"Test Uncommon: {len(test_df)} tracks")
print(f"Test Harmonix: {len(harmonix_test)} tracks")


#%% Feature extraction

feature_dic_train = build_feature_dictionnary(
    train_df.to_dict("records"),
    sr=22050,
    dic={},
    beat_track="other",
)

feature_dic_test = build_feature_dictionnary(
    test_df.to_dict("records"),
    sr=22050,
    dic={},
    beat_track="other",
)

if EVALUATE_HARMONIX:
    feature_dic_harmonix = build_feature_dictionnary(
        harmonix_test.to_dict("records"),
        sr=22050,
        dic={},
        beat_track="other",
    )


#% Training and testing arrays

X_train, y_train = training_arrays(
    train_df.to_dict("records"),
    feature_dic_train,
    tolerance=TOLERANCE
)

X_test, y_test = training_arrays(
    test_df.to_dict("records"),
    feature_dic_test,
    tolerance=TOLERANCE
)

if EVALUATE_HARMONIX:
    X_harmonix, y_harmonix = training_arrays(
        harmonix_test.to_dict("records"),
        feature_dic_harmonix,
        tolerance=TOLERANCE
    )


#%% Train classifiers

#IT IS POSSIBLE TO ADD SVM TO BUT IT TAKES TOO LONG FOR MY PC
if INCLUDE_SVM_MODEL :
    models = {
    "Random Forest": train_forest(X_train, y_train),
    "HistGradientBoosting": train_HistGradientBoostingClassifier( X_train, y_train),
    "SVM" : train_svm(X_train, y_train)
}

else :
    models = {
    "Random Forest": train_forest(X_train, y_train),
    "HistGradientBoosting": train_HistGradientBoostingClassifier( X_train, y_train),
    }



#%% Evaluation with average precision

results = []

for name, model in models.items():

    # Uncommon
    probabilities = downbeat_probabilities(model, X_test)
    ap_uncommon = average_precision_score(y_test, probabilities)

    result = {
        "Model": name,
        "AP Uncommon": ap_uncommon
    }

    # Harmonix
    if EVALUATE_HARMONIX:
        probabilities_h = downbeat_probabilities(model, X_harmonix)
        ap_harmonix = average_precision_score(
            y_harmonix, probabilities_h
        )
        result["AP Harmonix"] = ap_harmonix

    results.append(result)

results_df = pd.DataFrame(results)

print("\nFinal test results:")
print(results_df.to_string(index=False))

print(f"\nUncommon positive ratio: {np.mean(y_test):.3f}")

if EVALUATE_HARMONIX:
    print(f"Harmonix positive ratio: {np.mean(y_harmonix):.3f}")
    
    
    
    
    
    

#%% Naive evaluation without HMM, using scipy.signal.find_peaks

from evaluate import eval_downbeats
from scipy.signal import find_peaks

THRESHOLD = 0.3

metrics = ["F", "CMLc", "CMLt", "AMLc", "AMLt"]

results_without_hmm = []
decoded_tracks_naive = {}

for model_name, model in models.items():

    for _, r in test_df.iterrows():

        key = (r["dataset"], r["track_id"])
        item = feature_dic_test.get(key)

        if item is None:
            continue

        beat_times = np.asarray(item["beat_times"], dtype=float)
        X = item["X"]
        ref = np.asarray(r["downbeat_times"], dtype=float)

        if len(beat_times) == 0 or len(ref) == 0:
            continue

        # Classifier probabilities
        probabilities = downbeat_probabilities(model, X)

        # Naive downbeat detection
        indices, properties = find_peaks(
            probabilities,
            height=THRESHOLD
        )

        est = beat_times[indices]

        # Evaluation
        F, CMLc, CMLt, AMLc, AMLt = eval_downbeats(
            ref,
            est,
            tolerance=TOLERANCE
        )

        results_without_hmm.append({
            "Model": model_name,
            "Track": r["track_id"],
            "F": F,
            "CMLc": CMLc,
            "CMLt": CMLt,
            "AMLc": AMLc,
            "AMLt": AMLt
        })

        decoded_tracks_naive[(model_name, r["track_id"])] = {
            "beat_times": beat_times,
            "probabilities": probabilities,
            "ref": ref,
            "est": est
        }


results_without_hmm_df = pd.DataFrame(results_without_hmm)

summary_without_hmm = (
    results_without_hmm_df
    .groupby("Model")[metrics]
    .mean()
)

print("\nNaive evaluation without HMM:")
print(summary_without_hmm.round(4).to_string())


#%% Train HMM on training dataset

from HMM.train_hmm import train_hmm_from_ids
from HMM.hmm import (
    build_observation_scores,
    viterbi,
    DOWNBEAT_STATES
)

train_ids = train_df["track_id"].tolist()

A, pi = train_hmm_from_ids(
    train_ids,
    smoothing=1.0,
    a=0.3,
    p_stay=0.9
)

print("\nHMM trained on", len(train_ids), "tracks")



#%% Evaluation with trained HMM
from evaluate import eval_downbeats
results_hmm = []
decoded_tracks_hmm = {}

for model_name, model in models.items():

    for _, r in test_df.iterrows():

        key = (r["dataset"], r["track_id"])
        item = feature_dic_test.get(key)

        if item is None:
            continue

        beat_times = np.asarray(item["beat_times"], dtype=float)
        X = item["X"]
        ref = np.asarray(r["downbeat_times"], dtype=float)

        if len(beat_times) == 0 or len(ref) == 0:
            continue

        # Classifier probabilities
        probabilities = downbeat_probabilities(model, X)

        # HMM observations
        observation_scores = build_observation_scores(
            probabilities
        )

        # Viterbi decoding
        state_path = viterbi(
            observation_scores,
            A,
            pi
        )

        if state_path is None:
            print("Viterbi failed:", r["track_id"])
            continue

        # Identify downbeat states
        indices = np.where(
            np.isin(state_path, DOWNBEAT_STATES)
        )[0]

        # Convert indices to seconds
        est = beat_times[indices]

        # Evaluate
        F, CMLc, CMLt, AMLc, AMLt = eval_downbeats(
            ref,
            est,
            tolerance=TOLERANCE
        )

        results_hmm.append({
            "Model": model_name,
            "Track": r["track_id"],
            "F": F,
            "CMLc": CMLc,
            "CMLt": CMLt,
            "AMLc": AMLc,
            "AMLt": AMLt
        })

        decoded_tracks_hmm[(model_name, r["track_id"])] = {
            "beat_times": beat_times,
            "probabilities": probabilities,
            "ref": ref,
            "est": est,
            "state_path": state_path
        }


results_hmm_df = pd.DataFrame(results_hmm)

summary_hmm = (
    results_hmm_df
    .groupby("Model")[metrics]
    .mean()
)

print("\nEvaluation with trained HMM:")
print(summary_hmm.round(4).to_string())








#%% Comparison: naive vs HMM

comparison = pd.concat(
    {
        "Without HMM": summary_without_hmm,
        "With HMM": summary_hmm
    },
    names=["Method", "Model"]
)

print("\nFinal comparison:")
print(comparison.round(4).to_string())










#%% F-measure for each track

fig, ax = plt.subplots(figsize=(13, 5))


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Metadata for each track
track_info = {}

for _, r in test_df.iterrows():

    signatures = [
        tuple(ts) for ts in r["time_signatures"]
        if ts is not None
    ]

    # Most frequent time signature
    if signatures:
        major_signature = Counter(signatures).most_common(1)[0][0]
        signature_str = f"{major_signature[0]}/{major_signature[1]}"
    else:
        signature_str = "Unknown"

    track_info[str(r["track_id"])] = {
        "signature": signature_str,
        "meter_changes": bool(r["contains_meter_changes"])
    }


#% Plot HMM performance

pivot = results_hmm_df.pivot(
    index="Track",
    columns="Model",
    values="F"
)

fig, ax = plt.subplots(figsize=(15, 6))

pivot.plot(kind="bar", ax=ax)

# Custom labels
labels = []

for track in pivot.index:

    info = track_info[str(track)]

    signature = info["signature"]
    changes = info["meter_changes"]

    label = f"{track}\n({signature})"

    if changes:
        label += " *"

    labels.append(label)

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")

# Highlight tracks with meter changes
for tick, track in zip(ax.get_xticklabels(), pivot.index):

    if track_info[str(track)]["meter_changes"]:
        tick.set_color("red")
        tick.set_fontweight("bold")

ax.set_ylabel("F-measure")
ax.set_xlabel("Track (major time signature)")
ax.set_title("HMM downbeat tracking performance")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)

ax.text(
    0.01, 0.98,
    "* Tracks with meter changes",
    transform=ax.transAxes,
    va="top",
    fontsize=10,
    color="red"
)

plt.tight_layout()
plt.show()














#%% Test down beat coverage, recall in the beat estimation
#This part of the programm isn't useful for the execution

import numpy as np

n_ref = 0
n_matched = 0

for r in train_df.to_dict("records"):
    item = feature_dic_train.get((r["dataset"], r["track_id"]))

    if item is None:
        continue

    beats = np.asarray(item["beat_times"])
    downbeats = np.asarray(r["downbeat_times"])

    if len(beats) == 0:
        continue

    n_ref += len(downbeats)

    for d in downbeats:
        if np.min(np.abs(beats - d)) <= 0.11:
            n_matched += 1

print("Downbeats annotated :", n_ref)
print("Downbeats found :", n_matched)
print("coverage :", n_matched / n_ref)


for tol in [0,0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:

    n_ref = 0
    n_matched = 0

    for r in train_df.to_dict("records"):
        item = feature_dic_train.get((r["dataset"], r["track_id"]))

        if item is None:
            continue

        beats = np.asarray(item["beat_times"])
        downbeats = np.asarray(r["downbeat_times"])

        if len(beats) == 0:
            continue

        n_ref += len(downbeats)

        for d in downbeats:
            if np.min(np.abs(beats - d)) <= tol:
                n_matched += 1

    print(f"{tol:.2f} s : {100*n_matched/n_ref:.1f} %")
    

    





