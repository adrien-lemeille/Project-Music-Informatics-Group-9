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
    

#%% 5 folds cross validation version


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks

from get_data_frame import get_full_dataframe
from features import build_feature_dictionnary
from classifier import *

from evaluate import eval_downbeats
from HMM.train_hmm import train_hmm_from_ids
from HMM.hmm import (
    build_observation_scores,
    viterbi,
    DOWNBEAT_STATES
)


#% Parameters

SEED = 42
N_FOLDS = 5
TOLERANCE = 0.10
THRESHOLD = 0.3

INCLUDE_SVM_MODEL = True

HMM_SMOOTHING = 1.0
HMM_A = 0.3
HMM_P_STAY = 0.9

metrics = ["F", "CMLc", "CMLt", "AMLc", "AMLt"]


#% Load dataset

df = get_full_dataframe(
    include_harmonix=False,
    include_uncommon=True
)

uncommon_df = (
    df[df["dataset"] == "uncommon_meter"]
    .copy()
    .reset_index(drop=True)
)

print("Number of tracks:", len(uncommon_df))


#% Feature extraction (once for all tracks)

feature_dic = build_feature_dictionnary(
    uncommon_df.to_dict("records"),
    sr=22050,
    dic={},
    beat_track="other"
)


#% Cross-validation

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

results_ap = []
results_without_hmm = []
results_hmm = []

decoded_tracks_naive = {}
decoded_tracks_hmm = {}

for fold, (train_idx, test_idx) in enumerate(
    kf.split(uncommon_df), start=1
):

    print(f"\n{'=' * 50}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'=' * 50}")

    train_df = uncommon_df.iloc[train_idx].reset_index(drop=True)
    test_df = uncommon_df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} tracks")
    print(f"Test: {len(test_df)} tracks")

    #% Training and testing arrays

    X_train, y_train = training_arrays(
        train_df.to_dict("records"),
        feature_dic,
        tolerance=TOLERANCE
    )

    X_test, y_test = training_arrays(
        test_df.to_dict("records"),
        feature_dic,
        tolerance=TOLERANCE
    )

    #% Train classifiers

    models = {
        "Random Forest": train_forest(X_train, y_train),
        "HistGradientBoosting": train_HistGradientBoostingClassifier(
            X_train, y_train
        )
    }

    if INCLUDE_SVM_MODEL:
        models["SVM"] = train_svm(X_train, y_train)

    #% Train HMM on this fold's training tracks

    train_ids = train_df["track_id"].tolist()

    A, pi = train_hmm_from_ids(
        train_ids,
        smoothing=HMM_SMOOTHING,
        a=HMM_A,
        p_stay=HMM_P_STAY
    )

    print("HMM trained on", len(train_ids), "tracks")

    # Prior estimated only from training data
    prior = np.mean(y_train)

    #% Evaluation

    for model_name, model in models.items():

        # Average precision
        probabilities_test = downbeat_probabilities(model, X_test)

        ap = average_precision_score(
            y_test,
            probabilities_test
        )

        results_ap.append({
            "Fold": fold,
            "Model": model_name,
            "AP": ap
        })

        # Track-by-track evaluation
        for _, r in test_df.iterrows():

            key = (r["dataset"], r["track_id"])
            item = feature_dic.get(key)

            if item is None:
                continue

            beat_times = np.asarray(
                item["beat_times"], dtype=float
            )

            X = item["X"]

            ref = np.asarray(
                r["downbeat_times"], dtype=float
            )

            if len(beat_times) == 0 or len(ref) == 0:
                continue

            probabilities = downbeat_probabilities(
                model, X
            )

            #-----------------------------------------
            # Naive evaluation without HMM
            #-----------------------------------------

            indices, _ = find_peaks(
                probabilities,
                height=THRESHOLD
            )

            est_naive = beat_times[indices]

            F, CMLc, CMLt, AMLc, AMLt = eval_downbeats(
                ref,
                est_naive,
                tolerance=TOLERANCE
            )

            results_without_hmm.append({
                "Fold": fold,
                "Model": model_name,
                "Track": r["track_id"],
                "F": F,
                "CMLc": CMLc,
                "CMLt": CMLt,
                "AMLc": AMLc,
                "AMLt": AMLt
            })

            decoded_tracks_naive[
                (fold, model_name, r["track_id"])
            ] = {
                "beat_times": beat_times,
                "probabilities": probabilities,
                "ref": ref,
                "est": est_naive
            }

            #-----------------------------------------
            # Evaluation with HMM
            #-----------------------------------------

            observation_scores = build_observation_scores(
                probabilities,
                prior=prior
            )

            state_path = viterbi(
                observation_scores,
                A,
                pi
            )

            if state_path is None:
                print("Viterbi failed:", r["track_id"])
                continue

            indices = np.where(
                np.isin(state_path, DOWNBEAT_STATES)
            )[0]

            est_hmm = beat_times[indices]

            F, CMLc, CMLt, AMLc, AMLt = eval_downbeats(
                ref,
                est_hmm,
                tolerance=TOLERANCE
            )

            results_hmm.append({
                "Fold": fold,
                "Model": model_name,
                "Track": r["track_id"],
                "F": F,
                "CMLc": CMLc,
                "CMLt": CMLt,
                "AMLc": AMLc,
                "AMLt": AMLt
            })

            decoded_tracks_hmm[
                (fold, model_name, r["track_id"])
            ] = {
                "beat_times": beat_times,
                "probabilities": probabilities,
                "ref": ref,
                "est": est_hmm,
                "state_path": state_path
            }

    print(f"Fold {fold} completed.")


#% Results DataFrames

results_ap_df = pd.DataFrame(results_ap)

results_without_hmm_df = pd.DataFrame(
    results_without_hmm
)

results_hmm_df = pd.DataFrame(results_hmm)


#%% Average precision across folds

summary_ap = (
    results_ap_df
    .groupby("Model")["AP"]
    .agg(["mean", "std"])
)

print("\nAverage precision across folds:")
print(summary_ap.round(4).to_string())


#%% Evaluation without HMM

summary_without_hmm = (
    results_without_hmm_df
    .groupby("Model")[metrics]
    .mean()
)

print("\nNaive evaluation without HMM:")
print(summary_without_hmm.round(4).to_string())


#%% Evaluation with HMM

summary_hmm = (
    results_hmm_df
    .groupby("Model")[metrics]
    .mean()
)

print("\nEvaluation with HMM:")
print(summary_hmm.round(4).to_string())


#%% Final comparison

comparison = pd.concat(
    {
        "Without HMM": summary_without_hmm,
        "With HMM": summary_hmm
    },
    names=["Method", "Model"]
)

print("\nFinal 5-fold comparison:")
print(comparison.round(4).to_string())


#%% Fold-by-fold comparison

fold_without_hmm = (
    results_without_hmm_df
    .groupby(["Fold", "Model"])[metrics]
    .mean()
)

fold_hmm = (
    results_hmm_df
    .groupby(["Fold", "Model"])[metrics]
    .mean()
)

fold_comparison = pd.concat(
    {
        "Without HMM": fold_without_hmm,
        "With HMM": fold_hmm
    },
    names=["Method"]
)

print("\nResults per fold:")
print(fold_comparison.round(4).to_string())


#%% Mean and standard deviation across folds

fold_summary = (
    fold_comparison
    .groupby(level=["Method", "Model"])
    .agg(["mean", "std"])
)

print("\nCross-validation mean ± std:")
print(fold_summary.round(4).to_string())

#%%



#%% SVM - Cross-validation analysis and plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path
from collections import Counter


# ============================================================
# PARAMETERS
# ============================================================

MODEL_NAME = "SVM"

# Enter track IDs here
GOOD_TRACK_ID = "rBvAz3890-Y"
BAD_TRACK_ID = "NNMR5MCpYGU"

# Optional: restrict visualization to a time interval (seconds)
# None = entire track
TIME_WINDOW = None
# Example: TIME_WINDOW = (20, 50)

SAVE_FIGURES = True
FIGURE_DIR = Path("Figures") / "SVM"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PREPARE RESULTS
# ============================================================

svm_naive = results_without_hmm_df[
    results_without_hmm_df["Model"] == MODEL_NAME
].copy()

svm_hmm = results_hmm_df[
    results_hmm_df["Model"] == MODEL_NAME
].copy()

svm_naive["Track"] = svm_naive["Track"].astype(str)
svm_hmm["Track"] = svm_hmm["Track"].astype(str)

# Merge naive and HMM results for each track and fold
svm_results = pd.merge(
    svm_naive,
    svm_hmm,
    on=["Fold", "Model", "Track"],
    suffixes=("_naive", "_hmm"),
    validate="one_to_one"
)


# ============================================================
# METADATA: TIME SIGNATURES
# ============================================================

def get_track_metadata(track_df):

    metadata = track_df.copy()
    metadata["Track"] = metadata["track_id"].astype(str)

    # Read measure annotations from the same directory as train_hmm
    from HMM.train_hmm import ANNOTATION_DIR

    annotation_metadata = {}

    for path in Path(ANNOTATION_DIR).glob("*.json"):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        track_id = str(data["youtube_id"])

        signatures = [
            f"{m['time_sig_num']}/{m['time_sig_den']}"
            for m in data["measures"]
        ]

        if not signatures:
            continue

        annotation_metadata[track_id] = {
            "contains_change_of_meter":
                len(set(signatures)) > 1,

            "majority_time_signature":
                Counter(signatures).most_common(1)[0][0]
        }

    # Use annotation-derived metadata
    metadata["contains_change_of_meter"] = (
        metadata["Track"].map(
            lambda x: annotation_metadata.get(x, {}).get(
                "contains_change_of_meter", np.nan
            )
        )
    )

    metadata["majority_time_signature"] = (
        metadata["Track"].map(
            lambda x: annotation_metadata.get(x, {}).get(
                "majority_time_signature", np.nan
            )
        )
    )

    return metadata[
        [
            "Track",
            "contains_change_of_meter",
            "majority_time_signature"
        ]
    ].drop_duplicates(subset="Track")


track_metadata = get_track_metadata(uncommon_df)

svm_results = svm_results.merge(
    track_metadata,
    on="Track",
    how="left",
    validate="many_to_one"
)

print("\nSVM tracks available:", len(svm_results))

print("\nMeter change distribution:")
print(
    svm_results["contains_change_of_meter"]
    .value_counts(dropna=False)
)

print("\nMajority time signatures:")
print(
    svm_results["majority_time_signature"]
    .value_counts(dropna=False)
)


# ============================================================
# 1. F-MEASURE AND CMLc VS METER CHANGE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

change_groups = [False, True]
labels = ["No meter change", "Meter change"]

for ax, metric in zip(axes, ["F", "CMLc"]):

    means = []
    stds = []

    for change in change_groups:

        values = svm_results.loc[
            svm_results["contains_change_of_meter"] == change,
            f"{metric}_hmm"
        ].dropna()

        means.append(values.mean())
        stds.append(values.std() if len(values) > 1 else 0)

    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_title(f"{metric} vs meter change")
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("SVM + HMM - Effect of meter changes")
fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIGURE_DIR / "meter_change.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()


# ============================================================
# 2. F-MEASURE VS MAJORITY TIME SIGNATURE
# ============================================================

signature_stats = (
    svm_results
    .dropna(subset=["majority_time_signature"])
    .groupby("majority_time_signature")["F_hmm"]
    .agg(["mean", "std", "count"])
)

# Sort signatures by numerator, then denominator
signature_stats = signature_stats.loc[
    sorted(
        signature_stats.index,
        key=lambda s: tuple(map(int, s.split("/")))
    )
]

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    signature_stats.index,
    signature_stats["mean"]
)

ax.set_title("SVM + HMM - F-measure by majority time signature")
ax.set_xlabel("Majority time signature")
ax.set_ylabel("Mean F-measure")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

# Display number of tracks above bars
for i, (_, row) in enumerate(signature_stats.iterrows()):
    ax.text(
        i,
        row["mean"] + 0.015,
        f"n={int(row['count'])}",
        ha="center",
        fontsize=9
    )

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIGURE_DIR / "time_signature.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()


# ============================================================
# 3. BEFORE / AFTER HMM - F-MEASURE AND CMLc
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, metric in zip(axes, ["F", "CMLc"]):

    before = svm_results[f"{metric}_naive"]
    after = svm_results[f"{metric}_hmm"]

    ax.scatter(before, after, alpha=0.65)



    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"{metric} without HMM")
    ax.set_ylabel(f"{metric} with HMM")
    ax.set_title(f"{metric}: before vs after HMM")
    ax.grid(alpha=0.3)

fig.suptitle("SVM - Track-level comparison")
fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIGURE_DIR / "before_after_hmm.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()



fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.5),
    constrained_layout=True
)

metrics_to_plot = ["F", "CMLc"]
labels = ["Without HMM", "With HMM"]
colors = ["#7A9EBC", "orange"]

for ax, metric in zip(axes, metrics_to_plot):

    before = svm_results[f"{metric}_naive"].dropna()
    after = svm_results[f"{metric}_hmm"].dropna()

    means = [before.mean(), after.mean()]
    stds = [before.std(), after.std()]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=1.5,
        error_kw={"elinewidth": 1.4, "capthick": 1.4}
    )

    # Display mean values above bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_title(metric, fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.65, max(np.array(means) + np.array(stds)) + 0.12))

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "SVM — Impact of HMM on Downbeat Detection",

)

#%% SVM - HMM impact depending on meter changes

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5),
    constrained_layout=True
)

metrics_to_plot = ["F", "CMLc"]

groups = [
    ("No meter change", False),
    ("Meter change", True)
]

colors = ["#7A9EBC", "orange"]
x = np.arange(len(groups))
width = 0.32

for ax, metric in zip(axes, metrics_to_plot):

    for i, (suffix, label) in enumerate([
        ("naive", "Without HMM"),
        ("hmm", "With HMM")
    ]):

        means = []
        stds = []

        for group_name, has_change in groups:

            values = svm_results.loc[
                svm_results["contains_change_of_meter"] == has_change,
                f"{metric}_{suffix}"
            ].dropna()

            means.append(values.mean())
            stds.append(values.std() if len(values) > 1 else 0)

        positions = x + (i - 0.5) * width

        bars = ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=5,
            color=colors[i],
            label=label,
            edgecolor="white",
            linewidth=1.2
        )

        # Values above bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.015,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in groups])
    ax.set_ylabel("Score")
    ax.set_title(metric, fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(frameon=False, loc="upper left")

fig.suptitle(
    "SVM — Impact of HMM and Meter Changes",

)

if SAVE_FIGURES:
    fig.savefig(
        FIGURE_DIR / "svm_hmm_meter_change_comparison.png",

    )

plt.show()
# ============================================================
# 4 & 5. TRACK-LEVEL VISUALIZATION
# ============================================================

def plot_track_comparison(track_id, title, filename):

    track_id = str(track_id)

    matching_keys = [
        key for key in decoded_tracks_hmm
        if key[1] == MODEL_NAME and str(key[2]) == track_id
    ]

    if not matching_keys:
        print(f"Track {track_id} not found.")
        return

    key = matching_keys[0]

    hmm_data = decoded_tracks_hmm[key]
    naive_data = decoded_tracks_naive[key]

    beat_times = np.asarray(hmm_data["beat_times"])
    probabilities = np.asarray(hmm_data["probabilities"])

    ref = np.asarray(hmm_data["ref"])
    est_hmm = np.asarray(hmm_data["est"])
    est_naive = np.asarray(naive_data["est"])

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Classifier probabilities
    axes[0].plot(
        beat_times,
        probabilities,
        label="SVM probability"
    )



    axes[0].set_ylabel("Downbeat probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Downbeat positions
    axes[1].eventplot(
        [ref, est_naive, est_hmm],
        lineoffsets=[3, 2, 1],
        linelengths=0.65
    )

    axes[1].set_yticks([1, 2, 3])
    axes[1].set_yticklabels([
        "With HMM",
        "Without HMM",
        "Ground truth"
    ])

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0.4, 3.6)
    axes[1].grid(axis="x", alpha=0.3)

    if TIME_WINDOW is not None:
        axes[1].set_xlim(*TIME_WINDOW)

    row = svm_results[
        svm_results["Track"] == track_id
    ]

    if not row.empty:
        r = row.iloc[0]

        fig.suptitle(
            f"{title} - {track_id}\n"
            f"F: {r['F_naive']:.3f} -> {r['F_hmm']:.3f} | "
            f"CMLc: {r['CMLc_naive']:.3f} -> {r['CMLc_hmm']:.3f}"
        )
    else:
        fig.suptitle(f"{title} - {track_id}")

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# Good example
if GOOD_TRACK_ID != "ENTER_GOOD_TRACK_ID":
    plot_track_comparison(
        GOOD_TRACK_ID,
        "Successful downbeat detection",
        "good_track.png"
    )

# Bad example
if BAD_TRACK_ID != "ENTER_BAD_TRACK_ID":
    plot_track_comparison(
        BAD_TRACK_ID,
        "Unsuccessful downbeat detection",
        "bad_track.png"
    )
#%%
