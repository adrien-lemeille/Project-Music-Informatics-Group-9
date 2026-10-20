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



def filter_harmonix_recall(df, feature_dic, threshold=0.90, tol=0.1):

    keep = []

    for _, r in df.iterrows():

        if r["dataset"] != "harmonix":
            keep.append(True)
            continue

        item = feature_dic.get((r["dataset"], r["track_id"]))

        if item is None:
            keep.append(False)
            continue

        beats = np.asarray(item["beat_times"])
        downbeats = np.asarray(r["downbeat_times"])

        if len(beats) == 0 or len(downbeats) == 0:
            keep.append(False)
            continue

        recall = np.mean([
            np.min(np.abs(beats - d)) <= tol
            for d in downbeats
        ])

        keep.append(recall >= threshold)

    print(f"Harmonix tracks removed: {len(df) - sum(keep)}")

    return df.loc[keep].reset_index(drop=True)




def training_arrays(tracks, feature_dic, tolerance=0.07):
    """creates traing labels
    
    tarcks : annotaded tracks from the dataset
    dic : feature dictionnary

    X shaoe (N, D) et y de forme (N,), 1=downbeat.

    """
    from features import label_beats
    xs, ys = [], []
    for r in tracks:
        item = feature_dic.get((r['dataset'], r['track_id']))
        
        if item is None or not len(item['beat_times']):
            print('track',r['track_id'], 'not found it in the features dic')
            continue
    
        
        # Labels : 1=downbeat, 0=othe beat
        labels = label_beats(item['beat_times'], r['downbeat_times'], tolerance)

        xs.append(item['X'])
        ys.append(labels)
    
    return np.vstack(xs), np.concatenate(ys)


def train_svm(X, labels):
    """
    Train a SVM
    """
    return make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='scale',class_weight='balanced', probability=True)).fit(X, labels)



def train_forest(X, labels):
    """
    fit a forest classifier model
    """
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3,class_weight='balanced_subsample', n_jobs=-1).fit(X, labels)


def train_HistGradientBoostingClassifier(X, labels):
    model = HistGradientBoostingClassifier(
    max_iter=200,
    max_leaf_nodes=15,
    learning_rate=0.05,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=42
    )
    
    return model.fit(X, labels)


def downbeat_probabilities(model, X):
    """
    probability for each beat 
    """
    if len(X) == 0:
        return np.empty(0)
    
    return np.clip(model.predict_proba(X)[:, 1], 1e-5, 1-1e-5)


#%%
SEED = 42
TOLERANCE = 0.10

INCLUDE_HARMONIX_TRAIN = False
EVALUATE_HARMONIX = False

N_HARMONIX_TRAIN = 20  # Number of Harmonix tracks to put in the training


#%% Load datasets

df = get_full_dataframe(
    include_harmonix=True,
    include_uncommon=True
)

uncommon_df = df[df["dataset"] == "uncommon_meter"].copy()
harmonix_df = df[df["dataset"] == "harmonix"].copy()


#%% Train / test split

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
    chords=False
)

feature_dic_test = build_feature_dictionnary(
    test_df.to_dict("records"),
    sr=22050,
    dic={},
    beat_track="other",
    chords=False
)

if EVALUATE_HARMONIX:
    feature_dic_harmonix = build_feature_dictionnary(
        harmonix_test.to_dict("records"),
        sr=22050,
        dic={},
        beat_track="other",
        chords=False
    )


#%% Training and testing arrays

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

models = {
    "Random Forest": train_forest(X_train, y_train),
    "HistGradientBoosting": train_HistGradientBoostingClassifier( X_train, y_train)
}


#%% Evaluation

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
    
    
    
    
    
    
    
    
    
    
    

#%% Test down beat coverage, recall in the beat estimation
#This part of the programm isn't useful for the execution
"""
import numpy as np

n_ref = 0
n_matched = 0

for r in train_df.to_dict("records"):
    item = feature_dic.get((r["dataset"], r["track_id"]))

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
        item = feature_dic.get((r["dataset"], r["track_id"]))

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
    
    """
