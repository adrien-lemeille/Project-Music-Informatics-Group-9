import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

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

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
from sklearn.model_selection import train_test_split




from get_data_frame import get_full_dataframe
from features import build_feature_dictionnary
import pandas as pd

#%%

# Load annotated dataset
df = get_full_dataframe(include_harmonix = True)

# Test train split
train_df, test_df = train_test_split(df, test_size=0.2)

# features extraction
feature_dic = build_feature_dictionnary(train_df.to_dict("records"),sr=22050,dic = {},beat_track = 'other')

#%%

# Labels for training
X_train, y_train = training_arrays(
    train_df.to_dict("records"),
    feature_dic,
    tolerance=0.1)

print (np.sum(y_train)/len(y_train))

#%%

# Train
model_svm = train_svm(X_train, y_train)
#%%

model_forest = train_forest(X_train, y_train)

#%%

model_HistGradientBoostingClassifier = train_HistGradientBoostingClassifier(X_train, y_train)

#%%
# Labels for testing

# features extraction
feature_dic_test = build_feature_dictionnary(test_df.to_dict("records"),sr=22050,dic = {},beat_track = 'other')

X_test, y_test = training_arrays(
    test_df.to_dict("records"),
    feature_dic_test,
    tolerance=0.1)

#%%

y_test_est_HistGradientBoostingClassifier = downbeat_probabilities(model_HistGradientBoostingClassifier, X_test)


#%%
y_test_est_forest = downbeat_probabilities(model_forest, X_test)

#%%
y_test_est_forest = downbeat_probabilities(model_forest, X_test)

#%%

from sklearn.metrics import classification_report, confusion_matrix,average_precision_score

# Conversion en labels binaires
y_pred = (y_test_est >= 0.1).astype(int)



ap = average_precision_score(y_test, y_test_est)
print("Average Precision :", ap)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Beat", "Downbeat"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

ap = average_precision_score(y_test, y_test_est)
print("Average Precision :", ap)

#%% Test down beat coverage

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