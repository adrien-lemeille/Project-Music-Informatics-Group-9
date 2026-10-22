import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent




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

