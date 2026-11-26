"""Dataset loading for the UCI Human Activity Recognition (HAR) Using Smartphones dataset.

Two on-disk layouts are supported so the same code runs locally, in VS Code and on Kaggle:

1. The official UCI layout ("UCI HAR Dataset/train/X_train.txt", ...), as distributed by
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. The Kaggle CSV layout (train.csv / test.csv with "subject" and "Activity" columns), as in
   https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

Both contain the same 10,299 windows (7,352 train / 2,947 test), 561 features and 6 activities,
and the same official subject-based train/test partition (21 train subjects / 9 test subjects).
"""
import os
import glob
import numpy as np
import pandas as pd

# Official class order from activity_labels.txt (ids 1..6 -> encoded 0..5).
ACTIVITY_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                   "SITTING", "STANDING", "LAYING"]


def _candidate_roots():
    """Folders in which the dataset may live (local project, notebooks folder, Kaggle)."""
    here = os.getcwd()
    roots = [os.path.join(here, "data"), os.path.join(here, "..", "data"), here,
             "/kaggle/input"]
    env = os.environ.get("HAR_DATA_DIR")
    if env:
        roots.insert(0, env)
    return [r for r in roots if os.path.isdir(r)]


def find_dataset():
    """Return (layout, path). layout is 'uci' (folder with train/X_train.txt) or 'kaggle' (folder with train.csv)."""
    for root in _candidate_roots():
        for hit in glob.glob(os.path.join(root, "**", "X_train.txt"), recursive=True):
            folder = os.path.dirname(os.path.dirname(hit))  # .../UCI HAR Dataset
            if os.path.exists(os.path.join(folder, "features.txt")):
                return "uci", folder
        for hit in glob.glob(os.path.join(root, "**", "train.csv"), recursive=True):
            folder = os.path.dirname(hit)
            if os.path.exists(os.path.join(folder, "test.csv")):
                head = pd.read_csv(hit, nrows=2)
                if "Activity" in head.columns and "subject" in head.columns:
                    return "kaggle", folder
    raise FileNotFoundError(
        "UCI HAR dataset not found. Put the 'UCI HAR Dataset' folder inside data/ "
        "(see data/README.md) or attach the Kaggle dataset "
        "'uciml/human-activity-recognition-with-smartphones' to the notebook.")


def unique_feature_names(names):
    """features.txt contains 42 duplicated names (bandsEnergy blocks); append the index to keep them unique."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_har():
    """Load the official partition.

    Returns a dict with X_train, y_train (activity names), subject_train, X_test, y_test,
    subject_test (pandas objects), feature_names, layout and path.
    """
    layout, path = find_dataset()
    if layout == "uci":
        feats = pd.read_csv(os.path.join(path, "features.txt"), sep=r"\s+", header=None)[1].tolist()
        feats = unique_feature_names(feats)
        id2name = dict(zip(range(1, 7), ACTIVITY_LABELS))
        out = {}
        for part in ("train", "test"):
            X = pd.read_csv(os.path.join(path, part, f"X_{part}.txt"), sep=r"\s+", header=None)
            X.columns = feats
            y = pd.read_csv(os.path.join(path, part, f"y_{part}.txt"), header=None)[0].map(id2name)
            s = pd.read_csv(os.path.join(path, part, f"subject_{part}.txt"), header=None)[0]
            out[f"X_{part}"], out[f"y_{part}"], out[f"subject_{part}"] = X, y.rename("Activity"), s.rename("subject")
        out["feature_names"] = feats
    else:
        out = {}
        for part in ("train", "test"):
            df = pd.read_csv(os.path.join(path, f"{part}.csv"))
            out[f"y_{part}"] = df.pop("Activity")
            out[f"subject_{part}"] = df.pop("subject")
            out[f"X_{part}"] = df
        out["feature_names"] = list(out["X_train"].columns)
    out["layout"], out["path"] = layout, path
    return out


def describe_partition(data):
    """Small verification table: samples, subjects and features per official partition."""
    rows = []
    for part in ("train", "test"):
        rows.append({"partition": part, "samples": len(data[f"X_{part}"]),
                     "features": data[f"X_{part}"].shape[1],
                     "subjects": int(data[f"subject_{part}"].nunique()),
                     "classes": int(data[f"y_{part}"].nunique())})
    return pd.DataFrame(rows)
