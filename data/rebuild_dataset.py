"""Rebuild the UCI HAR dataset from the split archive parts stored in this folder.

GitHub's web upload limits file size, so the original dataset zip was split into parts of ~9 MB.
Run from the project root:   python data/rebuild_dataset.py
Result: data/UCI HAR Dataset/ (train/, test/, features.txt, activity_labels.txt, ...)
"""
import glob
import os
import zipfile

here = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(here, "UCI_HAR_Dataset.zip.part*")))
assert parts, "no archive parts found"
target = os.path.join(here, "UCI_HAR_Dataset.zip")
with open(target, "wb") as out:
    for p in parts:
        out.write(open(p, "rb").read())
zipfile.ZipFile(target).extractall(here)
print("extracted to", os.path.join(here, "UCI HAR Dataset"))
