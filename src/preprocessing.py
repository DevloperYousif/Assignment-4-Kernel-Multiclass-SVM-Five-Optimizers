"""Preprocessing: label encoding, subject-aware validation split, standardisation,
reproducible kernel-basis selection and the RBF median heuristic.

Every statistic is fitted on the TRAINING split only (no data leakage).
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from dataset_loader import ACTIVITY_LABELS  # local-import

SEED = 42


def encode_labels(y, classes=ACTIVITY_LABELS):
    """Map activity names to integers 0..K-1 using the fixed official class order."""
    mapping = {c: i for i, c in enumerate(classes)}
    return np.asarray([mapping[v] for v in y], dtype=np.int64)


def subject_group_split(subjects, val_fraction=0.2, seed=SEED):
    """Split the official training set into train / validation by SUBJECT.

    No volunteer appears in both parts, so validation measures generalisation to new people,
    exactly like the official test set (which contains 9 unseen subjects).
    """
    subjects = np.asarray(subjects)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(subjects)), groups=subjects))
    return np.sort(tr), np.sort(va)


class Standardizer:
    """StandardScaler written out explicitly: z = (x - mean) / std, fitted on training data only."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)  # constant columns stay unchanged
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def select_basis(y, n_basis, seed=SEED):
    """Reproducible class-stratified selection of kernel basis samples (indices into the TRAINING split).

    Each class receives a share proportional to its frequency (at least one sample), so every
    activity is represented among the basis functions. n_basis >= len(y) returns all indices (exact mode).
    """
    y = np.asarray(y)
    n = len(y)
    if n_basis >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts / n * n_basis).astype(int))
    # distribute the remaining slots to the largest fractional parts
    rest = n_basis - quota.sum()
    if rest > 0:
        frac = counts / n * n_basis - np.floor(counts / n * n_basis)
        for i in np.argsort(-frac)[:rest]:
            quota[i] += 1
    idx = []
    for c, q in zip(classes, quota):
        members = np.flatnonzero(y == c)
        idx.append(rng.choice(members, size=min(q, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def median_heuristic_gamma(X, n_samples=2000, seed=SEED):
    """gamma = 1 / median(||x_i - x_j||^2) over random pairs of training samples."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    S = X[idx]
    sq = np.sum(S ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * S @ S.T, 0.0)
    iu = np.triu_indices(len(S), k=1)
    return float(1.0 / np.median(d2[iu]))
