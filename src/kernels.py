"""Kernel functions implemented manually with NumPy.

All functions take X (n, d) and Z (m, d) and return the (n, m) matrix K[i, j] = k(X[i], Z[j]).
"""
import numpy as np


def squared_distances(X, Z):
    """||x - z||^2 = ||x||^2 + ||z||^2 - 2 x.z, vectorised.

    Floating-point cancellation can produce tiny negative numbers; they are clipped to 0.
    """
    xx = np.sum(X * X, axis=1)[:, None]
    zz = np.sum(Z * Z, axis=1)[None, :]
    return np.maximum(xx + zz - 2.0 * (X @ Z.T), 0.0)


def linear_kernel(X, Z):
    """K(x, z) = x^T z  (equivalent to the Linear SVM of the previous assignment)."""
    return X @ Z.T


def polynomial_kernel(X, Z, gamma=None, degree=3, coef0=1.0):
    """K(x, z) = (gamma x^T z + r)^d. Valid (PSD) for integer d >= 1, gamma > 0, r >= 0."""
    gamma = 1.0 / X.shape[1] if gamma is None else gamma
    return (gamma * (X @ Z.T) + coef0) ** degree


def rbf_kernel(X, Z, gamma):
    """K(x, z) = exp(-gamma ||x - z||^2). Always PSD; K(x, x) = 1."""
    return np.exp(-gamma * squared_distances(X, Z))


def sigmoid_kernel(X, Z, gamma=None, coef0=0.0):
    """K(x, z) = tanh(gamma x^T z + r). NOT PSD in general -> exploratory only."""
    gamma = 1.0 / X.shape[1] if gamma is None else gamma
    return np.tanh(gamma * (X @ Z.T) + coef0)


KERNELS = {"linear": linear_kernel, "polynomial": polynomial_kernel,
           "rbf": rbf_kernel, "sigmoid": sigmoid_kernel}


def make_kernel(name, **params):
    """Return a two-argument kernel function k(X, Z) with its parameters fixed."""
    fn = KERNELS[name]
    return lambda X, Z: fn(X, Z, **params)


def kernel_matrix(X, Z, kernel, batch_size=2048, dtype=np.float64):
    """Evaluate kernel(X, Z) in row batches to bound peak memory; returns an (n, m) array of `dtype`."""
    n = X.shape[0]
    out = np.empty((n, Z.shape[0]), dtype=dtype)
    for start in range(0, n, batch_size):
        out[start:start + batch_size] = kernel(X[start:start + batch_size], Z)
    return out


def min_eigenvalue(K):
    """Smallest eigenvalue of a symmetric matrix (PSD check: should be >= -tolerance)."""
    return float(np.linalg.eigvalsh((K + K.T) / 2.0)[0])
