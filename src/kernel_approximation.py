"""Explicit approximate RBF feature maps (Mode C) - Nystrom and Random Fourier Features.

With an explicit feature map phi(x), a LINEAR multi-class SVM with weights W and penalty (lam/2)||W||^2
in phi-space is an approximate kernel SVM: phi(x)^T phi(z) ~= k(x, z).

Nystrom: with basis Z and G = U diag(Lam) U^T, phi(x) = k(x, Z) U_r diag(Lam_r)^(-1/2) keeping eigenvalues
above a relative tolerance (stabilisation). Then phi(x)^T phi(z) = k(x,Z) G^+ k(Z,z).
For the SAME basis this is an exact re-parametrisation of the restricted kernel model: A = U_r Lam_r^(-1/2) W
gives H A = Phi W and Tr(A^T G A) = ||W||^2 (verified in tests/test_kernels.py).

Random Fourier Features (Rahimi & Recht, 2007): for k(x,z) = exp(-gamma||x-z||^2) draw
omega ~ N(0, 2 gamma I), u ~ Uniform[0, 2 pi], and phi(x) = sqrt(2/D) cos(x omega + u).
"""
import numpy as np

from kernels import rbf_kernel  # local-import


class Nystroem:
    def __init__(self, gamma, rel_tol=1e-10):
        self.gamma, self.rel_tol = gamma, rel_tol

    def fit(self, Z):
        self.Z = np.asarray(Z, float)
        G = rbf_kernel(self.Z, self.Z, self.gamma)
        lam, U = np.linalg.eigh(G)
        keep = lam > self.rel_tol * lam.max()            # drop numerically zero / negative eigenvalues
        self.n_components_ = int(keep.sum())
        self.map_ = U[:, keep] / np.sqrt(lam[keep])      # U_r Lam_r^(-1/2)
        return self

    def transform(self, X, batch_size=2048):
        out = []
        for s in range(0, len(X), batch_size):
            out.append(rbf_kernel(X[s:s + batch_size], self.Z, self.gamma) @ self.map_)
        return np.vstack(out)


class RandomFourierFeatures:
    def __init__(self, gamma, n_components=2000, seed=42):
        self.gamma, self.n_components, self.seed = gamma, n_components, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.omega_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.offset_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.omega_ + self.offset_)


def approximation_error(X, gamma, feature_map):
    """Relative Frobenius error ||K - Phi Phi^T||_F / ||K||_F on a representative subset X."""
    K = rbf_kernel(X, X, gamma)
    P = feature_map.transform(X)
    return float(np.linalg.norm(K - P @ P.T) / np.linalg.norm(K))
