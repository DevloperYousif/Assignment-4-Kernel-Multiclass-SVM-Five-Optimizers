"""Kernelized direct multi-class SVM (Crammer-Singer hinge loss) in coefficient space.

Representer form of the score functions (one function per class k):
    f_k(x) = sum_j A[j, k] * kernel(x, z_j) + b[k]
so with H[i, j] = kernel(x_i, z_j):          S = H A + b          (N x K)

Objective (RKHS-regularised Crammer-Singer hinge loss, margin 1):
    J(A, b) = (lam / 2) * Tr(A^T G A) + (1/N) * sum_i L_i
    L_i     = max(0, 1 + max_{j != y_i} S[i, j] - S[i, y_i])
where G[i, j] = kernel(z_i, z_j) is the basis Gram matrix, and Tr(A^T G A) = sum_k ||f_k||^2_H.

Explicit-feature mode: passing G=None means G = I, i.e. an ordinary linear multi-class SVM on the
columns of H. This is used for (a) the Linear SVM reference (H = X) and (b) the Nystrom /
Random-Fourier-Feature approximations (H = Phi), where ||W||^2 is the approximate RKHS norm.
"""
import numpy as np


# ----------------------------------------------------------------------------- core maths
def decision_scores(H, A, b):
    """S = H A + b."""
    return H @ A + b


def margin_terms(S, y):
    """Return (margins, j_star): margin_i = 1 + max_{j != y_i} S_ij - S_i,y_i and the arg-max wrong class.

    Tie rule: np.argmax returns the FIRST maximal incorrect class, so ties are broken by the smallest
    class index - deterministic and consistent for every optimizer.
    """
    n = S.shape[0]
    rows = np.arange(n)
    wrong = S.copy()
    wrong[rows, y] = -np.inf               # exclude the correct class from the inner max
    j_star = np.argmax(wrong, axis=1)
    margins = 1.0 + wrong[rows, j_star] - S[rows, y]
    return margins, j_star


def hinge_losses(S, y):
    """Per-sample Crammer-Singer hinge loss L_i = max(0, margin_i)."""
    margins, _ = margin_terms(S, y)
    return np.maximum(0.0, margins)


def score_subgradient(S, y):
    """Subgradient D = dL_i / dS_i for each sample.

    Active violation (margin_i > 0): D[i, y_i] = -1 and D[i, j*] = +1, all other entries 0.
    No violation (margin_i <= 0, including the kink margin_i == 0): the zero subgradient is used.
    """
    margins, j_star = margin_terms(S, y)
    D = np.zeros_like(S)
    active = margins > 0
    rows = np.flatnonzero(active)
    D[rows, y[rows]] = -1.0
    D[rows, j_star[rows]] = 1.0
    return D


def regularization(A, G, lam):
    """(lam/2) Tr(A^T G A) = (lam/2) sum_k a_k^T G a_k  (RKHS norm); with G=None -> (lam/2)||A||_F^2."""
    GA = A if G is None else G @ A
    return 0.5 * lam * float(np.sum(A * GA))


def objective(A, b, H, y, G, lam):
    """Return dict(hinge, reg, total) of the full objective on (H, y)."""
    S = decision_scores(H, A, b)
    hinge = float(np.mean(hinge_losses(S, y)))
    reg = regularization(A, G, lam)
    return {"hinge": hinge, "reg": reg, "total": hinge + reg}


def gradients(A, b, H, y, G, lam):
    """Subgradients of J on the (mini-)batch (H, y):

        dJ/dA = H^T D / n + lam * G A        dJ/db = sum(D, axis=0) / n

    The empirical term is averaged over the batch; the regularisation gradient is the exact
    full gradient lam*G*A in every update (it does not depend on the data). The bias is not regularised.
    """
    n = H.shape[0]
    D = score_subgradient(decision_scores(H, A, b), y)
    gA = H.T @ D / n + lam * (A if G is None else G @ A)
    gb = D.sum(axis=0) / n
    return gA, gb


# ----------------------------------------------------------------------------- model wrapper
class KernelMulticlassSVM:
    """Holds the basis, kernel, parameters (A, b) and everything needed for reproducible inference."""

    def __init__(self, kernel_name="rbf", kernel_params=None, lam=1e-3, n_classes=6,
                 class_names=None, explicit_features=False):
        self.kernel_name = kernel_name
        self.kernel_params = dict(kernel_params or {})
        self.lam = lam
        self.n_classes = n_classes
        self.class_names = list(class_names) if class_names is not None else [str(i) for i in range(n_classes)]
        self.explicit_features = explicit_features
        self.A = None
        self.b = None
        self.basis = None
        self.scaler_mean = None
        self.scaler_scale = None

    def kernel(self, X, Z):
        from kernels import make_kernel  # local-import
        return make_kernel(self.kernel_name, **self.kernel_params)(X, Z)

    def init_params(self, n_basis):
        """Identical deterministic initialisation for every optimizer: A = 0, b = 0."""
        self.A = np.zeros((n_basis, self.n_classes))
        self.b = np.zeros(self.n_classes)
        return self

    def features(self, X_scaled):
        """H = kernel(X, basis) (or X itself for the explicit linear model)."""
        return X_scaled if self.explicit_features else self.kernel(X_scaled, self.basis)

    def decision_function(self, X_raw):
        Xs = (np.asarray(X_raw, float) - self.scaler_mean) / self.scaler_scale
        return decision_scores(self.features(Xs), self.A, self.b)

    def predict(self, X_raw):
        return np.argmax(self.decision_function(X_raw), axis=1)

    # ---- serialisation: everything required for inference is stored in one .npz file
    def save(self, path, **extra):
        np.savez_compressed(path, A=self.A, b=self.b, basis=self.basis if self.basis is not None else np.zeros((0,)),
                            scaler_mean=self.scaler_mean, scaler_scale=self.scaler_scale,
                            lam=self.lam, kernel_name=self.kernel_name,
                            kernel_params_keys=np.array(list(self.kernel_params.keys())),
                            kernel_params_vals=np.array(list(self.kernel_params.values()), dtype=float),
                            class_names=np.array(self.class_names), explicit=self.explicit_features,
                            **{k: np.asarray(v) for k, v in extra.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        params = {str(k): float(v) for k, v in zip(z["kernel_params_keys"], z["kernel_params_vals"])}
        if "degree" in params:
            params["degree"] = int(params["degree"])
        m = cls(kernel_name=str(z["kernel_name"]), kernel_params=params, lam=float(z["lam"]),
                n_classes=len(z["class_names"]), class_names=[str(c) for c in z["class_names"]],
                explicit_features=bool(z["explicit"]))
        m.A, m.b, m.basis = z["A"], z["b"], z["basis"]
        m.scaler_mean, m.scaler_scale = z["scaler_mean"], z["scaler_scale"]
        return m
