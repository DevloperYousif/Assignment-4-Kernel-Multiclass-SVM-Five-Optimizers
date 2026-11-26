"""The five mandatory optimizers, written by hand, and the shared training loop.

All five minimise the SAME objective J(A, b) of kernel_multiclass_svm.py:
    GD      : one update per epoch on the full training set
    SGD     : one update per training sample (batch size 1)
    MBGD    : one update per mini-batch (default 64)
    AdaGrad : mini-batch subgradients + per-parameter accumulated squared gradients
    Adam    : mini-batch subgradients + bias-corrected first and second moments
"""
import time
import numpy as np

from kernel_multiclass_svm import gradients, decision_scores, hinge_losses, regularization  # local-import
from evaluation import accuracy, macro_f1  # local-import


# ----------------------------------------------------------------------------- update rules
class PlainGradientStep:
    """theta <- theta - lr * g   (used by GD, SGD and Mini-Batch GD; only the batch differs)."""

    def __init__(self, lr):
        self.lr = lr
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for p, g in zip(params, grads):
            p -= self.lr * g


class AdaGrad:
    """sq_grad_sum <- sq_grad_sum + g^2 ;  theta <- theta - lr * g / (sqrt(sq_grad_sum) + eps).

    `sq_grad_sum` is the AdaGrad accumulator (called G_t in the literature); it is deliberately NOT
    named G to avoid confusion with the kernel Gram matrix.
    """

    def __init__(self, lr, eps=1e-8):
        self.lr, self.eps = lr, eps
        self.sq_grad_sum = None
        self.t = 0

    def step(self, params, grads):
        if self.sq_grad_sum is None:                      # one accumulator per parameter array, starting at 0
            self.sq_grad_sum = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, acc in zip(params, grads, self.sq_grad_sum):
            acc += g * g
            p -= self.lr * g / (np.sqrt(acc) + self.eps)

    def effective_lr(self):
        """Median per-parameter step size lr / (sqrt(acc) + eps) of the coefficient matrix A
        (median, because coefficients that have received almost no gradient have huge, uninformative step sizes)."""
        return float(np.median(self.lr / (np.sqrt(self.sq_grad_sum[0]) + self.eps)))


class Adam:
    """m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; m_hat = m/(1-b1^t) ; v_hat = v/(1-b2^t) ;
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps).  t is the global update counter (starts at 1)."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        c1 = 1.0 - self.beta1 ** self.t
        c2 = 1.0 - self.beta2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def effective_lr(self):
        c2 = 1.0 - self.beta2 ** self.t
        return float(np.median(self.lr / (np.sqrt(self.v[0] / c2) + self.eps)))


OPTIMIZERS = {"GD": "Gradient Descent", "SGD": "Stochastic Gradient Descent",
              "MBGD": "Mini-Batch Gradient Descent", "AdaGrad": "AdaGrad", "Adam": "Adam"}
DEFAULT_BATCH = {"GD": None, "SGD": 1, "MBGD": 64, "AdaGrad": 64, "Adam": 64}


def make_optimizer(name, lr, **kw):
    if name in ("GD", "SGD", "MBGD"):
        return PlainGradientStep(lr)
    if name == "AdaGrad":
        return AdaGrad(lr, **kw)
    if name == "Adam":
        return Adam(lr, **kw)
    raise ValueError(name)


# ----------------------------------------------------------------------------- SGD fast path
def sgd_epoch_rotated(Ht, y, C, b, shrink, lr, order):
    """One SGD epoch in rotated coordinates C = U^T A, where G = U diag(Lam) U^T.

    The SGD step  A <- A - lr * (h_i^T d_i + lam G A)  becomes, after multiplying by U^T,
        C <- (1 - lr lam Lam) * C - lr * (h_i U)^T d_i
    which is the IDENTICAL iterate (U is orthogonal) but costs O(M K) instead of O(M^2 K) per update.
    Ht = H U is precomputed once. Equivalence with the naive update is verified in tests/test_optimizers.py.
    """
    for i in order:
        h = Ht[i]
        s = h @ C + b
        yi = y[i]
        true_score = s[yi]
        s[yi] = -np.inf
        j = int(np.argmax(s))
        active = 1.0 + s[j] - true_score > 0
        C *= shrink                                   # regularisation part, applied every update
        if active:                                    # data part: D_i = -1 at y_i, +1 at j*
            C[:, yi] += lr * h
            C[:, j] -= lr * h
            b[yi] += lr
            b[j] -= lr
    return C, b


# ----------------------------------------------------------------------------- training loop
def train(H, y, G, lam, optimizer, lr, epochs=100, batch_size=None, seed=42,
          H_val=None, y_val=None, n_classes=6, fast_sgd=True, opt_kwargs=None, verbose=False):
    """Train A, b from A = 0, b = 0 with the chosen optimizer.

    Returns (A, b, history) where history is a dict of per-epoch lists (index 0 = before training):
    train_hinge, train_reg, train_obj, val_hinge, val_obj, train_acc, val_acc, train_f1, val_f1,
    updates, samples, train_time, eff_lr; and 'diverged'.
    Only parameter updates (plus the SGD rotation precomputation) are timed; evaluation is not.
    """
    N, M = H.shape
    if batch_size is None:
        batch_size = DEFAULT_BATCH[optimizer] or N
    if optimizer == "GD":
        batch_size = N
    if optimizer == "SGD":
        batch_size = 1
    rng = np.random.default_rng(seed)
    A = np.zeros((M, n_classes))
    b = np.zeros(n_classes)
    opt = make_optimizer(optimizer, lr, **(opt_kwargs or {}))
    keys = ["train_hinge", "train_reg", "train_obj", "val_hinge", "val_obj", "train_acc", "val_acc",
            "train_f1", "val_f1", "updates", "samples", "train_time", "eff_lr"]
    hist = {k: [] for k in keys}
    hist["diverged"] = False
    state = {"updates": 0, "samples": 0, "time": 0.0}

    def record(eff):
        reg = regularization(A, G, lam)                     # computed once, shared by train / val objective
        S_tr = decision_scores(H, A, b)
        pred_tr = np.argmax(S_tr, axis=1)
        hinge_tr = float(np.mean(hinge_losses(S_tr, y)))
        hist["train_hinge"].append(hinge_tr); hist["train_reg"].append(reg); hist["train_obj"].append(hinge_tr + reg)
        hist["train_acc"].append(accuracy(y, pred_tr)); hist["train_f1"].append(macro_f1(y, pred_tr, n_classes))
        if H_val is not None:
            S_va = decision_scores(H_val, A, b)
            pred_va = np.argmax(S_va, axis=1)
            hinge_va = float(np.mean(hinge_losses(S_va, y_val)))
            hist["val_hinge"].append(hinge_va); hist["val_obj"].append(hinge_va + reg)
            hist["val_acc"].append(accuracy(y_val, pred_va)); hist["val_f1"].append(macro_f1(y_val, pred_va, n_classes))
        hist["updates"].append(state["updates"]); hist["samples"].append(state["samples"])
        hist["train_time"].append(state["time"]); hist["eff_lr"].append(eff)

    record(lr)
    rotated = optimizer == "SGD" and fast_sgd and G is not None
    if rotated:
        t0 = time.perf_counter()
        Lam, U = np.linalg.eigh(G)                    # G = U diag(Lam) U^T
        Ht = H @ U
        C = U.T @ A
        shrink = (1.0 - lr * lam * Lam)[:, None]
        state["time"] += time.perf_counter() - t0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(N)
        if rotated:
            C, b = sgd_epoch_rotated(Ht, y, C, b, shrink, lr, order)
            A = U @ C
            state["updates"] += N
        else:
            for start in range(0, N, batch_size):
                idx = order[start:start + batch_size]       # the last batch may be smaller
                gA, gb = gradients(A, b, H[idx], y[idx], G, lam)
                opt.step([A, b], [gA, gb])
                state["updates"] += 1
        state["samples"] += N
        state["time"] += time.perf_counter() - t0
        eff = opt.effective_lr() if hasattr(opt, "effective_lr") and opt.t > 0 else lr
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))) or np.abs(A).max() > 1e8:
            hist["diverged"] = True
            break
        record(eff)
        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d}  J={hist['train_obj'][-1]:.4f}  val F1={hist['val_f1'][-1] if H_val is not None else float('nan'):.4f}")
    return A, b, hist
