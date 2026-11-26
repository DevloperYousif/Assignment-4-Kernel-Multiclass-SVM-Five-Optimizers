import numpy as np
from kernels import rbf_kernel
from kernel_multiclass_svm import gradients, objective, hinge_losses, score_subgradient, regularization

rng = np.random.default_rng(1)
N, M, K = 30, 12, 4
X = rng.normal(size=(N, 3)); Zb = X[:M]
H = rbf_kernel(X, Zb, 0.5); G = rbf_kernel(Zb, Zb, 0.5)
y = rng.integers(0, K, N)
A = rng.normal(size=(M, K)); b = rng.normal(size=K) * 0.1
lam = 0.05


def J(A_, b_):
    return objective(A_, b_, H, y, G, lam)["total"]


def test_hinge_loss_matches_loop():
    S = H @ A + b
    ref = [max(0.0, 1 + max(S[i, j] for j in range(K) if j != y[i]) - S[i, y[i]]) for i in range(N)]
    assert np.allclose(hinge_losses(S, y), ref)


def test_inner_max_excludes_correct_class():
    S = np.array([[5.0, 1.0, 0.0]]); yy = np.array([0])
    assert np.isclose(hinge_losses(S, yy)[0], 0.0)          # margin 1 + 1 - 5 < 0
    S = np.array([[1.0, 0.5, 0.0]])
    assert np.isclose(hinge_losses(S, yy)[0], 0.5)


def test_subgradient_structure_and_tie_rule():
    S = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]); yy = np.array([1, 0])
    D = score_subgradient(S, yy)
    assert np.allclose(D[0], [1, -1, 0])                    # tie among wrong classes -> smallest index
    assert np.allclose(D[1], 0)                             # no violation -> zero
    assert np.allclose(D.sum(axis=1), 0)


def test_rkhs_regularization():
    assert np.isclose(regularization(A, G, lam), 0.5 * lam * np.trace(A.T @ G @ A))
    assert np.isclose(regularization(A, None, lam), 0.5 * lam * np.sum(A ** 2))


def test_analytical_gradient_matches_numerical():
    gA, gb = gradients(A, b, H, y, G, lam)
    S = H @ A + b
    m = np.sort(np.abs(1 + np.max(np.where(np.eye(K)[y] > 0, -np.inf, S), 1) - S[np.arange(N), y]))
    assert m[0] > 1e-4                                       # away from non-differentiable points
    eps = 1e-6
    num = np.zeros_like(A)
    for i in range(M):
        for k in range(K):
            E = np.zeros_like(A); E[i, k] = eps
            num[i, k] = (J(A + E, b) - J(A - E, b)) / (2 * eps)
    numb = np.array([(J(A, b + eps * np.eye(K)[k]) - J(A, b - eps * np.eye(K)[k])) / (2 * eps) for k in range(K)])
    assert np.allclose(gA, num, atol=1e-6) and np.allclose(gb, numb, atol=1e-6)


def test_minibatch_gradient_is_average_plus_full_regularizer():
    idx = np.arange(7)
    gA, gb = gradients(A, b, H[idx], y[idx], G, lam)
    parts = [gradients(A, b, H[[i]], y[[i]], G, lam) for i in idx]
    assert np.allclose(gA, np.mean([p[0] for p in parts], axis=0))   # same lam*G*A in each term
    assert np.allclose(gb, np.mean([p[1] for p in parts], axis=0))
