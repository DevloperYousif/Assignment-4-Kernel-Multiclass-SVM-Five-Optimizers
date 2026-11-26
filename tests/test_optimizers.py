import numpy as np
from kernels import rbf_kernel
from kernel_multiclass_svm import gradients
from optimizers import PlainGradientStep, AdaGrad, Adam, train

rng = np.random.default_rng(3)
N, K = 23, 3
X = rng.normal(size=(N, 2)); y = rng.integers(0, K, N)
G = rbf_kernel(X, X, 0.7); H = G
lam = 0.01


def test_gd_single_update_matches_formula():
    A, b, h = train(H, y, G, lam, "GD", 0.1, epochs=1, n_classes=K)
    gA, gb = gradients(np.zeros((N, K)), np.zeros(K), H, y, G, lam)
    assert np.allclose(A, -0.1 * gA) and np.allclose(b, -0.1 * gb) and h["updates"][-1] == 1


def test_sgd_one_sample_per_update_and_fast_path_equivalence():
    A1, b1, h1 = train(H, y, G, lam, "SGD", 0.05, epochs=3, n_classes=K, fast_sgd=True)
    A2, b2, h2 = train(H, y, G, lam, "SGD", 0.05, epochs=3, n_classes=K, fast_sgd=False)
    assert h1["updates"][-1] == 3 * N == h2["updates"][-1]
    assert np.allclose(A1, A2, atol=1e-10) and np.allclose(b1, b2, atol=1e-12)


def test_minibatch_counts_with_incomplete_last_batch():
    _, _, h = train(H, y, G, lam, "MBGD", 0.05, epochs=2, batch_size=5, n_classes=K)
    assert h["updates"][-1] == 2 * 5 and h["samples"][-1] == 2 * N     # 23 = 4*5 + 3


def test_minibatch_naive_reference():
    """Reproduce one MBGD epoch by hand with the same shuffling."""
    A, b, _ = train(H, y, G, lam, "MBGD", 0.05, epochs=1, batch_size=5, n_classes=K)
    order = np.random.default_rng(42).permutation(N)
    Ar, br = np.zeros((N, K)), np.zeros(K)
    for s in range(0, N, 5):
        idx = order[s:s + 5]
        gA, gb = gradients(Ar, br, H[idx], y[idx], G, lam)
        Ar -= 0.05 * gA; br -= 0.05 * gb
    assert np.allclose(A, Ar) and np.allclose(b, br)


def test_adagrad_accumulator():
    p = np.array([1.0, -2.0]); opt = AdaGrad(0.5, eps=1e-8)
    g1 = np.array([0.5, 1.0]); g2 = np.array([-1.0, 2.0])
    opt.step([p], [g1]); opt.step([p], [g2])
    acc = g1 ** 2 + g2 ** 2
    assert np.allclose(opt.sq_grad_sum[0], acc)
    expected = np.array([1.0, -2.0]) - 0.5 * g1 / (np.abs(g1) + 1e-8) - 0.5 * g2 / (np.sqrt(acc) + 1e-8)
    assert np.allclose(p, expected)


def test_adam_moments_and_bias_correction():
    p = np.array([0.0]); opt = Adam(0.1, 0.9, 0.999, 1e-8)
    opt.step([p], [np.array([2.0])])
    assert np.isclose(opt.m[0][0], 0.2) and np.isclose(opt.v[0][0], 0.004)
    assert np.isclose(p[0], -0.1, atol=1e-6)                   # first bias-corrected step = -lr * sign(g)
    opt.step([p], [np.array([2.0])])
    assert opt.t == 2 and np.isclose(p[0], -0.2, atol=1e-6)


def test_plain_step():
    p = np.array([1.0]); PlainGradientStep(0.5).step([p], [np.array([2.0])])
    assert np.isclose(p[0], 0.0)


def test_all_optimizers_decrease_objective():
    for opt, lr in [("GD", 0.05), ("SGD", 0.01), ("MBGD", 0.05), ("AdaGrad", 0.05), ("Adam", 0.01)]:
        _, _, h = train(H, y, G, lam, opt, lr, epochs=15, batch_size=8, n_classes=K)
        assert h["train_obj"][-1] < h["train_obj"][0], opt
