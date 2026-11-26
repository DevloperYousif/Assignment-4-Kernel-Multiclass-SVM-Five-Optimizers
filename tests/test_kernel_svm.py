import os, numpy as np
from kernels import rbf_kernel
from kernel_multiclass_svm import KernelMulticlassSVM, decision_scores
from optimizers import train
from preprocessing import select_basis, Standardizer, subject_group_split, median_heuristic_gamma

rng = np.random.default_rng(2)


def rings(n=300, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    t = r.uniform(0, 2 * np.pi, n)
    rad = 1.0 + 1.5 * y + r.normal(0, 0.15, n)
    return np.c_[rad * np.cos(t), rad * np.sin(t)], y


def test_scores_shape_and_prediction_labels():
    H = rng.random((10, 4)); A = rng.normal(size=(4, 3)); b = np.zeros(3)
    S = decision_scores(H, A, b)
    assert S.shape == (10, 3)
    pred = np.argmax(S, 1)
    assert set(pred) <= {0, 1, 2}


def test_rbf_svm_learns_nonlinear_rings_but_linear_cannot():
    X, y = rings()
    G = rbf_kernel(X, X, 1.0)
    A, b, h = train(G, y, G, 1e-3, "Adam", 0.05, epochs=60, batch_size=32, n_classes=3)
    acc_rbf = np.mean(np.argmax(G @ A + b, 1) == y)
    W, c, _ = train(X, y, None, 1e-3, "Adam", 0.05, epochs=60, batch_size=32, n_classes=3)
    acc_lin = np.mean(np.argmax(X @ W + c, 1) == y)
    assert acc_rbf > 0.95 and acc_lin < 0.6


def test_reproducible_training():
    X, y = rings(120)
    G = rbf_kernel(X, X, 1.0)
    r1 = train(G, y, G, 1e-3, "MBGD", 0.1, epochs=5, batch_size=16, n_classes=3)
    r2 = train(G, y, G, 1e-3, "MBGD", 0.1, epochs=5, batch_size=16, n_classes=3)
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])


def test_basis_selection_stratified_reproducible():
    y = np.repeat(np.arange(6), [100, 90, 80, 120, 130, 140])
    a, b2 = select_basis(y, 66, 42), select_basis(y, 66, 42)
    assert np.array_equal(a, b2) and len(a) == 66 and len(np.unique(y[a])) == 6
    assert np.array_equal(select_basis(y, 10_000, 42), np.arange(len(y)))


def test_standardizer_and_group_split():
    X = rng.normal(3, 2, size=(200, 4))
    s = Standardizer().fit(X)
    assert np.allclose(s.transform(X).mean(0), 0) and np.allclose(s.transform(X).std(0), 1)
    subj = np.repeat(np.arange(10), 20)
    tr, va = subject_group_split(subj, 0.2, 42)
    assert set(subj[tr]).isdisjoint(subj[va]) and len(tr) + len(va) == 200


def test_median_heuristic_positive():
    assert median_heuristic_gamma(rng.normal(size=(100, 5))) > 0


def test_numerical_stability_large_inputs():
    X = rng.normal(size=(50, 3)) * 1e3
    G = rbf_kernel(X, X, 1.0)
    assert np.all(np.isfinite(G)) and G.min() >= 0 and G.max() <= 1


def test_model_save_load_inference(tmp_path):
    X, y = rings(150)
    s = Standardizer().fit(X); Xs = s.transform(X)
    G = rbf_kernel(Xs, Xs, 1.0)
    A, b, _ = train(G, y, G, 1e-3, "Adam", 0.05, epochs=20, batch_size=32, n_classes=3)
    m = KernelMulticlassSVM("rbf", {"gamma": 1.0}, 1e-3, 3, ["a", "b", "c"])
    m.A, m.b, m.basis, m.scaler_mean, m.scaler_scale = A, b, Xs, s.mean_, s.scale_
    p = os.path.join(tmp_path, "m.npz"); m.save(p)
    m2 = KernelMulticlassSVM.load(p)
    assert np.allclose(m2.decision_function(X), G @ A + b) and np.array_equal(m2.predict(X), m.predict(X))
    assert m2.class_names == ["a", "b", "c"]
