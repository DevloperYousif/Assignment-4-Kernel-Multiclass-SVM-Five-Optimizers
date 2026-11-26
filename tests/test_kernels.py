import numpy as np
from kernels import (linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel, squared_distances,
                     kernel_matrix, make_kernel, min_eigenvalue)
from kernel_approximation import Nystroem, RandomFourierFeatures, approximation_error
from kernel_multiclass_svm import decision_scores, regularization

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 5)); Z = rng.normal(size=(15, 5))


def test_linear_kernel_matches_definition():
    assert np.allclose(linear_kernel(X, Z), np.array([[x @ z for z in Z] for x in X]))


def test_polynomial_kernel_matches_definition():
    K = polynomial_kernel(X, Z, gamma=0.3, degree=3, coef0=1.0)
    assert np.allclose(K, np.array([[(0.3 * x @ z + 1) ** 3 for z in Z] for x in X]))


def test_rbf_kernel_matches_definition_and_diagonal():
    K = rbf_kernel(X, Z, 0.2)
    ref = np.array([[np.exp(-0.2 * np.sum((x - z) ** 2)) for z in Z] for x in X])
    assert np.allclose(K, ref)
    assert np.allclose(np.diag(rbf_kernel(X, X, 0.2)), 1.0)


def test_sigmoid_kernel_definition():
    assert np.allclose(sigmoid_kernel(X, Z, 0.1, 0.5), np.tanh(0.1 * X @ Z.T + 0.5))


def test_squared_distances_nonnegative_with_duplicates():
    D = squared_distances(X * 1e4, X * 1e4)       # cancellation-prone; must be clipped at 0
    assert D.min() >= 0.0 and np.allclose(np.diag(D), 0.0, atol=1e-3 * D.max())


def test_kernel_matrix_dimensions_symmetry_and_batching():
    k = make_kernel("rbf", gamma=0.5)
    K = kernel_matrix(X, Z, k, batch_size=7)
    assert K.shape == (40, 15) and np.allclose(K, rbf_kernel(X, Z, 0.5))
    G = k(X, X)
    assert np.allclose(G, G.T)


def test_valid_kernels_are_psd():
    for G in (linear_kernel(X, X), polynomial_kernel(X, X, 0.2, 2, 1.0), rbf_kernel(X, X, 0.3)):
        assert min_eigenvalue(G) > -1e-8 * np.abs(G).max()


def test_sigmoid_can_be_indefinite():
    G = sigmoid_kernel(X, X, gamma=1.0, coef0=-1.0)
    assert min_eigenvalue(G) < -1e-6


def test_nystroem_full_basis_reconstructs_kernel():
    ny = Nystroem(0.2).fit(X)
    assert approximation_error(X, 0.2, ny) < 1e-6


def test_nystroem_is_reparametrisation_of_restricted_model():
    gamma = 0.2
    ny = Nystroem(gamma, rel_tol=0).fit(Z)
    W = rng.normal(size=(ny.n_components_, 3)); b = rng.normal(size=3)
    A = ny.map_ @ W                         # A = U Lam^(-1/2) W
    H = rbf_kernel(X, Z, gamma); G = rbf_kernel(Z, Z, gamma)
    assert np.allclose(decision_scores(H, A, b), decision_scores(ny.transform(X), W, b))
    assert np.isclose(regularization(A, G, 0.1), regularization(W, None, 0.1), rtol=1e-6)


def test_random_fourier_features_approximate_rbf():
    Xs = rng.normal(size=(60, 4)) * 0.5
    err_small = approximation_error(Xs, 0.5, RandomFourierFeatures(0.5, 200, 1).fit(Xs))
    err_big = approximation_error(Xs, 0.5, RandomFourierFeatures(0.5, 8000, 1).fit(Xs))
    assert err_big < err_small and err_big < 0.05
