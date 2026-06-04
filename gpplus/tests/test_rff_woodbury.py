"""Tests for RFF Woodbury utilities and MLL."""

import math

import torch

from gpplus.models import RFFGPR
from gpplus.training.rff_mll import RFFWoodburyMarginalLogLikelihood
from gpplus.utils.rff_utils import (
    featurize_rbf,
    init_rbf_weights,
    woodbury_factor,
    woodbury_log_det,
    woodbury_log_det_from_chol,
    woodbury_middle_matrix,
    woodbury_marginal_log_likelihood,
    woodbury_quadratic_form,
    woodbury_solve,
    woodbury_solve_from_chol,
)


def test_spectral_init_scales_with_lengthscale():
    torch.manual_seed(4)
    d, D = 5, 100
    ls = torch.tensor([1.0, 2.0, 0.5, 4.0, 8.0])
    w = init_rbf_weights(d, D, lengthscale=ls)
    w_unit = init_rbf_weights(d, D)
    ratio = (w.std(dim=1) / w_unit.std(dim=1)).mean()
    expected = (1.0 / ls).mean() / (1.0 / ls.new_ones(d)).mean()
    assert ratio > 0.5


def test_featurize_shape():
    torch.manual_seed(0)
    n, d, D = 20, 3, 10
    x = torch.randn(n, d)
    w = init_rbf_weights(d, D)
    ls = torch.ones(1, d)
    z = featurize_rbf(x, w, ls, D)
    assert z.shape == (n, 2 * D)


def test_woodbury_solve_matches_dense():
    torch.manual_seed(1)
    n, m = 30, 12
    z = torch.randn(n, m)
    noise = torch.tensor(0.5)
    b = torch.randn(n)
    alpha = woodbury_solve(noise, z, b, jitter=1e-6)
    sigma = noise * torch.eye(n) + z @ z.transpose(-1, -2)
    expected = torch.linalg.solve(sigma, b)
    assert torch.allclose(alpha, expected, atol=1e-5, rtol=1e-4)


def test_woodbury_inverse_matches_explicit_formula():
    """Literal Woodbury inverse with Phi (n,m) matches woodbury_solve."""
    torch.manual_seed(22)
    n, m = 18, 7
    phi = torch.randn(n, m)
    noise = torch.tensor(0.4)
    b = torch.randn(n)
    sigma_sq = noise
    inv_noise = 1.0 / sigma_sq
    inv_noise_b = b * inv_noise
    middle = torch.eye(m, dtype=phi.dtype) + phi.transpose(-1, -2) @ phi * inv_noise
    middle_rhs = phi.transpose(-1, -2) @ inv_noise_b
    inner = torch.linalg.solve(middle, middle_rhs)
    expected = inv_noise_b - phi @ inner * inv_noise
    alpha = woodbury_solve(noise, phi, b, jitter=0.0)
    assert torch.allclose(alpha, expected, atol=1e-5, rtol=1e-4)


def test_woodbury_middle_matrix_matches_factor():
    torch.manual_seed(23)
    n, m = 12, 5
    z = torch.randn(n, m)
    noise = torch.tensor(0.25)
    middle = woodbury_middle_matrix(noise, z, jitter=1e-6)
    chol, _ = woodbury_factor(noise, z, jitter=1e-6)
    recon = chol @ chol.transpose(-1, -2)
    assert torch.allclose(middle, recon, atol=1e-5, rtol=1e-4)


def test_woodbury_solve_from_chol_matches_dense():
    torch.manual_seed(11)
    n, m = 30, 12
    z = torch.randn(n, m)
    noise = torch.tensor(0.5)
    b = torch.randn(n)
    chol, noise_c = woodbury_factor(noise, z, jitter=1e-6)
    alpha = woodbury_solve_from_chol(noise_c, z, chol, b)
    sigma = noise * torch.eye(n) + z @ z.transpose(-1, -2)
    expected = torch.linalg.solve(sigma, b)
    assert torch.allclose(alpha, expected, atol=1e-5, rtol=1e-4)


def test_woodbury_mll_matches_dense():
    torch.manual_seed(2)
    n, m = 25, 8
    z = torch.randn(n, m)
    noise = torch.tensor(0.3)
    y = torch.randn(n)
    mll = woodbury_marginal_log_likelihood(noise, z, y, jitter=1e-6)
    sigma = noise * torch.eye(n) + z @ z.transpose(-1, -2)
    chol = torch.linalg.cholesky(sigma + 1e-6 * torch.eye(n))
    quad = torch.cholesky_solve(y.unsqueeze(-1), chol).squeeze(-1)
    log_det = 2.0 * chol.diagonal().log().sum()
    expected = -0.5 * (y * quad).sum() - 0.5 * log_det - 0.5 * n * math.log(2 * math.pi)
    assert torch.allclose(mll, expected, atol=1e-4, rtol=1e-3)


def test_woodbury_mll_single_factorization():
    """MLL with shared chol matches separate log_det / quad from same factor."""
    torch.manual_seed(12)
    n, m = 25, 8
    z = torch.randn(n, m)
    noise = torch.tensor(0.3)
    y = torch.randn(n)
    mll = woodbury_marginal_log_likelihood(noise, z, y, jitter=1e-6)
    chol, noise_c = woodbury_factor(noise, z, jitter=1e-6)
    quad = woodbury_quadratic_form(noise, z, y, jitter=1e-6, chol=chol, noise=noise_c)
    log_det = woodbury_log_det_from_chol(noise_c, z, chol)
    const = -0.5 * n * math.log(2 * math.pi)
    expected = const - 0.5 * quad - 0.5 * log_det
    assert torch.allclose(mll, expected, atol=1e-5, rtol=1e-4)


def test_rff_kernel_diag_near_one():
    """With 1/sqrt(D) features, k(x,x) should be O(1), not 1/D."""
    from gpplus.kernels import RFFKernel

    torch.manual_seed(20)
    x = torch.randn(5, 3)
    for D in (100, 500):
        k = RFFKernel(num_samples=D, num_dims=3)
        diag = k(x, x, diag=True)
        assert diag.mean() > 0.5, f"expected k(x,x) near 1, got mean {diag.mean():.4f} for D={D}"


def test_scaled_features_match_kernel_diag():
    torch.manual_seed(13)
    n, d = 30, 4
    train_x = torch.randn(n, d)
    train_y = torch.randn(n)
    model = RFFGPR(train_x, train_y, num_rff=12, ard=False)
    z = model.scaled_features(train_x)
    k_woodbury = (z * z).sum(dim=-1)
    k_kernel = model.covar_module(train_x, train_x, diag=True)
    assert torch.allclose(k_woodbury, k_kernel, atol=1e-5, rtol=1e-4)


def test_woodbury_mll_matches_exact_mll():
    """Woodbury MLL should match ExactMarginalLogLikelihood (per-datapoint scale)."""
    import gpytorch

    torch.manual_seed(21)
    n, d = 40, 3
    train_x = torch.randn(n, d)
    train_y = torch.randn(n)
    model = RFFGPR(train_x, train_y, num_rff=20, ard=False)
    model.train()
    mll_w = RFFWoodburyMarginalLogLikelihood(model.likelihood, model)
    mll_e = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    out = model(train_x)
    assert torch.allclose(mll_w(None, train_y), mll_e(out, train_y), atol=1e-4, rtol=1e-3)


def test_rff_gpr_woodbury_mll_runs():
    torch.manual_seed(3)
    n, d = 40, 2
    train_x = torch.randn(n, d)
    train_y = torch.sin(train_x[:, 0]) + 0.1 * torch.randn(n)
    model = RFFGPR(train_x, train_y, num_rff=15, ard=False)
    mll = RFFWoodburyMarginalLogLikelihood(model.likelihood, model)
    model.train()
    loss = -mll(None, train_y)
    loss.backward()
    assert torch.isfinite(loss)


def test_woodbury_log_det_shared_factor():
    torch.manual_seed(14)
    n, m = 20, 6
    z = torch.randn(n, m)
    noise = torch.tensor(0.2)
    chol, noise_c = woodbury_factor(noise, z, jitter=1e-6)
    ld1 = woodbury_log_det(noise, z, jitter=1e-6)
    ld2 = woodbury_log_det_from_chol(noise_c, z, chol)
    assert torch.allclose(ld1, ld2, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    test_featurize_shape()
    test_woodbury_inverse_matches_explicit_formula()
    test_woodbury_middle_matrix_matches_factor()
    test_woodbury_solve_matches_dense()
    test_woodbury_solve_from_chol_matches_dense()
    test_woodbury_mll_matches_dense()
    test_woodbury_mll_single_factorization()
    test_scaled_features_match_kernel_diag()
    test_woodbury_mll_matches_exact_mll()
    test_rff_gpr_woodbury_mll_runs()
    test_woodbury_log_det_shared_factor()
    print("All tests passed.")
