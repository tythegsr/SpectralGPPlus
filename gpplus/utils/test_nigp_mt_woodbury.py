"""Tests for multitask NIGP Woodbury (diag task noise)."""

from __future__ import annotations

import torch

from gpplus.models.rff_mtgpr import RFFMTGPR
from gpplus.training.nigp_mt_mll import NIGPMTWoodburyMarginalLogLikelihood
from gpplus.training.rff_mt_mll import RFFMTWoodburyMarginalLogLikelihood
from gpplus.utils.nigp_utils import (
    effective_noise_variance,
    effective_noise_variance_mt,
    posterior_mean_grad_wrt_x,
    posterior_mean_grad_wrt_x_mt,
)
from gpplus.utils.rff_utils import (
    woodbury_marginal_log_likelihood_diag_noise,
    woodbury_marginal_log_likelihood_mt_diag_noise,
)


def _make_mt_problem(
    n: int = 20,
    d: int = 3,
    t: int = 2,
    num_rff: int = 4,
    *,
    rank_kernel: int = 0,
    nigp: bool = True,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, t, generator=g, dtype=torch.float64)
    model = RFFMTGPR(
        x,
        y,
        num_rff=num_rff,
        ard=True,
        rff_sampling="rff",
        rank_kernel=rank_kernel,
        rank_likelihood=0,
        nigp=nigp,
    )
    return model, x, y


def test_rank0_mll_matches_independent_st_diag_noise():
    """Diagonal B: MT diag-noise MLL equals sum of ST diag-noise MLLs."""
    model, x, y = _make_mt_problem(rank_kernel=0, nigp=True, seed=1)
    model.train()
    phi = model.scaled_spatial_features(x)
    r_b = model.task_psd_factor()
    # Force diagonal R_B for the algebra gate (rank0 should already be ~diag).
    r_diag = torch.diag(torch.diagonal(r_b))
    mean = model.mean_module(x)
    y_c = y - mean
    task_noises = model.task_noises().detach()
    # Synthetic grads / d
    g = torch.Generator().manual_seed(2)
    grad_mu = torch.randn(x.shape[0], model.num_tasks, x.shape[-1], generator=g, dtype=torch.float64)
    d_nt = effective_noise_variance_mt(task_noises, model.input_noise_var.detach(), grad_mu)

    mll_mt = woodbury_marginal_log_likelihood_mt_diag_noise(
        d_nt, phi, r_diag, x.shape[0], y_c.reshape(-1), jitter=1e-6
    )
    scales = torch.diagonal(r_diag)
    mll_sum = y.new_zeros(())
    for t in range(model.num_tasks):
        mll_sum = mll_sum + woodbury_marginal_log_likelihood_diag_noise(
            d_nt[:, t], phi * scales[t], y_c[:, t], jitter=1e-6
        )
    assert torch.allclose(mll_mt, mll_sum, rtol=1e-8, atol=1e-8)


def test_freeze_matches_parent_mt_mll():
    model, x, y = _make_mt_problem(rank_kernel=0, nigp=True, seed=3)
    model.nigp_correction_enabled = False
    mll_nigp = NIGPMTWoodburyMarginalLogLikelihood(model.likelihood, model, jitter=1e-6)
    mll_parent = RFFMTWoodburyMarginalLogLikelihood(model.likelihood, model, jitter=1e-6)
    v1 = mll_nigp(None, y)
    v2 = mll_parent(None, y)
    assert torch.allclose(v1, v2, rtol=1e-10, atol=1e-10)


def test_icm_nigp_forward_backward():
    model, x, y = _make_mt_problem(rank_kernel=1, nigp=True, seed=4)
    model.train()
    mll = NIGPMTWoodburyMarginalLogLikelihood(model.likelihood, model, jitter=1e-5)
    loss = -mll(None, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model._rff_kernel.raw_lengthscale.grad is not None
    assert model.raw_input_noise.grad is not None
    assert torch.isfinite(model.raw_input_noise.grad).all()


def test_effective_noise_variance_mt_shape():
    t, n, d = 3, 5, 4
    task_noises = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float64)
    input_var = torch.ones(d, dtype=torch.float64) * 0.01
    grad = torch.randn(n, t, d, dtype=torch.float64)
    d_nt = effective_noise_variance_mt(task_noises, input_var, grad)
    assert d_nt.shape == (n, t)
    # Match per-task ST formula
    for ti in range(t):
        d_t = effective_noise_variance(task_noises[ti], input_var, grad[:, ti, :])
        assert torch.allclose(d_nt[:, ti], d_t)


def test_posterior_mean_grad_mt_finite():
    model, x, y = _make_mt_problem(rank_kernel=0, nigp=True, seed=5)
    mean = model.mean_module(x)
    y_c = (y - mean).reshape(-1)
    with torch.enable_grad():
        g = posterior_mean_grad_wrt_x_mt(
            model, x, y_c, model.task_noises(), jitter=1e-6
        )
    assert g.shape == (x.shape[0], model.num_tasks, x.shape[-1])
    assert torch.isfinite(g).all()


def test_analytic_mt_grad_mu_matches_autograd():
    """Analytic ICM+RFF ∇μ matches the per-task autograd fallback."""
    import gpplus.utils.nigp_utils as nigp_u

    model, x, y = _make_mt_problem(rank_kernel=1, nigp=True, seed=6)
    mean = model.mean_module(x)
    y_c = (y - mean).reshape(-1)
    tn = model.task_noises()

    g_an = posterior_mean_grad_wrt_x_mt(model, x, y_c, tn, jitter=1e-6)

    orig = nigp_u._rff_logscale_params
    nigp_u._rff_logscale_params = lambda _m: None
    try:
        g_ref = posterior_mean_grad_wrt_x_mt(model, x, y_c, tn, jitter=1e-6)
    finally:
        nigp_u._rff_logscale_params = orig

    assert g_an.shape == g_ref.shape
    assert torch.allclose(g_an, g_ref, rtol=1e-8, atol=1e-8)


def test_middle_matrix_blocked_matches_kron():
    n, m, t = 12, 8, 3
    g = torch.Generator().manual_seed(7)
    phi = torch.randn(n, m, generator=g, dtype=torch.float64)
    a = torch.randn(t, t, generator=g, dtype=torch.float64)
    r_b = torch.linalg.cholesky(a @ a.T + torch.eye(t, dtype=torch.float64))
    d = 0.05 + torch.rand(n, t, generator=g, dtype=torch.float64)

    from gpplus.utils.rff_utils import woodbury_middle_matrix_mt_diag_noise

    m_blocked = woodbury_middle_matrix_mt_diag_noise(d, phi, r_b, jitter=1e-6)
    # Reference kron accumulation
    inv = d.clamp_min(1e-12).reciprocal()
    eye = torch.eye(m * t, dtype=torch.float64)
    m_ref = eye.clone()
    for task in range(t):
        g_t = phi.T @ (phi * inv[:, task].unsqueeze(-1))
        outer = torch.outer(r_b[task], r_b[task])
        m_ref = m_ref + torch.kron(g_t, outer)
    m_ref = m_ref + 1e-6 * eye
    assert torch.allclose(m_blocked, m_ref, rtol=1e-10, atol=1e-10)


def test_mt_diag_custom_autograd_matches_reference():
    """Custom MT diag MLL value/grads match stock Chol autodiff."""
    from gpplus.utils.rff_utils import woodbury_marginal_log_likelihood_mt_diag_noise
    from gpplus.utils.woodbury_mll_autograd import woodbury_mt_diag_mll_apply

    n, m, t = 16, 10, 3
    g = torch.Generator().manual_seed(8)
    phi0 = torch.randn(n, m, generator=g, dtype=torch.float64)
    a = torch.randn(t, t, generator=g, dtype=torch.float64)
    r0 = torch.linalg.cholesky(a @ a.T + torch.eye(t, dtype=torch.float64))
    d0 = 0.08 + torch.rand(n, t, generator=g, dtype=torch.float64)
    y0 = torch.randn(n * t, generator=g, dtype=torch.float64)

    def _grads(fn):
        phi = phi0.detach().clone().requires_grad_(True)
        r_b = r0.detach().clone().requires_grad_(True)
        d = d0.detach().clone().requires_grad_(True)
        y = y0.detach().clone().requires_grad_(True)
        mll = fn(d, phi, r_b, y)
        mll.backward()
        return (
            mll.detach(),
            d.grad.detach(),
            phi.grad.detach(),
            r_b.grad.detach(),
            y.grad.detach(),
        )

    mll_c, gd_c, gp_c, gr_c, gy_c = _grads(
        lambda d, phi, r_b, y: woodbury_mt_diag_mll_apply(d, phi, r_b, y, jitter=1e-6)
    )
    mll_r, gd_r, gp_r, gr_r, gy_r = _grads(
        lambda d, phi, r_b, y: woodbury_marginal_log_likelihood_mt_diag_noise(
            d, phi, r_b, n, y, jitter=1e-6
        )
    )
    assert torch.allclose(mll_c, mll_r, rtol=1e-8, atol=1e-8)
    assert torch.allclose(gd_c, gd_r, rtol=1e-5, atol=1e-5)
    assert torch.allclose(gp_c, gp_r, rtol=1e-5, atol=1e-5)
    assert torch.allclose(gr_c, gr_r, rtol=1e-5, atol=1e-5)
    assert torch.allclose(gy_c, gy_r, rtol=1e-5, atol=1e-5)


def test_predict_diag_factor_reuse_matches():
    """Precomputed Chol/α predict matches factoring inside the helper."""
    from gpplus.utils.rff_utils import (
        icm_omega_rmatvec,
        woodbury_factor_mt_diag_noise,
        woodbury_predict_mt_diag_noise,
        woodbury_solve_mt_diag_noise_from_chol,
    )

    model, x, y = _make_mt_problem(n=24, rank_kernel=1, nigp=True, seed=9)
    phi = model.scaled_spatial_features(x)
    r_b = model.task_psd_factor().detach()
    mean = model.mean_module(x)
    y_c = (y - mean).reshape(-1)
    tn = model.task_noises().detach()
    g = torch.Generator().manual_seed(10)
    grad_mu = torch.randn(x.shape[0], model.num_tasks, x.shape[-1], generator=g, dtype=x.dtype)
    d_nt = effective_noise_variance_mt(tn, model.input_noise_var.detach(), grad_mu)

    # Split train/test
    n_tr = 16
    phi_tr, phi_te = phi[:n_tr], phi[n_tr:]
    d_tr, d_te = d_nt[:n_tr], d_nt[n_tr:]
    y_tr = y_c[: n_tr * model.num_tasks]

    ref = woodbury_predict_mt_diag_noise(
        d_tr, phi_tr, phi_te, r_b, n_tr, model.num_tasks, y_tr, d_test=d_te
    )
    chol, d_c = woodbury_factor_mt_diag_noise(d_tr, phi_tr, r_b)
    alpha = woodbury_solve_mt_diag_noise_from_chol(d_c, phi_tr, r_b, chol, y_tr.to(chol.dtype))
    v = icm_omega_rmatvec(phi_tr, r_b, alpha.to(phi_tr.dtype))
    reused = woodbury_predict_mt_diag_noise(
        d_tr,
        phi_tr,
        phi_te,
        r_b,
        n_tr,
        model.num_tasks,
        y_tr,
        d_test=d_te,
        chol=chol,
        d_factor=d_c,
        alpha=alpha,
        feature_weights=v,
    )
    for a, b in zip(ref, reused):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-8)
