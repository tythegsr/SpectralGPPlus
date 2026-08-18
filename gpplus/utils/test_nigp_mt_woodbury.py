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
