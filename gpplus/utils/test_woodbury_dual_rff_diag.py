"""Grad parity: fused diag-noise dual RFF MLL vs reference diag-noise path."""

from __future__ import annotations

import os

import torch

from gpplus.utils.rff_utils import (
    featurize_rbf,
    init_rbf_weights,
    woodbury_marginal_log_likelihood_diag_noise,
)
from gpplus.utils.woodbury_mll_autograd import woodbury_dual_rff_diag_mll_apply


def _make_problem(
    n: int = 48,
    d: int = 4,
    num_rff: int = 8,
    *,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
):
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=dtype).to(device)
    y = torch.randn(n, generator=g, dtype=dtype).to(device)
    weights = init_rbf_weights(
        d, num_rff, device=device, dtype=dtype, rff_sampling="rff"
    )
    ls = torch.randn(1, d, generator=g, dtype=dtype).to(device)
    os_ = torch.tensor(-0.25, dtype=dtype, device=device)
    # Heteroskedastic d > 0
    d_noise = 0.05 + 0.2 * torch.rand(n, generator=g, dtype=dtype).to(device)
    return d_noise, y, x, ls, os_, weights, num_rff


def _reference_mll(d, y, x, ls, os_, weights, num_rff, jitter: float):
    """Diag-noise MLL through featurize graph (materializing path)."""
    z = featurize_rbf(x, weights, ls, num_rff)
    scale = torch.pow(10.0, os_ / 2.0)
    while scale.dim() < z.dim():
        scale = scale.unsqueeze(-1)
    phi = z * scale
    return woodbury_marginal_log_likelihood_diag_noise(d, phi, y, jitter=jitter)


def test_fused_diag_mll_value_and_grads_match_reference():
    os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)
    os.environ.pop("GPPLUS_WOODBURY_FUSE_FEATURIZE", None)

    d0, y0, x, ls0, os0, weights, num_rff = _make_problem()
    jitter = 1e-6

    d_ref = d0.detach().clone().requires_grad_(True)
    y_ref = y0.detach().clone().requires_grad_(True)
    ls_ref = ls0.detach().clone().requires_grad_(True)
    os_ref = os0.detach().clone().requires_grad_(True)
    mll_ref = _reference_mll(d_ref, y_ref, x, ls_ref, os_ref, weights, num_rff, jitter)
    mll_ref.backward()

    d_f = d0.detach().clone().requires_grad_(True)
    y_f = y0.detach().clone().requires_grad_(True)
    ls_f = ls0.detach().clone().requires_grad_(True)
    os_f = os0.detach().clone().requires_grad_(True)
    mll_f = woodbury_dual_rff_diag_mll_apply(
        d_f, y_f, x, ls_f, os_f, weights, num_rff, jitter=jitter
    )
    mll_f.backward()

    assert torch.allclose(mll_f.detach(), mll_ref.detach(), rtol=1e-10, atol=1e-10)
    assert torch.allclose(d_f.grad, d_ref.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(y_f.grad, y_ref.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(ls_f.grad, ls_ref.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(os_f.grad, os_ref.grad, rtol=1e-8, atol=1e-8)


def test_nigp_mll_fused_matches_reference_env():
    """NIGPWoodbury MLL: fused path vs GPPLUS_WOODBURY_AUTOGRAD=reference."""
    from gpplus.models.rff_gpr import RFFGPR
    from gpplus.training.nigp_mll import NIGPWoodburyMarginalLogLikelihood

    g = torch.Generator().manual_seed(2)
    n, d, num_rff = 40, 3, 6
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    model = RFFGPR(
        x,
        y,
        num_rff=num_rff,
        ard=True,
        rff_sampling="rff",
        nigp=True,
    )
    model.train()
    # Ensure correction is on (default).
    model.nigp_correction_enabled = True

    mll = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood, model, woodbury_form="dual"
    )

    os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)
    os.environ["GPPLUS_WOODBURY_FUSE_FEATURIZE"] = "1"
    loss_f = -mll(None, model.train_targets)
    params = [p for p in model.parameters() if p.requires_grad]
    grads_f = torch.autograd.grad(loss_f, params, retain_graph=False, allow_unused=True)

    os.environ["GPPLUS_WOODBURY_AUTOGRAD"] = "reference"
    loss_r = -mll(None, model.train_targets)
    grads_r = torch.autograd.grad(loss_r, params, retain_graph=False, allow_unused=True)

    os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)
    os.environ.pop("GPPLUS_WOODBURY_FUSE_FEATURIZE", None)

    assert torch.allclose(loss_f.detach(), loss_r.detach(), rtol=1e-8, atol=1e-8)
    for gf, gr in zip(grads_f, grads_r):
        if gf is None and gr is None:
            continue
        assert gf is not None and gr is not None
        assert torch.allclose(gf, gr, rtol=1e-6, atol=1e-6)
