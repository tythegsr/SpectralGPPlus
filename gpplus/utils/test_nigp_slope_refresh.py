"""Paper-style outer-loop slope refresh tests for NIGP Woodbury MLL."""

from __future__ import annotations

import torch

import gpplus.training.nigp_mll as nigp_mll_mod
from gpplus.models.rff_gpr import RFFGPR
from gpplus.training.nigp_mll import NIGPWoodburyMarginalLogLikelihood
from gpplus.utils import nigp_utils


def _make_model(n=32, d=3, num_rff=8, seed=0):
    g = torch.Generator().manual_seed(seed)
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
    model.nigp_correction_enabled = True
    return model, y


def _count_grad_calls(mll, y, n_steps: int) -> int:
    calls = {"n": 0}
    orig = nigp_utils.posterior_mean_grad_wrt_x

    def counting_grad(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    nigp_utils.posterior_mean_grad_wrt_x = counting_grad
    nigp_mll_mod.posterior_mean_grad_wrt_x = counting_grad
    try:
        for _ in range(n_steps):
            mll(None, y)
    finally:
        nigp_utils.posterior_mean_grad_wrt_x = orig
        nigp_mll_mod.posterior_mean_grad_wrt_x = orig
    return calls["n"]


def test_slope_refresh_none_every_step():
    model, y = _make_model(seed=1)
    mll = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood, model, woodbury_form="dual"
    )
    n_steps = 7
    assert _count_grad_calls(mll, y, n_steps) == n_steps
    assert mll.nigp_slope_refresh_count == n_steps


def test_slope_refresh_once_then_reuse():
    model, y = _make_model(seed=2)
    mll = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood,
        model,
        woodbury_form="dual",
        nigp_slope_refreshes=1,
        nigp_active_epochs=20,
    )
    n_steps = 8
    assert _count_grad_calls(mll, y, n_steps) == 1
    assert mll.nigp_slope_refresh_count == 1


def test_slope_refresh_evenly_spaced():
    model, y = _make_model(seed=3)
    # A=20, R=4 → interval=5 → refresh at steps 0,5,10,15
    mll = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood,
        model,
        woodbury_form="dual",
        nigp_slope_refreshes=4,
        nigp_active_epochs=20,
    )
    assert _count_grad_calls(mll, y, 20) == 4
    assert mll.nigp_slope_refresh_count == 4


def test_slope_phase_resets_after_freeze():
    model, y = _make_model(seed=4)
    mll = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood,
        model,
        woodbury_form="dual",
        nigp_slope_refreshes=1,
        nigp_active_epochs=10,
    )
    assert _count_grad_calls(mll, y, 3) == 1

    model.nigp_correction_enabled = False
    mll(None, y)  # freeze path; ends slope phase
    model.nigp_correction_enabled = True
    assert _count_grad_calls(mll, y, 3) == 1
    assert mll.nigp_slope_refresh_count == 1


def test_cached_slopes_match_fresh_on_refresh_steps():
    model, y = _make_model(seed=5)
    mll_every = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood, model, woodbury_form="dual"
    )
    mll_cached = NIGPWoodburyMarginalLogLikelihood(
        model.likelihood,
        model,
        woodbury_form="dual",
        nigp_slope_refreshes=2,
        nigp_active_epochs=4,  # interval=2 → refresh at 0,2
    )
    # First step: both refresh → same loss
    v0 = mll_every(None, y)
    v0c = mll_cached(None, y)
    assert torch.allclose(v0, v0c, rtol=0, atol=0)
