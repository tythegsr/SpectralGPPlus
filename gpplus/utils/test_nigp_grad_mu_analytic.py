"""Analytic / shared-Φ posterior_mean_grad_wrt_x parity tests."""

from __future__ import annotations

import torch

from gpplus.models.rff_gpr import RFFGPR
from gpplus.utils.nigp_utils import (
    _posterior_mean_grad_wrt_x_autograd,
    posterior_mean_grad_wrt_x,
)


def _make_model(n=40, d=3, num_rff=6, seed=0):
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
    return model, x, y


def test_analytic_grad_mu_matches_autograd_same_points():
    model, x, y = _make_model(seed=1)
    mean = model.mean_module(x).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    g_an = posterior_mean_grad_wrt_x(model, x, y_c, noise)
    g_ref = _posterior_mean_grad_wrt_x_autograd(model, x, y_c, noise)
    assert torch.allclose(g_an, g_ref, rtol=1e-10, atol=1e-10)


def test_analytic_grad_mu_matches_autograd_train_neq_x():
    model, x_train, y = _make_model(n=30, seed=2)
    g = torch.Generator().manual_seed(3)
    x_test = torch.randn(12, x_train.shape[-1], generator=g, dtype=torch.float64)
    mean = model.mean_module(x_train).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    g_an = posterior_mean_grad_wrt_x(
        model, x_test, y_c, noise, train_x=x_train
    )
    g_ref = _posterior_mean_grad_wrt_x_autograd(
        model, x_test, y_c, noise, train_x=x_train
    )
    assert g_an.shape == x_test.shape
    assert torch.allclose(g_an, g_ref, rtol=1e-10, atol=1e-10)


def test_same_points_single_featurize_omega():
    """Same-points analytic path should call featurize_rbf_scaled_omega once."""
    model, x, y = _make_model(seed=4)
    mean = model.mean_module(x).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    import gpplus.utils.woodbury_mll_autograd as wma

    calls = {"n": 0}
    orig = wma.featurize_rbf_scaled_omega

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    wma.featurize_rbf_scaled_omega = counted
    try:
        g = posterior_mean_grad_wrt_x(model, x, y_c, noise)
    finally:
        wma.featurize_rbf_scaled_omega = orig

    assert calls["n"] == 1
    assert g.shape == x.shape


def test_diff_points_two_featurize_omega():
    model, x_train, y = _make_model(n=20, seed=5)
    x_test = torch.randn(8, x_train.shape[-1], dtype=torch.float64)
    mean = model.mean_module(x_train).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    import gpplus.utils.woodbury_mll_autograd as wma

    calls = {"n": 0}
    orig = wma.featurize_rbf_scaled_omega

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    wma.featurize_rbf_scaled_omega = counted
    try:
        posterior_mean_grad_wrt_x(model, x_test, y_c, noise, train_x=x_train)
    finally:
        wma.featurize_rbf_scaled_omega = orig

    assert calls["n"] == 2
