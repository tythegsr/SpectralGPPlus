"""Analytic / shared-Φ posterior_mean_grad_wrt_x parity tests."""

from __future__ import annotations

import time

import torch

from gpplus.models.rff_gpr import RFFGPR
from gpplus.training.eval import evaluate_rff_gp_model
from gpplus.utils.nigp_utils import (
    _posterior_feature_weights_homoskedastic,
    _posterior_mean_grad_wrt_x_autograd,
    _rff_logscale_params,
    posterior_mean_grad_wrt_x,
)
from gpplus.utils.woodbury_mll_autograd import featurize_rbf_scaled_omega


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


def test_feature_weights_parity_and_single_featurize():
    """Cached homo weights match default path and skip the train featurize."""
    model, x_train, y = _make_model(n=28, seed=6)
    x_test = torch.randn(10, x_train.shape[-1], dtype=torch.float64)
    mean = model.mean_module(x_train).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    z_train = model.scaled_features(x_train)
    homo_w = _posterior_feature_weights_homoskedastic(noise, z_train, y_c)

    g_default = posterior_mean_grad_wrt_x(
        model, x_test, y_c, noise, train_x=x_train
    )
    g_cached = posterior_mean_grad_wrt_x(
        model, x_test, y_c, noise, train_x=x_train, feature_weights=homo_w
    )
    assert torch.allclose(g_default, g_cached, rtol=1e-10, atol=1e-10)

    import gpplus.utils.woodbury_mll_autograd as wma

    calls = {"n": 0}
    orig = wma.featurize_rbf_scaled_omega

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    wma.featurize_rbf_scaled_omega = counted
    try:
        posterior_mean_grad_wrt_x(
            model, x_test, y_c, noise, train_x=x_train, feature_weights=homo_w
        )
    finally:
        wma.featurize_rbf_scaled_omega = orig
    assert calls["n"] == 1


def test_feature_weights_with_proj_reuse_zero_featurize():
    model, x_train, y = _make_model(n=24, seed=7)
    x_test = torch.randn(9, x_train.shape[-1], dtype=torch.float64)
    mean = model.mean_module(x_train).reshape(y.shape)
    y_c = y - mean
    noise = model.likelihood.noise.reshape(())

    z_train = model.scaled_features(x_train)
    homo_w = _posterior_feature_weights_homoskedastic(noise, z_train, y_c)
    lengthscale, outputscale, randn_weights, num_samples = _rff_logscale_params(model)
    phi, proj, omega, scale_out = featurize_rbf_scaled_omega(
        x_test, randn_weights, lengthscale, outputscale, num_samples
    )
    del phi

    g_ref = posterior_mean_grad_wrt_x(
        model, x_test, y_c, noise, train_x=x_train, feature_weights=homo_w
    )

    import gpplus.utils.woodbury_mll_autograd as wma

    calls = {"n": 0}
    orig = wma.featurize_rbf_scaled_omega

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    wma.featurize_rbf_scaled_omega = counted
    try:
        g = posterior_mean_grad_wrt_x(
            model,
            x_test,
            y_c,
            noise,
            train_x=x_train,
            feature_weights=homo_w,
            proj=proj,
            omega=omega,
            scale_out=scale_out,
        )
    finally:
        wma.featurize_rbf_scaled_omega = orig

    assert calls["n"] == 0
    assert torch.allclose(g, g_ref, rtol=1e-10, atol=1e-10)


def test_evaluate_rff_nigp_chunked_matches_full_batch():
    model, x_train, y = _make_model(n=50, d=4, num_rff=16, seed=8)
    model.eval()
    g = torch.Generator().manual_seed(9)
    x_test = torch.randn(64, x_train.shape[-1], generator=g, dtype=torch.float64)

    mean0, lo0, hi0, std0 = evaluate_rff_gp_model(model, x_test, chunk_size=0)
    mean1, lo1, hi1, std1 = evaluate_rff_gp_model(model, x_test, chunk_size=17)
    assert torch.allclose(mean0, mean1, rtol=1e-10, atol=1e-10)
    assert torch.allclose(lo0, lo1, rtol=1e-10, atol=1e-10)
    assert torch.allclose(hi0, hi1, rtol=1e-10, atol=1e-10)
    assert torch.allclose(std0, std1, rtol=1e-10, atol=1e-10)


def test_evaluate_rff_nigp_predict_parity():
    model, x_train, y = _make_model(n=40, d=3, num_rff=12, seed=10)
    model.eval()
    x_test = torch.randn(32, x_train.shape[-1], dtype=torch.float64)
    mean_e, lo_e, hi_e, _ = evaluate_rff_gp_model(model, x_test, chunk_size=0)
    mean_p, lo_p, hi_p = model.predict(x_test)
    assert torch.allclose(mean_e, mean_p, rtol=1e-9, atol=1e-9)
    assert torch.allclose(lo_e, lo_p, rtol=1e-9, atol=1e-9)
    assert torch.allclose(hi_e, hi_p, rtol=1e-9, atol=1e-9)


def test_nigp_predict_bench_smoke():
    """Smoke timing: amortized path should finish a modest batch quickly on CPU."""
    n_train, n_test, d, num_rff = 200, 1000, 8, 64
    g = torch.Generator().manual_seed(11)
    x = torch.randn(n_train, d, generator=g, dtype=torch.float64)
    y = torch.randn(n_train, generator=g, dtype=torch.float64)
    model = RFFGPR(
        x, y, num_rff=num_rff, ard=True, rff_sampling="rff", nigp=True
    )
    model.eval()
    x_test = torch.randn(n_test, d, generator=g, dtype=torch.float64)

    # Warmup
    evaluate_rff_gp_model(model, x_test[:32], chunk_size=0)
    t0 = time.perf_counter()
    evaluate_rff_gp_model(model, x_test, chunk_size=0)
    elapsed = time.perf_counter() - t0
    # Loose ceiling so CI stays green; real GPU 25k target is separate.
    assert elapsed < 30.0, f"unexpectedly slow NIGP predict: {elapsed:.2f}s"
