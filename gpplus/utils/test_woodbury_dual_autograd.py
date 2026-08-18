"""Tests for dual Woodbury MLL custom autograd."""

from __future__ import annotations

import os

import pytest
import torch

from gpplus.utils.rff_utils import (
    woodbury_marginal_log_likelihood_diag_noise,
    woodbury_marginal_log_likelihood_dual,
    woodbury_marginal_log_likelihood_dual_reference,
)
from gpplus.utils.woodbury_mll_autograd import WoodburyDualMLL, woodbury_dual_mll_apply


def _make_problem(
    n: int = 20,
    m: int = 6,
    *,
    batch: int | None = None,
    noise: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    g = torch.Generator().manual_seed(seed)
    if batch is None:
        phi = torch.randn(n, m, generator=g, dtype=dtype, device=device)
        y = torch.randn(n, generator=g, dtype=dtype, device=device)
        sigma = torch.tensor(noise, dtype=dtype, device=device)
    else:
        phi = torch.randn(batch, n, m, generator=g, dtype=dtype, device=device)
        y = torch.randn(batch, n, generator=g, dtype=dtype, device=device)
        sigma = torch.full((batch,), noise, dtype=dtype, device=device)
    return sigma, phi, y


def _grads(fn, noise, phi, y, jitter: float = 1e-6):
    noise = noise.detach().clone().requires_grad_(True)
    phi = phi.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    mll = fn(noise, phi, y, jitter=jitter)
    if mll.dim() > 0:
        mll = mll.sum()
    mll.backward()
    return mll.detach(), noise.grad.detach(), phi.grad.detach(), y.grad.detach()


def test_forward_matches_reference():
    noise, phi, y = _make_problem()
    custom = woodbury_dual_mll_apply(noise, phi, y, jitter=1e-6)
    ref = woodbury_marginal_log_likelihood_dual_reference(noise, phi, y, jitter=1e-6)
    assert torch.allclose(custom, ref, rtol=1e-12, atol=1e-12)


def test_parity_vs_reference_autodiff():
    noise, phi, y = _make_problem(n=24, m=8, noise=0.15)
    _, g_n_c, g_p_c, g_y_c = _grads(woodbury_dual_mll_apply, noise, phi, y)
    _, g_n_r, g_p_r, g_y_r = _grads(
        woodbury_marginal_log_likelihood_dual_reference, noise, phi, y
    )
    assert torch.allclose(g_n_c, g_n_r, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_p_c, g_p_r, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_y_c, g_y_r, rtol=1e-5, atol=1e-6)


def test_gradcheck_dual_mll():
    noise, phi, y = _make_problem(n=10, m=4, noise=0.2, seed=1)
    noise = noise.detach().clone().requires_grad_(True)
    phi = phi.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    jitter = phi.new_tensor(1e-8)

    def f(n_, p_, y_):
        return WoodburyDualMLL.apply(n_, p_, y_, jitter)

    assert torch.autograd.gradcheck(f, (noise, phi, y), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_batched_parity():
    noise, phi, y = _make_problem(n=16, m=5, batch=3, noise=0.12, seed=2)
    _, g_n_c, g_p_c, g_y_c = _grads(woodbury_dual_mll_apply, noise, phi, y)
    _, g_n_r, g_p_r, g_y_r = _grads(
        woodbury_marginal_log_likelihood_dual_reference, noise, phi, y
    )
    assert torch.allclose(g_n_c, g_n_r, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_p_c, g_p_r, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_y_c, g_y_r, rtol=1e-5, atol=1e-6)


def test_diag_noise_parity():
    """NIGP path: diag-noise → dual; custom backward must match reference."""
    g = torch.Generator().manual_seed(3)
    n, m = 18, 5
    phi = torch.randn(n, m, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    d = 0.05 + 0.1 * torch.rand(n, generator=g, dtype=torch.float64)

    def custom_fn(d_, phi_, y_, jitter=1e-6):
        # Force custom path (default)
        return woodbury_marginal_log_likelihood_diag_noise(d_, phi_, y_, jitter=jitter)

    def ref_fn(d_, phi_, y_, jitter=1e-6):
        os.environ["GPPLUS_WOODBURY_AUTOGRAD"] = "reference"
        try:
            return woodbury_marginal_log_likelihood_diag_noise(
                d_, phi_, y_, jitter=jitter
            )
        finally:
            os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)

    d_c = d.detach().clone().requires_grad_(True)
    p_c = phi.detach().clone().requires_grad_(True)
    y_c = y.detach().clone().requires_grad_(True)
    mll_c = custom_fn(d_c, p_c, y_c)
    mll_c.backward()

    d_r = d.detach().clone().requires_grad_(True)
    p_r = phi.detach().clone().requires_grad_(True)
    y_r = y.detach().clone().requires_grad_(True)
    mll_r = ref_fn(d_r, p_r, y_r)
    mll_r.backward()

    assert torch.allclose(mll_c.detach(), mll_r.detach(), rtol=1e-12, atol=1e-12)
    assert torch.allclose(d_c.grad, d_r.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(p_c.grad, p_r.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(y_c.grad, y_r.grad, rtol=1e-5, atol=1e-6)


def test_public_api_uses_custom_by_default():
    noise, phi, y = _make_problem()
    os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)
    noise = noise.detach().clone().requires_grad_(True)
    phi = phi.detach().clone().requires_grad_(True)
    mll = woodbury_marginal_log_likelihood_dual(noise, phi, y)
    mll.backward()
    assert noise.grad is not None and phi.grad is not None


def test_fused_rff_featurize_vjp_parity():
    """Fused RFF MLL grads match scaled_features → dual MLL path."""
    from gpplus.utils.rff_utils import featurize_rbf, init_rbf_weights
    from gpplus.utils.woodbury_mll_autograd import woodbury_dual_rff_mll_apply

    g = torch.Generator().manual_seed(7)
    n, d, D = 32, 5, 8
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    noise = torch.tensor(0.12, dtype=torch.float64)
    weights = init_rbf_weights(d, D, device=x.device, dtype=x.dtype, rff_sampling="rff")
    ls = torch.randn(1, d, generator=g, dtype=torch.float64)
    os_ = torch.tensor(-0.5, dtype=torch.float64)

    def run_fused(ls_, os_, noise_, y_):
        return woodbury_dual_rff_mll_apply(
            noise_, y_, x, ls_, os_, weights, D, jitter=1e-6
        )

    def run_ref(ls_, os_, noise_, y_):
        z = featurize_rbf(x, weights, ls_, D)
        scale = torch.pow(10.0, os_ / 2.0)
        phi = z * scale
        return woodbury_dual_mll_apply(noise_, phi, y_, jitter=1e-6)

    ls_f = ls.detach().clone().requires_grad_(True)
    os_f = os_.detach().clone().requires_grad_(True)
    n_f = noise.detach().clone().requires_grad_(True)
    y_f = y.detach().clone().requires_grad_(True)
    mll_f = run_fused(ls_f, os_f, n_f, y_f)
    mll_f.backward()

    ls_r = ls.detach().clone().requires_grad_(True)
    os_r = os_.detach().clone().requires_grad_(True)
    n_r = noise.detach().clone().requires_grad_(True)
    y_r = y.detach().clone().requires_grad_(True)
    mll_r = run_ref(ls_r, os_r, n_r, y_r)
    mll_r.backward()

    assert torch.allclose(mll_f.detach(), mll_r.detach(), rtol=1e-10, atol=1e-10)
    assert torch.allclose(ls_f.grad, ls_r.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(os_f.grad, os_r.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(n_f.grad, n_r.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(y_f.grad, y_r.grad, rtol=1e-5, atol=1e-6)


def test_fused_rff_float32_model_float64_factor():
    """Model float32 + default float64 Woodbury factors must backward cleanly."""
    from gpplus.utils.rff_utils import init_rbf_weights
    from gpplus.utils.woodbury_mll_autograd import woodbury_dual_rff_mll_apply

    g = torch.Generator().manual_seed(9)
    n, d, D = 28, 4, 6
    dtype = torch.float32
    x = torch.randn(n, d, generator=g, dtype=dtype)
    y = torch.randn(n, generator=g, dtype=dtype)
    noise = torch.tensor(0.15, dtype=dtype)
    weights = init_rbf_weights(d, D, device=x.device, dtype=dtype, rff_sampling="rff")
    ls = torch.randn(1, d, generator=g, dtype=dtype, requires_grad=True)
    os_ = torch.tensor(-0.3, dtype=dtype, requires_grad=True)
    noise_t = noise.detach().clone().requires_grad_(True)
    y_t = y.detach().clone().requires_grad_(True)

    mll = woodbury_dual_rff_mll_apply(noise_t, y_t, x, ls, os_, weights, D, jitter=1e-5)
    assert mll.dtype == torch.float64  # factor path promotes
    mll.backward()
    assert ls.grad is not None and ls.grad.dtype == dtype
    assert os_.grad is not None and os_.grad.dtype == dtype
    assert torch.isfinite(ls.grad).all()
    assert torch.isfinite(os_.grad).all()


def test_vjp_no_gphi_matches_gphi_path():
    """Streaming α/v/W VJP matches materializing full ∇_Φ."""
    from gpplus.utils.rff_utils import featurize_rbf, init_rbf_weights
    from gpplus.utils.woodbury_mll_autograd import (
        _phi_lam_inv,
        featurize_rbf_scaled_nograd,
        vjp_rff_from_alpha_v_w,
        vjp_rff_lengthscale_outputscale,
        woodbury_dual_mll_apply,
    )
    from gpplus.utils.rff_utils import woodbury_factor_dual

    g = torch.Generator().manual_seed(11)
    n, d, D = 24, 4, 6
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    noise = torch.tensor(0.2, dtype=torch.float64)
    weights = init_rbf_weights(d, D, device=x.device, dtype=x.dtype, rff_sampling="rff")
    ls = torch.randn(1, d, generator=g, dtype=torch.float64)
    os_ = torch.tensor(0.1, dtype=torch.float64)

    phi, proj, z_u, s_out, s_ls = featurize_rbf_scaled_nograd(x, weights, ls, os_, D)
    chol, noise_c, z_lin = woodbury_factor_dual(noise, phi, 1e-6)
    # Get alpha, v via one MLL step with grads on phi
    phi_t = phi.detach().clone().requires_grad_(True)
    noise_t = noise.detach().clone().requires_grad_(True)
    y_t = y.detach().clone().requires_grad_(True)
    mll = woodbury_dual_mll_apply(noise_t, phi_t, y_t, jitter=1e-6)
    mll.backward()
    g_phi = phi_t.grad

    # Reconstruct alpha, v from forward formula
    y64 = y.to(dtype=z_lin.dtype)
    phi_ty = z_lin.T @ y64
    v = torch.cholesky_solve(phi_ty.unsqueeze(-1), chol).squeeze(-1)
    alpha = (y64 - z_lin @ v) / noise_c
    w_phi = _phi_lam_inv(z_lin, chol)

    g_ls_a, g_os_a = vjp_rff_from_alpha_v_w(
        alpha, v, w_phi, torch.tensor(1.0, dtype=torch.float64),
        x, weights, ls, os_, proj, z_u, s_out, s_ls, D,
    )
    g_ls_b, g_os_b = vjp_rff_lengthscale_outputscale(
        g_phi, x, weights, ls, os_, proj, z_u, s_out, s_ls, D,
    )
    assert torch.allclose(g_ls_a, g_ls_b, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_os_a, g_os_b, rtol=1e-5, atol=1e-6)


def test_conditional_backward_noise_only_skips_phi_lam_inv(monkeypatch):
    """Noise-only grad must not require ΦΛ⁻¹ (conditional fused backward)."""
    import gpplus.utils.woodbury_mll_autograd as wma

    calls = {"phi": 0, "tr": 0}
    real_phi = wma._phi_lam_inv
    real_tr = wma._trace_inv_from_chol

    def wrap_phi(*a, **k):
        calls["phi"] += 1
        return real_phi(*a, **k)

    def wrap_tr(*a, **k):
        calls["tr"] += 1
        return real_tr(*a, **k)

    monkeypatch.setattr(wma, "_phi_lam_inv", wrap_phi)
    monkeypatch.setattr(wma, "_trace_inv_from_chol", wrap_tr)

    noise, phi, y = _make_problem(n=12, m=4)
    noise = noise.detach().clone().requires_grad_(True)
    phi = phi.detach()  # no grad on phi
    y = y.detach()
    mll = woodbury_dual_mll_apply(noise, phi, y)
    mll.backward()
    assert calls["tr"] == 1
    assert calls["phi"] == 0
    assert noise.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
