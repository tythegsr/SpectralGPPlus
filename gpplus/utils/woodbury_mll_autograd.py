"""Custom autograd for dual Woodbury Gaussian MLL.

Avoids differentiating through Cholesky by using closed-form gradients of

    Σ = σ² I_n + Φ Φᵀ,  Λ = Φᵀ Φ + σ² I_m
    α = Σ⁻¹ y,  v = Λ⁻¹ Φᵀ y

    ∇_Φ ℓ = α vᵀ − Φ Λ⁻¹
    ∂ℓ/∂σ² = ½ (‖α‖² − tr(Σ⁻¹)),  tr(Σ⁻¹) = (n−m)/σ² + tr(Λ⁻¹)
    ∇_y ℓ = −α

RFF fused path applies the Φ-gradient VJP into lengthscale/outputscale **without**
materializing a full ``(n, m)`` ``∇_Φ`` buffer (only ``W = Φ Λ⁻¹``).

NIGP diag-noise uses the same fused featurize VJP after row-scaling
``Φ̂ = D^{-1/2} Φ`` (see ``WoodburyDualRFFDiagMLL``).

Env flags
---------
- ``GPPLUS_WOODBURY_AUTOGRAD=reference`` — stock Cholesky autodiff (A/B).
- ``GPPLUS_WOODBURY_LINALG=float32`` — optional float32 factors (off by default).
- ``GPPLUS_WOODBURY_FUSE_FEATURIZE=0`` — disable fused RFF featurize VJP.
- ``GPPLUS_WOODBURY_CUDA_TIMING=1`` — print CUDA section averages (Gram/chol/…).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import torch
from torch import Tensor

_LOG10 = math.log(10.0)

# ---------------------------------------------------------------------------
# CUDA section timing (optional)
# ---------------------------------------------------------------------------

_TIMING_ENABLED: bool | None = None
_TIMING_MS: dict[str, list[float]] = defaultdict(list)
_TIMING_PRINT_EVERY = 50


def use_cuda_timing() -> bool:
    global _TIMING_ENABLED
    if _TIMING_ENABLED is None:
        _TIMING_ENABLED = os.environ.get("GPPLUS_WOODBURY_CUDA_TIMING", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return _TIMING_ENABLED


def reset_cuda_timing() -> None:
    _TIMING_MS.clear()


def set_cuda_timing(enabled: bool | None = True) -> None:
    """Enable/disable section timers (``None`` re-reads ``GPPLUS_WOODBURY_CUDA_TIMING``)."""
    global _TIMING_ENABLED
    if enabled is None:
        _TIMING_ENABLED = None
    else:
        _TIMING_ENABLED = bool(enabled)
        if not _TIMING_ENABLED:
            reset_cuda_timing()


def cuda_timing_averages() -> dict[str, float]:
    return {k: sum(v) / len(v) for k, v in _TIMING_MS.items() if v}


def _record_ms(name: str, ms: float) -> None:
    bucket = _TIMING_MS[name]
    bucket.append(ms)
    if len(bucket) % _TIMING_PRINT_EVERY == 0:
        keys = sorted(_TIMING_MS.keys())
        parts = []
        for k in keys:
            vals = _TIMING_MS[k]
            parts.append(f"{k}={sum(vals) / len(vals):.2f}ms")
        print(f"[woodbury cuda timing avg n={len(bucket)}] " + " ".join(parts))


class _CudaSection:
    """Context manager: CUDA event timer, or no-op."""

    __slots__ = ("name", "enabled", "start", "end", "device")

    def __init__(self, name: str, device: torch.device | None = None):
        self.name = name
        self.enabled = use_cuda_timing() and torch.cuda.is_available()
        self.device = device
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        if not self.enabled or self.start is None or self.end is None:
            return False
        self.end.record()
        self.end.synchronize()
        _record_ms(self.name, float(self.start.elapsed_time(self.end)))
        return False


def use_reference_woodbury_autograd() -> bool:
    return os.environ.get("GPPLUS_WOODBURY_AUTOGRAD", "").strip().lower() in (
        "reference",
        "ref",
    )


def use_float32_woodbury_linalg() -> bool:
    return os.environ.get("GPPLUS_WOODBURY_LINALG", "").strip().lower() in (
        "float32",
        "f32",
    )


def use_fused_featurize_vjp() -> bool:
    val = os.environ.get("GPPLUS_WOODBURY_FUSE_FEATURIZE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _trace_inv_from_chol(chol: Tensor) -> Tensor:
    with _CudaSection("tr_lam_inv"):
        lam_inv = torch.cholesky_inverse(chol)
        return lam_inv.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def _phi_lam_inv(z_lin: Tensor, chol: Tensor) -> Tensor:
    with _CudaSection("phi_lam_inv"):
        phi_t = z_lin.transpose(-1, -2).contiguous()
        lam_inv_phi_t = torch.cholesky_solve(phi_t, chol)
        return lam_inv_phi_t.transpose(-1, -2).contiguous()


def _dual_mll_forward_values(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float,
):
    """Shared dual MLL forward (no_grad factor)."""
    from .rff_utils import woodbury_factor_dual

    noise_clamped = noise_var.clamp_min(1e-12)
    with torch.no_grad():
        with _CudaSection("factor_dual"):
            chol, noise, z_lin = woodbury_factor_dual(
                noise_clamped.detach(), z_train.detach(), jitter
            )

    n = z_train.shape[-2]
    m = z_train.shape[-1]
    y = y_centered.to(dtype=z_lin.dtype)
    if y.dim() == z_lin.dim() - 1:
        y_col = y.unsqueeze(-1)
    else:
        y_col = y

    with _CudaSection("fwd_solves"):
        log_det_lam = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1)
        phi_ty = (z_lin.transpose(-1, -2) @ y_col).squeeze(-1)
        v = torch.cholesky_solve(phi_ty.unsqueeze(-1), chol).squeeze(-1)
        lambda_quad = (phi_ty * v).sum(dim=-1)
        y_norm_sq = (y * y).sum(dim=-1)

    noise_b = noise
    if noise_b.dim() > y_norm_sq.dim():
        noise_b = noise_b.reshape(noise_b.shape[: y_norm_sq.dim()])

    const = -0.5 * n * math.log(2.0 * math.pi)
    mll = (
        const
        - 0.5 * (n - m) * noise_b.log()
        - 0.5 * log_det_lam
        - 0.5 * y_norm_sq / noise_b
        + 0.5 * lambda_quad / noise_b
    )

    if z_lin.dim() == 2:
        alpha = (y - z_lin @ v) / noise_b
    else:
        alpha = (
            y - torch.matmul(z_lin, v.unsqueeze(-1)).squeeze(-1)
        ) / noise_b.unsqueeze(-1)
    return mll, z_lin, chol, v, alpha, noise_b, n, m


def _scale_grad_output(grad_output: Tensor, like: Tensor) -> Tensor:
    if grad_output.dim() == 0:
        return grad_output
    return grad_output.reshape(-1, *([1] * (like.dim() - 1)))


def _grad_phi_from_saved(
    z_lin: Tensor,
    chol: Tensor,
    v: Tensor,
    alpha: Tensor,
    grad_output: Tensor,
) -> Tensor:
    """Materialize ∇_Φ (used by WoodburyDualMLL feature path / tests)."""
    phi_lam_inv = _phi_lam_inv(z_lin, chol)
    g_phi = alpha.unsqueeze(-1) * v.unsqueeze(-2) - phi_lam_inv
    return g_phi * _scale_grad_output(grad_output, g_phi)


def _grad_noise_from_saved(
    alpha: Tensor,
    noise_b: Tensor,
    chol: Tensor,
    n: int,
    m: int,
    grad_output: Tensor,
    *,
    below_floor: Tensor,
    noise_dtype: torch.dtype,
    noise_var_shape: tuple,
) -> Tensor:
    tr_lam_inv = _trace_inv_from_chol(chol)
    alpha_sq = (alpha * alpha).sum(dim=-1)
    tr_sigma_inv = (n - m) / noise_b + tr_lam_inv
    g_sigma = 0.5 * (alpha_sq - tr_sigma_inv)
    g_sigma = g_sigma * (grad_output if grad_output.dim() == 0 else grad_output)
    g_sigma = g_sigma.to(dtype=noise_dtype)
    below = below_floor
    if below.shape != g_sigma.shape:
        below = below.reshape(g_sigma.shape) if below.numel() == g_sigma.numel() else below
    g_sigma = torch.where(below, torch.zeros_like(g_sigma), g_sigma)
    if g_sigma.shape != noise_var_shape:
        g_sigma = g_sigma.reshape(noise_var_shape)
    return g_sigma


def _grad_y_from_saved(
    alpha: Tensor, grad_output: Tensor, y_dtype: torch.dtype
) -> Tensor:
    g_y = -alpha
    if grad_output.dim() == 0:
        g_y = g_y * grad_output
    else:
        g_y = g_y * grad_output.unsqueeze(-1)
    return g_y.to(dtype=y_dtype)


class WoodburyDualMLL(torch.autograd.Function):
    """Fused forward + analytic backward for dual Woodbury MLL on features Φ."""

    @staticmethod
    def forward(
        ctx,
        noise_var: Tensor,
        z_train: Tensor,
        y_centered: Tensor,
        jitter: Tensor,
    ) -> Tensor:
        jitter_f = float(jitter.detach().item())
        below_floor = noise_var < 1e-12
        mll, z_lin, chol, v, alpha, noise_b, n, m = _dual_mll_forward_values(
            noise_var, z_train, y_centered, jitter_f
        )
        ctx.save_for_backward(z_lin, chol, v, alpha, noise_b)
        ctx.below_floor = below_floor
        ctx.n = n
        ctx.m = m
        ctx.noise_var_shape = tuple(noise_var.shape)
        ctx.z_dtype = z_train.dtype
        ctx.y_dtype = y_centered.dtype
        ctx.noise_dtype = noise_var.dtype
        return mll

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z_lin, chol, v, alpha, noise_b = ctx.saved_tensors
        n, m = ctx.n, ctx.m
        needs = ctx.needs_input_grad

        grad_noise = grad_z = grad_y = None
        if needs[1]:
            grad_z = _grad_phi_from_saved(z_lin, chol, v, alpha, grad_output).to(
                dtype=ctx.z_dtype
            )
        if needs[0]:
            grad_noise = _grad_noise_from_saved(
                alpha,
                noise_b,
                chol,
                n,
                m,
                grad_output,
                below_floor=ctx.below_floor,
                noise_dtype=ctx.noise_dtype,
                noise_var_shape=ctx.noise_var_shape,
            )
        if needs[2]:
            grad_y = _grad_y_from_saved(alpha, grad_output, ctx.y_dtype)
        return grad_noise, grad_z, grad_y, None


def woodbury_dual_mll_apply(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    jitter_t = z_train.new_tensor(float(jitter))
    return WoodburyDualMLL.apply(noise_var, z_train, y_centered, jitter_t)


def _lengthscale_column_scale(lengthscale: Tensor) -> Tensor:
    scale = torch.pow(10.0, lengthscale / 2.0)
    if scale.dim() >= 2 and scale.shape[-2] == 1:
        return scale.transpose(-1, -2)
    return scale.reshape(-1, 1)


def featurize_rbf_scaled_nograd(
    x: Tensor,
    randn_weights: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    num_samples: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Φ = 10^(outputscale/2) * RFF features."""
    D = num_samples
    scale_ls = _lengthscale_column_scale(lengthscale)
    omega = randn_weights * scale_ls
    if x.dim() == 2 and omega.dim() > 2:
        proj = torch.einsum("nd,...dD->...nD", x, omega)
    else:
        proj = x.matmul(omega)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    z_unscaled = proj.new_empty(proj.shape[:-1] + (2 * D,))
    z_unscaled[..., :D] = torch.cos(proj).mul(inv_sqrt_d)
    z_unscaled[..., D:] = torch.sin(proj).mul(inv_sqrt_d)
    scale_out = torch.pow(10.0, outputscale / 2.0)
    while scale_out.dim() < z_unscaled.dim():
        scale_out = scale_out.unsqueeze(-1)
    phi = z_unscaled * scale_out
    return phi, proj, z_unscaled, scale_out, scale_ls


def rff_grad_mu_from_proj(
    proj: Tensor,
    omega: Tensor,
    scale_out: Tensor,
    w: Tensor,
    num_samples: int,
) -> Tensor:
    """
    Analytic ``∇_x μ`` for ``μ = φ(x)ᵀ w`` with frozen RFF features.

    ``proj = x @ Ω``, ``Ω = randn_weights * 10^(ls/2)``, ``φ = s [cos, sin]/√D``.

        ∂μ/∂x = g @ Ωᵀ,
        g_k = (s/√D) (−w^{cos}_k sin(proj_k) + w^{sin}_k cos(proj_k))

    ``proj`` ``(n, D)``, ``omega`` ``(d, D)``, ``w`` ``(2D,)`` → ``(n, d)``.
    """
    D = int(num_samples)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    if w.dim() == 2:
        w = w[0]
    w = w.to(dtype=proj.dtype)
    omega = omega.to(dtype=proj.dtype)
    w_cos = w[..., :D]
    w_sin = w[..., D : 2 * D]
    s = scale_out.to(dtype=proj.dtype)
    while s.dim() < proj.dim():
        s = s.unsqueeze(-1)
    # Broadcast w over n: (D,) with (n, D) -> (n, D)
    g = s * inv_sqrt_d * (-w_cos * torch.sin(proj) + w_sin * torch.cos(proj))
    if omega.dim() == 2:
        return g.matmul(omega.transpose(0, 1))
    return torch.matmul(g, omega.transpose(-1, -2))


def featurize_rbf_scaled_omega(
    x: Tensor,
    randn_weights: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    num_samples: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Scaled RFF features plus frequencies for analytic ``∇_x μ``.

    Returns ``phi, proj, omega, scale_out`` with ``proj = x @ omega``.
    """
    D = num_samples
    scale_ls = _lengthscale_column_scale(lengthscale)
    omega = randn_weights * scale_ls
    if x.dim() == 2 and omega.dim() > 2:
        proj = torch.einsum("nd,...dD->...nD", x, omega)
    else:
        proj = x.matmul(omega)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    z_unscaled = proj.new_empty(proj.shape[:-1] + (2 * D,))
    z_unscaled[..., :D] = torch.cos(proj).mul(inv_sqrt_d)
    z_unscaled[..., D:] = torch.sin(proj).mul(inv_sqrt_d)
    scale_out = torch.pow(10.0, outputscale / 2.0)
    while scale_out.dim() < z_unscaled.dim():
        scale_out = scale_out.unsqueeze(-1)
    phi = z_unscaled * scale_out
    return phi, proj, omega, scale_out


def _align_rff_vjp_inputs(
    compute_dtype: torch.dtype,
    x: Tensor,
    randn_weights: Tensor,
    proj: Tensor,
    z_unscaled: Tensor,
    scale_out: Tensor,
    scale_ls: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Cast featurize-side tensors to factor dtype (float32 model → float64 Λ)."""
    return (
        x.to(dtype=compute_dtype),
        randn_weights.to(dtype=compute_dtype),
        proj.to(dtype=compute_dtype),
        z_unscaled.to(dtype=compute_dtype),
        scale_out.to(dtype=compute_dtype),
        scale_ls.to(dtype=compute_dtype),
    )


def vjp_rff_lengthscale_outputscale(
    g_phi: Tensor,
    x: Tensor,
    randn_weights: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    proj: Tensor,
    z_unscaled: Tensor,
    scale_out: Tensor,
    scale_ls: Tensor,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    """Reference VJP via full ``g_phi`` (tests / fallback)."""
    D = num_samples
    inv_sqrt_d = 1.0 / math.sqrt(D)
    x, randn_weights, proj, z_unscaled, scale_out, scale_ls = _align_rff_vjp_inputs(
        g_phi.dtype, x, randn_weights, proj, z_unscaled, scale_out, scale_ls
    )
    g_z = g_phi * scale_out
    g_proj = (-torch.sin(proj) * inv_sqrt_d) * g_z[..., :D] + (
        torch.cos(proj) * inv_sqrt_d
    ) * g_z[..., D:]
    g_w = x.transpose(-1, -2) @ g_proj
    g_scale_ls = (g_w * randn_weights).sum(dim=-1, keepdim=True)
    g_ls_col = g_scale_ls * scale_ls * (_LOG10 / 2.0)
    if lengthscale.dim() >= 2 and lengthscale.shape[-2] == 1:
        g_lengthscale = g_ls_col.transpose(-1, -2)
    else:
        g_lengthscale = g_ls_col.reshape(lengthscale.shape)
    if outputscale.numel() == 1:
        g_s = (g_phi * z_unscaled).sum()
        g_outputscale = (g_s * scale_out.reshape(()) * (_LOG10 / 2.0)).reshape(
            outputscale.shape
        )
    else:
        g_s_b = (g_phi * z_unscaled).flatten(start_dim=-2).sum(dim=-1)
        s_b = scale_out.reshape(g_s_b.shape)
        g_outputscale = (g_s_b * s_b * (_LOG10 / 2.0)).reshape(outputscale.shape)
    return g_lengthscale, g_outputscale


def vjp_rff_from_alpha_v_w(
    alpha: Tensor,
    v: Tensor,
    w_phi: Tensor,
    grad_output: Tensor,
    x: Tensor,
    randn_weights: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    proj: Tensor,
    z_unscaled: Tensor,
    scale_out: Tensor,
    scale_ls: Tensor,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    """
    RFF VJP for ``∇_Φ ℓ = α vᵀ − W`` without allocating full ``∇_Φ``.

    ``w_phi`` is ``Φ Λ⁻¹``. Only cos/sin halves (width ``D``) are materialized.
    Featurize tensors may be float32 while ``α/v/W`` are float64 (default promote).
    """
    D = num_samples
    inv_sqrt_d = 1.0 / math.sqrt(D)
    compute_dtype = alpha.dtype
    x, randn_weights, proj, z_unscaled, scale_out, scale_ls = _align_rff_vjp_inputs(
        compute_dtype, x, randn_weights, proj, z_unscaled, scale_out, scale_ls
    )
    go = _scale_grad_output(grad_output.to(dtype=compute_dtype), scale_out)
    s_go = scale_out * go
    a = alpha.unsqueeze(-1)

    with _CudaSection("rff_vjp"):
        g_z_cos = s_go * (a * v[..., :D] - w_phi[..., :D])
        g_z_sin = s_go * (a * v[..., D:] - w_phi[..., D:])
        g_proj = (-torch.sin(proj) * inv_sqrt_d) * g_z_cos + (
            torch.cos(proj) * inv_sqrt_d
        ) * g_z_sin
        g_w = x.transpose(-1, -2) @ g_proj
        g_scale_ls = (g_w * randn_weights).sum(dim=-1, keepdim=True)
        g_ls_col = g_scale_ls * scale_ls * (_LOG10 / 2.0)
        if lengthscale.dim() >= 2 and lengthscale.shape[-2] == 1:
            g_lengthscale = g_ls_col.transpose(-1, -2)
        else:
            g_lengthscale = g_ls_col.reshape(lengthscale.shape)

        # ⟨α vᵀ − W, Z⟩ = αᵀ (Z v) − ⟨W, Z⟩  (Z = unscaled features)
        z_v = torch.matmul(z_unscaled, v.unsqueeze(-1)).squeeze(-1)
        if z_unscaled.dim() == 2:
            term1 = (alpha * z_v).sum()
            term2 = (w_phi * z_unscaled).sum()
            go_s = go if go.dim() == 0 else go.reshape(())
            g_s = go_s * (term1 - term2)
            g_outputscale = (g_s * scale_out.reshape(()) * (_LOG10 / 2.0)).reshape(
                outputscale.shape
            )
        else:
            term1_b = (alpha * z_v).sum(dim=-1)
            term2_b = (w_phi * z_unscaled).flatten(start_dim=-2).sum(dim=-1)
            go_b = go if go.dim() == 0 else go
            g_s_b = go_b * (term1_b - term2_b)
            s_b = scale_out.reshape(g_s_b.shape)
            g_outputscale = (g_s_b * s_b * (_LOG10 / 2.0)).reshape(outputscale.shape)

    return g_lengthscale, g_outputscale


class WoodburyDualRFFMLL(torch.autograd.Function):
    """Dual Woodbury MLL with fused RFF featurize VJP (no full ∇_Φ buffer)."""

    @staticmethod
    def forward(
        ctx,
        noise_var: Tensor,
        y_centered: Tensor,
        x: Tensor,
        lengthscale: Tensor,
        outputscale: Tensor,
        randn_weights: Tensor,
        num_samples: Tensor,
        jitter: Tensor,
    ) -> Tensor:
        jitter_f = float(jitter.detach().item())
        D = int(num_samples.detach().item())
        below_floor = noise_var < 1e-12

        with torch.no_grad():
            with _CudaSection("featurize"):
                phi, proj, z_unscaled, scale_out, scale_ls = featurize_rbf_scaled_nograd(
                    x, randn_weights, lengthscale, outputscale, D
                )

        mll, z_lin, chol, v, alpha, noise_b, n, m = _dual_mll_forward_values(
            noise_var, phi, y_centered, jitter_f
        )

        ctx.save_for_backward(
            z_lin,
            chol,
            v,
            alpha,
            noise_b,
            x,
            randn_weights,
            lengthscale,
            outputscale,
            proj,
            z_unscaled,
            scale_out,
            scale_ls,
        )
        ctx.below_floor = below_floor
        ctx.n = n
        ctx.m = m
        ctx.D = D
        ctx.noise_var_shape = tuple(noise_var.shape)
        ctx.y_dtype = y_centered.dtype
        ctx.noise_dtype = noise_var.dtype
        ctx.ls_dtype = lengthscale.dtype
        ctx.os_dtype = outputscale.dtype
        return mll

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            z_lin,
            chol,
            v,
            alpha,
            noise_b,
            x,
            randn_weights,
            lengthscale,
            outputscale,
            proj,
            z_unscaled,
            scale_out,
            scale_ls,
        ) = ctx.saved_tensors
        n, m, D = ctx.n, ctx.m, ctx.D
        needs = ctx.needs_input_grad

        grad_noise = grad_y = None
        grad_ls = grad_os = None

        if needs[0]:
            grad_noise = _grad_noise_from_saved(
                alpha,
                noise_b,
                chol,
                n,
                m,
                grad_output,
                below_floor=ctx.below_floor,
                noise_dtype=ctx.noise_dtype,
                noise_var_shape=ctx.noise_var_shape,
            )
        if needs[1]:
            grad_y = _grad_y_from_saved(alpha, grad_output, ctx.y_dtype)

        if needs[3] or needs[4]:
            w_phi = _phi_lam_inv(z_lin, chol)
            g_ls, g_os = vjp_rff_from_alpha_v_w(
                alpha,
                v,
                w_phi,
                grad_output,
                x,
                randn_weights,
                lengthscale,
                outputscale,
                proj,
                z_unscaled,
                scale_out,
                scale_ls,
                D,
            )
            if needs[3]:
                grad_ls = g_ls.to(dtype=ctx.ls_dtype)
            if needs[4]:
                grad_os = g_os.to(dtype=ctx.os_dtype)

        return grad_noise, grad_y, None, grad_ls, grad_os, None, None, None


def woodbury_dual_rff_mll_apply(
    noise_var: Tensor,
    y_centered: Tensor,
    x: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    randn_weights: Tensor,
    num_samples: int,
    jitter: float = 1e-6,
) -> Tensor:
    return WoodburyDualRFFMLL.apply(
        noise_var,
        y_centered,
        x,
        lengthscale,
        outputscale,
        randn_weights,
        x.new_tensor(float(num_samples)),
        x.new_tensor(float(jitter)),
    )


class WoodburyDualRFFDiagMLL(torch.autograd.Function):
    """
    Dual Woodbury MLL for ``Σ = diag(d) + ΦΦᵀ`` with fused RFF featurize VJP.

    Row-scales to ``Φ̂ = D^{-1/2} Φ``, ``ŷ = D^{-1/2} y``, evaluates unit-noise
    dual MLL on ``(Φ̂, ŷ)``, then subtracts ``½ ∑ log d_i``. Lengthscale /
    outputscale grads use the same fused VJP as :class:`WoodburyDualRFFMLL`
    (no full ``∇_Φ`` buffer). ``d`` is typically NIGP effective noise with
    detached ``∇μ``.
    """

    @staticmethod
    def forward(
        ctx,
        d: Tensor,
        y_centered: Tensor,
        x: Tensor,
        lengthscale: Tensor,
        outputscale: Tensor,
        randn_weights: Tensor,
        num_samples: Tensor,
        jitter: Tensor,
    ) -> Tensor:
        jitter_f = float(jitter.detach().item())
        D = int(num_samples.detach().item())
        below_floor = d < 1e-12

        with torch.no_grad():
            with _CudaSection("featurize"):
                phi, proj, z_unscaled, scale_out, scale_ls = featurize_rbf_scaled_nograd(
                    x, randn_weights, lengthscale, outputscale, D
                )
            d_clamped = d.detach().clamp_min(1e-12)
            inv_sqrt = d_clamped.rsqrt()
            while inv_sqrt.dim() < phi.dim() - 1:
                inv_sqrt = inv_sqrt.unsqueeze(0)
            phi_hat = phi * inv_sqrt.unsqueeze(-1)
            y_hat = y_centered.detach() * inv_sqrt

        ones = phi_hat.new_ones(())
        if phi_hat.dim() == 3:
            ones = phi_hat.new_ones(phi_hat.shape[0])

        mll_tilde, z_lin, chol, v, alpha, _noise_b, n, m = _dual_mll_forward_values(
            ones, phi_hat, y_hat, jitter_f
        )
        log_d_sum = d_clamped.log().sum(dim=-1)
        mll = mll_tilde - 0.5 * log_d_sum

        # Align inv_sqrt / y_hat to factor dtype used by α, v, z_lin.
        inv_sqrt_b = inv_sqrt.to(dtype=z_lin.dtype)
        y_hat_b = y_hat.to(dtype=z_lin.dtype)
        d_b = d_clamped.to(dtype=z_lin.dtype)

        ctx.save_for_backward(
            z_lin,
            chol,
            v,
            alpha,
            inv_sqrt_b,
            y_hat_b,
            d_b,
            x,
            randn_weights,
            lengthscale,
            outputscale,
            proj,
            z_unscaled,
            scale_out,
            scale_ls,
        )
        ctx.below_floor = below_floor
        ctx.n = n
        ctx.m = m
        ctx.D = D
        ctx.d_shape = tuple(d.shape)
        ctx.y_dtype = y_centered.dtype
        ctx.d_dtype = d.dtype
        ctx.ls_dtype = lengthscale.dtype
        ctx.os_dtype = outputscale.dtype
        return mll

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            z_lin,
            chol,
            v,
            alpha,
            inv_sqrt,
            y_hat,
            d_clamped,
            x,
            randn_weights,
            lengthscale,
            outputscale,
            proj,
            z_unscaled,
            scale_out,
            scale_ls,
        ) = ctx.saved_tensors
        n, m, D = ctx.n, ctx.m, ctx.D
        needs = ctx.needs_input_grad

        grad_d = grad_y = None
        grad_ls = grad_os = None

        # ∇_Φ̂ ℓ̃ = α vᵀ − W; ∇_Φ ℓ = inv_sqrt ⊙_rows ∇_Φ̂ ℓ̃
        # ⇒ use fused VJP with α' = inv_sqrt ⊙ α and W' = inv_sqrt ⊙ W.
        w_hat = None
        if needs[0] or needs[3] or needs[4]:
            w_hat = _phi_lam_inv(z_lin, chol)

        if needs[3] or needs[4]:
            alpha_s = alpha * inv_sqrt
            w_s = w_hat * inv_sqrt.unsqueeze(-1)
            g_ls, g_os = vjp_rff_from_alpha_v_w(
                alpha_s,
                v,
                w_s,
                grad_output,
                x,
                randn_weights,
                lengthscale,
                outputscale,
                proj,
                z_unscaled,
                scale_out,
                scale_ls,
                D,
            )
            if needs[3]:
                grad_ls = g_ls.to(dtype=ctx.ls_dtype)
            if needs[4]:
                grad_os = g_os.to(dtype=ctx.os_dtype)

        if needs[1]:
            # ∇_y ℓ = inv_sqrt ⊙ (−α)
            g_y = -alpha * inv_sqrt
            if grad_output.dim() == 0:
                g_y = g_y * grad_output
            else:
                g_y = g_y * grad_output.unsqueeze(-1)
            grad_y = g_y.to(dtype=ctx.y_dtype)

        if needs[0]:
            # ∂ℓ/∂d_i = −½/d_i * (⟨∇_Φ̂_i, Φ̂_i⟩ − α_i ŷ_i + 1)
            phi_v = torch.matmul(z_lin, v.unsqueeze(-1)).squeeze(-1)
            row_inner = alpha * phi_v - (w_hat * z_lin).sum(dim=-1) - alpha * y_hat
            g_d = -0.5 / d_clamped * (row_inner + 1.0)
            if grad_output.dim() == 0:
                g_d = g_d * grad_output
            else:
                g_d = g_d * grad_output
            below = ctx.below_floor
            if below.shape != g_d.shape:
                below = (
                    below.reshape(g_d.shape)
                    if below.numel() == g_d.numel()
                    else below
                )
            g_d = torch.where(below, torch.zeros_like(g_d), g_d)
            g_d = g_d.to(dtype=ctx.d_dtype)
            if g_d.shape != ctx.d_shape:
                g_d = g_d.reshape(ctx.d_shape)
            grad_d = g_d

        return grad_d, grad_y, None, grad_ls, grad_os, None, None, None


def woodbury_dual_rff_diag_mll_apply(
    d: Tensor,
    y_centered: Tensor,
    x: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    randn_weights: Tensor,
    num_samples: int,
    jitter: float = 1e-6,
) -> Tensor:
    """Fused diag-noise dual Woodbury + RFF featurize VJP (NIGP MLL path)."""
    return WoodburyDualRFFDiagMLL.apply(
        d,
        y_centered,
        x,
        lengthscale,
        outputscale,
        randn_weights,
        x.new_tensor(float(num_samples)),
        x.new_tensor(float(jitter)),
    )


# ---------------------------------------------------------------------------
# Multitask ICM diag-noise Woodbury (NIGP) — closed-form grads, no Chol VJP
# ---------------------------------------------------------------------------


def _mt_diag_mll_backward_terms(
    chol: Tensor,
    d_nt: Tensor,
    phi: Tensor,
    r_b: Tensor,
    alpha: Tensor,
    *,
    chunk_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Analytic pieces for ``ℓ = log N(y; 0, diag(d)+ΩΩᵀ)``, ``Ω = Φ ⊗ R_B``.

    Returns ``(diag_sinv, grad_phi_base, grad_r_b_base, v)`` where
    ``∇_Φ ℓ = grad_output * grad_phi_base``, etc., and
    ``∂ℓ/∂d_p = grad_output * (-½)(diag_sinv_p − α_p²)``.
    """
    from .rff_utils import build_icm_joint_features, icm_omega_rmatvec

    n, m = phi.shape[-2], phi.shape[-1]
    t = r_b.shape[-1]
    dtype = chol.dtype
    device = chol.device
    phi_f = phi.to(dtype=dtype)
    r_f = r_b.to(dtype=dtype)
    d_f = d_nt.to(dtype=dtype).clamp_min(1e-12)
    alpha_f = alpha.to(dtype=dtype)
    if alpha_f.dim() > 1:
        alpha_f = alpha_f.reshape(-1)
    a_mt = alpha_f.reshape(n, t)
    inv_d = d_f.reciprocal()

    v = icm_omega_rmatvec(phi_f, r_f, alpha_f)
    v_mat = v.reshape(m, t)
    # α vᵀ contracted through Kronecker → Φ and R_B
    av_phi = (a_mt @ r_f) @ v_mat.transpose(-1, -2)
    av_rb = (phi_f.transpose(-1, -2) @ a_mt).transpose(-1, -2) @ v_mat

    diag_sinv = torch.empty(n, t, device=device, dtype=dtype)
    corr_phi = torch.zeros(n, m, device=device, dtype=dtype)
    corr_rb = torch.zeros(t, t, device=device, dtype=dtype)
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        end = min(start + step, n)
        phi_c = phi_f[start:end]
        c = end - start
        omega_c = build_icm_joint_features(phi_c, r_f)
        solved = torch.cholesky_solve(omega_c.transpose(-1, -2), chol)
        quad = (omega_c * solved.transpose(-1, -2)).sum(dim=-1).view(c, t)
        inv_c = inv_d[start:end]
        diag_sinv[start:end] = inv_c - quad * (inv_c * inv_c)
        # W = Λ^{-1} Ω M^{-1} = (Ω M^{-1}) / d ; solved = M^{-1} Ωᵀ
        w = solved.transpose(-1, -2) * inv_c.reshape(c * t, 1)
        w4 = w.view(c, t, m, t)
        corr_phi[start:end] = (w4 * r_f.view(1, t, 1, t)).sum(dim=(1, 3))
        corr_rb = corr_rb + torch.einsum("ctms,cm->ts", w4, phi_c)

    grad_phi = av_phi - corr_phi
    grad_rb = av_rb - corr_rb
    return diag_sinv, grad_phi, grad_rb, v


class WoodburyMTDiagMLL(torch.autograd.Function):
    """
    Multitask diag-noise Woodbury MLL with closed-form grads (no Chol VJP).

    Forward matches ``woodbury_marginal_log_likelihood_mt_diag_noise`` (dense
    Chol of ``M``). Backward uses matrix-calculus grads w.r.t. ``d``, ``Φ``,
    ``R_B``, and ``y``.
    """

    @staticmethod
    def forward(
        ctx,
        d: Tensor,
        phi: Tensor,
        r_b: Tensor,
        y_centered: Tensor,
        jitter: Tensor,
    ) -> Tensor:
        from .rff_utils import (
            _coerce_mt_diag_noise,
            flatten_multitask_targets,
            woodbury_factor_mt_diag_noise,
            woodbury_solve_mt_diag_noise_from_chol,
        )

        jitter_f = float(jitter.detach().item())
        n = phi.shape[-2]
        t = r_b.shape[-2]
        d_nt = _coerce_mt_diag_noise(d, n, t)
        y_shape = tuple(y_centered.shape)
        y_flat = flatten_multitask_targets(
            y_centered if y_centered.dim() > 1 else y_centered.reshape(n, t)
        )
        if y_flat.numel() != n * t:
            raise ValueError(
                f"y_centered has {y_flat.numel()} elems, expected n*T={n * t}"
            )

        below_floor = d_nt < 1e-12
        with torch.no_grad():
            chol, d_c = woodbury_factor_mt_diag_noise(
                d_nt.detach(), phi.detach(), r_b.detach(), jitter=jitter_f
            )
            y_f = y_flat.detach().to(dtype=chol.dtype)
            alpha = woodbury_solve_mt_diag_noise_from_chol(
                d_c, phi.detach(), r_b.detach(), chol, y_f
            )
            if alpha.dim() > 1:
                alpha = alpha.reshape(-1)
            quad = (y_f * alpha).sum()
            log_det_lam = d_c.log().sum()
            log_det_m = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum()
            n_t = float(y_f.numel())
            const = -0.5 * n_t * math.log(2.0 * math.pi)
            mll = const - 0.5 * quad - 0.5 * (log_det_lam + log_det_m)

        ctx.save_for_backward(chol, d_c, phi.detach(), r_b.detach(), alpha, y_f)
        ctx.below_floor = below_floor
        ctx.n = n
        ctx.t = t
        ctx.d_shape = tuple(d.shape)
        ctx.y_shape = y_shape
        ctx.d_dtype = d.dtype
        ctx.phi_dtype = phi.dtype
        ctx.rb_dtype = r_b.dtype
        ctx.y_dtype = y_centered.dtype
        return mll.to(dtype=y_centered.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        chol, d_c, phi, r_b, alpha, y_f = ctx.saved_tensors
        n, t = ctx.n, ctx.t
        needs = ctx.needs_input_grad
        grad_d = grad_phi = grad_rb = grad_y = None

        if not any(needs[:4]):
            return None, None, None, None, None

        diag_sinv, g_phi, g_rb, _v = _mt_diag_mll_backward_terms(
            chol, d_c, phi, r_b, alpha
        )
        scale = grad_output.reshape(())

        if needs[0]:
            a_mt = alpha.reshape(n, t)
            g_d = -0.5 * (diag_sinv - a_mt * a_mt) * scale
            below = ctx.below_floor
            if below.shape != g_d.shape:
                below = below.reshape(g_d.shape)
            g_d = torch.where(below, torch.zeros_like(g_d), g_d)
            g_d = g_d.to(dtype=ctx.d_dtype)
            if g_d.shape != ctx.d_shape:
                g_d = g_d.reshape(ctx.d_shape)
            grad_d = g_d

        if needs[1]:
            grad_phi = (g_phi * scale).to(dtype=ctx.phi_dtype)

        if needs[2]:
            grad_rb = (g_rb * scale).to(dtype=ctx.rb_dtype)

        if needs[3]:
            g_y = (-alpha * scale).to(dtype=ctx.y_dtype)
            grad_y = g_y.reshape(ctx.y_shape)

        return grad_d, grad_phi, grad_rb, grad_y, None


def woodbury_mt_diag_mll_apply(
    d: Tensor,
    phi: Tensor,
    r_b: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """MT ICM diag-noise Woodbury MLL with closed-form grads (NIGP hot path)."""
    return WoodburyMTDiagMLL.apply(
        d,
        phi,
        r_b,
        y_centered,
        phi.new_tensor(float(jitter)),
    )
