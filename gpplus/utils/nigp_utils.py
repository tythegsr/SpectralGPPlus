"""Classic McHutchon–Rasmussen NIGP helpers for RFF / Woodbury and exact GPs.

Independent input noise ``Σ_x = diag(σ_{x,1}^2, …, σ_{x,D}^2)`` yields
effective observation noise

    d_i = σ_y² + Σ_d σ_{x,d}² (∂_{x_d} μ(x_i))²

``μ`` is the **homoskedastic** posterior mean (input noise ignored when forming
dual weights / ``α``), matching the paper's two-step linearization that avoids
the circular dependence of ``μ`` on its own slope. ``∇μ`` is detached when
forming ``d``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..constraints import SoftClamp
from .rff_utils import woodbury_factor_dual

# Default SoftClamp on log10(σ_x): σ_x ∈ (1e-6, 1).
INPUT_NOISE_SOFTCLAMP_LOWER = -6.0
INPUT_NOISE_SOFTCLAMP_UPPER = 0.0
# Constructor placeholder (overwritten by parameter_initializer Uniform(-4, -1)).
INPUT_NOISE_PLACEHOLDER_STD = 10.0 ** (-2.5)


def nigp_correction_enabled(model: nn.Module) -> bool:
    """
    True when the NIGP input-noise term should enter the MLL / predictive noise.

    During :class:`~gpplus.training.callbacks.NIGPInputNoiseFreezeCallback`
    freeze windows this is set ``False`` so training matches a standard GP
    (paper step 1) before slopes are used with learnable ``σ_x``.
    """
    if not bool(getattr(model, "nigp", False)):
        return False
    if not hasattr(model, "raw_input_noise"):
        return False
    return bool(getattr(model, "nigp_correction_enabled", True))


def effective_noise_variance(
    noise_y: Tensor,
    input_noise_var: Tensor,
    grad_mu: Tensor,
) -> Tensor:
    """
    Per-point NIGP noise ``d_i = σ_y² + Σ_d σ_{x,d}² (∇μ_{i,d})²``.

    Parameters
    ----------
    noise_y :
        Homoskedastic output noise ``()`` / ``(B,)`` / ``(1,)``.
    input_noise_var :
        Per-dimension input noise variances ``(D,)`` or ``(B, D)``.
    grad_mu :
        Detached posterior-mean Jacobian ``(n, D)`` or ``(B, n, D)``.

    Returns
    -------
    d : ``(n,)`` or ``(B, n)``
    """
    g2 = grad_mu * grad_mu
    # (..., n, D) @ (..., D) -> (..., n)
    if input_noise_var.dim() == 1:
        input_term = (g2 * input_noise_var).sum(dim=-1)
    else:
        # (B, D) with (B, n, D) or (n, D)
        while input_noise_var.dim() < g2.dim():
            input_noise_var = input_noise_var.unsqueeze(-2)
        input_term = (g2 * input_noise_var).sum(dim=-1)

    noise = noise_y.clamp_min(1e-12)
    while noise.dim() < input_term.dim():
        noise = noise.unsqueeze(-1)
    return noise + input_term


def _posterior_feature_weights_homoskedastic(
    noise_y: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """Frozen dual weights ``w = Λ⁻¹ Φᵀ y`` for ``Σ = σ² I + ΦΦᵀ``."""
    chol, _noise, z_lin = woodbury_factor_dual(noise_y, z_train, jitter=jitter)
    y = y_centered.to(dtype=chol.dtype)
    if y.dim() == z_lin.dim() - 1:
        y_col = y.unsqueeze(-1)
    else:
        y_col = y
    phi_ty = z_lin.transpose(-1, -2) @ y_col
    return torch.cholesky_solve(phi_ty, chol).squeeze(-1)


def _same_feature_points(a: Tensor, b: Tensor) -> bool:
    """True when ``a`` and ``b`` are the same storage (share one Φ)."""
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.dtype == b.dtype


def _rff_logscale_params(model: nn.Module):
    """
    Return ``(lengthscale, outputscale, randn_weights, num_samples)`` for
    ``LogScaleKernel(RFFKernel)``, else ``None``.
    """
    from ..kernels import LogScaleKernel, RFFKernel

    covar = getattr(model, "covar_module", None)
    if not isinstance(covar, LogScaleKernel):
        return None
    base = covar.base_kernel
    if not isinstance(base, RFFKernel) or not hasattr(base, "randn_weights"):
        return None
    return base.lengthscale, covar.outputscale, base.randn_weights, int(base.num_samples)


def _mean_independent_of_x(mean_module: nn.Module) -> bool:
    from gpytorch.means import ConstantMean, ZeroMean

    return isinstance(mean_module, (ConstantMean, ZeroMean))


def _posterior_mean_grad_wrt_x_autograd(
    model: nn.Module,
    x: Tensor,
    y_centered: Tensor,
    noise_for_mu: Tensor,
    *,
    train_x: Tensor | None = None,
    jitter: float = 1e-6,
) -> Tensor:
    """Reference path: featurize-with-graph + ``torch.autograd.grad``."""
    weight_x = (train_x if train_x is not None else x).detach()
    noise_mu = noise_for_mu.detach()
    with torch.no_grad():
        z_det = model.scaled_features(weight_x)
        w = _posterior_feature_weights_homoskedastic(
            noise_mu, z_det, y_centered.detach(), jitter=jitter
        )

    x_req = x.detach().requires_grad_(True)
    z = model.scaled_features(x_req)
    mean = model.mean_module(x_req)
    if mean.dim() > 1 and mean.shape[0] == 1 and z.dim() == 2:
        mean = mean.squeeze(0)
    w_b = w.detach().to(dtype=z.dtype)
    if z.dim() == 2:
        if w_b.dim() == 2:
            w_b = w_b[0]
        f_rff = (z * w_b).sum(dim=-1)
    else:
        if w_b.dim() == 1:
            f_rff = (z * w_b).sum(dim=-1)
        else:
            f_rff = (z * w_b.unsqueeze(-2)).sum(dim=-1)
    mu = mean + f_rff
    grad = torch.autograd.grad(
        mu.sum(),
        x_req,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad.detach()


def posterior_mean_grad_wrt_x(
    model: nn.Module,
    x: Tensor,
    y_centered: Tensor,
    noise_for_mu: Tensor,
    *,
    train_x: Tensor | None = None,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Detached Jacobian ``∇_x μ`` of the Woodbury posterior mean at ``x``.

    Uses **homoskedastic** Woodbury with ``noise_for_mu`` (``σ_y`` only — never
    the NIGP ``d``) on ``train_x`` (defaults to ``x``) to form frozen feature
    weights ``w``, then

        μ(x) = m(x) + φ(x)ᵀ w

    and differentiates only through ``m`` and ``φ``. This is the paper's escape
    from the intractable self-consistent slope equation.

    For ``LogScaleKernel(RFFKernel)`` + x-independent mean, uses a single no-grad
    featurize when ``train_x`` is omitted or aliases ``x``, plus an analytic RFF
    Jacobian. When ``train_x`` is a distinct tensor, featurizes train for ``w`` and
    eval points for ``∇μ`` separately. Other models fall back to autograd.
    """
    from .woodbury_mll_autograd import (
        featurize_rbf_scaled_omega,
        rff_grad_mu_from_proj,
    )

    x_det = x.detach()
    weight_x = (train_x if train_x is not None else x).detach()
    same_points = train_x is None or _same_feature_points(weight_x, x_det)

    params = _rff_logscale_params(model)
    mean_module = getattr(model, "mean_module", None)
    if (
        params is None
        or mean_module is None
        or not _mean_independent_of_x(mean_module)
    ):
        return _posterior_mean_grad_wrt_x_autograd(
            model,
            x,
            y_centered,
            noise_for_mu,
            train_x=train_x,
            jitter=jitter,
        )

    lengthscale, outputscale, randn_weights, num_samples = params
    noise_mu = noise_for_mu.detach()
    y_c = y_centered.detach()

    with torch.no_grad():
        if same_points:
            phi, proj, omega, scale_out = featurize_rbf_scaled_omega(
                x_det,
                randn_weights,
                lengthscale,
                outputscale,
                num_samples,
            )
            w = _posterior_feature_weights_homoskedastic(
                noise_mu, phi, y_c, jitter=jitter
            )
            grad = rff_grad_mu_from_proj(
                proj, omega, scale_out, w, num_samples
            )
        else:
            phi_train, _, _, _ = featurize_rbf_scaled_omega(
                weight_x,
                randn_weights,
                lengthscale,
                outputscale,
                num_samples,
            )
            w = _posterior_feature_weights_homoskedastic(
                noise_mu, phi_train, y_c, jitter=jitter
            )
            _phi_x, proj, omega, scale_out = featurize_rbf_scaled_omega(
                x_det,
                randn_weights,
                lengthscale,
                outputscale,
                num_samples,
            )
            grad = rff_grad_mu_from_proj(
                proj, omega, scale_out, w, num_samples
            )
    return grad.to(dtype=x.dtype).detach()


def exact_posterior_mean_grad_wrt_x(
    model: nn.Module,
    x: Tensor,
    noise_for_mu: Tensor,
    *,
    train_x: Tensor | None = None,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Detached Jacobian ``∇_x μ`` of the exact-GP posterior mean at ``x``.

    Uses **homoskedastic** ``K + (σ² + jitter) I`` on ``train_x`` (defaults to
    ``model.train_inputs[0]``) to form frozen dual weights
    ``α = (K+σ²I)⁻¹ (y − m)``, then

        μ(x) = m(x) + k(x, X) α

    and differentiates only through ``m`` and ``k`` (classic NIGP linearization).
    """
    weight_x = train_x if train_x is not None else model.train_inputs[0]
    if isinstance(weight_x, (tuple, list)):
        weight_x = weight_x[0]
    weight_x = weight_x.detach()
    if weight_x.dim() == 3 and weight_x.shape[0] == 1:
        weight_x = weight_x.squeeze(0)

    train_y = model.train_targets
    if train_y.dim() == 2 and train_y.shape[0] == 1:
        train_y = train_y.squeeze(0)
    train_y = train_y.detach()

    with torch.no_grad():
        k_xx = model.covar_module(weight_x).to_dense()
        n = int(k_xx.shape[-1])
        eye = torch.eye(n, dtype=k_xx.dtype, device=k_xx.device)
        noise = noise_for_mu.detach().reshape(-1)
        if noise.numel() == 1:
            k_noisy = k_xx + eye * (float(noise.reshape(()).item()) + jitter)
        else:
            k_noisy = k_xx + torch.diag(noise.to(dtype=k_xx.dtype) + jitter)
        chol = torch.linalg.cholesky(k_noisy)
        mean_train = model.mean_module(weight_x)
        if mean_train.dim() > 1 and mean_train.shape[0] == 1:
            mean_train = mean_train.squeeze(0)
        mean_train = mean_train.reshape(train_y.shape)
        alpha = torch.cholesky_solve(
            (train_y - mean_train).unsqueeze(-1), chol
        ).squeeze(-1)

    x_req = x.detach().requires_grad_(True)
    mean_x = model.mean_module(x_req)
    k_star = model.covar_module(x_req, weight_x).to_dense()
    if mean_x.dim() > 1 and mean_x.shape[0] == 1:
        mean_x = mean_x.squeeze(0)
    if mean_x.shape != k_star.shape[:-1]:
        mean_x = mean_x.reshape(*k_star.shape[:-1])
    mu = mean_x + (k_star @ alpha.detach().to(dtype=k_star.dtype))
    grad = torch.autograd.grad(
        mu.sum(),
        x_req,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad.detach()


def effective_noise_variance_mt(
    task_noises: Tensor,
    input_noise_var: Tensor,
    grad_mu: Tensor,
) -> Tensor:
    """
    Per-(i,t) NIGP noise ``d_{i,t} = σ_t² + Σ_d σ_{x,d}² (∇μ_{i,t,d})²``.

    Parameters
    ----------
    task_noises :
        Per-task output noise ``(T,)``.
    input_noise_var :
        Per-dimension input noise variances ``(D,)``.
    grad_mu :
        Detached posterior-mean Jacobian ``(n, T, D)``.

    Returns
    -------
    d : ``(n, T)``
    """
    if grad_mu.dim() != 3:
        raise ValueError(f"grad_mu must be (n, T, D), got {tuple(grad_mu.shape)}")
    g2 = grad_mu * grad_mu
    if input_noise_var.dim() != 1:
        input_noise_var = input_noise_var.reshape(-1)
    input_term = (g2 * input_noise_var).sum(dim=-1)
    noise = task_noises.clamp_min(1e-12).reshape(1, -1)
    return noise + input_term


def _posterior_feature_weights_homoskedastic_mt(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Frozen dual feature weights ``v = Ωᵀ Σ⁻¹ y`` for homoskedastic MT Woodbury.

    ``μ_* = Ω_* v`` with ``Ω = Φ ⊗ R_B``.
    """
    from .rff_utils import (
        icm_omega_rmatvec,
        woodbury_factor_mt,
        woodbury_solve_mt_from_factor,
    )

    factor, noise = woodbury_factor_mt(task_noises, phi, r_b, jitter=jitter)
    alpha = woodbury_solve_mt_from_factor(noise, phi, r_b, n, factor, y_centered)
    return icm_omega_rmatvec(
        phi.to(dtype=alpha.dtype),
        r_b.to(dtype=alpha.dtype),
        alpha,
    )


def posterior_mean_grad_wrt_x_mt(
    model: nn.Module,
    x: Tensor,
    y_centered: Tensor,
    task_noises: Tensor,
    *,
    train_x: Tensor | None = None,
    jitter: float = 1e-6,
    feature_weights: Tensor | None = None,
) -> Tensor:
    """
    Detached Jacobian ``∇_x μ`` of the multitask Woodbury posterior mean.

    Returns ``(n, T, D)``. Uses **homoskedastic** MT Woodbury (``task_noises``
    only — never NIGP ``d``) to form frozen ``v = Ωᵀ Σ⁻¹ y``, then

        μ(x) = m(x) + unflatten(Ω(x) v)

    and differentiates only through ``m`` and spatial features.

    When ``feature_weights`` is provided it must be the frozen homoskedastic ``v``
    for ``train_x`` (or ``x`` when ``train_x`` is omitted), skipping the train solve.
    For LogScaleKernel+RFF with an x-independent mean, uses an analytic RFF
    Jacobian; otherwise falls back to per-task autograd.
    """
    from .rff_utils import (
        icm_omega_matvec,
        unflatten_multitask_targets,
    )
    from .woodbury_mll_autograd import (
        featurize_rbf_scaled_omega,
        rff_grad_mu_from_proj,
    )

    weight_x = (train_x if train_x is not None else x).detach()
    noise_mu = task_noises.detach()
    with torch.no_grad():
        r_b = model.task_psd_factor().detach()
        if feature_weights is None:
            phi_det = model.scaled_spatial_features(weight_x)
            n_train = weight_x.shape[0]
            v = _posterior_feature_weights_homoskedastic_mt(
                noise_mu,
                phi_det,
                r_b,
                n_train,
                y_centered.detach(),
                jitter=jitter,
            )
        else:
            v = feature_weights.detach()

    mean_module = getattr(model, "mean_module", None)
    params = _rff_logscale_params(model)
    if (
        params is not None
        and mean_module is not None
        and _mean_independent_of_x(mean_module)
    ):
        lengthscale, outputscale, randn_weights, num_samples = params
        x_det = x.detach()
        with torch.no_grad():
            _phi, proj, omega, scale_out = featurize_rbf_scaled_omega(
                x_det,
                randn_weights,
                lengthscale,
                outputscale,
                num_samples,
            )
            # μ_t = φᵀ w_t with w = V @ R_B^T, V = v.view(m, T)
            m = int(_phi.shape[-1])
            t = int(r_b.shape[-1])
            v_mat = v.detach().to(dtype=proj.dtype).reshape(m, t)
            w_mt = v_mat @ r_b.to(dtype=proj.dtype).transpose(-1, -2)  # (m, T)
            grads = []
            for task in range(t):
                grads.append(
                    rff_grad_mu_from_proj(
                        proj, omega, scale_out, w_mt[:, task], num_samples
                    )
                )
            return torch.stack(grads, dim=1).to(dtype=x.dtype).detach()

    x_req = x.detach().requires_grad_(True)
    phi = model.scaled_spatial_features(x_req)
    mean = model.mean_module(x_req)
    num_tasks = int(mean.shape[-1])
    v_b = v.detach().to(dtype=phi.dtype)
    r_b_b = r_b.to(dtype=phi.dtype)
    f_flat = icm_omega_matvec(phi, r_b_b, v_b)
    f_mt = unflatten_multitask_targets(f_flat, num_tasks)
    mu = mean + f_mt
    grads = []
    for t in range(num_tasks):
        g_t = torch.autograd.grad(
            mu[:, t].sum(),
            x_req,
            create_graph=False,
            retain_graph=(t < num_tasks - 1),
            allow_unused=False,
        )[0]
        if g_t is None:
            g_t = torch.zeros_like(x)
        grads.append(g_t.detach())
    return torch.stack(grads, dim=1)


def input_noise_softclamp(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> SoftClamp:
    """SoftClamp on log10(σ_x) with bounds ``[-6, 0]``."""
    c = SoftClamp(
        lower_bound=INPUT_NOISE_SOFTCLAMP_LOWER,
        upper_bound=INPUT_NOISE_SOFTCLAMP_UPPER,
        margin=1e-2,
    )
    if dtype is not None or device is not None:
        c = c.to(dtype=dtype or c.lower_bound.dtype, device=device or c.lower_bound.device)
    return c


def raw_input_noise_init_value(
    input_noise_init: float | None = None,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """
    Unconstrained ``raw`` for log10 SoftClamp input noise.

    ``σ_x = 10^{SoftClamp(raw)}``. ``input_noise_init`` is the physical std in
    ``(1e-6, 1)``; ``None`` uses the mid-range constructor placeholder
    ``10^{-2.5}`` (typically overwritten by the parameter initializer).
    """
    std = float(INPUT_NOISE_PLACEHOLDER_STD if input_noise_init is None else input_noise_init)
    lo = 10.0 ** INPUT_NOISE_SOFTCLAMP_LOWER
    hi = 10.0 ** INPUT_NOISE_SOFTCLAMP_UPPER
    if not (lo < std < hi):
        raise ValueError(
            f"input_noise_init must be in ({lo:g}, {hi:g}) for SoftClamp "
            f"[{INPUT_NOISE_SOFTCLAMP_LOWER}, {INPUT_NOISE_SOFTCLAMP_UPPER}], got {std}."
        )
    constraint = input_noise_softclamp(dtype=dtype, device=device)
    log10_std = torch.tensor(std, dtype=dtype, device=device).log10()
    return constraint.inverse_transform(log10_std)


__all__ = [
    "INPUT_NOISE_SOFTCLAMP_LOWER",
    "INPUT_NOISE_SOFTCLAMP_UPPER",
    "INPUT_NOISE_PLACEHOLDER_STD",
    "nigp_correction_enabled",
    "effective_noise_variance",
    "effective_noise_variance_mt",
    "posterior_mean_grad_wrt_x",
    "posterior_mean_grad_wrt_x_mt",
    "exact_posterior_mean_grad_wrt_x",
    "input_noise_softclamp",
    "raw_input_noise_init_value",
]
