"""
Metrics computed on training data for monitoring (e.g. NIS, LOO, Kernel Flows).
Used when log_nis / log_loo / log_kf / log_residual_mse are enabled in the trainer.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import gpytorch
import torch

from ..utils.metrics_functions import compute_nis

# ============================================================
# KF helpers
# ============================================================

def _get_metric_jitter(model, cholesky_jitter: Optional[float]) -> float:
    if cholesky_jitter is not None:
        return float(cholesky_jitter)
    return float(getattr(model, "cholesky_jitter", 1e-6))


def _make_generator(device: torch.device, seed: Optional[int]) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def _safe_mean(vals: List[float]) -> float:
    vals = [float(v) for v in vals if torch.isfinite(torch.tensor(v))]
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


# UPDATED
def _symmetrize_dense(K: torch.Tensor) -> torch.Tensor:
    return 0.5 * (K + K.transpose(-1, -2))


# UPDATED
def _quadratic_form_from_operator(
    K_op,
    r: torch.Tensor,
    *,
    jitter: float,
    max_tries: int = 6,
) -> float:
    """
    Compute r^T K^{-1} r using a Cholesky solve.

    Notes
    -----
    - K_op is expected to be a non-batched LinearOperator-like kernel object.
    - We add only numerical jitter here, not observation noise.
    - Falls back to dense Cholesky if the lazy path fails.
    - Uses adaptive jitter escalation for numerical robustness.
    """
    # First try lazy/operator route with escalating jitter
    for k in range(max_tries):
        jit_k = float(jitter * (10 ** k))
        try:
            K_j = K_op.add_jitter(jit_k)
            chol = K_j.cholesky()
            alpha = chol.solve(r.unsqueeze(-1)).squeeze(-1)
            val = (r * alpha).sum()
            val_f = float(val.item())
            if torch.isfinite(torch.tensor(val_f, device=r.device)):
                return val_f
        except Exception:
            pass

    # Dense fallback with escalating jitter
    try:
        K_dense = K_op.to_dense()
        K_dense = _symmetrize_dense(K_dense)  # UPDATED
        eye = torch.eye(K_dense.size(-1), device=K_dense.device, dtype=K_dense.dtype)

        for k in range(max_tries):
            jit_k = float(jitter * (10 ** k))
            try:
                K_try = K_dense + jit_k * eye
                L = torch.linalg.cholesky(K_try)
                alpha = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
                val = (r * alpha).sum()
                val_f = float(val.item())
                if torch.isfinite(torch.tensor(val_f, device=r.device)):
                    return val_f
            except Exception:
                continue
    except Exception:
        pass

    return float("nan")  # UPDATED


def compute_kf_metric(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    Nf: int,
    cholesky_jitter: Optional[float] = None,
    seed: Optional[int] = None,
    likelihood=None,  # kept for API compatibility; intentionally unused in prior-kernel KF
    n_subsamples: int = 10,
    subset_rates: Tuple[float, ...] = (0.50,),
    max_n: int = 2000,
    return_details: bool = False,
) -> float | Dict[str, float]:
    """
    Kernel Flows (KF) metric for a standard single-output GP.

    This implementation follows the updated KF logic:
      - use the PRIOR kernel matrix from model.covar_module(...)
      - build subset kernels from subset inputs directly
      - use Cholesky solves
      - average over multiple independent full/subset draws
      - support multiple subset rates (default 25%, 50%, 75%)

    For each random draw:
      1) sample a fixed-size "full" KF set of size Nf from the training set
      2) for each subset rate r in subset_rates, sample Ns = round(r * Nf) from those Nf points
      3) compute:
             rho = 1 - (v^T K_sub^{-1} v) / (u^T K_full^{-1} u)
         where:
             u uses r_full = y_full - m(X_full)
             v uses r_sub  = y_sub  - m(X_sub)
      4) average across draws, then average across subset rates

    Parameters
    ----------
    model : GP model
        Expected to have mean_module and covar_module.
    train_x : torch.Tensor
        Training inputs of shape (N, D).
    train_y : torch.Tensor
        Training targets of shape (N,) or (N, 1).
    Nf : int
        Fixed full-set size used inside the KF computation.
    cholesky_jitter : Optional[float]
        Numerical jitter for Cholesky solves.
    seed : Optional[int]
        Random seed for subset sampling.
    likelihood : optional
        Retained only for API compatibility. Not used here because KF is computed
        from the prior kernel, not posterior/noise-augmented covariance.
    n_subsamples : int
        Number of independent random KF draws to average.
    subset_rates : tuple of float
        Subset fractions to evaluate within each full set.
    max_n : int
        Safety cap to avoid very large KF computations.
    return_details : bool
        If True, returns a dict containing per-rate KFs and the averaged KF.
        Otherwise returns the averaged KF as a float.

    Returns
    -------
    float or Dict[str, float]
        By default returns the averaged KF across requested subset rates.
        With return_details=True returns:
            {
              "KF": ...,
              "KF_25pct": ...,
              "KF_50pct": ...,
              "KF_75pct": ...,
            }
    """
    try:
        jitter = _get_metric_jitter(model, cholesky_jitter)
        N_total = train_x.size(0)

        base_out: Dict[str, float] = {"KF": float("nan")}
        for rate in subset_rates:
            base_out[f"KF_{int(round(100 * rate))}pct"] = float("nan")

        if N_total < 2:
            return base_out if return_details else float("nan")
        if N_total > max_n:
            return base_out if return_details else float("nan")

        Nf = min(int(Nf), N_total)
        if Nf < 2:
            return base_out if return_details else float("nan")

        # UPDATED: work in float64 for better numerical stability
        train_x = train_x.to(dtype=torch.float64)
        train_y = train_y.to(dtype=torch.float64)

        # Build subset sizes for each requested rate.
        rate_to_ns: Dict[str, int] = {}
        for rate in subset_rates:
            Ns = int(round(float(rate) * Nf))
            Ns = min(max(Ns, 1), Nf - 1)
            label = f"{int(round(100 * rate))}pct"
            rate_to_ns[label] = Ns

        gen = _make_generator(train_x.device, seed)
        y_flat = train_y.view(-1)

        rho_draws_by_rate: Dict[str, List[float]] = {k: [] for k in rate_to_ns.keys()}

        # UPDATED
        eps_denom = 1e-12
        tol = 1e-10

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            for _ in range(int(n_subsamples)):
                full_inds = torch.randperm(N_total, device=train_x.device, generator=gen)[:Nf]
                x_full = train_x[full_inds]
                y_full = y_flat[full_inds]

                mean_full = model.mean_module(x_full).view(-1).to(dtype=torch.float64)
                r_full = y_full - mean_full

                # IMPORTANT: use prior kernel on the sampled full set only
                K_full = model.covar_module(x_full)
                u_sq = _quadratic_form_from_operator(K_full, r_full, jitter=jitter)

                # UPDATED: reject tiny / invalid denominators
                if (not torch.isfinite(torch.tensor(u_sq, device=train_x.device))) or (u_sq <= eps_denom):
                    continue

                for label, Ns in rate_to_ns.items():
                    sub_inds_local = torch.randperm(Nf, device=train_x.device, generator=gen)[:Ns]
                    x_sub = x_full[sub_inds_local]
                    y_sub = y_full[sub_inds_local]

                    mean_sub = model.mean_module(x_sub).view(-1).to(dtype=torch.float64)
                    r_sub = y_sub - mean_sub

                    # IMPORTANT: build subset kernel directly from subset inputs (not slice of full K)
                    K_sub = model.covar_module(x_sub)
                    v_sq = _quadratic_form_from_operator(K_sub, r_sub, jitter=jitter)

                    if not torch.isfinite(torch.tensor(v_sq, device=train_x.device)):
                        continue
                    if v_sq < -tol:  # UPDATED
                        continue

                    # UPDATED: compute bounded rho
                    ratio = v_sq / max(u_sq, eps_denom)
                    rho = 1.0 - ratio

                    # Tiny overshoots from floating point are OK; big ones get clipped.
                    if rho < 0.0 and rho > -tol:
                        rho = 0.0
                    elif rho > 1.0 and rho < 1.0 + tol:
                        rho = 1.0
                    else:
                        rho = float(min(max(rho, 0.0), 1.0))  # UPDATED hard clamp

                    if torch.isfinite(torch.tensor(rho, device=train_x.device)):
                        rho_draws_by_rate[label].append(float(rho))

        out = {}
        per_rate_vals = []
        for label in rate_to_ns.keys():
            val = _safe_mean(rho_draws_by_rate[label])
            out[f"KF_{label}"] = val
            if torch.isfinite(torch.tensor(val)):
                per_rate_vals.append(val)

        out["KF"] = _safe_mean(per_rate_vals)

        if return_details:
            return out
        return out["KF"]

    except Exception:
        if return_details:
            base_out: Dict[str, float] = {"KF": float("nan")}
            for rate in subset_rates:
                base_out[f"KF_{int(round(100 * rate))}pct"] = float("nan")
            return base_out
        return float("nan")

# def compute_kf_metric(
#     model,
#     train_x: torch.Tensor,
#     train_y: torch.Tensor,
#     *,
#     Nf: int,
#     cholesky_jitter: Optional[float] = None,
#     seed: Optional[int] = None,
#     likelihood=None,
# ) -> float:
#     """
#     Kernel Flows (KF) metric ρ_t = 1 - (||v||_K^2 / ||u||_K^2) per Algorithm 1 (line 15).

#     - ||u||_K^2 = r^T K_θ^{-1} r  (full residuals, full kernel + noise)
#     - ||v||_K^2 = r_sub^T K_sub^{-1} r_sub  (subset residuals, subset kernel + noise)
#     - r = y - m(X; β), and K = k(X,X) + σ_n^2 I.

#     Used to monitor geometric stability and detect overfitting. Higher ρ can indicate
#     the subset approximates the full fit well.

#     Args:
#         model: GP model (ExactGP or similar) with train data already set.
#         train_x: Training inputs (N, D).
#         train_y: Training targets (N,) or (N, 1).
#         Nf: Subset size for KF (batch size for KF).
#         cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.
#         seed: Optional seed for subset sampling (for reproducibility).
#         likelihood: Optional; if provided, use likelihood(prior) so K includes noise (recommended).

#     Returns:
#         ρ_t in [0, 1] or float('nan') on failure.
#     """
#     try:
#         import linear_operator

#         jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
#         N = train_x.size(0)
#         if Nf >= N or Nf < 1:
#             return float("nan")
#         if N > 2000:
#             return float("nan")

#         # Noise variance for K = prior_cov + noise_var*I
#         noise_var = 1e-6
#         if likelihood is not None and hasattr(likelihood, "noise"):
#             try:
#                 n = likelihood.noise
#                 noise_var = n.item() if n.numel() == 1 else n.flatten()[0].item()
#                 noise_var = max(noise_var, 1e-12)
#             except Exception:
#                 pass

#         with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
#             try:
#                 linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
#             except Exception:
#                 pass

#             # Single forward on full training data (ExactGP only allows this)
#             prior = model(train_x)
#             r = (train_y - prior.mean).view(-1)
#             K_prior = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
#             if K_prior is None:
#                 return float("nan")

#             # Dense path: build K = prior + noise*I once, then full solve and submatrix solve
#             K_prior_dense = K_prior.to_dense()
#             K_dense = K_prior_dense + noise_var * torch.eye(
#                 N, device=K_prior_dense.device, dtype=K_prior_dense.dtype
#             )
#             K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
#             u_sq = (r * K_inv_r).sum().item()
#             if u_sq <= 0 or not torch.isfinite(torch.tensor(u_sq)):
#                 return float("nan")

#             # Subset: use principal submatrix of K_dense (no second model call)
#             gen = torch.Generator(device=train_x.device)
#             if seed is not None:
#                 gen.manual_seed(seed)
#             inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]
#             r_sub = r[inds]
#             K_sub = K_dense[inds][:, inds]
#             K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
#             v_sq = (r_sub * K_sub_inv_r).sum().item()
#             if not torch.isfinite(torch.tensor(v_sq)):
#                 return float("nan")

#             rho = 1.0 - (v_sq / u_sq)
#             return float(rho)
#     except Exception:
#         return float("nan")


# def compute_kf_metric_tensor(
#     model,
#     train_x: torch.Tensor,
#     train_y: torch.Tensor,
#     *,
#     Nf: int,
#     cholesky_jitter: Optional[float] = None,
#     seed: Optional[int] = None,
#     likelihood=None,
# ) -> torch.Tensor:
#     """
#     Differentiable Kernel Flows metric ρ_t = 1 - (||v||_K^2 / ||u||_K^2) as a 0-dim tensor.
#     Same formula as compute_kf_metric but keeps the computation in the graph for use as a loss.
#     Use loss = -rho to maximize KF, or loss = nll + rho / nll - rho for composites.

#     Returns:
#         0-dim tensor ρ_t on the same device as train_x. Returns zeros (no gradient) if N>2000 or Nf invalid.
#     """
#     jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
#     N = train_x.size(0)
#     if Nf >= N or Nf < 1 or N > 2000:
#         return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
#     try:
#         with gpytorch.settings.cholesky_jitter(jitter):
#             prior = model(train_x)
#             r = (train_y - prior.mean).view(-1)
#             K_prior = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
#             if K_prior is None:
#                 return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
#             noise_var = torch.tensor(1e-6, device=train_x.device, dtype=train_x.dtype)
#             if likelihood is not None and hasattr(likelihood, "noise"):
#                 n = likelihood.noise
#                 noise_var = n.clamp(min=1e-12) if n.numel() == 1 else n.flatten()[0].clamp(min=1e-12)
#             K_prior_dense = K_prior.to_dense()
#             K_dense = K_prior_dense + noise_var * torch.eye(
#                 N, device=K_prior_dense.device, dtype=K_prior_dense.dtype
#             )
#             K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
#             u_sq = (r * K_inv_r).sum()
#             u_sq_safe = u_sq.clamp(min=1e-12)
#             gen = torch.Generator(device=train_x.device)
#             if seed is not None:
#                 gen.manual_seed(seed)
#             inds = torch.randperm(N, device=train_x.device, generator=gen)[:Nf]
#             r_sub = r[inds]
#             K_sub = K_dense[inds][:, inds]
#             K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
#             v_sq = (r_sub * K_sub_inv_r).sum()
#             rho = 1.0 - (v_sq / u_sq_safe)
#             return rho
#     except Exception:
#         return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)


def compute_kf_metric_tensor(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    Nf: int,
    cholesky_jitter: Optional[float] = None,
    seed: Optional[int] = None,
    likelihood=None,
) -> torch.Tensor:
    """
    Differentiable Kernel Flows metric ρ_t = 1 - (||v||_K^2 / ||u||_K^2) as a 0-dim tensor.

    This variant mirrors the original KF definition but keeps the computation in the
    autograd graph so it can be used directly as a loss. In our convention we will
    typically minimize ρ itself (loss = ρ), so lower values are better.

    Returns
    -------
    torch.Tensor
        0-dim tensor ρ_t on the same device as train_x. Returns a 0.0 tensor
        (no gradient) if N > 2000 or Nf is invalid.
    """
    jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
    N = train_x.size(0)
    if N < 2 or N > 2000:
        return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
    try:
        with gpytorch.settings.cholesky_jitter(jitter):
            prior = model(train_x)
            r = (train_y - prior.mean).view(-1)
            K_prior = getattr(prior, "lazy_covariance_matrix", None) or getattr(prior, "lazy_covar", None)
            if K_prior is None:
                return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
            # Use PRIOR kernel only here (no observation noise term), with jitter
            # applied purely for numerical stability.
            K_prior_dense = K_prior.to_dense()
            eye = torch.eye(N, device=train_x.device, dtype=train_x.dtype)
            K_dense = K_prior_dense + jitter * eye
            K_inv_r = torch.linalg.solve(K_dense, r.unsqueeze(-1)).squeeze(-1)
            u_sq = (r * K_inv_r).sum()
            u_sq_safe = u_sq.clamp(min=1e-12)
            gen = torch.Generator(device=train_x.device)
            if seed is not None:
                gen.manual_seed(seed)
            # Full vs half scheme: use all N points for the full norm (u_sq)
            # and a subset of size Ns = N // 2 for the subset norm (v_sq).
            Ns = max(1, N // 2)
            inds = torch.randperm(N, device=train_x.device, generator=gen)[:Ns]
            r_sub = r[inds]
            K_sub = K_dense[inds][:, inds]
            K_sub_inv_r = torch.linalg.solve(K_sub, r_sub.unsqueeze(-1)).squeeze(-1)
            v_sq = (r_sub * K_sub_inv_r).sum()
            rho = 1.0 - (v_sq / u_sq_safe)
            return rho
    except Exception:
        return torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)


def compute_rrmse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Relative root mean squared error (RRMSE) of training residuals:
    sqrt(MSE) / std(train_y), where MSE is computed on (train_y - model_mean).

    When train_y is standardized (mean 0, std 1), std(train_y)=1 so this equals
    RMSE in standardized space, which is the same as RRMSE on the original scale
    (no need for original y_mean/y_std). Lower is better.

    Args:
        model: GP model with train data already set.
        train_x: Training inputs.
        train_y: Training targets.
        cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.

    Returns:
        RRMSE (scalar). Returns float('nan') on failure.
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        r = (y_flat - mean)
        mse = (r**2).mean()
        # Normalize by std of training targets
        y_std = y_flat.std()
        eps = torch.tensor(1e-12, device=y_std.device, dtype=y_std.dtype)
        denom = torch.clamp(y_std, min=eps)
        rrmse = torch.sqrt(mse) / denom
        model.train()
        return float(rrmse.item()) if torch.isfinite(rrmse) else float("nan")
    except Exception:
        return float("nan")


def compute_mse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Mean squared error of training residuals: mean((train_y - model_mean)^2).
    Lower is better.
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        r = y_flat - mean
        mse = (r ** 2).mean()
        model.train()
        return float(mse.item()) if torch.isfinite(mse) else float("nan")
    except Exception:
        return float("nan")


def compute_r2(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Coefficient of determination R^2 on training data.

    R^2 = 1 - SSE / SST, where:
      - SSE = sum((y - y_hat)^2)
      - SST = sum((y - mean(y))^2)
    """
    try:
        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        model.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model(train_x)
            mean = out.mean.detach().view(-1)
        y_flat = train_y.view(-1)
        resid = y_flat - mean
        sse = (resid * resid).sum()
        y_mean = y_flat.mean()
        sst = ((y_flat - y_mean) ** 2).sum()
        eps = torch.tensor(1e-12, device=sst.device, dtype=sst.dtype)
        sst = torch.clamp(sst, min=eps)
        r2 = 1.0 - (sse / sst)
        model.train()
        return float(r2.item()) if torch.isfinite(r2) else float("nan")
    except Exception:
        return float("nan")


def compute_residual_mse(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """Backwards-compatible alias for compute_mse (residual MSE)."""
    return compute_mse(model, train_x, train_y, cholesky_jitter=cholesky_jitter)


def compute_training_nis(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Compute Normalized Interval Score (NIS) on training data.

    Uses the model's posterior at train_x to get mean and variance, then
    computes NIS (interval score normalized by std of y). Lower is better.

    Args:
        model: GP model (ExactGP or similar) with train data already set.
        train_x: Training inputs.
        train_y: Training targets.
        alpha: Nominal miscoverage for interval (default 0.05).

    Returns:
        NIS scalar (float). Returns float('nan') on failure.
    """
    try:
        with torch.no_grad():
            model.eval()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                out = model.likelihood(model(train_x))
            mean = out.mean
            var = out.variance
            std = var.sqrt()
            model.train()

        # compute_nis expects numpy or tensor; it converts to numpy
        result = compute_nis(
            train_y,
            y_hat=mean,
            output_std=std,
            alpha=alpha,
            normalize_by_y_std=True,
        )

        return result["NIS"]
    except Exception:
        return float("nan")


def compute_loo_pll(
    model,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
) -> float:
    """
    Compute leave-one-out pseudo log likelihood on training data.

    Uses GPyTorch's LeaveOneOutPseudoLikelihood. Higher is better.

    Args:
        model: GP model (ExactGP or similar).
        likelihood: Model's likelihood (e.g. model.likelihood).
        train_x: Training inputs.
        train_y: Training targets.
        cholesky_jitter: Optional jitter; if None, uses model.cholesky_jitter if set.

    Returns:
        LOO pseudo log likelihood (scalar). Returns float('nan') on failure.
    """
    try:
        import linear_operator

        jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
        # Stay in train mode to avoid GPInputWarning (eval + same input as train data)
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            try:
                linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
            except Exception:
                pass
            loo_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
            out = model(train_x)
            pll = loo_mll(out, train_y)
            if pll.dim() == 0:
                return -pll.item()
            return -pll.sum().item()
    except Exception:
        return float("nan")


def compute_training_metrics_batch(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cholesky_jitter: Optional[float] = None,
    log_nis: bool = True,
    log_loo: bool = True,
    log_kf: bool = True,
    log_mse: bool = True,
    log_r2: bool = True,
    kf_Nf: Optional[int] = None,
    kf_seed: Optional[int] = None,
    nis_alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute NIS, LOO_NLL, KF, MSE, R2 in one go: single model.eval(), one forward
    (prior = model(train_x)), one likelihood(prior) for NIS, then model.train().
    Returns only requested metrics to avoid redundant work.
    """
    out: Dict[str, float] = {}
    jitter = cholesky_jitter if cholesky_jitter is not None else getattr(model, "cholesky_jitter", 1e-6)
    train_x = train_x.to(dtype=getattr(model, "dtype", torch.float64))
    train_y = train_y.to(dtype=getattr(model, "dtype", torch.float64))
    N = train_x.size(0)

    try:
        import linear_operator
    except Exception:
        linear_operator = None  # continue without it; skip linear_operator.settings below

    try:
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
            if linear_operator is not None:
                try:
                    linear_operator.settings.cholesky_jitter(double_value=jitter, float_value=jitter)
                except Exception:
                    pass
            with warnings.catch_warnings():
                model.eval()
                warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                prior = model(train_x)
                pred = model.likelihood(prior)
            model.train()
            mean = pred.mean.detach().view(-1)
            y_flat = train_y.view(-1)

            if log_mse:
                r = y_flat - mean
                mse = (r ** 2).mean()
                out["MSE"] = float(mse.item()) if torch.isfinite(mse) else float("nan")
            if log_r2:
                resid = y_flat - mean
                sse = (resid * resid).sum()
                sst = ((y_flat - y_flat.mean()) ** 2).sum().clamp(min=1e-12)
                r2 = 1.0 - (sse / sst)
                out["R2"] = float(r2.item()) if torch.isfinite(r2) else float("nan")

            if log_nis:
                try:
                    var = pred.variance
                    std = var.sqrt()
                    nis_result = compute_nis(
                        train_y,
                        y_hat=mean,
                        output_std=std,
                        alpha=nis_alpha,
                        normalize_by_y_std=True,
                    )
                    out["NIS"] = nis_result["NIS"]
                except Exception:
                    out["NIS"] = float("nan")

            if log_loo:
                try:
                    # Use train-mode forward so LOO_NLL matches the LOO loss when loss_type=="loo"
                    # (loss is computed in train mode; prior above was from model.eval()).
                    model.train()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
                        prior_loo = model(train_x)
                    loo_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
                    pll = loo_mll(prior_loo, train_y)
                    val = pll.sum().item() if pll.dim() > 0 else pll.item()
                    out["LOO_NLL"] = float(-val) if torch.isfinite(torch.tensor(val)) else float("nan")
                except Exception:
                    out["LOO_NLL"] = float("nan")

            if log_kf:
                out["KF"] = float("nan")
                try:
                    Nf_val = kf_Nf if kf_Nf is not None else min(64, N)
                    kf_result = compute_kf_metric(
                        model,
                        train_x,
                        train_y,
                        Nf=Nf_val,
                        cholesky_jitter=jitter,
                        seed=kf_seed,
                        likelihood=model.likelihood,
                    )
                    out["KF"] = float(kf_result) if isinstance(kf_result, (int, float)) else float(kf_result.get("KF", float("nan")))
                except Exception:
                    out["KF"] = float("nan")
    finally:
        model.train()

    return out
