"""Helpers for S2 independent-task runners (band select, ARD extract, metrics)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def select_bands(x: torch.Tensor, band_indices: Sequence[int]) -> torch.Tensor:
    """Keep only ``band_indices`` from the last dimension of ``x``."""
    idx = torch.as_tensor(list(band_indices), dtype=torch.int64, device=x.device)
    return x.index_select(-1, idx)


def apply_x_transform(x: torch.Tensor, transform: str | None) -> torch.Tensor:
    """
    Elementwise input transform applied after band select, before PCA / X scaling.

    Supported: ``None`` / ``\"none\"``, ``\"log1p\"`` (``ln(1 + max(x, 0))``).
    """
    if transform is None or transform == "none":
        return x
    if transform == "log1p":
        return torch.log1p(x.clamp_min(0.0))
    raise ValueError(f"Unknown x_transform {transform!r}; expected 'none' or 'log1p'.")


def extract_ard_lengthscales(model) -> dict[str, Any]:
    """Extract transformed lengthscales from a GP / RFFGP model if present."""
    out: dict[str, Any] = {
        "lengthscale": [],
        "raw_lengthscale": [],
        "outputscale": None,
    }
    try:
        covar = model.covar_module
    except Exception:
        return out

    try:
        if hasattr(covar, "outputscale"):
            out["outputscale"] = float(covar.outputscale.detach().cpu().reshape(-1)[0].item())
    except Exception:
        pass

    base = getattr(covar, "base_kernel", covar)
    try:
        if hasattr(base, "lengthscale"):
            ls = base.lengthscale.detach().cpu().reshape(-1)
            out["lengthscale"] = [float(v) for v in ls.tolist()]
        if hasattr(base, "raw_lengthscale"):
            raw = base.raw_lengthscale.detach().cpu().reshape(-1)
            out["raw_lengthscale"] = [float(v) for v in raw.tolist()]
    except Exception:
        pass
    return out


def map_ard_to_bands(
    lengthscales: Sequence[float],
    band_indices: Sequence[int],
    *,
    wavelengths_nm: Sequence[float] | None = None,
    ard_space: str = "bands",
) -> dict[str, Any]:
    """Attach ARD values to original band indices or PCA component ids."""
    ls = [float(v) for v in lengthscales]
    if ard_space == "bands":
        if len(ls) not in (0, 1, len(band_indices)):
            # Soft mismatch: still return what we have.
            pass
        entries = []
        for i, band in enumerate(band_indices):
            entry = {"band_index": int(band), "lengthscale": float(ls[i]) if i < len(ls) else None}
            if wavelengths_nm is not None:
                entry["wavelength_nm"] = float(wavelengths_nm[int(band)])
            entries.append(entry)
        return {"ard_space": "bands", "entries": entries, "lengthscale": ls}

    entries = [{"component": i, "lengthscale": float(ls[i]) if i < len(ls) else None} for i in range(len(ls))]
    return {
        "ard_space": "pca_components",
        "physical_band_indices": [int(b) for b in band_indices],
        "entries": entries,
        "lengthscale": ls,
    }


def extract_nigp_input_noise(model) -> dict[str, Any] | None:
    """
    Extract learned NIGP per-dimension input noise if ``model.nigp`` is enabled.

    Returns ``None`` when NIGP is off or parameters are missing.
    """
    if not bool(getattr(model, "nigp", False)):
        return None
    if not hasattr(model, "raw_input_noise"):
        return None
    try:
        raw = model.raw_input_noise.detach().cpu().reshape(-1)
        std = model.input_noise.detach().cpu().reshape(-1)
        var = model.input_noise_var.detach().cpu().reshape(-1)
    except Exception:
        return None
    return {
        "raw_input_noise": [float(v) for v in raw.tolist()],
        "input_noise": [float(v) for v in std.tolist()],
        "input_noise_var": [float(v) for v in var.tolist()],
    }


def compute_nigp_grad_diagnostics(
    model,
    x: torch.Tensor,
    *,
    band_indices: Sequence[int] | None = None,
    wavelengths_nm: Sequence[float] | None = None,
    noise_space: str = "bands",
    jitter: float = 1e-6,
    max_points: int = 4096,
    seed: int = 0,
) -> dict[str, Any] | None:
    """
    Per-dimension NIGP diagnostics from ``∇_x μ`` and learned ``σ_x``.

    Computes on up to ``max_points`` rows of ``x`` (subsampled if larger):

    - ``grad_mu_abs_mean[d]`` = mean_i |∂_{x_d} μ(x_i)|
    - ``grad_mu_sq_mean[d]`` = mean_i (∂_{x_d} μ)^2
    - ``input_noise_contrib_mean[d]`` = mean_i σ_{x,d}^2 (∂_{x_d} μ)^2
      (the per-dim term that enters effective observation noise)
    - ``input_noise_contrib_frac[d]`` = normalized contrib across dimensions
    - ``effective_input_term_mean`` = mean_i Σ_d σ_{x,d}^2 (∂_{x_d} μ)^2

    Returns ``None`` when NIGP is off.
    """
    if not bool(getattr(model, "nigp", False)) or not hasattr(model, "raw_input_noise"):
        return None

    from gpplus.utils.nigp_utils import (
        exact_posterior_mean_grad_wrt_x,
        posterior_mean_grad_wrt_x,
    )

    try:
        from gpplus.models.rff_gpr import _drop_singleton_batch
    except Exception:  # pragma: no cover
        def _drop_singleton_batch(t):  # type: ignore[misc]
            return t.squeeze(0) if t.dim() > 1 and t.shape[0] == 1 else t

    # Align all tensors to the model device (x may be CPU while model is CUDA).
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    train_x = model.train_inputs[0]
    if isinstance(train_x, (tuple, list)):
        train_x = train_x[0]
    train_x = _drop_singleton_batch(train_x).to(device=device, dtype=dtype)
    train_y = _drop_singleton_batch(model.train_targets).to(device=device, dtype=dtype)

    noise = model.likelihood.noise
    if noise.dim() > 1:
        noise = noise.reshape(noise.shape[0])
    noise = noise.to(device=device, dtype=dtype)

    mean_train = model.mean_module(train_x)
    if mean_train.dim() > 1 and mean_train.shape[0] == 1:
        mean_train = mean_train.squeeze(0)
    y_centered = train_y - mean_train.reshape(train_y.shape)

    x_eval = x.detach().to(device=device, dtype=dtype)
    n = int(x_eval.shape[0])
    n_used = n
    if max_points is not None and n > int(max_points):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[: int(max_points)]
        x_eval = x_eval[idx.to(device=device)]
        n_used = int(x_eval.shape[0])

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.enable_grad():
            if hasattr(model, "scaled_features"):
                grad = posterior_mean_grad_wrt_x(
                    model,
                    x_eval,
                    y_centered,
                    noise,
                    train_x=train_x,
                    jitter=jitter,
                )
            else:
                grad = exact_posterior_mean_grad_wrt_x(
                    model,
                    x_eval,
                    noise,
                    train_x=train_x,
                    jitter=jitter,
                )
    finally:
        if was_training:
            model.train()

    grad = grad.detach()
    if grad.dim() == 1:
        grad = grad.unsqueeze(0)
    abs_g = grad.abs()
    g2 = grad * grad
    sx2 = model.input_noise_var.detach().to(device=grad.device, dtype=grad.dtype).reshape(-1)
    if sx2.numel() == 1 and grad.shape[-1] > 1:
        sx2 = sx2.expand(grad.shape[-1])
    contrib = g2 * sx2  # (n, D)
    contrib_mean = contrib.mean(dim=0)
    contrib_sum = float(contrib_mean.sum().clamp_min(1e-30).item())
    contrib_frac = contrib_mean / contrib_sum

    grad_mu_abs_mean = [float(v) for v in abs_g.mean(dim=0).cpu().tolist()]
    grad_mu_abs_median = [float(v) for v in abs_g.median(dim=0).values.cpu().tolist()]
    grad_mu_sq_mean = [float(v) for v in g2.mean(dim=0).cpu().tolist()]
    input_noise_contrib_mean = [float(v) for v in contrib_mean.cpu().tolist()]
    input_noise_contrib_frac = [float(v) for v in contrib_frac.cpu().tolist()]
    input_noise = [float(v) for v in model.input_noise.detach().cpu().reshape(-1).tolist()]
    input_noise_var = [float(v) for v in sx2.detach().cpu().tolist()]

    d = len(grad_mu_abs_mean)
    indices = list(band_indices) if band_indices is not None else list(range(d))
    entries: list[dict[str, Any]] = []
    for i in range(d):
        if noise_space == "pca_components":
            entry: dict[str, Any] = {"component": i}
        else:
            band = int(indices[i]) if i < len(indices) else i
            entry = {"band_index": band}
            if wavelengths_nm is not None and band < len(wavelengths_nm):
                entry["wavelength_nm"] = float(wavelengths_nm[band])
        entry["input_noise"] = input_noise[i] if i < len(input_noise) else None
        entry["input_noise_var"] = input_noise_var[i] if i < len(input_noise_var) else None
        entry["grad_mu_abs_mean"] = grad_mu_abs_mean[i]
        entry["grad_mu_abs_median"] = grad_mu_abs_median[i]
        entry["grad_mu_sq_mean"] = grad_mu_sq_mean[i]
        entry["input_noise_contrib_mean"] = input_noise_contrib_mean[i]
        entry["input_noise_contrib_frac"] = input_noise_contrib_frac[i]
        entries.append(entry)

    out: dict[str, Any] = {
        "nigp_space": noise_space,
        "n_points_total": n,
        "n_points_used": n_used,
        "max_points": int(max_points) if max_points is not None else None,
        "jitter": float(jitter),
        "input_noise": input_noise,
        "input_noise_var": input_noise_var,
        "grad_mu_abs_mean": grad_mu_abs_mean,
        "grad_mu_abs_median": grad_mu_abs_median,
        "grad_mu_sq_mean": grad_mu_sq_mean,
        "input_noise_contrib_mean": input_noise_contrib_mean,
        "input_noise_contrib_frac": input_noise_contrib_frac,
        "effective_input_term_mean": float(contrib.sum(dim=-1).mean().item()),
        "entries": entries,
    }
    if noise_space == "pca_components" and band_indices is not None:
        out["physical_band_indices"] = [int(b) for b in band_indices]
    return out


def map_input_noise_to_bands(
    input_noise: Sequence[float],
    band_indices: Sequence[int],
    *,
    input_noise_var: Sequence[float] | None = None,
    wavelengths_nm: Sequence[float] | None = None,
    noise_space: str = "bands",
) -> dict[str, Any]:
    """Attach per-dimension NIGP input-noise std/var to bands or PCA components."""
    stds = [float(v) for v in input_noise]
    vars_ = (
        [float(v) for v in input_noise_var]
        if input_noise_var is not None
        else [s * s for s in stds]
    )
    if noise_space == "bands":
        entries = []
        for i, band in enumerate(band_indices):
            entry = {
                "band_index": int(band),
                "input_noise": float(stds[i]) if i < len(stds) else None,
                "input_noise_var": float(vars_[i]) if i < len(vars_) else None,
            }
            if wavelengths_nm is not None:
                entry["wavelength_nm"] = float(wavelengths_nm[int(band)])
            entries.append(entry)
        return {
            "nigp_space": "bands",
            "entries": entries,
            "input_noise": stds,
            "input_noise_var": vars_,
        }

    entries = [
        {
            "component": i,
            "input_noise": float(stds[i]) if i < len(stds) else None,
            "input_noise_var": float(vars_[i]) if i < len(vars_) else None,
        }
        for i in range(len(stds))
    ]
    return {
        "nigp_space": "pca_components",
        "physical_band_indices": [int(b) for b in band_indices],
        "entries": entries,
        "input_noise": stds,
        "input_noise_var": vars_,
    }


def compute_per_task_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    task_names: Sequence[str],
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, len(task_names))
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, len(task_names))
    metrics: dict[str, float] = {}
    for t, name in enumerate(task_names):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        metrics.update(_scalar_error_metrics(yt, yp, prefix=f"{name}_"))
    return metrics


def _scalar_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    abs_err = np.abs(yp - yt)
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    std = float(np.std(yt))
    rrmse = rmse / std if std > 0 else float("inf")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}MedAE": medae,
        f"{prefix}RRMSE": rrmse,
        f"{prefix}R2": r2,
    }


def compute_log_scale_extra_metrics(
    y_true_physical: np.ndarray | torch.Tensor,
    *,
    log_mu: np.ndarray | torch.Tensor,
    point_mean_physical: np.ndarray | torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Extra test metrics for QoIs trained in ln-space.

    - ``*_log``: errors in ln-space (``ln(y_true)`` vs predictive ``log_mu``).
    - ``*_mean``: physical errors using log-normal mean as the point estimate.
    """
    yt = np.asarray(y_true_physical, dtype=np.float64).reshape(-1)
    mu = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    if yt.shape != mu.shape:
        raise ValueError(f"y_true and log_mu shape mismatch: {yt.shape} vs {mu.shape}")
    if np.any(yt <= 0):
        raise ValueError("log-scale metrics require strictly positive y_true.")

    out = _scalar_error_metrics(np.log(yt), mu, prefix="")
    # Rename to *_log
    log_out = {
        "RMSE_log": out["RMSE"],
        "MAE_log": out["MAE"],
        "MedAE_log": out["MedAE"],
        "RRMSE_log": out["RRMSE"],
        "R2_log": out["R2"],
    }
    if point_mean_physical is not None:
        ym = np.asarray(point_mean_physical, dtype=np.float64).reshape(-1)
        mean_out = _scalar_error_metrics(yt, ym, prefix="")
        log_out.update(
            {
                "RMSE_mean": mean_out["RMSE"],
                "MAE_mean": mean_out["MAE"],
                "MedAE_mean": mean_out["MedAE"],
                "RRMSE_mean": mean_out["RRMSE"],
                "R2_mean": mean_out["R2"],
            }
        )
    return log_out


def macro_rrmse(per_task: Mapping[str, float], task_names: Sequence[str]) -> float:
    vals = [float(per_task[f"{name}_RRMSE"]) for name in task_names]
    return float(np.mean(vals)) if vals else float("nan")


def macro_metric(
    per_task: Mapping[str, float],
    task_names: Sequence[str],
    suffix: str,
) -> float:
    """Mean of ``{name}_{suffix}`` over tasks that define the key."""
    vals = [
        float(per_task[f"{name}_{suffix}"])
        for name in task_names
        if f"{name}_{suffix}" in per_task
    ]
    return float(np.mean(vals)) if vals else float("nan")
