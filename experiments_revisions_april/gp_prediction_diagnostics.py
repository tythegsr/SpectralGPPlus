"""
GP prediction diagnostics: true vs predicted scatter and GP mean ± uncertainty (1D or marginal slices).

Used by synthetic benchmarks A4–A9 when defaults.PLOT_PREDICTION_DIAGNOSTICS is True.
Outputs: <save_path>/plots/prediction_diagnostics_<experiment_title>/run_XXX/ (title sanitized for paths).
Denormalization matches gpplus.utils.train_eval.train_eval_gp (scalar y mean/std, log-y path).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.train_eval import _torch_log_y_point_original

from plot_tabpfn1d_comparison import save_1d_train_gp_tabpfn_plot


def prediction_diagnostics_plot_subdir(experiment_title: str) -> str:
    """
    Directory segment under <save_path>/plots/, parallel to JSON names like gp_{title}.json.
    Sanitizes characters that are invalid on Windows paths.
    """
    t = (experiment_title or "experiment").strip()
    for c in '\\/:*?"<>|':
        t = t.replace(c, "_")
    t = t.rstrip(" .")
    return f"prediction_diagnostics_{t}"


def denormalize_gp_output(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    standardize_y_log_scale: bool,
    log_scale_C: float | None,
    log_y_point_inverse: str = "median",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map GP predictive mean/std from model target space to original y scale."""
    if y_train_mean is None or y_train_std is None:
        return mean, std

    mean_val = y_train_mean
    std_val = y_train_std
    if isinstance(mean_val, torch.Tensor):
        mean_val = mean_val.to(device=mean.device, dtype=mean.dtype).squeeze()
    else:
        mean_val = torch.tensor(float(mean_val), device=mean.device, dtype=mean.dtype)
    if isinstance(std_val, torch.Tensor):
        std_val = std_val.to(device=mean.device, dtype=mean.dtype).squeeze()
    else:
        std_val = torch.tensor(float(std_val), device=mean.device, dtype=mean.dtype)

    if standardize_y_log_scale and log_scale_C is not None:
        log_y_pred = (mean * std_val) + mean_val
        log_y_std = std * std_val
        max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
        log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
        exp_log_y = torch.exp(log_y_pred)
        y_pred = _torch_log_y_point_original(
            exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
        )
        output_std = exp_log_y * log_y_std
        return y_pred, output_std

    if standardize_y:
        return (mean * std_val) + mean_val, std * std_val

    return mean, std


def predict_gp_denormalized(
    model,
    X_scaled: torch.Tensor,
    *,
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    standardize_y_log_scale: bool,
    log_scale_C: float | None,
    log_y_point_inverse: str = "median",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    Xb = X_scaled.to(device=device, dtype=next(model.parameters()).dtype)
    mean, _, _, std = evaluate_gp_model(model, Xb)
    return denormalize_gp_output(
        mean,
        std,
        y_train_mean=y_train_mean,
        y_train_std=y_train_std,
        standardize_y=standardize_y,
        standardize_y_log_scale=standardize_y_log_scale,
        log_scale_C=log_scale_C,
        log_y_point_inverse=log_y_point_inverse,
    )


def save_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
) -> Path:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, c="C0", edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = max((hi - lo) * 0.02, 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y_true (test)")
    ax.set_ylabel("y_pred (GP)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    if len(y_true) > 1:
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        std_true = float(np.std(y_true))
        rrmse = rmse / std_true if std_true > 0 else float("inf")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        extra = f"RRMSE={rrmse:.4g}  R²={r2:.4f}"
        if subtitle:
            extra = f"{subtitle}\n{extra}"
        ax.text(0.02, 0.98, extra, transform=ax.transAxes, va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gp_prediction_panel_1d(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_pred_gp: np.ndarray,
    y_std_gp: np.ndarray,
    truth_fn: Callable[[torch.Tensor], torch.Tensor],
    out_dir: Path,
    title: str,
    run_index: int,
    interval_z: float = 1.96,
) -> Path:
    """1D: train points, true f(x) on test grid, GP mean ± PI."""
    out_dir = Path(out_dir)
    X_test_t = torch.as_tensor(x_test, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        y_true_test = truth_fn(X_test_t).detach().cpu().numpy().astype(np.float64).ravel()
    return save_1d_train_gp_tabpfn_plot(
        x_train,
        y_train,
        x_test,
        y_pred_gp,
        None,
        y_std_gp,
        None,
        out_dir,
        title=title,
        run_index=run_index,
        y_true_test=y_true_test,
        file_suffix="gp_diag",
        interval_z=interval_z,
    )


def save_gp_marginal_slices(
    *,
    model,
    X_train_orig: torch.Tensor,
    cont_cols: list[int],
    Xscaler,
    standardize_X: bool,
    truth_fn: Callable[[torch.Tensor], torch.Tensor],
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    standardize_y_log_scale: bool,
    log_scale_C: float | None,
    log_y_point_inverse: str,
    x_bounds: list[float],
    out_dir: Path,
    title: str,
    run_index: int,
    max_marginal_dims: int,
    n_grid: int = 200,
    interval_z: float = 1.96,
) -> list[Path]:
    """For D>1: partial dependence along the first K continuous dimensions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    median_row = X_train_orig.median(dim=0).values.detach().cpu()
    dtype = X_train_orig.dtype
    device_model = next(model.parameters()).device

    k_max = min(max_marginal_dims, len(cont_cols))
    for j in range(k_max):
        col = cont_cols[j]
        x_lo = float(X_train_orig[:, col].min().item())
        x_hi = float(X_train_orig[:, col].max().item())
        if x_bounds is not None and len(x_bounds) == 2:
            x_lo = min(x_lo, float(x_bounds[0]))
            x_hi = max(x_hi, float(x_bounds[1]))
        grid = torch.linspace(x_lo, x_hi, n_grid, dtype=dtype)

        X_list = []
        for val in grid:
            row = median_row.clone()
            row[col] = val
            X_list.append(row)
        X_orig = torch.stack(X_list, dim=0)

        X_scaled = X_orig.clone()
        if standardize_X and Xscaler is not None:
            X_scaled[:, cont_cols] = Xscaler.transform(X_scaled[:, cont_cols].to(X_train_orig.device))

        mean_y, std_y = predict_gp_denormalized(
            model,
            X_scaled.to(device_model),
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            standardize_y=standardize_y,
            standardize_y_log_scale=standardize_y_log_scale,
            log_scale_C=log_scale_C,
            log_y_point_inverse=log_y_point_inverse,
        )
        mean_np = mean_y.detach().cpu().numpy().ravel()
        std_np = std_y.detach().cpu().numpy().ravel()
        x_np = grid.detach().cpu().numpy().ravel()

        with torch.no_grad():
            y_true_np = truth_fn(X_orig.to(device=X_train_orig.device)).detach().cpu().numpy().ravel()

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.plot(x_np, y_true_np, color="#D95F02", lw=2.0, label="True f(x)", zorder=3)
        ax.fill_between(
            x_np,
            mean_np - interval_z * std_np,
            mean_np + interval_z * std_np,
            color="#1B9E77",
            alpha=0.2,
            linewidth=0.0,
            label=f"GP ±{interval_z}σ",
            zorder=1,
        )
        ax.plot(x_np, mean_np, color="#1B9E77", lw=1.8, label="GP mean", zorder=2)
        tr_x = X_train_orig[:, col].detach().cpu().numpy()
        y_min = float(np.nanmin(np.concatenate([y_true_np, mean_np - interval_z * std_np])))
        y_max = float(np.nanmax(np.concatenate([y_true_np, mean_np + interval_z * std_np])))
        rug_y = y_min - 0.05 * (y_max - y_min + 1e-9)
        ax.scatter(
            tr_x,
            np.full(tr_x.shape, rug_y),
            s=10,
            c="0.3",
            alpha=0.4,
            marker="|",
            linewidths=0.8,
            label="Train x",
        )

        ax.set_xlabel(f"x[{col}] (original scale)")
        ax.set_ylabel("y (original scale)")
        ax.set_title(f"{title}\nrun {run_index + 1} marginal dim {j} (col {col})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.28)
        fig.tight_layout()
        fp = out_dir / f"marginal_dim{j}_col{col}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        saved.append(fp)

    return saved


def run_gp_prediction_diagnostics(
    *,
    save_path: str | Path | None,
    run_index: int,
    experiment_title: str,
    dimensions: int,
    cont_cols: list[int],
    x_bounds: list[float],
    X_train_orig: torch.Tensor,
    X_train_raw_for_pfn: torch.Tensor,
    X_test_raw_for_pfn: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    y_pred_gp: np.ndarray,
    output_std_gp: np.ndarray,
    model,
    Xscaler,
    standardize_X: bool,
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    standardize_y_log_scale: bool = False,
    log_scale_C: float | None = None,
    log_y_point_inverse: str = "median",
    truth_fn: Callable[[torch.Tensor], torch.Tensor],
    plot_prediction_diagnostics: bool,
    diagnostic_run_indices: tuple[int, ...],
    max_marginal_dims: int,
) -> None:
    """Entry point from experiment loops."""
    if save_path is None or not plot_prediction_diagnostics:
        return
    if run_index not in diagnostic_run_indices:
        return

    base = (
        Path(save_path)
        / "plots"
        / prediction_diagnostics_plot_subdir(experiment_title)
        / f"run_{run_index:03d}"
    )
    base.mkdir(parents=True, exist_ok=True)

    y_test_np = y_test.detach().cpu().numpy().ravel()
    save_true_vs_pred_scatter(
        y_test_np,
        y_pred_gp,
        base / "gp_true_vs_pred_test.png",
        title=f"{experiment_title}\nGP test: true vs pred (run {run_index + 1})",
    )

    if dimensions == 1:
        x_tr = X_train_raw_for_pfn[:, cont_cols].detach().cpu().numpy().ravel()
        y_tr = y_train.detach().cpu().numpy().ravel()
        x_te = X_test_raw_for_pfn[:, cont_cols].detach().cpu().numpy().ravel()
        save_gp_prediction_panel_1d(
            x_train=x_tr,
            y_train=y_tr,
            x_test=x_te,
            y_pred_gp=y_pred_gp,
            y_std_gp=output_std_gp,
            truth_fn=truth_fn,
            out_dir=base,
            title=experiment_title,
            run_index=run_index,
        )
    else:
        save_gp_marginal_slices(
            model=model,
            X_train_orig=X_train_orig,
            cont_cols=cont_cols,
            Xscaler=Xscaler,
            standardize_X=standardize_X,
            truth_fn=truth_fn,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            standardize_y=standardize_y,
            standardize_y_log_scale=standardize_y_log_scale,
            log_scale_C=log_scale_C,
            log_y_point_inverse=log_y_point_inverse,
            x_bounds=x_bounds,
            out_dir=base,
            title=experiment_title,
            run_index=run_index,
            max_marginal_dims=max_marginal_dims,
        )
