"""Shared S2 end-of-run summaries, log-scale extras, and plot/artifact helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments_toa.s2_plotting import (
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import compute_log_scale_extra_metrics, macro_metric, macro_rrmse
from experiments_toa.s2_y_transform import task_uses_log_scale

logger = logging.getLogger(__name__)


def plot_per_task_validation_curves(
    metrics: dict,
    save_path: str | Path,
    task_name: str,
    json_path: str | None = None,
) -> list[str]:
    """Write validation PNGs for one independent QoI under ``{save_path}/validation/{task}``."""
    block_key = f"{task_name}_validation_metrics_by_init"
    if block_key not in metrics:
        return []
    from plot_validation_curves import plot_run

    task_metrics = {
        "monitor_validation": True,
        "validation_metrics_by_init": metrics[block_key],
        "best_init_index": metrics.get(f"{task_name}_best_init_index"),
        "best_val_NLL": metrics.get(f"{task_name}_best_val_NLL"),
        "best_val_RRMSE": metrics.get(f"{task_name}_best_val_RRMSE"),
        "title": metrics.get("title", ""),
    }
    if json_path:
        task_metrics["_source_file"] = json_path
    out_dir = Path(save_path) / "validation" / task_name
    try:
        return [str(p) for p in plot_run(task_metrics, out_dir)]
    except Exception as exc:
        logger.warning("Validation plot generation failed for %s: %s", task_name, exc)
        return []


def spectrum_ylabel_from_meta(data_meta: Mapping[str, Any] | None) -> str:
    meta = data_meta or {}
    if str(meta.get("input_variable", "")).endswith("radiance"):
        return "Radiance"
    return "Reflectance"


def collect_log_scale_prediction_arrays(
    inv: Any,
    pred_mean: torch.Tensor | np.ndarray,
    *,
    y_pred_mean_all: list[np.ndarray],
    y_pred_mode_all: list[np.ndarray],
    log_mu_all: list[np.ndarray],
    log_sigma_all: list[np.ndarray],
) -> None:
    """Append median/mean/mode/log arrays for one task (NaN placeholders when unused)."""
    pred_np = (
        pred_mean.detach().cpu().numpy()
        if isinstance(pred_mean, torch.Tensor)
        else np.asarray(pred_mean)
    )
    if inv.point_mean is not None:
        y_pred_mean_all.append(inv.point_mean.numpy())
    else:
        y_pred_mean_all.append(pred_np)
    if inv.point_mode is not None:
        y_pred_mode_all.append(inv.point_mode.numpy())
    else:
        y_pred_mode_all.append(np.full_like(pred_np, np.nan))
    if inv.log_mu is not None:
        log_mu_all.append(inv.log_mu.numpy())
    else:
        log_mu_all.append(np.full_like(pred_np, np.nan))
    if inv.log_sigma is not None:
        log_sigma_all.append(inv.log_sigma.numpy())
    else:
        log_sigma_all.append(np.full_like(pred_np, np.nan))


def apply_log_scale_extra_metrics(
    computed: dict,
    *,
    y_true: torch.Tensor | np.ndarray,
    inv: Any,
    task_name: str,
    log_scale: bool,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
) -> dict:
    """Mutate ``computed`` with log-scale extras when the QoI is trained in ln-space."""
    if not task_uses_log_scale(
        task_name, log_scale=log_scale, log_scale_tasks=log_scale_tasks
    ):
        return computed
    computed["log_scale"] = True
    if inv.log_mu is None:
        raise RuntimeError(f"log_mu missing after inverse for log-scale task {task_name}")
    extra = compute_log_scale_extra_metrics(
        y_true.cpu() if isinstance(y_true, torch.Tensor) else y_true,
        log_mu=inv.log_mu,
        point_mean_physical=inv.point_mean,
    )
    computed.update(extra)
    return computed


def print_task_test_metrics(
    task_name: str,
    computed: Mapping[str, float],
    *,
    log_scale: bool,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
) -> None:
    if task_uses_log_scale(
        task_name, log_scale=log_scale, log_scale_tasks=log_scale_tasks
    ):
        print(
            f"{task_name} Test (physical median) RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}  "
            f"MedAE: {computed['MedAE']:.6f}"
        )
        print(
            f"{task_name} Test (ln-space) RMSE_log: {computed['RMSE_log']:.6f}  "
            f"RRMSE_log: {computed['RRMSE_log']:.6f}  MAE_log: {computed['MAE_log']:.6f}  "
            f"MedAE_log: {computed['MedAE_log']:.6f}"
        )
        if "RRMSE_mean" in computed:
            print(
                f"{task_name} Test (physical mean) RMSE_mean: {computed['RMSE_mean']:.6f}  "
                f"RRMSE_mean: {computed['RRMSE_mean']:.6f}  "
                f"MedAE_mean: {computed.get('MedAE_mean', float('nan')):.6f}"
            )
    else:
        print(
            f"{task_name} Test RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}  "
            f"MedAE: {computed['MedAE']:.6f}"
        )


def attach_relative_error_fields(
    per_task: dict,
    *,
    names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rel_tolerance: float,
    compute_relative_error_metrics,
) -> dict[str, dict[str, float | int]]:
    """Fill per-task relative-error keys; return ``rel_metrics_by_task``."""
    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}
    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_true[:, t], y_pred[:, t], rel_tolerance=rel_tolerance
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_m["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])
    return rel_metrics_by_task


def attach_log_scale_aggregate_fields(
    per_task: dict,
    task_metrics: Mapping[str, Mapping[str, Any]],
    *,
    names: Sequence[str],
    log_scale: bool,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
) -> tuple[list[str], float, float]:
    """Copy per-task log extras into ``per_task`` and return macros."""
    log_task_names = [
        n
        for n in names
        if task_uses_log_scale(n, log_scale=log_scale, log_scale_tasks=log_scale_tasks)
    ]
    for name in log_task_names:
        tm = task_metrics[name]
        for key in (
            "RMSE_log",
            "MAE_log",
            "MedAE_log",
            "RRMSE_log",
            "R2_log",
            "RMSE_mean",
            "MAE_mean",
            "MedAE_mean",
            "RRMSE_mean",
            "R2_mean",
        ):
            if key in tm:
                per_task[f"{name}_{key}"] = float(tm[key])
    aggregate_rrmse_log = macro_metric(per_task, log_task_names, "RRMSE_log")
    aggregate_rrmse_mean = macro_metric(per_task, log_task_names, "RRMSE_mean")
    return log_task_names, aggregate_rrmse_log, aggregate_rrmse_mean


def print_end_of_run_summary(
    *,
    names: Sequence[str],
    per_task: Mapping[str, float],
    rel_metrics_by_task: Mapping[str, Mapping[str, float | int]],
    aggregate_rrmse: float,
    aggregate_rmse: float,
    log_task_names: Sequence[str],
    aggregate_rrmse_log: float,
    aggregate_rrmse_mean: float,
    rel_tolerance: float,
    total_train_time: float,
    format_relative_error_summary,
    aggregate_medae: float | None = None,
) -> None:
    medae_part = (
        f"  MedAE: {aggregate_medae:.6f}"
        if aggregate_medae is not None and aggregate_medae == aggregate_medae
        else ""
    )
    print(
        f"\nTest macro RRMSE (physical median): {aggregate_rrmse:.6f}  "
        f"RMSE: {aggregate_rmse:.6f}{medae_part}"
    )
    if log_task_names:
        print(
            f"Test macro RRMSE_log (ln-space, log QoIs): {aggregate_rrmse_log:.6f}  "
            f"macro RRMSE_mean (physical lognormal mean): {aggregate_rrmse_mean:.6f}"
        )
    for name in names:
        medae = per_task.get(f"{name}_MedAE", float("nan"))
        print(
            f"{name} RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"MAE: {per_task[f'{name}_MAE']:.6f}  "
            f"MedAE: {medae:.6f}"
        )
        if name in log_task_names:
            print(
                f"  ln-space RRMSE_log: {per_task[f'{name}_RRMSE_log']:.6f}  "
                f"RMSE_log: {per_task[f'{name}_RMSE_log']:.6f}  "
                f"MedAE_log: {per_task.get(f'{name}_MedAE_log', float('nan')):.6f}  |  "
                f"physical-mean RRMSE_mean: {per_task[f'{name}_RRMSE_mean']:.6f}"
            )
        print(
            format_relative_error_summary(
                name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance
            )
        )
    print(f"Total training time: {total_train_time:.1f}s")


def save_s2_summary_artifacts(
    *,
    save_path: str | Path,
    title: str,
    metrics: dict,
    names: Sequence[str],
    y_test_np: np.ndarray,
    y_pred_stacked: np.ndarray,
    y_std_stacked: np.ndarray,
    lower_stacked: np.ndarray,
    upper_stacked: np.ndarray,
    x_test_orig: np.ndarray,
    wavelengths_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    bands_by_task: Mapping[str, Sequence[int]],
    rel_metrics_by_task: Mapping[str, Mapping[str, float | int]],
    rel_tolerance: float,
    log_scale: bool,
    log_scale_tasks: frozenset[str] | set[str] | None,
    data_meta: Mapping[str, Any] | None,
    seed: int,
    posterior_n_examples: int,
    posterior_example_indices: list[int] | None,
    plot_validation: bool,
    plot_posterior: bool,
    monitor_validation: bool,
    n_val: int,
    save_metrics_json,
    y_pred_mean_stacked: np.ndarray | None = None,
    y_pred_mode_stacked: np.ndarray | None = None,
    log_mu_stacked: np.ndarray | None = None,
    log_sigma_stacked: np.ndarray | None = None,
) -> str:
    """Save NPZ + metrics JSON, optional validation/scatter/posterior plots. Returns JSON path."""
    example_indices = select_posterior_example_indices(
        y_test_np.shape[0],
        posterior_n_examples,
        seed=seed,
        explicit_indices=posterior_example_indices,
    )
    log_task_names_plot = [
        n
        for n in names
        if task_uses_log_scale(n, log_scale=log_scale, log_scale_tasks=log_scale_tasks)
    ]
    out_npz = save_s2_predictions_npz(
        save_path,
        title=title,
        task_names=names,
        y_true=y_test_np,
        y_pred=y_pred_stacked,
        y_std=y_std_stacked,
        lower=lower_stacked,
        upper=upper_stacked,
        x_test=x_test_orig,
        wavelengths_nm=wavelengths_np,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        bands_by_task=bands_by_task,
        y_pred_mean=y_pred_mean_stacked if log_task_names_plot else None,
        y_pred_mode=y_pred_mode_stacked if log_task_names_plot else None,
        log_mu=log_mu_stacked if log_task_names_plot else None,
        log_sigma=log_sigma_stacked if log_task_names_plot else None,
        log_scale_tasks=log_task_names_plot,
    )
    print(f"Saved predictions to {out_npz}")
    metrics["predictions_npz"] = str(out_npz)

    out_json = save_metrics_json(metrics, save_path, title)
    print(f"Saved metrics to {out_json}")

    if plot_validation and monitor_validation and n_val > 0:
        for task_name in names:
            for plot_path in plot_per_task_validation_curves(
                metrics, save_path, task_name, json_path=out_json
            ):
                print(f"Saved validation plot to {plot_path}")

    if plot_posterior:
        scatter_dir = Path(save_path) / "plots" / "scatter" / title
        for t, name in enumerate(names):
            plot_s2_task_scatter(
                y_true=y_test_np[:, t],
                y_pred=y_pred_stacked[:, t],
                lower=lower_stacked[:, t],
                upper=upper_stacked[:, t],
                task_name=name,
                out_path=scatter_dir / f"{name}_scatter.png",
                title=f"{title} | {name}",
            )
        post_dir = Path(save_path) / "plots" / "posterior" / title
        post_paths = plot_s2_posterior_examples(
            x_test=x_test_orig,
            wavelengths_nm=wavelengths_np,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
            task_names=names,
            example_indices=example_indices,
            save_dir=post_dir,
            title=title,
            y_std=y_std_stacked,
            rel_metrics_by_task=rel_metrics_by_task,
            rel_tolerance=rel_tolerance,
            log_scale_tasks=log_task_names_plot,
            y_pred_mean=y_pred_mean_stacked if log_task_names_plot else None,
            y_pred_mode=y_pred_mode_stacked if log_task_names_plot else None,
            log_mu=log_mu_stacked if log_task_names_plot else None,
            log_sigma=log_sigma_stacked if log_task_names_plot else None,
            spectrum_ylabel=spectrum_ylabel_from_meta(data_meta),
        )
        for p in post_paths[:3]:
            print(f"Saved posterior plot to {p}")

    return out_json


def stack_or_none(arrays: list[np.ndarray]) -> np.ndarray | None:
    if not arrays:
        return None
    return np.stack(arrays, axis=1)


def compute_aggregate_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_aggregate_rrmse(per_task: Mapping[str, float], names: Sequence[str]) -> float:
    return macro_rrmse(per_task, names)
