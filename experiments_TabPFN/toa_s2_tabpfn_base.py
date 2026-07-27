"""S2 independent TabPFN runner with per-QoI original-band subsets."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_DEFAULT_SAVE_DIR = "experiments_TabPFN/results/s2_toa_tabpfn"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _TABPFN_DIR)

from experiments_toa.s2_bands import band_config_metadata, load_task_band_config
from experiments_toa.s2_constants import S2_DEFAULT_BAND_CONFIG_PATH, S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_data import load_s2_toa_data
from experiments_toa.s2_reporting import (
    apply_log_scale_extra_metrics,
    attach_log_scale_aggregate_fields,
    attach_relative_error_fields,
    collect_log_scale_prediction_arrays,
    compute_aggregate_rmse,
    compute_aggregate_rrmse,
    print_end_of_run_summary,
    print_task_test_metrics,
    save_s2_summary_artifacts,
    stack_or_none,
)
from experiments_toa.s2_utils import apply_x_transform, compute_per_task_metrics, macro_metric, select_bands
from experiments_toa.s2_y_transform import (
    forward_y_s2,
    inverse_y_s2,
    resolve_log_scale,
    task_uses_log_scale,
)
from gpplus.utils import compute_metrics, set_seed
from mtgpr_experiment_utils import (
    compute_relative_error_metrics,
    format_relative_error_summary,
    save_metrics_json,
)
from toa_tabpfn_base import PFN_MODEL_VERSION_CHOICES, PdfMode, make_tabpfn_regressor


def run_s2_toa_tabpfn(
    n_train: int = 16000,
    n_test: int = 5000,
    seed: int = 42,
    save_path: str | None = None,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    pfn_device: str = "cuda",
    pfn_model_version: str = "auto",
    ignore_pretraining_limits: bool = False,
    posterior_pdf_mode: PdfMode = "tabpfn_bar",
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    log_scale: bool | None = None,
    x_transform: str | None = None,
) -> dict:
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR
    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    band_cfg_path = task_band_config or str(S2_DEFAULT_BAND_CONFIG_PATH)
    bands_by_task = load_task_band_config(band_cfg_path, task_names=names, input_dim=S2_INPUT_DIM)
    set_seed(seed)

    title = f"S2_TOA_nTrain{n_train}_nTest{n_test}_tabpfn"
    print("=" * 60)
    print(title)
    print(f"S2 TabPFN tasks={names}, band_config={band_cfg_path}")
    print("=" * 60)

    (
        x_train_full,
        y_train,
        x_val_full,
        y_val,
        x_test_full,
        y_test,
        train_idx,
        val_idx,
        test_idx,
        wavelengths,
        data_meta,
    ) = load_s2_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=0,
        seed=seed,
        data_path=data_path,
        input_variable=input_variable,  # type: ignore[arg-type]
        task_names=names,
    )
    log_scale, log_scale_task_set, log_scale_source = resolve_log_scale(
        log_scale, meta=data_meta
    )
    print(
        f"Output log_scale={log_scale} (source={log_scale_source}); "
        f"log tasks={sorted(log_scale_task_set) or '[]'}"
    )
    wavelengths_np = wavelengths.detach().cpu().numpy()
    x_test_orig = x_test_full.clone()

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, dict] = {}
    input_bands_by_task: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    y_pred_mean_all: list[np.ndarray] = []
    y_pred_mode_all: list[np.ndarray] = []
    log_mu_all: list[np.ndarray] = []
    log_sigma_all: list[np.ndarray] = []

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        x_tr = select_bands(x_train_full, band_indices)
        x_te = select_bands(x_test_full, band_indices)
        x_tr = apply_x_transform(x_tr, x_transform)
        x_te = apply_x_transform(x_te, x_transform)
        if x_transform and x_transform != "none" and task_idx == 0:
            print(f"X transform: {x_transform} (before TabPFN fit)")
        x_tr_np = x_tr.detach().cpu().numpy().astype(np.float32)
        x_te_np = x_te.detach().cpu().numpy().astype(np.float32)
        y_tr_raw = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_tr_model = forward_y_s2(
            y_tr_raw, task_name, log_scale=log_scale, log_scale_tasks=log_scale_task_set
        )
        y_tr = y_tr_model.detach().cpu().numpy().astype(np.float32)

        regressor = make_tabpfn_regressor(
            pfn_device=pfn_device,
            pfn_model_version=pfn_model_version,
            seed=seed,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )
        t0 = time.time()
        regressor.fit(x_tr_np, y_tr)
        train_time = time.time() - t0
        total_train_time += train_time

        t1 = time.time()
        full = regressor.predict(x_te_np, output_type="full", quantiles=[0.025, 0.975])
        prediction_time = time.time() - t1
        total_prediction_time += prediction_time

        pred_mean = np.asarray(full["mean"], dtype=np.float64).reshape(-1)
        if "quantiles" in full and isinstance(full["quantiles"], list) and len(full["quantiles"]) >= 2:
            lower = np.asarray(full["quantiles"][0], dtype=np.float64).reshape(-1)
            upper = np.asarray(full["quantiles"][1], dtype=np.float64).reshape(-1)
            pred_std = (upper - lower) / 4.0
        else:
            pred_std = np.full_like(pred_mean, np.nan)
            lower = pred_mean.copy()
            upper = pred_mean.copy()

        inv = inverse_y_s2(
            torch.as_tensor(pred_mean),
            torch.as_tensor(pred_std),
            torch.as_tensor(lower),
            torch.as_tensor(upper),
            task_name=task_name,
            y_scaler=None,
            standardize_y=False,
            log_scale=log_scale,
            log_scale_tasks=log_scale_task_set,
            extended=True,
        )
        pred_mean_t, pred_std_t, lower_t, upper_t = inv.as_tuple()
        pred_mean = pred_mean_t.numpy()
        pred_std = pred_std_t.numpy()
        lower = lower_t.numpy()
        upper = upper_t.numpy()

        computed = compute_metrics(
            y_te,
            pred_mean_t,
            output_std=pred_std_t,
            lower_95=lower_t,
            upper_95=upper_t,
            training_time=train_time,
            prediction_time=prediction_time,
        )
        apply_log_scale_extra_metrics(
            computed,
            y_true=y_te,
            inv=inv,
            task_name=task_name,
            log_scale=log_scale,
            log_scale_tasks=log_scale_task_set,
        )
        task_metrics[task_name] = dict(computed)
        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": [float(wavelengths_np[i]) for i in band_indices],
            "model_input_dim": int(x_tr.shape[-1]),
            "ard_space": "none",
        }
        y_pred_all.append(pred_mean)
        y_std_all.append(pred_std)
        lower_all.append(lower)
        upper_all.append(upper)
        collect_log_scale_prediction_arrays(
            inv,
            pred_mean_t,
            y_pred_mean_all=y_pred_mean_all,
            y_pred_mode_all=y_pred_mode_all,
            log_mu_all=log_mu_all,
            log_sigma_all=log_sigma_all,
        )
        print_task_test_metrics(
            task_name,
            computed,
            log_scale=log_scale,
            log_scale_tasks=log_scale_task_set,
        )

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_pred_mean_stacked = stack_or_none(y_pred_mean_all)
    y_pred_mode_stacked = stack_or_none(y_pred_mode_all)
    log_mu_stacked = stack_or_none(log_mu_all)
    log_sigma_stacked = stack_or_none(log_sigma_all)
    y_test_np = y_test.detach().cpu().numpy()
    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked, names)
    rel_metrics_by_task = attach_relative_error_fields(
        per_task,
        names=names,
        y_true=y_test_np,
        y_pred=y_pred_stacked,
        rel_tolerance=rel_tolerance,
        compute_relative_error_metrics=compute_relative_error_metrics,
    )
    aggregate_rmse = compute_aggregate_rmse(y_pred_stacked, y_test_np)
    aggregate_rrmse = compute_aggregate_rrmse(per_task, names)
    log_task_names, aggregate_rrmse_log, aggregate_rrmse_mean = attach_log_scale_aggregate_fields(
        per_task,
        task_metrics,
        names=names,
        log_scale=log_scale,
        log_scale_tasks=log_scale_task_set,
    )
    aggregate_medae = macro_metric(per_task, names, "MedAE")

    metrics: dict = {
        "title": title,
        "dataset": "s2",
        "input_dim_original": S2_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "model_class": "TabPFNRegressor",
        "pfn_device": pfn_device,
        "pfn_model_version": pfn_model_version,
        "posterior_pdf_mode": posterior_pdf_mode,
        "x_transform": x_transform or "none",
        "log_scale": bool(log_scale),
        "log_scale_tasks": [
            n
            for n in names
            if task_uses_log_scale(n, log_scale=log_scale, log_scale_tasks=log_scale_task_set)
        ],
        "log_scale_source": log_scale_source,
        "rel_tolerance": rel_tolerance,
        "task_band_config": str(band_cfg_path),
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wavelengths_np),
        "input_bands_by_task": input_bands_by_task,
        "data_meta": data_meta,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_RRMSE_mean": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "aggregate_RRMSE_log_tasks": aggregate_rrmse_log,
        "aggregate_RRMSE_lognormal_mean_tasks": aggregate_rrmse_mean,
        "RMSE": aggregate_rmse,
        **per_task,
    }
    for task_name in names:
        for key, value in task_metrics[task_name].items():
            metrics[f"{task_name}_{key}"] = value

    print_end_of_run_summary(
        names=names,
        per_task=per_task,
        rel_metrics_by_task=rel_metrics_by_task,
        aggregate_rrmse=aggregate_rrmse,
        aggregate_rmse=aggregate_rmse,
        log_task_names=log_task_names,
        aggregate_rrmse_log=aggregate_rrmse_log,
        aggregate_rrmse_mean=aggregate_rrmse_mean,
        rel_tolerance=rel_tolerance,
        total_train_time=total_train_time,
        format_relative_error_summary=format_relative_error_summary,
        aggregate_medae=aggregate_medae,
    )

    if save_path:
        save_s2_summary_artifacts(
            save_path=save_path,
            title=title,
            metrics=metrics,
            names=names,
            y_test_np=y_test_np,
            y_pred_stacked=y_pred_stacked,
            y_std_stacked=y_std_stacked,
            lower_stacked=lower_stacked,
            upper_stacked=upper_stacked,
            x_test_orig=x_test_orig.numpy(),
            wavelengths_np=wavelengths_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
            rel_metrics_by_task=rel_metrics_by_task,
            rel_tolerance=rel_tolerance,
            log_scale=log_scale,
            log_scale_tasks=log_scale_task_set,
            data_meta=data_meta,
            seed=seed,
            posterior_n_examples=posterior_n_examples,
            posterior_example_indices=posterior_example_indices,
            plot_validation=False,
            plot_posterior=plot_posterior,
            monitor_validation=False,
            n_val=0,
            save_metrics_json=save_metrics_json,
            y_pred_mean_stacked=y_pred_mean_stacked,
            y_pred_mode_stacked=y_pred_mode_stacked,
            log_mu_stacked=log_mu_stacked,
            log_sigma_stacked=log_sigma_stacked,
        )
    return metrics


__all__ = ["PFN_MODEL_VERSION_CHOICES", "run_s2_toa_tabpfn"]
