"""S4 independent TabPFN runner: snow-TOA / EMIT NetCDF train → ASD predict.

Fits one ``TabPFNRegressor`` per QoI on radiance + geometry aux (coszen, ele_km).
ASD evaluation runs in-process after fit (no ``.pt`` checkpoints).
"""

from __future__ import annotations

import json
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
_SORF_DIR = _ROOT / "experiments_SORF"
_DEFAULT_SAVE_DIR = "experiments_TabPFN/results/s4_emit_tabpfn"
_DEFAULT_DATA_PATH = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_flat_Sep03.nc"
)
_DEFAULT_ASD_PATH = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _TABPFN_DIR, _SORF_DIR)

from experiments_RFFMTGPR.emit_s4_mtgpr_base import (
    _load_wavelengths_nm,
    force_train_local_from_asd_appended,
    load_emit_s4_xy,
    parse_s4_task_names,
    split_emit_s4,
)
from experiments_SORF._eval_s4_asd_validation import TASK_LABEL_MAP, _load_asd_xy
from experiments_toa.s2_bands import band_config_metadata, load_task_band_config
from experiments_toa.s2_constants import (
    S2_TASK_NAMES,
    S4_BAND_CONFIG_ALIASES,
    S4_INPUT_DIM,
    S4_LOGIT_BOUNDS,
    S4_SPECTRAL_DIM,
    S4_TASK_NAMES,
)
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
from experiments_toa.s2_utils import (
    _scalar_error_metrics,
    compute_per_task_metrics,
    macro_metric,
    select_bands,
)
from experiments_toa.s2_y_transform import (
    forward_y_s2,
    inverse_y_s2,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from experiments_toa.s4_asd_posttrain import ASD_TASK_NAMES
from gpplus.utils import compute_metrics, set_seed
from mtgpr_experiment_utils import (
    compute_prediction_coverage_metrics,
    compute_relative_error_metrics,
    format_relative_error_summary,
    save_metrics_json,
)
from toa_tabpfn_base import PFN_MODEL_VERSION_CHOICES, PdfMode, make_tabpfn_regressor


def _task_input_columns(
    band_indices: Sequence[int],
    *,
    aux_indices: Sequence[int],
    n_spectral: int = S4_SPECTRAL_DIM,
) -> list[int]:
    """Spectral band indices plus fixed S4 aux columns (coszen, ele_km)."""
    cols = [int(i) for i in band_indices]
    for idx in aux_indices:
        if int(idx) < n_spectral:
            raise ValueError(f"aux index {idx} must be >= n_spectral={n_spectral}")
        cols.append(int(idx))
    return cols


def _unpack_tabpfn_full(full: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_mean = np.asarray(full["mean"], dtype=np.float64).reshape(-1)
    quantiles = full.get("quantiles")
    if isinstance(quantiles, list) and len(quantiles) >= 2:
        lower = np.asarray(quantiles[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(quantiles[1], dtype=np.float64).reshape(-1)
        pred_std = (upper - lower) / 4.0
    else:
        pred_std = np.full_like(pred_mean, np.nan)
        lower = pred_mean.copy()
        upper = pred_mean.copy()
    return pred_mean, pred_std, lower, upper


def _inverse_physical(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    task_name: str,
    warps,
):
    return inverse_y_s2(
        torch.as_tensor(pred_mean),
        torch.as_tensor(pred_std),
        torch.as_tensor(lower),
        torch.as_tensor(upper),
        task_name=task_name,
        y_scaler=None,
        standardize_y=False,
        warps=warps,
        extended=True,
    )


def run_s4_emit_tabpfn(
    n_train: int = 10000,
    n_val: int = 0,
    seed: int = 42,
    save_path: str | None = None,
    plot_posterior: bool = True,
    plot_asd: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | Path | None = None,
    asd_path: str | Path | None = None,
    eval_asd: bool = True,
    pfn_device: str = "cuda",
    pfn_model_version: str = "auto",
    ignore_pretraining_limits: bool = True,
    posterior_pdf_mode: PdfMode = "tabpfn_bar",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    log_scale: bool | None = None,
    log_scale_qoi: Sequence[str] | None = None,
    logit_scale_qoi: Sequence[str] | None = None,
    logit_bounds: dict[str, tuple[float, float]] | None = None,
    log_offsets: dict[str, float] | None = None,
    filter_valid_labels: bool = False,
    coszen_override: float | None = None,
) -> dict:
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR
    names = parse_s4_task_names(task_names)
    emit_path = Path(data_path) if data_path is not None else _DEFAULT_DATA_PATH
    asd_file = Path(asd_path) if asd_path is not None else _DEFAULT_ASD_PATH
    if logit_bounds is None:
        logit_bounds = dict(S4_LOGIT_BOUNDS)

    set_seed(seed)

    X_np, Y_np, source_idx, data_meta = load_emit_s4_xy(
        emit_path,
        task_names=names,
        filter_valid_labels=filter_valid_labels,
    )
    n_filtered = int(X_np.shape[0])
    input_dim = int(X_np.shape[1])
    aux_indices = list(data_meta.get("aux_indices", []))
    force_train_local = force_train_local_from_asd_appended(emit_path, source_idx)
    if force_train_local.size:
        print(
            f"Forcing {force_train_local.size} ASD-appended row(s) into train "
            f"(source indices {source_idx[force_train_local].tolist()})"
        )
    train_local, val_local, test_local = split_emit_s4(
        n_filtered,
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        force_train_local=force_train_local,
    )
    n_test = int(test_local.size)
    train_idx = source_idx[train_local]
    val_idx = source_idx[val_local]
    test_idx = source_idx[test_local]
    wl_np = _load_wavelengths_nm(emit_path, None)

    if task_band_config is not None:
        band_lookup_names = [S4_BAND_CONFIG_ALIASES.get(n, n) for n in names]
        skipped = [n for n in band_lookup_names if n not in S2_TASK_NAMES]
        if skipped:
            raise ValueError(
                f"task_band_config cannot be used with non-S2 QoI names {skipped}. "
                "Drop them from QOI or leave task_band_config=None."
            )
        bands_by_lookup = load_task_band_config(
            task_band_config,
            task_names=band_lookup_names,
            input_dim=S4_SPECTRAL_DIM,
        )
        bands_by_task = {
            n: bands_by_lookup[S4_BAND_CONFIG_ALIASES.get(n, n)] for n in names
        }
    else:
        bands_by_task = {name: list(range(S4_SPECTRAL_DIM)) for name in names}

    title = f"S4_EMIT_ST_nTrain{n_train}_nVal{n_val}_nTest{n_test}_tabpfn_T{len(names)}"
    print("=" * 60)
    print(title)
    print(
        f"S4 TabPFN tasks={names}, input_dim={input_dim}, aux={aux_indices}, "
        f"pfn_device={pfn_device}, pfn_model_version={pfn_model_version}"
    )
    print(
        f"Split: filtered={n_filtered}  train={n_train}  val={n_val}  test={n_test}  "
        f"(data {emit_path})"
    )
    print("=" * 60)

    x_train_full = torch.as_tensor(X_np[train_local], dtype=torch.float32)
    y_train = torch.as_tensor(Y_np[train_local], dtype=torch.float32)
    x_test_full = torch.as_tensor(X_np[test_local], dtype=torch.float32)
    y_test = torch.as_tensor(Y_np[test_local], dtype=torch.float32)
    del X_np, Y_np
    x_test_orig = x_test_full.clone()

    warps = resolve_y_warps(
        log_scale_qoi,
        logit_scale_qoi,
        log_scale=log_scale,
        meta=data_meta,
        logit_bounds=logit_bounds,
        log_offsets=log_offsets,
    )
    log_scale = warps.log_scale
    log_scale_task_set = warps.log_tasks
    log_scale_source = warps.log_source
    logit_scale_task_set = warps.logit_tasks
    print(
        f"Output warps: log={sorted(log_scale_task_set) or '[]'} "
        f"(source={log_scale_source}); "
        f"logit={sorted(logit_scale_task_set) or '[]'} "
        f"(source={warps.logit_source})"
        + (
            f"; log_offsets={warps.active_log_offsets()}"
            if warps.active_log_offsets()
            else ""
        )
    )
    if task_band_config is not None:
        print(f"Task band config: {task_band_config}")

    asd_tasks = [n for n in names if n in ASD_TASK_NAMES and n in TASK_LABEL_MAP]
    asd_x_np: np.ndarray | None = None
    asd_y_by_task: dict[str, np.ndarray] = {}
    asd_aux_meta: dict | None = None
    coszen_asd_original = None
    if eval_asd:
        if not asd_file.is_file():
            print(f"ASD eval skipped: ASD NetCDF missing: {asd_file}")
            eval_asd = False
        elif not asd_tasks:
            print(
                "ASD eval skipped: none of the trained QoIs have ASD labels "
                f"(trained={names}, asd={sorted(ASD_TASK_NAMES)})"
            )
            eval_asd = False
        else:
            asd_x_np, asd_y_by_task, asd_aux_meta, coszen_asd_original = _load_asd_xy(
                asd_file,
                asd_tasks,
                coszen_override=coszen_override,
            )
            print(f"Loaded ASD validation set: n={asd_x_np.shape[0]} from {asd_file}")
            print(f"  ASD tasks: {asd_tasks}")

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
    logit_mu_all: list[np.ndarray] = []
    logit_sigma_all: list[np.ndarray] = []

    asd_per_task: dict[str, dict] = {}
    asd_y_true_all: list[np.ndarray] = []
    asd_y_pred_all: list[np.ndarray] = []
    asd_y_pred_mean_all: list[np.ndarray] = []
    asd_y_pred_mode_all: list[np.ndarray] = []
    asd_log_mu_all: list[np.ndarray] = []
    asd_log_sigma_all: list[np.ndarray] = []
    asd_y_std_all: list[np.ndarray] = []
    asd_lower_all: list[np.ndarray] = []
    asd_upper_all: list[np.ndarray] = []
    asd_coverage_by_task: dict[str, dict] = {}
    asd_task_order: list[str] = []

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        task_cols = _task_input_columns(band_indices, aux_indices=aux_indices)
        x_tr = select_bands(x_train_full, task_cols)
        x_te = select_bands(x_test_full, task_cols)
        x_tr_np = x_tr.detach().cpu().numpy().astype(np.float32)
        x_te_np = x_te.detach().cpu().numpy().astype(np.float32)
        y_tr_raw = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_tr = (
            forward_y_s2(y_tr_raw, task_name, warps=warps)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

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

        pred_mean, pred_std, lower, upper = _unpack_tabpfn_full(full)
        inv = _inverse_physical(
            pred_mean, pred_std, lower, upper, task_name=task_name, warps=warps
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
            warps=warps,
        )
        task_metrics[task_name] = dict(computed)
        spectral_wl = [
            float(wl_np[i]) if i < len(wl_np) else float("nan") for i in band_indices
        ]
        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": spectral_wl,
            "model_input_dim": int(x_tr.shape[-1]),
            "ard_space": "none",
            "input_column_indices": list(task_cols),
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
            logit_mu_all=logit_mu_all,
            logit_sigma_all=logit_sigma_all,
        )
        print_task_test_metrics(task_name, computed, warps=warps)

        if eval_asd and task_name in asd_tasks and asd_x_np is not None:
            print(f"  ASD predict: {task_name}")
            x_asd = asd_x_np[:, task_cols].astype(np.float32)
            t_asd = time.time()
            full_asd = regressor.predict(
                x_asd, output_type="full", quantiles=[0.025, 0.975]
            )
            asd_pred_time = time.time() - t_asd
            pm, ps, lo, hi = _unpack_tabpfn_full(full_asd)
            inv_asd = _inverse_physical(
                pm, ps, lo, hi, task_name=task_name, warps=warps
            )
            pm_t, ps_t, lo_t, hi_t = inv_asd.as_tuple()
            y_true = torch.as_tensor(asd_y_by_task[task_name], dtype=torch.float32)
            n_eval = int(y_true.numel())
            nan_col = np.full(n_eval, np.nan, dtype=np.float64)
            computed_asd = compute_metrics(
                y_true,
                pm_t,
                output_std=ps_t,
                lower_95=lo_t,
                upper_95=hi_t,
                prediction_time=asd_pred_time,
            )
            scalar = _scalar_error_metrics(
                y_true.numpy(),
                pm_t.detach().cpu().numpy(),
                prefix=f"{task_name}_",
            )
            cov_m = compute_prediction_coverage_metrics(
                y_true.numpy(),
                pm_t.numpy(),
                ps_t.numpy(),
            )
            asd_coverage_by_task[task_name] = cov_m
            y_pred_np = pm_t.detach().cpu().numpy()
            y_pred_mean_np = (
                inv_asd.point_mean.detach().cpu().numpy()
                if inv_asd.point_mean is not None
                else nan_col
            )
            y_pred_mode_np = (
                inv_asd.point_mode.detach().cpu().numpy()
                if inv_asd.point_mode is not None
                else nan_col
            )
            row = {
                "asd_label": TASK_LABEL_MAP[task_name],
                "n_eval": n_eval,
                "prediction_time_s": float(asd_pred_time),
                "y_true": y_true.numpy().tolist(),
                "y_pred": y_pred_np.tolist(),
                "y_pred_mean": y_pred_mean_np.tolist(),
                "y_pred_mode": y_pred_mode_np.tolist(),
                "y_std": ps_t.detach().cpu().numpy().tolist(),
                "lower_95": lo_t.detach().cpu().numpy().tolist(),
                "upper_95": hi_t.detach().cpu().numpy().tolist(),
                **cov_m,
                **computed_asd,
                **scalar,
            }
            asd_per_task[task_name] = row
            print(
                f"  ASD RMSE={computed_asd['RMSE']:.6g}  "
                f"RRMSE={computed_asd['RRMSE']:.6g}  "
                f"MAE={computed_asd['MAE']:.6g}  "
                f"R2={scalar[f'{task_name}_R2']:.4f}"
            )
            for i, (yt, yp) in enumerate(zip(row["y_true"], row["y_pred"])):
                extra = ""
                if warps.uses_log(task_name):
                    extra = (
                        f"  mean={y_pred_mean_np[i]:.6g}  mode={y_pred_mode_np[i]:.6g}"
                    )
                print(f"    sample {i}: true={yt:.6g}  pred={yp:.6g}{extra}")

            asd_task_order.append(task_name)
            asd_y_true_all.append(y_true.numpy())
            asd_y_pred_all.append(y_pred_np)
            asd_y_pred_mean_all.append(y_pred_mean_np)
            asd_y_pred_mode_all.append(y_pred_mode_np)
            asd_log_mu_all.append(
                inv_asd.log_mu.detach().cpu().numpy()
                if inv_asd.log_mu is not None
                else nan_col
            )
            asd_log_sigma_all.append(
                inv_asd.log_sigma.detach().cpu().numpy()
                if inv_asd.log_sigma is not None
                else nan_col
            )
            asd_y_std_all.append(ps_t.numpy())
            asd_lower_all.append(lo_t.numpy())
            asd_upper_all.append(hi_t.numpy())

        del regressor
        if pfn_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_pred_mean_stacked = stack_or_none(y_pred_mean_all)
    y_pred_mode_stacked = stack_or_none(y_pred_mode_all)
    log_mu_stacked = stack_or_none(log_mu_all)
    log_sigma_stacked = stack_or_none(log_sigma_all)
    logit_mu_stacked = stack_or_none(logit_mu_all)
    logit_sigma_stacked = stack_or_none(logit_sigma_all)
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
    log_task_names, aggregate_rrmse_log, aggregate_rrmse_mean = (
        attach_log_scale_aggregate_fields(
            per_task,
            task_metrics,
            names=names,
            warps=warps,
        )
    )
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    aggregate_medae = macro_metric(per_task, names, "MedAE")

    metrics: dict = {
        "title": title,
        "dataset": "s4",
        "input_dim_original": S4_INPUT_DIM,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "model_class": "TabPFNRegressor",
        "pfn_device": pfn_device,
        "pfn_model_version": pfn_model_version,
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "posterior_pdf_mode": posterior_pdf_mode,
        "data_path": str(emit_path.resolve()),
        "log_scale": bool(log_scale),
        "log_scale_tasks": [n for n in names if task_uses_log_scale(n, warps=warps)],
        "log_scale_source": log_scale_source,
        "log_offsets": dict(warps.log_offsets),
        "logit_scale_tasks": list(logit_task_names),
        "logit_scale_source": warps.logit_source,
        "logit_bounds": {
            k: list(v) for k, v in warps.logit_bounds.items() if k in logit_scale_task_set
        },
        "rel_tolerance": rel_tolerance,
        "task_band_config": task_band_config,
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wl_np),
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

    Path(save_path).mkdir(parents=True, exist_ok=True)
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
        wavelengths_np=wl_np,
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
        test_idx=np.asarray(test_idx),
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
        n_val=n_val,
        save_metrics_json=save_metrics_json,
        y_pred_mean_stacked=y_pred_mean_stacked,
        y_pred_mode_stacked=y_pred_mode_stacked,
        log_mu_stacked=log_mu_stacked,
        log_sigma_stacked=log_sigma_stacked,
        warps=warps,
        logit_mu_stacked=logit_mu_stacked,
        logit_sigma_stacked=logit_sigma_stacked,
    )

    if eval_asd and asd_task_order and asd_x_np is not None:
        flat_metrics: dict[str, float] = {}
        for name, row in asd_per_task.items():
            for key, value in row.items():
                if key.startswith(f"{name}_") or key in {
                    "RMSE",
                    "MAE",
                    "MedAE",
                    "RRMSE",
                    "R2",
                    "NLPD",
                    "NIS",
                    "PICP_95",
                    "MPIW_95",
                }:
                    if key.startswith(f"{name}_"):
                        flat_metrics[key] = value
                    else:
                        flat_metrics[f"{name}_{key}"] = value

        aggregate_asd_rrmse = macro_metric(flat_metrics, asd_task_order, "RRMSE")
        aggregate_asd_r2 = macro_metric(flat_metrics, asd_task_order, "R2")
        missing_tasks = sorted(set(TASK_LABEL_MAP) - set(asd_task_order))
        asd_results = {
            "title": f"asd_validation_eval_{Path(save_path).name}",
            "ckpt_dir": str(Path(save_path).resolve()),
            "asd_path": str(asd_file.resolve()),
            "task_names": asd_task_order,
            "missing_tasks": missing_tasks,
            "checkpoint_title": title,
            "model_class": "TabPFNRegressor",
            "log_scale_tasks": sorted(warps.log_tasks),
            "logit_scale_tasks": sorted(warps.logit_tasks),
            "log_offsets": dict(warps.log_offsets),
            "label_map": {t: TASK_LABEL_MAP[t] for t in asd_task_order},
            "aux_meta": asd_aux_meta,
            "coszen_override": coszen_override,
            "coszen_asd_original": (
                coszen_asd_original.tolist()
                if coszen_asd_original is not None
                else None
            ),
            "aggregate_RRMSE": aggregate_asd_rrmse,
            "aggregate_R2": aggregate_asd_r2,
            "per_task": asd_per_task,
            "coverage_by_task": asd_coverage_by_task,
            **flat_metrics,
        }
        eval_dir = Path(save_path) / "asd_validation_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        json_path = eval_dir / "asd_validation_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                asd_results,
                f,
                indent=2,
                default=lambda o: float(o) if hasattr(o, "item") else o,
            )
        npz_path = eval_dir / "asd_validation_predictions.npz"
        np.savez_compressed(
            npz_path,
            task_names=np.array(asd_task_order),
            y_true=np.column_stack(asd_y_true_all),
            y_pred=np.column_stack(asd_y_pred_all),
            y_pred_mean=np.column_stack(asd_y_pred_mean_all),
            y_pred_mode=np.column_stack(asd_y_pred_mode_all),
            log_mu=np.column_stack(asd_log_mu_all),
            log_sigma=np.column_stack(asd_log_sigma_all),
            y_std=np.column_stack(asd_y_std_all),
            lower_95=np.column_stack(asd_lower_all),
            upper_95=np.column_stack(asd_upper_all),
            x_spectral=asd_x_np[:, :S4_SPECTRAL_DIM],
        )
        print(f"\nASD Aggregate RRMSE={aggregate_asd_rrmse:.6f}  R2={aggregate_asd_r2:.6f}")
        print(f"Saved ASD metrics: {json_path}")
        print(f"Saved ASD predictions: {npz_path}")
        metrics["asd_validation"] = {
            "aggregate_RRMSE": aggregate_asd_rrmse,
            "aggregate_R2": aggregate_asd_r2,
            "task_names": asd_task_order,
            "eval_dir": str(eval_dir),
        }

        if plot_asd:
            print("\nGenerating ASD validation plots...")
            from experiments_SORF._plot_s4_asd_validation import (
                generate_asd_validation_plots,
            )

            plot_paths = generate_asd_validation_plots(
                eval_dir,
                asd_path=asd_file,
                out_dir=eval_dir / "plots",
            )
            print(f"Saved {len(plot_paths)} ASD plots under {eval_dir / 'plots'}")

    return metrics


__all__ = [
    "PFN_MODEL_VERSION_CHOICES",
    "S4_TASK_NAMES",
    "parse_s4_task_names",
    "run_s4_emit_tabpfn",
]
