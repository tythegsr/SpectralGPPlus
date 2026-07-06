"""
Shared TOA benchmark runner using independent TabPFN regressors (one per task).

Trains two TabPFNRegressor models on y_cos and y_grain with shared raw inputs
and physical-scale targets (no X/Y scaling or log transforms).
Requires tabpfn >= 6.0 for n_train=49000 (TabPFN v2.5+).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIR = "experiments_TabPFN/results/toa_tabpfn"

PFN_MODEL_VERSION_CHOICES = ("auto", "v2.5", "v3.0")
PdfMode = Literal["gaussian", "tabpfn_bar"]


if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _TABPFN_DIR)

from experiments_toa.data import load_toa_data
from gpplus.utils import compute_metrics, set_seed
from mtgpr_experiment_utils import (
    compute_relative_error_metrics,
    format_relative_error_summary,
    save_metrics_json,
    unpack_train_val_test,
)
from plot_toa_posterior import (
    plot_toa_posterior_figures,
    save_predictions_npz,
    select_posterior_example_indices,
    wavelength_axis,
)
from toa_mtgpr_base import (
    TASK_NAMES,
    TOA_INPUT_DIM,
    compute_per_task_metrics,
    drop_input_columns,
    normalize_columns_to_drop,
)
from toa_y_transform import inverse_y_single

NUM_TASKS = 2


def make_tabpfn_regressor(
    *,
    pfn_device: str,
    pfn_model_version: str = "auto",
    seed: int = 42,
    ignore_pretraining_limits: bool = False,
):
    """Construct TabPFNRegressor for v2.5/v3.0 when available, else auto/default."""
    from tabpfn import TabPFNRegressor

    common: dict = {
        "device": pfn_device,
        "random_state": seed,
        "ignore_pretraining_limits": ignore_pretraining_limits,
    }
    ver = pfn_model_version.lower().replace("_", ".")
    if ver == "auto":
        return TabPFNRegressor(**common)

    try:
        from tabpfn.model_loading import ModelVersion

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": getattr(ModelVersion, "V2_5", None),
            "v3": getattr(ModelVersion, "V3", None),
            "v3.0": getattr(ModelVersion, "V3", None),
        }
        model_version = version_map.get(ver)
        if model_version is not None and hasattr(TabPFNRegressor, "create_default_for_version"):
            return TabPFNRegressor.create_default_for_version(model_version, **common)
    except (ImportError, AttributeError, ValueError):
        pass

    if ver in ("v2.5", "v3", "v3.0"):
        warnings.warn(
            f"TabPFN {pfn_model_version!r} selector unavailable in this tabpfn build; "
            "using model_path='auto' (default checkpoint for installed package).",
            stacklevel=2,
        )
        return TabPFNRegressor(model_path="auto", **common)

    raise ValueError(
        f"pfn_model_version must be one of {PFN_MODEL_VERSION_CHOICES}, got {pfn_model_version!r}"
    )


def _eval_tabpfn_task(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_train_phys: np.ndarray,
    y_test_phys: np.ndarray,
    *,
    task_name: str,
    regressor,
) -> dict:
    """Fit TabPFN on one scalar task; return predictions in physical units + bar-dist artifacts."""
    y_tr_np = np.asarray(y_train_phys, dtype=np.float32).ravel()

    t_train = time.time()
    regressor.fit(x_train_np, y_tr_np)
    train_time = time.time() - t_train

    t_pred = time.time()
    full = regressor.predict(
        x_test_np,
        output_type="full",
        quantiles=[0.025, 0.975],
    )
    prediction_time = time.time() - t_pred

    logits = full["logits"]
    criterion = full["criterion"]
    if isinstance(logits, np.ndarray):
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
    else:
        logits_t = logits.detach().float()
    if logits_t.ndim > 2:
        logits_t = logits_t.reshape(logits_t.shape[0], -1)

    pred_mean = full.get("mean")
    if pred_mean is None:
        raise RuntimeError("TabPFN full predict missing 'mean'")
    if isinstance(pred_mean, torch.Tensor):
        pred_mean_t = pred_mean.detach().cpu().float().reshape(-1)
    else:
        pred_mean_t = torch.as_tensor(pred_mean, dtype=torch.float32).reshape(-1)

    lower_t = upper_t = None
    if "quantiles" in full:
        q = full["quantiles"]
        if isinstance(q, list) and len(q) >= 2:
            lo, hi = q[0], q[1]
            lower_t = torch.as_tensor(lo, dtype=torch.float32).reshape(-1)
            upper_t = torch.as_tensor(hi, dtype=torch.float32).reshape(-1)

    pred_std_t = None
    if hasattr(criterion, "variance"):
        # criterion.borders live on pfn_device (e.g. cuda); logits must match.
        borders = getattr(criterion, "borders", None)
        logits_for_var = logits_t
        if borders is not None and logits_for_var.device != borders.device:
            logits_for_var = logits_for_var.to(device=borders.device)
        variance = criterion.variance(logits_for_var)
        pred_std_t = torch.sqrt(variance.detach().cpu().float()).reshape(-1)

    logits_t = logits_t.detach().cpu()

    if lower_t is None or upper_t is None:
        if pred_std_t is None:
            pred_std_t = torch.full_like(pred_mean_t, float("nan"))
        lower_t = pred_mean_t - 2.0 * pred_std_t
        upper_t = pred_mean_t + 2.0 * pred_std_t
    elif pred_std_t is None:
        pred_std_t = (upper_t - lower_t) / 4.0

    pred_mean_phys, pred_std_phys, lower_phys, upper_phys = inverse_y_single(
        pred_mean_t,
        pred_std_t,
        lower_t,
        upper_t,
        task_name=task_name,
        y_scaler=None,
        standardize_y=False,
        log_grain=False,
    )

    borders = criterion.borders.detach().cpu().numpy()
    logits_np = logits_t.detach().cpu().numpy()

    y_test_eval = torch.as_tensor(y_test_phys, dtype=torch.float64).reshape(-1)
    computed = compute_metrics(
        y_test_eval,
        pred_mean_phys,
        output_std=pred_std_phys,
        lower_95=lower_phys,
        upper_95=upper_phys,
        training_time=train_time,
        prediction_time=prediction_time,
        tabpfn_logits=logits_t,
        tabpfn_bar_dist=criterion,
        y_test_normalized=y_test_eval,
    )

    return {
        "metrics": computed,
        "train_time": train_time,
        "prediction_time": prediction_time,
        "pred_mean": pred_mean_phys.detach().cpu().numpy(),
        "pred_std": pred_std_phys.detach().cpu().numpy(),
        "lower": lower_phys.detach().cpu().numpy(),
        "upper": upper_phys.detach().cpu().numpy(),
        "logits": logits_np,
        "borders": borders,
        "train_scale": "physical",
    }


def run_toa_tabpfn(
    n_train: int = 49000,
    n_test: int = 5000,
    seed: int = 42,
    save_path: str | None = None,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    drop_columns: list[int] | None = None,
    pfn_device: str = "cuda",
    pfn_model_version: str = "auto",
    ignore_pretraining_limits: bool = False,
    posterior_pdf_mode: PdfMode = "tabpfn_bar",
) -> dict:
    """Train independent TabPFN regressors on TOA data and evaluate on held-out test points."""
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR

    set_seed(seed)
    title = f"TOA_nTrain{n_train}_nTest{n_test}_tabpfn"

    print("=" * 60)
    print(title)
    print(
        f"TabPFN (independent tasks), pfn_device={pfn_device}, "
        f"model_version={pfn_model_version}, tasks={TASK_NAMES}, y_scale=physical (no transforms)"
    )
    print(f"n_train={n_train}, n_test={n_test}, posterior_pdf_mode={posterior_pdf_mode}")
    print("=" * 60)

    data = load_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=0,
        seed=seed,
        data_path=data_path,
    )
    x_train, y_train, _x_val, _y_val, x_test, y_test, _train_idx, _val_idx, test_idx = (
        unpack_train_val_test(data)
    )

    x_test_orig = x_test.clone()
    input_dim_original = TOA_INPUT_DIM
    dropped_columns = normalize_columns_to_drop(drop_columns, input_dim=input_dim_original)
    x_train, kept_column_indices_t = drop_input_columns(
        x_train, dropped_columns, input_dim=input_dim_original
    )
    x_test, _ = drop_input_columns(x_test, dropped_columns, input_dim=input_dim_original)
    input_dim = int(x_train.shape[-1])
    kept_column_indices = kept_column_indices_t.tolist()
    if dropped_columns:
        print(
            f"Input columns: dropped {len(dropped_columns)} / {input_dim_original} "
            f"-> input_dim={input_dim}"
        )

    x_train_np = x_train.detach().cpu().numpy().astype(np.float32)
    x_test_np = x_test.detach().cpu().numpy().astype(np.float32)
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    borders_all: list[np.ndarray] = []
    train_scales: list[str] = []
    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}

    for task_idx, task_name in enumerate(TASK_NAMES):
        print(f"\n--- Task: {task_name} ---")
        regressor = make_tabpfn_regressor(
            pfn_device=pfn_device,
            pfn_model_version=pfn_model_version,
            seed=seed,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )
        result = _eval_tabpfn_task(
            x_train_np,
            x_test_np,
            y_train_np[:, task_idx],
            y_test_np[:, task_idx],
            task_name=task_name,
            regressor=regressor,
        )
        total_train_time += result["train_time"]
        total_prediction_time += result["prediction_time"]
        task_metrics[task_name] = result["metrics"]
        y_pred_all.append(result["pred_mean"])
        y_std_all.append(result["pred_std"])
        lower_all.append(result["lower"])
        upper_all.append(result["upper"])
        logits_all.append(result["logits"])
        borders_all.append(result["borders"])
        train_scales.append(result["train_scale"])

        computed = result["metrics"]
        print(
            f"{task_name} Test RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}"
        )
        print(f"{task_name} train time: {result['train_time']:.1f}s  predict: {result['prediction_time']:.1f}s")

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    tabpfn_logits_stacked = np.stack(logits_all, axis=1)
    tabpfn_borders_stacked = np.stack(borders_all, axis=0)
    tabpfn_train_scale = np.array(train_scales)

    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked)
    for t, name in enumerate(TASK_NAMES):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t],
            y_pred_stacked[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))
    metrics: dict = {
        "title": title,
        "input_dim": input_dim,
        "input_dim_original": input_dim_original,
        "dropped_columns": dropped_columns,
        "kept_column_indices": kept_column_indices,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": NUM_TASKS,
        "task_names": list(TASK_NAMES),
        "model_class": "TabPFNRegressor",
        "pfn_device": pfn_device,
        "pfn_model_version": pfn_model_version,
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "posterior_pdf_mode": posterior_pdf_mode,
        "standardize_x": False,
        "standardize_y": False,
        "log_grain": False,
        "rel_tolerance": rel_tolerance,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "RMSE": aggregate_rmse,
        **per_task,
    }

    for task_name in TASK_NAMES:
        tm = task_metrics[task_name]
        for key, value in tm.items():
            metrics[f"{task_name}_{key}"] = value

    print(f"\nTest aggregate RMSE: {aggregate_rmse:.6f}")
    for name in TASK_NAMES:
        print(
            f"{name} RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"RRMSE: {per_task[f'{name}_RRMSE']:.6f}"
        )
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
    print(f"Total training time: {total_train_time:.1f}s")

    if save_path:
        example_indices = select_posterior_example_indices(
            y_test_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )

        npz_pdf_mode = posterior_pdf_mode if posterior_pdf_mode != "gaussian" else None
        if posterior_pdf_mode == "tabpfn_bar":
            npz_pdf_mode = "tabpfn_bar"

        out_npz = save_predictions_npz(
            save_path,
            title,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            y_std=y_std_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
            x_test_orig=x_test_orig.numpy(),
            test_idx=test_idx.cpu().numpy(),
            task_names=TASK_NAMES,
            seed=seed,
            rel_tolerance=rel_tolerance,
            example_indices=example_indices,
            wavelength_nm=wavelength_axis(x_test_orig.shape[-1]),
            log_grain=False,
            posterior_pdf_mode=npz_pdf_mode,
            tabpfn_logits=tabpfn_logits_stacked if posterior_pdf_mode == "tabpfn_bar" else None,
            tabpfn_borders=tabpfn_borders_stacked if posterior_pdf_mode == "tabpfn_bar" else None,
            tabpfn_train_scale=tabpfn_train_scale if posterior_pdf_mode == "tabpfn_bar" else None,
        )
        print(f"Saved predictions to {out_npz}")
        metrics["predictions_npz"] = out_npz

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_posterior and example_indices:
            import logging

            from plot_validation_curves import sanitize_plot_subdir

            post_dir = Path(save_path) / "plots" / "posterior" / sanitize_plot_subdir(title)
            try:
                post_paths = plot_toa_posterior_figures(
                    x_test_orig.numpy(),
                    y_test_np,
                    y_pred_stacked,
                    y_std_stacked,
                    lower_stacked,
                    upper_stacked,
                    post_dir,
                    title=title,
                    example_indices=example_indices,
                    rel_metrics_by_task=rel_metrics_by_task,
                    rel_tolerance=rel_tolerance,
                    wavelength_nm=wavelength_axis(x_test_orig.shape[-1]),
                    log_grain=False,
                    pdf_mode=posterior_pdf_mode,
                    tabpfn_logits=tabpfn_logits_stacked if posterior_pdf_mode == "tabpfn_bar" else None,
                    tabpfn_borders=tabpfn_borders_stacked if posterior_pdf_mode == "tabpfn_bar" else None,
                    tabpfn_train_scale=tabpfn_train_scale if posterior_pdf_mode == "tabpfn_bar" else None,
                    posterior_pdf_mode=npz_pdf_mode,
                )
                for plot_path in post_paths:
                    print(f"Saved posterior plot to {plot_path}")
            except Exception as exc:
                logging.getLogger(__name__).warning("Posterior plot generation failed: %s", exc)

    return metrics
