"""Train PCA+SVGP on fsnow 70–100% sim data, then evaluate on EMIT snow subset.

Uses SMAC full-tune incumbent hyperparameters
(``results/Aug15/smac_s2_svgp_algae/incumbent.json``) with ``num_epochs=200``. Trains independent SVGP models for algae, grain_size, and
fsnow, saves checkpoints, then predicts on ``split_files/emit_data_snow_70to100.nc``.

Examples::

    python experiments_SVGP/S2_toa_SVGP_snow70_train_predict_emit.py
    python experiments_SVGP/S2_toa_SVGP_snow70_train_predict_emit.py --train-only
    python experiments_SVGP/S2_toa_SVGP_snow70_train_predict_emit.py --predict-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SVGP_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_PCA_DIR = _ROOT / "experiments_PCA"

for _p in (_ROOT, _GP_DIR, _MTGPR_DIR, _RFF_DIR, _PCA_DIR, _SVGP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_toa.export_emit_as_s2 import mapped_qois
from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
from experiments_toa.paths import pin_toa_import_paths
from experiments_toa.s2_plotting import (
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import _scalar_error_metrics, apply_x_transform, macro_metric, select_bands
from experiments_toa.s2_y_transform import inverse_y_s2, resolve_y_warps
from gp_experiment_utils import DEFAULT_ADAM_KWARGS
from gpplus.training import evaluate_svgp_gp_model
from gpplus.utils import compute_metrics
from toa_pca_utils import pca_from_dict, transform_pca
from toa_s2_gp_base import run_s2_toa_gp
from toa_svgp_checkpoint import load_toa_svgp_checkpoint

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR)

import gpplus

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
TASKS: list[str] = ["algae", "grain_size", "fsnow"]
TRAIN_DATA = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261808.nc"
EMIT_PATH = _ROOT / "split_files" / "emit_data_snow_70to100.nc"
INCUMBENT_JSON = (
    _ROOT / "experiments_SVGP" / "results" / "Aug15" / "smac_s2_svgp_algae" / "incumbent.json"
)
SAVE_ROOT = _ROOT / "experiments_SVGP" / "results" / "snow70_svgp_pca_smac_ep200"
TASK_BAND_CONFIG = _ROOT / "experiments_toa" / "configs" / "s2_task_bands_all.json"
INPUT_VARIABLE = "toa_reflectance"
N_TRAIN = 100_000
N_TEST = 10_000
NUM_EPOCHS = 800
LOG_EVERY_N_EPOCHS = 50
SEED = 42
DEVICE = "cuda"
DTYPE = "float32"
PREDICT_CHUNK_SIZE = 4096
MAX_EMIT_SAMPLES = 0  # 0 = no cap after filtering
NONDEFAULT_ONLY = True
FILTER_VALID_LABELS = True
PLOT = True
PLOT_POSTERIOR = True
POSTERIOR_N_EXAMPLES = 20
SCATTER_MAX_POINTS = 20000  # subsample EMIT scatter for readability; 0 = all
LOG_LEVEL = "INFO"
# ---------------------------------------------------------------------------

TASK_VALID_Y_RANGE: dict[str, tuple[float, float]] = {
    "algae": (1e-2, 6e5),
    "grain_size": (30.0, 1500.0),
    "fsnow": (0.7, 1.0),
}


def load_incumbent(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    inc = dict(data["incumbent"])
    snippet = data.get("ide_snippet") or {}
    return {
        "lr": float(inc["lr"]),
        "variational_lr": float(inc.get("variational_lr", inc["lr"])),
        "num_inducing": int(inc["num_inducing"]),
        "batch_size": int(inc["batch_size"]),
        "kl_beta": float(inc["kl_beta"]),
        "adam_stop_patience": int(inc["adam_stop_patience"]),
        "n_pca_components": int(inc["n_pca_components"]),
        "n_train": int(snippet.get("N_TRAIN", N_TRAIN)),
        "n_test": int(snippet.get("N_TEST", N_TEST)),
        "variant": str(inc.get("variant", "pca")),
    }


def run_train_phase(
    *,
    save_root: Path,
    incumbent_path: Path,
    tasks: list[str],
    data_path: Path,
    num_epochs: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    hp = load_incumbent(incumbent_path)
    if hp["variant"] != "pca":
        raise ValueError(f"Expected PCA variant in incumbent, got {hp['variant']!r}")

    save_root.mkdir(parents=True, exist_ok=True)
    print(f"Training SVGP+PCA on {data_path}")
    print(f"  tasks={tasks}  epochs={num_epochs}  save_root={save_root}")
    print(f"  incumbent={incumbent_path}")

    metrics = run_s2_toa_gp(
        svgp=True,
        n_train=hp["n_train"],
        n_test=hp["n_test"],
        num_inits=1,
        num_epochs=num_epochs,
        seed=seed,
        device=device,
        dtype=dtype,
        save_path=str(save_root),
        save_checkpoint=True,
        task_names=tasks,
        data_path=str(data_path),
        input_variable=INPUT_VARIABLE,
        task_band_config=str(TASK_BAND_CONFIG),
        x_transform="none",
        log_scale_qoi=[],
        logit_scale_qoi=[],
        n_pca_components=hp["n_pca_components"],
        nigp=False,
        num_inducing=hp["num_inducing"],
        batch_size=hp["batch_size"],
        variational_lr=hp["variational_lr"],
        kl_beta=hp["kl_beta"],
        adam_stop_patience=hp["adam_stop_patience"],
        optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": hp["lr"]},
        predict_chunk_size=PREDICT_CHUNK_SIZE,
        log_every_n_epochs=LOG_EVERY_N_EPOCHS,
        monitor_validation=True,
        plot_validation=PLOT,
        plot_posterior=PLOT and PLOT_POSTERIOR,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        ard=True,
        learn_inducing_locations=True,
    )
    return metrics


def _find_checkpoints(ckpt_dir: Path, tasks: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        stem = path.stem
        for name in sorted(tasks, key=len, reverse=True):
            if stem.endswith("_" + name):
                out[name] = path
                break
    missing = [t for t in tasks if t not in out]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for {missing} under {ckpt_dir}")
    return out


def _load_emit_xy(
    emit_path: Path,
    *,
    task_names: list[str],
    max_samples: int | None,
    seed: int,
    nondefault_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with h5py.File(emit_path, "r") as f:
        n_total = int(f["reflectance"].shape[0])
        state = np.asarray(f["state"][:], dtype=np.float64)
        if "state_feature_names" in f.attrs:
            names_attr = f.attrs["state_feature_names"]
            if isinstance(names_attr, bytes):
                names_attr = names_attr.decode("utf-8")
            state_names = [s.strip() for s in str(names_attr).split(",")]
        else:
            state_names = list(EMIT_STATE_FEATURE_NAMES)

        if nondefault_only:
            grain_col = EMIT_STATE_FEATURE_NAMES.index("grain_radius")
            mask = np.abs(state[:, grain_col] - 500.0) > 0.1
        else:
            mask = np.ones(n_total, dtype=bool)

        idx = np.flatnonzero(mask)
        if max_samples is not None and idx.size > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_samples, replace=False))

        X = np.asarray(f["reflectance"][idx], dtype=np.float64)
        sample = np.asarray(f["sample"][idx], dtype=np.int64)

    qoi = mapped_qois(state[idx])
    cols = [qoi[name] for name in task_names]
    Y = np.column_stack(cols)

    meta = {
        "emit_path": str(emit_path.resolve()),
        "n_total": n_total,
        "n_eval": int(idx.size),
        "nondefault_only": bool(nondefault_only),
        "max_samples": max_samples,
        "seed": seed,
        "state_feature_names": state_names,
        "task_names": list(task_names),
        "fsnow_label": "softmax_fraction(z_snow,z_pv,z_npv,z_soil)",
        "task_valid_y_range": {
            k: list(TASK_VALID_Y_RANGE[k]) for k in task_names if k in TASK_VALID_Y_RANGE
        },
        "sample_idx_min": int(sample.min()) if sample.size else None,
        "sample_idx_max": int(sample.max()) if sample.size else None,
    }
    return X, Y, idx, meta


def _valid_label_mask(y: np.ndarray, task_name: str) -> np.ndarray:
    bounds = TASK_VALID_Y_RANGE.get(task_name)
    if bounds is None:
        return np.ones(y.shape[0], dtype=bool)
    lo, hi = bounds
    return (y >= lo) & (y <= hi)


def _load_wavelengths_nm(emit_path: Path, train_data: Path) -> np.ndarray:
    for path in (emit_path, train_data):
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            if "wl" in f:
                return np.asarray(f["wl"][:], dtype=np.float64)
    raise FileNotFoundError(
        f"No 'wl' variable in {emit_path} or {train_data}; needed for posterior plots."
    )


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_points), replace=False))


def _save_emit_eval_plots(
    *,
    out_dir: Path,
    tasks: list[str],
    x_np: np.ndarray,
    wavelengths_nm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    valid_mask: np.ndarray,
    seed: int,
    posterior_n_examples: int,
    scatter_max_points: int,
    title: str,
) -> None:
    scatter_dir = out_dir / "plots" / "scatter"
    post_dir = out_dir / "plots" / "posterior"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    for t, name in enumerate(tasks):
        valid = valid_mask[:, t].astype(bool) & np.isfinite(y_true[:, t]) & np.isfinite(
            y_pred[:, t]
        )
        if not np.any(valid):
            print(f"  skip scatter for {name}: no valid rows")
            continue
        yt = y_true[valid, t]
        yp = y_pred[valid, t]
        lo = lower[valid, t]
        hi = upper[valid, t]
        pick = _subsample_indices(yt.shape[0], scatter_max_points, seed + t)
        n_plot = int(pick.size)
        n_all = int(yt.shape[0])
        extra = f" (n={n_plot}/{n_all})" if n_plot < n_all else f" (n={n_all})"
        path = plot_s2_task_scatter(
            y_true=yt[pick],
            y_pred=yp[pick],
            lower=lo[pick],
            upper=hi[pick],
            task_name=name,
            out_path=scatter_dir / f"{name}_scatter.png",
            title=f"{title} | {name}{extra}",
        )
        print(f"Saved scatter: {path}")

    row_ok = (
        valid_mask.astype(bool)
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(y_std)
    ).all(axis=1)
    pool = np.flatnonzero(row_ok)
    if posterior_n_examples <= 0:
        return
    if pool.size == 0:
        print("  skip posterior examples: no rows valid for all tasks")
        return
    local = select_posterior_example_indices(
        int(pool.size),
        posterior_n_examples,
        seed=seed,
    )
    example_indices = [int(pool[i]) for i in local]
    post_paths = plot_s2_posterior_examples(
        x_test=x_np,
        wavelengths_nm=wavelengths_nm,
        y_true=y_true,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        task_names=tasks,
        example_indices=example_indices,
        save_dir=post_dir,
        title=title,
        y_std=y_std,
        log_scale_tasks=[],
        spectrum_ylabel="Reflectance",
    )
    for p in post_paths[:3]:
        print(f"Saved posterior plot: {p}")
    if len(post_paths) > 3:
        print(f"  ... and {len(post_paths) - 3} more under {post_dir}")


def _preprocess_emit_x(
    x_np: np.ndarray,
    bundle,
    *,
    device: str,
) -> torch.Tensor:
    x = torch.as_tensor(x_np, dtype=bundle.dtype)
    if bundle.input_column_indices is not None:
        cols = bundle.input_column_indices.to(dtype=torch.long)
        if cols.numel() < x.shape[-1]:
            x = select_bands(x, cols.tolist())
    x = apply_x_transform(x, bundle.x_transform)
    if bundle.pca_meta is not None:
        pca_fit = pca_from_dict(bundle.pca_meta)
        x = transform_pca(pca_fit, x, dtype=bundle.dtype)
    if bundle.standardize_x and bundle.x_scaler is not None:
        x = bundle.x_scaler.transform(x)
    return x.to(device=device, dtype=bundle.dtype)


def evaluate_svgp_checkpoints_on_emit(
    ckpt_dir: Path,
    emit_path: Path,
    *,
    tasks: list[str],
    device: str = "cuda",
    predict_chunk_size: int = 4096,
    max_samples: int | None = None,
    seed: int = 42,
    nondefault_only: bool = True,
    filter_valid_labels: bool = True,
    save_dir: Path | None = None,
    plot: bool = True,
    plot_posterior: bool = True,
    posterior_n_examples: int = 20,
    scatter_max_points: int = 20000,
    train_data: Path | None = None,
) -> dict:
    ckpt_dir = Path(ckpt_dir)
    emit_path = Path(emit_path)
    if not emit_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {emit_path}")

    ckpts = _find_checkpoints(ckpt_dir, tasks)
    print(f"Evaluating tasks: {tasks}")
    print(f"Loading EMIT subset from {emit_path}")
    X_np, Y_np, eval_idx, data_meta = _load_emit_xy(
        emit_path,
        task_names=tasks,
        max_samples=max_samples,
        seed=seed,
        nondefault_only=nondefault_only,
    )
    print(
        f"  eval n={data_meta['n_eval']:,} / total={data_meta['n_total']:,} "
        f"(nondefault_only={nondefault_only})"
    )

    per_task: dict[str, dict] = {}
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    valid_mask_all: list[np.ndarray] = []

    for task_name in tasks:
        ckpt_path = ckpts[task_name]
        print(f"\n=== {task_name} <- {ckpt_path.name} ===")
        bundle = load_toa_svgp_checkpoint(ckpt_path, device=device)
        warps = resolve_y_warps(
            bundle.log_scale_qoi,
            bundle.logit_scale_qoi,
            log_scale=False,
        )
        t_col = tasks.index(task_name)
        y_true_full = torch.as_tensor(Y_np[:, t_col], dtype=bundle.dtype)
        x = _preprocess_emit_x(X_np, bundle, device=device)

        model = bundle.model
        model.eval()
        t0 = time.time()
        pred_mean, lower, upper, pred_std = evaluate_svgp_gp_model(
            model,
            x,
            chunk_size=predict_chunk_size,
        )
        pred_time = time.time() - t0

        inv = inverse_y_s2(
            pred_mean.detach().cpu(),
            pred_std.detach().cpu(),
            lower.detach().cpu(),
            upper.detach().cpu(),
            task_name=task_name,
            y_scaler=bundle.y_scaler,
            standardize_y=bundle.standardize_y,
            warps=warps,
            extended=True,
        )
        pred_mean_c, pred_std_c, lower_c, upper_c = inv.as_tuple()

        y_np = y_true_full.cpu().numpy()
        if filter_valid_labels:
            valid = _valid_label_mask(y_np, task_name)
        else:
            valid = np.ones(y_np.shape[0], dtype=bool)
        n_drop = int((~valid).sum())
        if n_drop:
            print(
                f"  dropping {n_drop}/{len(valid)} rows outside "
                f"{TASK_VALID_Y_RANGE.get(task_name)} for metrics"
            )

        y_true = y_true_full.cpu()[valid]
        pred_mean_m = pred_mean_c[valid]
        pred_std_m = pred_std_c[valid]
        lower_m = lower_c[valid]
        upper_m = upper_c[valid]

        computed = compute_metrics(
            y_true,
            pred_mean_m,
            output_std=pred_std_m,
            lower_95=lower_m,
            upper_95=upper_m,
            prediction_time=pred_time,
        )
        scalar = _scalar_error_metrics(
            y_true.numpy(),
            pred_mean_m.detach().cpu().numpy(),
            prefix=f"{task_name}_",
        )
        row = {
            "checkpoint": str(ckpt_path),
            "n_eval": int(y_true.numel()),
            "n_eval_before_label_filter": int(y_true_full.numel()),
            "n_dropped_invalid_label": n_drop,
            "prediction_time_s": float(pred_time),
            "best_train_loss": float(bundle.best_train_loss),
            **computed,
            **scalar,
        }
        per_task[task_name] = row
        print(
            f"  RMSE={computed['RMSE']:.6g}  RRMSE={computed['RRMSE']:.6g}  "
            f"MAE={computed['MAE']:.6g}  R2={scalar[f'{task_name}_R2']:.4f}  "
            f"n={int(y_true.numel())}  time={pred_time:.1f}s"
        )

        y_true_out = y_np.astype(np.float64, copy=True)
        y_pred_out = pred_mean_c.detach().cpu().numpy().astype(np.float64, copy=True)
        y_std_out = pred_std_c.detach().cpu().numpy().astype(np.float64, copy=True)
        lower_out = lower_c.detach().cpu().numpy().astype(np.float64, copy=True)
        upper_out = upper_c.detach().cpu().numpy().astype(np.float64, copy=True)
        if n_drop:
            y_true_out[~valid] = np.nan
            y_pred_out[~valid] = np.nan
            y_std_out[~valid] = np.nan
            lower_out[~valid] = np.nan
            upper_out[~valid] = np.nan
        y_true_all.append(y_true_out)
        y_pred_all.append(y_pred_out)
        y_std_all.append(y_std_out)
        lower_all.append(lower_out)
        upper_all.append(upper_out)
        valid_mask_all.append(valid.astype(np.uint8))

        del bundle, model, x, pred_mean, lower, upper, pred_std
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    flat_metrics: dict[str, Any] = {}
    for name, row in per_task.items():
        for k, v in row.items():
            if k.startswith(f"{name}_") or k in {
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
                if k.startswith(f"{name}_"):
                    flat_metrics[k] = v
                else:
                    flat_metrics[f"{name}_{k}"] = v

    aggregate_rrmse = macro_metric(flat_metrics, tasks, "RRMSE")
    aggregate_r2 = macro_metric(flat_metrics, tasks, "R2")

    results = {
        "title": f"emit_eval_{ckpt_dir.name}",
        "ckpt_dir": str(ckpt_dir.resolve()),
        "task_names": tasks,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_R2": aggregate_r2,
        "data_meta": data_meta,
        "per_task": per_task,
        **flat_metrics,
    }

    out_dir = Path(save_dir) if save_dir is not None else ckpt_dir / "emit_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)

    csv_path = out_dir / "per_task.csv"
    if per_task:
        fieldnames = ["task"] + sorted({k for row in per_task.values() for k in row})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name in tasks:
                writer.writerow({"task": name, **per_task[name]})

    npz_path = out_dir / "predictions.npz"
    y_true_stack = np.column_stack(y_true_all)
    y_pred_stack = np.column_stack(y_pred_all)
    y_std_stack = np.column_stack(y_std_all)
    lower_stack = np.column_stack(lower_all)
    upper_stack = np.column_stack(upper_all)
    valid_stack = np.column_stack(valid_mask_all)
    np.savez_compressed(
        npz_path,
        task_names=np.array(tasks),
        eval_idx=eval_idx,
        y_true=y_true_stack,
        y_pred=y_pred_stack,
        y_std=y_std_stack,
        lower=lower_stack,
        upper=upper_stack,
        valid_mask=valid_stack,
    )
    print(f"\nAggregate RRMSE={aggregate_rrmse:.6f}  R2={aggregate_r2:.6f}")
    print(f"Saved metrics: {json_path}")
    print(f"Saved per-task CSV: {csv_path}")
    print(f"Saved predictions: {npz_path}")

    if plot:
        wl = _load_wavelengths_nm(emit_path, Path(train_data) if train_data else TRAIN_DATA)
        _save_emit_eval_plots(
            out_dir=out_dir,
            tasks=tasks,
            x_np=X_np,
            wavelengths_nm=wl,
            y_true=y_true_stack,
            y_pred=y_pred_stack,
            y_std=y_std_stack,
            lower=lower_stack,
            upper=upper_stack,
            valid_mask=valid_stack,
            seed=seed,
            posterior_n_examples=posterior_n_examples if plot_posterior else 0,
            scatter_max_points=scatter_max_points,
            title=str(results["title"]),
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--predict-only", action="store_true")
    parser.add_argument("--save-root", type=Path, default=SAVE_ROOT)
    parser.add_argument("--incumbent", type=Path, default=INCUMBENT_JSON)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--emit-path", type=Path, default=EMIT_PATH)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"QoI tasks (default: {TASKS})",
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--predict-chunk-size", type=int, default=PREDICT_CHUNK_SIZE)
    parser.add_argument(
        "--max-emit-samples",
        type=int,
        default=MAX_EMIT_SAMPLES,
        help="Cap EMIT rows after filtering (0 = no cap)",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Include default-state EMIT rows (grain_radius near 500)",
    )
    parser.add_argument(
        "--no-filter-valid-labels",
        action="store_true",
        help="Do not drop out-of-range EMIT labels before scoring",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    gpplus.config.configure_logger(level=getattr(logging, args.log_level))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"
    elif device.startswith("cuda"):
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    dtype = torch.float32 if DTYPE == "float32" else torch.float64
    tasks = list(args.tasks) if args.tasks is not None else list(TASKS)
    max_emit = None if args.max_emit_samples == 0 else int(args.max_emit_samples)

    if not args.predict_only:
        run_train_phase(
            save_root=args.save_root,
            incumbent_path=args.incumbent,
            tasks=tasks,
            data_path=args.train_data,
            num_epochs=args.num_epochs,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if not args.train_only:
        evaluate_svgp_checkpoints_on_emit(
            args.save_root,
            args.emit_path,
            tasks=tasks,
            device=device,
            predict_chunk_size=args.predict_chunk_size,
            max_samples=max_emit,
            seed=args.seed,
            nondefault_only=not args.all_samples,
            filter_valid_labels=not args.no_filter_valid_labels,
            save_dir=args.save_root / "emit_eval",
            plot=PLOT,
            plot_posterior=PLOT and PLOT_POSTERIOR,
            posterior_n_examples=POSTERIOR_N_EXAMPLES,
            scatter_max_points=SCATTER_MAX_POINTS,
            train_data=args.train_data,
        )


if __name__ == "__main__":
    main()
