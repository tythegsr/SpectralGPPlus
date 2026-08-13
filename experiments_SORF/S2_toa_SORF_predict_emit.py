"""Evaluate saved S2 SORF checkpoints on merged emit_data.nc."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

import gpplus
from experiments_RFF.rff_gp_defaults import rff_eval_kwargs
from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
from experiments_toa.s2_y_transform import inverse_y_s2, resolve_y_warps
from experiments_toa.s2_utils import _scalar_error_metrics, macro_metric
from gpplus.training import evaluate_rff_gp_model
from gpplus.utils import compute_metrics
from toa_stgp_checkpoint import ToaStgpBundle, _dtype_from_str
from toa_mtgpr_checkpoint import CHECKPOINT_VERSION, scaler_from_dict
from gpplus.models import RFFGPR

# Checkpoint model_config stores run metadata; only constructor kwargs go to RFFGPR.
_RFFGPR_INIT_KEYS = {
    "num_rff",
    "ard",
    "rff_sampling",
    "correct_sorf",
    "spectral_kernel",
    "nigp",
    "batch_shape",
    "init_batch_size",
    "input_noise_init",
}

DEFAULT_CKPT_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug08"
    / "s2_toa_sorf_1inits_numrff1600_lr0.1_taskbandconfig_rbf_nigp_freezeepochnigp100_dtypefloat64"
)
DEFAULT_EMIT_PATH = _ROOT / "split_files" / "emit_data.nc"

# QoIs present in Aug08 float64 run / EMIT state mapping.
TASK_TO_STATE_COL = {
    "cos_i": 0,
    "grain_size": 2,
    "liquid_water": 3,
    "dust": 4,
    "algae": 5,
    "aot": 13,
    "cwv": 14,
}


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
            # Default vector dominates (~84%); keep physically active rows.
            mask = np.abs(state[:, 2] - 500.0) > 0.1
        else:
            mask = np.ones(n_total, dtype=bool)

        idx = np.flatnonzero(mask)
        if max_samples is not None and idx.size > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_samples, replace=False))

        X = np.asarray(f["reflectance"][idx], dtype=np.float64)
        sample = np.asarray(f["sample"][idx], dtype=np.int64)

    cols = []
    for name in task_names:
        if name not in TASK_TO_STATE_COL:
            raise KeyError(f"No EMIT state mapping for task {name!r}")
        col = TASK_TO_STATE_COL[name]
        cols.append(state[idx, col])
    Y = np.column_stack(cols)

    meta = {
        "emit_path": str(emit_path.resolve()),
        "n_total": n_total,
        "n_eval": int(idx.size),
        "nondefault_only": bool(nondefault_only),
        "max_samples": max_samples,
        "seed": seed,
        "state_feature_names": state_names,
        "task_to_state_col": {k: TASK_TO_STATE_COL[k] for k in task_names},
        "sample_idx_min": int(sample.min()) if sample.size else None,
        "sample_idx_max": int(sample.max()) if sample.size else None,
    }
    return X, Y, idx, meta


def _find_checkpoints(ckpt_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        stem = path.stem
        for name in sorted(TASK_TO_STATE_COL, key=len, reverse=True):
            if stem.endswith("_" + name):
                out[name] = path
                break
    return out


def _load_stgp_checkpoint(ckpt_path: Path, device: str) -> ToaStgpBundle:
    """Load STGP checkpoint, ignoring non-constructor model_config metadata."""
    path = Path(ckpt_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {payload.get('version')!r}")

    dtype = _dtype_from_str(payload["dtype"])
    raw_cfg = dict(payload["model_config"])
    model_config = {k: v for k, v in raw_cfg.items() if k in _RFFGPR_INIT_KEYS}
    train_x = payload["train_x"].to(dtype=dtype, device=device)
    train_y = payload["train_y"].to(dtype=dtype, device=device)

    model = RFFGPR(train_x, train_y, **model_config)
    ckpt_w = payload["state_dict"]["covar_module.base_kernel.randn_weights"]
    base = model._rff_kernel
    if tuple(base.randn_weights.shape) != tuple(ckpt_w.shape):
        base.register_buffer(
            "randn_weights",
            torch.empty(ckpt_w.shape, device=device, dtype=dtype),
        )
    model.load_state_dict(payload["state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.invalidate_feature_cache()

    x_scaler = scaler_from_dict(payload.get("x_scaler"))
    y_scaler = scaler_from_dict(payload.get("y_scaler"))
    if "input_column_indices" in payload:
        input_column_indices = payload["input_column_indices"].to(torch.int64)
    else:
        input_column_indices = torch.arange(train_x.shape[-1], dtype=torch.int64)

    return ToaStgpBundle(
        model=model,
        task_name=str(payload["task_name"]),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        standardize_x=bool(payload["standardize_x"]),
        standardize_y=bool(payload["standardize_y"]),
        x_standardize_method=int(payload["x_standardize_method"]),
        train_idx=payload["train_idx"],
        val_idx=payload["val_idx"],
        test_idx=payload["test_idx"],
        title=str(payload["title"]),
        seed=int(payload["seed"]),
        best_train_loss=float(payload["best_train_loss"]),
        n_train=int(payload["n_train"]),
        n_test=int(payload["n_test"]),
        n_val=int(payload["n_val"]),
        data_path=payload.get("data_path"),
        rel_tolerance=float(payload.get("rel_tolerance", 0.01)),
        dtype=dtype,
        log_grain=bool(payload.get("log_grain", False)),
        logit_cos=bool(payload.get("logit_cos", True)),
        input_column_indices=input_column_indices,
        rff_sampling=str(model_config.get("rff_sampling", "rff")),
    )


def evaluate_checkpoints_on_emit(
    ckpt_dir: Path,
    emit_path: Path,
    *,
    device: str = "cuda",
    predict_chunk_size: int = 512,
    max_samples: int | None = 50000,
    seed: int = 42,
    nondefault_only: bool = True,
    save_dir: Path | None = None,
) -> dict:
    ckpt_dir = Path(ckpt_dir)
    emit_path = Path(emit_path)
    if not emit_path.is_file():
        raise FileNotFoundError(f"Merged emit NetCDF not found: {emit_path}")

    ckpts = _find_checkpoints(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {ckpt_dir}")
    task_names = [t for t in TASK_TO_STATE_COL if t in ckpts]
    missing = [t for t in TASK_TO_STATE_COL if t not in ckpts]
    if missing:
        print(f"WARNING: missing checkpoints for {missing}")

    print(f"Loading emit subset from {emit_path}")
    X_np, Y_np, eval_idx, data_meta = _load_emit_xy(
        emit_path,
        task_names=task_names,
        max_samples=max_samples,
        seed=seed,
        nondefault_only=nondefault_only,
    )
    print(
        f"  eval n={data_meta['n_eval']:,} / total={data_meta['n_total']:,} "
        f"(nondefault_only={nondefault_only})"
    )

    # Training run used no y-warps (explicit empty lists).
    warps = resolve_y_warps(log_scale_qoi=[], logit_scale_qoi=[])

    per_task: dict[str, dict] = {}
    y_true_all = []
    y_pred_all = []
    y_std_all = []

    for task_name in task_names:
        ckpt_path = ckpts[task_name]
        print(f"\n=== {task_name} <- {ckpt_path.name} ===")
        bundle = _load_stgp_checkpoint(ckpt_path, device=device)
        t_col = task_names.index(task_name)
        y_true = torch.as_tensor(Y_np[:, t_col], dtype=bundle.dtype)
        x = torch.as_tensor(X_np, dtype=bundle.dtype)

        if bundle.input_column_indices is not None:
            cols = bundle.input_column_indices.to(dtype=torch.long)
            if cols.numel() < x.shape[-1]:
                x = x.index_select(-1, cols)

        if bundle.standardize_x and bundle.x_scaler is not None:
            x = bundle.x_scaler.transform(x)

        model = bundle.model
        model.eval()
        model.invalidate_feature_cache()
        x = x.to(device=model.train_inputs[0].device, dtype=bundle.dtype)

        t0 = time.time()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model,
            x,
            chunk_size=predict_chunk_size,
            **rff_eval_kwargs(bundle.dtype),
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
        computed = compute_metrics(
            y_true.cpu(),
            pred_mean_c,
            output_std=pred_std_c,
            lower_95=lower_c,
            upper_95=upper_c,
            prediction_time=pred_time,
        )
        # Also keep simple scalar metrics under task prefix for aggregation.
        scalar = _scalar_error_metrics(
            y_true.cpu().numpy(),
            pred_mean_c.detach().cpu().numpy(),
            prefix=f"{task_name}_",
        )
        row = {
            "checkpoint": str(ckpt_path),
            "n_eval": int(y_true.numel()),
            "prediction_time_s": float(pred_time),
            "best_train_loss": float(bundle.best_train_loss),
            **computed,
            **scalar,
        }
        per_task[task_name] = row
        print(
            f"  RMSE={computed['RMSE']:.6g}  RRMSE={computed['RRMSE']:.6g}  "
            f"MAE={computed['MAE']:.6g}  R2={scalar[f'{task_name}_R2']:.4f}  "
            f"time={pred_time:.1f}s"
        )

        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(pred_mean_c.detach().cpu().numpy())
        y_std_all.append(pred_std_c.detach().cpu().numpy())

        # Free GPU memory between tasks.
        del bundle, model, x, pred_mean, lower, upper, pred_std
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    flat_metrics = {}
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

    aggregate_rrmse = macro_metric(flat_metrics, task_names, "RRMSE")
    aggregate_r2 = macro_metric(flat_metrics, task_names, "R2")

    results = {
        "title": "emit_eval_aug08_sorf_float64",
        "ckpt_dir": str(ckpt_dir.resolve()),
        "task_names": task_names,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_R2": aggregate_r2,
        "data_meta": data_meta,
        "per_task": per_task,
        **flat_metrics,
    }

    save_dir = Path(save_dir) if save_dir is not None else ckpt_dir / "emit_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "emit_eval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
    npz_path = save_dir / "emit_eval_predictions.npz"
    np.savez_compressed(
        npz_path,
        task_names=np.array(task_names),
        eval_idx=eval_idx,
        y_true=np.column_stack(y_true_all),
        y_pred=np.column_stack(y_pred_all),
        y_std=np.column_stack(y_std_all),
    )
    print(f"\nAggregate RRMSE={aggregate_rrmse:.6f}  R2={aggregate_r2:.6f}")
    print(f"Saved metrics: {json_path}")
    print(f"Saved predictions: {npz_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--emit-path", type=Path, default=DEFAULT_EMIT_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--predict-chunk-size", type=int, default=512)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Cap evaluation rows after filtering (0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Include default-state rows (~84% near prior defaults)",
    )
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()
    gpplus.config.configure_logger(level=getattr(logging, args.log_level))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"

    max_samples = None if args.max_samples == 0 else int(args.max_samples)
    evaluate_checkpoints_on_emit(
        args.ckpt_dir,
        args.emit_path,
        device=device,
        predict_chunk_size=args.predict_chunk_size,
        max_samples=max_samples,
        seed=args.seed,
        nondefault_only=not args.all_samples,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
