"""Evaluate S4 SVGP checkpoints on experiments_toa ASD validation NetCDF."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SVGP_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_PCA_DIR = _ROOT / "experiments_PCA"

for _p in (_ROOT, _SVGP_DIR, _GP_DIR, _MTGPR_DIR, _PCA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_RFFMTGPR.emit_s4_mtgpr_base import append_s4_aux_inputs
from experiments_toa.s2_utils import _scalar_error_metrics, apply_x_transform, macro_metric
from experiments_toa.s2_y_transform import inverse_y_s2, resolve_y_warps
from gpplus.training import evaluate_svgp_gp_model
from gpplus.utils import compute_metrics
from gpplus.utils.fs_path import fs_path
from mtgpr_experiment_utils import compute_prediction_coverage_metrics
from toa_svgp_checkpoint import load_toa_svgp_checkpoint

DEFAULT_CKPT_DIR = (
    _ROOT
    / "experiments_SVGP"
    / "results"
    / "s4_emit_svgp_1inits_M512_batch1024_lr0.05_nigp_freezeepochnigp50_dtypefloat32"
)
DEFAULT_ASD_PATH = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"

TASK_LABEL_MAP: dict[str, str] = {
    "grain_size": "grain_radius_mean",
    "cos_i": "cos_i",
    "dust": "dust_conc_mean",
    "algae": "algae_conc_mean",
    "cwv": "cwv",
    "lwc": "lwc_mean",
    "aot": "aod",
}
DEFAULT_CHECKPOINT_TITLE = (
    "S4_EMIT_ST_nTrain20000_nVal5000_nTest95000_svgpM512_T2_nigp"
)


def _load_asd_xy(
    asd_path: Path,
    task_names: list[str],
    *,
    coszen_override: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict, np.ndarray | None]:
    with h5py.File(asd_path, "r") as f:
        x_spec = np.asarray(f["toa_radiance"][:], dtype=np.float64)
        coszen_asd = np.cos(np.radians(np.asarray(f["sza"][:], dtype=np.float64)))
        if coszen_override is not None:
            coszen = np.full_like(coszen_asd, float(coszen_override))
        else:
            coszen = coszen_asd
        ele_km = np.asarray(f["elevation_km"][:], dtype=np.float64).reshape(-1)
        x_np, aux_meta = append_s4_aux_inputs(x_spec, coszen, ele_km)
        y_by_task = {
            task: np.asarray(f[TASK_LABEL_MAP[task]][:], dtype=np.float64).reshape(-1)
            for task in task_names
        }
    return (
        x_np,
        y_by_task,
        aux_meta,
        coszen_asd if coszen_override is not None else None,
    )


def _s4_script_warp_defaults() -> tuple[list[str] | None, list[str] | None, dict[str, float] | None]:
    from experiments_SVGP.S4_emit_SVGP import LOGIT_SCALE_QOI, LOG_OFFSETS, LOG_SCALE_QOI

    return LOG_SCALE_QOI, LOGIT_SCALE_QOI, LOG_OFFSETS


def _infer_warps_from_checkpoints(ckpt_dir: Path):
    log_tasks: list[str] = []
    logit_tasks: list[str] = []
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        payload = torch.load(fs_path(path), map_location="cpu", weights_only=False)
        task = str(payload.get("task_name", ""))
        if not task:
            continue
        for name in payload.get("log_scale_qoi") or []:
            if name == task and task not in log_tasks:
                log_tasks.append(task)
        for name in payload.get("logit_scale_qoi") or []:
            if name == task and task not in logit_tasks:
                logit_tasks.append(task)
    log_scale_qoi, logit_scale_qoi, log_offsets = _s4_script_warp_defaults()
    return resolve_y_warps(
        log_tasks if log_tasks else log_scale_qoi,
        logit_tasks if logit_tasks else logit_scale_qoi,
        log_offsets=log_offsets,
    )


def _load_run_warps(ckpt_dir: Path):
    candidates = sorted(ckpt_dir.glob("gp_S4_*.json"))
    if candidates:
        with open(fs_path(candidates[-1]), encoding="utf-8") as f:
            meta = json.load(f)
        return resolve_y_warps(
            meta.get("log_scale_tasks"),
            meta.get("logit_scale_tasks"),
            log_offsets=meta.get("log_offsets"),
        )
    print("  (no gp_S4_*.json; inferring warps from checkpoints + S4_emit_SVGP defaults)")
    return _infer_warps_from_checkpoints(ckpt_dir)


def _checkpoint_path(ckpt_dir: Path, task_name: str, *, checkpoint_title: str) -> Path:
    return ckpt_dir / f"checkpoint_{checkpoint_title}_{task_name}.pt"


def _available_asd_tasks(
    ckpt_dir: Path,
    *,
    checkpoint_title: str,
    candidates: list[str],
) -> list[str]:
    return [
        task
        for task in candidates
        if os.path.isfile(
            fs_path(_checkpoint_path(ckpt_dir, task, checkpoint_title=checkpoint_title))
        )
    ]


def _detect_checkpoint_title(ckpt_dir: Path) -> str:
    task_suffixes = sorted(TASK_LABEL_MAP, key=len, reverse=True)
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        stem = path.stem
        if not stem.startswith("checkpoint_"):
            continue
        rest = stem[len("checkpoint_") :]
        for task_name in task_suffixes:
            suffix = f"_{task_name}"
            if rest.endswith(suffix):
                return rest[: -len(suffix)]
    return DEFAULT_CHECKPOINT_TITLE


def _find_checkpoint(ckpt_dir: Path, task_name: str, *, checkpoint_title: str) -> Path:
    path = _checkpoint_path(ckpt_dir, task_name, checkpoint_title=checkpoint_title)
    if not os.path.isfile(fs_path(path)):
        raise FileNotFoundError(f"Missing checkpoint for {task_name!r}: {path}")
    return path


def _prepare_x(bundle, x_np: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(x_np, dtype=bundle.dtype)
    cols = bundle.input_column_indices.to(dtype=torch.long)
    if cols.numel() < x.shape[-1]:
        x = x.index_select(-1, cols)
    x = apply_x_transform(x, bundle.x_transform or "none")
    if bundle.pca_meta is not None:
        from toa_pca_utils import pca_from_dict, transform_pca

        pca_fit = pca_from_dict(bundle.pca_meta)
        x = transform_pca(pca_fit, x, dtype=bundle.dtype)
    if bundle.standardize_x and bundle.x_scaler is not None:
        x = bundle.x_scaler.transform(x)
    return x.to(device=bundle.model.train_inputs[0].device, dtype=bundle.dtype)


def evaluate_s4_svgp_checkpoints_on_asd(
    ckpt_dir: Path,
    asd_path: Path,
    *,
    device: str = "cuda",
    predict_chunk_size: int = 512,
    tasks: list[str] | None = None,
    save_dir: Path | None = None,
    checkpoint_title: str | None = None,
    coszen_override: float | None = None,
) -> dict:
    ckpt_dir = Path(ckpt_dir)
    asd_path = Path(asd_path)
    title = checkpoint_title or _detect_checkpoint_title(ckpt_dir)
    requested = list(tasks) if tasks is not None else list(TASK_LABEL_MAP)
    unknown = [t for t in requested if t not in TASK_LABEL_MAP]
    if unknown:
        raise ValueError(f"No ASD label mapping for tasks: {unknown}")

    task_names = _available_asd_tasks(
        ckpt_dir, checkpoint_title=title, candidates=requested
    )
    skipped = [t for t in requested if t not in task_names]
    if skipped:
        print(f"  skipping tasks without checkpoints: {skipped}")
    if not task_names:
        raise FileNotFoundError(f"No ASD task checkpoints found under {ckpt_dir}")

    if coszen_override is not None:
        print(f"  coszen override: fixed at {coszen_override:.6f} for all scenes")

    x_np, y_by_task, aux_meta, coszen_asd_original = _load_asd_xy(
        asd_path,
        task_names,
        coszen_override=coszen_override,
    )
    print(f"Loaded ASD validation set: n={x_np.shape[0]} from {asd_path}")
    print(f"  aux inputs: {aux_meta['aux_inputs']}")
    print(f"  checkpoint title: {title}")

    warps = _load_run_warps(ckpt_dir)
    print(
        f"  output warps: log={sorted(warps.log_tasks)} "
        f"logit={sorted(warps.logit_tasks)}"
    )

    per_task: dict[str, dict] = {}
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_pred_mean_all: list[np.ndarray] = []
    y_pred_mode_all: list[np.ndarray] = []
    log_mu_all: list[np.ndarray] = []
    log_sigma_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    coverage_by_task: dict[str, dict] = {}

    for task_name in task_names:
        ckpt_path = _find_checkpoint(ckpt_dir, task_name, checkpoint_title=title)
        print(f"\n=== {task_name} <- {ckpt_path.name} ===")
        bundle = load_toa_svgp_checkpoint(ckpt_path, device=device)
        if getattr(bundle.model, "nigp", False):
            if hasattr(bundle.model, "nigp_correction_enabled"):
                bundle.model.nigp_correction_enabled = True

        x = _prepare_x(bundle, x_np)
        t0 = time.time()
        pred_mean, lower, upper, pred_std = evaluate_svgp_gp_model(
            bundle.model,
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
        y_true = torch.as_tensor(y_by_task[task_name], dtype=bundle.dtype)
        n_eval = int(y_true.numel())
        nan_col = np.full(n_eval, np.nan, dtype=np.float64)

        computed = compute_metrics(
            y_true,
            pred_mean_c,
            output_std=pred_std_c,
            lower_95=lower_c,
            upper_95=upper_c,
            prediction_time=pred_time,
        )
        scalar = _scalar_error_metrics(
            y_true.numpy(),
            pred_mean_c.detach().cpu().numpy(),
            prefix=f"{task_name}_",
        )
        cov_m = compute_prediction_coverage_metrics(
            y_true.numpy(),
            pred_mean_c.numpy(),
            pred_std_c.numpy(),
        )
        coverage_by_task[task_name] = cov_m
        y_pred_np = pred_mean_c.detach().cpu().numpy()
        y_pred_mean_np = (
            inv.point_mean.detach().cpu().numpy()
            if inv.point_mean is not None
            else nan_col
        )
        y_pred_mode_np = (
            inv.point_mode.detach().cpu().numpy()
            if inv.point_mode is not None
            else nan_col
        )
        row = {
            "checkpoint": str(ckpt_path),
            "asd_label": TASK_LABEL_MAP[task_name],
            "n_eval": n_eval,
            "prediction_time_s": float(pred_time),
            "y_true": y_true.numpy().tolist(),
            "y_pred": y_pred_np.tolist(),
            "y_pred_mean": y_pred_mean_np.tolist(),
            "y_pred_mode": y_pred_mode_np.tolist(),
            "y_std": pred_std_c.detach().cpu().numpy().tolist(),
            "lower_95": lower_c.detach().cpu().numpy().tolist(),
            "upper_95": upper_c.detach().cpu().numpy().tolist(),
            **cov_m,
            **computed,
            **scalar,
        }
        per_task[task_name] = row
        print(
            f"  RMSE={computed['RMSE']:.6g}  RRMSE={computed['RRMSE']:.6g}  "
            f"MAE={computed['MAE']:.6g}  R2={scalar[f'{task_name}_R2']:.4f}  "
            f"cov95={cov_m['coverage_95']:.2f}"
        )
        for i, (yt, yp) in enumerate(zip(row["y_true"], row["y_pred"])):
            print(f"    sample {i}: true={yt:.6g}  pred={yp:.6g}")

        y_true_all.append(y_true.numpy())
        y_pred_all.append(y_pred_np)
        y_pred_mean_all.append(y_pred_mean_np)
        y_pred_mode_all.append(y_pred_mode_np)
        log_mu_all.append(
            inv.log_mu.detach().cpu().numpy() if inv.log_mu is not None else nan_col
        )
        log_sigma_all.append(
            inv.log_sigma.detach().cpu().numpy()
            if inv.log_sigma is not None
            else nan_col
        )
        y_std_all.append(pred_std_c.numpy())
        lower_all.append(lower_c.numpy())
        upper_all.append(upper_c.numpy())

        del bundle
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    flat_metrics: dict[str, float] = {}
    for name, row in per_task.items():
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

    aggregate_rrmse = macro_metric(flat_metrics, task_names, "RRMSE")
    aggregate_r2 = macro_metric(flat_metrics, task_names, "R2")

    results = {
        "title": f"asd_validation_eval_{ckpt_dir.name}",
        "ckpt_dir": str(ckpt_dir.resolve()),
        "asd_path": str(asd_path.resolve()),
        "model_class": "SVGPR",
        "task_names": task_names,
        "missing_tasks": sorted({"fsnow", *skipped}),
        "checkpoint_title": title,
        "log_scale_tasks": sorted(warps.log_tasks),
        "logit_scale_tasks": sorted(warps.logit_tasks),
        "log_offsets": dict(warps.log_offsets),
        "label_map": TASK_LABEL_MAP,
        "aux_meta": aux_meta,
        "coszen_override": coszen_override,
        "coszen_asd_original": (
            coszen_asd_original.tolist() if coszen_asd_original is not None else None
        ),
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_R2": aggregate_r2,
        "per_task": per_task,
        "coverage_by_task": coverage_by_task,
        **flat_metrics,
    }

    save_dir = Path(save_dir) if save_dir is not None else ckpt_dir / "asd_validation_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "asd_validation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "item") else o,
        )
    npz_path = save_dir / "asd_validation_predictions.npz"
    np.savez_compressed(
        npz_path,
        task_names=np.array(task_names),
        y_true=np.column_stack(y_true_all),
        y_pred=np.column_stack(y_pred_all),
        y_pred_mean=np.column_stack(y_pred_mean_all),
        y_pred_mode=np.column_stack(y_pred_mode_all),
        log_mu=np.column_stack(log_mu_all),
        log_sigma=np.column_stack(log_sigma_all),
        y_std=np.column_stack(y_std_all),
        lower_95=np.column_stack(lower_all),
        upper_95=np.column_stack(upper_all),
        x_spectral=x_np[:, :285],
    )
    print(f"\nAggregate RRMSE={aggregate_rrmse:.6f}  R2={aggregate_r2:.6f}")
    print(f"Saved metrics: {json_path}")
    print(f"Saved predictions: {npz_path}")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--asd-path", type=Path, default=DEFAULT_ASD_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--predict-chunk-size", type=int, default=512)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-title", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--coszen-override", type=float, default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"
    elif device.startswith("cuda"):
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    evaluate_s4_svgp_checkpoints_on_asd(
        args.ckpt_dir,
        args.asd_path,
        device=device,
        predict_chunk_size=args.predict_chunk_size,
        tasks=args.tasks,
        save_dir=args.save_dir,
        checkpoint_title=args.checkpoint_title,
        coszen_override=args.coszen_override,
    )


if __name__ == "__main__":
    main()
