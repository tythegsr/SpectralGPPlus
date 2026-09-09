"""Evaluate a joint S4 RFFMTGPR checkpoint on ASD validation NetCDF."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"

for _p in (_ROOT, _MTGPR_DIR, _RFF_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_RFF.rff_gp_defaults import mt_eval_kwargs
from experiments_RFFMTGPR.emit_s4_mtgpr_base import append_s4_aux_inputs
from experiments_toa.s2_utils import _scalar_error_metrics, macro_metric
from experiments_toa.s2_y_transform import resolve_y_warps
from gpplus.training import evaluate_rff_mt_gp_model
from gpplus.utils import compute_metrics
from gpplus.utils.fs_path import fs_path
from mtgpr_experiment_utils import compute_prediction_coverage_metrics
from toa_mtgpr_checkpoint import load_toa_mtgpr_checkpoint
from toa_s2_mtgpr_base import inverse_y_s2_matrix

DEFAULT_CKPT_DIR = (
    _ROOT
    / "experiments_RFFMTGPR"
    / "results"
    / "Sept07"
    / "s4_emit_mtgpr_1inits_numrff500_lr0.005_nigp_freezeepochnigp800_sloperefreshes10_dtypefloat32"
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


def _load_run_meta(ckpt_dir: Path) -> dict:
    candidates = sorted(ckpt_dir.glob("gp_S4_*.json"))
    if not candidates:
        return {}
    with open(fs_path(candidates[-1]), encoding="utf-8") as f:
        return json.load(f)


def _load_run_warps(ckpt_dir: Path, meta: dict | None = None):
    meta = meta if meta is not None else _load_run_meta(ckpt_dir)
    if meta:
        return resolve_y_warps(
            meta.get("log_scale_tasks"),
            meta.get("logit_scale_tasks"),
            log_offsets=meta.get("log_offsets"),
        )
    print("  (no gp_S4_*.json; using S4_emit_MTGPR warp defaults)")
    from experiments_RFFMTGPR.S4_emit_MTGPR import (
        LOG_OFFSETS,
        LOG_SCALE_QOI,
        LOGIT_SCALE_QOI,
    )

    return resolve_y_warps(LOG_SCALE_QOI, LOGIT_SCALE_QOI, log_offsets=LOG_OFFSETS)


def _trained_task_names(meta: dict, *, fallback: list[str] | None = None) -> list[str]:
    names = meta.get("task_names") if meta else None
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    if fallback:
        return list(fallback)
    return list(TASK_LABEL_MAP)


def _find_mt_checkpoint(
    ckpt_dir: Path,
    *,
    checkpoint_title: str | None = None,
    meta: dict | None = None,
) -> Path:
    """Resolve the joint ``checkpoint_{title}.pt`` for an MTGPR run dir."""
    ckpt_dir = Path(ckpt_dir)
    titles: list[str] = []
    if checkpoint_title:
        titles.append(checkpoint_title)
    if meta and meta.get("title"):
        titles.append(str(meta["title"]))

    for title in titles:
        path = ckpt_dir / f"checkpoint_{title}.pt"
        if path.is_file():
            return path

    # Prefer joint checkpoints (no trailing _{task} suffix).
    task_suffixes = sorted(TASK_LABEL_MAP, key=len, reverse=True)
    joint: list[Path] = []
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        stem = path.stem
        if not stem.startswith("checkpoint_"):
            continue
        rest = stem[len("checkpoint_") :]
        if any(rest.endswith(f"_{t}") for t in task_suffixes):
            continue
        joint.append(path)
    if len(joint) == 1:
        return joint[0]
    if len(joint) > 1:
        raise FileNotFoundError(
            f"Multiple joint MTGPR checkpoints under {ckpt_dir}: "
            f"{[p.name for p in joint]}; pass checkpoint_title="
        )
    raise FileNotFoundError(f"No joint MTGPR checkpoint found under {ckpt_dir}")


def evaluate_s4_mtgpr_checkpoint_on_asd(
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
    """Evaluate one joint RFFMTGPR checkpoint on ASD field scenes.

    Writes the same ``asd_validation_metrics.json`` /
    ``asd_validation_predictions.npz`` layout as the ST SORF/SVGP evaluators so
    shared plotting and neighbor analysis continue to work.
    """
    ckpt_dir = Path(ckpt_dir)
    asd_path = Path(asd_path)
    meta = _load_run_meta(ckpt_dir)
    ckpt_path = _find_mt_checkpoint(
        ckpt_dir, checkpoint_title=checkpoint_title, meta=meta
    )
    trained_names = _trained_task_names(meta)
    title = checkpoint_title or str(meta.get("title") or ckpt_path.stem.replace("checkpoint_", "", 1))

    requested = list(tasks) if tasks is not None else list(TASK_LABEL_MAP)
    unknown = [t for t in requested if t not in TASK_LABEL_MAP]
    if unknown:
        raise ValueError(f"No ASD label mapping for tasks: {unknown}")

    # Keep trained column order; only score tasks that have ASD labels + were requested.
    eval_names = [t for t in trained_names if t in TASK_LABEL_MAP and t in requested]
    skipped_no_asd = [t for t in trained_names if t not in TASK_LABEL_MAP]
    skipped_missing = [t for t in requested if t not in trained_names]
    if skipped_no_asd:
        print(f"  trained tasks without ASD labels (ignored): {skipped_no_asd}")
    if skipped_missing:
        print(f"  requested ASD tasks not in checkpoint (skipped): {skipped_missing}")
    if not eval_names:
        raise FileNotFoundError(
            f"No overlapping ASD tasks between checkpoint tasks={trained_names} "
            f"and requested={requested}"
        )

    if coszen_override is not None:
        print(f"  coszen override: fixed at {coszen_override:.6f} for all scenes")

    x_np, y_by_task, aux_meta, coszen_asd_original = _load_asd_xy(
        asd_path,
        eval_names,
        coszen_override=coszen_override,
    )
    print(f"Loaded ASD validation set: n={x_np.shape[0]} from {asd_path}")
    print(f"  aux inputs: {aux_meta['aux_inputs']}")
    print(f"  checkpoint: {ckpt_path.name}")
    print(f"  trained tasks: {trained_names}")
    print(f"  ASD eval tasks: {eval_names}")

    warps = _load_run_warps(ckpt_dir, meta=meta)
    print(
        f"  output warps: log={sorted(warps.log_tasks)} "
        f"logit={sorted(warps.logit_tasks)}"
    )

    bundle = load_toa_mtgpr_checkpoint(fs_path(ckpt_path), device=device)
    if getattr(bundle.model, "nigp", False):
        bundle.model.nigp_correction_enabled = True

    n_model_tasks = int(bundle.model.num_tasks)
    if n_model_tasks != len(trained_names):
        if n_model_tasks == len(eval_names) and set(eval_names) <= set(trained_names):
            print(
                f"  warning: model T={n_model_tasks} != meta tasks={len(trained_names)}; "
                f"falling back to eval task order {eval_names}"
            )
            trained_names = list(eval_names)
        else:
            raise ValueError(
                f"Cannot align model outputs (T={n_model_tasks}) with "
                f"task_names={trained_names}"
            )

    x = torch.as_tensor(x_np, dtype=bundle.dtype)
    cols = bundle.input_column_indices
    if cols is not None:
        cols = cols.to(dtype=torch.long)
        if cols.numel() < x.shape[-1]:
            x = x.index_select(-1, cols)
    if bundle.standardize_x and bundle.x_scaler is not None:
        x = bundle.x_scaler.transform(x)
    x = x.to(device=bundle.model.train_inputs[0].device, dtype=bundle.dtype)

    t0 = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_mt_gp_model(
        bundle.model,
        x,
        chunk_size=predict_chunk_size,
        **mt_eval_kwargs(bundle.dtype),
    )
    pred_time = time.time() - t0

    inv = inverse_y_s2_matrix(
        pred_mean.detach().cpu(),
        pred_std.detach().cpu(),
        lower.detach().cpu(),
        upper.detach().cpu(),
        task_names=trained_names,
        y_scaler=bundle.y_scaler,
        standardize_y=bundle.standardize_y,
        warps=warps,
    )
    y_pred_all = inv.point.numpy()
    y_std_all = inv.std.numpy()
    lower_all = inv.lower.numpy()
    upper_all = inv.upper.numpy()
    y_pred_mean_all = (
        inv.point_mean.numpy() if inv.point_mean is not None else np.full_like(y_pred_all, np.nan)
    )
    y_pred_mode_all = (
        inv.point_mode.numpy() if inv.point_mode is not None else np.full_like(y_pred_all, np.nan)
    )
    log_mu_all = (
        inv.log_mu.numpy() if inv.log_mu is not None else np.full_like(y_pred_all, np.nan)
    )
    log_sigma_all = (
        inv.log_sigma.numpy() if inv.log_sigma is not None else np.full_like(y_pred_all, np.nan)
    )

    col_index = {name: i for i, name in enumerate(trained_names)}
    per_task: dict[str, dict] = {}
    coverage_by_task: dict[str, dict] = {}
    y_true_cols: list[np.ndarray] = []
    y_pred_cols: list[np.ndarray] = []
    y_pred_mean_cols: list[np.ndarray] = []
    y_pred_mode_cols: list[np.ndarray] = []
    log_mu_cols: list[np.ndarray] = []
    log_sigma_cols: list[np.ndarray] = []
    y_std_cols: list[np.ndarray] = []
    lower_cols: list[np.ndarray] = []
    upper_cols: list[np.ndarray] = []

    print(f"\n=== joint MTGPR ASD predict ({pred_time:.2f}s) ===")
    for task_name in eval_names:
        t = col_index[task_name]
        y_true = torch.as_tensor(y_by_task[task_name], dtype=bundle.dtype)
        y_pred_np = y_pred_all[:, t]
        y_std_np = y_std_all[:, t]
        lower_np = lower_all[:, t]
        upper_np = upper_all[:, t]
        y_pred_mean_np = y_pred_mean_all[:, t]
        y_pred_mode_np = y_pred_mode_all[:, t]
        log_mu_np = log_mu_all[:, t]
        log_sigma_np = log_sigma_all[:, t]
        n_eval = int(y_true.numel())

        computed = compute_metrics(
            y_true,
            torch.as_tensor(y_pred_np),
            output_std=torch.as_tensor(y_std_np),
            lower_95=torch.as_tensor(lower_np),
            upper_95=torch.as_tensor(upper_np),
            prediction_time=pred_time / max(len(eval_names), 1),
        )
        scalar = _scalar_error_metrics(
            y_true.numpy(),
            y_pred_np,
            prefix=f"{task_name}_",
        )
        cov_m = compute_prediction_coverage_metrics(
            y_true.numpy(),
            y_pred_np,
            y_std_np,
        )
        coverage_by_task[task_name] = cov_m
        row = {
            "checkpoint": str(ckpt_path),
            "asd_label": TASK_LABEL_MAP[task_name],
            "n_eval": n_eval,
            "prediction_time_s": float(pred_time / max(len(eval_names), 1)),
            "y_true": y_true.numpy().tolist(),
            "y_pred": y_pred_np.tolist(),
            "y_pred_mean": y_pred_mean_np.tolist(),
            "y_pred_mode": y_pred_mode_np.tolist(),
            "y_std": y_std_np.tolist(),
            "lower_95": lower_np.tolist(),
            "upper_95": upper_np.tolist(),
            **cov_m,
            **computed,
            **scalar,
        }
        per_task[task_name] = row
        print(
            f"  {task_name}: RMSE={computed['RMSE']:.6g}  RRMSE={computed['RRMSE']:.6g}  "
            f"MAE={computed['MAE']:.6g}  R2={scalar[f'{task_name}_R2']:.4f}"
        )
        for i, (yt, yp) in enumerate(zip(row["y_true"], row["y_pred"])):
            extra = ""
            if warps.uses_log(task_name):
                extra = (
                    f"  mean={y_pred_mean_np[i]:.6g}  mode={y_pred_mode_np[i]:.6g}"
                )
            print(f"    sample {i}: true={yt:.6g}  pred={yp:.6g}{extra}")

        y_true_cols.append(y_true.numpy())
        y_pred_cols.append(y_pred_np)
        y_pred_mean_cols.append(y_pred_mean_np)
        y_pred_mode_cols.append(y_pred_mode_np)
        log_mu_cols.append(log_mu_np)
        log_sigma_cols.append(log_sigma_np)
        y_std_cols.append(y_std_np)
        lower_cols.append(lower_np)
        upper_cols.append(upper_np)

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

    aggregate_rrmse = macro_metric(flat_metrics, eval_names, "RRMSE")
    aggregate_r2 = macro_metric(flat_metrics, eval_names, "R2")

    results = {
        "title": f"asd_validation_eval_{ckpt_dir.name}",
        "backend": "mtgpr",
        "ckpt_dir": str(ckpt_dir.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "asd_path": str(asd_path.resolve()),
        "task_names": eval_names,
        "trained_task_names": trained_names,
        "missing_tasks": sorted({"fsnow", *skipped_missing, *skipped_no_asd}),
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
        "prediction_time_s": float(pred_time),
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
        task_names=np.array(eval_names),
        y_true=np.column_stack(y_true_cols),
        y_pred=np.column_stack(y_pred_cols),
        y_pred_mean=np.column_stack(y_pred_mean_cols),
        y_pred_mode=np.column_stack(y_pred_mode_cols),
        log_mu=np.column_stack(log_mu_cols),
        log_sigma=np.column_stack(log_sigma_cols),
        y_std=np.column_stack(y_std_cols),
        lower_95=np.column_stack(lower_cols),
        upper_95=np.column_stack(upper_cols),
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
    parser.add_argument(
        "--checkpoint-title",
        type=str,
        default=None,
        help="Joint checkpoint title prefix (default: auto from gp_S4_*.json)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Subset of ASD-labeled tasks (default: {list(TASK_LABEL_MAP)})",
    )
    parser.add_argument(
        "--coszen-override",
        type=float,
        default=None,
        help="Use a fixed coszen aux input for all ASD scenes",
    )
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"
    elif device.startswith("cuda"):
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    evaluate_s4_mtgpr_checkpoint_on_asd(
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
