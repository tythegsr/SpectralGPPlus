"""Re-evaluate completed S4 SORF checkpoints on the held-out test split.

Discovers ``checkpoint_*.pt`` by task-name suffix (handles mixed T1/T7 titles),
predicts on the shared ``test_idx``, and writes a merged ``gp_*.json`` with
RRMSE + NIS for all completed QoIs. Does not retrain.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _SORF_DIR, _RFF_DIR, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments_RFF.rff_gp_defaults import rff_eval_kwargs  # noqa: E402
from experiments_RFF.toa_stgp_checkpoint import load_toa_stgp_checkpoint  # noqa: E402
from experiments_RFFMTGPR.emit_s4_mtgpr_base import (  # noqa: E402
    S4_TASK_NAMES,
    load_emit_s4_xy,
)
from experiments_RFFMTGPR.mtgpr_experiment_utils import (  # noqa: E402
    compute_relative_error_metrics,
    save_metrics_json,
)
from experiments_SORF.emit_s4_sorf_base import (  # noqa: E402
    _predict_worker_device,
    resolve_predict_n_jobs,
)
from experiments_toa.s2_utils import (  # noqa: E402
    compute_per_task_metrics,
    macro_metric,
    macro_rrmse,
)
from experiments_toa.s2_y_transform import inverse_y_s2, resolve_y_warps  # noqa: E402
from gpplus.training import evaluate_rff_gp_model  # noqa: E402
from gpplus.utils import compute_metrics  # noqa: E402
from gpplus.utils.fs_path import fs_path  # noqa: E402

# Canonical 7-QoI comparison set used by Sept08 S4 emit runs.
S4_COMPARE_TASKS = [
    "cos_i",
    "grain_size",
    "dust",
    "algae",
    "cwv",
    "lwc",
    "aot",
]

DEFAULT_CKPT_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Sept08"
    / "s4_emit_sorf_1inits_numrff2000_lr0.01_dtypefloat64"
)


def _s4_script_warp_defaults():
    from experiments_SORF.S4_emit_SORF import LOGIT_SCALE_QOI, LOG_OFFSETS, LOG_SCALE_QOI

    return LOG_SCALE_QOI, LOGIT_SCALE_QOI, LOG_OFFSETS


def _discover_checkpoints(ckpt_dir: Path, tasks: list[str]) -> dict[str, Path]:
    """Map task name -> checkpoint path via stem suffix (longest match first)."""
    out: dict[str, Path] = {}
    suffixes = sorted(tasks, key=len, reverse=True)
    for path in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        stem = path.stem
        for task in suffixes:
            if stem.endswith("_" + task):
                # Prefer newer / first found; keep first if duplicate.
                out.setdefault(task, path)
                break
    return out


def _infer_warps(ckpt_dir: Path, bundles_log_flags: dict[str, tuple[bool, bool]]):
    log_tasks = [t for t, (lg, _) in bundles_log_flags.items() if lg]
    logit_tasks = [t for t, (_, lt) in bundles_log_flags.items() if lt]
    log_scale_qoi, logit_scale_qoi, log_offsets = _s4_script_warp_defaults()
    return resolve_y_warps(
        log_tasks if log_tasks else log_scale_qoi,
        logit_tasks if logit_tasks else logit_scale_qoi,
        log_offsets=log_offsets,
    )


def _merged_title(first_title: str, n_tasks: int) -> str:
    """Replace ``_T{k}_`` in checkpoint title with completed task count."""
    import re

    updated, n = re.subn(r"_T\d+_", f"_T{n_tasks}_", first_title, count=1)
    if n:
        return updated
    return f"{first_title}_T{n_tasks}_reeval"


def _reeval_one_pack(
    pack: dict[str, Any],
    *,
    worker_device: str,
    predict_chunk_size: int,
    warps,
    rel_tolerance: float,
) -> dict[str, Any]:
    task_name = str(pack["task_name"])
    model = pack["model"]
    dtype = pack["dtype"]
    x_te = pack["x_te"]
    y_te = pack["y_te"]
    y_scaler = pack["y_scaler"]
    standardize_y = bool(pack["standardize_y"])
    best_train_loss = float(pack["best_train_loss"])
    ckpt_path = Path(pack["ckpt_path"])

    model = model.to(device=worker_device)
    x_te_dev = x_te.to(device=worker_device, dtype=dtype)
    if getattr(model, "nigp", False):
        model.nigp_correction_enabled = True
    model.eval()
    model.invalidate_feature_cache()
    t0 = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
        model,
        x_te_dev,
        chunk_size=predict_chunk_size,
        **rff_eval_kwargs(dtype),
    )
    if str(worker_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    pred_time = time.time() - t0

    inv = inverse_y_s2(
        pred_mean.detach().cpu(),
        pred_std.detach().cpu(),
        lower.detach().cpu(),
        upper.detach().cpu(),
        task_name=task_name,
        y_scaler=y_scaler,
        standardize_y=standardize_y,
        warps=warps,
        extended=True,
    )
    pred_mean_p, pred_std_p, lower_p, upper_p = inv.as_tuple()
    computed = compute_metrics(
        y_te.cpu(),
        pred_mean_p,
        output_std=pred_std_p,
        lower_95=lower_p,
        upper_95=upper_p,
        training_time=0.0,
        prediction_time=pred_time,
    )
    model.cpu()
    if str(worker_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "task_name": task_name,
        "prediction_time": pred_time,
        "pred_mean": pred_mean_p,
        "pred_std": pred_std_p,
        "lower": lower_p,
        "upper": upper_p,
        "y_te": y_te,
        "computed": computed,
        "best_train_loss": best_train_loss,
        "ckpt_path": ckpt_path,
        "model_config": dict(pack.get("model_config") or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--predict-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--predict-n-jobs",
        type=int,
        default=-1,
        help="Parallel QoI predicts (-1 = auto; 1 = sequential)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="QoI subset (default: Sept08 7-QoI comparison set)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_dir
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"

    task_candidates = list(args.tasks) if args.tasks else list(S4_COMPARE_TASKS)
    unknown = [t for t in task_candidates if t not in S4_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S4 tasks: {unknown}")

    ckpts = _discover_checkpoints(ckpt_dir, task_candidates)
    names = [t for t in task_candidates if t in ckpts]
    missing = [t for t in task_candidates if t not in ckpts]
    if not names:
        raise FileNotFoundError(f"No matching checkpoints under {ckpt_dir}")

    print(f"Found {len(names)} checkpoints in {ckpt_dir}")
    for t in names:
        print(f"  {t}: {ckpts[t].name}")
    if missing:
        print(f"Missing: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Peek metadata from first checkpoint (CPU).
    first = load_toa_stgp_checkpoint(ckpts[names[0]], device="cpu")
    data_path = first.data_path
    if not data_path or not Path(data_path).is_file():
        raise FileNotFoundError(f"Checkpoint data_path missing or not found: {data_path}")
    seed = int(first.seed)
    rel_tolerance = float(first.rel_tolerance)
    test_idx = first.test_idx.cpu().long()
    train_idx = first.train_idx.cpu().long()
    val_idx = first.val_idx.cpu().long()
    n_train = int(first.n_train)
    n_val = int(first.n_val)
    n_test = int(first.n_test)
    title = _merged_title(str(first.title), len(names))

    print(f"data_path={data_path}")
    print(f"device={device} out={out_dir}")
    print(f"output title={title}")

    X_np, Y_np, source_idx, data_meta = load_emit_s4_xy(
        data_path,
        task_names=names,
        filter_valid_labels=False,
    )
    # With filter_valid_labels=False, source_idx is identity; checkpoint indices
    # address rows in X_np/Y_np directly.
    name_to_col = {n: i for i, n in enumerate(names)}
    x_test_full = torch.as_tensor(X_np[test_idx.numpy()], dtype=torch.float64)
    y_test_full = torch.as_tensor(Y_np[test_idx.numpy()], dtype=torch.float64)

    # Collect log/logit flags while evaluating.
    log_flags: dict[str, tuple[bool, bool]] = {}
    # Pre-load flags from all ckpts for warps (cheap metadata).
    for t in names:
        payload = torch.load(fs_path(ckpts[t]), map_location="cpu", weights_only=False)
        log_flags[t] = (bool(payload.get("log_grain")), bool(payload.get("logit_cos")))
    warps = _infer_warps(ckpt_dir, log_flags)
    print(
        f"Output warps: log={sorted(warps.log_tasks)} "
        f"logit={sorted(warps.logit_tasks)} "
        f"offsets={warps.active_log_offsets()}"
    )

    predict_n_jobs = resolve_predict_n_jobs(
        device, len(names), None if args.predict_n_jobs < 0 else args.predict_n_jobs
    )
    print(f"predict_n_jobs={predict_n_jobs}")

    packs: list[dict[str, Any]] = []
    for task_name in names:
        print(f"\n=== Load {task_name} ===")
        bundle = load_toa_stgp_checkpoint(ckpts[task_name], device="cpu")
        if not torch.equal(bundle.test_idx.cpu().long(), test_idx):
            raise RuntimeError(f"{task_name} test_idx differs from {names[0]}")

        model = bundle.model
        dtype = bundle.dtype
        cols = bundle.input_column_indices
        if cols is None:
            x_te = x_test_full.to(dtype=dtype)
        else:
            cols_long = cols.to(dtype=torch.long)
            x_te = x_test_full.index_select(-1, cols_long).to(dtype=dtype)
        if bundle.standardize_x and bundle.x_scaler is not None:
            x_te = bundle.x_scaler.transform(x_te)

        y_te = y_test_full[:, name_to_col[task_name]].to(dtype=torch.float64)
        payload = torch.load(fs_path(ckpts[task_name]), map_location="cpu", weights_only=False)
        packs.append(
            {
                "task_name": task_name,
                "model": model.cpu(),
                "dtype": dtype,
                "x_te": x_te.detach().cpu(),
                "y_te": y_te.detach().cpu(),
                "y_scaler": bundle.y_scaler,
                "standardize_y": bundle.standardize_y,
                "best_train_loss": float(bundle.best_train_loss),
                "ckpt_path": ckpts[task_name],
                "model_config": dict(payload.get("model_config") or {}),
            }
        )
        del bundle

    results: list[dict[str, Any] | None] = [None] * len(packs)

    def _one(i: int, pack: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        worker_device = _predict_worker_device(device, i, predict_n_jobs)
        out = _reeval_one_pack(
            pack,
            worker_device=worker_device,
            predict_chunk_size=args.predict_chunk_size,
            warps=warps,
            rel_tolerance=rel_tolerance,
        )
        return i, out

    t_wall0 = time.time()
    if predict_n_jobs == 1:
        for i, pack in enumerate(packs):
            idx, out = _one(i, pack)
            results[idx] = out
            print(
                f"{out['task_name']} Test RMSE: {out['computed']['RMSE']:.6f}  "
                f"RRMSE: {out['computed']['RRMSE']:.6f}  "
                f"NIS: {out['computed'].get('NIS', float('nan')):.6f}  "
                f"pred={out['prediction_time']:.2f}s"
            )
    else:
        with ThreadPoolExecutor(max_workers=predict_n_jobs) as pool:
            futs = [pool.submit(_one, i, pack) for i, pack in enumerate(packs)]
            for fut in as_completed(futs):
                idx, out = fut.result()
                results[idx] = out
                print(
                    f"{out['task_name']} Test RMSE: {out['computed']['RMSE']:.6f}  "
                    f"RRMSE: {out['computed']['RRMSE']:.6f}  "
                    f"NIS: {out['computed'].get('NIS', float('nan')):.6f}  "
                    f"pred={out['prediction_time']:.2f}s"
                )
    wall_pred = time.time() - t_wall0
    ordered = [r for r in results if r is not None]
    if len(ordered) != len(packs):
        raise RuntimeError("reeval predict phase lost results")
    sum_pred = float(sum(float(r["prediction_time"]) for r in ordered))
    print(f"Predict wall: {wall_pred:.2f}s  sum: {sum_pred:.2f}s")

    task_metrics: dict = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    y_true_cols: list[np.ndarray] = []
    model_config_ref: dict = {}

    for out in ordered:
        task_name = str(out["task_name"])
        model_config_ref = dict(out.get("model_config") or model_config_ref)
        tm = {
            "best_train_loss": float(out["best_train_loss"]),
            "checkpoint_path": str(out["ckpt_path"]),
            "checkpoint_mtime": time.ctime(Path(out["ckpt_path"]).stat().st_mtime),
            "reeval": True,
            **{k: v for k, v in out["computed"].items()},
        }
        task_metrics[task_name] = tm
        y_pred_all.append(out["pred_mean"].numpy())
        y_std_all.append(out["pred_std"].numpy())
        lower_all.append(out["lower"].numpy())
        upper_all.append(out["upper"].numpy())
        y_true_cols.append(out["y_te"].cpu().numpy())

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_test_np = np.stack(y_true_cols, axis=1)

    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked, names)
    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t],
            y_pred_stacked[:, t],
            rel_tolerance=rel_tolerance,
        )
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_m["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))
    aggregate_rrmse = float(macro_rrmse(per_task, names))
    aggregate_medae = float(macro_metric(per_task, names, "MedAE"))

    metrics: dict = {
        "title": title,
        "dataset": "emit_s4",
        "task_names": names,
        "num_tasks": len(names),
        "n_train": n_train,
        "n_test": n_test,
        "n_val": n_val,
        "n_filtered": int(data_meta.get("n_filtered", X_np.shape[0])),
        "seed": seed,
        "num_rff": int(model_config_ref.get("num_rff", 0)),
        "rff_sampling": str(model_config_ref.get("rff_sampling", "sorf")),
        "correct_sorf": bool(model_config_ref.get("correct_sorf", True)),
        "nigp": bool(model_config_ref.get("nigp", False)),
        "rel_tolerance": rel_tolerance,
        "Training_Time": 0.0,
        "Prediction_Time": wall_pred,
        "Prediction_Time_Sum": sum_pred,
        "predict_n_jobs": int(predict_n_jobs),
        "Total_Time": wall_pred,
        "aggregate_RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "log_scale_tasks": sorted(warps.log_tasks),
        "logit_scale_tasks": sorted(warps.logit_tasks),
        "log_offsets": warps.active_log_offsets(),
        "aux_inputs": data_meta.get("aux_inputs"),
        "data_meta": {
            k: v
            for k, v in data_meta.items()
            if k
            not in (
                "task_valid_y_range",
            )
        },
        "reeval_meta": {
            "reeval_source_dir": str(ckpt_dir),
            "reeval_note": (
                "Re-evaluated S4 SORF checkpoints on held-out test split "
                f"({len(names)}/{len(task_candidates)} QoIs; suffix discovery)."
            ),
            "completed_tasks": names,
            "missing_tasks": missing,
            "checkpoint_mtimes": {
                t: time.ctime(ckpts[t].stat().st_mtime) for t in names
            },
            "source_idx_is_identity": bool(
                np.array_equal(source_idx, np.arange(len(source_idx)))
            ),
        },
        "task_metrics": task_metrics,
        **per_task,
    }
    for name in names:
        tm = task_metrics[name]
        for k in ("RMSE", "MAE", "MedAE", "RRMSE", "R2", "NIS", "NLPD", "CRPS", "NCRPS", "MSE"):
            if k in tm and f"{name}_{k}" not in metrics:
                metrics[f"{name}_{k}"] = tm[k]

    json_path = save_metrics_json(metrics, str(out_dir), title)
    npz_path = out_dir / f"predictions_{title}.npz"
    np.savez_compressed(
        fs_path(npz_path),
        y_true=y_test_np,
        y_pred=y_pred_stacked,
        y_std=y_std_stacked,
        lower=lower_stacked,
        upper=upper_stacked,
        task_names=np.asarray(names),
        test_idx=test_idx.numpy(),
        train_idx=train_idx.numpy(),
        val_idx=val_idx.numpy(),
    )

    print(f"\naggregate_RRMSE={aggregate_rrmse:.6f}")
    for n in names:
        print(f"  {n}: RRMSE={per_task[f'{n}_RRMSE']:.6f}  NIS={task_metrics[n]['NIS']:.6f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
