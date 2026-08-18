"""Re-evaluate surviving Aug05 NIGP_cuda_check SORF checkpoints into a new folder.

Only QoIs whose checkpoints were NOT overwritten by the later aot/liquid_water
re-run are evaluated: algae, cos_i, cwv, dust, grain_size.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_SORF_DIR = _ROOT / "experiments_SORF"
for p in (_ROOT, _RFF_DIR, _MTGPR_DIR, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments_toa.paths import pin_toa_import_paths  # noqa: E402
from experiments_toa.s2_data import load_s2_arrays  # noqa: E402
from experiments_toa.s2_reporting import (  # noqa: E402
    print_end_of_run_summary,
    save_s2_summary_artifacts,
)
from experiments_toa.s2_utils import (  # noqa: E402
    apply_x_transform,
    compute_per_task_metrics,
    macro_metric,
    macro_rrmse,
    select_bands,
)
from experiments_toa.s2_y_transform import YWarpConfig, inverse_y_s2  # noqa: E402
from gpplus.models import RFFGPR  # noqa: E402
from gpplus.training.eval import evaluate_rff_gp_model  # noqa: E402
from gpplus.utils import compute_metrics  # noqa: E402
from mtgpr_experiment_utils import (  # noqa: E402
    compute_relative_error_metrics,
    format_relative_error_summary,
    save_metrics_json,
)
from rff_gp_defaults import rff_eval_kwargs  # noqa: E402
from toa_mtgpr_checkpoint import CHECKPOINT_VERSION, scaler_from_dict  # noqa: E402
from toa_stgp_checkpoint import _dtype_from_str  # noqa: E402

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

SRC_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug05"
    / "NIGP_cuda_check"
    / "s2_toa_sorf_1inits_numrff600_lr0.04_taskbandconfig_rbf_dtypefloat64"
)

# Original afternoon checkpoints (not overwritten at ~17:05).
SURVIVING_TASKS = ("algae", "cos_i", "cwv", "dust", "grain_size")

OUT_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug06"
    / "NIGP_cuda_check_reeval_surviving_ckpt"
    / "s2_toa_sorf_1inits_numrff600_lr0.04_taskbandconfig_rbf_dtypefloat64"
)

RFFGPR_KEYS = {
    "num_rff",
    "ard",
    "rff_sampling",
    "correct_sorf",
    "spectral_kernel",
    "nigp",
    "batch_shape",
    "init_batch_size",
}


def _rffgpr_kwargs(model_config: dict) -> dict:
    kw = {k: model_config[k] for k in RFFGPR_KEYS if k in model_config}
        # Old softplus-era checkpoints may store nigp_input_noise_init; ignore for
        # constructor (state_dict overwrites raw). New code uses SoftClamp log10.
    # Drop None PCA flag if present as unused.
    kw.pop("n_pca_components", None)
    return kw


def load_bundle(ckpt_path: Path, device: str):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {payload.get('version')!r}")

    dtype = _dtype_from_str(payload["dtype"])
    model_config = dict(payload["model_config"])
    train_x = payload["train_x"].to(dtype=dtype, device=device)
    train_y = payload["train_y"].to(dtype=dtype, device=device)

    model = RFFGPR(train_x, train_y, **_rffgpr_kwargs(model_config))
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
    if getattr(model, "nigp", False):
        model.nigp_correction_enabled = True

    return {
        "model": model,
        "payload": payload,
        "model_config": model_config,
        "dtype": dtype,
        "x_scaler": scaler_from_dict(payload.get("x_scaler")),
        "y_scaler": scaler_from_dict(payload.get("y_scaler")),
        "task_name": str(payload["task_name"]),
        "title": str(payload["title"]),
        "seed": int(payload["seed"]),
        "best_train_loss": float(payload["best_train_loss"]),
        "n_train": int(payload["n_train"]),
        "n_test": int(payload["n_test"]),
        "n_val": int(payload["n_val"]),
        "data_path": payload.get("data_path"),
        "rel_tolerance": float(payload.get("rel_tolerance", 0.01)),
        "standardize_x": bool(payload["standardize_x"]),
        "standardize_y": bool(payload["standardize_y"]),
        "train_idx": payload["train_idx"].cpu().long(),
        "val_idx": payload["val_idx"].cpu().long(),
        "test_idx": payload["test_idx"].cpu().long(),
        "input_column_indices": payload["input_column_indices"].cpu().long(),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = {
        t: SRC_DIR / f"checkpoint_S2_TOA_nTrain16000_nTest5000_sorfD600_{t}.pt"
        for t in SURVIVING_TASKS
    }
    for t, p in ckpts.items():
        if not p.is_file():
            raise FileNotFoundError(p)
        print(f"source {t}: {p}  mtime={time.ctime(p.stat().st_mtime)}")

    # Peek first checkpoint for shared data / split metadata.
    first = load_bundle(ckpts[SURVIVING_TASKS[0]], device="cpu")
    data_path = first["data_path"]
    input_variable = str(first["model_config"].get("input_variable", "toa_reflectance"))
    x_transform = str(first["model_config"].get("x_transform", "none"))
    title = first["title"]
    seed = first["seed"]
    rel_tolerance = first["rel_tolerance"]
    test_idx = first["test_idx"]
    train_idx = first["train_idx"]
    val_idx = first["val_idx"]

    print(f"data_path={data_path}")
    print(f"input_variable={input_variable} x_transform={x_transform}")
    print(f"device={device} out={OUT_DIR}")

    X_np, Y_np, wl, all_names, data_meta = load_s2_arrays(
        data_path,
        input_variable=input_variable,
        task_names=None,
    )
    name_to_col = {n: i for i, n in enumerate(all_names)}
    for t in SURVIVING_TASKS:
        if t not in name_to_col:
            raise KeyError(f"Task {t} not in data columns {all_names}")

    X = torch.tensor(X_np, dtype=torch.float64)
    Y = torch.tensor(Y_np, dtype=torch.float64)
    x_test_full = X[test_idx]
    y_test_full = Y[test_idx]
    wavelengths_np = np.asarray(wl, dtype=np.float64)

    warps = YWarpConfig()

    names = list(SURVIVING_TASKS)
    per_task: dict = {}
    task_metrics: dict = {}
    bands_by_task: dict = {}
    rel_metrics_by_task: dict = {}
    y_pred_all = []
    y_std_all = []
    lower_all = []
    upper_all = []
    y_true_cols = []
    total_pred_time = 0.0

    meta_note = {
        "reeval_source_dir": str(SRC_DIR),
        "reeval_note": (
            "Re-evaluated surviving original Aug05 NIGP_cuda_check checkpoints "
            "(algae/cos_i/cwv/dust/grain_size). aot and liquid_water checkpoints "
            "were overwritten by a later partial re-run and are excluded."
        ),
        "excluded_tasks": ["aot", "liquid_water"],
        "checkpoint_mtimes": {
            t: time.ctime(ckpts[t].stat().st_mtime) for t in SURVIVING_TASKS
        },
    }

    for task_name in names:
        print(f"\n=== Re-eval {task_name} ===")
        bundle = load_bundle(ckpts[task_name], device=device)
        # Sanity: shared split
        if not torch.equal(bundle["test_idx"], test_idx):
            raise RuntimeError(f"{task_name} test_idx differs from algae checkpoint")

        # Copy checkpoint into out dir for provenance.
        shutil.copy2(ckpts[task_name], OUT_DIR / ckpts[task_name].name)

        model = bundle["model"]
        dtype = bundle["dtype"]
        band_indices = bundle["input_column_indices"].tolist()
        bands_by_task[task_name] = band_indices

        x_te = select_bands(x_test_full, band_indices).to(dtype=dtype)
        x_te = apply_x_transform(x_te, x_transform)
        if bundle["standardize_x"] and bundle["x_scaler"] is not None:
            x_te = bundle["x_scaler"].transform(x_te)
        x_te = x_te.to(device=device, dtype=dtype)

        y_te = y_test_full[:, name_to_col[task_name]].to(dtype=torch.float64)

        model.eval()
        model.invalidate_feature_cache()
        t0 = time.time()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model, x_te, chunk_size=512, **rff_eval_kwargs(dtype)
        )
        pred_time = time.time() - t0
        total_pred_time += pred_time

        inv = inverse_y_s2(
            pred_mean.detach().cpu(),
            pred_std.detach().cpu(),
            lower.detach().cpu(),
            upper.detach().cpu(),
            task_name=task_name,
            y_scaler=bundle["y_scaler"],
            standardize_y=bundle["standardize_y"],
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
        print(
            f"{task_name} Test RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}  "
            f"MedAE: {computed.get('MedAE', float('nan')):.6f}"
        )

        nigp_mapped = bundle["model_config"].get("nigp_mapping") or {}
        tm = {
            "best_train_loss": bundle["best_train_loss"],
            "checkpoint_path": str(OUT_DIR / ckpts[task_name].name),
            "checkpoint_source_path": str(ckpts[task_name]),
            "checkpoint_mtime": time.ctime(ckpts[task_name].stat().st_mtime),
            "n_bands": len(band_indices),
            "model_input_dim": int(x_te.shape[-1]),
            "reeval": True,
            **{k: v for k, v in computed.items()},
        }
        if getattr(model, "nigp", False):
            sx = model.input_noise.detach().float().cpu().numpy().tolist()
            tm["input_noise"] = sx
            tm["input_noise_var"] = (np.asarray(sx) ** 2).tolist()
            if "effective_input_term_mean" in nigp_mapped:
                tm["effective_input_term_mean"] = nigp_mapped["effective_input_term_mean"]
            # Prefer live mean sigma_x.
            tm["input_noise_mean"] = float(np.mean(sx))

        task_metrics[task_name] = tm
        y_pred_all.append(pred_mean_p.numpy())
        y_std_all.append(pred_std_p.numpy())
        lower_all.append(lower_p.numpy())
        upper_all.append(upper_p.numpy())
        y_true_cols.append(y_te.cpu().numpy())

        # Free GPU
        del model, bundle
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

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
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_m["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))
    aggregate_rrmse = macro_rrmse(per_task, names)
    aggregate_medae = macro_metric(per_task, names, "MedAE")

    metrics: dict = {
        "title": title,
        "task_names": names,
        "n_train": first["n_train"],
        "n_test": first["n_test"],
        "n_val": first["n_val"],
        "seed": seed,
        "num_rff": 600,
        "rff_sampling": "sorf",
        "nigp": True,
        "pac_bayes": False,
        "rel_tolerance": rel_tolerance,
        "Training_Time": 0.0,
        "Prediction_Time": total_pred_time,
        "Total_Time": total_pred_time,
        "aggregate_RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "x_transform": x_transform,
        "input_variable": input_variable,
        "data_meta": data_meta,
        "reeval_meta": meta_note,
        **per_task,
    }
    for name in names:
        tm = task_metrics[name]
        for k, v in tm.items():
            if k in ("best_train_loss",) or k.endswith("_path") or k in (
                "checkpoint_mtime",
                "reeval",
                "n_bands",
                "model_input_dim",
                "input_noise",
                "input_noise_var",
                "input_noise_mean",
                "effective_input_term_mean",
                "raw_input_noise",
            ):
                metrics[f"{name}_{k}" if not k.startswith(name) else k] = v
            elif k in (
                "RMSE",
                "MAE",
                "MedAE",
                "RRMSE",
                "R2",
                "NIS",
                "NLPD",
                "CRPS",
                "NCRPS",
                "MSE",
                "Training_Time",
                "Prediction_Time",
            ):
                # already in per_task for core metrics; keep task-level extras
                if f"{name}_{k}" not in metrics:
                    metrics[f"{name}_{k}"] = v
            else:
                metrics[f"{name}_{k}"] = v

    print_end_of_run_summary(
        names=names,
        per_task=per_task,
        rel_metrics_by_task=rel_metrics_by_task,
        aggregate_rmse=aggregate_rmse,
        aggregate_rrmse=aggregate_rrmse,
        log_task_names=[],
        aggregate_rrmse_log=float("nan"),
        aggregate_rrmse_mean=float("nan"),
        rel_tolerance=rel_tolerance,
        total_train_time=0.0,
        format_relative_error_summary=format_relative_error_summary,
        aggregate_medae=aggregate_medae,
    )

    save_s2_summary_artifacts(
        save_path=OUT_DIR,
        title=title,
        metrics=metrics,
        names=names,
        y_test_np=y_test_np,
        y_pred_stacked=y_pred_stacked,
        y_std_stacked=y_std_stacked,
        lower_stacked=lower_stacked,
        upper_stacked=upper_stacked,
        x_test_orig=x_test_full.numpy(),
        wavelengths_np=wavelengths_np,
        train_idx=train_idx.numpy(),
        val_idx=val_idx.numpy(),
        test_idx=test_idx.numpy(),
        bands_by_task=bands_by_task,
        rel_metrics_by_task=rel_metrics_by_task,
        rel_tolerance=rel_tolerance,
        log_scale=False,
        log_scale_tasks=frozenset(),
        data_meta=data_meta,
        seed=seed,
        posterior_n_examples=0,
        posterior_example_indices=None,
        plot_validation=False,
        plot_posterior=False,
        monitor_validation=False,
        n_val=0,
        save_metrics_json=save_metrics_json,
        warps=warps,
    )

    # Also write a small sidecar README
    readme = OUT_DIR / "REEVAL_README.txt"
    readme.write_text(
        "\n".join(
            [
                meta_note["reeval_note"],
                f"source: {SRC_DIR}",
                f"tasks: {', '.join(names)}",
                f"aggregate_RRMSE: {aggregate_rrmse}",
                "",
                "Per-task RRMSE:",
                *[f"  {n}: {per_task[f'{n}_RRMSE']}" for n in names],
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWrote metrics under {OUT_DIR}")


if __name__ == "__main__":
    main()
