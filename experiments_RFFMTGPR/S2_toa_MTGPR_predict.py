"""Reload a saved TOA RFFMTGPR checkpoint and evaluate on the test split."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.training import evaluate_rff_mt_gp_model
from load_experimental_data import load_toa_data
from mtgpr_experiment_utils import (
    compute_relative_error_metrics,
    format_relative_error_summary,
    unpack_train_val_test,
)
from plot_toa_posterior import (
    plot_toa_posterior_figures,
    save_predictions_npz,
    select_posterior_example_indices,
    wavelength_axis,
)
from plot_validation_curves import sanitize_plot_subdir
from toa_mtgpr_base import TASK_NAMES, compute_per_task_metrics, select_input_columns
from toa_mtgpr_checkpoint import load_toa_mtgpr_checkpoint
from toa_y_transform import inverse_y_predictions


def predict_from_checkpoint(
    bundle,
    *,
    predict_chunk_size: int = 512,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Load test data from checkpoint split metadata, predict, return original-scale arrays."""
    data = load_toa_data(
        n_train=bundle.n_train,
        n_test=bundle.n_test,
        n_val=bundle.n_val,
        seed=bundle.seed,
        data_path=bundle.data_path,
    )
    _x_train, _y_train, _x_val, _y_val, x_test, y_test, _train_idx, _val_idx, test_idx = (
        unpack_train_val_test(data)
    )

    if not torch.equal(test_idx.cpu(), bundle.test_idx.cpu()):
        raise RuntimeError(
            "Test split from load_toa_data does not match checkpoint test_idx; "
            "check data_path, seed, and split sizes."
        )

    x_test_orig = x_test.clone()
    dtype = bundle.dtype
    x_test = x_test.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    if bundle.input_column_indices.numel() < x_test.shape[-1]:
        x_test = select_input_columns(x_test, bundle.input_column_indices)

    if bundle.standardize_x and bundle.x_scaler is not None:
        x_test = bundle.x_scaler.transform(x_test)

    t0 = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_mt_gp_model(
        bundle.model,
        x_test.to(device=bundle.model.train_inputs[0].device),
        chunk_size=predict_chunk_size,
    )
    pred_time = time.time() - t0

    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu()
    lower = lower.detach().cpu()
    upper = upper.detach().cpu()

    pred_mean, pred_std, lower, upper = inverse_y_predictions(
        pred_mean,
        pred_std,
        lower,
        upper,
        y_scaler=bundle.y_scaler,
        standardize_y=bundle.standardize_y,
        log_grain=bundle.log_grain,
    )
    y_test_eval = y_test.cpu()

    y_pred_np = pred_mean.numpy()
    y_true_np = y_test_eval.numpy()

    return (
        y_true_np,
        y_pred_np,
        pred_std.numpy(),
        lower.numpy(),
        upper.numpy(),
        x_test_orig.numpy(),
        test_idx.cpu().numpy(),
        pred_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved TOA RFFMTGPR checkpoint on the held-out test split"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_*.pt saved by S1_toa_MTGPR.py",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Test points per Woodbury predict chunk (0 = single batch)",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Directory to write predictions NPZ (default: checkpoint parent dir)",
    )
    parser.add_argument(
        "--plot-posterior",
        action="store_true",
        help="Write 3-panel posterior figures for selected test examples",
    )
    parser.add_argument(
        "--posterior-n-examples",
        type=int,
        default=8,
        help="Number of test examples to plot when --plot-posterior is set",
    )
    parser.add_argument(
        "--posterior-example-indices",
        type=str,
        default=None,
        help="Comma-separated test row indices to plot (overrides --posterior-n-examples)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    gpplus.config.configure_logger(level=getattr(logging, args.log_level))

    ckpt_path = Path(args.checkpoint)
    bundle = load_toa_mtgpr_checkpoint(ckpt_path, device=args.device)
    print(f"Loaded checkpoint: {bundle.title}")
    print(f"  n_train={bundle.n_train}, n_test={bundle.n_test}, n_val={bundle.n_val}")
    print(f"  best_train_loss={bundle.best_train_loss:.4f}")
    print(f"  log_grain={bundle.log_grain}")

    (
        y_true_np,
        y_pred_np,
        pred_std_np,
        lower_np,
        upper_np,
        x_test_orig_np,
        test_idx_np,
        pred_time,
    ) = predict_from_checkpoint(bundle, predict_chunk_size=args.predict_chunk_size)

    per_task = compute_per_task_metrics(y_true_np, y_pred_np)
    aggregate_rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))

    print(f"\nTest aggregate RMSE: {aggregate_rmse:.6f}  (predict time: {pred_time:.1f}s)")
    rel_metrics_by_task: dict[str, dict] = {}
    for name in TASK_NAMES:
        rel_m = compute_relative_error_metrics(
            y_true_np[:, TASK_NAMES.index(name)],
            y_pred_np[:, TASK_NAMES.index(name)],
            rel_tolerance=bundle.rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        print(
            f"{name} RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"RRMSE: {per_task[f'{name}_RRMSE']:.6f}"
        )
        print(format_relative_error_summary(name, rel_m, rel_tolerance=bundle.rel_tolerance))

    if args.save_predictions is not None or args.plot_posterior:
        save_dir = args.save_predictions or str(ckpt_path.parent)

        posterior_example_indices = None
        if args.posterior_example_indices:
            posterior_example_indices = [
                int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()
            ]

        example_indices = select_posterior_example_indices(
            y_true_np.shape[0],
            args.posterior_n_examples,
            seed=bundle.seed,
            explicit_indices=posterior_example_indices,
        )

        if args.save_predictions is not None:
            out_npz = save_predictions_npz(
                save_dir,
                bundle.title,
                y_true=y_true_np,
                y_pred=y_pred_np,
                y_std=pred_std_np,
                lower=lower_np,
                upper=upper_np,
                x_test_orig=x_test_orig_np,
                test_idx=test_idx_np,
                task_names=TASK_NAMES,
                seed=bundle.seed,
                rel_tolerance=bundle.rel_tolerance,
                example_indices=example_indices,
                wavelength_nm=wavelength_axis(x_test_orig_np.shape[-1]),
                log_grain=bundle.log_grain,
            )
            print(f"Saved predictions to {out_npz}")

        if args.plot_posterior and example_indices:
            post_dir = Path(save_dir) / "plots" / "posterior" / sanitize_plot_subdir(bundle.title)
            post_paths = plot_toa_posterior_figures(
                x_test_orig_np,
                y_true_np,
                y_pred_np,
                pred_std_np,
                lower_np,
                upper_np,
                post_dir,
                title=bundle.title,
                example_indices=example_indices,
                rel_metrics_by_task=rel_metrics_by_task,
                rel_tolerance=bundle.rel_tolerance,
                wavelength_nm=wavelength_axis(x_test_orig_np.shape[-1]),
                log_grain=bundle.log_grain,
            )
            for plot_path in post_paths:
                print(f"Saved posterior plot to {plot_path}")


if __name__ == "__main__":
    main()
