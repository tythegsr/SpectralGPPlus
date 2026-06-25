"""
Multi-fidelity Wing benchmark with GPPlus SORF kernel (Woodbury inference).

Uses RFFGPR + LogScaleKernel(RFFKernel) only — same training path as A4_ackley_SORF.py.
Data from load_experimental_data.generate_mf_wing_data (10 continuous + source column).
When only one fidelity has samples, the source column is dropped automatically (10D inputs).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from load_experimental_data import generate_mf_wing_data
from sorf_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    compute_val_samples_per_source,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    save_metrics_json,
    scale_validation_tensors,
    summarize_validation_from_runs,
    unpack_train_val_test,
)

WING_CONT_DIM = 10
WING_INPUT_DIM_MF = 11
WING_NUM_SOURCES = 4
WING_SOURCE_NAMES = ("s0", "s1", "s2", "s3")


def _expand_per_source(values: int | float | list, length: int = WING_NUM_SOURCES) -> list:
    if isinstance(values, (int, float)):
        return [float(values)] * length
    if len(values) != length:
        raise ValueError(f"Expected length-{length} per-source list, got {len(values)}")
    return [float(v) for v in values]


def _active_source_indices(
    train_samples_per_source: list[int],
    test_samples_per_source: list[int],
) -> list[int]:
    """Source indices with at least one train or test sample."""
    active = []
    for idx, (n_train, n_test) in enumerate(zip(train_samples_per_source, test_samples_per_source)):
        if n_train > 0 or n_test > 0:
            active.append(idx)
    return active


def _prepare_wing_inputs(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    train_samples_per_source: list[int],
    test_samples_per_source: list[int],
    *,
    drop_source_column: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int | None, bool]:
    """
    Drop the source-id column when exactly one fidelity is used.

    Returns (x_train, x_test, input_dim, source_index, dropped_source_column).
    """
    active = _active_source_indices(train_samples_per_source, test_samples_per_source)
    if len(active) == 0:
        raise ValueError("At least one source must have train or test samples.")
    if len(active) > 1 and drop_source_column:
        raise ValueError(
            "drop_source_column=True requires a single active source; "
            f"active sources: {[WING_SOURCE_NAMES[i] for i in active]}"
        )

    single_fidelity = len(active) == 1
    should_drop = drop_source_column if drop_source_column is not None else single_fidelity

    if not should_drop:
        return x_train, x_test, WING_INPUT_DIM_MF, active[0] if single_fidelity else None, False

    if not single_fidelity:
        raise ValueError(
            "Cannot drop source column for multi-fidelity Wing; "
            f"active sources: {[WING_SOURCE_NAMES[i] for i in active]}"
        )

    return (
        x_train[:, :WING_CONT_DIM].contiguous(),
        x_test[:, :WING_CONT_DIM].contiguous(),
        WING_CONT_DIM,
        active[0],
        True,
    )


def run_wing_sorf(
    train_samples_per_source: list[int] | None = None,
    test_samples_per_source: list[int] | None = None,
    num_sorf: int | None = None,
    noise_train: float | list[float] = 0.0,
    noise_test: float | list[float] = 0.0,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 8,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_SORF/results/wing_sorf",
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    drop_source_column: bool | None = None,
    monitor_validation: bool = False,
    val_fraction: float = 0.2,
    validation_verbose: bool = True,
) -> dict:
    """
    Train SORF-GP on multi-fidelity Wing and evaluate on held-out test points.

    Parameters
    ----------
    train_samples_per_source : samples per fidelity source (s0–s3); default 25 each.
    test_samples_per_source : test samples per source; default 500 each.
    num_sorf : D in RFF (m = 2*D). Default: min(512, n_train // 3).
    num_epochs : 1 uses LBFGSScipy; >1 uses torch.optim.Adam.
    drop_source_column : If True, use 10 continuous inputs only (single fidelity).
        If None (default), drop automatically when exactly one source has samples.
    """
    if train_samples_per_source is None:
        train_samples_per_source = [25, 25, 25, 25]
    if test_samples_per_source is None:
        test_samples_per_source = [500, 500, 500, 500]
    if len(train_samples_per_source) != WING_NUM_SOURCES or len(test_samples_per_source) != WING_NUM_SOURCES:
        raise ValueError(
            f"train/test samples per source must have length {WING_NUM_SOURCES}; "
            f"got train={train_samples_per_source}, test={test_samples_per_source}"
        )

    set_seed(seed)
    n_train = sum(train_samples_per_source)
    n_test = sum(test_samples_per_source)
    if num_sorf is None:
        num_sorf = min(512, max(64, n_train // 3))

    train_noise = _expand_per_source(noise_train)
    test_noise = _expand_per_source(noise_test)

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    active_sources = _active_source_indices(train_samples_per_source, test_samples_per_source)
    single_fidelity = len(active_sources) == 1
    fidelity_source = active_sources[0] if single_fidelity else None
    will_drop_source = (
        drop_source_column if drop_source_column is not None else single_fidelity
    )

    title = (
        f"Wing_tr{train_samples_per_source}_te{test_samples_per_source}_"
        f"sorfD{num_sorf}_noiseTest{noise_test}_noiseTrain{noise_train}"
    )
    if will_drop_source and fidelity_source is not None:
        title = title.replace("Wing_", f"Wing_{WING_SOURCE_NAMES[fidelity_source]}_", 1)
    print("=" * 60)
    print(title)
    feature_dim = 2 * num_sorf
    input_dim_planned = WING_CONT_DIM if will_drop_source else WING_INPUT_DIM_MF
    print(
        f"SORF kernel (Woodbury), D={num_sorf}, m={feature_dim}, ARD={ard}, "
        f"input_dim={input_dim_planned}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}"
    )
    if single_fidelity:
        print(
            f"Single fidelity: {WING_SOURCE_NAMES[fidelity_source]} "
            f"(source column {'dropped' if will_drop_source else 'kept'})"
        )
    else:
        print(f"Multi-fidelity: active sources {[WING_SOURCE_NAMES[i] for i in active_sources]}")
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Woodbury: n_train={n_train}, n_test={n_test}, m/n={feature_dim / n_train:.4f}")
    if feature_dim >= n_train:
        print(
            f"WARNING: m={feature_dim} >= n_train={n_train}; Woodbury may not beat dense GP. "
            f"Consider num_sorf <= {max(1, n_train // 2 - 1)}."
        )
    print("=" * 60)

    val_samples_per_source = (
        compute_val_samples_per_source(train_samples_per_source, val_fraction)
        if monitor_validation
        else [0, 0, 0, 0]
    )
    n_val = sum(val_samples_per_source)
    if monitor_validation and validation_verbose:
        print(f"Validation monitoring: n_val={n_val} ({val_fraction:.0%} of n_train={n_train})")

    data = generate_mf_wing_data(
        train_samples_per_source=train_samples_per_source,
        test_samples_per_source=test_samples_per_source,
        val_samples_per_source=val_samples_per_source,
        seed=seed,
        train_noise=train_noise,
        test_noise=test_noise,
        noise_type=noise_type,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = unpack_train_val_test(data)

    x_train = x_train.to(dtype=dtype)
    x_val = x_val.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    x_train, x_test, input_dim, fidelity_source, dropped_source_column = _prepare_wing_inputs(
        x_train,
        x_test,
        train_samples_per_source,
        test_samples_per_source,
        drop_source_column=drop_source_column,
    )
    if dropped_source_column and x_val.numel() > 0:
        x_val = x_val[:, :WING_CONT_DIM].contiguous()
    if dropped_source_column:
        print(f"Using {input_dim} continuous inputs (dropped source-id column).")

    x_scaling_type = "None"
    x_scaler = None
    if standardize_x:
        if x_standardize_method == 0:
            x_scaler = StandardScaler()
            x_scaling_type = "StandardScaler (Gaussian)"
        elif x_standardize_method == 1:
            x_scaler = UniformScaler(scale_to_neg_one=False)
            x_scaling_type = "UniformScaler [0, 1]"
        elif x_standardize_method == 2:
            x_scaler = UniformScaler(scale_to_neg_one=True)
            x_scaling_type = "UniformScaler [-1, 1]"
        else:
            raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
        x_scaler.fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        print(f"X scaling: {x_scaling_type}")

    y_mean, y_std = None, None
    y_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.unsqueeze(-1))
        y_mean, y_std = y_scaler.mean.squeeze(), y_scaler.std.squeeze()
        y_train = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)
        y_test_scaled = y_scaler.transform(y_test.unsqueeze(-1)).squeeze(-1)
    else:
        y_test_scaled = y_test

    x_val_scaled, y_val_scaled = scale_validation_tensors(
        x_val,
        y_val,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
        dtype=dtype,
    )
    callbacks = []
    if monitor_validation and n_val > 0:
        callbacks.append(
            make_validation_callback(
                x_val_scaled,
                y_val_scaled,
                num_inits,
                chunk_size=predict_chunk_size,
                verbose=validation_verbose,
            )
        )

    model = RFFGPR(x_train, y_train, num_rff=num_sorf, ard=ard, rff_sampling="sorf")

    trainer = GPTrainer(
        model,
        mll_class=RFFWoodburyMarginalLogLikelihood,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        initializer_class=RFFParameterInitializer,
        n_jobs=n_jobs,
        inner_max_num_threads=1,
        cholesky_jitter=1e-6,
        callbacks=callbacks,
    )
    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All training runs failed. "
            + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
        )
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])
    learned_noise = extract_learned_likelihood_noise(model, y_std=y_std)

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
        model, x_test, chunk_size=predict_chunk_size
    )
    prediction_time = time.time() - t_pred
    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu()
    lower = lower.detach().cpu()
    upper = upper.detach().cpu()

    if standardize_y:
        pred_mean = pred_mean * y_std.cpu() + y_mean.cpu()
        pred_std = pred_std * y_std.cpu()
        lower = lower * y_std.cpu() + y_mean.cpu()
        upper = upper * y_std.cpu() + y_mean.cpu()
        y_test_eval = y_test.cpu()
    else:
        y_test_eval = y_test_scaled.cpu()

    computed = compute_metrics(
        y_test_eval,
        pred_mean,
        output_std=pred_std,
        lower_95=lower,
        upper_95=upper,
        training_time=train_time,
        prediction_time=prediction_time,
    )

    metrics = {
        "title": title,
        "input_dim": input_dim,
        "single_fidelity": single_fidelity,
        "fidelity_source": WING_SOURCE_NAMES[fidelity_source] if fidelity_source is not None else None,
        "fidelity_source_index": fidelity_source,
        "drop_source_column": dropped_source_column,
        "active_sources": [WING_SOURCE_NAMES[i] for i in active_sources],
        "n_sources": WING_NUM_SOURCES,
        "train_samples_per_source": train_samples_per_source,
        "test_samples_per_source": test_samples_per_source,
        "n_train": n_train,
        "n_test": n_test,
        "num_sorf": num_sorf,
        "rff_sampling": "sorf",
        "feature_dim": 2 * num_sorf,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "best_train_loss": best_loss,
        "noise_train": train_noise,
        "noise_test": test_noise,
        "noise_type": noise_type,
        **learned_noise,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        **computed,
    }
    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["val_fraction"] = val_fraction
        metrics["n_val"] = n_val
        metrics["val_samples_per_source"] = val_samples_per_source
        metrics.update(summarize_validation_from_runs(runs, best_run))

    print(
        f"\nTest RMSE: {computed['RMSE']:.6f}  RRMSE: {computed['RRMSE']:.6f}  "
        f"MAE: {computed['MAE']:.6f}"
    )
    if "NIS" in computed:
        print(f"NIS: {computed['NIS']:.4f}")
    print(f"Best training loss: {best_loss:.4f}  Time: {train_time:.1f}s")

    if save_path:
        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-fidelity Wing with GPPlus SORF")
    parser.add_argument(
        "--train-per-source",
        type=int,
        nargs=4,
        default=[400, 0, 0, 0],
        metavar=("S0", "S1", "S2", "S3"),
        help="Training samples per fidelity source",
    )
    parser.add_argument(
        "--test-per-source",
        type=int,
        nargs=4,
        default=[5000, 0, 0, 0],
        metavar=("S0", "S1", "S2", "S3"),
        help="Test samples per fidelity source",
    )
    parser.add_argument("--num-sorf", type=int, default=100, help="D (SORF frequencies); default min(512, n_train//3)")
    parser.add_argument("--noise-train", type=float, default=0.005, help="Train noise scale (all sources)")
    parser.add_argument("--noise-test", type=float, default=0.005, help="Test noise scale (all sources)")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=("gaussian", "uniform"))
    parser.add_argument("--num-inits", type=int, default=16)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Epochs per init: 1 uses LBFGSScipy; >1 uses torch.optim.Adam",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Adam learning rate (only when --num-epochs > 1; default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Test points per Woodbury predict chunk (0 = single batch)",
    )
    parser.add_argument(
        "--ard",
       type=bool,
       default=True,
       help="Automatic relevance determination",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel hyperparameter inits (-1 = all cores)",
    )
    parser.add_argument("--save-path", type=str, default="experiments_SORF/results/wing_sorf")
    args = parser.parse_args()

    gpplus.config.configure_logger()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    run_wing_sorf(
        train_samples_per_source=list(args.train_per_source),
        test_samples_per_source=list(args.test_per_source),
        num_sorf=args.num_sorf,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        noise_type=args.noise_type,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        save_path=args.save_path,
        ard=args.ard,
        n_jobs=n_jobs,
        predict_chunk_size=args.predict_chunk_size,
    )
