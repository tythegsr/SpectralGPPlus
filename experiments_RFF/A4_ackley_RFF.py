"""
Ackley XXD (XX=number of dimensions) benchmark with GPPlus RFF kernel (Woodbury inference).

Uses RFFGPR + LogScaleKernel(RFFKernel) only — no SEEK or other composite kernels.
Tuned defaults align with experiments_revisions_april/A4_ackley_GPvsPFN.py (Gaussian baseline).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR):
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
from load_experimental_data import generate_ackley_data
from rff_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    VAL_SEED_OFFSET,
    compute_n_val,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    save_metrics_json,
    plot_validation_curves_after_save,
    scale_validation_tensors,
    summarize_validation_from_runs,
    unpack_train_val_test,
)


def run_ackley_40d_rff(
    dimensions: int = 40,
    train_size: int = 40,
    num_rff: int | None = None,
    num_test: int = 5000,
    x_bounds: tuple[float, float] = (-5.0, 10.0),
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 8,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_RFF/results/ackley_40D_rff",
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    monitor_validation: bool = True,
    val_fraction: float = 0.2,
    validation_verbose: bool = True,
    plot_validation: bool = True,
) -> dict:
    """
    Train RFF-GP on Ackley and evaluate on held-out Sobol test points.

    Parameters
    ----------
    train_size : training points per input dimension (total train n = train_size * dimensions).
    num_rff : D in RFF (feature dimension m = 2*D). Default: min(512, n_train // 3).
    num_epochs : Training epochs per init. Use 1 with LBFGSScipy; increase for Adam.
    """
    set_seed(seed)
    n_train = train_size * dimensions
    if num_rff is None:
        num_rff = min(512, max(64, n_train // 3))

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    title = (
        f"Ackley_{dimensions}Dx_{train_size}Dn_{list(x_bounds)}_"
        f"rffD{num_rff}_noiseTest{noise_test}_noiseTrain{noise_train}"
    )
    print("=" * 60)
    print(title)
    feature_dim = 2 * num_rff
    print(
        f"RFF kernel (Woodbury), D={num_rff}, m={feature_dim}, ARD={ard}, "
        f"dtype={dtype}, inits={num_inits}, epochs={num_epochs}"
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Woodbury: n_train={n_train}, m/n={feature_dim / n_train:.4f}")
    if feature_dim >= n_train:
        print(
            f"WARNING: m={feature_dim} >= n_train={n_train}; Woodbury may not beat dense GP. "
            f"Consider num_rff <= {max(1, n_train // 2 - 1)}."
        )
    print("=" * 60)

    n_val = compute_n_val(n_train, val_fraction) if monitor_validation else 0
    if monitor_validation and validation_verbose:
        print(f"Validation monitoring: n_val={n_val} ({val_fraction:.0%} of n_train={n_train})")

    data = generate_ackley_data(
        n_train=n_train,
        n_test=num_test,
        n_val=n_val,
        dimensions=dimensions,
        x_bounds=list(x_bounds),
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = unpack_train_val_test(data)

    x_train = x_train.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

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

    model = RFFGPR(x_train, y_train, num_rff=num_rff, ard=ard)

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
        "dimensions": dimensions,
        "n_train": n_train,
        "n_test": num_test,
        "num_rff": num_rff,
        "feature_dim": 2 * num_rff,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "best_train_loss": best_loss,
        "noise_train": noise_train,
        "noise_test": noise_test,
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
        if plot_validation and monitor_validation and n_val > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ackley 40D with GPPlus RFF (no SEEK)")
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=0, help="train points per dimension")
    parser.add_argument("--num-rff", type=int, default=500, help="D (RFF frequencies); default min(512, n_train//3)")
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--noise-train", type=float, default=0.005)
    parser.add_argument("--noise-test", type=float, default=0.005)
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
        help="float32 is faster on CPU with similar quality for RFF",
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Test points per Woodbury predict chunk (0 = single batch)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel hyperparameter inits (-1 = all cores)",
    )
    parser.add_argument(
        "--ard",
       type=bool,
       default=False,
       help="Automatic relevance determination",
    )
    parser.add_argument("--save-path", type=str, default="experiments_RFF/results/ackley_test_val")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip validation curve plots after saving JSON",
    )
    args = parser.parse_args()

    gpplus.config.configure_logger()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    run_ackley_40d_rff(
        dimensions=args.dimensions,
        train_size=args.train_size,
        num_rff=args.num_rff,
        num_test=args.num_test,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        noise_type=args.noise_type,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=args.ard,
        save_path=args.save_path,
        n_jobs=n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        plot_validation=not args.no_plot,
    )
