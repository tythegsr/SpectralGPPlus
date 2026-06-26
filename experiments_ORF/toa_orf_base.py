"""
Shared TOA benchmark runner using ORF-GP (Woodbury inference).

Trains two independent RFFGPR models on y_cos and y_grain with shared scaled inputs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_ORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _ORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from load_experimental_data import load_toa_data
from orf_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    compute_n_val,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    plot_validation_curves_after_save,
    save_metrics_json,
    scale_validation_tensors,
    summarize_validation_from_runs,
    unpack_train_val_test,
)

TOA_INPUT_DIM = 285
NUM_TASKS = 2
TASK_NAMES = ("y_cos", "y_grain")


def compute_per_task_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    task_names: tuple[str, ...] = TASK_NAMES,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, len(task_names))
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, len(task_names))
    metrics: dict[str, float] = {}
    for t, name in enumerate(task_names):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        std = float(np.std(yt))
        rrmse = rmse / std if std > 0 else float("inf")
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_RRMSE"] = rrmse
        metrics[f"{name}_R2"] = r2
    return metrics


def run_toa_orf(
    n_train: int = 10000,
    n_test: int = 5000,
    num_orf: int | None = None,
    seed: int = 42,
    num_inits: int = 8,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_ORF/results/toa_orf",
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    monitor_validation: bool = False,
    val_fraction: float = 0.2,
    validation_verbose: bool = True,
    plot_validation: bool = True,
    data_path: str | None = None,
) -> dict:
    """Train ORF-GP on TOA data (two independent scalar models) and evaluate on held-out test points."""
    set_seed(seed)
    if num_orf is None:
        num_orf = min(512, max(64, n_train // 3))

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    title = f"TOA_nTrain{n_train}_nTest{n_test}_orfD{num_orf}"
    feature_dim = 2 * num_orf
    print("=" * 60)
    print(title)
    print(
        f"ORF kernel (Woodbury), D={num_orf}, m={feature_dim}, ARD={ard}, "
        f"dtype={dtype}, inits={num_inits}, epochs={num_epochs}, tasks={TASK_NAMES}"
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Woodbury: n_train={n_train}, m/n={feature_dim / n_train:.4f}")
    if feature_dim >= n_train:
        print(
            f"WARNING: m={feature_dim} >= n_train={n_train}; Woodbury may not beat dense GP. "
            f"Consider num_orf <= {max(1, n_train // 2 - 1)}."
        )
    print("=" * 60)

    n_val = compute_n_val(n_train, val_fraction) if monitor_validation else 0
    if monitor_validation and validation_verbose:
        print(f"Validation monitoring: n_val={n_val} ({val_fraction:.0%} of n_train={n_train})")

    data = load_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        seed=seed,
        data_path=data_path,
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

    if standardize_x and x_scaler is not None and x_val.numel() > 0:
        x_val = x_scaler.transform(x_val.to(dtype=dtype))
    else:
        x_val = x_val.to(dtype=dtype)

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, float | dict] = {}
    task_runs: dict[str, list] = {}
    task_best_runs: dict[str, dict] = {}
    y_pred_all = []

    for task_idx, task_name in enumerate(TASK_NAMES):
        print(f"\n--- Task: {task_name} ---")
        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_va = y_val[:, task_idx] if y_val.numel() > 0 else y_val

        y_mean, y_std = None, None
        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr.unsqueeze(-1))
            y_mean, y_std = y_scaler.mean.squeeze(), y_scaler.std.squeeze()
            y_tr_fit = y_scaler.transform(y_tr.unsqueeze(-1)).squeeze(-1)
            y_te_scaled = y_scaler.transform(y_te.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit = y_tr
            y_te_scaled = y_te

        x_val_scaled, y_val_scaled = scale_validation_tensors(
            x_val,
            y_va,
            x_scaler=None,
            y_scaler=y_scaler,
            standardize_x=False,
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

        model = RFFGPR(x_train, y_tr_fit, num_rff=num_orf, ard=ard, rff_sampling="orf")
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
        total_train_time += train_time

        successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
        if not successful:
            errors = [r.get("error", "unknown") for r in runs if r.get("error")]
            raise RuntimeError(
                f"All training runs failed for {task_name}. "
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
        total_prediction_time += prediction_time

        pred_mean = pred_mean.detach().cpu()
        pred_std = pred_std.detach().cpu()
        lower = lower.detach().cpu()
        upper = upper.detach().cpu()

        if standardize_y:
            pred_mean = pred_mean * y_std.cpu() + y_mean.cpu()
            pred_std = pred_std * y_std.cpu()
            lower = lower * y_std.cpu() + y_mean.cpu()
            upper = upper * y_std.cpu() + y_mean.cpu()
            y_test_eval = y_te.cpu()
        else:
            y_test_eval = y_te_scaled.cpu()

        computed = compute_metrics(
            y_test_eval,
            pred_mean,
            output_std=pred_std,
            lower_95=lower,
            upper_95=upper,
            training_time=train_time,
            prediction_time=prediction_time,
        )

        task_metrics[task_name] = {
            "best_train_loss": best_loss,
            **learned_noise,
            **{f"{task_name}_{k}": v for k, v in computed.items()},
        }
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run
        y_pred_all.append(pred_mean.numpy())

        print(
            f"{task_name} Test RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}"
        )
        print(f"{task_name} best loss: {best_loss:.4f}  train time: {train_time:.1f}s")

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_test_np = y_test.detach().cpu().numpy()
    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked)

    metrics: dict = {
        "title": title,
        "input_dim": TOA_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": NUM_TASKS,
        "task_names": list(TASK_NAMES),
        "num_orf": num_orf,
        "rff_sampling": "orf",
        "feature_dim": feature_dim,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        **per_task,
    }

    for task_name in TASK_NAMES:
        tm = task_metrics[task_name]
        metrics[f"{task_name}_best_train_loss"] = tm["best_train_loss"]
        metrics[f"{task_name}_raw_noise"] = tm["raw_noise"]
        metrics[f"{task_name}_noise"] = tm["noise"]
        metrics[f"{task_name}_noise_std"] = tm["noise_std"]
        for key, value in tm.items():
            if key.startswith(f"{task_name}_") and key not in metrics:
                metrics[key] = value

    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["val_fraction"] = val_fraction
        metrics["n_val"] = n_val
        for task_name in TASK_NAMES:
            val_summary = summarize_validation_from_runs(
                task_runs[task_name], task_best_runs[task_name]
            )
            for key, value in val_summary.items():
                metrics[f"{task_name}_{key}"] = value

    print(f"\nTotal training time: {total_train_time:.1f}s")

    if save_path:
        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")
        if plot_validation and monitor_validation and n_val > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")

    return metrics
