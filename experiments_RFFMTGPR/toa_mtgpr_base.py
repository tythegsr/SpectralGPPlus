"""
Shared TOA benchmark runner using joint RFFMTGPR (Woodbury inference).

Trains one RFFMTGPR on y_cos and y_grain jointly with shared scaled inputs.
Sampling mode (RFF, ORF, SORF) is selected via rff_sampling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpplus.models import RFFMTGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    RFFMTParameterInitializer,
    RFFMTWoodburyMarginalLogLikelihood,
    evaluate_rff_mt_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gpplus.utils.rff_utils import woodbury_jitter_for_dtype
from load_experimental_data import load_toa_data
from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    compute_n_val,
    compute_relative_error_metrics,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    plot_validation_curves_after_save,
    save_metrics_json,
    summarize_validation_from_runs,
    unpack_train_val_test,
    make_train_loss_callback,
)
from toa_mtgpr_checkpoint import checkpoint_path_for_run, save_toa_mtgpr_checkpoint
from toa_y_transform import forward_y, inverse_y_predictions

TOA_INPUT_DIM = 285
NUM_TASKS = 2
TASK_NAMES = ("y_cos", "y_grain")
RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")


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


def extract_mt_learned_noise(model) -> dict[str, float | list[float]]:
    noises = model.task_noises().detach().cpu().tolist()
    return {
        "task_noises": noises,
        "task_noise_stds": [float(np.sqrt(n)) for n in noises],
    }


def normalize_columns_to_drop(
    columns_to_drop: list[int] | np.ndarray | torch.Tensor | None,
    *,
    input_dim: int = TOA_INPUT_DIM,
) -> list[int]:
    """Validate 0-based column indices to remove from X (last dimension)."""
    if columns_to_drop is None:
        return []
    drop = torch.as_tensor(columns_to_drop, dtype=torch.int64).reshape(-1).unique(sorted=True)
    if drop.numel() == 0:
        return []
    if torch.any(drop < 0) or torch.any(drop >= input_dim):
        bad = drop[(drop < 0) | (drop >= input_dim)].tolist()
        raise ValueError(
            f"columns_to_drop out of range for input_dim={input_dim}: {bad}"
        )
    return [int(i) for i in drop.tolist()]


def kept_column_indices(
    columns_to_drop: list[int] | np.ndarray | torch.Tensor | None,
    *,
    input_dim: int = TOA_INPUT_DIM,
) -> torch.Tensor:
    """Column indices retained after dropping ``columns_to_drop``."""
    drop = set(normalize_columns_to_drop(columns_to_drop, input_dim=input_dim))
    keep = [i for i in range(input_dim) if i not in drop]
    if not keep:
        raise ValueError("Cannot drop all input columns.")
    return torch.tensor(keep, dtype=torch.int64)


def drop_input_columns(
    x: torch.Tensor,
    columns_to_drop: list[int] | np.ndarray | torch.Tensor | None,
    *,
    input_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove feature columns from ``x`` (shape ``[..., D]``).

    Returns
    -------
    x_reduced, kept_indices : reduced tensor and int64 indices of kept columns.
    """
    if input_dim is None:
        input_dim = int(x.shape[-1])
    keep = kept_column_indices(columns_to_drop, input_dim=input_dim)
    if not normalize_columns_to_drop(columns_to_drop, input_dim=input_dim):
        return x, keep
    idx = keep.to(device=x.device)
    return x.index_select(-1, idx), keep


def select_input_columns(
    x: torch.Tensor,
    column_indices: list[int] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Keep only ``column_indices`` from the last dimension of ``x``."""
    idx = torch.as_tensor(column_indices, dtype=torch.int64, device=x.device)
    return x.index_select(-1, idx)


def run_toa_mtgpr(
    n_train: int = 10000,
    n_test: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "orf",
    seed: int = 42,
    num_inits: int = 8,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    save_path: str | None = None,
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
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 8,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    rank_kernel: int = 1,
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 1,
    save_checkpoint: bool = True,
    log_grain: bool = True,
    drop_columns: list[int] | None = None,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.5,
) -> dict:
    """Train joint RFFMTGPR on TOA data and evaluate on held-out test points."""
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")

    if save_path is None:
        save_path = f"experiments_RFFMTGPR/results/toa_{rff_sampling}"

    set_seed(seed)
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

    title = f"TOA_nTrain{n_train}_nTest{n_test}_{rff_sampling}D{num_rff}_mt"
    feature_dim = 2 * num_rff
    joint_width = feature_dim * NUM_TASKS
    sampling_label = rff_sampling.upper()
    print("=" * 60)
    print(title)
    print(
        f"Joint {sampling_label}-MTGP (Woodbury), D={num_rff}, m={feature_dim}, m*T={joint_width}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}, tasks={TASK_NAMES}, "
        f"log_grain={log_grain}"
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
    print(f"Woodbury: n_train={n_train}, nT={n_train * NUM_TASKS}, joint cols={joint_width}")
    if joint_width >= n_train * NUM_TASKS:
        print(
            f"WARNING: m*T={joint_width} >= n*T={n_train * NUM_TASKS}; "
            "Woodbury may not beat dense GP."
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
    x_train, y_train, x_val, y_val, x_test, y_test, train_idx, val_idx, test_idx = unpack_train_val_test(data)

    x_test_orig = x_test.clone()
    input_dim_original = TOA_INPUT_DIM
    dropped_columns = normalize_columns_to_drop(drop_columns, input_dim=input_dim_original)
    x_train, kept_column_indices_t = drop_input_columns(
        x_train, dropped_columns, input_dim=input_dim_original
    )
    x_test, _ = drop_input_columns(x_test, dropped_columns, input_dim=input_dim_original)
    if x_val.numel() > 0:
        x_val, _ = drop_input_columns(x_val, dropped_columns, input_dim=input_dim_original)
    input_dim = int(x_train.shape[-1])
    kept_column_indices = kept_column_indices_t.tolist()
    if dropped_columns:
        print(
            f"Input columns: dropped {len(dropped_columns)} / {input_dim_original} "
            f"-> input_dim={input_dim}"
        )
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
    y_val = y_val.to(dtype=dtype)

    y_train_model = forward_y(y_train, log_grain=log_grain)
    y_test_model = forward_y(y_test, log_grain=log_grain)

    y_mean, y_std = None, None
    y_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_model)
        y_mean, y_std = y_scaler.mean.squeeze(0), y_scaler.std.squeeze(0)
        y_train_fit = y_scaler.transform(y_train_model)
        y_test_scaled = y_scaler.transform(y_test_model)
    else:
        y_train_fit = y_train_model
        y_test_scaled = y_test_model

    x_val_scaled = x_val
    y_val_scaled = y_val
    if y_val.numel() > 0:
        y_val_model = forward_y(y_val, log_grain=log_grain)
        if standardize_y and y_scaler is not None:
            y_val_scaled = y_scaler.transform(y_val_model)
        else:
            y_val_scaled = y_val_model
        x_val_scaled = x_val

    callbacks = []
    if num_epochs > 1 and training_verbose:
        callbacks.append(
            make_train_loss_callback(
                num_inits,
                num_epochs,
                verbose=training_verbose,
                log_every_n_epochs=log_every_n_epochs,
            )
        )
    if monitor_validation and n_val > 0:
        callbacks.append(
            make_validation_callback(
                x_val_scaled,
                y_val_scaled,
                num_inits,
                chunk_size=predict_chunk_size,
                verbose=validation_verbose,
                log_every_n_epochs=log_every_n_epochs,
            )
        )

    noise_prior = None
    initializer_kwargs: dict | None = None
    noise_prior_meta: dict | None = None
    if response_noise_prior:
        from gpplus.priors.response_noise import (
            build_multitask_noise_likelihood,
            empirical_task_noise_variances,
            log_normal_noise_prior_from_responses,
            task_noise_raw_init_from_variances,
        )

        target_vars = empirical_task_noise_variances(y_train_fit, fraction=noise_var_fraction)
        noise_prior = log_normal_noise_prior_from_responses(
            y_train_fit,
            fraction=noise_var_fraction,
            log_scale=noise_prior_log_scale,
            dtype=dtype,
            device=y_train_fit.device,
        )
        temp_lik = build_multitask_noise_likelihood(NUM_TASKS, rank=0)
        raw_init = task_noise_raw_init_from_variances(temp_lik, target_vars.to(dtype=dtype))
        initializer_kwargs = {
            "parameter_configs": {
                "raw_task_noises": {"method": "constant", "value": raw_init},
            }
        }
        noise_prior_meta = {
            "response_noise_prior": True,
            "noise_var_fraction": float(noise_var_fraction),
            "noise_prior_log_scale": float(noise_prior_log_scale),
            "noise_prior_loc": noise_prior.loc.detach().cpu().tolist(),
            "noise_prior_target_var": target_vars.detach().cpu().tolist(),
        }
        print(
            f"Response noise prior: LogNormal per task, fraction={noise_var_fraction}, "
            f"log_scale={noise_prior_log_scale}, target_var={noise_prior_meta['noise_prior_target_var']}"
        )

    model = RFFMTGPR(
        x_train,
        y_train_fit,
        num_tasks=NUM_TASKS,
        num_rff=num_rff,
        ard=ard,
        rff_sampling=rff_sampling,
        rank_kernel=rank_kernel,
        rank_likelihood=0,
        noise_prior=noise_prior,
    )
    trainer = GPTrainer(
        model,
        mll_class=RFFMTWoodburyMarginalLogLikelihood,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        initializer_class=RFFMTParameterInitializer,
        initializer_kwargs=initializer_kwargs,
        n_jobs=n_jobs,
        inner_max_num_threads=1,
        cholesky_jitter=woodbury_jitter_for_dtype(dtype),
        callbacks=callbacks,
        stop_conditions=[
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
        parallel_verbose=parallel_verbose,
    )
    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All training runs failed for joint TOA model. "
            + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
        )
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])
    learned_noise = extract_mt_learned_noise(model)

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_mt_gp_model(
        model, x_test, chunk_size=predict_chunk_size
    )
    prediction_time = time.time() - t_pred

    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu()
    lower = lower.detach().cpu()
    upper = upper.detach().cpu()

    pred_mean, pred_std, lower, upper = inverse_y_predictions(
        pred_mean,
        pred_std,
        lower,
        upper,
        y_scaler=y_scaler,
        standardize_y=standardize_y,
        log_grain=log_grain,
    )
    y_test_eval = y_test.cpu()

    y_pred_np = pred_mean.numpy()
    y_true_np = y_test_eval.numpy()
    per_task = compute_per_task_metrics(y_true_np, y_pred_np)

    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}
    for t, name in enumerate(TASK_NAMES):
        rel_m = compute_relative_error_metrics(
            y_true_np[:, t],
            y_pred_np[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    metrics: dict = {
        "title": title,
        "input_dim": input_dim,
        "input_dim_original": input_dim_original,
        "dropped_columns": dropped_columns,
        "kept_column_indices": kept_column_indices,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": NUM_TASKS,
        "task_names": list(TASK_NAMES),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "feature_dim": feature_dim,
        "joint_feature_dim": joint_width,
        "rank_kernel": rank_kernel,
        "ard": ard,
        "model_class": "RFFMTGPR",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_grain": log_grain,
        "response_noise_prior": bool(response_noise_prior),
        "best_train_loss": best_loss,
        "rel_tolerance": rel_tolerance,
        **learned_noise,
        "Training_Time": train_time,
        "Prediction_Time": prediction_time,
        "Total_Time": train_time + prediction_time,
        "RMSE": aggregate_rmse,
        **per_task,
    }
    if noise_prior_meta is not None:
        metrics.update(noise_prior_meta)

    for t, name in enumerate(TASK_NAMES):
        yt = y_test_eval[:, t].numpy()
        yp = pred_mean[:, t].numpy()
        computed = compute_metrics(
            torch.from_numpy(yt),
            torch.from_numpy(yp),
            output_std=pred_std[:, t],
            lower_95=lower[:, t],
            upper_95=upper[:, t],
            training_time=train_time / NUM_TASKS,
            prediction_time=prediction_time / NUM_TASKS,
        )
        for key, value in computed.items():
            metrics[f"{name}_{key}"] = value

    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["val_fraction"] = val_fraction
        metrics["n_val"] = n_val
        val_summary = summarize_validation_from_runs(runs, best_run)
        metrics.update(val_summary)

    print(f"\nTest aggregate RMSE: {aggregate_rmse:.6f}")
    for name in TASK_NAMES:
        print(
            f"{name} RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"RRMSE: {per_task[f'{name}_RRMSE']:.6f}"
        )
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
    print(f"best loss: {best_loss:.4f}  train time: {train_time:.1f}s")

    if save_path:
        if save_checkpoint:
            ckpt_path = save_toa_mtgpr_checkpoint(
                checkpoint_path_for_run(save_path, title),
                model=model,
                train_x=x_train.cpu(),
                train_y=y_train_fit.cpu(),
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                x_standardize_method=x_standardize_method,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                title=title,
                seed=seed,
                best_train_loss=best_loss,
                n_train=n_train,
                n_test=n_test,
                n_val=n_val,
                data_path=data_path,
                rel_tolerance=rel_tolerance,
                dtype=dtype,
                log_grain=log_grain,
                input_column_indices=kept_column_indices_t,
                model_config={
                    "num_tasks": NUM_TASKS,
                    "num_rff": num_rff,
                    "ard": ard,
                    "rff_sampling": rff_sampling,
                    "rank_kernel": rank_kernel,
                    "rank_likelihood": 0,
                },
            )
            metrics["checkpoint_path"] = str(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        from plot_toa_posterior import (
            plot_toa_posterior_figures,
            save_predictions_npz,
            select_posterior_example_indices,
            wavelength_axis,
        )

        example_indices = select_posterior_example_indices(
            y_true_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )

        out_npz = save_predictions_npz(
            save_path,
            title,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_std=pred_std.numpy(),
            lower=lower.numpy(),
            upper=upper.numpy(),
            x_test_orig=x_test_orig.numpy(),
            test_idx=test_idx.cpu().numpy(),
            task_names=TASK_NAMES,
            seed=seed,
            rel_tolerance=rel_tolerance,
            example_indices=example_indices,
            wavelength_nm=wavelength_axis(x_test_orig.shape[-1]),
            log_grain=log_grain,
        )
        print(f"Saved predictions to {out_npz}")

        if plot_validation and monitor_validation and n_val > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")

        if plot_posterior and example_indices:
            import logging

            from plot_validation_curves import sanitize_plot_subdir

            post_dir = Path(save_path) / "plots" / "posterior" / sanitize_plot_subdir(title)
            try:
                post_paths = plot_toa_posterior_figures(
                    x_test_orig.numpy(),
                    y_true_np,
                    y_pred_np,
                    pred_std.numpy(),
                    lower.numpy(),
                    upper.numpy(),
                    post_dir,
                    title=title,
                    example_indices=example_indices,
                    rel_metrics_by_task=rel_metrics_by_task,
                    rel_tolerance=rel_tolerance,
                    wavelength_nm=wavelength_axis(x_test_orig.shape[-1]),
                    log_grain=log_grain,
                )
                for plot_path in post_paths:
                    print(f"Saved posterior plot to {plot_path}")
            except Exception as exc:
                logging.getLogger(__name__).warning("Posterior plot generation failed: %s", exc)

    return metrics
