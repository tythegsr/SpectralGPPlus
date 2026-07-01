"""
Shared TOA benchmark runner using independent RFFGPR models (Woodbury inference).

Trains two separate RFFGPR models on y_cos and y_grain with shared scaled inputs.
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
_RFF_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIRS = {
    "rff": "experiments_RFF/results/toa_rff",
    "orf": "experiments_ORF/results/toa_orf",
    "sorf": "experiments_SORF/results/toa_sorf",
}


def _pin_experiment_paths() -> None:
    """Ensure MTGPR helpers/loaders resolve before per-folder copies."""
    ordered = (str(_MTGPR_DIR), str(_RFF_DIR), str(_ROOT))
    sys.path[:] = list(ordered) + [p for p in sys.path if p not in ordered]


_pin_experiment_paths()

from gpplus.models import RFFGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
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
    make_train_loss_callback,
    make_validation_callback,
    save_metrics_json,
    summarize_validation_from_runs,
    unpack_train_val_test,
)
from plot_toa_posterior import (
    plot_toa_posterior_figures,
    save_predictions_npz,
    select_posterior_example_indices,
    wavelength_axis,
)
from rff_experiment_utils import extract_learned_likelihood_noise
from toa_mtgpr_base import (
    TASK_NAMES,
    TOA_INPUT_DIM,
    compute_per_task_metrics,
    drop_input_columns,
    normalize_columns_to_drop,
)
from toa_stgp_checkpoint import checkpoint_path_for_run, save_toa_stgp_checkpoint
from toa_y_transform import forward_y_single, inverse_y_single

NUM_TASKS = 2
RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")


def _plot_task_validation_curves(
    metrics: dict,
    save_path: str | Path,
    task_name: str,
    json_path: str | None = None,
) -> list[str]:
    """Plot validation curves for one task using prefixed validation_metrics_by_init."""
    block_key = f"{task_name}_validation_metrics_by_init"
    if block_key not in metrics:
        return []
    from plot_validation_curves import plot_run

    task_metrics = {
        "monitor_validation": True,
        "validation_metrics_by_init": metrics[block_key],
        "best_init_index": metrics.get(f"{task_name}_best_init_index"),
        "best_val_NLL": metrics.get(f"{task_name}_best_val_NLL"),
        "best_val_RRMSE": metrics.get(f"{task_name}_best_val_RRMSE"),
        "title": metrics.get("title", ""),
    }
    if json_path:
        task_metrics["_source_file"] = json_path
    out_dir = Path(save_path) / "validation" / task_name
    try:
        return [str(p) for p in plot_run(task_metrics, out_dir)]
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Validation plot generation failed for %s: %s", task_name, exc
        )
        return []


def run_toa_stgp(
    n_train: int = 10000,
    n_test: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "rff",
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
    """Train independent RFFGPR models on TOA data and evaluate on held-out test points."""
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")

    if save_path is None:
        save_path = _DEFAULT_SAVE_DIRS[rff_sampling]

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

    title = f"TOA_nTrain{n_train}_nTest{n_test}_{rff_sampling}D{num_rff}"
    feature_dim = 2 * num_rff
    sampling_label = rff_sampling.upper()
    print("=" * 60)
    print(title)
    print(
        f"Independent {sampling_label}-GP (Woodbury), D={num_rff}, m={feature_dim}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}, "
        f"tasks={TASK_NAMES}, log_grain={log_grain}"
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
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

    data = load_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        seed=seed,
        data_path=data_path,
    )
    x_train, y_train, x_val, y_val, x_test, y_test, train_idx, val_idx, test_idx = unpack_train_val_test(
        data
    )

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

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, dict] = {}
    task_runs: dict[str, list] = {}
    task_best_runs: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}

    for task_idx, task_name in enumerate(TASK_NAMES):
        print(f"\n--- Task: {task_name} ---")
        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_va = y_val[:, task_idx] if y_val.numel() > 0 else y_val

        y_tr_model = forward_y_single(y_tr, task_name, log_grain=log_grain)
        y_te_model = forward_y_single(y_te, task_name, log_grain=log_grain)

        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))
            y_tr_fit = y_scaler.transform(y_tr_model.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit = y_tr_model

        y_val_scaled = y_va
        if y_va.numel() > 0:
            y_va_model = forward_y_single(y_va, task_name, log_grain=log_grain)
            if standardize_y and y_scaler is not None:
                y_val_scaled = y_scaler.transform(y_va_model.unsqueeze(-1)).squeeze(-1)
            else:
                y_val_scaled = y_va_model

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
                    x_val,
                    y_val_scaled,
                    num_inits,
                    chunk_size=predict_chunk_size,
                    verbose=validation_verbose,
                    log_every_n_epochs=log_every_n_epochs,
                )
            )

        likelihood = None
        initializer_kwargs: dict | None = None
        noise_prior_meta: dict | None = None
        if response_noise_prior:
            from gpplus.priors.response_noise import (
                build_scalar_noise_likelihood,
                empirical_scalar_noise_variance,
                log_normal_scalar_noise_prior_from_responses,
                scalar_noise_raw_init_from_variance,
            )

            target_var = empirical_scalar_noise_variance(y_tr_fit, fraction=noise_var_fraction)
            noise_prior = log_normal_scalar_noise_prior_from_responses(
                y_tr_fit,
                fraction=noise_var_fraction,
                log_scale=noise_prior_log_scale,
                dtype=dtype,
                device=y_tr_fit.device,
            )
            likelihood = build_scalar_noise_likelihood(noise_prior=noise_prior)
            raw_init = scalar_noise_raw_init_from_variance(
                likelihood, target_var.to(dtype=dtype)
            )
            initializer_kwargs = {
                "parameter_configs": {
                    "raw_noise": {"method": "constant", "value": raw_init},
                }
            }
            noise_prior_meta = {
                "noise_prior_target_var": float(target_var.detach().cpu()),
                "noise_prior_loc": float(noise_prior.loc.detach().cpu()),
            }
            print(
                f"Response noise prior ({task_name}): LogNormal, fraction={noise_var_fraction}, "
                f"log_scale={noise_prior_log_scale}, target_var={noise_prior_meta['noise_prior_target_var']}"
            )

        model = RFFGPR(
            x_train,
            y_tr_fit,
            likelihood=likelihood,
            num_rff=num_rff,
            ard=ard,
            rff_sampling=rff_sampling,
        )
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
        y_std_for_noise = y_scaler.std.squeeze() if y_scaler is not None else None
        learned_noise = extract_learned_likelihood_noise(model, y_std=y_std_for_noise)

        if save_checkpoint and save_path:
            ckpt_path = save_toa_stgp_checkpoint(
                checkpoint_path_for_run(save_path, title, task_name),
                model=model,
                task_name=task_name,
                train_x=x_train.cpu(),
                train_y=y_tr_fit.cpu(),
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
                    "num_rff": num_rff,
                    "ard": ard,
                    "rff_sampling": rff_sampling,
                },
            )
            learned_noise["checkpoint_path"] = str(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        model.eval()
        model.invalidate_feature_cache()
        t_pred = time.time()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model, x_test, chunk_size=predict_chunk_size
        )
        prediction_time = time.time() - t_pred
        total_prediction_time += prediction_time

        pred_mean, pred_std, lower, upper = inverse_y_single(
            pred_mean.detach().cpu(),
            pred_std.detach().cpu(),
            lower.detach().cpu(),
            upper.detach().cpu(),
            task_name=task_name,
            y_scaler=y_scaler,
            standardize_y=standardize_y,
            log_grain=log_grain,
        )
        y_test_eval = y_te.cpu()

        computed = compute_metrics(
            y_test_eval,
            pred_mean,
            output_std=pred_std,
            lower_95=lower,
            upper_95=upper,
            training_time=train_time,
            prediction_time=prediction_time,
        )

        tm: dict = {
            "best_train_loss": best_loss,
            **learned_noise,
            **{k: v for k, v in computed.items()},
        }
        if noise_prior_meta is not None:
            tm.update(noise_prior_meta)
        task_metrics[task_name] = tm
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run

        y_pred_all.append(pred_mean.numpy())
        y_std_all.append(pred_std.numpy())
        lower_all.append(lower.numpy())
        upper_all.append(upper.numpy())

        print(
            f"{task_name} Test RMSE: {computed['RMSE']:.6f}  "
            f"RRMSE: {computed['RRMSE']:.6f}  MAE: {computed['MAE']:.6f}"
        )
        print(f"{task_name} best loss: {best_loss:.4f}  train time: {train_time:.1f}s")

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_test_np = y_test.detach().cpu().numpy()
    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked)

    for t, name in enumerate(TASK_NAMES):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t],
            y_pred_stacked[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))
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
        "ard": ard,
        "model_class": "RFFGPR",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_grain": log_grain,
        "response_noise_prior": bool(response_noise_prior),
        "rel_tolerance": rel_tolerance,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "RMSE": aggregate_rmse,
        **per_task,
    }

    for task_name in TASK_NAMES:
        tm = task_metrics[task_name]
        metrics[f"{task_name}_best_train_loss"] = tm["best_train_loss"]
        metrics[f"{task_name}_raw_noise"] = tm["raw_noise"]
        metrics[f"{task_name}_noise"] = tm["noise"]
        metrics[f"{task_name}_noise_std"] = tm["noise_std"]
        if "checkpoint_path" in tm:
            metrics[f"{task_name}_checkpoint_path"] = tm["checkpoint_path"]
        for key, value in tm.items():
            if key in ("best_train_loss", "raw_noise", "noise", "noise_std", "checkpoint_path"):
                continue
            metrics[f"{task_name}_{key}"] = value

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

    print(f"\nTest aggregate RMSE: {aggregate_rmse:.6f}")
    for name in TASK_NAMES:
        print(
            f"{name} RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"RRMSE: {per_task[f'{name}_RRMSE']:.6f}"
        )
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
    print(f"Total training time: {total_train_time:.1f}s")

    if save_path:
        example_indices = select_posterior_example_indices(
            y_test_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )

        out_npz = save_predictions_npz(
            save_path,
            title,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            y_std=y_std_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
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
        metrics["predictions_npz"] = out_npz

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_validation and monitor_validation and n_val > 0:
            for task_name in TASK_NAMES:
                for plot_path in _plot_task_validation_curves(
                    metrics, save_path, task_name, json_path=out_json
                ):
                    print(f"Saved validation plot to {plot_path}")

        if plot_posterior and example_indices:
            import logging

            from plot_multid_slice_predictions import sanitize_plot_subdir

            post_dir = Path(save_path) / "plots" / "posterior" / sanitize_plot_subdir(title)
            try:
                post_paths = plot_toa_posterior_figures(
                    x_test_orig.numpy(),
                    y_test_np,
                    y_pred_stacked,
                    y_std_stacked,
                    lower_stacked,
                    upper_stacked,
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


def run_toa_rff(**kwargs) -> dict:
    """Train independent RFF-GP models on TOA (rff_sampling='rff')."""
    return run_toa_stgp(rff_sampling="rff", **kwargs)
