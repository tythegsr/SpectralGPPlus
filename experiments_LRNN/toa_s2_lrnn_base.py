"""S2 independent LRNN/DBK runner with per-QoI original-band subsets."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_LRNN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_DEFAULT_SAVE_PATH = "experiments_LRNN/results/s2_toa_lrnn"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _LRNN_DIR)

from experiments_toa.data import TOA_TEST_POOL_SIZE, TOA_TRAIN_POOL_SIZE, TOA_VAL_POOL_SIZE
from experiments_toa.s2_bands import band_config_metadata, load_task_band_config
from experiments_toa.s2_constants import S2_DEFAULT_BAND_CONFIG_PATH, S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_data import load_s2_toa_data
from experiments_toa.s2_plotting import (
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import compute_per_task_metrics, macro_rrmse, select_bands
from experiments_toa.s2_y_transform import forward_y_s2, inverse_y_s2, task_uses_log_scale
from gpplus.models import LRNNGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    DefaultParameterInitializer,
    GPTrainer,
    LRNNWoodburyMarginalLogLikelihood,
    MinLossChangeStopCondition,
    evaluate_lrnn_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gpplus.utils.lrnn_utils import woodbury_jitter_for_lrnn
from mtgpr_experiment_utils import (
    DEFAULT_LBFGS_KWARGS,
    compute_relative_error_metrics,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_train_loss_callback,
    make_validation_callback,
    save_metrics_json,
    summarize_validation_from_runs,
)
from rff_experiment_utils import extract_learned_likelihood_noise
from toa_lrnn_base import DEFAULT_LRNN_ADAM_KWARGS, DEFAULT_LRNN_ADAM_STOP_PATIENCE, resolve_activation
from toa_lrnn_checkpoint import checkpoint_path_for_run, save_toa_lrnn_checkpoint


def run_s2_toa_lrnn(
    n_train: int = 16000,
    n_test: int = 5000,
    hidden_dims: Sequence[int] = (128, 256, 512, 1024, 512, 256),
    feature_rank: int = 128,
    activation: str = "tanh",
    variance_correction: bool = True,
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 1000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    log_scale: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    monitor_validation: bool = True,
    validation_verbose: bool = True,
    plot_validation: bool = True,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 1,
    save_checkpoint: bool = True,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.01,
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
) -> dict:
    if save_path is None:
        save_path = _DEFAULT_SAVE_PATH
    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    band_cfg_path = task_band_config or str(S2_DEFAULT_BAND_CONFIG_PATH)
    bands_by_task = load_task_band_config(band_cfg_path, task_names=names, input_dim=S2_INPUT_DIM)
    set_seed(seed)

    act_cls = resolve_activation(activation)
    hidden = [int(h) for h in hidden_dims]
    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ]
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = dict(DEFAULT_LRNN_ADAM_KWARGS)
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=DEFAULT_LRNN_ADAM_STOP_PATIENCE),
        ]
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    hid_tag = "-".join(str(h) for h in hidden)
    title = f"S2_TOA_nTrain{n_train}_nTest{n_test}_lrnnR{feature_rank}_h{hid_tag}"
    print("=" * 60)
    print(title)
    print(f"S2 LRNN/DBK tasks={names}, band_config={band_cfg_path}")
    print("=" * 60)

    n_val = TOA_VAL_POOL_SIZE if monitor_validation else 0
    (
        x_train_full,
        y_train,
        x_val_full,
        y_val,
        x_test_full,
        y_test,
        train_idx,
        val_idx,
        test_idx,
        wavelengths,
        data_meta,
    ) = load_s2_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        seed=seed,
        data_path=data_path,
        input_variable=input_variable,  # type: ignore[arg-type]
        task_names=names,
    )
    wavelengths_np = wavelengths.detach().cpu().numpy()
    x_test_orig = x_test_full.clone()
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, dict] = {}
    task_runs: dict[str, list] = {}
    task_best_runs: dict[str, dict] = {}
    input_bands_by_task: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    rel_metrics_by_task: dict[str, dict] = {}

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        x_tr = select_bands(x_train_full, band_indices).to(dtype=dtype)
        x_te = select_bands(x_test_full, band_indices).to(dtype=dtype)
        x_va = (
            select_bands(x_val_full, band_indices).to(dtype=dtype)
            if x_val_full.numel() > 0
            else x_val_full.to(dtype=dtype)
        )

        x_scaler = None
        if standardize_x:
            if x_standardize_method == 0:
                x_scaler = StandardScaler()
            elif x_standardize_method == 1:
                x_scaler = UniformScaler(scale_to_neg_one=False)
            elif x_standardize_method == 2:
                x_scaler = UniformScaler(scale_to_neg_one=True)
            else:
                raise ValueError(f"x_standardize_method must be 0/1/2, got {x_standardize_method}")
            x_scaler.fit(x_tr)
            x_tr = x_scaler.transform(x_tr)
            x_te = x_scaler.transform(x_te)
            if x_va.numel() > 0:
                x_va = x_scaler.transform(x_va)

        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_va = y_val[:, task_idx] if y_val.numel() > 0 else y_val
        y_tr_model = forward_y_s2(y_tr, task_name, log_scale=log_scale)
        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))
            y_tr_fit = y_scaler.transform(y_tr_model.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit = y_tr_model
        y_val_scaled = y_va
        if y_va.numel() > 0:
            y_va_model = forward_y_s2(y_va, task_name, log_scale=log_scale)
            y_val_scaled = (
                y_scaler.transform(y_va_model.unsqueeze(-1)).squeeze(-1)
                if standardize_y and y_scaler is not None
                else y_va_model
            )

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
                    x_va,
                    y_val_scaled,
                    num_inits,
                    chunk_size=predict_chunk_size,
                    verbose=validation_verbose,
                    log_every_n_epochs=log_every_n_epochs,
                )
            )

        likelihood = None
        initializer_kwargs = None
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
            raw_init = scalar_noise_raw_init_from_variance(likelihood, target_var.to(dtype=dtype))
            initializer_kwargs = {
                "parameter_configs": {"raw_noise": {"method": "constant", "value": raw_init}}
            }

        model = LRNNGPR(
            x_tr,
            y_tr_fit,
            likelihood=likelihood,
            hidden_dims=hidden,
            feature_rank=feature_rank,
            activation=act_cls,
            variance_correction=variance_correction,
        )
        trainer = GPTrainer(
            model,
            mll_class=LRNNWoodburyMarginalLogLikelihood,
            num_epochs=num_epochs,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=dtype,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_class=DefaultParameterInitializer,
            initializer_kwargs=initializer_kwargs,
            n_jobs=n_jobs,
            cholesky_jitter=woodbury_jitter_for_lrnn(dtype),
            callbacks=callbacks,
            stop_conditions=stop_conditions,
            parallel_verbose=parallel_verbose,
        )
        t0 = time.time()
        runs = trainer.train()
        train_time = time.time() - t0
        total_train_time += train_time
        successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
        if not successful:
            raise RuntimeError(f"All training runs failed for {task_name}")
        best_run = min(successful, key=lambda r: r["loss"])
        model.load_state_dict(best_run["state_dict"])
        best_loss = float(best_run["loss"])
        y_std_for_noise = y_scaler.std.squeeze() if y_scaler is not None else None
        learned_noise = extract_learned_likelihood_noise(model, y_std=y_std_for_noise)
        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": [float(wavelengths_np[i]) for i in band_indices],
            "model_input_dim": int(x_tr.shape[-1]),
            "ard_space": "lrnn_features",
        }

        if save_checkpoint and save_path:
            ckpt_path = save_toa_lrnn_checkpoint(
                checkpoint_path_for_run(save_path, title, task_name),
                model=model,
                task_name=task_name,
                train_x=x_tr.cpu(),
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
                log_grain=task_uses_log_scale(task_name, log_scale=log_scale),
                logit_cos=False,
                input_column_indices=torch.as_tensor(band_indices, dtype=torch.int64),
                model_config={
                    "hidden_dims": hidden,
                    "feature_rank": feature_rank,
                    "activation": activation,
                    "variance_correction": variance_correction,
                    "dataset": "s2",
                    "band_indices": list(band_indices),
                    "log_scale": task_uses_log_scale(task_name, log_scale=log_scale),
                },
            )
            learned_noise["checkpoint_path"] = str(ckpt_path)

        model.eval()
        t1 = time.time()
        pred_mean, lower, upper, pred_std = evaluate_lrnn_gp_model(
            model, x_te, chunk_size=predict_chunk_size
        )
        prediction_time = time.time() - t1
        total_prediction_time += prediction_time
        inv = inverse_y_s2(
            pred_mean.detach().cpu(),
            pred_std.detach().cpu(),
            lower.detach().cpu(),
            upper.detach().cpu(),
            task_name=task_name,
            y_scaler=y_scaler,
            standardize_y=standardize_y,
            log_scale=log_scale,
            extended=True,
        )
        pred_mean, pred_std, lower, upper = inv.as_tuple()

        computed = compute_metrics(
            y_te.cpu(),
            pred_mean,
            output_std=pred_std,
            lower_95=lower,
            upper_95=upper,
            training_time=train_time,
            prediction_time=prediction_time,
        )
        if task_uses_log_scale(task_name, log_scale=log_scale):
            computed["log_scale"] = True
        tm = {"best_train_loss": best_loss, **learned_noise, **computed}
        task_metrics[task_name] = tm
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run
        y_pred_all.append(pred_mean.numpy())
        y_std_all.append(pred_std.numpy())
        lower_all.append(lower.numpy())
        upper_all.append(upper.numpy())
        print(f"{task_name} RRMSE: {computed['RRMSE']:.6f}")

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_test_np = y_test.detach().cpu().numpy()
    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked, names)
    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t], y_pred_stacked[:, t], rel_tolerance=rel_tolerance
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])

    aggregate_rrmse = macro_rrmse(per_task, names)
    metrics: dict = {
        "title": title,
        "dataset": "s2",
        "input_dim_original": S2_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "model_class": "LRNNGPR",
        "hidden_dims": hidden,
        "feature_rank": feature_rank,
        "activation": activation,
        "variance_correction": variance_correction,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "standardize_y": standardize_y,
        "log_scale": bool(log_scale),
        "log_scale_tasks": [n for n in names if task_uses_log_scale(n, log_scale=log_scale)],
        "task_band_config": str(band_cfg_path),
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wavelengths_np),
        "input_bands_by_task": input_bands_by_task,
        "data_meta": data_meta,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_RRMSE_mean": aggregate_rrmse,
        **per_task,
    }
    for task_name in names:
        for key, value in task_metrics[task_name].items():
            metrics[f"{task_name}_{key}"] = value
    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["n_val"] = n_val
        metrics["train_pool_size"] = TOA_TRAIN_POOL_SIZE
        metrics["val_pool_size"] = TOA_VAL_POOL_SIZE
        metrics["test_pool_size"] = TOA_TEST_POOL_SIZE
        for task_name in names:
            val_summary = summarize_validation_from_runs(
                task_runs[task_name], task_best_runs[task_name]
            )
            for key, value in val_summary.items():
                metrics[f"{task_name}_{key}"] = value

    if save_path:
        example_indices = select_posterior_example_indices(
            y_test_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        out_npz = save_s2_predictions_npz(
            save_path,
            title=title,
            task_names=names,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            y_std=y_std_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
            x_test=x_test_orig.numpy(),
            wavelengths_nm=wavelengths_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
        )
        metrics["predictions_npz"] = str(out_npz)
        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")
        if plot_posterior:
            scatter_dir = Path(save_path) / "plots" / "scatter" / title
            for t, name in enumerate(names):
                plot_s2_task_scatter(
                    y_true=y_test_np[:, t],
                    y_pred=y_pred_stacked[:, t],
                    lower=lower_stacked[:, t],
                    upper=upper_stacked[:, t],
                    task_name=name,
                    out_path=scatter_dir / f"{name}_scatter.png",
                )
            plot_s2_posterior_examples(
                x_test=x_test_orig.numpy(),
                wavelengths_nm=wavelengths_np,
                y_true=y_test_np,
                y_pred=y_pred_stacked,
                lower=lower_stacked,
                upper=upper_stacked,
                task_names=names,
                example_indices=example_indices,
                save_dir=Path(save_path) / "plots" / "posterior" / title,
                title=title,
                y_std=y_std_stacked,
                rel_metrics_by_task=rel_metrics_by_task,
                rel_tolerance=rel_tolerance,
                log_scale_tasks=[n for n in names if task_uses_log_scale(n, log_scale=log_scale)],
            )
    return metrics
