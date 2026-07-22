"""
S2 joint RFFMTGPR runner (shared-band multitask Woodbury).

Uses the band config ``default`` ranges for a single shared X (not per-QoI keeps).
Log-scale QoIs are transformed column-wise via ``s2_y_transform``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR)

from experiments_toa.data import (
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
)
from experiments_toa.s2_bands import band_config_metadata, expand_band_spec
from experiments_toa.s2_constants import (
    S2_DEFAULT_BAND_CONFIG_PATH,
    S2_INPUT_DIM,
    S2_TASK_NAMES,
)
from experiments_toa.s2_data import load_s2_toa_data
from experiments_toa.s2_plotting import (
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import (
    compute_log_scale_extra_metrics,
    compute_per_task_metrics,
    macro_metric,
    macro_rrmse,
    select_bands,
)
from experiments_toa.s2_y_transform import (
    InverseYOutput,
    forward_y_s2,
    task_uses_log_scale,
    _lognormal_original_scale,
)
from gpplus.models import RFFMTGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    RFFMTParameterInitializer,
    evaluate_rff_mt_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from experiments_RFF.rff_gp_defaults import (
    DEFAULT_MT_WOODBURY_METHOD,
    build_rff_multitask_noise_likelihood,
    merge_mt_noise_initializer_kwargs,
    mt_eval_kwargs,
    mt_mll_class,
    woodbury_jitter_for_dtype,
)
from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    DEFAULT_TOA_ADAM_LR,
    DEFAULT_TOA_ADAM_STOP_PATIENCE,
    compute_relative_error_metrics,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_train_loss_callback,
    make_validation_callback,
    plot_validation_curves_after_save,
    save_metrics_json,
    summarize_validation_from_runs,
)
from toa_mtgpr_base import extract_mt_learned_noise
from toa_mtgpr_checkpoint import checkpoint_path_for_run, save_toa_mtgpr_checkpoint

RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")


def _shared_default_bands(task_band_config: str | Path | None) -> list[int]:
    """Load the band config ``default`` ranges (shared X for joint MT)."""
    path = Path(task_band_config) if task_band_config is not None else S2_DEFAULT_BAND_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg_dim = int(payload.get("input_dim", S2_INPUT_DIM))
    if cfg_dim != S2_INPUT_DIM:
        raise ValueError(f"Config input_dim={cfg_dim} != expected {S2_INPUT_DIM}")
    return expand_band_spec(payload.get("default"), input_dim=S2_INPUT_DIM)


def forward_y_s2_matrix(
    y: torch.Tensor,
    task_names: Sequence[str],
    *,
    log_scale: bool = True,
) -> torch.Tensor:
    out = y.clone()
    for t, name in enumerate(task_names):
        out[:, t] = forward_y_s2(y[:, t], name, log_scale=log_scale)
    return out


def inverse_y_s2_matrix(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    task_names: Sequence[str],
    y_scaler: StandardScaler | None,
    standardize_y: bool,
    log_scale: bool = True,
) -> InverseYOutput:
    """Unstandardize multitask preds, then inverse log-scale columns."""
    pred_mean = pred_mean.clone()
    pred_std = pred_std.clone()
    lower = lower.clone()
    upper = upper.clone()
    if standardize_y and y_scaler is not None:
        y_mean = y_scaler.mean.squeeze(0)
        y_std = y_scaler.std.squeeze(0)
        pred_mean = pred_mean * y_std + y_mean
        pred_std = pred_std * y_std
        lower = lower * y_std + y_mean
        upper = upper * y_std + y_mean

    point_mean = pred_mean.clone()
    point_mode = pred_mean.clone()
    log_mu = torch.full_like(pred_mean, float("nan"))
    log_sigma = torch.full_like(pred_mean, float("nan"))

    for t, name in enumerate(task_names):
        if not task_uses_log_scale(name, log_scale=log_scale):
            continue
        mu_log = pred_mean[:, t]
        sigma_log = pred_std[:, t]
        median, mean, mode, ln_std, lo, hi = _lognormal_original_scale(mu_log, sigma_log)
        pred_mean[:, t] = median
        pred_std[:, t] = ln_std
        lower[:, t] = lo
        upper[:, t] = hi
        point_mean[:, t] = mean
        point_mode[:, t] = mode
        log_mu[:, t] = mu_log
        log_sigma[:, t] = sigma_log

    any_log = any(task_uses_log_scale(n, log_scale=log_scale) for n in task_names)
    return InverseYOutput(
        point=pred_mean,
        std=pred_std,
        lower=lower,
        upper=upper,
        point_mean=point_mean if any_log else None,
        point_mode=point_mode if any_log else None,
        log_mu=log_mu if any_log else None,
        log_sigma=log_sigma if any_log else None,
    )


def run_s2_toa_mtgpr(
    n_train: int = 16000,
    n_test: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "sorf",
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 200,
    device: str = "cuda",
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
    validation_verbose: bool = True,
    plot_validation: bool = True,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    rank_kernel: int = 1,
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 50,
    save_checkpoint: bool = True,
    log_scale: bool = True,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.5,
    correct_sorf: bool = True,
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
) -> dict:
    """Train joint RFFMTGPR on S2 QoIs with shared default-band inputs."""
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")
    correct_sorf = bool(correct_sorf) if rff_sampling == "sorf" else False

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    unknown = [n for n in names if n not in S2_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S2 task names: {unknown}")
    num_tasks = len(names)
    if num_tasks < 1:
        raise ValueError("Need at least one QoI for multitask training")

    shared_bands = _shared_default_bands(task_band_config)
    bands_by_task = {name: list(shared_bands) for name in names}

    if save_path is None:
        save_path = f"experiments_RFFMTGPR/results/s2_toa_mtgpr_{rff_sampling}"

    set_seed(seed)
    if num_rff is None:
        num_rff = min(512, max(64, n_train // 3))

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ]
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": DEFAULT_TOA_ADAM_LR}
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=DEFAULT_TOA_ADAM_STOP_PATIENCE),
        ]
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    title = f"S2_TOA_MT_nTrain{n_train}_nTest{n_test}_{rff_sampling}D{num_rff}_T{num_tasks}"
    if rff_sampling == "sorf":
        title = f"{title}_correctSorf{correct_sorf}"
    feature_dim = 2 * num_rff
    joint_width = feature_dim * num_tasks
    sampling_label = rff_sampling.upper()

    print("=" * 60)
    print(title)
    print(
        f"Joint {sampling_label}-MTGP (Woodbury), D={num_rff}, m={feature_dim}, m*T={joint_width}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}, tasks={names}, "
        f"n_bands={len(shared_bands)}, log_scale={log_scale}"
        + (f", correct_sorf={correct_sorf}" if rff_sampling == "sorf" else "")
    )
    print(f"Optimizer: {getattr(optimizer_class, '__name__', optimizer_class)}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
    print("=" * 60)

    n_val = TOA_VAL_POOL_SIZE if monitor_validation else 0
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
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

    x_test_orig = x_test.clone()
    wl_np = wavelengths.detach().cpu().numpy()
    x_train = select_bands(x_train, shared_bands).to(dtype=dtype)
    x_test = select_bands(x_test, shared_bands).to(dtype=dtype)
    if x_val.numel() > 0:
        x_val = select_bands(x_val, shared_bands).to(dtype=dtype)
    else:
        x_val = x_val.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)
    input_dim = int(x_train.shape[-1])

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
        if x_val.numel() > 0:
            x_val = x_scaler.transform(x_val)
        print(f"X scaling: {x_scaling_type}")

    y_train_model = forward_y_s2_matrix(y_train, names, log_scale=log_scale)
    y_test_model = forward_y_s2_matrix(y_test, names, log_scale=log_scale)

    y_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_model)
        y_train_fit = y_scaler.transform(y_train_model)
    else:
        y_train_fit = y_train_model

    x_val_scaled = x_val
    y_val_scaled = y_val
    if y_val.numel() > 0:
        y_val_model = forward_y_s2_matrix(y_val, names, log_scale=log_scale)
        y_val_scaled = y_scaler.transform(y_val_model) if standardize_y and y_scaler is not None else y_val_model

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
                woodbury_mt_method=DEFAULT_MT_WOODBURY_METHOD,
            )
        )

    noise_prior = None
    initializer_kwargs: dict | None = None
    parameter_configs: dict = {}
    noise_prior_meta: dict | None = None
    if response_noise_prior:
        from gpplus.priors.response_noise import (
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
        temp_lik = build_rff_multitask_noise_likelihood(num_tasks, rank=0)
        raw_init = task_noise_raw_init_from_variances(temp_lik, target_vars.to(dtype=dtype))
        parameter_configs["raw_task_noises"] = {"method": "constant", "value": raw_init}
        noise_prior_meta = {
            "response_noise_prior": True,
            "noise_var_fraction": float(noise_var_fraction),
            "noise_prior_log_scale": float(noise_prior_log_scale),
            "noise_prior_loc": noise_prior.loc.detach().cpu().tolist(),
            "noise_prior_target_var": target_vars.detach().cpu().tolist(),
        }
        print(
            f"Response noise prior: LogNormal per task, fraction={noise_var_fraction}, "
            f"log_scale={noise_prior_log_scale}"
        )

    if parameter_configs:
        initializer_kwargs = {"parameter_configs": parameter_configs}
    initializer_kwargs = merge_mt_noise_initializer_kwargs(initializer_kwargs)

    likelihood = build_rff_multitask_noise_likelihood(num_tasks, noise_prior=noise_prior)
    model = RFFMTGPR(
        x_train,
        y_train_fit,
        num_tasks=num_tasks,
        num_rff=num_rff,
        ard=ard,
        rff_sampling=rff_sampling,
        correct_sorf=correct_sorf,
        rank_kernel=rank_kernel,
        rank_likelihood=0,
        likelihood=likelihood,
    )
    trainer = GPTrainer(
        model,
        mll_class=mt_mll_class(),
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
        stop_conditions=stop_conditions,
        parallel_verbose=parallel_verbose,
    )
    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All training runs failed for joint S2 MT model. "
            + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
        )
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])
    learned_noise = extract_mt_learned_noise(model)
    final_lr = float(best_run["final_lr"]) if best_run.get("final_lr") is not None else None

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_mt_gp_model(
        model, x_test, chunk_size=predict_chunk_size, **mt_eval_kwargs(dtype)
    )
    prediction_time = time.time() - t_pred

    inv = inverse_y_s2_matrix(
        pred_mean.detach().cpu(),
        pred_std.detach().cpu(),
        lower.detach().cpu(),
        upper.detach().cpu(),
        task_names=names,
        y_scaler=y_scaler,
        standardize_y=standardize_y,
        log_scale=log_scale,
    )
    y_true_np = y_test.cpu().numpy()
    y_pred_np = inv.point.numpy()
    y_pred_mean_np = inv.point_mean.numpy() if inv.point_mean is not None else y_pred_np
    y_pred_mode_np = inv.point_mode.numpy() if inv.point_mode is not None else None
    pred_std_np = inv.std.numpy()
    lower_np = inv.lower.numpy()
    upper_np = inv.upper.numpy()

    per_task = compute_per_task_metrics(y_true_np, y_pred_np, task_names=names)

    rel_metrics_by_task: dict[str, dict] = {}
    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_true_np[:, t],
            y_pred_np[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        if (
            task_uses_log_scale(name, log_scale=log_scale)
            and inv.log_mu is not None
        ):
            extra = compute_log_scale_extra_metrics(
                y_true_np[:, t],
                log_mu=inv.log_mu[:, t].numpy(),
                point_mean_physical=(
                    y_pred_mean_np[:, t] if y_pred_mean_np is not None else None
                ),
            )
            for key, value in extra.items():
                per_task[f"{name}_{key}"] = float(value)

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    aggregate_rrmse = float(macro_rrmse(per_task, names))
    log_task_names = [n for n in names if task_uses_log_scale(n, log_scale=log_scale)]
    aggregate_rrmse_log = macro_metric(per_task, log_task_names, "RRMSE_log")
    aggregate_rrmse_lnorm_mean = macro_metric(per_task, log_task_names, "RRMSE_mean")

    prob_metrics: dict[str, float] = {}
    for t, name in enumerate(names):
        kwargs: dict = {
            "output_std": torch.as_tensor(pred_std_np[:, t]),
            "lower_95": torch.as_tensor(lower_np[:, t]),
            "upper_95": torch.as_tensor(upper_np[:, t]),
            "training_time": train_time / num_tasks,
            "prediction_time": prediction_time / num_tasks,
        }
        if (
            task_uses_log_scale(name, log_scale=log_scale)
            and inv.log_mu is not None
            and inv.log_sigma is not None
        ):
            kwargs["log_mu"] = inv.log_mu[:, t]
            kwargs["log_sigma"] = inv.log_sigma[:, t]
        computed = compute_metrics(
            torch.as_tensor(y_true_np[:, t]),
            torch.as_tensor(y_pred_np[:, t]),
            **kwargs,
        )
        for key, value in computed.items():
            prob_metrics[f"{name}_{key}"] = value

    print(f"\nTest macro RRMSE (physical median): {aggregate_rrmse:.6f}  RMSE: {aggregate_rmse:.6f}")
    if log_task_names:
        print(
            f"Test macro RRMSE_log (ln-space, log QoIs): {aggregate_rrmse_log:.6f}  "
            f"macro RRMSE_mean (physical lognormal mean): {aggregate_rrmse_lnorm_mean:.6f}"
        )
    for name in names:
        print(
            f"{name} RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"RMSE: {per_task[f'{name}_RMSE']:.6f}"
        )
        if name in log_task_names:
            print(
                f"  ln-space RRMSE_log: {per_task[f'{name}_RRMSE_log']:.6f}  "
                f"RMSE_log: {per_task[f'{name}_RMSE_log']:.6f}  |  "
                f"physical-mean RRMSE_mean: {per_task[f'{name}_RRMSE_mean']:.6f}"
            )
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
    print(f"Total training time: {train_time:.1f}s")

    log_tasks = log_task_names
    metrics: dict = {
        "title": title,
        "dataset": "s2",
        "input_dim": input_dim,
        "input_dim_original": S2_INPUT_DIM,
        "shared_band_indices": list(shared_bands),
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": num_tasks,
        "task_names": list(names),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "correct_sorf": correct_sorf,
        "feature_dim": feature_dim,
        "joint_feature_dim": joint_width,
        "rank_kernel": rank_kernel,
        "ard": ard,
        "model_class": "RFFMTGPR",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "initial_lr": float(optimizer_kwargs.get("lr")) if "lr" in optimizer_kwargs else None,
        "final_lr": final_lr,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_scale": log_scale,
        "log_scale_tasks": log_tasks,
        "response_noise_prior": bool(response_noise_prior),
        "best_train_loss": best_loss,
        "rel_tolerance": rel_tolerance,
        "input_variable": input_variable,
        "task_band_config": str(task_band_config or S2_DEFAULT_BAND_CONFIG_PATH),
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wl_np),
        "data_meta": data_meta,
        **learned_noise,
        "Training_Time": train_time,
        "Prediction_Time": prediction_time,
        "Total_Time": train_time + prediction_time,
        "RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_RRMSE_log_tasks": aggregate_rrmse_log,
        "aggregate_RRMSE_lognormal_mean_tasks": aggregate_rrmse_lnorm_mean,
        **per_task,
        **prob_metrics,
    }
    if noise_prior_meta is not None:
        metrics.update(noise_prior_meta)

    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["n_val"] = n_val
        metrics["train_pool_size"] = TOA_TRAIN_POOL_SIZE
        metrics["val_pool_size"] = TOA_VAL_POOL_SIZE
        metrics["test_pool_size"] = TOA_TEST_POOL_SIZE
        metrics.update(summarize_validation_from_runs(runs, best_run))

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
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
                log_grain=bool(log_tasks),
                logit_cos=False,
                input_column_indices=torch.as_tensor(shared_bands, dtype=torch.int64),
                model_config={
                    "num_tasks": num_tasks,
                    "num_rff": num_rff,
                    "ard": ard,
                    "rff_sampling": rff_sampling,
                    "correct_sorf": correct_sorf,
                    "rank_kernel": rank_kernel,
                    "rank_likelihood": 0,
                },
            )
            metrics["checkpoint_path"] = str(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        example_indices = select_posterior_example_indices(
            y_true_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        out_npz = save_s2_predictions_npz(
            save_path,
            title=title,
            task_names=names,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_std=pred_std_np,
            lower=lower_np,
            upper=upper_np,
            x_test=x_test_orig.numpy(),
            wavelengths_nm=wl_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
            y_pred_mean=y_pred_mean_np if log_tasks else None,
            y_pred_mode=y_pred_mode_np if log_tasks else None,
        )
        print(f"Saved predictions to {out_npz}")
        metrics["predictions_npz"] = str(out_npz)

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_validation and monitor_validation and n_val > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")

        if plot_posterior:
            scatter_dir = Path(save_path) / "plots" / "scatter" / title
            for t, name in enumerate(names):
                plot_s2_task_scatter(
                    y_true=y_true_np[:, t],
                    y_pred=y_pred_np[:, t],
                    lower=lower_np[:, t],
                    upper=upper_np[:, t],
                    task_name=name,
                    out_path=scatter_dir / f"{name}_scatter.png",
                    title=f"{title} | {name}",
                )
            if example_indices:
                post_dir = Path(save_path) / "plots" / "posterior" / title
                post_paths = plot_s2_posterior_examples(
                    x_test=x_test_orig.numpy(),
                    wavelengths_nm=wl_np,
                    y_true=y_true_np,
                    y_pred=y_pred_np,
                    lower=lower_np,
                    upper=upper_np,
                    task_names=names,
                    example_indices=example_indices,
                    save_dir=post_dir,
                    title=title,
                    y_std=pred_std_np,
                    rel_metrics_by_task=rel_metrics_by_task,
                    rel_tolerance=rel_tolerance,
                    log_scale_tasks=log_tasks,
                    y_pred_mean=y_pred_mean_np if log_tasks else None,
                    y_pred_mode=y_pred_mode_np if log_tasks else None,
                    log_mu=inv.log_mu.numpy() if log_tasks and inv.log_mu is not None else None,
                    log_sigma=inv.log_sigma.numpy() if log_tasks and inv.log_sigma is not None else None,
                    spectrum_ylabel=(
                        "Radiance"
                        if str(data_meta.get("input_variable", "")).endswith("radiance")
                        else "Reflectance"
                    ),
                )
                for p in post_paths:
                    print(f"Saved posterior plot to {p}")

    return metrics
