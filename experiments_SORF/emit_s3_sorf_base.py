"""S3 EMIT independent SORF-GP runner (Woodbury RFFGPR, no SVGP/VIRFF).

Uses the shared EMIT load/split from ``emit_s3_mtgpr_base`` and trains one
``RFFGPR`` per QoI with ``rff_sampling='sorf'`` and classic NIGP.
Optional per-QoI band configs and off-floor train masks address ISOFIT
floor/zero structure that synthetic Sobol labels do not have.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIR = "experiments_SORF/results/s3_emit_sorf"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

from experiments_toa.s2_bands import band_config_metadata, load_task_band_config
from experiments_toa.s2_constants import S3_LABEL_FLOOR_THRESHOLDS
from experiments_toa.s2_plotting import (
    plot_prediction_coverage,
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_reporting import plot_per_task_validation_curves
from experiments_toa.s2_utils import (
    compute_floor_bound_slice_metrics,
    compute_per_task_metrics,
    format_floor_bound_summary,
    macro_metric,
    macro_rrmse,
    select_bands,
)
from experiments_toa.s2_y_transform import (
    forward_y_s2,
    inverse_y_s2,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from gpplus.models import RFFGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    NIGPInputNoiseFreezeCallback,
    RFFParameterInitializer,
    evaluate_rff_gp_model,
    nigp_woodbury_mll_class,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from experiments_RFF.rff_gp_defaults import (
    DEFAULT_WOODBURY_FORM,
    build_rff_scalar_noise_likelihood,
    merge_rff_noise_initializer_kwargs,
    rff_eval_kwargs,
    rff_mll_class,
    woodbury_jitter_for_dtype,
)
from emit_s3_mtgpr_base import (
    EMIT_INPUT_DIM,
    S3_TASK_NAMES,
    TASK_VALID_Y_RANGE,
    _DEFAULT_EMIT_PATH,
    _DEFAULT_WL_SRC,
    _load_wavelengths_nm,
    _subsample_scatter,
    load_emit_s3_xy,
    parse_s3_task_names,
    split_emit_s3,
)
from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    DEFAULT_TOA_ADAM_LR,
    DEFAULT_TOA_ADAM_STOP_PATIENCE,
    compute_relative_error_metrics,
    compute_prediction_coverage_metrics,
    format_coverage_summary,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_train_loss_callback,
    make_validation_callback,
    save_metrics_json,
    summarize_validation_from_runs,
)
from rff_experiment_utils import extract_learned_likelihood_noise
from toa_stgp_checkpoint import checkpoint_path_for_run, save_toa_stgp_checkpoint

RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")


def _task_input_columns(
    band_indices: Sequence[int],
    *,
    has_elevation: bool,
    n_spectral: int = EMIT_INPUT_DIM,
) -> list[int]:
    """Spectral band indices, optionally appending the elevation column."""
    cols = [int(i) for i in band_indices]
    if has_elevation:
        cols.append(int(n_spectral))
    return cols


def _off_floor_mask(
    y: torch.Tensor,
    task_name: str,
    *,
    thresholds: Mapping[str, float],
) -> torch.Tensor:
    thr = float(thresholds[task_name])
    return y > thr


def run_s3_emit_sorf(
    n_train: int = 10000,
    n_val: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "sorf",
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 2000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 2048,
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
    wl_src: str | None = None,
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 50,
    save_checkpoint: bool = True,
    log_scale: bool | None = True,
    log_scale_qoi: Sequence[str] | None = None,
    logit_scale_qoi: Sequence[str] | None = None,
    logit_bounds: dict[str, tuple[float, float]] | None = None,
    log_offsets: Mapping[str, float] | None = None,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.5,
    correct_sorf: bool = True,
    task_names: Sequence[str] | None = None,
    nigp: bool = True,
    freeze_epoch_nigp: int = 100,
    nondefault_only: bool = True,
    filter_valid_labels: bool = True,
    include_elevation: bool = False,
    task_band_config: str | None = None,
    off_floor_train_tasks: Sequence[str] | None = None,
    floor_thresholds: Mapping[str, float] | None = None,
) -> dict:
    """Train independent SORF RFFGPR + NIGP models on EMIT snow70 (Woodbury)."""
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(
            f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}"
        )
    correct_sorf = bool(correct_sorf) if rff_sampling == "sorf" else False
    names = parse_s3_task_names(task_names)
    num_tasks = len(names)
    emit_path = Path(data_path) if data_path is not None else _DEFAULT_EMIT_PATH
    wl_path = Path(wl_src) if wl_src is not None else _DEFAULT_WL_SRC

    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR

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

    n_val_eff = int(n_val) if monitor_validation else 0
    X_np, Y_np, source_idx, data_meta = load_emit_s3_xy(
        emit_path,
        task_names=names,
        nondefault_only=nondefault_only,
        filter_valid_labels=filter_valid_labels,
        include_elevation=include_elevation,
    )
    n_filtered = int(X_np.shape[0])
    input_dim = int(X_np.shape[1])
    has_elevation = bool(data_meta.get("has_elevation", False))
    elevation_index = data_meta.get("elevation_index")
    train_local, val_local, test_local = split_emit_s3(
        n_filtered,
        n_train=n_train,
        n_val=n_val_eff,
        seed=seed,
    )
    n_test = int(test_local.size)
    train_idx = torch.as_tensor(source_idx[train_local], dtype=torch.int64)
    val_idx = torch.as_tensor(source_idx[val_local], dtype=torch.int64)
    test_idx = torch.as_tensor(source_idx[test_local], dtype=torch.int64)

    wl_np = _load_wavelengths_nm(emit_path, wl_path)
    floor_thr = dict(S3_LABEL_FLOOR_THRESHOLDS)
    if floor_thresholds:
        floor_thr.update({str(k): float(v) for k, v in floor_thresholds.items()})
    off_floor_train = frozenset(str(n) for n in (off_floor_train_tasks or ()))
    unknown_off = sorted(off_floor_train - set(names))
    if unknown_off:
        raise ValueError(f"off_floor_train_tasks not in run QoIs: {unknown_off}")
    missing_thr = sorted(n for n in off_floor_train if n not in floor_thr)
    if missing_thr:
        raise ValueError(
            f"off_floor_train_tasks need thresholds in S3_LABEL_FLOOR_THRESHOLDS "
            f"or floor_thresholds=...; missing {missing_thr}"
        )

    if task_band_config is not None:
        from experiments_toa.s2_constants import S2_TASK_NAMES

        skipped = [n for n in names if n not in S2_TASK_NAMES]
        if skipped:
            raise ValueError(
                f"task_band_config cannot be used with non-S2 QoI names {skipped}. "
                "Drop them from QOI or leave task_band_config=None."
            )
        # Band JSON is spectral-only (285); elevation is appended separately.
        bands_by_task = load_task_band_config(
            task_band_config, task_names=names, input_dim=EMIT_INPUT_DIM
        )
        shared_bands = sorted(set().union(*(bands_by_task[n] for n in names)))
    else:
        shared_bands = list(range(EMIT_INPUT_DIM))
        bands_by_task = {name: list(shared_bands) for name in names}
    input_column_indices = _task_input_columns(
        shared_bands, has_elevation=has_elevation
    )

    title = (
        f"S3_EMIT_ST_nTrain{n_train}_nVal{n_val_eff}_nTest{n_test}_"
        f"{rff_sampling}D{num_rff}_T{num_tasks}"
    )
    if rff_sampling == "sorf":
        title = f"{title}_correctSorf{correct_sorf}"
    if has_elevation:
        title = f"{title}_elev"
    feature_dim = 2 * num_rff
    sampling_label = rff_sampling.upper()

    print("=" * 60)
    print(title)
    print(
        f"Independent {sampling_label}-GP (Woodbury), D={num_rff}, m={feature_dim}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}, tasks={names}, "
        f"n_bands={EMIT_INPUT_DIM}, input_dim={input_dim}, elev={has_elevation}, "
        f"nigp={bool(nigp)}"
        + (f", correct_sorf={correct_sorf}" if rff_sampling == "sorf" else "")
    )
    print(f"Optimizer: {getattr(optimizer_class, '__name__', optimizer_class)}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
    print(
        f"Split: filtered={n_filtered}  train={n_train}  val={n_val_eff}  test={n_test}  "
        f"(EMIT {emit_path})"
    )
    print("=" * 60)

    x_train_full = torch.as_tensor(X_np[train_local], dtype=dtype)
    y_train = torch.as_tensor(Y_np[train_local], dtype=dtype)
    x_test_full = torch.as_tensor(X_np[test_local], dtype=dtype)
    y_test = torch.as_tensor(Y_np[test_local], dtype=dtype)
    if n_val_eff > 0:
        x_val_full = torch.as_tensor(X_np[val_local], dtype=dtype)
        y_val = torch.as_tensor(Y_np[val_local], dtype=dtype)
    else:
        x_val_full = x_train_full.new_zeros((0, x_train_full.shape[-1]))
        y_val = y_train.new_zeros((0, y_train.shape[-1]))
    del X_np, Y_np

    warps = resolve_y_warps(
        log_scale_qoi,
        logit_scale_qoi,
        log_scale=log_scale,
        meta=data_meta,
        logit_bounds=logit_bounds,
        log_offsets=log_offsets,
    )
    log_scale = warps.log_scale
    log_scale_task_set = warps.log_tasks
    log_scale_source = warps.log_source
    logit_scale_task_set = warps.logit_tasks
    print(
        f"Output warps: log={sorted(log_scale_task_set) or '[]'} "
        f"(source={log_scale_source}); "
        f"logit={sorted(logit_scale_task_set) or '[]'} "
        f"(source={warps.logit_source})"
        + (
            f"; log_offsets={warps.active_log_offsets()}"
            if warps.active_log_offsets()
            else ""
        )
    )
    if task_band_config is not None:
        print(f"Task band config: {task_band_config}")
    if off_floor_train:
        print(
            "Off-floor train masks: "
            + ", ".join(f"{n}>{floor_thr[n]:g}" for n in sorted(off_floor_train))
        )

    x_test_orig = x_test_full.clone()
    if nigp:
        print(
            "NIGP: on (independent input noise, sigma_x=10^SoftClamp(raw), "
            f"freeze_epochs={int(freeze_epoch_nigp)})"
        )

    task_metrics: dict[str, dict] = {}
    task_runs: dict[str, list] = {}
    task_best_runs: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    rel_metrics_by_task: dict[str, dict] = {}
    cov_metrics_by_task: dict[str, dict] = {}
    off_floor_train_counts: dict[str, dict[str, int]] = {}
    total_train_time = 0.0
    total_prediction_time = 0.0
    x_scaling_type = "None"

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        task_cols = _task_input_columns(band_indices, has_elevation=has_elevation)
        x_tr = select_bands(x_train_full, task_cols)
        x_te = select_bands(x_test_full, task_cols)
        x_va = (
            select_bands(x_val_full, task_cols)
            if x_val_full.numel() > 0
            else x_val_full
        )

        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_va = y_val[:, task_idx] if y_val.numel() > 0 else y_val
        if task_name in off_floor_train:
            tr_mask = _off_floor_mask(y_tr, task_name, thresholds=floor_thr)
            n_keep = int(tr_mask.sum().item())
            n_drop = int((~tr_mask).sum().item())
            if n_keep < 8:
                raise RuntimeError(
                    f"off_floor_train for {task_name!r} left only {n_keep} train rows "
                    f"(threshold={floor_thr[task_name]:g})."
                )
            x_tr = x_tr[tr_mask]
            y_tr = y_tr[tr_mask]
            off_floor_train_counts[task_name] = {
                "n_train_kept": n_keep,
                "n_train_dropped": n_drop,
            }
            print(
                f"Off-floor train {task_name}: kept {n_keep}/{n_keep + n_drop} "
                f"(y>{floor_thr[task_name]:g})"
            )
            if y_va.numel() > 0:
                va_mask = _off_floor_mask(y_va, task_name, thresholds=floor_thr)
                x_va = x_va[va_mask]
                y_va = y_va[va_mask]
                print(f"Off-floor val {task_name}: kept {int(va_mask.sum().item())}")

        x_scaler = None
        x_scaling_type = "None"
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
                raise ValueError(
                    f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}"
                )
            x_scaler.fit(x_tr)
            x_tr = x_scaler.transform(x_tr)
            x_te = x_scaler.transform(x_te)
            if x_va.numel() > 0:
                x_va = x_scaler.transform(x_va)
            print(f"X scaling ({task_name}): {x_scaling_type}  n_features={x_tr.shape[-1]}")

        y_tr_model = forward_y_s2(y_tr, task_name, warps=warps)

        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))
            y_tr_fit = y_scaler.transform(y_tr_model.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit = y_tr_model

        y_val_scaled = y_va
        if y_va.numel() > 0:
            y_va_model = forward_y_s2(y_va, task_name, warps=warps)
            if standardize_y and y_scaler is not None:
                y_val_scaled = y_scaler.transform(y_va_model.unsqueeze(-1)).squeeze(-1)
            else:
                y_val_scaled = y_va_model

        callbacks = []
        if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1:
            callbacks.append(
                NIGPInputNoiseFreezeCallback(
                    freeze_epochs=int(freeze_epoch_nigp),
                    verbose=training_verbose,
                )
            )
        if num_epochs > 1 and training_verbose:
            callbacks.append(
                make_train_loss_callback(
                    num_inits,
                    num_epochs,
                    verbose=training_verbose,
                    log_every_n_epochs=log_every_n_epochs,
                )
            )
        if monitor_validation and n_val_eff > 0:
            callbacks.append(
                make_validation_callback(
                    x_va,
                    y_val_scaled,
                    num_inits,
                    chunk_size=predict_chunk_size,
                    verbose=validation_verbose,
                    log_every_n_epochs=log_every_n_epochs,
                    woodbury_form=DEFAULT_WOODBURY_FORM,
                )
            )

        initializer_kwargs: dict | None = None
        if response_noise_prior:
            from gpplus.priors.response_noise import (
                empirical_task_noise_variances,
                log_normal_noise_prior_from_responses,
                scalar_noise_raw_init_from_variance,
            )

            target_var = empirical_task_noise_variances(
                y_tr_fit.unsqueeze(-1), fraction=noise_var_fraction
            ).reshape(-1)[0]
            noise_prior = log_normal_noise_prior_from_responses(
                y_tr_fit.unsqueeze(-1),
                fraction=noise_var_fraction,
                log_scale=noise_prior_log_scale,
                dtype=dtype,
                device=y_tr_fit.device,
            )
            likelihood = build_rff_scalar_noise_likelihood(noise_prior=noise_prior)
            raw_init = scalar_noise_raw_init_from_variance(likelihood, target_var.to(dtype=dtype))
            initializer_kwargs = {
                "parameter_configs": {"raw_noise": {"method": "constant", "value": raw_init}}
            }
        else:
            likelihood = build_rff_scalar_noise_likelihood()

        initializer_kwargs = merge_rff_noise_initializer_kwargs(initializer_kwargs)
        model = RFFGPR(
            x_tr,
            y_tr_fit,
            likelihood=likelihood,
            num_rff=num_rff,
            ard=ard,
            rff_sampling=rff_sampling,
            correct_sorf=correct_sorf,
            nigp=bool(nigp),
        )
        if nigp:
            task_mll_class = nigp_woodbury_mll_class(DEFAULT_WOODBURY_FORM)
        else:
            task_mll_class = rff_mll_class()

        trainer = GPTrainer(
            model,
            mll_class=task_mll_class,
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
            stop_conditions=stop_conditions,
            parallel_verbose=parallel_verbose,
            min_epochs=(
                int(freeze_epoch_nigp)
                if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1
                else 0
            ),
        )
        t_train = time.time()
        runs = trainer.train()
        train_time = time.time() - t_train
        total_train_time += train_time
        model = trainer.model

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
                n_val=n_val_eff,
                data_path=str(emit_path),
                rel_tolerance=rel_tolerance,
                dtype=dtype,
                log_grain=task_uses_log_scale(task_name, warps=warps),
                logit_cos=task_uses_logit_scale(task_name, warps=warps),
                input_column_indices=torch.as_tensor(task_cols, dtype=torch.int64),
                model_config={
                    "num_rff": num_rff,
                    "ard": ard,
                    "rff_sampling": rff_sampling,
                    "correct_sorf": correct_sorf,
                    "nigp": bool(nigp),
                    "include_elevation": bool(has_elevation),
                    "task_band_config": task_band_config,
                    "off_floor_train": task_name in off_floor_train,
                },
            )
            learned_noise["checkpoint_path"] = str(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        model.eval()
        model.invalidate_feature_cache()
        t_pred = time.time()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model, x_te, chunk_size=predict_chunk_size, **rff_eval_kwargs(dtype)
        )
        prediction_time = time.time() - t_pred
        total_prediction_time += prediction_time

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
        rel_m = compute_relative_error_metrics(
            y_te.cpu().numpy(),
            pred_mean.numpy(),
            rel_tolerance=rel_tolerance,
        )
        cov_m = compute_prediction_coverage_metrics(
            y_te.cpu().numpy(),
            pred_mean.numpy(),
            pred_std.numpy(),
        )
        rel_metrics_by_task[task_name] = rel_m
        cov_metrics_by_task[task_name] = cov_m
        tm = {
            "best_train_loss": best_loss,
            **learned_noise,
            **computed,
            f"{task_name}_max_rel_error": float(rel_m["max_rel_error"]),
            f"{task_name}_mean_rel_error": float(rel_m["mean_rel_error"]),
            f"{task_name}_median_rel_error": float(rel_m["median_rel_error"]),
            f"{task_name}_pct_within_1pct": float(rel_m["pct_within_1pct"]),
            f"{task_name}_coverage_50": float(cov_m["coverage_50"]),
            f"{task_name}_coverage_90": float(cov_m["coverage_90"]),
            f"{task_name}_coverage_95": float(cov_m["coverage_95"]),
        }
        if best_run.get("final_lr") is not None:
            tm["final_lr"] = float(best_run["final_lr"])
        task_metrics[task_name] = tm
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run
        y_pred_all.append(pred_mean.numpy())
        y_std_all.append(pred_std.numpy())
        lower_all.append(lower.numpy())
        upper_all.append(upper.numpy())
        print(format_relative_error_summary(task_name, rel_m, rel_tolerance=rel_tolerance))
        print(format_coverage_summary(task_name, cov_m))

    y_true_np = y_test.cpu().numpy()
    y_pred_np = np.column_stack(y_pred_all)
    pred_std_np = np.column_stack(y_std_all)
    lower_np = np.column_stack(lower_all)
    upper_np = np.column_stack(upper_all)
    per_task = compute_per_task_metrics(y_true_np, y_pred_np, task_names=names)
    slice_metrics = compute_floor_bound_slice_metrics(
        y_true_np, y_pred_np, task_names=names, thresholds=floor_thr
    )
    per_task.update(slice_metrics)
    for name in names:
        per_task[f"{name}_max_rel_error"] = float(rel_metrics_by_task[name]["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_metrics_by_task[name]["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_metrics_by_task[name]["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_metrics_by_task[name]["pct_within_1pct"])
        per_task[f"{name}_coverage_50"] = float(cov_metrics_by_task[name]["coverage_50"])
        per_task[f"{name}_coverage_90"] = float(cov_metrics_by_task[name]["coverage_90"])
        per_task[f"{name}_coverage_95"] = float(cov_metrics_by_task[name]["coverage_95"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    aggregate_rrmse = float(macro_rrmse(per_task, names))
    aggregate_medae = float(macro_metric(per_task, names, "MedAE"))
    log_task_names = [n for n in names if task_uses_log_scale(n, warps=warps)]
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    off_floor_names = [n for n in names if f"{n}_off_floor_RRMSE" in per_task]
    aggregate_rrmse_off_floor = macro_metric(per_task, off_floor_names, "off_floor_RRMSE")

    print(
        f"\nTest macro RRMSE (physical median): {aggregate_rrmse:.6f}  "
        f"RMSE: {aggregate_rmse:.6f}  MedAE: {aggregate_medae:.6f}"
    )
    if off_floor_names:
        print(f"Test macro RRMSE (off-floor subsets): {aggregate_rrmse_off_floor:.6f}")
    for name in names:
        print(
            f"{name} RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"MAE: {per_task[f'{name}_MAE']:.6f}  "
            f"MedAE: {per_task[f'{name}_MedAE']:.6f}"
        )
        summary = format_floor_bound_summary(name, per_task)
        if summary:
            print(summary)
    print(f"Total training time: {total_train_time:.1f}s")

    metrics: dict = {
        "title": title,
        "dataset": "emit_snow70",
        "input_dim": input_dim,
        "input_dim_original": EMIT_INPUT_DIM,
        "n_spectral_bands": EMIT_INPUT_DIM,
        "has_elevation": has_elevation,
        "elevation_index": elevation_index,
        "aux_inputs": list(data_meta.get("aux_inputs", [])),
        "shared_band_indices": list(shared_bands),
        "input_column_indices": list(input_column_indices),
        "task_band_config": task_band_config,
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wl_np),
        "off_floor_train_tasks": sorted(off_floor_train),
        "off_floor_train_counts": off_floor_train_counts,
        "floor_thresholds": {k: floor_thr[k] for k in names if k in floor_thr},
        "aggregate_RRMSE_off_floor": aggregate_rrmse_off_floor,
        "n_train": n_train,
        "n_val": n_val_eff,
        "n_test": n_test,
        "n_filtered": n_filtered,
        "split": "permutation_train_val_remaining_test",
        "num_tasks": num_tasks,
        "task_names": list(names),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "correct_sorf": correct_sorf,
        "feature_dim": feature_dim,
        "nigp": bool(nigp),
        "freeze_epoch_nigp": int(freeze_epoch_nigp),
        "ard": ard,
        "model_class": "RFFGPR",
        "train_mode": "independent",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_scale": bool(log_scale),
        "log_scale_tasks": log_task_names,
        "log_scale_source": log_scale_source,
        "log_offsets": warps.active_log_offsets(),
        "logit_scale_tasks": list(logit_task_names),
        "logit_scale_source": warps.logit_source,
        "rel_tolerance": rel_tolerance,
        "data_meta": data_meta,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        **per_task,
        "task_metrics": task_metrics,
    }
    for task_name, tm in task_metrics.items():
        if tm.get("checkpoint_path"):
            metrics[f"{task_name}_checkpoint_path"] = tm["checkpoint_path"]
        metrics[f"{task_name}_best_train_loss"] = tm["best_train_loss"]
    if monitor_validation and n_val_eff > 0:
        metrics["monitor_validation"] = True
        for task_name in names:
            val_summary = summarize_validation_from_runs(
                task_runs[task_name], task_best_runs[task_name]
            )
            for key, value in val_summary.items():
                metrics[f"{task_name}_{key}"] = value

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
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
            log_scale_tasks=log_task_names,
            logit_scale_tasks=logit_task_names,
        )
        print(f"Saved predictions to {out_npz}")
        metrics["predictions_npz"] = str(out_npz)

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_validation and monitor_validation and n_val_eff > 0:
            for task_name in names:
                for plot_path in plot_per_task_validation_curves(
                    metrics, save_path, task_name, json_path=out_json
                ):
                    print(f"Saved validation plot to {plot_path}")

        if plot_posterior:
            scatter_dir = Path(save_path) / "plots" / "scatter" / title
            for t, name in enumerate(names):
                yt_s, yp_s, lo_s, hi_s = _subsample_scatter(
                    y_true_np[:, t],
                    y_pred_np[:, t],
                    lower_np[:, t],
                    upper_np[:, t],
                    seed=seed + t,
                )
                plot_s2_task_scatter(
                    y_true=yt_s,
                    y_pred=yp_s,
                    lower=lo_s,
                    upper=hi_s,
                    task_name=name,
                    out_path=scatter_dir / f"{name}_scatter.png",
                    title=f"{title} | {name}",
                    test_rrmse=float(per_task[f"{name}_RRMSE"]),
                )
            if example_indices:
                post_dir = Path(save_path) / "plots" / "posterior" / title
                post_paths = plot_s2_posterior_examples(
                    x_test=x_test_orig.numpy()[:, :EMIT_INPUT_DIM],
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
                    log_scale_tasks=log_task_names,
                    spectrum_ylabel="Reflectance",
                )
                for p in post_paths:
                    print(f"Saved posterior plot to {p}")

            cov_dir = Path(save_path) / "plots" / "coverage" / title
            for cov_path in plot_prediction_coverage(
                task_names=names,
                coverage_by_task=cov_metrics_by_task,
                save_dir=cov_dir,
                title=title,
            ):
                print(f"Saved coverage plot to {cov_path}")

    return metrics


__all__ = [
    "EMIT_INPUT_DIM",
    "S3_TASK_NAMES",
    "TASK_VALID_Y_RANGE",
    "parse_s3_task_names",
    "run_s3_emit_sorf",
]
