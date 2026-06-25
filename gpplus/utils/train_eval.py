import time
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from gpplus.config import get_settings
from gpplus.training import GPTrainer
from gpplus.training.callbacks import FinalParameterStorageCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.training.stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
)
from gpplus.utils.metrics_functions import (
    compute_lognormal_interval_bounds,
    compute_metrics,
    compute_nis,
    compute_per_source_metrics,
)


def _print_train_eval_gp_eval_parity(model: torch.nn.Module, tag: str = "") -> None:
    """One-shot diagnostics for series vs parallel prediction parity (main process)."""
    label = f" {tag}" if tag else ""
    jitter = getattr(model, "cholesky_jitter", None)
    try:
        det = torch.are_deterministic_algorithms_enabled()
    except Exception:  # noqa: BLE001
        det = None
    nth = torch.get_num_threads()
    try:
        n_interop = torch.get_num_interop_threads()
    except Exception:  # noqa: BLE001
        n_interop = None
    tp_summary = "n/a"
    try:
        from threadpoolctl import threadpool_info

        info = threadpool_info()
        tp_summary = ", ".join(
            f"{e.get('user_api', e.get('prefix', '?'))}={e.get('num_threads', '?')}"
            for e in info
        )
    except Exception:  # noqa: BLE001
        pass
    print(
        f"[train_eval_gp eval parity{label}] cholesky_jitter={jitter!r} "
        f"deterministic_algorithms={det} torch_num_threads={nth} "
        f"torch_num_interop_threads={n_interop!r} threadpool_info={tp_summary}"
    )


def _validate_log_y_point_inverse(log_y_point_inverse: str) -> None:
    if log_y_point_inverse not in ("median", "mean"):
        raise ValueError(
            f"log_y_point_inverse must be 'median' or 'mean', got {log_y_point_inverse!r}."
        )


def _require_finite_log_scale_c(
    standardize_y_log_scale: bool,
    log_scale_C: float | None,
    y_train_mean,
    y_train_std,
    *,
    caller: str,
) -> None:
    if not standardize_y_log_scale or y_train_mean is None or y_train_std is None:
        return
    if log_scale_C is None:
        raise ValueError(
            f"{caller}: standardize_y_log_scale=True with y scaling requires a finite "
            "log_scale_C (shift used in log(y + C))."
        )
    try:
        c = float(log_scale_C)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{caller}: log_scale_C must be a finite float, got {log_scale_C!r}."
        ) from e
    if not np.isfinite(c):
        raise ValueError(
            f"{caller}: log_scale_C must be finite, got {log_scale_C!r}."
        )


# Cap σ in raw log space when applying E[Y]=exp(μ+σ²/2)−C. Uncapped σ with the GP's
# standardized predictive std can yield astronomically large exp(σ²/2) and meaningless RMSE.
_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN = 3.0


def _torch_log_y_point_original(
    exp_log_y: torch.Tensor,
    log_y_std: torch.Tensor,
    log_scale_C: float,
    log_y_point_inverse: str,
) -> torch.Tensor:
    """Map (exp(mu_log), sigma_log) to original-scale point prediction (median or log-normal mean)."""
    if log_y_point_inverse == "median":
        return exp_log_y - log_scale_C
    log_y_std_eff = torch.clamp(log_y_std, max=_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN)
    if torch.any(log_y_std > _LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN):
        warnings.warn(
            "train_eval_gp: log_y_std (raw log space) exceeded "
            f"{_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN} for at least one test point; "
            "clamping for log-normal *mean* back-transform only (median path unchanged). "
            "Large GP predictive std in standardized space × y_train_std causes this; "
            "consider log_y_point_inverse='median' or a better-calibrated model.",
            stacklevel=3,
        )
    half_var = 0.5 * (log_y_std_eff**2)
    return exp_log_y * torch.exp(half_var) - log_scale_C


def _np_log_y_point_original(
    exp_log_y: np.ndarray,
    log_y_std: np.ndarray,
    log_scale_C: float,
    log_y_point_inverse: str,
) -> np.ndarray:
    if log_y_point_inverse == "median":
        return exp_log_y - log_scale_C
    cap = _LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN
    if np.any(log_y_std > cap):
        warnings.warn(
            "train_eval_PFN: log_y_std (raw log space) exceeded "
            f"{cap} for at least one test point; clamping for log-normal mean back-transform only.",
            stacklevel=3,
        )
    log_y_std_eff = np.minimum(log_y_std, cap)
    half_var = 0.5 * (log_y_std_eff**2)
    return exp_log_y * np.exp(half_var) - log_scale_C


def _process_trainer_info(stored_params, train_results, num_epochs):
    """
    Process stored parameters from callback into a structured trainer log.
    Delegates to the shared trainer_analysis module.
    """
    from gpplus.training.trainer_analysis import build_trainer_analysis_payload
    return build_trainer_analysis_payload(stored_params, train_results, num_epochs)


def _compute_log_space_metrics(
    y_true_original,
    y_pred_original,
    log_scale_C: float,
    output_std_original=None,
    lower_original=None,
    upper_original=None,
) -> dict:
    """
    Compute error metrics on log(y + C) space from original-scale y_true/y_pred.
    Only finite pairs with y + C > 0 are used.
    """
    y_true = (
        y_true_original.detach().cpu().numpy().reshape(-1)
        if isinstance(y_true_original, torch.Tensor)
        else np.asarray(y_true_original).reshape(-1)
    )
    y_pred = (
        y_pred_original.detach().cpu().numpy().reshape(-1)
        if isinstance(y_pred_original, torch.Tensor)
        else np.asarray(y_pred_original).reshape(-1)
    )
    c_val = float(log_scale_C)
    y_true_shift = y_true + c_val
    y_pred_shift = y_pred + c_val
    valid = (
        np.isfinite(y_true_shift)
        & np.isfinite(y_pred_shift)
        & (y_true_shift > 0)
        & (y_pred_shift > 0)
    )
    if output_std_original is not None:
        out_std = (
            output_std_original.detach().cpu().numpy().reshape(-1)
            if isinstance(output_std_original, torch.Tensor)
            else np.asarray(output_std_original).reshape(-1)
        )
    else:
        out_std = None
    if lower_original is not None:
        lower_arr = (
            lower_original.detach().cpu().numpy().reshape(-1)
            if isinstance(lower_original, torch.Tensor)
            else np.asarray(lower_original).reshape(-1)
        )
    else:
        lower_arr = None
    if upper_original is not None:
        upper_arr = (
            upper_original.detach().cpu().numpy().reshape(-1)
            if isinstance(upper_original, torch.Tensor)
            else np.asarray(upper_original).reshape(-1)
        )
    else:
        upper_arr = None

    if not np.any(valid):
        return {
            "RMSE_log": None,
            "MAE_log": None,
            "RRMSE_log": None,
            "NIS_log": None,
            "NIS_width_log": None,
            "NIS_outside_log": None,
            "log_eval_valid_fraction": 0.0,
        }
    y_true_log = np.log(y_true_shift[valid])
    y_pred_log = np.log(y_pred_shift[valid])
    rmse_log = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
    mae_log = float(np.mean(np.abs(y_true_log - y_pred_log)))
    std_log = float(np.std(y_true_log))
    rrmse_log = float(rmse_log / std_log) if std_log > 0 else float("inf")
    log_metrics = {
        "RMSE_log": rmse_log,
        "MAE_log": mae_log,
        "RRMSE_log": rrmse_log,
        "log_eval_valid_fraction": float(np.mean(valid)),
    }
    nis_log = None
    if (lower_arr is not None) and (upper_arr is not None):
        lower_shift = lower_arr + c_val
        upper_shift = upper_arr + c_val
        valid_bounds = (
            valid
            & np.isfinite(lower_shift)
            & np.isfinite(upper_shift)
            & (lower_shift > 0)
            & (upper_shift > 0)
        )
        if np.any(valid_bounds):
            nis_log = compute_nis(
                np.log(y_true_shift[valid_bounds]),
                lower=np.log(lower_shift[valid_bounds]),
                upper=np.log(upper_shift[valid_bounds]),
                alpha=0.05,
            )
    elif out_std is not None:
        # Delta-method approximation: sigma_log ≈ sigma_y / (y_hat + C)
        denom = y_pred_shift
        valid_std = valid & np.isfinite(out_std) & (out_std >= 0) & (denom > 0)
        if np.any(valid_std):
            log_sigma = out_std[valid_std] / denom[valid_std]
            nis_log = compute_nis(
                np.log(y_true_shift[valid_std]),
                y_hat=np.log(y_pred_shift[valid_std]),
                output_std=log_sigma,
                alpha=0.05,
            )
    if nis_log is not None:
        log_metrics["NIS_log"] = float(nis_log["NIS"])
        log_metrics["NIS_width_log"] = float(nis_log["NIS_width"])
        log_metrics["NIS_outside_log"] = float(nis_log["NIS_outside"])
    else:
        log_metrics["NIS_log"] = None
        log_metrics["NIS_width_log"] = None
        log_metrics["NIS_outside_log"] = None
    return log_metrics


def train_eval_gp(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    seed: int,
    num_inits: int,
    lr: float,
    convergence_patience: int = 10,
    min_epochs: int = 0,
    min_loss_change: float = 1e-7,
    optimizer_class=None,
    optimizer_kwargs: dict | None = None,
    initializer_class=None,
    initializer_kwargs: dict | None = None,
    device: str = "cpu",
    # dtype: torch.dtype = torch.float64,
    y_train_mean: torch.Tensor | dict | None = None,
    y_train_std: torch.Tensor | dict | None = None,
    standardize_y_log_scale: bool = False,
    log_scale_C: float | None = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    log_y_point_inverse: str = "median",  # "median": exp(mu)-C; "mean": exp(mu+sigma^2/2)-C in raw log space
    source_cols: int | list[int] | None = None,
    trainer_info: bool = False,
    cholesky_jitter: float = 1e-6,  # Jitter for Cholesky; use larger (e.g. 1e-5, 1e-4) for large n
    fold_index: int | None = None,  # Fold index for multi-fold experiments (sets fold_index on callbacks)
    callbacks: list | None = None,  # Optional list of callbacks (if None, creates default callbacks)
    callback_save_path: str | None = None,  # Base path for saving callback data (if None, uses default paths)
    log_lbfgs_inner: bool = True,  # Enabled by default: log/store per-iteration loss inside LBFGSScipy step(); metrics via callbacks
    lbfgs_inner_extra_metrics: list | None = None,  # Optional [(name, fn(context)->float), ...] for LBFGSInnerMetricsCallbackV3
    n_jobs: int | None = None,  # Joblib worker count for multi-init training. None=trainer default; 1=force series.
    inner_max_num_threads: int | None = 1,  # BLAS threads cap per worker AND in main during eval when set.
):
    """
    Train a GP model and evaluate metrics on the provided test set.

    Uses :class:`~gpplus.training.GPTrainer` (``training_single_run``) and
    :func:`~gpplus.training.eval.evaluate_gp_model` for training and prediction.
    """
    # Set optimizer kwargs based on optimizer type (Adam convenience)
    if optimizer_class is torch.optim.Adam or (
        isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.Adam)
    ):
        optimizer_kwargs = {"lr": lr if lr is not None else 0.01}

    # Use provided callbacks; if none are supplied, disable callbacks by default.
    if callbacks is None:
        callbacks = [FinalParameterStorageCallback(save_file=None, verbose=False)]


        # callbacks = 
            # # Jitter tracking across runs/epochs
            # callbacks.append(
            #     JitterTrackingCallback(
            #         save_file=jitter_save_file,
            #         verbose=True,
            #     )
            # )


    # Set fold_index on callbacks that support it
    if fold_index is not None:
        for cb in callbacks:
            if hasattr(cb, "set_fold_index"):
                cb.set_fold_index(fold_index)

    trainer = GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_inits=num_inits,
        stop_conditions=[
            ConvergencePatienceStopCondition(patience=convergence_patience),
            MinLossChangeStopCondition(min_loss_change=min_loss_change),
        ],
        min_epochs=min_epochs,
        callbacks=callbacks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        initializer_class=initializer_class,
        initializer_kwargs=initializer_kwargs,
        cholesky_jitter=cholesky_jitter,
        n_jobs=n_jobs,
        inner_max_num_threads=inner_max_num_threads,
    )

    t_train_start = time.time()
    train_results = trainer.train()
    training_time = float(getattr(trainer, "full_train_time", None) or (time.time() - t_train_start))

    # Measure prediction time
    t_pred_start = time.time()
    # Always evaluate on the same device as the model's training inputs / parameters
    if hasattr(model, "train_inputs") and model.train_inputs:
        eval_device = model.train_inputs[0].device
    else:
        eval_device = next(model.parameters()).device
    if X_test.device != eval_device:
        X_test = X_test.to(eval_device)

    # After parallel (loky) training, re-apply main-process settings before prediction
    # so ill-conditioned GP solves match series / worker numerics.
    try:
        torch.manual_seed(int(seed) if seed is not None else 0)
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        get_settings().apply()
    except Exception:  # noqa: BLE001
        pass

    eval_blas_ctx = nullcontext()
    if inner_max_num_threads is not None and str(device).lower().startswith("cpu"):
        try:
            from threadpoolctl import threadpool_limits

            eval_blas_ctx = threadpool_limits(limits=inner_max_num_threads)
        except ImportError:
            pass

    _prev_torch_threads = torch.get_num_threads()
    _prev_torch_interop = None
    try:
        _prev_torch_interop = torch.get_num_interop_threads()
    except Exception:  # noqa: BLE001
        _prev_torch_interop = None

    with eval_blas_ctx:
        if inner_max_num_threads is not None and str(device).lower().startswith("cpu"):
            try:
                nt = max(1, int(inner_max_num_threads))
                torch.set_num_threads(nt)
                try:
                    torch.set_num_interop_threads(1)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass

        _print_train_eval_gp_eval_parity(model, tag="before_predict")

        try:
            y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
        finally:
            try:
                torch.set_num_threads(_prev_torch_threads)
            except Exception:  # noqa: BLE001
                pass
            if _prev_torch_interop is not None:
                try:
                    torch.set_num_interop_threads(_prev_torch_interop)
                except Exception:  # noqa: BLE001
                    pass

    prediction_time = time.time() - t_pred_start

    # Check for NaN/Inf in model outputs before denormalization
    if torch.any(~torch.isfinite(y_pred)):
        nan_count = torch.sum(~torch.isfinite(y_pred)).item()
        import warnings

        warnings.warn(
            f"train_eval_gp: Model predictions contain {nan_count} NaN/Inf values before denormalization. "
            "This may indicate model training issues. Replacing with mean prediction (0 in standardized space)."
        )
        y_pred = torch.where(torch.isfinite(y_pred), y_pred, torch.zeros_like(y_pred))

    if torch.any(~torch.isfinite(output_std)):
        nan_count = torch.sum(~torch.isfinite(output_std)).item()
        import warnings

        warnings.warn(
            f"train_eval_gp: Model output_std contains {nan_count} NaN/Inf values before denormalization. "
            "This may indicate model training issues."
        )
        output_std = torch.where(torch.isfinite(output_std), output_std, torch.ones_like(output_std) * 1e-6)

    _validate_log_y_point_inverse(log_y_point_inverse)
    _require_finite_log_scale_c(
        standardize_y_log_scale,
        log_scale_C,
        y_train_mean,
        y_train_std,
        caller="train_eval_gp",
    )

    # Denormalization logic (unchanged from v1/v2)
    log_mu_for_metrics = None
    log_sigma_for_metrics = None
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            if source_cols is not None:
                is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
                if is_onehot:
                    onehot_cols = X_test[:, source_cols]
                    source_indices_test = torch.argmax(onehot_cols, dim=1)
                else:
                    source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
                    source_indices_test = X_test[:, source_col].long()

                y_pred_denorm = torch.zeros_like(y_pred)
                output_std_denorm = torch.zeros_like(output_std)
                log_mu_denorm = torch.zeros_like(y_pred) if standardize_y_log_scale else None
                log_sigma_denorm = torch.zeros_like(output_std) if standardize_y_log_scale else None
                unique_sources = torch.unique(source_indices_test)

                for source_idx in unique_sources:
                    source_mask = source_indices_test == source_idx
                    source_key = source_idx.item()

                    if source_key in y_train_mean:
                        mean = y_train_mean[source_key]
                        std = y_train_std[source_key]
                    elif 0 in y_train_mean:
                        mean = y_train_mean[0]
                        std = y_train_std[0]
                    else:
                        first_key = list(y_train_mean.keys())[0]
                        mean = y_train_mean[first_key]
                        std = y_train_std[first_key]

                    if isinstance(mean, torch.Tensor):
                        mean = mean.squeeze()
                    if isinstance(std, torch.Tensor):
                        std = std.squeeze()

                    if standardize_y_log_scale:
                        log_y_pred = (y_pred[source_mask] * std) + mean
                        log_y_std = output_std[source_mask] * std
                        max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                        log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                        exp_log_y = torch.exp(log_y_pred)
                        y_pred_denorm[source_mask] = _torch_log_y_point_original(
                            exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                        )
                        output_std_denorm[source_mask] = exp_log_y * log_y_std
                        log_mu_denorm[source_mask] = log_y_pred
                        log_sigma_denorm[source_mask] = log_y_std
                    else:
                        y_pred_denorm[source_mask] = (y_pred[source_mask] * std) + mean
                        output_std_denorm[source_mask] = output_std[source_mask] * std

                y_pred = y_pred_denorm
                output_std = output_std_denorm
                if standardize_y_log_scale:
                    log_mu_for_metrics = log_mu_denorm
                    log_sigma_for_metrics = log_sigma_denorm
            else:
                first_key = list(y_train_mean.keys())[0]
                mean = y_train_mean[first_key]
                std = y_train_std[first_key]

                if isinstance(mean, torch.Tensor):
                    mean = mean.squeeze()
                if isinstance(std, torch.Tensor):
                    std = std.squeeze()
                if standardize_y_log_scale:
                    log_y_pred = (y_pred * std) + mean
                    log_y_std = output_std * std
                    max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                    log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                    exp_log_y = torch.exp(log_y_pred)
                    y_pred = _torch_log_y_point_original(
                        exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                    )
                    output_std = exp_log_y * log_y_std
                    log_mu_for_metrics = log_y_pred
                    log_sigma_for_metrics = log_y_std
                else:
                    y_pred = (y_pred * std) + mean
                    output_std = output_std * std
        else:
            mean_val = y_train_mean.squeeze() if isinstance(y_train_mean, torch.Tensor) else y_train_mean
            std_val = y_train_std.squeeze() if isinstance(y_train_std, torch.Tensor) else y_train_std

            if standardize_y_log_scale:
                log_y_pred = (y_pred * std_val) + mean_val
                log_y_std = output_std * std_val
                max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
                log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
                exp_log_y = torch.exp(log_y_pred)
                y_pred = _torch_log_y_point_original(
                    exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                )
                output_std = exp_log_y * log_y_std
                log_mu_for_metrics = log_y_pred
                log_sigma_for_metrics = log_y_std
            else:
                y_pred = (y_pred * std_val) + mean_val
                output_std = output_std * std_val

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)
    log_mu_np = (
        log_mu_for_metrics.detach().cpu().numpy().reshape(-1)
        if isinstance(log_mu_for_metrics, torch.Tensor)
        else None
    )
    log_sigma_np = (
        log_sigma_for_metrics.detach().cpu().numpy().reshape(-1)
        if isinstance(log_sigma_for_metrics, torch.Tensor)
        else None
    )

    if standardize_y_log_scale and np.any(~np.isfinite(y_pred_np)):
        n_bad = int(np.sum(~np.isfinite(y_pred_np)))
        warnings.warn(
            f"train_eval_gp: {n_bad} non-finite values in y_pred after log-scale denorm "
            "(before NaN replacement).",
            stacklevel=2,
        )

    # Check for NaN/Inf values and replace with reasonable defaults
    if np.any(~np.isfinite(y_pred_np)):
        nan_mask = ~np.isfinite(y_pred_np)
        if standardize_y_log_scale:
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            if len(valid_preds) > 0:
                replacement = np.median(valid_preds)
            else:
                if y_train_mean is not None:
                    if isinstance(y_train_mean, dict):
                        log_mean_val = list(y_train_mean.values())[0]
                        log_std_val = list(y_train_std.values())[0]
                    else:
                        log_mean_val = y_train_mean
                        log_std_val = y_train_std
                    if isinstance(log_mean_val, torch.Tensor):
                        log_mean_val = (
                            log_mean_val.item()
                            if log_mean_val.numel() == 1
                            else log_mean_val.cpu().numpy()
                        )
                    if isinstance(log_std_val, torch.Tensor):
                        log_std_val = (
                            log_std_val.item()
                            if log_std_val.numel() == 1
                            else log_std_val.cpu().numpy()
                        )
                    mu_f = float(log_mean_val)
                    if log_y_point_inverse == "mean":
                        sig_f = float(log_std_val)
                        replacement = float(
                            np.exp(mu_f + 0.5 * sig_f * sig_f) - float(log_scale_C)
                        )
                    else:
                        replacement = float(np.exp(mu_f) - float(log_scale_C))
                else:
                    replacement = 0.0
            y_pred_np[nan_mask] = replacement
        else:
            valid_preds = y_pred_np[np.isfinite(y_pred_np)]
            replacement = np.median(valid_preds) if len(valid_preds) > 0 else 0.0
            y_pred_np[nan_mask] = replacement

    if np.any(~np.isfinite(output_std_np)):
        nan_mask = ~np.isfinite(output_std_np)
        valid_std = output_std_np[np.isfinite(output_std_np)]
        replacement = np.median(valid_std) if len(valid_std) > 0 else 1.0
        output_std_np[nan_mask] = replacement
    if log_mu_np is not None:
        if log_mu_np.shape != y_pred_np.shape:
            raise ValueError(
                f"train_eval_gp: log_mu shape mismatch, got {log_mu_np.shape} vs {y_pred_np.shape}."
            )
        bad_mu = ~np.isfinite(log_mu_np)
        if np.any(bad_mu):
            valid_mu = log_mu_np[np.isfinite(log_mu_np)]
            mu_repl = np.median(valid_mu) if len(valid_mu) > 0 else 0.0
            log_mu_np[bad_mu] = mu_repl
    if log_sigma_np is not None:
        if log_sigma_np.shape != y_pred_np.shape:
            raise ValueError(
                f"train_eval_gp: log_sigma shape mismatch, got {log_sigma_np.shape} vs {y_pred_np.shape}."
            )
        bad_sigma = ~np.isfinite(log_sigma_np)
        if np.any(bad_sigma):
            valid_sigma = log_sigma_np[np.isfinite(log_sigma_np)]
            sigma_repl = np.median(valid_sigma) if len(valid_sigma) > 0 else 0.0
            log_sigma_np[bad_sigma] = sigma_repl
        log_sigma_np = np.maximum(log_sigma_np, 0.0)

    y_true_flat = (
        y_test.detach().cpu().numpy().reshape(-1)
        if isinstance(y_test, torch.Tensor)
        else np.asarray(y_test).reshape(-1)
    )
    if y_true_flat.shape[0] != y_pred_np.shape[0]:
        raise ValueError(
            "train_eval_gp: y_test and predictions length mismatch — "
            f"len(y_test)={y_true_flat.shape[0]}, len(y_pred)={y_pred_np.shape[0]}. "
            "Expected one prediction per row of X_test (same order as y_test)."
        )

    gp_metric = compute_metrics(
        y_test,
        y_pred_np,
        output_std_np,
        training_time=training_time,
        prediction_time=prediction_time,
        log_mu=log_mu_np,
        log_sigma=log_sigma_np,
        log_scale_C=log_scale_C,
        use_log_quantile_nis=(
            standardize_y_log_scale
            and (log_mu_np is not None)
            and (log_sigma_np is not None)
            and (log_scale_C is not None)
        ),
    )
    if (
        standardize_y_log_scale
        and y_train_mean is not None
        and y_train_std is not None
        and log_scale_C is not None
    ):
        lower_original = None
        upper_original = None
        if (log_mu_np is not None) and (log_sigma_np is not None):
            lower_original, upper_original = compute_lognormal_interval_bounds(
                log_mu_np,
                log_sigma_np,
                float(log_scale_C),
                alpha=0.05,
            )
        gp_metric.update(
            _compute_log_space_metrics(
                y_test,
                y_pred_np,
                float(log_scale_C),
                output_std_original=output_std_np,
                lower_original=lower_original,
                upper_original=upper_original,
            )
        )
    # When trainer provides timing breakdown: report exactly Total_Time, Full_Train_Time, Train_Time, Log_Time, Prediction_Time at top (no duplicates)
    if (
        hasattr(trainer, "full_train_time")
        and trainer.full_train_time is not None
        and hasattr(trainer, "train_time")
        and trainer.train_time is not None
        and hasattr(trainer, "log_time")
        and trainer.log_time is not None
    ):
        _time_keys = ("Total_Time", "Full_Train_Time", "Train_Time", "Log_Time", "Prediction_Time")
        _skip = ("Total_Time", "Training_Time", "Prediction_Time")
        new_metric = {
            "Total_Time": gp_metric["Total_Time"],
            "Full_Train_Time": float(trainer.full_train_time),
            "Train_Time": float(trainer.train_time),
            "Log_Time": float(trainer.log_time),
            "Prediction_Time": gp_metric["Prediction_Time"],
        }
        for k, v in gp_metric.items():
            if k not in _skip:
                new_metric[k] = v
        gp_metric = new_metric

    # Extract noise and noise_std (will be added in correct order later)
    noise_std = None
    noise_std_original_scale = None
    try:
        nc = model.likelihood.noise_covar
        tx = model.train_inputs[0]
        if type(nc).__name__ == "LogScaleMLPNoise":
            noise_variance = nc.variance_at(tx).mean().detach().cpu()
        else:
            noise_variance = model.likelihood.noise.detach().cpu()
        noise_std = np.sqrt(noise_variance)

        if y_train_std is not None:
            if isinstance(y_train_std, dict):
                if 0 in y_train_std:
                    std_to_use = y_train_std[0]
                    mean_to_use = (
                        y_train_mean[0]
                        if isinstance(y_train_mean, dict) and 0 in y_train_mean
                        else None
                    )
                else:
                    std_to_use = list(y_train_std.values())[0]
                    mean_to_use = (
                        list(y_train_mean.values())[0]
                        if isinstance(y_train_mean, dict)
                        else None
                    )
            else:
                std_to_use = y_train_std
                mean_to_use = y_train_mean

            if standardize_y_log_scale:
                if mean_to_use is not None:
                    if isinstance(mean_to_use, torch.Tensor):
                        log_mean = (
                            mean_to_use.item()
                            if mean_to_use.numel() == 1
                            else mean_to_use.cpu().numpy()
                        )
                    else:
                        log_mean = float(mean_to_use) if np.isscalar(mean_to_use) else mean_to_use

                    if isinstance(std_to_use, torch.Tensor):
                        log_std = (
                            std_to_use.item()
                            if std_to_use.numel() == 1
                            else std_to_use.cpu().numpy()
                        )
                    else:
                        log_std = float(std_to_use) if np.isscalar(std_to_use) else std_to_use

                    log_noise_std = noise_std * log_std
                    noise_std_original_scale = np.exp(log_mean) * log_noise_std
                else:
                    if isinstance(std_to_use, torch.Tensor):
                        std_val = (
                            std_to_use.item()
                            if std_to_use.numel() == 1
                            else std_to_use.cpu().numpy()
                        )
                    else:
                        std_val = std_to_use
                    noise_std_original_scale = noise_std * std_val
            else:
                if isinstance(std_to_use, torch.Tensor):
                    std_val = (
                        std_to_use.item()
                        if std_to_use.numel() == 1
                        else std_to_use.cpu().numpy()
                    )
                else:
                    std_val = std_to_use
                noise_std_original_scale = noise_std * std_val
        else:
            noise_std_original_scale = noise_std
    except Exception as e:
        import logging

        logging.warning(f"Could not extract noise std: {e}")

    # Always extract directly from the model after training (best model already loaded)
    best_model_metrics = None

    best_run = (
        min(
            [r for r in train_results if r.get("loss") is not None],
            key=lambda x: x.get("loss", float("inf")),
            default=None,
        )
        if train_results
        else None
    )

    num_epochs_actual = trainer.num_epochs if hasattr(trainer, "num_epochs") else None
    best_epoch_value = None
    jitter_value = None
    jitter_max_value = None

    if best_run is not None:
        callback_data = best_run.get("callback_data", {}) or {}
        for cb_key, stored_params_list in callback_data.items():
            if "FinalParameterStorage" in cb_key or "ParameterStorage" in cb_key:
                if stored_params_list:
                    import math

                    def _loss_or_inf(rec):
                        val = rec.get("best_loss")
                        return float(val) if val is not None else math.inf

                    record = min(stored_params_list, key=_loss_or_inf)
                    best_epoch_value = record.get("best_epoch", best_epoch_value)
                    jitter_value = record.get("jitter", jitter_value)
                    jitter_max_value = record.get("jitter_max", jitter_max_value)
                    if num_epochs_actual is None:
                        num_epochs_actual = record.get("num_epochs", num_epochs_actual)
                break

    if hasattr(trainer, "cholesky_jitter"):
        temp_callback = FinalParameterStorageCallback(verbose=False)
        extracted_params = temp_callback._extract_final_parameters(
            model,
            epoch=0,
            best_loss=best_run.get("loss") if best_run else None,
            cholesky_jitter=trainer.cholesky_jitter,
            best_epoch=None,
            jitter_max=None,
        )
        if extracted_params:
            lengthscales_extracted = extracted_params.get("lengthscales")
            cat_lengthscales_extracted = extracted_params.get("cat_lengthscales")
            source_lengthscales_extracted = extracted_params.get("source_lengthscales")
            periods_extracted = extracted_params.get("periods")
            best_model_metrics = {
                "num_epochs": num_epochs_actual
                if num_epochs_actual is not None
                else extracted_params.get("num_epochs"),
                "best_epoch": best_epoch_value,
                "best_loss": best_run.get("loss") if best_run else extracted_params.get("best_loss"),
                "jitter": jitter_value if jitter_value is not None else extracted_params.get("jitter"),
                "jitter_max": jitter_max_value,
                "raw_noise": extracted_params.get("raw_noise"),
                "outputscale": extracted_params.get("outputscale"),
                "raw_power": extracted_params.get("raw_power"),
                "power": extracted_params.get("power"),
                "lengthscales": lengthscales_extracted,
                "cat_lengthscales": cat_lengthscales_extracted,
                "source_lengthscales": source_lengthscales_extracted,
                "periods": periods_extracted,
                "raw_periods": extracted_params.get("raw_periods"),
            }

    if best_model_metrics:
        gp_metric["jitter"] = best_model_metrics.get("jitter")
        if best_model_metrics.get("jitter_max") is not None:
            gp_metric["jitter_max"] = best_model_metrics.get("jitter_max")

        def add_metric(name, value):
            if value is None:
                return
            if hasattr(value, "numpy"):
                value = value.numpy()
            if hasattr(value, "size") and value.size == 1:
                gp_metric[name] = float(value.item() if hasattr(value, "item") else value.flat[0])
            elif hasattr(value, "__len__") and len(value) > 1:
                for i, v in enumerate(value):
                    gp_metric[f"{name}_{i}"] = float(v)
            else:
                gp_metric[name] = float(value)

        try:
            raw_noise = model.likelihood.raw_noise.detach().cpu()
            add_metric(
                "raw_noise",
                raw_noise.numpy().flatten() if raw_noise.numel() > 1 else raw_noise.item(),
            )
        except Exception:
            add_metric("raw_noise", best_model_metrics.get("raw_noise"))

        add_metric("noise", noise_std)
        add_metric("noise_std", noise_std_original_scale)
        gp_metric["outputscale"] = best_model_metrics.get("outputscale")

        raw_power = best_model_metrics.get("raw_power")
        power = best_model_metrics.get("power")
        if raw_power is not None:
            gp_metric["raw_power"] = (
                float(raw_power)
                if isinstance(raw_power, (int, float))
                else float(raw_power.item() if hasattr(raw_power, "item") else raw_power)
            )
        if power is not None:
            gp_metric["power"] = (
                float(power)
                if isinstance(power, (int, float))
                else float(power.item() if hasattr(power, "item") else power)
            )

        gp_metric.update(
            {
                "num_epochs": best_model_metrics.get("num_epochs"),
                "best_epoch": best_model_metrics.get("best_epoch"),
            }
        )

        lengthscales = best_model_metrics.get("lengthscales")
        if lengthscales is None:
            pass
        elif isinstance(lengthscales, (list, tuple)):
            if len(lengthscales) > 0:
                for i, ls_val in enumerate(lengthscales):
                    gp_metric[f"cont_lengthscale_{i}"] = ls_val
        else:
            gp_metric["cont_lengthscale_0"] = lengthscales

        cat_lengthscales = best_model_metrics.get("cat_lengthscales")
        if cat_lengthscales is not None:
            if isinstance(cat_lengthscales, (list, tuple)):
                if len(cat_lengthscales) > 0:
                    for i, ls_val in enumerate(cat_lengthscales):
                        gp_metric[f"cat_lengthscale_{i}"] = ls_val
            else:
                gp_metric["cat_lengthscale_0"] = cat_lengthscales

        source_lengthscales = best_model_metrics.get("source_lengthscales")
        if source_lengthscales is not None:
            if isinstance(source_lengthscales, (list, tuple)):
                if len(source_lengthscales) > 0:
                    for i, ls_val in enumerate(source_lengthscales):
                        gp_metric[f"source_lengthscale_{i}"] = ls_val
            else:
                gp_metric["source_lengthscale_0"] = source_lengthscales

        periods = best_model_metrics.get("periods")
        if periods is not None:
            if isinstance(periods, (list, tuple)):
                if len(periods) > 0:
                    for i, period_val in enumerate(periods):
                        gp_metric[f"cont_period_{i}"] = period_val
            else:
                gp_metric["cont_period_0"] = periods

        raw_periods = best_model_metrics.get("raw_periods")
        if raw_periods is not None:
            if isinstance(raw_periods, (list, tuple)):
                if len(raw_periods) > 0:
                    for i, period_val in enumerate(raw_periods):
                        gp_metric[f"raw_period_{i}"] = period_val
            else:
                gp_metric["raw_period_0"] = raw_periods
    else:
        pass

        def add_metric(name, value):
            if value is None:
                return
            if hasattr(value, "size"):
                if value.size == 1:
                    gp_metric[name] = float(value.item() if hasattr(value, "item") else value)
                else:
                    for i, v in enumerate(value):
                        gp_metric[f"{name}_{i}"] = float(v)
            elif isinstance(value, (list, tuple)) and len(value) > 1:
                for i, v in enumerate(value):
                    gp_metric[f"{name}_{i}"] = float(v)
            else:
                gp_metric[name] = float(value)

        add_metric("noise", noise_std)
        add_metric("noise_std", noise_std_original_scale)

    # Per-source metrics
    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]

        gp_per_source_metric = compute_per_source_metrics(
            y_test,
            y_pred_np,
            output_std_np,
            X_test,
            source_columns=source_cols,
            training_time=training_time,
            prediction_time=prediction_time,
        )

        for source_name, source_metrics in gp_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                gp_metric[f"{source_name}_{metric_name}"] = metric_value

    # Store y_train_mean/std used for this run
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            for source_key, mean_val in y_train_mean.items():
                gp_metric[f"y_train_mean_source_{source_key}"] = float(
                    mean_val.item() if hasattr(mean_val, "item") else mean_val
                )
            for source_key, std_val in y_train_std.items():
                gp_metric[f"y_train_std_source_{source_key}"] = float(
                    std_val.item() if hasattr(std_val, "item") else std_val
                )
        else:
            if hasattr(y_train_mean, "item"):
                if y_train_mean.numel() == 1:
                    mean_val = y_train_mean.item()
                else:
                    mean_val = y_train_mean.detach().cpu().tolist()
            else:
                mean_val = y_train_mean
            if hasattr(y_train_std, "item"):
                if y_train_std.numel() == 1:
                    std_val = y_train_std.item()
                else:
                    std_val = y_train_std.detach().cpu().tolist()
            else:
                std_val = y_train_std
            gp_metric["y_train_mean"] = mean_val
            gp_metric["y_train_std"] = std_val

    # Trainer info structure (includes optional lbfgs_inner_metrics)
    gp_trainer_info = None
    if trainer_info:
        from gpplus.training.trainer_analysis import build_stored_params_from_results, build_trainer_analysis_payload
        all_stored_params = build_stored_params_from_results(train_results)

        if all_stored_params:
            gp_trainer_info = build_trainer_analysis_payload(all_stored_params, train_results, num_epochs)
        else:
            # Fallback when no callback data: extract from model and best run
            best_run = (
                min(
                    [r for r in train_results if r.get("loss") is not None],
                    key=lambda x: x.get("loss", float("inf")),
                    default=None,
                )
                if train_results
                else None
            )
            if best_run is not None and hasattr(trainer, "cholesky_jitter"):
                temp_cb = FinalParameterStorageCallback(verbose=False)
                extracted = temp_cb._extract_final_parameters(
                    model,
                    epoch=best_run.get("run_index", best_run.get("init_index", 0)),
                    best_loss=best_run.get("loss"),
                    cholesky_jitter=trainer.cholesky_jitter,
                    best_epoch=None,
                )
                if extracted:
                    run_id = int(best_run.get("run_index", best_run.get("init_index", 0))) + 1
                    record = {
                        "run": run_id,
                        "num_epochs": extracted.get(
                            "num_epochs",
                            getattr(trainer, "num_epochs", None),
                        ),
                        "best_epoch": extracted.get("best_epoch"),
                        "best_loss": best_run.get("loss"),
                        "jitter": extracted.get("jitter"),
                        "initial": None,
                        "final": extracted,
                    }
                    all_stored_params = [record]
                    gp_trainer_info = build_trainer_analysis_payload(all_stored_params, train_results, num_epochs)

            if gp_trainer_info is None:
                gp_trainer_info = {
                    "inits": [],
                    "best_parameters": None,
                    "average_final_parameters": {},
                }
        # Add trainer timing to gp_trainer_info for JSON output
        if gp_trainer_info is not None and hasattr(trainer, "full_train_time"):
            if trainer.full_train_time is not None:
                gp_trainer_info["full_train_time"] = float(trainer.full_train_time)
            if trainer.train_time is not None:
                gp_trainer_info["train_time"] = float(trainer.train_time)
            if trainer.log_time is not None:
                gp_trainer_info["log_time"] = float(trainer.log_time)

    # Always return 4 values (gp_trainer_info may be None)
    return gp_metric, y_pred_np, output_std_np, gp_trainer_info


def train_eval_PFN(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    amp_device: str,
    amp_dtype,
    regressor=None,
    y_train_mean=None,
    y_train_std=None,
    standardize_y_log_scale: bool = False,
    log_scale_C: float | None = None,  # C used in log(y + C) transformation. If None, will use LogScaler's C from fit.
    log_y_point_inverse: str = "median",
    source_cols=None,
):
    """
    Thin wrapper around PFN evaluation identical to v1/v2.
    PFN evaluation helper (TabPFN); independent of GP training stack.
    """
    import numpy as np
    import torch

    try:
        from tabpfn import TabPFNRegressor as StandardTabPFNRegressor

        is_standard_tabpfn = isinstance(regressor, StandardTabPFNRegressor)
    except ImportError:
        is_standard_tabpfn = False

    if is_standard_tabpfn:
        X_train_np = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        X_test_np = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_train_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_test_np = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

        if y_train_np.ndim > 1:
            y_train_np = y_train_np.ravel()
        if y_test_np.ndim > 1:
            y_test_np = y_test_np.ravel()

        t_fit_start = time.time()
        if not hasattr(regressor, "feature_names_in_"):
            regressor.fit(X_train_np, y_train_np)
        training_time = time.time() - t_fit_start

        t_pred_start = time.time()
        y_var_tabpfn = None
        lower_95_test = None
        upper_95_test = None

        full_predictions = regressor.predict(
            X_test_np,
            output_type="full",
            quantiles=[0.025, 0.975],
        )
        t_predict_call = time.time() - t_pred_start
        print(f"[TIMER] predict(output_type='full') took: {t_predict_call:.4f}s")

        if "logits" in full_predictions and "criterion" in full_predictions:
            logits = full_predictions["logits"]
            criterion = full_predictions["criterion"]
            if hasattr(criterion, "variance"):
                t_var_calc_start = time.time()
                if isinstance(logits, np.ndarray):
                    logits = torch.tensor(logits)
                variance = criterion.variance(logits)
                y_var_tabpfn = variance.detach().cpu().numpy()
                t_var_calc = time.time() - t_var_calc_start
                print(f"[TIMER] criterion.variance(logits) calculation took: {t_var_calc:.4f}s")
                y_pred_tabpfn = full_predictions.get("mean")

                tabpfn_logits_for_crps = logits.detach().cpu()
                tabpfn_bar_dist_for_crps = criterion
                print(f"[CRPS] Storing logits shape: {logits.shape} for CRPS")

        if "quantiles" in full_predictions:
            q_95 = full_predictions["quantiles"]
            if isinstance(q_95, list) and len(q_95) >= 2:
                lower_95_test, upper_95_test = q_95[0], q_95[1]

        if isinstance(y_pred_tabpfn, torch.Tensor):
            y_pred_tabpfn = y_pred_tabpfn.detach().cpu().numpy()
        if isinstance(y_var_tabpfn, torch.Tensor):
            y_var_tabpfn = y_var_tabpfn.detach().cpu().numpy()
        if y_pred_tabpfn.ndim > 1:
            y_pred_tabpfn = y_pred_tabpfn.ravel()
        if isinstance(y_var_tabpfn, np.ndarray) and y_var_tabpfn.ndim > 1:
            y_var_tabpfn = y_var_tabpfn.ravel()
        elif not isinstance(y_var_tabpfn, np.ndarray):
            y_var_tabpfn = np.array(y_var_tabpfn)
        prediction_time = time.time() - t_pred_start

        y_pred_test = y_pred_tabpfn
        output_std_test = np.sqrt(y_var_tabpfn)
        if "tabpfn_logits_for_crps" in locals():
            tabpfn_logits_test = tabpfn_logits_for_crps
            tabpfn_bar_dist_test = tabpfn_bar_dist_for_crps
    else:
        t_fit_start = time.time()
        X_all = np.concatenate([X_train, X_test], axis=0)
        Y_all = np.concatenate([y_train, np.zeros_like(y_test)], axis=0)

        X_all = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1)
        Y_all = torch.tensor(Y_all, dtype=torch.float32).reshape(-1, 1, 1)
        training_time = time.time() - t_fit_start

        single_eval_pos = len(X_train)
        t_pred_start = time.time()
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_all, Y_all, single_eval_pos)
            logits = out["standard"]
            y_mean = regressor.predict_mean(logits)
            y_var = regressor.predict_variance(logits)

            logits_test = logits[-len(y_test) :]
            tabpfn_logits_for_crps = logits_test.detach().cpu()
            tabpfn_bar_dist_for_crps = regressor.bardist_
            print(f"[CRPS] Storing logits shape: {logits_test.shape} for CRPS")

        y_pred = y_mean.detach().cpu().numpy().reshape(-1)
        output_std = (y_var.detach().cpu().numpy().reshape(-1)) ** 0.5
        prediction_time = time.time() - t_pred_start

        y_pred_test = y_pred[-len(y_test) :]
        output_std_test = output_std[-len(y_test) :]
        lower_95_test = None
        upper_95_test = None
        try:
            logits_test_for_q = logits[-len(y_test) :]
            q025 = regressor.bardist_.icdf(logits_test_for_q, 0.025).detach().cpu().numpy().reshape(-1)
            q975 = regressor.bardist_.icdf(logits_test_for_q, 0.975).detach().cpu().numpy().reshape(-1)
            lower_95_test, upper_95_test = q025, q975
        except Exception:
            lower_95_test, upper_95_test = None, None
        if "tabpfn_logits_for_crps" in locals():
            tabpfn_logits_test = tabpfn_logits_for_crps
            tabpfn_bar_dist_test = tabpfn_bar_dist_for_crps

    y_test_normalized = y_test.copy() if isinstance(y_test, np.ndarray) else np.array(y_test)

    _validate_log_y_point_inverse(log_y_point_inverse)
    _require_finite_log_scale_c(
        standardize_y_log_scale,
        log_scale_C,
        y_train_mean,
        y_train_std,
        caller="train_eval_PFN",
    )

    if (y_train_mean is not None) and (y_train_std is not None):
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            if source_cols is not None:
                if isinstance(X_test, np.ndarray):
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                else:
                    X_test_tensor = X_test

                is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
                if is_onehot:
                    onehot_cols = X_test_tensor[:, source_cols]
                    source_indices_test = torch.argmax(onehot_cols, dim=1)
                else:
                    source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
                    source_indices_test = X_test_tensor[:, source_col].long()

                source_indices_test_np = source_indices_test.detach().cpu().numpy()

                y_pred_test_denorm = y_pred_test.copy()
                output_std_test_denorm = output_std_test.copy()
                lower_95_test_denorm = lower_95_test.copy() if lower_95_test is not None else None
                upper_95_test_denorm = upper_95_test.copy() if upper_95_test is not None else None
                unique_sources = np.unique(source_indices_test_np)

                for source_idx in unique_sources:
                    source_mask = source_indices_test_np == source_idx
                    source_key = int(source_idx)

                    if source_key in y_train_mean:
                        mean = float(y_train_mean[source_key])
                        std = float(y_train_std[source_key])
                    elif 0 in y_train_mean:
                        mean = float(y_train_mean[0])
                        std = float(y_train_std[0])
                    else:
                        first_key = list(y_train_mean.keys())[0]
                        mean = float(y_train_mean[first_key])
                        std = float(y_train_std[first_key])

                    if standardize_y_log_scale:
                        log_y_pred = (y_pred_test[source_mask] * std) + mean
                        log_y_std = output_std_test[source_mask] * std
                        max_log_val = 700.0
                        log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                        exp_log_y = np.exp(log_y_pred)
                        y_pred_test_denorm[source_mask] = _np_log_y_point_original(
                            exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                        )
                        output_std_test_denorm[source_mask] = exp_log_y * log_y_std

                        if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                            log_lower = (lower_95_test[source_mask] * std) + mean
                            log_upper = (upper_95_test[source_mask] * std) + mean
                            log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                            log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                            lower_95_test_denorm[source_mask] = np.exp(log_lower) - log_scale_C
                            upper_95_test_denorm[source_mask] = np.exp(log_upper) - log_scale_C
                    else:
                        y_pred_test_denorm[source_mask] = (y_pred_test[source_mask] * std) + mean
                        output_std_test_denorm[source_mask] = output_std_test[source_mask] * std
                        if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                            lower_95_test_denorm[source_mask] = (lower_95_test[source_mask] * std) + mean
                            upper_95_test_denorm[source_mask] = (upper_95_test[source_mask] * std) + mean

                y_pred_test = y_pred_test_denorm
                output_std_test = output_std_test_denorm
                if lower_95_test_denorm is not None and upper_95_test_denorm is not None:
                    lower_95_test = lower_95_test_denorm
                    upper_95_test = upper_95_test_denorm
            else:
                first_key = list(y_train_mean.keys())[0]
                mean = float(y_train_mean[first_key])
                std = float(y_train_std[first_key])
                if standardize_y_log_scale:
                    log_y_pred = (y_pred_test * std) + mean
                    log_y_std = output_std_test * std
                    max_log_val = 700.0
                    log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                    exp_log_y = np.exp(log_y_pred)
                    y_pred_test = _np_log_y_point_original(
                        exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                    )
                    output_std_test = exp_log_y * log_y_std
                    if lower_95_test is not None and upper_95_test is not None:
                        log_lower = (lower_95_test * std) + mean
                        log_upper = (upper_95_test * std) + mean
                        log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                        log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                        lower_95_test = np.exp(log_lower) - log_scale_C
                        upper_95_test = np.exp(log_upper) - log_scale_C
                else:
                    y_pred_test = (y_pred_test * std) + mean
                    output_std_test = output_std_test * std
                    if lower_95_test is not None and upper_95_test is not None:
                        lower_95_test = (lower_95_test * std) + mean
                        upper_95_test = (upper_95_test * std) + mean
        else:
            if standardize_y_log_scale:
                log_y_pred = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                log_y_std = output_std_test * float(y_train_std)
                max_log_val = 700.0
                log_y_pred = np.clip(log_y_pred, -max_log_val, max_log_val)
                exp_log_y = np.exp(log_y_pred)
                y_pred_test = _np_log_y_point_original(
                    exp_log_y, log_y_std, float(log_scale_C), log_y_point_inverse
                )
                output_std_test = exp_log_y * log_y_std
                if lower_95_test is not None and upper_95_test is not None:
                    log_lower = (lower_95_test * float(y_train_std)) + float(y_train_mean)
                    log_upper = (upper_95_test * float(y_train_std)) + float(y_train_mean)
                    log_lower = np.clip(log_lower, -max_log_val, max_log_val)
                    log_upper = np.clip(log_upper, -max_log_val, max_log_val)
                    lower_95_test = np.exp(log_lower) - log_scale_C
                    upper_95_test = np.exp(log_upper) - log_scale_C
            else:
                y_pred_test = (y_pred_test * float(y_train_std)) + float(y_train_mean)
                output_std_test = output_std_test * float(y_train_std)
                if lower_95_test is not None and upper_95_test is not None:
                    lower_95_test = (lower_95_test * float(y_train_std)) + float(y_train_mean)
                    upper_95_test = (upper_95_test * float(y_train_std)) + float(y_train_mean)

    if standardize_y_log_scale and isinstance(y_pred_test, np.ndarray):
        if np.any(~np.isfinite(y_pred_test)):
            n_bad = int(np.sum(~np.isfinite(y_pred_test)))
            warnings.warn(
                f"train_eval_PFN: {n_bad} non-finite values in y_pred_test after log-scale denorm.",
                stacklevel=2,
            )

    tabpfn_logits_for_metrics = tabpfn_logits_test if "tabpfn_logits_test" in locals() else None
    tabpfn_bar_dist_for_metrics = tabpfn_bar_dist_test if "tabpfn_bar_dist_test" in locals() else None
    y_test_for_crps = y_test_normalized if "y_test_normalized" in locals() else y_test

    y_true_pfn = (
        y_test.detach().cpu().numpy().reshape(-1)
        if isinstance(y_test, torch.Tensor)
        else np.asarray(y_test).reshape(-1)
    )
    y_pred_flat = np.asarray(y_pred_test).reshape(-1)
    if y_true_pfn.shape[0] != y_pred_flat.shape[0]:
        raise ValueError(
            "train_eval_PFN: y_test and predictions length mismatch — "
            f"len(y_test)={y_true_pfn.shape[0]}, len(y_pred)={y_pred_flat.shape[0]}."
        )

    metrics = compute_metrics(
        y_test,
        y_pred_test,
        output_std_test,
        training_time=training_time,
        prediction_time=prediction_time,
        tabpfn_logits=tabpfn_logits_for_metrics,
        tabpfn_bar_dist=tabpfn_bar_dist_for_metrics,
        y_test_normalized=y_test_for_crps,
        lower_95=lower_95_test,
        upper_95=upper_95_test,
    )
    if (
        standardize_y_log_scale
        and y_train_mean is not None
        and y_train_std is not None
        and log_scale_C is not None
    ):
        metrics.update(
            _compute_log_space_metrics(
                y_test,
                y_pred_test,
                float(log_scale_C),
                output_std_original=output_std_test,
                lower_original=lower_95_test,
                upper_original=upper_95_test,
            )
        )

    if isinstance(source_cols, int) or (isinstance(source_cols, (list, tuple)) and len(source_cols) > 0):
        if isinstance(source_cols, (list, tuple)) and len(source_cols) == 1:
            source_cols = source_cols[0]

        pfn_per_source_metric = compute_per_source_metrics(
            y_test, y_pred_test, output_std_test, X_test, source_columns=source_cols
        )

        for source_name, source_metrics in pfn_per_source_metric["per_source"].items():
            for metric_name, metric_value in source_metrics.items():
                metrics[f"{source_name}_{metric_name}"] = metric_value

    return metrics, y_pred_test, output_std_test

