"""Shared GPyTorch training/eval utilities for GPvsPFN experiments.

This module exists to avoid duplicating the large `train_eval_gp_gpytorch_default` helper
across each `*_gpytorch.py` experiment script.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import numpy as np
import torch
import gpytorch
import linear_operator

try:
    from linear_operator.utils.errors import NotPSDError, NanError
except Exception:  # pragma: no cover
    from linear_operator.utils.errors import NotPSDError  # type: ignore

    NanError = NotPSDError  # type: ignore

from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.metrics_functions import compute_metrics

# Allow this helper to work both when running problem scripts from this folder
# (where `defaults_gpytorch` is importable as a local module) and when importing
# via `experiments.Problems.*` from the repo root.
try:
    import defaults_gpytorch as defaults
except ModuleNotFoundError:  # pragma: no cover
    from experiments.Problems import defaults_gpytorch as defaults


def train_eval_gp_gpytorch_default(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    num_inits: int = 1,
    seed: int | None = None,    
    device: str = "cpu",
    y_train_mean: torch.Tensor | None = None,
    y_train_std: torch.Tensor | None = None,
    convergence_patience: int | None = None,
    min_loss_change: float = 1e-7,
    optimizer_class=None,
    lr: float | None = None,
    standardize_y_log_scale: bool = False,
    y_train_min: float | None = None,
    iteration_callbacks: list[Callable[..., None]] | None = None,
):
    """
    Train a GP model using gpytorch components (ExactMarginalLogLikelihood).

    - Supports multiple runs with different initializations, selecting the best run by loss.
    - Supports LBFGS (with closure) and standard optimizers (Adam/SGD/etc.).
    - Adds robust NaN/NotPSD handling via jitter escalation.
    - Optional iteration_callbacks: list of callables invoked once per outer
      optimization step (\"epoch\") with (iteration=epoch, loss=loss_val, run_idx=run_idx).
      This is intentionally lightweight and independent of gpplus trainer classes.

    Returns:
        gp_metric: dict of computed metrics
        y_pred: numpy array of predictions (denormalized if mean/std provided)
        output_std: numpy array of predictive std (denormalized if mean/std provided)
    """

    # Move model and data to device (keep y_test on CPU for metric computation)
    model = model.to(device)
    X_test = X_test.to(device)
    # Keep y_test on CPU - it's in original scale and used only for metrics
    y_test_cpu = y_test.detach().clone().cpu() if isinstance(y_test, torch.Tensor) else y_test

    # Store original model state for reinitialization
    import copy

    original_state = copy.deepcopy(model.state_dict())

    # Track best run
    best_loss = float("inf")
    best_state_dict = None
    best_run_index = None
    best_epoch = None
    run_results: list[dict[str, Any]] = []

    # Track final jitter value used during training (for evaluation)
    final_jitter = 1e-6  # Default gpytorch jitter

    # Track best loss per run (for diagnostics / logging).
    best_loss_per_run: list[float] = [float("inf")] * num_inits

    # Training time tracking
    t_train_start = time.time()
    logging_time = 0.0  # time spent on printing / callbacks during training (not optimizer work)

    # Multiple runs with different initializations
    # Use seed to create reproducible random state for each run
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    for run_idx in range(num_inits):
        # Reinitialize model parameters for each run (except first run which uses original initialization)
        if run_idx > 0:
            # Reinitialize hyperparameters with seed-based random initialization
            # This ensures reproducibility while providing different initializations per run
            with torch.no_grad():
                # Reinitialize likelihood noise
                if hasattr(model.likelihood, "raw_noise"):
                    model.likelihood.raw_noise.data.normal_(mean=0, std=0.1, generator=rng)

                # Reinitialize mean module
                if hasattr(model.mean_module, "constant"):
                    model.mean_module.constant.data.normal_(mean=0, std=0.1, generator=rng)

                # Reinitialize kernel parameters
                if hasattr(model.covar_module, "base_kernel"):
                    if hasattr(model.covar_module.base_kernel, "raw_lengthscale"):
                        # For ARD kernels
                        model.covar_module.base_kernel.raw_lengthscale.data.normal_(
                            mean=0, std=0.1, generator=rng
                        )
                    elif hasattr(model.covar_module.base_kernel, "lengthscale"):
                        model.covar_module.base_kernel.lengthscale.data.normal_(
                            mean=1.0, std=0.1, generator=rng
                        )

                # Reinitialize outputscale
                if hasattr(model.covar_module, "raw_outputscale"):
                    model.covar_module.raw_outputscale.data.normal_(mean=0, std=0.1, generator=rng)

        # Set model and likelihood to training mode
        model.train()
        model.likelihood.train()

        # Exact MLL
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # Determine optimizer class and learning rate
        if optimizer_class is None:
            optimizer_class = torch.optim.LBFGS

        if lr is None:
            if optimizer_class == torch.optim.LBFGS:
                lr = getattr(defaults, "LBFGS_LR", 1.0)
            elif optimizer_class == torch.optim.Adam:
                lr = getattr(defaults, "ADAM_LR", 1e-3)
            else:
                lr = getattr(defaults, "TRAINER_LR", 0.1)

        is_lbfgs = optimizer_class == torch.optim.LBFGS

        # Create optimizer
        if is_lbfgs:
            optimizer = optimizer_class(
                model.parameters(),
                lr=lr,
                max_iter=getattr(defaults, "LBFGS_MAX_ITER", 20),
            )
        elif optimizer_class == torch.optim.Adam:
            optimizer = optimizer_class(
                model.parameters(),
                lr=lr,
                betas=getattr(defaults, "ADAM_BETAS", (0.9, 0.999)),
                eps=getattr(defaults, "ADAM_EPS", 1e-8),
                weight_decay=getattr(defaults, "ADAM_WEIGHT_DECAY", 0.0),
            )
        else:
            optimizer = optimizer_class(model.parameters(), lr=lr)

        # Define closure function for LBFGS (and re-used for other optimizers) with NaN/NotPSD protection
        def closure():
            optimizer.zero_grad()
            try:
                # Access training data from the model (stored by ExactGP)
                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets)

                # Check for NaN or Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    # Return a large penalty value to discourage this parameter configuration
                    loss = torch.tensor(
                        1e6, dtype=loss.dtype, device=loss.device, requires_grad=True
                    )
                    return loss

                loss.backward()

                # Check for NaN in gradients
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            # Zero out NaN/Inf gradients
                            param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0.0

                return loss
            except NotPSDError:
                # Catch NotPSDError - return a large penalty value
                model_dtype = next(model.parameters()).dtype
                model_device = next(model.parameters()).device
                loss = torch.tensor(
                    1e6, dtype=model_dtype, device=model_device, requires_grad=True
                )
                return loss
            except (RuntimeError, ValueError) as e:
                # Catch errors like NaN in covariance matrix
                if (
                    "NaN" in str(e)
                    or "nan" in str(e).lower()
                    or "NotPSD" in str(e)
                    or "not p.d." in str(e).lower()
                ):
                    # Return a large penalty value
                    model_dtype = next(model.parameters()).dtype
                    model_device = next(model.parameters()).device
                    loss = torch.tensor(
                        1e6, dtype=model_dtype, device=model_device, requires_grad=True
                    )
                    return loss
                # Re-raise if it's a different error
                raise

        # Training loop for this run with convergence checking
        run_best_loss = float("inf")
        run_best_epoch = 0
        run_best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None
        early_stop_triggered = False

        # Set jitter for numerical stability - start with default, increase if needed
        jitter = 1e-6  # Default gpytorch jitter
        max_jitter = 1e-3  # Maximum jitter to try
        # Track jitter for this run (will be updated if increased)
        run_jitter = jitter

        for epoch in range(num_epochs):
            # Use jitter settings context manager for Cholesky decomposition
            loss_val = None  # Initialize to None for each epoch
            epoch_successful = False
            skip_run = False

            while not epoch_successful and not skip_run:
                with (
                    gpytorch.settings.cholesky_jitter(jitter),
                    linear_operator.settings.cholesky_jitter(
                        float_value=jitter, double_value=jitter
                    ),
                ):
                    try:
                        if is_lbfgs:
                            loss = optimizer.step(closure)
                        else:
                            loss = closure()
                            optimizer.step()

                        # Check if loss is NaN or Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NaN/Inf loss detected. Skipping this run."
                            )
                            # Skip this run - restore original state and break
                            model.load_state_dict(original_state)
                            skip_run = True
                            break

                        # Track best epoch and loss for this run
                        loss_val = loss.item()
                        epoch_successful = True
                    except NotPSDError:
                        # Try increasing jitter if we haven't reached max
                        if jitter < max_jitter:
                            jitter = min(jitter * 10, max_jitter)
                            run_jitter = jitter  # Update run jitter
                            print(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NotPSDError detected. Increasing jitter to {jitter:.1e}."
                            )
                            # Retry this epoch with higher jitter
                            epoch_successful = False
                        else:
                            print(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NotPSDError persists even with jitter={jitter:.1e}. Skipping this run."
                            )
                            # Skip this run - restore original state and break
                            model.load_state_dict(original_state)
                            skip_run = True
                            break
                    except (RuntimeError, ValueError) as e:
                        # Catch NaN errors or NotPSD errors in covariance matrix
                        error_str = str(e).lower()
                        if "nan" in error_str or "nanerror" in error_str:
                            print(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NaN error detected: {e}. Skipping this run."
                            )
                            model.load_state_dict(original_state)
                            skip_run = True
                            break
                        if (
                            "notpsd" in error_str
                            or "not p.d." in error_str
                            or "not positive definite" in error_str
                        ):
                            if jitter < max_jitter:
                                jitter = min(jitter * 10, max_jitter)
                                run_jitter = jitter
                                print(
                                    f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NotPSD error detected. Increasing jitter to {jitter:.1e}."
                                )
                                epoch_successful = False
                                continue
                            print(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{num_epochs}: NotPSD error persists even with jitter={jitter:.1e}. Skipping this run."
                            )
                            model.load_state_dict(original_state)
                            skip_run = True
                            break
                        raise

            # If we need to skip this run, break out of the for loop
            if skip_run or loss_val is None:
                break

            # Optionally invoke any simple iteration callbacks (track time separately as logging_time)
            if iteration_callbacks:
                t_log_start = time.time()
                for cb in iteration_callbacks:
                    try:
                        cb(iteration=epoch, loss=loss_val, run_idx=run_idx)
                    except TypeError:
                        # Callback may only accept (iteration, loss)
                        try:
                            cb(iteration=epoch, loss=loss_val)
                        except Exception as e:
                            warnings.warn(f"Iteration callback raised an error: {e}")
                    except Exception as e:
                        warnings.warn(f"Iteration callback raised an error: {e}")
                logging_time += time.time() - t_log_start

            if loss_val < run_best_loss:
                run_best_loss = loss_val
                run_best_epoch = epoch
                run_best_state_dict = copy.deepcopy(model.state_dict())
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Check for early stopping conditions
            early_stop_reason = ""

            # Condition 1: No improvement for convergence_patience epochs
            if convergence_patience is not None and no_improvement_epochs >= convergence_patience:
                early_stop_triggered = True
                early_stop_reason = f"No improvement for {convergence_patience} epochs"

            # Condition 2: Absolute loss change is below threshold (OR condition)
            if previous_loss is not None:
                loss_change = abs(previous_loss - loss_val)
                if loss_change < min_loss_change:
                    early_stop_triggered = True
                    if early_stop_reason:
                        early_stop_reason += (
                            f" OR absolute loss change below {min_loss_change:.1e}"
                        )
                    else:
                        early_stop_reason = f"absolute loss change below {min_loss_change:.1e}"

            if early_stop_triggered:
                # Early stopping triggered - restore best model state
                if run_best_state_dict is not None:
                    model.load_state_dict(run_best_state_dict)
                t_log_start = time.time()
                print(
                    f"  Early stopping at epoch {epoch + 1}: {early_stop_reason}. Best loss: {run_best_loss:.6f}"
                )
                logging_time += time.time() - t_log_start
                break

            # Update previous_loss for next epoch
            previous_loss = loss_val

        # After training loop, ensure we're using the best model state
        if run_best_state_dict is not None:
            model.load_state_dict(run_best_state_dict)

        # Store run results
        final_loss = run_best_loss
        best_loss_per_run[run_idx] = float(final_loss)
        run_results.append(
            {
                "run_index": run_idx,
                "loss": final_loss,
                "best_epoch": run_best_epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
        )

        # Update best run if this is better
        if final_loss < best_loss:
            best_loss = final_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_run_index = run_idx
            best_epoch = run_best_epoch
            final_jitter = run_jitter  # Update final jitter to the best run's jitter

    total_training_wall_time = time.time() - t_train_start
    training_time = total_training_wall_time - logging_time

    # Check if any run was successful
    if best_state_dict is None:
        print(f"  ERROR: All {num_inits} training runs failed due to numerical instability.")
        print("  This indicates severe numerical issues. Possible causes:")
        print("    - Data scaling problems")
        print("    - Incompatible hyperparameter initialization")
        print(f"    - Insufficient jitter (max tried: {max_jitter})")
        print("    - Model/data mismatch")
        # Return dummy metrics with NaN values - skip compute_metrics since it can't handle NaN
        y_pred_np = np.full(len(y_test_cpu), np.nan)
        output_std_np = np.full(len(y_test_cpu), np.nan)
        # Create metrics dict manually with NaN values
        gp_metric = {
            "Total_Time": total_training_wall_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": 0.0,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,  # Record the jitter that was attempted
            "evaluation_error": f"All {num_inits} training runs failed - no valid model",
            "all_runs_failed": True,
            "all_metrics_nan": True,
            "best_loss_per_run": best_loss_per_run,
        }
        return gp_metric, y_pred_np, output_std_np

    # Load best model state
    model.load_state_dict(best_state_dict)
    model.eval()
    model.likelihood.eval()

    # Validate model parameters before evaluation
    has_nan_inf = False
    param_info: list[str] = []

    # Check likelihood noise
    if hasattr(model.likelihood, "raw_noise"):
        noise_val = model.likelihood.raw_noise.detach()
        if torch.isnan(noise_val).any() or torch.isinf(noise_val).any():
            has_nan_inf = True
            param_info.append(f"likelihood.raw_noise: {noise_val}")

    # Check outputscale
    if hasattr(model.covar_module, "raw_outputscale"):
        outputscale_val = model.covar_module.raw_outputscale.detach()
        if torch.isnan(outputscale_val).any() or torch.isinf(outputscale_val).any():
            has_nan_inf = True
            param_info.append(f"covar_module.raw_outputscale: {outputscale_val}")

    # Check lengthscales
    if hasattr(model.covar_module, "base_kernel") and hasattr(
        model.covar_module.base_kernel, "raw_lengthscale"
    ):
        lengthscale_val = model.covar_module.base_kernel.raw_lengthscale.detach()
        if torch.isnan(lengthscale_val).any() or torch.isinf(lengthscale_val).any():
            has_nan_inf = True
            param_info.append(f"base_kernel.raw_lengthscale: {lengthscale_val}")

    if has_nan_inf:
        print(
            f"  WARNING: Model has invalid hyperparameters (NaN/Inf detected): {param_info}"
        )
        print("  This indicates numerical instability. Skipping evaluation for this run.")
        y_pred_np = np.full(len(y_test_cpu), np.nan)
        output_std_np = np.full(len(y_test_cpu), np.nan)
        gp_metric = {
            "Total_Time": training_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": 0.0,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,
            "evaluation_error": f"NaN/Inf in parameters: {param_info}",
            "all_metrics_nan": True,
        }
        return gp_metric, y_pred_np, output_std_np

    # Evaluation with error handling and jitter settings
    t_pred_start = time.time()
    try:
        eval_jitter = final_jitter
        with (
            gpytorch.settings.cholesky_jitter(eval_jitter),
            linear_operator.settings.cholesky_jitter(
                float_value=eval_jitter, double_value=eval_jitter
            ),
        ):
            y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
    except (NanError, NotPSDError) as e:
        print(f"  WARNING: Evaluation failed due to numerical instability: {e}")
        print("  Attempting evaluation with increased jitter...")
        try:
            with (
                gpytorch.settings.cholesky_jitter(1e-3),
                linear_operator.settings.cholesky_jitter(
                    float_value=1e-3, double_value=1e-3
                ),
            ):
                y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
                final_jitter = 1e-3
        except Exception as e2:
            print(f"  ERROR: Evaluation failed even with maximum jitter: {e2}")
            y_pred_np = np.full(len(y_test_cpu), np.nan)
            output_std_np = np.full(len(y_test_cpu), np.nan)
            gp_metric = {
                "Total_Time": training_time,
                "Training_Time": training_time,
                "Logging_Time": logging_time,
                "Prediction_Time": 0.0,
                "RRMSE": np.nan,
                "RMSE": np.nan,
                "MSE": np.nan,
                "NIS": np.nan,
                "NIS_width": np.nan,
                "NIS_outside": np.nan,
                "jitter": final_jitter,
                "evaluation_error": str(e2),
                "all_metrics_nan": True,
            }
            return gp_metric, y_pred_np, output_std_np
    except (RuntimeError, ValueError) as e:
        error_str = str(e).lower()
        if (
            "nan" in error_str
            or "notpsd" in error_str
            or "not p.d." in error_str
            or "not positive definite" in error_str
        ):
            print(f"  WARNING: Evaluation failed due to numerical instability: {e}")
            print("  Attempting evaluation with increased jitter...")
            try:
                with (
                    gpytorch.settings.cholesky_jitter(1e-3),
                    linear_operator.settings.cholesky_jitter(
                        float_value=1e-3, double_value=1e-3
                    ),
                ):
                    y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
                    final_jitter = 1e-3
            except Exception as e2:
                print(f"  ERROR: Evaluation failed even with maximum jitter: {e2}")
                y_pred_np = np.full(len(y_test_cpu), np.nan)
                output_std_np = np.full(len(y_test_cpu), np.nan)
                gp_metric = {
                    "Total_Time": training_time,
                    "Training_Time": training_time,
                    "Logging_Time": logging_time,
                    "Prediction_Time": 0.0,
                    "RRMSE": np.nan,
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "NIS": np.nan,
                    "NIS_width": np.nan,
                    "NIS_outside": np.nan,
                    "jitter": final_jitter,
                    "evaluation_error": str(e2),
                    "all_metrics_nan": True,
                }
                return gp_metric, y_pred_np, output_std_np
        else:
            raise

    prediction_time = time.time() - t_pred_start

    # Denormalize if needed
    if y_train_mean is not None and y_train_std is not None:
        y_pred = (y_pred * y_train_std) + y_train_mean
        output_std = output_std * y_train_std

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)

    # Check if predictions are all NaN before computing metrics
    if np.all(np.isnan(y_pred_np)):
        print("  WARNING: All predictions are NaN. Skipping metric computation.")
        gp_metric = {
            "Total_Time": training_time + logging_time + prediction_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": prediction_time,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,
            "evaluation_error": "All predictions are NaN",
            "all_metrics_nan": True,
        }
    else:
        gp_metric = compute_metrics(
            y_test_cpu,
            y_pred_np,
            output_std_np,
            training_time=training_time,
            prediction_time=prediction_time,
        )

    # Check if all metrics are NaN (indicates complete failure)
    excluded_keys = [
        "training_time",
        "prediction_time",
        "num_epochs",
        "best_epoch",
        "evaluation_error",
        "all_runs_failed",
        "all_metrics_nan",
    ]
    metric_values = [
        v
        for k, v in gp_metric.items()
        if k not in excluded_keys and isinstance(v, (int, float))
    ]
    if len(metric_values) > 0 and all(np.isnan(v) for v in metric_values):
        print(
            "  WARNING: All computed metrics are NaN. This indicates evaluation failed completely."
        )
        gp_metric["evaluation_error"] = "All metrics are NaN - evaluation failed"
        gp_metric["all_metrics_nan"] = True

    # Attach best loss per run (outer loop): useful for logging/comparison without full history
    gp_metric["best_loss_per_run"] = best_loss_per_run

    # Extract hyperparameters from the trained model
    gp_metric["jitter"] = final_jitter

    # Extract raw_noise
    try:
        raw_noise = model.likelihood.raw_noise.detach().cpu()
        gp_metric["raw_noise"] = float(raw_noise.item()) if raw_noise.numel() == 1 else float(
            raw_noise.numpy().flatten()[0]
        )
    except Exception:
        gp_metric["raw_noise"] = np.nan

    # Extract noise (transformed) and compute noise_std
    noise_std_original_scale = None
    try:
        noise_variance = model.likelihood.noise.detach().cpu()
        noise_val = float(noise_variance.item()) if noise_variance.numel() == 1 else float(
            noise_variance.numpy().flatten()[0]
        )
        gp_metric["noise"] = noise_val
        noise_std = float(np.sqrt(noise_val))

        # Convert to original output scale if y was standardized
        if y_train_std is not None:
            if isinstance(y_train_std, dict):
                std_to_use = y_train_std[0] if 0 in y_train_std else list(y_train_std.values())[0]
            else:
                std_to_use = y_train_std.item() if hasattr(y_train_std, "item") else y_train_std
            noise_std_original_scale = noise_std * std_to_use
        else:
            noise_std_original_scale = noise_std
        gp_metric["noise_std"] = float(noise_std_original_scale)
    except Exception as e:
        import logging

        logging.warning(f"Could not extract noise: {e}")
        gp_metric["noise"] = np.nan
        gp_metric["noise_std"] = np.nan

    # Extract outputscale
    try:
        outputscale = model.covar_module.outputscale.detach().cpu()
        gp_metric["outputscale"] = float(outputscale.item()) if outputscale.numel() == 1 else float(
            outputscale.numpy().flatten()[0]
        )
    except Exception:
        gp_metric["outputscale"] = np.nan

    # Extract lengthscales (for ARD kernels)
    try:
        if hasattr(model.covar_module, "base_kernel") and hasattr(
            model.covar_module.base_kernel, "lengthscale"
        ):
            lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu()
            for i, ls_val in enumerate(lengthscales.numpy().flatten()):
                gp_metric[f"cont_lengthscale_{i}"] = float(ls_val)
        elif hasattr(model.covar_module, "lengthscale"):
            lengthscale = model.covar_module.lengthscale.detach().cpu()
            if lengthscale.numel() == 1:
                gp_metric["cont_lengthscale_0"] = float(lengthscale.item())
            else:
                for i, ls_val in enumerate(lengthscale.numpy().flatten()):
                    gp_metric[f"cont_lengthscale_{i}"] = float(ls_val)
    except Exception as e:
        import logging

        logging.warning(f"Could not extract lengthscales: {e}")

    # Add training metadata
    gp_metric["num_epochs"] = num_epochs
    gp_metric["best_epoch"] = best_epoch if best_epoch is not None else num_epochs

    # Add y_train_mean and y_train_std if provided
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
            gp_metric["y_train_mean"] = float(
                y_train_mean.item() if hasattr(y_train_mean, "item") else y_train_mean
            )
            gp_metric["y_train_std"] = float(
                y_train_std.item() if hasattr(y_train_std, "item") else y_train_std
            )

    return gp_metric, y_pred_np, output_std_np
