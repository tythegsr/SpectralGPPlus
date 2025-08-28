# =============================================================================
# SHARED METRICS COMPUTATION FUNCTIONS
# =============================================================================
# This script contains the shared metrics computation functions used by
# GIT_O_M2AX2.py and GIT_O_DNS_ROM4.py to ensure consistency.
# =============================================================================

import numpy as np
import torch
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_hat, output_std=None, start_time=None):
    """
    Compute basic metrics for predictions.
    
    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional)
    
    Returns:
        dict: Dictionary with computed metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)
    
    # Only add time if start_time is provided
    if start_time is not None:
        metrics = { 
            'Time': time.time() - start_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
            # "MAE": mean_absolute_error(y_true, y_hat),
            # "R2": r2_score(y_true, y_hat)
        }
    else:
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
            # "MAE": mean_absolute_error(y_true, y_hat),
            # "R2": r2_score(y_true, y_hat)
        }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96 
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics
    
    return metrics


def compute_per_source_metrics(y_true, y_hat, output_std, X_test, source_columns, start_time=None):
    """
    Compute metrics for each source separately.
    
    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        X_test: Test features (2D array) containing source information
        source_columns: Either a single column index (int) or list of column indices for source identification
        start_time: Start time for timing (optional)
    
    Returns:
        dict: Dictionary with overall metrics and per-source metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)
    
    # Store the overall standard deviation for consistent normalization
    overall_std = y_true.std()
    
    # Handle source_columns parameter
    if isinstance(source_columns, int):
        # Single column case - filter by value in that column
        source_values = np.unique(X_test[:, source_columns])
        num_sources = len(source_values)
        source_indices = {}
        for i, val in enumerate(source_values):
            source_indices[f"source_{i}"] = (X_test[:, source_columns] == val)
    else:
        # Multiple columns case - one-hot encoded sources
        # Convert to list if it's a numpy array
        if isinstance(source_columns, np.ndarray):
            source_columns = source_columns.tolist()
        
        num_sources = len(source_columns)
        source_indices = {}
        for i in range(num_sources):
            source_indices[f"source_{i}"] = (X_test[:, source_columns[i]] == 1)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_hat, output_std, start_time)
    # Add sample size to overall metrics (as integer)
    overall_metrics["num_samples"] = int(len(y_true))
    
    # Compute per-source metrics
    per_source_metrics = {}
    for source_name, source_mask in source_indices.items():
        if np.sum(source_mask) > 0:  # Only compute if source has data
            source_y_true = y_true[source_mask]
            source_y_hat = y_hat[source_mask]
            source_output_std = output_std[source_mask] if output_std is not None else None
            
            # Compute source metrics with consistent normalization
            source_rmse = np.sqrt(mean_squared_error(source_y_true, source_y_hat))
            source_rrmse = source_rmse / overall_std  # Use overall std for consistency
            
            # Only add time if start_time is provided
            if start_time is not None:
                source_metrics = {
                    "Time": time.time() - start_time,
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            else:
                source_metrics = {
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            
            # Add NIS if output_std is provided
            if source_output_std is not None:
                z = 1.96 
                L = source_y_hat - z * source_output_std
                U = source_y_hat + z * source_output_std
                width = U - L
                below = (L - source_y_true) * (source_y_true < L)
                above = (source_y_true - U) * (source_y_true > U)
                interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
                source_nis = interval_score.mean() / source_y_true.std()  # Use per-source target std for NIS
                source_metrics["NIS"] = source_nis
            
            per_source_metrics[source_name] = source_metrics
    
    # Combine overall and per-source metrics
    all_metrics = {
        "overall": overall_metrics,
        "per_source": per_source_metrics,
        "num_sources": num_sources
    }
    
    return all_metrics
