import logging
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
try:
    # Optional dependency used for exact CRPS computation (TabPFN bucket sampling).
    # If unavailable, we still support Gaussian closed-form CRPS via compute_crps_gaussian.
    from CRPS.CRPS import CRPS as pscore
except ModuleNotFoundError:  # pragma: no cover
    pscore = None

# from sklearn.metrics import mean_absolute_error, r2_score

# Use a non-interactive backend to avoid Tkinter dependency in non-main threads
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception as e:
    logging.warning(f"Failed to set matplotlib backend: {e}")


def compute_crps_gaussian(y_true, y_hat, output_std):
    """
    Compute Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
    
    Uses the closed-form analytical formula for Gaussian CRPS (exact, no sampling needed).
    
    For Gaussian predictions N(μ, σ²), the closed-form CRPS is:
    CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    where z = (y_true - μ) / σ, Φ is the standard normal CDF, φ is the standard normal PDF
    
    Args:
        y_true: True values (1D array)
        y_hat: Predicted means (1D array)
        output_std: Predicted standard deviations (1D array)
    
    Returns:
        float: Mean CRPS across all predictions
    """
    from scipy.stats import norm
    
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
    if isinstance(output_std, torch.Tensor):
        output_std = output_std.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_hat = np.asarray(y_hat).flatten()
    output_std = np.asarray(output_std).flatten()
    
    # Avoid division by zero
    output_std = np.maximum(output_std, 1e-10)
    
    # Standardized residuals
    z = (y_true - y_hat) / output_std
    
    # Standard normal CDF and PDF
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # CRPS formula for Gaussian distribution (closed-form, exact)
    crps = output_std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
    
    return crps.mean()


def compute_nlpd_gaussian(y_true, y_hat, output_std, eps: float = 1e-12):
    """
    Compute Gaussian negative log predictive density (NLPD) as mean over samples.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy()

    y_true = np.asarray(y_true).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)
    output_std = np.asarray(output_std).reshape(-1)

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_hat)
        & np.isfinite(output_std)
        & (output_std > 0.0)
    )
    if not np.any(valid_mask):
        return None

    y_true_valid = y_true[valid_mask]
    y_hat_valid = y_hat[valid_mask]
    sigma = np.maximum(output_std[valid_mask], eps)
    sigma2 = sigma * sigma

    nlpd = 0.5 * np.log(2.0 * np.pi * sigma2) + 0.5 * ((y_true_valid - y_hat_valid) ** 2 / sigma2)
    return float(np.mean(nlpd))


def logits_to_ensemble(logits, bar_dist, n_samples=1000):
    """
    Convert TabPFN logits to ensemble members by sampling from bar distribution.
    
    Args:
        logits: TabPFN logits (2D array), shape (n, n_buckets)
        bar_dist: BarDistribution object with sample() method
        n_samples: Number of samples per prediction
    
    Returns:
        np.ndarray: Ensemble members, shape (n, n_samples)
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu()
    else:
        logits = torch.tensor(logits, dtype=torch.float32)
    
    # Move bar_dist to CPU if needed
    if hasattr(bar_dist, 'borders') and bar_dist.borders.device.type == 'cuda':
        bar_dist = bar_dist.cpu()
    
    ensemble = []
    for i in range(len(logits)):
        samples = np.array([bar_dist.sample(logits[i].unsqueeze(0)).item() for _ in range(n_samples)])
        ensemble.append(samples)
    
    return np.array(ensemble)


def compute_nis(
    y_true,
    *,
    y_hat=None,
    output_std=None,
    lower=None,
    upper=None,
    alpha: float = 0.05,
    z: float = 1.96,
    normalize_by_y_std: bool = True,
    eps: float = 1e-12,
):
    """
    Compute Normalized Interval Score (NIS) for prediction intervals.

    You can provide either:
    - `y_hat` + `output_std` (assumes Gaussian; constructs a (1-alpha) interval), OR
    - explicit `lower` and `upper` bounds for the (1-alpha) interval.

    Interval score (Gneiting & Raftery) for central (1-alpha) prediction interval:
        IS = (U - L) + (2/alpha) * (L - y) * 1[y < L] + (2/alpha) * (y - U) * 1[y > U]

    This function returns the mean interval score normalized by std(y_true) by default.

    Returns:
        dict with keys: "NIS", "NIS_width", "NIS_outside"
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if y_hat is not None and isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy()
    if lower is not None and isinstance(lower, torch.Tensor):
        lower = lower.detach().cpu().numpy()
    if upper is not None and isinstance(upper, torch.Tensor):
        upper = upper.detach().cpu().numpy()

    y_true = np.asarray(y_true).reshape(-1)

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Construct bounds
    if lower is None or upper is None:
        if y_hat is None or output_std is None:
            raise ValueError("Provide either (lower, upper) or (y_hat, output_std).")

        y_hat = np.asarray(y_hat).reshape(-1)
        output_std = np.asarray(output_std).reshape(-1)

        # Avoid degenerate/negative std
        output_std = np.maximum(output_std, 0.0)

        lower = y_hat - z * output_std
        upper = y_hat + z * output_std
    else:
        lower = np.asarray(lower).reshape(-1)
        upper = np.asarray(upper).reshape(-1)

    if lower.shape != y_true.shape or upper.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, lower {lower.shape}, upper {upper.shape}"
        )

    width = upper - lower
    below = (lower - y_true) * (y_true < lower)
    above = (y_true - upper) * (y_true > upper)
    outside_penalty = (2.0 / alpha) * below + (2.0 / alpha) * above
    interval_score = width + outside_penalty

    denom = float(np.std(y_true))
    if not normalize_by_y_std:
        denom = 1.0
    denom = max(denom, eps)

    return {
        "NIS": float(np.mean(interval_score) / denom),
        "NIS_width": float(np.mean(width) / denom),
        "NIS_outside": float(np.mean(outside_penalty) / denom),
    }


def compute_lognormal_interval_bounds(
    log_mu,
    log_sigma,
    log_scale_C: float,
    *,
    alpha: float = 0.05,
):
    """
    Compute central prediction-interval bounds on original scale from
    Normal predictive parameters in log space.

    If log(y + C) ~ N(log_mu, log_sigma^2), then:
      L = exp(q_{alpha/2}) - C
      U = exp(q_{1-alpha/2}) - C
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    try:
        c_val = float(log_scale_C)
    except (TypeError, ValueError) as e:
        raise ValueError(f"log_scale_C must be a finite float, got {log_scale_C!r}.") from e
    if not np.isfinite(c_val):
        raise ValueError(f"log_scale_C must be finite, got {log_scale_C!r}.")

    if isinstance(log_mu, torch.Tensor):
        log_mu = log_mu.detach().cpu().numpy()
    if isinstance(log_sigma, torch.Tensor):
        log_sigma = log_sigma.detach().cpu().numpy()

    log_mu = np.asarray(log_mu).reshape(-1)
    log_sigma = np.asarray(log_sigma).reshape(-1)
    if log_mu.shape != log_sigma.shape:
        raise ValueError(f"Shape mismatch: log_mu {log_mu.shape}, log_sigma {log_sigma.shape}")

    # Predictive std should be nonnegative; guard against tiny negative numerical artifacts.
    log_sigma = np.maximum(log_sigma, 0.0)

    # Normal inverse-CDF values for alpha/2 and 1-alpha/2.
    dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    q_low = float(dist.icdf(torch.tensor(alpha / 2.0)).item())
    q_high = float(dist.icdf(torch.tensor(1.0 - alpha / 2.0)).item())

    lower_log = log_mu + q_low * log_sigma
    upper_log = log_mu + q_high * log_sigma

    # Clamp exponent arguments to avoid overflow.
    max_log_val = 700.0 if lower_log.dtype == np.float32 else 1000.0
    lower_log = np.clip(lower_log, -max_log_val, max_log_val)
    upper_log = np.clip(upper_log, -max_log_val, max_log_val)

    lower = np.exp(lower_log) - c_val
    upper = np.exp(upper_log) - c_val
    return lower, upper


def compute_nis_from_log_params(
    y_true,
    *,
    log_mu,
    log_sigma,
    log_scale_C: float,
    alpha: float = 0.05,
    normalize_by_y_std: bool = True,
    eps: float = 1e-12,
):
    """
    Compute NIS using 2.5/97.5% (or alpha-adjusted) quantiles implied by
    Normal predictive parameters in log space.
    """
    lower, upper = compute_lognormal_interval_bounds(
        log_mu,
        log_sigma,
        log_scale_C,
        alpha=alpha,
    )
    return compute_nis(
        y_true,
        lower=lower,
        upper=upper,
        alpha=alpha,
        normalize_by_y_std=normalize_by_y_std,
        eps=eps,
    )


def compute_metrics(
    y_true,
    y_hat,
    output_std=None,
    start_time=None,
    training_time=None,
    prediction_time=None,
    tabpfn_logits=None,
    tabpfn_bar_dist=None,
    y_test_normalized=None,
    lower_95=None,
    upper_95=None,
    log_mu=None,
    log_sigma=None,
    log_scale_C=None,
    use_log_quantile_nis: bool = False,
):
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)
        tabpfn_logits: TabPFN logits for exact CRPS (optional, shape: n_test x n_buckets)
        tabpfn_bar_dist: TabPFN BarDistribution object (required if tabpfn_logits provided)
        y_test_normalized: Normalized y_test for CRPS (logits are in normalized space!)
        lower_95/upper_95: Optional explicit interval bounds on original scale.
        log_mu/log_sigma/log_scale_C: Optional predictive Normal parameters in log(y+C)
            used to compute quantile-based NIS on original scale.
        use_log_quantile_nis: If True and log params are provided, prefer log-quantile NIS.

    Returns:
        dict: Dictionary with computed metrics including time information
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)
    if lower_95 is not None and isinstance(lower_95, torch.Tensor):
        lower_95 = lower_95.detach().cpu().numpy().reshape(-1)
    if upper_95 is not None and isinstance(upper_95, torch.Tensor):
        upper_95 = upper_95.detach().cpu().numpy().reshape(-1)
    if log_mu is not None and isinstance(log_mu, torch.Tensor):
        log_mu = log_mu.detach().cpu().numpy().reshape(-1)
    if log_sigma is not None and isinstance(log_sigma, torch.Tensor):
        log_sigma = log_sigma.detach().cpu().numpy().reshape(-1)

    # Handle time metrics
    if training_time is not None and prediction_time is not None:
        # New approach: separate training and prediction times
        total_time = training_time + prediction_time
        metrics = {
            "Total_Time": total_time,
            "Training_Time": training_time,
            "Prediction_Time": prediction_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
            "MAE": np.mean(np.abs(y_true - y_hat)),
        }
    elif start_time is not None:
        # Legacy approach: single start_time
        elapsed_time = time.time() - start_time
        metrics = {
            "Time": elapsed_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
            "MAE": np.mean(np.abs(y_true - y_hat)),
        }
    else:
        # No time information
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
            "MAE": np.mean(np.abs(y_true - y_hat)),
        }

    # Add NIS (and CRPS, if std available) when we have any uncertainty information
    has_bounds = (lower_95 is not None) and (upper_95 is not None)
    has_log_params = (
        use_log_quantile_nis
        and (log_mu is not None)
        and (log_sigma is not None)
        and (log_scale_C is not None)
    )
    if (output_std is not None) or has_bounds or has_log_params:
        if has_log_params:
            nis_metrics = compute_nis_from_log_params(
                y_true,
                log_mu=log_mu,
                log_sigma=log_sigma,
                log_scale_C=float(log_scale_C),
                alpha=0.05,
            )
        else:
            # If bounds are provided, compute_nis will use them; otherwise it falls back to std.
            nis_metrics = compute_nis(
                y_true,
                y_hat=y_hat,
                output_std=output_std,
                lower=lower_95,
                upper=upper_95,
                alpha=0.05,
            )
        metrics.update(nis_metrics)
        
        # CRPS (Continuous Ranked Probability Score)
        # - Exact (bucket-sampling) CRPS requires optional CRPS package + TabPFN logits/bar_dist
        # - Gaussian CRPS is always computed (closed form) when output_std is provided
        if (
            (pscore is not None)
            and (output_std is not None)
            and (tabpfn_logits is not None)
            and (tabpfn_bar_dist is not None)
        ):
            # Exact CRPS using bar distribution buckets
            # Note: logits and bar_dist are in normalized space, need to transform back
            
            # Get logits and probabilities
            if isinstance(tabpfn_logits, torch.Tensor):
                logits_torch = tabpfn_logits.cpu()
            else:
                logits_torch = torch.tensor(tabpfn_logits, dtype=torch.float32)
            
            probs = torch.softmax(logits_torch, dim=-1).numpy()
            borders = tabpfn_bar_dist.borders.cpu().numpy()
            bucket_centers = (borders[:-1] + borders[1:]) / 2
            
            # Transform bucket centers back to original scale
            # If y_test_normalized is provided, we need to denormalize
            if y_test_normalized is not None:
                # Infer normalization parameters from y_true and y_test_normalized
                # y_normalized = (y - mean) / std, so y = y_normalized * std + mean
                y_mean = y_true.mean()
                y_std = y_true.std()
                bucket_centers_original = bucket_centers * y_std + y_mean
            else:
                bucket_centers_original = bucket_centers
            
            # Use all bins (no subsampling)
            n_bins = len(bucket_centers_original)
            
            # Compute CRPS using weighted buckets
            crps_values = []
            fcrps_values = []
            n_samples = 1000
            for i in range(len(y_true)):
                # Sample from buckets according to probabilities
                ensemble = np.random.choice(bucket_centers_original, size=n_samples, p=probs[i])
                crps_result = pscore(ensemble, y_true[i]).compute()
                crps_values.append(crps_result[0])
                fcrps_values.append(crps_result[1])
            
            crps_exact = np.mean(crps_values)
            fcrps_exact = np.mean(fcrps_values)
            metrics["CRPS_exact"] = crps_exact
            metrics["NCRPS_exact"] = crps_exact / y_true.std()
            metrics["fCRPS_exact"] = fcrps_exact
            metrics["NfCRPS_exact"] = fcrps_exact / y_true.std()
            print(f"[CRPS] Using exact CRPS with {n_bins} bins, {n_samples} samples (denormalized)")
        elif (tabpfn_logits is not None) and (tabpfn_bar_dist is not None) and (pscore is None):
            logging.warning(
                "CRPS package not installed; skipping exact CRPS (CRPS_exact/NCRPS_exact). "
                "Install CRPS or set tabpfn_logits=None."
            )
        
        if output_std is not None:
            # Always compute Gaussian CRPS for comparison
            crps = compute_crps_gaussian(y_true, y_hat, output_std)
            metrics["CRPS"] = crps
            # Normalized CRPS (similar to RRMSE normalization)
            metrics["NCRPS"] = crps / y_true.std()

            # Gaussian negative log predictive density (NLPD)
            nlpd = compute_nlpd_gaussian(y_true, y_hat, output_std)
            if nlpd is not None:
                metrics["NLPD"] = nlpd
        
        return metrics

    return metrics


def adjust_predictive_variance_for_test_noise(output_std, test_noise_std):
    """
    Adjust predictive standard deviation to account for additional test noise.

    When test data has noise added that is not accounted for in the model's
    predictive variance, this function adds the test noise variance to the
    predictive variance to get the total uncertainty.

    Args:
        output_std: Predictive standard deviation from the model (includes training noise)
        test_noise_std: Standard deviation of the noise added to test targets

    Returns:
        Adjusted standard deviation: sqrt(predictive_variance + test_noise_variance)

    Example:
        If model predicts with std=0.1 and test noise has std=0.05:
        >>> adjusted_std = adjust_predictive_variance_for_test_noise(0.1, 0.05)
        >>> # adjusted_std = sqrt(0.1^2 + 0.05^2) = sqrt(0.01 + 0.0025) ≈ 0.112

    Note:
        This is useful when evaluating with noisy test data. The model's predictive
        variance includes training noise, but not test noise. Adding test noise variance
        gives the total uncertainty for comparing against noisy test targets.
    """
    if isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy()
    if isinstance(test_noise_std, torch.Tensor):
        test_noise_std = test_noise_std.detach().cpu().numpy()

    # Convert to numpy arrays if needed
    output_std = np.asarray(output_std)
    test_noise_std = np.asarray(test_noise_std)

    # Add variances (var = std^2)
    adjusted_variance = output_std**2 + test_noise_std**2
    adjusted_std = np.sqrt(adjusted_variance)

    return adjusted_std


def format_metric_value(key: str, value: float, precision: int = 4) -> str:
    """
    Format a metric value appropriately based on its key.

    Args:
        key: The metric key (e.g., 'jitter', 'noise', 'RRMSE')
        value: The value to format
        precision: Number of decimal places (for non-scientific notation)

    Returns:
        Formatted string representation of the value
    """
    if key in ["jitter", "jitter_max", "noise", "noise_std"]:
        # Use scientific notation for jitter / jitter_max / noise / noise_std
        return f"{value:.6e}"
    elif key in ["num_epochs", "best_epoch"]:
        # Integer values
        return f"{int(value)}"
    else:
        # Default formatting
        return f"{value:.{precision}f}"


def analyze_metrics(metrics_list, print_summary: bool = False, label: str = None, title: str = None):
    """
    Summarize metrics across seeds for all available metrics (RRMSE, NIS, CRPS, etc.), 
    including per-source statistics.

    Args:
        metrics_list: list of dicts, each containing metric values for a seed

    Returns:
        dict with per-metric summary: {metric: {mean, std, median, min, max}}
        and per-source summaries for RRMSE, NIS, CRPS, etc.
    """
    import numpy as np
    import pandas as pd

    if metrics_list is None or len(metrics_list) == 0:
        return {}

    df = pd.DataFrame(metrics_list)

    # Mean/std for all available metric columns
    # Detailed stats for all metrics in the DataFrame
    detailed = {}
    for m in df.columns:
        vals = df[m].dropna().values
        if len(vals) == 0:
            continue
        # Try to convert to float, skip if not numeric
        try:
            vals = vals.astype(float)
        except (ValueError, TypeError):
            continue
        detailed[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": int(len(vals)),
        }

    # Handle individual lengthscale metrics (lengthscale_0, lengthscale_1, etc.)
    lengthscale_columns = [col for col in df.columns if col.startswith("lengthscale_")]
    for lengthscale_col in sorted(lengthscale_columns):  # Sort to ensure consistent ordering
        vals = df[lengthscale_col].dropna().values
        if len(vals) == 0:
            continue
        vals = vals.astype(float)
        detailed[lengthscale_col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": int(len(vals)),
        }

    # Extract per-source metrics for RRMSE, NIS, CRPS, etc.
    per_source_stats = {}
    source_columns = [col for col in df.columns if col.startswith("source_") and
                     ("_RRMSE" in col or "_NIS" in col or "_CRPS" in col or "_NCRPS" in col or "_NLPD" in col)]

    if source_columns:
        # Group by source
        sources = {}
        for col in source_columns:
            source_name = col.split("_")[0] + "_" + col.split("_")[1]  # e.g., 'source_0'
            metric_name = col.split("_", 2)[2]  # e.g., 'RRMSE' or 'NIS'

            if source_name not in sources:
                sources[source_name] = {}
            sources[source_name][metric_name] = col

        # Compute statistics for each source
        for source_name, metrics in sources.items():
            per_source_stats[source_name] = {}
            for metric_name, col_name in metrics.items():
                if col_name in df.columns:
                    vals = df[col_name].dropna().values
                    if len(vals) > 0:
                        vals = vals.astype(float)
                        per_source_stats[source_name][metric_name] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "median": float(np.median(vals)),
                            "min": float(np.min(vals)),
                            "max": float(np.max(vals)),
                            "count": int(len(vals)),
                        }

    if print_summary and len(detailed) > 0:
        label_print = label or "Summary"

        if title:
            print(f"\n{label_print} over {len(metrics_list)} seeds for {title}:")
        else:
            print(f"\n{label_print} over {len(metrics_list)} seeds:")

        for m, s in detailed.items():
            # Format output based on metric type
            if m.startswith("lengthscale_"):
                # For individual lengthscales
                print(f"  {m}: median={s['median']:.6f} | min={s['min']:.6f} | max={s['max']:.6f} | mean={s['mean']:.6f} ± {s['std']:.6f} (n={s['count']})")
            elif m in ["num_epochs", "best_epoch"]:
                # For integer metrics, show as integers
                print(f"  {m}: median={s['median']:.0f} | min={s['min']:.0f} | max={s['max']:.0f} | mean={s['mean']:.1f} ± {s['std']:.1f} (n={s['count']})")
            elif m in ["jitter", "jitter_max", "noise", "noise_std"]:
                # Use scientific notation for jitter / jitter_max / noise / noise_std
                print(f"  {m}: median={s['median']:.6e} | min={s['min']:.6e} | max={s['max']:.6e} | mean={s['mean']:.6e} ± {s['std']:.6e} (n={s['count']})")
            else:
                print(f"  {m}: median={s['median']:.6f} | min={s['min']:.6f} | max={s['max']:.6f} | mean={s['mean']:.6f} ± {s['std']:.6f} (n={s['count']})")

        # Print per-source statistics
        if per_source_stats:
            print(f"\n{label_print} Per-Source Statistics:")
            for source_name, source_metrics in per_source_stats.items():
                print(f"  {source_name}:")
                for metric_name, stats in source_metrics.items():
                    print(
                        f"    {metric_name}: median={stats['median']:.6f} | "
                        f"min={stats['min']:.6f} | max={stats['max']:.6f} | "
                        f"mean={stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})"
                    )

    # Add per-source stats to the return value
    if per_source_stats:
        detailed["per_source"] = per_source_stats

    return detailed


def plot_metrics(*args, labels: list = None, title: str = None, save_path: str = None, subplots: bool = True):
    """
    Plot per-seed metric VALUES (not aggregates) for multiple runs as violin plots.

    Args:
        metrics_lists: list of lists, where each inner list is a metrics_list
                       (the same structure you pass to analyze_metrics), i.e.,
                       a list of dicts with keys like 'RRMSE', 'NIS', 'Time'.
        labels: optional list of names for each metrics_list; defaults to
                ["run_0", ...].
        subplots: if True (default), returns both individual plots AND combined plots.
                 if False, returns only individual plots.

    Returns:
        dict: Dictionary containing 'individual' and optionally 'combined' figure objects.
              - 'individual': dict with 'RRMSE' and 'NIS' figure objects
              - 'combined': figure object with both metrics in subplots (only when subplots=True)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Debug: print save_path if provided
    if save_path is not None:
        print(f"[DEBUG plot_metrics] save_path provided: {save_path}")
    else:
        print("[DEBUG plot_metrics] save_path is None - plots will not be saved")

    # Normalize inputs: allow plot_metric_values(run1, run2, ...) or plot_metric_values([run1, run2, ...])
    if len(args) == 1 and isinstance(args[0], list) and (len(args[0]) == 0 or isinstance(args[0][0], (dict, list))):
        metrics_lists = args[0]
    else:
        metrics_lists = list(args)

    if labels is None:
        labels = [f"run_{i}" for i in range(len(metrics_lists))]

    def extract(vals_list, key):
        out = []
        for ml in metrics_lists:
            # ml should be list[dict]
            if not isinstance(ml, list):
                out.append(np.array([], dtype=float))
                continue
            arr = [d[key] for d in ml if isinstance(d, dict) and key in d and d[key] is not None]
            out.append(np.array(arr, dtype=float) if len(arr) > 0 else np.array([], dtype=float))
        return out

    # Determine a representative seed count (use min across lists to be safe)
    seed_counts = [len(ml) if isinstance(ml, list) else 0 for ml in metrics_lists]
    n_seeds = min(seed_counts) if len(seed_counts) > 0 else 0

    def create_violin_plot(ax, data, metric, labels, n_seeds):
        """Helper function to create a violin plot on the given axis."""
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#888888")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        # Overlay mean (blue) and median (red) lines
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(mean_v, i - 0.25, i + 0.25, colors="blue", linewidth=2)
            ax.hlines(med_v, i - 0.25, i + 0.25, colors="red", linewidth=2)

        # Legend: blue = mean, red = median
        try:
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], color="blue", lw=2, label="Mean"),
                Line2D([0], [0], color="red", lw=2, label="Median"),
            ]
            ax.legend(handles=legend_handles, loc="upper right", frameon=False)
        except Exception as e:
            logging.warning(f"Failed to add legend to plot: {e}")

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} distribution (n={n_seeds})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    def save_figure(fig, metric_name, save_path, title):
        """Helper function to save a figure."""
        if save_path is not None:
            from pathlib import Path

            p = Path(save_path)
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[WARNING] Failed to create directory {save_path}: {e}")
                return
            fname = f"{metric_name}" + (f"_{title}" if title else "") + ".png"
            full_path = p / fname
            try:
                fig.savefig(str(full_path), dpi=300, bbox_inches="tight")
                print(f"[INFO] Saved plot to: {full_path}")
                # Close only when we actually save, to avoid leaking figures
                try:
                    plt.close(fig)
                except Exception as e:
                    logging.warning(f"Failed to close figure: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to save plot to {full_path}: {e}")
                import traceback

                traceback.print_exc()

    # Always create individual plots
    individual_figs = {}
    for metric in ["RRMSE", "NIS", "NCRPS"]:
        # Special handling for NCRPS: show TabPFN NfCRPS_exact, TabPFN NCRPS, and GP NCRPS
        if metric == "NCRPS":
            crps_data = []
            crps_labels = []
            
            # Extract TabPFN NfCRPS_exact (from first metrics list) - first
            if len(metrics_lists) > 0:
                tabpfn_nfcrps_exact = extract([metrics_lists[0]], "NfCRPS_exact")[0]
                if len(tabpfn_nfcrps_exact) > 0:
                    crps_data.append(tabpfn_nfcrps_exact)
                    crps_labels.append("TabPFN NfCRPS_exact")
                
                # Extract TabPFN NCRPS (from first metrics list) - second
                tabpfn_ncrps = extract([metrics_lists[0]], "NCRPS")[0]
                if len(tabpfn_ncrps) > 0:
                    crps_data.append(tabpfn_ncrps)
                    crps_labels.append("TabPFN NCRPS")
            
            # Extract GP NCRPS (from second metrics list) - third
            if len(metrics_lists) > 1:
                gp_ncrps = extract([metrics_lists[1]], "NCRPS")[0]
                if len(gp_ncrps) > 0:
                    crps_data.append(gp_ncrps)
                    crps_labels.append("GP NCRPS")
            
            if len(crps_data) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                create_violin_plot(ax, crps_data, metric, crps_labels, n_seeds)
                
                if title:
                    try:
                        fig.suptitle(title)
                    except Exception as e:
                        logging.warning(f"Failed to set figure title: {e}")
                
                plt.tight_layout()
                save_figure(fig, metric.lower(), save_path, title)
                individual_figs[metric] = fig
        else:
            # Regular metrics (RRMSE, NIS)
            data = extract(metrics_lists, metric)
            # Skip if no data for this metric
            if all(len(d) == 0 for d in data):
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            create_violin_plot(ax, data, metric, labels, n_seeds)

            if title:
                try:
                    fig.suptitle(title)
                except Exception as e:
                    logging.warning(f"Failed to set figure title: {e}")

            plt.tight_layout()
            save_figure(fig, metric.lower(), save_path, title)
            individual_figs[metric] = fig

    result = {"individual": individual_figs}

    # Create combined plot if subplots=True
    if subplots:
        # Create one figure with three subplots
        combined_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 4))

        # Plot RRMSE
        rrmse_data = extract(metrics_lists, "RRMSE")
        create_violin_plot(ax1, rrmse_data, "RRMSE", labels, n_seeds)

        # Plot NIS
        nis_data = extract(metrics_lists, "NIS")
        create_violin_plot(ax2, nis_data, "NIS", labels, n_seeds)

        # Plot NCRPS: show TabPFN NfCRPS_exact, TabPFN NCRPS, and GP NCRPS
        crps_data = []
        crps_labels = []
        
        # Extract TabPFN NfCRPS_exact (from first metrics list) - first
        if len(metrics_lists) > 0:
            tabpfn_nfcrps_exact = extract([metrics_lists[0]], "NfCRPS_exact")[0]
            if len(tabpfn_nfcrps_exact) > 0:
                crps_data.append(tabpfn_nfcrps_exact)
                crps_labels.append("TabPFN NfCRPS_exact")
            
            # Extract TabPFN NCRPS (from first metrics list) - second
            tabpfn_ncrps = extract([metrics_lists[0]], "NCRPS")[0]
            if len(tabpfn_ncrps) > 0:
                crps_data.append(tabpfn_ncrps)
                crps_labels.append("TabPFN NCRPS")
        
        # Extract GP NCRPS (from second metrics list) - third
        if len(metrics_lists) > 1:
            gp_ncrps = extract([metrics_lists[1]], "NCRPS")[0]
            if len(gp_ncrps) > 0:
                crps_data.append(gp_ncrps)
                crps_labels.append("GP NCRPS")
        
        if len(crps_data) > 0:
            create_violin_plot(ax3, crps_data, "NCRPS", crps_labels, n_seeds)
        else:
            # Hide the third subplot if no NCRPS data
            ax3.set_visible(False)

        # Set overall title if provided
        if title:
            try:
                combined_fig.suptitle(title)
            except Exception as e:
                logging.warning(f"Failed to set combined figure title: {e}")

        plt.tight_layout()
        save_figure(combined_fig, "metrics_combined", save_path, title)
        result["combined"] = combined_fig

    return result


def compute_per_source_metrics(
    y_true, y_hat, output_std, X_test, source_columns, start_time=None, training_time=None, prediction_time=None
):
    """
    Compute metrics for each source separately.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        X_test: Test features (2D array) containing source information
        source_columns: Either a single column index (int) or list of column indices for source identification
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)

    Returns:
        dict: Dictionary with overall metrics and per-source metrics including time information
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

    # Handle source_columns parameter
    if isinstance(source_columns, int):
        # Single column case - filter by value in that column
        source_values = np.unique(X_test[:, source_columns])
        num_sources = len(source_values)
        source_indices = {}
        for i, val in enumerate(source_values):
            source_indices[f"source_{i}"] = X_test[:, source_columns] == val
    else:
        # Multiple columns case - one-hot encoded sources
        # Convert to list if it's a numpy array
        if isinstance(source_columns, np.ndarray):
            source_columns = source_columns.tolist()

        num_sources = len(source_columns)
        source_indices = {}
        for i in range(num_sources):
            source_indices[f"source_{i}"] = X_test[:, source_columns[i]] == 1

    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_hat, output_std, start_time, training_time, prediction_time)
    # Add sample size to overall metrics (as integer)
    overall_metrics["num_samples"] = int(len(y_true))

    # Compute per-source metrics
    per_source_metrics = {}
    for source_name, source_mask in source_indices.items():
        if np.sum(source_mask) > 0:  # Only compute if source has data
            source_y_true = y_true[source_mask]
            source_y_hat = y_hat[source_mask]
            source_output_std = output_std[source_mask] if output_std is not None else None

            # Compute source metrics with source-specific normalization
            source_rmse = np.sqrt(mean_squared_error(source_y_true, source_y_hat))
            source_std = source_y_true.std()
            source_rrmse = source_rmse / source_std if source_std > 0 else np.inf

            # Per-source metrics (no time metrics - times are the same for all sources)
            source_metrics = {
                "RRMSE": source_rrmse,
                "RMSE": source_rmse,
                "MSE": mean_squared_error(source_y_true, source_y_hat),
                "num_samples": int(len(source_y_true)),  # Number of predictions for this source
            }

            # Add NIS and CRPS if output_std is provided
            if source_output_std is not None:
                nis_metrics = compute_nis(
                    source_y_true,
                    y_hat=source_y_hat,
                    output_std=source_output_std,
                    alpha=0.05,
                )
                source_metrics.update(nis_metrics)
                
                # CRPS
                source_crps = compute_crps_gaussian(source_y_true, source_y_hat, source_output_std)
                source_metrics["CRPS"] = source_crps
                source_metrics["NCRPS"] = source_crps / source_y_true.std()
                source_nlpd = compute_nlpd_gaussian(source_y_true, source_y_hat, source_output_std)
                if source_nlpd is not None:
                    source_metrics["NLPD"] = source_nlpd

            per_source_metrics[source_name] = source_metrics

    # Combine overall and per-source metrics
    all_metrics = {"overall": overall_metrics, "per_source": per_source_metrics, "num_sources": num_sources}

    return all_metrics


def extract_parameter_statistics(gp_parameters_file="gp_parameters.json"):
    """
    Extract parameter statistics from the gp_parameters.json file.

    Args:
        gp_parameters_file: Path to the gp_parameters.json file

    Returns:
        dict: Parameter statistics including initial, final, and deltas for each parameter
    """
    import json
    from pathlib import Path

    import numpy as np

    try:
        # Read the gp_parameters.json file
        param_file = Path(gp_parameters_file)
        if not param_file.exists():
            return {"error": f"Parameter file {gp_parameters_file} not found"}

        with open(param_file, "r") as f:
            parameters_data = json.load(f)

        if not parameters_data:
            return {"error": "No parameter data found"}

        # Extract parameter statistics
        param_stats = {
            "raw_noise": {"initial": [], "final": [], "deltas": []},
            "raw_outputscale": {"initial": [], "final": [], "deltas": []},
            "raw_lengthscales": {"initial": [], "final": [], "deltas": []},
        }

        # Collect all parameter values across runs
        for run_data in parameters_data:
            for param_name in ["raw_noise", "raw_outputscale", "raw_lengthscales"]:
                if param_name in run_data.get("initial", {}):
                    initial_val = run_data["initial"][param_name]
                    final_val = run_data["final"][param_name]
                    delta_val = run_data["deltas"][param_name]

                    # Handle different data types
                    if param_name == "raw_lengthscales":
                        # For lengthscales, store as lists
                        param_stats[param_name]["initial"].append(initial_val if initial_val is not None else [])
                        param_stats[param_name]["final"].append(final_val if final_val is not None else [])
                        param_stats[param_name]["deltas"].append(delta_val if delta_val is not None else [])
                    else:
                        # For scalar parameters
                        param_stats[param_name]["initial"].append(initial_val if initial_val is not None else 0.0)
                        param_stats[param_name]["final"].append(final_val if final_val is not None else 0.0)
                        param_stats[param_name]["deltas"].append(delta_val if delta_val is not None else 0.0)

        # Compute summary statistics for each parameter
        summary_stats = {}
        for param_name, param_data in param_stats.items():
            summary_stats[param_name] = {}

            for stat_type in ["initial", "final", "deltas"]:
                values = param_data[stat_type]

                if param_name == "raw_lengthscales":
                    # For lengthscales, compute stats across all dimensions
                    if values and len(values) > 0 and len(values[0]) > 0:
                        # Flatten all lengthscale values
                        flat_values = []
                        for val_list in values:
                            if val_list:  # Check if not empty
                                flat_values.extend(val_list)

                        if flat_values:
                            summary_stats[param_name][stat_type] = {
                                "mean": float(np.mean(flat_values)),
                                "std": float(np.std(flat_values, ddof=1)) if len(flat_values) > 1 else 0.0,
                                "median": float(np.median(flat_values)),
                                "min": float(np.min(flat_values)),
                                "max": float(np.max(flat_values)),
                                "count": len(flat_values),
                                "raw_values": values,  # Keep raw values for reference
                            }
                        else:
                            summary_stats[param_name][stat_type] = {
                                "mean": 0.0,
                                "std": 0.0,
                                "median": 0.0,
                                "min": 0.0,
                                "max": 0.0,
                                "count": 0,
                                "raw_values": values,
                            }
                    else:
                        summary_stats[param_name][stat_type] = {
                            "mean": 0.0,
                            "std": 0.0,
                            "median": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "count": 0,
                            "raw_values": values,
                        }
                else:
                    # For scalar parameters
                    if values:
                        values_array = np.array(values)
                        summary_stats[param_name][stat_type] = {
                            "mean": float(np.mean(values_array)),
                            "std": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                            "median": float(np.median(values_array)),
                            "min": float(np.min(values_array)),
                            "max": float(np.max(values_array)),
                            "count": len(values_array),
                            "raw_values": values,
                        }
                    else:
                        summary_stats[param_name][stat_type] = {
                            "mean": 0.0,
                            "std": 0.0,
                            "median": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "count": 0,
                            "raw_values": values,
                        }

        # Add metadata
        summary_stats["metadata"] = {
            "total_runs": len(parameters_data),
            "parameter_file": str(param_file),
            "kernel_types": list(
                set([run.get("initial", {}).get("kernel_type", "Unknown") for run in parameters_data])
            ),
            "input_dims": list(set([run.get("initial", {}).get("input_dim", "Unknown") for run in parameters_data])),
        }

        return summary_stats

    except Exception as e:
        return {"error": f"Failed to extract parameter statistics: {str(e)}"}
