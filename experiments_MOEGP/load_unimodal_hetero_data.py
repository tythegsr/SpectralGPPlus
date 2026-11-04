"""A05: unimodal mean with skewed contaminated heteroscedastic noise.

Zero-mean noise at each x (asymmetric Laplace core + rare wide Gaussian
contamination) so p(y|x) stays unimodal while shape, skew, and scale vary
across the domain. Richer than plain Student-t heteroscedasticity.
"""

from __future__ import annotations

import numpy as np
import torch

A05_SLOPE = 0.5
A05_INTERCEPT = 2.0
A05_X_LO, A05_X_HI = 0.0, 10.0
A05_SIGMA_MIN = 0.2
A05_SIGMA_MAX = 2.0
A05_SIGMA_POWER = 1.5
A05_TAU_MIN = 0.35  # P(positive side) at x_lo
A05_TAU_MAX = 0.75  # P(positive side) at x_hi
A05_EPS_CONTAM = 0.12
A05_KAPPA = 4.0
A05_Y_MIN, A05_Y_MAX = -12.0, 28.0


def a05_mean(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """True mean function μ(x) = slope·x + intercept."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    return A05_SLOPE * x_np + A05_INTERCEPT


def a05_noise_scale(
    x: np.ndarray | torch.Tensor,
    *,
    sigma_min: float = A05_SIGMA_MIN,
    sigma_max: float = A05_SIGMA_MAX,
    power: float = A05_SIGMA_POWER,
    x_lo: float = A05_X_LO,
    x_hi: float = A05_X_HI,
) -> np.ndarray:
    """Nonlinear heteroscedastic scale σ(x)."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    denom = max(x_hi - x_lo, 1e-12)
    w = np.clip((x_np - x_lo) / denom, 0.0, 1.0)
    return sigma_min + (sigma_max - sigma_min) * np.power(w, power)


def _a05_tau(
    x: np.ndarray,
    *,
    tau_min: float = A05_TAU_MIN,
    tau_max: float = A05_TAU_MAX,
    x_lo: float = A05_X_LO,
    x_hi: float = A05_X_HI,
) -> np.ndarray:
    denom = max(x_hi - x_lo, 1e-12)
    w = np.clip((x - x_lo) / denom, 0.0, 1.0)
    return tau_min + (tau_max - tau_min) * w


def _sample_asymmetric_laplace(
    rng: np.random.Generator,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Zero-mean asymmetric Laplace with scale ``sigma`` and positive-side prob ``tau``.

    Positive and negative exponential rates are chosen so Var ≈ σ² and the
    sign probability equals τ (skew grows with τ away from 0.5).
    """
    n = sigma.shape[0]
    # Side probability τ; scales b+/b- so mixture variance tracks σ².
    # For two-sided Exp with P(+)=τ, scale b+: Var = 2(τ b+² + (1-τ) b-²).
    # Set b+ = σ / sqrt(2τ), b- = σ / sqrt(2(1-τ)) when τ∈(0,1) so Var=σ²
    # under equal-moment construction when τ=0.5 (b+=b-=σ/√2 → Var=σ²).
    tau = np.clip(tau, 1e-3, 1.0 - 1e-3)
    b_pos = sigma / np.sqrt(2.0 * tau)
    b_neg = sigma / np.sqrt(2.0 * (1.0 - tau))
    pos = rng.random(n) < tau
    noise = np.empty(n, dtype=np.float64)
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos:
        noise[pos] = rng.exponential(b_pos[pos])
    if n_neg:
        noise[~pos] = -rng.exponential(b_neg[~pos])
    # Remove mean shift of asymmetric Laplace so noise is zero-mean.
    # E[ε] = τ b+ - (1-τ) b-
    mean_shift = tau * b_pos - (1.0 - tau) * b_neg
    return noise - mean_shift


def generate_unimodal_hetero_data(
    n_samples: int = 400,
    seed: int = 42,
    *,
    x_lo: float = A05_X_LO,
    x_hi: float = A05_X_HI,
    sigma_min: float = A05_SIGMA_MIN,
    sigma_max: float = A05_SIGMA_MAX,
    sigma_power: float = A05_SIGMA_POWER,
    tau_min: float = A05_TAU_MIN,
    tau_max: float = A05_TAU_MAX,
    eps_contam: float = A05_EPS_CONTAM,
    kappa: float = A05_KAPPA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unimodal linear mean + skewed contaminated heteroscedastic noise.

    Returns:
        X: shape ``[n, 1]``
        y: shape ``[n]``
        is_outlier: bool array (contamination draws)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_lo, x_hi, size=n_samples)
    sigma = a05_noise_scale(
        x,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        power=sigma_power,
        x_lo=x_lo,
        x_hi=x_hi,
    )
    tau = _a05_tau(x, tau_min=tau_min, tau_max=tau_max, x_lo=x_lo, x_hi=x_hi)
    is_outlier = rng.random(n_samples) < eps_contam
    noise = np.empty(n_samples, dtype=np.float64)
    core = ~is_outlier
    if core.any():
        noise[core] = _sample_asymmetric_laplace(rng, sigma[core], tau[core])
    if is_outlier.any():
        noise[is_outlier] = rng.normal(0.0, kappa * sigma[is_outlier])
    y = a05_mean(x) + noise
    return (
        x.reshape(-1, 1).astype(np.float64),
        y.astype(np.float64),
        is_outlier.astype(bool),
    )


def train_test_split_unimodal_hetero(
    n_samples: int = 400,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> dict[str, np.ndarray]:
    """Generate A05 data and split into train/test."""
    from sklearn.model_selection import train_test_split

    X, y, is_outlier = generate_unimodal_hetero_data(
        n_samples=n_samples, seed=seed, **gen_kwargs
    )
    X_train, X_test, y_train, y_test, o_train, o_test = train_test_split(
        X,
        y,
        is_outlier,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "outlier_train": o_train,
        "outlier_test": o_test,
    }


def make_unimodal_hetero_tensors(
    n_samples: int = 400,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience: train/test tensors for MidlineSparse runners."""
    data = train_test_split_unimodal_hetero(
        n_samples=n_samples, test_size=test_size, seed=seed, **gen_kwargs
    )
    return (
        torch.as_tensor(data["X_train"], dtype=torch.float64),
        torch.as_tensor(data["y_train"], dtype=torch.float64),
        torch.as_tensor(data["X_test"], dtype=torch.float64),
        torch.as_tensor(data["y_test"], dtype=torch.float64),
    )
