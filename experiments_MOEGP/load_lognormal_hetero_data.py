"""A06: cubic + steep bump mean with strong additive centered lognormal noise.

y = f(x) + (κ |f(x)| + ε₀) (L − E[L]),  L ~ LogNormal(0, σ(x)²),
E[L] = exp(σ(x)² / 2),

f(x) = 0.1 x³ − x² + 3x − 9 + A exp(−(x−c)² / (2 w²)) on x ∈ [0, 10].
"""

from __future__ import annotations

import numpy as np
import torch

A06_X_LO, A06_X_HI = 0.0, 10.0
# Steep localized bump (near-discontinuous slopes around x=c).
A06_BUMP_A = 18.0
A06_BUMP_C = 5.0
A06_BUMP_W = 0.35
# Strong heteroscedastic log-noise scale.
A06_SIGMA_MIN = 0.35
A06_SIGMA_MAX = 0.95
A06_SIGMA_POWER = 1.0
A06_KAPPA = 0.65
A06_NOISE_FLOOR = 0.8  # ε₀ so noise does not vanish at roots of f
# Generous envelope for density / marginal panels.
A06_Y_MIN, A06_Y_MAX = -60.0, 80.0


def a06_mean(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """True mean: cubic trend + narrow Gaussian bump."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    cubic = 0.1 * x_np**3 - x_np**2 + 3.0 * x_np - 9.0
    bump = A06_BUMP_A * np.exp(-0.5 * ((x_np - A06_BUMP_C) / A06_BUMP_W) ** 2)
    return cubic + bump


def a06_sigma_log(
    x: np.ndarray | torch.Tensor,
    *,
    sigma_min: float = A06_SIGMA_MIN,
    sigma_max: float = A06_SIGMA_MAX,
    power: float = A06_SIGMA_POWER,
    x_lo: float = A06_X_LO,
    x_hi: float = A06_X_HI,
) -> np.ndarray:
    """Log-normal scale σ(x) for L ~ LogNormal(0, σ(x)²)."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    denom = max(x_hi - x_lo, 1e-12)
    w = np.clip((x_np - x_lo) / denom, 0.0, 1.0)
    return sigma_min + (sigma_max - sigma_min) * np.power(w, power)


def a06_noise(
    rng: np.random.Generator,
    x: np.ndarray,
    *,
    kappa: float = A06_KAPPA,
    noise_floor: float = A06_NOISE_FLOOR,
    sigma_min: float = A06_SIGMA_MIN,
    sigma_max: float = A06_SIGMA_MAX,
    sigma_power: float = A06_SIGMA_POWER,
    x_lo: float = A06_X_LO,
    x_hi: float = A06_X_HI,
) -> np.ndarray:
    """Additive mean-zero lognormal noise scaled by κ|f(x)| + ε₀."""
    x = np.asarray(x, dtype=np.float64).ravel()
    sigma = a06_sigma_log(
        x,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        power=sigma_power,
        x_lo=x_lo,
        x_hi=x_hi,
    )
    # L ~ LogNormal(0, σ²) ⇔ log L ~ N(0, σ²); E[L] = exp(σ²/2)
    log_l = rng.normal(0.0, sigma)
    l = np.exp(log_l)
    e_l = np.exp(0.5 * sigma**2)
    mu = a06_mean(x)
    scale = kappa * np.abs(mu) + float(noise_floor)
    return scale * (l - e_l)


def generate_lognormal_hetero_data(
    n_samples: int = 1800,
    seed: int = 42,
    *,
    x_lo: float = A06_X_LO,
    x_hi: float = A06_X_HI,
    kappa: float = A06_KAPPA,
    noise_floor: float = A06_NOISE_FLOOR,
    sigma_min: float = A06_SIGMA_MIN,
    sigma_max: float = A06_SIGMA_MAX,
    sigma_power: float = A06_SIGMA_POWER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic+bump mean + strong additive centered lognormal heteroscedastic noise.

    Returns:
        X: shape ``[n, 1]``
        y: shape ``[n]``
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_lo, x_hi, size=n_samples)
    noise = a06_noise(
        rng,
        x,
        kappa=kappa,
        noise_floor=noise_floor,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_power=sigma_power,
        x_lo=x_lo,
        x_hi=x_hi,
    )
    y = a06_mean(x) + noise
    return x.reshape(-1, 1).astype(np.float64), y.astype(np.float64)


def train_test_split_lognormal_hetero(
    n_samples: int = 1800,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> dict[str, np.ndarray]:
    """Generate A06 data and split into train/test (legacy fractional API)."""
    from sklearn.model_selection import train_test_split

    X, y = generate_lognormal_hetero_data(
        n_samples=n_samples, seed=seed, **gen_kwargs
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def make_lognormal_hetero_tensors(
    n_samples: int = 1800,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience: train/test tensors for MidlineSparse runners."""
    data = train_test_split_lognormal_hetero(
        n_samples=n_samples, test_size=test_size, seed=seed, **gen_kwargs
    )
    return (
        torch.as_tensor(data["X_train"], dtype=torch.float64),
        torch.as_tensor(data["y_train"], dtype=torch.float64),
        torch.as_tensor(data["X_test"], dtype=torch.float64),
        torch.as_tensor(data["y_test"], dtype=torch.float64),
    )
