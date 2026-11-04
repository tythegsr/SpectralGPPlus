"""ScoringBench-style 1D bimodal synthetic data (local copy for MOEGPwD)."""

from __future__ import annotations

import numpy as np
import torch


BIMODAL_M1, BIMODAL_C1 = 2.0, 5.0
BIMODAL_M2, BIMODAL_C2 = -1.0, 15.0
BIMODAL_NOISE_STD = 2.0
BIMODAL_X_LO, BIMODAL_X_HI = 0.0, 10.0


def bimodal_mode_means(x: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return the two true mode mean functions at ``x``."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    y1 = BIMODAL_M1 * x_np + BIMODAL_C1
    y2 = BIMODAL_M2 * x_np + BIMODAL_C2
    return y1, y2


def generate_bimodal_data(
    n_samples: int = 1000,
    seed: int = 42,
    *,
    noise_std: float = BIMODAL_NOISE_STD,
    x_lo: float = BIMODAL_X_LO,
    x_hi: float = BIMODAL_X_HI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ScoringBench bimodal toy data.

    Returns:
        X: shape ``[n, 1]``
        y: shape ``[n]``
        mode_mask: bool array, True iff sample came from mode 1
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_lo, x_hi, n_samples)
    y1 = BIMODAL_M1 * x + BIMODAL_C1 + rng.normal(0.0, noise_std, n_samples)
    y2 = BIMODAL_M2 * x + BIMODAL_C2 + rng.normal(0.0, noise_std, n_samples)
    mode_mask = rng.binomial(1, 0.5, n_samples) == 1
    y = np.where(mode_mask, y1, y2)
    return x.reshape(-1, 1).astype(np.float64), y.astype(np.float64), mode_mask


def train_test_split_bimodal(
    n_samples: int = 1000,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> dict[str, np.ndarray]:
    """Generate data and split into train/test."""
    from sklearn.model_selection import train_test_split

    X, y, mode_mask = generate_bimodal_data(n_samples=n_samples, seed=seed, **gen_kwargs)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X,
        y,
        mode_mask,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "mode_train": m_train,
        "mode_test": m_test,
    }
