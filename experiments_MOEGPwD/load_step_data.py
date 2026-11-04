"""1D piecewise-constant step toy for sharp-edge MoE experiments."""

from __future__ import annotations

import numpy as np
import torch


DEFAULT_STEP_VALUES = (-0.4, -0.2, 0.1, 0.4)
DEFAULT_X_BOUNDS = (-0.5, 0.5)


def step_function(
    x: np.ndarray | torch.Tensor,
    *,
    x_bounds: tuple[float, float] = DEFAULT_X_BOUNDS,
    step_values: tuple[float, ...] = DEFAULT_STEP_VALUES,
) -> np.ndarray:
    """Piecewise-constant levels on equal-width bins along x."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    l_bound, u_bound = x_bounds
    n_steps = len(step_values)
    width = (u_bound - l_bound) / n_steps
    bin_idx = np.floor((x_np - l_bound) / width).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_steps - 1)
    levels = np.asarray(step_values, dtype=np.float64)
    return levels[bin_idx]


def generate_step_data(
    n_train: int = 200,
    n_test: int = 500,
    seed: int = 42,
    *,
    x_bounds: tuple[float, float] = DEFAULT_X_BOUNDS,
    step_values: tuple[float, ...] = DEFAULT_STEP_VALUES,
    noise_train: float = 0.02,
    noise_test: float = 0.02,
) -> dict[str, np.ndarray]:
    """Sobol-like uniform samples with additive Gaussian noise; true bin labels."""
    rng = np.random.default_rng(seed)
    l_bound, u_bound = x_bounds
    x_train = rng.uniform(l_bound, u_bound, size=n_train)
    x_test = rng.uniform(l_bound, u_bound, size=n_test)

    y_train_true = step_function(x_train, x_bounds=x_bounds, step_values=step_values)
    y_test_true = step_function(x_test, x_bounds=x_bounds, step_values=step_values)

    y_train = y_train_true + rng.normal(0.0, noise_train, size=n_train)
    y_test = y_test_true + rng.normal(0.0, noise_test, size=n_test)

    n_steps = len(step_values)
    width = (u_bound - l_bound) / n_steps
    bin_train = np.clip(np.floor((x_train - l_bound) / width).astype(np.int64), 0, n_steps - 1)
    bin_test = np.clip(np.floor((x_test - l_bound) / width).astype(np.int64), 0, n_steps - 1)

    return {
        "X_train": x_train.reshape(-1, 1).astype(np.float64),
        "X_test": x_test.reshape(-1, 1).astype(np.float64),
        "y_train": y_train.astype(np.float64),
        "y_test": y_test.astype(np.float64),
        "y_train_true": y_train_true.astype(np.float64),
        "y_test_true": y_test_true.astype(np.float64),
        "bin_train": bin_train,
        "bin_test": bin_test,
        "step_values": np.asarray(step_values, dtype=np.float64),
        "x_bounds": np.asarray(x_bounds, dtype=np.float64),
    }
