"""Co-located trimodal synthetic data with unequal mode widths (A04)."""

from __future__ import annotations

import numpy as np
import torch

# Three linear modes with distinct slopes/intercepts and unequal noise σ.
TRIMODAL_M = (2.0, -0.5, 1.0)
TRIMODAL_C = (5.0, 12.0, -2.0)
TRIMODAL_SIGMA = (0.8, 2.0, 4.0)  # narrow / medium / wide
TRIMODAL_X_LO, TRIMODAL_X_HI = 0.0, 10.0
TRIMODAL_Y_MIN, TRIMODAL_Y_MAX = -15.0, 40.0


def trimodal_mode_means(
    x: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three true mode mean functions at ``x``."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().ravel()
    else:
        x_np = np.asarray(x, dtype=np.float64).ravel()
    y0 = TRIMODAL_M[0] * x_np + TRIMODAL_C[0]
    y1 = TRIMODAL_M[1] * x_np + TRIMODAL_C[1]
    y2 = TRIMODAL_M[2] * x_np + TRIMODAL_C[2]
    return y0, y1, y2


def generate_trimodal_data(
    n_samples: int = 400,
    seed: int = 42,
    *,
    x_lo: float = TRIMODAL_X_LO,
    x_hi: float = TRIMODAL_X_HI,
    weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Co-located trimodal toy: at each x, y is drawn from one of three Gaussians.

    Returns:
        X: shape ``[n, 1]``
        y: shape ``[n]``
        mode_idx: int array in {0,1,2}
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_lo, x_hi, n_samples)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mode_idx = rng.choice(3, size=n_samples, p=w)
    y = np.empty(n_samples, dtype=np.float64)
    for k in range(3):
        mask = mode_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        mu = TRIMODAL_M[k] * x[mask] + TRIMODAL_C[k]
        y[mask] = mu + rng.normal(0.0, TRIMODAL_SIGMA[k], size=n_k)
    return x.reshape(-1, 1).astype(np.float64), y, mode_idx.astype(np.int64)


def train_test_split_trimodal(
    n_samples: int = 400,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> dict[str, np.ndarray]:
    """Generate trimodal data and split into train/test."""
    from sklearn.model_selection import train_test_split

    X, y, mode_idx = generate_trimodal_data(n_samples=n_samples, seed=seed, **gen_kwargs)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X,
        y,
        mode_idx,
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


def make_trimodal_tensors(
    n_samples: int = 400,
    test_size: float = 0.3,
    seed: int = 42,
    **gen_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience: train/test tensors for MidlineSparse runners."""
    data = train_test_split_trimodal(
        n_samples=n_samples, test_size=test_size, seed=seed, **gen_kwargs
    )
    return (
        torch.as_tensor(data["X_train"], dtype=torch.float64),
        torch.as_tensor(data["y_train"], dtype=torch.float64),
        torch.as_tensor(data["X_test"], dtype=torch.float64),
        torch.as_tensor(data["y_test"], dtype=torch.float64),
    )
