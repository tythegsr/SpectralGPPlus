"""Data loaders for RFFMTGPR experiments."""

from __future__ import annotations

import os

import numpy as np
import torch


def load_toa_data(
    n_train: int,
    n_test: int,
    n_val: int = 0,
    seed: int = 42,
    data_path: str | None = None,
) -> tuple:
    """
    Load TOA flattened data and return train/val/test splits.

    Dataset keys: X (n, 285), y_cos (n,), y_grain (n,).
    Targets are stacked as y with shape (n, 2).

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        n_val: Number of validation samples (optional).
        seed: Random seed for shuffled split.
        data_path: Path to toa_data_flattened.npz (defaults to repo root).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
        (val tensors/indices are empty when ``n_val == 0``).
        Indices are global positions in the dataset permutation (int64 tensors).
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "toa_data_flattened.npz")
    data_path = os.path.abspath(data_path)

    data = np.load(data_path)
    X = torch.tensor(data["X"], dtype=torch.float64)
    y_cos = torch.tensor(data["y_cos"], dtype=torch.float64)
    y_grain = torch.tensor(data["y_grain"], dtype=torch.float64)
    y = torch.stack([y_cos, y_grain], dim=1)

    n_total = X.shape[0]
    needed = n_train + n_test + n_val
    if needed > n_total:
        raise ValueError(
            f"Requested n_train={n_train} + n_test={n_test} + n_val={n_val} "
            f"exceeds dataset size {n_total}"
        )

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val : n_test + n_val + n_train]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    if n_val > 0:
        X_val = X[val_idx]
        y_val = y[val_idx]
    else:
        X_val = X.new_zeros((0, X.shape[-1]))
        y_val = y.new_zeros((0, y.shape[-1]))
        val_idx = perm.new_zeros((0,), dtype=torch.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
