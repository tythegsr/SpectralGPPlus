"""Single source of truth for TOA dataset loading and fixed train/val/test pools."""

from __future__ import annotations

import os

import numpy as np
import torch

# Fixed restricted pools for TOA splits (seed-based permutation).
# Smaller n_train / n_val / n_test take prefixes of these pools so val/test
# stay identical when training size is reduced.
TOA_TRAIN_POOL_SIZE = 49000
TOA_VAL_POOL_SIZE = 4900
TOA_TEST_POOL_SIZE = 5000


def load_toa_data(
    n_train: int,
    n_test: int = TOA_TEST_POOL_SIZE,
    n_val: int = 0,
    seed: int = 42,
    data_path: str | None = None,
    train_pool_size: int = TOA_TRAIN_POOL_SIZE,
    val_pool_size: int = TOA_VAL_POOL_SIZE,
    test_pool_size: int = TOA_TEST_POOL_SIZE,
) -> tuple:
    """
    Load TOA flattened data and return train/val/test splits.

    Dataset keys: X (n, 285), y_cos (n,), y_grain (n,).
    Targets are stacked as y with shape (n, 2).

    Always reserves fixed pools from a seed-based permutation:

        test_pool  = perm[:test_pool_size]
        val_pool   = perm[test_pool_size : test_pool_size + val_pool_size]
        train_pool = perm[test_pool_size + val_pool_size :
                          test_pool_size + val_pool_size + train_pool_size]

    Then takes prefixes ``train_pool[:n_train]``, ``val_pool[:n_val]``,
    ``test_pool[:n_test]``. Pools are reserved even when ``n_val == 0``, so
    train/test indices stay aligned across runners that skip validation.

    Args:
        n_train: Number of training samples (must be <= train_pool_size).
        n_test: Number of test samples (must be <= test_pool_size).
        n_val: Number of validation samples (must be <= val_pool_size;
            0 returns empty val tensors/indices).
        seed: Random seed for shuffled split.
        data_path: Path to toa_data_flattened.npz (defaults to repo root).
        train_pool_size: Fixed training pool size (default 49000).
        val_pool_size: Fixed validation pool size (default 4900).
        test_pool_size: Fixed test pool size (default 5000).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
        (val tensors/indices are empty when ``n_val == 0``).
        Indices are global positions in the dataset (int64 tensors).
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
    pool_total = train_pool_size + val_pool_size + test_pool_size
    if pool_total > n_total:
        raise ValueError(
            f"Restricted pools train={train_pool_size} + val={val_pool_size} + "
            f"test={test_pool_size} = {pool_total} exceed dataset size {n_total}"
        )
    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError(
            f"n_train, n_val, n_test must be >= 0, got "
            f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )
    if n_train > train_pool_size:
        raise ValueError(f"n_train={n_train} exceeds train_pool_size={train_pool_size}")
    if n_val > val_pool_size:
        raise ValueError(f"n_val={n_val} exceeds val_pool_size={val_pool_size}")
    if n_test > test_pool_size:
        raise ValueError(f"n_test={n_test} exceeds test_pool_size={test_pool_size}")

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)

    test_pool = perm[:test_pool_size]
    val_pool = perm[test_pool_size : test_pool_size + val_pool_size]
    train_pool = perm[
        test_pool_size + val_pool_size : test_pool_size + val_pool_size + train_pool_size
    ]

    test_idx = test_pool[:n_test]
    train_idx = train_pool[:n_train]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    if n_val > 0:
        val_idx = val_pool[:n_val]
        X_val = X[val_idx]
        y_val = y[val_idx]
    else:
        val_idx = perm.new_zeros((0,), dtype=torch.int64)
        X_val = X.new_zeros((0, X.shape[-1]))
        y_val = y.new_zeros((0, y.shape[-1]))

    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
