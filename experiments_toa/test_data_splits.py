"""Tests for canonical TOA data splits in ``experiments_toa.data``."""

from __future__ import annotations

import pytest

from experiments_toa.data import (
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
    load_toa_data,
)


def test_toa_pool_constants() -> None:
    assert TOA_TRAIN_POOL_SIZE == 49000
    assert TOA_VAL_POOL_SIZE == 4900
    assert TOA_TEST_POOL_SIZE == 5000


def test_n_val_zero_preserves_train_and_test_splits() -> None:
    """Val pool is still reserved when n_val=0 so train/test indices stay aligned."""
    seed = 42
    n_train = 1000
    n_test = 500

    with_val, without_val = (
        load_toa_data(n_train=n_train, n_test=n_test, n_val=n_val, seed=seed)
        for n_val in (TOA_VAL_POOL_SIZE, 0)
    )
    _, _, _, _, _, _, train_a, _, test_a = with_val
    _, _, _, _, _, _, train_b, val_b, test_b = without_val

    assert val_b.numel() == 0
    assert torch_equal(train_a, train_b)
    assert torch_equal(test_a, test_b)


def test_smaller_n_train_is_prefix_of_full_train_pool() -> None:
    seed = 7
    full = load_toa_data(
        n_train=TOA_TRAIN_POOL_SIZE,
        n_test=TOA_TEST_POOL_SIZE,
        n_val=TOA_VAL_POOL_SIZE,
        seed=seed,
    )
    small = load_toa_data(
        n_train=1000,
        n_test=TOA_TEST_POOL_SIZE,
        n_val=TOA_VAL_POOL_SIZE,
        seed=seed,
    )
    train_full, _, _, _, test_full, _, idx_full, val_full, test_idx_full = full
    train_small, _, _, _, test_small, _, idx_small, val_small, test_idx_small = small

    assert torch_equal(idx_small, idx_full[:1000])
    assert torch_equal(test_idx_small, test_idx_full)
    assert torch_equal(val_small, val_full)
    assert torch_equal(train_small, train_full[:1000])
    assert torch_equal(test_small, test_full)


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a, b)
