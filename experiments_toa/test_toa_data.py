"""Tests for TOA data loading and maximin train/val/test pools."""

from __future__ import annotations

import torch

from experiments_toa.data import (
    TOA_COS_MAX,
    TOA_COS_MIN,
    TOA_GRAIN_MAX,
    TOA_GRAIN_MIN,
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
    build_maximin_pools,
    load_toa_data,
    normalize_design_coords,
    select_maximin_indices,
)


def test_normalize_design_coords_corners():
    y = torch.tensor(
        [
            [TOA_COS_MIN, TOA_GRAIN_MIN],
            [TOA_COS_MAX, TOA_GRAIN_MAX],
            [0.5 * (TOA_COS_MIN + TOA_COS_MAX), 0.5 * (TOA_GRAIN_MIN + TOA_GRAIN_MAX)],
        ],
        dtype=torch.float64,
    )
    coords = normalize_design_coords(y)
    assert torch.allclose(coords[0], torch.tensor([0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(coords[1], torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(coords[2], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_select_maximin_prefers_spread_on_toy_grid():
    # 3x3 grid in (cos, grain); pick 4 points — should cover corners / spread.
    cos_levels = torch.tensor([TOA_COS_MIN, 0.5 * (TOA_COS_MIN + TOA_COS_MAX), TOA_COS_MAX])
    grain_levels = torch.tensor([TOA_GRAIN_MIN, 0.5 * (TOA_GRAIN_MIN + TOA_GRAIN_MAX), TOA_GRAIN_MAX])
    rows = []
    for c in cos_levels:
        for g in grain_levels:
            rows.append([float(c), float(g)])
    y = torch.tensor(rows, dtype=torch.float64)
    candidate_idx = torch.arange(9, dtype=torch.int64)

    selected = select_maximin_indices(candidate_idx, y, n_select=4)
    assert selected.numel() == 4
    assert set(selected.tolist()).issubset(set(range(9)))
    assert len(set(selected.tolist())) == 4

    # First point is nearest center (index 4 on the 3x3 grid).
    assert int(selected[0].item()) == 4

    # Remaining three should be among the four corners {0, 2, 6, 8}.
    corners = {0, 2, 6, 8}
    assert set(selected[1:].tolist()).issubset(corners)


def test_select_maximin_deterministic_and_empty():
    y = torch.tensor(
        [
            [TOA_COS_MIN, TOA_GRAIN_MIN],
            [TOA_COS_MAX, TOA_GRAIN_MIN],
            [TOA_COS_MIN, TOA_GRAIN_MAX],
            [TOA_COS_MAX, TOA_GRAIN_MAX],
        ],
        dtype=torch.float64,
    )
    cand = torch.arange(4, dtype=torch.int64)
    a = select_maximin_indices(cand, y, 3)
    b = select_maximin_indices(cand, y, 3)
    assert torch.equal(a, b)
    empty = select_maximin_indices(cand, y, 0)
    assert empty.numel() == 0


def test_build_maximin_pools_disjoint_on_toy_grid():
    # 5x5 factorial-like grid → enough points for small pools.
    cos = torch.linspace(TOA_COS_MIN, TOA_COS_MAX, 5)
    grain = torch.linspace(TOA_GRAIN_MIN, TOA_GRAIN_MAX, 5)
    rows = [[float(c), float(g)] for c in cos for g in grain]
    y = torch.tensor(rows, dtype=torch.float64)
    test_p, val_p, train_p = build_maximin_pools(
        y, train_pool_size=10, val_pool_size=5, test_pool_size=5
    )
    assert test_p.numel() == 5
    assert val_p.numel() == 5
    assert train_p.numel() == 10
    s_test, s_val, s_train = set(test_p.tolist()), set(val_p.tolist()), set(train_p.tolist())
    assert s_test.isdisjoint(s_val)
    assert s_test.isdisjoint(s_train)
    assert s_val.isdisjoint(s_train)
    assert len(s_test | s_val | s_train) == 20


def test_load_toa_random_matches_legacy_prefix():
    n_train, n_test, n_val = 50, 20, 10
    seed = 42
    out_a = load_toa_data(
        n_train=n_train, n_test=n_test, n_val=n_val, seed=seed, train_subset="random"
    )
    out_b = load_toa_data(
        n_train=n_train, n_test=n_test, n_val=n_val, seed=seed, train_subset="random"
    )
    train_idx_a, val_idx_a, test_idx_a = out_a[6], out_a[7], out_a[8]
    train_idx_b, val_idx_b, test_idx_b = out_b[6], out_b[7], out_b[8]
    assert torch.equal(train_idx_a, train_idx_b)
    assert torch.equal(val_idx_a, val_idx_b)
    assert torch.equal(test_idx_a, test_idx_b)
    assert train_idx_a.numel() == n_train
    assert val_idx_a.numel() == n_val
    assert test_idx_a.numel() == n_test


def test_load_toa_maximin_nested_train_fixed_val_test(tmp_path):
    """Small pool sizes so full-dataset FPS stays cheap in CI."""
    seed = 42
    pool_kw = dict(
        train_pool_size=80,
        val_pool_size=30,
        test_pool_size=40,
        seed=seed,
        train_subset="maximin",
        cache_dir=str(tmp_path),
    )
    small = load_toa_data(n_train=20, n_test=40, n_val=30, **pool_kw)
    large = load_toa_data(n_train=60, n_test=40, n_val=30, **pool_kw)
    train_s, val_s, test_s = small[6], small[7], small[8]
    train_l, val_l, test_l = large[6], large[7], large[8]

    assert torch.equal(val_s, val_l)
    assert torch.equal(test_s, test_l)
    assert torch.equal(train_s, train_l[:20])
    assert train_s.numel() == 20
    assert train_l.numel() == 60
    assert len(set(train_l.tolist())) == 60
    assert set(train_l.tolist()).isdisjoint(set(test_l.tolist()))
    assert set(train_l.tolist()).isdisjoint(set(val_l.tolist()))
    assert set(val_l.tolist()).isdisjoint(set(test_l.tolist()))


def test_load_toa_maximin_n_val_zero_aligns_train_test(tmp_path):
    pool_kw = dict(
        train_pool_size=50,
        val_pool_size=20,
        test_pool_size=25,
        seed=7,
        train_subset="maximin",
        cache_dir=str(tmp_path),
    )
    with_val = load_toa_data(n_train=30, n_test=25, n_val=20, **pool_kw)
    without = load_toa_data(n_train=30, n_test=25, n_val=0, **pool_kw)
    assert without[7].numel() == 0
    assert torch.equal(with_val[6], without[6])
    assert torch.equal(with_val[8], without[8])


def test_load_toa_maximin_differs_from_random(tmp_path):
    kw = dict(
        n_train=40,
        n_test=20,
        n_val=15,
        seed=42,
        train_pool_size=60,
        val_pool_size=20,
        test_pool_size=25,
        cache_dir=str(tmp_path),
    )
    rand = load_toa_data(train_subset="random", **kw)
    maxim = load_toa_data(train_subset="maximin", **kw)
    # Val/test are no longer the random pools under maximin mode.
    assert not torch.equal(rand[7], maxim[7])
    assert not torch.equal(rand[8], maxim[8])
    assert not torch.equal(rand[6], maxim[6])


def test_load_toa_maximin_cache_hit(tmp_path):
    kw = dict(
        n_train=10,
        n_test=8,
        n_val=5,
        seed=42,
        train_pool_size=30,
        val_pool_size=10,
        test_pool_size=12,
        train_subset="maximin",
        cache_dir=str(tmp_path),
    )
    a = load_toa_data(**kw)
    cache_files = list(tmp_path.glob("toa_maximin_pools_*.pt"))
    assert len(cache_files) == 1
    b = load_toa_data(**kw)
    assert torch.equal(a[6], b[6])
    assert torch.equal(a[7], b[7])
    assert torch.equal(a[8], b[8])
