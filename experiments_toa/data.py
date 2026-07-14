"""Single source of truth for TOA dataset loading and fixed train/val/test pools."""

from __future__ import annotations

import hashlib
import os
from typing import Literal

import numpy as np
import torch

# Fixed restricted pools for TOA splits.
# Smaller n_train / n_val / n_test take prefixes of these pools so val/test
# stay identical when training size is reduced.
TOA_TRAIN_POOL_SIZE = 49000
TOA_VAL_POOL_SIZE = 4900
TOA_TEST_POOL_SIZE = 5000

# Full-factorial design ranges for (cos, grain) normalization to [0, 1]^2.
TOA_COS_MIN = 0.06
TOA_COS_MAX = 1.0
TOA_GRAIN_MIN = 30.0
TOA_GRAIN_MAX = 1500.0

_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

TrainSubsetMode = Literal["random", "maximin"]


def normalize_design_coords(y: torch.Tensor) -> torch.Tensor:
    """
    Map stacked TOA targets ``y`` (cos, grain) to ``[0, 1]^2`` using full-factorial ranges.

    Args:
        y: Tensor of shape ``(..., 2)`` with columns ``(cos, grain)``.

    Returns:
        Normalized coordinates with the same leading shape and dtype/device as ``y``.
    """
    if y.shape[-1] != 2:
        raise ValueError(f"Expected y with last dim 2 (cos, grain), got shape {tuple(y.shape)}")
    cos = (y[..., 0] - TOA_COS_MIN) / (TOA_COS_MAX - TOA_COS_MIN)
    grain = (y[..., 1] - TOA_GRAIN_MIN) / (TOA_GRAIN_MAX - TOA_GRAIN_MIN)
    return torch.stack([cos, grain], dim=-1)


def _normalize_design_coords_np(y: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`normalize_design_coords` for FPS loops."""
    if y.shape[-1] != 2:
        raise ValueError(f"Expected y with last dim 2 (cos, grain), got shape {y.shape}")
    cos = (y[..., 0] - TOA_COS_MIN) / (TOA_COS_MAX - TOA_COS_MIN)
    grain = (y[..., 1] - TOA_GRAIN_MIN) / (TOA_GRAIN_MAX - TOA_GRAIN_MIN)
    return np.stack([cos, grain], axis=-1).astype(np.float64, copy=False)


def _greedy_maximin_local(coords: np.ndarray, n_select: int) -> np.ndarray:
    """
    Greedy farthest-point order on normalized ``coords`` ``(n, 2)``.

    Returns local indices of length ``n_select``. Starts at the point nearest
    the center of the unit box.
    """
    n_cand = int(coords.shape[0])
    if n_select < 0 or n_select > n_cand:
        raise ValueError(
            f"n_select={n_select} must satisfy 0 <= n_select <= n_cand={n_cand}"
        )
    if n_select == 0:
        return np.zeros(0, dtype=np.int64)

    selected = np.empty(n_select, dtype=np.int64)
    center = np.array([0.5, 0.5], dtype=np.float64)
    first = int(np.argmin(np.sum((coords - center) ** 2, axis=1)))
    selected[0] = first
    min_dist_sq = np.sum((coords - coords[first]) ** 2, axis=1)
    min_dist_sq[first] = -1.0

    for k in range(1, n_select):
        nxt = int(np.argmax(min_dist_sq))
        selected[k] = nxt
        diff = coords - coords[nxt]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
        min_dist_sq[nxt] = -1.0

    return selected


def select_maximin_indices(
    candidate_idx: torch.Tensor,
    y: torch.Tensor,
    n_select: int,
) -> torch.Tensor:
    """
    Greedy farthest-point (maximin) subset of ``candidate_idx`` in normalized (cos, grain).

    Starts at the candidate closest to the center of the unit box, then iteratively adds
    the point that maximizes minimum Euclidean distance to the selected set.

    Args:
        candidate_idx: Global dataset indices available for selection (1-D int64).
        y: Full dataset targets of shape ``(n_total, 2)`` with columns ``(cos, grain)``.
        n_select: Number of points to select (must be ``0 <= n_select <= len(candidate_idx)``).

    Returns:
        Selected global indices as a 1-D int64 tensor of length ``n_select``.
    """
    if candidate_idx.ndim != 1:
        raise ValueError(f"candidate_idx must be 1-D, got shape {tuple(candidate_idx.shape)}")
    n_cand = int(candidate_idx.numel())
    if n_select < 0 or n_select > n_cand:
        raise ValueError(
            f"n_select={n_select} must satisfy 0 <= n_select <= len(candidate_idx)={n_cand}"
        )
    if n_select == 0:
        return candidate_idx.new_zeros((0,), dtype=torch.int64)

    cand_np = candidate_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    y_np = y.detach().cpu().numpy()
    coords = _normalize_design_coords_np(y_np[cand_np])
    local = _greedy_maximin_local(coords, n_select)
    return torch.as_tensor(cand_np[local], dtype=torch.int64)


def build_maximin_pools(
    y: torch.Tensor | np.ndarray,
    train_pool_size: int = TOA_TRAIN_POOL_SIZE,
    val_pool_size: int = TOA_VAL_POOL_SIZE,
    test_pool_size: int = TOA_TEST_POOL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sequential greedy maximin pools over the full design: test → val → train.

    Each pool is built independently on the remaining candidates (each FPS restarts
    from the center of that candidate set).

    Returns:
        ``(test_pool, val_pool, train_pool)`` as int64 tensors.
    """
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)
    n_total = int(y_np.shape[0])
    pool_total = train_pool_size + val_pool_size + test_pool_size
    if pool_total > n_total:
        raise ValueError(
            f"Pools train={train_pool_size} + val={val_pool_size} + "
            f"test={test_pool_size} = {pool_total} exceed dataset size {n_total}"
        )

    remaining = np.arange(n_total, dtype=np.int64)
    coords_all = _normalize_design_coords_np(y_np)

    def _take(n: int) -> np.ndarray:
        nonlocal remaining
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        local = _greedy_maximin_local(coords_all[remaining], n)
        chosen = remaining[local]
        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[local] = False
        remaining = remaining[mask]
        return chosen

    test_pool = _take(test_pool_size)
    val_pool = _take(val_pool_size)
    train_pool = _take(train_pool_size)
    return (
        torch.as_tensor(test_pool, dtype=torch.int64),
        torch.as_tensor(val_pool, dtype=torch.int64),
        torch.as_tensor(train_pool, dtype=torch.int64),
    )


def _maximin_cache_path(
    data_path: str,
    seed: int,
    train_pool_size: int,
    val_pool_size: int,
    test_pool_size: int,
    cache_dir: str,
) -> str:
    digest = hashlib.sha1(os.path.abspath(data_path).encode("utf-8")).hexdigest()[:10]
    name = (
        f"toa_maximin_pools_seed{seed}_"
        f"tr{train_pool_size}_va{val_pool_size}_te{test_pool_size}_{digest}.pt"
    )
    return os.path.join(cache_dir, name)


def get_maximin_pools(
    y: torch.Tensor | np.ndarray,
    data_path: str,
    seed: int = 42,
    train_pool_size: int = TOA_TRAIN_POOL_SIZE,
    val_pool_size: int = TOA_VAL_POOL_SIZE,
    test_pool_size: int = TOA_TEST_POOL_SIZE,
    cache_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load cached maximin pools or build and cache them.

    Pool construction is deterministic from ``y`` (seed is only part of the cache key).
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = _maximin_cache_path(
        data_path, seed, train_pool_size, val_pool_size, test_pool_size, cache_dir
    )
    data_path_abs = os.path.abspath(data_path)

    if os.path.isfile(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            int(payload.get("seed", -1)) == seed
            and int(payload.get("train_pool_size", -1)) == train_pool_size
            and int(payload.get("val_pool_size", -1)) == val_pool_size
            and int(payload.get("test_pool_size", -1)) == test_pool_size
            and os.path.abspath(str(payload.get("data_path", ""))) == data_path_abs
        ):
            return (
                payload["test_pool"].to(dtype=torch.int64),
                payload["val_pool"].to(dtype=torch.int64),
                payload["train_pool"].to(dtype=torch.int64),
            )

    print(
        f"Building maximin TOA pools "
        f"(test={test_pool_size}, val={val_pool_size}, train={train_pool_size}); "
        f"caching to {path}"
    )
    test_pool, val_pool, train_pool = build_maximin_pools(
        y,
        train_pool_size=train_pool_size,
        val_pool_size=val_pool_size,
        test_pool_size=test_pool_size,
    )
    torch.save(
        {
            "seed": seed,
            "train_pool_size": train_pool_size,
            "val_pool_size": val_pool_size,
            "test_pool_size": test_pool_size,
            "data_path": data_path_abs,
            "test_pool": test_pool,
            "val_pool": val_pool,
            "train_pool": train_pool,
        },
        path,
    )
    return test_pool, val_pool, train_pool


def load_toa_data(
    n_train: int,
    n_test: int = TOA_TEST_POOL_SIZE,
    n_val: int = 0,
    seed: int = 42,
    data_path: str | None = None,
    train_pool_size: int = TOA_TRAIN_POOL_SIZE,
    val_pool_size: int = TOA_VAL_POOL_SIZE,
    test_pool_size: int = TOA_TEST_POOL_SIZE,
    train_subset: TrainSubsetMode = "random",
    cache_dir: str | None = None,
) -> tuple:
    """
    Load TOA flattened data and return train/val/test splits.

    Dataset keys: X (n, 285), y_cos (n,), y_grain (n,).
    Targets are stacked as y with shape (n, 2).

    Always reserves fixed pools; smaller sizes take prefixes so val/test stay
    identical when ``n_train`` changes (and pools are reserved even when
    ``n_val == 0``).

    ``train_subset``:

    - ``"random"`` (default): seeded ``randperm`` pools; prefixes of those pools.
    - ``"maximin"``: sequential space-filling pools over normalized (cos, grain)
      in order test → val → train (cached on disk). Prefixes of the ordered
      pools; val/test are fixed across methods and training sizes.

    Args:
        n_train: Number of training samples (must be <= train_pool_size).
        n_test: Number of test samples (must be <= test_pool_size).
        n_val: Number of validation samples (must be <= val_pool_size;
            0 returns empty val tensors/indices).
        seed: Random seed for shuffled split (random mode); also used in the
            maximin cache key (construction itself is deterministic from ``y``).
        data_path: Path to toa_data_flattened.npz (defaults to repo root).
        train_pool_size: Fixed training pool size (default 49000).
        val_pool_size: Fixed validation pool size (default 4900).
        test_pool_size: Fixed test pool size (default 5000).
        train_subset: ``"random"`` or ``"maximin"``.
        cache_dir: Directory for maximin pool cache (default ``experiments_toa/cache``).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
        (val tensors/indices are empty when ``n_val == 0``).
        Indices are global positions in the dataset (int64 tensors).
    """
    if train_subset not in ("random", "maximin"):
        raise ValueError(
            f"train_subset must be 'random' or 'maximin', got {train_subset!r}"
        )

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

    if train_subset == "maximin":
        test_pool, val_pool, train_pool = get_maximin_pools(
            y,
            data_path=data_path,
            seed=seed,
            train_pool_size=train_pool_size,
            val_pool_size=val_pool_size,
            test_pool_size=test_pool_size,
            cache_dir=cache_dir,
        )
    else:
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
        val_idx = torch.zeros((0,), dtype=torch.int64)
        X_val = X.new_zeros((0, X.shape[-1]))
        y_val = y.new_zeros((0, y.shape[-1]))

    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
