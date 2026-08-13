"""Shared TOA experiment utilities (data splits, import paths)."""

from experiments_toa.data import (
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
    build_maximin_pools,
    get_maximin_pools,
    load_toa_data,
    normalize_design_coords,
    resolve_toa_pool_sizes,
    select_maximin_indices,
)
from experiments_toa.s2_constants import S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_data import load_s2_toa_data

__all__ = [
    "TOA_TEST_POOL_SIZE",
    "TOA_TRAIN_POOL_SIZE",
    "TOA_VAL_POOL_SIZE",
    "S2_INPUT_DIM",
    "S2_TASK_NAMES",
    "build_maximin_pools",
    "get_maximin_pools",
    "load_toa_data",
    "load_s2_toa_data",
    "normalize_design_coords",
    "resolve_toa_pool_sizes",
    "select_maximin_indices",
]
