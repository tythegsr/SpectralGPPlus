"""Focused tests for S2 PCA partitioned exact GPR."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _PCA_DIR)

from experiments_toa.s2_constants import S2_DEFAULT_DATA_PATH
from toa_pca_partition_gpr_base import ENSEMBLE_MODE_NAMES, _aggregate_ensemble
from toa_pca_utils import make_train_partitions


def test_make_train_partitions_count_when_n_exceeds_size():
    n_train, partition_size = 40, 15
    z = torch.randn(n_train, 4)
    y = torch.randn(n_train, 1)
    train_idx = torch.arange(n_train)
    parts = make_train_partitions(
        z, y, train_idx, partition_size=partition_size, seed=7, shuffle=True
    )
    assert len(parts) == math.ceil(n_train / partition_size)
    assert sum(int(p["n_partition"]) for p in parts) == n_train
    assert max(int(p["n_partition"]) for p in parts) <= partition_size
    # Disjoint coverage of training rows.
    covered = torch.cat([p["train_indices"] for p in parts]).sort().values
    assert torch.equal(covered, train_idx)


def test_aggregate_ensemble_full_matches_mean_and_law_of_total_variance():
    rng = np.random.default_rng(0)
    k, n_test, p = 3, 5, 2
    mu = rng.normal(size=(k, n_test))
    std = rng.uniform(0.1, 1.0, size=(k, n_test))
    centroids = rng.normal(size=(k, p))
    z_test = rng.normal(size=(n_test, p))
    agg_mu, agg_std, _ = _aggregate_ensemble(
        mu,
        std,
        centroids,
        z_test,
        mode="full",
        single_partition_index=0,
        top_m=2,
    )
    assert agg_mu.shape == (n_test,)
    np.testing.assert_allclose(agg_mu, mu.mean(axis=0))
    within = np.mean(std**2, axis=0)
    between = np.var(mu, axis=0)
    np.testing.assert_allclose(agg_std, np.sqrt(np.maximum(within + between, 0.0)))
    assert "full" in ENSEMBLE_MODE_NAMES


@pytest.mark.skipif(not S2_DEFAULT_DATA_PATH.is_file(), reason="S2 NetCDF not present")
def test_s2_pca_gpr_smoke_n_partitions(tmp_path):
    from toa_s2_pca_gpr_base import run_s2_toa_pca_gpr

    n_train, partition_size = 40, 15
    metrics = run_s2_toa_pca_gpr(
        n_train=n_train,
        n_test=20,
        n_components=5,
        partition_size=partition_size,
        num_inits=1,
        num_epochs=1,
        seed=0,
        device="cpu",
        dtype=torch.float64,
        save_path=str(tmp_path / "s2_pca_gpr_smoke"),
        ard=True,
        n_jobs=1,
        monitor_validation=False,
        plot_posterior=False,
        plot_validation=False,
        task_names=["cos_i"],
        partition_shuffle=True,
        top_m_partitions=2,
    )
    expected_k = int(math.ceil(n_train / partition_size))
    assert metrics["n_partitions"] == expected_k
    assert metrics["n_partitions_by_task"]["cos_i"] == expected_k
    assert metrics["primary_ensemble_mode"] == "full"
    assert metrics["pca_partition_mode"] == "independent_gpr_per_task_partition"
    assert metrics["model_class"] == "GPR_partitioned"
    assert len(metrics["partition_runs"]) == expected_k
    assert set(metrics["ensemble_modes"].keys()) == set(ENSEMBLE_MODE_NAMES)
    assert Path(metrics["predictions_npz"]).is_file()
