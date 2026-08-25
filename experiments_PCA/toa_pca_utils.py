"""PCA fit/transform utilities for TOA PCA+GP experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass
class ToaPCAFit:
    """Fitted PCA state (train-only)."""

    pca: PCA
    n_components: int
    input_dim: int
    svd_solver: str

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    @property
    def total_variance_explained(self) -> float:
        return float(self.cumulative_variance[-1]) if self.n_components > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "input_dim_after_drop": self.input_dim,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_variance": self.cumulative_variance.tolist(),
            "total_variance_explained": self.total_variance_explained,
            "svd_solver": self.svd_solver,
            "mean": self.pca.mean_.tolist(),
            "components": self.pca.components_.tolist(),
            "explained_variance_": self.pca.explained_variance_.tolist(),
        }


def pca_from_dict(meta: dict[str, Any]) -> ToaPCAFit:
    """Rebuild sklearn PCA from :meth:`ToaPCAFit.to_dict` payload."""
    n_components = int(meta["n_components"])
    input_dim = int(meta["input_dim_after_drop"])
    svd_solver = str(meta.get("svd_solver", "randomized"))
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    pca.mean_ = np.asarray(meta["mean"], dtype=np.float64)
    pca.components_ = np.asarray(meta["components"], dtype=np.float64)
    pca.explained_variance_ratio_ = np.asarray(
        meta["explained_variance_ratio"], dtype=np.float64
    )
    if "explained_variance_" in meta:
        pca.explained_variance_ = np.asarray(meta["explained_variance_"], dtype=np.float64)
    else:
        pca.explained_variance_ = pca.explained_variance_ratio_.copy()
    pca.n_features_in_ = input_dim
    pca.n_components_ = n_components
    return ToaPCAFit(
        pca=pca,
        n_components=n_components,
        input_dim=input_dim,
        svd_solver=svd_solver,
    )


def fit_pca_on_train(
    x_train: torch.Tensor | np.ndarray,
    *,
    n_components: int,
    svd_solver: str = "randomized",
    random_state: int = 42,
) -> ToaPCAFit:
    """Fit sklearn PCA on training inputs only."""
    x_np = np.asarray(x_train, dtype=np.float64)
    if x_np.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {x_np.shape}")
    input_dim = x_np.shape[1]
    n_comp = min(int(n_components), input_dim, x_np.shape[0])
    if n_comp < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    pca = PCA(n_components=n_comp, svd_solver=svd_solver, random_state=random_state)
    pca.fit(x_np)
    return ToaPCAFit(
        pca=pca,
        n_components=n_comp,
        input_dim=input_dim,
        svd_solver=svd_solver,
    )


def transform_pca(
    fit: ToaPCAFit,
    x: torch.Tensor | np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Project inputs with a fitted PCA."""
    x_np = np.asarray(x, dtype=np.float64)
    z_np = fit.pca.transform(x_np)
    return torch.as_tensor(z_np, dtype=dtype)


def make_train_partitions(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    partition_size: int,
    seed: int,
    shuffle: bool = True,
) -> list[dict[str, torch.Tensor | int]]:
    """
    Split training tensors into disjoint partitions.

    Returns list of dicts with keys: z, y, train_indices, partition_index, n_partition.
    """
    n = z_train.shape[0]
    if partition_size < 1:
        raise ValueError(f"partition_size must be >= 1, got {partition_size}")

    order = torch.arange(n, dtype=torch.int64)
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        order = order[torch.randperm(n, generator=gen)]

    partitions: list[dict[str, torch.Tensor | int]] = []
    k = 0
    for start in range(0, n, partition_size):
        end = min(start + partition_size, n)
        idx = order[start:end]
        partitions.append(
            {
                "partition_index": k,
                "n_partition": int(idx.numel()),
                "z": z_train.index_select(0, idx),
                "y": y_train.index_select(0, idx),
                "train_indices": train_idx.index_select(0, idx),
            }
        )
        k += 1
    return partitions
