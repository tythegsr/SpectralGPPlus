"""Load the S2 11-QoI NetCDF TOA dataset and build fixed train/val/test pools."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

from experiments_toa.data import (
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
    resolve_toa_pool_sizes,
)
from experiments_toa.s2_constants import (
    S2_AUX_INPUT_NAMES,
    S2_DEFAULT_DATA_PATH,
    S2_INPUT_DIM,
    S2_INPUT_VARIABLES,
    S2_TASK_NAMES,
)
from experiments_toa.s2_y_transform import (
    _attr_to_str,
    infer_log_scale_from_attrs,
)

InputVariable = Literal["toa_reflectance", "toa_radiance"]


def default_s2_data_path() -> str:
    return str(S2_DEFAULT_DATA_PATH)


def read_elevation_array(h5_file) -> np.ndarray | None:
    """Return 1-D elevation (m) if present, else None."""
    if "elevation" not in h5_file:
        return None
    elev = np.asarray(h5_file["elevation"][:], dtype=np.float64).reshape(-1)
    if elev.size == 0:
        return None
    if not np.isfinite(elev).all():
        raise ValueError("Non-finite values found in elevation")
    return elev


def append_elevation(X: np.ndarray, elevation: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """Append elevation as the last feature column when available."""
    n_spectral = int(X.shape[1])
    meta = {
        "has_elevation": False,
        "n_spectral_bands": n_spectral,
        "input_dim": n_spectral,
        "aux_inputs": [],
        "elevation_index": None,
    }
    if elevation is None:
        return X, meta
    elev = np.asarray(elevation, dtype=np.float64).reshape(-1)
    if elev.shape[0] != X.shape[0]:
        raise ValueError(
            f"elevation length {elev.shape[0]} does not match n_samples {X.shape[0]}"
        )
    X_out = np.column_stack([X, elev])
    meta.update(
        {
            "has_elevation": True,
            "input_dim": int(X_out.shape[1]),
            "aux_inputs": list(S2_AUX_INPUT_NAMES),
            "elevation_index": n_spectral,
        }
    )
    return X_out, meta


def _dataset_fingerprint(path: str | Path) -> str:
    abs_path = os.path.abspath(str(path))
    st = os.stat(abs_path)
    payload = f"{abs_path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _read_dataset_attrs(h5_file) -> dict[str, str]:
    """Decode NetCDF/HDF5 root attrs to plain strings for log-scale inference."""
    out: dict[str, str] = {}
    for key in h5_file.attrs:
        out[str(key)] = _attr_to_str(h5_file.attrs[key])
    return out


def load_s2_arrays(
    data_path: str | Path | None = None,
    *,
    input_variable: InputVariable = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    include_elevation: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """
    Load raw S2 arrays from NetCDF via h5py.

    Returns
    -------
    X : (n, 285) or (n, 286) float64
        Spectral columns first; if elevation is present and included, it is
        appended as the last column (never part of band keep/drop ranges).
    Y : (n, T) float64
    wavelengths_nm : (285,) float64
    task_names : list[str]
    meta : dict
        Includes ``log_scale``, ``log_scale_tasks``, ``log_scale_source`` inferred
        from NetCDF attrs (``output_log_scale`` / ``log_uniform_qois``), plus
        ``has_elevation`` / ``n_spectral_bands`` / ``elevation_index``.
    """
    if input_variable not in S2_INPUT_VARIABLES:
        raise ValueError(
            f"input_variable must be one of {S2_INPUT_VARIABLES}, got {input_variable!r}"
        )
    if data_path is None:
        data_path = default_s2_data_path()
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"S2 NetCDF not found: {data_path}")

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    unknown = [n for n in names if n not in S2_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S2 task names: {unknown}")
    if not names:
        raise ValueError("task_names must be non-empty")

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "h5py is required to load the S2 NetCDF dataset. Install with: pip install h5py"
        ) from exc

    with h5py.File(data_path, "r") as f:
        if input_variable not in f:
            raise KeyError(f"Missing variable {input_variable!r} in {data_path}")
        if "wl" not in f:
            raise KeyError(f"Missing wavelength variable 'wl' in {data_path}")
        X = np.asarray(f[input_variable][:], dtype=np.float64)
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        elev = read_elevation_array(f)
        cols = []
        for name in names:
            if name not in f:
                raise KeyError(f"Missing QoI {name!r} in {data_path}")
            cols.append(np.asarray(f[name][:], dtype=np.float64))
        Y = np.column_stack(cols)
        dataset_attrs = _read_dataset_attrs(f)

    if X.ndim != 2 or X.shape[1] != S2_INPUT_DIM:
        raise ValueError(f"Expected X shape (n, {S2_INPUT_DIM}), got {X.shape}")
    if wl.shape != (S2_INPUT_DIM,):
        raise ValueError(f"Expected wl shape ({S2_INPUT_DIM},), got {wl.shape}")
    if Y.shape != (X.shape[0], len(names)):
        raise ValueError(f"Expected Y shape ({X.shape[0]}, {len(names)}), got {Y.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in X")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values found in Y")

    use_elev = elev is not None if include_elevation is None else bool(include_elevation)
    if use_elev and elev is None:
        raise KeyError(f"include_elevation=True but 'elevation' missing in {data_path}")
    X, elev_meta = append_elevation(X, elev if use_elev else None)

    log_scale, log_tasks, log_source = infer_log_scale_from_attrs(dataset_attrs)
    meta = {
        "data_path": os.path.abspath(str(data_path)),
        "dataset_fingerprint": _dataset_fingerprint(data_path),
        "input_variable": input_variable,
        "input_dim": int(elev_meta["input_dim"]),
        "n_spectral_bands": int(elev_meta["n_spectral_bands"]),
        "has_elevation": bool(elev_meta["has_elevation"]),
        "aux_inputs": list(elev_meta["aux_inputs"]),
        "elevation_index": elev_meta["elevation_index"],
        "n_samples": int(X.shape[0]),
        "task_names": list(names),
        "num_tasks": len(names),
        "dataset_attrs": dataset_attrs,
        "log_scale": bool(log_scale),
        "log_scale_tasks": sorted(log_tasks),
        "log_scale_source": log_source,
    }
    return X, Y, wl, list(names), meta


def load_s2_toa_data(
    n_train: int,
    n_test: int = TOA_TEST_POOL_SIZE,
    n_val: int = 0,
    seed: int = 42,
    data_path: str | Path | None = None,
    train_pool_size: int = TOA_TRAIN_POOL_SIZE,
    val_pool_size: int = TOA_VAL_POOL_SIZE,
    test_pool_size: int = TOA_TEST_POOL_SIZE,
    input_variable: InputVariable = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    include_elevation: bool | None = None,
) -> tuple:
    """
    Load S2 TOA data and return train/val/test splits.

    Uses deterministic random pools (same prefix semantics as S1).
    Pool sizes start at the legacy 49k / 4.9k / 5k defaults and grow when
    ``n_train`` / ``n_test`` / ``n_val`` (or leftover rows) require it.
    Maximin is intentionally not used for the 11-D QoI design.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    train_idx, val_idx, test_idx,
    wavelengths_nm, meta
    """
    X_np, Y_np, wl, names, meta = load_s2_arrays(
        data_path,
        input_variable=input_variable,
        task_names=task_names,
        include_elevation=include_elevation,
    )
    X = torch.tensor(X_np, dtype=torch.float64)
    y = torch.tensor(Y_np, dtype=torch.float64)
    wavelengths = torch.tensor(wl, dtype=torch.float64)

    n_total = X.shape[0]
    train_pool_size, val_pool_size, test_pool_size = resolve_toa_pool_sizes(
        n_total,
        n_train,
        n_val,
        n_test,
        train_pool_size=train_pool_size,
        val_pool_size=val_pool_size,
        test_pool_size=test_pool_size,
    )

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

    # print(y_train)
    print(X_train[0])
    print(y_train[0])

    if n_val > 0:
        val_idx = val_pool[:n_val]
        X_val = X[val_idx]
        y_val = y[val_idx]
    else:
        val_idx = torch.zeros((0,), dtype=torch.int64)
        X_val = X.new_zeros((0, X.shape[-1]))
        y_val = y.new_zeros((0, y.shape[-1]))

    meta = {
        **meta,
        "seed": int(seed),
        "train_pool_size": int(train_pool_size),
        "val_pool_size": int(val_pool_size),
        "test_pool_size": int(test_pool_size),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "split_mode": "random",
    }
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_idx,
        val_idx,
        test_idx,
        wavelengths,
        meta,
    )
