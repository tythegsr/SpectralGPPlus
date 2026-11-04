"""Load TabArena-v0.1 ``concrete_compressive_strength`` (OpenML 46917).

Single-dataset helper for foundation compare — not full TabArena CV.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# TabArena suite 457 / OpenML dataset id (see TabArena metadata CSV).
TABARENA_CONCRETE_OPENML_ID = 46917
TABARENA_CONCRETE_TARGET = "ConcreteCompressiveStrength"
# 80/20 of N=1030 → 824 / 206
TABARENA_CONCRETE_N = 1030
TABARENA_CONCRETE_N_TRAIN = 824
TABARENA_CONCRETE_N_TEST = 206


def load_tabarena_concrete_xy(
    *,
    dataset_id: int = TABARENA_CONCRETE_OPENML_ID,
    target_name: str = TABARENA_CONCRETE_TARGET,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Download OpenML regression table → float64 ``X`` ``[N,d]``, ``y`` ``[N]``.

    Non-numeric feature columns are dropped. Target is ``target_name`` when
    present, else the dataset default target attribute.
    """
    import openml

    ds = openml.datasets.get_dataset(int(dataset_id), download_data=True)
    # Newer openml returns (X, y, categorical_indicator, attribute_names)
    X_df, y_s, categorical_indicator, attribute_names = ds.get_data(
        target=ds.default_target_attribute,
        dataset_format="dataframe",
    )
    # Prefer explicit TabArena target name if columns still include it
    if hasattr(y_s, "name") and y_s.name:
        y_col = str(y_s.name)
    else:
        y_col = str(ds.default_target_attribute)

    if target_name and target_name != y_col:
        # Some OpenML dumps keep the TabArena name as default; if not, y_s is fine.
        pass

    # Features: numeric only
    X_num = X_df.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        raise RuntimeError(
            f"OpenML dataset {dataset_id} has no numeric features after filtering."
        )
    X = np.asarray(X_num.to_numpy(), dtype=np.float64)
    y = np.asarray(y_s.to_numpy(), dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"X/y length mismatch: X={X.shape[0]} y={y.shape[0]} (dataset {dataset_id})"
        )
    # Drop rows with NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    meta = {
        "openml_id": int(dataset_id),
        "dataset_name": getattr(ds, "name", "concrete_compressive_strength"),
        "target": y_col,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "feature_names": list(X_num.columns),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "categorical_indicator": categorical_indicator,
        "attribute_names": attribute_names,
    }
    return X, y, meta


def split_tabarena_concrete(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int = TABARENA_CONCRETE_N_TRAIN,
    n_test: int = TABARENA_CONCRETE_N_TEST,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded permutation: first ``n_test`` → test, next ``n_train`` → train."""
    n = int(X.shape[0])
    if n_train + n_test > n:
        raise ValueError(
            f"Need n_train+n_test={n_train + n_test} <= n={n} for concrete split"
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    te = perm[:n_test]
    tr = perm[n_test : n_test + n_train]
    return (
        np.asarray(X[tr], dtype=np.float64),
        np.asarray(y[tr], dtype=np.float64),
        np.asarray(X[te], dtype=np.float64),
        np.asarray(y[te], dtype=np.float64),
    )
