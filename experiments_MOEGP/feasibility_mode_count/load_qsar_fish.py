"""Load UCI QSAR fish toxicity for foundation compare."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# UCI 504 / QSAR fish toxicity: N=908, d=6, target LC50
QSAR_FISH_N = 908
QSAR_FISH_N_TRAIN = 726  # ~80%
QSAR_FISH_N_TEST = 182  # ~20%


def load_qsar_fish_xy() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return float64 ``X`` ``[N,d]``, ``y`` ``[N]`` via experiments_SORF loader."""
    import sys

    sorf = Path(__file__).resolve().parents[2] / "experiments_SORF"
    if str(sorf) not in sys.path:
        sys.path.insert(0, str(sorf))
    from load_uci_regression_data import load_qsar_fish_toxicity

    X_t, y_t = load_qsar_fish_toxicity(print_info=False)
    X = np.asarray(X_t.detach().cpu().numpy(), dtype=np.float64)
    y = np.asarray(y_t.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    meta = {
        "dataset_name": "qsar_fish_toxicity",
        "uci_id": 504,
        "target": "LC50",
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }
    return X, y, meta


def split_qsar_fish(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int = QSAR_FISH_N_TRAIN,
    n_test: int = QSAR_FISH_N_TEST,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded permutation: first ``n_test`` → test, next ``n_train`` → train."""
    n = int(X.shape[0])
    if n_train + n_test > n:
        raise ValueError(
            f"Need n_train+n_test={n_train + n_test} <= n={n} for fish split"
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
