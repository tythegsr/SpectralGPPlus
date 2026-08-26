"""EMIT S4 joint loader (radiance + geometry aux → S2-style QoIs).

Trains on processed EMIT NetCDF with radiance spectra plus coszen, ele_km,
and RAA_TRUE as model inputs. Labels include cos_i from calc_new_angles (sinA/cosA
+ SZA/SAA/slope) and lwc from state.
Also exports the shared loader/split used by independent S4 SORF.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
_DEFAULT_EMIT_PATH = (
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
    / "emit_test_data_processed_90to100_20262508.nc"
)
_DEFAULT_WL_SRC = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261208.nc"
)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.export_emit_as_s2 import mapped_qois
from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
from experiments_toa.s2_constants import (
    S4_AUX_INPUT_NAMES,
    S4_INPUT_DIM,
    S4_SPECTRAL_DIM,
    S4_TASK_NAMES,
)
from experiments_toa.s2_data import read_elevation_array

from experiments_toa.emit_geometry import (
    OBS_SAA,
    OBS_SZA,
    OBS_VAA,
    calc_cos_i_from_state_obs,
    elevation_to_km,
    relative_azimuth,
)

TASK_VALID_Y_RANGE: dict[str, tuple[float, float]] = {
    "algae": (1e-2, 6e5),
    "cos_i": (0.06, 1.0),
    "grain_size": (30.0, 1500.0),
    "fsnow": (0.9, 1.0),
    "fPV": (0.0, 1.0),
    "fNPV": (0.0, 1.0),
    "fsoil": (0.0, 1.0),
    "lwc": (0.0, 1e6),
}

SCATTER_MAX_POINTS = 20_000


def extract_s4_aux_inputs(
    obs: np.ndarray,
    elevation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coszen, ele_km, RAA_TRUE arrays (N,) each."""
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] < 9:
        raise ValueError(f"obs must be (N, >=9), got {obs.shape}")
    sza = obs[:, OBS_SZA]
    coszen = np.cos(np.radians(sza))
    raa_true = np.array(
        [relative_azimuth(saa, vaa) for saa, vaa in zip(obs[:, OBS_SAA], obs[:, OBS_VAA])],
        dtype=np.float64,
    )
    if elevation is None:
        raise KeyError("S4 aux inputs require elevation in the NetCDF")
    ele_km = elevation_to_km(elevation)
    return coszen, ele_km, raa_true


def append_s4_aux_inputs(
    X: np.ndarray,
    coszen: np.ndarray,
    ele_km: np.ndarray,
    raa_true: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Append coszen, ele_km, RAA_TRUE after spectral columns."""
    n_spectral = int(X.shape[1])
    n = int(X.shape[0])
    for name, arr in (
        ("coszen", coszen),
        ("ele_km", ele_km),
        ("RAA_TRUE", raa_true),
    ):
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.shape[0] != n:
            raise ValueError(f"{name} length {a.shape[0]} != n_samples {n}")
    aux = np.column_stack([coszen, ele_km, raa_true])
    X_out = np.column_stack([X, aux])
    aux_indices = list(range(n_spectral, n_spectral + len(S4_AUX_INPUT_NAMES)))
    meta = {
        "has_aux_inputs": True,
        "n_spectral_bands": n_spectral,
        "input_dim": int(X_out.shape[1]),
        "aux_inputs": list(S4_AUX_INPUT_NAMES),
        "aux_indices": aux_indices,
        "coszen_index": aux_indices[0],
        "ele_km_index": aux_indices[1],
        "raa_true_index": aux_indices[2],
    }
    return X_out, meta


def mapped_qois_s4(state: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Map ISOFIT state + obs geometry to S4 task labels."""
    state = np.asarray(state, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    base = mapped_qois(state)
    return {
        "grain_size": base["grain_size"],
        "cos_i": calc_cos_i_from_state_obs(state, obs),
        "lwc": base["liquid_water"],
        "dust": base["dust"],
        "algae": base["algae"],
        "fsnow": base["fsnow"],
        "fPV": base["fPV"],
        "fNPV": base["fNPV"],
        "fsoil": base["fsoil"],
        "cwv": base["cwv"],
    }


def parse_s4_task_names(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize a QoI list; ``None`` means all S4 tasks."""
    if raw is None:
        return list(S4_TASK_NAMES)
    values = [raw] if isinstance(raw, str) else list(raw)
    names = [
        name
        for value in values
        for name in (x.strip() for x in str(value).split(","))
        if name
    ]
    unknown = [n for n in names if n not in S4_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S4 task names: {unknown}. Valid: {list(S4_TASK_NAMES)}")
    if not names:
        raise ValueError("QoI selection produced an empty list")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate QoI names are not allowed: {duplicates}")
    return names


def _load_wavelengths_nm(emit_path: Path, wl_src: Path | None) -> np.ndarray:
    for path in (emit_path, wl_src):
        if path is None or not Path(path).is_file():
            continue
        import h5py

        with h5py.File(path, "r") as f:
            if "wl" in f:
                return np.asarray(f["wl"][:], dtype=np.float64)
    raise FileNotFoundError(
        f"No 'wl' dataset in {emit_path} and fallback wl_src={wl_src} is missing."
    )


def _valid_label_mask(y: np.ndarray, task_names: Sequence[str]) -> np.ndarray:
    mask = np.ones(y.shape[0], dtype=bool)
    for t, name in enumerate(task_names):
        bounds = TASK_VALID_Y_RANGE.get(name)
        if bounds is None:
            continue
        lo, hi = bounds
        mask &= y[:, t] >= lo
        if hi is not None and math.isfinite(hi):
            mask &= y[:, t] <= hi
    return mask


def load_emit_s4_xy(
    emit_path: str | Path,
    *,
    task_names: Sequence[str],
    filter_valid_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load EMIT radiance + geometry aux and S4 QoI labels.

    X columns: radiance (285), coszen, ele_km, RAA_TRUE (288 total).
    """
    import h5py

    emit_path = Path(emit_path)
    if not emit_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {emit_path}")
    names = list(task_names)
    unknown = [n for n in names if n not in S4_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S4 task names: {unknown}")

    with h5py.File(emit_path, "r") as f:
        for key in ("radiance", "obs", "state"):
            if key not in f:
                raise KeyError(f"Missing {key!r} in {emit_path}")
        n_total = int(f["radiance"].shape[0])
        n_bands = int(f["radiance"].shape[1])
        if n_bands != S4_SPECTRAL_DIM:
            raise ValueError(f"Expected {S4_SPECTRAL_DIM} bands, got {n_bands}")
        state = np.asarray(f["state"][:], dtype=np.float64)
        obs = np.asarray(f["obs"][:], dtype=np.float64)
        elev_all = read_elevation_array(f)
        if "state_feature_names" in f.attrs:
            names_attr = f.attrs["state_feature_names"]
            if isinstance(names_attr, bytes):
                names_attr = names_attr.decode("utf-8")
            state_names = [s.strip() for s in str(names_attr).split(",")]
        else:
            state_names = list(EMIT_STATE_FEATURE_NAMES)
        idx = np.arange(n_total, dtype=np.int64)
        X_spec = np.asarray(f["radiance"][:], dtype=np.float64)

    qoi = mapped_qois_s4(state, obs)
    missing = [n for n in names if n not in qoi]
    if missing:
        raise KeyError(f"mapped_qois_s4 missing {missing}")
    Y = np.column_stack([qoi[n] for n in names])

    if filter_valid_labels:
        valid = _valid_label_mask(Y, names)
        X_spec = X_spec[valid]
        Y = Y[valid]
        state = state[valid]
        obs = obs[valid]
        idx = idx[valid]
        if elev_all is not None:
            elev_all = elev_all[valid]

    if elev_all is None:
        raise KeyError(f"elevation missing in {emit_path} (required for ele_km aux input)")

    coszen, ele_km, raa_true = extract_s4_aux_inputs(obs, elev_all)
    X, aux_meta = append_s4_aux_inputs(X_spec, coszen, ele_km, raa_true)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in EMIT inputs")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values found in EMIT labels")

    meta = {
        "emit_path": str(emit_path.resolve()),
        "n_total": n_total,
        "n_filtered": int(idx.size),
        "filter_valid_labels": bool(filter_valid_labels),
        "state_feature_names": state_names,
        "task_names": names,
        "fsnow_label": "softmax_fraction(z_snow,z_pv,z_npv,z_soil)",
        "task_valid_y_range": {
            k: list(v) for k, v in TASK_VALID_Y_RANGE.items() if k in names
        },
        "input_variable": "radiance",
        **aux_meta,
        "input_dim": int(aux_meta["input_dim"]),
    }
    return X, Y, idx, meta


def split_emit_s4(
    n_filtered: int,
    *,
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded permutation: train, val, remaining test (no reserved TOA pools)."""
    if n_train < 0 or n_val < 0:
        raise ValueError(f"n_train and n_val must be >= 0, got {n_train}, {n_val}")
    if n_train + n_val >= n_filtered:
        raise ValueError(
            f"Need leftover test rows: n_train={n_train} + n_val={n_val} "
            f">= n_filtered={n_filtered}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_filtered)
    train_local = perm[:n_train]
    val_local = perm[n_train : n_train + n_val]
    test_local = perm[n_train + n_val :]
    return train_local, val_local, test_local


def _subsample_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    seed: int,
    max_points: int = SCATTER_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(y_true.shape[0])
    if max_points <= 0 or n <= max_points:
        return y_true, y_pred, lower, upper
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return y_true[idx], y_pred[idx], lower[idx], upper[idx]


__all__ = [
    "S4_AUX_INPUT_NAMES",
    "S4_INPUT_DIM",
    "S4_SPECTRAL_DIM",
    "S4_TASK_NAMES",
    "TASK_VALID_Y_RANGE",
    "_DEFAULT_EMIT_PATH",
    "_DEFAULT_WL_SRC",
    "_load_wavelengths_nm",
    "_subsample_scatter",
    "append_s4_aux_inputs",
    "calc_cos_i_from_state_obs",
    "extract_s4_aux_inputs",
    "load_emit_s4_xy",
    "mapped_qois_s4",
    "parse_s4_task_names",
    "split_emit_s4",
]
