"""S4 joint loader (radiance + geometry aux → S2-style QoIs).

Accepts either:
- processed EMIT NetCDF (``radiance`` / ``obs`` / ``state`` / ``elevation``), or
- snow-TOA simulation NetCDF (``toa_radiance`` + ``coszen`` / ``ele_km`` +
  named QoI variables).

X is always spectral (285) plus coszen, ele_km (RAA dropped; fixed in current
data). Also exports the shared loader/split used by independent S4 SORF.
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
    S4_BAND_CONFIG_ALIASES,
    S4_INPUT_DIM,
    S4_SPECTRAL_DIM,
    S4_TASK_NAMES,
)
from experiments_toa.s2_data import read_elevation_array

from experiments_toa.emit_geometry import (
    OBS_SZA,
    calc_cos_i_from_state_obs,
    elevation_to_km,
)

# NetCDF schemas accepted by ``load_emit_s4_xy``.
S4_SCHEMA_EMIT = "emit"
S4_SCHEMA_SNOW_TOA = "snow_toa"

TASK_VALID_Y_RANGE: dict[str, tuple[float, float]] = {
    "algae": (1e-2, 6e5),
    "aot": (0.04, 1.0),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return coszen, ele_km arrays (N,) each."""
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] < 9:
        raise ValueError(f"obs must be (N, >=9), got {obs.shape}")
    sza = obs[:, OBS_SZA]
    coszen = np.cos(np.radians(sza))
    if elevation is None:
        raise KeyError("S4 aux inputs require elevation in the NetCDF")
    ele_km = elevation_to_km(elevation)
    return coszen, ele_km


def append_s4_aux_inputs(
    X: np.ndarray,
    coszen: np.ndarray,
    ele_km: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Append coszen, ele_km after spectral columns."""
    n_spectral = int(X.shape[1])
    n = int(X.shape[0])
    for name, arr in (
        ("coszen", coszen),
        ("ele_km", ele_km),
    ):
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.shape[0] != n:
            raise ValueError(f"{name} length {a.shape[0]} != n_samples {n}")
    aux = np.column_stack([coszen, ele_km])
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
        "aot": base["aot"],
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


def detect_s4_nc_schema(h5_file) -> str:
    """Return ``emit`` or ``snow_toa`` from NetCDF/HDF5 keys."""
    keys = set(h5_file.keys())
    if {"radiance", "obs", "state"}.issubset(keys):
        return S4_SCHEMA_EMIT
    if "toa_radiance" in keys and set(S4_AUX_INPUT_NAMES).issubset(keys):
        return S4_SCHEMA_SNOW_TOA
    raise KeyError(
        "Unrecognized S4 NetCDF schema. Expected either "
        "(radiance, obs, state) [EMIT] or "
        f"(toa_radiance, {', '.join(S4_AUX_INPUT_NAMES)}) [snow TOA]. "
        f"Found keys: {sorted(keys)}"
    )


def _s4_snow_qoi_key(task_name: str) -> str:
    """Map S4 task name to snow-TOA NetCDF variable (``lwc`` → ``liquid_water``)."""
    return S4_BAND_CONFIG_ALIASES.get(task_name, task_name)


def _load_emit_s4_xy_emit(
    f,
    *,
    emit_path: Path,
    names: list[str],
    filter_valid_labels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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

    coszen, ele_km = extract_s4_aux_inputs(obs, elev_all)
    X, aux_meta = append_s4_aux_inputs(X_spec, coszen, ele_km)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in EMIT inputs")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values found in EMIT labels")

    meta = {
        "emit_path": str(emit_path.resolve()),
        "schema": S4_SCHEMA_EMIT,
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


def _load_emit_s4_xy_snow_toa(
    f,
    *,
    emit_path: Path,
    names: list[str],
    filter_valid_labels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_total = int(f["toa_radiance"].shape[0])
    n_bands = int(f["toa_radiance"].shape[1])
    if n_bands != S4_SPECTRAL_DIM:
        raise ValueError(f"Expected {S4_SPECTRAL_DIM} bands, got {n_bands}")
    X_spec = np.asarray(f["toa_radiance"][:], dtype=np.float64)
    coszen = np.asarray(f["coszen"][:], dtype=np.float64).reshape(-1)
    ele_km = np.asarray(f["ele_km"][:], dtype=np.float64).reshape(-1)
    for name, arr in (("coszen", coszen), ("ele_km", ele_km)):
        if arr.shape[0] != n_total:
            raise ValueError(f"{name} length {arr.shape[0]} != n_samples {n_total}")

    qoi_keys = {_s4_snow_qoi_key(n): n for n in names}
    missing_keys = [k for k in qoi_keys if k not in f]
    if missing_keys:
        raise KeyError(
            f"Missing snow-TOA QoI variables {missing_keys} in {emit_path} "
            f"(S4 tasks: {[qoi_keys[k] for k in missing_keys]})"
        )
    Y = np.column_stack(
        [np.asarray(f[_s4_snow_qoi_key(n)][:], dtype=np.float64).reshape(-1) for n in names]
    )
    idx = np.arange(n_total, dtype=np.int64)

    if filter_valid_labels:
        valid = _valid_label_mask(Y, names)
        X_spec = X_spec[valid]
        Y = Y[valid]
        coszen = coszen[valid]
        ele_km = ele_km[valid]
        idx = idx[valid]

    X, aux_meta = append_s4_aux_inputs(X_spec, coszen, ele_km)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in snow-TOA inputs")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values found in snow-TOA labels")

    meta = {
        "emit_path": str(emit_path.resolve()),
        "schema": S4_SCHEMA_SNOW_TOA,
        "n_total": n_total,
        "n_filtered": int(idx.size),
        "filter_valid_labels": bool(filter_valid_labels),
        "state_feature_names": None,
        "task_names": names,
        "qoi_variable_map": {n: _s4_snow_qoi_key(n) for n in names},
        "fsnow_label": "fsnow",
        "task_valid_y_range": {
            k: list(v) for k, v in TASK_VALID_Y_RANGE.items() if k in names
        },
        "input_variable": "toa_radiance",
        **aux_meta,
        "input_dim": int(aux_meta["input_dim"]),
    }
    return X, Y, idx, meta


def load_emit_s4_xy(
    emit_path: str | Path,
    *,
    task_names: Sequence[str],
    filter_valid_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load S4 inputs/labels from EMIT processed or snow-TOA simulation NetCDF.

    Supported schemas
    -----------------
    emit
        ``radiance`` + ``obs`` + ``state`` (+ ``elevation``); aux derived from
        geometry; QoIs from ``mapped_qois_s4``.
    snow_toa
        ``toa_radiance`` + ``coszen`` / ``ele_km``; QoIs as named variables
        (``lwc`` reads ``liquid_water``).

    X columns: spectral (285), coszen, ele_km (287 total).
    """
    import h5py

    emit_path = Path(emit_path)
    if not emit_path.is_file():
        raise FileNotFoundError(f"S4 NetCDF not found: {emit_path}")
    names = list(task_names)
    unknown = [n for n in names if n not in S4_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S4 task names: {unknown}")

    with h5py.File(emit_path, "r") as f:
        schema = detect_s4_nc_schema(f)
        if schema == S4_SCHEMA_EMIT:
            return _load_emit_s4_xy_emit(
                f,
                emit_path=emit_path,
                names=names,
                filter_valid_labels=filter_valid_labels,
            )
        return _load_emit_s4_xy_snow_toa(
            f,
            emit_path=emit_path,
            names=names,
            filter_valid_labels=filter_valid_labels,
        )


def split_emit_s4(
    n_filtered: int,
    *,
    n_train: int,
    n_val: int,
    seed: int,
    force_train_local: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded permutation: train, val, remaining test (no reserved TOA pools).

    ``force_train_local`` (optional): local row indices in the filtered array that
    must appear in the train split (e.g. ASD scenes appended to a sim NetCDF).
    """
    if n_train < 0 or n_val < 0:
        raise ValueError(f"n_train and n_val must be >= 0, got {n_train}, {n_val}")
    if n_train + n_val >= n_filtered:
        raise ValueError(
            f"Need leftover test rows: n_train={n_train} + n_val={n_val} "
            f">= n_filtered={n_filtered}"
        )
    if force_train_local is None:
        force = np.asarray([], dtype=np.int64)
    else:
        force = np.unique(np.asarray(force_train_local, dtype=np.int64))
    if force.size:
        if np.any(force < 0) or np.any(force >= n_filtered):
            raise ValueError(
                f"force_train_local out of range for n_filtered={n_filtered}: {force}"
            )
        if int(force.size) > int(n_train):
            raise ValueError(
                f"force_train_local has {force.size} rows but n_train={n_train}"
            )
    remaining = np.setdiff1d(np.arange(n_filtered, dtype=np.int64), force)
    rng = np.random.default_rng(seed)
    perm_rem = rng.permutation(remaining)
    n_force = int(force.size)
    n_fill = int(n_train) - n_force
    train_local = np.concatenate([force, perm_rem[:n_fill]])
    val_local = perm_rem[n_fill : n_fill + int(n_val)]
    test_local = perm_rem[n_fill + int(n_val) :]
    return train_local, val_local, test_local


def force_train_local_from_asd_appended(
    emit_path: str | Path,
    source_idx: np.ndarray,
) -> np.ndarray:
    """Local indices of NetCDF rows tagged by ``asd_appended_n`` (trailing rows)."""
    import h5py

    emit_path = Path(emit_path)
    with h5py.File(emit_path, "r") as f:
        n_add = int(f.attrs.get("asd_appended_n", 0) or 0)
        if "toa_radiance" in f:
            n_total = int(f["toa_radiance"].shape[0])
        elif "radiance" in f:
            n_total = int(f["radiance"].shape[0])
        else:
            n_total = int(np.max(source_idx) + 1) if source_idx.size else 0
    if n_add <= 0:
        return np.asarray([], dtype=np.int64)
    forced_src = set(range(n_total - n_add, n_total))
    local = [
        i for i, s in enumerate(np.asarray(source_idx).reshape(-1)) if int(s) in forced_src
    ]
    return np.asarray(local, dtype=np.int64)


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
    "S4_SCHEMA_EMIT",
    "S4_SCHEMA_SNOW_TOA",
    "S4_SPECTRAL_DIM",
    "S4_TASK_NAMES",
    "TASK_VALID_Y_RANGE",
    "_DEFAULT_EMIT_PATH",
    "_DEFAULT_WL_SRC",
    "_load_wavelengths_nm",
    "_subsample_scatter",
    "append_s4_aux_inputs",
    "calc_cos_i_from_state_obs",
    "detect_s4_nc_schema",
    "extract_s4_aux_inputs",
    "force_train_local_from_asd_appended",
    "load_emit_s4_xy",
    "mapped_qois_s4",
    "parse_s4_task_names",
    "split_emit_s4",
]
