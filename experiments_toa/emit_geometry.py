"""EMIT obs/state geometry helpers (cos_i, coszen, RAA, elevation)."""

from __future__ import annotations

import numpy as np

from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES

# ISOFIT obs column defaults (canonical order).
OBS_VAA = 1
OBS_SAA = 3
OBS_SZA = 4
OBS_SLOPE = 6

EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
COS_I_CLIP_LO = 0.06


def relative_azimuth(saa: float, vaa: float) -> float:
    delta = abs(float(saa) - float(vaa))
    return float(min(delta, 360.0 - delta))


def elevation_to_km(elevation: np.ndarray) -> np.ndarray:
    elev = np.asarray(elevation, dtype=np.float64).reshape(-1)
    out = np.empty_like(elev)
    out[elev > 20.0] = elev[elev > 20.0] / 1000.0
    out[elev <= 20.0] = elev[elev <= 20.0]
    return out


def calc_cos_i_from_state_obs(state: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Vectorized ``calc_new_angles_cosi`` (matches isofit_emit_verification)."""
    state = np.asarray(state, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    sin_a = state[:, EMIT_STATE_INDEX["sinA"]]
    cos_a = state[:, EMIT_STATE_INDEX["cosA"]]
    sza = obs[:, OBS_SZA]
    saa = obs[:, OBS_SAA]
    slope = obs[:, OBS_SLOPE]
    aspect = np.degrees(np.arctan2(sin_a, cos_a))
    aspect = np.where(aspect < 0.0, aspect + 360.0, aspect)
    sza_rad = np.radians(sza)
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    cosi = (
        np.sin(sza_rad) * np.sin(slope_rad) * np.cos(np.radians(saa) - aspect_rad)
        + np.cos(sza_rad) * np.cos(slope_rad)
    )
    return np.clip(cosi, COS_I_CLIP_LO, 1.0)
