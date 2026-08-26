"""Constants and defaults for the S2 11-QoI TOA dataset."""

from __future__ import annotations

from pathlib import Path

S2_INPUT_DIM = 285

S2_TASK_NAMES: tuple[str, ...] = (
    "algae",
    "aot",
    "cos_i",
    "cwv",
    "dust",
    "fNPV",
    "fPV",
    "fsnow",
    "fsoil",
    "grain_size",
    "liquid_water",
)

# Positive, heavy-tailed QoIs trained in log space (log-normal inverse at predict).
S2_LOG_SCALE_TASK_NAMES: frozenset[str] = frozenset(
    {
        "algae",
        "dust",
        "grain_size",
        "liquid_water",
        "lwc",
    }
)

# Additive offsets for log(y + C) warps (zero-inflated / near-zero QoIs).
# Bare log(y) is used when a task is absent from this map (effective C=0).
S2_LOG_OFFSETS: dict[str, float] = {
    "dust": 1.0,
}

# Default affine boxes for logit warps: map physical y in [a, b] to (0, 1) then logit.
# Match TOA design ranges from experiments_toa/toa_log_data.py (not raw [0, 1]).
S2_LOGIT_BOUNDS: dict[str, tuple[float, float]] = {
    "cos_i": (0.06, 1.0),
    "aot": (0.04, 1.0),
    "cwv": (0.2, 5.2),
    "fsnow": (0.0, 10.0),
    "fPV": (0.0, 10.0),
    "fNPV": (0.0, 10.0),
    "fsoil": (0.0, 10.0),
}

# ISOFIT / EMIT label floors and lower bounds for conditional metrics and
# off-floor training masks (y > threshold = "off floor / off bound").
S3_LABEL_FLOOR_THRESHOLDS: dict[str, float] = {
    "aot": 0.03,
    "dust": 1e-6,
    "cwv": 0.06,
}

# EMIT-appropriate logit boxes (ISOFIT physical ranges), not synthetic Sobol boxes.
S3_LOGIT_BOUNDS: dict[str, tuple[float, float]] = {
    "aot": (0.017, 0.61),
    "cwv": (0.05, 0.70),
}

# S4 EMIT: cos_i from obs, cwv from ISOFIT state (90–100% snow subset).
S4_LOGIT_BOUNDS: dict[str, tuple[float, float]] = {
    "cos_i": (0.06, 1.0),
    "cwv": (0.05, 0.70),
}

S4_TASK_NAMES: tuple[str, ...] = (
    "grain_size",
    "cos_i",
    "lwc",
    "dust",
    "algae",
    "fsnow",
    "fPV",
    "fNPV",
    "fsoil",
    "cwv",
)

S4_SPECTRAL_DIM = 285
S4_AUX_INPUT_NAMES: tuple[str, ...] = ("coszen", "ele_km", "RAA_TRUE")
S4_INPUT_DIM = S4_SPECTRAL_DIM + len(S4_AUX_INPUT_NAMES)

# Band-config JSON keys use liquid_water; S4 task name is lwc.
S4_BAND_CONFIG_ALIASES: dict[str, str] = {"lwc": "liquid_water"}

S2_INPUT_VARIABLES: tuple[str, ...] = ("toa_reflectance", "toa_radiance")

# Optional non-spectral model inputs (always kept; never in band drop lists).
# Stored as NetCDF variables and appended after the 285 spectral columns.
S2_AUX_INPUT_NAMES: tuple[str, ...] = ("elevation",)

# Absorption / low-SNR bands identified from the joint correlation analysis.
S2_DEFAULT_DROP_INDICES: tuple[int, ...] = (
    *range(131, 143),
    *range(191, 213),
    *range(278, 285),
)

# Retained band ranges after applying S2_DEFAULT_DROP_INDICES.
S2_DEFAULT_KEEP_RANGES: tuple[tuple[int, int], ...] = (
    (0, 130),
    (143, 190),
    (213, 277),
)

_S2_DATA_DIR = Path(__file__).resolve().parent / "data 11 QoI"
S2_DEFAULT_DATA_PATH = _S2_DATA_DIR / "snow_toa_simulations_20262107.nc"
S2_DEFAULT_BAND_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "s2_task_bands_default.json"
)
