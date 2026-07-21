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
    }
)

S2_INPUT_VARIABLES: tuple[str, ...] = ("toa_reflectance", "toa_radiance")

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
