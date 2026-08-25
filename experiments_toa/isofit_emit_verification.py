"""Multi-sample ISOFIT EMIT forward-model verification.

Expects a processed NetCDF (e.g. ``emit_test_data_processed_20262508.nc``) with:

data_vars
    radiance, reflectance, elevation, obs, state
coords
    sample, radiance_feature, reflectance_feature, elevation_feature,
    obs_feature, state_feature

For each selected sample: parse obs geometry, compare obs ``cosine i`` to
``calc_new_angles`` (sinA/cosA + SZA/SAA/slope), run the DISORT/MODTRAN forward
model with both cosi sources, and score simulated vs measured radiance and
reflectance.

Writes per-sample spectrum plots, a metrics table (CSV + markdown), and
aggregate RMSE / cosi summaries.

Softmax convention (critical)
-----------------------------
EMIT stores z_snow, z_pv, z_npv, z_soil as **logits**. The forward model applies
``softmax`` internally; pass logits directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_DATA = Path(r"C:/Users/tylerj/isofit/disort_data_for_tyler")
MODTRAN_PATH = str(_DATA / "lut.zarr")
DISORT_PATH = str(_DATA / "disort_snow_lut_EMIT.nc")
ENDMEMBER_PATH = str(_DATA / "endmembers.csv")
EMIT_WAVE_PATH = str(_DATA / "emit-wave.txt")
NOISE_PATH = str(_DATA / "emit_noise.txt")
DEFAULT_EMIT_NC = (
    _DATA / "data" / "emit_test_data_processed_20262508.nc"
)
DEFAULT_OUT = str(_DATA / "data" / "isofit_emit_verification")
COMMON_DIR: Path | None = None

EMIT_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "sinA",
    "cosA",
    "grain_radius",
    "liquid_water",
    "dust",
    "algae",
    "z_snow",
    "z_pv",
    "z_npv",
    "z_soil",
    "veg_rank",
    "npv_rank",
    "soil_rank",
    "AOT660",
    "H20STR",
)

# Canonical ISOFIT obs band order / names (case/spacing tolerant matching)
OBS_FEATURE_NAMES: tuple[str, ...] = (
    "path length",
    "to-sensor azimuth",
    "to-sensor zenith",
    "to-sun azimuth",
    "to-sun zenith",
    "phase",
    "slope",
    "aspect",
    "cosine i",
    "UTC time",
    "earth-sun distance",
)

EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
IDX_Z = tuple(EMIT_STATE_INDEX[n] for n in ("z_snow", "z_pv", "z_npv", "z_soil"))

OBS_PATH_LENGTH = 0
OBS_VAA = 1
OBS_VZA = 2
OBS_SAA = 3
OBS_SZA = 4
OBS_PHASE = 5
OBS_SLOPE = 6
OBS_ASPECT = 7
OBS_COS_I = 8
OBS_UTC = 9
OBS_ESD = 10

ELE_DEFAULT_KM = 3.0
SVF_DEFAULT = 1.0  # synthetic-sim fallback when slope/svf unavailable
NIR_TARGET_NM = 900.0
COS_I_CLIP_LO = 0.06

EMIT_COLOR = "#1f4e79"
SIM_OBS_COLOR = "#b85c38"
SIM_CALC_COLOR = "#2d6a4f"


def _ensure_common_import(common_dir: Path | None) -> None:
    dirs: list[Path] = []
    if common_dir is not None:
        dirs.append(common_dir)
    if COMMON_DIR is not None:
        dirs.append(COMMON_DIR)
    dirs.append(_DATA)
    for d in dirs:
        p = str(d)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from common import VectorInterpolator, calculate_resample_matrix  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Cannot import common.VectorInterpolator. Pass --common-dir to the "
            "folder that contains common.py, or add it to PYTHONPATH."
        ) from exc


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        x = obj.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _norm_feat_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _decode_feature_labels(raw: Any, n: int, fallback: tuple[str, ...]) -> list[str]:
    """Turn NetCDF coord / attr labels into string names; else use fallback order."""
    if raw is None:
        if n <= len(fallback):
            return list(fallback[:n])
        return [str(i) for i in range(n)]
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S", "O"):
        out: list[str] = []
        for x in arr.tolist():
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return out
    # integer indices -> fallback names
    if n <= len(fallback):
        return list(fallback[:n])
    return [str(i) for i in range(n)]


def _index_map(names: list[str], canonical: tuple[str, ...]) -> dict[str, int]:
    """Map canonical names -> column index via fuzzy match; else positional."""
    norm_to_i = {_norm_feat_name(n): i for i, n in enumerate(names)}
    out: dict[str, int] = {}
    for j, canon in enumerate(canonical):
        key = _norm_feat_name(canon)
        if key in norm_to_i:
            out[canon] = norm_to_i[key]
        elif j < len(names):
            out[canon] = j
    return out


# ---------------------------------------------------------------------------
# Geometry / cosi
# ---------------------------------------------------------------------------


@dataclass
class ObsGeometry:
    path_length_m: float
    observer_azimuth: float
    observer_zenith: float  # VZA
    solar_azimuth: float
    solar_zenith: float
    phase: float
    slope: float
    dem_aspect: float
    obs_cos_i: float
    utc: float
    earth_sun_distance: float | None
    relative_azimuth: float  # RAA_TRUE
    raa_disort: float  # RAA_DISORT = 180 - RAA_TRUE
    coszen: float
    svf: float  # SVF_TRUE (sky-view factor)
    svf_source: str


def relative_azimuth(saa: float, vaa: float) -> float:
    delta = abs(float(saa) - float(vaa))
    return float(min(delta, 360.0 - delta))


def svf_from_slope(slope_deg: float) -> float:
    """ISOFIT apply_oe skyview='slope' approximation: cos^2(slope/2)."""
    if not math.isfinite(slope_deg):
        return SVF_DEFAULT
    svf = float(np.cos(np.radians(float(slope_deg)) / 2.0) ** 2)
    return max(0.0, min(svf, 1.0))


def parse_obs_vector(
    obs: np.ndarray,
    *,
    col_map: dict[str, int] | None = None,
    svf_override: float | None = None,
) -> ObsGeometry:
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if obs.size < 9:
        raise ValueError(f"obs vector needs >= 9 bands, got {obs.size}")

    def get(canon: str, default_idx: int) -> float:
        if col_map is not None and canon in col_map:
            return float(obs[col_map[canon]])
        return float(obs[default_idx])

    saa = get("to-sun azimuth", OBS_SAA)
    vaa = get("to-sensor azimuth", OBS_VAA)
    sza = get("to-sun zenith", OBS_SZA)
    vza = get("to-sensor zenith", OBS_VZA)
    slope = get("slope", OBS_SLOPE)
    raa_true = relative_azimuth(saa, vaa)
    raa_disort = 180.0 - raa_true
    esd = None
    if col_map is not None and "earth-sun distance" in col_map:
        esd = float(obs[col_map["earth-sun distance"]])
    elif obs.size > OBS_ESD:
        esd = float(obs[OBS_ESD])

    if svf_override is not None and math.isfinite(svf_override):
        svf = float(max(0.0, min(svf_override, 1.0)))
        svf_source = "netcdf.svf"
    else:
        svf = svf_from_slope(slope)
        svf_source = "slope_cos2" if math.isfinite(slope) else "default"

    return ObsGeometry(
        path_length_m=get("path length", OBS_PATH_LENGTH),
        observer_azimuth=vaa,
        observer_zenith=vza,
        solar_azimuth=saa,
        solar_zenith=sza,
        phase=get("phase", OBS_PHASE) if obs.size > OBS_PHASE else float("nan"),
        slope=slope,
        dem_aspect=get("aspect", OBS_ASPECT),
        obs_cos_i=get("cosine i", OBS_COS_I),
        utc=get("UTC time", OBS_UTC) if obs.size > OBS_UTC else float("nan"),
        earth_sun_distance=esd,
        relative_azimuth=raa_true,
        raa_disort=raa_disort,
        coszen=float(np.cos(np.radians(sza))),
        svf=svf,
        svf_source=svf_source,
    )


def aspect_from_sin_cos(sin_a: float, cos_a: float) -> float:
    aspect = float(np.degrees(math.atan2(sin_a, cos_a)))
    if aspect < 0.0:
        aspect += 360.0
    return aspect


def cosi_from_angles(
    sza_deg: float,
    saa_deg: float,
    slope_deg: float,
    aspect_deg: float,
    *,
    clip: bool = True,
) -> float:
    sza = np.radians(sza_deg)
    slope = np.radians(slope_deg)
    aspect = np.radians(aspect_deg)
    cosi = float(
        np.sin(sza) * np.sin(slope) * np.cos(np.radians(saa_deg) - aspect)
        + np.cos(sza) * np.cos(slope)
    )
    if clip:
        cosi = max(COS_I_CLIP_LO, min(cosi, 1.0))
    return cosi


def calc_new_angles_cosi(
    sin_a: float,
    cos_a: float,
    sza_deg: float,
    saa_deg: float,
    slope_deg: float,
) -> tuple[float, float]:
    aspect_deg = aspect_from_sin_cos(sin_a, cos_a)
    cosi = cosi_from_angles(sza_deg, saa_deg, slope_deg, aspect_deg, clip=True)
    return aspect_deg, cosi


# ---------------------------------------------------------------------------
# NetCDF loader
# ---------------------------------------------------------------------------


@dataclass
class EmitDataset:
    path: Path
    n_samples: int
    sample_ids: np.ndarray
    state: np.ndarray
    reflectance: np.ndarray
    radiance: np.ndarray
    elevation_m: np.ndarray
    obs: np.ndarray
    svf: np.ndarray | None
    state_names: list[str]
    obs_names: list[str]
    obs_col_map: dict[str, int]
    state_col_map: dict[str, int]
    wavelengths: np.ndarray | None
    global_attrs: dict[str, Any]


def _read_coord_or_attr(f: h5py.File, *keys: str) -> Any | None:
    for k in keys:
        if k in f:
            return f[k][:]
        if k in f.attrs:
            return f.attrs[k]
    return None


def load_emit_dataset(nc_path: Path) -> EmitDataset:
    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {nc_path}")

    with h5py.File(nc_path, "r") as f:
        required = ("state", "reflectance", "radiance", "obs", "elevation")
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(
                f"{nc_path} missing data_vars {missing}; "
                f"found {sorted(f.keys())}"
            )
        state = np.asarray(f["state"][:], dtype=np.float64)
        reflectance = np.asarray(f["reflectance"][:], dtype=np.float64)
        radiance = np.asarray(f["radiance"][:], dtype=np.float64)
        obs = np.asarray(f["obs"][:], dtype=np.float64)
        elev = np.asarray(f["elevation"][:], dtype=np.float64).reshape(state.shape[0], -1)
        elevation_m = elev[:, 0]
        n = int(state.shape[0])
        sample_ids = (
            np.asarray(f["sample"][:], dtype=np.int64)
            if "sample" in f
            else np.arange(n, dtype=np.int64)
        )
        svf = None
        for svf_key in ("svf", "skyview_factor", "SVF_TRUE"):
            if svf_key in f:
                svf = np.asarray(f[svf_key][:], dtype=np.float64).reshape(n, -1)[:, 0]
                break
        global_attrs: dict[str, Any] = {}
        for ak in (
            "RAA_TRUE",
            "VZA",
            "VZA_TRUE",
            "RAA_DISORT",
            "SVF_TRUE",
            "skyview_factor",
        ):
            if ak in f.attrs:
                v = f.attrs[ak]
                if isinstance(v, (bytes, np.bytes_)):
                    global_attrs[ak] = v.decode("utf-8", errors="replace")
                else:
                    try:
                        global_attrs[ak] = float(v)
                    except (TypeError, ValueError):
                        global_attrs[ak] = v
        state_raw = _read_coord_or_attr(
            f, "state_feature", "state_feature_name", "state_feature_names"
        )
        obs_raw = _read_coord_or_attr(
            f, "obs_feature", "obs_feature_name", "obs_feature_names"
        )
        # attrs may store comma-separated names
        if isinstance(state_raw, (bytes, np.bytes_)):
            state_raw = state_raw.decode("utf-8").split(",")
        if isinstance(obs_raw, (bytes, np.bytes_)):
            obs_raw = obs_raw.decode("utf-8").split(",")
        if "state_feature_names" in f.attrs and state_raw is None:
            v = f.attrs["state_feature_names"]
            if isinstance(v, (bytes, np.bytes_)):
                state_raw = v.decode("utf-8").split(",")
        wavelengths = None
        for key in ("wavelength", "wavelengths", "wl"):
            if key in f:
                wavelengths = np.asarray(f[key][:], dtype=np.float64)
                break

    state_names = _decode_feature_labels(
        state_raw, state.shape[1], EMIT_STATE_FEATURE_NAMES
    )
    obs_names = _decode_feature_labels(obs_raw, obs.shape[1], OBS_FEATURE_NAMES)
    obs_col_map = _index_map(obs_names, OBS_FEATURE_NAMES)
    state_col_map = _index_map(state_names, EMIT_STATE_FEATURE_NAMES)

    if reflectance.shape[0] != n or radiance.shape[0] != n or obs.shape[0] != n:
        raise ValueError(
            f"Length mismatch: state={n}, rfl={reflectance.shape[0]}, "
            f"rdn={radiance.shape[0]}, obs={obs.shape[0]}"
        )
    return EmitDataset(
        path=nc_path,
        n_samples=n,
        sample_ids=sample_ids,
        state=state,
        reflectance=reflectance,
        radiance=radiance,
        elevation_m=elevation_m,
        obs=obs,
        svf=svf,
        state_names=state_names,
        obs_names=obs_names,
        obs_col_map=obs_col_map,
        state_col_map=state_col_map,
        wavelengths=wavelengths,
        global_attrs=global_attrs,
    )


def state_vector_for_forward(state_row: np.ndarray, col_map: dict[str, int]) -> np.ndarray:
    """Reorder / extract state into canonical EMIT_STATE_FEATURE_NAMES order."""
    state_row = np.asarray(state_row, dtype=np.float64).ravel()
    out = np.empty(len(EMIT_STATE_FEATURE_NAMES), dtype=np.float64)
    for i, name in enumerate(EMIT_STATE_FEATURE_NAMES):
        if name in col_map:
            out[i] = state_row[col_map[name]]
        elif i < state_row.size:
            out[i] = state_row[i]
        else:
            raise ValueError(f"Cannot find state feature {name}")
    return out


def state_to_forward_args(state: np.ndarray) -> dict[str, float]:
    state = np.asarray(state, dtype=np.float64).ravel()
    idx = EMIT_STATE_INDEX
    return {
        "grain": float(state[idx["grain_radius"]]),
        "lwc": float(state[idx["liquid_water"]]),
        "dust": float(state[idx["dust"]]),
        "algae": float(state[idx["algae"]]),
        "fsnow": float(state[idx["z_snow"]]),
        "fPV": float(state[idx["z_pv"]]),
        "fNPV": float(state[idx["z_npv"]]),
        "fsoil": float(state[idx["z_soil"]]),
        "cwv": float(state[idx["H20STR"]]),
        "aot": float(state[idx["AOT660"]]),
    }


def load_wavelengths(
    *,
    ds: EmitDataset,
    wl_path: Path | None,
    emit_wave_path: Path,
) -> np.ndarray:
    if wl_path is not None:
        specs = pd.read_csv(wl_path, sep=r"\s+", names=["idx", "wl", "fwhm"])
        return np.asarray(specs.wl.values, dtype=np.float64)
    if ds.wavelengths is not None:
        return np.asarray(ds.wavelengths, dtype=np.float64)
    specs = pd.read_csv(emit_wave_path, sep=r"\s+", names=["idx", "wl", "fwhm"])
    return np.asarray(specs.wl.values, dtype=np.float64)


def select_indices(
    n: int,
    *,
    indices: list[int] | None,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    if indices:
        idx = np.asarray(indices, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= n):
            raise IndexError(f"indices out of range for n={n}: {idx.tolist()}")
        return idx
    n_take = min(int(n_samples), n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_take, replace=False))


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


@dataclass
class ForwardModel:
    wl_mod: np.ndarray
    wl_emit: np.ndarray
    ds_dis_wl: np.ndarray
    h_matrix: np.ndarray
    endmembers: np.ndarray
    lut_coszen: float
    solar_irr: np.ndarray
    emit_noise: pd.DataFrame
    v_interp_latm: Any
    v_interp_sphalb: Any
    v_interp_lraw: dict[str, Any]
    v_interp_r_dd: Any
    v_interp_r_hd: Any

    def simulate_pixel(
        self,
        *,
        cos_i: float,
        ele_km: float,
        coszen: float,
        vza: float,
        raa: float,
        grain: float,
        lwc: float,
        dust: float,
        algae: float,
        fsnow: float,
        f_pv: float,
        f_npv: float,
        f_soil: float,
        cwv: float,
        aot: float,
        raa_disort: float | None = None,
        svf: float = 1.0,
        add_noise: bool = True,
        noise_rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward model using per-pixel VZA, RAA_TRUE, RAA_DISORT, SVF from obs.

        Parameters mirror synthetic-sim naming:
          vza         -> VZA / observer zenith (from obs)
          raa         -> RAA_TRUE (relative azimuth from obs SAA/VAA)
          raa_disort  -> RAA_DISORT (= 180 - RAA_TRUE); computed if None
          svf         -> SVF_TRUE (sky-view factor)
        """
        if raa_disort is None:
            raa_disort = 180.0 - float(raa)
        svf = float(max(0.0, min(svf, 1.0)))
        lookup_pt = np.array(
            [
                np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0))),
                vza,
                raa_disort,
                grain,
                algae,
                dust,
                lwc,
            ],
            dtype=np.float64,
        )
        rho_dd_22 = self.v_interp_r_dd(lookup_pt)
        rho_hd_22 = self.v_interp_r_hd(lookup_pt)
        rho_dd = np.interp(self.wl_mod, self.ds_dis_wl, rho_dd_22)
        rho_hd = np.interp(self.wl_mod, self.ds_dis_wl, rho_hd_22)

        f = softmax(np.array([fsnow, f_pv, f_npv, f_soil], dtype=np.float64))
        rho_dd = rho_dd * f[0] + np.dot(self.endmembers, f[1:])
        rho_hd = rho_hd * f[0] + np.dot(self.endmembers, f[1:])

        if not (0.0 < coszen <= 1.0):
            raise ValueError(f"coszen must be in (0, 1], got {coszen}")
        # MODTRAN observer_zenith axis is stored as (180 - VZA), same as sims.
        atm_pt = np.array([ele_km, 180.0 - vza, raa, aot, cwv], dtype=np.float64)
        l_atm = self.v_interp_latm(atm_pt)
        s_alb = self.v_interp_sphalb(atm_pt)
        l_raw = [
            self.v_interp_lraw[k](atm_pt)
            for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
        ]
        solar_irr_emit = np.dot(self.h_matrix, self.solar_irr)

        eq_11_term = 1.0 - (s_alb * rho_hd)
        l_dir_dir = (l_raw[0] / coszen) * cos_i
        l_dif_dir = l_raw[1] * (cos_i / coszen)
        # Diffuse illumination scaled by sky-view factor (ISOFIT geom.svf).
        l_dir_dif = l_raw[2] * svf
        l_dif_dif = l_raw[3] * svf
        l_tot = l_dir_dir + l_dif_dir + l_dir_dif + l_dif_dif
        l_dif_dir = l_dif_dir / eq_11_term
        l_dif_dif = l_dif_dif / eq_11_term
        atm_surface_scattering = s_alb * rho_hd

        toa_rdn = (
            l_atm
            + l_dir_dir * rho_dd
            + l_dif_dir * rho_hd
            + l_dir_dif * rho_hd
            + l_dif_dif * rho_hd
            + (l_tot * atm_surface_scattering * rho_hd) / eq_11_term
        )
        rdn_emit = np.dot(self.h_matrix, toa_rdn)
        if add_noise:
            rdn_out = apply_noise(
                rdn_emit, self.wl_emit, self.emit_noise, rng=noise_rng
            )
        else:
            rdn_out = rdn_emit
        toa_ref = rdn_out * np.pi / (solar_irr_emit * coszen)
        return rdn_out, toa_ref


def apply_noise(
    rdn: np.ndarray,
    wl: np.ndarray,
    noise_df: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    a = np.interp(wl, noise_df["wvl"], noise_df["a"])
    b = np.interp(wl, noise_df["wvl"], noise_df["b"])
    c = np.interp(wl, noise_df["wvl"], noise_df["c"])
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c
    sy = np.diagflat(np.power(nedl, 2))
    if rng is None:
        noise = np.random.multivariate_normal(np.zeros(rdn.shape), sy)
    else:
        noise = rng.multivariate_normal(np.zeros(rdn.shape), sy)
    return rdn + noise


def load_forward_model(*, wl_emit: np.ndarray) -> ForwardModel:
    from common import VectorInterpolator, calculate_resample_matrix

    ds_mod = xr.open_zarr(MODTRAN_PATH)
    wl_mod = np.asarray(ds_mod.wl.values, dtype=np.float64)
    target_dims = (
        "surface_elevation_km",
        "observer_zenith",
        "relative_azimuth",
        "AOT550",
        "H2OSTR",
        "wl",
    )
    modtran_grid = [
        ds_mod[k].values
        for k in [
            "surface_elevation_km",
            "observer_zenith",
            "relative_azimuth",
            "AOT550",
            "H2OSTR",
        ]
    ]
    v_interp_latm = VectorInterpolator(
        modtran_grid, ds_mod.rhoatm.transpose(*target_dims).values, version="mlg"
    )
    v_interp_sphalb = VectorInterpolator(
        modtran_grid, ds_mod.sphalb.transpose(*target_dims).values, version="mlg"
    )
    v_interp_lraw = {
        k: VectorInterpolator(
            modtran_grid, ds_mod[k].transpose(*target_dims).values, version="mlg"
        )
        for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
    }

    ds_dis = xr.load_dataset(DISORT_PATH)
    disort_grid = [
        ds_dis[k].values
        for k in ["sza", "vza", "raa", "grain_radius", "algae_conc", "dust_conc", "lwc"]
    ]
    v_interp_r_dd = VectorInterpolator(disort_grid, ds_dis.r_dd.values, version="mlg")
    v_interp_r_hd = VectorInterpolator(disort_grid, ds_dis.r_hd.values, version="mlg")

    emit_specs = pd.read_csv(EMIT_WAVE_PATH, sep=r"\s+", names=["idx", "wl", "fwhm"])
    emit_noise = pd.read_csv(
        NOISE_PATH, sep=r"\s+", names=["wvl", "a", "b", "c", "rmse"], comment="#"
    )
    h_matrix = calculate_resample_matrix(
        wl_mod, emit_specs.wl.values, emit_specs.fwhm.values
    )
    endmembers = np.array(pd.read_csv(ENDMEMBER_PATH))[:, 1:]

    if wl_emit.shape[0] != emit_specs.wl.shape[0]:
        raise ValueError(
            f"wl_emit length {wl_emit.shape[0]} != emit-wave bands {emit_specs.wl.shape[0]}"
        )

    return ForwardModel(
        wl_mod=wl_mod,
        wl_emit=wl_emit,
        ds_dis_wl=np.asarray(ds_dis.wavelength.values, dtype=np.float64),
        h_matrix=h_matrix,
        endmembers=endmembers,
        lut_coszen=float(ds_mod.coszen),
        solar_irr=np.asarray(ds_mod.solar_irr.values, dtype=np.float64),
        emit_noise=emit_noise,
        v_interp_latm=v_interp_latm,
        v_interp_sphalb=v_interp_sphalb,
        v_interp_lraw=v_interp_lraw,
        v_interp_r_dd=v_interp_r_dd,
        v_interp_r_hd=v_interp_r_hd,
    )


def spectrum_rmse(sim: np.ndarray, meas: np.ndarray) -> float:
    delta = np.asarray(sim, dtype=np.float64) - np.asarray(meas, dtype=np.float64)
    return float(np.sqrt(np.mean(delta**2)))


# ---------------------------------------------------------------------------
# Plots / tables
# ---------------------------------------------------------------------------


def _format_param_block(title: str, params: dict[str, Any]) -> str:
    lines = [title]
    for k, v in params.items():
        if isinstance(v, bool):
            lines.append(f"  {k}={v}")
        elif isinstance(v, (int, np.integer)):
            lines.append(f"  {k}={int(v)}")
        elif isinstance(v, (float, np.floating)):
            lines.append(f"  {k}={float(v):.5g}")
        else:
            lines.append(f"  {k}={v}")
    return "\n".join(lines)


def plot_sample_spectra(
    wl: np.ndarray,
    measured: np.ndarray,
    sim_obs: np.ndarray,
    sim_calc: np.ndarray,
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    params_measured: dict[str, Any],
    params_obs_cosi: dict[str, Any],
    params_calc_cosi: dict[str, Any],
) -> None:
    """Plot measured vs two sims, with all forward-model inputs annotated."""
    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[2.2, 1.0, 1.55], hspace=0.22
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")

    ax0.plot(wl, measured, color=EMIT_COLOR, lw=1.8, label="measured")
    ax0.plot(wl, sim_obs, color=SIM_OBS_COLOR, lw=1.4, ls="--", label="sim (obs cosi)")
    ax0.plot(
        wl, sim_calc, color=SIM_CALC_COLOR, lw=1.4, ls=":", label="sim (calc cosi)"
    )
    ax0.set_ylabel(ylabel)
    ax0.set_title(title)
    ax0.legend(frameon=False, fontsize=8)

    ax1.plot(wl, sim_obs - measured, color=SIM_OBS_COLOR, lw=1.2, label="sim(obs)-meas")
    ax1.plot(
        wl,
        sim_calc - measured,
        color=SIM_CALC_COLOR,
        lw=1.2,
        ls="--",
        label="sim(calc)-meas",
    )
    ax1.axhline(0.0, color="0.5", lw=0.8)
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("residual")
    ax1.legend(frameon=False, fontsize=8)

    col_meas = _format_param_block("MEASURED (pixel inputs)", params_measured)
    col_obs = _format_param_block("SIM (obs cos_i) FM inputs", params_obs_cosi)
    col_calc = _format_param_block("SIM (calc cos_i) FM inputs", params_calc_cosi)
    ax_txt.text(
        0.01,
        0.98,
        col_meas,
        transform=ax_txt.transAxes,
        fontsize=6.5,
        family="monospace",
        va="top",
        ha="left",
        color=EMIT_COLOR,
    )
    ax_txt.text(
        0.34,
        0.98,
        col_obs,
        transform=ax_txt.transAxes,
        fontsize=6.5,
        family="monospace",
        va="top",
        ha="left",
        color=SIM_OBS_COLOR,
    )
    ax_txt.text(
        0.67,
        0.98,
        col_calc,
        transform=ax_txt.transAxes,
        fontsize=6.5,
        family="monospace",
        va="top",
        ha="left",
        color=SIM_CALC_COLOR,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cosi_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(df["obs_cos_i"], df["calc_cos_i"], s=28, c=SIM_OBS_COLOR, alpha=0.85)
    lo = float(min(df["obs_cos_i"].min(), df["calc_cos_i"].min()))
    hi = float(max(df["obs_cos_i"].max(), df["calc_cos_i"].max()))
    pad = 0.02 * max(hi - lo, 1e-3)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, label="1:1")
    ax.set_xlabel("obs cosine i")
    ax.set_ylabel("calc_new_angles cosi")
    ax.set_title("obs vs calculated cosi")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_table_image(agg_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 1.2 + 0.35 * len(agg_df)))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(agg_df.values.astype(float), 6).tolist()
        if agg_df.select_dtypes(include=[np.number]).shape[1] == agg_df.shape[1]
        else [
            [
                f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row
            ]
            for row in agg_df.values.tolist()
        ],
        colLabels=list(agg_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Aggregate verification metrics", pad=12)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    # Prefer pandas to_markdown when tabulate is available; else plain pipe table.
    try:
        text = df.to_markdown(index=False, floatfmt=".6g")
    except (ImportError, ValueError):
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for row in df.itertuples(index=False):
            cells = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    cells.append(f"{float(v):.6g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        text = "\n".join(lines)
    path.write_text(text + "\n", encoding="utf-8")


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "VZA",
        "RAA_TRUE",
        "RAA_DISORT",
        "SVF_TRUE",
        "obs_cos_i",
        "calc_cos_i",
        "cos_i_abs_diff",
        "dem_formula_cos_i",
        "rfl_rmse_obs_cosi",
        "rfl_rmse_calc_cosi",
        "rdn_rmse_obs_cosi",
        "rdn_rmse_calc_cosi",
    ]
    rows = []
    for stat_name, fn in (
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("std", np.nanstd),
        ("min", np.nanmin),
        ("max", np.nanmax),
    ):
        row: dict[str, Any] = {"stat": stat_name}
        for c in numeric_cols:
            if c in df.columns:
                row[c] = float(fn(df[c].to_numpy(dtype=np.float64)))
        rows.append(row)
    row_match: dict[str, Any] = {"stat": "frac_cos_i_match"}
    for c in numeric_cols:
        row_match[c] = float("nan")
    if "cos_i_match" in df.columns:
        row_match["cos_i_abs_diff"] = float(np.mean(df["cos_i_match"].to_numpy()))
    rows.append(row_match)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(
    ds: EmitDataset,
    i: int,
    model: ForwardModel,
    wl: np.ndarray,
    *,
    ele_km_override: float | None,
    cos_i_atol: float,
    add_noise: bool,
    noise_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = state_vector_for_forward(ds.state[i], ds.state_col_map)
    svf_override = None
    svf_attr_source: str | None = None
    if ds.svf is not None:
        svf_override = float(ds.svf[i])
        svf_attr_source = "netcdf.svf"
    elif "SVF_TRUE" in ds.global_attrs:
        try:
            svf_override = float(ds.global_attrs["SVF_TRUE"])
            svf_attr_source = "netcdf.attr.SVF_TRUE"
        except (TypeError, ValueError):
            svf_override = None
    geom = parse_obs_vector(
        ds.obs[i], col_map=ds.obs_col_map, svf_override=svf_override
    )
    if svf_attr_source is not None:
        geom.svf_source = svf_attr_source
    # Prefer per-pixel obs geometry; allow NetCDF global attrs only as documentation
    # (obs always wins for VZA / RAA when present).
    vza = geom.observer_zenith
    raa_true = geom.relative_azimuth
    raa_disort = geom.raa_disort
    svf_true = geom.svf

    elev_m = float(ds.elevation_m[i])
    if ele_km_override is not None:
        ele_km = float(ele_km_override)
        ele_source = "cli --ele-km"
    elif math.isfinite(elev_m):
        # elevation stored in meters in processed EMIT files
        ele_km = elev_m / 1000.0 if elev_m > 20.0 else elev_m
        ele_source = "netcdf.elevation"
    else:
        ele_km = ELE_DEFAULT_KM
        ele_source = "default"

    sin_a = float(state[EMIT_STATE_INDEX["sinA"]])
    cos_a = float(state[EMIT_STATE_INDEX["cosA"]])
    inverted_aspect, calc_cos_i = calc_new_angles_cosi(
        sin_a, cos_a, geom.solar_zenith, geom.solar_azimuth, geom.slope
    )
    dem_formula_cos_i = cosi_from_angles(
        geom.solar_zenith, geom.solar_azimuth, geom.slope, geom.dem_aspect, clip=True
    )
    cos_i_abs_diff = abs(geom.obs_cos_i - calc_cos_i)
    cos_i_match = bool(cos_i_abs_diff <= cos_i_atol)

    fwd = state_to_forward_args(state)
    # Full kwargs that enter simulate_pixel (geometry + surface/atm state).
    base_fm = {
        "ele_km": ele_km,
        "coszen": geom.coszen,
        "VZA": vza,
        "RAA_TRUE": raa_true,
        "RAA_DISORT": raa_disort,
        "SVF_TRUE": svf_true,
        "grain": fwd["grain"],
        "lwc": fwd["lwc"],
        "dust": fwd["dust"],
        "algae": fwd["algae"],
        "fsnow": fwd["fsnow"],
        "fPV": fwd["fPV"],
        "fNPV": fwd["fNPV"],
        "fsoil": fwd["fsoil"],
        "cwv": fwd["cwv"],
        "aot": fwd["aot"],
        "add_noise": bool(add_noise),
    }
    # Measured: pixel inputs (obs cosi + same state/geometry used for sims).
    params_measured = {
        "cos_i": geom.obs_cos_i,
        **base_fm,
        "sza": geom.solar_zenith,
        "saa": geom.solar_azimuth,
        "slope": geom.slope,
        "sinA": sin_a,
        "cosA": cos_a,
        "svf_source": geom.svf_source,
        "ele_source": ele_source,
    }
    params_obs_cosi = {"cos_i": geom.obs_cos_i, **base_fm}
    params_calc_cosi = {"cos_i": calc_cos_i, **base_fm}

    # Independent RNGs so obs-cosi and calc-cosi sims get distinct noise draws,
    # but both are reproducible from --seed + sample index.
    rng_obs = np.random.default_rng(noise_seed + 17 * int(i) + 1) if add_noise else None
    rng_calc = np.random.default_rng(noise_seed + 17 * int(i) + 2) if add_noise else None

    common_kw = dict(
        ele_km=ele_km,
        coszen=geom.coszen,
        vza=vza,
        raa=raa_true,
        raa_disort=raa_disort,
        svf=svf_true,
        grain=fwd["grain"],
        lwc=fwd["lwc"],
        dust=fwd["dust"],
        algae=fwd["algae"],
        fsnow=fwd["fsnow"],
        f_pv=fwd["fPV"],
        f_npv=fwd["fNPV"],
        f_soil=fwd["fsoil"],
        cwv=fwd["cwv"],
        aot=fwd["aot"],
        add_noise=add_noise,
    )
    sim_rdn_obs, sim_rfl_obs = model.simulate_pixel(
        cos_i=geom.obs_cos_i, noise_rng=rng_obs, **common_kw
    )
    sim_rdn_calc, sim_rfl_calc = model.simulate_pixel(
        cos_i=calc_cos_i, noise_rng=rng_calc, **common_kw
    )

    meas_rfl = ds.reflectance[i]
    meas_rdn = ds.radiance[i]
    row = {
        "nc_index": int(i),
        "sample_id": int(ds.sample_ids[i]),
        "ele_km": ele_km,
        "ele_source": ele_source,
        "sza": geom.solar_zenith,
        "saa": geom.solar_azimuth,
        "VZA": vza,
        "RAA_TRUE": raa_true,
        "RAA_DISORT": raa_disort,
        "SVF_TRUE": svf_true,
        "svf_source": geom.svf_source,
        "slope": geom.slope,
        "dem_aspect": geom.dem_aspect,
        "inverted_aspect": inverted_aspect,
        "sinA": sin_a,
        "cosA": cos_a,
        "obs_cos_i": geom.obs_cos_i,
        "calc_cos_i": calc_cos_i,
        "dem_formula_cos_i": dem_formula_cos_i,
        "cos_i_abs_diff": cos_i_abs_diff,
        "cos_i_match": cos_i_match,
        "rfl_rmse_obs_cosi": spectrum_rmse(sim_rfl_obs, meas_rfl),
        "rfl_rmse_calc_cosi": spectrum_rmse(sim_rfl_calc, meas_rfl),
        "rdn_rmse_obs_cosi": spectrum_rmse(sim_rdn_obs, meas_rdn),
        "rdn_rmse_calc_cosi": spectrum_rmse(sim_rdn_calc, meas_rdn),
        "params_measured": params_measured,
        "params_obs_cosi": params_obs_cosi,
        "params_calc_cosi": params_calc_cosi,
    }
    spectra = {
        "meas_rfl": meas_rfl,
        "meas_rdn": meas_rdn,
        "sim_rfl_obs": sim_rfl_obs,
        "sim_rfl_calc": sim_rfl_calc,
        "sim_rdn_obs": sim_rdn_obs,
        "sim_rdn_calc": sim_rdn_calc,
    }
    return row, spectra, params_measured, params_obs_cosi, params_calc_cosi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    global MODTRAN_PATH, DISORT_PATH, ENDMEMBER_PATH, EMIT_WAVE_PATH, NOISE_PATH

    if args.modtran_path:
        MODTRAN_PATH = str(args.modtran_path)
    if args.disort_path:
        DISORT_PATH = str(args.disort_path)
    if args.endmember_path:
        ENDMEMBER_PATH = str(args.endmember_path)
    if args.emit_wave_path:
        EMIT_WAVE_PATH = str(args.emit_wave_path)
    if args.noise_path:
        NOISE_PATH = str(args.noise_path)

    _ensure_common_import(args.common_dir)

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    emit_nc = Path(args.emit_nc)
    print(f"Loading {emit_nc} ...")
    ds = load_emit_dataset(emit_nc)
    print(
        f"  n={ds.n_samples}  state_dim={ds.state.shape[1]}  obs_dim={ds.obs.shape[1]}  "
        f"bands={ds.reflectance.shape[1]}"
    )
    print(f"  obs_features={ds.obs_names}")
    print(f"  state_features={ds.state_names}")
    if ds.svf is not None:
        print("  per-sample SVF array found in NetCDF")
    elif "SVF_TRUE" in ds.global_attrs:
        print(f"  global SVF_TRUE attr={ds.global_attrs['SVF_TRUE']}")
    else:
        print("  SVF from slope: cos^2(slope/2) per sample")
    if ds.global_attrs:
        print(f"  geometry attrs={ds.global_attrs}")

    wl = load_wavelengths(
        ds=ds,
        wl_path=Path(args.wl) if args.wl else None,
        emit_wave_path=Path(EMIT_WAVE_PATH),
    )
    if ds.reflectance.shape[1] != wl.shape[0]:
        raise ValueError(
            f"reflectance bands {ds.reflectance.shape[1]} != wl {wl.shape[0]}"
        )

    indices = select_indices(
        ds.n_samples,
        indices=args.indices,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
    )
    n_plot = min(int(args.n_plot), len(indices))
    print(f"Evaluating {len(indices)} samples (plotting {n_plot})...")
    print(f"Instrument noise: {'ON' if args.add_noise else 'OFF'}")

    print("Loading forward model LUTs...")
    model = load_forward_model(wl_emit=wl)

    rows: list[dict[str, Any]] = []
    for k, i in enumerate(indices):
        row, spectra, params_measured, params_obs_cosi, params_calc_cosi = evaluate_sample(
            ds,
            int(i),
            model,
            wl,
            ele_km_override=float(args.ele_km) if args.ele_km is not None else None,
            cos_i_atol=float(args.cos_i_atol),
            add_noise=bool(args.add_noise),
            noise_seed=int(args.seed),
        )
        rows.append(row)
        if k < n_plot:
            sid = row["sample_id"]
            plot_sample_spectra(
                wl,
                spectra["meas_rfl"],
                spectra["sim_rfl_obs"],
                spectra["sim_rfl_calc"],
                ylabel="TOA reflectance",
                title=(
                    f"sample_id={sid}  rfl RMSE obs={row['rfl_rmse_obs_cosi']:.5f}  "
                    f"calc={row['rfl_rmse_calc_cosi']:.5f}"
                ),
                out_path=plots_dir / f"sample_{sid}_reflectance.png",
                params_measured=params_measured,
                params_obs_cosi=params_obs_cosi,
                params_calc_cosi=params_calc_cosi,
            )
            plot_sample_spectra(
                wl,
                spectra["meas_rdn"],
                spectra["sim_rdn_obs"],
                spectra["sim_rdn_calc"],
                ylabel="TOA radiance",
                title=(
                    f"sample_id={sid}  rdn RMSE obs={row['rdn_rmse_obs_cosi']:.5f}  "
                    f"calc={row['rdn_rmse_calc_cosi']:.5f}"
                ),
                out_path=plots_dir / f"sample_{sid}_radiance.png",
                params_measured=params_measured,
                params_obs_cosi=params_obs_cosi,
                params_calc_cosi=params_calc_cosi,
            )
        if (k + 1) == 1 or (k + 1) == len(indices) or (k + 1) % 10 == 0:
            print(f"  done {k + 1}/{len(indices)}", flush=True)

    df = pd.DataFrame(rows)
    # Nested param dicts are for plots/JSON; keep flat metrics for CSV/MD tables.
    param_cols = ["params_measured", "params_obs_cosi", "params_calc_cosi"]
    df_flat = df.drop(columns=[c for c in param_cols if c in df.columns])
    agg = aggregate_metrics(df_flat)

    df_flat.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    agg.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    write_markdown_table(df_flat, out_dir / "per_sample_metrics.md")
    write_markdown_table(agg, out_dir / "aggregate_metrics.md")
    plot_cosi_scatter(df_flat, out_dir / "cosi_obs_vs_calc.png")
    plot_aggregate_table_image(agg, out_dir / "aggregate_metrics_table.png")

    summary = {
        "emit_nc": str(emit_nc),
        "n_evaluated": int(len(df)),
        "n_plot": n_plot,
        "add_noise": bool(args.add_noise),
        "cos_i_atol": float(args.cos_i_atol),
        "obs_features": ds.obs_names,
        "state_features": ds.state_names,
        "aggregate": agg.to_dict(orient="records"),
        "per_sample": rows,
    }
    with open(out_dir / "verification.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print("\nAggregate metrics:")
    print(agg.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emit-nc",
        type=Path,
        default=DEFAULT_EMIT_NC,
        help="processed EMIT NetCDF (radiance/reflectance/elevation/obs/state)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=330893,
        help="number of random samples to evaluate (ignored if --indices set)",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="explicit NetCDF row indices to evaluate",
    )
    p.add_argument(
        "--n-plot",
        type=int,
        default=100,
        help="how many samples get radiance/reflectance comparison plots",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ele-km",
        type=float,
        default=None,
        help="override elevation (km); default uses NetCDF elevation",
    )
    p.add_argument("--wl", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument(
        "--add-noise",
        dest="add_noise",
        action="store_true",
        default=True,
        help="add EMIT parametric instrument noise to simulated radiance (default: on)",
    )
    p.add_argument(
        "--no-noise",
        dest="add_noise",
        action="store_false",
        help="disable instrument noise on simulated radiance",
    )
    p.add_argument("--cos-i-atol", type=float, default=1e-3)
    p.add_argument("--common-dir", type=Path, default=None)
    p.add_argument("--modtran-path", type=Path, default=None)
    p.add_argument("--disort-path", type=Path, default=None)
    p.add_argument("--endmember-path", type=Path, default=None)
    p.add_argument("--emit-wave-path", type=Path, default=None)
    p.add_argument("--noise-path", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
