import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

# NOTE: This is the nice interpolator Jouni built
from common import VectorInterpolator, calculate_resample_matrix

# ToDo: update these paths
# Atmosphere LUT (MODTRAN / sRTMnet). NetCDF, not the snow-surface DISORT LUT.
MODTRAN_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/modtran_blue_hook_v3.nc"
# Snow-surface BRDF LUT (grain / algae / dust / lwc / viewing geometry).
DISORT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/disort_snow_lut_EMIT.nc"
ENDMEMBER_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/endmembers.csv"
EMIT_WAVE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit-wave.txt"
NOISE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit_noise.txt"
OUTPUT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/data/snow_toa_fsnow_90to100_20260209.nc"

SVF_TRUE = 1.0
VZA_TRUE = 0.0

# # NOTE: background reflectance should be assumed equal to the target pixel
# per convo with Niklas today.
BG_RFL = "None"

# Column order: cos_i, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot,
#               ele_km, RAA_TRUE, sza, slope, saa_minus_aspect
# cos_i is DERIVED from (sza, slope, saa_minus_aspect); coszen = cos(sza).
# Set LO==HI on any geometry dim to fix it (same pattern as RAA).
# Zero-floored free dims (LWC / dust / algae): mixture of log-uniform and
# linear-uniform decode. MIX_LOG_WEIGHT is P(log); 1.0 = pure log, 0.0 = pure lin.
# Small positive floors keep log-space defined.
N_SAMPLES = 120000
SOBOL_SEED = 42
MIX_LOG_WEIGHT = 0.2
GRAIN_DUST_THRESHOLD = 400.0

COSI_LO, COSI_HI = 0.06, 1.0
GRAIN_LO, GRAIN_HI = 30.0, 1500.0
# Design windows still span [0, hi] physically; log decode needs lo > 0.
LWC_LO, LWC_HI = 1e-2, 25.0
DUST_LO, DUST_HI = 1e-2, 1000.0
ALGAE_LO, ALGAE_HI = 1e-2, 1e5
# When grain_size < GRAIN_DUST_THRESHOLD, redraw impurities on a low log-uniform band
# (ASD-like) instead of pinning to the global floor.
FINE_DUST_LO, FINE_DUST_HI = 1.0, 200.0
FINE_ALGAE_LO, FINE_ALGAE_HI = 1.0, 100.0
FSNOW_LO, FSNOW_HI = 0.9, 1.0  # fsnow-only
FPV_LO, FPV_HI = 0.0, 0.1
FNPV_LO, FNPV_HI = 0.0, 0.1
FSOIL_LO, FSOIL_HI = 0.0, 0.1
# Atmosphere LUT (modtran_blue_hook_v3.nc) design window.
# Keep design windows inside the LUT axis maxima (not 70.01 / 0.5501).
CWV_LO, CWV_HI = 0.05, 0.55  # H2OSTR
AOT_LO, AOT_HI = 0.01, 0.07  # AOT550
ELE_LO, ELE_HI = 0.0, 5.0  # km
RAA_LO, RAA_HI = 0.0, 0.0  # deg (RAA_TRUE); LUT is nadir/RAA=0
SZA_LO, SZA_HI = 0.0, 70.0  # deg (solar zenith)
SLOPE_LO, SLOPE_HI = 0.0, 70.0  # deg (surface slope; non-flat branch)
# ASD-like flat terrain: P(slope in [FLAT_SLOPE_LO, FLAT_SLOPE_HI]) = FLAT_SLOPE_WEIGHT.
FLAT_SLOPE_WEIGHT = 0.15
FLAT_SLOPE_LO, FLAT_SLOPE_HI = 0.0, 2.0  # deg
# Relative solar-aspect angle: SAA - aspect (deg). cos() is even & 360-periodic.
SAA_ASPECT_LO, SAA_ASPECT_HI = 0.0, 360.0

LOG_UNIFORM_QOIS = ("liquid_water", "dust", "algae", "grain_size")
# Task: cos_i (derived) + 10 QoIs + ele/RAA/sza/slope/saa_aspect.
# Sobol free dims: slope + 10 QoIs + ele/RAA/sza/saa_aspect + mix gates (no cos_i).
N_TASK_DIMS = 16
N_FREE_SOBOL = 15  # slope, grain..aot (10), ele, raa, sza, saa_aspect
N_MIX_GATES = 4  # flat-slope + lwc/dust/algae
N_SOBOL_DIMS = N_FREE_SOBOL + N_MIX_GATES  # 19
COL_ELE = 11
COL_RAA = 12
COL_SZA = 13
COL_SLOPE = 14
COL_SAA_ASPECT = 15
# Sobol u columns (not task columns):
U_SLOPE = 0
U_SAA_ASPECT = 14
MIX_GATE_SLOPE = 15
MIX_GATE_LWC = 16
MIX_GATE_DUST = 17
MIX_GATE_ALGAE = 18

# Preferred atmosphere interpolator axis order. Only names present on the LUT
# are used, so old lut.zarr (observer_zenith) and blue_hook (solar_zenith) both work.
_ELE_DIM_CANDIDATES = ("surface_elevation_km", "ele_km", "elevation_km")
_OBS_DIM_CANDIDATES = ("observer_zenith",)
_SZA_DIM_CANDIDATES = ("solar_zenith", "sza", "SZA")
_RAA_DIM_CANDIDATES = ("relative_azimuth",)
_AOT_DIM_CANDIDATES = ("AOT550",)
_H2O_DIM_CANDIDATES = ("H2OSTR",)
_WL_DIM_CANDIDATES = ("wl", "wavelength")
_MODTRAN_AXIS_GROUPS = (
    _ELE_DIM_CANDIDATES,
    _OBS_DIM_CANDIDATES,
    _SZA_DIM_CANDIDATES,
    _RAA_DIM_CANDIDATES,
    _AOT_DIM_CANDIDATES,
    _H2O_DIM_CANDIDATES,
)


def _first_present(names, available):
    for name in names:
        if name in available:
            return name
    return None


def _wavelengths_to_nm(wl, *, label: str) -> np.ndarray:
    """EMIT/DISORT are in nm; some MODTRAN LUTs store µm."""
    wl = np.asarray(wl, dtype=np.float64).reshape(-1)
    wmax = float(np.nanmax(wl)) if wl.size else float("nan")
    if np.isfinite(wmax) and wmax < 20.0:
        print(f"  {label}: converting wavelengths µm → nm (max was {wmax:g})")
        return wl * 1000.0
    return wl


def _clip_query_to_grid(values, grids, names):
    """Clip interpolator queries into LUT bounds (OOB → NaNs with VectorInterpolator)."""
    out = np.asarray(values, dtype=np.float64).copy()
    msgs = []
    for i, (val, grid, name) in enumerate(zip(out, grids, names)):
        g = np.asarray(grid, dtype=np.float64).reshape(-1)
        lo, hi = float(np.min(g)), float(np.max(g))
        clipped = float(np.clip(val, lo, hi))
        if clipped != float(val):
            msgs.append(f"{name}:{val:g}->{clipped:g}")
        out[i] = clipped
    return out, msgs


def _snap_query_to_nearest_grid(values, grids):
    """Snap each axis to the nearest LUT node."""
    out = np.asarray(values, dtype=np.float64).copy()
    for i, grid in enumerate(grids):
        g = np.asarray(grid, dtype=np.float64).reshape(-1)
        out[i] = float(g[int(np.argmin(np.abs(g - out[i])))])
    return out


def _fill_nan_spectra_nearest(arr):
    """Fill all-NaN spectra at grid nodes from the nearest finite node.

    Atmosphere LUTs often leave ~1% of geometry nodes unset. Multilinear
    interpolation through any NaN corner returns an all-NaN spectrum.
    """
    from scipy.ndimage import distance_transform_edt

    out = np.array(arr, dtype=np.float64, copy=True)
    if out.ndim < 2:
        raise ValueError(f"expected (*grid, wl), got shape {out.shape}")
    spatial = out.shape[:-1]
    flat = out.reshape(-1, out.shape[-1])
    valid_flat = np.all(np.isfinite(flat), axis=1)
    n_bad = int(np.count_nonzero(~valid_flat))
    if n_bad == 0:
        return out, 0
    if not np.any(valid_flat):
        raise ValueError("atmosphere LUT has no finite spectra to fill from")
    valid = valid_flat.reshape(spatial)
    nan_mask = ~valid
    # Distance to nearest False (= valid) along spatial dims; indices of that node.
    indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    nearest = out[tuple(indices)]
    out[nan_mask] = nearest[nan_mask]
    return out, n_bad


def _finite_frac(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.isfinite(x)))


def select_modtran_axes(ds, data_var="rhoatm"):
    """LUT spatial axes in a stable order, skipping names the file does not have."""
    available = set(ds[data_var].dims)
    axes = []
    for group in _MODTRAN_AXIS_GROUPS:
        name = _first_present(group, available)
        if name is not None:
            axes.append(name)
    return axes


def build_atm_pt(
    ele_km,
    raa_true,
    aot,
    cwv,
    sza=None,
    *,
    vza_true=None,
    axes,
):
    """Atmosphere LUT query vector, one value per ``axes`` name (LUT dim order)."""
    if vza_true is None:
        vza_true = VZA_TRUE
    values = {
        "surface_elevation_km": ele_km,
        "ele_km": ele_km,
        "elevation_km": ele_km,
        "observer_zenith": 180.0 - float(vza_true),
        "relative_azimuth": raa_true,
        "AOT550": aot,
        "H2OSTR": cwv,
        "solar_zenith": sza,
        "sza": sza,
        "SZA": sza,
    }
    missing = [ax for ax in axes if ax not in values]
    if missing:
        raise KeyError(f"No atm_pt mapping for LUT axes {missing}")
    return np.array([values[ax] for ax in axes], dtype=np.float64)


def lin_map(u, lo, hi):
    return u * (hi - lo) + lo


def log_map(u, lo, hi):
    """Map u in [0, 1] -> log-uniform physical values on [lo, hi]."""
    if not (lo > 0.0 and hi > lo):
        raise ValueError(f"log_map requires 0 < lo < hi, got lo={lo}, hi={hi}")
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    return np.power(10.0, u * (log_hi - log_lo) + log_lo)


def mix_map(u_val, u_gate, lo, hi, w):
    """Mixture decode: P(log-uniform)=w, else linear-uniform (same [lo, hi])."""
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"MIX_LOG_WEIGHT must be in [0, 1], got {w}")
    if w >= 1.0:
        return log_map(u_val, lo, hi)
    if w <= 0.0:
        return lin_map(u_val, lo, hi)
    use_log = u_gate < w
    out = np.empty_like(u_val, dtype=np.float64)
    if np.any(use_log):
        out[use_log] = log_map(u_val[use_log], lo, hi)
    if np.any(~use_log):
        out[~use_log] = lin_map(u_val[~use_log], lo, hi)
    return out


def atm_midlat_winter_cwv_upperbound(altitude_km):
    """ISOFIT ModtranRT.modtran_water_upperbound_polynomials()['ATM_MIDLAT_WINTER'].

    altitude_km is ground altitude in km (elev_m / 1000 when elevation is meters).
    """
    z = np.asarray(altitude_km, dtype=np.float64)
    return np.maximum(
        1.371226
        + (-0.442087 * z)
        + (4.485325e-02 * z**2)
        + (-1.130163e-03 * z**3),
        0.25,
    )


def atm_midlat_winter_aot_lowerbound(altitude_km):
    """ISOFIT ModtranRT.modtran_aot_lowerbound_polynomials()['ATM_MIDLAT_WINTER'].

    altitude_km is ground altitude in km (elev_m / 1000 when elevation is meters).
    """
    z = np.asarray(altitude_km, dtype=np.float64)
    return (
        0.024748
        + (-0.001654 * z)
        + (-5.083805e-07 * z**2)
        + (7.252007e-08 * z**3)
    )


def cos_i_from_geometry(sza_deg, slope_deg, saa_minus_aspect_deg):
    """ISOFIT local incidence: cos_i from SZA, slope, and (SAA - aspect).

    Matches ``experiments_toa.emit_geometry.calc_cos_i_from_state_obs`` /
    ``isofit_emit_verification.cosi_from_angles`` (without clipping).
    """
    sza = np.radians(np.asarray(sza_deg, dtype=np.float64))
    slope = np.radians(np.asarray(slope_deg, dtype=np.float64))
    delta = np.radians(np.asarray(saa_minus_aspect_deg, dtype=np.float64))
    return np.sin(sza) * np.sin(slope) * np.cos(delta) + np.cos(sza) * np.cos(slope)


def slope_mix_map(u_val, u_gate, *, flat_weight=FLAT_SLOPE_WEIGHT):
    """Mixture slope decode: P(flat ASD-like)=flat_weight, else non-flat range.

    Flat branch: uniform on [FLAT_SLOPE_LO, FLAT_SLOPE_HI].
    Non-flat branch: uniform on [FLAT_SLOPE_HI, SLOPE_HI] when that interval is
    non-empty, else [SLOPE_LO, SLOPE_HI].
    """
    if not (0.0 <= flat_weight <= 1.0):
        raise ValueError(f"FLAT_SLOPE_WEIGHT must be in [0, 1], got {flat_weight}")
    u_val = np.asarray(u_val, dtype=np.float64)
    u_gate = np.asarray(u_gate, dtype=np.float64)
    out = np.empty_like(u_val, dtype=np.float64)
    use_flat = u_gate < flat_weight
    if np.any(use_flat):
        out[use_flat] = lin_map(u_val[use_flat], FLAT_SLOPE_LO, FLAT_SLOPE_HI)
    if np.any(~use_flat):
        nonflat_lo = FLAT_SLOPE_HI if FLAT_SLOPE_HI < SLOPE_HI else SLOPE_LO
        out[~use_flat] = lin_map(u_val[~use_flat], nonflat_lo, SLOPE_HI)
    return out


def decode_sobol_tasks(u, mix_log_weight=MIX_LOG_WEIGHT):
    """
    Decode Sobol u in [0,1]^{19} to physical tasks (n, 16).

    Free Sobol dims: slope, grain..aot, ele_km, RAA_TRUE, sza, saa_minus_aspect,
    then mix gates for flat-slope, liquid_water, dust, algae.
    Column 0 (cos_i) is derived from sza, slope, saa_minus_aspect.
    """
    if u.ndim != 2 or u.shape[1] != N_SOBOL_DIMS:
        raise ValueError(f"expected u shape (n, {N_SOBOL_DIMS}), got {u.shape}")
    tasks = np.empty((u.shape[0], N_TASK_DIMS), dtype=np.float64)
    slope = slope_mix_map(u[:, U_SLOPE], u[:, MIX_GATE_SLOPE])
    tasks[:, 1] = lin_map(u[:, 1], GRAIN_LO, GRAIN_HI)
    tasks[:, 2] = mix_map(u[:, 2], u[:, MIX_GATE_LWC], LWC_LO, LWC_HI, mix_log_weight)
    tasks[:, 3] = mix_map(u[:, 3], u[:, MIX_GATE_DUST], DUST_LO, DUST_HI, mix_log_weight)
    tasks[:, 4] = mix_map(u[:, 4], u[:, MIX_GATE_ALGAE], ALGAE_LO, ALGAE_HI, mix_log_weight)
    # Fine grains: low log-uniform dust/algae (reuse same Sobol coords, truncated range).
    fine = tasks[:, 1] < GRAIN_DUST_THRESHOLD
    if np.any(fine):
        tasks[fine, 3] = log_map(u[fine, 3], FINE_DUST_LO, FINE_DUST_HI)
        tasks[fine, 4] = log_map(u[fine, 4], FINE_ALGAE_LO, FINE_ALGAE_HI)

    # Fix fractional snow cover to between 70-100%
    u_snow = u[:, 5]
    u_pv = u[:, 6]
    u_npv = u[:, 7]
    u_soil = u[:, 8]

    fsnow = lin_map(u_snow, FSNOW_LO, FSNOW_HI)
    f_rest = 1.0 - fsnow

    raw = np.stack([u_pv, u_npv, u_soil], axis=1)
    raw_sum = np.sum(raw, axis=1, keepdims=True)
    raw_norm = raw / raw_sum  # normalize to sum=1
    fpv, fnpv, fsoil = (raw_norm * f_rest[:, None]).T

    tasks[:, 5] = fsnow
    tasks[:, 6] = fpv
    tasks[:, 7] = fnpv
    tasks[:, 8] = fsoil

    tasks[:, 9] = lin_map(u[:, 9], CWV_LO, CWV_HI)
    tasks[:, COL_ELE] = lin_map(u[:, 11], ELE_LO, ELE_HI)
    tasks[:, 10] = lin_map(u[:, 10], AOT_LO, AOT_HI)
    tasks[:, COL_RAA] = lin_map(u[:, 12], RAA_LO, RAA_HI)
    tasks[:, COL_SZA] = lin_map(u[:, 13], SZA_LO, SZA_HI)
    tasks[:, COL_SLOPE] = slope
    tasks[:, COL_SAA_ASPECT] = lin_map(u[:, U_SAA_ASPECT], SAA_ASPECT_LO, SAA_ASPECT_HI)
    tasks[:, 0] = cos_i_from_geometry(
        tasks[:, COL_SZA],
        tasks[:, COL_SLOPE],
        tasks[:, COL_SAA_ASPECT],
    )
    return tasks


def apply_grain_dust_and_cwv_filter(tasks):
    """Keep CWV and cos_i in design window (fine-grain dust/algae set in decode).

    Rejects self-shadowed / out-of-range local incidence (cos_i outside [COSI_LO, COSI_HI]).
    AOT is sampled over the atmosphere LUT window [AOT_LO, AOT_HI]; no extra AOT filter.
    """
    tasks = np.asarray(tasks, dtype=np.float64).copy()

    cwv_max = atm_midlat_winter_cwv_upperbound(tasks[:, COL_ELE])
    keep_cwv = tasks[:, 9] <= cwv_max
    keep_cosi = (tasks[:, 0] >= COSI_LO) & (tasks[:, 0] <= COSI_HI)
    keep = keep_cwv & keep_cosi
    n_rejected_cwv = int(np.count_nonzero(~keep_cwv))
    n_rejected_cosi = int(np.count_nonzero(keep_cwv & ~keep_cosi))
    kept = tasks[keep]
    n_fine_grain = int(np.count_nonzero(kept[:, 1] < GRAIN_DUST_THRESHOLD))
    return kept, n_fine_grain, n_rejected_cwv, n_rejected_cosi


def sample_valid_tasks(n_samples, mix_log_weight=MIX_LOG_WEIGHT, seed=SOBOL_SEED):
    """Rejection-sample Sobol until n_samples pass CWV / cos_i rules."""
    sampler = qmc.Sobol(d=N_SOBOL_DIMS, scramble=True, seed=seed)
    kept = []
    n_drawn = 0
    n_fine_grain_total = 0
    n_rejected_cwv_total = 0
    n_rejected_cosi_total = 0
    # SciPy Sobol prefers powers of 2; draw in balanced chunks.
    chunk = 1 << int(np.ceil(np.log2(max(n_samples, 2))))

    while sum(block.shape[0] for block in kept) < n_samples:
        u = sampler.random(chunk)
        n_drawn += chunk
        decoded = decode_sobol_tasks(u, mix_log_weight=mix_log_weight)
        good, n_fine, n_rej_cwv, n_rej_cosi = apply_grain_dust_and_cwv_filter(decoded)
        n_fine_grain_total += n_fine
        n_rejected_cwv_total += n_rej_cwv
        n_rejected_cosi_total += n_rej_cosi
        if good.shape[0]:
            kept.append(good)

    tasks = np.concatenate(kept, axis=0)[:n_samples]
    stats = {
        "n_drawn": n_drawn,
        "n_forced_dust": n_fine_grain_total,  # fine-grain low-impurity remaps (legacy key)
        "n_fine_grain": n_fine_grain_total,
        "n_rejected_cwv": n_rejected_cwv_total,
        "n_rejected_cosi": n_rejected_cosi_total,
        "n_kept": int(tasks.shape[0]),
    }
    return tasks, stats


# Generate filtered Sobol samples BEFORE building LUTs
tasks, sample_stats = sample_valid_tasks(N_SAMPLES, mix_log_weight=MIX_LOG_WEIGHT)

print(
    "Sample filter (before LUT build): "
    f"drawn={sample_stats['n_drawn']}, "
    f"fine_grain(grain<{GRAIN_DUST_THRESHOLD:g}; "
    f"dust logU[{FINE_DUST_LO:g},{FINE_DUST_HI:g}], "
    f"algae logU[{FINE_ALGAE_LO:g},{FINE_ALGAE_HI:g}])={sample_stats['n_fine_grain']}, "
    f"rejected_cwv={sample_stats['n_rejected_cwv']}, "
    f"rejected_cos_i={sample_stats['n_rejected_cosi']}, "
    f"kept={sample_stats['n_kept']}"
)
print(
    "Geometry sampling: "
    f"sza=[{SZA_LO:g},{SZA_HI:g}], "
    f"slope=[{SLOPE_LO:g},{SLOPE_HI:g}] with "
    f"P(flat [{FLAT_SLOPE_LO:g},{FLAT_SLOPE_HI:g}])={FLAT_SLOPE_WEIGHT:g}, "
    f"saa-aspect=[{SAA_ASPECT_LO:g},{SAA_ASPECT_HI:g}]; "
    f"cos_i derived in [{COSI_LO:g},{COSI_HI:g}], coszen=cos(sza)"
)
_flat_frac = float(np.mean(tasks[:, COL_SLOPE] <= FLAT_SLOPE_HI))
print(f"  realized flat-slope fraction (slope<={FLAT_SLOPE_HI:g}): {_flat_frac:.3f}")

print(tasks)
print(
    f"Mixture decode (MIX_LOG_WEIGHT={MIX_LOG_WEIGHT:g}) for "
    f"{', '.join(LOG_UNIFORM_QOIS)} "
    f"(floors: lwc={LWC_LO:g}, dust={DUST_LO:g}, algae={ALGAE_LO:g})"
)

mean_per_feat = np.mean(tasks, axis=0)
print("mean per feature: ", mean_per_feat)

std_per_feat = np.std(tasks, axis=0)
print("std per feature: ", std_per_feat)

# Sanity: mix means sit between pure-log and linear midpoints
print(
    "linear midpoints (for contrast): "
    f"lwc={0.5 * (LWC_LO + LWC_HI):.1f}, "
    f"dust={0.5 * (DUST_LO + DUST_HI):.1f}, "
    f"algae={0.5 * (ALGAE_LO + ALGAE_HI):.1f}"
)


ds_mod = xr.open_dataset(MODTRAN_PATH) if str(MODTRAN_PATH).endswith(".nc") else xr.open_zarr(MODTRAN_PATH)
wl_name = _first_present(_WL_DIM_CANDIDATES, set(ds_mod.rhoatm.dims))
if wl_name is None:
    raise KeyError(f"Atmosphere LUT has no wavelength dim; rhoatm dims={ds_mod.rhoatm.dims}")
wl_mod = _wavelengths_to_nm(ds_mod[wl_name].values, label="modtran")
modtran_axes = select_modtran_axes(ds_mod)
if not modtran_axes:
    raise KeyError(f"No recognized atmosphere LUT axes in {tuple(ds_mod.rhoatm.dims)}")
print(f"MODTRAN LUT axes: {modtran_axes}")
print(
    "  axis ranges: "
    + ", ".join(
        f"{ax}=[{float(np.min(ds_mod[ax].values)):g},{float(np.max(ds_mod[ax].values)):g}]"
        for ax in modtran_axes
    )
)
if "observer_zenith" not in modtran_axes:
    print("  (no observer_zenith on this LUT; VZA is not an interpolation axis)")
if any(a in modtran_axes for a in _SZA_DIM_CANDIDATES):
    print("  using solar zenith as an interpolation axis")
target_dims = tuple(modtran_axes) + (wl_name,)
modtran_grid = [np.asarray(ds_mod[k].values) for k in modtran_axes]

# # NOTE: Loading Atmospheric radiative transfer LUT
print("building modtran interpolators")
for ax, g in zip(modtran_axes, modtran_grid):
    print(f"  {ax} nodes ({len(g)}): {np.array2string(np.asarray(g), precision=4)}")
_rhoatm = np.asarray(ds_mod.rhoatm.transpose(*target_dims).values)
_sphalb = np.asarray(ds_mod.sphalb.transpose(*target_dims).values)
print(
    f"  LUT finite_frac (raw): rhoatm={_finite_frac(_rhoatm):.4f}  "
    f"sphalb={_finite_frac(_sphalb):.4f}"
)
_rhoatm, n_fill_rho = _fill_nan_spectra_nearest(_rhoatm)
_sphalb, n_fill_sph = _fill_nan_spectra_nearest(_sphalb)
print(
    f"  filled NaN geometry nodes: rhoatm={n_fill_rho}  sphalb={n_fill_sph}  "
    f"(post finite_frac rhoatm={_finite_frac(_rhoatm):.4f})"
)
v_interp_Latm = VectorInterpolator(modtran_grid, _rhoatm, version="mlg")
v_interp_sphalb = VectorInterpolator(modtran_grid, _sphalb, version="mlg")
v_interp_Lraw = {}
for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]:
    arr_k, n_fill_k = _fill_nan_spectra_nearest(
        np.asarray(ds_mod[k].transpose(*target_dims).values)
    )
    if n_fill_k:
        print(f"  filled NaN geometry nodes: {k}={n_fill_k}")
    v_interp_Lraw[k] = VectorInterpolator(modtran_grid, arr_k, version="mlg")

# Tasks were drawn before LUT load; hard-clip atmosphere QoIs onto actual axes.
_atm_col = {
    "surface_elevation_km": COL_ELE,
    "ele_km": COL_ELE,
    "elevation_km": COL_ELE,
    "solar_zenith": COL_SZA,
    "sza": COL_SZA,
    "SZA": COL_SZA,
    "AOT550": 10,
    "H2OSTR": 9,
}
for ax, col in _atm_col.items():
    if ax not in modtran_axes:
        continue
    g = modtran_grid[modtran_axes.index(ax)]
    lo, hi = float(np.min(g)), float(np.max(g))
    before = tasks[:, col].copy()
    tasks[:, col] = np.clip(tasks[:, col], lo, hi)
    n_clip = int(np.count_nonzero(tasks[:, col] != before))
    if n_clip:
        print(f"  clipped {n_clip} task values onto LUT axis {ax}=[{lo:g},{hi:g}]")

# SZA may have been clipped onto the atmosphere LUT; refresh derived cos_i.
tasks[:, 0] = cos_i_from_geometry(
    tasks[:, COL_SZA],
    tasks[:, COL_SLOPE],
    tasks[:, COL_SAA_ASPECT],
)
n_cosi_oob = int(
    np.count_nonzero((tasks[:, 0] < COSI_LO) | (tasks[:, 0] > COSI_HI))
)
if n_cosi_oob:
    print(
        f"  warning: {n_cosi_oob} samples have cos_i outside "
        f"[{COSI_LO:g},{COSI_HI:g}] after SZA LUT clip; clipping cos_i"
    )
    tasks[:, 0] = np.clip(tasks[:, 0], COSI_LO, COSI_HI)

# # NOTE: Loading Snow Surface radiative transfer LUT
print("loading disort")
ds_dis = xr.load_dataset(DISORT_PATH)
disort_axis_names = ["sza", "vza", "raa", "grain_radius", "algae_conc", "dust_conc", "lwc"]
disort_grid = [np.asarray(ds_dis[k].values) for k in disort_axis_names]
print(
    "  DISORT ranges: "
    + ", ".join(
        f"{n}=[{float(np.min(g)):g},{float(np.max(g)):g}]"
        for n, g in zip(disort_axis_names, disort_grid)
    )
)
# Nadir VZA=0 is often outside the snow LUT; use nearest in-grid VZA for DISORT only.
VZA_DISORT = float(np.clip(VZA_TRUE, float(np.min(disort_grid[1])), float(np.max(disort_grid[1]))))
if VZA_DISORT != float(VZA_TRUE):
    print(f"  DISORT VZA clamped: {VZA_TRUE:g} → {VZA_DISORT:g} (LUT bounds)")
v_interp_r_dd = VectorInterpolator(disort_grid, ds_dis.r_dd.values, version="mlg")
v_interp_r_hd = VectorInterpolator(disort_grid, ds_dis.r_hd.values, version="mlg")

emit_specs = pd.read_csv(EMIT_WAVE_PATH, sep=r'\s+', names=['idx', 'wl', 'fwhm'])
emit_noise = pd.read_csv(NOISE_PATH, sep=r'\s+', names=['wvl', 'a', 'b', 'c', 'rmse'], comment='#')

emit_wl = _wavelengths_to_nm(emit_specs.wl.values, label="emit")
disort_wl = _wavelengths_to_nm(ds_dis.wavelength.values, label="disort")
H_matrix = calculate_resample_matrix(wl_mod, emit_wl, emit_specs.fwhm.values)

endmembers_emit = np.array(pd.read_csv(ENDMEMBER_PATH))[:, 1:]
if endmembers_emit.shape[0] == wl_mod.size:
    endmembers = endmembers_emit
else:
    # Background endmembers are on EMIT bands; mix snow+soil on the atmosphere LUT grid.
    endmembers = np.column_stack(
        [
            np.interp(wl_mod, emit_wl, endmembers_emit[:, j])
            for j in range(endmembers_emit.shape[1])
        ]
    )
print(
    f"wl grids: modtran={wl_mod.size} [{wl_mod.min():.4g},{wl_mod.max():.4g}]  "
    f"emit={emit_wl.size} [{emit_wl.min():.4g},{emit_wl.max():.4g}]  "
    f"disort={disort_wl.size} [{disort_wl.min():.4g},{disort_wl.max():.4g}]  "
    f"endmembers={endmembers.shape}"
)
solar_irr = np.asarray(ds_mod.solar_irr.values, dtype=np.float64).reshape(-1)
if solar_irr.size != wl_mod.size:
    raise ValueError(
        f"solar_irr length {solar_irr.size} != modtran wl {wl_mod.size}"
    )
print(
    f"  solar_irr finite={_finite_frac(solar_irr):.3f}  "
    f"H_matrix={tuple(np.asarray(H_matrix).shape)}"
)

coszen_var = ds_mod["coszen"] if "coszen" in ds_mod else None
if coszen_var is not None and np.ndim(np.asarray(coszen_var)) == 0:
    COSZEN_LUT = float(coszen_var)
elif "coszen" in ds_mod.attrs:
    COSZEN_LUT = float(ds_mod.attrs["coszen"])
else:
    COSZEN_LUT = None
if COSZEN_LUT is not None and not np.isfinite(COSZEN_LUT):
    COSZEN_LUT = None
print(f"Scalar LUT coszen attr: {COSZEN_LUT} (per-sample SZA is used for coszen)")


def softmax(z):
    "Used to maintain sum-to-1 condition and positive fractional covers"
    return np.exp(z) / np.sum(np.exp(z))

def E_to_L(E, coszen):
    """Convert irradiance to radiance (ISOFIT convention)."""
    return E * coszen / np.pi


def transm_to_rdn(transm, coszen, solar_irr):
    """Convert a unitless atmospheric vector to radiance units (ISOFIT convention)."""
    return transm * E_to_L(solar_irr, coszen)


def apply_noise(rdn, wl, noise_df):
    """EMIT parametric NEDL; independent draws (Sy is diagonal — skip SVD)."""
    rdn = np.asarray(rdn, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(rdn)):
        n_bad = int(np.count_nonzero(~np.isfinite(rdn)))
        raise ValueError(
            f"TOA radiance has {n_bad}/{rdn.size} non-finite values before noise"
        )
    a = np.interp(wl, noise_df["wvl"], noise_df["a"])
    b = np.interp(wl, noise_df["wvl"], noise_df["b"])
    c = np.interp(wl, noise_df["wvl"], noise_df["c"])
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c
    nedl = np.maximum(nedl, 0.0)
    return rdn + np.random.normal(loc=0.0, scale=nedl)

# NOTE: this is similar (hopefully) to the forward.py in isofit
def simulate_pixel(
    cosi, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot,
    ele_km, raa_true, sza, wl_mod, ds_dis_wl, solar_irr_emit, coszen,
    *,
    diagnose=False,
):
    raa_disort = 180.0 - raa_true
    # cosi → local incidence as SZA-like angle on the DISORT sza axis
    sza_surface = float(np.degrees(np.arccos(np.clip(cosi, -1.0, 1.0))))
    lookup_pt_raw = np.array(
        [sza_surface, VZA_DISORT, raa_disort, grain, algae, dust, lwc],
        dtype=np.float64,
    )
    lookup_pt, disort_clip_msgs = _clip_query_to_grid(
        lookup_pt_raw, disort_grid, disort_axis_names
    )
    rho_dd_22 = v_interp_r_dd(lookup_pt)
    rho_hd_22 = v_interp_r_hd(lookup_pt)

    # DISORT reflectance → atmosphere LUT wavelength grid
    rho_dd = np.interp(wl_mod, ds_dis_wl, rho_dd_22)
    rho_hd = np.interp(wl_mod, ds_dis_wl, rho_hd_22)
    f = np.array([fsnow, fPV, fNPV, fsoil])

    rho_dd = rho_dd * f[0] + np.dot(endmembers, f[1:])
    rho_hd = rho_hd * f[0] + np.dot(endmembers, f[1:])

    coszen = max(float(coszen), 1e-6)
    atm_pt_raw = build_atm_pt(
        ele_km,
        raa_true,
        aot,
        cwv,
        sza,
        axes=modtran_axes,
    )
    atm_pt, atm_clip_msgs = _clip_query_to_grid(atm_pt_raw, modtran_grid, modtran_axes)

    # LUT outputs are unitless (reflectance-like); convert to radiance via ISOFIT convention.
    rhoatm = v_interp_Latm(atm_pt)
    s_alb = v_interp_sphalb(atm_pt)
    transm_raw = [v_interp_Lraw[k](atm_pt) for k in ['dir-dir', 'dif-dir', 'dir-dif', 'dif-dif']]
    solar_irr_local = np.asarray(solar_irr, dtype=np.float64).reshape(-1)
    solar_irr_emit = np.dot(H_matrix, solar_irr_local)

    # Convert unitless LUT vectors to radiance (µW/cm²/sr/nm).
    L_atm = transm_to_rdn(rhoatm, coszen, solar_irr_local)
    L_raw = [transm_to_rdn(t, coszen, solar_irr_local) for t in transm_raw]

    # ISOFIT 6c: form L_tot before eq. 11 on diffuse terms; residual is
    # (L_tot * s * rho^2) / (1 - s * rho), not L_raw * rho / (1 - s * rho).
    eq_11_term = np.maximum(1.0 - (s_alb * rho_hd), 1e-8)
    L_dir_dir = (L_raw[0] / coszen) * cosi
    L_dif_dir = L_raw[1] * (cosi / coszen)
    L_dir_dif = L_raw[2]  # flat background: cos_i_bg = coszen
    L_dif_dif = L_raw[3]
    l_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
    L_dif_dir = L_dif_dir / eq_11_term
    L_dif_dif = L_dif_dif / eq_11_term
    atm_surface_scattering = s_alb * rho_hd

    toa_rdn = (
        L_atm
        + L_dir_dir * rho_dd
        + L_dif_dir * rho_hd
        + L_dir_dif * rho_hd
        + L_dif_dif * rho_hd
        + (l_tot * atm_surface_scattering * rho_hd) / eq_11_term
    )

    rdn_emit = np.dot(H_matrix, toa_rdn)

    if diagnose:
        print(
            "  smoke disort_pt="
            + np.array2string(lookup_pt, precision=4, separator=",")
            + (f" clipped[{','.join(disort_clip_msgs)}]" if disort_clip_msgs else "")
        )
        print(
            "  smoke atm_pt="
            + np.array2string(atm_pt, precision=4, separator=",")
            + (f" clipped[{','.join(atm_clip_msgs)}]" if atm_clip_msgs else "")
        )
        print(
            "  smoke finite_frac: "
            f"rho_dd={_finite_frac(rho_dd_22):.3f} rho_hd={_finite_frac(rho_hd_22):.3f} "
            f"rhoatm={_finite_frac(rhoatm):.3f} L_atm={_finite_frac(L_atm):.3f} "
            f"s_alb={_finite_frac(s_alb):.3f} "
            f"Lraw={[round(_finite_frac(x), 3) for x in L_raw]} "
            f"toa={_finite_frac(toa_rdn):.3f} rdn_emit={_finite_frac(rdn_emit):.3f} "
            f"solar_emit={_finite_frac(solar_irr_emit):.3f}"
        )

    if not np.all(np.isfinite(rdn_emit)):
        raise ValueError(
            "TOA radiance non-finite before noise "
            f"(finite_frac={_finite_frac(rdn_emit):.3f}; "
            f"disort_clip={disort_clip_msgs or 'none'}; "
            f"atm_clip={atm_clip_msgs or 'none'}; "
            f"atm_pt={np.array2string(atm_pt, precision=6)}; "
            f"rho_dd={_finite_frac(rho_dd_22):.3f}; "
            f"rhoatm={_finite_frac(rhoatm):.3f}; "
            f"L_atm={_finite_frac(L_atm):.3f})"
        )

    rdn_noisy = apply_noise(rdn_emit, emit_wl, emit_noise)
    toa_ref = rdn_noisy * np.pi / (solar_irr_emit * coszen)

    return rdn_noisy, toa_ref


def _row_kwargs(row):
    return dict(
        cosi=row[0],
        grain=row[1],
        lwc=row[2],
        dust=row[3],
        algae=row[4],
        fsnow=row[5],
        fPV=row[6],
        fNPV=row[7],
        fsoil=row[8],
        cwv=row[9],
        aot=row[10],
        ele_km=row[COL_ELE],
        raa_true=row[COL_RAA],
        sza=row[COL_SZA],
        wl_mod=wl_mod,
        ds_dis_wl=disort_wl,
        solar_irr_emit=None,
        coszen=float(np.cos(np.radians(row[COL_SZA]))),
    )


# Number of Sobol Samples
n_samples = tasks.shape[0]

print("Smoke-testing first Sobol sample (diagnose=True)...")
_ = simulate_pixel(**_row_kwargs(tasks[0]), diagnose=True)

print(f"Starting parallel simulation on {n_samples} Sobol samples...")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_pixel)(
        row[0],  # COSI
        row[1],  # grain
        row[2],  # LWC
        row[3],  # dust
        row[4],  # algae
        row[5],  # f_snow
        row[6],  # f_pv
        row[7],  # f_npv
        row[8],  # f_soil
        row[9],  # CWV
        row[10],  # AOT
        row[COL_ELE],  # ele_km
        row[COL_RAA],  # RAA_TRUE
        row[COL_SZA],  # sza
        wl_mod,
        disort_wl,
        None,
        np.cos(np.radians(row[COL_SZA])),
    )
    for row in tasks
)

results_rdn = np.array([r[0] for r in results])
results_ref = np.array([r[1] for r in results])

ds_out = xr.Dataset(
    data_vars={
        "toa_radiance": (["sample", "wl"], results_rdn.astype(np.float32)),
        "toa_reflectance": (["sample", "wl"], results_ref.astype(np.float32)),
        "cos_i": (["sample"], tasks[:, 0]),
        "grain_size": (["sample"], tasks[:, 1]),
        "liquid_water": (["sample"], tasks[:, 2]),
        "dust": (["sample"], tasks[:, 3]),
        "algae": (["sample"], tasks[:, 4]),
        "fsnow": (["sample"], tasks[:, 5]),
        "fPV": (["sample"], tasks[:, 6]),
        "fNPV": (["sample"], tasks[:, 7]),
        "fsoil": (["sample"], tasks[:, 8]),
        "cwv": (["sample"], tasks[:, 9]),
        "aot": (["sample"], tasks[:, 10]),
        "ele_km": (["sample"], tasks[:, COL_ELE]),
        "RAA_TRUE": (["sample"], tasks[:, COL_RAA]),
        "sza": (["sample"], tasks[:, COL_SZA]),
        "slope": (["sample"], tasks[:, COL_SLOPE]),
        "saa_minus_aspect": (["sample"], tasks[:, COL_SAA_ASPECT]),
        "coszen": (["sample"], np.cos(np.radians(tasks[:, COL_SZA]))),
    },
    coords={
        "sample": np.arange(len(tasks)),
        "wl": emit_wl,
    },
    attrs={
        "skyview_factor": SVF_TRUE,
        "VZA": VZA_TRUE,
        "VZA_DISORT": VZA_DISORT,
        "Constant_background_reflectance": BG_RFL,
        "ele_km_range": f"[{ELE_LO:g}, {ELE_HI:g}]",
        "RAA_TRUE_range": f"[{RAA_LO:g}, {RAA_HI:g}]",
        "sza_range": f"[{SZA_LO:g}, {SZA_HI:g}]",
        "slope_range": f"[{SLOPE_LO:g}, {SLOPE_HI:g}]",
        "flat_slope_weight": float(FLAT_SLOPE_WEIGHT),
        "flat_slope_range": f"[{FLAT_SLOPE_LO:g}, {FLAT_SLOPE_HI:g}]",
        "saa_minus_aspect_range": f"[{SAA_ASPECT_LO:g}, {SAA_ASPECT_HI:g}]",
        "cos_i_range": f"[{COSI_LO:g}, {COSI_HI:g}]",
        "aot_range": f"[{AOT_LO:g}, {AOT_HI:g}]",
        "cwv_range": f"[{CWV_LO:g}, {CWV_HI:g}]",
        # NetCDF attrs cannot be None; this LUT has no scalar coszen (per-sample sza used).
        "coszen_lut": float(COSZEN_LUT) if COSZEN_LUT is not None else float("nan"),
        "modtran_path": str(MODTRAN_PATH),
        "modtran_axes": ",".join(modtran_axes),
        #ToDo: update this path
        "noise_model": "https://github.com/isofit/isofit-data/blob/main/emit_noise.txt",
        "RT_atmosphere": "sRTMnet 6c",
        "RT_surface": "DISORT based snow surface LUT",
        "sampling": (
            f"Sobol({N_SOBOL_DIMS}); mixture decode for liquid_water,dust,algae "
            f"(P(log)={MIX_LOG_WEIGHT:g}, floors {LWC_LO:g},{DUST_LO:g},{ALGAE_LO:g}); "
            "linear-uniform for grain_size,cwv,aot,ele_km,RAA_TRUE,sza,saa_minus_aspect; "
            f"slope mix: P([{FLAT_SLOPE_LO:g},{FLAT_SLOPE_HI:g}])={FLAT_SLOPE_WEIGHT:g}, "
            f"else [{FLAT_SLOPE_HI:g},{SLOPE_HI:g}]; "
            "cos_i=sin(sza)*sin(slope)*cos(saa-aspect)+cos(sza)*cos(slope); "
            "coszen=cos(sza); "
            f"when grain_size<{GRAIN_DUST_THRESHOLD:g}: "
            f"dust log-uniform [{FINE_DUST_LO:g},{FINE_DUST_HI:g}], "
            f"algae log-uniform [{FINE_ALGAE_LO:g},{FINE_ALGAE_HI:g}]; "
            "cwv kept under ATM_MIDLAT_WINTER(ele_km); "
            f"cos_i kept in [{COSI_LO:g},{COSI_HI:g}]"
        ),
        "log_uniform_qois": ",".join(LOG_UNIFORM_QOIS),
        "mix_log_weight": float(MIX_LOG_WEIGHT),
        "fine_dust_range": f"[{FINE_DUST_LO:g}, {FINE_DUST_HI:g}]",
        "fine_algae_range": f"[{FINE_ALGAE_LO:g}, {FINE_ALGAE_HI:g}]",
        "n_drawn": sample_stats["n_drawn"],
        "n_forced_dust": sample_stats["n_forced_dust"],
        "n_fine_grain": sample_stats["n_fine_grain"],
        "n_rejected_cwv": sample_stats["n_rejected_cwv"],
        "n_rejected_cosi": sample_stats["n_rejected_cosi"],
        "n_kept": sample_stats["n_kept"],
    }
)
ds_out.to_netcdf(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
