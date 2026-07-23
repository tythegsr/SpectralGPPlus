"""
Generate S2 11-QoI TOA simulations with log-uniform sampling for heavy-tailed QoIs.

Corrects the linear Sobol decode used for Otto/Will RNTD-style runs:
  - cos_i, fsnow, fPV, fNPV, fsoil, cwv, aot  -> linear-uniform in physical units
  - grain_size, liquid_water, dust, algae     -> log-uniform in physical units
    (Sobol u mapped through log10, then 10** so DISORT/MODTRAN still see physical values)

NetCDF stores physical QoIs (same schema as before). GP training may still apply log10
to algae/dust/grain_size/liquid_water at fit time.

Requires the ISOFIT/DISORT helper module ``common`` (VectorInterpolator,
calculate_resample_matrix) on PYTHONPATH, plus the LUT paths below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Paths — update for your machine
# ---------------------------------------------------------------------------
_DATA = Path(r"C:/Users/tylerj/isofit/disort_data_for_tyler")
MODTRAN_PATH = _DATA / "lut.zarr"
DISORT_PATH = _DATA / "disort_snow_lut_EMIT.nc"
ENDMEMBER_PATH = _DATA / "endmembers.csv"
EMIT_WAVE_PATH = _DATA / "emit-wave.txt"
NOISE_PATH = _DATA / "emit_noise.txt"
OUTPUT_PATH = _DATA / "data" / "snow_toa_simulations_loguniform_20262107.nc"

# Optional: directory containing ``common.py`` (VectorInterpolator, ...)
COMMON_DIR: Path | None = None  # e.g. Path(r"C:/Users/tylerj/isofit/...")

N_SAMPLES = 59000
SOBOL_SEED = 42
N_JOBS = -1

ELE_TRUE = 3.0
SVF_TRUE = 0.95
RAA_TRUE = 164.0
VZA_TRUE = 6.0
RAA_DISORT = 180.0 - RAA_TRUE
BG_RFL = "None"

# Linear physical bounds (unchanged design)
COSI_LO, COSI_HI = 0.06, 1.0
FSNOW_LO, FSNOW_HI = 0.0, 10.0
FPV_LO, FPV_HI = 0.0, 10.0
FNPV_LO, FNPV_HI = 0.0, 10.0
FSOIL_LO, FSOIL_HI = 0.0, 10.0
CWV_LO, CWV_HI = 0.2, 5.2
AOT_LO, AOT_HI = 0.04, 1.0

# Log-uniform physical bounds (fallback if LUT mins are 0 / missing)
GRAIN_LO_FALLBACK, GRAIN_HI_FALLBACK = 30.0, 1500.0
LWC_LO_FALLBACK, LWC_HI_FALLBACK = 1e-2, 25.0
DUST_LO_FALLBACK, DUST_HI_FALLBACK = 1e-2, 4000.0
ALGAE_LO_FALLBACK, ALGAE_HI_FALLBACK = 1e-2, 6e5

# Column order in ``tasks`` / ``simulate_pixel``
IDX_COSI, IDX_GRAIN, IDX_LWC, IDX_DUST, IDX_ALGAE = 0, 1, 2, 3, 4
IDX_FSNOW, IDX_FPV, IDX_FNPV, IDX_FSOIL, IDX_CWV, IDX_AOT = 5, 6, 7, 8, 9, 10


def _ensure_common_import() -> None:
    if COMMON_DIR is not None:
        p = str(COMMON_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from common import VectorInterpolator, calculate_resample_matrix  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Cannot import common.VectorInterpolator. Set COMMON_DIR to the folder "
            "that contains common.py, or add it to PYTHONPATH."
        ) from exc


def lin_map(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return u * (hi - lo) + lo


def log_map(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map u in [0,1] -> log-uniform physical values on [lo, hi]."""
    if not (lo > 0.0 and hi > lo):
        raise ValueError(f"log_map requires 0 < lo < hi, got lo={lo}, hi={hi}")
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    return np.power(10.0, u * (log_hi - log_lo) + log_lo)


def _log_sample_bounds(
    values: np.ndarray,
    design_lo: float,
    design_hi: float,
    *,
    name: str = "axis",
) -> tuple[float, float]:
    """
    Physical [lo, hi] for log-uniform Sobol decode.

    Uses the **design** window (e.g. dust 1e-2..4000), not the first positive LUT
    node. Many snow LUTs only tabulate coarse positive knots (0, 2000, 4000) —
    sampling should still cover the design range; DISORT interpolates between nodes.

    Caps ``hi`` to the LUT max when the LUT is narrower than the design.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    uniq = np.unique(v)
    pos = np.unique(v[v > 0.0]) if v.size else np.array([], dtype=np.float64)
    print(f"  LUT {name}: unique={uniq[:12]}{'...' if uniq.size > 12 else ''} (n={uniq.size})")

    lo = float(design_lo)
    hi = float(design_hi)
    if pos.size:
        lut_lo = float(pos.min())
        lut_hi = float(pos.max())
        if lo < lut_lo:
            print(
                f"  LUT {name}: design lo={lo:g} is below first positive LUT node "
                f"{lut_lo:g}; DISORT will interpolate from 0/{lut_lo:g}"
            )
        hi = min(hi, lut_hi)
    if not (lo > 0.0 and hi > lo):
        raise ValueError(
            f"Invalid log bounds for {name}: lo={lo}, hi={hi} "
            f"(design=[{design_lo}, {design_hi}])"
        )
    return lo, hi


def decode_sobol_tasks(
    u: np.ndarray,
    *,
    grain_lo: float,
    grain_hi: float,
    lwc_lo: float,
    lwc_hi: float,
    dust_lo: float,
    dust_hi: float,
    algae_lo: float,
    algae_hi: float,
) -> np.ndarray:
    """
    Decode Sobol samples in [0,1]^11 to physical QoI matrix (n, 11).

    Log-uniform: grain, lwc, dust, algae.
    Linear-uniform: cosi, fsnow, fPV, fNPV, fsoil, cwv, aot.
    """
    if u.ndim != 2 or u.shape[1] != 11:
        raise ValueError(f"expected u shape (n, 11), got {u.shape}")
    tasks = np.empty_like(u, dtype=np.float64)
    tasks[:, IDX_COSI] = lin_map(u[:, IDX_COSI], COSI_LO, COSI_HI)
    tasks[:, IDX_GRAIN] = log_map(u[:, IDX_GRAIN], grain_lo, grain_hi)
    tasks[:, IDX_LWC] = log_map(u[:, IDX_LWC], lwc_lo, lwc_hi)
    tasks[:, IDX_DUST] = log_map(u[:, IDX_DUST], dust_lo, dust_hi)
    tasks[:, IDX_ALGAE] = log_map(u[:, IDX_ALGAE], algae_lo, algae_hi)
    tasks[:, IDX_FSNOW] = lin_map(u[:, IDX_FSNOW], FSNOW_LO, FSNOW_HI)
    tasks[:, IDX_FPV] = lin_map(u[:, IDX_FPV], FPV_LO, FPV_HI)
    tasks[:, IDX_FNPV] = lin_map(u[:, IDX_FNPV], FNPV_LO, FNPV_HI)
    tasks[:, IDX_FSOIL] = lin_map(u[:, IDX_FSOIL], FSOIL_LO, FSOIL_HI)
    tasks[:, IDX_CWV] = lin_map(u[:, IDX_CWV], CWV_LO, CWV_HI)
    tasks[:, IDX_AOT] = lin_map(u[:, IDX_AOT], AOT_LO, AOT_HI)
    return tasks


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def apply_noise(rdn: np.ndarray, wl: np.ndarray, noise_df: pd.DataFrame) -> np.ndarray:
    a = np.interp(wl, noise_df["wvl"], noise_df["a"])
    b = np.interp(wl, noise_df["wvl"], noise_df["b"])
    c = np.interp(wl, noise_df["wvl"], noise_df["c"])
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c
    sy = np.diagflat(np.power(nedl, 2))
    return rdn + np.random.multivariate_normal(np.zeros(rdn.shape), sy)


def main() -> None:
    _ensure_common_import()
    from common import VectorInterpolator, calculate_resample_matrix

    print("Loading MODTRAN / DISORT LUTs...")
    ds_mod = xr.open_zarr(MODTRAN_PATH)
    ds_dis = xr.load_dataset(DISORT_PATH)

    grain_lo, grain_hi = _log_sample_bounds(
        ds_dis["grain_radius"].values,
        GRAIN_LO_FALLBACK,
        GRAIN_HI_FALLBACK,
        name="grain_radius",
    )
    algae_lo, algae_hi = _log_sample_bounds(
        ds_dis["algae_conc"].values,
        ALGAE_LO_FALLBACK,
        ALGAE_HI_FALLBACK,
        name="algae_conc",
    )
    dust_lo, dust_hi = _log_sample_bounds(
        ds_dis["dust_conc"].values,
        DUST_LO_FALLBACK,
        DUST_HI_FALLBACK,
        name="dust_conc",
    )
    lwc_lo, lwc_hi = _log_sample_bounds(
        ds_dis["lwc"].values,
        LWC_LO_FALLBACK,
        LWC_HI_FALLBACK,
        name="lwc",
    )

    print("Log-uniform physical bounds (simulation units):")
    print(f"  grain_size:   [{grain_lo:g}, {grain_hi:g}]")
    print(f"  liquid_water: [{lwc_lo:g}, {lwc_hi:g}]")
    print(f"  dust:         [{dust_lo:g}, {dust_hi:g}]")
    print(f"  algae:        [{algae_lo:g}, {algae_hi:g}]")

    sampler = qmc.Sobol(d=11, scramble=True, seed=SOBOL_SEED)
    # SciPy Sobol requires n = 2^k for balance; draw then truncate
    n_pow2 = 1 << int(np.ceil(np.log2(max(N_SAMPLES, 2))))
    u = sampler.random(n_pow2)[:N_SAMPLES]
    tasks = decode_sobol_tasks(
        u,
        grain_lo=grain_lo,
        grain_hi=grain_hi,
        lwc_lo=lwc_lo,
        lwc_hi=lwc_hi,
        dust_lo=dust_lo,
        dust_hi=dust_hi,
        algae_lo=algae_lo,
        algae_hi=algae_hi,
    )

    names = [
        "cos_i",
        "grain_size",
        "liquid_water",
        "dust",
        "algae",
        "fsnow",
        "fPV",
        "fNPV",
        "fsoil",
        "cwv",
        "aot",
    ]
    print("mean per feature (physical):", dict(zip(names, np.mean(tasks, axis=0))))
    print("std  per feature (physical):", dict(zip(names, np.std(tasks, axis=0))))
    for j, name in [
        (IDX_GRAIN, "grain_size"),
        (IDX_LWC, "liquid_water"),
        (IDX_DUST, "dust"),
        (IDX_ALGAE, "algae"),
    ]:
        y = np.log10(tasks[:, j])
        print(f"  log10({name}): mean={y.mean():.3f} std={y.std():.3f} "
              f"range=[{y.min():.3f}, {y.max():.3f}]")

    wl_mod = ds_mod.wl.values
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

    # sRTMnet 6c LUTs store rhoatm + dir-* coupling products already in radiance
    # (RT_mode=rdn). Only transm-mode LUTs need solar_irr * coszen / pi.
    rt_mode = str(ds_mod.attrs.get("RT_mode", ds_mod.attrs.get("rt_mode", "rdn"))).lower()
    print(f"Atmospheric LUT RT_mode={rt_mode}")

    print("Building MODTRAN interpolators...")
    v_interp_rhoatm = VectorInterpolator(
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

    print("Building DISORT interpolators...")
    disort_grid = [
        ds_dis[k].values
        for k in ["sza", "vza", "raa", "grain_radius", "algae_conc", "dust_conc", "lwc"]
    ]
    v_interp_r_dd = VectorInterpolator(disort_grid, ds_dis.r_dd.values, version="mlg")
    v_interp_r_hd = VectorInterpolator(disort_grid, ds_dis.r_hd.values, version="mlg")

    emit_specs = pd.read_csv(
        EMIT_WAVE_PATH, sep=r"\s+", names=["idx", "wl", "fwhm"]
    )
    emit_noise = pd.read_csv(
        NOISE_PATH,
        sep=r"\s+",
        names=["wvl", "a", "b", "c", "rmse"],
        comment="#",
    )
    h_matrix = calculate_resample_matrix(
        wl_mod, emit_specs.wl.values, emit_specs.fwhm.values
    )
    endmembers = np.array(pd.read_csv(ENDMEMBER_PATH))[:, 1:]
    ds_dis_wl = ds_dis.wavelength.values
    solar_irr = ds_mod.solar_irr.values
    coszen = float(ds_mod.coszen)

    def simulate_pixel(
        cosi: float,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        # Physical units into DISORT (not log)
        lookup_pt = np.array(
            [
                np.degrees(np.arccos(cosi)),
                VZA_TRUE,
                RAA_DISORT,
                grain,
                algae,
                dust,
                lwc,
            ]
        )
        rho_dd_22 = v_interp_r_dd(lookup_pt)
        rho_hd_22 = v_interp_r_hd(lookup_pt)
        rho_dd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_dd_22)
        rho_hd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_hd_22)

        f = softmax(np.array([fsnow, f_pv, f_npv, f_soil]))
        rho_dd = rho_dd * f[0] + np.dot(endmembers, f[1:])
        rho_hd = rho_hd * f[0] + np.dot(endmembers, f[1:])

        atm_pt = np.array([ELE_TRUE, 180.0 - VZA_TRUE, RAA_TRUE, aot, cwv])
        rho_atm = v_interp_rhoatm(atm_pt)
        s_alb = v_interp_sphalb(atm_pt)
        # Coupling terms from LUT; convert transmittance -> radiance only in transm mode.
        scale = (solar_irr * coszen / np.pi) if rt_mode == "transm" else 1.0
        l_raw = [v_interp_lraw[k](atm_pt) * scale for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]]
        l_atm = rho_atm * scale if rt_mode == "transm" else rho_atm
        solar_irr_emit = np.dot(h_matrix, solar_irr)

        # ISOFIT 6c: L_tot before eq. 11 on diffuse; residual is L_tot * s * rho^2 / (1 - s * rho).
        eq_11_term = 1.0 - (s_alb * rho_hd)
        l_dir_dir = (l_raw[0] / coszen) * cosi
        l_dif_dir = l_raw[1] * (cosi / coszen)
        l_dir_dif = l_raw[2]  # flat background: cos_i_bg = coszen
        l_dif_dif = l_raw[3]
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

        rdn_emit = np.dot(h_matrix, toa_rdn)
        rdn_noisy = apply_noise(rdn_emit, emit_specs.wl.values, emit_noise)
        toa_ref = rdn_noisy * np.pi / (solar_irr_emit * coszen)
        return rdn_noisy, toa_ref

    # Smoke test: one physical sample through the RT
    rdn0, ref0 = simulate_pixel(*tasks[0])
    if not (np.isfinite(rdn0).all() and np.isfinite(ref0).all()):
        raise RuntimeError("Smoke-test spectrum contains non-finite values.")
    print(f"Smoke test OK: rdn range [{rdn0.min():.4g}, {rdn0.max():.4g}]")

    print(f"Starting parallel simulation on {N_SAMPLES} samples (n_jobs={N_JOBS})...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(simulate_pixel)(
            row[IDX_COSI],
            row[IDX_GRAIN],
            row[IDX_LWC],
            row[IDX_DUST],
            row[IDX_ALGAE],
            row[IDX_FSNOW],
            row[IDX_FPV],
            row[IDX_FNPV],
            row[IDX_FSOIL],
            row[IDX_CWV],
            row[IDX_AOT],
        )
        for row in tasks
    )

    results_rdn = np.asarray([r[0] for r in results], dtype=np.float32)
    results_ref = np.asarray([r[1] for r in results], dtype=np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            "toa_radiance": (["sample", "wl"], results_rdn),
            "toa_reflectance": (["sample", "wl"], results_ref),
            "cos_i": (["sample"], tasks[:, IDX_COSI]),
            "grain_size": (["sample"], tasks[:, IDX_GRAIN]),
            "liquid_water": (["sample"], tasks[:, IDX_LWC]),
            "dust": (["sample"], tasks[:, IDX_DUST]),
            "algae": (["sample"], tasks[:, IDX_ALGAE]),
            "fsnow": (["sample"], tasks[:, IDX_FSNOW]),
            "fPV": (["sample"], tasks[:, IDX_FPV]),
            "fNPV": (["sample"], tasks[:, IDX_FNPV]),
            "fsoil": (["sample"], tasks[:, IDX_FSOIL]),
            "cwv": (["sample"], tasks[:, IDX_CWV]),
            "aot": (["sample"], tasks[:, IDX_AOT]),
        },
        coords={
            "sample": np.arange(len(tasks)),
            "wl": emit_specs.wl.values,
        },
        attrs={
            "skyview_factor": SVF_TRUE,
            "RAA": RAA_TRUE,
            "VZA": VZA_TRUE,
            "Constant_background_reflectance": BG_RFL,
            "elev_km": ELE_TRUE,
            "noise_model": "https://github.com/isofit/isofit-data/blob/main/emit_noise.txt",
            "RT_atmosphere": "sRTMnet 6c",
            "RT_mode": rt_mode,
            "RT_surface": "DISORT based snow surface LUT",
            "sampling": (
                "Sobol(11); log-uniform physical decode for "
                "grain_size, liquid_water, dust, algae; "
                "linear-uniform for remaining QoIs"
            ),
            "log_uniform_qois": "grain_size,liquid_water,dust,algae",
            "output_log_scale": "true",
            "grain_size_bounds": f"[{grain_lo}, {grain_hi}]",
            "liquid_water_bounds": f"[{lwc_lo}, {lwc_hi}]",
            "dust_bounds": f"[{dust_lo}, {dust_hi}]",
            "algae_bounds": f"[{algae_lo}, {algae_hi}]",
        },
    )
    ds_out.to_netcdf(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
