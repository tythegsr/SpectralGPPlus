import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

# NOTE: This is the nice interpolator Jouni built
from common import VectorInterpolator, calculate_resample_matrix

# ToDo: update these paths
MODTRAN_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/lut.zarr"
DISORT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/disort_snow_lut_EMIT.nc"
ENDMEMBER_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/endmembers.csv"
EMIT_WAVE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit-wave.txt"
NOISE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit_noise.txt"
OUTPUT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/data/snow_toa_fsnow_70to100_20261808.nc"

SVF_TRUE = 1.0
VZA_TRUE = 6.0

# # NOTE: background reflectance should be assumed equal to the target pixel
# per convo with Niklas today.
BG_RFL = "None"

# Column order: cos_i, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot,
#               ele_km, RAA_TRUE
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
FSNOW_LO, FSNOW_HI = 0.9, 1.0  # fsnow-only
FPV_LO, FPV_HI = 0.0, 0.1
FNPV_LO, FNPV_HI = 0.0, 0.1
FSOIL_LO, FSOIL_HI = 0.0, 0.1
CWV_LO, CWV_HI = 0.1, 1.4
AOT_LO, AOT_HI = 0.02, 0.02
ELE_LO, ELE_HI = 2.0, 4.5  # km
RAA_LO, RAA_HI = 110.0, 180.0  # deg (RAA_TRUE)

LOG_UNIFORM_QOIS = ("liquid_water", "dust", "algae", "grain_size")
# Task: 11 QoIs + ele_km + RAA_TRUE. Sobol: task coords + mix gates for lwc/dust/algae.
N_TASK_DIMS = 13
N_MIX_GATES = 3
N_SOBOL_DIMS = N_TASK_DIMS + N_MIX_GATES  # 16
COL_ELE = 11
COL_RAA = 12


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


def decode_sobol_tasks(u, mix_log_weight=MIX_LOG_WEIGHT):
    """
    Decode Sobol u in [0,1]^{16} to physical tasks (n, 13).

    Columns 0..10: QoI coordinates.
    Columns 11,12: ele_km, RAA_TRUE.
    Columns 13,14,15: mix gates for liquid_water, dust, algae.
    """
    if u.ndim != 2 or u.shape[1] != N_SOBOL_DIMS:
        raise ValueError(f"expected u shape (n, {N_SOBOL_DIMS}), got {u.shape}")
    tasks = np.empty((u.shape[0], N_TASK_DIMS), dtype=np.float64)
    tasks[:, 0] = lin_map(u[:, 0], COSI_LO, COSI_HI)
    tasks[:, 1] = lin_map(u[:, 1], GRAIN_LO, GRAIN_HI)
    tasks[:, 2] = mix_map(u[:, 2], u[:, 13], LWC_LO, LWC_HI, mix_log_weight)
    tasks[:, 3] = mix_map(u[:, 3], u[:, 14], DUST_LO, DUST_HI, mix_log_weight)
    tasks[:, 4] = mix_map(u[:, 4], u[:, 15], ALGAE_LO, ALGAE_HI, mix_log_weight)

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
    tasks[:, 10] = lin_map(u[:, 10], AOT_LO, AOT_HI)
    tasks[:, COL_ELE] = lin_map(u[:, 11], ELE_LO, ELE_HI)
    tasks[:, COL_RAA] = lin_map(u[:, 12], RAA_LO, RAA_HI)
    return tasks


def apply_grain_dust_and_cwv_filter(tasks):
    """Force near-zero dust for fine grains; keep CWV under ATM_MIDLAT_WINTER(ele_km)."""
    tasks = np.asarray(tasks, dtype=np.float64).copy()
    fine = tasks[:, 1] < GRAIN_DUST_THRESHOLD
    tasks[fine, 3] = DUST_LO

    cwv_max = atm_midlat_winter_cwv_upperbound(tasks[:, COL_ELE])
    keep = tasks[:, 9] <= cwv_max
    n_rejected_cwv = int(np.count_nonzero(~keep))
    kept = tasks[keep]
    n_forced_dust = int(np.count_nonzero(kept[:, 1] < GRAIN_DUST_THRESHOLD))
    return kept, n_forced_dust, n_rejected_cwv


def sample_valid_tasks(n_samples, mix_log_weight=MIX_LOG_WEIGHT, seed=SOBOL_SEED):
    """Rejection-sample Sobol until n_samples pass CWV / grain-dust rules."""
    sampler = qmc.Sobol(d=N_SOBOL_DIMS, scramble=True, seed=seed)
    kept = []
    n_drawn = 0
    n_forced_dust_total = 0
    n_rejected_cwv_total = 0
    # SciPy Sobol prefers powers of 2; draw in balanced chunks.
    chunk = 1 << int(np.ceil(np.log2(max(n_samples, 2))))

    while sum(block.shape[0] for block in kept) < n_samples:
        u = sampler.random(chunk)
        n_drawn += chunk
        decoded = decode_sobol_tasks(u, mix_log_weight=mix_log_weight)
        good, n_forced, n_rej = apply_grain_dust_and_cwv_filter(decoded)
        n_forced_dust_total += n_forced
        n_rejected_cwv_total += n_rej
        if good.shape[0]:
            kept.append(good)

    tasks = np.concatenate(kept, axis=0)[:n_samples]
    stats = {
        "n_drawn": n_drawn,
        "n_forced_dust": n_forced_dust_total,
        "n_rejected_cwv": n_rejected_cwv_total,
        "n_kept": int(tasks.shape[0]),
    }
    return tasks, stats


# Generate filtered Sobol samples BEFORE building LUTs
tasks, sample_stats = sample_valid_tasks(N_SAMPLES, mix_log_weight=MIX_LOG_WEIGHT)

print(
    "Sample filter (before LUT build): "
    f"drawn={sample_stats['n_drawn']}, "
    f"forced_dust(grain<{GRAIN_DUST_THRESHOLD:g})={sample_stats['n_forced_dust']}, "
    f"rejected_cwv={sample_stats['n_rejected_cwv']}, "
    f"kept={sample_stats['n_kept']}"
)

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


ds_mod = xr.open_zarr(MODTRAN_PATH)
wl_mod = ds_mod.wl.values
target_dims = ("surface_elevation_km", "observer_zenith", "relative_azimuth", "AOT550", "H2OSTR", "wl")
modtran_grid = [ds_mod[k].values for k in ["surface_elevation_km", "observer_zenith", "relative_azimuth", "AOT550", "H2OSTR"]]

# # NOTE: Loading Atmospheric radiative transfer LUT
print("building modtran interpolators")
v_interp_Latm = VectorInterpolator(modtran_grid, ds_mod.rhoatm.transpose(*target_dims).values, version="mlg")
v_interp_sphalb = VectorInterpolator(modtran_grid, ds_mod.sphalb.transpose(*target_dims).values, version="mlg")
v_interp_Lraw = {k: VectorInterpolator(modtran_grid, ds_mod[k].transpose(*target_dims).values, version="mlg")
                 for k in ['dir-dir', 'dif-dir', 'dir-dif', 'dif-dif']}

# # NOTE: Loading Snow Surface radiative transfer LUT
print("loading disort")
ds_dis = xr.load_dataset(DISORT_PATH)
disort_grid = [ds_dis[k].values for k in ["sza", "vza", "raa", "grain_radius", "algae_conc", "dust_conc","lwc"]]
v_interp_r_dd = VectorInterpolator(disort_grid, ds_dis.r_dd.values, version="mlg")
v_interp_r_hd = VectorInterpolator(disort_grid, ds_dis.r_hd.values, version="mlg")

emit_specs = pd.read_csv(EMIT_WAVE_PATH, sep=r'\s+', names=['idx', 'wl', 'fwhm'])
emit_noise = pd.read_csv(NOISE_PATH, sep=r'\s+', names=['wvl', 'a', 'b', 'c', 'rmse'], comment='#')

H_matrix = calculate_resample_matrix(wl_mod, emit_specs.wl.values, emit_specs.fwhm.values)

endmembers = np.array(pd.read_csv(ENDMEMBER_PATH))[:,1:]

# Fixed LUT solar zenith cosine (not swept); stored per sample as model aux.
COSZEN_LUT = float(ds_mod.coszen)


def softmax(z):
    "Used to maintain sum-to-1 condition and positive fractional covers"
    return np.exp(z) / np.sum(np.exp(z))

def apply_noise(rdn, wl, noise_df):
    a = np.interp(wl, noise_df['wvl'], noise_df['a']) 
    b = np.interp(wl, noise_df['wvl'], noise_df['b']) 
    c = np.interp(wl, noise_df['wvl'], noise_df['c']) 
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c 
    Sy = np.diagflat(np.power(nedl, 2))
    return rdn + np.random.multivariate_normal(np.zeros(rdn.shape), Sy)

# NOTE: this is similar (hopefully) to the forward.py in isofit
def simulate_pixel(
    cosi, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot,
    ele_km, raa_true, wl_mod, ds_dis_wl, solar_irr_emit, coszen,
):
    raa_disort = 180.0 - raa_true
    lookup_pt = np.array([np.degrees(np.arccos(cosi)), VZA_TRUE, raa_disort, grain, algae, dust, lwc])
    # DISORT interpolators give radiance at ds_dis.wavelength (22)
    rho_dd_22 = v_interp_r_dd(lookup_pt)
    rho_hd_22 = v_interp_r_hd(lookup_pt)

    # Resample DISORT radiance to MODTRAN wavelengths (285)
    rho_dd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_dd_22)
    rho_hd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_hd_22)
    # f = softmax(np.array([fsnow, fPV, fNPV, fsoil]))
    f = np.array([fsnow, fPV, fNPV, fsoil])

    rho_dd = rho_dd * f[0] + np.dot(endmembers, f[1:])
    rho_hd = rho_hd * f[0] + np.dot(endmembers, f[1:])

    coszen = float(coszen)
    # print("coszen: ", coszen)
    solar_irr = ds_mod.solar_irr.values
    # print("solar_irr: ", solar_irr)
    atm_pt = np.array([
        ele_km,            # surface_elevation_km
        180.0 - VZA_TRUE,  # observer_zenith
        raa_true,          # relative_azimuth (RAA_TRUE)
        aot,               # AOT550
        cwv,               # H2OSTR
    ])

    L_atm = v_interp_Latm(atm_pt)
    s_alb = v_interp_sphalb(atm_pt)
    L_raw = [v_interp_Lraw[k](atm_pt) for k in ['dir-dir', 'dif-dir', 'dir-dif', 'dif-dif']]
    solar_irr_emit = np.dot(H_matrix, solar_irr)

    # eq_11_term = 1 - (s_alb * rho_hd)
    # L_dir_dir = (L_raw[0] / coszen) * cosi
    # L_dif_dir = (L_raw[1] * (cosi / coszen)) / eq_11_term
    # L_dir_dif = (L_raw[2] / coszen) * coszen
    # L_dif_dif = L_raw[3] / eq_11_term
    # L_path_sum = sum(L_raw)

    # # # NOTE: for BG_RFL we can just assume same as target
    # toa_rdn = (
    #     L_atm +
    #     L_dir_dir * rho_dd +
    #     L_dif_dir * rho_hd +
    #     L_dir_dif * rho_hd + 
    #     L_dif_dif * rho_hd +
    #     (L_path_sum * rho_hd) / (1 - s_alb*rho_hd)
    # )

    # ISOFIT 6c: form L_tot before eq. 11 on diffuse terms; residual is
    # (L_tot * s * rho^2) / (1 - s * rho), not L_raw * rho / (1 - s * rho).
    eq_11_term = 1.0 - (s_alb * rho_hd)
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

    # NOTE: The parameteric noise model is used on the TOA RDN and then converted to TOA reflectance
    rdn_emit = np.dot(H_matrix, toa_rdn)
    rdn_noisy = apply_noise(rdn_emit, emit_specs.wl.values, emit_noise)
    toa_ref = rdn_noisy * np.pi / (solar_irr_emit * coszen)
    
    return rdn_noisy, toa_ref

# Number of Sobol Samples
n_samples = tasks.shape[0]

print(f"Starting parallel simulation on {n_samples} Sobol samples...")

# print(ds_mod)
# print(ds_mod.data_vars)
# print(ds_mod.coords)

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
        wl_mod,
        ds_dis.wavelength.values,
        ds_mod.solar_irr,
        COSZEN_LUT,
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
        "coszen": (["sample"], np.full(n_samples, COSZEN_LUT, dtype=np.float64)),
    },
    coords={
        "sample": np.arange(len(tasks)),
        "wl": emit_specs.wl.values,
    },
    attrs={
        "skyview_factor": SVF_TRUE,
        "VZA": VZA_TRUE,
        "Constant_background_reflectance": BG_RFL,
        "ele_km_range": f"[{ELE_LO:g}, {ELE_HI:g}]",
        "RAA_TRUE_range": f"[{RAA_LO:g}, {RAA_HI:g}]",
        "coszen_lut": COSZEN_LUT,
        #ToDo: update this path
        "noise_model": "https://github.com/isofit/isofit-data/blob/main/emit_noise.txt",
        "RT_atmosphere": "sRTMnet 6c",
        "RT_surface": "DISORT based snow surface LUT",
        "sampling": (
            f"Sobol({N_SOBOL_DIMS}); mixture decode for liquid_water,dust,algae "
            f"(P(log)={MIX_LOG_WEIGHT:g}, floors {LWC_LO:g},{DUST_LO:g},{ALGAE_LO:g}); "
            "linear-uniform for cos_i,grain_size,cwv,aot,ele_km,RAA_TRUE; "
            f"dust forced to {DUST_LO:g} when grain_size<{GRAIN_DUST_THRESHOLD:g}; "
            "cwv kept under ATM_MIDLAT_WINTER(ele_km)"
        ),
        "log_uniform_qois": ",".join(LOG_UNIFORM_QOIS),
        "mix_log_weight": float(MIX_LOG_WEIGHT),
        "n_drawn": sample_stats["n_drawn"],
        "n_forced_dust": sample_stats["n_forced_dust"],
        "n_rejected_cwv": sample_stats["n_rejected_cwv"],
        "n_kept": sample_stats["n_kept"],
    }
)
ds_out.to_netcdf(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
