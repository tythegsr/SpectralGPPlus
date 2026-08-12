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
OUTPUT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/data/snow_toa_fsnow_only_20260308.nc"

ELE_TRUE = 3.0
SVF_TRUE = 0.95
RAA_TRUE = 164.0
VZA_TRUE = 6.0

# NOTE: this convention is flipped from ISOFIT RAA. Still spans 0-180 but just something to be aware of.
RAA_DISORT = 180.0 - RAA_TRUE 

# # NOTE: background reflectance should be assumed equal to the target pixel
# per convo with Niklas today.
BG_RFL = "None"

COSI_RANGE = np.linspace(0.06, 1.0, 200)
GRAIN_RANGE = np.linspace(30, 1500, 100)
LWC_RANGE = np.linspace(0, 25, 5)
DUST_RANGE = np.linspace(0, 4000, 8)
ALGAE_RANGE = np.linspace(0, 6e5, 6)
FSNOW_RANGE = np.linspace(0, 10, 10)
FPV_RANGE = np.linspace(0, 10, 10)
FNPV_RANGE = np.linspace(0, 10, 10)
FSOIL_RANGE = np.linspace(0, 10, 10)
CWV_RANGE = np.linspace(0.2, 5.2, 25)
AOT_RANGE = np.linspace(0.04, 1.0, 50)

# Column order: cos_i, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot
# Zero-floored free dims (LWC / dust / algae): mixture of log-uniform and
# linear-uniform decode. MIX_LOG_WEIGHT is P(log); 1.0 = pure log, 0.0 = pure lin.
# Small positive floors keep log-space defined.
N_SAMPLES = 59000
SOBOL_SEED = 42
MIX_LOG_WEIGHT = 0.5

COSI_LO, COSI_HI = 0.06, 1.0
GRAIN_LO, GRAIN_HI = 30.0, 1500.0
# Design windows still span [0, hi] physically; log decode needs lo > 0.
LWC_LO, LWC_HI = 1e-2, 25.0
DUST_LO, DUST_HI = 1e-2, 4000.0
ALGAE_LO, ALGAE_HI = 1e-2, 6e5
FSNOW_LO, FSNOW_HI = 10.0, 10.0  # fsnow-only
FPV_LO, FPV_HI = 0.0, 0.0
FNPV_LO, FNPV_HI = 0.0, 0.0
FSOIL_LO, FSOIL_HI = 0.0, 0.0
CWV_LO, CWV_HI = 0.2, 5.2
AOT_LO, AOT_HI = 0.04, 1.0

LOG_UNIFORM_QOIS = ("liquid_water", "dust", "algae")
# Sobol: 11 QoI coords + one mix gate per mixed QoI (independent Bernoulli).
N_QOI_DIMS = 11
N_SOBOL_DIMS = N_QOI_DIMS + len(LOG_UNIFORM_QOIS)  # 14


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


def decode_sobol_tasks(u, mix_log_weight=MIX_LOG_WEIGHT):
    """
    Decode Sobol u in [0,1]^{14} to physical QoIs (n, 11).

    Columns 0..10: QoI coordinates.
    Columns 11,12,13: mix gates for liquid_water, dust, algae.
    """
    if u.ndim != 2 or u.shape[1] != N_SOBOL_DIMS:
        raise ValueError(f"expected u shape (n, {N_SOBOL_DIMS}), got {u.shape}")
    tasks = np.empty((u.shape[0], N_QOI_DIMS), dtype=np.float64)
    tasks[:, 0] = lin_map(u[:, 0], COSI_LO, COSI_HI)
    tasks[:, 1] = lin_map(u[:, 1], GRAIN_LO, GRAIN_HI)
    tasks[:, 2] = mix_map(u[:, 2], u[:, 11], LWC_LO, LWC_HI, mix_log_weight)
    tasks[:, 3] = mix_map(u[:, 3], u[:, 12], DUST_LO, DUST_HI, mix_log_weight)
    tasks[:, 4] = mix_map(u[:, 4], u[:, 13], ALGAE_LO, ALGAE_HI, mix_log_weight)
    tasks[:, 5] = lin_map(u[:, 5], FSNOW_LO, FSNOW_HI)
    tasks[:, 6] = lin_map(u[:, 6], FPV_LO, FPV_HI)
    tasks[:, 7] = lin_map(u[:, 7], FNPV_LO, FNPV_HI)
    tasks[:, 8] = lin_map(u[:, 8], FSOIL_LO, FSOIL_HI)
    tasks[:, 9] = lin_map(u[:, 9], CWV_LO, CWV_HI)
    tasks[:, 10] = lin_map(u[:, 10], AOT_LO, AOT_HI)
    return tasks


# Generate random samples for the 11-dim dataset (+ mix gates)
sampler = qmc.Sobol(d=N_SOBOL_DIMS, scramble=True, seed=SOBOL_SEED)
# SciPy Sobol prefers n = 2^k for balance; draw then truncate
n_pow2 = 1 << int(np.ceil(np.log2(max(N_SAMPLES, 2))))
sample = sampler.random(n_pow2)[:N_SAMPLES]
tasks = decode_sobol_tasks(sample, mix_log_weight=MIX_LOG_WEIGHT)

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
def simulate_pixel(cosi, grain, lwc, dust, algae, fsnow, fPV, fNPV, fsoil, cwv, aot, wl_mod, ds_dis_wl, solar_irr_emit, coszen):

    lookup_pt = np.array([np.degrees(np.arccos(cosi)), VZA_TRUE, RAA_DISORT, grain, algae, dust, lwc])
    # DISORT interpolators give radiance at ds_dis.wavelength (22)
    rho_dd_22 = v_interp_r_dd(lookup_pt)
    rho_hd_22 = v_interp_r_hd(lookup_pt)

    # Resample DISORT radiance to MODTRAN wavelengths (285)
    rho_dd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_dd_22)
    rho_hd = np.interp(ds_mod.wl.values, ds_dis_wl, rho_hd_22)
    f = softmax(np.array([fsnow, fPV, fNPV, fsoil]))

    rho_dd = rho_dd * f[0] + np.dot(endmembers, f[1:])
    rho_hd = rho_hd * f[0] + np.dot(endmembers, f[1:])

    coszen = float(ds_mod.coszen)
    solar_irr = ds_mod.solar_irr.values
    atm_pt = np.array([
        ELE_TRUE,          # surface_elevation_km
        180.0 - VZA_TRUE,  # observer_zenith
        RAA_TRUE,          # relative_azimuth
        aot,               # AOT550
        cwv,               # H2OSTR
    ])

    L_atm = v_interp_Latm(atm_pt)
    s_alb = v_interp_sphalb(atm_pt)
    L_raw = [v_interp_Lraw[k](atm_pt) for k in ['dir-dir', 'dif-dir', 'dir-dif', 'dif-dif']]
    solar_irr_emit = np.dot(H_matrix, solar_irr)

    eq_11_term = 1 - (s_alb * rho_hd)
    L_dir_dir = (L_raw[0] / coszen) * cosi
    L_dif_dir = (L_raw[1] * (cosi / coszen)) / eq_11_term
    L_dir_dif = (L_raw[2] / coszen) * coszen
    L_dif_dif = L_raw[3] / eq_11_term
    L_path_sum = sum(L_raw)

    # # NOTE: for BG_RFL we can just assume same as target
    toa_rdn = (
        L_atm +
        L_dir_dir * rho_dd +
        L_dif_dir * rho_hd +
        L_dir_dif * rho_hd + 
        L_dif_dif * rho_hd +
        (L_path_sum * rho_hd) / (1 - s_alb*rho_hd)
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
        row[0], # COSI
        row[1],  # grain
        row[2],  # LWC
        row[3],  # dust
        row[4],  # algae
        row[5],  # f_snow
        row[6],  # f_pv
        row[7],  # f_npv
        row[8],  # f_soil
        row[9],  # CWV
        row[10], # AOT
        wl_mod, 
        ds_dis.wavelength.values, 
        ds_mod.solar_irr, 
        ds_mod.coszen
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
        #ToDo: update this path
        "noise_model": "https://github.com/isofit/isofit-data/blob/main/emit_noise.txt",
        "RT_atmosphere": "sRTMnet 6c",
        "RT_surface": "DISORT based snow surface LUT",
        "sampling": (
            f"Sobol({N_SOBOL_DIMS}); mixture decode for liquid_water,dust,algae "
            f"(P(log)={MIX_LOG_WEIGHT:g}, floors {LWC_LO:g},{DUST_LO:g},{ALGAE_LO:g}); "
            "linear-uniform for cos_i,grain_size,cwv,aot; fsnow fixed; fractions fixed at 0"
        ),
        "log_uniform_qois": ",".join(LOG_UNIFORM_QOIS),
        "mix_log_weight": float(MIX_LOG_WEIGHT),
    }
)
ds_out.to_netcdf(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
