# This is essentially what I ran for the data for Otto/Will's RNTD Niklas was mentioning.

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
OUTPUT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/data/snow_toa_simulations_20261607.nc"

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

# Generate random samples for the 11-dim dataset
sampler = qmc.Sobol(d=11, scramble=True, seed=42)
sample = sampler.random(59000)

# Scale the data
l_bound = np.array([0.06, 30, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.04])
u_bound = np.array([1.0, 1500, 25, 4000, 6e5, 10, 10, 10, 10, 5.2, 1.0])

tasks = sample * (u_bound - l_bound) + l_bound

print(tasks)

mean_per_feat = np.mean(tasks, axis=0)
print("mean per feature: ", mean_per_feat)

std_per_feat = np.std(tasks, axis=0)
print("std per feature: ", std_per_feat)


ds_mod = xr.open_zarr(MODTRAN_PATH)
wl_mod = ds_mod.wl.values
target_dims = ("surface_elevation_km", "observer_zenith", "relative_azimuth", "AOT550", "H2OSTR", "wl")
modtran_grid = [ds_mod[k].values for k in ["surface_elevation_km", "observer_zenith", "relative_azimuth", "AOT550", "H2OSTR"]]

# sRTMnet 6c LUTs store rhoatm + dir-* coupling products already in radiance (RT_mode=rdn).
# Only transm-mode LUTs need multiplication by solar_irr * coszen / pi.
RT_MODE = str(ds_mod.attrs.get("RT_mode", ds_mod.attrs.get("rt_mode", "rdn"))).lower()
print(f"Atmospheric LUT RT_mode={RT_MODE}")

# # NOTE: Loading Atmospheric radiative transfer LUT
print("building modtran interpolators")
v_interp_rhoatm = VectorInterpolator(modtran_grid, ds_mod.rhoatm.transpose(*target_dims).values, version="mlg")
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

    rho_atm = v_interp_rhoatm(atm_pt)
    s_alb = v_interp_sphalb(atm_pt)
    # Coupling terms from LUT; convert transmittance -> radiance only in transm mode.
    scale = (solar_irr * coszen / np.pi) if RT_MODE == "transm" else 1.0
    L_raw = [v_interp_Lraw[k](atm_pt) * scale for k in ['dir-dir', 'dif-dir', 'dir-dif', 'dif-dif']]
    L_atm = rho_atm * scale if RT_MODE == "transm" else rho_atm
    solar_irr_emit = np.dot(H_matrix, solar_irr)

    # ISOFIT 6c: apply cosi scaling, form L_tot *before* eq. 11 on diffuse terms,
    # then residual uses (L_tot * s * rho^2) / (1 - s * rho) — not L_tot * rho / (...).
    eq_11_term = 1.0 - (s_alb * rho_hd)
    L_dir_dir = (L_raw[0] / coszen) * cosi
    L_dif_dir = L_raw[1] * (cosi / coszen)
    L_dir_dif = L_raw[2]  # flat background: cos_i_bg = coszen
    L_dif_dif = L_raw[3]
    L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
    L_dif_dir = L_dif_dir / eq_11_term
    L_dif_dif = L_dif_dif / eq_11_term
    atm_surface_scattering = s_alb * rho_hd

    # # NOTE: for BG_RFL we can just assume same as target (homogeneous)
    toa_rdn = (
        L_atm
        + L_dir_dir * rho_dd
        + L_dif_dir * rho_hd
        + L_dir_dif * rho_hd
        + L_dif_dif * rho_hd
        + (L_tot * atm_surface_scattering * rho_hd) / eq_11_term
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
        "RT_mode": RT_MODE,
        "RT_surface": "DISORT based snow surface LUT"
    }
)
ds_out.to_netcdf(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
