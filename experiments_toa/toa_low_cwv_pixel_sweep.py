"""Low-CWV pixel parameter sweep: stored truth vs forward-model spectra.

Loads a synthetic NetCDF from ``toa_simulations_august.py``, selects a low-CWV
pixel as baseline, and overlays the stored ``toa_reflectance`` (truth, with noise)
against noise-free forward-model spectra while sweeping user-chosen QoI parameters.

Forward physics match ``toa_simulations_august.py`` (physical cover fractions,
ISOFIT 6c RT). See that file as source of truth for ``simulate_pixel``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import xarray as xr

_NC_NAME = "snow_toa_fsnow_90to100_constrained_20262608.nc"


def _resolve_data_dir() -> Path:
    """Folder with LUTs / common.py (august-style disort_data_for_tyler)."""
    candidates = [
        _SCRIPT_DIR,
        Path(r"C:/Users/tylerj/isofit/disort_data_for_tyler"),
        _SCRIPT_DIR.parent / "disort_data_for_tyler",
    ]
    for d in candidates:
        if (d / "lut.zarr").exists() or (d / "common.py").exists():
            return d
    return candidates[0]


def _resolve_default_nc() -> Path:
    """NetCDF from august output or SpectralGPPlus data folder."""
    data_dir = _resolve_data_dir()
    candidates = [
        data_dir / "data" / _NC_NAME,
        _SCRIPT_DIR / "data" / _NC_NAME,
        _SCRIPT_DIR / "data 11 QoI" / _NC_NAME,
        _ROOT / "experiments_toa" / "data 11 QoI" / _NC_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Prefer august output location for a clear error if missing.
    return data_dir / "data" / _NC_NAME


def _resolve_default_out() -> Path:
    data_dir = _resolve_data_dir()
    if (_SCRIPT_DIR / "data 11 QoI").exists():
        return _SCRIPT_DIR / "reports" / "low_cwv_pixel_sweep"
    return data_dir / "reports" / "low_cwv_pixel_sweep"


_DATA = _resolve_data_dir()
DEFAULT_NC = _resolve_default_nc()
DEFAULT_OUT = _resolve_default_out()


def _ensure_common_import(common_dir: Path | None) -> None:
    dirs: list[Path] = []
    if common_dir is not None:
        dirs.append(common_dir)
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


MODTRAN_PATH = str(_DATA / "lut.zarr")
DISORT_PATH = str(_DATA / "disort_snow_lut_EMIT.nc")
ENDMEMBER_PATH = str(_DATA / "endmembers.csv")
EMIT_WAVE_PATH = str(_DATA / "emit-wave.txt")
NOISE_PATH = str(_DATA / "emit_noise.txt")

# Used when --sweep is omitted (one figure per inner list).
DEFAULT_SWEEPS: list[list[str]] = [
    ["cwv", "0.1", "0.2", "0.125", "0.15", "0.175"],
    # ["dust", "0.01", "10", "100", "500"],
]

VZA_TRUE = 6.0

PARAM_NAMES: tuple[str, ...] = (
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
    "ele_km",
    "RAA_TRUE",
    "coszen",
)

PARAM_ALIASES: dict[str, str] = {
    "grain": "grain_size",
    "lwc": "liquid_water",
    "cosi": "cos_i",
    "raa": "RAA_TRUE",
    "raa_true": "RAA_TRUE",
    "ele": "ele_km",
    "elevation": "ele_km",
}

TRUE_COLOR = "#1f4e79"
SWEEP_CMAP = "viridis"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.decode("latin-1", errors="replace")
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


def canonical_param(name: str) -> str:
    key = name.strip()
    lowered = key.lower()
    if lowered in PARAM_ALIASES:
        return PARAM_ALIASES[lowered]
    if key in PARAM_NAMES:
        return key
    allowed = sorted(set(PARAM_NAMES) | set(PARAM_ALIASES))
    raise ValueError(
        f"Unknown parameter {name!r}. Allowed: {', '.join(allowed)}"
    )


@dataclass
class BaselinePixel:
    sample_index: int
    wl: np.ndarray
    true_reflectance: np.ndarray
    params: dict[str, float] = field(default_factory=dict)
    nc_attrs: dict[str, Any] = field(default_factory=dict)
    nc_path: str = ""


def select_sample_index(
    cwv: np.ndarray,
    *,
    sample_index: int | None,
    cwv_pick: str,
    cwv_pctl: float,
) -> int:
    n = int(cwv.size)
    if sample_index is not None:
        if not (0 <= sample_index < n):
            raise ValueError(f"--sample-index {sample_index} out of range [0, {n})")
        return int(sample_index)

    cwv_pick = cwv_pick.lower()
    if cwv_pick == "min":
        return int(np.argmin(cwv))
    if cwv_pick == "pctl":
        if not (0.0 < cwv_pctl <= 100.0):
            raise ValueError(f"--cwv-pctl must be in (0, 100], got {cwv_pctl}")
        threshold = float(np.percentile(cwv, cwv_pctl))
        candidates = np.flatnonzero(cwv <= threshold + 1e-12)
        if candidates.size == 0:
            return int(np.argmin(cwv))
        return int(candidates[0])
    raise ValueError(f"Unknown --cwv-pick {cwv_pick!r}; use 'min' or 'pctl'")


def load_baseline_pixel(
    nc_path: Path,
    *,
    sample_index: int | None,
    cwv_pick: str,
    cwv_pctl: float,
) -> BaselinePixel:
    with h5py.File(nc_path, "r") as f:
        missing = [name for name in PARAM_NAMES if name not in f]
        if missing:
            raise ValueError(f"NetCDF missing variables: {missing}")
        if "toa_reflectance" not in f:
            raise ValueError("NetCDF missing toa_reflectance")
        if "wl" not in f:
            raise ValueError("NetCDF missing wl")

        cwv = np.asarray(f["cwv"][:], dtype=np.float64)
        idx = select_sample_index(
            cwv,
            sample_index=sample_index,
            cwv_pick=cwv_pick,
            cwv_pctl=cwv_pctl,
        )
        params = {name: float(f[name][idx]) for name in PARAM_NAMES}
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        true_refl = np.asarray(f["toa_reflectance"][idx], dtype=np.float64)
        attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
        return BaselinePixel(
            sample_index=idx,
            wl=wl,
            true_reflectance=true_refl,
            params=params,
            nc_attrs=attrs,
            nc_path=str(nc_path),
        )


@dataclass
class ForwardModel:
    """Forward model matching ``toa_simulations_august.simulate_pixel``."""

    wl_mod: np.ndarray
    wl_emit: np.ndarray
    ds_dis_wl: np.ndarray
    h_matrix: np.ndarray
    endmembers: np.ndarray
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
        ele_km: float,
        raa_true: float,
        coszen: float,
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        raa_disort = 180.0 - raa_true
        lookup_pt = np.array(
            [
                np.degrees(np.arccos(cos_i)),
                VZA_TRUE,
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

        f = np.array([fsnow, f_pv, f_npv, f_soil], dtype=np.float64)
        rho_dd = rho_dd * f[0] + np.dot(self.endmembers, f[1:])
        rho_hd = rho_hd * f[0] + np.dot(self.endmembers, f[1:])

        coszen = float(coszen)
        atm_pt = np.array(
            [ele_km, 180.0 - VZA_TRUE, raa_true, aot, cwv], dtype=np.float64
        )
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
        l_dir_dif = l_raw[2]
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

        rdn_emit = np.dot(self.h_matrix, toa_rdn)
        if add_noise:
            rdn_out = apply_noise(rdn_emit, self.wl_emit, self.emit_noise)
        else:
            rdn_out = rdn_emit
        toa_ref = rdn_out * np.pi / (solar_irr_emit * coszen)
        return rdn_out, toa_ref


def apply_noise(
    rdn: np.ndarray, wl: np.ndarray, noise_df: pd.DataFrame
) -> np.ndarray:
    a = np.interp(wl, noise_df["wvl"], noise_df["a"])
    b = np.interp(wl, noise_df["wvl"], noise_df["b"])
    c = np.interp(wl, noise_df["wvl"], noise_df["c"])
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c
    sy = np.diagflat(np.power(nedl, 2))
    return rdn + np.random.multivariate_normal(np.zeros(rdn.shape), sy)


def load_forward_model(*, wl_emit: np.ndarray, common_dir: Path | None) -> ForwardModel:
    _ensure_common_import(common_dir)
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
        solar_irr=np.asarray(ds_mod.solar_irr.values, dtype=np.float64),
        emit_noise=emit_noise,
        v_interp_latm=v_interp_latm,
        v_interp_sphalb=v_interp_sphalb,
        v_interp_lraw=v_interp_lraw,
        v_interp_r_dd=v_interp_r_dd,
        v_interp_r_hd=v_interp_r_hd,
    )


def params_to_sim_kwargs(params: dict[str, float]) -> dict[str, float]:
    return {
        "cos_i": params["cos_i"],
        "grain": params["grain_size"],
        "lwc": params["liquid_water"],
        "dust": params["dust"],
        "algae": params["algae"],
        "fsnow": params["fsnow"],
        "f_pv": params["fPV"],
        "f_npv": params["fNPV"],
        "f_soil": params["fsoil"],
        "cwv": params["cwv"],
        "aot": params["aot"],
        "ele_km": params["ele_km"],
        "raa_true": params["RAA_TRUE"],
        "coszen": params["coszen"],
    }


def simulate_from_params(
    model: ForwardModel,
    params: dict[str, float],
    *,
    add_noise: bool = False,
) -> np.ndarray:
    _, toa_ref = model.simulate_pixel(**params_to_sim_kwargs(params), add_noise=add_noise)
    return toa_ref


def spectrum_metrics(sim: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    sim = np.asarray(sim, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    delta = sim - truth
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_bias": float(np.mean(delta)),
        "mean_abs_rel_error": float(
            np.mean(np.abs(delta) / np.maximum(np.abs(truth), 1e-6))
        ),
    }


@dataclass
class SweepSpec:
    param: str
    values: list[float]


def parse_sweep_args(sweep_args: list[list[str]]) -> list[SweepSpec]:
    if not sweep_args:
        raise ValueError("At least one --sweep PARAM v1 v2 ... is required")
    specs: list[SweepSpec] = []
    for raw in sweep_args:
        if len(raw) < 2:
            raise ValueError(f"--sweep requires PARAM and at least one value, got {raw!r}")
        param = canonical_param(raw[0])
        try:
            values = [float(v) for v in raw[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric sweep values for {param}: {raw[1:]}") from exc
        specs.append(SweepSpec(param=param, values=values))
    return specs


def run_parameter_sweep(
    model: ForwardModel,
    baseline: BaselinePixel,
    spec: SweepSpec,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_vals = len(spec.values)
    n_bands = baseline.wl.size
    spectra = np.empty((n_vals, n_bands), dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for i, value in enumerate(spec.values):
        params = dict(baseline.params)
        params[spec.param] = float(value)
        sim = simulate_from_params(model, params, add_noise=False)
        spectra[i] = sim
        metrics = spectrum_metrics(sim, baseline.true_reflectance)
        rows.append(
            {
                "param": spec.param,
                "value": float(value),
                "is_baseline_value": bool(
                    np.isclose(value, baseline.params[spec.param], rtol=0, atol=1e-9)
                ),
                **metrics,
            }
        )
    return spectra, rows


def plot_parameter_sweep(
    wl: np.ndarray,
    true_refl: np.ndarray,
    spec: SweepSpec,
    spectra: np.ndarray,
    *,
    sample_index: int,
    baseline_cwv: float,
    out_path: Path,
) -> None:
    n_vals = len(spec.values)
    cmap = plt.get_cmap(SWEEP_CMAP)
    colors = [cmap(i / max(n_vals - 1, 1)) for i in range(n_vals)]

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
    )

    ax0.plot(
        wl,
        true_refl,
        color=TRUE_COLOR,
        lw=2.2,
        label="stored truth (toa_reflectance)",
        zorder=10,
    )
    for i, (value, color) in enumerate(zip(spec.values, colors)):
        ax0.plot(
            wl,
            spectra[i],
            color=color,
            lw=1.4,
            label=f"{spec.param}={value:g}",
        )

    ax0.set_ylabel("TOA reflectance")
    ax0.set_title(
        f"Sample {sample_index} (baseline cwv={baseline_cwv:.4g}): sweep {spec.param}"
    )
    ax0.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    ax0.set_ylim(bottom=0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1.axhline(0.0, color="0.55", lw=0.8)
    for i, (value, color) in enumerate(zip(spec.values, colors)):
        delta = spectra[i] - true_refl
        ax1.plot(wl, delta, color=color, lw=1.2, label=f"{spec.param}={value:g}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("Δ reflectance (sim − truth)")
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_baseline_json(baseline: BaselinePixel, out_dir: Path) -> None:
    payload = {
        "nc_path": baseline.nc_path,
        "sample_index": baseline.sample_index,
        "params": baseline.params,
        "nc_attrs": baseline.nc_attrs,
        "cwv": baseline.params["cwv"],
        "mean_true_reflectance": float(np.mean(baseline.true_reflectance)),
    }
    path = out_dir / "baseline_pixel.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)


def run(args: argparse.Namespace) -> None:
    nc_path = Path(args.nc)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"NetCDF not found: {nc_path}\n"
            f"  Generate with toa_simulations_august.py or pass --nc explicitly.\n"
            f"  Expected default (august output): {_resolve_data_dir() / 'data' / _NC_NAME}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_pixel(
        nc_path,
        sample_index=args.sample_index,
        cwv_pick=args.cwv_pick,
        cwv_pctl=args.cwv_pctl,
    )
    print(f"Loaded {nc_path}")
    print(
        f"Baseline sample_index={baseline.sample_index} cwv={baseline.params['cwv']:.6g} "
        f"(pick={args.cwv_pick})"
    )
    for name in PARAM_NAMES:
        print(f"  {name}: {baseline.params[name]:.6g}")

    write_baseline_json(baseline, out_dir)

    print("Loading forward model LUTs...")
    model = load_forward_model(wl_emit=baseline.wl, common_dir=args.common_dir)

    sweep_specs = parse_sweep_args(args.sweep)
    sweep_summary: dict[str, Any] = {
        "baseline_sample_index": baseline.sample_index,
        "baseline_cwv": baseline.params["cwv"],
        "sweeps": {},
    }

    for spec in sweep_specs:
        print(f"Sweeping {spec.param}: {spec.values}")
        spectra, rows = run_parameter_sweep(model, baseline, spec)
        sweep_summary["sweeps"][spec.param] = {
            "values": spec.values,
            "curves": rows,
        }
        plot_path = out_dir / f"{spec.param}_sweep.png"
        plot_parameter_sweep(
            baseline.wl,
            baseline.true_reflectance,
            spec,
            spectra,
            sample_index=baseline.sample_index,
            baseline_cwv=baseline.params["cwv"],
            out_path=plot_path,
        )
        print(f"  Wrote {plot_path}")

    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(sweep_summary), f, indent=2)

    print(f"Wrote outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", type=Path, default=DEFAULT_NC, help="Synthetic NetCDF path")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Use this sample row instead of low-CWV selection",
    )
    p.add_argument(
        "--cwv-pick",
        choices=("min", "pctl"),
        default="min",
        help="How to pick low-CWV pixel when --sample-index is unset",
    )
    p.add_argument(
        "--cwv-pctl",
        type=float,
        default=5.0,
        help="Percentile threshold when --cwv-pick pctl (default 5)",
    )
    p.add_argument(
        "--common-dir",
        type=Path,
        default=None,
        help="Directory containing common.py (default: disort_data_for_tyler)",
    )
    p.add_argument(
        "--sweep",
        nargs="+",
        action="append",
        metavar=("PARAM", "VALUE"),
        help=(
            "Repeatable: PARAM v1 v2 ... (one figure per --sweep). "
            f"If omitted, defaults to: {DEFAULT_SWEEPS}"
        ),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.sweep:
        args.sweep = [list(s) for s in DEFAULT_SWEEPS]
        print(f"No --sweep given; using defaults: {DEFAULT_SWEEPS}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
