"""Compare EMIT real pixels vs synthetic TOA domain (state + reflectance).

Streaming over emit_data.nc (~1.75M x 285). Writes JSON, markdown, and PNGs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
from experiments_toa.s2_constants import S2_TASK_NAMES

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMIT = _ROOT / "split_files" / "emit_data.nc"
DEFAULT_SIM = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261208.nc"
)
DEFAULT_UNFILTERED = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_simulations_20262107.nc"
)
DEFAULT_OUT = _ROOT / "experiments_toa" / "reports" / "emit_vs_sim"

EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
SIM_QOI_NAMES = list(S2_TASK_NAMES)

# Canonical synthetic design boxes (physical units). Fraction boxes are [0, 1]
# after softmax; the 70-100 file already stores post-softmax covers.
DESIGN_BOX: dict[str, tuple[float, float]] = {
    "cos_i": (0.06, 1.0),
    "grain_size": (30.0, 1500.0),
    "liquid_water": (1e-2, 25.0),
    "dust": (1e-2, 4000.0),
    "algae": (1e-2, 6e5),
    "aot": (0.04, 1.0),
    "cwv": (0.2, 5.2),
    "fsnow": (0.0, 1.0),
    "fPV": (0.0, 1.0),
    "fNPV": (0.0, 1.0),
    "fsoil": (0.0, 1.0),
}

# 70-100 file snow-fraction filter (post-softmax).
FSNOW70_BOX: dict[str, tuple[float, float]] = {
    **DESIGN_BOX,
    "fsnow": (0.70, 1.0),
}

MAPPED_QOI = (
    "grain_size",
    "liquid_water",
    "dust",
    "algae",
    "aot",
    "cwv",
    "fsnow",
    "fPV",
    "fNPV",
    "fsoil",
)

WRAP_WIDTH = 1280  # packed sample index 0..n-1; not a recovered geographic grid
CHUNK = 8192
SUBSAMPLE = 50000
PCA_N = 20000
HEAT_DS_W = 48
HEAT_DS_H = 32
SPEC_CANVAS_STEP = 7
NIR_TARGET_NM = 900.0
GRAIN_DEFAULT = 500.0
GRAIN_DEFAULT_TOL = 0.1


def _as_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return str(v.reshape(-1)[0])
        return ",".join(str(x) for x in v.tolist()[:12])
    return str(v)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        x = obj.item()
        if isinstance(x, float) and (not math.isfinite(x)):
            return None
        return x
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _finite_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    n_all = int(x.size)
    n_nonfinite = int((~np.isfinite(x)).sum())
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "n_all": n_all,
            "n_nonfinite": n_nonfinite,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "n_zero": 0,
            "frac_zero": 0.0,
        }
    qs = np.percentile(x, [1, 5, 50, 95, 99])
    n_zero = int((np.abs(x) < 1e-12).sum())
    return {
        "n": int(x.size),
        "n_all": n_all,
        "n_nonfinite": n_nonfinite,
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
        "n_zero": n_zero,
        "frac_zero": float(n_zero / x.size),
    }


def _in_box(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.isfinite(x) & (x >= lo) & (x <= hi)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)


def fractions_look_physical(fsnow, fpv, fnpv, fsoil) -> bool:
    stacked = np.column_stack([fsnow, fpv, fnpv, fsoil]).astype(np.float64)
    s = stacked.sum(axis=1)
    return bool(
        stacked.min() >= -1e-4
        and stacked.max() <= 1.05
        and float(np.median(s)) > 0.8
        and float(np.median(s)) < 1.2
    )


def read_root_attrs(h5) -> dict[str, str]:
    return {str(k): _as_str(h5.attrs[k]) for k in h5.attrs}


def infer_wrap_grid(n: int, width: int = WRAP_WIDTH) -> tuple[int, int, int]:
    rows = int(math.ceil(n / width))
    rem = int(n % width)
    return rows, width, rem


def pack_to_grid(values: np.ndarray, width: int, fill=np.nan) -> np.ndarray:
    n = int(values.size)
    rows, width, rem = infer_wrap_grid(n, width)
    grid = np.full((rows, width), fill, dtype=np.float64)
    grid.ravel()[:n] = values.astype(np.float64, copy=False)
    return grid


def block_downsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = grid.shape
    ys = np.linspace(0, h, out_h + 1, dtype=int)
    xs = np.linspace(0, w, out_w + 1, dtype=int)
    out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            block = grid[ys[i] : max(ys[i + 1], ys[i] + 1), xs[j] : max(xs[j + 1], xs[j] + 1)]
            m = np.isfinite(block)
            if m.any():
                out[i, j] = float(block[m].mean())
    return out


def nearest_band(wl: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wl - target_nm)))


def load_sim_file(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        qoi = {name: np.asarray(f[name][:], dtype=np.float64) for name in SIM_QOI_NAMES}
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        refl = np.asarray(f["toa_reflectance"][:], dtype=np.float64)
        attrs = read_root_attrs(f)
    phys = fractions_look_physical(qoi["fsnow"], qoi["fPV"], qoi["fNPV"], qoi["fsoil"])
    if phys:
        frac = {
            "fsnow": qoi["fsnow"],
            "fPV": qoi["fPV"],
            "fNPV": qoi["fNPV"],
            "fsoil": qoi["fsoil"],
        }
        frac_kind = "already_physical"
    else:
        stacked = softmax_rows(
            np.column_stack([qoi["fsnow"], qoi["fPV"], qoi["fNPV"], qoi["fsoil"]])
        )
        frac = {
            "fsnow": stacked[:, 0],
            "fPV": stacked[:, 1],
            "fNPV": stacked[:, 2],
            "fsoil": stacked[:, 3],
        }
        frac_kind = "softmax_of_logits"
    pixel_mean = np.nanmean(refl, axis=1)
    return {
        "path": str(path.resolve()),
        "n": int(refl.shape[0]),
        "wl": wl,
        "refl": refl,
        "qoi": qoi,
        "frac": frac,
        "frac_kind": frac_kind,
        "attrs": attrs,
        "pixel_mean": pixel_mean,
        "spec_mean": np.nanmean(refl, axis=0),
        "spec_std": np.nanstd(refl, axis=0),
        "spec_p10": np.nanpercentile(refl, 10, axis=0),
        "spec_p50": np.nanpercentile(refl, 50, axis=0),
        "spec_p90": np.nanpercentile(refl, 90, axis=0),
        "refl_min": float(np.nanmin(refl)),
        "refl_max": float(np.nanmax(refl)),
        "n_neg": int(np.nansum(refl < 0)),
        "n_gt1": int(np.nansum(refl > 1)),
        "n_nonfinite": int(np.nansum(~np.isfinite(refl))),
    }


def stream_emit_reflectance(
    path: Path,
    *,
    masks: dict[str, np.ndarray],
    subsample_idx: np.ndarray,
    nir_idx: int,
) -> dict:
    n = None
    d = None
    spec_sum = None
    spec_sumsq = None
    spec_count = None
    strat_sum: dict[str, np.ndarray] = {}
    strat_count: dict[str, int] = {}
    pixel_mean = None
    nir = None
    n_neg = 0
    n_gt1 = 0
    n_nonfinite = 0
    sub_pos = {int(i): k for k, i in enumerate(subsample_idx)}
    sub_x = None
    next_sub = 0
    sub_sorted = subsample_idx

    with h5py.File(path, "r") as f:
        ds = f["reflectance"]
        n, d = int(ds.shape[0]), int(ds.shape[1])
        spec_sum = np.zeros(d, dtype=np.float64)
        spec_sumsq = np.zeros(d, dtype=np.float64)
        spec_count = np.zeros(d, dtype=np.float64)
        pixel_mean = np.empty(n, dtype=np.float64)
        nir = np.empty(n, dtype=np.float64)
        sub_x = np.empty((subsample_idx.size, d), dtype=np.float64)
        for name, mask in masks.items():
            strat_sum[name] = np.zeros(d, dtype=np.float64)
            strat_count[name] = 0
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            x = np.asarray(ds[start:stop], dtype=np.float64)
            finite = np.isfinite(x)
            n_nonfinite += int((~finite).sum())
            n_neg += int(np.sum(finite & (x < 0)))
            n_gt1 += int(np.sum(finite & (x > 1)))
            x_masked = np.where(finite, x, np.nan)
            pixel_mean[start:stop] = np.nanmean(x_masked, axis=1)
            nir[start:stop] = x_masked[:, nir_idx]
            spec_sum += np.nansum(x_masked, axis=0)
            spec_sumsq += np.nansum(x_masked**2, axis=0)
            spec_count += np.sum(finite, axis=0)
            for name, mask in masks.items():
                m = mask[start:stop]
                if m.any():
                    strat_sum[name] += np.nansum(x_masked[m], axis=0)
                    strat_count[name] += int(m.sum())
            while next_sub < sub_sorted.size and sub_sorted[next_sub] < stop:
                i = int(sub_sorted[next_sub])
                sub_x[sub_pos[i]] = x[i - start]
                next_sub += 1

    spec_mean = spec_sum / np.clip(spec_count, 1.0, None)
    var = spec_sumsq / np.clip(spec_count, 1.0, None) - spec_mean**2
    spec_std = np.sqrt(np.clip(var, 0.0, None))
    strat_mean = {
        k: (strat_sum[k] / max(strat_count[k], 1)) for k in masks
    }
    return {
        "n": n,
        "d": d,
        "pixel_mean": pixel_mean,
        "nir": nir,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "strat_mean": strat_mean,
        "strat_count": strat_count,
        "sub_x": sub_x,
        "n_neg": n_neg,
        "n_gt1": n_gt1,
        "n_nonfinite": n_nonfinite,
        "mean_pixel_mean": float(np.nanmean(pixel_mean)),
    }


def coverage_table(values: dict[str, np.ndarray], box: dict[str, tuple[float, float]]) -> dict:
    out = {}
    joint = None
    for name in MAPPED_QOI:
        x = values[name]
        lo, hi = box[name]
        inside = _in_box(x, lo, hi)
        out[name] = {
            "lo": lo,
            "hi": hi,
            "frac_inside": float(inside.mean()),
            "n_inside": int(inside.sum()),
            "n": int(x.size),
            **{k: v for k, v in _finite_stats(x).items() if k not in {"n", "n_all"}},
        }
        joint = inside if joint is None else (joint & inside)
    out["joint_mapped"] = {
        "frac_inside": float(joint.mean()) if joint is not None else 0.0,
        "n_inside": int(joint.sum()) if joint is not None else 0,
        "n": int(next(iter(values.values())).size),
    }
    return out


def pca_project(train: np.ndarray, test: np.ndarray, n_comp: int = 2):
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    mu = np.nanmean(train, axis=0)
    x = np.nan_to_num(train - mu, nan=0.0)
    xt = np.nan_to_num(test - mu, nan=0.0)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:n_comp]
    var = (s**2) / max(x.shape[0] - 1, 1)
    evr = var[:n_comp] / max(var.sum(), 1e-30)
    return x @ comp.T, xt @ comp.T, [float(v) for v in evr]


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    qs = np.linspace(0.0, 1.0, 256)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def save_heatmap(path: Path, grid: np.ndarray, title: str, cbar: str, *, log: bool = False):
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    data = np.array(grid, dtype=np.float64, copy=True)
    if log:
        pos = data[np.isfinite(data) & (data > 0)]
        floor = float(np.percentile(pos, 1)) if pos.size else 1e-6
        data = np.log10(np.clip(data, floor, None))
        cbar = f"log10({cbar})"
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(f"packed column (wrap width={WRAP_WIDTH})")
    ax.set_ylabel("packed row")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_mask_heatmap(path: Path, mask_grid: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.imshow(mask_grid.astype(float), aspect="auto", interpolation="nearest", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel(f"packed column (wrap width={WRAP_WIDTH})")
    ax.set_ylabel("packed row")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_hist_overlay(path: Path, series: dict[str, np.ndarray], title: str, xlabel: str):
    """Linear-x density overlay. EMIT retrievals are not log-uniform; do not log-bin."""
    cleaned: dict[str, np.ndarray] = {}
    for name, x in series.items():
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size:
            cleaned[name] = x
    if not cleaned:
        return
    lo = min(float(v.min()) for v in cleaned.values())
    hi = max(float(v.max()) for v in cleaned.values())
    if hi <= lo:
        hi = lo + 1.0
    if lo >= 0.0:
        lo = 0.0
    bins = np.linspace(lo, hi, 41)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, x in cleaned.items():
        ax.hist(x, bins=bins, density=True, histtype="step", label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(path: Path, stats: dict) -> None:
    emit = stats["emit"]
    sim = stats["sim70"]
    lines = [
        "# EMIT vs synthetic TOA domain",
        "",
        f"- EMIT: `{emit['path']}`  n={emit['n']:,}  bands={emit['n_bands']}",
        f"- Synthetic (fsnow 70–100): `{sim['path']}`  n={sim['n']:,}",
    ]
    if stats.get("unfiltered"):
        u = stats["unfiltered"]
        lines.append(f"- Unfiltered synthetic: `{u['path']}`  n={u['n']:,}")
    lines += [
        "",
        "## Headline",
        "",
        (
            f"Broadband mean reflectance is **{emit['mean_pixel_mean']:.3f}** on EMIT vs "
            f"**{sim['mean_pixel_mean']:.3f}** on the 70–100% snow synthetic set "
            f"(ratio EMIT/sim = {emit['mean_pixel_mean'] / sim['mean_pixel_mean']:.3f})."
        ),
        "",
        "### Why",
        "",
    ]
    for b in stats["diagnosis"]:
        lines.append(f"- {b}")
    lines += ["", "## EMIT grain isolation", ""]
    g = emit["grain"]
    lines += [
        f"- `grain_radius == 0`: {g['n_zero']:,} pixels ({100 * g['frac_zero']:.4f}%).",
        f"- `grain_radius ≈ 500` (|x-500|≤{GRAIN_DEFAULT_TOL}): {g['n_near_500']:,} "
        f"({100 * g['frac_near_500']:.2f}%). This is a sharp histogram spike.",
        f"- `grain_radius` in [30, 1500] excluding ≈500: {g['n_in_design_not_500']:,} "
        f"({100 * g['frac_in_design_not_500']:.2f}%).",
        "",
        "Zero-grain pixels are rare. The 500 µm mode is the dominant discrete mass "
        "and co-occurs with other stuck-looking state values (see stats.json).",
        "",
        "## State ranges vs synthetic box",
        "",
        "| quantity | EMIT min | EMIT max | EMIT mean | sim70 min | sim70 max | % EMIT in sim70 box |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cov = stats["coverage_sim70"]
    for name in MAPPED_QOI:
        e = emit["mapped_stats"][name]
        s = sim["mapped_stats"][name]
        c = cov[name]
        lines.append(
            f"| {name} | {e['min']:.4g} | {e['max']:.4g} | {e['mean']:.4g} | "
            f"{s['min']:.4g} | {s['max']:.4g} | {100 * c['frac_inside']:.1f}% |"
        )
    j = cov["joint_mapped"]
    lines += [
        "",
        f"Joint coverage (all mapped QoIs inside the 70–100 file ranges): "
        f"**{100 * j['frac_inside']:.2f}%** ({j['n_inside']:,} / {j['n']:,}).",
        "",
        "## Reflectance scale",
        "",
        f"- EMIT reflectance min/max: {emit['refl_min']:.4g} / {emit['refl_max']:.4g}; "
        f"neg={emit['n_neg']:,}, >1={emit['n_gt1']:,}.",
        f"- Sim70 reflectance min/max: {sim['refl_min']:.4g} / {sim['refl_max']:.4g}; "
        f"neg={sim['n_neg']:,}, >1={sim['n_gt1']:,}.",
        "",
        "EMIT values sit in (0, 1) like true reflectance. Sim70 can exceed 1 and dip negative "
        "(noisy TOA conversion). Scales match well enough that a 0–10000 integer scaling is not the issue.",
        "",
        "## Spatial heatmaps",
        "",
        (
            f"EMIT `sample` is packed `0..n-1` with no gaps, so original scene geometry is not recoverable. "
            f"Heatmaps wrap at width {WRAP_WIDTH} (typical EMIT cross-track). Treat them as index maps, not maps."
        ),
        "",
        "Figures are in this directory (`heatmap_*.png`, `hist_*.png`, `spectra_*.png`, `pca_overlap.png`).",
        "",
        "## Unit / parameterization mismatches",
        "",
        "- Sim70 stores **physical fractional covers** already (fsnow in [0.70, 1.00]); "
        "the unfiltered file stores **pre-softmax logits in [0, 10]**.",
        "- EMIT `z_snow/z_pv/z_npv/z_soil` are logits in [-5, 5]; comparison uses softmax.",
        "- EMIT has `sinA/cosA`, not `cos_i`. Synthetic geometry is fixed (VZA=6°, RAA=164°, elev=3 km).",
        "- EMIT `AOT660` vs synthetic `aot` (AOT550 in the forward model).",
        "- EMIT `H20STR` max is far below the synthetic CWV upper bound 5.2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_diagnosis(stats: dict) -> list[str]:
    emit = stats["emit"]
    sim = stats["sim70"]
    ratio = emit["mean_pixel_mean"] / max(sim["mean_pixel_mean"], 1e-12)
    g = emit["grain"]
    snow = emit["mapped_stats"]["fsnow"]
    out = [
        (
            f"Mean reflectance ratio EMIT/sim70 = {ratio:.3f}. "
            "The synthetic file is snow-bright by construction (fsnow 0.70–1.00)."
        ),
        (
            f"EMIT softmax snow fraction mean={snow['mean']:.3f} (p50={snow['p50']:.3f}), "
            f"vs sim70 mean={sim['mapped_stats']['fsnow']['mean']:.3f}. Mixed / lower snow cover "
            "on EMIT is the primary brightness gap."
        ),
        (
            f"Only {g['n_zero']:,} EMIT pixels have grain_radius == 0 "
            f"({100 * g['frac_zero']:.4f}%). They are not driving the scene-mean offset."
        ),
        (
            f"{g['n_near_500']:,} EMIT pixels ({100 * g['frac_near_500']:.1f}%) sit at grain≈500. "
            "That is a discrete spike, not the sim Sobol continuum; it is consistent with an "
            "ISOFIT init/unused mode but is not proven from ISOFIT source here."
        ),
    ]
    cwv = emit["mapped_stats"]["cwv"]
    out.append(
        f"EMIT H20STR is [{cwv['min']:.3g}, {cwv['max']:.3g}] vs synthetic CWV box [0.2, 5.2]. "
        "Almost the entire EMIT water-vapor range sits at the dry end of (or below) the sim box."
    )
    aot = emit["mapped_stats"]["aot"]
    out.append(
        f"EMIT AOT660 max={aot['max']:.3g} vs sim aot up to ~1.0. Aerosol optical depth "
        "domains only partially overlap, and the band (660 vs 550) differs."
    )
    j = stats["coverage_sim70"]["joint_mapped"]["frac_inside"]
    out.append(
        f"Only {100 * j:.2f}% of EMIT pixels fall inside the joint 70–100 synthetic hyper-rectangle "
        "on the mapped QoIs (after softmax on EMIT fractions)."
    )
    high = emit["strat_mean_pixel"].get("snow70_valid_grain")
    if high is not None and sim["mean_pixel_mean"]:
        out.append(
            f"Restricting EMIT to softmax snow≥0.70 and grain in [30,1500] excluding ≈500 "
            f"gives mean reflectance {high:.3f} vs sim70 {sim['mean_pixel_mean']:.3f} "
            f"(ratio {high / sim['mean_pixel_mean']:.3f}). Residual gap after this cut is "
            "geometry / RT / AOT-band mismatch, not just the snow-fraction filter."
        )
    return out


def run(emit_path: Path, sim_path: Path, unfiltered_path: Path | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    sim70 = load_sim_file(sim_path)
    unfiltered = load_sim_file(unfiltered_path) if unfiltered_path and unfiltered_path.is_file() else None
    wl = sim70["wl"]
    nir_idx = nearest_band(wl, NIR_TARGET_NM)

    with h5py.File(emit_path, "r") as f:
        state = np.asarray(f["state"][:], dtype=np.float64)
        sample = np.asarray(f["sample"][:], dtype=np.int64)
        emit_attrs = read_root_attrs(f)
        n_emit = int(f["reflectance"].shape[0])
        n_bands = int(f["reflectance"].shape[1])
        names_attr = emit_attrs.get("state_feature_names", ",".join(EMIT_STATE_FEATURE_NAMES))
        state_names = [s.strip() for s in names_attr.split(",")]

    grain = state[:, EMIT_STATE_INDEX["grain_radius"]]
    z = state[:, [EMIT_STATE_INDEX[k] for k in ("z_snow", "z_pv", "z_npv", "z_soil")]]
    emit_frac = softmax_rows(z)
    emit_mapped = {
        "grain_size": grain,
        "liquid_water": state[:, EMIT_STATE_INDEX["liquid_water"]],
        "dust": state[:, EMIT_STATE_INDEX["dust"]],
        "algae": state[:, EMIT_STATE_INDEX["algae"]],
        "aot": state[:, EMIT_STATE_INDEX["AOT660"]],
        "cwv": state[:, EMIT_STATE_INDEX["H20STR"]],
        "fsnow": emit_frac[:, 0],
        "fPV": emit_frac[:, 1],
        "fNPV": emit_frac[:, 2],
        "fsoil": emit_frac[:, 3],
    }

    grain0 = np.abs(grain) < 1e-12
    grain_near0 = grain < 1.0
    grain500 = np.abs(grain - GRAIN_DEFAULT) <= GRAIN_DEFAULT_TOL
    grain_design = _in_box(grain, 30.0, 1500.0)
    grain_ok = grain_design & (~grain500)
    snow70 = emit_mapped["fsnow"] >= 0.70
    snow70_valid = snow70 & grain_ok

    masks = {
        "all": np.ones(n_emit, dtype=bool),
        "grain0": grain0,
        "grain500": grain500,
        "grain_design_not_500": grain_ok,
        "snow70": snow70,
        "snow70_valid_grain": snow70_valid,
    }

    rng = np.random.default_rng(42)
    sub_n = min(SUBSAMPLE, n_emit)
    subsample_idx = np.sort(rng.choice(n_emit, size=sub_n, replace=False))

    emit_stream = stream_emit_reflectance(
        emit_path, masks=masks, subsample_idx=subsample_idx, nir_idx=nir_idx
    )

    emit_state_stats = {
        name: _finite_stats(state[:, i]) for i, name in enumerate(state_names)
    }
    emit_mapped_stats = {name: _finite_stats(v) for name, v in emit_mapped.items()}
    sim_mapped = {name: sim70["frac"][name] if name in sim70["frac"] else sim70["qoi"][name] for name in MAPPED_QOI}
    sim_mapped_stats = {name: _finite_stats(sim_mapped[name]) for name in MAPPED_QOI}

    coverage_sim70 = coverage_table(emit_mapped, {k: (float(np.min(sim_mapped[k])), float(np.max(sim_mapped[k]))) for k in MAPPED_QOI})
    coverage_design = coverage_table(emit_mapped, DESIGN_BOX)

    # Co-occurrence of grain≈500 with other zeros
    z_zero = np.all(np.abs(z) < 1e-12, axis=1)
    lw0 = np.abs(emit_mapped["liquid_water"]) < 1e-12
    grain500_and_z0 = int((grain500 & z_zero).sum())
    grain500_and_lw0 = int((grain500 & lw0).sum())

    strat_pixel_mean = {}
    for name, mask in masks.items():
        if mask.any():
            strat_pixel_mean[name] = float(np.nanmean(emit_stream["pixel_mean"][mask]))
        else:
            strat_pixel_mean[name] = None

    zero_idx = np.flatnonzero(grain0)
    zero_summary = {
        "n": int(zero_idx.size),
        "sample_idx": zero_idx[:200].astype(int).tolist(),
        "state_median": {
            name: float(np.median(state[grain0, i])) if grain0.any() else None
            for i, name in enumerate(state_names)
        },
        "frac_median": {
            "fsnow": float(np.median(emit_mapped["fsnow"][grain0])) if grain0.any() else None,
            "fPV": float(np.median(emit_mapped["fPV"][grain0])) if grain0.any() else None,
            "fNPV": float(np.median(emit_mapped["fNPV"][grain0])) if grain0.any() else None,
            "fsoil": float(np.median(emit_mapped["fsoil"][grain0])) if grain0.any() else None,
        },
        "mean_reflectance": strat_pixel_mean.get("grain0"),
    }

    # PCA + Wasserstein on subsamples
    sim_sub = rng.choice(sim70["n"], size=min(PCA_N, sim70["n"]), replace=False)
    emit_pca_idx = rng.choice(sub_n, size=min(PCA_N, sub_n), replace=False)
    sim_pc, emit_pc, evr = pca_project(sim70["refl"][sim_sub], emit_stream["sub_x"][emit_pca_idx])

    wass_mean = wasserstein_1d(emit_stream["pixel_mean"][subsample_idx], sim70["pixel_mean"])
    wass_bands = []
    step = max(1, n_bands // 20)
    for b in range(0, n_bands, step):
        wass_bands.append(
            {
                "band": int(b),
                "wl_nm": float(wl[b]),
                "wasserstein": wasserstein_1d(emit_stream["sub_x"][:, b], sim70["refl"][sim_sub, b]),
                "mean_gap": float(emit_stream["spec_mean"][b] - sim70["spec_mean"][b]),
            }
        )

    # Heatmaps
    mean_grid = pack_to_grid(emit_stream["pixel_mean"], WRAP_WIDTH)
    nir_grid = pack_to_grid(emit_stream["nir"], WRAP_WIDTH)
    save_heatmap(fig_dir / "heatmap_mean_reflectance.png", mean_grid, "EMIT packed-index mean reflectance (all 285 bands)", "mean reflectance")
    save_heatmap(fig_dir / "heatmap_nir900.png", nir_grid, f"EMIT packed-index reflectance at {wl[nir_idx]:.0f} nm", "reflectance")
    save_mask_heatmap(fig_dir / "heatmap_grain0.png", pack_to_grid(grain0.astype(float), WRAP_WIDTH), "EMIT grain_radius == 0")
    save_mask_heatmap(fig_dir / "heatmap_grain500.png", pack_to_grid(grain500.astype(float), WRAP_WIDTH), "EMIT grain_radius ≈ 500")
    save_mask_heatmap(fig_dir / "heatmap_snow70.png", pack_to_grid(snow70.astype(float), WRAP_WIDTH), "EMIT softmax snow fraction ≥ 0.70")

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    axes = axes.ravel()
    log_state = {"grain_radius", "liquid_water", "dust", "algae"}
    for i, name in enumerate(state_names):
        ax = axes[i]
        grid = pack_to_grid(state[:, i], WRAP_WIDTH)
        data = grid.copy()
        cbar = name
        if name in log_state:
            pos = data[np.isfinite(data) & (data > 0)]
            floor = float(np.percentile(pos, 1)) if pos.size else 1e-6
            data = np.log10(np.clip(data, floor, None))
            cbar = f"log10({name})"
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for j in range(len(state_names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"EMIT 15-D state (packed wrap width={WRAP_WIDTH})", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_state_grid.png", dpi=130)
    plt.close(fig)

    # Histograms
    plot_hist_overlay(
        fig_dir / "hist_mean_reflectance.png",
        {
            "EMIT": emit_stream["pixel_mean"][subsample_idx],
            "sim 70-100": sim70["pixel_mean"],
            **({"unfiltered sim": unfiltered["pixel_mean"]} if unfiltered else {}),
        },
        "Broadband mean reflectance",
        "mean reflectance (285 bands)",
    )
    plot_hist_overlay(
        fig_dir / "hist_grain.png",
        {
            "EMIT grain_radius": grain,
            "sim70 grain_size": sim70["qoi"]["grain_size"],
        },
        "Grain size / radius",
        "µm",
    )
    plot_hist_overlay(
        fig_dir / "hist_fsnow_frac.png",
        {
            "EMIT softmax fsnow": emit_mapped["fsnow"],
            "sim70 fsnow": sim_mapped["fsnow"],
        },
        "Snow fractional cover",
        "snow fraction",
    )
    for name in ("algae", "dust", "liquid_water", "aot", "cwv"):
        plot_hist_overlay(
            fig_dir / f"hist_{name}.png",
            {
                f"EMIT {name}": emit_mapped[name],
                f"sim70 {name}": sim_mapped[name],
            },
            name,
            name,
        )

    # Spectra
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, emit_stream["spec_mean"], label="EMIT mean", color="#1f4e79")
    ax.fill_between(
        wl,
        emit_stream["spec_mean"] - emit_stream["spec_std"],
        emit_stream["spec_mean"] + emit_stream["spec_std"],
        alpha=0.18,
        color="#1f4e79",
        label="EMIT ±1 std",
    )
    ax.plot(wl, sim70["spec_mean"], label="sim 70–100 mean", color="#b85c38")
    ax.fill_between(
        wl,
        sim70["spec_mean"] - sim70["spec_std"],
        sim70["spec_mean"] + sim70["spec_std"],
        alpha=0.18,
        color="#b85c38",
        label="sim ±1 std",
    )
    if unfiltered:
        ax.plot(wl, unfiltered["spec_mean"], label="unfiltered sim mean", color="#5c5c5c", linestyle="--")
    ax.set_title("Mean TOA reflectance spectrum")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_mean.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, np.nanpercentile(emit_stream["sub_x"], 10, axis=0), color="#1f4e79", linestyle=":", label="EMIT p10")
    ax.plot(wl, np.nanpercentile(emit_stream["sub_x"], 50, axis=0), color="#1f4e79", label="EMIT p50")
    ax.plot(wl, np.nanpercentile(emit_stream["sub_x"], 90, axis=0), color="#1f4e79", linestyle="--", label="EMIT p90")
    ax.plot(wl, sim70["spec_p10"], color="#b85c38", linestyle=":", label="sim70 p10")
    ax.plot(wl, sim70["spec_p50"], color="#b85c38", label="sim70 p50")
    ax.plot(wl, sim70["spec_p90"], color="#b85c38", linestyle="--", label="sim70 p90")
    ax.set_title("Reflectance percentile envelopes (EMIT subsample vs sim70)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_percentiles.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    labels = {
        "all": "EMIT all",
        "grain0": "EMIT grain=0",
        "grain500": "EMIT grain≈500",
        "grain_design_not_500": "EMIT grain in [30,1500] \\ ≈500",
        "snow70": "EMIT snow≥0.70",
        "snow70_valid_grain": "EMIT snow≥0.70 and valid grain",
    }
    for k, lab in labels.items():
        if emit_stream["strat_count"].get(k, 0) > 0:
            ax.plot(wl, emit_stream["strat_mean"][k], label=f"{lab} (n={emit_stream['strat_count'][k]:,})")
    ax.plot(wl, sim70["spec_mean"], color="black", linewidth=1.6, label="sim 70–100")
    ax.set_title("EMIT mean spectra stratified by grain / snow vs synthetic")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_stratified.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    hb = ax.hexbin(
        emit_mapped["fsnow"][subsample_idx],
        emit_stream["pixel_mean"][subsample_idx],
        gridsize=40,
        mincnt=3,
        cmap="viridis",
    )
    ax.scatter(
        sim_mapped["fsnow"][:: max(1, sim70["n"] // 4000)],
        sim70["pixel_mean"][:: max(1, sim70["n"] // 4000)],
        s=6,
        c="#b85c38",
        alpha=0.25,
        label="sim 70–100 (thinned)",
    )
    ax.set_xlabel("snow fraction (EMIT: softmax z_*)")
    ax.set_ylabel("broadband mean reflectance")
    ax.set_title("Snow fraction vs brightness")
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, label="EMIT count")
    fig.tight_layout()
    fig.savefig(fig_dir / "hexbin_snow_vs_refl.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(sim_pc[:, 0], sim_pc[:, 1], s=4, alpha=0.25, label="sim 70–100", c="#b85c38")
    ax.scatter(emit_pc[:, 0], emit_pc[:, 1], s=4, alpha=0.25, label="EMIT", c="#1f4e79")
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% sim variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% sim variance)")
    ax.set_title("Reflectance PCA (fit on sim, project EMIT)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_overlap.png", dpi=140)
    plt.close(fig)

    ds_mean = block_downsample(mean_grid, HEAT_DS_H, HEAT_DS_W)
    ds_g500 = block_downsample(pack_to_grid(grain500.astype(float), WRAP_WIDTH), HEAT_DS_H, HEAT_DS_W)
    ds_snow = block_downsample(pack_to_grid(emit_mapped["fsnow"], WRAP_WIDTH), HEAT_DS_H, HEAT_DS_W)

    spec_idx = np.arange(0, n_bands, SPEC_CANVAS_STEP)
    stats = {
        "emit": {
            "path": str(emit_path.resolve()),
            "n": n_emit,
            "n_bands": n_bands,
            "attrs": emit_attrs,
            "state_names": state_names,
            "sample_min": int(sample.min()),
            "sample_max": int(sample.max()),
            "sample_packed_consecutive": bool(sample.min() == 0 and sample.max() == n_emit - 1),
            "wrap_width": WRAP_WIDTH,
            "wrap_rows": infer_wrap_grid(n_emit)[0],
            "wrap_remainder": infer_wrap_grid(n_emit)[2],
            "mean_pixel_mean": emit_stream["mean_pixel_mean"],
            "refl_min": float(np.nanmin(emit_stream["sub_x"])),
            "refl_max": float(np.nanmax(emit_stream["sub_x"])),
            "n_neg": emit_stream["n_neg"],
            "n_gt1": emit_stream["n_gt1"],
            "n_nonfinite": emit_stream["n_nonfinite"],
            "nir_nm": float(wl[nir_idx]),
            "state_stats": emit_state_stats,
            "mapped_stats": emit_mapped_stats,
            "grain": {
                "n_zero": int(grain0.sum()),
                "frac_zero": float(grain0.mean()),
                "n_near_zero_lt1": int(grain_near0.sum()),
                "n_near_500": int(grain500.sum()),
                "frac_near_500": float(grain500.mean()),
                "n_in_design": int(grain_design.sum()),
                "n_in_design_not_500": int(grain_ok.sum()),
                "frac_in_design_not_500": float(grain_ok.mean()),
                "n_grain500_and_z_all_zero": grain500_and_z0,
                "n_grain500_and_liquid_water_zero": grain500_and_lw0,
                "frac_grain500_with_z_all_zero": float(grain500_and_z0 / max(int(grain500.sum()), 1)),
            },
            "zero_grain": zero_summary,
            "strat_count": {k: int(v) for k, v in emit_stream["strat_count"].items()},
            "strat_mean_pixel": strat_pixel_mean,
            "spec_mean": emit_stream["spec_mean"].tolist(),
            "spec_std": emit_stream["spec_std"].tolist(),
        },
        "sim70": {
            "path": sim70["path"],
            "n": sim70["n"],
            "attrs": sim70["attrs"],
            "frac_kind": sim70["frac_kind"],
            "mean_pixel_mean": float(np.nanmean(sim70["pixel_mean"])),
            "refl_min": sim70["refl_min"],
            "refl_max": sim70["refl_max"],
            "n_neg": sim70["n_neg"],
            "n_gt1": sim70["n_gt1"],
            "n_nonfinite": sim70["n_nonfinite"],
            "mapped_stats": sim_mapped_stats,
            "spec_mean": sim70["spec_mean"].tolist(),
            "spec_std": sim70["spec_std"].tolist(),
        },
        "unfiltered": None
        if unfiltered is None
        else {
            "path": unfiltered["path"],
            "n": unfiltered["n"],
            "frac_kind": unfiltered["frac_kind"],
            "mean_pixel_mean": float(np.nanmean(unfiltered["pixel_mean"])),
            "refl_min": unfiltered["refl_min"],
            "refl_max": unfiltered["refl_max"],
            "n_neg": unfiltered["n_neg"],
            "n_gt1": unfiltered["n_gt1"],
            "mapped_stats": {
                name: _finite_stats(
                    unfiltered["frac"][name] if name in unfiltered["frac"] else unfiltered["qoi"][name]
                )
                for name in MAPPED_QOI
            },
            "spec_mean": unfiltered["spec_mean"].tolist(),
        },
        "coverage_sim70": coverage_sim70,
        "coverage_design": coverage_design,
        "pca": {"explained_variance_ratio": evr},
        "wasserstein_mean_reflectance": wass_mean,
        "wasserstein_bands": wass_bands,
        "wl": wl.tolist(),
        "canvas": {
            "wl": [float(wl[i]) for i in spec_idx],
            "emit_spec_mean": [float(emit_stream["spec_mean"][i]) for i in spec_idx],
            "sim_spec_mean": [float(sim70["spec_mean"][i]) for i in spec_idx],
            "unf_spec_mean": None
            if unfiltered is None
            else [float(unfiltered["spec_mean"][i]) for i in spec_idx],
            "heatmap_mean": np.round(np.nan_to_num(ds_mean, nan=0.0), 4).tolist(),
            "heatmap_grain500": np.round(np.nan_to_num(ds_g500, nan=0.0), 4).tolist(),
            "heatmap_snow": np.round(np.nan_to_num(ds_snow, nan=0.0), 4).tolist(),
            "heatmap_vmin": float(np.nanpercentile(ds_mean, 2)),
            "heatmap_vmax": float(np.nanpercentile(ds_mean, 98)),
        },
    }
    stats["mean_reflectance_ratio_emit_over_sim70"] = float(
        stats["emit"]["mean_pixel_mean"] / max(stats["sim70"]["mean_pixel_mean"], 1e-12)
    )
    stats["diagnosis"] = build_diagnosis(stats)

    json_path = out_dir / "stats.json"
    json_path.write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", stats)
    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Figures in {fig_dir}")
    print("EMIT mean", stats["emit"]["mean_pixel_mean"], "sim70", stats["sim70"]["mean_pixel_mean"])
    print("grain0", stats["emit"]["grain"]["n_zero"], "grain500", stats["emit"]["grain"]["n_near_500"])
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit", type=Path, default=DEFAULT_EMIT)
    p.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    p.add_argument("--unfiltered", type=Path, default=DEFAULT_UNFILTERED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-unfiltered", action="store_true")
    args = p.parse_args()
    unf = None if args.no_unfiltered else args.unfiltered
    run(args.emit, args.sim, unf, args.out)


if __name__ == "__main__":
    main()
