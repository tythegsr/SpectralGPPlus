"""Analyze a processed EMIT NetCDF (radiance + reflectance + elevation + state).

Writes stats.json, report.md, and figures under --out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
DEFAULT_SRC = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "emit_test_data_processed_20262008.nc"
)
DEFAULT_WL_SRC = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261808.nc"
)
DEFAULT_OUT = _ROOT / "experiments_toa" / "reports" / "emit_processed_20262008"

EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
IDX_Z = tuple(EMIT_STATE_INDEX[n] for n in ("z_snow", "z_pv", "z_npv", "z_soil"))
WRAP_WIDTH = 1280
SUBSAMPLE = 50000
PCA_N = 20000
NIR_TARGET_NM = 900.0
GRAIN_DEFAULT = 500.0
GRAIN_DEFAULT_TOL = 0.1
CHUNK = 8192


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
        }
    qs = np.percentile(x, [1, 5, 50, 95, 99])
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
    }


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)


def pack_to_grid(values: np.ndarray, width: int = WRAP_WIDTH, fill=np.nan) -> np.ndarray:
    n = int(values.size)
    rows = int(math.ceil(n / width))
    grid = np.full((rows, width), fill, dtype=np.float64)
    grid.ravel()[:n] = values.astype(np.float64, copy=False)
    return grid


def nearest_band(wl: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wl - target_nm)))


def save_heatmap(path: Path, grid: np.ndarray, title: str, cbar: str):
    data = np.asarray(grid, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vmin, vmax = np.percentile(finite, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(f"packed column (wrap width={WRAP_WIDTH})")
    ax.set_ylabel("packed row")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_hist(path: Path, series: dict[str, np.ndarray], title: str, xlabel: str, *, logx: bool = False):
    cleaned: dict[str, np.ndarray] = {}
    for name, x in series.items():
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if logx:
            x = x[x > 0]
        if x.size:
            cleaned[name] = x
    if not cleaned:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if logx:
        lo = min(float(v.min()) for v in cleaned.values())
        hi = max(float(v.max()) for v in cleaned.values())
        bins = np.logspace(np.log10(lo), np.log10(hi), 41)
        ax.set_xscale("log")
    else:
        lo = min(float(v.min()) for v in cleaned.values())
        hi = max(float(v.max()) for v in cleaned.values())
        if hi <= lo:
            hi = lo + 1.0
        if lo >= 0.0:
            lo = min(0.0, lo)
        bins = np.linspace(lo, hi, 41)
    for name, x in cleaned.items():
        ax.hist(x, bins=bins, density=True, histtype="step", label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if len(cleaned) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stream_band_moments(ds, n: int, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """One-pass mean/std over samples for each band."""
    sum1 = np.zeros(n_bands, dtype=np.float64)
    sum2 = np.zeros(n_bands, dtype=np.float64)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        chunk = np.asarray(ds[start:stop], dtype=np.float64)
        sum1 += chunk.sum(axis=0)
        sum2 += np.square(chunk, dtype=np.float64).sum(axis=0)
    mean = sum1 / n
    var = np.maximum(sum2 / n - np.square(mean), 0.0)
    return mean, np.sqrt(var)


def stream_pixel_means(ds, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        chunk = np.asarray(ds[start:stop], dtype=np.float32)
        out[start:stop] = chunk.mean(axis=1)
    return out


def mapped_qois(state: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    frac = softmax_rows(state[:, list(IDX_Z)])
    return {
        "sinA": state[:, EMIT_STATE_INDEX["sinA"]],
        "cosA": state[:, EMIT_STATE_INDEX["cosA"]],
        "grain_size": state[:, EMIT_STATE_INDEX["grain_radius"]],
        "liquid_water": state[:, EMIT_STATE_INDEX["liquid_water"]],
        "dust": state[:, EMIT_STATE_INDEX["dust"]],
        "algae": state[:, EMIT_STATE_INDEX["algae"]],
        "fsnow": frac[:, 0],
        "fPV": frac[:, 1],
        "fNPV": frac[:, 2],
        "fsoil": frac[:, 3],
        "aot": state[:, EMIT_STATE_INDEX["AOT660"]],
        "cwv": state[:, EMIT_STATE_INDEX["H20STR"]],
        "elevation": None,  # filled by caller
    }


def write_report(path: Path, stats: dict) -> None:
    s = stats
    lines = [
        "# Processed EMIT scene analysis",
        "",
        f"- Source: `{s['path']}`",
        f"- n={s['n']:,}  bands={s['n_bands']}  dtype radiance/reflectance={s['dtypes']['radiance']}/{s['dtypes']['reflectance']}",
        f"- Wavelengths from `{s['wl_src']}`",
        "",
        "## Headline",
        "",
        (
            f"Broadband mean **reflectance** = {s['reflectance']['pixel_mean']['mean']:.4f} "
            f"(p50={s['reflectance']['pixel_mean']['p50']:.4f}); "
            f"mean **radiance** = {s['radiance']['pixel_mean']['mean']:.4f}; "
            f"mean **elevation** = {s['elevation']['mean']:.1f} m "
            f"[{s['elevation']['min']:.1f}, {s['elevation']['max']:.1f}]."
        ),
        "",
        f"- Softmax snow fraction mean={s['qoi']['fsnow']['mean']:.3f} "
        f"(min={s['qoi']['fsnow']['min']:.3f}, max={s['qoi']['fsnow']['max']:.3f}).",
        f"- Reflectance values >1: {s['reflectance']['n_gt1']:,} "
        f"({100 * s['reflectance']['frac_gt1']:.2f}%); "
        f"<0: {s['reflectance']['n_lt0']:,}.",
        f"- Radiance values <0: {s['radiance']['n_lt0']:,}.",
        f"- Correlation(elevation, mean reflectance) = {s['corr']['elev_vs_mean_ref']:.3f}; "
        f"elev vs fsnow = {s['corr']['elev_vs_fsnow']:.3f}; "
        f"mean rad vs mean ref = {s['corr']['mean_rad_vs_mean_ref']:.3f}.",
        "",
        "## Grain isolation",
        "",
        f"- grain_radius == 0: {s['grain']['n_zero']:,}",
        f"- grain_radius ≈ 500: {s['grain']['n_near_500']:,}",
        "",
        "## QoI summary",
        "",
        "| quantity | min | max | mean | p50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, st in s["qoi"].items():
        lines.append(
            f"| {name} | {st['min']:.4g} | {st['max']:.4g} | {st['mean']:.4g} | {st['p50']:.4g} |"
        )
    lines += [
        "",
        "## Spectral notes",
        "",
        f"- NIR (~{s['nir_nm']:.0f} nm) mean reflectance = {s['reflectance']['nir_mean']:.4f}",
        f"- NIR (~{s['nir_nm']:.0f} nm) mean radiance = {s['radiance']['nir_mean']:.4f}",
        f"- PCA on reflectance subsample (n={s['pca']['n']:,}): "
        f"PC1–3 explain {100 * s['pca']['var_pc123']:.1f}% variance; "
        f"components to 99% = {s['pca']['n_comp_99']}.",
        "",
        "Figures are in `figures/` (`hist_*.png`, `spectra_*.png`, `heatmap_*.png`, `pca_*.png`, `scatter_*.png`).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(src: Path, wl_src: Path, out_dir: Path) -> dict:
    src = Path(src)
    wl_src = Path(wl_src)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(wl_src, "r") as f_wl:
        wl = np.asarray(f_wl["wl"][:], dtype=np.float64)

    with h5py.File(src, "r") as f:
        for key in ("radiance", "reflectance", "elevation", "state"):
            if key not in f:
                raise KeyError(f"Missing '{key}' in {src}")
        n = int(f["reflectance"].shape[0])
        n_bands = int(f["reflectance"].shape[1])
        if wl.size != n_bands:
            raise ValueError(f"wl length {wl.size} != n_bands {n_bands}")
        dtypes = {
            "radiance": str(f["radiance"].dtype),
            "reflectance": str(f["reflectance"].dtype),
            "elevation": str(f["elevation"].dtype),
            "state": str(f["state"].dtype),
        }

        print(f"Loading state / elevation ({n:,} rows)...")
        state = np.asarray(f["state"][:], dtype=np.float32)
        elev = np.asarray(f["elevation"][:], dtype=np.float32).reshape(n)
        if state.shape[1] != len(EMIT_STATE_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(EMIT_STATE_FEATURE_NAMES)} state features, got {state.shape[1]}"
            )

        print("Streaming radiance / reflectance band moments and pixel means...")
        rad_mean, rad_std = stream_band_moments(f["radiance"], n, n_bands)
        ref_mean, ref_std = stream_band_moments(f["reflectance"], n, n_bands)
        pixel_mean_rad = stream_pixel_means(f["radiance"], n)
        pixel_mean_ref = stream_pixel_means(f["reflectance"], n)

        nir_idx = nearest_band(wl, NIR_TARGET_NM)
        nir_rad = np.empty(n, dtype=np.float32)
        nir_ref = np.empty(n, dtype=np.float32)
        n_ref_gt1 = 0
        n_ref_lt0 = 0
        n_rad_lt0 = 0
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            rad = np.asarray(f["radiance"][start:stop], dtype=np.float32)
            ref = np.asarray(f["reflectance"][start:stop], dtype=np.float32)
            nir_rad[start:stop] = rad[:, nir_idx]
            nir_ref[start:stop] = ref[:, nir_idx]
            n_ref_gt1 += int((ref > 1.0).sum())
            n_ref_lt0 += int((ref < 0.0).sum())
            n_rad_lt0 += int((rad < 0.0).sum())

        rng = np.random.default_rng(0)
        sub_n = min(SUBSAMPLE, n)
        sub_idx = np.sort(rng.choice(n, size=sub_n, replace=False))
        rad_sub = np.asarray(f["radiance"][sub_idx], dtype=np.float32)
        ref_sub = np.asarray(f["reflectance"][sub_idx], dtype=np.float32)

        pca_n = min(PCA_N, n)
        pca_idx = np.sort(rng.choice(n, size=pca_n, replace=False))
        ref_pca = np.asarray(f["reflectance"][pca_idx], dtype=np.float64)

    qoi = mapped_qois(state)
    qoi["elevation"] = elev.astype(np.float64, copy=False)
    grain = qoi["grain_size"]
    n_grain0 = int((np.abs(grain) < 1e-12).sum())
    n_grain500 = int((np.abs(grain - GRAIN_DEFAULT) <= GRAIN_DEFAULT_TOL).sum())

    # Correlations (subsample for speed/robustness)
    corr_idx = sub_idx
    elev_s = elev[corr_idx].astype(np.float64)
    fsnow_s = qoi["fsnow"][corr_idx]
    mean_ref_s = pixel_mean_ref[corr_idx].astype(np.float64)
    mean_rad_s = pixel_mean_rad[corr_idx].astype(np.float64)

    def _corr(a, b) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    corr = {
        "elev_vs_mean_ref": _corr(elev_s, mean_ref_s),
        "elev_vs_fsnow": _corr(elev_s, fsnow_s),
        "mean_rad_vs_mean_ref": _corr(mean_rad_s, mean_ref_s),
        "elev_vs_mean_rad": _corr(elev_s, mean_rad_s),
        "fsnow_vs_mean_ref": _corr(fsnow_s, mean_ref_s),
    }

    # PCA on reflectance
    print("PCA on reflectance subsample...")
    x = ref_pca - ref_pca.mean(axis=0, keepdims=True)
    # economy SVD
    _, svals, _ = np.linalg.svd(x, full_matrices=False)
    var = (svals ** 2) / max(pca_n - 1, 1)
    var_ratio = var / max(var.sum(), 1e-30)
    cum = np.cumsum(var_ratio)
    n_comp_99 = int(np.searchsorted(cum, 0.99) + 1)

    # Figures
    print("Writing figures...")
    plot_hist(
        fig_dir / "hist_mean_reflectance.png",
        {"mean reflectance": pixel_mean_ref[sub_idx]},
        "Broadband mean reflectance",
        "mean reflectance",
    )
    plot_hist(
        fig_dir / "hist_mean_radiance.png",
        {"mean radiance": pixel_mean_rad[sub_idx]},
        "Broadband mean radiance",
        "mean radiance",
    )
    plot_hist(
        fig_dir / "hist_elevation.png",
        {"elevation": elev[sub_idx]},
        "Surface elevation",
        "elevation (m)",
    )
    plot_hist(
        fig_dir / "hist_fsnow.png",
        {"fsnow": qoi["fsnow"]},
        "Softmax snow fraction",
        "fsnow",
    )
    for name in ("grain_size", "liquid_water", "dust", "algae", "aot", "cwv"):
        plot_hist(
            fig_dir / f"hist_{name}.png",
            {name: qoi[name]},
            name,
            name,
            logx=name in ("dust", "algae"),
        )

    # Spectra
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, ref_mean, color="#1f4e79", label="mean")
    ax.fill_between(wl, ref_mean - ref_std, ref_mean + ref_std, alpha=0.18, color="#1f4e79", label="±1 std")
    ax.set_title("Mean TOA reflectance spectrum")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_reflectance_mean.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, rad_mean, color="#0b6e4f", label="mean")
    ax.fill_between(wl, rad_mean - rad_std, rad_mean + rad_std, alpha=0.18, color="#0b6e4f", label="±1 std")
    ax.set_title("Mean TOA radiance spectrum")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("radiance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_radiance_mean.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, np.nanpercentile(ref_sub, 10, axis=0), linestyle=":", label="p10")
    ax.plot(wl, np.nanpercentile(ref_sub, 50, axis=0), label="p50")
    ax.plot(wl, np.nanpercentile(ref_sub, 90, axis=0), linestyle="--", label="p90")
    ax.set_title("Reflectance percentiles (subsample)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_reflectance_percentiles.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(wl, np.nanpercentile(rad_sub, 10, axis=0), linestyle=":", label="p10")
    ax.plot(wl, np.nanpercentile(rad_sub, 50, axis=0), label="p50")
    ax.plot(wl, np.nanpercentile(rad_sub, 90, axis=0), linestyle="--", label="p90")
    ax.set_title("Radiance percentiles (subsample)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("radiance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectra_radiance_percentiles.png", dpi=140)
    plt.close(fig)

    # Heatmaps
    save_heatmap(
        fig_dir / "heatmap_mean_reflectance.png",
        pack_to_grid(pixel_mean_ref),
        "Packed-index mean reflectance",
        "mean reflectance",
    )
    save_heatmap(
        fig_dir / "heatmap_mean_radiance.png",
        pack_to_grid(pixel_mean_rad),
        "Packed-index mean radiance",
        "mean radiance",
    )
    save_heatmap(
        fig_dir / "heatmap_elevation.png",
        pack_to_grid(elev),
        "Packed-index surface elevation",
        "elevation (m)",
    )
    save_heatmap(
        fig_dir / "heatmap_nir_reflectance.png",
        pack_to_grid(nir_ref),
        f"Packed-index reflectance at {wl[nir_idx]:.0f} nm",
        "reflectance",
    )
    save_heatmap(
        fig_dir / "heatmap_fsnow.png",
        pack_to_grid(qoi["fsnow"]),
        "Packed-index softmax snow fraction",
        "fsnow",
    )

    # State grid
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    axes = axes.ravel()
    log_state = {"grain_radius", "liquid_water", "dust", "algae"}
    for i, name in enumerate(EMIT_STATE_FEATURE_NAMES):
        ax = axes[i]
        grid = pack_to_grid(state[:, i])
        data = grid.copy()
        if name in log_state:
            pos = data[np.isfinite(data) & (data > 0)]
            floor = float(np.percentile(pos, 1)) if pos.size else 1e-6
            data = np.log10(np.clip(data, floor, None))
            cbar = f"log10({name})"
        else:
            cbar = name
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[-1].set_visible(False)
    fig.suptitle(f"15-D state + elevation elsewhere (wrap width={WRAP_WIDTH})", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_state_grid.png", dpi=130)
    plt.close(fig)

    # Scatter: elevation relationships
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].scatter(elev_s, mean_ref_s, s=2, alpha=0.15, c="#1f4e79")
    axes[0].set_xlabel("elevation (m)")
    axes[0].set_ylabel("mean reflectance")
    axes[0].set_title(f"r={corr['elev_vs_mean_ref']:.3f}")
    axes[1].scatter(elev_s, fsnow_s, s=2, alpha=0.15, c="#0b6e4f")
    axes[1].set_xlabel("elevation (m)")
    axes[1].set_ylabel("fsnow")
    axes[1].set_title(f"r={corr['elev_vs_fsnow']:.3f}")
    axes[2].scatter(mean_rad_s, mean_ref_s, s=2, alpha=0.15, c="#b85c38")
    axes[2].set_xlabel("mean radiance")
    axes[2].set_ylabel("mean reflectance")
    axes[2].set_title(f"r={corr['mean_rad_vs_mean_ref']:.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_elevation_rad_ref.png", dpi=140)
    plt.close(fig)

    # PCA scree
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    k = min(40, var_ratio.size)
    ax.bar(np.arange(1, k + 1), var_ratio[:k], color="#1f4e79")
    ax.set_xlabel("principal component")
    ax.set_ylabel("fraction of variance")
    ax.set_title(f"Reflectance PCA scree (n={pca_n:,})")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_reflectance_scree.png", dpi=140)
    plt.close(fig)

    qoi_stats = {k: _finite_stats(v) for k, v in qoi.items() if v is not None}
    stats = {
        "path": str(src.resolve()),
        "wl_src": str(wl_src.resolve()),
        "n": n,
        "n_bands": n_bands,
        "dtypes": dtypes,
        "nir_nm": float(wl[nir_idx]),
        "nir_band_index": nir_idx,
        "elevation": _finite_stats(elev),
        "radiance": {
            "pixel_mean": _finite_stats(pixel_mean_rad),
            "band_mean": rad_mean.tolist(),
            "band_std": rad_std.tolist(),
            "nir_mean": float(np.mean(nir_rad)),
            "n_lt0": n_rad_lt0,
            "frac_lt0": float(n_rad_lt0 / (n * n_bands)),
            "subsample_percentiles": {
                "p10": np.nanpercentile(rad_sub, 10, axis=0).tolist(),
                "p50": np.nanpercentile(rad_sub, 50, axis=0).tolist(),
                "p90": np.nanpercentile(rad_sub, 90, axis=0).tolist(),
            },
        },
        "reflectance": {
            "pixel_mean": _finite_stats(pixel_mean_ref),
            "band_mean": ref_mean.tolist(),
            "band_std": ref_std.tolist(),
            "nir_mean": float(np.mean(nir_ref)),
            "n_gt1": n_ref_gt1,
            "frac_gt1": float(n_ref_gt1 / (n * n_bands)),
            "n_lt0": n_ref_lt0,
            "frac_lt0": float(n_ref_lt0 / (n * n_bands)),
            "subsample_percentiles": {
                "p10": np.nanpercentile(ref_sub, 10, axis=0).tolist(),
                "p50": np.nanpercentile(ref_sub, 50, axis=0).tolist(),
                "p90": np.nanpercentile(ref_sub, 90, axis=0).tolist(),
            },
        },
        "qoi": qoi_stats,
        "grain": {
            "n_zero": n_grain0,
            "n_near_500": n_grain500,
            "tol": GRAIN_DEFAULT_TOL,
        },
        "corr": corr,
        "pca": {
            "n": pca_n,
            "var_ratio_top10": var_ratio[:10].tolist(),
            "var_pc123": float(var_ratio[:3].sum()),
            "n_comp_99": n_comp_99,
        },
        "wl": wl.tolist(),
    }

    json_path = out_dir / "stats.json"
    json_path.write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", stats)
    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Figures in {fig_dir}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--wl-src", type=Path, default=DEFAULT_WL_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    run(args.src, args.wl_src, args.out)


if __name__ == "__main__":
    main()
