"""Histograms of S2 TOA radiance and QoIs (raw + log transforms)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_constants import (
    S2_DEFAULT_DATA_PATH,
    S2_LOG_SCALE_TASK_NAMES,
    S2_TASK_NAMES,
)
from experiments_toa.s2_data import read_elevation_array

QOI_NAMES = list(S2_TASK_NAMES)
BANDS_PER_PAGE = 24  # 4 x 6
NCOLS = 4


def _qoi_names_in_file(f) -> list[str]:
    names = [n for n in QOI_NAMES if n in f]
    if not names:
        raise KeyError(f"No S2 QoI datasets found among {QOI_NAMES}")
    return names


def _load(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        if "wl" not in f:
            raise KeyError(f"Missing wavelength variable 'wl' in {path}")
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        if "toa_reflectance" in f:
            spec = np.asarray(f["toa_reflectance"][:], dtype=np.float64)
            spec_name = "toa_reflectance"
        elif "toa_radiance" in f:
            spec = np.asarray(f["toa_radiance"][:], dtype=np.float64)
            spec_name = "toa_radiance"
        else:
            raise KeyError(f"Missing toa_radiance/toa_reflectance in {path}")
        names = _qoi_names_in_file(f)
        Y = np.column_stack([np.asarray(f[n][:], dtype=np.float64) for n in names])
        elev = read_elevation_array(f)
    return wl, spec, Y, names, spec_name, elev


def _plot_elevation_histogram(elev: np.ndarray, out: Path) -> None:
    col = _finite(elev)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(col, bins=60, color="seagreen", edgecolor="white", linewidth=0.3)
    ax.set_title("Elevation input (m)")
    ax.set_xlabel("elevation (m)")
    ax.set_ylabel("count")
    st = _stats(col)
    ax.text(
        0.98,
        0.95,
        f"n={st['n']}\n[{st['min']:.3g}, {st['max']:.3g}]\n"
        f"μ={st['mean']:.3g}\nσ={st['std']:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _stats(x: np.ndarray) -> dict:
    x = _finite(x)
    if x.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "p01": None, "p99": None}
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p01": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def radiance_log1p(rad: np.ndarray) -> np.ndarray:
    """ln(1 + x); clip tiny negatives from numerical noise before transform."""
    return np.log1p(np.clip(rad, 0.0, None))


def _plot_qoi_histograms(
    Y: np.ndarray,
    out: Path,
    *,
    log_space: bool,
    names: Sequence[str] | None = None,
) -> None:
    """Grid of QoI histograms: raw physical, or log10 for log-scale QoIs only."""
    names = list(names) if names is not None else QOI_NAMES
    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    title = (
        "QoI histograms (log10) — expect ~flat for log-uniform design dims"
        if log_space
        else "QoI histograms (raw / physical)"
    )
    for j, name in enumerate(names):
        ax = axes[j]
        col = Y[:, j]
        if log_space:
            if name not in S2_LOG_SCALE_TASK_NAMES:
                ax.set_visible(False)
                continue
            col = np.log10(np.clip(_finite(col), 1e-30, None))
            label = f"log10({name})"
            color = "coral"
        else:
            col = _finite(col)
            label = name
            color = "steelblue"
        ax.hist(col, bins=60, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("count")
        st = _stats(col)
        ax.text(
            0.98,
            0.95,
            f"n={st['n']}\n[{st['min']:.3g}, {st['max']:.3g}]\n"
            f"μ={st['mean']:.3g}\nσ={st['std']:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        )
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_all_band_histograms_pdf(
    X: np.ndarray,
    wl: np.ndarray,
    *,
    out: Path,
    title: str,
    color: str = "darkseagreen",
) -> None:
    """One histogram panel per band; multi-page PDF."""
    n_bands = X.shape[1]
    nrows = int(np.ceil(BANDS_PER_PAGE / NCOLS))
    with PdfPages(out) as pdf:
        for start in range(0, n_bands, BANDS_PER_PAGE):
            end = min(start + BANDS_PER_PAGE, n_bands)
            fig, axes = plt.subplots(nrows, NCOLS, figsize=(14, 2.6 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for k, j in enumerate(range(start, end)):
                ax = axes[k]
                col = _finite(X[:, j])
                ax.hist(col, bins=50, color=color, edgecolor="white", linewidth=0.2)
                ax.set_title(f"band {j} ({wl[j]:.0f} nm)", fontsize=8)
                ax.tick_params(labelsize=7)
            for k in range(end - start, len(axes)):
                axes[k].set_visible(False)
            fig.suptitle(f"{title}  (bands {start}–{end - 1})", fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig, dpi=120)
            plt.close(fig)


def _plot_wavelength_value_density(
    X: np.ndarray,
    wl: np.ndarray,
    *,
    out: Path,
    title: str,
    ylabel: str,
    n_value_bins: int = 80,
    max_samples: int = 20000,
) -> None:
    """2D density: wavelength (columns) vs value (rows), log1p(count) color."""
    n, d = X.shape
    if n > max_samples:
        rng = np.random.default_rng(0)
        X = X[rng.choice(n, size=max_samples, replace=False)]

    y_lo = float(np.nanpercentile(X, 0.5))
    y_hi = float(np.nanpercentile(X, 99.5))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
        y_lo, y_hi = float(np.nanmin(X)), float(np.nanmax(X))
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
    y_edges = np.linspace(y_lo, y_hi, n_value_bins + 1)
    counts = np.zeros((n_value_bins, d), dtype=np.float64)
    for j in range(d):
        c, _ = np.histogram(X[:, j], bins=y_edges)
        counts[:, j] = c

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        np.log1p(counts),
        aspect="auto",
        origin="lower",
        extent=(float(wl[0]), float(wl[-1]), y_lo, y_hi),
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log1p(count)")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_histograms(data_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {data_path}")
    wl, spec, Y, names, spec_name, elev = _load(data_path)
    is_radiance = spec_name == "toa_radiance"
    spec_label = "radiance" if is_radiance else "reflectance"
    print(
        f"  {spec_name}={spec.shape} Y={Y.shape} wl={wl.min():.1f}-{wl.max():.1f} nm"
    )
    print(f"  qoi={names}")
    if elev is not None:
        print(f"  elevation input present: [{elev.min():.1f}, {elev.max():.1f}] m")

    print("QoI raw histograms...")
    _plot_qoi_histograms(
        Y, out_dir / "qoi_histograms_raw.png", log_space=False, names=names
    )
    print("QoI log10 histograms (algae/dust/grain_size/liquid_water)...")
    _plot_qoi_histograms(
        Y, out_dir / "qoi_histograms_log10.png", log_space=True, names=names
    )
    if elev is not None:
        print("Elevation input histogram...")
        _plot_elevation_histogram(elev, out_dir / "elevation_histogram.png")

    spec_log = radiance_log1p(spec)
    n_bands = spec.shape[1]
    prefix = spec_name

    print(f"Full-band {spec_label} histograms (raw) -> PDF...")
    _plot_all_band_histograms_pdf(
        spec,
        wl,
        out=out_dir / f"{prefix}_histograms_raw.pdf",
        title=f"TOA {spec_label} — raw",
        color="darkseagreen",
    )
    print(f"Full-band {spec_label} histograms (log1p) -> PDF...")
    _plot_all_band_histograms_pdf(
        spec_log,
        wl,
        out=out_dir / f"{prefix}_histograms_log1p.pdf",
        title=f"TOA {spec_label} — log1p",
        color="steelblue",
    )

    print("Wavelength-value density (raw)...")
    _plot_wavelength_value_density(
        spec,
        wl,
        out=out_dir / f"{prefix}_density_raw.png",
        title=f"TOA {spec_label} density (raw)",
        ylabel=spec_label.capitalize(),
    )
    print("Wavelength-value density (log1p)...")
    _plot_wavelength_value_density(
        spec_log,
        wl,
        out=out_dir / f"{prefix}_density_log1p.png",
        title=f"TOA {spec_label} density (log1p)",
        ylabel=f"log1p({spec_label})",
    )

    meta = {
        "data_path": str(data_path.resolve()),
        "n_samples": int(spec.shape[0]),
        "n_bands": int(n_bands),
        "spectral_variable": spec_name,
        "has_elevation_input": elev is not None,
        "qoi_names": names,
        "log_scale_qois": sorted(S2_LOG_SCALE_TASK_NAMES),
        "qoi_stats_raw": {name: _stats(Y[:, j]) for j, name in enumerate(names)},
        "qoi_stats_log10": {
            name: _stats(np.log10(np.clip(_finite(Y[:, j]), 1e-30, None)))
            for j, name in enumerate(names)
            if name in S2_LOG_SCALE_TASK_NAMES
        },
        "elevation_stats": _stats(elev) if elev is not None else None,
        "transform_spectral": f"log1p = ln(1 + max({spec_label}, 0))",
        "band_stats_raw": {
            str(j): {**_stats(spec[:, j]), "wavelength_nm": float(wl[j])}
            for j in range(n_bands)
        },
        "band_stats_log1p": {
            str(j): {**_stats(spec_log[:, j]), "wavelength_nm": float(wl[j])}
            for j in range(n_bands)
        },
    }
    summary_path = out_dir / "histogram_summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {summary_path}")
    return meta


def run_radiance_histograms(data_path: Path, out_dir: Path) -> dict:
    """Back-compat alias."""
    return run_histograms(data_path, out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=S2_DEFAULT_DATA_PATH)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_toa_July21",
    )
    args = p.parse_args()
    run_histograms(args.data_path, args.out_dir)


if __name__ == "__main__":
    main()
