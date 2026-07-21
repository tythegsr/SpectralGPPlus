"""Histograms of S2 TOA radiance (all wavelengths): raw and log1p."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_constants import S2_DEFAULT_DATA_PATH, S2_TASK_NAMES

QOI_NAMES = list(S2_TASK_NAMES)
BANDS_PER_PAGE = 24  # 4 x 6
NCOLS = 4


def _load_radiance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        rad = np.asarray(f["toa_radiance"][:], dtype=np.float64)
    return wl, rad


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


def run_radiance_histograms(data_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading radiance from {data_path}")
    wl, rad = _load_radiance(data_path)
    print(f"  rad={rad.shape} wl={wl.min():.1f}-{wl.max():.1f} nm")

    rad_log = radiance_log1p(rad)
    n_bands = rad.shape[1]

    print("Full-band radiance histograms (raw) -> PDF...")
    _plot_all_band_histograms_pdf(
        rad,
        wl,
        out=out_dir / "toa_radiance_histograms_raw.pdf",
        title="TOA radiance — raw",
        color="darkseagreen",
    )
    print("Full-band radiance histograms (log1p) -> PDF...")
    _plot_all_band_histograms_pdf(
        rad_log,
        wl,
        out=out_dir / "toa_radiance_histograms_log1p.pdf",
        title="TOA radiance — log1p",
        color="steelblue",
    )

    print("Wavelength-value density (raw)...")
    _plot_wavelength_value_density(
        rad,
        wl,
        out=out_dir / "toa_radiance_density_raw.png",
        title="TOA radiance density (raw)",
        ylabel="Radiance",
    )
    print("Wavelength-value density (log1p)...")
    _plot_wavelength_value_density(
        rad_log,
        wl,
        out=out_dir / "toa_radiance_density_log1p.png",
        title="TOA radiance density (log1p)",
        ylabel="log1p(radiance)",
    )

    meta = {
        "data_path": str(data_path.resolve()),
        "n_samples": int(rad.shape[0]),
        "n_bands": int(n_bands),
        "transform": "log1p = ln(1 + max(radiance, 0))",
        "band_stats_raw": {
            str(j): {**_stats(rad[:, j]), "wavelength_nm": float(wl[j])} for j in range(n_bands)
        },
        "band_stats_log1p": {
            str(j): {**_stats(rad_log[:, j]), "wavelength_nm": float(wl[j])} for j in range(n_bands)
        },
    }
    summary_path = out_dir / "radiance_histogram_summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {summary_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=S2_DEFAULT_DATA_PATH)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_toa_July21",
    )
    args = p.parse_args()
    run_radiance_histograms(args.data_path, args.out_dir)


if __name__ == "__main__":
    main()
