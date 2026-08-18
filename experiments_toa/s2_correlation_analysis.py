"""Correlation / drop-band analysis for S2 11-QoI TOA NetCDF datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_constants import S2_TASK_NAMES

QOI_NAMES = list(S2_TASK_NAMES)


def _safe_max_abs_corr(
    corr_col: np.ndarray,
    wl: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize max |r| for one QoI; handles constant (all-NaN) columns."""
    abs_r = np.abs(np.asarray(corr_col, dtype=np.float64))
    if not np.any(np.isfinite(abs_r)):
        return {
            "max_abs_r": float("nan"),
            "band_index": None,
            "wavelength_nm": None,
            "constant_qoi": True,
        }
    idx = int(np.nanargmax(abs_r))
    return {
        "max_abs_r": float(abs_r[idx]),
        "band_index": idx,
        "wavelength_nm": float(wl[idx]),
        "constant_qoi": False,
    }


def _qoi_names_in_file(f) -> list[str]:
    names = [n for n in QOI_NAMES if n in f]
    if not names:
        raise KeyError(f"No S2 QoI datasets found among {QOI_NAMES}")
    return names


def _load(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    with h5py.File(path, "r") as f:
        if "wl" not in f:
            raise KeyError(f"Missing wavelength variable 'wl' in {path}")
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        if "toa_reflectance" in f:
            refl = np.asarray(f["toa_reflectance"][:], dtype=np.float64)
        elif "reflectance" in f:
            refl = np.asarray(f["reflectance"][:], dtype=np.float64)
        else:
            raise KeyError(f"Missing reflectance in {path}")
        rad = (
            np.asarray(f["toa_radiance"][:], dtype=np.float64)
            if "toa_radiance" in f
            else None
        )
        names = _qoi_names_in_file(f)
        Y = np.column_stack([np.asarray(f[n][:], dtype=np.float64) for n in names])
    return wl, refl, rad, Y, names


def _suggest_drop_indices(X: np.ndarray) -> list[int]:
    """Flag absorption / low-SNR bands from std and off-diagonal correlation."""
    std = X.std(axis=0)
    med_std = float(np.median(std[std > 0])) if np.any(std > 0) else 1.0
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    drop: list[int] = []
    for j in range(n):
        off = np.abs(corr[j, :])
        off[j] = np.nan
        mean_abs = float(np.nanmean(off))
        frac_weak = float(np.nanmean(off < 0.05))
        if std[j] < 0.05 * med_std or mean_abs < 0.05 or frac_weak > 0.85:
            drop.append(j)
    return drop


def _plot_qoi_corr(
    Y: np.ndarray, out: Path, names: Sequence[str] | None = None
) -> np.ndarray:
    names = list(names) if names is not None else QOI_NAMES
    corr = np.corrcoef(Y, rowvar=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("QoI–QoI Pearson correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return corr


def _plot_band_corr(corr: np.ndarray, wl: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", origin="lower")
    n = corr.shape[0]
    ticks = np.linspace(0, n - 1, 8, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{wl[i]:.0f}" for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{wl[i]:.0f}" for i in ticks])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_band_corr_by_index(corr: np.ndarray, drop: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", origin="lower")
    for j in drop:
        ax.axvline(j, color="cyan", lw=0.4, alpha=0.7)
        ax.axhline(j, color="cyan", lw=0.4, alpha=0.7)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Band index")
    ax.set_title("Reflectance band correlation (cyan = drop candidates)")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_band_vs_qoi(
    band_qoi: np.ndarray,
    wl: np.ndarray,
    drop: list[int],
    out: Path,
    *,
    title: str = "Pearson r: reflectance bands vs QoI",
    names: Sequence[str] | None = None,
) -> None:
    names = list(names) if names is not None else QOI_NAMES
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        band_qoi,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="lower",
        extent=(-0.5, band_qoi.shape[1] - 0.5, -0.5, band_qoi.shape[0] - 0.5),
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    yticks = np.linspace(0, band_qoi.shape[0] - 1, 10, dtype=int)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{i}\n{wl[i]:.0f}nm" for i in yticks], fontsize=7)
    for j in drop:
        ax.axhline(j, color="cyan", lw=0.35, alpha=0.6)
    ax.set_ylabel("Band index / wavelength")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_joint(
    corr_full: np.ndarray,
    wl: np.ndarray,
    drop: list[int],
    out: Path,
    *,
    title: str = "Joint reflectance + QoI Pearson correlation",
    names: Sequence[str] | None = None,
) -> None:
    names = list(names) if names is not None else QOI_NAMES
    n_b = len(wl)
    n_q = len(names)
    fig = plt.figure(figsize=(12, 11))
    gs = GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[4, 1.2], hspace=0.08, wspace=0.08)
    ax_bb = fig.add_subplot(gs[0, 0])
    ax_bq = fig.add_subplot(gs[0, 1], sharey=ax_bb)
    ax_qb = fig.add_subplot(gs[1, 0], sharex=ax_bb)
    ax_qq = fig.add_subplot(gs[1, 1])

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax_bb.imshow(corr_full[:n_b, :n_b], cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    ax_bq.imshow(corr_full[:n_b, n_b:], cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    ax_qb.imshow(corr_full[n_b:, :n_b], cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    im = ax_qq.imshow(corr_full[n_b:, n_b:], cmap="RdBu_r", norm=norm, origin="lower", aspect="equal")

    for j in drop:
        ax_bb.axvline(j, color="cyan", lw=0.3, alpha=0.5)
        ax_bb.axhline(j, color="cyan", lw=0.3, alpha=0.5)

    ax_bq.set_xticks(range(n_q))
    ax_bq.set_xticklabels(names, rotation=90, fontsize=7)
    ax_qb.set_yticks(range(n_q))
    ax_qb.set_yticklabels(names, fontsize=7)
    ax_qq.set_xticks(range(n_q))
    ax_qq.set_yticks(range(n_q))
    ax_qq.set_xticklabels(names, rotation=90, fontsize=7)
    ax_qq.set_yticklabels(names, fontsize=7)
    for i in range(n_q):
        for j in range(n_q):
            ax_qq.text(
                j,
                i,
                f"{corr_full[n_b + i, n_b + j]:.2f}",
                ha="center",
                va="center",
                fontsize=5,
            )

    ticks = np.linspace(0, n_b - 1, 6, dtype=int)
    ax_bb.set_yticks(ticks)
    ax_bb.set_yticklabels([str(i) for i in ticks])
    ax_qb.set_xticks(ticks)
    ax_qb.set_xticklabels([str(i) for i in ticks])
    ax_bb.set_ylabel("Band index")
    ax_qb.set_xlabel("Band index")
    ax_bb.set_title("Band–band")
    ax_bq.set_title("Band–QoI")
    ax_qb.set_title("QoI–band")
    ax_qq.set_title("QoI–QoI")
    fig.colorbar(im, ax=[ax_bb, ax_bq, ax_qb, ax_qq], fraction=0.02, pad=0.02)
    fig.suptitle(title, y=0.98)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis(data_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {data_path}")
    wl, refl, rad, Y, names = _load(data_path)
    rad_shape = None if rad is None else rad.shape
    print(f"  refl={refl.shape} rad={rad_shape} Y={Y.shape} wl={wl.min():.1f}-{wl.max():.1f} nm")
    print(f"  qoi={names}")

    meta = {
        "data_path": str(data_path.resolve()),
        "n_samples": int(refl.shape[0]),
        "n_bands": int(refl.shape[1]),
        "n_qoi": int(Y.shape[1]),
        "qoi_names": names,
        "missing_qoi": [n for n in QOI_NAMES if n not in names],
        "has_toa_radiance": rad is not None,
        "wl_min_nm": float(wl.min()),
        "wl_max_nm": float(wl.max()),
        "qoi_ranges": {
            name: {
                "min": float(Y[:, j].min()),
                "max": float(Y[:, j].max()),
                "mean": float(Y[:, j].mean()),
                "std": float(Y[:, j].std()),
            }
            for j, name in enumerate(names)
        },
    }

    print("QoI–QoI correlation...")
    qoi_corr = _plot_qoi_corr(Y, out_dir / "qoi_correlation_matrix.png", names=names)
    off = qoi_corr[np.triu_indices(len(names), 1)]
    meta["qoi_offdiag_abs_pearson_mean"] = float(np.nanmean(np.abs(off)))
    meta["qoi_offdiag_abs_pearson_max"] = float(np.nanmax(np.abs(off)))

    print("Band–band correlations...")
    corr_refl = np.corrcoef(refl, rowvar=False)
    _plot_band_corr(
        corr_refl,
        wl,
        "TOA reflectance band correlation",
        out_dir / "toa_reflectance_band_correlation.png",
    )
    if rad is not None:
        corr_rad = np.corrcoef(rad, rowvar=False)
        _plot_band_corr(
            corr_rad,
            wl,
            "TOA radiance band correlation",
            out_dir / "toa_radiance_band_correlation.png",
        )

    drop = _suggest_drop_indices(refl)
    meta["bands_to_drop_indices"] = drop
    meta["bands_to_drop_wavelengths_nm"] = [float(wl[i]) for i in drop]
    _plot_band_corr_by_index(
        corr_refl, drop, out_dir / "toa_reflectance_band_correlation_by_index.png"
    )

    drop_path = out_dir / "bands_to_drop_indices.txt"
    drop_path.write_text(
        "# Band indices recommended to drop (low signal / absorption-like)\n"
        "# Criteria: std < 5% of median std, or mean |off-diag r| < 0.05, or >85% of |r|<0.05\n"
        f"# n_drop={len(drop)} of {refl.shape[1]}\n"
        f"drop_indices = {drop}\n"
        f"drop_wavelengths_nm = {[float(wl[i]) for i in drop]}\n",
        encoding="utf-8",
    )

    print("Band vs QoI / joint...")
    band_qoi_refl = np.zeros((refl.shape[1], Y.shape[1]), dtype=np.float64)
    for t in range(Y.shape[1]):
        for j in range(refl.shape[1]):
            band_qoi_refl[j, t] = np.corrcoef(refl[:, j], Y[:, t])[0, 1]
    _plot_band_vs_qoi(
        band_qoi_refl,
        wl,
        drop,
        out_dir / "toa_reflectance_vs_qoi_correlation.png",
        title="Pearson r: reflectance bands vs QoI",
        names=names,
    )

    joint_refl = np.concatenate([refl, Y], axis=1)
    corr_full_refl = np.corrcoef(joint_refl, rowvar=False)
    _plot_joint(
        corr_full_refl,
        wl,
        drop,
        out_dir / "toa_reflectance_qoi_joint_correlation.png",
        title="Joint reflectance + QoI Pearson correlation",
        names=names,
    )

    meta["band_qoi_max_abs_pearson"] = {
        name: _safe_max_abs_corr(band_qoi_refl[:, t], wl)
        for t, name in enumerate(names)
    }

    if rad is not None:
        band_qoi_rad = np.zeros((rad.shape[1], Y.shape[1]), dtype=np.float64)
        for t in range(Y.shape[1]):
            for j in range(rad.shape[1]):
                band_qoi_rad[j, t] = np.corrcoef(rad[:, j], Y[:, t])[0, 1]
        _plot_band_vs_qoi(
            band_qoi_rad,
            wl,
            drop,
            out_dir / "toa_radiance_vs_qoi_correlation.png",
            title="Pearson r: radiance bands vs QoI",
            names=names,
        )

        joint_rad = np.concatenate([rad, Y], axis=1)
        corr_full_rad = np.corrcoef(joint_rad, rowvar=False)
        _plot_joint(
            corr_full_rad,
            wl,
            drop,
            out_dir / "toa_radiance_qoi_joint_correlation.png",
            title="Joint radiance + QoI Pearson correlation",
            names=names,
        )

        meta["radiance_band_qoi_max_abs_pearson"] = {
            name: _safe_max_abs_corr(band_qoi_rad[:, t], wl)
            for t, name in enumerate(names)
        }

        band_rr = np.array(
            [np.corrcoef(refl[:, j], rad[:, j])[0, 1] for j in range(refl.shape[1])]
        )
        meta["rad_vs_refl_per_band_pearson"] = {
            "mean": float(np.nanmean(band_rr)),
            "min": float(np.nanmin(band_rr)),
            "max": float(np.nanmax(band_rr)),
        }

    (out_dir / "analysis_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote artifacts under {out_dir}")
    print(f"  drop bands: {len(drop)}")
    print(f"  QoI off-diag |r| mean={meta['qoi_offdiag_abs_pearson_mean']:.2e}")
    for name, info in meta["band_qoi_max_abs_pearson"].items():
        r = info["max_abs_r"]
        r_s = f"{r:.3f}" if r == r else "nan"
        print(f"  refl {name:14s} max|r|={r_s} @ band {info['band_index']}")
    if "radiance_band_qoi_max_abs_pearson" in meta:
        for name, info in meta["radiance_band_qoi_max_abs_pearson"].items():
            r = info["max_abs_r"]
            r_s = f"{r:.3f}" if r == r else "nan"
            print(f"  rad  {name:14s} max|r|={r_s} @ band {info['band_index']}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 TOA correlation analysis")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(
            # Path(__file__).resolve().parent
            # / "data 11 QoI"
            # / "snow_toa_simulations_20262107.nc"
            Path(__file__).resolve().parent
            / "data 11 QoI"
            / "snow_toa_fsnow_70to100_20261208.nc"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "analysis_toa_August17"),
    )
    args = parser.parse_args()
    run_analysis(Path(args.data_path), Path(args.out_dir))


if __name__ == "__main__":
    main()
