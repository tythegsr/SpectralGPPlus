"""Mean TOA spectrum: filtered EMIT vs sim 70–100, with p10–p90 coverage."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMIT = _ROOT / "split_files" / "emit_data_filtered.nc"
DEFAULT_SIM = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261208.nc"
)
DEFAULT_OUT = (
    _ROOT / "experiments_toa" / "reports" / "emit_vs_sim" / "figures"
    / "spectra_mean_p10p90_filtered.png"
)
EMIT_COLOR = "#1f4e79"
SIM_COLOR = "#b85c38"


def _band_stats(path: Path, dset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as f:
        x = np.asarray(f[dset][:], dtype=np.float32)
    n = int(x.shape[0])
    mean = np.nanmean(x, axis=0).astype(np.float64)
    p10 = np.nanpercentile(x, 10, axis=0).astype(np.float64)
    p90 = np.nanpercentile(x, 90, axis=0).astype(np.float64)
    return mean, p10, p90, n


def main() -> None:
    emit_path = DEFAULT_EMIT
    sim_path = DEFAULT_SIM
    out_path = DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(sim_path, "r") as f:
        wl = np.asarray(f["wl"][:], dtype=np.float64)

    emit_mean, emit_p10, emit_p90, n_emit = _band_stats(emit_path, "reflectance")
    sim_mean, sim_p10, sim_p90, n_sim = _band_stats(sim_path, "toa_reflectance")
    if emit_mean.shape != sim_mean.shape:
        raise ValueError(
            f"Band mismatch: EMIT {emit_mean.shape} vs sim {sim_mean.shape}"
        )

    delta = sim_mean - emit_mean
    overlap = (np.minimum(emit_p90, sim_p90) - np.maximum(emit_p10, sim_p10)) > 0
    emit_bb = float(np.nanmean(emit_mean))
    sim_bb = float(np.nanmean(sim_mean))
    print(f"EMIT filtered n={n_emit:,}  mean reflectance={emit_bb:.4f}")
    print(f"sim 70-100   n={n_sim:,}  mean reflectance={sim_bb:.4f}")
    print(f"ratio EMIT/sim={emit_bb / sim_bb:.3f}  mean(sim-EMIT)={float(np.nanmean(delta)):.4f}")
    print(f"p10-p90 envelopes overlap on {int(overlap.sum())}/{overlap.size} bands")

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
    )
    ax0.fill_between(
        wl, emit_p10, emit_p90, color=EMIT_COLOR, alpha=0.22, linewidth=0, label="EMIT p10–p90"
    )
    ax0.fill_between(
        wl, sim_p10, sim_p90, color=SIM_COLOR, alpha=0.22, linewidth=0, label="sim p10–p90"
    )
    ax0.plot(wl, emit_mean, color=EMIT_COLOR, lw=1.8, label=f"EMIT mean (n={n_emit:,})")
    ax0.plot(wl, sim_mean, color=SIM_COLOR, lw=1.8, label=f"sim 70–100 mean (n={n_sim:,})")
    ax0.set_ylabel("TOA reflectance")
    ax0.set_title("Filtered EMIT vs synthetic mean spectrum (p10–p90 coverage)")
    ax0.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax0.set_ylim(bottom=0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1.axhline(0.0, color="0.55", lw=0.8)
    ax1.fill_between(wl, 0.0, delta, where=delta >= 0, color=SIM_COLOR, alpha=0.28, linewidth=0)
    ax1.fill_between(wl, 0.0, delta, where=delta < 0, color=EMIT_COLOR, alpha=0.28, linewidth=0)
    ax1.plot(wl, delta, color="#2b2b2b", lw=1.5, label="sim mean − EMIT mean")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("Δ reflectance")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
