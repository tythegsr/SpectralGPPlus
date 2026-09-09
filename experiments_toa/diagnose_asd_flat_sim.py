"""Diagnose why flat-surface synthetic training fails on ASD (esp. grain_size).

Writes easy-to-read plots + JSON into a dedicated output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_flat_Sep03.nc"
DEFAULT_EVAL = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Sept03_log_final"
    / "s4_emit_aotbelow02_sorf_1inits_numrff1000_lr0.04_nigp_freezeepochnigp200_sloperefreshes10_dtypefloat32"
    / "asd_validation_eval"
)
DEFAULT_OUT = _ROOT / "experiments_toa" / "asd_flat_sim_diagnosis"

QOI_ASD = {
    "grain_size": "grain_radius_mean",
    "cos_i": "cos_i",
    "dust": "dust_conc_mean",
    "algae": "algae_conc_mean",
    "lwc": "lwc_mean",
    "aot": "aod",
    "cwv": "cwv",
}
QOI_SIM = {
    "grain_size": "grain_size",
    "cos_i": "cos_i",
    "dust": "dust",
    "algae": "algae",
    "lwc": "liquid_water",
    "aot": "aot",
    "cwv": "cwv",
}
STD_ASD = {
    "grain_size": "grain_radius_std",
    "dust": "dust_conc_std",
    "algae": "algae_conc_std",
    "lwc": "lwc_std",
}

# Large, plain style for readability
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
C_ASD = "#1f77b4"
C_PRED = "#ff7f0e"
C_NN = "#2ca02c"
C_BAND = "#98df8a"
C_STD = "#9467bd"


def _rrmse(yt: np.ndarray, yp: np.ndarray) -> float:
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    s = float(np.std(yt))
    if s <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((yp - yt) ** 2)) / s)


def _rel_rmse_rows(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.mean(np.abs(q), axis=1, keepdims=True), 1e-8)
    return np.sqrt(((g[None, :, :] - q[:, None, :]) ** 2).mean(axis=2)) / scale


def find_nn(asd_rad: np.ndarray, sim_rad: np.ndarray, k: int = 8, chunk: int = 8000):
    n_asd, n_sim = asd_rad.shape[0], sim_rad.shape[0]
    best_d = np.full((n_asd, k), np.inf)
    best_i = np.full((n_asd, k), -1, dtype=np.int64)
    for start in range(0, n_sim, chunk):
        end = min(start + chunk, n_sim)
        d = _rel_rmse_rows(asd_rad, sim_rad[start:end])
        for i in range(n_asd):
            cand_d = np.concatenate([best_d[i], d[i]])
            cand_i = np.concatenate([best_i[i], np.arange(start, end)])
            o = np.argsort(cand_d)[:k]
            best_d[i] = cand_d[o]
            best_i[i] = cand_i[o]
    return best_i, best_d


def plot_scorecard(synth: dict, asd: dict, out: Path) -> None:
    tasks = ["cos_i", "grain_size"]
    labels = ["cos incidence", "grain size"]
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b1 = ax.bar(
        x - w / 2,
        [synth[t] for t in tasks],
        w,
        label="Synthetic holdout (same model)",
        color="#4C78A8",
    )
    b2 = ax.bar(
        x + w / 2,
        [asd[t] for t in tasks],
        w,
        label="ASD field validation",
        color="#F58518",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RRMSE  (RMSE / std of truth)")
    ax.set_title("Flat-surface model: synthetic holdout vs ASD field")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.15,
                f"{h:.3g}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    ax.annotate(
        "cos_i recovered\nwith flat slope",
        xy=(0 + w / 2, asd["cos_i"]),
        xytext=(0.35, asd["cos_i"] * 8),
        arrowprops={"arrowstyle": "->", "color": "#333"},
        fontsize=11,
    )
    ax.annotate(
        "grain still fails\n(~same as before flat)",
        xy=(1 + w / 2, asd["grain_size"]),
        xytext=(0.55, asd["grain_size"] * 0.25),
        arrowprops={"arrowstyle": "->", "color": "#333"},
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_bars(
    dates: list[str],
    yt: np.ndarray,
    yp: np.ndarray,
    ys: np.ndarray,
    ystd_asd: np.ndarray | None,
    title: str,
    ylabel: str,
    out: Path,
    *,
    logy: bool = False,
) -> None:
    n = len(yt)
    x = np.arange(n)
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    asd_yerr = ystd_asd if ystd_asd is not None else None
    ax.bar(
        x - w / 2,
        yt,
        w,
        yerr=asd_yerr,
        label="ASD truth (±1σ from solutions.csv)" if asd_yerr is not None else "ASD truth",
        color=C_ASD,
        error_kw={"ecolor": C_STD, "elinewidth": 2, "capsize": 4},
    )
    ax.bar(
        x + w / 2,
        yp,
        w,
        yerr=1.96 * ys,
        label="Model prediction (95% CI)",
        color=C_PRED,
        error_kw={"ecolor": C_PRED, "elinewidth": 1.5, "capsize": 3, "alpha": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}\n{d}" for i, d in enumerate(dates)], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    # annotate relative error
    for i in range(n):
        rel = (yp[i] - yt[i]) / max(abs(yt[i]), 1e-9)
        ax.text(
            i,
            max(yt[i], yp[i]) * (1.08 if not logy else 1.25),
            f"{rel:+.0%}",
            ha="center",
            fontsize=10,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_spectra_nn(
    wl: np.ndarray,
    asd_rad: np.ndarray,
    sim_rad: np.ndarray,
    nn_i: np.ndarray,
    nn_d: np.ndarray,
    dates: list[str],
    out: Path,
) -> None:
    n = asd_rad.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        ax.plot(wl, asd_rad[i], color=C_ASD, lw=2.4, label="ASD", zorder=3)
        ax.plot(
            wl,
            sim_rad[nn_i[i, 0]],
            color=C_NN,
            lw=1.8,
            label=f"Closest flat sim (dist={nn_d[i,0]:.3f})",
            zorder=2,
        )
        for r in range(1, min(4, nn_i.shape[1])):
            ax.plot(wl, sim_rad[nn_i[i, r]], color=C_BAND, lw=1.0, alpha=0.55, zorder=1)
        ax.set_title(f"Sample {i+1}  ({dates[i]})", fontweight="semibold")
        ax.set_ylabel("TOA radiance")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            ax.text(
                0.02,
                0.02,
                "pale green = next 3 closest",
                transform=ax.transAxes,
                fontsize=9,
                color="#555",
                va="bottom",
            )
    for ax in axes[-3:]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "ASD spectra vs nearest FLAT synthetic spectra\n"
        "(closer lines = better radiance match)",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_nn_qoi_panel(
    dates: list[str],
    asd_q: dict[str, np.ndarray],
    asd_std: dict[str, np.ndarray],
    sim_q: dict[str, np.ndarray],
    nn_i: np.ndarray,
    pred: dict[str, np.ndarray],
    out: Path,
) -> None:
    tasks = ["grain_size", "cos_i", "dust", "algae", "lwc"]
    titles = {
        "grain_size": "Grain size (µm)",
        "cos_i": "cos incidence",
        "dust": "Dust",
        "algae": "Algae",
        "lwc": "Liquid water (LWC)",
    }
    # Algae ASD ±1σ is tiny on a log axis; keep algae/lwc linear so std bars read clearly.
    log_tasks = {"grain_size", "dust"}
    n = len(dates)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        yt = asd_q[task]
        nn_vals = sim_q[task][nn_i]
        lo, hi = nn_vals.min(axis=1), nn_vals.max(axis=1)
        med = np.median(nn_vals, axis=1)
        ax.fill_between(
            x, lo, hi, color=C_BAND, alpha=0.45, label="Range of 8 closest flat sims"
        )
        ax.plot(x, med, "s-", color=C_NN, ms=8, lw=2, label="Median of those 8 sims")
        yerr = asd_std.get(task)
        has_std = yerr is not None and np.any(np.isfinite(yerr) & (yerr > 0))
        ax.errorbar(
            x,
            yt,
            yerr=yerr if has_std else None,
            fmt="D",
            ms=9,
            color=C_ASD,
            ecolor=C_STD,
            elinewidth=2.4,
            capsize=5,
            capthick=1.6,
            label="ASD truth (±1σ)" if has_std else "ASD truth (no σ in CSV)",
            zorder=5,
        )
        if task in pred:
            ax.plot(
                x,
                pred[task],
                "o-",
                color=C_PRED,
                ms=9,
                lw=2,
                label="Model prediction",
                zorder=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)])
        std_note = ""
        if has_std:
            std_note = f"\nASD σ range [{np.nanmin(yerr):.3g}, {np.nanmax(yerr):.3g}]"
        ax.set_title(titles[task] + std_note, fontweight="semibold")
        ax.grid(axis="y", alpha=0.25)
        if task in log_tasks:
            ax.set_yscale("log")
        elif task == "algae" and has_std:
            # ASD algae σ is ~1 on a ~20 mean; zoom so ±1σ bars are visible.
            pad = max(3.0, 4.0 * float(np.nanmax(yerr)))
            y0 = float(np.nanmin(yt - yerr) - pad)
            y1 = float(np.nanmax(yt + yerr) + pad)
            ax.set_ylim(max(0.0, y0), y1)
            ax.text(
                0.98,
                0.04,
                f"sim NN span [{lo.min():.3g}, {hi.max():.3g}] (clipped)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#2ca02c",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
            )
        if task == "grain_size":
            ax.legend(loc="lower left", frameon=True, fontsize=9)
    for ax in axes[len(tasks) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Do the closest flat spectra carry the right labels?\n"
        "Purple bars = ASD ±1σ from asd_solutions.csv (grain / dust / algae / LWC). "
        "If green band misses blue±σ, radiance twins have the wrong physics labels.",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_error_alignment(
    dates: list[str],
    asd_g: np.ndarray,
    nn1_g: np.ndarray,
    pred_g: np.ndarray,
    asd_c: np.ndarray,
    nn1_c: np.ndarray,
    pred_c: np.ndarray,
    out: Path,
) -> None:
    n = len(dates)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # grain
    ax = axes[0]
    ax.axhline(0, color="#888", lw=1)
    ax.plot(x, pred_g - asd_g, "o-", color=C_PRED, ms=10, lw=2, label="Model − ASD")
    ax.plot(x, nn1_g - asd_g, "s--", color=C_NN, ms=9, lw=2, label="Closest-sim label − ASD")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n)])
    ax.set_ylabel("Grain size error (µm)")
    ax.set_title("Grain: model error tracks sim-label mismatch", fontweight="semibold")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    # cos_i
    ax = axes[1]
    ax.axhline(0, color="#888", lw=1)
    ax.plot(x, pred_c - asd_c, "o-", color=C_PRED, ms=10, lw=2, label="Model − ASD")
    ax.plot(x, nn1_c - asd_c, "s--", color=C_NN, ms=9, lw=2, label="Closest-sim label − ASD")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n)])
    ax.set_ylabel("cos_i error")
    ax.set_title("cos_i: flat slope mostly fixes this", fontweight="semibold")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Orange ≈ green means the model is doing what the synthetic data teaches\n"
        "(not randomly wrong)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_residual_spectra(
    wl: np.ndarray,
    asd_rad: np.ndarray,
    sim_rad: np.ndarray,
    nn_i: np.ndarray,
    nn_d: np.ndarray,
    dates: list[str],
    out: Path,
) -> None:
    n = asd_rad.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        resid = sim_rad[nn_i[i, 0]] - asd_rad[i]
        ax.axhline(0, color="#666", lw=1)
        ax.plot(wl, resid, color="#d62728", lw=1.4)
        ax.set_title(f"S{i+1}  dist={nn_d[i,0]:.3f}", fontweight="semibold")
        ax.set_ylabel("Closest flat sim − ASD")
        ax.grid(alpha=0.2)
    for ax in axes[-3:]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Where radiance still disagrees (flat sim − ASD)\n"
        "Systematic residuals hint at RT / impurity / grain effects, not slope",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_takeaway_slide(
    grain_block: dict,
    cosi_block: dict,
    mean_dist: float,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "Flat-surface synthetic → ASD: what changed?", fontsize=18, fontweight="bold")
    ax.text(
        0.02,
        0.78,
        f"• cos_i: largely FIXED  (ASD RRMSE {cosi_block['model_rrmse']:.3f}; "
        f"synth holdout {cosi_block['synth_rrmse']:.4f})",
        fontsize=13,
        color="#1a7f37",
    )
    ax.text(
        0.02,
        0.66,
        f"• grain_size: STILL BROKEN  (ASD RRMSE {grain_block['model_rrmse']:.3f}; "
        f"synth holdout {grain_block['synth_rrmse']:.3f})",
        fontsize=13,
        color="#b42318",
    )
    ax.text(
        0.02,
        0.52,
        f"• Closest flat spectra match radiance well (mean rel. RMSE {mean_dist:.3f})",
        fontsize=13,
    )
    ax.text(
        0.02,
        0.40,
        f"• But those twins still have wrong grain labels "
        f"(NN1-label RRMSE {grain_block['nn1_label_rrmse']:.3f} ≈ model {grain_block['model_rrmse']:.3f})",
        fontsize=13,
    )
    ax.text(
        0.02,
        0.28,
        f"• Model error correlates with NN1 label error "
        f"(corr={grain_block['corr_model_nn1']:.2f}) → not a training bug",
        fontsize=13,
    )
    ax.text(
        0.02,
        0.14,
        "Next levers: impurity/grain coupling, RT consistency (MODTRAN vs sRTMnet),\n"
        "  ASD-like algae/dust priors across ALL grain sizes — not more slope tweaks.",
        fontsize=13,
        fontweight="semibold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asd", type=Path, default=DEFAULT_ASD)
    p.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    p.add_argument(
        "--gp-json",
        type=Path,
        default=None,
        help="Optional path to gp_S4_*.json (avoids Windows long-path issues)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((args.eval_dir / "asd_validation_metrics.json").read_text(encoding="utf-8"))
    title = metrics.get("checkpoint_title")
    gp = None
    if args.gp_json is not None and Path(args.gp_json).is_file():
        gp = json.loads(Path(args.gp_json).read_text(encoding="utf-8"))
    else:
        parent = Path(args.eval_dir).parent
        for jp in sorted(parent.glob("gp_S4_*.json")):
            try:
                g = json.loads(jp.read_text(encoding="utf-8"))
            except OSError:
                continue
            if title is None or g.get("title") == title or "nTest25000" in jp.name:
                gp = g
                break
        if gp is None:
            # long-path Windows fallback
            long_parent = Path("\\\\?\\" + str(parent.resolve()))
            try:
                names = list(long_parent.iterdir())
            except OSError:
                names = []
            for jp in names:
                if jp.name.startswith("gp_S4_") and "nTest25000" in jp.name:
                    gp = json.loads(jp.read_text(encoding="utf-8"))
                    break
    if gp is None:
        print("WARNING: could not load gp_S4_*.json; synth holdout RRMSE will be NaN")
        gp = {}

    pred = {}
    pred_std = {}
    for t, d in metrics["per_task"].items():
        pred[t] = np.asarray(d["y_pred"], float)
        pred_std[t] = np.asarray(d["y_std"], float)

    with h5py.File(args.asd, "r") as fa, h5py.File(args.sim, "r") as fs:
        wl = np.asarray(fa["wl"][:], float)
        asd_rad = np.asarray(fa["toa_radiance"][:], float)
        dates = [
            d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in fa["date"][:]
        ]
        asd_q = {t: np.asarray(fa[QOI_ASD[t]][:], float) for t in QOI_ASD}
        asd_std = {
            t: np.asarray(fa[STD_ASD[t]][:], float) for t in STD_ASD if STD_ASD[t] in fa
        }
        sim_rad = np.asarray(fs["toa_radiance"][:], float)
        sim_q = {t: np.asarray(fs[QOI_SIM[t]][:], float) for t in QOI_SIM}
        slope = np.asarray(fs["slope"][:], float)
        sim_attrs = {
            k: (v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v))
            for k, v in fs.attrs.items()
        }

    print(f"Finding k={args.k} neighbors in flat sim ({sim_rad.shape[0]} spectra)...")
    nn_i, nn_d = find_nn(asd_rad, sim_rad, k=args.k)

    synth_rrmse = {
        "cos_i": float(gp["cos_i_RRMSE"]) if gp else float("nan"),
        "grain_size": float(gp["grain_size_RRMSE"]) if gp else float("nan"),
    }
    asd_rrmse = {
        "cos_i": float(metrics["per_task"]["cos_i"]["RRMSE"]),
        "grain_size": float(metrics["per_task"]["grain_size"]["RRMSE"]),
    }

    plot_scorecard(synth_rrmse, asd_rrmse, plots / "00_scorecard_synth_vs_asd.png")
    plot_per_sample_bars(
        dates,
        asd_q["cos_i"],
        pred["cos_i"],
        pred_std["cos_i"],
        None,
        "cos_i on ASD — flat training largely works",
        "cos incidence",
        plots / "01_cos_i_asd_vs_pred.png",
    )
    plot_per_sample_bars(
        dates,
        asd_q["grain_size"],
        pred["grain_size"],
        pred_std["grain_size"],
        asd_std.get("grain_size"),
        "grain_size on ASD — still systematically low / outside ASD ±1σ",
        "grain size (µm)",
        plots / "02_grain_asd_vs_pred.png",
        logy=True,
    )
    plot_spectra_nn(wl, asd_rad, sim_rad, nn_i, nn_d, dates, plots / "03_spectra_asd_vs_flat_nn.png")
    plot_nn_qoi_panel(
        dates, asd_q, asd_std, sim_q, nn_i, pred, plots / "04_labels_of_closest_flat_spectra.png"
    )
    plot_error_alignment(
        dates,
        asd_q["grain_size"],
        sim_q["grain_size"][nn_i[:, 0]],
        pred["grain_size"],
        asd_q["cos_i"],
        sim_q["cos_i"][nn_i[:, 0]],
        pred["cos_i"],
        plots / "05_model_error_follows_sim_label_gap.png",
    )
    plot_residual_spectra(
        wl, asd_rad, sim_rad, nn_i, nn_d, dates, plots / "06_spectral_residuals_flat_nn1.png"
    )

    grain_block = {
        "synth_rrmse": synth_rrmse["grain_size"],
        "model_rrmse": asd_rrmse["grain_size"],
        "nn1_label_rrmse": _rrmse(asd_q["grain_size"], sim_q["grain_size"][nn_i[:, 0]]),
        "nn_med_label_rrmse": _rrmse(
            asd_q["grain_size"], np.median(sim_q["grain_size"][nn_i], axis=1)
        ),
        "corr_model_nn1": float(
            np.corrcoef(
                pred["grain_size"] - asd_q["grain_size"],
                sim_q["grain_size"][nn_i[:, 0]] - asd_q["grain_size"],
            )[0, 1]
        ),
        "frac_pred_inside_asd_1sigma": float(
            np.mean(np.abs(pred["grain_size"] - asd_q["grain_size"]) <= asd_std["grain_size"])
        )
        if "grain_size" in asd_std
        else None,
        "nn_span_median": float(np.median(np.ptp(sim_q["grain_size"][nn_i], axis=1))),
    }
    cosi_block = {
        "synth_rrmse": synth_rrmse["cos_i"],
        "model_rrmse": asd_rrmse["cos_i"],
        "nn1_label_rrmse": _rrmse(asd_q["cos_i"], sim_q["cos_i"][nn_i[:, 0]]),
        "corr_model_nn1": float(
            np.corrcoef(
                pred["cos_i"] - asd_q["cos_i"],
                sim_q["cos_i"][nn_i[:, 0]] - asd_q["cos_i"],
            )[0, 1]
        ),
        "cov95": float(metrics["per_task"]["cos_i"].get("coverage_95", float("nan"))),
    }
    plot_takeaway_slide(grain_block, cosi_block, float(nn_d[:, 0].mean()), plots / "07_takeaways.png")

    per_sample = []
    for i in range(len(dates)):
        row = {
            "sample": i + 1,
            "date": dates[i],
            "nn1_spectral_rel_rmse": float(nn_d[i, 0]),
            "asd_grain": float(asd_q["grain_size"][i]),
            "nn1_grain": float(sim_q["grain_size"][nn_i[i, 0]]),
            "pred_grain": float(pred["grain_size"][i]),
            "asd_cos_i": float(asd_q["cos_i"][i]),
            "nn1_cos_i": float(sim_q["cos_i"][nn_i[i, 0]]),
            "pred_cos_i": float(pred["cos_i"][i]),
            "nn1_dust": float(sim_q["dust"][nn_i[i, 0]]),
            "nn1_algae": float(sim_q["algae"][nn_i[i, 0]]),
            "asd_dust": float(asd_q["dust"][i]),
            "asd_algae": float(asd_q["algae"][i]),
        }
        per_sample.append(row)

    summary = {
        "eval_dir": str(args.eval_dir),
        "sim_path": str(args.sim),
        "checkpoint_title": title,
        "slope_min_max": [float(slope.min()), float(slope.max())],
        "sim_attrs": {k: sim_attrs[k] for k in sim_attrs if any(
            s in k.lower() for s in ("slope", "flat", "algae", "dust", "sampling", "range")
        )},
        "synth_holdout_rrmse": synth_rrmse,
        "asd_model_rrmse": asd_rrmse,
        "grain": grain_block,
        "cos_i": cosi_block,
        "mean_nn1_spectral_rel_rmse": float(nn_d[:, 0].mean()),
        "per_sample": per_sample,
        "plots": sorted(str(p.relative_to(out)) for p in plots.glob("*.png")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # also CSV for easy reading
    with open(out / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)

    print(f"\nWrote diagnosis to {out}")
    print(f"  cos_i  synth={synth_rrmse['cos_i']:.4g}  ASD={asd_rrmse['cos_i']:.4g}")
    print(f"  grain  synth={synth_rrmse['grain_size']:.4g}  ASD={asd_rrmse['grain_size']:.4g}")
    print(f"  grain NN1-label RRMSE={grain_block['nn1_label_rrmse']:.4g}  corr={grain_block['corr_model_nn1']:.3f}")
    print(f"  mean NN1 spectral dist={float(nn_d[:,0].mean()):.4f}")
    for name in sorted(plots.glob("*.png")):
        print(f"  plot: {name.name}")


if __name__ == "__main__":
    main()
