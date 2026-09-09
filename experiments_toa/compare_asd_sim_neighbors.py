"""Find synthetic spectra nearest to ASD validation scenes and diagnose RRMSE gap.

For each ASD TOA radiance spectrum, finds k nearest neighbors in the synthetic
snow-TOA file (relative RMSE / cosine distance on radiance), plots overlays,
and compares QoI labels among neighbors vs ASD truth and optional model preds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_low_part_Sep03.nc"
)
DEFAULT_OUT = _ROOT / "experiments_toa" / "asd_sim_nn_compare"
DEFAULT_ASD_EVAL = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Sept03_log_final"
    / "s4_emit_aotbelow02_sorf_1inits_numrff1000_lr0.04_nigp_freezeepochnigp200_sloperefreshes10_dtypefloat32"
    / "asd_validation_eval"
)

QOI_MAP = {
    "grain_size": ("grain_radius_mean", "grain_size"),
    "cos_i": ("cos_i", "cos_i"),
    "dust": ("dust_conc_mean", "dust"),
    "algae": ("algae_conc_mean", "algae"),
    "cwv": ("cwv", "cwv"),
    "lwc": ("lwc_mean", "liquid_water"),
    "aot": ("aod", "aot"),
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 160,
    }
)


def _decode_dates(raw) -> list[str]:
    out = []
    for d in raw:
        if isinstance(d, (bytes, bytearray)):
            d = d.decode()
        out.append(str(d))
    return out


def _rel_rmse(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """(n_q, n_g) relative RMSE: rms(g-q) / mean(|q|)."""
    q = np.asarray(query, float)
    g = np.asarray(gallery, float)
    scale = np.maximum(np.mean(np.abs(q), axis=1, keepdims=True), 1e-8)
    diff2 = ((g[None, :, :] - q[:, None, :]) ** 2).mean(axis=2)
    return np.sqrt(diff2) / scale


def _cosine_dist(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = np.asarray(query, float)
    g = np.asarray(gallery, float)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return 1.0 - (qn @ gn.T)


def _load_sorted(ds: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    vals = np.asarray(ds[sorted_idx], float)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return vals[inv]


def find_neighbors(
    asd_rad: np.ndarray,
    sim_rad: np.ndarray,
    *,
    k: int = 5,
    metric: str = "rel_rmse",
    chunk: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nn_idx [n_asd,k], nn_dist [n_asd,k])."""
    n_asd = asd_rad.shape[0]
    n_sim = sim_rad.shape[0]
    best_idx = np.full((n_asd, k), -1, dtype=np.int64)
    best_dist = np.full((n_asd, k), np.inf, dtype=np.float64)

    for start in range(0, n_sim, chunk):
        end = min(start + chunk, n_sim)
        gal = sim_rad[start:end]
        if metric == "cosine":
            d = _cosine_dist(asd_rad, gal)
        else:
            d = _rel_rmse(asd_rad, gal)
        for i in range(n_asd):
            di = d[i]
            # merge with current top-k
            cand_d = np.concatenate([best_dist[i], di])
            cand_i = np.concatenate(
                [best_idx[i], np.arange(start, end, dtype=np.int64)]
            )
            order = np.argsort(cand_d)[:k]
            best_dist[i] = cand_d[order]
            best_idx[i] = cand_i[order]
    return best_idx, best_dist


def plot_spectra_grid(
    wl: np.ndarray,
    asd_rad: np.ndarray,
    sim_rad: np.ndarray,
    nn_idx: np.ndarray,
    nn_dist: np.ndarray,
    dates: list[str],
    out_path: Path,
    *,
    metric_name: str,
) -> None:
    n = asd_rad.shape[0]
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        ax.plot(wl, asd_rad[i], color="#4C78A8", lw=2.0, label="ASD", zorder=3)
        for r, (j, dist) in enumerate(zip(nn_idx[i], nn_dist[i])):
            alpha = 0.85 - 0.12 * r
            ax.plot(
                wl,
                sim_rad[j],
                lw=1.1,
                alpha=max(alpha, 0.35),
                label=f"NN{r+1} ({metric_name}={dist:.3g})",
            )
        ax.set_title(f"S{i+1}  {dates[i]}")
        ax.set_ylabel("TOA radiance")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols) : n]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        f"ASD vs nearest synthetic spectra ({metric_name})",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_qoi_neighbor_panels(
    asd_qoi: dict[str, np.ndarray],
    sim_qoi: dict[str, np.ndarray],
    nn_idx: np.ndarray,
    asd_std: dict[str, np.ndarray] | None,
    model_pred: dict[str, np.ndarray] | None,
    out_path: Path,
    tasks: list[str],
) -> None:
    n = next(iter(asd_qoi.values())).shape[0]
    k = nn_idx.shape[1]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.2))
    axes = np.atleast_1d(axes)
    x = np.arange(n)
    for ax, task in zip(axes, tasks):
        yt = asd_qoi[task]
        nn_vals = sim_qoi[task][nn_idx]  # (n, k)
        nn_med = np.median(nn_vals, axis=1)
        nn_lo = np.min(nn_vals, axis=1)
        nn_hi = np.max(nn_vals, axis=1)
        ax.fill_between(
            x,
            nn_lo,
            nn_hi,
            color="#54A24B",
            alpha=0.25,
            label=f"sim NN1–{k} range",
            zorder=1,
        )
        ax.plot(x, nn_med, "s-", color="#54A24B", ms=6, label="sim NN median", zorder=2)
        yerr = None
        if asd_std is not None and task in asd_std:
            yerr = asd_std[task]
        ax.errorbar(
            x,
            yt,
            yerr=yerr,
            fmt="D",
            color="#4C78A8",
            ms=7,
            ecolor="#9E9AC8",
            elinewidth=1.6,
            capsize=3,
            label="ASD (±1σ)" if yerr is not None else "ASD",
            zorder=4,
        )
        if model_pred is not None and task in model_pred:
            ax.plot(
                x,
                model_pred[task],
                "o-",
                color="#F58518",
                ms=7,
                label="model pred",
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)])
        ax.set_title(task)
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.25)
        if task in {"grain_size", "dust", "algae"} and np.nanmax(np.abs(yt)) > 50:
            ax.set_yscale("symlog", linthresh=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "QoI: ASD truth vs spectral nearest-neighbor synthetic labels (+ model)",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_vs_nn(
    asd_qoi: dict[str, np.ndarray],
    sim_qoi: dict[str, np.ndarray],
    nn_idx: np.ndarray,
    model_pred: dict[str, np.ndarray] | None,
    out_path: Path,
    tasks: list[str],
) -> None:
    """Show that model errors track NN-label mismatch more than spectral distance."""
    if model_pred is None:
        return
    n = next(iter(asd_qoi.values())).shape[0]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8))
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        yt = asd_qoi[task]
        yp = model_pred[task]
        nn1 = sim_qoi[task][nn_idx[:, 0]]
        model_err = yp - yt
        nn_err = nn1 - yt
        ax.axhline(0, color="#888888", lw=1)
        ax.scatter(
            np.arange(n),
            model_err,
            s=60,
            color="#F58518",
            label="model − ASD",
            zorder=3,
        )
        ax.scatter(
            np.arange(n) + 0.12,
            nn_err,
            s=60,
            marker="s",
            color="#54A24B",
            label="NN1 label − ASD",
            zorder=3,
        )
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([f"S{i+1}" for i in range(n)])
        ax.set_title(task)
        ax.set_ylabel("error")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Model error vs nearest-sim label mismatch (same spectral neighbor)",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _rrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Match experiments_toa / gpplus: RMSE / std(y_true)."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    denom = float(np.std(yt))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((yp - yt) ** 2)) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd", type=Path, default=DEFAULT_ASD)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--asd-eval-dir", type=Path, default=DEFAULT_ASD_EVAL)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--metric", choices=("rel_rmse", "cosine"), default="rel_rmse")
    parser.add_argument("--subsample", type=int, default=0, help="0 = use full sim")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    with h5py.File(args.asd, "r") as fa, h5py.File(args.sim, "r") as fs:
        wl = np.asarray(fa["wl"][:], float)
        asd_rad = np.asarray(fa["toa_radiance"][:], float)
        dates = _decode_dates(fa["date"][:])
        n_sim = int(fs["toa_radiance"].shape[0])
        rng = np.random.default_rng(args.seed)
        if args.subsample and args.subsample < n_sim:
            sim_idx = np.sort(rng.choice(n_sim, args.subsample, replace=False))
        else:
            sim_idx = np.arange(n_sim, dtype=np.int64)
        print(f"Loading {len(sim_idx)}/{n_sim} synthetic spectra...")
        sim_rad = _load_sorted(fs["toa_radiance"], sim_idx)

        asd_qoi = {}
        sim_qoi_full = {}
        asd_std = {}
        for task, (asd_key, sim_key) in QOI_MAP.items():
            asd_qoi[task] = np.asarray(fa[asd_key][:], float)
            sim_qoi_full[task] = np.asarray(fs[sim_key][:], float)
            std_key = {
                "grain_size": "grain_radius_std",
                "dust": "dust_conc_std",
                "algae": "algae_conc_std",
                "lwc": "lwc_std",
            }.get(task)
            if std_key and std_key in fa:
                asd_std[task] = np.asarray(fa[std_key][:], float)

        # aux for interpretation
        asd_sza = np.asarray(fa["sza"][:], float)
        asd_elev = np.asarray(fa["elevation_km"][:], float)
        sim_sza = np.asarray(fs["sza"][:], float)
        sim_elev = np.asarray(fs["ele_km"][:], float)
        sim_slope = np.asarray(fs["slope"][:], float)
        sim_coszen = np.asarray(fs["coszen"][:], float) if "coszen" in fs else np.cos(
            np.radians(sim_sza)
        )

    print(f"Finding k={args.k} neighbors by {args.metric}...")
    nn_local, nn_dist = find_neighbors(
        asd_rad, sim_rad, k=args.k, metric=args.metric
    )
    nn_idx = sim_idx[nn_local]  # global indices into full sim
    sim_qoi = {t: v[nn_idx] for t, v in ((t, sim_qoi_full[t]) for t in QOI_MAP)}
    # also keep full-indexed arrays for neighbor panels
    sim_qoi_nn = {t: sim_qoi_full[t] for t in QOI_MAP}

    # optional model preds
    model_pred = None
    model_std = None
    asd_eval_metrics = None
    eval_dir = Path(args.asd_eval_dir)
    metrics_path = eval_dir / "asd_validation_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as f:
            asd_eval_metrics = json.load(f)
        model_pred = {}
        model_std = {}
        for t, d in asd_eval_metrics.get("per_task", {}).items():
            if "y_pred" in d:
                model_pred[t] = np.asarray(d["y_pred"], float)
            if "y_std" in d:
                model_std[t] = np.asarray(d["y_std"], float)
        print(f"Loaded model ASD preds from {metrics_path}")

    # plots
    plot_spectra_grid(
        wl,
        asd_rad,
        sim_rad,
        nn_local,
        nn_dist,
        dates,
        plots / f"00_spectra_nn_{args.metric}.png",
        metric_name=args.metric,
    )

    focus = ["grain_size", "cos_i"]
    plot_qoi_neighbor_panels(
        asd_qoi,
        sim_qoi_nn,
        nn_idx,
        asd_std,
        model_pred,
        plots / "01_qoi_asd_vs_nn_labels.png",
        focus,
    )
    plot_residual_vs_nn(
        asd_qoi,
        sim_qoi_nn,
        nn_idx,
        model_pred,
        plots / "02_model_err_vs_nn_label_err.png",
        focus,
    )

    # per-scene residual spectra for NN1
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), sharex=True)
    axes = axes.ravel()
    for i in range(asd_rad.shape[0]):
        ax = axes[i]
        j_loc = nn_local[i, 0]
        resid = sim_rad[j_loc] - asd_rad[i]
        ax.axhline(0, color="#888", lw=0.8)
        ax.plot(wl, resid, color="#E45756", lw=1.1)
        ax.set_title(
            f"S{i+1} NN1 residual  {args.metric}={nn_dist[i,0]:.3g}"
        )
        ax.set_ylabel("sim − ASD")
        ax.grid(alpha=0.2)
    for ax in axes[-3:]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle("Nearest-neighbor spectral residual (sim − ASD)", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(plots / "03_nn1_spectral_residuals.png", bbox_inches="tight")
    plt.close(fig)

    # geometry of NN1 vs ASD
    nn1 = nn_idx[:, 0]
    summary_rows = []
    for i in range(asd_rad.shape[0]):
        row = {
            "sample": i + 1,
            "date": dates[i],
            f"nn1_{args.metric}": float(nn_dist[i, 0]),
            "asd_sza": float(asd_sza[i]),
            "nn1_sza": float(sim_sza[nn1[i]]),
            "asd_elev_km": float(asd_elev[i]),
            "nn1_elev_km": float(sim_elev[nn1[i]]),
            "nn1_slope_deg": float(sim_slope[nn1[i]]),
            "nn1_coszen": float(sim_coszen[nn1[i]]),
            "asd_coszen": float(np.cos(np.radians(asd_sza[i]))),
        }
        for task in QOI_MAP:
            row[f"asd_{task}"] = float(asd_qoi[task][i])
            row[f"nn1_{task}"] = float(sim_qoi_full[task][nn1[i]])
            nn_vals = sim_qoi_full[task][nn_idx[i]]
            row[f"nn_med_{task}"] = float(np.median(nn_vals))
            row[f"nn_span_{task}"] = float(np.max(nn_vals) - np.min(nn_vals))
            if model_pred and task in model_pred:
                row[f"pred_{task}"] = float(model_pred[task][i])
                row[f"pred_err_{task}"] = float(model_pred[task][i] - asd_qoi[task][i])
                row[f"nn1_err_{task}"] = float(
                    sim_qoi_full[task][nn1[i]] - asd_qoi[task][i]
                )
        summary_rows.append(row)

    # aggregate diagnostics
    diagnostics: dict = {
        "asd_path": str(args.asd),
        "sim_path": str(args.sim),
        "n_sim_used": int(len(sim_idx)),
        "n_sim_total": int(n_sim),
        "k": int(args.k),
        "metric": args.metric,
        "per_sample": summary_rows,
    }

    for task in focus:
        nn1_labels = sim_qoi_full[task][nn1]
        yt = asd_qoi[task]
        block = {
            "asd_vs_nn1_label_RRMSE": _rrmse(yt, nn1_labels),
            "asd_vs_nn_median_label_RRMSE": _rrmse(
                yt, np.median(sim_qoi_full[task][nn_idx], axis=1)
            ),
            "nn1_label_mae": float(np.mean(np.abs(nn1_labels - yt))),
            "nn_span_median": float(
                np.median(np.ptp(sim_qoi_full[task][nn_idx], axis=1))
            ),
            "nn_span_mean": float(np.mean(np.ptp(sim_qoi_full[task][nn_idx], axis=1))),
        }
        if model_pred and task in model_pred:
            yp = model_pred[task]
            block["model_vs_asd_RRMSE"] = _rrmse(yt, yp)
            block["model_vs_asd_mae"] = float(np.mean(np.abs(yp - yt)))
            # correlation between model error and nn1 label error
            me = yp - yt
            ne = nn1_labels - yt
            if np.std(me) > 0 and np.std(ne) > 0:
                block["corr_model_err_nn1_err"] = float(np.corrcoef(me, ne)[0, 1])
            # how often model closer to nn1 label than to ASD
            block["frac_pred_closer_to_nn1_than_asd"] = float(
                np.mean(np.abs(yp - nn1_labels) < np.abs(yp - yt))
            )
            if asd_std and task in asd_std:
                block["frac_pred_inside_asd_1sigma"] = float(
                    np.mean(np.abs(yp - yt) <= asd_std[task])
                )
                block["frac_nn1_inside_asd_1sigma"] = float(
                    np.mean(np.abs(nn1_labels - yt) <= asd_std[task])
                )
        diagnostics[task] = block

    diagnostics["mean_nn1_spectral_dist"] = float(np.mean(nn_dist[:, 0]))
    diagnostics["median_nn1_spectral_dist"] = float(np.median(nn_dist[:, 0]))

    # geometry mismatch summary
    diagnostics["geometry"] = {
        "asd_elev_km_unique": sorted(set(float(x) for x in asd_elev)),
        "nn1_elev_km_mean": float(np.mean(sim_elev[nn1])),
        "nn1_slope_deg_mean": float(np.mean(sim_slope[nn1])),
        "nn1_frac_slope_lt_2": float(np.mean(sim_slope[nn1] < 2.0)),
        "asd_vs_nn1_sza_mae": float(np.mean(np.abs(sim_sza[nn1] - asd_sza))),
        "asd_coszen_vs_nn1_cos_i_mae": float(
            np.mean(np.abs(sim_qoi_full["cos_i"][nn1] - np.cos(np.radians(asd_sza))))
        ),
    }

    if asd_eval_metrics is not None:
        diagnostics["asd_eval_checkpoint"] = asd_eval_metrics.get("checkpoint_title")
        diagnostics["asd_eval_aggregate_RRMSE"] = asd_eval_metrics.get("aggregate_RRMSE")

    out_json = out_dir / "asd_sim_nn_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nWrote {out_json}")
    print(f"Plots under {plots}")
    for task in focus:
        b = diagnostics[task]
        print(f"\n=== {task} ===")
        for k, v in b.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4g}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
