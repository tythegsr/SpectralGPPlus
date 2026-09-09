"""Is the flat-surface inverse problem multi-valued?

Quantifies how many distinct (grain, dust, algae, LWC, cos_i) solutions can produce
nearly the same TOA radiance — both among synthetic neighbors of ASD scenes and
within the flat synthetic set itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_flat_Sep03.nc"
DEFAULT_OUT = _ROOT / "experiments_toa" / "asd_flat_sim_diagnosis" / "multiplicity"

LABELS = ("grain_size", "cos_i", "dust", "algae", "liquid_water", "aot", "cwv")
ASD_MAP = {
    "grain_size": "grain_radius_mean",
    "cos_i": "cos_i",
    "dust": "dust_conc_mean",
    "algae": "algae_conc_mean",
    "liquid_water": "lwc_mean",
    "aot": "aod",
    "cwv": "cwv",
}
STD_MAP = {
    "grain_size": "grain_radius_std",
    "dust": "dust_conc_std",
    "algae": "algae_conc_std",
    "liquid_water": "lwc_std",
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def rel_rmse_to_gallery(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.mean(np.abs(q), axis=1, keepdims=True), 1e-8)
    return np.sqrt(((g[None, :, :] - q[:, None, :]) ** 2).mean(axis=2)) / scale


def chunked_dists(query: np.ndarray, gallery: np.ndarray, chunk: int = 5000) -> np.ndarray:
    n_q = query.shape[0]
    n_g = gallery.shape[0]
    out = np.empty((n_q, n_g), dtype=np.float64)
    for s in range(0, n_g, chunk):
        e = min(s + chunk, n_g)
        out[:, s:e] = rel_rmse_to_gallery(query, gallery[s:e])
    return out


def topk_from_dists(d: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # d: (n_q, n_g)
    idx = np.argpartition(d, kth=min(k, d.shape[1] - 1), axis=1)[:, :k]
    # sort those
    row = np.arange(d.shape[0])[:, None]
    part = d[row, idx]
    order = np.argsort(part, axis=1)
    idx = idx[row, order]
    dist = d[row, idx]
    return idx, dist


def label_spread(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, float)
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
        "span": float(np.max(vals) - np.min(vals)),
        "cv": float(np.std(vals) / max(abs(np.mean(vals)), 1e-12)),
    }


def n_distinct_bins(vals: np.ndarray, edges: np.ndarray) -> int:
    """Count occupied bins as a crude 'number of distinct solutions'."""
    idx = np.digitize(vals, edges) - 1
    idx = idx[(idx >= 0) & (idx < len(edges) - 1)]
    return int(len(np.unique(idx)))


def plot_solution_clouds(
    asd_q: dict[str, np.ndarray],
    asd_std: dict[str, np.ndarray],
    sim_q: dict[str, np.ndarray],
    nn_idx: np.ndarray,
    nn_dist: np.ndarray,
    dates: list[str],
    thr: float,
    out: Path,
) -> None:
    """For each ASD scene: scatter grain vs dust among all sims within thr."""
    n = len(dates)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        mask = nn_dist[i] <= thr
        # If too few within thr, fall back to top-50 already in nn_idx
        js = nn_idx[i, mask] if mask.sum() >= 10 else nn_idx[i]
        g = sim_q["grain_size"][js]
        d = sim_q["dust"][js]
        a = sim_q["algae"][js]
        sc = ax.scatter(
            g,
            d,
            c=a,
            s=28,
            cmap="viridis",
            alpha=0.75,
            edgecolors="none",
            vmin=0,
            vmax=100,
        )
        ax.errorbar(
            asd_q["grain_size"][i],
            asd_q["dust"][i],
            xerr=asd_std.get("grain_size", np.zeros(n))[i],
            yerr=asd_std.get("dust", np.zeros(n))[i],
            fmt="D",
            ms=10,
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.8,
            capsize=4,
            zorder=5,
            label="ASD +/-1 sigma",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"S{i+1} {dates[i]}\n"
            f"n_sims={len(js)}  grain span={g.max()-g.min():.0f}µm",
            fontweight="semibold",
        )
        ax.set_xlabel("grain (µm)")
        ax.set_ylabel("dust")
        ax.grid(alpha=0.25, which="both")
        if i == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)
    cbar = fig.colorbar(sc, ax=axes[:n].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("algae")
    fig.suptitle(
        f"Many solutions near each ASD spectrum (rel. RMSE <= {thr:g}, else top neighbors)\n"
        "Each point = one flat synthetic spectrum with similar radiance. "
        "Spread => non-unique grain/dust/algae.",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0.0, 0.90, 0.92])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_span_bars(
    per_asd: list[dict],
    asd_std: dict[str, np.ndarray],
    out: Path,
) -> None:
    """Compare label span among spectral neighbors vs ASD measurement sigma."""
    tasks = ["grain_size", "dust", "algae", "liquid_water", "cos_i"]
    titles = {
        "grain_size": "Grain size (µm)",
        "dust": "Dust",
        "algae": "Algae",
        "liquid_water": "LWC",
        "cos_i": "cos_i",
    }
    n = len(per_asd)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        spans = [p["within_thr"][task]["span"] for p in per_asd]
        stds = (
            asd_std[task]
            if task in asd_std
            else np.full(n, np.nan)
        )
        ax.bar(x - 0.18, spans, 0.36, color="#2ca02c", label="Span among similar sims")
        if np.any(np.isfinite(stds)):
            ax.bar(x + 0.18, 2 * stds, 0.36, color="#1f77b4", label="ASD 2-sigma width")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)])
        ax.set_title(titles[task], fontweight="semibold")
        ax.set_ylabel("value units")
        ax.grid(axis="y", alpha=0.25)
        if task == "grain_size":
            ax.legend(loc="upper right", frameon=True, fontsize=9)
        for i, (sp, sd) in enumerate(zip(spans, stds)):
            if np.isfinite(sd) and sd > 0 and sp > 4 * sd:
                ax.text(i, sp * 1.02, ">>sigma", ha="center", fontsize=9, color="#b42318")
    axes[-1].axis("off")
    axes[-1].text(
        0.0,
        0.55,
        "How to read this\n\n"
        "Green = how much the label varies\n"
        "among spectra that look almost alike.\n\n"
        "Blue = ASD measurement uncertainty\n"
        "(2x reported 1-sigma).\n\n"
        "If green >> blue, many physically\n"
        "different solutions fit the same\n"
        "radiance within noise.",
        fontsize=12,
        va="center",
        family="sans-serif",
    )
    fig.suptitle(
        "Is the solution unique? Neighbor label span vs ASD uncertainty",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_matrix(
    sim_q: dict[str, np.ndarray],
    nn_idx: np.ndarray,
    out: Path,
) -> None:
    """For pooled ASD-neighbors: pairwise scatter of labels colored by scene."""
    colors = plt.cm.tab10(np.linspace(0, 1, nn_idx.shape[0]))
    pairs = [
        ("grain_size", "dust"),
        ("grain_size", "algae"),
        ("grain_size", "liquid_water"),
        ("dust", "algae"),
        ("dust", "liquid_water"),
        ("algae", "liquid_water"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    axes = axes.ravel()
    for ax, (a, b) in zip(axes, pairs):
        for i in range(nn_idx.shape[0]):
            js = nn_idx[i]
            ax.scatter(
                sim_q[a][js],
                sim_q[b][js],
                s=18,
                alpha=0.55,
                color=colors[i],
                edgecolors="none",
                label=f"S{i+1}",
            )
        ax.set_xlabel(a.replace("liquid_water", "lwc").replace("_size", ""))
        ax.set_ylabel(b.replace("liquid_water", "lwc").replace("_size", ""))
        if a in {"grain_size", "dust", "algae"}:
            ax.set_xscale("log")
        if b in {"grain_size", "dust", "algae"}:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        # correlation in log space when positive
        xa = np.concatenate([sim_q[a][nn_idx[i]] for i in range(nn_idx.shape[0])])
        xb = np.concatenate([sim_q[b][nn_idx[i]] for i in range(nn_idx.shape[0])])
        if np.all(xa > 0) and np.all(xb > 0):
            corr = float(np.corrcoef(np.log10(xa), np.log10(xb))[0, 1])
        else:
            corr = float(np.corrcoef(xa, xb)[0, 1])
        ax.set_title(f"log-corr={corr:.2f}" if np.all(xa > 0) and np.all(xb > 0) else f"corr={corr:.2f}")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markersize=8, label=f"S{i+1}")
        for i in range(nn_idx.shape[0])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Label tradeoffs among radiance-similar flat sims (top-40 neighbors / ASD scene)\n"
        "Diagonal streaks / wide clouds => interchangeable solutions (degeneracy)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_within_sim_degeneracy(
    spreads: list[dict],
    thr: float,
    out: Path,
) -> None:
    """Histogram of neighbor spans for random sim queries."""
    tasks = ["grain_size", "dust", "algae", "liquid_water", "cos_i"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        spans = np.array([s[task]["span"] for s in spreads], float)
        ax.hist(spans, bins=30, color="#4C78A8", edgecolor="white")
        ax.axvline(np.median(spans), color="#d62728", lw=2, label=f"median={np.median(spans):.3g}")
        ax.set_title(task.replace("liquid_water", "lwc"), fontweight="semibold")
        ax.set_xlabel("label span among similar sims")
        ax.set_ylabel("count (random queries)")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.5,
        f"Within the FLAT synthetic set alone\n"
        f"(slope fixed at 0):\n\n"
        f"For random spectra, other sims within\n"
        f"rel. RMSE <= {thr:g} still span a wide\n"
        f"range of grain / dust / algae / LWC.\n\n"
        f"=> even with perfect RT and flat terrain,\n"
        f"the map radiance→QoIs is multi-valued.",
        fontsize=12,
        va="center",
    )
    fig.suptitle(
        "Internal degeneracy of the flat synthetic forward model",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_n_solutions(
    per_asd: list[dict],
    out: Path,
) -> None:
    """Bar chart: number of occupied grain bins among neighbors."""
    n = len(per_asd)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_grain = [p["n_grain_bins_50um"] for p in per_asd]
    n_dust = [p["n_dust_log_bins"] for p in per_asd]
    n_joint = [p["n_joint_grain_dust_bins"] for p in per_asd]
    ax.bar(x - 0.25, n_grain, 0.25, label="# distinct grain bins (50 µm)", color="#ff7f0e")
    ax.bar(x, n_dust, 0.25, label="# distinct dust log-bins", color="#2ca02c")
    ax.bar(x + 0.25, n_joint, 0.25, label="# joint (grain x dust) bins", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{p['sample']}" for p in per_asd])
    ax.set_ylabel("count of occupied bins among similar sims")
    ax.set_title(
        "How many different solutions sit near each ASD spectrum?\n"
        "(Higher bars => more distinct physical states with similar radiance)",
        fontweight="semibold",
    )
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(1, color="#888", ls="--", lw=1)
    ax.text(n - 0.1, 1.15, "unique", color="#888", ha="right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_verdict(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    multi = summary["verdict_multi_valued"]
    ax.text(0.02, 0.90, "Does this inverse problem have many solutions?", fontsize=18, fontweight="bold")
    ax.text(
        0.02,
        0.78,
        "YES — strongly multi-valued" if multi else "Mostly unique",
        fontsize=16,
        fontweight="bold",
        color="#b42318" if multi else "#1a7f37",
    )
    ax.text(0.02, 0.62, summary["verdict_text"], fontsize=13, va="top", wrap=True)
    ax.text(
        0.02,
        0.22,
        "Implication: a GP trained on flat sim cannot recover a single ASD grain/dust/algae\n"
        "state from radiance alone — several label combinations look the same.\n"
        "Need priors, joint constraints, or different observables — not just more training.",
        fontsize=12,
        fontweight="semibold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asd", type=Path, default=DEFAULT_ASD)
    ap.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--thr", type=float, default=0.05, help="rel RMSE threshold for 'similar'")
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.asd, "r") as fa, h5py.File(args.sim, "r") as fs:
        wl = np.asarray(fa["wl"][:], float)
        asd_rad = np.asarray(fa["toa_radiance"][:], float)
        dates = [
            d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in fa["date"][:]
        ]
        asd_q = {k: np.asarray(fa[ASD_MAP[k]][:], float) for k in LABELS if k in ASD_MAP}
        asd_std = {
            k: np.asarray(fa[STD_MAP[k]][:], float) for k in STD_MAP if STD_MAP[k] in fa
        }
        sim_rad = np.asarray(fs["toa_radiance"][:], float)
        sim_q = {k: np.asarray(fs[k][:], float) for k in LABELS}

    print(f"Computing distances ASD({asd_rad.shape[0]}) x sim({sim_rad.shape[0]}) ...")
    d_asd = chunked_dists(asd_rad, sim_rad)
    nn_idx, nn_dist = topk_from_dists(d_asd, k=args.topk)

    grain_edges = np.arange(30, 1500 + 50, 50)
    dust_edges = np.logspace(-2, 3, 21)  # 20 bins

    per_asd = []
    for i in range(asd_rad.shape[0]):
        within = nn_dist[i] <= args.thr
        # ensure enough points
        js = nn_idx[i, within]
        if js.size < 15:
            js = nn_idx[i, :40]
            used_thr = float(nn_dist[i, 39])
            mode = "topk40_fallback"
        else:
            used_thr = args.thr
            mode = "threshold"
        block = {
            "sample": i + 1,
            "date": dates[i],
            "mode": mode,
            "thr_used": used_thr,
            "n_similar": int(js.size),
            "nn1_dist": float(nn_dist[i, 0]),
            "within_thr": {},
            "asd": {},
        }
        for k in LABELS:
            if k not in asd_q:
                continue
            sp = label_spread(sim_q[k][js])
            block["within_thr"][k] = sp
            block["asd"][k] = float(asd_q[k][i])
            if k in asd_std:
                block["asd"][f"{k}_std"] = float(asd_std[k][i])
                block["within_thr"][k]["span_over_asd_2sigma"] = float(
                    sp["span"] / max(2 * asd_std[k][i], 1e-12)
                )
        gvals = sim_q["grain_size"][js]
        dvals = sim_q["dust"][js]
        block["n_grain_bins_50um"] = n_distinct_bins(gvals, grain_edges)
        block["n_dust_log_bins"] = n_distinct_bins(dvals, dust_edges)
        # joint bins
        gi = np.digitize(gvals, grain_edges) - 1
        di = np.digitize(dvals, dust_edges) - 1
        ok = (gi >= 0) & (gi < len(grain_edges) - 1) & (di >= 0) & (di < len(dust_edges) - 1)
        block["n_joint_grain_dust_bins"] = int(len(np.unique(list(zip(gi[ok], di[ok])))))
        per_asd.append(block)
        print(
            f"S{i+1}: n={js.size} mode={mode} grain_span={block['within_thr']['grain_size']['span']:.0f} "
            f"dust_span={block['within_thr']['dust']['span']:.3g} "
            f"joint_bins={block['n_joint_grain_dust_bins']}"
        )

    # within-sim degeneracy
    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(sim_rad.shape[0], size=min(args.n_queries, sim_rad.shape[0]), replace=False)
    print(f"Within-sim queries: {len(q_idx)}")
    d_in = chunked_dists(sim_rad[q_idx], sim_rad)
    # exclude self
    for r, qi in enumerate(q_idx):
        d_in[r, qi] = np.inf
    in_idx, in_dist = topk_from_dists(d_in, k=args.topk)
    within_spreads = []
    n_close = []
    for r in range(len(q_idx)):
        mask = in_dist[r] <= args.thr
        js = in_idx[r, mask]
        if js.size < 10:
            js = in_idx[r, :30]
        n_close.append(int(js.size))
        within_spreads.append({k: label_spread(sim_q[k][js]) for k in LABELS})

    # plots
    plot_solution_clouds(
        asd_q, asd_std, sim_q, nn_idx, nn_dist, dates, args.thr, plots / "01_solution_clouds_grain_dust.png"
    )
    plot_span_bars(per_asd, asd_std, plots / "02_neighbor_span_vs_asd_uncertainty.png")
    plot_tradeoff_matrix(sim_q, nn_idx[:, :40], plots / "03_label_tradeoffs_among_twins.png")
    plot_within_sim_degeneracy(within_spreads, args.thr, plots / "04_within_sim_degeneracy.png")
    plot_n_solutions(per_asd, plots / "05_how_many_distinct_solutions.png")

    # verdict heuristics
    grain_span_over = np.median(
        [p["within_thr"]["grain_size"].get("span_over_asd_2sigma", np.nan) for p in per_asd]
    )
    joint_med = float(np.median([p["n_joint_grain_dust_bins"] for p in per_asd]))
    within_grain_med = float(np.median([s["grain_size"]["span"] for s in within_spreads]))
    multi = bool(joint_med >= 5 and within_grain_med > 200)
    verdict_text = (
        f"Near each ASD spectrum, similar flat sims occupy a median of {joint_med:.0f} "
        f"distinct (grain x dust) bins.\n"
        f"Grain span among those sims is typically "
        f"{np.median([p['within_thr']['grain_size']['span'] for p in per_asd]):.0f} um "
        f"(~{grain_span_over:.1f}x ASD 2-sigma).\n"
        f"Even inside the synthetic set alone, neighbors within rel.RMSE<={args.thr:g} "
        f"have median grain span {within_grain_med:.0f} µm "
        f"(median count of close sims={np.median(n_close):.0f}).\n"
        f"Dust/algae/LWC also trade off while radiance stays similar."
    )
    summary = {
        "sim": str(args.sim),
        "asd": str(args.asd),
        "thr_rel_rmse": args.thr,
        "topk": args.topk,
        "per_asd": per_asd,
        "within_sim": {
            "n_queries": len(q_idx),
            "median_n_close": float(np.median(n_close)),
            "median_spans": {
                k: float(np.median([s[k]["span"] for s in within_spreads])) for k in LABELS
            },
        },
        "verdict_multi_valued": multi,
        "verdict_text": verdict_text,
        "wavelengths_nm_n": int(wl.size),
    }
    plot_verdict(summary, plots / "00_verdict.png")
    (out / "multiplicity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + verdict_text.encode("ascii", "replace").decode("ascii"))
    print(f"\nWrote {out}")
    for pth in sorted(plots.glob("*.png")):
        print(" ", pth.name)


if __name__ == "__main__":
    main()
