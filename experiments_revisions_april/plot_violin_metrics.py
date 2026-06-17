"""
Violin plots for RFF batch results under RESULTS_ROOT.

Scans gp_*.json under each example subdir, groups by dimensions and n_train (rows),
noise level (color-coded violins), and plots a grid: rows = increasing n_train,
columns = RRMSE, NIS, noise_std, Total_Time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpplus.utils.metrics_functions import analyze_metrics

# ---------------------------------------------------------------------------
# User configuration (match experiments_RFF/run_rff_experiments.py layout)
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("experiments_RFF/results/rff_batch")
PLOT_OUTPUT_DIR = RESULTS_ROOT / "plots"

EXAMPLES = {
    "ackley": {
        "subdir": "ackley",
        "label": "Ackley RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "wing_sf": {
        "subdir": "wing_s0",
        "label": "Wing s0 RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_abs_x": {
        "subdir": "tabpfn1d_abs_x",
        "label": "TabPFN1D abs_x RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_abs_x_ood": {
        "subdir": "tabpfn1d_abs_x_ood",
        "label": "TabPFN1D abs_x RFF OOD",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_sin_2pi_x": {
        "subdir": "tabpfn1d_sin_2pi_x",
        "label": "TabPFN1D sin_2pi_x RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_sin_2pi_x_ood": {
        "subdir": "tabpfn1d_sin_2pi_x_ood",
        "label": "TabPFN1D sin_2pi_x RFF OOD",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_step": {
        "subdir": "tabpfn1d_step",
        "label": "TabPFN1D step RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_step_ood": {
        "subdir": "tabpfn1d_step_ood",
        "label": "TabPFN1D step RFF OOD",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_x_squared": {
        "subdir": "tabpfn1d_x_squared",
        "label": "TabPFN1D x_squared RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_x_squared_ood": {
        "subdir": "tabpfn1d_x_squared_ood",
        "label": "TabPFN1D x_squared RFF OOD",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_linear_homoscedastic": {
        "subdir": "tabpfn1d_linear_homoscedastic",
        "label": "TabPFN1D linear_homoscedastic RFF",
        "noise_levels": [0.0, 0.005, 0.05],
    },
    "tabpfn1d_linear_homoscedastic_ood": {
        "subdir": "tabpfn1d_linear_homoscedastic_ood",
        "label": "TabPFN1D linear_homoscedastic RFF OOD",
        "noise_levels": [0.0, 0.005, 0.05],
    },
}

METRICS = ["RRMSE", "NIS", "noise_std", "Total_Time"]
NOISE_ROUND_DECIMALS = 6

NOISE_COLORS: dict[float, str] = {
    0.0: "#FFA500",
    0.005: "#90EE90",
    0.05: "#FFB6C1",
}
DEFAULT_NOISE_COLOR = "#888888"


def _noise_key(record: dict) -> float | None:
    for key in ("noise_test", "noise_train"):
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, list):
                val = val[0] if val else None
            if val is not None:
                return round(float(val), NOISE_ROUND_DECIMALS)
    return None


def _noise_color(noise: float) -> str:
    nk = round(float(noise), NOISE_ROUND_DECIMALS)
    if nk in NOISE_COLORS:
        return NOISE_COLORS[nk]
    print(f"[WARN] No color configured for noise={noise}; using grey.")
    return DEFAULT_NOISE_COLOR


def _dimensions_key(record: dict) -> int | None:
    for key in ("dimensions", "input_dim"):
        if key in record and record[key] is not None:
            return int(record[key])
    return None


def _dim_suffix(dimensions: int | None) -> str:
    return f"_D{dimensions}" if dimensions is not None else ""


def _load_gp_jsons(example_dir: Path) -> list[dict]:
    records = []
    for path in sorted(example_dir.glob("**/gp_*.json")):
        if path.name == "manifest.json":
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["_source_file"] = str(path)
                data["_model"] = "GP"
                records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {path}: {exc}")
    return records


def _extract_tabpfn_records(records: list[dict]) -> list[dict]:
    """Build flat TabPFN metric records from nested ``tabpfn_metrics`` in GP JSONs."""
    pfn_records: list[dict] = []
    for rec in records:
        pfn = rec.get("tabpfn_metrics")
        if not isinstance(pfn, dict):
            continue
        pfn_rec = dict(pfn)
        pfn_rec["_model"] = "TabPFN"
        pfn_rec["_source_file"] = rec.get("_source_file", "")
        for key in ("noise_test", "noise_train", "n_train", "dimensions", "train_size", "title"):
            if key in rec and key not in pfn_rec:
                pfn_rec[key] = rec[key]
        pfn_records.append(pfn_rec)
    return pfn_records


def _split_by_dimensions(records: list[dict]) -> dict[int | None, list[dict]]:
    buckets: dict[int | None, list[dict]] = {}
    for rec in records:
        d = _dimensions_key(rec)
        buckets.setdefault(d, []).append(rec)
    return buckets


def _group_by_train_and_noise(
    records: list[dict],
    noise_levels: list[float],
) -> dict[int, dict[float, list[dict]]]:
    targets = [round(float(n), NOISE_ROUND_DECIMALS) for n in noise_levels]
    grouped: dict[int, dict[float, list[dict]]] = {}

    for rec in records:
        nk = _noise_key(rec)
        if nk is None:
            continue
        if nk not in targets:
            print(
                f"[WARN] noise={nk} not in configured noise_levels; "
                f"skipping {rec.get('_source_file', '')}"
            )
            continue
        if rec.get("n_train") is None:
            print(f"[WARN] Missing n_train; skipping {rec.get('_source_file', '')}")
            continue

        n_train = int(rec["n_train"])
        if n_train not in grouped:
            grouped[n_train] = {t: [] for t in targets}
        grouped[n_train][nk].append(rec)

    return {k: grouped[k] for k in sorted(grouped.keys())}


def _format_median_label(metric: str, value: float) -> str:
    if metric == "Total_Time":
        return f"median={value:.2f}s"
    if metric == "NIS":
        return f"median={value:.4f}"
    if metric == "noise_std":
        return f"median={value:.6f}"
    return f"median={value:.6f}"


def _row_label(n_train: int, dimensions: int | None, sample_record: dict | None) -> str:
    base = f"n_train={n_train}"
    if sample_record and "dimensions" in sample_record and dimensions and dimensions > 0:
        train_size = n_train // dimensions
        return f"{base}\n{train_size}Dn"
    return base


def _metric_arrays(
    runs_by_noise: dict[float, list[dict]],
    noise_levels: list[float],
    metric: str,
) -> tuple[list[np.ndarray], list[str], int]:
    data: list[np.ndarray] = []
    short_labels: list[str] = []
    for noise in noise_levels:
        nk = round(float(noise), NOISE_ROUND_DECIMALS)
        runs = runs_by_noise.get(nk, [])
        arr = [
            float(d[metric])
            for d in runs
            if isinstance(d, dict) and metric in d and d[metric] is not None
        ]
        data.append(np.array(arr, dtype=float) if arr else np.array([], dtype=float))
        short_labels.append(f"{noise:g}")
    counts = [len(a) for a in data if len(a) > 0]
    n_seeds = min(counts) if counts else 0
    return data, short_labels, n_seeds


def _create_violin_plot(
    ax,
    data: list[np.ndarray],
    metric: str,
    noise_levels: list[float],
    colors: list[str],
    n_seeds: int,
    *,
    show_stat_legend: bool = False,
) -> None:
    """Violin plot with noise-colored bodies, blue mean and red median lines."""
    if not data or all(arr.size == 0 for arr in data):
        ax.set_visible(False)
        return

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        color = colors[i] if i < len(colors) else DEFAULT_NOISE_COLOR
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    tick_labels: list[str] = []
    for i, (noise, arr) in enumerate(zip(noise_levels, data), start=1):
        if arr.size == 0:
            tick_labels.append(f"{noise:g}\n—")
            continue
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        ax.hlines(mean_v, i - 0.25, i + 0.25, colors="blue", linewidth=2)
        ax.hlines(med_v, i - 0.25, i + 0.25, colors="red", linewidth=2)
        tick_labels.append(f"{noise:g}\n{_format_median_label(metric, med_v)}")

    if show_stat_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color="blue", lw=2, label="Mean"),
                Line2D([0], [0], color="red", lw=2, label="Median"),
            ],
            loc="upper right",
            frameon=False,
            fontsize=8,
        )

    ax.set_xticks(np.arange(1, len(noise_levels) + 1))
    ax.set_xticklabels(tick_labels, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_ha("center")
    if metric == "Total_Time":
        ylabel = "Total_Time (s)"
    elif metric == "noise_std":
        ylabel = "noise_std (learned)"
    else:
        ylabel = metric
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (n={n_seeds})", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _create_dual_violin_plot(
    ax,
    gp_data: list[np.ndarray],
    pfn_data: list[np.ndarray],
    metric: str,
    noise_levels: list[float],
    colors: list[str],
    n_seeds_gp: int,
    n_seeds_pfn: int,
    *,
    show_stat_legend: bool = False,
) -> None:
    """Side-by-side GP vs TabPFN violins at each noise level."""
    if (not gp_data or all(a.size == 0 for a in gp_data)) and (
        not pfn_data or all(a.size == 0 for a in pfn_data)
    ):
        ax.set_visible(False)
        return

    width = 0.35
    tick_labels: list[str] = []
    for i, noise in enumerate(noise_levels, start=1):
        gp_arr = gp_data[i - 1] if i - 1 < len(gp_data) else np.array([], dtype=float)
        pfn_arr = pfn_data[i - 1] if i - 1 < len(pfn_data) else np.array([], dtype=float)
        color = colors[i - 1] if i - 1 < len(colors) else DEFAULT_NOISE_COLOR

        if gp_arr.size > 0:
            parts = ax.violinplot([gp_arr], positions=[i - width / 2], widths=width, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.55)
            ax.hlines(float(np.mean(gp_arr)), i - width / 2 - 0.12, i - width / 2 + 0.12, colors="blue", linewidth=1.5)
            ax.hlines(float(np.median(gp_arr)), i - width / 2 - 0.12, i - width / 2 + 0.12, colors="red", linewidth=1.5)

        if pfn_arr.size > 0:
            parts = ax.violinplot([pfn_arr], positions=[i + width / 2], widths=width, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("#7570B3")
                pc.set_alpha=0.35
                pc.set_hatch("//")
            ax.hlines(float(np.mean(pfn_arr)), i + width / 2 - 0.12, i + width / 2 + 0.12, colors="blue", linewidth=1.5)
            ax.hlines(float(np.median(pfn_arr)), i + width / 2 - 0.12, i + width / 2 + 0.12, colors="red", linewidth=1.5)

        med_gp = float(np.median(gp_arr)) if gp_arr.size > 0 else float("nan")
        med_pfn = float(np.median(pfn_arr)) if pfn_arr.size > 0 else float("nan")
        tick_labels.append(f"{noise:g}\nGP {med_gp:.4g}\nPFN {med_pfn:.4g}")

    if show_stat_legend:
        ax.legend(
            handles=[
                Patch(facecolor="0.85", edgecolor="black", alpha=0.55, label="GP"),
                Patch(facecolor="0.85", edgecolor="#7570B3", alpha=0.35, hatch="//", label="TabPFN"),
                Line2D([0], [0], color="blue", lw=1.5, label="Mean"),
                Line2D([0], [0], color="red", lw=1.5, label="Median"),
            ],
            loc="upper right",
            frameon=False,
            fontsize=7,
        )

    ax.set_xticks(np.arange(1, len(noise_levels) + 1))
    ax.set_xticklabels(tick_labels, fontsize=7)
    for tick in ax.get_xticklabels():
        tick.set_ha("center")
    if metric == "Total_Time":
        ylabel = "Total_Time (s)"
    elif metric == "noise_std":
        ylabel = "noise_std (learned)"
    else:
        ylabel = metric
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} GP vs TabPFN (n_gp={n_seeds_gp}, n_pfn={n_seeds_pfn})", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_metrics_grid_comparison(
    grouped_gp: dict[int, dict[float, list[dict]]],
    grouped_pfn: dict[int, dict[float, list[dict]]],
    noise_levels: list[float],
    label: str,
    save_dir: Path,
    dimensions: int | None,
) -> None:
    n_rows = len(grouped_gp)
    if n_rows == 0:
        return

    colors = [_noise_color(n) for n in noise_levels]
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.8 * n_rows), squeeze=False)

    for row_idx, (n_train, runs_by_noise_gp) in enumerate(grouped_gp.items()):
        runs_by_noise_pfn = grouped_pfn.get(n_train, {round(float(n), NOISE_ROUND_DECIMALS): [] for n in noise_levels})
        sample = next((r for runs in runs_by_noise_gp.values() for r in runs), None)
        row_title = _row_label(n_train, dimensions, sample)

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            gp_data, _, n_seeds_gp = _metric_arrays(runs_by_noise_gp, noise_levels, metric)
            pfn_data, _, n_seeds_pfn = _metric_arrays(runs_by_noise_pfn, noise_levels, metric)
            _create_dual_violin_plot(
                ax,
                gp_data,
                pfn_data,
                metric,
                noise_levels,
                colors,
                n_seeds_gp,
                n_seeds_pfn,
                show_stat_legend=(row_idx == 0 and col_idx == 0),
            )
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n{ax.get_ylabel()}", fontsize=9)

    dim_part = f" (D={dimensions})" if dimensions is not None else ""
    fig.suptitle(f"{label} — GP vs TabPFN{dim_part}", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    save_dir.mkdir(parents=True, exist_ok=True)
    safe_title = label.replace(" ", "_")
    out_path = save_dir / f"metrics_gp_vs_tabpfn_{safe_title}{_dim_suffix(dimensions)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved {out_path}")
    plt.close(fig)


def _add_figure_legends(fig, noise_levels: list[float]) -> None:
    noise_handles = [
        Patch(
            facecolor=_noise_color(n),
            edgecolor="black",
            alpha=0.7,
            label=f"noise={n:g}",
        )
        for n in noise_levels
    ]
    stat_handles = [
        Line2D([0], [0], color="blue", lw=2, label="Mean"),
        Line2D([0], [0], color="red", lw=2, label="Median"),
    ]
    fig.legend(
        handles=noise_handles + stat_handles,
        loc="lower center",
        ncol=len(noise_levels) + 2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )


def _plot_metrics_grid(
    grouped_by_train: dict[int, dict[float, list[dict]]],
    noise_levels: list[float],
    label: str,
    save_dir: Path,
    dimensions: int | None,
) -> None:
    n_rows = len(grouped_by_train)
    if n_rows == 0:
        return

    colors = [_noise_color(n) for n in noise_levels]
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.8 * n_rows), squeeze=False)

    for row_idx, (n_train, runs_by_noise) in enumerate(grouped_by_train.items()):
        sample = next(
            (r for runs in runs_by_noise.values() for r in runs),
            None,
        )
        row_title = _row_label(n_train, dimensions, sample)

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            data, _, n_seeds = _metric_arrays(runs_by_noise, noise_levels, metric)
            show_stat = row_idx == 0 and col_idx == 0
            _create_violin_plot(
                ax,
                data,
                metric,
                noise_levels,
                colors,
                n_seeds,
                show_stat_legend=show_stat,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n{ax.get_ylabel()}", fontsize=9)

    dim_part = f" (D={dimensions})" if dimensions is not None else ""
    fig.suptitle(f"{label}{dim_part}", fontsize=12)
    _add_figure_legends(fig, noise_levels)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    save_dir.mkdir(parents=True, exist_ok=True)
    safe_title = label.replace(" ", "_")
    out_path = save_dir / f"metrics_combined_{safe_title}{_dim_suffix(dimensions)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved {out_path}")
    plt.close(fig)


def _plot_single_metric_grid(
    grouped_by_train: dict[int, dict[float, list[dict]]],
    noise_levels: list[float],
    metric: str,
    label: str,
    save_dir: Path,
    dimensions: int | None,
) -> None:
    n_rows = len(grouped_by_train)
    if n_rows == 0:
        return

    colors = [_noise_color(n) for n in noise_levels]
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    has_data = False
    for row_idx, (n_train, runs_by_noise) in enumerate(grouped_by_train.items()):
        ax = axes_flat[row_idx]
        data, _, n_seeds = _metric_arrays(runs_by_noise, noise_levels, metric)
        if all(arr.size == 0 for arr in data):
            ax.set_visible(False)
            continue
        has_data = True
        sample = next((r for runs in runs_by_noise.values() for r in runs), None)
        _create_violin_plot(
            ax,
            data,
            metric,
            noise_levels,
            colors,
            n_seeds,
            show_stat_legend=(row_idx == 0),
        )
        row_title = _row_label(n_train, dimensions, sample)
        if metric == "Total_Time":
            ylabel = "Total_Time (s)"
        elif metric == "noise_std":
            ylabel = "noise_std (learned)"
        else:
            ylabel = metric
        ax.set_ylabel(f"{row_title}\n{ylabel}", fontsize=9)

    if not has_data:
        plt.close(fig)
        print(f"[WARN] No data for {metric} in {label}; skipping plot.")
        return

    dim_part = f" (D={dimensions})" if dimensions is not None else ""
    fig.suptitle(f"{label}{dim_part} — {metric}", fontsize=12)
    _add_figure_legends(fig, noise_levels)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    save_dir.mkdir(parents=True, exist_ok=True)
    safe_title = label.replace(" ", "_")
    fname = "cost" if metric == "Total_Time" else metric.lower()
    out_path = save_dir / f"{fname}_{safe_title}{_dim_suffix(dimensions)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved {out_path}")
    plt.close(fig)


def _print_median_table(
    example_label: str,
    grouped_by_train: dict[int, dict[float, list[dict]]],
    noise_levels: list[float],
    dimensions: int | None,
) -> None:
    dim_part = f" D={dimensions}" if dimensions is not None else ""
    print(f"\nMedian summary: {example_label}{dim_part}")
    metric_cols = "  ".join(f"{m:>12}" for m in METRICS)
    header = f"{'n_train':>10}  {'noise':>8}  {metric_cols}  {'n':>6}"
    print(header)
    for n_train, runs_by_noise in grouped_by_train.items():
        for noise in noise_levels:
            nk = round(float(noise), NOISE_ROUND_DECIMALS)
            runs = runs_by_noise.get(nk, [])
            if not runs:
                dashes = "  ".join(f"{'—':>12}" for _ in METRICS)
                print(f"{n_train:>10}  {noise:>8}  {dashes}  {0:>6}")
                continue
            medians = {}
            for metric in METRICS:
                vals = [float(r[metric]) for r in runs if metric in r and r[metric] is not None]
                medians[metric] = float(np.median(vals)) if vals else float("nan")
            metric_vals = "  ".join(
                f"{medians[m]:>12.6f}" if m != "Total_Time" else f"{medians[m]:>12.2f}"
                for m in METRICS
            )
            print(f"{n_train:>10}  {noise:>8}  {metric_vals}  {len(runs):>6}")


def plot_example(
    name: str,
    cfg: dict,
    *,
    results_root: Path,
    plot_output_dir: Path,
) -> None:
    subdir = cfg["subdir"]
    label = cfg.get("label", name)
    noise_levels = cfg.get("noise_levels", [])

    example_dir = results_root / subdir
    if not example_dir.is_dir():
        print(f"[WARN] Missing example dir: {example_dir}")
        return

    records = _load_gp_jsons(example_dir)
    if not records:
        print(f"[WARN] No gp_*.json under {example_dir}")
        return

    pfn_records = _extract_tabpfn_records(records)
    has_tabpfn = len(pfn_records) > 0

    save_dir = plot_output_dir / subdir
    by_dimensions = _split_by_dimensions(records)

    for dimensions, dim_records in sorted(
        by_dimensions.items(),
        key=lambda x: (x[0] is None, x[0] if x[0] is not None else -1),
    ):
        grouped_by_train = _group_by_train_and_noise(dim_records, noise_levels)
        if not grouped_by_train:
            print(f"[WARN] No grouped data for {label}{_dim_suffix(dimensions)}")
            continue

        _print_median_table(label, grouped_by_train, noise_levels, dimensions)
        _plot_metrics_grid(grouped_by_train, noise_levels, label, save_dir, dimensions)

        if has_tabpfn:
            dim_pfn = [r for r in pfn_records if _dimensions_key(r) == dimensions]
            grouped_pfn = _group_by_train_and_noise(dim_pfn, noise_levels)
            if grouped_pfn:
                _plot_metrics_grid_comparison(
                    grouped_by_train, grouped_pfn, noise_levels, label, save_dir, dimensions
                )

        for metric in METRICS:
            _plot_single_metric_grid(
                grouped_by_train, noise_levels, metric, label, save_dir, dimensions
            )

        for n_train, runs_by_noise in grouped_by_train.items():
            for noise in noise_levels:
                nk = round(float(noise), NOISE_ROUND_DECIMALS)
                runs = runs_by_noise.get(nk, [])
                if runs:
                    analyze_metrics(
                        runs,
                        print_summary=True,
                        label=f"n_train={n_train}, noise={noise}",
                        title=label,
                    )


def plot_all(
    *,
    results_root: Path | None = None,
    plot_output_dir: Path | None = None,
    examples: dict | None = None,
) -> None:
    results_root = Path(results_root or RESULTS_ROOT)
    plot_output_dir = Path(plot_output_dir or PLOT_OUTPUT_DIR)
    examples = examples or EXAMPLES

    if not results_root.is_dir():
        raise FileNotFoundError(f"RESULTS_ROOT does not exist: {results_root}")

    plot_output_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in examples.items():
        print("=" * 60)
        print(f"Plotting: {name}")
        plot_example(name, cfg, results_root=results_root, plot_output_dir=plot_output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Violin plots for RFF batch results")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--examples", nargs="*", default=None)
    args = parser.parse_args()

    examples = EXAMPLES
    if args.examples:
        unknown = set(args.examples) - set(EXAMPLES)
        if unknown:
            raise SystemExit(f"Unknown examples: {unknown}. Available: {list(EXAMPLES)}")
        examples = {k: EXAMPLES[k] for k in args.examples}

    plot_all(
        results_root=Path(args.results_root) if args.results_root else None,
        plot_output_dir=Path(args.plot_dir) if args.plot_dir else None,
        examples=examples,
    )


if __name__ == "__main__":
    main()
