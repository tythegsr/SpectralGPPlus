"""
Violin plots and per-problem cost tables for April28 GP+ / TabPFN benchmark results.

Loads nested JSON from results_April28 folders and june16_pfnv2_only, filters to
10Dn/40Dn and noise 0.005/0.05, and produces:
  - GP+ vs TabPFN v2.5 vs TabPFN v2.0 violin grids (6 benchmarks)
  - GP+ vs TabPFN v2.5 vs TabPFN v2.0 for real datasets (Elevators, Pumadyn32)
  - Three-way comparison (GP+, GP+ PE, TabPFN v2.5) for Dixon-Price D20 and Rosenbrock D40
  - Per-problem PNG cost tables
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_revisions_april.plot_violin_metrics import (
    METRICS,
    NOISE_ROUND_DECIMALS,
    _dim_suffix,
    _metric_arrays,
    _print_median_table,
)

# April28 noise colors: 0.005 orange (left), 0.05 green (right)
APRIL28_NOISE_COLORS: dict[float, str] = {
    0.005: "#FFA500",
    0.05: "#90EE90",
}
DEFAULT_NOISE_COLOR = "#888888"

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
RESULTS_APRIL28 = _ROOT / "results_April28"
GP_ROOT = RESULTS_APRIL28 / "20_runs_Gaussian"
TABPFN_ROOT = RESULTS_APRIL28 / "20_runs_TabPFN"
TABPFN_V2_ROOT = _ROOT / "june16_pfnv2_only"
GP_REAL_ROOT = RESULTS_APRIL28 / "20_runs_real_datasets"
TABPFN_REAL_ROOT = RESULTS_APRIL28 / "20_runs_real_datasets_pfn"
PE_DIXON_ROOT = RESULTS_APRIL28 / "20_runs_logging_full_PE_constant_mean"
PE_ROSENBROCK_ROOT = RESULTS_APRIL28 / "20_runs_logging_full_PE_no_constant_mean"
DEFAULT_PLOT_DIR = RESULTS_APRIL28 / "plots"

NOISE_LEVELS = [0.005, 0.05]
TARGET_DNS = (10, 40)
REAL_DATASET_DNS = (10, 20, 40)
REAL_DATASET_NOISE = 0.0
REAL_DATASET_NOISE_LEVELS = [REAL_DATASET_NOISE]

# Plot typography (minimum 20pt throughout)
AXIS_LABEL_FS = 20
TICK_FS = 20
SUBPLOT_TITLE_FS = 20
ROW_LABEL_FS = 20
SUPTITLE_FS = 22
LEGEND_FS = 20
TABLE_FS = 20
TABLE_TITLE_FS = 20

METRICS_CORE = ["RRMSE", "NIS", "Total_Time"]

# Violin geometry: wide bodies with modest gap between model columns
VIOLIN_WIDTH = 0.38
NOISE_PAIR_OFFSET = 0.19  # center-to-center offset for the two noise violins
STAT_LINE_HALF_WIDTH = 0.16
MODEL_COL_SPACING = 0.98  # distance between model column centers

GP_VS_TABPFN_MODEL_ORDER = ["GP+", "TabPFN v2.5", "TabPFN v2.0"]
GP_VS_TABPFN_SUPTITLE = "GP+ vs TabPFN v2.5 vs TabPFN v2.0"
MODEL_COL_START = 0.55

BENCHMARK_PROBLEMS = {
    "ackley": {"subdir": "ackley", "label": "Ackley"},
    "rosenbrock": {"subdir": "rosenbrock", "label": "Rosenbrock"},
    "griewank": {"subdir": "griewank", "label": "Griewank"},
    "dixon_price": {"subdir": "dixon_price", "label": "Dixon-Price"},
    "wing": {"subdir": "wing", "label": "Wing Weight"},
    "buckling": {"subdir": "buckling", "label": "Buckling"},
}

THREE_WAY = {
    "dixon_price": {"dx": 20, "pe_root": PE_DIXON_ROOT, "label": "Dixon-Price"},
    "rosenbrock": {"dx": 40, "pe_root": PE_ROSENBROCK_ROOT, "label": "Rosenbrock"},
}

REAL_DATASET_PROBLEMS = {
    "elevators": {"subdir": "elevators", "label": "Elevators", "input_dim": 18},
    "pumadyn32": {"subdir": "pumadyn32", "label": "Pumadyn32", "input_dim": 32},
}

def _april28_noise_color(noise: float) -> str:
    nk = round(float(noise), NOISE_ROUND_DECIMALS)
    return APRIL28_NOISE_COLORS.get(nk, DEFAULT_NOISE_COLOR)


def _april28_row_label(dn: int) -> str:
    """Row label for train size per dimension, e.g. N=10D_x with subscript x."""
    return rf"$N={dn}D_{{x}}$"


def _april28_dim_title_part(dimensions: int | None) -> str:
    if dimensions is None:
        return ""
    return rf" ($D_x={dimensions}$)"


def _metric_ylabel(metric: str) -> str:
    if metric == "Total_Time":
        return "Cost (s)"
    if metric == "noise_std":
        return "noise_std (learned)"
    return metric


def _log_cost_tick_label(value: float, _pos) -> str:
    """Label only decade ticks as 10^n."""
    if value <= 0:
        return ""
    exponent = int(np.round(np.log10(value)))
    if abs(value - (10**exponent)) / value > 0.01:
        return ""
    return rf"$10^{{{exponent}}}$"


def _format_log_cost_axis(ax, values: list[float]) -> None:
    """Log y-axis: 10^n labels only; minor gridlines at 2–9 within each decade."""
    vals = np.asarray(values, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return

    log_min = float(np.log10(vals.min()))
    log_max = float(np.log10(vals.max()))
    span = log_max - log_min
    pad = max(0.2, 0.12 * span)
    if span < 0.35:
        pad = 0.45
    lo = 10 ** (log_min - pad)
    hi = 10 ** (log_max + pad)

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=15))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10), numticks=15)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(_log_cost_tick_label))
    ax.tick_params(axis="y", which="major", pad=2)
    ax.grid(axis="y", which="major", linestyle=":", alpha=0.55)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.25)


def _create_model_column_violin_plot(
    ax,
    model_series: list[tuple[str, list[np.ndarray]]],
    metric: str,
    noise_levels: list[float],
    *,
    log_y: bool = False,
) -> bool:
    """One x-axis column per model; within each column, noise violins left-to-right."""
    if not model_series:
        ax.set_visible(False)
        return False

    has_data = False
    n_models = len(model_series)
    log_values: list[float] = []
    if len(noise_levels) == 1:
        noise_offsets = [0.0]
    else:
        noise_offsets = [-NOISE_PAIR_OFFSET, NOISE_PAIR_OFFSET]
    col_centers = [MODEL_COL_START + model_idx * MODEL_COL_SPACING for model_idx in range(n_models)]

    for model_idx, (model_label, noise_arrays) in enumerate(model_series):
        col_center = col_centers[model_idx]
        for noise_idx, noise in enumerate(noise_levels):
            arr = (
                noise_arrays[noise_idx]
                if noise_idx < len(noise_arrays)
                else np.array([], dtype=float)
            )
            if arr.size == 0:
                continue
            if log_y:
                arr = arr[arr > 0]
                log_values.extend(arr.tolist())
            if arr.size == 0:
                continue
            has_data = True
            pos = col_center + (
                noise_offsets[noise_idx] if noise_idx < len(noise_offsets) else 0.0
            )
            color = _april28_noise_color(noise)
            parts = ax.violinplot(
                [arr], positions=[pos], widths=VIOLIN_WIDTH, showextrema=True
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            ax.hlines(
                float(np.mean(arr)),
                pos - STAT_LINE_HALF_WIDTH,
                pos + STAT_LINE_HALF_WIDTH,
                colors="blue",
                linewidth=2.0,
            )
            ax.hlines(
                float(np.median(arr)),
                pos - STAT_LINE_HALF_WIDTH,
                pos + STAT_LINE_HALF_WIDTH,
                colors="red",
                linewidth=2.0,
            )

    if not has_data:
        ax.set_visible(False)
        return False

    ax.set_xticks(col_centers)
    ax.set_xticklabels([name for name, _ in model_series], fontsize=TICK_FS)
    pad = VIOLIN_WIDTH / 2 + NOISE_PAIR_OFFSET + 0.08
    ax.set_xlim(col_centers[0] - pad, col_centers[-1] + pad)
    ax.set_title(_metric_ylabel(metric), fontsize=SUBPLOT_TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if log_y:
        _format_log_cost_axis(ax, log_values)
    else:
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    return True


def _add_april28_figure_legend(fig, noise_levels: list[float]) -> None:
    stat_handles = [
        Line2D([0], [0], color="blue", lw=3, label="Mean"),
        Line2D([0], [0], color="red", lw=3, label="Median"),
    ]
    if len(noise_levels) == 1 and round(float(noise_levels[0]), NOISE_ROUND_DECIMALS) == round(
        REAL_DATASET_NOISE, NOISE_ROUND_DECIMALS
    ):
        handles = stat_handles
        ncol = 2
    else:
        noise_handles = [
            Patch(
                facecolor=_april28_noise_color(n),
                edgecolor="black",
                alpha=0.7,
                label=f"noise={n:g}",
            )
            for n in noise_levels
        ]
        handles = noise_handles + stat_handles
        ncol = len(noise_levels) + 2
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=LEGEND_FS,
    )


def _plot_model_comparison_grid(
    grouped_by_model: dict[str, dict[int, dict[float, list[dict]]]],
    model_order: list[str],
    noise_levels: list[float],
    label: str,
    save_dir: Path,
    dimensions: int | None,
    *,
    filename_prefix: str,
    suptitle_suffix: str,
    metric_filter: list[str] | None = None,
    log_y: bool = False,
) -> None:
    """Grid: rows=Dn, cols=metrics; each subplot has model columns with noise violins."""
    metrics = metric_filter or METRICS

    all_dns: set[int] = set()
    for model_data in grouped_by_model.values():
        all_dns.update(model_data.keys())
    if not all_dns:
        return

    n_rows = len(sorted(all_dns))
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5 * n_cols, 6.8 * n_rows), squeeze=False)

    for row_idx, dn in enumerate(sorted(all_dns)):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            model_series: list[tuple[str, list[np.ndarray]]] = []

            for model_name in model_order:
                runs_by_noise = grouped_by_model.get(model_name, {}).get(
                    dn, {round(float(n), NOISE_ROUND_DECIMALS): [] for n in noise_levels}
                )
                arrays, _, _ = _metric_arrays(runs_by_noise, noise_levels, metric)
                model_series.append((model_name, arrays))

            _create_model_column_violin_plot(
                ax,
                model_series,
                metric,
                noise_levels,
                log_y=log_y and metric == "Total_Time",
            )

            if col_idx == 0:
                ax.set_ylabel(_april28_row_label(dn), fontsize=ROW_LABEL_FS)

    dim_part = _april28_dim_title_part(dimensions)
    fig.suptitle(f"{label}{dim_part} — {suptitle_suffix}", fontsize=SUPTITLE_FS)
    _add_april28_figure_legend(fig, noise_levels)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    save_dir.mkdir(parents=True, exist_ok=True)
    safe_title = label.replace(" ", "_").replace("-", "")
    out_path = save_dir / f"{filename_prefix}_{safe_title}{_dim_suffix(dimensions)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved {out_path}")
    plt.close(fig)


def _plot_comparison_metric_grids(
    grouped_by_model: dict[str, dict[int, dict[float, list[dict]]]],
    model_order: list[str],
    noise_levels: list[float],
    label: str,
    save_dir: Path,
    dimensions: int | None,
    *,
    metrics_prefix: str,
    suptitle_suffix: str,
) -> None:
    """Full 4-metric grid plus 3-metric grid without noise_std."""
    _plot_model_comparison_grid(
        grouped_by_model,
        model_order,
        noise_levels,
        label,
        save_dir,
        dimensions,
        filename_prefix=metrics_prefix,
        suptitle_suffix=suptitle_suffix,
        metric_filter=list(METRICS),
    )
    _plot_model_comparison_grid(
        grouped_by_model,
        model_order,
        noise_levels,
        label,
        save_dir,
        dimensions,
        filename_prefix=f"{metrics_prefix}_no_noisestd",
        suptitle_suffix=suptitle_suffix,
        metric_filter=METRICS_CORE,
        log_y=True,
    )
    _plot_model_comparison_grid(
        grouped_by_model,
        model_order,
        noise_levels,
        label,
        save_dir,
        dimensions,
        filename_prefix="cost",
        suptitle_suffix="Cost (s)",
        metric_filter=["Total_Time"],
        log_y=True,
    )

RE_BENCHMARK = re.compile(
    r"^(?:gp|pfn)_(?P<title>[A-Za-z]+)_(?P<dx>\d+)Dx_(?P<dn>\d+)Dn_.*"
    r"noiseTest(?P<noise>[\d.]+)_noiseTrain[\d.]+_x\d+\.json$"
)
RE_SF = re.compile(
    r"^(?:gp|pfn)_(?P<title>wing|buckling)_SF_(?P<dn>\d+)Dn_.*"
    r"noiseTest(?P<noise>[\d.]+)_noiseTrain[\d.]+_x\d+\.json$"
)
RE_REAL_DATASET = re.compile(
    r"^(?:gp|pfn)_(?P<title>elevators|puma32H)_(?P<dx>\d+)Dx_(?P<dn>\d+)Dn_.*_x\d+\.json$"
)

TITLE_TO_PROBLEM = {
    "Ackley": "ackley",
    "Rosenbrock": "rosenbrock",
    "Griewank": "griewank",
    "DixonPrice": "dixon_price",
    "wing": "wing",
    "buckling": "buckling",
}

REAL_TITLE_TO_PROBLEM = {
    "elevators": "elevators",
    "puma32H": "pumadyn32",
}


def _parse_filename(name: str) -> dict | None:
    m = RE_BENCHMARK.match(name)
    if m:
        title = m.group("title")
        problem = TITLE_TO_PROBLEM.get(title)
        if problem is None:
            return None
        dx = int(m.group("dx"))
        dn = int(m.group("dn"))
        return {
            "problem": problem,
            "input_dim": dx,
            "train_size": dn,
            "n_train": dx * dn,
            "noise": round(float(m.group("noise")), NOISE_ROUND_DECIMALS),
        }
    m = RE_SF.match(name)
    if m:
        title = m.group("title")
        dn = int(m.group("dn"))
        return {
            "problem": title,
            "input_dim": None,
            "train_size": dn,
            "n_train": dn,
            "noise": round(float(m.group("noise")), NOISE_ROUND_DECIMALS),
        }
    return None


def _parse_real_dataset_filename(name: str) -> dict | None:
    m = RE_REAL_DATASET.match(name)
    if not m:
        return None
    title = m.group("title")
    problem = REAL_TITLE_TO_PROBLEM.get(title)
    if problem is None:
        return None
    dx = int(m.group("dx"))
    dn = int(m.group("dn"))
    return {
        "problem": problem,
        "input_dim": dx,
        "train_size": dn,
        "n_train": dx * dn,
        "noise": round(REAL_DATASET_NOISE, NOISE_ROUND_DECIMALS),
    }


def _normalize_run_metrics(run: dict) -> dict:
    out = dict(run)
    if "Training_Time" in out and "Train_Time" not in out:
        out["Train_Time"] = out["Training_Time"]
    return out


def _load_april28_records(
    root: Path,
    model_label: str,
    *,
    file_glob: str,
    problems: set[str] | None = None,
) -> list[dict]:
    """Load per-run flat records from April28 nested JSON files."""
    seen_stems: set[str] = set()
    records: list[dict] = []

    for path in sorted(root.glob(f"**/{file_glob}")):
        if "trainer_analysis" in path.parts or "GP_Trainer_Analysis" in path.name:
            continue
        meta = _parse_filename(path.name)
        if meta is None:
            continue
        if problems is not None and meta["problem"] not in problems:
            continue
        if meta["train_size"] not in TARGET_DNS:
            continue
        if meta["noise"] not in [round(n, NOISE_ROUND_DECIMALS) for n in NOISE_LEVELS]:
            continue

        stem_key = path.name
        if stem_key in seen_stems:
            continue
        seen_stems.add(stem_key)

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        if path.name.startswith("pfn_"):
            metrics_list = data.get("tabpfn_data", {}).get("metrics", [])
        else:
            metrics_list = data.get("gp_data", {}).get("metrics", [])

        if not metrics_list:
            print(f"[WARN] No metrics in {path}")
            continue

        for run in metrics_list:
            if not isinstance(run, dict):
                continue
            flat = _normalize_run_metrics(run)
            flat.update(
                {
                    "problem": meta["problem"],
                    "model": model_label,
                    "input_dim": meta["input_dim"],
                    "dimensions": meta["input_dim"],
                    "train_size": meta["train_size"],
                    "n_train": meta["n_train"],
                    "noise_test": meta["noise"],
                    "noise_train": meta["noise"],
                    "_source_file": str(path),
                }
            )
            records.append(flat)

    return records


def _load_real_dataset_records(
    root: Path,
    model_label: str,
    *,
    file_glob: str,
    problems: set[str] | None = None,
) -> list[dict]:
    """Load per-run flat records from real-dataset JSON files (no noise in filename)."""
    seen_stems: set[str] = set()
    records: list[dict] = []

    for path in sorted(root.glob(f"**/{file_glob}")):
        if "trainer_analysis" in path.parts or "GP_Trainer_Analysis" in path.name:
            continue
        meta = _parse_real_dataset_filename(path.name)
        if meta is None:
            continue
        if problems is not None and meta["problem"] not in problems:
            continue
        if meta["train_size"] not in REAL_DATASET_DNS:
            continue

        stem_key = path.name
        if stem_key in seen_stems:
            continue
        seen_stems.add(stem_key)

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        if path.name.startswith("pfn_"):
            metrics_list = data.get("tabpfn_data", {}).get("metrics", [])
        else:
            metrics_list = data.get("gp_data", {}).get("metrics", [])

        if not metrics_list:
            print(f"[WARN] No metrics in {path}")
            continue

        for run in metrics_list:
            if not isinstance(run, dict):
                continue
            flat = _normalize_run_metrics(run)
            flat.update(
                {
                    "problem": meta["problem"],
                    "model": model_label,
                    "input_dim": meta["input_dim"],
                    "dimensions": meta["input_dim"],
                    "train_size": meta["train_size"],
                    "n_train": meta["n_train"],
                    "noise_test": meta["noise"],
                    "noise_train": meta["noise"],
                    "_source_file": str(path),
                }
            )
            records.append(flat)

    return records


def _filter_records(
    records: list[dict],
    *,
    problem: str | None = None,
    input_dim: int | None = None,
    model: str | None = None,
) -> list[dict]:
    out = records
    if problem is not None:
        out = [r for r in out if r["problem"] == problem]
    if input_dim is not None:
        out = [r for r in out if r.get("input_dim") == input_dim]
    if model is not None:
        out = [r for r in out if r["model"] == model]
    return out


def _group_by_dn_and_noise(
    records: list[dict],
    noise_levels: list[float],
    *,
    target_dns: tuple[int, ...] = TARGET_DNS,
) -> dict[int, dict[float, list[dict]]]:
    """Group flat per-run records by train_size (Dn) and noise."""
    targets = [round(float(n), NOISE_ROUND_DECIMALS) for n in noise_levels]
    grouped: dict[int, dict[float, list[dict]]] = {}

    for rec in records:
        dn = rec.get("train_size")
        noise = rec.get("noise_test")
        if dn is None or noise is None:
            continue
        nk = round(float(noise), NOISE_ROUND_DECIMALS)
        if nk not in targets or int(dn) not in target_dns:
            continue
        dn = int(dn)
        if dn not in grouped:
            grouped[dn] = {t: [] for t in targets}
        grouped[dn][nk].append(rec)

    return {k: grouped[k] for k in sorted(grouped.keys())}


def _dn_to_n_train(dn: int, dimensions: int | None) -> int:
    if dimensions is None:
        return dn
    return dimensions * dn


def _grouped_dn_as_n_train(
    grouped_dn: dict[int, dict[float, list[dict]]],
    dimensions: int | None,
) -> dict[int, dict[float, list[dict]]]:
    """Convert Dn-keyed groups to n_train keys for reuse of violin helpers."""
    out: dict[int, dict[float, list[dict]]] = {}
    for dn, runs_by_noise in grouped_dn.items():
        n_train = _dn_to_n_train(dn, dimensions)
        out[n_train] = runs_by_noise
    return out


def _timing_stats(runs: list[dict]) -> dict[str, float]:
    def _med_std(key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in runs if key in r and r[key] is not None]
        if not vals:
            return float("nan"), float("nan")
        return float(np.median(vals)), float(np.std(vals))

    train_med, train_std = _med_std("Train_Time")
    pred_med, pred_std = _med_std("Prediction_Time")
    total_med, total_std = _med_std("Total_Time")
    return {
        "train_med": train_med,
        "train_std": train_std,
        "pred_med": pred_med,
        "pred_std": pred_std,
        "total_med": total_med,
        "total_std": total_std,
    }


def _fmt_med_std(med: float, std: float, *, decimals: int = 4) -> str:
    if np.isnan(med):
        return "—"
    if decimals == 2:
        return f"{med:.2f} ± {std:.2f}"
    return f"{med:.4f} ± {std:.4f}"


def _render_cost_table_png(
    rows: list[list[str]],
    columns: list[str],
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        print(f"[WARN] No cost table rows for {title}; skipping.")
        return

    n_rows = len(rows)
    fig_h = max(4.0, 0.65 * (n_rows + 2))
    fig_w = max(16.0, 1.8 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FS)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=TABLE_FS)
        elif row > 0 and rows[row - 1][-1] == "Combined":
            cell.set_facecolor("#FFF2CC")
        else:
            cell.set_text_props(fontsize=TABLE_FS)

    ax.set_title(title, fontsize=TABLE_TITLE_FS, fontweight="bold", pad=24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved {out_path}")
    plt.close(fig)


def _build_gp_vs_tabpfn_cost_rows(
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    problem: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    cfg = BENCHMARK_PROBLEMS[problem]
    label = cfg["label"]

    gp_prob = _filter_records(gp_records, problem=problem)
    pfn_v25_prob = _filter_records(pfn_v25_records, problem=problem)
    pfn_v2_prob = _filter_records(pfn_v2_records, problem=problem)

    dx_values = sorted(
        {
            r["input_dim"]
            for r in gp_prob + pfn_v25_prob + pfn_v2_prob
            if r.get("input_dim") is not None
        },
        key=lambda x: x,
    )
    if not dx_values:
        dx_values = [None]

    model_recs = [
        ("GP+", gp_prob),
        ("TabPFN v2.5", pfn_v25_prob),
        ("TabPFN v2.0", pfn_v2_prob),
    ]

    for dx in dx_values:
        for dn in TARGET_DNS:
            for noise in NOISE_LEVELS:
                nk = round(float(noise), NOISE_ROUND_DECIMALS)
                for model_label, recs in model_recs:
                    subset = [
                        r
                        for r in recs
                        if r["train_size"] == dn
                        and round(float(r["noise_test"]), NOISE_ROUND_DECIMALS) == nk
                        and (dx is None or r.get("input_dim") == dx)
                    ]
                    if not subset:
                        continue
                    stats = _timing_stats(subset)
                    n_train = subset[0]["n_train"]
                    dx_str = str(dx) if dx is not None else "—"
                    rows.append(
                        [
                            label,
                            dx_str,
                            str(dn),
                            str(n_train),
                            f"{noise:g}",
                            model_label,
                            _fmt_med_std(stats["train_med"], stats["train_std"]),
                            _fmt_med_std(stats["pred_med"], stats["pred_std"]),
                            _fmt_med_std(stats["total_med"], stats["total_std"], decimals=2),
                        ]
                    )
    return rows


def _build_real_dataset_cost_rows(
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    problem: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    cfg = REAL_DATASET_PROBLEMS[problem]
    label = cfg["label"]
    dx = cfg["input_dim"]

    gp_prob = _filter_records(gp_records, problem=problem)
    pfn_v25_prob = _filter_records(pfn_v25_records, problem=problem)
    pfn_v2_prob = _filter_records(pfn_v2_records, problem=problem)

    model_recs = [
        ("GP+", gp_prob),
        ("TabPFN v2.5", pfn_v25_prob),
        ("TabPFN v2.0", pfn_v2_prob),
    ]

    for dn in REAL_DATASET_DNS:
        for model_label, recs in model_recs:
            subset = [
                r
                for r in recs
                if r["train_size"] == dn and r.get("input_dim") == dx
            ]
            if not subset:
                continue
            stats = _timing_stats(subset)
            n_train = subset[0]["n_train"]
            rows.append(
                [
                    label,
                    str(dx),
                    str(dn),
                    str(n_train),
                    "—",
                    model_label,
                    _fmt_med_std(stats["train_med"], stats["train_std"]),
                    _fmt_med_std(stats["pred_med"], stats["pred_std"]),
                    _fmt_med_std(stats["total_med"], stats["total_std"], decimals=2),
                ]
            )
    return rows


def _build_three_way_cost_rows(
    gp_records: list[dict],
    pe_records: list[dict],
    pfn_records: list[dict],
    problem: str,
    dx: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    label = THREE_WAY[problem]["label"]

    for dn in TARGET_DNS:
        for noise in NOISE_LEVELS:
            nk = round(float(noise), NOISE_ROUND_DECIMALS)
            model_stats: dict[str, dict[str, float]] = {}
            n_train = None

            for model_label, recs in [
                ("GP+", _filter_records(gp_records, problem=problem, input_dim=dx)),
                ("GP+ (PE)", _filter_records(pe_records, problem=problem, input_dim=dx)),
                ("TabPFN v2.5", _filter_records(pfn_records, problem=problem, input_dim=dx)),
            ]:
                subset = [
                    r
                    for r in recs
                    if r["train_size"] == dn
                    and round(float(r["noise_test"]), NOISE_ROUND_DECIMALS) == nk
                ]
                if not subset:
                    continue
                n_train = subset[0]["n_train"]
                stats = _timing_stats(subset)
                model_stats[model_label] = stats
                rows.append(
                    [
                        label,
                        str(dx),
                        str(dn),
                        str(n_train),
                        f"{noise:g}",
                        model_label,
                        _fmt_med_std(stats["train_med"], stats["train_std"]),
                        _fmt_med_std(stats["pred_med"], stats["pred_std"]),
                        _fmt_med_std(stats["total_med"], stats["total_std"], decimals=2),
                    ]
                )

            if len(model_stats) >= 2:
                total_combined = sum(s["total_med"] for s in model_stats.values() if not np.isnan(s["total_med"]))
                rows.append(
                    [
                        label,
                        str(dx),
                        str(dn),
                        str(n_train) if n_train is not None else "—",
                        f"{noise:g}",
                        "Combined",
                        "—",
                        "—",
                        f"{total_combined:.2f}",
                    ]
                )
    return rows


COST_COLUMNS = [
    "Problem",
    "Dx",
    "Dn",
    "n_train",
    "Noise",
    "Model",
    "Train (med±std)",
    "Predict (med±std)",
    "Total (med±std)",
]


def _plot_three_way_grid(
    grouped_gp: dict[int, dict[float, list[dict]]],
    grouped_pe: dict[int, dict[float, list[dict]]],
    grouped_pfn: dict[int, dict[float, list[dict]]],
    noise_levels: list[float],
    label: str,
    save_dir: Path,
    dimensions: int,
) -> None:
    if not grouped_gp:
        print(f"[WARN] No three-way data for {label} D={dimensions}")
        return

    grouped_by_model = {
        "GP+": grouped_gp,
        "GP+ (PE)": grouped_pe,
        "TabPFN v2.5": grouped_pfn,
    }
    model_order = ["GP+", "GP+ (PE)", "TabPFN v2.5"]

    _plot_comparison_metric_grids(
        grouped_by_model,
        model_order,
        noise_levels,
        label,
        save_dir,
        dimensions,
        metrics_prefix="metrics_three_way",
        suptitle_suffix="GP+ vs GP+ (PE) vs TabPFN v2.5",
    )


def _plot_gp_vs_tabpfn_problem(
    problem: str,
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    plot_dir: Path,
) -> None:
    cfg = BENCHMARK_PROBLEMS[problem]
    label = cfg["label"]
    save_dir = plot_dir / "gp_vs_tabpfn" / problem

    gp_prob = _filter_records(gp_records, problem=problem)
    pfn_v25_prob = _filter_records(pfn_v25_records, problem=problem)
    pfn_v2_prob = _filter_records(pfn_v2_records, problem=problem)

    dx_values = sorted(
        {
            r["input_dim"]
            for r in gp_prob + pfn_v25_prob + pfn_v2_prob
            if r.get("input_dim") is not None
        },
        key=lambda x: x,
    )
    if not dx_values:
        dx_values = [None]

    for dx in dx_values:
        gp_dx = _filter_records(gp_prob, input_dim=dx) if dx is not None else gp_prob
        pfn_v25_dx = (
            _filter_records(pfn_v25_prob, input_dim=dx) if dx is not None else pfn_v25_prob
        )
        pfn_v2_dx = (
            _filter_records(pfn_v2_prob, input_dim=dx) if dx is not None else pfn_v2_prob
        )

        grouped_gp = _group_by_dn_and_noise(gp_dx, NOISE_LEVELS)
        grouped_pfn_v25 = _group_by_dn_and_noise(pfn_v25_dx, NOISE_LEVELS)
        grouped_pfn_v2 = _group_by_dn_and_noise(pfn_v2_dx, NOISE_LEVELS)
        if not grouped_gp and not grouped_pfn_v25 and not grouped_pfn_v2:
            continue

        grouped_nt_gp = _grouped_dn_as_n_train(grouped_gp, dx)

        if grouped_nt_gp:
            _print_median_table(label, grouped_nt_gp, NOISE_LEVELS, dx)

        all_dns = sorted(
            set(grouped_gp.keys()) | set(grouped_pfn_v25.keys()) | set(grouped_pfn_v2.keys())
        )
        if all_dns:
            grouped_by_model = {
                "GP+": {d: grouped_gp.get(d, {}) for d in all_dns},
                "TabPFN v2.5": {d: grouped_pfn_v25.get(d, {}) for d in all_dns},
                "TabPFN v2.0": {d: grouped_pfn_v2.get(d, {}) for d in all_dns},
            }
            _plot_comparison_metric_grids(
                grouped_by_model,
                GP_VS_TABPFN_MODEL_ORDER,
                NOISE_LEVELS,
                label,
                save_dir,
                dx,
                metrics_prefix="metrics_gp_vs_tabpfn",
                suptitle_suffix=GP_VS_TABPFN_SUPTITLE,
            )

    cost_rows = _build_gp_vs_tabpfn_cost_rows(
        gp_records, pfn_v25_records, pfn_v2_records, problem
    )
    safe_label = label.replace(" ", "_")
    _render_cost_table_png(
        cost_rows,
        COST_COLUMNS,
        f"{label} — Cost Table ({GP_VS_TABPFN_SUPTITLE})",
        save_dir / f"cost_table_{safe_label}.png",
        )


def _plot_real_dataset_problem(
    problem: str,
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    plot_dir: Path,
) -> None:
    cfg = REAL_DATASET_PROBLEMS[problem]
    label = cfg["label"]
    dx = cfg["input_dim"]
    save_dir = plot_dir / "real_datasets" / problem
    noise_levels = REAL_DATASET_NOISE_LEVELS

    gp_prob = _filter_records(gp_records, problem=problem, input_dim=dx)
    pfn_v25_prob = _filter_records(pfn_v25_records, problem=problem, input_dim=dx)
    pfn_v2_prob = _filter_records(pfn_v2_records, problem=problem, input_dim=dx)

    grouped_gp = _group_by_dn_and_noise(
        gp_prob, noise_levels, target_dns=REAL_DATASET_DNS
    )
    grouped_pfn_v25 = _group_by_dn_and_noise(
        pfn_v25_prob, noise_levels, target_dns=REAL_DATASET_DNS
    )
    grouped_pfn_v2 = _group_by_dn_and_noise(
        pfn_v2_prob, noise_levels, target_dns=REAL_DATASET_DNS
    )
    if not grouped_gp and not grouped_pfn_v25 and not grouped_pfn_v2:
        print(f"[WARN] No real-dataset data for {label}")
        return

    grouped_nt_gp = _grouped_dn_as_n_train(grouped_gp, dx)
    if grouped_nt_gp:
        _print_median_table(label, grouped_nt_gp, noise_levels, dx)

    all_dns = sorted(
        set(grouped_gp.keys()) | set(grouped_pfn_v25.keys()) | set(grouped_pfn_v2.keys())
    )
    grouped_by_model = {
        "GP+": {d: grouped_gp.get(d, {}) for d in all_dns},
        "TabPFN v2.5": {d: grouped_pfn_v25.get(d, {}) for d in all_dns},
        "TabPFN v2.0": {d: grouped_pfn_v2.get(d, {}) for d in all_dns},
    }
    _plot_comparison_metric_grids(
        grouped_by_model,
        GP_VS_TABPFN_MODEL_ORDER,
        noise_levels,
        label,
        save_dir,
        dx,
        metrics_prefix="metrics_gp_vs_tabpfn",
        suptitle_suffix=GP_VS_TABPFN_SUPTITLE,
    )

    cost_rows = _build_real_dataset_cost_rows(
        gp_records, pfn_v25_records, pfn_v2_records, problem
    )
    safe_label = label.replace(" ", "_")
    _render_cost_table_png(
        cost_rows,
        COST_COLUMNS,
        f"{label} — Cost Table ({GP_VS_TABPFN_SUPTITLE})",
        save_dir / f"cost_table_{safe_label}.png",
    )


def plot_real_datasets(
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    plot_dir: Path,
    problems: list[str] | None = None,
) -> None:
    problem_list = problems or list(REAL_DATASET_PROBLEMS.keys())
    for problem in problem_list:
        if problem not in REAL_DATASET_PROBLEMS:
            print(f"[WARN] Unknown real-dataset problem: {problem}")
            continue
        print("=" * 60)
        print(f"Real dataset: {problem}")
        _plot_real_dataset_problem(
            problem, gp_records, pfn_v25_records, pfn_v2_records, plot_dir
        )


def plot_gp_vs_tabpfn(
    gp_records: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    plot_dir: Path,
    problems: list[str] | None = None,
) -> None:
    problem_list = problems or list(BENCHMARK_PROBLEMS.keys())
    for problem in problem_list:
        print("=" * 60)
        print(f"GP vs TabPFN: {problem}")
        _plot_gp_vs_tabpfn_problem(
            problem, gp_records, pfn_v25_records, pfn_v2_records, plot_dir
        )


def plot_three_way(
    gp_records: list[dict],
    pe_dixon: list[dict],
    pe_rosenbrock: list[dict],
    pfn_records: list[dict],
    plot_dir: Path,
) -> None:
    pe_by_problem = {
        "dixon_price": pe_dixon,
        "rosenbrock": pe_rosenbrock,
    }

    for problem, cfg in THREE_WAY.items():
        print("=" * 60)
        print(f"Three-way: {problem} D={cfg['dx']}")
        dx = cfg["dx"]
        label = cfg["label"]
        save_dir = plot_dir / "three_way" / f"{problem}_D{dx}"

        gp_dx = _filter_records(gp_records, problem=problem, input_dim=dx)
        pe_dx = _filter_records(pe_by_problem[problem], problem=problem, input_dim=dx)
        pfn_dx = _filter_records(pfn_records, problem=problem, input_dim=dx)

        grouped_gp = _group_by_dn_and_noise(gp_dx, NOISE_LEVELS)
        grouped_pe = _group_by_dn_and_noise(pe_dx, NOISE_LEVELS)
        grouped_pfn = _group_by_dn_and_noise(pfn_dx, NOISE_LEVELS)

        if not grouped_gp:
            print(f"[WARN] No GP+ data for {label} D={dx}")
            continue

        _plot_three_way_grid(
            grouped_gp, grouped_pe, grouped_pfn, NOISE_LEVELS, label, save_dir, dx
        )

        cost_rows = _build_three_way_cost_rows(
            gp_records, pe_by_problem[problem], pfn_records, problem, dx
        )
        safe_label = label.replace(" ", "_")
        _render_cost_table_png(
            cost_rows,
            COST_COLUMNS,
            f"{label} D={dx} — Cost Table (GP+ vs GP+ PE vs TabPFN v2.5)",
            save_dir / f"cost_table_{safe_label}_D{dx}.png",
        )


def plot_cost_tables_only(
    gp_records: list[dict],
    pe_dixon: list[dict],
    pe_rosenbrock: list[dict],
    pfn_v25_records: list[dict],
    pfn_v2_records: list[dict],
    plot_dir: Path,
) -> None:
    for problem in BENCHMARK_PROBLEMS:
        cfg = BENCHMARK_PROBLEMS[problem]
        save_dir = plot_dir / "gp_vs_tabpfn" / problem
        cost_rows = _build_gp_vs_tabpfn_cost_rows(
            gp_records, pfn_v25_records, pfn_v2_records, problem
        )
        safe_label = cfg["label"].replace(" ", "_")
        _render_cost_table_png(
            cost_rows,
            COST_COLUMNS,
            f"{cfg['label']} — Cost Table ({GP_VS_TABPFN_SUPTITLE})",
            save_dir / f"cost_table_{safe_label}.png",
        )

    pe_by_problem = {"dixon_price": pe_dixon, "rosenbrock": pe_rosenbrock}
    for problem, cfg in THREE_WAY.items():
        dx = cfg["dx"]
        save_dir = plot_dir / "three_way" / f"{problem}_D{dx}"
        cost_rows = _build_three_way_cost_rows(
            gp_records, pe_by_problem[problem], pfn_v25_records, problem, dx
        )
        safe_label = cfg["label"].replace(" ", "_")
        _render_cost_table_png(
            cost_rows,
            COST_COLUMNS,
            f"{cfg['label']} D={dx} — Cost Table (GP+ vs GP+ PE vs TabPFN v2.5)",
            save_dir / f"cost_table_{safe_label}_D{dx}.png",
        )


def load_all_records(
    *,
    tabpfn_v2_root: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    problems = set(BENCHMARK_PROBLEMS.keys()) | set(THREE_WAY.keys())
    v2_root = tabpfn_v2_root or TABPFN_V2_ROOT

    gp_records = _load_april28_records(
        GP_ROOT, "GP+", file_glob="gp_*.json", problems=problems
    )
    pfn_v25_records = _load_april28_records(
        TABPFN_ROOT, "TabPFN v2.5", file_glob="pfn_*.json", problems=problems
    )
    pfn_v2_records: list[dict] = []
    if v2_root.is_dir():
        pfn_v2_records = _load_april28_records(
            v2_root, "TabPFN v2.0", file_glob="pfn_*.json", problems=problems
        )
    else:
        print(f"[WARN] TabPFN v2.0 root not found: {v2_root}")

    pe_dixon = _load_april28_records(
        PE_DIXON_ROOT, "GP+ (PE)", file_glob="gp_*.json", problems={"dixon_price"}
    )
    pe_rosenbrock = _load_april28_records(
        PE_ROSENBROCK_ROOT,
        "GP+ (PE)",
        file_glob="gp_*.json",
        problems={"rosenbrock"},
    )
    print(
        f"[INFO] Loaded {len(gp_records)} GP+ runs, "
        f"{len(pfn_v25_records)} TabPFN v2.5 runs, "
        f"{len(pfn_v2_records)} TabPFN v2.0 runs, "
        f"{len(pe_dixon)} PE Dixon runs, {len(pe_rosenbrock)} PE Rosenbrock runs"
    )
    return gp_records, pfn_v25_records, pfn_v2_records, pe_dixon, pe_rosenbrock


def load_real_dataset_records(
    *,
    tabpfn_v2_root: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    problems = set(REAL_DATASET_PROBLEMS.keys())
    v2_root = tabpfn_v2_root or TABPFN_V2_ROOT

    gp_records = _load_real_dataset_records(
        GP_REAL_ROOT, "GP+", file_glob="gp_*.json", problems=problems
    )
    pfn_v25_records = _load_real_dataset_records(
        TABPFN_REAL_ROOT, "TabPFN v2.5", file_glob="pfn_*.json", problems=problems
    )
    pfn_v2_records: list[dict] = []
    if v2_root.is_dir():
        pfn_v2_records = _load_real_dataset_records(
            v2_root, "TabPFN v2.0", file_glob="pfn_*.json", problems=problems
        )
    else:
        print(f"[WARN] TabPFN v2.0 root not found: {v2_root}")

    print(
        f"[INFO] Real datasets: {len(gp_records)} GP+ runs, "
        f"{len(pfn_v25_records)} TabPFN v2.5 runs, "
        f"{len(pfn_v2_records)} TabPFN v2.0 runs"
    )
    return gp_records, pfn_v25_records, pfn_v2_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="April28 GP+ / TabPFN violin plots and cost tables"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_DIR),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--tabpfn-v2-root",
        type=str,
        default=str(TABPFN_V2_ROOT),
        help="Root directory for TabPFN v2.0 pfn_*.json results",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["gp_vs_tabpfn", "real_datasets", "three_way", "cost_tables"],
        choices=["gp_vs_tabpfn", "real_datasets", "three_way", "cost_tables"],
        help="Which plot sections to generate (default: all)",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Subset of benchmark problems for gp_vs_tabpfn",
    )
    parser.add_argument(
        "--real-problems",
        nargs="*",
        default=None,
        help="Subset of real-dataset problems (elevators, pumadyn32)",
    )
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gp_records, pfn_v25_records, pfn_v2_records, pe_dixon, pe_rosenbrock = load_all_records(
        tabpfn_v2_root=Path(args.tabpfn_v2_root),
    )
    sections = set(args.sections)

    if "gp_vs_tabpfn" in sections:
        plot_gp_vs_tabpfn(
            gp_records, pfn_v25_records, pfn_v2_records, plot_dir, problems=args.problems
        )

    if "real_datasets" in sections:
        real_gp, real_pfn_v25, real_pfn_v2 = load_real_dataset_records(
            tabpfn_v2_root=Path(args.tabpfn_v2_root),
        )
        plot_real_datasets(
            real_gp,
            real_pfn_v25,
            real_pfn_v2,
            plot_dir,
            problems=args.real_problems,
        )

    if "three_way" in sections:
        plot_three_way(gp_records, pe_dixon, pe_rosenbrock, pfn_v25_records, plot_dir)

    if "cost_tables" in sections and "gp_vs_tabpfn" not in sections and "three_way" not in sections:
        plot_cost_tables_only(
            gp_records, pe_dixon, pe_rosenbrock, pfn_v25_records, pfn_v2_records, plot_dir
        )

    print(f"\n[DONE] Plots written to {plot_dir}")


if __name__ == "__main__":
    main()
