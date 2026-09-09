"""Publication-style plots for S4 SORF ASD validation evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
if str(_MTGPR_DIR) not in sys.path:
    sys.path.insert(0, str(_MTGPR_DIR))

from experiments_toa.s2_plotting import plot_prediction_coverage, plot_s2_posterior_examples
from gpplus.utils.fs_path import ensure_dir, fs_path
from mtgpr_experiment_utils import compute_prediction_coverage_metrics

DEFAULT_EVAL_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug29"
    / "s4_emit_aotbelow02_sorf_1inits_numrff600_lr0.1_nigp_freezeepochnigp300_dtypefloat32"
    / "asd_validation_eval"
)
DEFAULT_ASD_PATH = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_ASD_SOLUTIONS_CSV = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_solutions.csv"
)
DEFAULT_TRAIN_METRICS = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug29"
    / "s4_emit_aotbelow02_sorf_1inits_numrff600_lr0.1_nigp_freezeepochnigp300_dtypefloat32"
    / "gp_S4_EMIT_ST_nTrain100000_nVal5000_nTest15000_sorfD600_T8_correctSorfTrue.json"
)

TASK_LABELS: dict[str, str] = {
    "grain_size": "grain size (µm)",
    "cos_i": "cos incidence",
    "dust": "dust conc.",
    "algae": "algae conc.",
    "cwv": "column water vapor",
    "lwc": "liquid water content",
    "aot": "aerosol optical depth",
}

# QoI → asd_solutions.csv std column (and matching NetCDF variable).
TASK_ASD_STD_COLUMNS: dict[str, str] = {
    "grain_size": "grain_radius_std",
    "algae": "algae_conc_std",
    "dust": "dust_conc_std",
    "lwc": "lwc_std",
}

DEFAULT_TASK_ORDER = ["grain_size", "cos_i", "dust", "algae", "cwv", "lwc", "aot"]
SAMPLE_COLORS = list(plt.cm.tab10.colors[:6])
ASD_STD_COLOR = "#9E9AC8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _ordered_task_names(task_names: list[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(DEFAULT_TASK_ORDER)}
    return sorted(task_names, key=lambda name: order.get(name, len(DEFAULT_TASK_ORDER)))


def _task_column_map(arrays: dict[str, np.ndarray]) -> tuple[list[str], dict[str, int]]:
    raw_task_names = [str(t) for t in arrays["task_names"]]
    task_names = _ordered_task_names(raw_task_names)
    return task_names, {name: raw_task_names.index(name) for name in task_names}


def _task_subplot_grid(n_tasks: int) -> tuple[int, int, tuple[float, float]]:
    if n_tasks <= 6:
        nrows, ncols = 2, 3
    elif n_tasks <= 8:
        nrows, ncols = 2, 4
    else:
        ncols = 3
        nrows = int(np.ceil(n_tasks / ncols))
    figsize = (4.2 * ncols, 4.0 * nrows)
    return nrows, ncols, figsize


def _make_task_axes(n_tasks: int):
    nrows, ncols, figsize = _task_subplot_grid(n_tasks)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)
    return fig, axes_flat


def _default_train_metrics(eval_dir: Path) -> Path:
    run_dir = Path(eval_dir).parent
    candidates = sorted(run_dir.glob("gp_S4_*.json"))
    if candidates:
        return candidates[-1]
    return DEFAULT_TRAIN_METRICS


def _load_eval(eval_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    eval_dir = Path(eval_dir)
    with open(eval_dir / "asd_validation_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    npz = np.load(eval_dir / "asd_validation_predictions.npz", allow_pickle=True)
    arrays = {k: np.asarray(npz[k]) for k in npz.files}
    return metrics, arrays


def _load_asd_meta(asd_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(asd_path, "r") as f:
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        radiance = np.asarray(f["toa_radiance"][:], dtype=np.float64)
        dates = np.asarray(
            [d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in f["date"][:]]
        )
        cos_i = np.asarray(f["cos_i"][:], dtype=np.float64)
        sza = np.asarray(f["sza"][:], dtype=np.float64)
        aod = np.asarray(f["aod"][:], dtype=np.float64)
    return {
        "wl": wl,
        "radiance": radiance,
        "dates": dates,
        "cos_i": cos_i,
        "sza": sza,
        "aod": aod,
    }


def _normalize_date_key(value: object) -> str:
    text = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
    text = text.strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    return text.replace("-", "")


def load_asd_ref_stds(
    *,
    dates: np.ndarray | list[str],
    solutions_csv: Path | None = DEFAULT_ASD_SOLUTIONS_CSV,
    asd_path: Path | None = DEFAULT_ASD_PATH,
    task_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Per-sample ASD solution stds aligned to validation dates.

    Prefers ``asd_solutions.csv`` (user-facing source). Falls back to matching
    NetCDF ``*_std`` variables when a CSV column is missing.
    """
    date_keys = [_normalize_date_key(d) for d in dates]
    n = len(date_keys)
    wanted = (
        list(task_names)
        if task_names is not None
        else list(TASK_ASD_STD_COLUMNS.keys())
    )
    out: dict[str, np.ndarray] = {
        t: np.full(n, np.nan, dtype=np.float64)
        for t in wanted
        if t in TASK_ASD_STD_COLUMNS
    }
    if not out:
        return out

    csv_by_date: dict[str, dict[str, float]] = {}
    csv_path = Path(solutions_csv) if solutions_csv is not None else None
    if csv_path is not None and csv_path.is_file():
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = _normalize_date_key(row.get("date", ""))
                csv_by_date[key] = row
        for i, key in enumerate(date_keys):
            row = csv_by_date.get(key)
            if row is None:
                continue
            for task, col in TASK_ASD_STD_COLUMNS.items():
                if task not in out or col not in row or row[col] in ("", None):
                    continue
                out[task][i] = float(row[col])

    missing_tasks = [
        t for t, arr in out.items() if not np.any(np.isfinite(arr))
    ]
    nc_path = Path(asd_path) if asd_path is not None else None
    if missing_tasks and nc_path is not None and nc_path.is_file():
        with h5py.File(nc_path, "r") as f:
            nc_dates = [
                _normalize_date_key(d) for d in f["date"][:]
            ]
            index = {d: i for i, d in enumerate(nc_dates)}
            for task in missing_tasks:
                col = TASK_ASD_STD_COLUMNS[task]
                if col not in f:
                    continue
                vals = np.asarray(f[col][:], dtype=np.float64)
                for i, key in enumerate(date_keys):
                    j = index.get(key)
                    if j is None:
                        continue
                    out[task][i] = float(vals[j])

    return {t: arr for t, arr in out.items() if np.any(np.isfinite(arr))}


def _asd_std_for_task(
    asd_ref_std: dict[str, np.ndarray] | None,
    task: str,
    n_samples: int,
) -> np.ndarray | None:
    if not asd_ref_std or task not in asd_ref_std:
        return None
    std = np.asarray(asd_ref_std[task], dtype=np.float64).reshape(-1)
    if std.shape[0] != n_samples or not np.any(np.isfinite(std)):
        return None
    return np.where(np.isfinite(std), np.maximum(std, 0.0), np.nan)


def _sim_holdout_rrmse(train_metrics_path: Path, task_names: list[str]) -> dict[str, float]:
    path = Path(train_metrics_path)
    if not path.is_file():
        return {t: np.nan for t in task_names}
    with open(path, encoding="utf-8") as f:
        train = json.load(f)
    return {t: float(train.get(f"{t}_RRMSE", np.nan)) for t in task_names}


def _format_r2(r2: float) -> str:
    if not np.isfinite(r2):
        return "nan"
    if abs(r2) > 100:
        return f"{r2:.2e}"
    return f"{r2:.3f}"


def _resolve_intervals(
    arrays: dict[str, np.ndarray],
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper predictive intervals, preferring saved GP bounds."""
    y_pred = arrays["y_pred"]
    y_std = arrays["y_std"]
    if level == 0.95 and "lower_95" in arrays and "upper_95" in arrays:
        return arrays["lower_95"], arrays["upper_95"]
    cov = compute_prediction_coverage_metrics(
        np.zeros_like(y_pred),
        y_pred,
        y_std,
        levels=(level,),
    )
    _ = cov  # intervals built below via same z as metrics helper
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + 0.5 * level))
    ys = np.maximum(y_std, 0.0)
    return y_pred - z * ys, y_pred + z * ys


def _coverage_by_task(metrics: dict, arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    if "coverage_by_task" in metrics:
        return metrics["coverage_by_task"]
    task_names = [str(t) for t in arrays["task_names"]]
    out: dict[str, dict] = {}
    for i, task in enumerate(task_names):
        out[task] = compute_prediction_coverage_metrics(
            arrays["y_true"][:, i],
            arrays["y_pred"][:, i],
            arrays["y_std"][:, i],
        )
    return out


def plot_interval_coverage_grid(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    out_path: Path,
    asd_ref_std: dict[str, np.ndarray] | None = None,
) -> Path:
    """Per-task panels: truth vs 50% / 95% predictive intervals for each sample."""
    task_names, col_by_task = _task_column_map(arrays)
    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]
    lower_95, upper_95 = _resolve_intervals(arrays, level=0.95)
    lower_50, upper_50 = _resolve_intervals(arrays, level=0.50)
    n_samples = y_true.shape[0]

    fig, axes = _make_task_axes(len(task_names))
    x = np.arange(n_samples)
    used_asd_std = False

    for i, task in enumerate(task_names):
        ax = axes[i]
        col = col_by_task[task]
        yt = y_true[:, col]
        yp = y_pred[:, col]
        lo95 = lower_95[:, col]
        hi95 = upper_95[:, col]
        lo50 = lower_50[:, col]
        hi50 = upper_50[:, col]
        covered = (yt >= lo95) & (yt <= hi95)
        asd_std = _asd_std_for_task(asd_ref_std, task, n_samples)

        for j in range(n_samples):
            color = "#54A24B" if covered[j] else "#E45756"
            ax.fill_between(
                [j - 0.32, j + 0.32],
                [lo95[j], lo95[j]],
                [hi95[j], hi95[j]],
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            ax.fill_between(
                [j - 0.22, j + 0.22],
                [lo50[j], lo50[j]],
                [hi50[j], hi50[j]],
                color=color,
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
            ax.plot([j - 0.22, j + 0.22], [lo95[j], lo95[j]], color=color, lw=1.0, alpha=0.7, zorder=3)
            ax.plot([j - 0.22, j + 0.22], [hi95[j], hi95[j]], color=color, lw=1.0, alpha=0.7, zorder=3)
            if asd_std is not None and np.isfinite(asd_std[j]):
                used_asd_std = True
                ax.errorbar(
                    j,
                    yt[j],
                    yerr=asd_std[j],
                    fmt="none",
                    ecolor=ASD_STD_COLOR,
                    elinewidth=2.4,
                    capsize=5,
                    capthick=1.4,
                    zorder=4,
                )
            ax.plot(j, yp[j], "o", color="#F58518", ms=7, mec="white", mew=0.8, zorder=5)
            ax.plot(
                j,
                yt[j],
                marker="D",
                color="#4C78A8",
                ms=7,
                mec="white",
                mew=0.8,
                zorder=6,
            )

        cov_block = _coverage_by_task(metrics, arrays).get(task, {})
        c95 = float(cov_block.get("coverage_95", np.nan))
        title = (
            f"{TASK_LABELS.get(task, task)}\n"
            f"95% coverage: {c95:.0%} ({int(covered.sum())}/{n_samples} inside)"
        )
        if asd_std is not None:
            inside_asd = np.isfinite(asd_std) & (np.abs(yp - yt) <= asd_std)
            n_fin = int(np.isfinite(asd_std).sum())
            if n_fin:
                title += (
                    f"\npred inside ASD ±1σ: "
                    f"{int(inside_asd.sum())}/{n_fin}"
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{j + 1}" for j in range(n_samples)], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.2)
        if task in {"algae", "dust", "grain_size"}:
            vals = np.r_[yt, yp, lo95, hi95]
            if asd_std is not None:
                vals = np.r_[vals, yt - asd_std, yt + asd_std]
            if np.nanmax(np.abs(vals)) > 100:
                ax.set_yscale("symlog", linthresh=10)

    legend_handles = [
        Patch(facecolor="#4C78A8", label="ASD reference"),
        Patch(facecolor="#F58518", label="predictive mean"),
        Patch(facecolor="#54A24B", alpha=0.35, label="inside 95% interval"),
        Patch(facecolor="#E45756", alpha=0.35, label="outside 95% interval"),
    ]
    if used_asd_std:
        legend_handles.append(Patch(facecolor=ASD_STD_COLOR, label="ASD ±1σ (solutions.csv)"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Predictive interval coverage on ASD validation  (dark band = 50%, light = 95%)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_coverage_matrix(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    """Heatmap of whether truth falls inside nominal intervals."""
    task_names, col_by_task = _task_column_map(arrays)
    n_samples = arrays["y_true"].shape[0]
    levels = [0.50, 0.90, 0.95]
    mats = []
    yt = arrays["y_true"]
    for level in levels:
        lo, hi = _resolve_intervals(arrays, level=level)
        mat = np.zeros((len(task_names), n_samples), dtype=np.float64)
        for r, task in enumerate(task_names):
            col = col_by_task[task]
            mat[r] = ((yt[:, col] >= lo[:, col]) & (yt[:, col] <= hi[:, col])).astype(np.float64)
        mats.append(mat)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, level, mat in zip(axes, levels, mats):
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks(np.arange(n_samples))
        ax.set_xticklabels([f"S{j + 1}" for j in range(n_samples)])
        ax.set_yticks(np.arange(len(task_names)))
        ax.set_yticklabels([TASK_LABELS.get(t, t) for t in task_names])
        ax.set_title(f"{int(round(level * 100))}% nominal interval")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                sym = "✓" if mat[r, c] > 0.5 else "✗"
                ax.text(c, r, sym, ha="center", va="center", fontsize=11, color="#222222")

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="covered")
    fig.suptitle("Interval coverage by QoI and sample", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_posterior_coverage_examples(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    asd_meta: dict[str, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """S1-style posterior density panels (truth vs predictive distribution).

    Non-log QoIs: predictive mean only. LOG_SCALE_QOI: median (``y_pred``) plus
    mean and mode markers when present in the eval NPZ.
    """
    task_names = [str(t) for t in arrays["task_names"]]
    lower, upper = _resolve_intervals(arrays, level=0.95)
    if "x_spectral" in arrays:
        x_test = arrays["x_spectral"]
    else:
        x_test = asd_meta["radiance"]
    n_samples = arrays["y_true"].shape[0]
    out_dir = ensure_dir(out_dir)
    log_scale_tasks = [str(t) for t in metrics.get("log_scale_tasks", [])]
    logit_scale_tasks = [str(t) for t in metrics.get("logit_scale_tasks", [])]
    warped = bool(log_scale_tasks or logit_scale_tasks)
    y_pred_mean = arrays.get("y_pred_mean") if warped else None
    y_pred_mode = arrays.get("y_pred_mode") if warped else None
    log_mu = arrays.get("log_mu") if log_scale_tasks else None
    log_sigma = arrays.get("log_sigma") if log_scale_tasks else None
    saved = plot_s2_posterior_examples(
        x_test=x_test,
        wavelengths_nm=asd_meta["wl"],
        y_true=arrays["y_true"],
        y_pred=arrays["y_pred"],
        lower=lower,
        upper=upper,
        task_names=task_names,
        example_indices=list(range(n_samples)),
        save_dir=out_dir,
        title="ASD validation",
        y_std=arrays["y_std"],
        log_scale_tasks=log_scale_tasks,
        logit_scale_tasks=logit_scale_tasks,
        y_pred_mean=y_pred_mean,
        y_pred_mode=y_pred_mode,
        log_mu=log_mu,
        log_sigma=log_sigma,
        spectrum_ylabel="TOA radiance",
    )
    return [Path(p) for p in saved]


def generate_coverage_plots(
    eval_dir: Path,
    *,
    asd_path: Path = DEFAULT_ASD_PATH,
    asd_solutions_csv: Path = DEFAULT_ASD_SOLUTIONS_CSV,
    out_dir: Path | None = None,
) -> list[Path]:
    eval_dir = Path(eval_dir)
    out_dir = Path(out_dir) if out_dir is not None else eval_dir / "plots" / "coverage"
    ensure_dir(out_dir)

    metrics, arrays = _load_eval(eval_dir)
    asd_meta = _load_asd_meta(asd_path)
    asd_ref_std = load_asd_ref_stds(
        dates=asd_meta["dates"],
        solutions_csv=asd_solutions_csv,
        asd_path=asd_path,
        task_names=[str(t) for t in arrays["task_names"]],
    )
    coverage_by_task = _coverage_by_task(metrics, arrays)
    task_names = [str(t) for t in arrays["task_names"]]

    outputs: list[Path] = [
        Path(p)
        for p in plot_prediction_coverage(
            task_names=task_names,
            coverage_by_task=coverage_by_task,
            save_dir=out_dir,
            title="ASD validation — predictive interval calibration",
        )
    ]
    outputs.extend(
        [
            plot_interval_coverage_grid(
                metrics,
                arrays,
                out_dir / "interval_coverage_grid.png",
                asd_ref_std=asd_ref_std,
            ),
            plot_coverage_matrix(metrics, arrays, out_dir / "coverage_matrix.png"),
        ]
    )
    outputs.extend(
        plot_posterior_coverage_examples(
            metrics,
            arrays,
            asd_meta,
            out_dir / "posterior",
        )
    )
    return outputs


def plot_metrics_comparison(
    metrics: dict,
    sim_rrmse: dict[str, float],
    out_path: Path,
) -> Path:
    task_names = _ordered_task_names([str(t) for t in metrics["task_names"]])
    asd_rrmse = [float(metrics["per_task"][t]["RRMSE"]) for t in task_names]
    sim_vals = [sim_rrmse.get(t, np.nan) for t in task_names]
    r2_vals = [float(metrics["per_task"][t].get(f"{t}_R2", metrics["per_task"][t].get("R2", np.nan))) for t in task_names]

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    x = np.arange(len(task_names))
    width = 0.36
    ax0.bar(
        x - width / 2,
        sim_vals,
        width,
        label="simulation holdout",
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
    )
    bars = ax0.bar(
        x + width / 2,
        asd_rrmse,
        width,
        label="ASD validation (n=6)",
        color="#F58518",
        edgecolor="white",
        linewidth=0.6,
    )
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], rotation=22, ha="right")
    ax0.set_ylabel("RRMSE (log scale)")
    ax0.set_title("Generalization gap: simulation vs field ASD")
    ax0.legend(loc="upper left", frameon=True)
    ax0.grid(axis="y", alpha=0.25, which="both")

    for bar, val in zip(bars, asd_rrmse):
        if val > 10:
            ax0.annotate(
                f"{val:.0e}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    colors = ["#54A24B" if r >= 0 else "#E45756" for r in r2_vals]
    ax1.barh(
        [TASK_LABELS.get(t, t) for t in task_names],
        r2_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax1.axvline(0.0, color="#333333", lw=0.8, alpha=0.5)
    ax1.axvline(1.0, color="#333333", lw=0.8, alpha=0.25, linestyle=":")
    ax1.set_xlim(-1.5, 1.05)
    ax1.set_xlabel("R² on ASD validation")
    ax1.set_title("Per-task fit quality (ASD)")
    ax1.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "S4 SORF (num_rff=600) — ASD field validation",
        fontsize=13,
        fontweight="semibold",
        y=1.02,
    )
    fig.subplots_adjust(top=0.88, wspace=0.28)
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_task_scatter_grid(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    out_path: Path,
    asd_ref_std: dict[str, np.ndarray] | None = None,
) -> Path:
    task_names, col_by_task = _task_column_map(arrays)
    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]
    y_std = arrays["y_std"]
    n_tasks = len(task_names)
    n_samples = y_true.shape[0]
    used_asd_std = False

    fig, axes = _make_task_axes(n_tasks)

    for i, task in enumerate(task_names):
        ax = axes[i]
        col = col_by_task[task]
        yt = y_true[:, col]
        yp = y_pred[:, col]
        ys = y_std[:, col]
        lower = arrays["lower_95"][:, col] if "lower_95" in arrays else yp - 1.96 * ys
        upper = arrays["upper_95"][:, col] if "upper_95" in arrays else yp + 1.96 * ys
        asd_std = _asd_std_for_task(asd_ref_std, task, n_samples)

        for j in range(len(yt)):
            xerr = None
            if asd_std is not None and np.isfinite(asd_std[j]):
                xerr = [[asd_std[j]], [asd_std[j]]]
                used_asd_std = True
            ax.errorbar(
                yt[j],
                yp[j],
                xerr=xerr,
                yerr=[[yp[j] - lower[j]], [upper[j] - yp[j]]],
                fmt="o",
                ms=7,
                mfc=SAMPLE_COLORS[j],
                mec="white",
                mew=0.8,
                ecolor=SAMPLE_COLORS[j],
                elinewidth=1.2,
                capsize=3,
                alpha=0.92,
                zorder=3,
            )
            if xerr is not None:
                # Draw ASD ±1σ in a distinct color on top of the shared errorbar color.
                ax.errorbar(
                    yt[j],
                    yp[j],
                    xerr=xerr,
                    fmt="none",
                    ecolor=ASD_STD_COLOR,
                    elinewidth=2.0,
                    capsize=4,
                    capthick=1.2,
                    zorder=4,
                )

        lo_vals = [yt.min(), yp.min(), lower.min()]
        hi_vals = [yt.max(), yp.max(), upper.max()]
        if asd_std is not None:
            lo_vals.extend((yt - asd_std)[np.isfinite(asd_std)])
            hi_vals.extend((yt + asd_std)[np.isfinite(asd_std)])
        lo = float(np.nanmin(lo_vals))
        hi = float(np.nanmax(hi_vals))
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="#444444", ls="--", lw=1.0, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        row = metrics["per_task"][task]
        rrmse = float(row["RRMSE"])
        r2 = float(row.get(f"{task}_R2", row.get("R2", np.nan)))
        note = f"RRMSE={rrmse:.3g}\nR²={_format_r2(r2)}"
        if asd_std is not None:
            inside = np.isfinite(asd_std) & (np.abs(yp - yt) <= asd_std)
            n_fin = int(np.isfinite(asd_std).sum())
            if n_fin:
                note += f"\nin ASD±1σ: {int(inside.sum())}/{n_fin}"
        ax.set_title(f"{TASK_LABELS.get(task, task)}")
        ax.set_xlabel("ASD reference")
        ax.set_ylabel("SORF prediction")
        ax.text(
            0.04,
            0.96,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
        )

    legend_handles = [
        Patch(facecolor=SAMPLE_COLORS[j], edgecolor="white", label=f"Sample {j + 1}")
        for j in range(n_samples)
    ]
    if used_asd_std:
        legend_handles.append(Patch(facecolor=ASD_STD_COLOR, label="ASD ±1σ (x)"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(7, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Predicted vs reference (y: 95% predictive intervals; x: ASD ±1σ when available)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sample_comparison(
    metrics: dict,
    out_path: Path,
    asd_ref_std: dict[str, np.ndarray] | None = None,
) -> Path:
    task_names = _ordered_task_names([str(t) for t in metrics["task_names"]])
    n_samples = len(metrics["per_task"][task_names[0]]["y_true"])
    used_asd_std = False

    fig, axes = _make_task_axes(len(task_names))
    x = np.arange(n_samples)
    width = 0.36

    for i, task in enumerate(task_names):
        ax = axes[i]
        yt = np.asarray(metrics["per_task"][task]["y_true"], dtype=np.float64)
        yp = np.asarray(metrics["per_task"][task]["y_pred"], dtype=np.float64)
        ys = np.asarray(metrics["per_task"][task].get("y_std", np.zeros(n_samples)), dtype=np.float64)
        asd_std = _asd_std_for_task(asd_ref_std, task, n_samples)
        asd_yerr = None
        if asd_std is not None:
            asd_yerr = np.where(np.isfinite(asd_std), asd_std, 0.0)
            used_asd_std = True
        ax.bar(
            x - width / 2,
            yt,
            width,
            yerr=asd_yerr,
            label="ASD reference",
            color="#4C78A8",
            edgecolor="white",
            error_kw={"ecolor": ASD_STD_COLOR, "elinewidth": 1.6, "capsize": 3},
        )
        ax.bar(
            x + width / 2,
            yp,
            width,
            yerr=1.96 * np.maximum(ys, 0.0),
            label="SORF pred.",
            color="#F58518",
            edgecolor="white",
            error_kw={"ecolor": "#F58518", "elinewidth": 1.2, "capsize": 3, "alpha": 0.85},
        )
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{j + 1}" for j in range(n_samples)], fontsize=8)
        if task in {"algae", "dust", "grain_size"} and np.nanmax(np.abs(np.r_[yt, yp])) > 100:
            ax.set_yscale("symlog", linthresh=10)
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if used_asd_std:
        handles.append(Patch(facecolor=ASD_STD_COLOR, label="ASD ±1σ"))
        labels.append("ASD ±1σ")
    axes[0].legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle(
        "Per-sample reference vs prediction (ASD ±1σ; pred 95% CI)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_residual_heatmap(metrics: dict, out_path: Path) -> Path:
    task_names = _ordered_task_names([str(t) for t in metrics["task_names"]])
    n_samples = len(metrics["per_task"][task_names[0]]["y_true"])
    rel_err = np.zeros((len(task_names), n_samples), dtype=np.float64)

    for i, task in enumerate(task_names):
        yt = np.asarray(metrics["per_task"][task]["y_true"], dtype=np.float64)
        yp = np.asarray(metrics["per_task"][task]["y_pred"], dtype=np.float64)
        denom = np.maximum(np.abs(yt), 1e-6)
        rel_err[i] = (yp - yt) / denom

    vmax = np.nanpercentile(np.abs(rel_err), 95)
    vmax = max(vmax, 0.5)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(
        rel_err,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_samples))
    ax.set_xticklabels([f"S{j + 1}" for j in range(n_samples)])
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in task_names])
    ax.set_xlabel("ASD validation sample")
    ax.set_title("Relative error  (pred − ref) / |ref|")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("relative error")

    for i in range(rel_err.shape[0]):
        for j in range(rel_err.shape[1]):
            val = rel_err[i, j]
            txt_color = "white" if abs(val) > 0.55 * vmax else "#222222"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_color, fontsize=8)

    fig.tight_layout()
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectra(asd_meta: dict[str, np.ndarray], out_path: Path) -> Path:
    wl = asd_meta["wl"]
    rad = asd_meta["radiance"]
    n = rad.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    axes = axes.ravel()
    norm = mcolors.Normalize(vmin=wl.min(), vmax=wl.max())
    cmap = plt.cm.viridis

    for i in range(min(6, n)):
        ax = axes[i]
        y = rad[i]
        color = cmap(norm(wl.mean()))
        ax.plot(wl, y, color=SAMPLE_COLORS[i], lw=1.3)
        ax.fill_between(wl, y, alpha=0.12, color=SAMPLE_COLORS[i])
        ax.set_title(f"Sample {i + 1}  |  cos i={asd_meta['cos_i'][i]:.2f}  AOD={asd_meta['aod'][i]:.3f}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("TOA radiance")
        ax.grid(alpha=0.2)

    fig.suptitle("ASD validation TOA radiance spectra", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dashboard(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    asd_meta: dict[str, np.ndarray],
    sim_rrmse: dict[str, float],
    out_path: Path,
    asd_ref_std: dict[str, np.ndarray] | None = None,
) -> Path:
    """Single-page overview for slides / reports."""
    task_names = _ordered_task_names([str(t) for t in metrics["task_names"]])
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.15, 1.0], hspace=0.38, wspace=0.28)

    # Top-left: RRMSE comparison
    ax_rrmse = fig.add_subplot(gs[0, :2])
    x = np.arange(len(task_names))
    width = 0.36
    sim_vals = [sim_rrmse.get(t, np.nan) for t in task_names]
    asd_vals = [float(metrics["per_task"][t]["RRMSE"]) for t in task_names]
    ax_rrmse.bar(x - width / 2, sim_vals, width, label="sim holdout", color="#4C78A8")
    ax_rrmse.bar(x + width / 2, asd_vals, width, label="ASD validation", color="#F58518")
    ax_rrmse.set_yscale("log")
    ax_rrmse.set_xticks(x)
    ax_rrmse.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], rotation=18, ha="right")
    ax_rrmse.set_ylabel("RRMSE")
    ax_rrmse.set_title("RRMSE: simulation holdout vs ASD field data")
    ax_rrmse.legend(frameon=True)
    ax_rrmse.grid(axis="y", alpha=0.25, which="both")

    # Top-right: best tasks scatter (grain_size, cos_i)
    for k, task in enumerate(["grain_size", "cos_i"]):
        if task not in task_names:
            continue
        ax = fig.add_subplot(gs[0, 2] if k == 0 else gs[1, 2])
        raw = [str(t) for t in arrays["task_names"]]
        idx = raw.index(task)
        yt = arrays["y_true"][:, idx]
        yp = arrays["y_pred"][:, idx]
        asd_std = _asd_std_for_task(asd_ref_std, task, len(yt))
        for j in range(len(yt)):
            if asd_std is not None and np.isfinite(asd_std[j]):
                ax.errorbar(
                    yt[j],
                    yp[j],
                    xerr=asd_std[j],
                    fmt="o",
                    ms=7,
                    color=SAMPLE_COLORS[j],
                    ecolor=ASD_STD_COLOR,
                    elinewidth=1.6,
                    capsize=3,
                    mec="white",
                    mew=0.8,
                    zorder=3,
                )
            else:
                ax.scatter(
                    yt[j],
                    yp[j],
                    s=55,
                    color=SAMPLE_COLORS[j],
                    edgecolors="white",
                    linewidth=0.8,
                    zorder=3,
                )
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        if asd_std is not None:
            lo = min(lo, float(np.nanmin(yt - asd_std)))
            hi = max(hi, float(np.nanmax(yt + asd_std)))
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#666666", lw=1)
        r2 = float(metrics["per_task"][task].get(f"{task}_R2", np.nan))
        title = f"{TASK_LABELS.get(task, task)}  (R²={_format_r2(r2)})"
        if asd_std is not None:
            inside = np.isfinite(asd_std) & (np.abs(yp - yt) <= asd_std)
            n_fin = int(np.isfinite(asd_std).sum())
            if n_fin:
                title += f"\nin ASD±1σ: {int(inside.sum())}/{n_fin}"
        ax.set_title(title)
        ax.set_xlabel("reference")
        ax.set_ylabel("prediction")

    # Middle row: spectra (span 2 cols) + residual heatmap
    ax_spec = fig.add_subplot(gs[1, :2])
    wl = asd_meta["wl"]
    for i in range(asd_meta["radiance"].shape[0]):
        ax_spec.plot(wl, asd_meta["radiance"][i], color=SAMPLE_COLORS[i], lw=1.0, alpha=0.9, label=f"S{i + 1}")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("TOA radiance")
    ax_spec.set_title("Input spectra (ASD validation)")
    ax_spec.legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
    ax_spec.grid(alpha=0.2)

    ax_hm = fig.add_subplot(gs[2, :])
    rel_err = []
    for task in task_names:
        yt = np.asarray(metrics["per_task"][task]["y_true"])
        yp = np.asarray(metrics["per_task"][task]["y_pred"])
        rel_err.append((yp - yt) / np.maximum(np.abs(yt), 1e-6))
    rel_err = np.asarray(rel_err)
    vmax = max(np.nanpercentile(np.abs(rel_err), 95), 0.5)
    im = ax_hm.imshow(rel_err, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_hm.set_xticks(np.arange(rel_err.shape[1]))
    ax_hm.set_xticklabels([f"S{j + 1}" for j in range(rel_err.shape[1])])
    ax_hm.set_yticks(np.arange(len(task_names)))
    ax_hm.set_yticklabels([TASK_LABELS.get(t, t) for t in task_names])
    ax_hm.set_title("Relative error heatmap")
    fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01, label="(pred−ref)/|ref|")

    fig.suptitle(
        "S4 SORF ASD validation overview  |  ASD ±1σ from asd_solutions.csv where available",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.savefig(fs_path(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_asd_validation_plots(
    eval_dir: Path,
    *,
    asd_path: Path = DEFAULT_ASD_PATH,
    asd_solutions_csv: Path = DEFAULT_ASD_SOLUTIONS_CSV,
    train_metrics_path: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    eval_dir = Path(eval_dir)
    out_dir = Path(out_dir) if out_dir is not None else eval_dir / "plots"
    ensure_dir(out_dir)

    metrics, arrays = _load_eval(eval_dir)
    asd_meta = _load_asd_meta(asd_path)
    asd_ref_std = load_asd_ref_stds(
        dates=asd_meta["dates"],
        solutions_csv=asd_solutions_csv,
        asd_path=asd_path,
        task_names=[str(t) for t in metrics["task_names"]],
    )
    train_metrics_path = Path(train_metrics_path) if train_metrics_path is not None else _default_train_metrics(eval_dir)
    sim_rrmse = _sim_holdout_rrmse(train_metrics_path, metrics["task_names"])

    outputs = [
        plot_metrics_comparison(metrics, sim_rrmse, out_dir / "01_metrics_comparison.png"),
        plot_task_scatter_grid(
            metrics,
            arrays,
            out_dir / "02_scatter_with_intervals.png",
            asd_ref_std=asd_ref_std,
        ),
        plot_sample_comparison(
            metrics,
            out_dir / "03_sample_bars.png",
            asd_ref_std=asd_ref_std,
        ),
        plot_residual_heatmap(metrics, out_dir / "04_relative_error_heatmap.png"),
        plot_spectra(asd_meta, out_dir / "05_input_spectra.png"),
        plot_dashboard(
            metrics,
            arrays,
            asd_meta,
            sim_rrmse,
            out_dir / "00_dashboard.png",
            asd_ref_std=asd_ref_std,
        ),
    ]
    outputs.extend(
        generate_coverage_plots(
            eval_dir,
            asd_path=asd_path,
            asd_solutions_csv=asd_solutions_csv,
            out_dir=out_dir / "coverage",
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--asd-path", type=Path, default=DEFAULT_ASD_PATH)
    parser.add_argument(
        "--asd-solutions",
        type=Path,
        default=DEFAULT_ASD_SOLUTIONS_CSV,
        help="CSV with per-date ASD QoI stds (grain_radius_std, ...)",
    )
    parser.add_argument("--train-metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Generate only uncertainty / coverage figures",
    )
    args = parser.parse_args()

    if args.coverage_only:
        paths = generate_coverage_plots(
            args.eval_dir,
            asd_path=args.asd_path,
            asd_solutions_csv=args.asd_solutions,
            out_dir=(args.out_dir / "coverage") if args.out_dir else None,
        )
    else:
        train_metrics = args.train_metrics or _default_train_metrics(args.eval_dir)
        paths = generate_asd_validation_plots(
            args.eval_dir,
            asd_path=args.asd_path,
            asd_solutions_csv=args.asd_solutions,
            train_metrics_path=train_metrics,
            out_dir=args.out_dir,
        )
    print(f"Saved {len(paths)} plots to {paths[0].parent}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
