"""Regenerate report-ready tables/plots for the Second Interim Report.

Reads existing ``gp_*.json`` + ``predictions_*.npz`` (no re-training).
Sample posterior/scatter figures omit aggregate metrics; those go in tables only.

Usage::

    python experiments_toa/regen_interim2_figures.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MTGPR = _ROOT / "experiments_RFFMTGPR"
if str(_MTGPR) not in sys.path:
    sys.path.insert(0, str(_MTGPR))

from experiments_toa.s2_plotting import (  # noqa: E402
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    select_posterior_example_indices,
)
from plot_toa_posterior import plot_toa_posterior_figures  # noqa: E402

OUT_DIR = _ROOT / "docs" / "overleaf" / "figures" / "interim2"

# Design ranges from experiments_toa/toa_simulations.py Sobol decode.
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "cos_i": (0.06, 1.0),
    "y_cos": (0.06, 1.0),
    "grain_size": (30.0, 1500.0),
    "y_grain": (30.0, 1500.0),
    "liquid_water": (0.0, 25.0),
    "dust": (0.0, 4000.0),
    "algae": (0.0, 6e5),
    "cwv": (0.2, 5.2),
    "aot": (0.04, 1.0),
    "fsnow": (0.0, 10.0),
    "fPV": (0.0, 10.0),
    "fNPV": (0.0, 10.0),
    "fsoil": (0.0, 10.0),
}

RUNS: dict[str, Path] = {
    "july16_rff_lr0.01": _ROOT
    / "experiments_RFF/results/July16/toa_rff_1inits_numrff1600_lr0.01_noisevarfrac0.001_noisepriorlogscale0.5",
    "july16_orf_lr0.01": _ROOT
    / "experiments_ORF/results/July16/toa_orf_1inits_numorfNone_lr0.01_noisevarfrac0.001_noisepriorlogscale0.5",
    "july16_orf_lr0.1": _ROOT
    / "experiments_ORF/results/July16/toa_orf_1inits_numorfNone_lr0.1_noisevarfrac0.001_noisepriorlogscale0.5",
    "july16_orf_lr1.0": _ROOT
    / "experiments_ORF/results/July16/toa_orf_1inits_numorfNone_lr1.0_noisevarfrac0.001_noisepriorlogscale0.5",
    "july16_sorf_lr0.01": _ROOT
    / "experiments_SORF/results/July16/toa_sorf_1inits_numrff1600_lr0.01_noisevarfrac0.001_noisepriorlogscale0.5_dtypefloat64",
    "july16_sorf_lr0.1": _ROOT
    / "experiments_SORF/results/July16/toa_sorf_1inits_numrff1600_lr0.1_noisevarfrac0.001_noisepriorlogscale0.5_dtypefloat64",
    "july16_sorf_lr1.0": _ROOT
    / "experiments_SORF/results/July16/toa_sorf_1inits_numrff1600_lr1.0_noisevarfrac0.001_noisepriorlogscale0.5_dtypefloat64",
    "july23_s2_d400": _ROOT
    / "experiments_SORF/results/July23/s2_toa_sorf_1inits_numrff400_lr0.01_taskbandconfig_dtypefloat64",
    "july29_look": _ROOT
    / "experiments_SORF/results/July29/adjusted_init_ls_LOOK"
    / "s2_toa_sorf_1inits_numrff1600_lr0.01_taskbandconfig_dtypefloat64",
    "july29_pacbayes": _ROOT
    / "experiments_SORF/results/July29/adjusted_init_ls_pacbayes_LOOK"
    / "s2_toa_sorf_1inits_numrff1600_lr0.01_taskbandconfig_dtypefloat64",
}

ANALYSIS_COPIES: list[tuple[Path, str]] = [
    (
        _ROOT / "experiments_toa/analysis_toa_July21/toa_radiance_vs_qoi_correlation.png",
        "corr_radiance_vs_qoi_11qoi.png",
    ),
    (
        _ROOT / "experiments_toa/analysis_toa_July21/per_qoi_corr_spectrum.png",
        "bands_per_qoi_11qoi.png",
    ),
    (
        _ROOT
        / "experiments_toa/analysis_toa_fsnow_only_July27/toa_radiance_vs_qoi_correlation.png",
        "corr_radiance_vs_qoi_fsnow_only.png",
    ),
    (
        _ROOT / "experiments_toa/analysis_toa_fsnow_only_July27/per_qoi_corr_spectrum.png",
        "bands_per_qoi_fsnow_only.png",
    ),
]


def _find_json(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("gp_*.json"))
    if not matches:
        raise FileNotFoundError(f"No gp_*.json in {run_dir}")
    return matches[0]


def _find_npz(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("predictions_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No predictions_*.npz in {run_dir}")
    return matches[0]


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads(_find_json(run_dir).read_text(encoding="utf-8"))


def _task_names_from_metrics(metrics: dict[str, Any]) -> list[str]:
    names = metrics.get("task_names")
    if names:
        return [str(n) for n in names]
    # Fallback: keys ending in _RRMSE that are not aggregate / best_val.
    out = []
    for k, v in metrics.items():
        if not k.endswith("_RRMSE"):
            continue
        if k.startswith("aggregate") or "best_val" in k:
            continue
        out.append(k[: -len("_RRMSE")])
    return sorted(out)


def write_metrics_table(run_id: str, metrics: dict[str, Any], out_dir: Path) -> Path:
    names = _task_names_from_metrics(metrics)
    rows: list[dict[str, Any]] = []
    for name in names:
        rows.append(
            {
                "task": name,
                "RMSE": metrics.get(f"{name}_RMSE"),
                "MAE": metrics.get(f"{name}_MAE"),
                "RRMSE": metrics.get(f"{name}_RRMSE"),
                "R2": metrics.get(f"{name}_R2"),
                "NLPD": metrics.get(f"{name}_NLPD"),
            }
        )

    csv_path = out_dir / f"metrics_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "RMSE", "MAE", "RRMSE", "R2", "NLPD"]
        )
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / f"metrics_{run_id}.tex"
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"QoI & RMSE & MAE & RRMSE & $R^2$ & NLPD \\",
        r"\midrule",
    ]
    for row in rows:
        def _fmt(x: Any, prec: int = 4) -> str:
            if x is None:
                return "---"
            try:
                v = float(x)
            except (TypeError, ValueError):
                return "---"
            if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
                return f"{v:.3e}"
            return f"{v:.{prec}f}"

        lines.append(
            f"{row['task']} & {_fmt(row['RMSE'])} & {_fmt(row['MAE'])} & "
            f"{_fmt(row['RRMSE'], 4)} & {_fmt(row['R2'], 4)} & {_fmt(row['NLPD'], 3)} \\\\"
        )
    agg = metrics.get("aggregate_RRMSE")
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{aggregate}} & --- & --- & {_fmt(agg, 4)} & --- & --- \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "aggregate_RRMSE": metrics.get("aggregate_RRMSE"),
        "Training_Time": metrics.get("Training_Time"),
        "initial_lr": metrics.get("initial_lr"),
        "num_rff": metrics.get("num_rff"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "task_names": names,
        "pac_bayes": metrics.get("pac_bayes"),
        "pac_bayes_temperature": metrics.get("pac_bayes_temperature"),
        "pac_bayes_prior_std": metrics.get("pac_bayes_prior_std"),
        "pac_bayes_posterior_std": metrics.get("pac_bayes_posterior_std"),
    }
    (out_dir / f"summary_{run_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return tex_path


def _plot_scatter_with_bounds(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
    out_path: Path,
    show_bounds: bool,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")

    stats: dict[str, float] = {
        "n": float(len(y_pred)),
        "frac_below_floor": 0.0,
        "frac_above_ceil": 0.0,
    }
    bounds = PHYSICAL_BOUNDS.get(task_name)
    if show_bounds and bounds is not None:
        flo, ceil = bounds
        if flo is not None:
            ax.axhline(flo, color="C3", ls=":", lw=1.2, label=f"floor={flo:g}")
            n_lo = int(np.sum(y_pred < flo))
            stats["frac_below_floor"] = n_lo / max(len(y_pred), 1)
        if ceil is not None:
            ax.axhline(ceil, color="C1", ls=":", lw=1.2, label=f"ceil={ceil:g}")
            n_hi = int(np.sum(y_pred > ceil))
            stats["frac_above_ceil"] = n_hi / max(len(y_pred), 1)
        note = (
            f"below floor: {100 * stats['frac_below_floor']:.1f}%; "
            f"above ceil: {100 * stats['frac_above_ceil']:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    ax.set_xlabel(f"True {task_name}")
    ax.set_ylabel(f"Predicted {task_name}")
    ax.set_title(f"{task_name}: predicted vs true (full test set)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stats


def regenerate_s2_run(
    run_id: str,
    run_dir: Path,
    out_dir: Path,
    *,
    n_examples: int = 4,
    show_bounds: bool = False,
) -> None:
    metrics = _load_metrics(run_dir)
    write_metrics_table(run_id, metrics, out_dir)

    data = np.load(_find_npz(run_dir), allow_pickle=True)
    task_names = [str(n) for n in np.asarray(data["task_names"]).tolist()]
    y_true = np.asarray(data["y_true"], dtype=np.float64)
    y_pred = np.asarray(data["y_pred"], dtype=np.float64)
    y_std = np.asarray(data["y_std"], dtype=np.float64)
    lower = np.asarray(data["lower"], dtype=np.float64)
    upper = np.asarray(data["upper"], dtype=np.float64)

    if "x_test" in data.files:
        x_test = np.asarray(data["x_test"], dtype=np.float64)
    else:
        x_test = np.asarray(data["x_test_orig"], dtype=np.float64)
    if "wavelengths_nm" in data.files:
        wl = np.asarray(data["wavelengths_nm"], dtype=np.float64)
    else:
        wl = np.asarray(data["wavelength_nm"], dtype=np.float64)

    scatter_dir = out_dir / run_id / "scatter"
    oob_stats: dict[str, dict[str, float]] = {}
    for t, name in enumerate(task_names):
        oob_stats[name] = _plot_scatter_with_bounds(
            y_true=y_true[:, t],
            y_pred=y_pred[:, t],
            task_name=name,
            out_path=scatter_dir / f"{name}_scatter.png",
            show_bounds=show_bounds,
        )
        # Also a clean copy without bound lines for main report when not LOOK.
        if show_bounds:
            plot_s2_task_scatter(
                y_true=y_true[:, t],
                y_pred=y_pred[:, t],
                lower=None,
                upper=None,
                task_name=name,
                out_path=scatter_dir / f"{name}_scatter_clean.png",
                title=f"{name}: predicted vs true (full test set)",
            )

    if show_bounds:
        (out_dir / f"oob_{run_id}.json").write_text(
            json.dumps(oob_stats, indent=2), encoding="utf-8"
        )

    # Sample-only posteriors: deliberately omit rel_metrics_by_task.
    if "example_indices" in data.files:
        idxs = [int(i) for i in np.asarray(data["example_indices"]).tolist()][
            :n_examples
        ]
    else:
        idxs = select_posterior_example_indices(len(y_true), n_examples, seed=42)

    is_july16 = task_names == ["y_cos", "y_grain"] or set(task_names) == {
        "y_cos",
        "y_grain",
    }
    post_dir = out_dir / run_id / "posterior"
    post_dir.mkdir(parents=True, exist_ok=True)

    if is_july16:
        # S1-style 3-panel figures; rel_metrics_by_task=None keeps aggregates off-plot.
        plot_toa_posterior_figures(
            x_test,
            y_true,
            y_pred,
            y_std,
            lower,
            upper,
            post_dir,
            title=run_id,
            example_indices=idxs,
            rel_metrics_by_task=None,
            wavelength_nm=wl,
            log_grain=bool(data["log_grain"]) if "log_grain" in data.files else True,
            logit_cos=bool(data["logit_cos"]) if "logit_cos" in data.files else False,
            y_pred_mean=np.asarray(data["y_pred_mean"])
            if "y_pred_mean" in data.files
            else None,
            y_pred_mode=np.asarray(data["y_pred_mode"])
            if "y_pred_mode" in data.files
            else None,
            log_mu=np.asarray(data["log_mu"]) if "log_mu" in data.files else None,
            log_sigma=np.asarray(data["log_sigma"])
            if "log_sigma" in data.files
            else None,
        )
    else:
        plot_s2_posterior_examples(
            x_test=x_test,
            wavelengths_nm=wl,
            y_true=y_true,
            y_pred=y_pred,
            lower=lower,
            upper=upper,
            task_names=task_names,
            example_indices=idxs,
            save_dir=post_dir,
            title=run_id,
            y_std=y_std,
            rel_metrics_by_task=None,
            log_scale_tasks=[
                str(n)
                for n in np.asarray(data["log_scale_tasks"]).tolist()
            ]
            if "log_scale_tasks" in data.files
            else None,
            y_pred_mean=np.asarray(data["y_pred_mean"])
            if "y_pred_mean" in data.files
            else None,
            y_pred_mode=np.asarray(data["y_pred_mode"])
            if "y_pred_mode" in data.files
            else None,
            log_mu=np.asarray(data["log_mu"]) if "log_mu" in data.files else None,
            log_sigma=np.asarray(data["log_sigma"])
            if "log_sigma" in data.files
            else None,
            spectrum_ylabel="Radiance",
        )

    # Ensure saved figures carry a clear single-sample caption in the filename list.
    (out_dir / run_id / "example_indices.json").write_text(
        json.dumps({"example_indices": idxs, "note": "single test samples only"}, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] {run_id}: tables + {len(task_names)} scatters + {len(idxs)} posteriors")


def write_lr_cost_table(out_dir: Path) -> None:
    """ORF/SORF July16 lr sweep: aggregate RRMSE vs Training_Time."""
    rows = []
    for run_id, method, lr in [
        ("july16_orf_lr0.01", "ORF", 0.01),
        ("july16_orf_lr0.1", "ORF", 0.1),
        ("july16_orf_lr1.0", "ORF", 1.0),
        ("july16_sorf_lr0.01", "SORF", 0.01),
        ("july16_sorf_lr0.1", "SORF", 0.1),
        ("july16_sorf_lr1.0", "SORF", 1.0),
        ("july16_rff_lr0.01", "RFF", 0.01),
    ]:
        m = _load_metrics(RUNS[run_id])
        rows.append(
            {
                "method": method,
                "lr": lr,
                "aggregate_RRMSE": float(m["aggregate_RRMSE"]),
                "Training_Time_s": float(m["Training_Time"]),
                "y_cos_RRMSE": float(m.get("y_cos_RRMSE", float("nan"))),
                "y_grain_RRMSE": float(m.get("y_grain_RRMSE", float("nan"))),
            }
        )

    csv_path = out_dir / "lr_cost_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "lr_cost_performance.tex"
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & lr & Agg.\ RRMSE & Train time (s) & cos\_i RRMSE & grain RRMSE \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} & {r['lr']:g} & {r['aggregate_RRMSE']:.5f} & "
            f"{r['Training_Time_s']:.0f} & {r['y_cos_RRMSE']:.5f} & "
            f"{r['y_grain_RRMSE']:.5f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    # Simple cost-performance scatter.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    markers = {"RFF": "o", "ORF": "s", "SORF": "^"}
    for method in ("RFF", "ORF", "SORF"):
        sub = [r for r in rows if r["method"] == method]
        ax.scatter(
            [r["Training_Time_s"] for r in sub],
            [r["aggregate_RRMSE"] for r in sub],
            marker=markers[method],
            s=70,
            label=method,
        )
        for r in sub:
            ax.annotate(
                f"lr={r['lr']:g}",
                (r["Training_Time_s"], r["aggregate_RRMSE"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Aggregate test RRMSE")
    ax.set_title("July16 2-QoI: cost vs performance (lr sweep)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lr_cost_performance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] lr cost-performance table + plot -> {csv_path.name}")


def write_method_compare_table(out_dir: Path) -> None:
    """Matched lr=0.01 RFF / ORF / SORF (float64) comparison."""
    specs = [
        ("RFF", "july16_rff_lr0.01"),
        ("ORF", "july16_orf_lr0.01"),
        ("SORF", "july16_sorf_lr0.01"),
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Agg.\ RRMSE & cos\_i RRMSE & grain RRMSE & Train time (s) \\",
        r"\midrule",
    ]
    for label, run_id in specs:
        m = _load_metrics(RUNS[run_id])
        lines.append(
            f"{label} & {float(m['aggregate_RRMSE']):.5f} & "
            f"{float(m['y_cos_RRMSE']):.5f} & {float(m['y_grain_RRMSE']):.5f} & "
            f"{float(m['Training_Time']):.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out_dir / "july16_method_compare_lr0.01.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("[ok] july16_method_compare_lr0.01.tex")


def copy_analysis_figures(out_dir: Path) -> None:
    for src, dst_name in ANALYSIS_COPIES:
        if not src.is_file():
            print(f"[warn] missing analysis figure: {src}")
            continue
        shutil.copy2(src, out_dir / dst_name)
        print(f"[ok] copied {dst_name}")


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_analysis_figures(out_dir)
    write_lr_cost_table(out_dir)
    write_method_compare_table(out_dir)

    # July16 matched methods + best lr variants (plots for main three + best sorf).
    for run_id in (
        "july16_rff_lr0.01",
        "july16_orf_lr0.01",
        "july16_sorf_lr0.01",
        "july16_sorf_lr0.1",
    ):
        regenerate_s2_run(run_id, RUNS[run_id], out_dir, n_examples=3, show_bounds=False)

    regenerate_s2_run(
        "july23_s2_d400", RUNS["july23_s2_d400"], out_dir, n_examples=3, show_bounds=False
    )
    regenerate_s2_run(
        "july29_look", RUNS["july29_look"], out_dir, n_examples=3, show_bounds=True
    )
    regenerate_s2_run(
        "july29_pacbayes",
        RUNS["july29_pacbayes"],
        out_dir,
        n_examples=3,
        show_bounds=True,
    )

    # Also emit metrics-only for remaining lr sweep runs (tables already in lr csv).
    for run_id in ("july16_orf_lr0.1", "july16_orf_lr1.0", "july16_sorf_lr1.0"):
        write_metrics_table(run_id, _load_metrics(RUNS[run_id]), out_dir)

    print(f"\nDone. Figures/tables in {out_dir}")


if __name__ == "__main__":
    main()
