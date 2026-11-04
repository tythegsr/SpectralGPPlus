"""Ablate MidlineSparseMoE-GP ``k_max`` on the foundation-compare problems.

Same protocol / metrics / marginal+posterior plots as
``compare_foundation_baselines.py``, but only MidlineSparse, with one run
per ``k_max`` in a user-specified list.

Example:
  conda activate gpplus2026
  python k_max_ablation_midline.py --problems A01,A05 --k-max 2,4,6,8,10
  python k_max_ablation_midline.py --problems all --k-max 1,2,3,4,6,8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
for p in (_ROOT, _EXP, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from compare_foundation_baselines import (
    ALL_PROBLEMS,
    COMPARE_COLS,
    load_problem,
)
from foundation_adapters import fit_predict_midline_sparse
from plot_foundation_marginals import (
    probe_x_values,
    save_marginals_per_probe,
    save_posterior_vs_data,
    slice_pred,
)
from scoringbench_metrics import compute_scoringbench_style_metrics, print_metrics_table

OUT_DIR = _THIS / "results" / "k_max_ablation"
FIG_DIR = OUT_DIR / "figures"

# Distinct colors for k_max variants (tab10 / tab20)
def _kmax_color(k: int, idx: int) -> str:
    cmap = plt.cm.tab10
    return cmap(idx % 10)


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _safe_savefig(fig, out_path: Path, *, retries: int = 6) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to save {out_path}")


def _parse_problems(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() in ("all", "*"):
        return list(ALL_PROBLEMS)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in ALL_PROBLEMS]
    if bad:
        raise SystemExit(f"Unknown problems {bad}; allowed: {ALL_PROBLEMS}")
    return items


def _parse_kmax(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = int(tok)
        if k < 1:
            raise SystemExit(f"k_max must be >= 1, got {k}")
        vals.append(k)
    if not vals:
        raise SystemExit("Provide at least one k_max, e.g. --k-max 2,4,6,8")
    # unique, preserve order
    seen: set[int] = set()
    out: list[int] = []
    for k in vals:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def run_midline_kmax(
    problem: dict[str, Any],
    *,
    k_max: int,
    seed: int,
    x_probe: np.ndarray,
    x_grid: np.ndarray,
) -> dict[str, Any]:
    """Fit MidlineSparse at ``k_max``; predict on test + probes + grid."""
    x_tr, y_tr = problem["x_tr"], problem["y_tr"]
    x_te, y_te = problem["x_te"], problem["y_te"]
    Xte = x_te.detach().cpu().numpy()
    y_np = y_te.detach().cpu().numpy().ravel()
    n_te = int(Xte.shape[0])
    d = int(Xte.shape[1])

    Xp = np.asarray(x_probe, dtype=np.float64).reshape(-1, d)
    Xg = np.asarray(x_grid, dtype=np.float64).reshape(-1, d)
    X_eval = np.vstack([Xte, Xp, Xg])
    x_eval_t = torch.as_tensor(X_eval, dtype=torch.float64)
    n_probe, n_grid = int(Xp.shape[0]), int(Xg.shape[0])

    # A01 uses non-champion in foundation compare; keep problem flag
    pred_all = fit_predict_midline_sparse(
        x_tr,
        y_tr,
        x_eval_t,
        k_max=int(k_max),
        num_steps=int(problem["num_steps"]),
        seed=seed,
        champion=bool(problem["champion"]),
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
    )

    pred = slice_pred(pred_all, 0, n_te)
    pred_probe = slice_pred(pred_all, n_te, n_te + n_probe)
    pred_grid = slice_pred(pred_all, n_te + n_probe, n_te + n_probe + n_grid)

    metrics = compute_scoringbench_style_metrics(
        y_np,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
    )
    extras = dict(pred_all.get("extras") or {})
    key = f"kmax{k_max}"
    label = rf"$K_{{\max}}={k_max}$"
    row: dict[str, Any] = {
        "Model": label,
        "model_key": key,
        "k_max": int(k_max),
        "source": "live",
        **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
    }
    for k in ("k_eff", "k_modes_mean", "k_modes_round", "k_hat", "wall_s"):
        if k in extras:
            row[k] = extras[k]
    return {
        "row": row,
        "metrics": metrics,
        "pred_probe": pred_probe,
        "pred_grid": pred_grid,
        "extras": extras,
        "model_key": key,
        "label": label,
    }


def save_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model",
        "model_key",
        "k_max",
        "source",
    ] + COMPARE_COLS + [
        "k_eff",
        "k_modes_mean",
        "k_modes_round",
        "k_hat",
        "wall_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def save_metrics_table_png(rows: list[dict], out_path: Path, *, title: str) -> Path:
    col_labels = ["Model", "k_max"] + COMPARE_COLS + ["k_eff", "k_modes_round"]
    cell_text = []
    for row in rows:
        cells = [str(row.get("Model", "")), str(row.get("k_max", ""))]
        for c in COMPARE_COLS + ["k_eff", "k_modes_round"]:
            v = row.get(c)
            cells.append(
                "—"
                if v is None or (isinstance(v, float) and not np.isfinite(float(v)))
                else f"{float(v):.4f}"
            )
        cell_text.append(cells)

    fig, ax = plt.subplots(figsize=(12.5, 2.0 + 0.38 * len(rows)), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.08, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_metrics_vs_kmax(
    rows: list[dict],
    *,
    out_path: Path,
    problem_name: str,
    true_k_modes: int,
) -> Path:
    """Line plots of ScoringBench metrics (and mode stats) vs k_max."""
    valid = [
        r
        for r in rows
        if r.get("k_max") is not None and r.get("RMSE") is not None
    ]
    if not valid:
        raise ValueError("no valid rows for metrics-vs-k_max plot")
    valid = sorted(valid, key=lambda r: int(r["k_max"]))
    ks = np.array([int(r["k_max"]) for r in valid], dtype=int)

    metric_keys = [
        ("RMSE", "RMSE"),
        ("CRPS", "CRPS"),
        ("log_score", "log_score (NLL)"),
        ("NIS", "NIS"),
        ("k_eff", r"$K_{\mathrm{eff}}$"),
        ("k_modes_mean", r"$K_{\mathrm{modes}}$ (mean)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), dpi=140)
    axes_flat = axes.ravel()
    for ax, (key, ylab) in zip(axes_flat, metric_keys, strict=True):
        ys = []
        for r in valid:
            v = r.get(key)
            ys.append(float(v) if v is not None and np.isfinite(float(v)) else np.nan)
        ax.plot(ks, ys, "o-", color="#2b6cb0", lw=2.0, ms=7)
        if key in ("k_eff", "k_modes_mean"):
            ax.axhline(
                true_k_modes,
                color="#c53030",
                ls="--",
                lw=1.4,
                label=rf"true $K_{{\mathrm{{modes}}}}={true_k_modes}$",
            )
            ax.legend(fontsize=8, loc="best")
        ax.set_xlabel(r"$K_{\max}$")
        ax.set_ylabel(ylab)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{problem_name}: MidlineSparse metrics vs $K_{{\\max}}$",
        fontsize=12,
    )
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def run_problem_ablation(
    problem_name: str,
    k_max_list: list[int],
    *,
    n_samples: int = 1000,
    test_size: float = 0.2,
    seed: int = 42,
    n_probes: int = 5,
    marginal_plots: bool = True,
) -> dict[str, Any]:
    problem = load_problem(
        problem_name, n_samples=n_samples, test_size=test_size, seed=seed
    )
    y_test = problem["y_te"].detach().cpu().numpy().ravel()
    x_tr_np = problem["x_tr"].detach().cpu().numpy()
    x_te_np = problem["x_te"].detach().cpu().numpy()
    y_tr_np = problem["y_tr"].detach().cpu().numpy().ravel()
    x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=n_probes)
    x_lo = float(min(x_tr_np.min(), x_te_np.min()))
    x_hi = float(max(x_tr_np.max(), x_te_np.max()))
    x_grid = np.linspace(x_lo, x_hi, 200).reshape(-1, 1)

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    probe_preds: dict[str, dict[str, np.ndarray]] = {}
    grid_preds: dict[str, dict[str, np.ndarray]] = {}
    model_labels: dict[str, str] = {}
    model_order: list[str] = []

    # Register colors for plot_foundation_marginals MODEL_COLORS fallback via labels only;
    # inject into plot module color map for nicer colors.
    import plot_foundation_marginals as pfm

    for idx, k_max in enumerate(k_max_list):
        key = f"kmax{k_max}"
        model_order.append(key)
        model_labels[key] = rf"$K_{{\max}}={k_max}$"
        pfm.MODEL_COLORS[key] = _kmax_color(k_max, idx)

        print(
            f"[{problem_name}] MidlineSparse k_max={k_max} "
            f"(champion={problem['champion']}) ...",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            out = run_midline_kmax(
                problem,
                k_max=k_max,
                seed=seed,
                x_probe=x_probes,
                x_grid=x_grid,
            )
            rows.append(out["row"])
            results[key] = {
                "metrics": out["metrics"],
                "extras": out["extras"],
                "k_max": k_max,
            }
            probe_preds[key] = out["pred_probe"]
            grid_preds[key] = out["pred_grid"]
            m = out["metrics"]
            ex = out["extras"]
            print(
                f"  done in {time.perf_counter() - t0:.1f}s | "
                f"RMSE={m['RMSE']:.4f} CRPS={m['CRPS']:.4f} "
                f"log_score={m['log_score']:.4f} "
                f"K_eff={ex.get('k_eff', float('nan')):.2f} "
                f"K_modes~{ex.get('k_modes_mean', float('nan')):.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", flush=True)
            rows.append(
                {
                    "Model": rf"$K_{{\max}}={k_max}$",
                    "model_key": key,
                    "k_max": k_max,
                    "source": "error",
                    "error": str(exc),
                    **{c: None for c in COMPARE_COLS},
                }
            )
            results[key] = {"error": str(exc), "k_max": k_max}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{problem_name.lower()}_kmax"

    table_json = OUT_DIR / f"{stem}_metrics.json"
    table_csv = OUT_DIR / f"{stem}_metrics.csv"
    table_png = FIG_DIR / f"{stem}_metrics_table.png"
    vs_png = FIG_DIR / f"{stem}_metrics_vs_kmax.png"
    marginal_paths: list[str] = []
    posterior_png: Path | None = None

    save_csv(rows, table_csv)
    save_metrics_table_png(
        rows,
        table_png,
        title=(
            f"{problem_name} MidlineSparse $K_{{\\max}}$ ablation — "
            f"n={n_samples}, test={test_size}, seed={seed}"
        ),
    )
    try:
        save_metrics_vs_kmax(
            rows,
            out_path=vs_png,
            problem_name=problem_name,
            true_k_modes=int(problem["true_k_modes"]),
        )
        print(f"Wrote {vs_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"  metrics-vs-k_max plot skipped: {exc}")
        vs_png = None  # type: ignore[assignment]

    if marginal_plots and probe_preds:
        x_all = np.vstack([x_tr_np, x_te_np])
        y_all = np.concatenate([y_tr_np, y_test])
        m_paths = save_marginals_per_probe(
            problem_name=problem_name,
            x_probes=x_probes,
            model_preds=probe_preds,
            model_labels=model_labels,
            model_order=model_order,
            x_data=x_all,
            y_data=y_all,
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            out_dir=FIG_DIR,
            stem=stem,
        )
        for p in m_paths:
            print(f"Wrote {p}")
            marginal_paths.append(str(p))

    if marginal_plots and grid_preds:
        posterior_png = FIG_DIR / f"{stem}_posterior_vs_data.png"
        save_posterior_vs_data(
            problem_name=problem_name,
            x_train=x_tr_np,
            y_train=y_tr_np,
            x_test=x_te_np,
            y_test=y_test,
            x_grid=x_grid,
            model_preds_grid=grid_preds,
            model_labels=model_labels,
            model_order=model_order,
            out_path=posterior_png,
        )
        print(f"Wrote {posterior_png}")

    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = []
    n_tr = int(problem["x_tr"].shape[0])
    for r in rows:
        if r.get("source") == "error":
            continue
        cost_rows.append(
            make_cost_row(
                script="k_max_ablation_midline",
                problem=problem_name,
                model=str(r.get("Model") or r.get("model_key")),
                wall_s=r.get("wall_s"),
                n_train=n_tr,
                k_max=r.get("k_max"),
                num_steps=int(problem["num_steps"]),
                gate_type="mlp",
                num_inits=2,
                num_stage2_inits=1,
                champion=bool(problem["champion"]),
            )
        )
    cost_art = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR,
        stem=f"{stem}_cost",
        title=f"{problem_name} k_max ablation — fit cost",
    )

    payload = {
        "protocol": {
            "problem": problem_name,
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
            "true_k_modes": problem["true_k_modes"],
            "k_max_list": k_max_list,
            "x_probes": x_probes.tolist(),
            "midline": {
                "champion": problem["champion"],
                "num_steps": problem["num_steps"],
            },
        },
        "rows": rows,
        "results": results,
        "artifacts": {
            "json": str(table_json),
            "csv": str(table_csv),
            "table_png": str(table_png),
            "metrics_vs_kmax_png": str(vs_png) if vs_png else None,
            "marginals_png": marginal_paths,
            "posterior_png": str(posterior_png) if posterior_png else None,
            "cost": cost_art,
        },
    }
    _json_dump(table_json, payload)
    print_metrics_table(
        rows, cols=["Model"] + COMPARE_COLS + ["k_eff", "k_modes_mean"]
    )
    print(f"Wrote {table_json}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_png}")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="all",
        help=f"Comma list or 'all'. Allowed: {', '.join(ALL_PROBLEMS)}",
    )
    parser.add_argument(
        "--k-max",
        default="1,2,4,5,6,8,10",
        dest="k_max",
        help="Comma-separated k_max values to ablate (default: 2,4,6,8,10)",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-probes", type=int, default=5)
    parser.add_argument(
        "--no-marginal-plots",
        action="store_true",
        help="Skip marginal / posterior figures (metrics only)",
    )
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    problems = _parse_problems(args.problems)
    k_max_list = _parse_kmax(args.k_max)

    all_payloads: dict[str, Any] = {}
    for pname in problems:
        all_payloads[pname] = run_problem_ablation(
            pname,
            k_max_list,
            n_samples=args.n_samples,
            test_size=args.test_size,
            seed=args.seed,
            n_probes=args.n_probes,
            marginal_plots=not args.no_marginal_plots,
        )

    summary_path = OUT_DIR / "summary.json"
    _json_dump(
        summary_path,
        {
            "problems": problems,
            "k_max_list": k_max_list,
            "protocol": {
                "n_samples": args.n_samples,
                "test_size": args.test_size,
                "seed": args.seed,
            },
            "payloads": {
                k: {"rows": v["rows"], "protocol": v["protocol"]}
                for k, v in all_payloads.items()
            },
        },
    )
    print(f"Wrote {summary_path}")
    return all_payloads


if __name__ == "__main__":
    main()
