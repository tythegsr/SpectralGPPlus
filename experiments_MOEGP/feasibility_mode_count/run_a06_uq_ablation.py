"""A06 MidlineSparse UQ ablation: k_max grid at fixed Stage-2 budget.

Protocol (shared with foundation compare):
  n_train=800, n_test=1000, seed=42 (fixed test).

Grid:
  k_max ∈ {2, 4, 6, 8, 10}, num_steps=2400, champion λ stack.

Also fits Exact GP+ and TabPFN-3.5 once as reference rows.

Example:
  conda activate gpplus2026
  python run_a06_uq_ablation.py
  python run_a06_uq_ablation.py --k-max 2,6,10 --skip-tabpfn
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
from compare_foundation_baselines import load_problem_pool, slice_train
from foundation_adapters import (
    fit_predict_exact_gp,
    fit_predict_midline_sparse,
    fit_predict_tabpfn,
)
from uq_calibration import compute_uq_metrics

OUT_DIR = _THIS / "results" / "a06_uq_ablation"
FIG_DIR = OUT_DIR / "figures"

DEFAULT_K_MAX = (2, 4, 6, 8, 10)
DEFAULT_NUM_STEPS = 2400
DEFAULT_N_TRAIN = 800
DEFAULT_N_TEST = 1000

UQ_COLS = ["RMSE", "CRPS", "NIS", "log_score", "cov50", "cov90", "width90"]


def _json_dump(path: Path, obj: Any) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _eval_row(
    *,
    model_key: str,
    label: str,
    y: np.ndarray,
    pred: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uq = compute_uq_metrics(
        y,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
    )
    extras = extras or {}
    row: dict[str, Any] = {
        "Model": label,
        "model_key": model_key,
        **{c: float(uq[c]) for c in UQ_COLS if c in uq},
        "wall_s": extras.get("wall_s"),
        "k_max": extras.get("k_max"),
        "k_eff": extras.get("k_eff"),
        "k_modes_mean": extras.get("k_modes_mean"),
        "num_steps": extras.get("num_steps"),
    }
    return row


def _save_metrics_table(rows: list[dict[str, Any]], out_path: Path, title: str) -> Path:
    cols = ["Model", "k_max", "RMSE", "CRPS", "NIS", "log_score", "cov50", "cov90", "wall_s"]
    cell_text = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
            elif c in ("Model",):
                cells.append(str(v))
            elif c in ("k_max",):
                cells.append(str(int(v)) if v is not None else "—")
            elif c == "wall_s":
                cells.append(f"{float(v):.1f}")
            elif c in ("cov50", "cov90"):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(f"{float(v):.4f}")
        cell_text.append(cells)

    fig, ax = plt.subplots(figsize=(14.0, 1.8 + 0.38 * len(rows)), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r >= 1 and str(rows[r - 1].get("model_key", "")).startswith("midline_k"):
            cell.set_facecolor("#ebf8ff")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def _parse_ints(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals or any(v < 1 for v in vals):
        raise SystemExit(f"Invalid int list {raw!r}")
    return vals


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-max",
        default=",".join(str(k) for k in DEFAULT_K_MAX),
        help="Comma list of k_max values (default: 2,4,6,8,10)",
    )
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--skip-exact-gp", action="store_true")
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    k_list = _parse_ints(args.k_max)
    n_train = int(args.n_train)
    n_test = int(args.n_test)
    num_steps = int(args.num_steps)
    seed = int(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pool = load_problem_pool("A06", n_train_max=n_train, n_test=n_test, seed=seed)
    problem = slice_train(pool, n_train)
    y_te = problem["y_te"].detach().cpu().numpy().ravel()
    x_te = problem["x_te"]

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    if not args.skip_exact_gp:
        print("[A06] Exact GP+ ...", flush=True)
        t0 = time.perf_counter()
        eg = fit_predict_exact_gp(
            problem["x_tr"], problem["y_tr"], x_te, seed=seed, device="cpu"
        )
        wall = time.perf_counter() - t0
        extras = dict(eg.get("extras") or {})
        extras.setdefault("wall_s", wall)
        extras["k_max"] = 1
        extras["num_steps"] = None
        row = _eval_row(
            model_key="exact_gp",
            label="Exact GP+ (GPR)",
            y=y_te,
            pred=eg,
            extras=extras,
        )
        rows.append(row)
        results["exact_gp"] = {"metrics": {c: row.get(c) for c in UQ_COLS}, "extras": extras}
        print(
            f"  Exact GP: CRPS={row['CRPS']:.4f} cov90={row['cov90']:.3f} "
            f"wall={wall:.1f}s",
            flush=True,
        )

    if not args.skip_tabpfn:
        print("[A06] TabPFN-3.5 ...", flush=True)
        t0 = time.perf_counter()
        try:
            tp = fit_predict_tabpfn(
                problem["x_tr"].detach().cpu().numpy(),
                problem["y_tr"].detach().cpu().numpy().ravel(),
                problem["x_te"].detach().cpu().numpy(),
                version="v3.5",
                seed=seed,
                device=device,
            )
            wall = time.perf_counter() - t0
            extras = dict(tp.get("extras") or {})
            extras.setdefault("wall_s", wall)
            extras["k_max"] = None
            extras["num_steps"] = None
            row = _eval_row(
                model_key="tabpfn35",
                label="TabPFN-3.5",
                y=y_te,
                pred=tp,
                extras=extras,
            )
            rows.append(row)
            results["tabpfn35"] = {
                "metrics": {c: row.get(c) for c in UQ_COLS},
                "extras": extras,
            }
            print(
                f"  TabPFN: CRPS={row['CRPS']:.4f} cov90={row['cov90']:.3f} "
                f"wall={wall:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  TabPFN FAILED: {exc}", flush=True)
            rows.append(
                {
                    "Model": "TabPFN-3.5",
                    "model_key": "tabpfn35",
                    "error": str(exc),
                    **{c: None for c in UQ_COLS},
                }
            )

    for k_max in k_list:
        mk = f"midline_k{k_max}"
        label = f"2-Stage Mixture GP (K={k_max}, steps={num_steps})"
        print(f"[A06] MidlineSparse k_max={k_max}, num_steps={num_steps} ...", flush=True)
        t0 = time.perf_counter()
        mid = fit_predict_midline_sparse(
            problem["x_tr"],
            problem["y_tr"],
            x_te,
            k_max=int(k_max),
            num_steps=num_steps,
            seed=seed,
            champion=True,
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
        )
        wall = time.perf_counter() - t0
        extras = dict(mid.get("extras") or {})
        extras.setdefault("wall_s", wall)
        extras["k_max"] = int(k_max)
        extras["num_steps"] = int(num_steps)
        row = _eval_row(
            model_key=mk,
            label=label,
            y=y_te,
            pred=mid,
            extras=extras,
        )
        rows.append(row)
        results[mk] = {"metrics": {c: row.get(c) for c in UQ_COLS}, "extras": extras}
        print(
            f"  K={k_max}: CRPS={row['CRPS']:.4f} cov90={row['cov90']:.3f} "
            f"k_eff={row.get('k_eff')} wall={wall:.1f}s",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"a06_n{n_train}_steps{num_steps}_kgrid"
    csv_path = OUT_DIR / f"{stem}_metrics.csv"
    json_path = OUT_DIR / f"{stem}_metrics.json"
    png_path = FIG_DIR / f"{stem}_metrics_table.png"

    fieldnames = [
        "Model",
        "model_key",
        "k_max",
        "num_steps",
        *UQ_COLS,
        "wall_s",
        "k_eff",
        "k_modes_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {
        "protocol": {
            "problem": "A06",
            "n_train": n_train,
            "n_test": n_test,
            "test_shared": True,
            "seed": seed,
            "num_steps": num_steps,
            "k_max_grid": k_list,
            "champion": True,
        },
        "rows": rows,
        "results": results,
    }
    _json_dump(json_path, payload)
    _save_metrics_table(
        rows,
        png_path,
        title=(
            f"A06 UQ ablation — n_train={n_train}, n_test={n_test}, "
            f"num_steps={num_steps}, seed={seed}"
        ),
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    return payload


if __name__ == "__main__":
    main()
