"""UQ-first beat-foundation diagnostics for MidlineSparse vs Exact GP / TabPFN.

Runs (foundation protocol: n=1000, test_size=0.2, seed=42):
  1. Stage-1-only Midline vs Stage-2 champion vs Exact GP vs TabPFN-3.5
     on A01, unimodal, A04, A05 — primary CRPS / NIS / coverage; log secondary.
  2. PIT + coverage figures.
  3. Conditional marginals with Midline expert annotation.
  4. Loss stress (champion / nll_only / force_merge / kmax1) on unimodal + A05.
  5. Writes results/beat_foundation_uq/ + memo-ready summary.json.

Example:
  conda activate gpplus2026
  python run_beat_foundation_uq.py
  python run_beat_foundation_uq.py --problems unimodal,A05 --skip-tabpfn
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
from compare_foundation_baselines import load_problem
from foundation_adapters import (
    fit_predict_exact_gp,
    fit_predict_midline_sparse,
    fit_predict_tabpfn,
)
from plot_foundation_marginals import (
    probe_x_values,
    save_marginals_per_probe,
    slice_pred,
)
from uq_calibration import (
    compute_uq_metrics,
    mixture_cdf_at_y,
    save_pit_coverage_figure,
)

OUT_DIR = _THIS / "results" / "beat_foundation_uq"
FIG_DIR = OUT_DIR / "figures"

STAGE_PROBLEMS = ("A01", "unimodal", "A04", "A05")
STRESS_PROBLEMS = ("unimodal", "A05")

UQ_COLS = [
    "CRPS",
    "NIS",
    "cov50",
    "cov90",
    "width90",
    "RMSE",
    "pit_ks",
    "log_score",
]

MODEL_LABELS = {
    "midline_stage1": "Midline Stage-1 (mu0,sigma0)",
    "midline_stage2": "Midline Stage-2 champion",
    "exact_gp": "Exact GP+ (GPR)",
    "tabpfn35": "TabPFN-3.5",
    "champion": "champion (lean: usage+merge+alive+delta+alpha)",
    "nll_only": "nll_only (structure+orphan/delta/alpha/alive off)",
    "force_merge": "force_merge (merge_p0=1)",
    "kmax1": "kmax1 (K_max=1)",
}


def _json_dump(path: Path, obj: Any) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _slice_mixture(pred: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Slice mixture + TabPFN/Kumo PPD fields for probes/test."""
    return slice_pred(pred, start, stop)


def _row_from_uq(
    *,
    model_key: str,
    uq: dict[str, float],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Model": MODEL_LABELS.get(model_key, model_key),
        "model_key": model_key,
        **{c: float(uq[c]) for c in UQ_COLS if c in uq},
    }
    extras = extras or {}
    for k in (
        "k_eff",
        "k_modes_mean",
        "k_modes_round",
        "merge_pressure",
        "stage1_gp_nll",
        "wall_s",
        "k_max",
    ):
        if k in extras:
            row[k] = extras[k]
    return row


def _eval_pred(
    y: np.ndarray, pred: dict[str, np.ndarray]
) -> tuple[dict[str, float], np.ndarray]:
    uq = compute_uq_metrics(
        y,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
    )
    pit = mixture_cdf_at_y(
        y, pred["expert_means"], pred["expert_stds"], pred["gate_weights"]
    )
    return uq, pit


def _fit_midline_bundle(
    problem: dict[str, Any],
    x_eval_t: torch.Tensor,
    *,
    seed: int,
    k_max: int | None = None,
    merge_pressure_init: float = 0.0,
    lambda_overrides: dict[str, Any] | None = None,
    adaptive_merge_prune: bool = True,
) -> dict[str, Any]:
    return fit_predict_midline_sparse(
        problem["x_tr"],
        problem["y_tr"],
        x_eval_t,
        k_max=int(k_max if k_max is not None else problem["k_max"]),
        num_steps=int(problem["num_steps"]),
        seed=seed,
        champion=bool(problem["champion"]),
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
        include_stage1=True,
        merge_pressure_init=merge_pressure_init,
        lambda_overrides=lambda_overrides,
        adaptive_merge_prune=adaptive_merge_prune,
    )


def run_stage_compare(
    problem_name: str,
    *,
    seed: int = 42,
    n_samples: int = 1000,
    test_size: float = 0.2,
    n_probes: int = 5,
    skip_tabpfn: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    problem = load_problem(
        problem_name, n_samples=n_samples, test_size=test_size, seed=seed
    )
    y_te = problem["y_te"].detach().cpu().numpy().ravel()
    x_tr_np = problem["x_tr"].detach().cpu().numpy()
    x_te_np = problem["x_te"].detach().cpu().numpy()
    y_tr_np = problem["y_tr"].detach().cpu().numpy().ravel()
    x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=n_probes)
    d = int(x_te_np.shape[1])
    Xp = np.asarray(x_probes, dtype=np.float64).reshape(-1, d)
    X_eval = np.vstack([x_te_np, Xp])
    x_eval_t = torch.as_tensor(X_eval, dtype=torch.float64)
    n_te = int(x_te_np.shape[0])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict[str, Any]] = []
    probe_preds: dict[str, dict[str, np.ndarray]] = {}
    pits: dict[str, np.ndarray] = {}
    cov_rows: dict[str, dict[str, float]] = {}
    results: dict[str, Any] = {}

    # --- Midline: one fit → Stage-2 + Stage-1 ---
    print(f"[{problem_name}] MidlineSparse (Stage-1 + Stage-2) ...", flush=True)
    t0 = time.perf_counter()
    mid = _fit_midline_bundle(problem, x_eval_t, seed=seed)
    print(f"  Midline wall {time.perf_counter() - t0:.1f}s", flush=True)

    for mk, pred_all in (
        ("midline_stage2", mid),
        ("midline_stage1", mid["stage1"]),
    ):
        pred = _slice_mixture(pred_all, 0, n_te)
        pred_probe = _slice_mixture(pred_all, n_te, n_te + len(x_probes))
        uq, pit = _eval_pred(y_te, pred)
        extras = dict(mid.get("extras") or {}) if mk == "midline_stage2" else {
            "stage1_gp_nll": (mid.get("extras") or {}).get("stage1_gp_nll"),
            "k_max": 1,
            "k_eff": 1.0,
            "k_modes_mean": 1.0,
            "k_modes_round": 1,
        }
        row = _row_from_uq(model_key=mk, uq=uq, extras=extras)
        rows.append(row)
        results[mk] = {"metrics": uq, "extras": extras}
        probe_preds[mk] = pred_probe
        pits[mk] = pit
        cov_rows[mk] = uq
        print(
            f"  {MODEL_LABELS[mk]}: CRPS={uq['CRPS']:.4f} cov90={uq['cov90']:.3f} "
            f"NIS={uq['NIS']:.4f} log={uq['log_score']:.4f}",
            flush=True,
        )

    # --- Exact GP ---
    print(f"[{problem_name}] Exact GP ...", flush=True)
    t0 = time.perf_counter()
    eg = fit_predict_exact_gp(
        problem["x_tr"], problem["y_tr"], x_eval_t, seed=seed, device="cpu"
    )
    print(f"  Exact GP wall {time.perf_counter() - t0:.1f}s", flush=True)
    pred = _slice_mixture(eg, 0, n_te)
    pred_probe = _slice_mixture(eg, n_te, n_te + len(x_probes))
    uq, pit = _eval_pred(y_te, pred)
    row = _row_from_uq(model_key="exact_gp", uq=uq, extras=eg.get("extras"))
    rows.append(row)
    results["exact_gp"] = {"metrics": uq, "extras": eg.get("extras")}
    probe_preds["exact_gp"] = pred_probe
    pits["exact_gp"] = pit
    cov_rows["exact_gp"] = uq
    print(
        f"  Exact GP: CRPS={uq['CRPS']:.4f} cov90={uq['cov90']:.3f} "
        f"NIS={uq['NIS']:.4f}",
        flush=True,
    )

    # --- TabPFN-3.5 ---
    if not skip_tabpfn:
        print(f"[{problem_name}] TabPFN-3.5 ...", flush=True)
        t0 = time.perf_counter()
        try:
            tp = fit_predict_tabpfn(
                x_tr_np,
                y_tr_np,
                X_eval,
                version="v3.5",
                seed=seed,
                device=device,
            )
            print(f"  TabPFN wall {time.perf_counter() - t0:.1f}s", flush=True)
            pred = _slice_mixture(tp, 0, n_te)
            pred_probe = _slice_mixture(tp, n_te, n_te + len(x_probes))
            uq, pit = _eval_pred(y_te, pred)
            row = _row_from_uq(model_key="tabpfn35", uq=uq, extras=tp.get("extras"))
            rows.append(row)
            results["tabpfn35"] = {"metrics": uq, "extras": tp.get("extras")}
            probe_preds["tabpfn35"] = pred_probe
            pits["tabpfn35"] = pit
            cov_rows["tabpfn35"] = uq
            print(
                f"  TabPFN-3.5: CRPS={uq['CRPS']:.4f} cov90={uq['cov90']:.3f} "
                f"NIS={uq['NIS']:.4f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  TabPFN FAILED: {exc}", flush=True)
            rows.append(
                {
                    "Model": MODEL_LABELS["tabpfn35"],
                    "model_key": "tabpfn35",
                    "error": str(exc),
                    **{c: None for c in UQ_COLS},
                }
            )
            results["tabpfn35"] = {"error": str(exc)}

    order = [
        k
        for k in (
            "midline_stage1",
            "midline_stage2",
            "exact_gp",
            "tabpfn35",
        )
        if k in probe_preds
    ]

    stem = f"{problem_name.lower()}_stage"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pit_path = FIG_DIR / f"{stem}_pit_coverage.png"
    save_pit_coverage_figure(
        problem_name=problem_name,
        model_pits=pits,
        model_labels=MODEL_LABELS,
        model_order=order,
        coverage_rows=cov_rows,
        out_path=pit_path,
    )
    print(f"Wrote {pit_path}", flush=True)

    x_all = np.vstack([x_tr_np, x_te_np])
    y_all = np.concatenate([y_tr_np, y_te])
    m_paths = save_marginals_per_probe(
        problem_name=problem_name,
        x_probes=x_probes,
        model_preds=probe_preds,
        model_labels=MODEL_LABELS,
        model_order=order,
        x_data=x_all,
        y_data=y_all,
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
        out_dir=FIG_DIR,
        stem=stem,
        annotate_experts_for=["midline_stage2"],
    )
    for p in m_paths:
        print(f"Wrote {p}", flush=True)

    table_png = FIG_DIR / f"{stem}_metrics_table.png"
    _save_uq_table_png(
        rows,
        table_png,
        title=f"{problem_name} Stage-1 vs Stage-2 vs FM (UQ-first)",
    )
    print(f"Wrote {table_png}", flush=True)

    csv_path = OUT_DIR / f"{stem}_metrics.csv"
    _save_csv(rows, csv_path)

    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = [
        make_cost_row(
            script="run_beat_foundation_uq.stage",
            problem=problem_name,
            model=str(r.get("Model") or r.get("model_key")),
            wall_s=r.get("wall_s"),
            n_train=int(problem["x_tr"].shape[0]),
            k_max=int(problem["k_max"])
            if str(r.get("model_key", "")).startswith("midline")
            else None,
            num_steps=int(problem.get("num_steps", 800))
            if str(r.get("model_key", "")).startswith("midline")
            else None,
            gate_type="mlp"
            if str(r.get("model_key", "")).startswith("midline")
            else None,
            notes=str(r.get("model_key") or ""),
        )
        for r in rows
        if r.get("CRPS") is not None
    ]
    cost_art = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR,
        stem=f"{stem}_cost",
        title=f"{problem_name} UQ stage compare — fit cost",
    )

    payload = {
        "problem": problem_name,
        "kind": "stage_compare",
        "protocol": {
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
            "true_k_modes": problem["true_k_modes"],
            "k_max": problem["k_max"],
            "champion": problem["champion"],
        },
        "rows": rows,
        "results": results,
        "artifacts": {
            "pit_coverage": str(pit_path),
            "marginals": [str(p) for p in m_paths],
            "table_png": str(table_png),
            "csv": str(csv_path),
            "cost": cost_art,
        },
        "verdict": _stage_verdict(problem_name, rows),
    }
    json_path = OUT_DIR / f"{stem}_metrics.json"
    _json_dump(json_path, payload)
    print(f"Wrote {json_path}", flush=True)
    print(f"  VERDICT: {payload['verdict']}", flush=True)
    return payload


def _stage_verdict(problem_name: str, rows: list[dict[str, Any]]) -> str:
    by = {r["model_key"]: r for r in rows if r.get("CRPS") is not None}

    def crps(k: str) -> float:
        return float(by[k]["CRPS"]) if k in by else float("inf")

    def cov90(k: str) -> float:
        return float(by[k]["cov90"]) if k in by and by[k].get("cov90") is not None else float("nan")

    s1, s2 = crps("midline_stage1"), crps("midline_stage2")
    eg, tp = crps("exact_gp"), crps("tabpfn35")
    best_fm = min(eg, tp)

    if not np.isfinite(s1) or not np.isfinite(s2):
        return "inconclusive (missing Midline scores)"

    # Stage-2 hurts UQ vs Stage-1 by a meaningful margin
    if s2 > s1 * 1.05 and s2 > s1 + 0.01:
        if abs(s1 - eg) / max(eg, 1e-9) < 0.08:
            return (
                f"{problem_name}: Stage-2 worsens CRPS vs Stage-1≈ExactGP "
                f"→ loss / anti-merge (γ/rec/sep) hurting UQ"
            )
        return (
            f"{problem_name}: Stage-2 worsens CRPS vs Stage-1; "
            f"Stage-1 also off ExactGP → check both loss and Stage-1 noise"
        )

    # Stage-1 already loses to FMs
    if s1 > best_fm * 1.08 and s1 > best_fm + 0.005:
        return (
            f"{problem_name}: Stage-1 CRPS lags ExactGP/TabPFN "
            f"→ Stage-1 / noise / likelihood family (not λ_rec/sep)"
        )

    # Stage-2 helps or matches and beats / matches FMs on UQ
    if s2 <= best_fm * 1.03:
        c90 = cov90("midline_stage2")
        cal = (
            "coverage OK"
            if np.isfinite(c90) and abs(c90 - 0.9) < 0.08
            else f"coverage check (cov90={c90:.2f})"
        )
        return (
            f"{problem_name}: Stage-2 UQ competitive with FMs ({cal}); "
            f"leave champion λs; do not chase log"
        )

    return (
        f"{problem_name}: Stage-2 helps vs Stage-1 but still lags FMs on CRPS "
        f"→ mixture capacity / predictive shape vs TabPFN"
    )


LOSS_VARIANTS: dict[str, dict[str, Any]] = {
    "champion": {
        "k_max": None,
        "merge_pressure_init": 0.0,
        "lambda_overrides": None,
    },
    "nll_only": {
        "k_max": None,
        "merge_pressure_init": 0.0,
        "lambda_overrides": {
            "lambda_rec": 0.0,
            "lambda_sep": 0.0,
            "lambda_overlap": 0.0,
            "lambda_gamma": 0.0,
            "lambda_orphan": 0.0,
            "lambda_delta": 0.0,
            "lambda_alpha": 0.0,
            "alive_weight": 0.0,
            "lambda_k": 0.0,
            "lambda_k_start": 0.0,
        },
    },
    "force_merge": {
        "k_max": None,
        "merge_pressure_init": 1.0,
        "lambda_overrides": None,
    },
    "kmax1": {
        "k_max": 1,
        "merge_pressure_init": 0.0,
        "lambda_overrides": None,
        "adaptive_merge_prune": False,
    },
}


def run_loss_stress(
    problem_name: str,
    *,
    seed: int = 42,
    n_samples: int = 1000,
    test_size: float = 0.2,
    skip_tabpfn: bool = False,
    device: str | None = None,
    fm_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem = load_problem(
        problem_name, n_samples=n_samples, test_size=test_size, seed=seed
    )
    y_te = problem["y_te"].detach().cpu().numpy().ravel()
    x_te_np = problem["x_te"].detach().cpu().numpy()
    x_tr_np = problem["x_tr"].detach().cpu().numpy()
    y_tr_np = problem["y_tr"].detach().cpu().numpy().ravel()
    x_eval_t = torch.as_tensor(x_te_np, dtype=torch.float64)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for vname, cfg in LOSS_VARIANTS.items():
        print(f"[{problem_name}] loss stress: {vname} ...", flush=True)
        t0 = time.perf_counter()
        km = cfg["k_max"]
        mid = fit_predict_midline_sparse(
            problem["x_tr"],
            problem["y_tr"],
            x_eval_t,
            k_max=int(km if km is not None else problem["k_max"]),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            include_stage1=False,
            merge_pressure_init=float(cfg["merge_pressure_init"]),
            lambda_overrides=cfg.get("lambda_overrides"),
            adaptive_merge_prune=bool(cfg.get("adaptive_merge_prune", True)),
        )
        uq, _pit = _eval_pred(y_te, mid)
        extras = dict(mid.get("extras") or {})
        extras["variant"] = vname
        extras["wall_s"] = time.perf_counter() - t0
        row = _row_from_uq(model_key=vname, uq=uq, extras=extras)
        rows.append(row)
        results[vname] = {"metrics": uq, "extras": extras}
        print(
            f"  {vname}: CRPS={uq['CRPS']:.4f} cov90={uq['cov90']:.3f} "
            f"k_modes={extras.get('k_modes_mean')} merge_p={extras.get('merge_pressure')}",
            flush=True,
        )

    # FM reference rows (reuse stage-compare cache when available)
    for mk, label_key in (("exact_gp", "exact_gp"), ("tabpfn35", "tabpfn35")):
        if fm_cache and mk in fm_cache and "metrics" in fm_cache[mk]:
            uq = fm_cache[mk]["metrics"]
            rows.append(_row_from_uq(model_key=mk, uq=uq, extras=fm_cache[mk].get("extras")))
            results[mk] = fm_cache[mk]
            continue
        if mk == "tabpfn35" and skip_tabpfn:
            continue
        print(f"[{problem_name}] loss stress FM ref: {mk} ...", flush=True)
        if mk == "exact_gp":
            pred = fit_predict_exact_gp(
                problem["x_tr"], problem["y_tr"], x_eval_t, seed=seed, device="cpu"
            )
        else:
            try:
                pred = fit_predict_tabpfn(
                    x_tr_np,
                    y_tr_np,
                    x_te_np,
                    version="v3.5",
                    seed=seed,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {exc}", flush=True)
                continue
        uq, _ = _eval_pred(y_te, pred)
        rows.append(_row_from_uq(model_key=mk, uq=uq, extras=pred.get("extras")))
        results[mk] = {"metrics": uq, "extras": pred.get("extras")}

    stem = f"{problem_name.lower()}_loss_stress"
    table_png = FIG_DIR / f"{stem}_metrics_table.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _save_uq_table_png(
        rows,
        table_png,
        title=f"{problem_name} loss stress (judge by CRPS / cov90)",
    )
    csv_path = OUT_DIR / f"{stem}_metrics.csv"
    _save_csv(rows, csv_path)
    payload = {
        "problem": problem_name,
        "kind": "loss_stress",
        "rows": rows,
        "results": results,
        "artifacts": {"table_png": str(table_png), "csv": str(csv_path)},
        "verdict": _loss_stress_verdict(problem_name, rows),
    }
    json_path = OUT_DIR / f"{stem}_metrics.json"
    _json_dump(json_path, payload)
    print(f"Wrote {json_path}", flush=True)
    print(f"  VERDICT: {payload['verdict']}", flush=True)
    return payload


def _loss_stress_verdict(problem_name: str, rows: list[dict[str, Any]]) -> str:
    by = {r["model_key"]: r for r in rows if r.get("CRPS") is not None}
    if "champion" not in by:
        return f"{problem_name}: incomplete stress table"

    champ = float(by["champion"]["CRPS"])
    best_var = min(
        (k for k in ("champion", "nll_only", "force_merge", "kmax1") if k in by),
        key=lambda k: float(by[k]["CRPS"]),
    )
    best_crps = float(by[best_var]["CRPS"])
    fm_keys = [k for k in ("exact_gp", "tabpfn35") if k in by]
    best_fm = min((float(by[k]["CRPS"]) for k in fm_keys), default=float("inf"))

    if best_var != "champion" and best_crps < champ * 0.97:
        closes = best_crps <= best_fm * 1.05 if np.isfinite(best_fm) else False
        return (
            f"{problem_name}: {best_var} improves CRPS vs champion "
            f"({'closes FM gap' if closes else 'still lags FM'}) "
            f"→ weaken rec/γ when merge signal is low"
        )
    if np.isfinite(best_fm) and champ > best_fm * 1.08:
        return (
            f"{problem_name}: no loss variant closes FM CRPS gap "
            f"→ Stage-1 / noise model is the bottleneck"
        )
    return (
        f"{problem_name}: champion UQ already near best among variants "
        f"(and near FMs); keep λ stack"
    )


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Model", "model_key"] + UQ_COLS + [
        "k_eff",
        "k_modes_mean",
        "k_modes_round",
        "merge_pressure",
        "stage1_gp_nll",
        "wall_s",
        "k_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _save_uq_table_png(rows: list[dict], out_path: Path, *, title: str) -> Path:
    extra = ["wall_s"] if any(r.get("wall_s") is not None for r in rows) else []
    metric_cols = UQ_COLS + extra
    cols = ["Model"] + metric_cols
    cell_text = []
    for row in rows:
        cells = [str(row.get("Model", ""))]
        for c in metric_cols:
            v = row.get(c)
            if v is None or not np.isfinite(float(v)):
                cells.append("—")
            elif c == "wall_s":
                cells.append(f"{float(v):.1f}")
            else:
                cells.append(f"{float(v):.4f}")
        cell_text.append(cells)
    fig_w = max(14.0, 1.2 * len(cols) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 1.8 + 0.38 * len(rows)), dpi=140)
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
        elif r >= 1:
            mk = rows[r - 1].get("model_key", "")
            if "midline" in str(mk) or mk in ("champion", "nll_only", "force_merge", "kmax1"):
                cell.set_facecolor("#ebf8ff")
            elif r % 2 == 0:
                cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def write_memo_section(summary: dict[str, Any]) -> Path:
    """Append §8.5 beat-foundation UQ diagnostics to FEASIBILITY_MEMO.md."""
    memo = _THIS / "FEASIBILITY_MEMO.md"
    text = memo.read_text(encoding="utf-8")
    marker = "### 8.5 Beat-foundation UQ diagnostics"
    block_lines = [
        "",
        marker,
        "",
        "UQ-first comparison of MidlineSparse Stage-1 / Stage-2 against Exact GP "
        "and TabPFN-3.5 (protocol aligned with foundation compare: "
        "`n=1000`, `test_size=0.2`, seed 42).",
        "",
        "**Primary metrics:** CRPS, NIS, cov50/cov90, PIT. "
        "`log_score` is secondary only.",
        "",
        "```bash",
        "python run_beat_foundation_uq.py",
        "# → results/beat_foundation_uq/",
        "```",
        "",
        "#### Stage-1 vs Stage-2 vs FM (CRPS / cov90)",
        "",
        "| Problem | Stage-1 | Stage-2 | ExactGP | TabPFN | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for pname in STAGE_PROBLEMS:
        payload = summary.get("stage_compare", {}).get(pname)
        if not payload:
            block_lines.append(f"| {pname} | — | — | — | — | (not run) |")
            continue
        by = {r["model_key"]: r for r in payload.get("rows", [])}

        def _cell(k: str) -> str:
            r = by.get(k) or {}
            c, cov = r.get("CRPS"), r.get("cov90")
            if c is None or not np.isfinite(float(c)):
                return "—"
            if cov is None or not np.isfinite(float(cov)):
                return f"CRPS {float(c):.4f}"
            return f"CRPS {float(c):.4f} / cov90 {float(cov):.2f}"

        block_lines.append(
            f"| {pname} | {_cell('midline_stage1')} | {_cell('midline_stage2')} | "
            f"{_cell('exact_gp')} | {_cell('tabpfn35')} | {payload.get('verdict', '')} |"
        )

    block_lines.extend(
        [
            "",
            "Full numeric tables (CRPS, NIS, cov50/90, RMSE, PIT, log footnote) live in "
            "`results/beat_foundation_uq/*_stage_metrics.csv`.",
            "",
            "#### Loss stress (unimodal, A05) — CRPS / cov90",
            "",
            "| Problem | champion | nll_only | force_merge | kmax1 | ExactGP | TabPFN | Verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for pname in STRESS_PROBLEMS:
        payload = summary.get("loss_stress", {}).get(pname)
        if not payload:
            block_lines.append(f"| {pname} | — | — | — | — | — | — | (not run) |")
            continue
        by = {r["model_key"]: r for r in payload.get("rows", [])}

        def _cell(k: str) -> str:
            r = by.get(k) or {}
            c, cov = r.get("CRPS"), r.get("cov90")
            if c is None or not np.isfinite(float(c)):
                return "—"
            if cov is None or not np.isfinite(float(cov)):
                return f"{float(c):.4f}"
            return f"{float(c):.4f} / {float(cov):.2f}"

        block_lines.append(
            f"| {pname} | {_cell('champion')} | {_cell('nll_only')} | {_cell('force_merge')} | "
            f"{_cell('kmax1')} | {_cell('exact_gp')} | {_cell('tabpfn35')} | "
            f"{payload.get('verdict', '')} |"
        )

    block_lines.extend(
        [
            "",
            "Artifacts: `results/beat_foundation_uq/*_stage_*`, "
            "`*_loss_stress_*`, PIT/coverage PNGs, expert-annotated marginals.",
            "",
        ]
    )
    new_block = "\n".join(block_lines)

    if marker in text:
        # Replace existing section through next ### or Appendix
        start = text.index(marker)
        rest = text[start:]
        end_rel = len(rest)
        for stop in ("\n### ", "\n## "):
            idx = rest.find(stop, len(marker))
            if idx != -1:
                end_rel = min(end_rel, idx)
        text = text[:start].rstrip() + "\n" + new_block + text[start + end_rel :]
    else:
        # Insert before Appendix
        app = "\n## Appendix: how to reproduce"
        if app in text:
            text = text.replace(app, new_block + app, 1)
        else:
            text = text.rstrip() + "\n" + new_block

    memo.write_text(text, encoding="utf-8")
    return memo


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="A01,unimodal,A04,A05",
        help="Comma list for stage compare",
    )
    parser.add_argument(
        "--stress-problems",
        default="unimodal,A05",
        help="Comma list for loss stress",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-probes", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Skip TabPFN (Exact GP + Midline only)",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip loss-stress block",
    )
    parser.add_argument(
        "--skip-memo",
        action="store_true",
        help="Do not rewrite FEASIBILITY_MEMO §8.5",
    )
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    stress = [p.strip() for p in args.stress_problems.split(",") if p.strip()]

    summary: dict[str, Any] = {
        "protocol": {
            "n_samples": args.n_samples,
            "test_size": args.test_size,
            "seed": args.seed,
            "primary": "CRPS/NIS/coverage/PIT",
            "secondary": "log_score",
        },
        "stage_compare": {},
        "loss_stress": {},
    }

    for pname in problems:
        payload = run_stage_compare(
            pname,
            seed=args.seed,
            n_samples=args.n_samples,
            test_size=args.test_size,
            n_probes=args.n_probes,
            skip_tabpfn=args.skip_tabpfn,
            device=args.device,
        )
        summary["stage_compare"][pname] = {
            "rows": payload["rows"],
            "verdict": payload["verdict"],
            "results": payload["results"],
            "artifacts": payload["artifacts"],
        }

    if not args.skip_stress:
        for pname in stress:
            fm_cache = summary["stage_compare"].get(pname, {}).get("results")
            payload = run_loss_stress(
                pname,
                seed=args.seed,
                n_samples=args.n_samples,
                test_size=args.test_size,
                skip_tabpfn=args.skip_tabpfn,
                device=args.device,
                fm_cache=fm_cache,
            )
            summary["loss_stress"][pname] = {
                "rows": payload["rows"],
                "verdict": payload["verdict"],
                "artifacts": payload["artifacts"],
            }

    summary_path = OUT_DIR / "summary.json"
    _json_dump(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)

    if not args.skip_memo:
        memo_path = write_memo_section(summary)
        print(f"Updated {memo_path} §8.5", flush=True)

    return summary


if __name__ == "__main__":
    main()
