"""Stage-2 loss / NLL trajectories + step-budget sweep (UQ-first).

Answers: is Stage-2 still improving at the default step count, or plateaus earlier?

For each problem (default: unimodal, A05, A01):
  1. Fit champion MidlineSparse with ``num_steps=max(budgets)`` and dense logging.
  2. Plot train loss, NLL, merge_pressure, K_eff vs step.
  3. Score held-out CRPS / cov90 at checkpoints in ``budgets`` by replaying
     history (refit at each budget — honest comparison).

Example:
  conda activate gpplus2026
  python run_stage2_training_curves.py
  python run_stage2_training_curves.py --problems unimodal,A05 --budgets 400,800,1600,2400
"""

from __future__ import annotations

import argparse
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
from uq_calibration import compute_uq_metrics

OUT_DIR = _THIS / "results" / "stage2_training_curves"
FIG_DIR = OUT_DIR / "figures"


def _json_dump(path: Path, obj: Any) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _slim_history(hist: list[dict]) -> list[dict]:
    keep = (
        "step",
        "loss",
        "nll",
        "k_eff",
        "merge_pressure",
        "k_hat",
        "lambda_k_t",
        "lambda_rec_t",
        "usage_sum",
        "rec_pen",
        "gamma_pen",
    )
    return [{k: row[k] for k in keep if k in row} for row in hist]


def _tail_slope(hist: list[dict], *, frac: float = 0.25) -> dict[str, float]:
    """Relative NLL change over the last ``frac`` of logged steps."""
    if len(hist) < 3:
        return {"nll_rel_drop_tail": float("nan"), "still_decreasing": float("nan")}
    steps = [int(r["step"]) for r in hist]
    nlls = [float(r["nll"]) for r in hist]
    s_max = max(steps)
    cut = s_max * (1.0 - frac)
    tail = [(s, n) for s, n in zip(steps, nlls) if s >= cut]
    if len(tail) < 2:
        tail = list(zip(steps[-3:], nlls[-3:]))
    n0, n1 = float(tail[0][1]), float(tail[-1][1])
    rel = (n0 - n1) / max(abs(n0), 1e-9)
    # "Still decreasing" if last quarter drops >2% relative
    return {
        "nll_start_tail": n0,
        "nll_end": n1,
        "nll_rel_drop_tail": rel,
        "still_decreasing": float(rel > 0.02),
        "nll_min": float(min(nlls)),
        "nll_at_end": n1,
        "best_step": float(steps[int(np.argmin(nlls))]),
    }


def fit_with_history(
    problem: dict[str, Any],
    *,
    num_steps: int,
    seed: int,
    log_every: int,
) -> tuple[dict[str, Any], list[dict], dict[str, float]]:
    """Fit MidlineSparse and return (pred_on_test, history, uq_metrics)."""
    from gpplus.models import MidlineSparseMoEGPR
    from diagnostics import dual_report

    x_tr, y_tr = problem["x_tr"], problem["y_tr"]
    x_te, y_te = problem["x_te"], problem["y_te"]
    champion = bool(problem["champion"])
    k_max = int(problem["k_max"])

    model = MidlineSparseMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        gate_type="mlp",
        offset_type="mlp",
        tau_max=2.5,
        sep_length=2.0,
        keff_tau=0.02,
        free_expert_scale=False,
        use_full_gp_variance=False,
        gamma_init_logit=2.2 if champion else 0.0,
    )
    stage2_device = "cuda" if torch.cuda.is_available() else "cpu"
    fit_kwargs: dict[str, Any] = dict(
        num_inits=2,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        seed=seed,
        device="cpu",
        n_jobs=1,
        adaptive_merge_prune=True,
        try_both_gates=False,
        num_stage2_inits=1,
        log_every=log_every,
        stage2_device=stage2_device,
    )
    if champion:
        # Lean post-factorial champion (matches foundation_adapters).
        fit_kwargs.update(
            lambda_k=0.01,
            lambda_sep=0.0,
            sep_margin=2.0,
            lambda_overlap=0.0,
            overlap_margin=1.0,
            lambda_alpha=0.05,
            lambda_orphan=0.0,
            lambda_delta=0.01,
            lambda_gamma=0.0,
            lambda_rec=0.0,
            rec_margin=1.0,
            lambda_k_start=0.0,
            alive_weight=0.75,
        )
    else:
        fit_kwargs.update(
            lambda_k=0.02 if k_max == 2 else 0.08,
            lambda_sep=0.05,
            lambda_alpha=0.05,
            lambda_orphan=0.05,
        )

    t0 = time.perf_counter()
    model.fit(**fit_kwargs)
    wall = time.perf_counter() - t0
    hist = list(getattr(model, "_fit_history", []) or [])

    pred = model.predict(x_te)
    em = pred["expert_means"].detach().cpu().numpy()
    es = pred["expert_stds"].detach().cpu().numpy()
    gw = pred["gate_weights"].detach().cpu().numpy()
    alpha = pred["alpha"].detach().cpu().numpy()
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, gw.shape)
    mass = np.clip(gw * alpha, 0.0, None)
    mass = mass / np.clip(mass.sum(axis=-1, keepdims=True), 1e-12, None)
    mix = (em * mass).sum(axis=-1)
    y_np = y_te.detach().cpu().numpy().ravel()
    uq = compute_uq_metrics(y_np, mix, em, es, mass)
    uq["wall_s"] = wall
    uq["merge_pressure"] = float(getattr(model, "merge_pressure", 0.0))
    uq["k_eff"] = float(model.k_eff().detach())
    uq["k_hat"] = int(model.k_hat())
    try:
        report = dual_report(
            model,
            np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist(),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            n_grid=400,
        )
        uq["k_modes_mean"] = float(report["k_modes_mean"])
    except Exception:  # noqa: BLE001
        pass
    return {"mixture_mean": mix, "expert_means": em, "expert_stds": es, "gate_weights": mass}, hist, uq


def plot_curves(
    problem_name: str,
    hist: list[dict],
    *,
    out_path: Path,
    budgets: list[int],
) -> Path:
    steps = [int(r["step"]) for r in hist]
    loss = [float(r["loss"]) for r in hist]
    nll = [float(r["nll"]) for r in hist]
    merge = [float(r.get("merge_pressure", 0.0)) for r in hist]
    keff = [float(r.get("k_eff", float("nan"))) for r in hist]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=140)
    ax = axes[0, 0]
    ax.plot(steps, loss, "o-", color="#2b6cb0", ms=3.5, lw=1.4, label="train loss")
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel("loss")
    ax.set_title("Composite train loss")
    ax.grid(True, alpha=0.3)
    for b in budgets:
        ax.axvline(b, color="0.6", ls=":", lw=1.0)

    ax = axes[0, 1]
    ax.plot(steps, nll, "o-", color="#c53030", ms=3.5, lw=1.4, label="train NLL")
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel("NLL")
    ax.set_title("Train mixture NLL")
    ax.grid(True, alpha=0.3)
    for b in budgets:
        ax.axvline(b, color="0.6", ls=":", lw=1.0)
    # mark best NLL
    if nll:
        ib = int(np.argmin(nll))
        ax.scatter([steps[ib]], [nll[ib]], color="#38a169", s=50, zorder=5, label=f"best@{steps[ib]}")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, merge, "o-", color="#d69e2e", ms=3.5, lw=1.4)
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel("merge_pressure")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Merge pressure (anti-merge fade)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, keff, "o-", color="#805ad5", ms=3.5, lw=1.4)
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$")
    ax.set_title("Effective experts")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{problem_name}: Stage-2 training curves (vertical lines = CRPS budgets)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def plot_budget_bars(
    problem_name: str,
    budget_rows: list[dict],
    *,
    out_path: Path,
) -> Path:
    budgets = [int(r["num_steps"]) for r in budget_rows]
    crps = [float(r["CRPS"]) for r in budget_rows]
    cov90 = [float(r["cov90"]) for r in budget_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=140)
    axes[0].bar([str(b) for b in budgets], crps, color="#2b6cb0")
    axes[0].set_xlabel("num_steps")
    axes[0].set_ylabel("held-out CRPS")
    axes[0].set_title("CRPS vs Stage-2 budget")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar([str(b) for b in budgets], cov90, color="#38a169")
    axes[1].axhline(0.9, color="#e53e3e", ls="--", lw=1.2)
    axes[1].set_xlabel("num_steps")
    axes[1].set_ylabel("cov90")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("90% coverage vs budget")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{problem_name}: does more Stage-2 help UQ?", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def run_problem(
    problem_name: str,
    *,
    budgets: list[int],
    seed: int = 42,
    n_samples: int = 1000,
    test_size: float = 0.2,
    log_every: int = 100,
) -> dict[str, Any]:
    problem = load_problem(
        problem_name, n_samples=n_samples, test_size=test_size, seed=seed
    )
    budgets = sorted(set(int(b) for b in budgets))
    max_steps = max(budgets)

    print(f"[{problem_name}] long run num_steps={max_steps} log_every={log_every} ...", flush=True)
    _pred, hist, uq_long = fit_with_history(
        problem, num_steps=max_steps, seed=seed, log_every=log_every
    )
    slope = _tail_slope(hist)
    print(
        f"  long-run: CRPS={uq_long['CRPS']:.4f} cov90={uq_long['cov90']:.3f} "
        f"NLL_end={slope.get('nll_end', float('nan')):.4f} "
        f"best_step={slope.get('best_step')} "
        f"tail_rel_drop={slope.get('nll_rel_drop_tail', float('nan')):.3f} "
        f"still_decreasing={bool(slope.get('still_decreasing', 0))}",
        flush=True,
    )

    curve_png = FIG_DIR / f"{problem_name.lower()}_stage2_curves.png"
    plot_curves(problem_name, hist, out_path=curve_png, budgets=budgets)
    print(f"Wrote {curve_png}", flush=True)

    budget_rows: list[dict[str, Any]] = []
    for b in budgets:
        if b == max_steps:
            row = {
                "num_steps": b,
                "CRPS": float(uq_long["CRPS"]),
                "cov90": float(uq_long["cov90"]),
                "NIS": float(uq_long["NIS"]),
                "RMSE": float(uq_long["RMSE"]),
                "log_score": float(uq_long["log_score"]),
                "k_eff": float(uq_long.get("k_eff", float("nan"))),
                "merge_pressure": float(uq_long.get("merge_pressure", float("nan"))),
                "wall_s": float(uq_long.get("wall_s", float("nan"))),
                "from_long_run": True,
            }
        else:
            print(f"[{problem_name}] budget={b} ...", flush=True)
            _p, h_b, uq_b = fit_with_history(
                problem, num_steps=b, seed=seed, log_every=max(log_every, b // 10 or 1)
            )
            row = {
                "num_steps": b,
                "CRPS": float(uq_b["CRPS"]),
                "cov90": float(uq_b["cov90"]),
                "NIS": float(uq_b["NIS"]),
                "RMSE": float(uq_b["RMSE"]),
                "log_score": float(uq_b["log_score"]),
                "k_eff": float(uq_b.get("k_eff", float("nan"))),
                "merge_pressure": float(uq_b.get("merge_pressure", float("nan"))),
                "wall_s": float(uq_b.get("wall_s", float("nan"))),
                "from_long_run": False,
                "tail": _tail_slope(h_b),
            }
            print(
                f"  budget {b}: CRPS={row['CRPS']:.4f} cov90={row['cov90']:.3f}",
                flush=True,
            )
        budget_rows.append(row)

    bar_png = FIG_DIR / f"{problem_name.lower()}_stage2_budget_uq.png"
    plot_budget_bars(problem_name, budget_rows, out_path=bar_png)
    print(f"Wrote {bar_png}", flush=True)

    # Verdict
    crps_by = {int(r["num_steps"]): float(r["CRPS"]) for r in budget_rows}
    default = 800 if 800 in crps_by else budgets[len(budgets) // 2]
    best_b = min(crps_by, key=lambda k: crps_by[k])
    improve = (crps_by[default] - crps_by[best_b]) / max(crps_by[default], 1e-9)
    if best_b > default and improve > 0.03:
        verdict = (
            f"{problem_name}: more Stage-2 helps - best CRPS at {best_b} steps "
            f"({100 * improve:.1f}% better than {default}); consider raising num_steps"
        )
    elif bool(slope.get("still_decreasing", 0)) and best_b >= default:
        verdict = (
            f"{problem_name}: train NLL still dropping in last quarter but "
            f"held-out CRPS gain is small (best@{best_b}); optional modest increase"
        )
    else:
        verdict = (
            f"{problem_name}: Stage-2 plateaus by ~{default} steps "
            f"(best CRPS@{best_b}); more steps unlikely to fix UQ gap"
        )

    payload = {
        "problem": problem_name,
        "protocol": {
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
            "budgets": budgets,
            "champion": problem["champion"],
            "k_max": problem["k_max"],
        },
        "long_run": {
            "num_steps": max_steps,
            "uq": uq_long,
            "tail_diagnostics": slope,
            "history": _slim_history(hist),
        },
        "budget_rows": budget_rows,
        "verdict": verdict,
        "artifacts": {"curves": str(curve_png), "budget_bars": str(bar_png)},
    }
    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = [
        make_cost_row(
            script="run_stage2_training_curves",
            problem=problem_name,
            model=f"MidlineSparse@{int(r['num_steps'])}",
            wall_s=r.get("wall_s"),
            n_train=int(problem["x_tr"].shape[0]),
            k_max=int(problem["k_max"]),
            num_steps=int(r["num_steps"]),
            gate_type="mlp",
            num_inits=2,
            num_stage2_inits=1,
            champion=bool(problem["champion"]),
            notes="budget sweep",
        )
        for r in budget_rows
    ]
    cost_art = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR,
        stem=f"{problem_name.lower()}_cost",
        title=f"{problem_name} Stage-2 budget sweep — fit cost",
    )
    payload["artifacts"]["cost"] = cost_art
    out_json = OUT_DIR / f"{problem_name.lower()}_stage2_curves.json"
    _json_dump(out_json, payload)
    print(f"Wrote {out_json}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="unimodal,A05,A01")
    parser.add_argument("--budgets", default="400,800,1600,2400")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"problems": {}, "budgets": budgets}
    for pname in problems:
        summary["problems"][pname] = run_problem(
            pname,
            budgets=budgets,
            seed=args.seed,
            n_samples=args.n_samples,
            test_size=args.test_size,
            log_every=args.log_every,
        )

    # Compact table
    lines = [
        "",
        "### Stage-2 training curves (step budget)",
        "",
        "| Problem | best CRPS @ steps | default-800 CRPS | train NLL still dropping? | Verdict |",
        "|---|---:|---:|---|---|",
    ]
    for pname, payload in summary["problems"].items():
        rows = payload["budget_rows"]
        by = {int(r["num_steps"]): r for r in rows}
        best = min(rows, key=lambda r: float(r["CRPS"]))
        c800 = by.get(800, {}).get("CRPS")
        still = payload["long_run"]["tail_diagnostics"].get("still_decreasing", 0)
        lines.append(
            f"| {pname} | {float(best['CRPS']):.4f} @ {best['num_steps']} | "
            f"{float(c800):.4f} | {bool(still)} | {payload['verdict']} |"
        )
    summary["markdown_table"] = "\n".join(lines)
    _json_dump(OUT_DIR / "summary.json", summary)
    # Windows consoles may not support some Unicode; print ASCII-safe.
    print(summary["markdown_table"].encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {OUT_DIR / 'summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
