"""A02-focused probes for MidlineSparseHardXGateMoEGPR (hard w(x) router).

Isolated from champion MidlineSparse / foundation_adapters wiring. Soft
constant/MLP contrasts are fit here only for comparison. Hard routing is
top-1 one-hot, always on the simplex.

Also emits foundation-style ScoringBench tables (RMSE/RRMSE/NIS/CRPS/log_score)
and per-probe marginal + posterior-vs-data figures.

Run (gpplus2026):
  python run_midline_hard_xgate.py
  python run_midline_hard_xgate.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
from gpplus.models import MidlineSparseHardXGateMoEGPR, MidlineSparseMoEGPR

from compare_foundation_baselines import ALL_PROBLEMS, load_problem
from diagnostics import dual_report
from foundation_adapters import fit_predict_midline_sparse
from plot_foundation_marginals import (
    MODEL_COLORS,
    probe_x_values,
    save_marginals_per_probe,
    save_posterior_vs_data,
    slice_pred,
)
from plot_midline_sparse import save_training_loss_subplots
from run_midline_sparse import make_step
from scoringbench_metrics import compute_scoringbench_style_metrics, print_metrics_table

OUT_DIR = _THIS / "results" / "hard_xgate"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMPARE_COLS = ["RMSE", "RRMSE", "NIS", "CRPS", "log_score"]
# Soft constant is most diagnostic on separated regimes (A02); other problems
# always get hard_x + soft MLP contrast.
SOFT_CONSTANT_PROBLEMS = frozenset({"A02"})
MODEL_LABELS = {
    "hard_x": "MidlineSparse hard-xgate",
    "soft_mlp": "MidlineSparse soft MLP",
    "soft_constant": "MidlineSparse soft constant",
}
# Colors used by plot_foundation_marginals (fallback C{j} if missing).
MODEL_COLORS.update(
    {
        "hard_x": "#c05621",
        "soft_mlp": "#2b6cb0",
        "soft_constant": "#718096",
    }
)


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _mixture_from_model(
    model: MidlineSparseHardXGateMoEGPR | MidlineSparseMoEGPR,
    x_te: torch.Tensor,
    *,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Foundation-style mixture dict; gate mass = π·α renormalized (simplex)."""
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
    return {
        "mixture_mean": mix,
        "expert_means": em,
        "expert_stds": es,
        "gate_weights": mass,
        "extras": dict(extras or {}),
    }


def fit_predict_hard_xgate(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    k_max: int = 4,
    num_inits: int = 2,
    num_steps: int = 500,
    gumbel_tau: float = 0.5,
    seed: int = 42,
    y_min: float = -0.5,
    y_max: float = 1.5,
    x_probe: list[float] | None = None,
    return_model: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], MidlineSparseHardXGateMoEGPR]:
    t0 = time.perf_counter()
    model = MidlineSparseHardXGateMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        offset_type="mlp",
        router_hidden=(32, 32),
        gumbel_tau=gumbel_tau,
        use_full_gp_variance=False,
        gamma_init_logit=2.2,
        free_expert_scale=False,
    )
    model.fit(
        num_inits=num_inits,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        lambda_k=0.02,
        lambda_sep=0.1,
        lambda_alpha=0.05,
        lambda_orphan=0.25,
        lambda_delta=0.02,
        lambda_gamma=0.1,
        lambda_rec=0.15,
        lambda_overlap=0.08,
        sep_margin=2.0,
        overlap_margin=1.0,
        rec_margin=1.0,
        lambda_k_start=0.0,
        seed=seed,
        device="cpu",
        n_jobs=1,
        adaptive_merge_prune=True,
        try_both_gates=False,
        num_stage2_inits=1,
        log_every=100,
        stage2_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    wall = time.perf_counter() - t0
    if x_probe is None:
        x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
    report = dual_report(model, x_probe, y_min=y_min, y_max=y_max, n_grid=400)
    hist = list(getattr(model, "_fit_history", []) or [])
    fit_history = [
        {
            "step": int(r["step"]),
            "loss": float(r["loss"]),
            "nll": float(r["nll"]),
            "k_eff": float(r.get("k_eff", float("nan"))),
            "k_hat": int(r.get("k_hat", -1)),
            "merge_pressure": float(r.get("merge_pressure", 0.0)),
        }
        for r in hist
        if "step" in r and "loss" in r
    ]
    extras = {
        "wall_s": wall,
        "k_eff": float(model.k_eff().detach()),
        "k_hat": int(model.k_hat()),
        "k_modes_mean": float(report["k_modes_mean"]),
        "k_modes_round": int(report["k_modes_round"]),
        "router_type": "hard_x",
        "gumbel_tau": gumbel_tau,
        "k_max": k_max,
        "merge_pressure": float(getattr(model, "merge_pressure", 0.0)),
        "fit_history": fit_history,
    }
    out = _mixture_from_model(model, x_eval, extras=extras)
    if return_model:
        return out, model
    return out


def save_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "model_key", "source"] + COMPARE_COLS + [
        "k_eff",
        "k_modes_mean",
        "k_modes_round",
        "wall_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def save_metrics_table_png(
    rows: list[dict],
    out_path: Path,
    *,
    title: str,
) -> Path:
    col_labels = ["Model"] + COMPARE_COLS
    cell_text = []
    for row in rows:
        cells = [str(row["Model"])]
        for c in COMPARE_COLS:
            v = row.get(c)
            cells.append(
                "—"
                if v is None or not np.isfinite(float(v))
                else f"{float(v):.4f}"
            )
        cell_text.append(cells)

    fig, ax = plt.subplots(figsize=(11.5, 2.0 + 0.38 * len(rows)), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.12, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r >= 1 and rows[r - 1].get("model_key") == "hard_x":
            cell.set_facecolor("#fffaf0")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def emit_scoringbench(
    *,
    problem_name: str,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    model_preds_te: dict[str, dict[str, np.ndarray]],
    model_preds_probe: dict[str, dict[str, np.ndarray]],
    model_preds_grid: dict[str, dict[str, np.ndarray]],
    model_order: list[str],
    model_extras: dict[str, dict[str, Any]],
    x_probes: np.ndarray,
    x_grid: np.ndarray,
    y_min: float,
    y_max: float,
    stem: str,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Write ScoringBench CSV/JSON/PNG + foundation-style marginals/posterior."""
    y_np = y_te.detach().cpu().numpy().ravel()
    x_tr_np = x_tr.detach().cpu().numpy()
    x_te_np = x_te.detach().cpu().numpy()
    y_tr_np = y_tr.detach().cpu().numpy().ravel()

    rows: list[dict[str, Any]] = []
    metrics_by_key: dict[str, Any] = {}
    for mk in model_order:
        pred = model_preds_te[mk]
        metrics = compute_scoringbench_style_metrics(
            y_np,
            pred["mixture_mean"],
            pred["expert_means"],
            pred["expert_stds"],
            pred["gate_weights"],
        )
        extras = model_extras.get(mk, {})
        row: dict[str, Any] = {
            "Model": MODEL_LABELS.get(mk, mk),
            "model_key": mk,
            "source": "hard_xgate_probe",
            **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
        }
        for k in ("k_eff", "k_modes_mean", "k_modes_round", "wall_s"):
            if k in extras:
                row[k] = extras[k]
        rows.append(row)
        metrics_by_key[mk] = metrics
        print(
            f"  [{problem_name}/{mk}] "
            f"RMSE={metrics['RMSE']:.4f} CRPS={metrics['CRPS']:.4f} "
            f"log_score={metrics['log_score']:.4f}",
            flush=True,
        )

    print_metrics_table(rows, cols=["Model"] + COMPARE_COLS)
    table_json = OUT_DIR / f"{stem}_metrics.json"
    table_csv = OUT_DIR / f"{stem}_metrics.csv"
    table_png = FIG_DIR / f"{stem}_metrics_table.png"
    save_csv(rows, table_csv)
    _json_dump(
        table_json,
        {
            "protocol": {
                "problem": problem_name,
                "n_train": int(x_tr.shape[0]),
                "n_test": int(x_te.shape[0]),
                "n_probes": int(len(x_probes)),
            },
            "rows": rows,
            "metrics": metrics_by_key,
            "extras": model_extras,
        },
    )
    save_metrics_table_png(
        rows,
        table_png,
        title=f"{problem_name}: hard-xgate vs soft MidlineSparse (ScoringBench)",
    )
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_json}")
    print(f"Wrote {table_png}")

    marginal_paths: list[str] = []
    posterior_path: str | None = None
    if write_plots and model_preds_probe:
        x_all = np.vstack([x_tr_np, x_te_np])
        y_all = np.concatenate([y_tr_np, y_np])
        m_paths = save_marginals_per_probe(
            problem_name=problem_name,
            x_probes=x_probes,
            model_preds=model_preds_probe,
            model_labels=MODEL_LABELS,
            model_order=model_order,
            x_data=x_all,
            y_data=y_all,
            y_min=y_min,
            y_max=y_max,
            out_dir=FIG_DIR,
            stem=stem,
        )
        for p in m_paths:
            print(f"Wrote {p}")
            marginal_paths.append(str(p))
    if write_plots and model_preds_grid:
        pp = FIG_DIR / f"{stem}_posterior_vs_data.png"
        save_posterior_vs_data(
            problem_name=problem_name,
            x_train=x_tr_np,
            y_train=y_tr_np,
            x_test=x_te_np,
            y_test=y_np,
            x_grid=x_grid,
            model_preds_grid=model_preds_grid,
            model_labels=MODEL_LABELS,
            model_order=model_order,
            out_path=pp,
        )
        print(f"Wrote {pp}")
        posterior_path = str(pp)

    return {
        "rows": rows,
        "metrics": metrics_by_key,
        "table_csv": str(table_csv),
        "table_json": str(table_json),
        "table_png": str(table_png),
        "marginal_paths": marginal_paths,
        "posterior_path": posterior_path,
    }


def _eval_pack(
    x_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    n_probes: int = 5,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, int, int, int]:
    x_tr_np = x_tr.detach().cpu().numpy()
    x_te_np = x_te.detach().cpu().numpy()
    x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=n_probes)
    x_lo = float(min(x_tr_np.min(), x_te_np.min()))
    x_hi = float(max(x_tr_np.max(), x_te_np.max()))
    x_grid = np.linspace(x_lo, x_hi, 200).reshape(-1, 1)
    x_probe_t = torch.as_tensor(x_probes.reshape(-1, 1), dtype=x_tr.dtype)
    x_grid_t = torch.as_tensor(x_grid, dtype=x_tr.dtype)
    x_eval = torch.cat([x_te, x_probe_t, x_grid_t], dim=0)
    return x_probes, x_grid, x_eval, int(x_te.shape[0]), int(x_probe_t.shape[0]), int(x_grid_t.shape[0])


def _split_eval(
    pred_all: dict[str, Any],
    n_te: int,
    n_probe: int,
    n_grid: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    i0, i1 = 0, n_te
    pred_te = slice_pred(pred_all, i0, i1)
    i0 = i1
    i1 = i0 + n_probe
    pred_probe = slice_pred(pred_all, i0, i1)
    i0 = i1
    i1 = i0 + n_grid
    pred_grid = slice_pred(pred_all, i0, i1)
    return pred_te, pred_probe, pred_grid


def _test_nll(model: MidlineSparseHardXGateMoEGPR, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model.predict(x)
    means = pred["expert_means"]
    stds = pred["expert_stds"]
    pi = pred["gate_weights"]
    y = y.reshape(-1)
    log_comps = []
    for k in range(model.k_max):
        var = stds[:, k].pow(2).clamp_min(1e-12)
        lp = (
            torch.log(pi[:, k].clamp_min(1e-12))
            - 0.5 * (torch.log(2 * torch.pi * var) + (y - means[:, k]).pow(2) / var)
        )
        log_comps.append(lp)
    return float((-torch.logsumexp(torch.stack(log_comps, dim=-1), dim=-1).mean()).item())


def _gate_simplex_ok(w: torch.Tensor, atol: float = 1e-5) -> bool:
    if (w < -atol).any():
        return False
    return bool(torch.allclose(w.sum(dim=-1), torch.ones(w.shape[0], dtype=w.dtype), atol=atol))


def _hard_fraction(w: torch.Tensor, atol: float = 1e-6) -> float:
    """Fraction of rows that are (near) one-hot."""
    maxes = w.max(dim=-1).values
    return float((maxes >= 1.0 - atol).float().mean().item())


def fit_hard_xgate(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    name: str,
    true_k_modes: int,
    k_max: int = 4,
    num_inits: int = 2,
    num_steps: int = 500,
    gumbel_tau: float = 0.5,
    seed: int = 42,
    y_min: float = -0.5,
    y_max: float = 1.5,
    return_model: bool = False,
) -> dict | tuple[dict, MidlineSparseHardXGateMoEGPR]:
    t0 = time.perf_counter()
    model = MidlineSparseHardXGateMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        offset_type="mlp",
        router_hidden=(32, 32),
        gumbel_tau=gumbel_tau,
        use_full_gp_variance=False,
        gamma_init_logit=2.2,
        free_expert_scale=False,
    )
    model.fit(
        num_inits=num_inits,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        lambda_k=0.02,
        lambda_sep=0.1,
        lambda_alpha=0.05,
        lambda_orphan=0.25,
        lambda_delta=0.02,
        lambda_gamma=0.1,
        lambda_rec=0.15,
        lambda_overlap=0.08,
        sep_margin=2.0,
        overlap_margin=1.0,
        rec_margin=1.0,
        lambda_k_start=0.0,
        seed=seed,
        device="cpu",
        n_jobs=1,
        adaptive_merge_prune=True,
        try_both_gates=False,
        num_stage2_inits=1,
        log_every=100,
        stage2_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    wall = time.perf_counter() - t0
    x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
    report = dual_report(model, x_probe, y_min=y_min, y_max=y_max, n_grid=400)
    nll = _test_nll(model, x_te, y_te)
    pred = model.predict(x_te)
    w = pred["gate_weights"]
    row = {
        "name": name,
        "router_type": "hard_x",
        "true_k_modes": true_k_modes,
        "k_max": k_max,
        "gumbel_tau": gumbel_tau,
        "num_steps": num_steps,
        "k_eff": report.get("k_eff", float(model.k_eff().detach())),
        "k_hat": report.get("k_hat", model.k_hat()),
        "k_modes_mean": report["k_modes_mean"],
        "k_modes_round": report["k_modes_round"],
        "test_nll": nll,
        "wall_s": wall,
        "gate_simplex_ok": _gate_simplex_ok(w),
        "hard_onehot_frac": _hard_fraction(w),
        "mean_active_expert": float(w.argmax(dim=-1).float().mean().item()),
        "alpha": model.get_alpha(deterministic=True).detach().cpu().tolist(),
        "merge_pressure": float(getattr(model, "merge_pressure", 0.0)),
    }
    print(
        f"[{name}] hard_x Kmax={k_max} | K_eff={row['k_eff']:.2f} "
        f"K_hat={row['k_hat']} K_modes~{row['k_modes_mean']:.2f} "
        f"nll={nll:.3f} onehot={row['hard_onehot_frac']:.2f} "
        f"simplex_ok={row['gate_simplex_ok']} t={wall:.1f}s"
    )
    if return_model:
        return row, model
    return row


def fit_soft_contrast(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    name: str,
    gate_type: str,
    k_max: int = 4,
    num_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Soft MidlineSparse contrast (champion path); not modified."""
    t0 = time.perf_counter()
    model = MidlineSparseMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        gate_type=gate_type,  # type: ignore[arg-type]
        offset_type="mlp",
        use_full_gp_variance=False,
        gamma_init_logit=2.2,
    )
    model.fit(
        num_inits=2,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        lambda_k=0.02,
        lambda_sep=0.1,
        lambda_alpha=0.05,
        lambda_orphan=0.25,
        lambda_delta=0.02,
        lambda_gamma=0.1,
        lambda_rec=0.15,
        lambda_overlap=0.08,
        sep_margin=2.0,
        overlap_margin=1.0,
        rec_margin=1.0,
        lambda_k_start=0.0,
        seed=seed,
        device="cpu",
        n_jobs=1,
        adaptive_merge_prune=True,
        try_both_gates=False,
        num_stage2_inits=1,
        log_every=100,
        stage2_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Local NLL (same formula as hard path)
    pred = model.predict(x_te)
    means, stds, pi = pred["expert_means"], pred["expert_stds"], pred["gate_weights"]
    y = y_te.reshape(-1)
    log_comps = []
    for k in range(model.k_max):
        var = stds[:, k].pow(2).clamp_min(1e-12)
        lp = (
            torch.log(pi[:, k].clamp_min(1e-12))
            - 0.5 * (torch.log(2 * torch.pi * var) + (y - means[:, k]).pow(2) / var)
        )
        log_comps.append(lp)
    nll = float((-torch.logsumexp(torch.stack(log_comps, dim=-1), dim=-1).mean()).item())
    report = dual_report(
        model,
        np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist(),
        y_min=-0.5,
        y_max=1.5,
        n_grid=400,
    )
    row = {
        "name": name,
        "router_type": gate_type,
        "k_max": k_max,
        "k_eff": float(model.k_eff().detach()),
        "k_hat": model.k_hat(),
        "k_modes_mean": report["k_modes_mean"],
        "test_nll": nll,
        "wall_s": time.perf_counter() - t0,
        "gate_simplex_ok": _gate_simplex_ok(pi),
        "hard_onehot_frac": _hard_fraction(pi),
    }
    print(
        f"[{name}] soft/{gate_type} Kmax={k_max} | "
        f"K_modes~{row['k_modes_mean']:.2f} nll={nll:.3f} t={row['wall_s']:.1f}s"
    )
    return row


def _write_figures(
    model: MidlineSparseHardXGateMoEGPR,
    *,
    stem: str,
    title_prefix: str,
    enabled: bool = True,
) -> list[str]:
    if not enabled:
        return []
    from plot_midline_hard_xgate import save_hard_xgate_figures

    return save_hard_xgate_figures(
        model,
        stem=stem,
        fig_dir=FIG_DIR,
        title_prefix=title_prefix,
    )


def run_smoke(*, write_plots: bool = True) -> dict:
    gpplus.config.configure_logger()
    xs_tr, ys_tr, xs_te, ys_te = make_step(120, 80, seed=0)
    row, model = fit_hard_xgate(
        xs_tr,
        ys_tr,
        xs_te,
        ys_te,
        name="smoke_A02_hard_x",
        true_k_modes=1,
        k_max=3,
        num_inits=1,
        num_steps=200,
        seed=0,
        return_model=True,
    )
    assert isinstance(model, MidlineSparseHardXGateMoEGPR)
    assert row["gate_simplex_ok"], "hard w(x) left the simplex"
    assert row["hard_onehot_frac"] >= 0.99, "eval gate should be one-hot"
    figures = _write_figures(
        model,
        stem="smoke_a02_hard_x",
        title_prefix="smoke A02",
        enabled=write_plots,
    )
    summary = {"smoke": row, "figures": figures, "ok": True}
    _json_dump(OUT_DIR / "hard_xgate_smoke.json", summary)
    return summary


def _run_problem_bundle(
    *,
    problem_name: str,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    y_min: float,
    y_max: float,
    stem: str,
    k_max: int = 4,
    num_steps: int = 500,
    seed: int = 42,
    write_plots: bool = True,
    soft_gates: Sequence[str] = ("mlp",),
    champion: bool = True,
) -> dict[str, Any]:
    x_probes, x_grid, x_eval, n_te, n_probe, n_grid = _eval_pack(x_tr, x_te)
    probe_list = x_probes.tolist()
    model_order = ["hard_x"]
    preds_te: dict[str, dict[str, np.ndarray]] = {}
    preds_probe: dict[str, dict[str, np.ndarray]] = {}
    preds_grid: dict[str, dict[str, np.ndarray]] = {}
    extras: dict[str, dict[str, Any]] = {}
    router_figs: list[str] = []

    print(f"[{problem_name}] fitting hard_x ...", flush=True)
    hard_all, hard_model = fit_predict_hard_xgate(
        x_tr,
        y_tr,
        x_eval,
        k_max=k_max,
        num_inits=2,
        num_steps=num_steps,
        seed=seed,
        y_min=y_min,
        y_max=y_max,
        x_probe=probe_list,
        return_model=True,
    )
    assert isinstance(hard_model, MidlineSparseHardXGateMoEGPR)
    te, pr, gr = _split_eval(hard_all, n_te, n_probe, n_grid)
    preds_te["hard_x"] = te
    preds_probe["hard_x"] = pr
    preds_grid["hard_x"] = gr
    extras["hard_x"] = dict(hard_all.get("extras") or {})
    # Validity check on raw hard w(x), not α-scaled mass used for scoring.
    w_raw = hard_model.get_hard_weights(x_te).detach()
    extras["hard_x"]["gate_simplex_ok"] = _gate_simplex_ok(w_raw)
    extras["hard_x"]["hard_onehot_frac"] = _hard_fraction(w_raw)
    router_figs.extend(
        _write_figures(
            hard_model,
            stem=f"{stem}_hard_x",
            title_prefix=problem_name,
            enabled=write_plots,
        )
    )

    gate_to_key = {"mlp": "soft_mlp", "constant": "soft_constant"}
    for gate in soft_gates:
        mk = gate_to_key[gate]
        print(f"[{problem_name}] fitting {mk} ...", flush=True)
        soft_all = fit_predict_midline_sparse(
            x_tr,
            y_tr,
            x_eval,
            k_max=k_max,
            num_steps=num_steps,
            seed=seed,
            champion=champion,
            y_min=y_min,
            y_max=y_max,
            x_probe=probe_list,
            gate_type=gate,
            adaptive_merge_prune=True,
        )
        te_s, pr_s, gr_s = _split_eval(soft_all, n_te, n_probe, n_grid)
        preds_te[mk] = te_s
        preds_probe[mk] = pr_s
        preds_grid[mk] = gr_s
        extras[mk] = dict(soft_all.get("extras") or {})
        extras[mk]["router_type"] = gate
        model_order.append(mk)

    sb = emit_scoringbench(
        problem_name=problem_name,
        x_tr=x_tr,
        y_tr=y_tr,
        x_te=x_te,
        y_te=y_te,
        model_preds_te=preds_te,
        model_preds_probe=preds_probe,
        model_preds_grid=preds_grid,
        model_order=model_order,
        model_extras=extras,
        x_probes=x_probes,
        x_grid=x_grid,
        y_min=y_min,
        y_max=y_max,
        stem=stem,
        write_plots=write_plots,
    )
    loss_figs: list[str] = []
    if write_plots:
        histories = {
            mk: list(extras.get(mk, {}).get("fit_history") or [])
            for mk in model_order
        }
        loss_path = save_training_loss_subplots(
            histories,
            out_path=FIG_DIR / f"{stem}_training_loss.png",
            model_labels=MODEL_LABELS,
            model_order=model_order,
            model_colors=MODEL_COLORS,
            title=f"{problem_name}  |  Stage-2 training loss",
        )
        print(f"Wrote {loss_path}", flush=True)
        loss_figs.append(str(loss_path))
    sb["router_figures"] = router_figs
    sb["training_loss_figures"] = loss_figs
    sb["extras"] = extras
    return sb


def run_problems(
    problems: Sequence[str] | None = None,
    *,
    n_samples: int = 1000,
    test_size: float = 0.2,
    seed: int = 42,
    write_plots: bool = True,
    k_max: int | None = None,
) -> dict[str, Any]:
    """Hard-xgate ScoringBench + marginals on foundation problem suite."""
    gpplus.config.configure_logger()
    names = list(problems) if problems else list(ALL_PROBLEMS)
    payloads: dict[str, Any] = {}
    for name in names:
        problem = load_problem(
            name, n_samples=n_samples, test_size=test_size, seed=seed
        )
        soft_gates: list[str] = ["mlp"]
        if name in SOFT_CONSTANT_PROBLEMS:
            soft_gates.append("constant")
        km = int(k_max) if k_max is not None else int(problem["k_max"])
        # Hard router with many unused slots is slow; keep overcomplete but
        # bounded for this branch unless caller overrides.
        if k_max is None:
            km = min(km, 6)
        stem = name.lower()
        print(
            f"\n=== {name}: n_train={problem['x_tr'].shape[0]} "
            f"k_max={km} steps={problem['num_steps']} soft={soft_gates} ===",
            flush=True,
        )
        payloads[name] = _run_problem_bundle(
            problem_name=name,
            x_tr=problem["x_tr"],
            y_tr=problem["y_tr"],
            x_te=problem["x_te"],
            y_te=problem["y_te"],
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            stem=stem,
            k_max=km,
            num_steps=int(problem["num_steps"]),
            seed=seed,
            write_plots=write_plots,
            soft_gates=soft_gates,
            champion=bool(problem["champion"]),
        )

    checks: dict[str, Any] = {}
    if "A02" in payloads:
        hard_ex = payloads["A02"]["extras"]["hard_x"]
        soft_const_modes = float(
            payloads["A02"]["extras"].get("soft_constant", {}).get("k_modes_mean", np.nan)
        )
        checks = {
            "hard_simplex_ok": bool(hard_ex.get("gate_simplex_ok", False)),
            "hard_onehot_at_eval": bool(hard_ex.get("hard_onehot_frac", 0.0) >= 0.99),
            "a02_modes_le_soft_const": bool(
                float(hard_ex.get("k_modes_mean", np.inf)) <= soft_const_modes + 0.5
            )
            if np.isfinite(soft_const_modes)
            else None,
        }

    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = []
    for pname, payload in payloads.items():
        extras = payload.get("extras") or {}
        for mk, ex in extras.items():
            cost_rows.append(
                make_cost_row(
                    script="run_midline_hard_xgate",
                    problem=pname,
                    model=MODEL_LABELS.get(mk, mk),
                    wall_s=(ex or {}).get("wall_s"),
                    n_train=n_samples,  # approx; actual train size from protocol
                    k_max=(ex or {}).get("k_max"),
                    num_steps=None,
                    gate_type=str((ex or {}).get("router_type") or (ex or {}).get("gate_type") or mk),
                    num_inits=2 if mk == "hard_x" else 2,
                    num_stage2_inits=1,
                    champion=None,
                    notes=mk,
                )
            )
    cost_art = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR,
        stem="cost",
        title="hard_xgate: fit cost (wall-clock)",
    )

    summary = {
        "problems": names,
        "protocol": {
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
        },
        "checks": checks,
        "cost": cost_art,
        "payloads": {
            k: {
                "rows": v.get("rows"),
                "table_csv": v.get("table_csv"),
                "table_png": v.get("table_png"),
                "marginal_paths": v.get("marginal_paths"),
                "posterior_path": v.get("posterior_path"),
                "router_figures": v.get("router_figures"),
                "extras": v.get("extras"),
            }
            for k, v in payloads.items()
        },
    }
    _json_dump(OUT_DIR / "hard_xgate_summary.json", summary)
    print(f"Wrote {OUT_DIR / 'hard_xgate_summary.json'}")
    if checks:
        print(json.dumps(checks, indent=2))
    return summary


# Back-compat alias used by older docs / callers.
def run_a02_probe(*, write_plots: bool = True) -> dict:
    return run_problems(
        ["A02", "unimodal", "A01", "A05", "A04"],
        n_samples=1000,
        test_size=0.2,
        seed=42,
        write_plots=write_plots,
        k_max=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Short A02 smoke fit")
    parser.add_argument(
        "--problems",
        default="all",
        help=f"Comma list or 'all'. Allowed: {', '.join(ALL_PROBLEMS)}",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="Override k_max (default: min(problem k_max, 6))",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure writing (JSON metrics only)",
    )
    args = parser.parse_args()
    write_plots = not args.no_plots
    if args.smoke:
        run_smoke(write_plots=write_plots)
        return

    if args.problems.strip().lower() == "all":
        problems = list(ALL_PROBLEMS)
    else:
        problems = [p.strip() for p in args.problems.split(",") if p.strip()]
        bad = [p for p in problems if p not in ALL_PROBLEMS]
        if bad:
            raise SystemExit(f"Unknown problems {bad}; choose from {ALL_PROBLEMS}")

    run_problems(
        problems,
        n_samples=args.n_samples,
        test_size=args.test_size,
        seed=args.seed,
        write_plots=write_plots,
        k_max=args.k_max,
    )


if __name__ == "__main__":
    main()
