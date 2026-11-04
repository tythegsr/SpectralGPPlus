"""Smoke + Gate-2 probes for MidlineSparseMoE-GP.

Example suite (soft MLP, 1 init; true p(y|x) + local KDE marginals)::

  python run_midline_sparse.py --mode examples
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
for p in (_ROOT, _EXP, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.models import MidlineSparseMoEGPR

from diagnostics import dual_report
from load_bimodal_data import train_test_split_bimodal
from load_trimodal_data import (
    TRIMODAL_Y_MAX,
    TRIMODAL_Y_MIN,
    make_trimodal_tensors,
)
from load_unimodal_hetero_data import (
    A05_Y_MAX,
    A05_Y_MIN,
    make_unimodal_hetero_tensors,
)

OUT_DIR = _THIS / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def make_unimodal(n: int = 300, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=(n, 1))
    y = 0.5 * x[:, 0] + 2.0 + 0.3 * rng.normal(size=n)
    idx = rng.permutation(n)
    n_te = max(1, int(0.3 * n))
    te, tr = idx[:n_te], idx[n_te:]
    return (
        torch.as_tensor(x[tr], dtype=torch.float64),
        torch.as_tensor(y[tr], dtype=torch.float64),
        torch.as_tensor(x[te], dtype=torch.float64),
        torch.as_tensor(y[te], dtype=torch.float64),
    )


def make_step(n_train: int = 200, n_test: int = 200, noise: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(0.0, 1.0, size=(n_train, 1))
    y_tr = (x_tr[:, 0] >= 0.5).astype(np.float64) + noise * rng.normal(size=n_train)
    x_te = rng.uniform(0.0, 1.0, size=(n_test, 1))
    y_te = (x_te[:, 0] >= 0.5).astype(np.float64) + noise * rng.normal(size=n_test)
    return (
        torch.as_tensor(x_tr, dtype=torch.float64),
        torch.as_tensor(y_tr, dtype=torch.float64),
        torch.as_tensor(x_te, dtype=torch.float64),
        torch.as_tensor(y_te, dtype=torch.float64),
    )


def _test_nll(model: MidlineSparseMoEGPR, x: torch.Tensor, y: torch.Tensor) -> float:
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


def fit_and_report(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    name: str,
    true_k_modes: int,
    k_max: int = 6,
    gate_type: str = "mlp",
    offset_type: str = "mlp",
    gamma_type: str = "constant",
    # Defaults = lean champion (usage + merge + alive + delta + alpha).
    lambda_k: float = 0.01,
    lambda_sep: float = 0.0,
    sep_margin: float = 2.0,
    lambda_overlap: float = 0.0,
    overlap_margin: float = 1.0,
    lambda_beta: float = 0.0,
    free_expert_scale: bool = False,
    lambda_k_start: float | None = 0.0,
    lambda_alpha: float = 0.05,
    lambda_orphan: float = 0.0,
    lambda_delta: float = 0.01,
    lambda_gamma: float = 0.0,
    lambda_rec: float = 0.0,
    rec_margin: float = 1.0,
    alive_weight: float = 0.75,
    lambda_mode: float = 0.0,
    use_full_gp_variance: bool = False,
    gamma_init_logit: float = 2.2,
    num_inits: int = 2,
    num_steps: int = 800,
    y_min: float = -2.0,
    y_max: float = 30.0,
    seed: int = 42,
    x_probe: list[float] | None = None,
    return_model: bool = False,
    adaptive_merge_prune: bool = True,
    try_both_gates: bool | None = None,
    num_stage2_inits: int = 1,
    log_every: int = 100,
    stage2_device: str | None = None,
) -> dict | tuple[dict, MidlineSparseMoEGPR]:
    t0 = time.perf_counter()
    both_gates = bool(try_both_gates) if try_both_gates is not None else False
    if stage2_device is None:
        stage2_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MidlineSparseMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        gate_type=gate_type,  # type: ignore[arg-type]
        offset_type=offset_type,  # type: ignore[arg-type]
        gamma_type=gamma_type,  # type: ignore[arg-type]
        tau_max=2.5,
        sep_length=2.0,
        keff_tau=0.02,
        free_expert_scale=free_expert_scale,
        use_full_gp_variance=use_full_gp_variance,
        gamma_init_logit=gamma_init_logit,
    )
    model.fit(
        num_inits=num_inits,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        lambda_k=lambda_k,
        lambda_sep=lambda_sep,
        lambda_alpha=lambda_alpha,
        lambda_orphan=lambda_orphan,
        lambda_mode=lambda_mode,
        lambda_overlap=lambda_overlap,
        lambda_beta=lambda_beta,
        lambda_delta=lambda_delta,
        lambda_gamma=lambda_gamma,
        lambda_rec=lambda_rec,
        sep_margin=sep_margin,
        overlap_margin=overlap_margin,
        rec_margin=rec_margin,
        lambda_k_start=lambda_k_start,
        alive_weight=alive_weight,
        seed=seed,
        device="cpu",
        n_jobs=1,
        adaptive_merge_prune=adaptive_merge_prune,
        try_both_gates=both_gates,
        num_stage2_inits=int(num_stage2_inits),
        log_every=int(log_every),
        stage2_device=stage2_device,
    )
    wall = time.perf_counter() - t0
    if x_probe is None:
        x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
    report = dual_report(model, x_probe, y_min=y_min, y_max=y_max, n_grid=400)
    nll = _test_nll(model, x_te, y_te)
    per_x_rows = report.get("per_x", [])
    per_x = {float(r["x"]): int(r["k_modes"]) for r in per_x_rows}

    def _k_at(target: float) -> int | None:
        if not per_x:
            return None
        best = min(per_x.keys(), key=lambda xv: abs(xv - target))
        return per_x[best] if abs(best - target) < 0.25 else None

    selected_gate = str(getattr(model, "gate_type", gate_type))
    row = {
        "name": name,
        "true_k_modes": true_k_modes,
        "gate_type": selected_gate,
        "gate_type_requested": gate_type,
        "offset_type": offset_type,
        "k_max": k_max,
        "lambda_k": lambda_k,
        "lambda_sep": lambda_sep,
        "sep_margin": sep_margin,
        "lambda_overlap": lambda_overlap,
        "overlap_margin": overlap_margin,
        "lambda_beta": lambda_beta,
        "free_expert_scale": free_expert_scale,
        "lambda_k_start": lambda_k_start,
        "lambda_orphan": lambda_orphan,
        "lambda_delta": lambda_delta,
        "lambda_gamma": lambda_gamma,
        "lambda_rec": lambda_rec,
        "rec_margin": rec_margin,
        "lambda_mode": lambda_mode,
        "use_full_gp_variance": use_full_gp_variance,
        "gamma_init_logit": gamma_init_logit,
        "gamma_type": gamma_type,
        "num_steps": num_steps,
        "adaptive_merge_prune": bool(adaptive_merge_prune),
        "merge_pressure": float(getattr(model, "merge_pressure", 0.0)),
        "k_eff": report.get("k_eff", float(model.k_eff().detach())),
        "k_hat": report.get("k_hat", model.k_hat()),
        "k_modes_mean": report["k_modes_mean"],
        "k_modes_round": report["k_modes_round"],
        "k_modes_at_x1": _k_at(1.0),
        "k_modes_at_x5": _k_at(5.0),
        "k_modes_at_x7": _k_at(7.0),
        "per_x_k_modes": {f"{xv:g}": km for xv, km in per_x.items()},
        "soft_peak_count_mean": report.get("soft_peak_count_mean"),
        "alpha": report.get("alpha", model.get_alpha(deterministic=True).detach().cpu().tolist()),
        "beta": model.get_beta().detach().cpu().tolist(),
        "gamma": (
            model.get_gamma(x_tr).mean(dim=0).detach().cpu().tolist()
            if gamma_type == "mlp"
            else model.get_gamma().detach().cpu().tolist()
        ),
        "mean_gamma": float(
            model.get_gamma(x_tr).detach().mean()
            if gamma_type == "mlp"
            else model.get_gamma().detach().mean()
        ),
        "test_nll": nll,
        "wall_s": wall,
        "fit_history_tail": model._fit_history[-3:] if model._fit_history else [],
    }
    print(
        f"[{name}] gate={selected_gate} offset={offset_type} gamma={gamma_type} "
        f"Kmax={k_max} | "
        f"K_eff={row['k_eff']:.2f} K_hat={row['k_hat']} "
        f"K_modes~{row['k_modes_mean']:.2f} merge_p={row['merge_pressure']:.2f} "
        f"nll={nll:.3f} t={wall:.1f}s"
    )
    if return_model:
        return row, model
    return row


def run_smoke() -> dict:
    gpplus.config.configure_logger()
    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=42)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)

    a01 = fit_and_report(
        x_tr,
        y_tr,
        x_te,
        y_te,
        name="A01_smoke",
        true_k_modes=2,
        k_max=6,
        gate_type="constant",
        y_min=-2.0,
        y_max=30.0,
    )
    xu, yu, xute, yute = make_unimodal(300, 7)
    uni = fit_and_report(
        xu,
        yu,
        xute,
        yute,
        name="unimodal_smoke",
        true_k_modes=1,
        k_max=6,
        gate_type="constant",
        y_min=-2.0,
        y_max=12.0,
        seed=7,
    )
    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    a04 = fit_and_report(
        x4,
        y4,
        x4te,
        y4te,
        name="A04_smoke",
        true_k_modes=3,
        k_max=6,
        gate_type="constant",
        y_min=TRIMODAL_Y_MIN,
        y_max=TRIMODAL_Y_MAX,
        num_steps=900,
        seed=42,
    )
    x5, y5, x5te, y5te = make_unimodal_hetero_tensors(
        n_samples=400, test_size=0.3, seed=42
    )
    a05 = fit_and_report(
        x5,
        y5,
        x5te,
        y5te,
        name="A05_smoke",
        true_k_modes=1,
        k_max=6,
        gate_type="constant",
        y_min=A05_Y_MIN,
        y_max=A05_Y_MAX,
        num_steps=900,
        seed=42,
        x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
    )
    out = {
        "A01_smoke": a01,
        "unimodal_smoke": uni,
        "A04_smoke": a04,
        "A05_smoke": a05,
    }
    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = []
    for key, row, n_tr, kmax, steps, gate in (
        ("A01", a01, int(x_tr.shape[0]), 6, 800, "constant"),
        ("unimodal", uni, int(xu.shape[0]), 6, 800, "constant"),
        ("A04", a04, int(x4.shape[0]), 6, 900, "constant"),
        ("A05", a05, int(x5.shape[0]), 6, 900, "constant"),
    ):
        cost_rows.append(
            make_cost_row(
                script="run_midline_sparse.smoke",
                problem=key,
                model="MidlineSparse",
                wall_s=row.get("wall_s"),
                n_train=n_tr,
                k_max=kmax,
                num_steps=steps,
                gate_type=gate,
                num_inits=2,
                num_stage2_inits=1,
                champion=True,
            )
        )
    out["cost"] = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR / "smoke",
        stem="cost",
        title="smoke: MidlineSparse fit cost",
    )
    _json_dump(OUT_DIR / "midline_sparse_smoke.json", out)
    return out


def run_gate2() -> dict:
    """Gate-2 probes from the design plan."""
    gpplus.config.configure_logger()
    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=42)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)

    # 2a: K_max=2 should show ~2 modes; light usage penalty so both arms stay
    a01_k2 = fit_and_report(
        x_tr,
        y_tr,
        x_te,
        y_te,
        name="gate2a_A01_Kmax2",
        true_k_modes=2,
        k_max=2,
        lambda_k=0.02,
        num_steps=900,
    )
    # 2b: overcomplete K_max=6 sparsity → K_eff~2 (Option E), modes~2
    a01_k6 = fit_and_report(
        x_tr,
        y_tr,
        x_te,
        y_te,
        name="gate2b_A01_Kmax6",
        true_k_modes=2,
        k_max=6,
        lambda_k=0.08,
        num_steps=1000,
    )
    xu, yu, xute, yute = make_unimodal(300, 7)
    uni = fit_and_report(
        xu,
        yu,
        xute,
        yute,
        name="gate2b_unimodal_Kmax6",
        true_k_modes=1,
        k_max=6,
        lambda_k=0.10,
        y_min=-2.0,
        y_max=12.0,
        seed=7,
        num_steps=900,
    )
    # 2c: K_eff trajectory already in fit_history
    traj_ok = False
    if a01_k6.get("fit_history_tail"):
        # Compare first logged vs last from a fresh short check — use history from model re-fit stored
        traj_ok = a01_k6["k_eff"] < a01_k6["k_max"] - 0.5

    # 2d: A02 constant vs MLP
    xs_tr, ys_tr, xs_te, ys_te = make_step(200, 200, seed=42)
    a02_const = fit_and_report(
        xs_tr,
        ys_tr,
        xs_te,
        ys_te,
        name="gate2d_A02_constant",
        true_k_modes=1,
        k_max=4,
        gate_type="constant",
        y_min=-0.5,
        y_max=1.5,
        lambda_k=0.08,
        num_steps=500,
        try_both_gates=False,
    )
    a02_mlp = fit_and_report(
        xs_tr,
        ys_tr,
        xs_te,
        ys_te,
        name="gate2d_A02_mlp",
        true_k_modes=1,
        k_max=4,
        gate_type="mlp",
        y_min=-0.5,
        y_max=1.5,
        lambda_k=0.08,
        num_steps=500,
        try_both_gates=False,
    )

    # A04: co-located trimodal with unequal widths (champion p12+rec defaults)
    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    a04 = fit_and_report(
        x4,
        y4,
        x4te,
        y4te,
        name="gate2_A04_trimodal",
        true_k_modes=3,
        k_max=6,
        gate_type="constant",
        y_min=TRIMODAL_Y_MIN,
        y_max=TRIMODAL_Y_MAX,
        num_steps=900,
        seed=42,
    )

    # A05: unimodal skewed contaminated heteroscedastic (soft negative control)
    x5, y5, x5te, y5te = make_unimodal_hetero_tensors(
        n_samples=400, test_size=0.3, seed=42
    )
    a05 = fit_and_report(
        x5,
        y5,
        x5te,
        y5te,
        name="gate2_A05_unimodal_hetero",
        true_k_modes=1,
        k_max=6,
        gate_type="constant",
        y_min=A05_Y_MIN,
        y_max=A05_Y_MAX,
        num_steps=900,
        seed=42,
        x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
    )

    # Tightened bands after active-Sep + orphan-α + sharper keff_tau
    pass_2a = (
        a01_k2["k_modes_mean"] >= 1.5
        and 1.7 <= a01_k2["k_eff"] <= 2.3
        and a01_k2["test_nll"] <= 2.8
    )
    pass_2b = (
        1.5 <= a01_k6["k_eff"] <= 2.5
        and a01_k6["k_hat"] in (1, 2, 3)
        and 1.5 <= a01_k6["k_modes_mean"] <= 2.5
        and uni["k_modes_round"] <= 1
        and uni["k_eff"] <= 1.5
    )
    pass_2c = bool(traj_ok)
    # Mode-structure: both gates locally unimodal; do not regress NLL badly
    pass_2d = (
        a02_const["k_modes_mean"] <= 1.25
        and a02_mlp["k_modes_mean"] <= 1.25
        and a02_mlp["test_nll"] <= a02_const["test_nll"] + 0.2
    )
    # Soft A04/A05 checks (reported; do not block overall Gate-2)
    pass_a04 = bool(
        a04["k_modes_mean"] >= 2.0 and 1.5 <= a04["k_eff"] <= 4.0
    )
    pass_a05 = bool(a05["k_modes_round"] <= 1 and a05["k_eff"] <= 1.5)

    overall = bool(pass_2a and pass_2b and pass_2c and pass_2d)
    summary = {
        "rows": {
            "a01_k2": a01_k2,
            "a01_k6": a01_k6,
            "unimodal": uni,
            "a02_const": a02_const,
            "a02_mlp": a02_mlp,
            "a04_trimodal": a04,
            "a05_unimodal_hetero": a05,
        },
        "pass": {
            "2a_peak_recovery_Kmax2": pass_2a,
            "2b_sparse_Keff_and_modes": pass_2b,
            "2c_Keff_below_Kmax": pass_2c,
            "2d_A02_gate_contrast": pass_2d,
            "a04_trimodal_recovery": pass_a04,
            "a05_no_invented_modes": pass_a05,
            "overall_gate2_pass": overall,
        },
    }
    from cost_tables import make_cost_row, write_cost_artifacts

    gate2_cost = []
    for key, row in summary["rows"].items():
        gate2_cost.append(
            make_cost_row(
                script="run_midline_sparse.gate2",
                problem=key,
                model="MidlineSparse",
                wall_s=row.get("wall_s"),
                n_train=None,
                k_max=row.get("k_max"),
                num_steps=row.get("num_steps"),
                gate_type=row.get("gate_type"),
                num_inits=2,
                num_stage2_inits=1,
                champion=True,
            )
        )
    summary["cost"] = write_cost_artifacts(
        gate2_cost,
        out_dir=OUT_DIR / "gate2",
        stem="cost",
        title="gate2: MidlineSparse fit cost",
    )
    _json_dump(OUT_DIR / "midline_sparse_gate2.json", summary)
    print(json.dumps(summary["pass"], indent=2))
    return summary


def run_gate2_with_mode_loss() -> dict:
    """v2: same Gate-2 A01/uni with lambda_mode > 0 (only call if Gate 2 passed)."""
    gpplus.config.configure_logger()
    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=42)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)
    row = fit_and_report(
        x_tr,
        y_tr,
        x_te,
        y_te,
        name="v2_A01_mode_loss",
        true_k_modes=2,
        k_max=6,
        lambda_k=0.08,
        lambda_mode=0.05,
        num_steps=800,
    )
    xu, yu, xute, yute = make_unimodal(300, 7)
    uni = fit_and_report(
        xu,
        yu,
        xute,
        yute,
        name="v2_uni_mode_loss",
        true_k_modes=1,
        k_max=6,
        lambda_k=0.1,
        lambda_mode=0.05,
        y_min=-2.0,
        y_max=12.0,
        seed=7,
        num_steps=700,
    )
    out = {"A01": row, "unimodal": uni}
    _json_dump(OUT_DIR / "midline_sparse_v2_mode_loss.json", out)
    return out


def run_a04_ablation() -> dict:
    """Phase-1 grid: existing distinction levers on A04 (no new regularizer)."""
    gpplus.config.configure_logger()
    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    # Fixed probe grid including x=1 and x=5
    x_probe = sorted(set(np.linspace(0.0, 10.0, 9).tolist() + [1.0, 5.0]))

    lambda_ks = [0.02, 0.04, 0.08]
    lambda_seps = [0.02, 0.05, 0.1]
    sep_margins = [2.0, 4.0]
    step_list = [900, 1500]

    rows: list[dict] = []
    for lam_k in lambda_ks:
        for lam_sep in lambda_seps:
            for margin in sep_margins:
                for steps in step_list:
                    name = (
                        f"a04_lk{lam_k:g}_ls{lam_sep:g}_m{margin:g}_s{steps}"
                    )
                    row = fit_and_report(
                        x4,
                        y4,
                        x4te,
                        y4te,
                        name=name,
                        true_k_modes=3,
                        k_max=6,
                        gate_type="constant",
                        y_min=TRIMODAL_Y_MIN,
                        y_max=TRIMODAL_Y_MAX,
                        lambda_k=lam_k,
                        lambda_sep=lam_sep,
                        sep_margin=margin,
                        num_steps=steps,
                        num_inits=2,
                        seed=42,
                        x_probe=x_probe,
                    )
                    # Drop bulky history from ablation table
                    row.pop("fit_history_tail", None)
                    rows.append(row)

    # Best by k_modes_mean, then lower NLL
    ranked = sorted(
        rows,
        key=lambda r: (-float(r["k_modes_mean"]), float(r["test_nll"])),
    )
    best = ranked[0] if ranked else None
    recovered = bool(best is not None and best["k_modes_mean"] >= 2.5)
    out = {
        "rows": rows,
        "best": best,
        "recovered_k_modes_ge_2_5": recovered,
        "n_configs": len(rows),
        "criterion": "k_modes_mean >= 2.5 (then min test_nll)",
    }
    _json_dump(OUT_DIR / "a04_robustness_ablation.json", out)
    print(
        f"[A04 ablation] n={len(rows)} best K_modes={best['k_modes_mean']:.2f} "
        f"nll={best['test_nll']:.3f} recovered={recovered}"
        if best
        else "[A04 ablation] empty"
    )
    return out


def run_a04_p1_verify() -> dict:
    """P1 free-scale multi-seed A04 + quick A01/uni no-regress checks."""
    gpplus.config.configure_logger()
    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    x_probe = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
    seeds = [42, 43, 44, 45, 46]
    a04_rows: list[dict] = []
    for seed in seeds:
        row = fit_and_report(
            x4,
            y4,
            x4te,
            y4te,
            name=f"a04_p1_s{seed}",
            true_k_modes=3,
            k_max=6,
            gate_type="constant",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            lambda_k=0.02,
            lambda_sep=0.1,
            sep_margin=2.0,
            lambda_overlap=0.08,
            overlap_margin=1.0,
            free_expert_scale=True,
            lambda_beta=0.01,
            num_steps=900,
            num_inits=2,
            seed=seed,
            x_probe=x_probe,
        )
        row.pop("fit_history_tail", None)
        a04_rows.append(row)

    # Baseline without free scale (seed 42) for comparison
    base = fit_and_report(
        x4,
        y4,
        x4te,
        y4te,
        name="a04_baseline_no_p1_s42",
        true_k_modes=3,
        k_max=6,
        gate_type="constant",
        y_min=TRIMODAL_Y_MIN,
        y_max=TRIMODAL_Y_MAX,
        lambda_k=0.02,
        lambda_sep=0.1,
        sep_margin=2.0,
        lambda_overlap=0.08,
        free_expert_scale=False,
        num_steps=900,
        seed=42,
        x_probe=x_probe,
    )
    base.pop("fit_history_tail", None)

    km_means = [float(r["k_modes_mean"]) for r in a04_rows]
    mean_km = float(np.mean(km_means))
    any_both = any(
        (r.get("k_modes_at_x1") or 0) >= 2 and (r.get("k_modes_at_x5") or 0) >= 2
        for r in a04_rows
    )
    accept_soft = bool(mean_km >= 2.0 and any_both)

    # Gate-2 A01 Kmax=2 + unimodal (free scale off)
    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=42)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)
    a01 = fit_and_report(
        x_tr,
        y_tr,
        x_te,
        y_te,
        name="p1_check_A01_Kmax2",
        true_k_modes=2,
        k_max=2,
        lambda_k=0.02,
        num_steps=900,
        free_expert_scale=False,
    )
    a01.pop("fit_history_tail", None)
    xu, yu, xute, yute = make_unimodal(300, 7)
    uni = fit_and_report(
        xu,
        yu,
        xute,
        yute,
        name="p1_check_unimodal",
        true_k_modes=1,
        k_max=6,
        lambda_k=0.10,
        y_min=-2.0,
        y_max=12.0,
        seed=7,
        num_steps=900,
        free_expert_scale=False,
    )
    uni.pop("fit_history_tail", None)
    a01_ok = bool(
        a01["k_modes_mean"] >= 1.5
        and 1.7 <= a01["k_eff"] <= 2.3
        and a01["test_nll"] <= 2.8
    )
    uni_ok = bool(uni["k_modes_round"] <= 1 and uni["k_eff"] <= 1.5)

    out = {
        "a04_p1_seeds": a04_rows,
        "a04_baseline_no_p1": base,
        "summary": {
            "mean_k_modes": mean_km,
            "per_seed_k_modes": km_means,
            "any_seed_x1_and_x5_ge2": any_both,
            "accept_soft": accept_soft,
            "better_than_baseline_s42": bool(
                a04_rows[0]["k_modes_mean"] >= base["k_modes_mean"]
            ),
        },
        "gate2_sanity": {
            "a01": a01,
            "unimodal": uni,
            "a01_ok": a01_ok,
            "uni_ok": uni_ok,
            "no_regress": bool(a01_ok and uni_ok),
        },
    }
    _json_dump(OUT_DIR / "a04_p1_free_scale.json", out)
    print(
        f"[A04 P1] mean K_modes={mean_km:.2f} accept={accept_soft} "
        f"A01_ok={a01_ok} uni_ok={uni_ok}"
    )
    return out


def run_a04_beta_prior_ablation() -> dict:
    """Ablate λ_β ∈ {0, 1e-4, 0.01} to test whether NLL wants β>1."""
    gpplus.config.configure_logger()
    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    x_probe = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
    lambdas = [0.0, 1e-4, 0.01]
    seeds = [42, 43, 44]
    rows: list[dict] = []
    for lam_b in lambdas:
        for seed in seeds:
            row = fit_and_report(
                x4,
                y4,
                x4te,
                y4te,
                name=f"a04_lbeta{lam_b:g}_s{seed}",
                true_k_modes=3,
                k_max=6,
                gate_type="constant",
                y_min=TRIMODAL_Y_MIN,
                y_max=TRIMODAL_Y_MAX,
                lambda_k=0.02,
                lambda_sep=0.1,
                sep_margin=2.0,
                lambda_overlap=0.08,
                free_expert_scale=True,
                lambda_beta=lam_b,
                num_steps=900,
                num_inits=2,
                seed=seed,
                x_probe=x_probe,
            )
            row.pop("fit_history_tail", None)
            beta = row.get("beta") or []
            row["beta_max"] = float(max(beta)) if beta else 1.0
            row["beta_mean"] = float(sum(beta) / len(beta)) if beta else 1.0
            rows.append(row)

    by_lam: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r['lambda_beta']:g}"
        by_lam.setdefault(key, []).append(r)

    summary = {}
    beta_unused = True
    for key, group in by_lam.items():
        bmax = max(float(r["beta_max"]) for r in group)
        bmean = float(np.mean([r["beta_mean"] for r in group]))
        km = float(np.mean([r["k_modes_mean"] for r in group]))
        summary[key] = {
            "mean_k_modes": km,
            "mean_beta": bmean,
            "max_beta": bmax,
            "n": len(group),
        }
        # "Used" if any run pushes beta meaningfully above init (~1.13) or >>1.05
        if bmax > 1.15:
            beta_unused = False

    out = {
        "rows": rows,
        "summary_by_lambda_beta": summary,
        "beta_unused_under_zero_prior": bool(
            summary.get("0", {}).get("max_beta", 1.0) <= 1.15
        ),
        "recommend_p4_anneal": beta_unused
        or bool(summary.get("0", {}).get("max_beta", 1.0) <= 1.15),
        "criterion": "beta_max > 1.15 indicates NLL uses free scale",
    }
    _json_dump(OUT_DIR / "a04_beta_prior_ablation.json", out)
    print(
        f"[A04 beta ablation] unused@λβ=0="
        f"{out['beta_unused_under_zero_prior']} recommend_P4={out['recommend_p4_anneal']}"
    )
    for key, s in summary.items():
        print(
            f"  λβ={key}: mean_km={s['mean_k_modes']:.2f} "
            f"mean_β={s['mean_beta']:.3f} max_β={s['max_beta']:.3f}"
        )
    return out


def run_orphan_delta_reg_test() -> dict:
    """Compare A04 with/without orphan coupling + δ L2; write under results/orphan_delta_reg/."""
    from plot_midline_sparse import (
        save_alpha_gate_bars,
        save_conditional_pdf,
        save_scatter_midline_experts,
        save_training_curves,
        save_training_loss_subplots,
    )
    from load_trimodal_data import (
        TRIMODAL_SIGMA,
        trimodal_mode_means,
    )

    gpplus.config.configure_logger()
    out_root = OUT_DIR / "orphan_delta_reg"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    x_probe = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
    x_grid = np.linspace(float(x4.min()), float(x4.max()), 200)
    true_means = trimodal_mode_means(x_grid)
    true_labels = [
        rf"true mode $k={k}$ ($\sigma={TRIMODAL_SIGMA[k]:g}$)" for k in range(3)
    ]

    configs = [
        {
            "tag": "baseline",
            "title": "A04 baseline (weak orphan, no $\\delta$ L2)",
            "lambda_orphan": 0.05,
            "lambda_delta": 0.0,
        },
        {
            "tag": "treated",
            "title": "A04 treated (orphan=0.25 + $\\pi$–$\\alpha$ couple + $\\delta$ L2)",
            "lambda_orphan": 0.25,
            "lambda_delta": 0.02,
        },
    ]

    rows: list[dict] = []
    loss_histories: dict[str, list[dict]] = {}
    for cfg in configs:
        tag = cfg["tag"]
        row, model = fit_and_report(
            x4,
            y4,
            x4te,
            y4te,
            name=f"a04_{tag}",
            true_k_modes=3,
            k_max=6,
            gate_type="constant",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            lambda_k=0.02,
            lambda_sep=0.1,
            sep_margin=2.0,
            lambda_overlap=0.08,
            free_expert_scale=False,
            lambda_beta=0.0,
            lambda_k_start=0.0,
            lambda_orphan=float(cfg["lambda_orphan"]),
            lambda_delta=float(cfg["lambda_delta"]),
            num_steps=900,
            seed=42,
            x_probe=x_probe,
            return_model=True,
        )
        pred = model.predict(
            torch.linspace(
                float(x4.min()), float(x4.max()), 200, dtype=torch.float64
            ).unsqueeze(-1)
        )
        means = pred["expert_means"].detach().cpu().numpy()
        alpha = pred["alpha"].detach().cpu().numpy()
        if alpha.ndim == 2:
            alpha_bar = alpha.mean(axis=0)
        else:
            alpha_bar = alpha
        pi = pred["gate_weights"].detach().cpu().numpy().mean(axis=0)
        mass = alpha_bar * pi
        active = mass >= 0.03
        min_active_mean = float(means[:, active].min()) if active.any() else float("nan")
        dead_mask = alpha_bar < 0.1
        max_dead_pi = float(pi[dead_mask].max()) if dead_mask.any() else 0.0
        max_live_pi = float(pi[~dead_mask].max()) if (~dead_mask).any() else 0.0

        row.pop("fit_history_tail", None)
        row["tag"] = tag
        row["min_active_expert_mean"] = min_active_mean
        row["max_dead_pi"] = max_dead_pi
        row["max_live_pi"] = max_live_pi
        row["no_runaway"] = bool(min_active_mean > -15.0)
        row["no_dead_pi_dominance"] = bool(max_dead_pi < max_live_pi)
        rows.append(row)

        save_scatter_midline_experts(
            model,
            out_path=fig_dir / f"a04_{tag}_scatter.png",
            title=cfg["title"],
            true_mode_means=true_means,
            true_mode_labels=true_labels,
        )
        x_med = float(np.median(x4.numpy()))
        save_conditional_pdf(
            model,
            x_value=x_med,
            out_path=fig_dir / f"a04_{tag}_pdf.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"{cfg['title']}  |  p(y|x={x_med:.1f})",
        )
        save_conditional_pdf(
            model,
            x_value=1.0,
            out_path=fig_dir / f"a04_{tag}_pdf_x1.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"{cfg['title']}  |  p(y|x=1)",
        )
        save_alpha_gate_bars(
            model,
            out_path=fig_dir / f"a04_{tag}_activity.png",
            title=f"A04 {tag} activity",
        )
        save_training_curves(
            model._fit_history,
            out_path=fig_dir / f"a04_{tag}_train.png",
            title=f"A04 {tag} Stage-2",
        )
        loss_histories[tag] = list(model._fit_history or [])

    loss_fig = save_training_loss_subplots(
        loss_histories,
        out_path=fig_dir / "a04_training_loss.png",
        model_labels={c["tag"]: c["title"] for c in configs},
        model_order=[c["tag"] for c in configs],
        title="A04 orphan/δ-reg  |  Stage-2 training loss",
    )
    print(f"Wrote {loss_fig}")

    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=42)
    a01 = fit_and_report(
        torch.as_tensor(data["X_train"], dtype=torch.float64),
        torch.as_tensor(data["y_train"], dtype=torch.float64),
        torch.as_tensor(data["X_test"], dtype=torch.float64),
        torch.as_tensor(data["y_test"], dtype=torch.float64),
        name="a01_sanity",
        true_k_modes=2,
        k_max=2,
        lambda_k=0.02,
        num_steps=900,
        seed=42,
    )
    a01.pop("fit_history_tail", None)
    a01_ok = bool(
        a01["k_modes_mean"] >= 1.5
        and 1.7 <= a01["k_eff"] <= 2.3
        and a01["test_nll"] <= 2.8
    )

    treated = next(r for r in rows if r["tag"] == "treated")
    out = {
        "rows": rows,
        "a01_sanity": a01,
        "checks": {
            "treated_no_runaway": treated["no_runaway"],
            "treated_no_dead_pi_dominance": treated["no_dead_pi_dominance"],
            "a01_ok": a01_ok,
            "pass": bool(
                treated["no_runaway"]
                and treated["no_dead_pi_dominance"]
                and a01_ok
            ),
        },
        "figure_dir": str(fig_dir),
        "training_loss_figure": str(loss_fig),
    }
    _json_dump(out_root / "metrics.json", out)
    print(
        f"[orphan_delta_reg] treated runaway_ok={treated['no_runaway']} "
        f"dead_pi_ok={treated['no_dead_pi_dominance']} a01_ok={a01_ok} "
        f"→ {out_root}"
    )
    return out


def run_noise_alleviation_probes() -> dict:
    """A04 singles (1/2/3) + combos (1+2, 1+2+3) vs baseline → results/noise_alleviation/."""
    from plot_midline_sparse import (
        save_alpha_gate_bars,
        save_conditional_pdf,
        save_scatter_midline_experts,
        save_training_curves,
        save_training_loss_subplots,
    )
    from load_trimodal_data import (
        TRIMODAL_SIGMA,
        trimodal_mode_means,
    )

    gpplus.config.configure_logger()
    out_root = OUT_DIR / "noise_alleviation"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x4, y4, x4te, y4te = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=42)
    x_probe = [0.0, 1.0, 2.5, 5.0, 7.0, 7.5, 10.0]
    x_grid = np.linspace(float(x4.min()), float(x4.max()), 200)
    true_means = trimodal_mode_means(x_grid)
    true_labels = [
        rf"true mode $k={k}$ ($\sigma={TRIMODAL_SIGMA[k]:g}$)" for k in range(3)
    ]

    # Shared A04 treated baseline knobs; arms toggle only their levers.
    base = {
        "lambda_orphan": 0.25,
        "lambda_delta": 0.02,
        "lambda_k": 0.02,
        "lambda_k_start": 0.0,
        "lambda_sep": 0.1,
        "sep_margin": 2.0,
        "lambda_overlap": 0.08,
        "overlap_margin": 1.0,
        "free_expert_scale": False,
        "lambda_beta": 0.0,
        "use_full_gp_variance": True,
        "lambda_gamma": 0.0,
        "gamma_init_logit": 1.1,
    }
    lever1 = {"use_full_gp_variance": False}
    lever2 = {"lambda_gamma": 0.1, "gamma_init_logit": 2.2}
    lever3 = {
        "lambda_overlap": 0.25,
        "overlap_margin": 1.5,
        "lambda_sep": 0.2,
        "sep_margin": 3.0,
    }

    arms: list[dict] = [
        {"tag": "baseline", "title": "A04 baseline (treated)", "levers": [], **base},
        {
            "tag": "p1_cap_sigma0",
            "title": "A04 P1: capped $\\sigma_0$",
            "levers": [1],
            **{**base, **lever1},
        },
        {
            "tag": "p2_gamma_prior",
            "title": "A04 P2: $\\gamma$ carve-out prior",
            "levers": [2],
            **{**base, **lever2},
        },
        {
            "tag": "p3_mean_sep",
            "title": "A04 P3: stronger mean separation",
            "levers": [3],
            **{**base, **lever3},
        },
        {
            "tag": "p12_cap_gamma",
            "title": "A04 P1+P2: cap $\\sigma_0$ + $\\gamma$ prior",
            "levers": [1, 2],
            **{**base, **lever1, **lever2},
        },
        {
            "tag": "p123_all",
            "title": "A04 P1+P2+P3: all noise levers",
            "levers": [1, 2, 3],
            **{**base, **lever1, **lever2, **lever3},
        },
    ]

    rows: list[dict] = []
    loss_histories: dict[str, list[dict]] = {}
    for arm in arms:
        tag = arm["tag"]
        row, model = fit_and_report(
            x4,
            y4,
            x4te,
            y4te,
            name=f"a04_{tag}",
            true_k_modes=3,
            k_max=6,
            gate_type="constant",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            lambda_k=float(arm["lambda_k"]),
            lambda_sep=float(arm["lambda_sep"]),
            sep_margin=float(arm["sep_margin"]),
            lambda_overlap=float(arm["lambda_overlap"]),
            overlap_margin=float(arm["overlap_margin"]),
            free_expert_scale=bool(arm["free_expert_scale"]),
            lambda_beta=float(arm["lambda_beta"]),
            lambda_k_start=float(arm["lambda_k_start"]),
            lambda_orphan=float(arm["lambda_orphan"]),
            lambda_delta=float(arm["lambda_delta"]),
            lambda_gamma=float(arm["lambda_gamma"]),
            use_full_gp_variance=bool(arm["use_full_gp_variance"]),
            gamma_init_logit=float(arm["gamma_init_logit"]),
            num_steps=900,
            seed=42,
            x_probe=x_probe,
            return_model=True,
        )
        loss_histories[tag] = list(model._fit_history or [])
        pred = model.predict(
            torch.linspace(
                float(x4.min()), float(x4.max()), 200, dtype=torch.float64
            ).unsqueeze(-1)
        )
        means = pred["expert_means"].detach().cpu().numpy()
        alpha = pred["alpha"].detach().cpu().numpy()
        alpha_bar = alpha.mean(axis=0) if alpha.ndim == 2 else alpha
        pi = pred["gate_weights"].detach().cpu().numpy().mean(axis=0)
        mass = alpha_bar * pi
        active = mass >= 0.03
        min_active_mean = float(means[:, active].min()) if active.any() else float("nan")

        row.pop("fit_history_tail", None)
        row["tag"] = tag
        row["levers"] = list(arm["levers"])
        row["min_active_expert_mean"] = min_active_mean
        row["no_runaway"] = bool(min_active_mean > -15.0)
        rows.append(row)

        save_scatter_midline_experts(
            model,
            out_path=fig_dir / f"{tag}_scatter.png",
            title=arm["title"],
            true_mode_means=true_means,
            true_mode_labels=true_labels,
        )
        x_med = float(np.median(x4.numpy()))
        save_conditional_pdf(
            model,
            x_value=x_med,
            out_path=fig_dir / f"{tag}_pdf.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"{arm['title']}  |  p(y|x={x_med:.1f})",
        )
        save_conditional_pdf(
            model,
            x_value=1.0,
            out_path=fig_dir / f"{tag}_pdf_x1.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"{arm['title']}  |  p(y|x=1)",
        )
        save_conditional_pdf(
            model,
            x_value=7.0,
            out_path=fig_dir / f"{tag}_pdf_x7.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"{arm['title']}  |  p(y|x=7)",
        )
        save_alpha_gate_bars(
            model,
            out_path=fig_dir / f"{tag}_activity.png",
            title=f"{tag} activity",
        )
        save_training_curves(
            model._fit_history,
            out_path=fig_dir / f"{tag}_train.png",
            title=f"{tag} Stage-2",
        )

    loss_fig = save_training_loss_subplots(
        loss_histories,
        out_path=fig_dir / "a04_training_loss.png",
        model_labels={a["tag"]: a["title"] for a in arms},
        model_order=[a["tag"] for a in arms],
        title="A04 noise levers  |  Stage-2 training loss",
    )
    print(f"Wrote {loss_fig}")

    baseline = next(r for r in rows if r["tag"] == "baseline")
    summary = []
    for r in rows:
        summary.append(
            {
                "tag": r["tag"],
                "levers": r["levers"],
                "k_modes_mean": r["k_modes_mean"],
                "k_modes_at_x1": r["k_modes_at_x1"],
                "k_modes_at_x5": r["k_modes_at_x5"],
                "k_modes_at_x7": r["k_modes_at_x7"],
                "k_hat": r["k_hat"],
                "k_eff": r["k_eff"],
                "test_nll": r["test_nll"],
                "mean_gamma": r["mean_gamma"],
                "no_runaway": r["no_runaway"],
                "delta_k_modes_vs_baseline": float(
                    r["k_modes_mean"] - baseline["k_modes_mean"]
                ),
            }
        )

    out = {
        "rows": rows,
        "summary": summary,
        "figure_dir": str(fig_dir),
        "training_loss_figure": str(loss_fig),
    }
    _json_dump(out_root / "metrics.json", out)
    print(f"[noise_alleviation] wrote {len(rows)} arms → {out_root}")
    for s in summary:
        print(
            f"  {s['tag']}: K_modes~{s['k_modes_mean']:.2f} "
            f"x1={s['k_modes_at_x1']} x7={s['k_modes_at_x7']} "
            f"γ̄={s['mean_gamma']:.3f} Δkm={s['delta_k_modes_vs_baseline']:+.2f}"
        )
    return out


# Champion: lean post-factorial (usage + merge + alive + delta + alpha).
_CHAMPION = {
    "lambda_orphan": 0.0,
    "lambda_delta": 0.01,
    "lambda_k": 0.01,
    "lambda_k_start": 0.0,
    "lambda_sep": 0.0,
    "sep_margin": 2.0,
    "lambda_overlap": 0.0,
    "overlap_margin": 1.0,
    # β free-scale explored; inert on A04 — keep muted (β≡1).
    "free_expert_scale": False,
    "lambda_beta": 0.0,
    "use_full_gp_variance": False,
    "lambda_gamma": 0.0,
    "gamma_init_logit": 2.2,
    "lambda_rec": 0.0,
    "rec_margin": 1.0,
    "lambda_alpha": 0.05,
    "alive_weight": 0.75,
}

_CLASSIC_LIGHT = {
    "lambda_orphan": 0.05,
    "lambda_delta": 0.0,
    "lambda_k": 0.02,
    "lambda_k_start": None,
    "lambda_sep": 0.05,
    "sep_margin": 2.0,
    "lambda_overlap": 0.0,
    "overlap_margin": 1.0,
    "free_expert_scale": False,
    "lambda_beta": 0.01,
    "use_full_gp_variance": True,
    "lambda_gamma": 0.0,
    "gamma_init_logit": 1.1,
    "lambda_rec": 0.0,
    "rec_margin": 1.0,
}


def _agg_numeric(rows: list[dict], keys: list[str]) -> dict:
    out: dict = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


def run_limitation_stress(
    seeds: list[int] | None = None,
) -> dict:
    """Multi-seed / multi-problem stress of champion (p12) vs classic controls.

    Writes ``results/limitation_stress/metrics.json``.
    """
    gpplus.config.configure_logger()
    seeds = list(seeds or [7, 21, 42, 99, 123])
    out_root = OUT_DIR / "limitation_stress"
    out_root.mkdir(parents=True, exist_ok=True)

    problems: list[dict] = []

    def _fit_cfg(
        *,
        problem: str,
        cfg_name: str,
        cfg: dict,
        seed: int,
        x_tr,
        y_tr,
        x_te,
        y_te,
        true_k: int,
        k_max: int,
        gate: str,
        y_min: float,
        y_max: float,
        x_probe: list[float] | None,
        num_steps: int = 900,
    ) -> dict:
        row = fit_and_report(
            x_tr,
            y_tr,
            x_te,
            y_te,
            name=f"{problem}_{cfg_name}_s{seed}",
            true_k_modes=true_k,
            k_max=k_max,
            gate_type=gate,
            offset_type="mlp",
            y_min=y_min,
            y_max=y_max,
            lambda_k=float(cfg["lambda_k"]),
            lambda_sep=float(cfg["lambda_sep"]),
            sep_margin=float(cfg["sep_margin"]),
            lambda_overlap=float(cfg["lambda_overlap"]),
            overlap_margin=float(cfg["overlap_margin"]),
            free_expert_scale=bool(cfg["free_expert_scale"]),
            lambda_beta=float(cfg["lambda_beta"]),
            lambda_k_start=cfg["lambda_k_start"],
            lambda_alpha=float(cfg.get("lambda_alpha", 0.05)),
            lambda_orphan=float(cfg["lambda_orphan"]),
            lambda_delta=float(cfg["lambda_delta"]),
            lambda_gamma=float(cfg["lambda_gamma"]),
            lambda_rec=float(cfg.get("lambda_rec", 0.0)),
            rec_margin=float(cfg.get("rec_margin", 1.0)),
            alive_weight=float(cfg.get("alive_weight", 0.75)),
            use_full_gp_variance=bool(cfg["use_full_gp_variance"]),
            gamma_init_logit=float(cfg["gamma_init_logit"]),
            num_steps=num_steps,
            seed=seed,
            x_probe=x_probe,
        )
        row.pop("fit_history_tail", None)
        row["problem"] = problem
        row["cfg"] = cfg_name
        row["seed"] = seed
        # Diagnostic flags
        alpha = np.asarray(row.get("alpha", []), dtype=float)
        gamma = np.asarray(row.get("gamma", []), dtype=float)
        row["n_alpha_alive"] = int((alpha >= 0.5).sum()) if alpha.size else 0
        row["mean_gamma_alive"] = (
            float(gamma[alpha >= 0.5].mean())
            if alpha.size and (alpha >= 0.5).any()
            else float(row.get("mean_gamma", float("nan")))
        )
        return row

    for seed in seeds:
        # --- A04 trimodal (champion only; classic known-weak) ---
        x4, y4, x4te, y4te = make_trimodal_tensors(
            n_samples=400, test_size=0.3, seed=seed
        )
        problems.append(
            _fit_cfg(
                problem="A04",
                cfg_name="champion_p12",
                cfg=_CHAMPION,
                seed=seed,
                x_tr=x4,
                y_tr=y4,
                x_te=x4te,
                y_te=y4te,
                true_k=3,
                k_max=6,
                gate="constant",
                y_min=TRIMODAL_Y_MIN,
                y_max=TRIMODAL_Y_MAX,
                x_probe=[0.0, 1.0, 2.5, 5.0, 7.0, 10.0],
            )
        )
        problems.append(
            _fit_cfg(
                problem="A04",
                cfg_name="classic",
                cfg=_CLASSIC_LIGHT,
                seed=seed,
                x_tr=x4,
                y_tr=y4,
                x_te=x4te,
                y_te=y4te,
                true_k=3,
                k_max=6,
                gate="constant",
                y_min=TRIMODAL_Y_MIN,
                y_max=TRIMODAL_Y_MAX,
                x_probe=[0.0, 1.0, 2.5, 5.0, 7.0, 10.0],
            )
        )

        # --- A01 bimodal: champion transfer vs classic Kmax=2 ---
        data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=seed)
        xa = torch.as_tensor(data["X_train"], dtype=torch.float64)
        ya = torch.as_tensor(data["y_train"], dtype=torch.float64)
        xate = torch.as_tensor(data["X_test"], dtype=torch.float64)
        yate = torch.as_tensor(data["y_test"], dtype=torch.float64)
        problems.append(
            _fit_cfg(
                problem="A01",
                cfg_name="champion_p12",
                cfg=_CHAMPION,
                seed=seed,
                x_tr=xa,
                y_tr=ya,
                x_te=xate,
                y_te=yate,
                true_k=2,
                k_max=6,
                gate="constant",
                y_min=-2.0,
                y_max=30.0,
                x_probe=[0.0, 1.0, 5.0, 7.0, 10.0],
            )
        )
        problems.append(
            _fit_cfg(
                problem="A01",
                cfg_name="classic_k2",
                cfg=_CLASSIC_LIGHT,
                seed=seed,
                x_tr=xa,
                y_tr=ya,
                x_te=xate,
                y_te=yate,
                true_k=2,
                k_max=2,
                gate="constant",
                y_min=-2.0,
                y_max=30.0,
                x_probe=[0.0, 1.0, 5.0, 7.0, 10.0],
            )
        )

        # --- Unimodal negative control ---
        xu, yu, xute, yute = make_unimodal(300, seed)
        problems.append(
            _fit_cfg(
                problem="unimodal",
                cfg_name="champion_p12",
                cfg=_CHAMPION,
                seed=seed,
                x_tr=xu,
                y_tr=yu,
                x_te=xute,
                y_te=yute,
                true_k=1,
                k_max=6,
                gate="constant",
                y_min=-2.0,
                y_max=12.0,
                x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
            )
        )
        problems.append(
            _fit_cfg(
                problem="unimodal",
                cfg_name="classic",
                cfg=_CLASSIC_LIGHT,
                seed=seed,
                x_tr=xu,
                y_tr=yu,
                x_te=xute,
                y_te=yute,
                true_k=1,
                k_max=6,
                gate="constant",
                y_min=-2.0,
                y_max=12.0,
                x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
            )
        )

        # --- A05 unimodal skewed contaminated heteroscedastic ---
        x5, y5, x5te, y5te = make_unimodal_hetero_tensors(
            n_samples=400, test_size=0.3, seed=seed
        )
        problems.append(
            _fit_cfg(
                problem="A05",
                cfg_name="champion_p12",
                cfg=_CHAMPION,
                seed=seed,
                x_tr=x5,
                y_tr=y5,
                x_te=x5te,
                y_te=y5te,
                true_k=1,
                k_max=6,
                gate="constant",
                y_min=A05_Y_MIN,
                y_max=A05_Y_MAX,
                x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
            )
        )
        problems.append(
            _fit_cfg(
                problem="A05",
                cfg_name="classic",
                cfg=_CLASSIC_LIGHT,
                seed=seed,
                x_tr=x5,
                y_tr=y5,
                x_te=x5te,
                y_te=y5te,
                true_k=1,
                k_max=6,
                gate="constant",
                y_min=A05_Y_MIN,
                y_max=A05_Y_MAX,
                x_probe=[0.0, 2.5, 5.0, 7.5, 10.0],
            )
        )

        # --- A02 step: MLP gate (x-dependent); champion transfer ---
        xs, ys, xste, yste = make_step(200, 200, 0.05, seed)
        problems.append(
            _fit_cfg(
                problem="A02",
                cfg_name="champion_mlp",
                cfg=_CHAMPION,
                seed=seed,
                x_tr=xs,
                y_tr=ys,
                x_te=xste,
                y_te=yste,
                true_k=2,
                k_max=4,
                gate="mlp",
                y_min=-0.5,
                y_max=1.5,
                x_probe=[0.1, 0.3, 0.5, 0.7, 0.9],
            )
        )
        problems.append(
            _fit_cfg(
                problem="A02",
                cfg_name="classic_mlp",
                cfg=_CLASSIC_LIGHT,
                seed=seed,
                x_tr=xs,
                y_tr=ys,
                x_te=xste,
                y_te=yste,
                true_k=2,
                k_max=4,
                gate="mlp",
                y_min=-0.5,
                y_max=1.5,
                x_probe=[0.1, 0.3, 0.5, 0.7, 0.9],
            )
        )

    # Aggregate + pass rates
    groups: dict[str, list[dict]] = {}
    for r in problems:
        key = f"{r['problem']}/{r['cfg']}"
        groups.setdefault(key, []).append(r)

    metric_keys = [
        "k_modes_mean",
        "k_eff",
        "k_hat",
        "test_nll",
        "mean_gamma",
        "k_modes_at_x1",
        "k_modes_at_x7",
    ]
    aggregates = {k: _agg_numeric(v, metric_keys) for k, v in groups.items()}

    def _rate(rows: list[dict], pred) -> float:
        if not rows:
            return float("nan")
        return float(sum(1 for r in rows if pred(r)) / len(rows))

    pass_rates = {
        "A04/champion_p12": {
            "modes_ge_2": _rate(
                groups.get("A04/champion_p12", []),
                lambda r: r["k_modes_mean"] >= 2.0,
            ),
            "x1_ge_2": _rate(
                groups.get("A04/champion_p12", []),
                lambda r: (r.get("k_modes_at_x1") or 0) >= 2,
            ),
            "x7_ge_2": _rate(
                groups.get("A04/champion_p12", []),
                lambda r: (r.get("k_modes_at_x7") or 0) >= 2,
            ),
            "k_hat_in_2_4": _rate(
                groups.get("A04/champion_p12", []),
                lambda r: 2 <= int(r["k_hat"]) <= 4,
            ),
        },
        "A01/champion_p12": {
            "modes_ge_1_5": _rate(
                groups.get("A01/champion_p12", []),
                lambda r: r["k_modes_mean"] >= 1.5,
            ),
            "keff_band": _rate(
                groups.get("A01/champion_p12", []),
                lambda r: 1.5 <= r["k_eff"] <= 2.5,
            ),
        },
        "A01/classic_k2": {
            "modes_ge_1_5": _rate(
                groups.get("A01/classic_k2", []),
                lambda r: r["k_modes_mean"] >= 1.5,
            ),
            "keff_band": _rate(
                groups.get("A01/classic_k2", []),
                lambda r: 1.5 <= r["k_eff"] <= 2.5,
            ),
        },
        "unimodal/champion_p12": {
            "no_extra_modes": _rate(
                groups.get("unimodal/champion_p12", []),
                lambda r: r["k_modes_mean"] <= 1.5,
            ),
            "k_hat_le_2": _rate(
                groups.get("unimodal/champion_p12", []),
                lambda r: int(r["k_hat"]) <= 2,
            ),
        },
        "unimodal/classic": {
            "no_extra_modes": _rate(
                groups.get("unimodal/classic", []),
                lambda r: r["k_modes_mean"] <= 1.5,
            ),
            "k_hat_le_2": _rate(
                groups.get("unimodal/classic", []),
                lambda r: int(r["k_hat"]) <= 2,
            ),
        },
        "A05/champion_p12": {
            "no_extra_modes": _rate(
                groups.get("A05/champion_p12", []),
                lambda r: r["k_modes_mean"] <= 1.5,
            ),
            "k_hat_le_2": _rate(
                groups.get("A05/champion_p12", []),
                lambda r: int(r["k_hat"]) <= 2,
            ),
            "keff_le_1_5": _rate(
                groups.get("A05/champion_p12", []),
                lambda r: r["k_eff"] <= 1.5,
            ),
        },
        "A05/classic": {
            "no_extra_modes": _rate(
                groups.get("A05/classic", []),
                lambda r: r["k_modes_mean"] <= 1.5,
            ),
            "k_hat_le_2": _rate(
                groups.get("A05/classic", []),
                lambda r: int(r["k_hat"]) <= 2,
            ),
        },
        "A02/champion_mlp": {
            "modes_ge_1_5": _rate(
                groups.get("A02/champion_mlp", []),
                lambda r: r["k_modes_mean"] >= 1.5,
            ),
        },
        "A02/classic_mlp": {
            "modes_ge_1_5": _rate(
                groups.get("A02/classic_mlp", []),
                lambda r: r["k_modes_mean"] >= 1.5,
            ),
        },
    }

    # Auto weaknesses / next levers
    weaknesses: list[str] = []
    next_levers: list[str] = []
    a04c = aggregates.get("A04/champion_p12", {})
    a04cl = aggregates.get("A04/classic", {})
    if a04c.get("k_modes_mean", {}).get("mean") is not None:
        if a04c["k_modes_mean"]["std"] > 0.4:
            weaknesses.append(
                f"A04 champion seed-fragile: K_modes std="
                f"{a04c['k_modes_mean']['std']:.2f}"
            )
            next_levers.append(
                "Multi-init Stage-2 / residual quantile seeds (P3) for A04 stability"
            )
        if pass_rates["A04/champion_p12"]["x1_ge_2"] < 0.8:
            weaknesses.append(
                f"A04 x=1 peak recovery only "
                f"{pass_rates['A04/champion_p12']['x1_ge_2']:.0%} of seeds"
            )
            next_levers.append(
                "Tune std0_cap_frac grid (0.2–0.4) or stronger γ prior on A04"
            )
        if (
            a04c["k_modes_mean"]["mean"] < 2.5
            and a04c["k_modes_mean"]["mean"]
            > (a04cl.get("k_modes_mean", {}).get("mean") or 0)
        ):
            next_levers.append(
                "Promote champion_p12 into A04 smoke/default; still short of K=3"
            )
    uni_ch = pass_rates["unimodal/champion_p12"]
    if uni_ch["no_extra_modes"] < 0.8:
        weaknesses.append(
            f"Champion invents modes on unimodal "
            f"({uni_ch['no_extra_modes']:.0%} clean)"
        )
        next_levers.append(
            "Problem-adaptive λ_k / disable γ prior when residuals look unimodal"
        )
    a05_ch = pass_rates["A05/champion_p12"]
    if a05_ch["no_extra_modes"] < 0.8:
        weaknesses.append(
            f"Champion invents modes on A05 hetero "
            f"({a05_ch['no_extra_modes']:.0%} clean)"
        )
        next_levers.append(
            "A05: mute γ / raise λ_k under skewed contaminated residuals"
        )
    a01_ch = pass_rates["A01/champion_p12"]
    a01_cl = pass_rates["A01/classic_k2"]
    if a01_ch["modes_ge_1_5"] < a01_cl["modes_ge_1_5"]:
        weaknesses.append("Champion transfers poorly to A01 vs classic Kmax=2")
        next_levers.append("Keep classic A01 defaults; do not globalize p12")
    a02_ch = aggregates.get("A02/champion_mlp", {})
    if (a02_ch.get("k_modes_mean", {}).get("mean") or 0) < 1.5:
        weaknesses.append("A02 step: champion MLP fails to recover ~2 modes")
        next_levers.append(
            "A02-specific: sharper gate temperature / region-local experts (P6)"
        )
    if not next_levers:
        next_levers.append(
            "Adopt p12 as A04 default; stress unequal-π / closer means next"
        )

    out = {
        "seeds": seeds,
        "champion": _CHAMPION,
        "classic": _CLASSIC_LIGHT,
        "rows": problems,
        "aggregates": aggregates,
        "pass_rates": pass_rates,
        "weaknesses": weaknesses,
        "next_levers": next_levers,
    }
    _json_dump(out_root / "metrics.json", out)
    print(f"[limitation_stress] {len(problems)} fits → {out_root}")
    for key, agg in sorted(aggregates.items()):
        km = agg["k_modes_mean"]
        print(
            f"  {key}: K_modes={km['mean']:.2f}±{km['std']:.2f} "
            f"[{km['min']:.2f},{km['max']:.2f}] "
            f"nll={agg['test_nll']['mean']:.3f}"
        )
    print("weaknesses:")
    for w in weaknesses:
        print(f"  - {w}")
    print("next_levers:")
    for n in next_levers:
        print(f"  - {n}")
    return out


def run_recovery_probe(seeds: list[int] | None = None) -> dict:
    """Champion p12 (no-rec ablation) vs default p12+rec on A01/A04/A05."""
    from diagnostics import recovery_certificate

    gpplus.config.configure_logger()
    seeds = list(seeds or [7, 21, 42, 99, 123])
    out_root = OUT_DIR / "recovery_probe"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # champion_p12_rec matches production defaults; champion_p12 ablates λ_rec.
    arms = [
        {"tag": "champion_p12", "lambda_rec": 0.0, "rec_margin": 1.0},
        {"tag": "champion_p12_rec", "lambda_rec": 0.15, "rec_margin": 1.0},
    ]
    rows: list[dict] = []

    for seed in seeds:
        data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=seed)
        xa = torch.as_tensor(data["X_train"], dtype=torch.float64)
        ya = torch.as_tensor(data["y_train"], dtype=torch.float64)
        xate = torch.as_tensor(data["X_test"], dtype=torch.float64)
        yate = torch.as_tensor(data["y_test"], dtype=torch.float64)
        x4, y4, x4te, y4te = make_trimodal_tensors(
            n_samples=400, test_size=0.3, seed=seed
        )
        x5, y5, x5te, y5te = make_unimodal_hetero_tensors(
            n_samples=400, test_size=0.3, seed=seed
        )

        for arm in arms:
            for problem, bundle in (
                (
                    "A01",
                    {
                        "x_tr": xa,
                        "y_tr": ya,
                        "x_te": xate,
                        "y_te": yate,
                        "true_k": 2,
                        "k_max": 6,
                        "y_min": -2.0,
                        "y_max": 30.0,
                        "x_probe": [0.0, 1.0, 5.0, 7.0, 10.0],
                        "pass_km": 1.5,
                    },
                ),
                (
                    "A04",
                    {
                        "x_tr": x4,
                        "y_tr": y4,
                        "x_te": x4te,
                        "y_te": y4te,
                        "true_k": 3,
                        "k_max": 6,
                        "y_min": TRIMODAL_Y_MIN,
                        "y_max": TRIMODAL_Y_MAX,
                        "x_probe": [0.0, 1.0, 2.5, 5.0, 7.0, 10.0],
                        "pass_km": 2.5,
                    },
                ),
                (
                    "A05",
                    {
                        "x_tr": x5,
                        "y_tr": y5,
                        "x_te": x5te,
                        "y_te": y5te,
                        "true_k": 1,
                        "k_max": 6,
                        "y_min": A05_Y_MIN,
                        "y_max": A05_Y_MAX,
                        "x_probe": [0.0, 2.5, 5.0, 7.5, 10.0],
                        "pass_km_max": 1.5,
                    },
                ),
            ):
                cfg = {**_CHAMPION, **arm}
                row, model = fit_and_report(
                    bundle["x_tr"],
                    bundle["y_tr"],
                    bundle["x_te"],
                    bundle["y_te"],
                    name=f"{problem}_{arm['tag']}_s{seed}",
                    true_k_modes=int(bundle["true_k"]),
                    k_max=int(bundle["k_max"]),
                    gate_type="constant",
                    y_min=float(bundle["y_min"]),
                    y_max=float(bundle["y_max"]),
                    lambda_k=float(cfg["lambda_k"]),
                    lambda_sep=float(cfg["lambda_sep"]),
                    sep_margin=float(cfg["sep_margin"]),
                    lambda_overlap=float(cfg["lambda_overlap"]),
                    overlap_margin=float(cfg["overlap_margin"]),
                    free_expert_scale=bool(cfg["free_expert_scale"]),
                    lambda_beta=float(cfg["lambda_beta"]),
                    lambda_k_start=float(cfg["lambda_k_start"]),
                    lambda_orphan=float(cfg["lambda_orphan"]),
                    lambda_delta=float(cfg["lambda_delta"]),
                    lambda_gamma=float(cfg["lambda_gamma"]),
                    lambda_rec=float(cfg["lambda_rec"]),
                    rec_margin=float(cfg["rec_margin"]),
                    use_full_gp_variance=bool(cfg["use_full_gp_variance"]),
                    gamma_init_logit=float(cfg["gamma_init_logit"]),
                    num_steps=900,
                    seed=seed,
                    x_probe=list(bundle["x_probe"]),
                    return_model=True,
                )
                cert = recovery_certificate(
                    model,
                    list(bundle["x_probe"]),
                    y_min=float(bundle["y_min"]),
                    y_max=float(bundle["y_max"]),
                    rec_margin=float(cfg["rec_margin"]),
                )
                row.pop("fit_history_tail", None)
                row["problem"] = problem
                row["tag"] = arm["tag"]
                row["seed"] = seed
                if "pass_km_max" in bundle:
                    row["pass_k_modes"] = bool(
                        row["k_modes_mean"] <= float(bundle["pass_km_max"])
                    )
                else:
                    row["pass_k_modes"] = bool(
                        row["k_modes_mean"] >= float(bundle["pass_km"])
                    )
                row["certificate_pass"] = bool(cert["certificate_pass"])
                row["certificate_pass_rate"] = float(cert["certificate_pass_rate"])
                row["certificate"] = cert
                rows.append(row)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r['problem']}/{r['tag']}"
        groups.setdefault(key, []).append(r)

    def _rate(rs, pred):
        return float(sum(1 for r in rs if pred(r)) / max(len(rs), 1))

    summary = {}
    for key, rs in groups.items():
        km = np.asarray([r["k_modes_mean"] for r in rs], dtype=float)
        summary[key] = {
            "n": len(rs),
            "k_modes_mean": float(km.mean()),
            "k_modes_std": float(km.std(ddof=0)),
            "pass_k_modes_rate": _rate(rs, lambda r: r["pass_k_modes"]),
            "certificate_pass_rate": _rate(rs, lambda r: r["certificate_pass"]),
            "mean_cert_probe_rate": float(
                np.mean([r["certificate_pass_rate"] for r in rs])
            ),
            "test_nll_mean": float(np.mean([r["test_nll"] for r in rs])),
        }

    out = {
        "seeds": seeds,
        "champion": _CHAMPION,
        "rows": rows,
        "summary": summary,
    }
    _json_dump(out_root / "metrics.json", out)
    print(f"[recovery_probe] {len(rows)} fits → {out_root}")
    for k, s in sorted(summary.items()):
        print(
            f"  {k}: km={s['k_modes_mean']:.2f}±{s['k_modes_std']:.2f} "
            f"pass_km={s['pass_k_modes_rate']:.0%} "
            f"cert={s['certificate_pass_rate']:.0%} "
            f"nll={s['test_nll_mean']:.3f}"
        )
    return out


def run_examples(
    *,
    problems: list[str] | None = None,
    n_samples: int = 1000,
    test_size: float = 0.2,
    seed: int = 42,
    write_plots: bool = True,
    k_max: int = 5,
) -> dict:
    """Foundation toy examples: MidlineSparse soft MLP, 1 Stage-1 init.

    Per problem: ScoringBench metrics + true ``p(y|x)`` / local KDE / model
    marginals + posterior-vs-data with 50%/90% mixture CI bands + coverage
    bar charts + Stage-2 training-loss figure. Uses ``k_max=5`` (not the
    foundation-compare default of 10).
    """
    from compare_foundation_baselines import (
        ALL_PROBLEMS,
        COMPARE_COLS,
        load_problem,
        save_csv,
        save_metrics_table_png,
    )
    from foundation_adapters import fit_predict_midline_sparse
    from plot_foundation_marginals import (
        MODEL_COLORS,
        probe_x_values,
        save_marginals_per_probe,
        save_posterior_vs_data,
        slice_pred,
    )
    from cost_tables import make_cost_row, write_cost_artifacts
    from plot_midline_sparse import save_training_loss_subplots
    from scoringbench_metrics import compute_scoringbench_style_metrics, print_metrics_table
    from uq_calibration import interval_coverage, save_coverage_bar_chart

    gpplus.config.configure_logger()
    names = list(problems or ALL_PROBLEMS)
    out_root = OUT_DIR / "examples"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_key = "midline"
    model_labels = {model_key: "MidlineSparse soft MLP"}
    MODEL_COLORS.setdefault(model_key, "#2b6cb0")

    payloads: dict[str, dict] = {}
    loss_histories: dict[str, list[dict]] = {}
    loss_labels: dict[str, str] = {}
    coverage_by_problem: dict[str, dict[str, float]] = {}
    cost_rows: list[dict] = []
    scoring_rows: list[dict] = []

    for problem_name in names:
        problem = load_problem(
            problem_name, n_samples=n_samples, test_size=test_size, seed=seed
        )
        x_tr = problem["x_tr"]
        y_tr = problem["y_tr"]
        x_te = problem["x_te"]
        y_te = problem["y_te"]
        x_tr_np = x_tr.detach().cpu().numpy()
        y_tr_np = y_tr.detach().cpu().numpy().ravel()
        x_te_np = x_te.detach().cpu().numpy()
        y_te_np = y_te.detach().cpu().numpy().ravel()
        d = int(x_te_np.shape[1])
        x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=9)
        x_lo = float(min(x_tr_np.min(), x_te_np.min()))
        x_hi = float(max(x_tr_np.max(), x_te_np.max()))
        x_grid = np.linspace(x_lo, x_hi, 200)
        Xp = np.asarray(x_probes, dtype=np.float64).reshape(-1, d)
        Xg = np.asarray(x_grid, dtype=np.float64).reshape(-1, d)
        n_te, n_probe, n_grid = int(x_te_np.shape[0]), int(Xp.shape[0]), int(Xg.shape[0])
        x_eval = torch.as_tensor(
            np.vstack([x_te_np, Xp, Xg]), dtype=torch.float64
        )

        print(
            f"[{problem_name}] MidlineSparse MLP | 1 init | "
            f"Kmax={k_max} steps={problem['num_steps']} "
            f"champion={problem['champion']}",
            flush=True,
        )
        pred_all = fit_predict_midline_sparse(
            x_tr,
            y_tr,
            x_eval,
            k_max=int(k_max),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            gate_type="mlp",
            num_inits=1,
            num_stage2_inits=1,
            x_probe=x_probes.tolist(),
        )
        pred_te = slice_pred(pred_all, 0, n_te)
        pred_probe = slice_pred(pred_all, n_te, n_te + n_probe)
        pred_grid = slice_pred(pred_all, n_te + n_probe, n_te + n_probe + n_grid)
        extras = dict(pred_all.get("extras") or {})
        metrics = compute_scoringbench_style_metrics(
            y_te_np,
            pred_te["mixture_mean"],
            pred_te["expert_means"],
            pred_te["expert_stds"],
            pred_te["gate_weights"],
        )
        cov = interval_coverage(
            y_te_np,
            pred_te["expert_means"],
            pred_te["expert_stds"],
            pred_te["gate_weights"],
            levels=(0.5, 0.9),
        )
        coverage_by_problem[problem_name] = cov
        row = {
            "Model": model_labels[model_key],
            "model_key": model_key,
            "source": "examples",
            "gate_type": "mlp",
            "num_inits": 1,
            "num_stage2_inits": 1,
            **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
            "cov50": cov.get("cov50"),
            "cov90": cov.get("cov90"),
            "width50": cov.get("width50"),
            "width90": cov.get("width90"),
            "k_eff": extras.get("k_eff"),
            "k_hat": extras.get("k_hat"),
            "k_modes_mean": extras.get("k_modes_mean"),
            "k_modes_round": extras.get("k_modes_round"),
            "wall_s": extras.get("wall_s"),
        }
        stem = problem_name.lower()
        scoring_rows.append(row)
        # Per-problem ScoringBench table (RMSE/RRMSE/NIS/CRPS/log_score)
        table_csv = out_root / f"{stem}_metrics.csv"
        table_json = out_root / f"{stem}_metrics.json"
        table_png = fig_dir / f"{stem}_metrics_table.png"
        save_csv([row], table_csv)
        _json_dump(
            table_json,
            {
                "protocol": {
                    "problem": problem_name,
                    "k_max": int(k_max),
                    "gate_type": "mlp",
                    "num_inits": 1,
                    "n_train": int(x_tr.shape[0]),
                    "n_test": int(x_te.shape[0]),
                },
                "rows": [row],
                "metrics": metrics,
                "coverage": cov,
            },
        )
        save_metrics_table_png(
            [row],
            table_png,
            title=(
                f"{problem_name} MidlineSparse MLP — ScoringBench "
                f"(k_max={k_max}, 1 init)"
            ),
        )
        print_metrics_table([row], cols=["Model"] + COMPARE_COLS)
        print(f"Wrote {table_csv}")
        print(f"Wrote {table_json}")
        print(f"Wrote {table_png}")

        marginal_paths: list[str] = []
        posterior_path: str | None = None
        coverage_bar_path: str | None = None
        if write_plots:
            x_all = np.vstack([x_tr_np, x_te_np])
            y_all = np.concatenate([y_tr_np, y_te_np])
            m_paths = save_marginals_per_probe(
                problem_name=problem_name,
                x_probes=x_probes,
                model_preds={model_key: pred_probe},
                model_labels=model_labels,
                model_order=[model_key],
                x_data=x_all,
                y_data=y_all,
                y_min=float(problem["y_min"]),
                y_max=float(problem["y_max"]),
                out_dir=fig_dir,
                stem=stem,
            )
            for p in m_paths:
                print(f"Wrote {p}")
                marginal_paths.append(str(p))
            pp = fig_dir / f"{stem}_posterior_vs_data.png"
            save_posterior_vs_data(
                problem_name=problem_name,
                x_train=x_tr_np,
                y_train=y_tr_np,
                x_test=x_te_np,
                y_test=y_te_np,
                x_grid=x_grid,
                model_preds_grid={model_key: pred_grid},
                model_labels=model_labels,
                model_order=[model_key],
                out_path=pp,
                y_min=float(problem["y_min"]),
                y_max=float(problem["y_max"]),
            )
            print(f"Wrote {pp}")
            posterior_path = str(pp)
            cb = save_coverage_bar_chart(
                coverage_rows={model_key: cov},
                labels=model_labels,
                order=[model_key],
                out_path=fig_dir / f"{stem}_coverage_bars.png",
                title=f"{problem_name}: 50%/90% interval coverage (test)",
            )
            print(f"Wrote {cb}")
            coverage_bar_path = str(cb)

        loss_histories[problem_name] = list(extras.get("fit_history") or [])
        loss_labels[problem_name] = problem_name
        cost_rows.append(
            make_cost_row(
                script="run_midline_sparse.examples",
                problem=problem_name,
                model=model_labels[model_key],
                wall_s=extras.get("wall_s"),
                n_train=int(x_tr.shape[0]),
                k_max=int(k_max),
                num_steps=int(problem["num_steps"]),
                gate_type="mlp",
                num_inits=1,
                num_stage2_inits=1,
                champion=bool(problem["champion"]),
            )
        )
        payloads[problem_name] = {
            "row": row,
            "metrics": metrics,
            "coverage": cov,
            "extras": extras,
            "marginal_paths": marginal_paths,
            "posterior_path": posterior_path,
            "coverage_bar_path": coverage_bar_path,
            "table_csv": str(table_csv),
            "table_json": str(table_json),
            "table_png": str(table_png),
        }
        print(
            f"  RMSE={metrics.get('RMSE', float('nan')):.4f} "
            f"CRPS={metrics.get('CRPS', float('nan')):.4f} "
            f"log={metrics.get('log_score', float('nan')):.4f} "
            f"cov50={cov.get('cov50', float('nan')):.3f} "
            f"cov90={cov.get('cov90', float('nan')):.3f} "
            f"K_eff={extras.get('k_eff')} K_modes~{extras.get('k_modes_mean')} "
            f"wall={extras.get('wall_s', float('nan')):.1f}s",
            flush=True,
        )

    loss_fig: str | None = None
    coverage_summary_fig: str | None = None
    cost_artifacts = write_cost_artifacts(
        cost_rows,
        out_dir=out_root,
        stem="cost",
        title="examples: MidlineSparse MLP fit cost (wall-clock)",
    )
    # Cross-problem ScoringBench summary (one row per problem)
    summary_rows = []
    for pname in names:
        r = dict(payloads[pname]["row"])
        r["Model"] = pname
        r["model_key"] = pname
        summary_rows.append(r)
    all_csv = out_root / "metrics_all.csv"
    all_json = out_root / "metrics_all.json"
    all_png = fig_dir / "metrics_table_all.png"
    save_csv(summary_rows if summary_rows else scoring_rows, all_csv)
    _json_dump(all_json, {"rows": summary_rows, "compare_cols": list(COMPARE_COLS)})
    save_metrics_table_png(
        summary_rows if summary_rows else scoring_rows,
        all_png,
        title=f"examples MidlineSparse MLP — ScoringBench (k_max={k_max})",
    )
    print_metrics_table(
        summary_rows if summary_rows else scoring_rows,
        cols=["Model"] + COMPARE_COLS,
    )
    print(f"Wrote {all_csv}")
    print(f"Wrote {all_json}")
    print(f"Wrote {all_png}")

    if write_plots and loss_histories:
        p = save_training_loss_subplots(
            loss_histories,
            out_path=fig_dir / "training_loss_all.png",
            model_labels=loss_labels,
            model_order=list(loss_histories.keys()),
            title="Examples  |  MidlineSparse MLP (1 init) Stage-2 loss",
        )
        print(f"Wrote {p}")
        loss_fig = str(p)
    if write_plots and coverage_by_problem:
        cs = save_coverage_bar_chart(
            coverage_rows=coverage_by_problem,
            labels={k: k for k in coverage_by_problem},
            order=list(coverage_by_problem.keys()),
            out_path=fig_dir / "coverage_bars_all.png",
            title="Examples: 50%/90% interval coverage by problem (test)",
        )
        print(f"Wrote {cs}")
        coverage_summary_fig = str(cs)

    summary = {
        "protocol": {
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
            "gate_type": "mlp",
            "num_inits": 1,
            "num_stage2_inits": 1,
            "k_max": int(k_max),
            "problems": names,
            "interval_levels": [0.5, 0.9],
        },
        "payloads": payloads,
        "scoring_rows": summary_rows,
        "metrics_all": {
            "csv": str(all_csv),
            "json": str(all_json),
            "png": str(all_png),
        },
        "training_loss_figure": loss_fig,
        "coverage_summary_figure": coverage_summary_fig,
        "cost": cost_artifacts,
        "cost_rows": cost_rows,
    }
    _json_dump(out_root / "examples_summary.json", summary)
    print(f"Wrote {out_root / 'examples_summary.json'}")
    return summary


def run_merge_prune_probe(
    *,
    k_max_list: list[int] | None = None,
    n_samples: int = 1000,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Compare adaptive_merge_prune on/off across foundation problems × K_max.

    Protocol matches ``compare_foundation_baselines``: n=1000, test_size=0.2,
    and each problem's ``champion`` flag from ``load_problem`` (A01 classic).
    Fitter never sees true_k_modes; acceptance uses truth only for reporting.
    """
    from compare_foundation_baselines import ALL_PROBLEMS, load_problem
    from cost_tables import make_cost_row, write_cost_artifacts
    from foundation_adapters import fit_predict_midline_sparse
    from plot_midline_sparse import save_training_loss_subplots
    from scoringbench_metrics import compute_scoringbench_style_metrics

    gpplus.config.configure_logger()
    k_max_list = list(k_max_list or [1, 2, 4, 5, 6, 8, 10])
    out_root = OUT_DIR / "merge_prune_probe"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cost_rows: list[dict] = []
    training_loss_figures: dict[str, str] = {}
    protocol = {
        "n_samples": int(n_samples),
        "test_size": float(test_size),
        "seed": int(seed),
        "k_max_list": list(k_max_list),
        "aligned_with": "compare_foundation_baselines",
    }

    for problem_name in ALL_PROBLEMS:
        problem = load_problem(
            problem_name, n_samples=n_samples, test_size=test_size, seed=seed
        )
        x_tr = problem["x_tr"]
        y_tr = problem["y_tr"]
        x_te = problem["x_te"]
        y_te = problem["y_te"]
        true_k = int(problem["true_k_modes"])
        y_min = float(problem["y_min"])
        y_max = float(problem["y_max"])
        champion = bool(problem.get("champion", True))
        loss_histories: dict[str, list[dict]] = {}
        loss_order: list[str] = []
        for k_max in k_max_list:
            for adaptive in (False, True):
                tag = "adaptive_on" if adaptive else "adaptive_off"
                pred = fit_predict_midline_sparse(
                    x_tr,
                    y_tr,
                    x_te,
                    k_max=int(k_max),
                    num_steps=int(problem.get("num_steps", 800)),
                    seed=seed,
                    champion=champion,
                    y_min=y_min,
                    y_max=y_max,
                    adaptive_merge_prune=adaptive,
                )
                metrics = compute_scoringbench_style_metrics(
                    y_te.detach().cpu().numpy().ravel(),
                    pred["mixture_mean"],
                    pred["expert_means"],
                    pred["expert_stds"],
                    pred["gate_weights"],
                )
                ex = pred.get("extras") or {}
                mk = f"k{k_max}_{tag}"
                loss_histories[mk] = list(ex.get("fit_history") or [])
                loss_order.append(mk)
                row = {
                    "problem": problem_name,
                    "true_k_modes": true_k,
                    "champion": champion,
                    "k_max": int(k_max),
                    "tag": tag,
                    "adaptive_merge_prune": adaptive,
                    "RMSE": float(metrics.get("RMSE", float("nan"))),
                    "CRPS": float(metrics.get("CRPS", float("nan"))),
                    "log_score": float(metrics.get("log_score", float("nan"))),
                    "k_eff": ex.get("k_eff"),
                    "k_hat": ex.get("k_hat"),
                    "k_modes_mean": ex.get("k_modes_mean"),
                    "k_modes_round": ex.get("k_modes_round"),
                    "merge_pressure": ex.get("merge_pressure"),
                    "gate_type": ex.get("gate_type"),
                    "wall_s": ex.get("wall_s"),
                    "adaptive_diagnostics": ex.get("adaptive_diagnostics"),
                }
                rows.append(row)
                cost_rows.append(
                    make_cost_row(
                        script="run_midline_sparse.merge_prune_probe",
                        problem=problem_name,
                        model=f"MidlineSparse {tag}",
                        wall_s=ex.get("wall_s"),
                        n_train=int(x_tr.shape[0]),
                        k_max=int(k_max),
                        num_steps=int(problem.get("num_steps", 800)),
                        gate_type=str(ex.get("gate_type") or "mlp"),
                        num_inits=2,
                        num_stage2_inits=1,
                        champion=champion,
                        notes=tag,
                    )
                )
                print(
                    f"[merge_prune] {problem_name} champ={champion} "
                    f"Kmax={k_max} {tag}: "
                    f"RMSE={row['RMSE']:.3f} log={row['log_score']:.3f} "
                    f"km={row['k_modes_mean']} keff={row['k_eff']} "
                    f"khat={row['k_hat']} merge_p={row['merge_pressure']}"
                )
        stem = problem_name.lower()
        loss_path = save_training_loss_subplots(
            loss_histories,
            out_path=fig_dir / f"{stem}_training_loss.png",
            model_order=loss_order,
            model_labels={k: k.replace("_", " ") for k in loss_order},
            title=f"{problem_name}  |  Stage-2 training loss",
        )
        print(f"Wrote {loss_path}")
        training_loss_figures[problem_name] = str(loss_path)

    # Acceptance summaries (evaluation only; not used in fitting).
    accept: dict[str, dict] = {}
    for problem_name in ALL_PROBLEMS:
        on_rows = [
            r
            for r in rows
            if r["problem"] == problem_name and r["adaptive_merge_prune"]
        ]
        off_rows = [
            r
            for r in rows
            if r["problem"] == problem_name and not r["adaptive_merge_prune"]
        ]

        def _finite_log(rs: list[dict]) -> bool:
            return all(
                np.isfinite(r["log_score"]) and abs(r["log_score"]) < 50.0 for r in rs
            )

        def _khat_ge1(rs: list[dict], kmin: int = 2) -> bool:
            return all(
                int(r.get("k_hat") or 0) >= 1 for r in rs if int(r["k_max"]) >= kmin
            )

        true_k = int(on_rows[0]["true_k_modes"]) if on_rows else 0
        if problem_name == "A02":
            accept[problem_name] = {
                "no_catastrophe_on": _finite_log(on_rows) and _khat_ge1(on_rows),
                "rmse_le_0_15_on": all(
                    float(r["RMSE"]) <= 0.15
                    for r in on_rows
                    if int(r["k_max"]) >= 2
                ),
                "no_catastrophe_off": _finite_log(off_rows),
            }
        elif problem_name in ("A05", "unimodal"):
            large = [r for r in on_rows if int(r["k_max"]) >= 6]
            accept[problem_name] = {
                "modes_le_1_25_on": all(
                    (r.get("k_modes_mean") or 99) <= 1.25 for r in large
                ),
                "keff_le_1_5_on": all(
                    (r.get("k_eff") or 99) <= 1.5 for r in large
                ),
            }
        elif problem_name == "A01":
            on_km = np.mean([r.get("k_modes_mean") or 0 for r in on_rows if r["k_max"] >= 2])
            off_km = np.mean([r.get("k_modes_mean") or 0 for r in off_rows if r["k_max"] >= 2])
            accept[problem_name] = {
                "modes_ge_1_5_on": bool(on_km >= 1.5),
                "no_regression_vs_off": bool(on_km + 0.25 >= off_km),
                "true_k_modes": true_k,
            }
        elif problem_name == "A04":
            on_km = np.mean([r.get("k_modes_mean") or 0 for r in on_rows if r["k_max"] >= 4])
            off_km = np.mean([r.get("k_modes_mean") or 0 for r in off_rows if r["k_max"] >= 4])
            accept[problem_name] = {
                "modes_ge_2_on": bool(on_km >= 2.0),
                "no_regression_vs_off": bool(on_km + 0.25 >= off_km),
                "true_k_modes": true_k,
            }

    cost_artifacts = write_cost_artifacts(
        cost_rows,
        out_dir=out_root,
        stem="cost",
        title="merge_prune_probe: MidlineSparse fit cost",
    )
    out = {
        "protocol": protocol,
        "seed": seed,
        "k_max_list": k_max_list,
        "rows": rows,
        "accept": accept,
        "training_loss_figures": training_loss_figures,
        "cost": cost_artifacts,
    }
    _json_dump(out_root / "metrics.json", out)
    print(
        f"[merge_prune_probe] protocol n={n_samples} test={test_size} "
        f"(foundation-aligned); {len(rows)} fits -> {out_root}"
    )
    print(json.dumps(accept, indent=2).encode("ascii", "replace").decode("ascii"))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "examples",
            "gate2",
            "v2",
            "charts",
            "a04_ablation",
            "a04_p1",
            "a04_beta_ablation",
            "orphan_delta_reg",
            "noise_alleviation",
            "limitation_stress",
            "limitation_charts",
            "recovery_probe",
            "merge_prune_probe",
            "merge_prune_charts",
            "all",
        ),
        default="examples",
        help=(
            "examples: A01/A02/A04/A05/unimodal with soft MLP + 1 init; "
            "true p(y|x)/KDE/model marginals + posterior + training loss. "
            "smoke: quick constant-gate fits. all: smoke+gate2(+v2)+charts."
        ),
    )
    args = parser.parse_args()
    if args.mode == "charts":
        from plot_midline_sparse import generate_all_charts

        generate_all_charts(out_dir=OUT_DIR / "figures")
    elif args.mode == "examples":
        run_examples()
    elif args.mode == "limitation_charts":
        from plot_midline_sparse import generate_limitation_stress_charts

        generate_limitation_stress_charts(
            metrics_path=OUT_DIR / "limitation_stress" / "metrics.json",
            out_dir=OUT_DIR / "limitation_stress" / "figures",
        )
    elif args.mode == "a04_ablation":
        run_a04_ablation()
    elif args.mode == "a04_p1":
        run_a04_p1_verify()
    elif args.mode == "a04_beta_ablation":
        run_a04_beta_prior_ablation()
    elif args.mode == "orphan_delta_reg":
        run_orphan_delta_reg_test()
    elif args.mode == "noise_alleviation":
        run_noise_alleviation_probes()
    elif args.mode == "limitation_stress":
        run_limitation_stress()
        from plot_midline_sparse import generate_limitation_stress_charts

        generate_limitation_stress_charts(
            metrics_path=OUT_DIR / "limitation_stress" / "metrics.json",
            out_dir=OUT_DIR / "limitation_stress" / "figures",
        )
    elif args.mode == "recovery_probe":
        run_recovery_probe()
        from plot_midline_sparse import generate_recovery_probe_charts

        generate_recovery_probe_charts(
            metrics_path=OUT_DIR / "recovery_probe" / "metrics.json",
            out_dir=OUT_DIR / "recovery_probe" / "figures",
        )
    elif args.mode == "merge_prune_probe":
        run_merge_prune_probe()
        from plot_midline_sparse import generate_merge_prune_probe_charts

        generate_merge_prune_probe_charts(
            metrics_path=OUT_DIR / "merge_prune_probe" / "metrics.json",
            out_dir=OUT_DIR / "merge_prune_probe" / "figures",
        )
    elif args.mode == "merge_prune_charts":
        from plot_midline_sparse import generate_merge_prune_probe_charts

        generate_merge_prune_probe_charts(
            metrics_path=OUT_DIR / "merge_prune_probe" / "metrics.json",
            out_dir=OUT_DIR / "merge_prune_probe" / "figures",
        )
    elif args.mode in ("smoke", "all"):
        run_smoke()
    gate2 = None
    if args.mode in ("gate2", "all"):
        gate2 = run_gate2()
    if args.mode == "v2":
        run_gate2_with_mode_loss()
    elif args.mode == "all" and gate2 is not None:
        if gate2["pass"]["overall_gate2_pass"]:
            print("Gate 2 PASSED — running v2 mode-loss.")
            run_gate2_with_mode_loss()
        else:
            print("Gate 2 FAILED — skipping v2 mode-loss per plan.")
            _json_dump(
                OUT_DIR / "midline_sparse_v2_mode_loss.json",
                {"skipped": True, "reason": "gate2_failed", "pass": gate2["pass"]},
            )
        print("Generating charts...")
        from plot_midline_sparse import generate_all_charts

        generate_all_charts(out_dir=OUT_DIR / "figures")
