"""Cheap empirical probes for mode-count + soft MoE-GP feasibility.

Runs:
  A01 co-located (constant gate, K_max in {2,3,4}; MLP gate contrast)
  Unimodal control (K_max=4)
  A02 separated step (MLP vs constant)
  Residual-GMM / BIC baselines

Outputs JSON under results/ and prints a summary table.
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
from gpplus.models import MOEGPR

from diagnostics import average_posthoc_k_modes, expert_activity
from estimate_mode_count import (
    estimate_mode_count,
    linear_midline_residuals,
)
from load_bimodal_data import train_test_split_bimodal


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


def _fit_soft_moe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    num_experts: int,
    gate_type: str,
    assignment_init: str = "residual",
    num_em_iters: int = 3,
    num_inits: int = 2,
    num_epochs: int = 1,
    gate_epochs: int = 150,
    orthogonality_weight: float = 3.0,
    seed: int = 42,
    device: str = "cpu",
) -> MOEGPR:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MOEGPR(
        x_train,
        y_train,
        num_experts=num_experts,
        gating_hidden=(32, 32),
        assignment_init=assignment_init,  # type: ignore[arg-type]
        gate_type=gate_type,  # type: ignore[arg-type]
        load_balance_weight=0.1,
        entropy_weight=0.01,
        diversity_weight=0.1,
        orthogonality_weight=orthogonality_weight,
        orth_refit_threshold=0.25,
        responsibility_temperature=0.5,
        soft_init_peak=0.95,
    )
    model = model.to(device=device, dtype=torch.float64)
    model.fit(
        fit_mode="soft",
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=1e-2,
        device=device,
        n_jobs=1,
    )
    return model


def _soft_nll(model: MOEGPR, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model.predict(x)
    means = pred["expert_means"].detach().cpu().numpy()
    stds = pred["expert_stds"].detach().cpu().numpy()
    w = pred["gate_weights"].detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().ravel()
    from scipy.stats import norm

    n, k = means.shape
    log_comp = np.zeros((n, k))
    for j in range(k):
        log_comp[:, j] = np.log(np.clip(w[:, j], 1e-12, None)) + norm.logpdf(
            y_np, means[:, j], np.clip(stds[:, j], 1e-6, None)
        )
    return float(-np.mean(np.logaddexp.reduce(log_comp, axis=1)))


def probe_a01(
    *,
    n_samples: int = 400,
    k_list: list[int] | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict]:
    if k_list is None:
        k_list = [2, 3, 4]
    data = train_test_split_bimodal(n_samples=n_samples, test_size=0.3, seed=seed)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)

    # Residual-GMM baseline (linear midline)
    residual_base = estimate_mode_count(
        linear_midline_residuals(x_tr, y_tr), k_max=6, criterion="bic"
    )

    x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
    rows: list[dict] = []

    for gate_type in ("constant", "mlp"):
        for k in k_list:
            t0 = time.perf_counter()
            model = _fit_soft_moe(
                x_tr,
                y_tr,
                num_experts=k,
                gate_type=gate_type,
                seed=seed,
                device=device,
                orthogonality_weight=3.0 if gate_type == "constant" else 1.0,
            )
            wall = time.perf_counter() - t0
            act = expert_activity(model, mass_eps=0.05)
            modes = average_posthoc_k_modes(
                model, x_probe, y_min=-2.0, y_max=30.0, n_grid=400
            )
            nll = _soft_nll(model, x_te, y_te)
            row = {
                "dataset": "A01_colocated",
                "true_k_modes": 2,
                "gate_type": gate_type,
                "k_max": k,
                "k_modes_mean": modes["k_modes_mean"],
                "k_modes_median": modes["k_modes_median"],
                "k_modes_round": modes["k_modes_round"],
                "k_experts_active": act["k_experts_active"],
                "mean_r": act["mean_r"],
                "mean_abs_expert_corr": act["mean_abs_expert_corr"],
                "k_eff_renorm_sum": act["k_eff_renorm_sum"],
                "test_mixture_nll": nll,
                "wall_s": wall,
                "residual_gmm_k_hat": residual_base.k_hat,
                "per_x_k_modes": [r["k_modes"] for r in modes["per_x"]],
            }
            print(
                f"[A01] gate={gate_type} K={k} | "
                f"K_modes~{modes['k_modes_mean']:.2f} "
                f"K_active={act['k_experts_active']} "
                f"corr={act['mean_abs_expert_corr']:.3f} "
                f"nll={nll:.3f} t={wall:.1f}s"
            )
            rows.append(row)
    return rows


def make_unimodal_data(
    n: int = 300,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def probe_unimodal(*, k_max: int = 4, seed: int = 7, device: str = "cpu") -> dict:
    x_tr, y_tr, x_te, y_te = make_unimodal_data(n=300, seed=seed)
    residual_base = estimate_mode_count(
        linear_midline_residuals(x_tr, y_tr), k_max=6, criterion="bic"
    )
    t0 = time.perf_counter()
    model = _fit_soft_moe(
        x_tr,
        y_tr,
        num_experts=k_max,
        gate_type="constant",
        seed=seed,
        device=device,
    )
    wall = time.perf_counter() - t0
    x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
    act = expert_activity(model, mass_eps=0.05)
    modes = average_posthoc_k_modes(model, x_probe, y_min=-2.0, y_max=12.0, n_grid=400)
    nll = _soft_nll(model, x_te, y_te)
    row = {
        "dataset": "unimodal_control",
        "true_k_modes": 1,
        "gate_type": "constant",
        "k_max": k_max,
        "k_modes_mean": modes["k_modes_mean"],
        "k_modes_median": modes["k_modes_median"],
        "k_modes_round": modes["k_modes_round"],
        "k_experts_active": act["k_experts_active"],
        "mean_r": act["mean_r"],
        "mean_abs_expert_corr": act["mean_abs_expert_corr"],
        "k_eff_renorm_sum": act["k_eff_renorm_sum"],
        "test_mixture_nll": nll,
        "wall_s": wall,
        "residual_gmm_k_hat": residual_base.k_hat,
        "per_x_k_modes": [r["k_modes"] for r in modes["per_x"]],
    }
    print(
        f"[UNI] K={k_max} | K_modes~{modes['k_modes_mean']:.2f} "
        f"K_active={act['k_experts_active']} residual_k={residual_base.k_hat} "
        f"nll={nll:.3f} t={wall:.1f}s"
    )
    return row


def make_step_data(
    n_train: int = 200,
    n_test: int = 200,
    noise: float = 0.05,
    seed: int = 42,
) -> dict:
    """Separated regimes: y ≈ 0 for x<0.5, y ≈ 1 for x>=0.5 (A02-style)."""
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(0.0, 1.0, size=(n_train, 1))
    y_tr = (x_tr[:, 0] >= 0.5).astype(np.float64) + noise * rng.normal(size=n_train)
    x_te = rng.uniform(0.0, 1.0, size=(n_test, 1))
    y_te = (x_te[:, 0] >= 0.5).astype(np.float64) + noise * rng.normal(size=n_test)
    return {
        "X_train": x_tr,
        "y_train": y_tr,
        "X_test": x_te,
        "y_test": y_te,
    }


def probe_a02(*, k_max: int = 4, seed: int = 42, device: str = "cpu") -> list[dict]:
    data = make_step_data(n_train=200, n_test=200, seed=seed)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)
    residual_base = estimate_mode_count(
        linear_midline_residuals(x_tr, y_tr), k_max=6, criterion="bic"
    )
    x_probe = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
    rows = []
    for gate_type in ("mlp", "constant"):
        t0 = time.perf_counter()
        model = _fit_soft_moe(
            x_tr,
            y_tr,
            num_experts=k_max,
            gate_type=gate_type,
            assignment_init="x" if gate_type == "mlp" else "residual",
            seed=seed,
            device=device,
            orthogonality_weight=1.0,
        )
        wall = time.perf_counter() - t0
        act = expert_activity(model, mass_eps=0.05)
        modes = average_posthoc_k_modes(
            model, x_probe, y_min=-0.5, y_max=1.5, n_grid=300
        )
        nll = _soft_nll(model, x_te, y_te)
        row = {
            "dataset": "A02_separated_step",
            "true_k_modes_per_x": 1,  # locally unimodal; regime switch in x
            "true_regimes": 2,
            "gate_type": gate_type,
            "k_max": k_max,
            "k_modes_mean": modes["k_modes_mean"],
            "k_modes_median": modes["k_modes_median"],
            "k_modes_round": modes["k_modes_round"],
            "k_experts_active": act["k_experts_active"],
            "mean_r": act["mean_r"],
            "mean_abs_expert_corr": act["mean_abs_expert_corr"],
            "k_eff_renorm_sum": act["k_eff_renorm_sum"],
            "test_mixture_nll": nll,
            "wall_s": wall,
            "residual_gmm_k_hat": residual_base.k_hat,
            "per_x_k_modes": [r["k_modes"] for r in modes["per_x"]],
        }
        print(
            f"[A02] gate={gate_type} K={k_max} | "
            f"K_modes~{modes['k_modes_mean']:.2f} "
            f"K_active={act['k_experts_active']} nll={nll:.3f} t={wall:.1f}s"
        )
        rows.append(row)
    return rows


def bic_over_fixed_k(a01_rows: list[dict]) -> list[dict]:
    """Treat test NLL * n + K log n as crude IC over constant-gate A01 runs."""
    const = [r for r in a01_rows if r["gate_type"] == "constant"]
    if not const:
        return []
    # n not stored; approximate from wall runs — use n_train≈280 for n_samples=400
    n = 280
    out = []
    for r in const:
        nll = r["test_mixture_nll"]
        k = r["k_max"]
        # proxy: use NLL as approx mean NLL; BIC ~ 2*n*NLL + k log n (params≈k)
        bic = 2.0 * n * nll + k * np.log(n)
        aic = 2.0 * n * nll + 2.0 * k
        out.append(
            {
                "k_max": k,
                "test_nll": nll,
                "aic_proxy": aic,
                "bic_proxy": bic,
                "k_modes_round": r["k_modes_round"],
            }
        )
    best_bic = min(out, key=lambda d: d["bic_proxy"])
    for d in out:
        d["selected_by_bic"] = d["k_max"] == best_bic["k_max"]
    return out


def main():
    gpplus.config.configure_logger()
    device = "cpu"
    print("=== Phase 3 feasibility probes ===")
    a01 = probe_a01(n_samples=400, k_list=[2, 3, 4], seed=42, device=device)
    uni = probe_unimodal(k_max=4, seed=7, device=device)
    a02 = probe_a02(k_max=4, seed=42, device=device)
    bic = bic_over_fixed_k(a01)

    summary = {
        "A01": a01,
        "unimodal": uni,
        "A02": a02,
        "bic_proxy_A01_constant": bic,
        "pass_criteria_notes": _evaluate_pass(a01, uni, a02),
    }
    out_path = OUT_DIR / "probe_summary.json"
    _json_dump(out_path, summary)
    print(f"\nWrote {out_path}")
    print(json.dumps(summary["pass_criteria_notes"], indent=2))
    return summary


def _evaluate_pass(a01, uni, a02) -> dict:
    """Check plan Phase-3 pass criteria."""
    # Post-hoc K_modes tracks truth better than active experts on overfitted K
    overfit = [
        r
        for r in a01
        if r["gate_type"] == "constant" and r["k_max"] >= 3
    ]
    modes_ok = False
    details = []
    for r in overfit:
        km = r["k_modes_round"]
        ka = r["k_experts_active"]
        better = abs(km - 2) < abs(ka - 2) or (km == 2 and ka != 2)
        details.append(
            {
                "k_max": r["k_max"],
                "k_modes_round": km,
                "k_experts_active": ka,
                "modes_closer_to_2_than_active": better,
            }
        )
        if better:
            modes_ok = True

    # x-gate fails on A01 vs helps on A02
    a01_const = min(
        (r["test_mixture_nll"] for r in a01 if r["gate_type"] == "constant" and r["k_max"] == 2),
        default=None,
    )
    a01_mlp = min(
        (r["test_mixture_nll"] for r in a01 if r["gate_type"] == "mlp" and r["k_max"] == 2),
        default=None,
    )
    a02_mlp = next((r["test_mixture_nll"] for r in a02 if r["gate_type"] == "mlp"), None)
    a02_const = next((r["test_mixture_nll"] for r in a02 if r["gate_type"] == "constant"), None)
    gate_contrast = {
        "a01_mlp_worse_or_similar_nll": (
            a01_mlp is not None and a01_const is not None and a01_mlp >= a01_const - 0.05
        ),
        "a02_mlp_better_nll": (
            a02_mlp is not None and a02_const is not None and a02_mlp < a02_const
        ),
        "a01_const_nll": a01_const,
        "a01_mlp_nll": a01_mlp,
        "a02_mlp_nll": a02_mlp,
        "a02_const_nll": a02_const,
    }

    uni_ok = uni["k_modes_round"] <= 1 or uni["k_modes_mean"] <= 1.5

    return {
        "posthoc_k_modes_beats_active_experts_on_overfit": modes_ok,
        "overfit_details": details,
        "x_gate_fails_A01_helps_A02": gate_contrast,
        "unimodal_k_modes_near_1": uni_ok,
        "unimodal_row": {
            "k_modes_mean": uni["k_modes_mean"],
            "k_experts_active": uni["k_experts_active"],
            "residual_gmm_k_hat": uni["residual_gmm_k_hat"],
        },
        "overall_phase3_pass": bool(
            modes_ok
            and gate_contrast["a01_mlp_worse_or_similar_nll"]
            and gate_contrast["a02_mlp_better_nll"]
            and uni_ok
        ),
    }


if __name__ == "__main__":
    main()
