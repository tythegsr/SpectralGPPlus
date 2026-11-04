"""ScoringBench 1D bimodal toy with Mixture-of-Experts exact GPR."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_EXP_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _EXP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from moe_gp_base import run_bimodal_moe_gp


def run_a01_bimodal_moe_gp(
    n_samples: int = 1000,
    test_size: float = 0.3,
    num_experts: int = 2,
    gating_hidden: tuple[int, ...] = (64, 64),
    assignment_init: str = "residual",
    fit_mode: str = "hard",
    gate_type: str | None = None,
    load_balance_weight: float = 0.1,
    entropy_weight: float = 0.01,
    diversity_weight: float = 0.1,
    orthogonality_weight: float = 3.0,
    orth_refit_threshold: float = 0.25,
    responsibility_temperature: float = 0.5,
    soft_init_peak: float = 0.95,
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 200,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str | None = "experiments_MOEGP/results/bimodal_moe_gp",
    plot: bool = True,
    cond_x: float = 9.0,
    **kwargs,
) -> dict:
    return run_bimodal_moe_gp(
        n_samples=n_samples,
        test_size=test_size,
        num_experts=num_experts,
        gating_hidden=gating_hidden,
        assignment_init=assignment_init,
        fit_mode=fit_mode,
        gate_type=gate_type,
        load_balance_weight=load_balance_weight,
        entropy_weight=entropy_weight,
        diversity_weight=diversity_weight,
        orthogonality_weight=orthogonality_weight,
        orth_refit_threshold=orth_refit_threshold,
        responsibility_temperature=responsibility_temperature,
        soft_init_peak=soft_init_peak,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        save_path=save_path or "experiments_MOEGP/results/bimodal_moe_gp",
        plot=plot,
        cond_x=cond_x,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE-GP on ScoringBench 1D bimodal toy")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-experts", type=int, default=2, help="Controllable number of GP experts")
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="soft",
        choices=["hard", "soft"],
        help="hard: subset EM; soft: all-data weighted experts + mixture-NLL gate",
    )
    parser.add_argument(
        "--gate-type",
        type=str,
        default="mlp",
        choices=["constant", "mlp"],
        help="Default: mlp for hard, constant for soft",
    )
    parser.add_argument(
        "--assignment-init",
        type=str,
        default="residual",
        choices=["residual", "x", "quadrant", "random"],
        help="First hard labels (hard mode only; soft starts from uniform r)",
    )
    parser.add_argument("--load-balance-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.5,
        help="Soft mode: encourage expert mean curves to differ",
    )
    parser.add_argument(
        "--orthogonality-weight",
        type=float,
        default=10.0,
        help="Soft mode: blend strength exponent; larger = harder pull to specialized seed",
    )
    parser.add_argument(
        "--orth-refit-threshold",
        type=float,
        default=0.25,
        help="Soft mode: if orth blend >= this, refit experts in the same EM iter",
    )
    parser.add_argument(
        "--responsibility-temperature",
        type=float,
        default=0.5,
        help="Soft mode: <1 sharpens r_ik so experts specialize",
    )
    parser.add_argument(
        "--soft-init-peak",
        type=float,
        default=0.95,
        help="Soft mode: initial weight on seeded expert (rest shared)",
    )
    parser.add_argument("--num-em-iters", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_MOEGP/results/bimodal_moe_gp",
    )
    parser.add_argument("--cond-x", type=float, default=9.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_a01_bimodal_moe_gp(
        n_samples=args.n_samples,
        test_size=args.test_size,
        num_experts=args.num_experts,
        assignment_init=args.assignment_init,
        fit_mode=args.fit_mode,
        gate_type=args.gate_type,
        load_balance_weight=args.load_balance_weight,
        entropy_weight=args.entropy_weight,
        diversity_weight=args.diversity_weight,
        orthogonality_weight=args.orthogonality_weight,
        orth_refit_threshold=args.orth_refit_threshold,
        responsibility_temperature=args.responsibility_temperature,
        soft_init_peak=args.soft_init_peak,
        num_em_iters=args.num_em_iters,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        gate_epochs=args.gate_epochs,
        gate_lr=args.gate_lr,
        seed=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
        plot=not args.no_plot,
        cond_x=args.cond_x,
    )
