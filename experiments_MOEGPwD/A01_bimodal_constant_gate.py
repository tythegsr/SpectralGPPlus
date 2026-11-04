"""ScoringBench 1D bimodal toy with constant-gate MoE-GP (co-located modes)."""

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
from moe_wd_base import run_bimodal_constant_gate


def run_a01(
    n_samples: int = 1000,
    test_size: float = 0.3,
    num_experts: int = 2,
    temperature: float = 1.0,
    assignment_init: str = "residual",
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 200,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str | None = "experiments_MOEGPwD/results/bimodal_constant_gate",
    plot: bool = True,
    cond_x: float = 9.0,
    **kwargs,
) -> dict:
    return run_bimodal_constant_gate(
        n_samples=n_samples,
        test_size=test_size,
        num_experts=num_experts,
        temperature=temperature,
        assignment_init=assignment_init,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        save_path=save_path or "experiments_MOEGPwD/results/bimodal_constant_gate",
        plot=plot,
        cond_x=cond_x,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE-GP constant gate on ScoringBench bimodal toy")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--assignment-init",
        type=str,
        default="residual",
        choices=["residual", "x", "quadrant", "random"],
        help="How to choose the first hard expert labels before EM",
    )
    parser.add_argument("--num-em-iters", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=8)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_MOEGPwD/results/bimodal_constant_gate",
    )
    parser.add_argument("--cond-x", type=float, default=9.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_a01(
        n_samples=args.n_samples,
        test_size=args.test_size,
        num_experts=args.num_experts,
        temperature=args.temperature,
        assignment_init=args.assignment_init,
        num_em_iters=args.num_em_iters,
        num_inits=args.num_inits,
        gate_epochs=args.gate_epochs,
        seed=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
        plot=not args.no_plot,
        cond_x=args.cond_x,
    )
