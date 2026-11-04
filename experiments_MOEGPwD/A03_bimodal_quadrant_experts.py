"""ScoringBench bimodal X as four line-segment experts (TL/BL/TR/BR) + MLP gate."""

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
from moe_wd_base import run_bimodal_quadrant_experts


def run_a03(
    n_samples: int = 1000,
    test_size: float = 0.3,
    temperature: float = 0.5,
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 300,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str | None = "experiments_MOEGPwD/results/bimodal_quadrant_experts",
    plot: bool = True,
    cond_x: float = 9.0,
    **kwargs,
) -> dict:
    """Four experts for the four rays of the crossing-mode X.

    The MLP gate should activate the left pair (TL+BL) on the left of the
    crossing and the right pair (TR+BR) on the right, while each pair still
    mixes the two output modes.
    """
    return run_bimodal_quadrant_experts(
        n_samples=n_samples,
        test_size=test_size,
        temperature=temperature,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        save_path=save_path or "experiments_MOEGPwD/results/bimodal_quadrant_experts",
        plot=plot,
        cond_x=cond_x,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MoE-GP with 4 quadrant experts on ScoringBench bimodal X"
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num-em-iters", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=8)
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_MOEGPwD/results/bimodal_quadrant_experts",
    )
    parser.add_argument("--cond-x", type=float, default=9.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_a03(
        n_samples=args.n_samples,
        test_size=args.test_size,
        temperature=args.temperature,
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
