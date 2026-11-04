"""1D step toy with sharp MLP-gate MoE-GP (low softmax temperature)."""

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
from moe_wd_base import run_step_sharp_gate


def run_a02(
    n_train: int = 200,
    n_test: int = 500,
    num_experts: int = 4,
    temperature: float = 0.1,
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 300,
    gate_lr: float = 1e-2,
    noise_train: float = 0.02,
    noise_test: float = 0.02,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str | None = "experiments_MOEGPwD/results/step_sharp_gate",
    plot: bool = True,
    **kwargs,
) -> dict:
    return run_step_sharp_gate(
        n_train=n_train,
        n_test=n_test,
        num_experts=num_experts,
        temperature=temperature,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        noise_train=noise_train,
        noise_test=noise_test,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        save_path=save_path or "experiments_MOEGPwD/results/step_sharp_gate",
        plot=plot,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE-GP sharp MLP gate on 1D step toy")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-em-iters", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=8)
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--noise-train", type=float, default=0.02)
    parser.add_argument("--noise-test", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_MOEGPwD/results/step_sharp_gate",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_a02(
        n_train=args.n_train,
        n_test=args.n_test,
        num_experts=args.num_experts,
        temperature=args.temperature,
        num_em_iters=args.num_em_iters,
        num_inits=args.num_inits,
        gate_epochs=args.gate_epochs,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        seed=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
        plot=not args.no_plot,
    )
