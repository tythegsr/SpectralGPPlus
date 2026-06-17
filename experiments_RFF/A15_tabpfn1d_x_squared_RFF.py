"""1D f(x)=x^2 benchmark with GPPlus RFF kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from load_experimental_data import generate_tabpfn_1d_x_squared_data, tabpfn_1d_x_squared_function
from tabpfn1d_rff_base import run_tabpfn1d_rff


def run_tabpfn1d_x_squared_rff(
    train_size: int = 10,
    num_rff: int | None = None,
    num_test: int = 5000,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    test_outside_margin: float = 0.0,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 16,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_RFF/results/tabpfn1d_x_squared_rff",
    ard: bool = True,
    n_jobs: int | None = None,
    plot_1d: bool = True,
    **kwargs,
) -> dict:
    if x_bounds is None:
        x_bounds = [-0.5, 0.5]
    return run_tabpfn1d_rff(
        problem_name="x_squared",
        generate_data_fn=generate_tabpfn_1d_x_squared_data,
        true_fn=tabpfn_1d_x_squared_function,
        train_size=train_size,
        num_rff=num_rff,
        num_test=num_test,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        test_outside_margin=test_outside_margin,
        noise_train=noise_train,
        noise_test=noise_test,
        noise_type=noise_type,
        seed=seed,
        num_inits=num_inits,
        num_epochs=num_epochs,
        device=device,
        dtype=dtype,
        save_path=save_path,
        ard=ard,
        n_jobs=n_jobs,
        plot_1d=plot_1d,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="1D x^2 with GPPlus RFF")
    parser.add_argument("--train-size", type=int, default=10)
    parser.add_argument("--num-rff", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--test-outside-margin", type=float, default=0.0)
    parser.add_argument("--noise-train", type=float, default=0.005)
    parser.add_argument("--noise-test", type=float, default=0.005)
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="experiments_RFF/results/tabpfn1d_x_squared_rff")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_tabpfn1d_x_squared_rff(
        train_size=args.train_size,
        num_rff=args.num_rff,
        num_test=args.num_test,
        test_outside_margin=args.test_outside_margin,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        num_inits=args.num_inits,
        seed=args.seed,
        save_path=args.save_path,
    )
