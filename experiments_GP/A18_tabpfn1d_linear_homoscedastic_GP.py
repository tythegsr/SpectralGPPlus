"""1D linear f(x)=slope*x+intercept benchmark with GPPlus exact GP."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_GP_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _GP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from load_experimental_data import (
    generate_tabpfn_1d_linear_homoscedastic_data,
    tabpfn_1d_linear_function,
)
from tabpfn1d_gp_base import run_tabpfn1d_gp


def run_tabpfn1d_linear_homoscedastic_gp(
    train_size: int = 40,
    num_test: int = 5000,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    test_outside_margin: float = 0.0,
    slope: float = 1.0,
    intercept: float = 0.0,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 16,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_GP/results/tabpfn1d_linear_homoscedastic_gp",
    ard: bool = True,
    n_jobs: int | None = None,
    plot_1d: bool = True,
    **kwargs,
) -> dict:
    if x_bounds is None:
        x_bounds = [-1.0, 1.0]

    def true_fn(X: torch.Tensor) -> torch.Tensor:
        return tabpfn_1d_linear_function(X, slope=slope, intercept=intercept)

    return run_tabpfn1d_gp(
        problem_name="linear_homoscedastic",
        generate_data_fn=generate_tabpfn_1d_linear_homoscedastic_data,
        true_fn=true_fn,
        train_size=train_size,
        num_test=num_test,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        test_outside_margin=test_outside_margin,
        generate_data_kwargs={"slope": slope, "intercept": intercept},
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

    parser = argparse.ArgumentParser(description="1D linear with GPPlus exact GP")
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--test-outside-margin", type=float, default=0.0)
    parser.add_argument("--noise-train", type=float, default=0.005)
    parser.add_argument("--noise-test", type=float, default=0.005)
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="experiments_GP/results/tabpfn1d_linear_homoscedastic_gp")
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_tabpfn1d_linear_homoscedastic_gp(
        train_size=args.train_size,
        num_test=args.num_test,
        test_outside_margin=args.test_outside_margin,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        num_inits=args.num_inits,
        seed=args.seed,
        save_path=args.save_path,
    )
