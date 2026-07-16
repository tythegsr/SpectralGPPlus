"""Shared 20-D / 5-task synthetic multitask problem.

One Sobol design ``u ∈ [0, 1]^{20}`` is mapped per-task into each benchmark's native
domain, then evaluated. The model sees the shared unit-cube inputs ``X = u`` and
targets ``Y`` with shape ``(n, 5)``.

Tasks (column order in ``Y``):
  0 ackley       — ``u`` → ``[-5, 10]^{20}``
  1 rosenbrock   — ``u`` → ``[-5, 10]^{20}``
  2 griewank     — ``u`` → ``[-600, 600]^{20}``
  3 borehole     — first 8 coords of ``u`` → physical Borehole bounds; dims 9–20 unused
  4 dixon_price  — ``u`` → ``[-10, 10]^{20}``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parents[1]
_SORF = _ROOT / "experiments_SORF"
if str(_SORF) not in sys.path:
    sys.path.insert(0, str(_SORF))

from load_experimental_data import (  # noqa: E402
    ackley_function,
    borehole_mixed_variables,
    dixon_price_function,
    griewank_function,
    rosenbrock_function,
)

TASK_NAMES = ("ackley", "rosenbrock", "griewank", "borehole", "dixon_price")
INPUT_DIM = 20
NUM_TASKS = 5

# Borehole physical bounds (same as generate_mf_borehole_data)
_BOREHOLE_LO = torch.tensor(
    [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0], dtype=torch.float64
)
_BOREHOLE_HI = torch.tensor(
    [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0], dtype=torch.float64
)


def _affine_cube(u: Tensor, lo: float, hi: float) -> Tensor:
    return u * (hi - lo) + lo


def evaluate_5task_outputs(u: Tensor) -> Tensor:
    """
    Map shared unit-cube ``u`` ``(n, 20)`` to multitask targets ``(n, 5)``.

    Borehole uses only the first 8 columns of ``u`` (physical box); remaining
    coordinates are ignored for that task.
    """
    if u.dim() != 2 or u.shape[-1] != INPUT_DIM:
        raise ValueError(f"u must have shape (n, {INPUT_DIM}), got {tuple(u.shape)}")

    u64 = u.to(dtype=torch.float64)
    y_ack = ackley_function(_affine_cube(u64, -5.0, 10.0), INPUT_DIM)
    y_ros = rosenbrock_function(_affine_cube(u64, -5.0, 10.0), INPUT_DIM)
    y_gri = griewank_function(_affine_cube(u64, -600.0, 600.0), INPUT_DIM)
    x_bore = u64[:, :8] * (_BOREHOLE_HI - _BOREHOLE_LO) + _BOREHOLE_LO
    y_bor = borehole_mixed_variables(x_bore, source="s0")
    y_dix = dixon_price_function(_affine_cube(u64, -10.0, 10.0), INPUT_DIM)
    return torch.stack([y_ack, y_ros, y_gri, y_bor, y_dix], dim=-1)


def generate_5task_20d_data(
    n_train: int,
    n_test: int,
    *,
    seed: int = 0,
    train_noise: float = 0.01,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate shared-``X`` multitask data.

    Returns
    -------
    X_train, Y_train, X_test, Y_test
        ``X_*`` are in ``[0, 1]^{20}`` (Sobol). ``Y_*`` have shape ``(n, 5)``.
        Noise (if any) is ``fraction * per-task test std``, applied independently
        per task (mirrors existing synthetic generators).
    """
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive.")

    torch.manual_seed(seed)
    total = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=INPUT_DIM, scramble=True, seed=seed)
    u_all = sobol.draw(total).to(dtype=torch.float64)
    y_all = evaluate_5task_outputs(u_all)

    X_train = u_all[:n_train]
    Y_train = y_all[:n_train].clone()
    X_test = u_all[n_train:]
    Y_test = y_all[n_train:].clone()

    y_test_std = Y_test.std(dim=0).clamp_min(1e-12)

    def _add_noise(y: Tensor, frac: float) -> Tensor:
        if frac <= 0:
            return y
        scale = frac * y_test_std
        if noise_type == "gaussian":
            noise = torch.randn_like(y) * scale
        elif noise_type == "uniform":
            noise = (torch.rand_like(y) - 0.5) * 2.0 * scale
        else:
            raise ValueError(f"noise_type must be 'gaussian' or 'uniform', got {noise_type!r}")
        return y + noise

    Y_train = _add_noise(Y_train, train_noise)
    Y_test = _add_noise(Y_test, test_noise)
    return X_train, Y_train, X_test, Y_test
