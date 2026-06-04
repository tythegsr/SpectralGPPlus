"""
Benchmark RFF Woodbury training step and chunked prediction.

Run: python -m gpplus.tests.benchmark_rff_woodbury
"""

from __future__ import annotations

import time

import torch

from gpplus.models import RFFGPR
from gpplus.training.eval import evaluate_rff_gp_model
from gpplus.training.rff_mll import RFFWoodburyMarginalLogLikelihood


def _time_mll_backward(model, mll, train_y, repeats: int = 5) -> float:
    model.train()
    for _ in range(2):
        loss = -mll(None, train_y)
        loss.backward()
        model.zero_grad()
    start = time.perf_counter()
    for _ in range(repeats):
        loss = -mll(None, train_y)
        loss.backward()
        model.zero_grad()
    return (time.perf_counter() - start) / repeats


def _time_predict(model, test_x, chunk_size: int | None, repeats: int = 3) -> float:
    model.eval()
    kwargs = {"jitter": 1e-6}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    for _ in range(1):
        evaluate_rff_gp_model(model, test_x, **kwargs)
    start = time.perf_counter()
    for _ in range(repeats):
        evaluate_rff_gp_model(model, test_x, **kwargs)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    torch.manual_seed(0)
    n, d, num_rff = 1600, 40, 200
    n_test = 5000
    m = 2 * num_rff

    train_x = torch.randn(n, d, dtype=torch.float64)
    train_y = torch.randn(n, dtype=torch.float64)
    test_x = torch.randn(n_test, d, dtype=torch.float64)

    model = RFFGPR(train_x, train_y, num_rff=num_rff, ard=True)
    mll = RFFWoodburyMarginalLogLikelihood(model.likelihood, model)

    print(f"n={n}, m={m} (2*num_rff), m/n={m/n:.3f}, n_test={n_test}")
    t_mll = _time_mll_backward(model, mll, train_y)
    print(f"MLL + backward (avg): {t_mll * 1000:.2f} ms")

    t_full = _time_predict(model, test_x, chunk_size=0)
    print(f"Predict full batch (chunk_size=0): {t_full * 1000:.2f} ms")

    t_chunk = _time_predict(model, test_x, chunk_size=512)
    print(f"Predict chunked (512): {t_chunk * 1000:.2f} ms")


if __name__ == "__main__":
    main()
