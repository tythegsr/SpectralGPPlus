"""Microbenchmark: Tier-2 Chol(M) vs Tier-3 eigen factor for multitask Woodbury MLL.

Synthetic Phi shaped like TOA defaults (n=16000, T=2, m=2D) without loading the dataset.
Times ``woodbury_factor_mt`` + one ``woodbury_marginal_log_likelihood_mt`` forward
(no autograd) for method in {chol, eigen} at D in {400, 800, 1600}.

Usage:
  python experiments_RFFMTGPR/bench_woodbury_mt_eigen.py
  python experiments_RFFMTGPR/bench_woodbury_mt_eigen.py --device cuda --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from gpplus.utils.rff_utils import woodbury_factor_mt, woodbury_marginal_log_likelihood_mt


def _make_inputs(n: int, d_freq: int, num_tasks: int, device: torch.device, dtype: torch.dtype):
    m = 2 * d_freq
    torch.manual_seed(0)
    phi = torch.randn(n, m, device=device, dtype=dtype) / (m**0.5)
    a = torch.randn(num_tasks, num_tasks, device=device, dtype=dtype)
    r_b = a @ a.T + 0.1 * torch.eye(num_tasks, device=device, dtype=dtype)
    task_noises = torch.full((num_tasks,), 0.05, device=device, dtype=dtype)
    y = torch.randn(n * num_tasks, device=device, dtype=dtype)
    return phi, r_b, task_noises, y


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, repeats: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
        _sync(device)
    times: list[float] = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def bench_one(
    n: int,
    d_freq: int,
    num_tasks: int,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
) -> dict:
    phi, r_b, task_noises, y = _make_inputs(n, d_freq, num_tasks, device, dtype)
    m = phi.shape[-1]
    m_t = m * num_tasks

    def run_factor():
        return woodbury_factor_mt(task_noises, phi, r_b, jitter=1e-6, method=method)

    def run_mll():
        return woodbury_marginal_log_likelihood_mt(
            task_noises, phi, r_b, n, y, jitter=1e-6, method=method
        )

    # Peak dense M size if Chol were used (bytes for float64)
    dense_m_bytes = m_t * m_t * 8

    factor_times = _time_call(run_factor, repeats=repeats, warmup=warmup, device=device)
    mll_times = _time_call(run_mll, repeats=repeats, warmup=warmup, device=device)

    # Force one value for sanity
    mll_val = float(run_mll().detach().cpu())

    return {
        "D": d_freq,
        "m": m,
        "mT": m_t,
        "method": method,
        "factor_median_s": statistics.median(factor_times),
        "mll_median_s": statistics.median(mll_times),
        "mll_value": mll_val,
        "dense_M_GiB": dense_m_bytes / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16000)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--d-list", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print(
        f"n={args.n} T={args.tasks} dtype={args.dtype} device={device} "
        f"repeats={args.repeats} (median)"
    )
    print(
        f"{'D':>5} {'mT':>6} {'method':>6} {'factor_s':>10} {'mll_s':>10} "
        f"{'speedup':>8} {'dense_M_GiB':>12} {'mll':>12}"
    )

    for d_freq in args.d_list:
        rows = {}
        for method in ("chol", "eigen"):
            # Skip chol when dense M is huge on purpose? Still try; OOM is informative.
            try:
                rows[method] = bench_one(
                    args.n,
                    d_freq,
                    args.tasks,
                    method,
                    device,
                    dtype,
                    args.repeats,
                    args.warmup,
                )
            except Exception as exc:  # noqa: BLE001 — report OOM / linalg failures
                print(f"{d_freq:5d} {'?':>6} {method:>6} FAILED: {exc}")
                rows[method] = None

        if rows.get("chol") and rows.get("eigen"):
            speed = rows["chol"]["mll_median_s"] / max(rows["eigen"]["mll_median_s"], 1e-12)
            for method in ("chol", "eigen"):
                r = rows[method]
                sp = speed if method == "eigen" else 1.0
                print(
                    f"{r['D']:5d} {r['mT']:6d} {r['method']:>6} "
                    f"{r['factor_median_s']:10.4f} {r['mll_median_s']:10.4f} "
                    f"{sp:8.2f}x {r['dense_M_GiB']:12.3f} {r['mll_value']:12.4g}"
                )
            # Value parity check (same inputs)
            diff = abs(rows["chol"]["mll_value"] - rows["eigen"]["mll_value"])
            print(f"      |mll_chol - mll_eigen| = {diff:.3e}")


if __name__ == "__main__":
    main()
