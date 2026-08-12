"""Micro-bench: dual Woodbury MLL fwd+bwd — custom / fused-RFF / reference.

Example (repo root)::

    python -m experiments_RFF.bench_woodbury_dual_autograd
    python -m experiments_RFF.bench_woodbury_dual_autograd --n 16000 --m 4000
    set GPPLUS_WOODBURY_CUDA_TIMING=1
    python -m experiments_RFF.bench_woodbury_dual_autograd --n 4096 --m 800 --cuda-timing
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from gpplus.utils.rff_utils import (
    featurize_rbf,
    init_rbf_weights,
    woodbury_marginal_log_likelihood_dual,
    woodbury_marginal_log_likelihood_dual_reference,
)
from gpplus.utils.woodbury_mll_autograd import (
    cuda_timing_averages,
    reset_cuda_timing,
    set_cuda_timing,
    vjp_rff_from_alpha_v_w,
    vjp_rff_lengthscale_outputscale,
    woodbury_dual_mll_apply,
    woodbury_dual_rff_mll_apply,
    _phi_lam_inv,
    featurize_rbf_scaled_nograd,
)
from gpplus.utils.rff_utils import woodbury_factor_dual


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_phi_problem(
    n: int,
    m: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.randn(n, m, generator=g, dtype=dtype).to(device)
    y = torch.randn(n, generator=g, dtype=dtype).to(device)
    noise = torch.tensor(0.1, dtype=dtype, device=device)
    return noise, phi, y


def _make_rff_problem(
    n: int,
    d: int,
    num_rff: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=dtype).to(device)
    y = torch.randn(n, generator=g, dtype=dtype).to(device)
    noise = torch.tensor(0.1, dtype=dtype, device=device)
    weights = init_rbf_weights(
        d, num_rff, device=device, dtype=dtype, rff_sampling="rff"
    )
    ls = torch.randn(1, d, generator=g, dtype=dtype).to(device)
    os_ = torch.tensor(-0.5, dtype=dtype, device=device)
    return noise, y, x, ls, os_, weights, num_rff


def _step_phi(fn, noise, phi, y, jitter: float):
    noise_ = noise.detach().clone().requires_grad_(True)
    phi_ = phi.detach().clone().requires_grad_(True)
    y_ = y.detach().clone().requires_grad_(True)
    mll = fn(noise_, phi_, y_, jitter=jitter)
    mll.backward()
    return mll.detach()


def bench_phi_ms(fn, noise, phi, y, device, warmup: int, iters: int, jitter: float) -> float:
    for _ in range(warmup):
        _step_phi(fn, noise, phi, y, jitter)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _step_phi(fn, noise, phi, y, jitter)
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def _step_rff_fused(noise, y, x, ls, os_, weights, num_rff, jitter):
    n_ = noise.detach().clone().requires_grad_(True)
    y_ = y.detach().clone().requires_grad_(True)
    ls_ = ls.detach().clone().requires_grad_(True)
    os2 = os_.detach().clone().requires_grad_(True)
    mll = woodbury_dual_rff_mll_apply(n_, y_, x, ls_, os2, weights, num_rff, jitter=jitter)
    mll.backward()
    return mll.detach()


def _step_rff_unfused(noise, y, x, ls, os_, weights, num_rff, jitter):
    n_ = noise.detach().clone().requires_grad_(True)
    y_ = y.detach().clone().requires_grad_(True)
    ls_ = ls.detach().clone().requires_grad_(True)
    os2 = os_.detach().clone().requires_grad_(True)
    z = featurize_rbf(x, weights, ls_, num_rff)
    phi = z * torch.pow(10.0, os2 / 2.0)
    mll = woodbury_dual_mll_apply(n_, phi, y_, jitter=jitter)
    mll.backward()
    return mll.detach()


def bench_rff_ms(step_fn, args, device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        step_fn(*args)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step_fn(*args)
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def _read_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def bench_vjp_no_gphi_ms(
    n: int,
    d: int,
    num_rff: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, bool]:
    """Compare streaming VJP vs materializing full ∇_Φ."""
    noise, y, x, ls, os_, weights, D = _make_rff_problem(n, d, num_rff, device, dtype)
    phi, proj, z_u, s_out, s_ls = featurize_rbf_scaled_nograd(x, weights, ls, os_, D)
    chol, noise_c, z_lin = woodbury_factor_dual(noise, phi, 1e-6)
    y64 = y.to(dtype=z_lin.dtype)
    phi_ty = z_lin.T @ y64
    v = torch.cholesky_solve(phi_ty.unsqueeze(-1), chol).squeeze(-1)
    alpha = (y64 - z_lin @ v) / noise_c
    w_phi = _phi_lam_inv(z_lin, chol)
    go = torch.tensor(1.0, dtype=dtype, device=device)

    # materialize g_phi once for parity + g_phi path
    g_phi = alpha.unsqueeze(-1) * v.unsqueeze(-2) - w_phi

    def step_stream():
        return vjp_rff_from_alpha_v_w(
            alpha, v, w_phi, go, x, weights, ls, os_, proj, z_u, s_out, s_ls, D
        )

    def step_full():
        return vjp_rff_lengthscale_outputscale(
            g_phi, x, weights, ls, os_, proj, z_u, s_out, s_ls, D
        )

    g_a = step_stream()
    g_b = step_full()
    ok = torch.allclose(g_a[0], g_b[0], rtol=1e-5, atol=1e-6) and torch.allclose(
        g_a[1], g_b[1], rtol=1e-5, atol=1e-6
    )

    for _ in range(warmup):
        step_stream()
        step_full()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step_stream()
    _sync(device)
    ms_stream = (time.perf_counter() - t0) / iters * 1000.0
    t0 = time.perf_counter()
    for _ in range(iters):
        step_full()
    _sync(device)
    ms_full = (time.perf_counter() - t0) / iters * 1000.0
    return ms_stream, ms_full, ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--m", type=int, default=800, help="feature width (2 * num_rff)")
    p.add_argument("--d", type=int, default=64, help="input dim for fused-RFF bench")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--jitter", type=float, default=1e-6)
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument(
        "--cuda-timing",
        action="store_true",
        help="Enable GPPLUS_WOODBURY_CUDA_TIMING section averages during fused bench",
    )
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.cuda_timing:
        os.environ["GPPLUS_WOODBURY_CUDA_TIMING"] = "1"
        set_cuda_timing(True)
        reset_cuda_timing()

    num_rff = args.m // 2
    print(
        f"device={device} dtype={args.dtype} n={args.n} m={args.m} d={args.d} "
        f"num_rff={num_rff} warmup={args.warmup} iters={args.iters}"
    )
    noise, phi, y = _make_phi_problem(args.n, args.m, device, dtype)
    os.environ.pop("GPPLUS_WOODBURY_AUTOGRAD", None)
    os.environ.pop("GPPLUS_WOODBURY_LINALG", None)

    ms_custom = bench_phi_ms(
        woodbury_dual_mll_apply, noise, phi, y, device, args.warmup, args.iters, args.jitter
    )
    ms_ref = bench_phi_ms(
        woodbury_marginal_log_likelihood_dual_reference,
        noise,
        phi,
        y,
        device,
        args.warmup,
        args.iters,
        args.jitter,
    )

    rff = _make_rff_problem(args.n, args.d, num_rff, device, dtype)
    rff_args = (*rff, args.jitter)

    peak_fused = peak_unfused = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ms_fused = bench_rff_ms(_step_rff_fused, rff_args, device, args.warmup, args.iters)
    if device.type == "cuda":
        peak_fused = _read_peak_mb(device)
        torch.cuda.reset_peak_memory_stats(device)
    ms_unfused = bench_rff_ms(_step_rff_unfused, rff_args, device, args.warmup, args.iters)
    if device.type == "cuda":
        peak_unfused = _read_peak_mb(device)

    ms_vjp_stream, ms_vjp_full, vjp_ok = bench_vjp_no_gphi_ms(
        args.n, args.d, num_rff, device, dtype, args.warmup, args.iters
    )

    # Grad parity smoke (phi path)
    n0 = noise.detach().clone().requires_grad_(True)
    p0 = phi.detach().clone().requires_grad_(True)
    y0 = y.detach().clone().requires_grad_(True)
    woodbury_dual_mll_apply(n0, p0, y0, jitter=args.jitter).backward()
    n1 = noise.detach().clone().requires_grad_(True)
    p1 = phi.detach().clone().requires_grad_(True)
    y1 = y.detach().clone().requires_grad_(True)
    woodbury_marginal_log_likelihood_dual_reference(
        n1, p1, y1, jitter=args.jitter
    ).backward()
    g_ok = torch.allclose(p0.grad, p1.grad, rtol=1e-4, atol=1e-5) and torch.allclose(
        n0.grad, n1.grad, rtol=1e-4, atol=1e-5
    )

    mll_api = woodbury_marginal_log_likelihood_dual(noise, phi, y, jitter=args.jitter)
    print(f"custom_autograd     {ms_custom:.2f} ms/step")
    print(f"reference_chol      {ms_ref:.2f} ms/step")
    print(f"speedup_vs_ref      {ms_ref / ms_custom:.2f}x")
    print(f"fused_rff_mll       {ms_fused:.2f} ms/step")
    print(f"unfused_rff_mll     {ms_unfused:.2f} ms/step")
    print(f"speedup_fuse_featurize {ms_unfused / ms_fused:.2f}x")
    print(f"vjp_no_gphi         {ms_vjp_stream:.2f} ms/call")
    print(f"vjp_full_gphi       {ms_vjp_full:.2f} ms/call")
    print(f"speedup_vjp_no_gphi {ms_vjp_full / max(ms_vjp_stream, 1e-9):.2f}x")
    print(f"vjp_parity          {'ok' if vjp_ok else 'FAIL'}")
    if peak_fused is not None:
        print(f"peak_mem_fused_mb   {peak_fused:.1f}")
        print(f"peak_mem_unfused_mb {peak_unfused:.1f}")
    print(f"grad_parity         {'ok' if g_ok else 'FAIL'}")
    print(f"public_mll          {float(mll_api.detach().cpu()):.6f}")

    if args.cuda_timing:
        avgs = cuda_timing_averages()
        if avgs:
            parts = " ".join(f"{k}={v:.2f}ms" for k, v in sorted(avgs.items()))
            print(f"cuda_section_avg    {parts}")
        else:
            print("cuda_section_avg    (no samples — CPU or timing disabled)")

    return 0 if (g_ok and vjp_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
