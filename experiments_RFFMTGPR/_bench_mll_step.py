"""Micro-bench: fwd+bwd Woodbury MLL step with TF32 on/off + accuracy gate."""
from __future__ import annotations

import argparse
import time

import torch

from gpplus.training.trainer_utils import (
    configure_woodbury_matmul_precision,
    enable_fast_float32_matmul,
)
from gpplus.utils.rff_utils import (
    featurize_rbf,
    woodbury_marginal_log_likelihood_mt,
)


def _set_tf32(enabled: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_problem(n: int, m: int, t: int, d: int, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n, d, generator=g).to(device=device, dtype=torch.float32)
    w = torch.randn(d, m // 2, generator=g).to(device=device, dtype=torch.float32)
    ls = torch.randn(d, generator=g).to(device=device, dtype=torch.float32)
    phi = featurize_rbf(x, w, ls, num_samples=m // 2).detach()
    f = torch.randn(t, 1, generator=g).to(device=device, dtype=torch.float32)
    v = torch.rand(t, generator=g).to(device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(0.5 * (b + b.T) + 1e-8 * torch.eye(t, device=device))
    tn = (0.05 + 0.03 * torch.rand(t, generator=g)).to(device=device, dtype=torch.float32)
    y = torch.randn(n * t, generator=g).to(device=device, dtype=torch.float32)
    return phi, r_b, tn, y


def _mll_with_grad(phi, r_b, tn, y, n: int):
    phi_ = phi.detach().clone().requires_grad_(True)
    rb_ = r_b.detach().clone().requires_grad_(True)
    tn_ = tn.detach().clone().requires_grad_(True)
    mll = woodbury_marginal_log_likelihood_mt(tn_, phi_, rb_, n, y, promote_features=False)
    mll.backward()
    return mll.detach(), phi_.grad.detach(), rb_.grad.detach(), tn_.grad.detach()


def bench_ms(phi, r_b, tn, y, n: int, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _mll_with_grad(phi, r_b, tn, y, n)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mll_with_grad(phi, r_b, tn, y, n)
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--m", type=int, default=800, help="feature width 2D")
    p.add_argument("--t", type=int, default=2)
    p.add_argument("--d", type=int, default=270)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} n={args.n} m={args.m} T={args.t} d={args.d}")

    phi, r_b, tn, y = _make_problem(args.n, args.m, args.t, args.d, device)

    # Default Woodbury path: TF32 off (highest). Opt-in TF32 is measured only
    # as a contrast — it is NOT the production default (can collapse validation).
    configure_woodbury_matmul_precision(device=device, dtype=torch.float32)
    _set_tf32(False)
    mll0, g_phi0, g_rb0, g_tn0 = _mll_with_grad(phi, r_b, tn, y, args.n)

    if device.type == "cuda":
        ms_off = bench_ms(phi, r_b, tn, y, args.n, device, args.warmup, args.iters)

        enable_fast_float32_matmul(device=device, dtype=torch.float32)
        _set_tf32(True)
        mll1, g_phi1, g_rb1, g_tn1 = _mll_with_grad(phi, r_b, tn, y, args.n)
        mll_rel = float((mll1 - mll0).abs() / mll0.abs().clamp_min(1e-12))
        phi_med = float(
            ((g_phi1 - g_phi0).abs() / g_phi0.abs().clamp_min(1e-12)).median()
        )
        phi_max = float(
            ((g_phi1 - g_phi0).abs() / g_phi0.abs().clamp_min(1e-12)).max()
        )
        print(
            f"TF32 vs off (opt-in only): mll_rel={mll_rel:.3e} "
            f"phi_grad_med_rel={phi_med:.3e} phi_grad_max_rel={phi_max:.3e}"
        )
        ms_on = bench_ms(phi, r_b, tn, y, args.n, device, args.warmup, args.iters)
        print(f"step_ms TF32-off={ms_off:.2f}  TF32-on={ms_on:.2f}  speedup={ms_off/max(ms_on,1e-9):.2f}x")
        print("NOTE: production Woodbury training keeps TF32 off.")
        _set_tf32(False)
        configure_woodbury_matmul_precision(device=device, dtype=torch.float32)
    else:
        ms = bench_ms(phi, r_b, tn, y, args.n, device, args.warmup, args.iters)
        print(f"step_ms (cpu, no TF32)={ms:.2f}")
        print("skip TF32 accuracy/speed comparison (CUDA not available)")

    # Featurize buffer sanity vs legacy cat formula
    torch.manual_seed(0)
    x = torch.randn(512, args.d, device=device, dtype=torch.float32)
    w = torch.randn(args.d, args.m // 2, device=device, dtype=torch.float32)
    ls = torch.randn(args.d, device=device, dtype=torch.float32)
    z_new = featurize_rbf(x, w, ls, num_samples=args.m // 2)
    scale = torch.pow(10.0, ls / 2.0).reshape(-1, 1)
    proj = x.matmul(w * scale)
    inv = 1.0 / (args.m // 2) ** 0.5
    z_legacy = inv * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    max_abs = float((z_new - z_legacy).abs().max())
    print(f"featurize buffer vs cat max_abs={max_abs:.3e}")
    if max_abs > 1e-5:
        raise AssertionError(f"featurize buffer mismatch: {max_abs}")

    print("BENCH_MLL_STEP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
