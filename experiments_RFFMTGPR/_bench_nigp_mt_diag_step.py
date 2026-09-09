"""Micro-bench: MT NIGP diag-noise Woodbury fwd+bwd and chunked predict."""
from __future__ import annotations

import argparse
import time

import torch

from gpplus.utils.rff_utils import (
    featurize_rbf,
    woodbury_factor_mt_diag_noise,
    woodbury_marginal_log_likelihood_mt_diag_noise,
    woodbury_predict_mt_diag_noise,
    woodbury_solve_mt_diag_noise_from_chol,
    icm_omega_rmatvec,
)
from gpplus.utils.woodbury_mll_autograd import woodbury_mt_diag_mll_apply


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_problem(n: int, m: int, t: int, d_in: int, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n, d_in, generator=g).to(device=device, dtype=torch.float32)
    w = torch.randn(d_in, m // 2, generator=g).to(device=device, dtype=torch.float32)
    ls = torch.randn(d_in, generator=g).to(device=device, dtype=torch.float32)
    phi = featurize_rbf(x, w, ls, num_samples=m // 2).detach()
    f = torch.randn(t, 1, generator=g).to(device=device, dtype=torch.float32)
    v = torch.rand(t, generator=g).to(device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(0.5 * (b + b.T) + 1e-5 * torch.eye(t, device=device))
    d_nt = 0.05 + 0.05 * torch.rand(n, t, generator=g).to(device=device, dtype=torch.float32)
    y = torch.randn(n * t, generator=g).to(device=device, dtype=torch.float32)
    return phi, r_b, d_nt, y


def _time(fn, device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--t", type=int, default=4)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    phi, r_b, d_nt, y = _make_problem(args.n, args.m, args.t, args.d, device)

    def ref_step():
        phi_ = phi.detach().requires_grad_(True)
        rb_ = r_b.detach().requires_grad_(True)
        d_ = d_nt.detach().requires_grad_(True)
        mll = woodbury_marginal_log_likelihood_mt_diag_noise(
            d_, phi_, rb_, args.n, y, jitter=1e-5
        )
        mll.backward()

    def custom_step():
        phi_ = phi.detach().requires_grad_(True)
        rb_ = r_b.detach().requires_grad_(True)
        d_ = d_nt.detach().requires_grad_(True)
        mll = woodbury_mt_diag_mll_apply(d_, phi_, rb_, y, jitter=1e-5)
        mll.backward()

    ms_ref = _time(ref_step, device, args.warmup, args.iters)
    ms_custom = _time(custom_step, device, args.warmup, args.iters)
    print(f"fwd+bwd reference Chol-autodiff: {ms_ref:.1f} ms")
    print(f"fwd+bwd custom autograd:         {ms_custom:.1f} ms")

    # Predict: refactor-every-chunk vs factor-once
    n_te = args.n
    phi_te = phi
    d_te = d_nt

    def predict_naive():
        with torch.no_grad():
            for start in range(0, n_te, args.chunk):
                woodbury_predict_mt_diag_noise(
                    d_nt,
                    phi,
                    phi_te[start : start + args.chunk],
                    r_b,
                    args.n,
                    args.t,
                    y,
                    d_test=d_te[start : start + args.chunk],
                )

    def predict_reuse():
        with torch.no_grad():
            chol, d_c = woodbury_factor_mt_diag_noise(d_nt, phi, r_b, jitter=1e-5)
            alpha = woodbury_solve_mt_diag_noise_from_chol(d_c, phi, r_b, chol, y.to(chol.dtype))
            v = icm_omega_rmatvec(phi, r_b, alpha.to(phi.dtype))
            for start in range(0, n_te, args.chunk):
                woodbury_predict_mt_diag_noise(
                    d_nt,
                    phi,
                    phi_te[start : start + args.chunk],
                    r_b,
                    args.n,
                    args.t,
                    y,
                    d_test=d_te[start : start + args.chunk],
                    chol=chol,
                    d_factor=d_c,
                    alpha=alpha,
                    feature_weights=v,
                )

    ms_naive = _time(predict_naive, device, args.warmup, args.iters)
    ms_reuse = _time(predict_reuse, device, args.warmup, args.iters)
    n_chunks = (n_te + args.chunk - 1) // args.chunk
    print(f"predict {n_chunks} chunks refactor-each: {ms_naive:.1f} ms")
    print(f"predict {n_chunks} chunks factor-once:   {ms_reuse:.1f} ms")


if __name__ == "__main__":
    main()
