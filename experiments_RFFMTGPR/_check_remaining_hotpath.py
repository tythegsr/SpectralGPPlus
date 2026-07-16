"""Accuracy gate: mixed Omega solve + ARD W-scale vs legacy full-float64 path."""
from __future__ import annotations

import math
import sys

import torch

from gpplus.utils.rff_utils import (
    featurize_rbf,
    woodbury_marginal_log_likelihood_mt,
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs() / b.abs().clamp_min(1e-12))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1)
    b_f = b.reshape(-1)
    denom = a_f.norm() * b_f.norm()
    if float(denom) < 1e-20:
        return 1.0
    return float((a_f @ b_f) / denom)


def check_featurize_ard(device: str) -> None:
    torch.manual_seed(1)
    n, d, D = 512, 32, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    w = torch.randn(d, D, device=device, dtype=torch.float32)
    ls = torch.randn(d, device=device, dtype=torch.float32)  # ARD log10-ish

    # Legacy formula: scale rows of x
    scale = torch.pow(10.0, ls / 2.0)
    legacy = (1.0 / math.sqrt(D)) * torch.cat(
        [
            torch.cos(x.mul(scale).matmul(w)),
            torch.sin(x.mul(scale).matmul(w)),
        ],
        dim=-1,
    )
    new = featurize_rbf(x, w, ls, num_samples=D)
    max_abs = float((legacy - new).abs().max())
    print(f"featurize ARD max_abs_diff={max_abs:.3e}")
    if max_abs > 1e-5:
        raise AssertionError(f"featurize_rbf ARD mismatch: max_abs={max_abs}")

    # Isotropic scalar lengthscale
    ls0 = torch.tensor(-1.5, device=device, dtype=torch.float32)
    scale0 = torch.pow(10.0, ls0 / 2.0)
    legacy0 = (1.0 / math.sqrt(D)) * torch.cat(
        [
            torch.cos(x.mul(scale0).matmul(w)),
            torch.sin(x.mul(scale0).matmul(w)),
        ],
        dim=-1,
    )
    new0 = featurize_rbf(x, w, ls0, num_samples=D)
    max_abs0 = float((legacy0 - new0).abs().max())
    print(f"featurize isotropic max_abs_diff={max_abs0:.3e}")
    if max_abs0 > 1e-5:
        raise AssertionError(f"featurize_rbf isotropic mismatch: max_abs={max_abs0}")


def check_mll_gate(
    *,
    n: int,
    m: int,
    t: int,
    device: str,
    tag: str,
    mll_rel_tol: float = 1e-6,
    grad_med_rel_tol: float = 1e-5,
) -> None:
    torch.manual_seed(0)
    phi = torch.randn(n, m, device=device, dtype=torch.float32)
    f = torch.randn(t, 1, device=device, dtype=torch.float32)
    v = torch.rand(t, device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(0.5 * (b + b.T) + 1e-8 * torch.eye(t, device=device))
    task_noises = torch.tensor([0.05, 0.08][:t], device=device, dtype=torch.float32)
    if t > 2:
        task_noises = 0.05 + 0.03 * torch.rand(t, device=device, dtype=torch.float32)
    y = torch.randn(n * t, device=device, dtype=torch.float32)

    phi1 = phi.clone().requires_grad_(True)
    phi2 = phi.clone().requires_grad_(True)
    rb1 = r_b.clone().requires_grad_(True)
    rb2 = r_b.clone().requires_grad_(True)
    tn1 = task_noises.clone().requires_grad_(True)
    tn2 = task_noises.clone().requires_grad_(True)

    mll_mixed = woodbury_marginal_log_likelihood_mt(
        tn1, phi1, rb1, n, y, promote_features=False
    )
    mll_full = woodbury_marginal_log_likelihood_mt(
        tn2, phi2, rb2, n, y, promote_features=True
    )
    mll_mixed.backward()
    mll_full.backward()

    mll_rel = _rel(mll_mixed.detach(), mll_full.detach())
    phi_med = float(
        ((phi1.grad - phi2.grad).abs() / phi2.grad.abs().clamp_min(1e-12)).median()
    )
    # Relatives ignore near-zero entries (floor 1e-3 of max |grad|).
    def _robust_max_rel(g_a: torch.Tensor, g_b: torch.Tensor) -> float:
        floor = 1e-3 * float(g_b.abs().max().clamp_min(1e-12))
        mask = g_b.abs() >= floor
        if not bool(mask.any()):
            return 0.0
        return float(((g_a - g_b).abs()[mask] / g_b.abs()[mask]).max())

    phi_rel = _robust_max_rel(phi1.grad, phi2.grad)
    rb_rel = _robust_max_rel(rb1.grad, rb2.grad)
    tn_rel = _robust_max_rel(tn1.grad, tn2.grad)
    phi_cos = _cos(phi1.grad, phi2.grad)
    rb_cos = _cos(rb1.grad, rb2.grad)
    tn_cos = _cos(tn1.grad, tn2.grad)

    print(
        f"[{tag}] device={device} n={n} m={m} T={t} "
        f"mll_rel={mll_rel:.3e} phi_med_rel={phi_med:.3e} "
        f"phi_rel={phi_rel:.3e} rb_rel={rb_rel:.3e} tn_rel={tn_rel:.3e} "
        f"phi_cos={phi_cos:.8f} rb_cos={rb_cos:.8f} tn_cos={tn_cos:.8f}"
    )
    if mll_rel > mll_rel_tol:
        raise AssertionError(f"{tag}: MLL rel diff {mll_rel} > {mll_rel_tol}")
    if phi_med > grad_med_rel_tol:
        raise AssertionError(f"{tag}: phi grad median rel {phi_med} > {grad_med_rel_tol}")
    # Max-rel outliers ~1e-4 are expected under float32 Omega matvecs; median+MLL
    # stay at 1e-7–1e-9. Fail closed on ≥0.1% robust drift.
    if phi_rel > 1e-3:
        raise AssertionError(f"{tag}: phi grad robust max rel {phi_rel} > 1e-3")
    if rb_rel > 1e-3:
        raise AssertionError(f"{tag}: r_b grad robust max rel {rb_rel} > 1e-3")
    if tn_rel > 1e-3:
        raise AssertionError(f"{tag}: task_noise grad robust max rel {tn_rel} > 1e-3")
    # Cosine can slightly exceed 1 in float32 noise; require near-alignment.
    if phi_cos < 1.0 - 1e-5:
        raise AssertionError(f"{tag}: phi grad cosine {phi_cos} too low")
    if rb_cos < 1.0 - 1e-5:
        raise AssertionError(f"{tag}: r_b grad cosine {rb_cos} too low")
    if tn_cos < 1.0 - 1e-5:
        raise AssertionError(f"{tag}: task_noise grad cosine {tn_cos} too low")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("accuracy gate device:", device)
    check_featurize_ard(device)
    check_mll_gate(n=2048, m=128, t=2, device=device, tag="synthetic")
    # TOA-shaped smoke: n~4k, m=2*400=800
    check_mll_gate(
        n=4096,
        m=800,
        t=2,
        device=device,
        tag="toa_smoke",
        mll_rel_tol=1e-6,
        grad_med_rel_tol=1e-5,
    )
    print("ACCURACY_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
