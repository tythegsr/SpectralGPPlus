"""Accuracy gate: feature-space predictive variance + mixed-precision predict."""
from __future__ import annotations

import sys

import torch

from gpplus.utils.rff_utils import (
    apply_middle_inverse_eigen,
    build_icm_joint_features,
    woodbury_factor,
    woodbury_factor_mt,
    woodbury_predictive_mean,
    woodbury_predictive_mean_mt,
    woodbury_predictive_var_diag,
    woodbury_predictive_var_diag_dense_ref,
    woodbury_predictive_var_diag_mt,
    woodbury_predictive_var_diag_mt_dense_ref,
)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _max_rel(a: torch.Tensor, b: torch.Tensor, floor: float = 1e-8) -> float:
    denom = b.abs().clamp_min(floor)
    return float(((a - b).abs() / denom).max())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1)
    b_f = b.reshape(-1)
    denom = a_f.norm() * b_f.norm()
    if float(denom) < 1e-20:
        return 1.0
    return float((a_f @ b_f) / denom)


def check_st_var(
    *,
    n_train: int,
    n_test: int,
    m: int,
    device: str,
    tag: str,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-4,
) -> None:
    torch.manual_seed(0)
    # float64: dense Sigma^{-1} path is algebraically equal at jitter=0.
    z_train = torch.randn(n_train, m, device=device, dtype=torch.float64)
    z_test = torch.randn(n_test, m, device=device, dtype=torch.float64)
    noise_var = torch.tensor(0.05, device=device, dtype=torch.float64)
    chol, noise = woodbury_factor(noise_var, z_train, jitter=0.0)

    f_var = woodbury_predictive_var_diag(
        noise_var, z_train, z_test, jitter=0.0, chol=chol, noise=noise
    )
    f_var_ref = woodbury_predictive_var_diag_dense_ref(
        noise_var, z_train, z_test, jitter=0.0, chol=chol, noise=noise
    )
    max_abs = _max_abs(f_var, f_var_ref)
    max_rel = _max_rel(f_var, f_var_ref)
    print(
        f"[{tag}] ST f_var dense vs feature-space: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    if max_abs > abs_tol and max_rel > rel_tol:
        raise AssertionError(
            f"{tag}: ST f_var drift max_abs={max_abs} max_rel={max_rel}"
        )


def check_mt_var_vs_materialised(
    *,
    n_train: int,
    n_test: int,
    m: int,
    t: int,
    device: str,
    tag: str,
    method: str = "eigen",
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-5,
) -> None:
    """Feature-space path vs materialised ``Omega_* M^{-1} Omega_*^T`` diag."""
    torch.manual_seed(1)
    phi_train = torch.randn(n_train, m, device=device, dtype=torch.float32)
    phi_test = torch.randn(n_test, m, device=device, dtype=torch.float32)
    f = torch.randn(t, 1, device=device, dtype=torch.float32)
    v = torch.rand(t, device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(
        0.5 * (b + b.T) + 1e-8 * torch.eye(t, device=device)
    )
    task_noises = 0.05 + 0.03 * torch.rand(t, device=device, dtype=torch.float32)

    factor, noise = woodbury_factor_mt(
        task_noises,
        phi_train,
        r_b,
        jitter=0.0,
        method=method,  # type: ignore[arg-type]
    )
    f_var = woodbury_predictive_var_diag_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        jitter=0.0,
        factor=factor,
        noise=noise,
        method=method,  # type: ignore[arg-type]
    )

    if factor.kind == "eigen":
        omega = build_icm_joint_features(
            phi_test.to(dtype=factor.q_g.dtype),
            r_b.to(dtype=factor.q_s.dtype),
        )
        solved = apply_middle_inverse_eigen(factor, omega.transpose(-1, -2))
        ref = (omega * solved.transpose(-1, -2)).sum(dim=-1).clamp_min(0.0)
    else:
        omega = build_icm_joint_features(
            phi_test.to(dtype=factor.chol.dtype),
            r_b.to(dtype=factor.chol.dtype),
        )
        solved = torch.cholesky_solve(omega.transpose(-1, -2), factor.chol)
        ref = (omega * solved.transpose(-1, -2)).sum(dim=-1).clamp_min(0.0)
    ref = ref.to(dtype=f_var.dtype)

    max_abs = _max_abs(f_var, ref)
    max_rel = _max_rel(f_var, ref)
    print(
        f"[{tag}] MT({method}) f_var materialised Omega M^{-1}: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    if max_abs > abs_tol and max_rel > rel_tol:
        raise AssertionError(
            f"{tag}: MT f_var drift max_abs={max_abs} max_rel={max_rel}"
        )


def check_mt_var_vs_dense_f64(
    *,
    n_train: int,
    n_test: int,
    m: int,
    t: int,
    device: str,
    tag: str,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
) -> None:
    """Dense Kronecker Sigma path vs feature-space in float64 (jitter=0)."""
    torch.manual_seed(1)
    phi_train = torch.randn(n_train, m, device=device, dtype=torch.float64)
    phi_test = torch.randn(n_test, m, device=device, dtype=torch.float64)
    f = torch.randn(t, 1, device=device, dtype=torch.float64)
    v = torch.rand(t, device=device, dtype=torch.float64) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(
        0.5 * (b + b.T) + 1e-12 * torch.eye(t, device=device, dtype=torch.float64)
    )
    task_noises = 0.05 + 0.03 * torch.rand(t, device=device, dtype=torch.float64)

    factor, noise = woodbury_factor_mt(
        task_noises, phi_train, r_b, jitter=0.0, method="eigen"
    )
    f_var = woodbury_predictive_var_diag_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        jitter=0.0,
        factor=factor,
        noise=noise,
        method="eigen",
    )
    f_var_ref = woodbury_predictive_var_diag_mt_dense_ref(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        jitter=0.0,
        factor=factor,
        noise=noise,
        method="eigen",
    )
    max_abs = _max_abs(f_var, f_var_ref)
    max_rel = _max_rel(f_var, f_var_ref)
    print(
        f"[{tag}] MT dense vs feature-space (f64): "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    if max_abs > abs_tol and max_rel > rel_tol:
        raise AssertionError(
            f"{tag}: MT f64 dense drift max_abs={max_abs} max_rel={max_rel}"
        )


def check_mt_mean_mixed_vs_promote(
    *,
    n_train: int,
    n_test: int,
    m: int,
    t: int,
    device: str,
    tag: str,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> None:
    """Mixed-precision predict mean vs full float64 feature promote (same factor)."""
    torch.manual_seed(2)
    phi_train = torch.randn(n_train, m, device=device, dtype=torch.float32)
    phi_test = torch.randn(n_test, m, device=device, dtype=torch.float32)
    f = torch.randn(t, 1, device=device, dtype=torch.float32)
    v = torch.rand(t, device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(
        0.5 * (b + b.T) + 1e-8 * torch.eye(t, device=device)
    )
    task_noises = 0.05 + 0.03 * torch.rand(t, device=device, dtype=torch.float32)
    y = torch.randn(n_train * t, device=device, dtype=torch.float32)

    factor, noise = woodbury_factor_mt(task_noises, phi_train, r_b, method="eigen")
    mean_mixed = woodbury_predictive_mean_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        y,
        factor=factor,
        noise=noise,
        method="eigen",
    )
    lin = factor.q_g.dtype
    mean_full = woodbury_predictive_mean_mt(
        task_noises.to(lin),
        phi_train.to(lin),
        phi_test.to(lin),
        r_b.to(lin),
        n_train,
        y.to(lin),
        factor=factor,
        noise=noise.to(lin),
        method="eigen",
    ).to(dtype=mean_mixed.dtype)

    max_abs = _max_abs(mean_mixed, mean_full)
    med_rel = float(
        (
            (mean_mixed - mean_full).abs()
            / mean_full.abs().clamp_min(1e-12)
        ).median()
    )
    cos = _cos(mean_mixed, mean_full)
    print(
        f"[{tag}] MT mean mixed vs float64 promote: "
        f"max_abs={max_abs:.3e} med_rel={med_rel:.3e} cos={cos:.8f}"
    )
    if cos < 1.0 - 1e-4:
        raise AssertionError(f"{tag}: MT mean cosine {cos} too low")
    if med_rel > rel_tol:
        raise AssertionError(f"{tag}: MT mean median rel {med_rel} > {rel_tol}")
    if max_abs > abs_tol:
        raise AssertionError(f"{tag}: MT mean max_abs {max_abs} > {abs_tol}")


def check_st_mean_mixed_vs_promote(
    *,
    n_train: int,
    n_test: int,
    m: int,
    device: str,
    tag: str,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> None:
    torch.manual_seed(3)
    z_train = torch.randn(n_train, m, device=device, dtype=torch.float32)
    z_test = torch.randn(n_test, m, device=device, dtype=torch.float32)
    y = torch.randn(n_train, device=device, dtype=torch.float32)
    noise_var = torch.tensor(0.05, device=device, dtype=torch.float32)
    chol, noise = woodbury_factor(noise_var, z_train)

    mean_mixed = woodbury_predictive_mean(
        noise_var, z_train, z_test, y, chol=chol, noise=noise
    )
    lin = chol.dtype
    mean_full = woodbury_predictive_mean(
        noise_var.to(lin),
        z_train.to(lin),
        z_test.to(lin),
        y.to(lin),
        chol=chol,
        noise=noise.to(lin),
    ).to(dtype=mean_mixed.dtype)

    max_abs = _max_abs(mean_mixed, mean_full)
    med_rel = float(
        (
            (mean_mixed - mean_full).abs()
            / mean_full.abs().clamp_min(1e-12)
        ).median()
    )
    cos = _cos(mean_mixed, mean_full)
    print(
        f"[{tag}] ST mean mixed vs float64 promote: "
        f"max_abs={max_abs:.3e} med_rel={med_rel:.3e} cos={cos:.8f}"
    )
    if cos < 1.0 - 1e-4:
        raise AssertionError(f"{tag}: ST mean cosine {cos} too low")
    if med_rel > rel_tol:
        raise AssertionError(f"{tag}: ST mean median rel {med_rel} > {rel_tol}")


def check_mt_var_mixed_vs_promote(
    *,
    n_train: int,
    n_test: int,
    m: int,
    t: int,
    device: str,
    tag: str,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-5,
) -> None:
    """Feature-space var from float32 Phi vs casting Phi to factor dtype first."""
    torch.manual_seed(4)
    phi_train = torch.randn(n_train, m, device=device, dtype=torch.float32)
    phi_test = torch.randn(n_test, m, device=device, dtype=torch.float32)
    f = torch.randn(t, 1, device=device, dtype=torch.float32)
    v = torch.rand(t, device=device, dtype=torch.float32) + 0.1
    b = f @ f.T + torch.diag(v)
    r_b = torch.linalg.cholesky(
        0.5 * (b + b.T) + 1e-8 * torch.eye(t, device=device)
    )
    task_noises = 0.05 + 0.03 * torch.rand(t, device=device, dtype=torch.float32)

    factor, noise = woodbury_factor_mt(task_noises, phi_train, r_b, method="eigen")
    var_mixed = woodbury_predictive_var_diag_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        factor=factor,
        noise=noise,
        method="eigen",
    )
    lin = factor.q_g.dtype
    var_full = woodbury_predictive_var_diag_mt(
        task_noises.to(lin),
        phi_train.to(lin),
        phi_test.to(lin),
        r_b.to(lin),
        n_train,
        factor=factor,
        noise=noise.to(lin),
        method="eigen",
    ).to(dtype=var_mixed.dtype)

    max_abs = _max_abs(var_mixed, var_full)
    max_rel = _max_rel(var_mixed, var_full)
    print(
        f"[{tag}] MT var mixed vs float64 promote: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    if max_abs > abs_tol and max_rel > rel_tol:
        raise AssertionError(
            f"{tag}: MT var mixed drift max_abs={max_abs} max_rel={max_rel}"
        )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("predictive accuracy gate device:", device)

    check_st_var(n_train=512, n_test=256, m=64, device=device, tag="st_synth")
    check_mt_var_vs_dense_f64(
        n_train=256, n_test=256, m=64, t=2, device=device, tag="mt_dense_f64"
    )
    check_mt_var_vs_materialised(
        n_train=512, n_test=256, m=64, t=2, device=device, tag="mt_synth_eigen"
    )
    check_mt_var_vs_materialised(
        n_train=256,
        n_test=128,
        m=32,
        t=2,
        device=device,
        tag="mt_synth_chol",
        method="chol",
    )
    check_st_mean_mixed_vs_promote(
        n_train=512,
        n_test=256,
        m=64,
        device=device,
        tag="st_mean_mixed",
        abs_tol=2e-2,
        rel_tol=1e-2,
    )
    check_mt_mean_mixed_vs_promote(
        n_train=512,
        n_test=256,
        m=64,
        t=2,
        device=device,
        tag="mt_mean_mixed",
        abs_tol=2e-2,
        rel_tol=1e-2,
    )
    check_mt_var_mixed_vs_promote(
        n_train=512, n_test=256, m=64, t=2, device=device, tag="mt_var_mixed"
    )

    # TOA-shaped smoke: chunk >= 256, m = 2D for D=400.
    check_mt_var_vs_materialised(
        n_train=1024,
        n_test=512,
        m=800,
        t=2,
        device=device,
        tag="toa_smoke_eigen",
        abs_tol=1e-5,
        rel_tol=1e-4,
    )
    check_mt_mean_mixed_vs_promote(
        n_train=1024,
        n_test=512,
        m=800,
        t=2,
        device=device,
        tag="toa_smoke_mean",
        abs_tol=5e-2,
        rel_tol=2e-2,
    )
    check_mt_var_mixed_vs_promote(
        n_train=1024,
        n_test=512,
        m=800,
        t=2,
        device=device,
        tag="toa_smoke_var_mixed",
        abs_tol=1e-5,
        rel_tol=1e-4,
    )

    print("PREDICTIVE_ACCURACY_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
