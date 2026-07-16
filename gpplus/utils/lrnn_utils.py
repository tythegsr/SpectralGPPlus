"""Helpers for low-rank NN (Deep Basis) kernels and variance correction.

Aligned with Zhu, Yuchi, Xie (2025), arXiv:2505.18526 §3.2–3.3.

Dual Woodbury (``Λ = ΦᵀΦ + σ² I``) lives in :mod:`gpplus.utils.rff_utils` and is
re-exported here for LRNN call sites.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .rff_utils import (
    woodbury_factor_dual,
    woodbury_jitter_for_dtype,
    woodbury_marginal_log_likelihood_dual,
    woodbury_predict,
    woodbury_predictive_mean_dual,
    woodbury_predictive_obs_std,
    woodbury_predictive_var_diag_dual,
)

# Aliases kept for existing LRNN imports.
_lambda_cholesky = woodbury_factor_dual
woodbury_jitter_for_lrnn = woodbury_jitter_for_dtype


def kernel_diag_from_features(phi: Tensor) -> Tensor:
    """``k(x_i, x_i) = ||φ(x_i)||²`` for rows of ``phi`` ``(n, r)``."""
    return (phi * phi).sum(dim=-1)


def lrnn_trace_penalty(noise_var: Tensor, kernel_diag: Tensor) -> Tensor:
    """
    Trace regularization from DBK variance correction (paper eq. 11).

    ``tr(C_XX) / (2 σ²)`` with
    ``c_ii = max_j k(x_j, x_j) - k(x_i, x_i)``, so
    ``tr(C) = n * max(k_ii) - sum(k_ii)``.

    Subtract this from ``log p(y)`` when maximizing the corrected lower bound.
    """
    noise = noise_var.clamp_min(1e-12)
    k_max = kernel_diag.max()
    tr_c = kernel_diag.numel() * k_max - kernel_diag.sum()
    return tr_c / (2.0 * noise)


def lrnn_correction_diag(kernel_diag: Tensor, reference_max: Tensor | None = None) -> Tensor:
    """
    Diagonal correction ``c(x, x) = max_ref - k(x, x)``.

    If ``reference_max`` is None, use ``max(kernel_diag)`` (training-set max).
    """
    k_max = kernel_diag.max() if reference_max is None else reference_max
    return (k_max - kernel_diag).clamp_min(0.0)


def lrnn_corrected_noise(
    noise_var: Tensor,
    kernel_diag: Tensor,
    reference_max: Tensor | None = None,
) -> Tensor:
    """``σ̂²(x) = σ² + c(x, x)`` with ``c`` from :func:`lrnn_correction_diag`."""
    noise = noise_var.clamp_min(1e-12)
    return noise + lrnn_correction_diag(kernel_diag, reference_max=reference_max)


def _dbk_noise_ratios(
    noise_var: Tensor,
    kernel_diag: Tensor,
    reference_max: Tensor | None = None,
) -> Tensor:
    """
    Per-point noise ratios ``D_i = σ̂²(x_i) / σ²`` from paper eq. (12).

    Clamped below ``1`` so normalization never up-weights rows (which destabilizes
    ``Φ^T D^{-1} Φ`` when ``σ̂²`` is only slightly above ``σ²``).
    """
    noise = noise_var.clamp_min(1e-12)
    corrected = lrnn_corrected_noise(noise, kernel_diag, reference_max=reference_max)
    return (corrected / noise).clamp_min(1.0)


def _dbk_normalize_train(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    kernel_diag: Tensor,
    reference_max: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Paper eq. (12) training normalization:

    ``Φ̂_i = Φ_i / sqrt(D_i)``, ``ŷ_i = y_i / sqrt(D_i)`` with ``D_i = σ̂²(x_i)/σ²``.
    """
    d_ratio = _dbk_noise_ratios(noise_var, kernel_diag, reference_max=reference_max)
    inv_sqrt_d = d_ratio.rsqrt()
    z_hat = z_train * inv_sqrt_d.unsqueeze(-1)
    y_hat = y_centered * inv_sqrt_d
    return z_hat, y_hat, d_ratio


def woodbury_predict_heteroscedastic(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    kernel_diag_train: Tensor,
    kernel_diag_test: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Latent posterior mean/variance with DBK diagonal correction (paper eq. 12).

    Uses ``D^{-1/2}`` row normalization on training rows, then dual Woodbury
    with base noise ``σ²``. Test latent variance uses unscaled ``φ*``.
    """
    del kernel_diag_test  # reserved for future test-side normalization
    k_max_train = kernel_diag_train.max()
    z_hat, y_hat, _ = _dbk_normalize_train(
        noise_var, z_train, y_centered, kernel_diag_train, reference_max=k_max_train
    )
    noise = noise_var.clamp_min(1e-12)
    chol, noise_clamped, z_lin = woodbury_factor_dual(noise, z_hat, jitter=jitter)
    f_mean = woodbury_predictive_mean_dual(
        noise_clamped, z_lin, z_test, y_hat, chol=chol
    )
    f_var = woodbury_predictive_var_diag_dual(noise_clamped, z_test, chol=chol)
    return f_mean, f_var


def woodbury_marginal_log_likelihood_lrnn(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    *,
    variance_correction: bool = True,
) -> Tensor:
    """
    Homoscedastic Woodbury MLL on ``Σ = σ² I + Φ Φ^T``, optionally with DBK
    trace regularization subtracted from ``log p(y)``.

    Uses the dual ``Λ = Φ^T Φ + σ² I_r`` factorization for numerical stability.
    """
    mll = woodbury_marginal_log_likelihood_dual(
        noise_var, z_train, y_centered, jitter=jitter
    )
    if variance_correction:
        k_diag = kernel_diag_from_features(z_train)
        mll = mll - lrnn_trace_penalty(noise_var, k_diag)
    return mll


def woodbury_predict_lrnn(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    *,
    variance_correction: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Posterior mean, latent var, and observation std for LRNN / DBK.

    With ``variance_correction=True``, uses diagonal correction (paper §3.3)
    via the eq. (12) ``D^{-1/2}`` normalization at train time and
    ``σ̂²(x*)`` at test time. Without correction, falls back to homoscedastic Woodbury.
    """
    if not variance_correction:
        f_mean, f_var = woodbury_predict(
            noise_var, z_train, z_test, y_centered, jitter=jitter, woodbury_form="dual"
        )
        obs_std = woodbury_predictive_obs_std(f_var, noise_var)
        return f_mean, f_var, obs_std

    k_train = kernel_diag_from_features(z_train)
    k_test = kernel_diag_from_features(z_test)
    k_max_train = k_train.max()

    f_mean, f_var = woodbury_predict_heteroscedastic(
        noise_var,
        z_train,
        z_test,
        y_centered,
        k_train,
        k_test,
        jitter=jitter,
    )
    noise_test = lrnn_corrected_noise(noise_var, k_test, reference_max=k_max_train)
    obs_std = (f_var.clamp_min(0.0) + noise_test).sqrt()
    return f_mean, f_var, obs_std
