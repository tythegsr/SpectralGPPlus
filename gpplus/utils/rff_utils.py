"""Random Fourier Feature helpers and Woodbury linear algebra for RFF-GP.

Woodbury marginal covariance (centered observations, homoskedastic noise ``noise_var``):

    Sigma = noise_var * I_n + Phi Phi^T,

with **Phi = z_train** of shape ``(n, m)`` (one row per training point, ``m = 2 * num_rff``).

The inverse is never formed as an ``n x n`` matrix. With ``M = I_m + Phi^T Phi / noise_var``,

    Sigma^{-1} = (noise_var I)^{-1}
                 - (noise_var I)^{-1} Phi M^{-1} Phi^T (noise_var I)^{-1}.

Equivalently, if ``Z = Phi^T`` is ``(m, n)`` (features x points),

    Sigma^{-1} = (noise_var I)^{-1}
                 - (noise_var I)^{-1} Z^T (I + Z (noise_var I)^{-1} Z^T)^{-1} Z (noise_var I)^{-1}.

Log-determinant (matrix determinant lemma):

    log|Sigma| = n log(noise_var) + log|M|.

See ``docs/overleaf/rff_woodbury_derivation.tex`` for a full derivation.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _sample_orf_weights(
    num_dims: int,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Yu et al. full ORF (arXiv:1610.09072 Eq. 2): W = S @ Q with chi(d) column scales.

    Returns W of shape (num_dims, num_samples); column j is w_j = s_j * q_j.
    When num_samples > num_dims, draws independent ORF blocks of size num_dims.
    """
    d = num_dims
    cols: list[Tensor] = []
    remaining = num_samples
    while remaining > 0:
        block = min(d, remaining)
        g = torch.randn(d, d, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(g)
        # chi(d): norm of a d-dimensional standard Gaussian vector
        s = torch.randn(block, d, device=device, dtype=dtype).norm(dim=1)
        cols.append(q[:, :block] * s.unsqueeze(0))
        remaining -= block
    return torch.cat(cols, dim=1)[:, :num_samples]


def init_rbf_weights(
    num_dims: int,
    num_samples: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    lengthscale: Tensor | None = None,
    orthogonal: bool = False,
) -> Tensor:
    """
    Draw RBF random frequencies W with shape (num_dims, num_samples).

    When ``orthogonal=True``, uses full ORF (Yu et al. 1610.09072): random
    orthogonal Q from QR plus per-frequency chi(d) scaling (S @ Q).

    When ``lengthscale`` is provided (GPPlus 10^(raw/2) per dimension), scales
    draws as omega_d ~ N(0, 1/lengthscale_d^2) after the base draw (i.i.d. or ORF).
    """
    dev = device or torch.device("cpu")
    dt = dtype or torch.float32
    if orthogonal:
        w = _sample_orf_weights(num_dims, num_samples, device=dev, dtype=dt)
    else:
        w = torch.randn(num_dims, num_samples, device=dev, dtype=dt)
    if lengthscale is not None:
        inv_ls = 1.0 / lengthscale.clamp_min(1e-12)
        w = w * inv_ls.unsqueeze(-1)
    return w


def featurize_rbf(
    x: Tensor,
    randn_weights: Tensor,
    lengthscale: Tensor,
    num_samples: int | None = None,
) -> Tensor:
    """
    Map inputs to RFF features Z of shape (..., n, 2D).

    Uses GPPlus/GaussianKernel-style input scaling 10^(lengthscale/2), then x @ W.
    """
    D = num_samples if num_samples is not None else randn_weights.shape[-1]
    scale = torch.pow(10.0, lengthscale / 2.0)
    proj = x.mul(scale).matmul(randn_weights)
    scale = 1.0 / math.sqrt(D)
    return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


_warned_woodbury_rank = False


def _check_woodbury_rank(z_train: Tensor) -> None:
    global _warned_woodbury_rank
    n, m = z_train.shape[-2], z_train.shape[-1]
    if m >= n and not _warned_woodbury_rank:
        logger.warning(
            "Woodbury feature dimension m=%s >= n=%s; low-rank solve may not reduce cost. "
            "Use num_rff < n_train/2 for benefit.",
            m,
            n,
        )
        _warned_woodbury_rank = True


def woodbury_middle_matrix(
    noise_var: Tensor,
    z_train: Tensor,
    jitter: float = 0.0,
) -> Tensor:
    """
    Woodbury middle matrix ``M = I_m + Phi^T (noise_var I)^{-1} Phi`` with ``Phi = z_train``.

    Parameters
    ----------
    z_train : (n, m) feature matrix Phi (rows = training points).
    """
    noise = noise_var.clamp_min(1e-12)
    m = z_train.shape[-1]
    ztz = torch.matmul(z_train.transpose(-1, -2), z_train)
    eye = torch.eye(m, device=z_train.device, dtype=z_train.dtype)
    middle = eye + ztz / noise
    if jitter > 0:
        middle = middle + jitter * eye
    return middle


def woodbury_factor(
    noise_var: Tensor,
    z_train: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Cholesky factor ``L`` with ``L L^T = M`` for the Woodbury middle matrix ``M``.

    ``M = I_m + Phi^T Phi / noise_var`` where ``Phi = z_train`` is ``(n, m)``.
    This is the ``m x m`` matrix in ``Sigma^{-1}`` for
    ``Sigma = noise_var I_n + Phi Phi^T`` (not an ``n x n`` factorization of ``Sigma``).

    Parameters
    ----------
    z_train : (n, m) scaled feature matrix Phi.

    Returns
    -------
    chol : (m, m) lower-triangular Cholesky factor of ``M``.
    noise : clamped noise variance scalar tensor.
    """
    _check_woodbury_rank(z_train)
    noise = noise_var.clamp_min(1e-12)
    middle = woodbury_middle_matrix(noise, z_train, jitter=jitter)
    chol = torch.linalg.cholesky(middle)
    return chol, noise


def woodbury_solve_from_chol(
    noise: Tensor,
    z_train: Tensor,
    chol: Tensor,
    b: Tensor,
) -> Tensor:
    """
    Compute ``Sigma^{-1} b`` via Woodbury (``Phi = z_train`` is ``(n, m)``).

    Implements

        inv_noise_b = (noise I)^{-1} b
        inner = M^{-1} Phi^T inv_noise_b
        Sigma^{-1} b = inv_noise_b - (noise I)^{-1} Phi inner

    with ``M = I + Phi^T Phi / noise`` factored as ``chol``.

    Parameters
    ----------
    z_train : (n, m) feature matrix Phi.
    b : (n,) or (n, k)
    """
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    inv_noise_b = b / noise
    middle_rhs = z_train.transpose(-1, -2) @ inv_noise_b
    inner = torch.cholesky_solve(middle_rhs, chol)
    correction = z_train @ inner / noise
    out = inv_noise_b - correction
    return out.squeeze(-1) if squeeze else out


def woodbury_solve(
    noise_var: Tensor,
    z_train: Tensor,
    b: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """
    Compute ``Sigma^{-1} b`` for ``Sigma = noise_var I_n + Phi Phi^T``, ``Phi = z_train``.

    Uses an ``m x m`` Cholesky of ``M = I + Phi^T Phi / noise_var``, not ``n x n`` Cholesky of ``Sigma``.

    Parameters
    ----------
    z_train : (n, m) feature matrix Phi.
    b : (n,) or (n, k)
    """
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    return woodbury_solve_from_chol(noise, z_train, chol, b)


def woodbury_log_det_from_chol(
    noise: Tensor,
    z_train: Tensor,
    chol: Tensor,
) -> Tensor:
    """
    ``log|Sigma|`` for ``Sigma = noise I_n + Phi Phi^T`` via ``log|Sigma| = n log(noise) + log|M|``.

    ``chol`` is the Cholesky factor of ``M = I + Phi^T Phi / noise``.
    """
    n = z_train.shape[-2]
    log_det_middle = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum()
    return n * noise.log() + log_det_middle


def woodbury_log_det(
    noise_var: Tensor,
    z_train: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """``log|Sigma|`` for ``Sigma = noise_var I_n + Phi Phi^T`` (Woodbury determinant lemma)."""
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    return woodbury_log_det_from_chol(noise, z_train, chol)


def woodbury_quadratic_form(
    noise_var: Tensor,
    z_train: Tensor,
    y: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """y^T Sigma^{-1} y (scalar)."""
    alpha = woodbury_solve(
        noise_var, z_train, y, jitter=jitter, chol=chol, noise=noise
    )
    return (y * alpha).sum()


def woodbury_marginal_log_likelihood(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Gaussian marginal log-density for ``y_centered ~ N(0, Sigma)``.

    ``Sigma = noise_var I_n + Phi Phi^T`` with ``Phi = z_train`` ``(n, m)``.
    Evaluates ``-1/2 y^T Sigma^{-1} y - 1/2 log|Sigma|`` using Woodbury on ``M`` only
    (never materializes ``n x n`` ``Sigma``). ``y_centered`` must already subtract the GP mean.
    One ``m x m`` Cholesky factorization per call.
    """
    n = y_centered.shape[-1]
    chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    quad = woodbury_quadratic_form(
        noise_var, z_train, y_centered, jitter=jitter, chol=chol, noise=noise
    )
    log_det = woodbury_log_det_from_chol(noise, z_train, chol)
    const = -0.5 * n * math.log(2.0 * math.pi)
    return const - 0.5 * quad - 0.5 * log_det


def woodbury_predictive_mean(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """Posterior mean of latent f at test points: Z_test Z_train^T Sigma^{-1} y."""
    alpha = woodbury_solve(
        noise_var, z_train, y_centered, jitter=jitter, chol=chol, noise=noise
    )
    return z_test @ (z_train.transpose(-1, -2) @ alpha)


def woodbury_predictive_var_diag(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """
    Diagonal posterior variance of latent f at test points.

    Var(f_i) = ||z_i||^2 - k_{i,n} Sigma^{-1} k_{n,i}^T.
    """
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    prior_var = (z_test * z_test).sum(dim=-1)
    cross = z_test @ z_train.transpose(-1, -2)
    alpha = woodbury_solve_from_chol(noise, z_train, chol, cross.transpose(-1, -2))
    explained = (cross * alpha.transpose(-1, -2)).sum(dim=-1)
    return prior_var - explained.clamp_min(0.0)


def woodbury_predict(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Latent posterior mean and diagonal variance at test points (single factorization).

    Returns
    -------
    f_mean, f_var : tensors of shape (n_test,).
    """
    chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    f_mean = woodbury_predictive_mean(
        noise_var, z_train, z_test, y_centered, jitter=jitter, chol=chol, noise=noise
    )
    f_var = woodbury_predictive_var_diag(
        noise_var, z_train, z_test, jitter=jitter, chol=chol, noise=noise
    )
    return f_mean, f_var
