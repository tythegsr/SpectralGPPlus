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
from typing import Callable, Literal

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_WOODBURY_JITTER_DEFAULT = 1e-6


def woodbury_jitter_for_dtype(dtype: torch.dtype) -> float:
    """Default Woodbury diagonal jitter (linalg may run in float64 even for float32 inputs)."""
    del dtype  # same jitter in promoted precision
    return _WOODBURY_JITTER_DEFAULT


def _woodbury_linalg_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote float32 inputs to float64 for Woodbury Cholesky/solve stability."""
    return torch.float64 if dtype == torch.float32 else dtype


def _symmetrize_matrix(m: Tensor) -> Tensor:
    return 0.5 * (m + m.transpose(-1, -2))


def _woodbury_cholesky_factor(
    build_middle: Callable[[float], Tensor],
    jitter: float,
    *,
    max_attempts: int = 10,
    jitter_scale: float = 10.0,
) -> tuple[Tensor, float]:
    """
    Cholesky of a Woodbury middle matrix, rebuilding M(j) with escalating jitter.

    ``build_middle(j)`` must return M already including ``j * I`` on the diagonal.
    """
    base = float(jitter)
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        j = base * (jitter_scale**attempt)
        middle = _symmetrize_matrix(build_middle(j))
        try:
            chol = torch.linalg.cholesky(middle)
            if attempt > 0:
                logger.warning(
                    "Woodbury Cholesky recovered with jitter=%.2e (requested %.2e)",
                    j,
                    base,
                )
            return chol, j
        except torch.linalg.LinAlgError as exc:
            last_err = exc
    assert last_err is not None
    raise last_err

RffSampling = Literal["rff", "orf", "sorf"]
RFF_SAMPLING_MODES = frozenset({"rff", "orf", "sorf"})


def _validate_rff_sampling(rff_sampling: str) -> RffSampling:
    if rff_sampling not in RFF_SAMPLING_MODES:
        raise ValueError(f"rff_sampling must be one of {sorted(RFF_SAMPLING_MODES)}, got {rff_sampling!r}.")
    return rff_sampling  # type: ignore[return-value]


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht(x: Tensor, dim: int = -1) -> Tensor:
    """Normalized fast Walsh-Hadamard transform along ``dim`` (size must be a power of 2)."""
    dim = dim % x.dim()
    n = x.shape[dim]
    if n & (n - 1):
        raise ValueError(f"FWHT dimension must be a power of 2, got {n}.")
    out = x.movedim(dim, -1).clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = out[..., i : i + h]
            b = out[..., i + h : i + 2 * h]
            out[..., i : i + h] = a + b
            out[..., i + h : i + 2 * h] = a - b
        h *= 2
    out = out / math.sqrt(n)
    return out.movedim(-1, dim)


def _rademacher(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    signs = torch.randint(0, 2, (length,), device=device, dtype=torch.int8)
    return (2 * signs - 1).to(dtype)


def _sorf_apply_columns(
    d1: Tensor,
    d2: Tensor,
    d3: Tensor,
    num_cols: int,
    scale: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fwht_dim: int = 0,
) -> Tensor:
    """Apply sqrt(d) * H D1 H D2 H D3 to the first ``num_cols`` basis vectors (batched)."""
    d_pad = d1.shape[-1]
    x = torch.eye(d_pad, num_cols, device=device, dtype=dtype)
    if d1.dim() > 1:
        x = x.unsqueeze(0).expand(d1.shape[0], -1, -1)
        fwht_dim = 1
    x = x * d3.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim)
    x = x * d2.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim)
    x = x * d1.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim)
    return x * scale


def _sorf_block(
    num_dims: int,
    num_cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """SORF block: columns of sqrt(d) * H D1 H D2 H D3, truncated to num_dims."""
    d_pad = _next_pow2(num_dims)
    d1 = _rademacher(d_pad, device, dtype)
    d2 = _rademacher(d_pad, device, dtype)
    d3 = _rademacher(d_pad, device, dtype)
    scale = math.sqrt(num_dims)
    x = _sorf_apply_columns(
        d1, d2, d3, num_cols, scale, device=device, dtype=dtype, fwht_dim=0
    )
    return x[:num_dims]


def _sample_sorf_weights(
    num_dims: int,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Yu et al. SORF (arXiv:1610.09072 Eq. 5): W = sqrt(d) * H D1 H D2 H D3.

    Returns W of shape (num_dims, num_samples). When num_samples > num_dims,
    draws independent SORF blocks of size num_dims (same convention as ORF).
    """
    d = num_dims
    cols: list[Tensor] = []
    remaining = num_samples
    while remaining > 0:
        block = min(d, remaining)
        cols.append(_sorf_block(d, block, device=device, dtype=dtype))
        remaining -= block
    return torch.cat(cols, dim=1)[:, :num_samples]


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
    rff_sampling: RffSampling = "rff",
) -> Tensor:
    """
    Draw RBF random frequencies W with shape (num_dims, num_samples).

    ``rff_sampling``:
      - ``"rff"``: i.i.d. Gaussian columns.
      - ``"orf"``: full ORF (Yu et al. Eq. 2): QR orthogonal Q plus chi(d) scaling.
      - ``"sorf"``: structured ORF (Yu et al. Eq. 5): Walsh-Hadamard with Rademacher signs.

    When ``lengthscale`` is provided (GPPlus 10^(raw/2) per dimension), scales
    draws as omega_d ~ N(0, 1/lengthscale_d^2) after the base draw.
    """
    rff_sampling = _validate_rff_sampling(rff_sampling)
    dev = device or torch.device("cpu")
    dt = dtype or torch.float32
    if rff_sampling == "orf":
        w = _sample_orf_weights(num_dims, num_samples, device=dev, dtype=dt)
    elif rff_sampling == "sorf":
        w = _sample_sorf_weights(num_dims, num_samples, device=dev, dtype=dt)
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
    lin_dtype = _woodbury_linalg_dtype(z_train.dtype)
    noise = noise_var.clamp_min(1e-12).to(lin_dtype)
    z_c = z_train.to(lin_dtype)

    def build_middle(j: float) -> Tensor:
        return woodbury_middle_matrix(noise, z_c, jitter=j)

    chol, _ = _woodbury_cholesky_factor(build_middle, jitter)
    return chol, noise_var.clamp_min(1e-12)


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
    dtype = chol.dtype
    noise = noise.to(dtype)
    z_train = z_train.to(dtype)
    b = b.to(dtype)
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


def woodbury_predictive_obs_std(f_var: Tensor, noise_var: Tensor) -> Tensor:
    """sqrt(max(f_var, 0) + noise_var); observation noise always contributes."""
    return (f_var.clamp_min(0.0) + noise_var).sqrt()


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


# ---------------------------------------------------------------------------
# Multitask ICM Woodbury (Sigma = Lambda + Omega Omega^T, Omega = Phi kron R_B)
# ---------------------------------------------------------------------------

_warned_woodbury_mt_rank = False


def task_psd_factor(task_covar_matrix, jitter: float = 1e-8) -> Tensor:
    """PSD square root R with B = R R^T from GPyTorch task covar_matrix."""
    B = task_covar_matrix.to_dense()
    B = 0.5 * (B + B.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(B)
    evals = evals.clamp(min=jitter)
    R = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.transpose(-1, -2)
    return R.contiguous()


def flatten_multitask_targets(y: Tensor) -> Tensor:
    """GPyTorch vec order: task index fastest (row-major flatten of (n, T))."""
    if y.dim() == 1:
        return y
    return y.reshape(-1)


def unflatten_multitask_targets(y_flat: Tensor, num_tasks: int) -> Tensor:
    return y_flat.reshape(-1, num_tasks)


def build_icm_joint_features(phi: Tensor, task_psd: Tensor) -> Tensor:
    """
    Joint ICM features Omega = Phi kron R_B.

    Parameters
    ----------
    phi : (n, m) spatial RFF features
    task_psd : (T, T) PSD factor with B = task_psd @ task_psd.T
    """
    return torch.kron(phi.contiguous(), task_psd.contiguous())


def _multitask_noise_per_row(task_noises: Tensor, n: int) -> Tensor:
    """Observation noise for each of n*T vec entries (Lambda = I_n kron diag(task_noises))."""
    T = task_noises.shape[-1]
    return task_noises.view(1, T).expand(n, T).reshape(-1)


def _apply_lambda_inv_rows(task_noises: Tensor, n: int, x: Tensor) -> Tensor:
    """Multiply rows of x (n*T, ...) by 1/task_noises per task within each spatial block."""
    T = task_noises.shape[-1]
    inv = task_noises.clamp_min(1e-12).reciprocal()
    if x.dim() == 1:
        x = x.view(n, T)
        return (x * inv).reshape(-1)
    x = x.view(n, T, -1)
    return (x * inv.view(1, T, 1)).reshape(n * T, -1)


def _check_woodbury_mt_rank(omega: Tensor, n: int, num_tasks: int) -> None:
    global _warned_woodbury_mt_rank
    nT = n * num_tasks
    mT = omega.shape[-1]
    if mT >= nT and not _warned_woodbury_mt_rank:
        logger.warning(
            "Multitask Woodbury feature width m*T=%s >= n*T=%s; low-rank solve may not reduce cost.",
            mT,
            nT,
        )
        _warned_woodbury_mt_rank = True


def woodbury_middle_matrix_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    jitter: float = 0.0,
) -> Tensor:
    """M = I + Omega^T Lambda^{-1} Omega with Lambda = I_n kron diag(task_noises)."""
    T = task_noises.shape[-1]
    mT = omega.shape[-1]
    inv_noise = task_noises.clamp_min(1e-12).reciprocal()
    omega_scaled = omega.view(n, T, mT) * inv_noise.view(1, T, 1)
    omega_scaled = omega_scaled.reshape(n * T, mT)
    middle = torch.matmul(omega.transpose(-1, -2), omega_scaled)
    eye = torch.eye(mT, device=omega.device, dtype=omega.dtype)
    middle = eye + middle
    if jitter > 0:
        middle = middle + jitter * eye
    return middle


def woodbury_factor_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Cholesky of multitask Woodbury middle matrix M."""
    _check_woodbury_mt_rank(omega, n, task_noises.shape[-1])
    lin_dtype = _woodbury_linalg_dtype(omega.dtype)
    noise = task_noises.clamp_min(1e-12).to(lin_dtype)
    omega_c = omega.to(lin_dtype)

    def build_middle(j: float) -> Tensor:
        return woodbury_middle_matrix_mt(noise, omega_c, n, jitter=j)

    chol, _ = _woodbury_cholesky_factor(build_middle, jitter)
    return chol, task_noises.clamp_min(1e-12)


def woodbury_solve_mt_from_chol(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    chol: Tensor,
    b: Tensor,
) -> Tensor:
    """Sigma^{-1} b for Sigma = Lambda + Omega Omega^T."""
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    dtype = chol.dtype
    task_noises = task_noises.to(dtype)
    omega = omega.to(dtype)
    b = b.to(dtype)
    lam_inv_b = _apply_lambda_inv_rows(task_noises, n, b)
    middle_rhs = omega.transpose(-1, -2) @ lam_inv_b
    inner = torch.cholesky_solve(middle_rhs, chol)
    lam_inv_omega = _apply_lambda_inv_rows(task_noises, n, omega)
    correction = lam_inv_omega @ inner
    out = lam_inv_b - correction
    return out.squeeze(-1) if squeeze else out


def woodbury_solve_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    b: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    if chol is None or noise is None:
        chol, noise = woodbury_factor_mt(task_noises, omega, n, jitter=jitter)
    return woodbury_solve_mt_from_chol(noise, omega, n, chol, b)


def woodbury_log_det_mt_from_chol(
    task_noises: Tensor,
    n: int,
    chol: Tensor,
) -> Tensor:
    T = task_noises.shape[-1]
    log_det_lam = n * task_noises.clamp_min(1e-12).log().sum()
    log_det_middle = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum()
    return log_det_lam + log_det_middle


def woodbury_log_det_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    if chol is None or noise is None:
        chol, noise = woodbury_factor_mt(task_noises, omega, n, jitter=jitter)
    return woodbury_log_det_mt_from_chol(noise, n, chol)


def woodbury_marginal_log_likelihood_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """Gaussian log-density for vec(y) ~ N(0, Lambda + Omega Omega^T)."""
    nT = y_centered.shape[-1]
    chol, noise = woodbury_factor_mt(task_noises, omega, n, jitter=jitter)
    y_c = y_centered.to(chol.dtype)
    alpha = woodbury_solve_mt_from_chol(noise, omega, n, chol, y_c)
    quad = (y_c * alpha).sum()
    log_det = woodbury_log_det_mt_from_chol(noise, n, chol)
    const = -0.5 * nT * math.log(2.0 * math.pi)
    return const - 0.5 * quad - 0.5 * log_det


def woodbury_predictive_mean_mt(
    task_noises: Tensor,
    omega_train: Tensor,
    omega_test: Tensor,
    n_train: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """Latent posterior mean vec; reshape caller to (n_test, T)."""
    lin_dtype = chol.dtype if chol is not None else _woodbury_linalg_dtype(omega_train.dtype)
    omega_train = omega_train.to(lin_dtype)
    omega_test = omega_test.to(lin_dtype)
    alpha = woodbury_solve_mt(
        task_noises,
        omega_train,
        n_train,
        y_centered,
        jitter=jitter,
        chol=chol,
        noise=noise,
    )
    return omega_test @ (omega_train.transpose(-1, -2) @ alpha)


def woodbury_predictive_var_diag_mt(
    task_noises: Tensor,
    omega_train: Tensor,
    omega_test: Tensor,
    n_train: int,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """Diagonal latent posterior variance for each vec entry (length n_test*T)."""
    if chol is None or noise is None:
        chol, noise = woodbury_factor_mt(task_noises, omega_train, n_train, jitter=jitter)
    lin_dtype = chol.dtype
    omega_train = omega_train.to(lin_dtype)
    omega_test = omega_test.to(lin_dtype)
    noise = noise.to(lin_dtype)
    prior_var = (omega_test * omega_test).sum(dim=-1)
    cross = omega_test @ omega_train.transpose(-1, -2)
    alpha = woodbury_solve_mt_from_chol(noise, omega_train, n_train, chol, cross.transpose(-1, -2))
    explained = (cross * alpha.transpose(-1, -2)).sum(dim=-1)
    return prior_var - explained.clamp_min(0.0)


def woodbury_predict_mt(
    task_noises: Tensor,
    omega_train: Tensor,
    omega_test: Tensor,
    n_train: int,
    num_tasks: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Latent posterior mean (n_test, T) and diagonal variance (n_test, T).
    """
    chol, noise = woodbury_factor_mt(task_noises, omega_train, n_train, jitter=jitter)
    mean_flat = woodbury_predictive_mean_mt(
        task_noises,
        omega_train,
        omega_test,
        n_train,
        y_centered,
        jitter=jitter,
        chol=chol,
        noise=noise,
    )
    var_flat = woodbury_predictive_var_diag_mt(
        task_noises,
        omega_train,
        omega_test,
        n_train,
        jitter=jitter,
        chol=chol,
        noise=noise,
    )
    n_test = omega_test.shape[0] // num_tasks
    return (
        unflatten_multitask_targets(mean_flat, num_tasks),
        unflatten_multitask_targets(var_flat, num_tasks),
    )
