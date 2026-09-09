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
from typing import Callable, Literal, NamedTuple, Union

import torch
from torch import Tensor

from .line_profile import profile

logger = logging.getLogger(__name__)

_WOODBURY_JITTER_DEFAULT = 1e-6
WoodburyForm = Literal["primal", "dual"]
WoodburyMtMethod = Literal["chol", "eigen", "dual_eigen", "primal_eigen"]


class WoodburyMtCholFactor(NamedTuple):
    """Dense Cholesky factor of ``M = I + (Phi^T Phi) kron (R_B^T D^{-1} R_B)``."""

    chol: Tensor
    kind: Literal["chol"] = "chol"


class WoodburyMtEigFactor(NamedTuple):
    """
    Product-eigenbasis factor for multitask Woodbury (Tier-3).

    ``form="primal"``: ``M = I + G ⊗ S`` with ``S = R_Bᵀ D⁻¹ R_B``;
    eigenvalues ``1 + jitter + λ_g[i] μ_s[j]``.

    ``form="dual"``: ``Λ = G ⊗ B + additive`` with ``B = R_Bᵀ R_B`` and
    ``additive[j] = task_noises[j]`` (exact when all task noises equal ``σ²``:
    ``Λ = G ⊗ B + σ² I``); eigenvalues ``jitter + λ_g[i] μ_b[j] + noise_add[j]``.
    """

    q_g: Tensor
    evals_g: Tensor
    q_s: Tensor
    evals_s: Tensor
    jitter: float
    form: Literal["primal", "dual"] = "primal"
    noise_add: Tensor | None = None  # (T,) dual additive term; None for primal
    kind: Literal["eigen"] = "eigen"


WoodburyMtFactor = Union[WoodburyMtCholFactor, WoodburyMtEigFactor]


def woodbury_jitter_for_dtype(dtype: torch.dtype) -> float:
    """Default Woodbury diagonal jitter (linalg may run in float64 even for float32 inputs)."""
    del dtype  # same jitter in promoted precision
    return _WOODBURY_JITTER_DEFAULT


def _woodbury_linalg_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype for small Woodbury factors (``G``, ``S``, ``M``).

    Float32 feature maps promote these small matrices to float64. The eigen MT
    path forms ``G = Phi^T Phi`` in ``Phi``'s dtype first (mixed precision).

    Opt-in override: ``GPPLUS_WOODBURY_LINALG=float32`` forces float32 factors
    even for float64 features (speed/accuracy trade; off by default).
    """
    try:
        from .woodbury_mll_autograd import use_float32_woodbury_linalg

        if use_float32_woodbury_linalg():
            return torch.float32
    except Exception:
        pass
    return torch.float64 if dtype == torch.float32 else dtype


def _symmetrize_matrix(m: Tensor) -> Tensor:
    return 0.5 * (m + m.transpose(-1, -2))


def _woodbury_psd_project_cholesky(middle: Tensor, eig_floor: float) -> Tensor:
    """Symmetrize, clamp eigenvalues, and return a true Cholesky factor."""
    middle = _symmetrize_matrix(middle)
    evals, evecs = torch.linalg.eigh(middle)
    evals = evals.clamp_min(max(float(eig_floor), 0.0))
    middle_psd = _symmetrize_matrix((evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2))
    return torch.linalg.cholesky(middle_psd)


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

    Recovery order:
    1. Cholesky at the requested jitter.
    2. Eigenvalue PSD projection at the **same** jitter, then a true Cholesky
       (fixes float32 Gram roundoff without inflating the noise scale).
    3. Escalate jitter and retry Cholesky / PSD projection.
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

        # Prefer PSD projection at this jitter before escalating further.
        # Important: return a real Cholesky factor — ``cholesky_solve`` rejects
        # dense ``Q diag(sqrt(λ))`` factors (they satisfy LLᵀ=A but are not
        # triangular Cholesky factors).
        try:
            chol = _woodbury_psd_project_cholesky(middle, eig_floor=max(j, 1e-12))
            if attempt == 0:
                logger.warning(
                    "Woodbury Cholesky used PSD eigenvalue projection at jitter=%.2e "
                    "(float32 Gram roundoff; requested %.2e).",
                    j,
                    base,
                )
            else:
                logger.warning(
                    "Woodbury Cholesky recovered via PSD projection with jitter=%.2e "
                    "(requested %.2e).",
                    j,
                    base,
                )
            return chol, j
        except torch.linalg.LinAlgError as exc:
            last_err = exc

    raise RuntimeError(
        f"Woodbury Cholesky failed after {max_attempts} jitter/PSD attempts "
        f"(requested jitter={base:.2e}). Last error: {last_err}"
    )


def _eigh_psd(m: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetrize, eigendecompose, and clamp eigenvalues to be non-negative."""
    evals, evecs = torch.linalg.eigh(_symmetrize_matrix(m))
    return evals.clamp_min(0.0), evecs


def _m_inv_diag_eigen(factor: WoodburyMtEigFactor) -> Tensor:
    """``(m, T)`` diagonal of the middle-matrix inverse in the product eigenbasis."""
    geom = factor.evals_g.unsqueeze(-1) * factor.evals_s.unsqueeze(0)
    if factor.form == "dual":
        noise_add = factor.noise_add
        if noise_add is None:
            raise ValueError("dual WoodburyMtEigFactor requires noise_add.")
        return (float(factor.jitter) + geom + noise_add.unsqueeze(0)).reciprocal()
    return (1.0 + float(factor.jitter) + geom).reciprocal()


def apply_middle_inverse_eigen(factor: WoodburyMtEigFactor, rhs: Tensor) -> Tensor:
    """
    Apply middle inverse in the product eigenbasis without forming a dense matrix.

    Primal: ``M^{-1}``. Dual: ``Λ^{-1}`` with ``Λ = G ⊗ B + diag(noise_add)``.

    ``rhs`` is ``(m*T,)`` or ``(m*T, K)`` with the same Kronecker layout as
    ``torch.kron(G, S)`` (feature index slowest within each block row = reshape ``(m, T)``).
    """
    squeeze = rhs.dim() == 1
    if squeeze:
        rhs = rhs.unsqueeze(-1)
    m = factor.q_g.shape[-1]
    t = factor.q_s.shape[-1]
    k = rhs.shape[-1]
    x = rhs.reshape(m, t, k)
    # Y = Q_g^T X Q_s   (Q from eigh: A = Q diag(Λ) Q^T)
    y = torch.einsum("ia,ijk,jb->abk", factor.q_g, x, factor.q_s)
    y = y * _m_inv_diag_eigen(factor).unsqueeze(-1)
    # Z = Q_g Y Q_s^T
    z = torch.einsum("ia,abk,jb->ijk", factor.q_g, y, factor.q_s)
    out = z.reshape(m * t, k)
    return out.squeeze(-1) if squeeze else out


def _woodbury_eigen_factor_from_grams(
    g: Tensor,
    s: Tensor,
    jitter: float,
    *,
    form: Literal["primal", "dual"] = "primal",
    noise_add: Tensor | None = None,
) -> WoodburyMtEigFactor:
    """Build Tier-3 factor from spatial Gram ``G`` and task Gram ``S``/``B``.

    Eigenvalues are clamped non-negative in :func:`_eigh_psd`.
    """
    evals_g, q_g = _eigh_psd(g)
    evals_s, q_s = _eigh_psd(s)
    if form == "dual":
        if noise_add is None:
            raise ValueError("dual eigen factor requires noise_add.")
        noise_add = noise_add.to(dtype=evals_g.dtype, device=evals_g.device).reshape(-1)
        if noise_add.numel() != evals_s.numel():
            raise ValueError(
                f"noise_add length {noise_add.numel()} != task eigen dim {evals_s.numel()}"
            )
    return WoodburyMtEigFactor(
        q_g=q_g,
        evals_g=evals_g,
        q_s=q_s,
        evals_s=evals_s,
        jitter=float(jitter),
        form=form,
        noise_add=noise_add,
    )


RffSampling = Literal["rff", "orf", "sorf"]
RFF_SAMPLING_MODES = frozenset({"rff", "orf", "sorf"})

SpectralKernel = Literal["rbf", "matern32"]
SPECTRAL_KERNEL_MODES = frozenset({"rbf", "matern32"})
# Matérn-ν spectral measure is multivariate Student-t with df = 2ν; ν=3/2 → df=3.
_MATERN32_DF = 3.0


def _validate_rff_sampling(rff_sampling: str) -> RffSampling:
    if rff_sampling not in RFF_SAMPLING_MODES:
        raise ValueError(f"rff_sampling must be one of {sorted(RFF_SAMPLING_MODES)}, got {rff_sampling!r}.")
    return rff_sampling  # type: ignore[return-value]


def _validate_spectral_kernel(spectral_kernel: str) -> SpectralKernel:
    if spectral_kernel not in SPECTRAL_KERNEL_MODES:
        raise ValueError(
            f"spectral_kernel must be one of {sorted(SPECTRAL_KERNEL_MODES)}, got {spectral_kernel!r}."
        )
    return spectral_kernel  # type: ignore[return-value]


def _matern32_scale_mixture(w: Tensor) -> Tensor:
    """
    Convert Gaussian frequency columns to Matérn-3/2 (multivariate Student-t, df=3).

    Each column ``ω ∈ R^d`` shares one radial scale ``u ~ Chi²(3)``:
    ``ω ← ω * sqrt(3 / u)``. Coordinates are not scaled independently.
    """
    num_samples = w.shape[-1]
    # Chi²(df) as sum of df squared standard normals (one draw per column).
    u = torch.randn(
        int(_MATERN32_DF),
        num_samples,
        device=w.device,
        dtype=w.dtype,
    ).pow(2).sum(dim=0)
    scale = (_MATERN32_DF / u.clamp_min(1e-12)).sqrt()
    return w * scale.unsqueeze(0)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht(x: Tensor, dim: int = -1, *, correct: bool = False) -> Tensor:
    """Normalized fast Walsh-Hadamard transform along ``dim`` (size must be a power of 2).

    ``correct=False`` (default): legacy in-place butterfly that aliases ``a``/``b`` views.
    This matches historical TOA SORF checkpoints (column norms ~0.56 at d=270).

    ``correct=True``: snapshot ``a``/``b`` before writing (true FWHT; Yu-scale norms).
    """
    dim = dim % x.dim()
    n = x.shape[dim]
    if n & (n - 1):
        raise ValueError(f"FWHT dimension must be a power of 2, got {n}.")
    out = x.movedim(dim, -1).clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            if correct:
                a = out[..., i : i + h].clone()
                b = out[..., i + h : i + 2 * h].clone()
            else:
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
    correct_sorf: bool = False,
) -> Tensor:
    """Apply sqrt(d) * H D1 H D2 H D3 to the first ``num_cols`` basis vectors (batched)."""
    d_pad = d1.shape[-1]
    x = torch.eye(d_pad, num_cols, device=device, dtype=dtype)
    if d1.dim() > 1:
        x = x.unsqueeze(0).expand(d1.shape[0], -1, -1)
        fwht_dim = 1
    x = x * d3.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim, correct=correct_sorf)
    x = x * d2.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim, correct=correct_sorf)
    x = x * d1.unsqueeze(-1)
    x = _fwht(x, dim=fwht_dim, correct=correct_sorf)
    return x * scale


def _sorf_block(
    num_dims: int,
    num_cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    correct_sorf: bool = False,
) -> Tensor:
    """SORF block: columns of sqrt(d) * H D1 H D2 H D3, truncated to num_dims."""
    d_pad = _next_pow2(num_dims)
    d1 = _rademacher(d_pad, device, dtype)
    d2 = _rademacher(d_pad, device, dtype)
    d3 = _rademacher(d_pad, device, dtype)
    # Scale by sqrt(d), not sqrt(d_pad): matches Yu Eq. 5 and historical TOA checkpoints.
    scale = math.sqrt(num_dims)
    x = _sorf_apply_columns(
        d1,
        d2,
        d3,
        num_cols,
        scale,
        device=device,
        dtype=dtype,
        fwht_dim=0,
        correct_sorf=correct_sorf,
    )
    return x[:num_dims]


def _sample_sorf_weights(
    num_dims: int,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    correct_sorf: bool = False,
) -> Tensor:
    """
    Yu et al. SORF (arXiv:1610.09072 Eq. 5): W = sqrt(d) * H D1 H D2 H D3.

    Returns W of shape (num_dims, num_samples). When num_samples > num_dims,
    draws independent SORF blocks of size num_dims (same convention as ORF).

    ``correct_sorf=False`` uses the legacy aliased FWHT (TOA-compatible).
    ``correct_sorf=True`` uses a true FWHT (textbook Yu scale).
    """
    d = num_dims
    cols: list[Tensor] = []
    remaining = num_samples
    while remaining > 0:
        block = min(d, remaining)
        cols.append(
            _sorf_block(d, block, device=device, dtype=dtype, correct_sorf=correct_sorf)
        )
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
    correct_sorf: bool = False,
    spectral_kernel: SpectralKernel = "rbf",
) -> Tensor:
    """
    Draw random frequencies W with shape (num_dims, num_samples).

    ``rff_sampling``:
      - ``"rff"``: i.i.d. Gaussian columns.
      - ``"orf"``: full ORF (Yu et al. Eq. 2): QR orthogonal Q plus chi(d) scaling.
      - ``"sorf"``: structured ORF (Yu et al. Eq. 5): Walsh-Hadamard with Rademacher signs.

    ``spectral_kernel``:
      - ``"rbf"``: leave Gaussian (or ORF/SORF) draws as-is (default).
      - ``"matern32"``: apply a per-column Chi²(3) scale mixture (Student-t, df=3).
        Works with ``rff`` / ``orf`` / ``sorf``; ORF/SORF keep their structured Gaussian
        part and only the radial law is fattened toward Matérn-3/2.

    ``correct_sorf`` only affects ``"sorf"``: True = true FWHT; False = legacy aliased FWHT.

    When ``lengthscale`` is provided (GPPlus 10^(raw/2) per dimension), scales
    draws as omega_d ~ N(0, 1/lengthscale_d^2) after the base draw.
    """
    rff_sampling = _validate_rff_sampling(rff_sampling)
    spectral_kernel = _validate_spectral_kernel(spectral_kernel)
    dev = device or torch.device("cpu")
    dt = dtype or torch.float32
    if rff_sampling == "orf":
        w = _sample_orf_weights(num_dims, num_samples, device=dev, dtype=dt)
    elif rff_sampling == "sorf":
        w = _sample_sorf_weights(
            num_dims,
            num_samples,
            device=dev,
            dtype=dt,
            correct_sorf=correct_sorf,
        )
    else:
        w = torch.randn(num_dims, num_samples, device=dev, dtype=dt)
    if spectral_kernel == "matern32":
        w = _matern32_scale_mixture(w)
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
    Lengthscales multiply columns of ``W`` (shape ``(d, D)`` or ``(..., d, D)``)
    rather than rows of ``x`` — algebraically identical, cheaper for large ``n``.

    Supports a leading init-batch on ``lengthscale`` ``(..., 1, d)`` and optional
    batched ``randn_weights`` ``(..., d, D)`` with unbatched ``x`` ``(n, d)``.

    Cos/sin are written into one ``(..., n, 2D)`` buffer (no ``cat`` of temporaries).
    """
    D = num_samples if num_samples is not None else randn_weights.shape[-1]
    # lengthscale: (1, d) | (d,) | (B, 1, d) -> scale with trailing (d, 1) for W columns
    scale = torch.pow(10.0, lengthscale / 2.0)
    if scale.dim() >= 2 and scale.shape[-2] == 1:
        scale = scale.transpose(-1, -2)  # (..., d, 1)
    else:
        scale = scale.reshape(-1, 1)  # unbatched ARD / scalar fallback
    w = randn_weights * scale
    if x.dim() == 2 and w.dim() > 2:
        proj = torch.einsum("nd,...dD->...nD", x, w)
    else:
        proj = x.matmul(w)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    out = proj.new_empty(proj.shape[:-1] + (2 * D,))
    # Write into one (..., 2D) buffer (no cat); scale on the slices for autograd safety.
    out[..., :D] = torch.cos(proj).mul(inv_sqrt_d)
    out[..., D:] = torch.sin(proj).mul(inv_sqrt_d)
    return out


def _expand_noise_for_square(noise: Tensor, square: Tensor) -> Tensor:
    """Broadcast noise ``()`` / ``(B,)`` against square mats ``(..., m, m)``."""
    while noise.dim() < square.dim():
        noise = noise.unsqueeze(-1)
    return noise


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
    *,
    promote_features: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Cholesky factor ``L`` with ``L L^T = M`` for the Woodbury middle matrix ``M``.

    ``M = I_m + Phi^T Phi / noise_var`` where ``Phi = z_train`` is ``(n, m)``.
    This is the ``m x m`` matrix in ``Sigma^{-1}`` for
    ``Sigma = noise_var I_n + Phi Phi^T`` (not an ``n x n`` factorization of ``Sigma``).

    Parameters
    ----------
    z_train : (n, m) scaled feature matrix Phi.
    promote_features :
        If True, cast full ``Phi`` to the factor dtype before ``Phi^T Phi`` (legacy).
        Default False: form Gram in ``Phi``'s dtype, then promote the small ``m x m``
        matrix for Cholesky (mixed precision; matches multitask Woodbury).

    Returns
    -------
    chol : (m, m) lower-triangular Cholesky factor of ``M``.
    noise : clamped noise variance scalar tensor.
    """
    _check_woodbury_rank(z_train)
    lin_dtype = _woodbury_linalg_dtype(z_train.dtype)
    noise = noise_var.clamp_min(1e-12).to(lin_dtype)

    if promote_features or z_train.dtype == lin_dtype:
        z_for_g = z_train.to(lin_dtype)
        ztz = torch.matmul(z_for_g.transpose(-1, -2), z_for_g)
    else:
        ztz = torch.matmul(z_train.transpose(-1, -2), z_train).to(lin_dtype)

    def build_middle(j: float) -> Tensor:
        m = ztz.shape[-1]
        eye = torch.eye(m, device=ztz.device, dtype=ztz.dtype)
        noise_b = _expand_noise_for_square(noise, ztz)
        middle = eye + ztz / noise_b
        if j > 0:
            middle = middle + j * eye
        return middle

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

    Mixed precision: Phi matvecs stay in ``z_train.dtype``; only the small
    ``m``-dimensional middle solve uses ``chol.dtype`` (float64 when features
    are float32). Returns in the factor dtype for stable quadratic forms.

    Parameters
    ----------
    z_train : (n, m) feature matrix Phi.
    b : (n,) or (n, k)
    """
    # b: (n,) | (n, k) | (B, n) | (B, n, k)
    squeeze = b.dim() == z_train.dim() - 1
    if squeeze:
        b = b.unsqueeze(-1)
    factor_dtype = chol.dtype
    phi_dtype = z_train.dtype
    noise_f = _expand_noise_for_square(noise.to(dtype=phi_dtype), b)
    b_f = b.to(dtype=phi_dtype)
    inv_noise_b = b_f / noise_f
    middle_rhs = (z_train.transpose(-1, -2) @ inv_noise_b).to(dtype=factor_dtype)
    inner = torch.cholesky_solve(middle_rhs, chol)
    inner_f = inner.to(dtype=phi_dtype)
    correction = z_train @ inner_f / noise_f
    out = (inv_noise_b - correction).to(dtype=factor_dtype)
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
    z_train : (n, m) or (B, n, m) feature matrix Phi.
    b : matching ``(..., n)`` or ``(..., n, k)``
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
    Returns a scalar or ``(B,)`` when ``chol`` is batched.
    """
    n = z_train.shape[-2]
    log_det_middle = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1)
    return n * noise.reshape(noise.shape[: log_det_middle.dim()]).log() + log_det_middle


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
    """y^T Sigma^{-1} y — scalar, or ``(B,)`` when ``y`` has a leading batch dim."""
    alpha = woodbury_solve(
        noise_var, z_train, y, jitter=jitter, chol=chol, noise=noise
    )
    return (y * alpha).sum(dim=-1)


@profile
def woodbury_marginal_log_likelihood(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    *,
    woodbury_form: WoodburyForm = "primal",
) -> Tensor:
    """
    Gaussian marginal log-density for ``y_centered ~ N(0, Sigma)``.

    ``Sigma = noise_var I_n + Phi Phi^T`` with ``Phi = z_train`` ``(n, m)``.
    Default ``woodbury_form="primal"`` uses ``M = I + ΦᵀΦ/σ²``.
    ``woodbury_form="dual"`` factors ``Λ = ΦᵀΦ + σ² I`` (stabler for small ``σ²``).
    """
    if woodbury_form == "dual":
        return woodbury_marginal_log_likelihood_dual(
            noise_var, z_train, y_centered, jitter=jitter
        )
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
    """
    Posterior mean of latent ``f`` at test points.

    Uses the stable primal identity ``f_* = Φ_* M^{-1} (Φᵀ y) / σ²`` rather than
    forming ``Σ^{-1} y`` in float32 (which is inaccurate when ``σ²`` is tiny).
    """
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    y = y_centered.to(dtype=z_train.dtype)
    if y.dim() == z_train.dim() - 1:
        y_col = y.unsqueeze(-1)
    else:
        y_col = y
    phi_ty = (z_train.transpose(-1, -2) @ y_col).squeeze(-1).to(dtype=chol.dtype)
    noise_c = noise.to(dtype=chol.dtype)
    while noise_c.dim() < phi_ty.dim():
        noise_c = noise_c.unsqueeze(-1)
    w = torch.cholesky_solve((phi_ty / noise_c).unsqueeze(-1), chol).squeeze(-1)
    return (z_test.to(dtype=chol.dtype) @ w.unsqueeze(-1)).squeeze(-1).to(dtype=z_test.dtype)


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

    Exact feature-space form: ``Var(f_i) = z_i^T M^{-1} z_i`` with
    ``M = I + Phi^T Phi / noise`` (equivalent to prior − explained cross form).
    """
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    lin_dtype = chol.dtype
    z_test_c = z_test.to(lin_dtype)
    # Batched: M^{-1} Z_*^T via Cholesky, then row-wise quadratic forms.
    solved = torch.cholesky_solve(z_test_c.transpose(-1, -2), chol)
    f_var = (z_test_c * solved.transpose(-1, -2)).sum(dim=-1)
    return f_var.clamp_min(0.0).to(dtype=z_test.dtype)


def woodbury_predictive_var_diag_dense_ref(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    jitter: float = 1e-6,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """
    Dense cross-cov + ``Sigma^{-1}`` reference for accuracy gates (ST).

    Matches the pre-feature-space predictive path. With ``jitter>0`` on
    ``Chol(M)``, this differs from ``z M^{-1} z`` at ``O(jitter)``; gates
    that check algebraic agreement should use ``jitter=0``.
    """
    if chol is None or noise is None:
        chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    lin_dtype = chol.dtype
    z_train_c = z_train.to(lin_dtype)
    z_test_c = z_test.to(lin_dtype)
    prior_var = (z_test_c * z_test_c).sum(dim=-1)
    cross = z_test_c @ z_train_c.transpose(-1, -2)
    alpha = woodbury_solve_from_chol(noise, z_train_c, chol, cross.transpose(-1, -2))
    explained = (cross * alpha.transpose(-1, -2)).sum(dim=-1)
    return (prior_var - explained.clamp_min(0.0)).to(z_test.dtype)


def woodbury_predictive_obs_std(f_var: Tensor, noise_var: Tensor) -> Tensor:
    """sqrt(max(f_var, 0) + noise_var); observation noise always contributes."""
    return (f_var.clamp_min(0.0) + noise_var).sqrt()


def woodbury_predict(
    noise_var: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    *,
    woodbury_form: WoodburyForm = "primal",
) -> tuple[Tensor, Tensor]:
    """
    Latent posterior mean and diagonal variance at test points (single factorization).

    Default ``woodbury_form="primal"`` uses ``M = I + ΦᵀΦ/σ²``.
    ``woodbury_form="dual"`` uses ``Λ = ΦᵀΦ + σ² I``.

    Returns
    -------
    f_mean, f_var : tensors of shape (n_test,).
    """
    if woodbury_form == "dual":
        chol, noise, z_lin = woodbury_factor_dual(noise_var, z_train, jitter=jitter)
        f_mean = woodbury_predictive_mean_dual(
            noise, z_lin, z_test, y_centered, chol=chol
        )
        f_var = woodbury_predictive_var_diag_dual(noise, z_test, chol=chol)
        return f_mean, f_var
    chol, noise = woodbury_factor(noise_var, z_train, jitter=jitter)
    f_mean = woodbury_predictive_mean(
        noise_var, z_train, z_test, y_centered, jitter=jitter, chol=chol, noise=noise
    )
    f_var = woodbury_predictive_var_diag(
        noise_var, z_train, z_test, jitter=jitter, chol=chol, noise=noise
    )
    return f_mean, f_var


# ---------------------------------------------------------------------------
# Dual Woodbury: factor Λ = ΦᵀΦ + σ² I_m  (equivalent to primal M = I + ΦᵀΦ/σ²
# via Λ = σ² M; more stable when σ² is small).
# ---------------------------------------------------------------------------


def woodbury_factor_dual(
    noise_var: Tensor,
    z_train: Tensor,
    jitter: float = 1e-6,
    *,
    promote_features: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Cholesky factor of ``Λ = ΦᵀΦ + σ² I_m``.

    By default matches primal mixed precision: form ``ΦᵀΦ`` in ``z_train.dtype``,
    then promote the small Gram for Cholesky. Set ``promote_features=True`` to
    cast full ``Φ`` before the Gram (legacy / max agreement with float64 GEMM).

    Returns
    -------
    chol : (m, m) factor of Λ
    noise : clamped noise (factor dtype)
    z_lin : Φ in the factor dtype (for dual matvecs / solves)
    """
    from .woodbury_mll_autograd import _CudaSection

    lin_dtype = _woodbury_linalg_dtype(z_train.dtype)
    noise = noise_var.clamp_min(1e-12).to(lin_dtype)

    with _CudaSection("gram"):
        if promote_features or z_train.dtype == lin_dtype:
            z_lin = z_train.to(dtype=lin_dtype, copy=False).contiguous()
            # Contiguous Φ → one GEMM; result is symmetric up to fp roundoff.
            if z_lin.dim() == 2:
                ztz = torch.mm(z_lin.transpose(0, 1), z_lin)
            else:
                ztz = torch.matmul(z_lin.transpose(-1, -2), z_lin)
        else:
            z_src = z_train.contiguous()
            if z_src.dim() == 2:
                ztz = torch.mm(z_src.transpose(0, 1), z_src).to(lin_dtype)
            else:
                ztz = torch.matmul(z_src.transpose(-1, -2), z_src).to(lin_dtype)
            z_lin = z_train.to(dtype=lin_dtype, copy=False).contiguous()

    def build_lambda(j: float) -> Tensor:
        # Diagonal add; ``_woodbury_cholesky_factor`` symmetrizes for recovery.
        noise_b = _expand_noise_for_square(noise, ztz)
        middle = ztz.clone()
        diag = torch.diagonal(middle, dim1=-2, dim2=-1)
        add = noise_b
        while add.dim() > diag.dim():
            add = add.squeeze(-1)
        diag.add_(add + j)
        return middle

    with _CudaSection("cholesky"):
        chol, _ = _woodbury_cholesky_factor(
            build_lambda, jitter, max_attempts=12, jitter_scale=10.0
        )
    return chol, noise, z_lin


# Backward-compatible alias used by LRNN.
_lambda_cholesky = woodbury_factor_dual


def woodbury_solve_dual(
    noise: Tensor,
    z_train: Tensor,
    chol: Tensor,
    b: Tensor,
) -> Tensor:
    """
    ``Σ⁻¹ b`` via dual Woodbury: ``(b - Φ Λ⁻¹ Φᵀ b) / σ²``.

    ``chol`` factors ``Λ = ΦᵀΦ + σ² I``.
    """
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    factor_dtype = chol.dtype
    phi_dtype = z_train.dtype
    noise_f = noise.to(dtype=phi_dtype)
    b_f = b.to(dtype=phi_dtype)
    phi_tb = (z_train.transpose(-1, -2) @ b_f).to(dtype=factor_dtype)
    v = torch.cholesky_solve(phi_tb, chol)
    v_f = v.to(dtype=phi_dtype)
    out = ((b_f - z_train @ v_f) / noise_f).to(dtype=factor_dtype)
    return out.squeeze(-1) if squeeze else out


def woodbury_marginal_log_likelihood_dual_reference(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Dual Woodbury MLL with stock PyTorch autodiff through Cholesky.

    Used for tests and when ``GPPLUS_WOODBURY_AUTOGRAD=reference``.
    """
    n = z_train.shape[-2]
    m = z_train.shape[-1]
    chol, noise, z_lin = woodbury_factor_dual(noise_var, z_train, jitter)
    y = y_centered.to(dtype=z_lin.dtype)
    if y.dim() == z_lin.dim() - 1:
        y_col = y.unsqueeze(-1)
    else:
        y_col = y

    log_det_lam = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1)
    phi_ty = (z_lin.transpose(-1, -2) @ y_col).squeeze(-1)
    inner = torch.cholesky_solve(phi_ty.unsqueeze(-1), chol).squeeze(-1)
    lambda_quad = (phi_ty * inner).sum(dim=-1)
    y_norm_sq = (y * y).sum(dim=-1)

    noise_b = noise.reshape(noise.shape[: y_norm_sq.dim()]) if noise.dim() > y_norm_sq.dim() else noise
    if noise_b.dim() < y_norm_sq.dim():
        # scalar noise with batched y — broadcast
        pass

    const = -0.5 * n * math.log(2.0 * math.pi)
    noise_log = -0.5 * (n - m) * noise_b.log()
    log_det_term = -0.5 * log_det_lam
    y_term = -0.5 * y_norm_sq / noise_b
    lambda_term = 0.5 * lambda_quad / noise_b
    return const + noise_log + log_det_term + y_term + lambda_term


def woodbury_marginal_log_likelihood_dual(
    noise_var: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Gaussian MLL via ``Λ = ΦᵀΦ + σ² I_m``.

    ``log|Σ| = (n-m) log σ² + log|Λ|`` and the quadratic form uses
    ``yᵀ Σ⁻¹ y = ||y||²/σ² - ||Λ⁻¹/² Φᵀ y||²/σ²``.

    Supports batched ``z_train`` ``(B, n, m)``, ``noise_var`` ``(B,)``, and
    ``y_centered`` ``(B, n)`` returning ``(B,)``.

    By default uses custom analytic autograd (see
    ``gpplus.utils.woodbury_mll_autograd``). Set
    ``GPPLUS_WOODBURY_AUTOGRAD=reference`` for stock Cholesky autodiff.
    """
    from .woodbury_mll_autograd import (
        use_reference_woodbury_autograd,
        woodbury_dual_mll_apply,
    )

    if use_reference_woodbury_autograd():
        return woodbury_marginal_log_likelihood_dual_reference(
            noise_var, z_train, y_centered, jitter=jitter
        )
    return woodbury_dual_mll_apply(noise_var, z_train, y_centered, jitter=jitter)


def woodbury_predictive_mean_dual(
    noise: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    chol: Tensor,
) -> Tensor:
    """Posterior mean ``φ*ᵀ Λ⁻¹ Φᵀ y`` (equivalent to primal Woodbury mean)."""
    del noise  # noise is already baked into chol(Λ)
    y = y_centered.to(dtype=chol.dtype).reshape(-1)
    z_lin = z_train.to(dtype=chol.dtype)
    phi_ty = z_lin.transpose(-1, -2) @ y
    w = torch.cholesky_solve(phi_ty.unsqueeze(-1), chol).squeeze(-1)
    return (z_test.to(dtype=chol.dtype) @ w).to(dtype=z_test.dtype)


def woodbury_predictive_var_diag_dual(
    noise: Tensor,
    z_test: Tensor,
    chol: Tensor,
) -> Tensor:
    """Posterior latent variance ``σ² φ*ᵀ Λ⁻¹ φ*``."""
    z_test_c = z_test.to(dtype=chol.dtype)
    solved = torch.cholesky_solve(z_test_c.transpose(-1, -2), chol)
    f_var = noise.to(dtype=chol.dtype) * (z_test_c * solved.transpose(-1, -2)).sum(dim=-1)
    return f_var.clamp_min(0.0).to(dtype=z_test.dtype)


# ---------------------------------------------------------------------------
# Heteroscedastic diagonal noise: Σ = diag(d) + Φ Φᵀ
# Implemented via D^{-1/2} row scaling → dual Woodbury with unit noise.
# ---------------------------------------------------------------------------


def _diag_noise_row_scale(
    d: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Scale rows by ``d^{-1/2}`` so ``Σ = diag(d)+ΦΦᵀ`` maps to ``I + Φ̂Φ̂ᵀ``.

    Returns
    -------
    z_hat, y_hat, d_clamped
    """
    d_clamped = d.clamp_min(1e-12)
    inv_sqrt = d_clamped.rsqrt()
    while inv_sqrt.dim() < z_train.dim() - 1:
        inv_sqrt = inv_sqrt.unsqueeze(0)
    z_hat = z_train * inv_sqrt.unsqueeze(-1)
    y_hat = y_centered * inv_sqrt
    return z_hat, y_hat, d_clamped


def woodbury_marginal_log_likelihood_diag_noise(
    d: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Gaussian MLL for ``y ~ N(0, diag(d) + Φ Φᵀ)`` with ``Φ = z_train``.

    Uses ``Φ̂ = D^{-1/2} Φ``, ``ŷ = D^{-1/2} y`` and dual Woodbury on
    ``Σ̃ = I + Φ̂ Φ̂ᵀ``, then adds the Jacobian ``-½ ∑ log d_i``.

    ``d`` shape ``(n,)`` or ``(B, n)``; ``z_train`` ``(n, m)`` or ``(B, n, m)``.
    """
    z_hat, y_hat, d_clamped = _diag_noise_row_scale(d, z_train, y_centered)
    ones = z_hat.new_ones(())
    if z_hat.dim() == 3:
        ones = z_hat.new_ones(z_hat.shape[0])
    mll_tilde = woodbury_marginal_log_likelihood_dual(
        ones, z_hat, y_hat, jitter=jitter
    )
    log_d_sum = d_clamped.log().sum(dim=-1)
    return mll_tilde - 0.5 * log_d_sum


def woodbury_posterior_weights_diag_noise(
    d: Tensor,
    z_train: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Dual weights ``w = Λ̂⁻¹ Φ̂ᵀ ŷ`` for ``Σ = diag(d)+ΦΦᵀ``.

    Returns ``(w, chol, z_hat)`` with ``Λ̂ = Φ̂ᵀ Φ̂ + I``.
    """
    z_hat, y_hat, _ = _diag_noise_row_scale(d, z_train, y_centered)
    ones = z_hat.new_ones(())
    if z_hat.dim() == 3:
        ones = z_hat.new_ones(z_hat.shape[0])
    chol, _noise, z_lin = woodbury_factor_dual(ones, z_hat, jitter=jitter)
    y = y_hat.to(dtype=chol.dtype)
    if y.dim() == z_lin.dim() - 1:
        y_col = y.unsqueeze(-1)
    else:
        y_col = y
    phi_ty = z_lin.transpose(-1, -2) @ y_col
    w = torch.cholesky_solve(phi_ty, chol).squeeze(-1)
    return w, chol, z_hat


def woodbury_predict_diag_noise(
    d_train: Tensor,
    z_train: Tensor,
    z_test: Tensor,
    y_centered: Tensor,
    jitter: float = 1e-6,
    d_test: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Posterior mean, latent var, and observation std for ``Σ_train = diag(d)+ΦΦᵀ``.

    Latent variance uses ``φ*ᵀ Λ̂⁻¹ φ*``. Observation std is
    ``sqrt(f_var + d_test)`` when ``d_test`` is provided, else ``sqrt(f_var)``
    (latent only).
    """
    w, chol, _z_hat = woodbury_posterior_weights_diag_noise(
        d_train, z_train, y_centered, jitter=jitter
    )
    z_test_c = z_test.to(dtype=chol.dtype)
    if w.dim() == 1:
        f_mean = (z_test_c @ w).to(dtype=z_test.dtype)
    else:
        f_mean = (z_test_c @ w.unsqueeze(-1)).squeeze(-1).to(dtype=z_test.dtype)
    ones = z_test.new_ones(())
    f_var = woodbury_predictive_var_diag_dual(ones, z_test, chol=chol)
    if d_test is None:
        obs_std = f_var.clamp_min(0.0).sqrt()
    else:
        obs_std = (f_var.clamp_min(0.0) + d_test.clamp_min(0.0)).sqrt()
    return f_mean, f_var, obs_std


# ---------------------------------------------------------------------------
# Multitask ICM Woodbury (Sigma = Lambda + Omega Omega^T, Omega = Phi kron R_B)
# Hot path builds M = I + (Phi^T Phi) kron (R_B^T D^{-1} R_B) without materializing Omega.
# ---------------------------------------------------------------------------

_warned_woodbury_mt_rank = False


def task_psd_factor(task_covar_matrix, jitter: float = 1e-8) -> Tensor:
    """
    Factor ``R`` with ``B = R R^T`` from a GPyTorch task ``covar_matrix``.

    Prefers Cholesky of the densified ``B`` (cheaper + friendlier autograd than
    ``eigh`` for small ``T``). Falls back to a symmetric PSD square root if
    Cholesky fails after jitter escalation.
    """
    B = task_covar_matrix.to_dense()
    B = 0.5 * (B + B.transpose(-1, -2))
    t = B.shape[-1]
    eye = torch.eye(t, device=B.device, dtype=B.dtype)
    base = float(jitter)
    last_err: Exception | None = None
    for attempt in range(10):
        j = base * (10.0**attempt)
        try:
            return torch.linalg.cholesky(B + j * eye)
        except torch.linalg.LinAlgError as exc:
            last_err = exc
    evals, evecs = torch.linalg.eigh(B)
    evals = evals.clamp(min=base)
    R = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.transpose(-1, -2)
    if last_err is not None:
        logger.warning(
            "task_psd_factor: Cholesky failed (%s); using eigh PSD square root.",
            last_err,
        )
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

    Kept for tests / debugging. Training and predict use Kronecker matvecs instead.

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


def woodbury_task_gram_mt(task_noises: Tensor, r_b: Tensor) -> Tensor:
    """S = R_B^T D^{-1} R_B with D = diag(task_noises)."""
    inv = task_noises.clamp_min(1e-12).reciprocal()
    r_scaled = r_b * inv.unsqueeze(-1)
    return r_b.transpose(-1, -2) @ r_scaled


def woodbury_middle_matrix_mt_from_phi(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    jitter: float = 0.0,
) -> Tensor:
    """M = I + (Phi^T Phi) kron (R_B^T D^{-1} R_B); never forms Omega."""
    g = phi.transpose(-1, -2) @ phi
    s = woodbury_task_gram_mt(task_noises, r_b)
    m = phi.shape[-1]
    t = r_b.shape[-1]
    m_t = m * t
    eye = torch.eye(m_t, device=phi.device, dtype=phi.dtype)
    middle = eye + torch.kron(g.contiguous(), s.contiguous())
    if jitter > 0:
        middle = middle + jitter * eye
    return middle


def woodbury_middle_matrix_mt(
    task_noises: Tensor,
    omega: Tensor,
    n: int,
    jitter: float = 0.0,
) -> Tensor:
    """Reference M = I + Omega^T Lambda^{-1} Omega (materializes Gram of Omega)."""
    t = task_noises.shape[-1]
    m_t = omega.shape[-1]
    inv_noise = task_noises.clamp_min(1e-12).reciprocal()
    omega_scaled = omega.view(n, t, m_t) * inv_noise.view(1, t, 1)
    omega_scaled = omega_scaled.reshape(n * t, m_t)
    middle = torch.matmul(omega.transpose(-1, -2), omega_scaled)
    eye = torch.eye(m_t, device=omega.device, dtype=omega.dtype)
    middle = eye + middle
    if jitter > 0:
        middle = middle + jitter * eye
    return middle


def icm_omega_matvec(phi: Tensor, r_b: Tensor, v: Tensor) -> Tensor:
    """
    Apply Omega v with Omega = Phi kron R_B (task index fastest).

    phi : (n, m), r_b : (T, T), v : (m*T,) or (m*T, K) -> (n*T,) or (n*T, K).
    Equivalent to build_icm_joint_features(phi, r_b) @ v.

    Uses one fused GEMM ``Phi @ (C R_B^T)`` with ``T*K`` packed columns so wide
    multi-RHS solves (predictive variance) stay ``O(n m T K)`` without a
    ``bmm`` batch dimension of size ``K``.
    """
    n, m = phi.shape[-2], phi.shape[-1]
    t = r_b.shape[-1]
    squeeze = v.dim() == 1
    if squeeze:
        v = v.unsqueeze(-1)
    k = v.shape[-1]
    # C:(m,T,K); tmp[m,s,k] = sum_t C[m,t,k] R_B[s,t] ; out = Phi @ tmp
    c = v.reshape(m, t, k)
    tmp = torch.matmul(c.permute(0, 2, 1), r_b.transpose(-1, -2)).permute(0, 2, 1)
    out = torch.matmul(phi, tmp.reshape(m, t * k)).reshape(n, t, k).reshape(n * t, k)
    return out.squeeze(-1) if squeeze else out


def icm_omega_rmatvec(phi: Tensor, r_b: Tensor, y: Tensor) -> Tensor:
    """
    Apply Omega^T y with Omega = Phi kron R_B.

    y : (n*T,) or (n*T, K) -> (m*T,) or (m*T, K).
    Equivalent to build_icm_joint_features(phi, r_b).T @ y.

    Fused ``Phi^T @ (Y R_B)`` with ``T*K`` packed columns (see matvec).
    """
    n, m = phi.shape[-2], phi.shape[-1]
    t = r_b.shape[-1]
    squeeze = y.dim() == 1
    if squeeze:
        y = y.unsqueeze(-1)
    k = y.shape[-1]
    # Y:(n,T,K); tmp[n,s,k] = sum_t Y[n,t,k] R_B[t,s] ; out = Phi^T @ tmp
    y_b = y.reshape(n, t, k)
    tmp = torch.matmul(y_b.permute(0, 2, 1), r_b).permute(0, 2, 1)
    out = torch.matmul(phi.transpose(-1, -2), tmp.reshape(n, t * k)).reshape(m, t, k).reshape(m * t, k)
    return out.squeeze(-1) if squeeze else out


def _icm_prior_var_diag(phi: Tensor, r_b: Tensor) -> Tensor:
    """Row-wise ||Omega||^2 without forming Omega."""
    phi_sq = (phi * phi).sum(dim=-1)  # (n,)
    rb_row_sq = (r_b * r_b).sum(dim=-1)  # (T,)
    return (phi_sq.unsqueeze(-1) * rb_row_sq.unsqueeze(0)).reshape(-1)


def _icm_cross_cov(phi_test: Tensor, phi_train: Tensor, r_b: Tensor) -> Tensor:
    """Omega_test Omega_train^T = (Phi_test Phi_train^T) kron (R_B R_B^T)."""
    kx = phi_test @ phi_train.transpose(-1, -2)
    b = r_b @ r_b.transpose(-1, -2)
    return torch.kron(kx.contiguous(), b.contiguous())


def _check_woodbury_mt_rank_phi(phi: Tensor, n: int, num_tasks: int) -> None:
    global _warned_woodbury_mt_rank
    n_t = n * num_tasks
    m_t = phi.shape[-1] * num_tasks
    if m_t >= n_t and not _warned_woodbury_mt_rank:
        logger.warning(
            "Multitask Woodbury feature width m*T=%s >= n*T=%s; low-rank solve may not reduce cost.",
            m_t,
            n_t,
        )
        _warned_woodbury_mt_rank = True


def _normalize_mt_method(method: WoodburyMtMethod) -> WoodburyMtMethod:
    """Map alias ``eigen`` -> ``primal_eigen`` (library default product eigenbasis)."""
    if method == "eigen":
        return "primal_eigen"
    return method


def woodbury_factor_mt(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    jitter: float = 1e-6,
    method: WoodburyMtMethod = "eigen",
    *,
    promote_features: bool = False,
) -> tuple[WoodburyMtFactor, Tensor]:
    """
    Factor the multitask Woodbury middle matrix.

    Parameters
    ----------
    method :
        ``"eigen"`` / ``"primal_eigen"`` (default): product eigenbasis of
        ``M = I + G ⊗ (R_Bᵀ D⁻¹ R_B)``.
        ``"dual_eigen"``: dual product eigenbasis with eigenvalues
        ``λ_g[i] μ_b[j] + task_noises[j]`` (avoids ``D⁻¹`` in the task Gram).
        ``"chol"``: dense ``M`` via ``torch.kron`` and Cholesky (Tier-2 oracle).
    promote_features :
        If True, cast full ``Phi`` to the factor dtype before the Gram (legacy).
        Default False: form ``G = Phi^T Phi`` in ``Phi``'s dtype, then promote
        only the small ``G`` / ``S`` for the eigen / Chol factor (mixed precision).
        Chol always needs a consistent middle-matrix dtype and still promotes
        ``Phi`` when the factor dtype differs.
    """
    method = _normalize_mt_method(method)
    n = phi.shape[-2]
    _check_woodbury_mt_rank_phi(phi, n, task_noises.shape[-1])
    lin_dtype = _woodbury_linalg_dtype(phi.dtype)
    noise = task_noises.clamp_min(1e-12).to(lin_dtype)
    r_c = r_b.to(lin_dtype)
    noise_out = task_noises.clamp_min(1e-12)

    if method == "chol":
        phi_c = phi.to(lin_dtype)

        def build_middle(j: float) -> Tensor:
            return woodbury_middle_matrix_mt_from_phi(noise, phi_c, r_c, jitter=j)

        chol, _ = _woodbury_cholesky_factor(build_middle, jitter)
        return WoodburyMtCholFactor(chol=chol), noise_out

    if method not in ("dual_eigen", "primal_eigen"):
        raise ValueError(
            f"method must be 'chol', 'eigen', 'dual_eigen', or 'primal_eigen', got {method!r}."
        )

    if promote_features or phi.dtype == lin_dtype:
        phi_for_g = phi.to(lin_dtype)
        g = phi_for_g.transpose(-1, -2) @ phi_for_g
    else:
        # Mixed precision: float32 GEMM for G, float64 eigen of small G/S.
        g = phi.transpose(-1, -2) @ phi
        g = g.to(lin_dtype)

    if method == "dual_eigen":
        # B = R_Bᵀ R_B (no D⁻¹); additive task noise in the eigenvalues.
        b_gram = r_c.transpose(-1, -2) @ r_c
        factor = _woodbury_eigen_factor_from_grams(
            g, b_gram, jitter, form="dual", noise_add=noise
        )
    else:
        s = woodbury_task_gram_mt(noise, r_c)
        factor = _woodbury_eigen_factor_from_grams(g, s, jitter, form="primal")
    return factor, noise_out


def woodbury_solve_mt_from_chol(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    chol: Tensor,
    b: Tensor,
) -> Tensor:
    """Sigma^{-1} b using a dense Cholesky factor of ``M`` (legacy / oracle path)."""
    return woodbury_solve_mt_from_factor(
        task_noises,
        phi,
        r_b,
        n,
        WoodburyMtCholFactor(chol=chol),
        b,
    )


def woodbury_solve_mt_from_factor(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    factor: WoodburyMtFactor,
    b: Tensor,
) -> Tensor:
    """
    Sigma^{-1} b for Sigma = Lambda + Omega Omega^T with Omega = Phi kron R_B.

    Mixed precision: Omega matvecs stay in ``phi.dtype``; only the small
    ``(m*T,)`` middle solve uses the factor dtype (float64 when features are
    float32). Returns in the factor dtype for stable quadratic forms.

    Dual eigen uses ``Σ⁻¹ b = Λ_noise⁻¹ (b - Ω Λ_dual⁻¹ Ωᵀ b)`` (exact when
    task noises are equal).
    """
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    if factor.kind == "chol":
        factor_dtype = factor.chol.dtype
        use_dual = False
    else:
        factor_dtype = factor.q_g.dtype
        use_dual = factor.form == "dual"
    phi_dtype = phi.dtype

    task_noises_f = task_noises.to(dtype=phi_dtype)
    r_b_f = r_b.to(dtype=phi_dtype)
    b_f = b.to(dtype=phi_dtype)

    if use_dual:
        # Dual: middle_rhs = Ωᵀ b, inner = Λ⁻¹ Ωᵀ b, out = Λ_noise⁻¹ (b - Ω inner)
        middle_rhs = icm_omega_rmatvec(phi, r_b_f, b_f)
        middle_rhs_f = middle_rhs.to(dtype=factor_dtype)
        inner = apply_middle_inverse_eigen(factor, middle_rhs_f)
        inner_f = inner.to(dtype=phi_dtype)
        correction = icm_omega_matvec(phi, r_b_f, inner_f)
        out = _apply_lambda_inv_rows(task_noises_f, n, b_f - correction)
        return out.to(dtype=factor_dtype).squeeze(-1) if squeeze else out.to(dtype=factor_dtype)

    inv = task_noises_f.clamp_min(1e-12).reciprocal()
    lam_inv_b = _apply_lambda_inv_rows(task_noises_f, n, b_f)
    middle_rhs = icm_omega_rmatvec(phi, r_b_f, lam_inv_b)

    middle_rhs_f = middle_rhs.to(dtype=factor_dtype)
    if factor.kind == "chol":
        inner = torch.cholesky_solve(middle_rhs_f, factor.chol)
    else:
        inner = apply_middle_inverse_eigen(factor, middle_rhs_f)
    inner_f = inner.to(dtype=phi_dtype)

    # Lambda^{-1} Omega = Phi kron (D^{-1} R_B)
    r_scaled = r_b_f * inv.unsqueeze(-1)
    correction = icm_omega_matvec(phi, r_scaled, inner_f)
    out = (lam_inv_b - correction).to(dtype=factor_dtype)
    return out.squeeze(-1) if squeeze else out


def woodbury_solve_mt(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    b: Tensor,
    jitter: float = 1e-6,
    factor: WoodburyMtFactor | None = None,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
    method: WoodburyMtMethod = "eigen",
) -> Tensor:
    if factor is None and chol is not None:
        factor = WoodburyMtCholFactor(chol=chol)
    if factor is None or noise is None:
        factor, noise = woodbury_factor_mt(
            task_noises, phi, r_b, jitter=jitter, method=method
        )
    return woodbury_solve_mt_from_factor(noise, phi, r_b, n, factor, b)


def woodbury_log_det_mt_from_chol(
    task_noises: Tensor,
    n: int,
    chol: Tensor,
) -> Tensor:
    return woodbury_log_det_mt_from_factor(
        task_noises, n, WoodburyMtCholFactor(chol=chol)
    )


def woodbury_log_det_mt_from_factor(
    task_noises: Tensor,
    n: int,
    factor: WoodburyMtFactor,
) -> Tensor:
    log_det_lam = n * task_noises.clamp_min(1e-12).log().sum()
    if factor.kind == "chol":
        log_det_middle = 2.0 * torch.diagonal(factor.chol, dim1=-2, dim2=-1).log().sum()
    elif factor.form == "dual":
        # |Σ| = |Λ_noise| * |Λ_dual| / |diag(noise_add)_{mT}|
        # so log|M_primal| ≡ log|Λ_dual| - m * sum_j log(noise_add[j]).
        noise_add = factor.noise_add
        assert noise_add is not None
        geom = factor.evals_g.unsqueeze(-1) * factor.evals_s.unsqueeze(0)
        diag = float(factor.jitter) + geom + noise_add.unsqueeze(0)
        m = factor.evals_g.numel()
        log_det_middle = diag.log().sum() - m * noise_add.clamp_min(1e-12).log().sum()
    else:
        diag = 1.0 + float(factor.jitter) + factor.evals_g.unsqueeze(-1) * factor.evals_s.unsqueeze(
            0
        )
        log_det_middle = diag.log().sum()
    return log_det_lam + log_det_middle


def woodbury_log_det_mt(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    jitter: float = 1e-6,
    factor: WoodburyMtFactor | None = None,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
    method: WoodburyMtMethod = "eigen",
) -> Tensor:
    if factor is None and chol is not None:
        factor = WoodburyMtCholFactor(chol=chol)
    if factor is None or noise is None:
        factor, noise = woodbury_factor_mt(
            task_noises, phi, r_b, jitter=jitter, method=method
        )
    return woodbury_log_det_mt_from_factor(noise, n, factor)


def woodbury_marginal_log_likelihood_mt(
    task_noises: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
    method: WoodburyMtMethod = "eigen",
    *,
    promote_features: bool = False,
) -> Tensor:
    """Gaussian log-density for vec(y) ~ N(0, Lambda + Omega Omega^T), Omega = Phi kron R_B."""
    n_t = y_centered.shape[-1]
    factor, noise = woodbury_factor_mt(
        task_noises,
        phi,
        r_b,
        jitter=jitter,
        method=method,
        promote_features=promote_features,
    )
    dtype = factor.chol.dtype if factor.kind == "chol" else factor.q_g.dtype
    y_c = y_centered.to(dtype)
    alpha = woodbury_solve_mt_from_factor(noise, phi, r_b, n, factor, y_c)
    quad = (y_c * alpha).sum(dim=-1)
    log_det = woodbury_log_det_mt_from_factor(noise, n, factor)
    const = -0.5 * n_t * math.log(2.0 * math.pi)
    return const - 0.5 * quad - 0.5 * log_det


def woodbury_predictive_mean_mt(
    task_noises: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    n_train: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
    factor: WoodburyMtFactor | None = None,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
    method: WoodburyMtMethod = "eigen",
) -> Tensor:
    """Latent posterior mean vec; reshape caller to (n_test, T)."""
    if factor is None and chol is not None:
        factor = WoodburyMtCholFactor(chol=chol)
    alpha = woodbury_solve_mt(
        task_noises,
        phi_train,
        r_b,
        n_train,
        y_centered,
        jitter=jitter,
        factor=factor,
        noise=noise,
        method=method,
    )
    # Mixed precision: Omega matvecs in phi.dtype (alpha may be factor dtype).
    alpha_f = alpha.to(dtype=phi_train.dtype)
    r_b_f = r_b.to(dtype=phi_train.dtype)
    return icm_omega_matvec(
        phi_test, r_b_f, icm_omega_rmatvec(phi_train, r_b_f, alpha_f)
    )


def _woodbury_predictive_var_diag_mt_eigen(
    phi_test: Tensor,
    r_b: Tensor,
    factor: WoodburyMtEigFactor,
) -> Tensor:
    """
    ``diag(Omega_* M^{-1} Omega_*^T)`` via product eigenbasis (no dense cross-cov).

    For task-s row of ``kron(phi_i, R_B)``, the quadratic form reduces to
    ``sum_{a,b} U[i,a]^2 V[s,b]^2 / (1+jitter+λ_a μ_b)`` with
    ``U = Phi_* Q_g``, ``V = R_B Q_s``.
    """
    # Promote only small Q / feature projections used in the eigen algebra.
    q_g = factor.q_g
    q_s = factor.q_s
    d_inv = _m_inv_diag_eigen(factor)  # (m, T)
    u = torch.matmul(phi_test.to(dtype=q_g.dtype), q_g)  # (n_*, m)
    v = torch.matmul(r_b.to(dtype=q_s.dtype), q_s)  # (T, T)
    # W[a,s] = sum_b V[s,b]^2 * d_inv[a,b]
    w = torch.matmul(d_inv, v.square().transpose(-1, -2))  # (m, T)
    f_var = torch.matmul(u.square(), w)  # (n_*, T)
    # Dual: latent var is σ² φᵀ Λ⁻¹ φ; noise_add holds per-mode σ² (equal when isotropic).
    if factor.form == "dual":
        assert factor.noise_add is not None
        f_var = f_var * factor.noise_add.to(dtype=f_var.dtype).unsqueeze(0)
    return f_var.reshape(-1).clamp_min(0.0)


def _woodbury_predictive_var_diag_mt_chol(
    phi_test: Tensor,
    r_b: Tensor,
    factor: WoodburyMtCholFactor,
) -> Tensor:
    """``diag(Omega_* M^{-1} Omega_*^T)`` via Chol(M) on materialised ``Omega_*``."""
    omega = build_icm_joint_features(
        phi_test.to(dtype=factor.chol.dtype),
        r_b.to(dtype=factor.chol.dtype),
    )
    solved = torch.cholesky_solve(omega.transpose(-1, -2), factor.chol)
    f_var = (omega * solved.transpose(-1, -2)).sum(dim=-1)
    return f_var.clamp_min(0.0)


def woodbury_predictive_var_diag_mt(
    task_noises: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    n_train: int,
    jitter: float = 1e-6,
    factor: WoodburyMtFactor | None = None,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
    method: WoodburyMtMethod = "eigen",
) -> Tensor:
    """
    Diagonal latent posterior variance for each vec entry (length n_test*T).

    Exact feature-space form ``ω_i^T M^{-1} ω_i`` (``Omega_* M^{-1} Omega_*^T`` diag).
    Does not form dense ``kron(Kx, B)`` or multi-RHS ``Sigma^{-1}``.
    """
    _ = n_train  # kept for API parity with mean / dense ref callers
    if factor is None and chol is not None:
        factor = WoodburyMtCholFactor(chol=chol)
    if factor is None or noise is None:
        factor, noise = woodbury_factor_mt(
            task_noises, phi_train, r_b, jitter=jitter, method=method
        )
    if factor.kind == "eigen":
        out = _woodbury_predictive_var_diag_mt_eigen(phi_test, r_b, factor)
    else:
        out = _woodbury_predictive_var_diag_mt_chol(phi_test, r_b, factor)
    return out.to(dtype=phi_test.dtype)


def woodbury_predictive_var_diag_mt_dense_ref(
    task_noises: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    n_train: int,
    jitter: float = 1e-6,
    factor: WoodburyMtFactor | None = None,
    chol: Tensor | None = None,
    noise: Tensor | None = None,
    method: WoodburyMtMethod = "eigen",
) -> Tensor:
    """
    Dense Kronecker cross-cov + ``Sigma^{-1}`` reference for accuracy gates (MT).

    Matches the pre-feature-space path (``kron(Kx, B)`` multi-RHS). With
    ``jitter>0`` on the middle factor, this differs from
    ``Omega_* M^{-1} Omega_*^T`` at ``O(jitter)``; algebraic gates should use
    ``jitter=0``.
    """
    if factor is None and chol is not None:
        factor = WoodburyMtCholFactor(chol=chol)
    if factor is None or noise is None:
        factor, noise = woodbury_factor_mt(
            task_noises, phi_train, r_b, jitter=jitter, method=method
        )
    lin_dtype = factor.chol.dtype if factor.kind == "chol" else factor.q_g.dtype
    phi_train_c = phi_train.to(lin_dtype)
    phi_test_c = phi_test.to(lin_dtype)
    r_b_c = r_b.to(lin_dtype)
    noise_c = noise.to(lin_dtype)
    prior_var = _icm_prior_var_diag(phi_test_c, r_b_c)
    cross = _icm_cross_cov(phi_test_c, phi_train_c, r_b_c)
    alpha = woodbury_solve_mt_from_factor(
        noise_c, phi_train_c, r_b_c, n_train, factor, cross.transpose(-1, -2)
    )
    explained = (cross * alpha.transpose(-1, -2)).sum(dim=-1)
    return (prior_var - explained.clamp_min(0.0)).to(dtype=phi_test.dtype)


def woodbury_predict_mt(
    task_noises: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    n_train: int,
    num_tasks: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
    method: WoodburyMtMethod = "eigen",
) -> tuple[Tensor, Tensor]:
    """Latent posterior mean (n_test, T) and diagonal variance (n_test, T)."""
    factor, noise = woodbury_factor_mt(
        task_noises, phi_train, r_b, jitter=jitter, method=method
    )
    mean_flat = woodbury_predictive_mean_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        y_centered,
        jitter=jitter,
        factor=factor,
        noise=noise,
        method=method,
    )
    var_flat = woodbury_predictive_var_diag_mt(
        task_noises,
        phi_train,
        phi_test,
        r_b,
        n_train,
        jitter=jitter,
        factor=factor,
        noise=noise,
        method=method,
    )
    return (
        unflatten_multitask_targets(mean_flat, num_tasks),
        unflatten_multitask_targets(var_flat, num_tasks),
    )


# ---------------------------------------------------------------------------
# Multitask Woodbury with general per-(i,t) diagonal noise (NIGP)
# Σ = diag(d) + Ω Ωᵀ, Ω = Φ ⊗ R_B.  Middle assembled without materializing Ω.
# ---------------------------------------------------------------------------


def _coerce_mt_diag_noise(d: Tensor, n: int, num_tasks: int) -> Tensor:
    """Return ``d`` as ``(n, T)``."""
    if d.dim() == 1:
        if d.numel() != n * num_tasks:
            raise ValueError(
                f"flat d length {d.numel()} != n*T={n * num_tasks}"
            )
        return d.reshape(n, num_tasks)
    if d.shape[-2:] != (n, num_tasks) and d.shape != (n, num_tasks):
        raise ValueError(f"d must be (n, T) or (n*T,), got {tuple(d.shape)}")
    return d.reshape(n, num_tasks)


def _task_factor_is_diagonal(r_b: Tensor, *, atol: float = 1e-5) -> bool:
    off = r_b - torch.diag_embed(torch.diagonal(r_b, dim1=-2, dim2=-1))
    return bool(off.abs().max().item() <= atol)


def woodbury_middle_matrix_mt_diag_noise(
    d: Tensor,
    phi: Tensor,
    r_b: Tensor,
    jitter: float = 0.0,
) -> Tensor:
    """
    ``M = I + Ωᵀ Λ⁻¹ Ω`` for ``Λ = diag(d)``, ``Ω = Φ ⊗ R_B``.

    Uses ``M = I + Σ_t kron(G^{(t)}, r_t r_tᵀ)`` with
    ``G^{(t)} = Φᵀ diag(1/d_{:,t}) Φ``. Weighted Grams are formed in ``phi``
    dtype (mixed precision); ``M`` accumulates in the caller's dtype.
    """
    n, m = phi.shape[-2], phi.shape[-1]
    t = r_b.shape[-2]
    r = r_b.shape[-1]
    d_nt = _coerce_mt_diag_noise(d, n, t).clamp_min(1e-12)
    m_r = m * r
    eye = torch.eye(m_r, device=phi.device, dtype=phi.dtype)
    middle = eye.clone()
    inv = d_nt.reciprocal()
    # View as (m, r, m, r) for blocked Kronecker accumulation.
    middle_blocks = middle.view(m, r, m, r)
    for task in range(t):
        # Gram in phi dtype; cast once into M dtype.
        g_t = phi.transpose(-1, -2) @ (phi * inv[:, task].unsqueeze(-1).to(dtype=phi.dtype))
        g_t = g_t.to(dtype=middle.dtype)
        r_t = r_b[task].to(dtype=middle.dtype)
        # kron(G, r rᵀ)_{ia,jb} = G_ij * r_a * r_b
        middle_blocks.add_(
            g_t.unsqueeze(1).unsqueeze(3) * r_t.view(1, r, 1, 1) * r_t.view(1, 1, 1, r)
        )
    if jitter > 0:
        middle = middle + jitter * eye
    return middle


def woodbury_factor_mt_diag_noise(
    d: Tensor,
    phi: Tensor,
    r_b: Tensor,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Cholesky of ``M = I + Ωᵀ Λ⁻¹ Ω`` for general per-(i,t) diag noise.

    Returns ``(chol, d_nt)`` with ``d_nt`` clamped ``(n, T)`` in factor dtype.
    Weighted Grams use ``phi`` dtype; only the middle factor is promoted.
    """
    n = phi.shape[-2]
    t = r_b.shape[-2]
    d_nt = _coerce_mt_diag_noise(d, n, t)
    lin_dtype = _woodbury_linalg_dtype(phi.dtype)
    # Keep phi in its dtype for Grams; promote r_b / d for M accumulation.
    r_c = r_b.to(dtype=lin_dtype)
    d_c = d_nt.to(dtype=lin_dtype)

    def build_middle(j: float) -> Tensor:
        # Form M in lin_dtype: Grams stay in phi.dtype inside the helper via phi,
        # then cast blocks into lin_dtype eye.
        n_loc, m_loc = phi.shape[-2], phi.shape[-1]
        t_loc = r_c.shape[-2]
        r_loc = r_c.shape[-1]
        d_use = d_c.clamp_min(1e-12)
        inv = d_use.reciprocal()
        m_r = m_loc * r_loc
        eye = torch.eye(m_r, device=phi.device, dtype=lin_dtype)
        middle = eye.clone()
        blocks = middle.view(m_loc, r_loc, m_loc, r_loc)
        inv_phi = inv.to(dtype=phi.dtype)
        for task in range(t_loc):
            g_t = phi.transpose(-1, -2) @ (phi * inv_phi[:, task].unsqueeze(-1))
            g_t = g_t.to(dtype=lin_dtype)
            r_t = r_c[task]
            blocks.add_(
                g_t.unsqueeze(1).unsqueeze(3)
                * r_t.view(1, r_loc, 1, 1)
                * r_t.view(1, 1, 1, r_loc)
            )
        if j > 0:
            middle = middle + j * eye
        return middle

    chol, _ = _woodbury_cholesky_factor(build_middle, jitter, max_attempts=12, jitter_scale=10.0)
    return chol, d_c


def _apply_general_lambda_inv_rows(d_nt: Tensor, x: Tensor) -> Tensor:
    """``Λ⁻¹ x`` for ``Λ = diag(d)`` with ``d`` shaped ``(n, T)`` (task fastest)."""
    n, t = d_nt.shape[-2], d_nt.shape[-1]
    inv = d_nt.clamp_min(1e-12).reciprocal()
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(-1)
    out = (x.reshape(n, t, -1) * inv.unsqueeze(-1)).reshape(n * t, -1)
    return out.squeeze(-1) if squeeze else out


def woodbury_solve_mt_diag_noise_from_chol(
    d_nt: Tensor,
    phi: Tensor,
    r_b: Tensor,
    chol: Tensor,
    b: Tensor,
) -> Tensor:
    """``Σ⁻¹ b`` for ``Σ = diag(d) + ΩΩᵀ`` given Chol(``M``)."""
    n = phi.shape[-2]
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    factor_dtype = chol.dtype
    phi_f = phi.to(dtype=factor_dtype)
    r_f = r_b.to(dtype=factor_dtype)
    d_f = d_nt.to(dtype=factor_dtype)
    b_f = b.to(dtype=factor_dtype)

    lam_inv_b = _apply_general_lambda_inv_rows(d_f, b_f)
    middle_rhs = icm_omega_rmatvec(phi_f, r_f, lam_inv_b)
    inner = torch.cholesky_solve(middle_rhs, chol)
    omega_inner = icm_omega_matvec(phi_f, r_f, inner)
    correction = _apply_general_lambda_inv_rows(d_f, omega_inner)
    out = lam_inv_b - correction
    return out.squeeze(-1) if squeeze else out


def _mll_mt_diag_noise_independent(
    d_nt: Tensor,
    phi: Tensor,
    r_b: Tensor,
    y_nt: Tensor,
    jitter: float,
) -> Tensor:
    """Independent-task path when ``R_B`` is diagonal: sum of ST diag-noise MLLs."""
    scales = torch.diagonal(r_b)
    total = y_nt.new_zeros(())
    t = d_nt.shape[-1]
    for task in range(t):
        phi_t = phi * scales[task]
        total = total + woodbury_marginal_log_likelihood_diag_noise(
            d_nt[:, task], phi_t, y_nt[:, task], jitter=jitter
        )
    return total


def woodbury_marginal_log_likelihood_mt_diag_noise(
    d: Tensor,
    phi: Tensor,
    r_b: Tensor,
    n: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
) -> Tensor:
    """
    Gaussian MLL for ``vec(y) ~ N(0, diag(d) + ΩΩᵀ)``, ``Ω = Φ ⊗ R_B``.

    ``d`` is ``(n, T)`` or flat ``(n*T,)``. When ``R_B`` is diagonal, uses an
    exact independent-task fast path (sum of ST diag-noise MLLs).
    """
    t = r_b.shape[-2]
    d_nt = _coerce_mt_diag_noise(d, n, t)
    y_nt = unflatten_multitask_targets(y_centered, t)
    if y_nt.shape[-2] != n:
        raise ValueError(f"y length mismatch: expected n={n}, got {y_nt.shape[-2]}")

    if _task_factor_is_diagonal(r_b):
        return _mll_mt_diag_noise_independent(d_nt, phi, r_b, y_nt, jitter)

    chol, d_c = woodbury_factor_mt_diag_noise(d_nt, phi, r_b, jitter=jitter)
    y_flat = flatten_multitask_targets(y_nt).to(dtype=chol.dtype)
    alpha = woodbury_solve_mt_diag_noise_from_chol(d_c, phi, r_b, chol, y_flat)
    quad = (y_flat * alpha).sum()
    log_det_lam = d_c.log().sum()
    log_det_m = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum()
    n_t = y_flat.shape[-1]
    const = -0.5 * n_t * math.log(2.0 * math.pi)
    return const - 0.5 * quad - 0.5 * (log_det_lam + log_det_m)


def woodbury_predict_mt_diag_noise_from_factor(
    chol: Tensor,
    d_c: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    num_tasks: int,
    alpha: Tensor,
    *,
    feature_weights: Tensor | None = None,
    d_test: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Predict from a precomputed diag-noise Chol(``M``) and ``α = Σ⁻¹ y``.

    ``feature_weights`` is ``v = Ωᵀ α`` when already available (avoids a train
    rmatvec per chunk). Latent variance uses a structured multi-RHS solve so
    full ``Ω_*`` is not required for the mean path; variance still forms a
    test-chunk ``Ω_*`` for the quadratic form (chunk-sized, not re-factored).
    """
    r_f = r_b.to(dtype=phi_train.dtype)
    phi_tr_f = phi_train.to(dtype=phi_train.dtype)
    phi_te_f = phi_test.to(dtype=phi_train.dtype)
    if feature_weights is None:
        v = icm_omega_rmatvec(phi_tr_f, r_f, alpha.to(dtype=phi_train.dtype))
    else:
        v = feature_weights.to(dtype=phi_train.dtype)
    mean_flat = icm_omega_matvec(phi_te_f, r_f, v)
    # Latent var: diag(Ω_* M^{-1} Ω_*^T) via chunk-sized Ω_* (Chol reused).
    omega_star = build_icm_joint_features(
        phi_test.to(dtype=chol.dtype), r_b.to(dtype=chol.dtype)
    )
    solved = torch.cholesky_solve(omega_star.transpose(-1, -2), chol)
    var_flat = (omega_star * solved.transpose(-1, -2)).sum(dim=-1).clamp_min(0.0)
    f_mean = unflatten_multitask_targets(mean_flat, num_tasks)
    f_var = unflatten_multitask_targets(var_flat.to(dtype=phi_test.dtype), num_tasks)
    if d_test is None:
        obs_std = f_var.clamp_min(0.0).sqrt()
    else:
        d_te = _coerce_mt_diag_noise(d_test, phi_test.shape[-2], num_tasks)
        obs_std = woodbury_predictive_obs_std(f_var, d_te)
    return f_mean, f_var, obs_std


def woodbury_predict_mt_diag_noise(
    d_train: Tensor,
    phi_train: Tensor,
    phi_test: Tensor,
    r_b: Tensor,
    n_train: int,
    num_tasks: int,
    y_centered: Tensor,
    jitter: float = 1e-6,
    d_test: Tensor | None = None,
    *,
    chol: Tensor | None = None,
    d_factor: Tensor | None = None,
    alpha: Tensor | None = None,
    feature_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Posterior mean ``(n*, T)``, latent var ``(n*, T)``, obs std ``(n*, T)``.

    Observation std uses ``d_test`` when provided (NIGP), else train task-averaged
    is not used — caller should pass ``d_test`` or use latent-only intervals.

    Optional ``chol`` / ``d_factor`` / ``alpha`` / ``feature_weights`` reuse a
    train factorization across test chunks (eval hot path).
    """
    d_nt = _coerce_mt_diag_noise(d_train, n_train, num_tasks)
    y_flat = flatten_multitask_targets(y_centered)

    if _task_factor_is_diagonal(r_b):
        scales = torch.diagonal(r_b)
        means = []
        vars_ = []
        stds = []
        y_nt = unflatten_multitask_targets(y_flat, num_tasks)
        d_te = (
            None
            if d_test is None
            else _coerce_mt_diag_noise(d_test, phi_test.shape[-2], num_tasks)
        )
        for task in range(num_tasks):
            phi_tr = phi_train * scales[task]
            phi_te = phi_test * scales[task]
            d_te_t = None if d_te is None else d_te[:, task]
            f_mean, f_var, obs_std = woodbury_predict_diag_noise(
                d_nt[:, task],
                phi_tr,
                phi_te,
                y_nt[:, task],
                jitter=jitter,
                d_test=d_te_t,
            )
            means.append(f_mean)
            vars_.append(f_var)
            stds.append(obs_std)
        f_mean = torch.stack(means, dim=-1)
        f_var = torch.stack(vars_, dim=-1)
        obs_std = torch.stack(stds, dim=-1)
        return f_mean, f_var, obs_std

    if chol is None or d_factor is None:
        chol, d_c = woodbury_factor_mt_diag_noise(d_nt, phi_train, r_b, jitter=jitter)
    else:
        d_c = d_factor
    if alpha is None:
        alpha = woodbury_solve_mt_diag_noise_from_chol(
            d_c, phi_train, r_b, chol, y_flat.to(dtype=chol.dtype)
        )
    return woodbury_predict_mt_diag_noise_from_factor(
        chol,
        d_c,
        phi_train,
        phi_test,
        r_b,
        num_tasks,
        alpha,
        feature_weights=feature_weights,
        d_test=d_test,
    )
