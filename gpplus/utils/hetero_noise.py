"""Additive hetero output noise + small Noise SORF helpers for RFFGPR.

Notation (see Final_Report_JPL/notes/heteroscedastic_nigp_combined):

- ``sigma_y_hom``: scalar homoscedastic likelihood noise ``σ_{y,hom}²``
- ``sigma_y_het(x)``: free hetero nugget ``σ_{y,het}²(x)`` from Noise SORF MAP mean
- ``sigma_nigp(x)``: NIGP term ``(∇μ)ᵀ Σ_x ∇μ``
- ``sigma_eff(x) = sigma_y_het(x) + sigma_nigp(x)`` (additive mode)

Noise SORF v1 is MAP: only the predictive *mean* of ``ℓ_het = log10 σ_{y,het}²``
is used. The Noise-SORF predictive *variance* is discarded for the Woodbury
diagonal (not integrated into ``σ_eff``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ..constraints import SoftClamp
from .rff_utils import featurize_rbf, init_rbf_weights

HeteroCombinationMode = Literal["none", "additive", "hetero_only", "multiplicative"]

# SoftClamp on log10(σ²): σ² ∈ (1e-7, 1e3), matching LogGaussianLikelihood order.
HET_LOG10_NOISE_LOWER = -7.0
HET_LOG10_NOISE_UPPER = 3.0


def resolve_hetero_mode(mode: str | None, *, nigp: bool) -> HeteroCombinationMode:
    """Normalize mode; ``None`` / empty → ``none``."""
    if mode is None or str(mode).strip() == "":
        return "none"
    m = str(mode).strip().lower()
    if m not in ("none", "additive", "hetero_only", "multiplicative"):
        raise ValueError(
            f"hetero_mode must be one of none|additive|hetero_only|multiplicative, got {mode!r}"
        )
    if m == "multiplicative":
        raise NotImplementedError(
            "hetero_mode='multiplicative' is reserved; use 'additive' "
            "(see Final_Report_JPL/notes/heteroscedastic_nigp_combined)."
        )
    if m == "additive" and not nigp:
        # Additive without NIGP is equivalent to hetero-only for σ_eff.
        return "hetero_only"
    return m  # type: ignore[return-value]


def combine_observation_noise(
    sigma_y_het: Tensor,
    sigma_nigp: Tensor | None,
    mode: HeteroCombinationMode,
) -> Tensor:
    """
    Build ``σ_eff`` from hetero nugget and optional NIGP term.

    Parameters
    ----------
    sigma_y_het :
        Per-point (or broadcastable) free output noise ``σ_{y,het}²``.
    sigma_nigp :
        Per-point NIGP term; ignored for ``hetero_only`` / ``none``.
    mode :
        Combination rule.
    """
    if mode == "none":
        return sigma_y_het.clamp_min(1e-12)
    if mode == "hetero_only":
        return sigma_y_het.clamp_min(1e-12)
    if mode == "additive":
        if sigma_nigp is None:
            raise ValueError("additive mode requires sigma_nigp")
        return (sigma_y_het + sigma_nigp).clamp_min(1e-12)
    raise NotImplementedError(f"Unsupported hetero_mode={mode!r}")


def hetero_folder_tag(
    *,
    hetero_mode: HeteroCombinationMode | str,
    epochs_hom: int | None = None,
    epochs_het: int | None = None,
    num_rff_het: int | None = None,
) -> str:
    """Save-path slug fragment, e.g. ``_hetero_additive_epohom200_epohet200_rffhet64``."""
    mode = resolve_hetero_mode(str(hetero_mode), nigp=True) if hetero_mode != "none" else "none"
    if mode == "none":
        return ""
    parts = [f"_hetero_{mode}"]
    if epochs_hom is not None:
        parts.append(f"_epohom{int(epochs_hom)}")
    if epochs_het is not None:
        parts.append(f"_epohet{int(epochs_het)}")
    if num_rff_het is not None:
        parts.append(f"_rffhet{int(num_rff_het)}")
    return "".join(parts)


def nigp_term_from_grad(
    input_noise_var: Tensor,
    grad_mu: Tensor,
) -> Tensor:
    """``σ_nigp² = Σ_d σ_{x,d}² (∇μ_d)²`` with shape matching ``grad_mu`` leading dims."""
    g2 = grad_mu * grad_mu
    if input_noise_var.dim() == 1:
        return (g2 * input_noise_var).sum(dim=-1)
    iv = input_noise_var
    while iv.dim() < g2.dim():
        iv = iv.unsqueeze(-2)
    return (g2 * iv).sum(dim=-1)


class NoiseSORFLogNoise(nn.Module):
    """
    Small SORF map for ``ℓ_het(x) = log10 σ_{y,het}²(x)`` (MAP).

    Forward returns **variance** ``σ_{y,het}²(x) = 10^{SoftClamp(ℓ_het)}``.
    The feature weights are fixed random SORF frequencies; only the linear head
    (and optional log-lengthscale) are trainable. Predictive variance of this
    noise field is **not** returned or used in ``σ_eff`` (v1 MAP).
    """

    def __init__(
        self,
        input_dim: int,
        num_rff_het: int = 64,
        *,
        correct_sorf: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if int(num_rff_het) < 1:
            raise ValueError(f"num_rff_het must be >= 1, got {num_rff_het}")
        self.input_dim = int(input_dim)
        self.num_rff_het = int(num_rff_het)
        self.correct_sorf = bool(correct_sorf)
        dev = torch.device("cpu") if device is None else torch.device(device)
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
        with torch.random.fork_rng(devices=[]):
            if gen is not None:
                torch.manual_seed(int(seed))
            w = init_rbf_weights(
                self.input_dim,
                self.num_rff_het,
                device=torch.device("cpu"),
                dtype=dtype,
                rff_sampling="sorf",
                correct_sorf=self.correct_sorf,
                spectral_kernel="rbf",
            )
        self.register_buffer("randn_weights", w.to(device=dev, dtype=dtype))
        # lengthscale in same LogScaleKernel convention (raw ≈ 0 → scale 1)
        self.raw_lengthscale = nn.Parameter(torch.zeros(1, self.input_dim, dtype=dtype, device=dev))
        feat_dim = 2 * self.num_rff_het
        self.weight = nn.Parameter(torch.zeros(feat_dim, dtype=dtype, device=dev))
        self.bias = nn.Parameter(torch.full((), -2.0, dtype=dtype, device=dev))  # ~1e-2 variance
        self._log10_constraint = SoftClamp(
            HET_LOG10_NOISE_LOWER, HET_LOG10_NOISE_UPPER, margin=1e-2
        )

    def log10_noise_mean(self, x: Tensor) -> Tensor:
        """MAP mean ``μ_ℓ(x)`` after SoftClamp (shape ``(n,)``)."""
        x = x.to(dtype=self.weight.dtype, device=self.weight.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        ls = self.raw_lengthscale
        z = featurize_rbf(x, self.randn_weights, ls, num_samples=self.num_rff_het)
        # z: (n, 2D)
        raw = z @ self.weight + self.bias
        return self._log10_constraint.transform(raw)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``σ_{y,het}²(x)`` of shape ``(n,)`` from the MAP log-noise mean."""
        return torch.pow(10.0, self.log10_noise_mean(x)).clamp_min(1e-12)

    def init_from_homoscedastic_noise(self, noise_variance: Tensor) -> None:
        """
        Constant MAP field matching scalar ``σ_{y,hom}²``: zero feature weights,
        bias so ``10^{SoftClamp(bias)} = noise_variance``.
        """
        var = torch.as_tensor(
            noise_variance, dtype=self.bias.dtype, device=self.bias.device
        )
        if var.numel() != 1:
            var = var.reshape(-1)[0]
        else:
            var = var.reshape(())
        log10 = torch.log10(var.clamp_min(1e-12))
        with torch.no_grad():
            self.weight.zero_()
            self.bias.copy_(self._log10_constraint.inverse_transform(log10))

    def set_trainable(self, trainable: bool) -> None:
        for p in self.parameters():
            p.requires_grad_(bool(trainable))


__all__ = [
    "HeteroCombinationMode",
    "HET_LOG10_NOISE_LOWER",
    "HET_LOG10_NOISE_UPPER",
    "NoiseSORFLogNoise",
    "combine_observation_noise",
    "hetero_folder_tag",
    "nigp_term_from_grad",
    "resolve_hetero_mode",
]
