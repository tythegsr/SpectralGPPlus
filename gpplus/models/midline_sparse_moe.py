"""MidlineSparseMoE-GP: one GP core + sparse additive mean offsets.

Two-stage training:
  1. ``fit_core`` — fit a single exact GPR midline (once).
  2. ``fit_offsets`` — learn overcomplete offsets with Hard-Concrete sparsity
     and a constant (or MLP) gate; no outer loop over K.

Offsets default to input-dependent MLP ``δ_k(x)``. Component variance uses a
multiplicative carve-out of the GP predictive variance, optionally scaled::

    Var_k(x) = σ₀²(x) * (1 - α_k γ_k(x)) * β_k + ε,   γ_k ∈ (0,1), β_k ≥ 1

``γ`` is per-expert constant by default (``gamma_type='constant'``); set
``gamma_type='mlp'`` for input-dependent ``γ_k(x)=σ(MLP(x))``.

With ``free_expert_scale=False`` (default / champion), ``β_k ≡ 1``.
Optional ``free_expert_scale=True`` learns ``β_k = 1 + softplus(raw_β_k)``;
explored on A04 and found inert — keep muted in production fits.

Effective expert count (Option E)::

    ã_k = σ((α_k π̄_k - ε) / τ),   K_eff = Σ_k ã_k s_k

so unused (low mass) slots drop out and duplicate curves (low s_k) count less.
Stage-2 sparsifies via Σ ã_k, Hard-Concrete complexity, bidirectional π–α
coupling (α-on/no-mass and mass-on/α-off), optional δ L2, and usage-weighted
separation. Optional active-pair overlap penalty encourages
``|μ_i-μ_j|/(σ_i+σ_j)`` above a margin so mixture peaks stay distinct without
prescribing a target mode count.

With ``adaptive_merge_prune=True`` (default in ``fit``), a soft online NLL
merge probe sets ``merge_pressure`` (merge / separate / partial mix) and fades
anti-merge weights; usage L0 scales with ``K_max`` only, and multi-init
rejects total collapse vs Stage-1 GP NLL — without targeting a mode count.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import gpytorch
import numpy as np
import torch
from torch import nn

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood
from .gpr import GPR
from .moe_gpr import ConstantGate, GatingNetwork

GateType = Literal["constant", "mlp"]
OffsetType = Literal["constant", "mlp"]
MidlineBackend = Literal["exact", "ni_sorf"]
GammaType = Literal["constant", "mlp"]


def _gaussian_log_prob(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = (std * std).clamp_min(1e-12)
    return -0.5 * (torch.log(2.0 * torch.pi * var) + (y - mean).pow(2) / var)


def _residual_unimodal_score(
    resid: np.ndarray,
    *,
    bw_factor: float = 0.55,
) -> float:
    """Soft score in [0,1] from Stage-1 residual KDE peak count.

    Rough mapping (data-driven, no true-K / problem id):
      ~1 peak → merge, ~2 peaks → partial mix, ≥3 peaks → separate.

    Counts KDE local maxima with relative height ≥ 8% of the peak.
    """
    r = np.asarray(resid, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 16:
        return 0.5
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return 0.5
    try:
        kde = gaussian_kde(r)
        kde.set_bandwidth(float(kde.factor) * float(bw_factor))
        lo, hi = np.quantile(r, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.5
        grid = np.linspace(float(lo), float(hi), 400)
        dens = np.asarray(kde(grid), dtype=np.float64)
        dmax = float(np.max(dens)) if dens.size else 0.0
        if dmax <= 1e-12:
            return 0.5
        thr = 0.08 * dmax
        peaks = 0
        for i in range(1, dens.size - 1):
            if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1] and dens[i] >= thr:
                peaks += 1
    except Exception:  # noqa: BLE001
        return 0.5
    # peaks=1 → ~0.82, peaks=2 → ~0.38, peaks=3 → ~0.08, peaks=4 → ~0.01
    z = float(np.clip((1.75 - float(peaks)) / 0.5, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _soft_peak_count_torch(
    density: torch.Tensor,
    *,
    rel_thr: float = 0.05,
    temp: float = 0.02,
) -> torch.Tensor:
    """Expected soft local-max count on a 1D density grid (training regularizer)."""
    d = density.reshape(-1)
    if d.numel() < 3:
        return d.new_zeros(())
    dmax = d.max().clamp_min(1e-12)
    thr = rel_thr * dmax
    t = (temp * dmax).clamp_min(1e-8)
    left = d[1:-1] - d[:-2]
    right = d[1:-1] - d[2:]
    mid = d[1:-1]
    return (
        torch.sigmoid(left / t) * torch.sigmoid(right / t) * torch.sigmoid((mid - thr) / t)
    ).sum()


class HardConcreteGate(nn.Module):
    """Hard-Concrete / Concrete L0 gate (Louizos et al.), one scalar per expert."""

    def __init__(
        self,
        num_experts: int,
        *,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        init = torch.full((num_experts,), 0.0, dtype=dtype)
        if num_experts >= 1:
            init[0] = 3.0
        if num_experts >= 2:
            init[1] = 3.0
        self.log_alpha = nn.Parameter(init)

    def _stretch(self, s: torch.Tensor) -> torch.Tensor:
        return s * (self.zeta - self.gamma) + self.gamma

    def forward(self, *, deterministic: bool | None = None) -> torch.Tensor:
        if deterministic is None:
            deterministic = not self.training
        if deterministic:
            s = torch.sigmoid(self.log_alpha)
            return self._stretch(s).clamp(0.0, 1.0)
        u = torch.rand_like(self.log_alpha).clamp(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.beta)
        return self._stretch(s).clamp(0.0, 1.0)

    def complexity_loss(self) -> torch.Tensor:
        thresh = -self.gamma / self.zeta
        return torch.sigmoid(
            self.log_alpha
            - self.beta
            * torch.log(
                torch.tensor(thresh, dtype=self.log_alpha.dtype, device=self.log_alpha.device)
            )
        ).sum()


class OffsetNetwork(nn.Module):
    """Shared MLP trunk with K heads: ``δ(x) ∈ R^K``."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dims: Sequence[int] = (32, 32),
        *,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_experts)))
        self.network = nn.Sequential(*layers).to(dtype=dtype)
        # Small random weights so early δ_k(x) ≈ bias (seeded after core fit)
        with torch.no_grad():
            for mod in self.network:
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, mean=0.0, std=0.05)
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def seed_bias(self, values: torch.Tensor) -> None:
        """Set last-layer bias to residual quantile seeds (centered)."""
        last = None
        for mod in self.network:
            if isinstance(mod, nn.Linear):
                last = mod
        if last is None:
            return
        with torch.no_grad():
            last.bias.copy_(values.to(dtype=last.bias.dtype, device=last.bias.device))


class MidlineSparseMoEGPR(nn.Module):
    """GP midline + Hard-Concrete-sparse additive offsets (constant or MLP)."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        k_max: int = 6,
        gate_type: GateType = "mlp",
        offset_type: OffsetType = "mlp",
        gamma_type: GammaType = "constant",
        gating_hidden: Sequence[int] = (32, 32),
        offset_hidden: Sequence[int] = (32, 32),
        gamma_hidden: Sequence[int] | None = None,
        temperature: float = 1.0,
        tau_max: float = 3.0,  # unused; kept for API compat with older callers
        sep_length: float = 2.0,
        var_floor: float = 1e-6,
        use_full_gp_variance: bool = True,
        free_expert_scale: bool = False,
        gamma_init_logit: float = 1.1,
        keff_eps: float = 0.05,
        keff_tau: float = 0.02,
        midline_backend: MidlineBackend = "exact",
        ni_sorf_num_rff: int = 2400,
        ni_sorf_num_epochs: int = 1000,
        ni_sorf_lr: float = 1e-2,
        ni_sorf_freeze_epoch_nigp: int = 200,
        ni_sorf_nigp_slope_refreshes: int = 10,
        var_expand: bool = False,
        var_expand_max_log: float = 2.0,
        het_stage1: bool = False,
        het_clip: tuple[float, float] = (0.2, 5.0),
        stage1: str = "gpplus",
        loo_mean_terms: bool = True,
        loo_backend: str = "auto",
        loo_nigp: bool = False,
        calib_frac: float = 0.0,
        calib_rotate: bool = False,
        het_auto: bool = False,
        vexp_cal_only: bool = False,
        lambda_vexp: float = 0.05,
    ):
        super().__init__()
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor.")
        if train_y.ndim > 1:
            train_y = train_y.reshape(-1)
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")
        if gate_type not in ("constant", "mlp"):
            raise ValueError(f"gate_type must be 'constant' or 'mlp', got {gate_type!r}")
        if offset_type not in ("constant", "mlp"):
            raise ValueError(f"offset_type must be 'constant' or 'mlp', got {offset_type!r}")
        if gamma_type not in ("constant", "mlp"):
            raise ValueError(f"gamma_type must be 'constant' or 'mlp', got {gamma_type!r}")
        if keff_tau <= 0:
            raise ValueError(f"keff_tau must be > 0, got {keff_tau}")
        if midline_backend not in ("exact", "ni_sorf"):
            raise ValueError(
                f"midline_backend must be 'exact' or 'ni_sorf', got {midline_backend!r}"
            )

        self.register_buffer("train_x", train_x.detach().clone())
        self.register_buffer("train_y", train_y.detach().clone())
        self.k_max = int(k_max)
        self.num_experts = self.k_max
        self.gate_type: GateType = gate_type
        self.offset_type: OffsetType = offset_type
        self.gamma_type: GammaType = gamma_type
        self.midline_backend: MidlineBackend = midline_backend
        self.ni_sorf_num_rff = int(ni_sorf_num_rff)
        self.ni_sorf_num_epochs = int(ni_sorf_num_epochs)
        self.ni_sorf_lr = float(ni_sorf_lr)
        self.ni_sorf_freeze_epoch_nigp = int(ni_sorf_freeze_epoch_nigp)
        self.ni_sorf_nigp_slope_refreshes = int(ni_sorf_nigp_slope_refreshes)
        self.tau_max = float(tau_max)  # deprecated (additive τ removed)
        self.sep_length = float(sep_length)
        self.var_floor = float(var_floor)
        self.keff_eps = float(keff_eps)
        self.keff_tau = float(keff_tau)
        # If True, mixture uses full GP predictive σ₀ as variance ceiling;
        # experts carve down via γ_k. If False, use capped Stage-2 σ₀.
        self.use_full_gp_variance = bool(use_full_gp_variance)
        # If True, learn β_k ≥ 1 so Var_k can exceed σ₀² (unequal widths).
        self.free_expert_scale = bool(free_expert_scale)
        self.dtype = train_x.dtype
        input_dim = int(train_x.shape[-1])
        self._gamma_init_logit = float(gamma_init_logit)

        self.core: Any = None  # GPR | RFFGPR
        self.register_buffer("_mu0_train", torch.zeros(train_x.shape[0], dtype=self.dtype))
        self.register_buffer("_std0_train", torch.ones(train_x.shape[0], dtype=self.dtype))
        self.register_buffer("_std0_stage2", torch.ones(train_x.shape[0], dtype=self.dtype))
        self._core_fitted = False
        self.std0_cap_frac: float = 0.3
        # Behboodian-scale min spacing for residual δ seeds (set before seeding).
        self.rec_margin: float = 1.0

        # γ ∈ (0,1): fraction of GP variance an active expert may remove.
        # Var_k = σ₀² (1 - α_k γ_k) β_k + ε  (γ_k or γ_k(x)).
        # Default logit≈1.1 → γ≈0.75; probe arms may pass ~2.2 → γ≈0.9.
        g_hid = (
            tuple(int(h) for h in gamma_hidden)
            if gamma_hidden is not None
            else tuple(int(h) for h in offset_hidden)
        )
        self._gamma_hidden = g_hid
        if gamma_type == "constant":
            self.raw_gamma: nn.Parameter | None = nn.Parameter(
                torch.full((self.k_max,), float(gamma_init_logit), dtype=self.dtype)
            )
            self.gamma_net: OffsetNetwork | None = None
        else:
            self.raw_gamma = None
            self.gamma_net = OffsetNetwork(
                input_dim=input_dim,
                num_experts=self.k_max,
                hidden_dims=g_hid,
                dtype=self.dtype,
            )
            # Seed last-layer bias so early γ_k(x) ≈ sigmoid(gamma_init_logit).
            self.gamma_net.seed_bias(
                torch.full((self.k_max,), float(gamma_init_logit), dtype=self.dtype)
            )
        # β_k = 1 + softplus(raw_β); init raw≈-2 → β≈1.13, near identity.
        self.raw_beta = nn.Parameter(torch.full((self.k_max,), -2.0, dtype=self.dtype))
        # Optional input-dependent variance *expansion*: Var_k ×= exp(s_k(x)),
        # s_k ∈ [0, var_expand_max_log], so experts may widen beyond σ₀²
        # (carve-out γ alone can only shrink).
        self.var_expand = bool(var_expand)
        self.var_expand_max_log = float(var_expand_max_log)
        self.scale_net: OffsetNetwork | None = None
        if self.var_expand:
            self.scale_net = self._new_scale_net()
        # Optional heteroscedastic Stage-1: σ₀(x) ×= sqrt(f(x)) with f from a
        # log-variance MLP on GP residuals, normalised to mean 1 on train.
        self.het_stage1 = bool(het_stage1)
        self.het_clip = (float(het_clip[0]), float(het_clip[1]))
        self.het_net: OffsetNetwork | None = None
        self._het_log_norm: float = 0.0
        self._std0_cap_value: float | None = None
        # Self-configuring ("auto") options. ``stage1='loo'`` fits the midline on
        # the closed-form LOO likelihood and hands Stage 2 the LOO mean/std at
        # training points, so Stage 2 sees honest (not interpolated) residuals.
        if stage1 not in ("gpplus", "loo"):
            raise ValueError(f"stage1 must be 'gpplus' or 'loo', got {stage1!r}")
        self.stage1 = stage1
        self.loo_mean_terms = bool(loo_mean_terms)
        # "exact" | "sorf" | "auto" (padded-SORF Bayesian linear core when n is large).
        self.loo_backend = str(loo_backend)
        self.loo_nigp = bool(loo_nigp)
        # Fraction of points whose NLL trains only the calibration parameters
        # (temperature, het shrinkage, optional expansion net), never the networks.
        self.calib_frac = float(calib_frac)
        self.calib_rotate = bool(calib_rotate)
        self.het_auto = bool(het_auto)
        self.vexp_cal_only = bool(vexp_cal_only)
        self.lambda_vexp = float(lambda_vexp)
        self.log_temp = nn.Parameter(torch.zeros((), dtype=self.dtype))
        self.raw_het_w = nn.Parameter(torch.tensor(-1.0, dtype=self.dtype))
        self.register_buffer("_het_h_train", torch.zeros(train_x.shape[0], dtype=self.dtype))
        self.alpha_gate = HardConcreteGate(self.k_max, dtype=self.dtype)

        if offset_type == "constant":
            self.delta = nn.Parameter(torch.zeros(self.k_max, dtype=self.dtype))
            self.offset_net: OffsetNetwork | None = None
        else:
            self.delta = None  # type: ignore[assignment]
            self.offset_net = OffsetNetwork(
                input_dim=input_dim,
                num_experts=self.k_max,
                hidden_dims=offset_hidden,
                dtype=self.dtype,
            )

        if gate_type == "constant":
            self.gating: nn.Module = ConstantGate(
                num_experts=self.k_max,
                temperature=temperature,
                dtype=self.dtype,
            )
        else:
            self.gating = GatingNetwork(
                input_dim=input_dim,
                num_experts=self.k_max,
                hidden_dims=gating_hidden,
                temperature=temperature,
                dtype=self.dtype,
            )

        self.mode_loss_weight: float = 0.0
        self._fit_history: list[dict] = []
        self._stage1_gp_nll: float | None = None
        self._gating_hidden = tuple(int(h) for h in gating_hidden)
        self._offset_hidden = tuple(int(h) for h in offset_hidden)
        self._gate_temperature = float(temperature)
        self.merge_pressure: float = 0.0
        self._adaptive_diagnostics: dict = {}

    def _new_scale_net(self) -> OffsetNetwork:
        net = OffsetNetwork(
            input_dim=int(self.train_x.shape[-1]),
            num_experts=self.k_max,
            hidden_dims=self._gamma_hidden,
            dtype=self.dtype,
        ).to(device=self.train_x.device)
        # sigmoid(-4) ≈ 0.018 → s ≈ 0 (no expansion) at init.
        net.seed_bias(
            torch.full((self.k_max,), -4.0, dtype=self.dtype, device=self.train_x.device)
        )
        return net

    def get_var_scale(self, x: torch.Tensor) -> torch.Tensor | None:
        """``exp(s_k(x))`` ``[N,K]`` with ``s ∈ [0, var_expand_max_log]``, or None."""
        if self.scale_net is None:
            return None
        self.scale_net.to(device=x.device)
        s = self.var_expand_max_log * torch.sigmoid(self.scale_net(x))
        return torch.exp(s)

    def _het_factor(self, x: torch.Tensor) -> torch.Tensor | None:
        """Std multiplier ``sqrt(f(x))`` ``[N]`` from the het Stage-1 net, or None."""
        if self.het_net is None:
            return None
        self.het_net.to(device=x.device)
        with torch.no_grad():
            h = self.het_net(x.to(dtype=self.dtype)).reshape(-1) - self._het_log_norm
            f = torch.exp(h).clamp(self.het_clip[0], self.het_clip[1])
        return f.sqrt()

    def _het_log(self, x: torch.Tensor) -> torch.Tensor | None:
        """Clipped, train-normalised log variance ratio ``h(x)`` ``[N]``, or None."""
        if self.het_net is None:
            return None
        self.het_net.to(device=x.device)
        with torch.no_grad():
            h = self.het_net(x.to(dtype=self.dtype)).reshape(-1) - self._het_log_norm
            lo, hi = np.log(self.het_clip[0]), np.log(self.het_clip[1])
        return h.clamp(float(lo), float(hi))

    def _het_auto_factor(self, x: torch.Tensor) -> torch.Tensor | None:
        """Std multiplier ``exp(w·h(x)/2)`` with learned shrinkage ``w ∈ (0,1)``."""
        if not self.het_auto:
            return None
        h = self._het_log(x)
        if h is None:
            return None
        w = torch.sigmoid(self.raw_het_w.detach()).to(device=h.device, dtype=h.dtype)
        return torch.exp(0.5 * w * h)

    def _fit_het_stage1(
        self, x: torch.Tensor, resid: torch.Tensor, *, seed: int, steps: int = 600
    ) -> None:
        """Gaussian NLL on residuals with log-variance MLP ``h(x)``."""
        torch.manual_seed(int(seed) + 101)
        net = OffsetNetwork(
            input_dim=int(x.shape[-1]),
            num_experts=1,
            hidden_dims=(32, 32),
            dtype=self.dtype,
        )
        r2 = resid.reshape(-1).to(dtype=self.dtype).pow(2)
        net.seed_bias(torch.log(r2.mean().clamp_min(1e-8)).reshape(1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
        xc = x.to(dtype=self.dtype).cpu()
        r2 = r2.cpu()
        for _ in range(int(steps)):
            opt.zero_grad()
            h = net(xc).reshape(-1)
            loss = 0.5 * (h + r2 * torch.exp(-h)).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            h = net(xc).reshape(-1)
            self._het_log_norm = float(torch.log(torch.exp(h).mean()).item())
        self.het_net = net

    def _build_gating(self, gate_type: GateType) -> nn.Module:
        input_dim = int(self.train_x.shape[-1])
        if gate_type == "constant":
            mod: nn.Module = ConstantGate(
                num_experts=self.k_max,
                temperature=self._gate_temperature,
                dtype=self.dtype,
            )
        else:
            mod = GatingNetwork(
                input_dim=input_dim,
                num_experts=self.k_max,
                hidden_dims=self._gating_hidden,
                temperature=self._gate_temperature,
                dtype=self.dtype,
            )
        # Match Stage-2 working device (train_x may be on CUDA during fit).
        return mod.to(device=self.train_x.device)

    def _set_gate_type(self, gate_type: GateType) -> None:
        if gate_type not in ("constant", "mlp"):
            raise ValueError(f"gate_type must be 'constant' or 'mlp', got {gate_type!r}")
        self.gate_type = gate_type
        self.gating = self._build_gating(gate_type)

    @staticmethod
    def _mixture_nll(
        y: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        log_comps = []
        for k in range(means.shape[1]):
            log_comps.append(
                torch.log(pi[:, k].clamp_min(1e-12))
                + _gaussian_log_prob(y, means[:, k], stds[:, k])
            )
        return -torch.logsumexp(torch.stack(log_comps, dim=-1), dim=-1).mean()

    def _gaussian_nll(
        self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return -_gaussian_log_prob(y, mean, std).mean()

    def _stage2_param_state(self) -> dict:
        """State dict for Stage-2 modules (excludes core GP)."""
        state = {
            "gate_type": self.gate_type,
            "gamma_type": self.gamma_type,
            "alpha_gate": {k: v.detach().cpu().clone() for k, v in self.alpha_gate.state_dict().items()},
            "raw_beta": self.raw_beta.detach().cpu().clone(),
            "gating": {k: v.detach().cpu().clone() for k, v in self.gating.state_dict().items()},
            "merge_pressure": float(self.merge_pressure),
            "gamma_init_logit": float(self._gamma_init_logit),
            "log_temp": self.log_temp.detach().cpu().clone(),
            "raw_het_w": self.raw_het_w.detach().cpu().clone(),
        }
        if self.gamma_type == "constant":
            assert self.raw_gamma is not None
            state["raw_gamma"] = self.raw_gamma.detach().cpu().clone()
        else:
            assert self.gamma_net is not None
            state["gamma_net"] = {
                k: v.detach().cpu().clone() for k, v in self.gamma_net.state_dict().items()
            }
        if self.offset_type == "constant":
            assert self.delta is not None
            state["delta"] = self.delta.detach().cpu().clone()
        else:
            assert self.offset_net is not None
            state["offset_net"] = {
                k: v.detach().cpu().clone() for k, v in self.offset_net.state_dict().items()
            }
        if self.scale_net is not None:
            state["scale_net"] = {
                k: v.detach().cpu().clone() for k, v in self.scale_net.state_dict().items()
            }
        return state

    def _load_stage2_param_state(self, state: dict) -> None:
        self._set_gate_type(state["gate_type"])
        # Ensure rebuilt modules sit on the current Stage-2 device.
        self.alpha_gate.to(device=self.train_x.device)
        self.gating.to(device=self.train_x.device)
        if self.offset_net is not None:
            self.offset_net.to(device=self.train_x.device)
        if self.gamma_net is not None:
            self.gamma_net.to(device=self.train_x.device)
        self.alpha_gate.load_state_dict(state["alpha_gate"])
        with torch.no_grad():
            self.raw_beta.copy_(state["raw_beta"].to(device=self.raw_beta.device))
            if "log_temp" in state:
                self.log_temp.copy_(state["log_temp"].to(device=self.log_temp.device))
                self.raw_het_w.copy_(state["raw_het_w"].to(device=self.raw_het_w.device))
            if self.gamma_type == "constant":
                assert self.raw_gamma is not None
                self.raw_gamma.copy_(state["raw_gamma"].to(device=self.raw_gamma.device))
            else:
                assert self.gamma_net is not None
                self.gamma_net.load_state_dict(state["gamma_net"])
        self.gating.load_state_dict(state["gating"])
        self.merge_pressure = float(state.get("merge_pressure", 0.0))
        if "gamma_init_logit" in state:
            self._gamma_init_logit = float(state["gamma_init_logit"])
        if self.offset_type == "constant":
            assert self.delta is not None
            with torch.no_grad():
                self.delta.copy_(state["delta"].to(device=self.delta.device))
        else:
            assert self.offset_net is not None
            self.offset_net.load_state_dict(state["offset_net"])
        if self.scale_net is not None and "scale_net" in state:
            self.scale_net.to(device=self.train_x.device)
            self.scale_net.load_state_dict(state["scale_net"])

    def _reinit_stage2(
        self,
        *,
        seed: int,
        gate_type: GateType | None = None,
        seed_sep_scale: float = 1.0,
        gamma_init_logit: float | None = None,
    ) -> None:
        """Re-sample Stage-2 parameters for a fresh init (core GP unchanged)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if gate_type is not None:
            self._set_gate_type(gate_type)
        # Fresh Hard-Concrete logits
        with torch.no_grad():
            init = torch.full((self.k_max,), 0.0, dtype=self.dtype)
            if self.k_max >= 1:
                init[0] = 3.0
            if self.k_max >= 2:
                init[1] = 3.0
            self.alpha_gate.log_alpha.copy_(init)
            g0 = (
                float(gamma_init_logit)
                if gamma_init_logit is not None
                else float(self._gamma_init_logit)
            )
            self._gamma_init_logit = g0
            if self.gamma_type == "constant":
                assert self.raw_gamma is not None
                self.raw_gamma.fill_(g0)
            self.raw_beta.fill_(-2.0)
            self.log_temp.fill_(0.0)
            self.raw_het_w.fill_(-1.0)
        # Rebuild gating weights
        self.gating = self._build_gating(self.gate_type)
        if self.offset_type == "constant":
            assert self.delta is not None
            with torch.no_grad():
                self.delta.zero_()
        else:
            assert self.offset_net is not None
            self.offset_net = OffsetNetwork(
                input_dim=int(self.train_x.shape[-1]),
                num_experts=self.k_max,
                hidden_dims=self._offset_hidden,
                dtype=self.dtype,
            ).to(device=self.train_x.device)
        if self.gamma_type == "mlp":
            self.gamma_net = OffsetNetwork(
                input_dim=int(self.train_x.shape[-1]),
                num_experts=self.k_max,
                hidden_dims=self._gamma_hidden,
                dtype=self.dtype,
            ).to(device=self.train_x.device)
            self.gamma_net.seed_bias(
                torch.full((self.k_max,), g0, dtype=self.dtype, device=self.train_x.device)
            )
        if self.var_expand:
            self.scale_net = self._new_scale_net()
        self._seed_offsets_from_residuals(sep_scale=seed_sep_scale)
        self.merge_pressure = 0.0

    def _move_stage2_tensors(self, device: str | torch.device) -> None:
        """Move Stage-2 modules/buffers; leave Exact GP ``core`` untouched."""
        dev = torch.device(device) if isinstance(device, str) else device
        self.alpha_gate.to(dev)
        self.gating.to(dev)
        if self.offset_net is not None:
            self.offset_net.to(dev)
        if self.gamma_net is not None:
            self.gamma_net.to(dev)
        if self.scale_net is not None:
            self.scale_net.to(dev)

        def _param(p: nn.Parameter) -> nn.Parameter:
            return nn.Parameter(p.detach().to(device=dev, dtype=p.dtype))

        if self.raw_gamma is not None:
            self.raw_gamma = _param(self.raw_gamma)
        self.raw_beta = _param(self.raw_beta)
        self.log_temp = _param(self.log_temp)
        self.raw_het_w = _param(self.raw_het_w)
        if self.delta is not None:
            self.delta = _param(self.delta)
        for name in (
            "train_x",
            "train_y",
            "_mu0_train",
            "_std0_train",
            "_std0_stage2",
            "_het_h_train",
        ):
            self.register_buffer(name, getattr(self, name).to(device=dev))
        self._stage2_device = str(dev)

    # ------------------------------------------------------------------ core
    def fit_core(
        self,
        *,
        num_inits: int = 4,
        num_epochs: int = 1,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
    ) -> "MidlineSparseMoEGPR":
        """Stage 1: Exact GPR or NI-SORF midline (see ``midline_backend``)."""
        x = self.train_x
        y = self.train_y
        best_loss = float("nan")

        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import fit_ni_sorf_core, predict_ni_sorf

            core_device = device
            if device == "cpu" and torch.cuda.is_available():
                # Prefer CUDA for NI-SORF Woodbury training when available
                core_device = "cuda"
            core, info = fit_ni_sorf_core(
                x,
                y,
                num_rff=self.ni_sorf_num_rff,
                num_epochs=self.ni_sorf_num_epochs,
                num_inits=max(1, int(num_inits)),
                lr=self.ni_sorf_lr,
                seed=seed,
                device=core_device,
                dtype=self.dtype,
                freeze_epoch_nigp=self.ni_sorf_freeze_epoch_nigp,
                nigp_slope_refreshes=self.ni_sorf_nigp_slope_refreshes,
            )
            self.core = core
            best_loss = float(info.get("best_loss", float("nan")))
            mu, std = predict_ni_sorf(core, x, dtype=self.dtype)
            mu = mu.to(dtype=self.dtype)
            std = std.to(dtype=self.dtype)
        elif self.stage1 == "loo":
            from .loo_gp_core import make_loo_core

            core = make_loo_core(
                int(x.shape[-1]),
                int(x.shape[0]),
                backend=self.loo_backend,
                mean_terms=self.loo_mean_terms,
                nigp=self.loo_nigp,
                seed=seed,
                dtype=self.dtype,
            )
            core.fit(x.cpu(), y.cpu(), num_inits=max(1, int(num_inits)), seed=seed)
            self.core = core
            best_loss = float(core.best_loss)
            mu, std = core.loo_train()
            mu = mu.reshape(-1).detach().to(dtype=self.dtype).cpu()
            std = std.reshape(-1).detach().clamp_min(1e-4).to(dtype=self.dtype).cpu()
        else:
            from ..training.trainer import GPTrainer
            from ..training.eval import evaluate_gp_model

            x_d = x.to(device=device)
            y_d = y.to(device=device)
            input_dim = int(x_d.shape[-1])
            core = GPR(
                x_d,
                y_d,
                likelihood=LogGaussianLikelihood(),
                mean_module=gpytorch.means.ConstantMean(),
                kernel_module=LogScaleKernel(GaussianKernel(ard_num_dims=input_dim)),
            )
            trainer = GPTrainer(
                core,
                num_epochs=num_epochs,
                num_inits=num_inits,
                seed=seed,
                device=device,
                dtype=self.dtype,
                n_jobs=n_jobs if n_jobs is not None else 1,
            )
            runs = trainer.train()
            valid = [
                r
                for r in runs
                if r.get("loss") is not None and r.get("state_dict") is not None
            ]
            if not valid:
                raise RuntimeError("All GPTrainer runs failed for MidlineSparseMoE core.")
            best = min(valid, key=lambda r: r["loss"])
            core.load_state_dict(best["state_dict"])
            core.eval()
            self.core = core
            best_loss = float(best["loss"])
            mu, _lo, _hi, std = evaluate_gp_model(core, x_d)
            mu = mu.reshape(-1).detach().to(dtype=self.dtype).cpu()
            std = std.reshape(-1).detach().clamp_min(1e-4).to(dtype=self.dtype).cpu()

        self._mu0_train = mu.reshape(-1).detach().to(dtype=self.dtype)
        self._std0_train = std.reshape(-1).detach().clamp_min(1e-4).to(dtype=self.dtype)
        resid = (y.reshape(-1).cpu() - self._mu0_train.cpu()).detach()
        resid_std = float(resid.std().clamp_min(1e-3))
        cap = max(self.std0_cap_frac * resid_std, 0.05)
        self._std0_cap_value = float(cap)
        self._std0_stage2 = self._std0_train.clamp(max=cap)
        if self.het_stage1:
            self._fit_het_stage1(x.cpu(), resid, seed=seed)
            het = self._het_factor(x.cpu())
            assert het is not None
            het = het.to(dtype=self.dtype, device=self._std0_train.device)
            self._std0_train = self._std0_train * het
            self._std0_stage2 = self._std0_stage2 * het
        elif self.het_auto:
            # Fit h(x) on standardized residuals so it models the *shape* of the
            # variance relative to σ₀; the learned w decides how much to apply.
            z = resid / self._std0_train.cpu().clamp_min(1e-6)
            self._fit_het_stage1(x.cpu(), z, seed=seed)
            h = self._het_log(x.cpu())
            assert h is not None
            self._het_h_train = h.to(dtype=self.dtype, device=self._std0_train.device)
        self._core_fitted = True
        with torch.no_grad():
            self._stage1_gp_nll = float(
                self._gaussian_nll(
                    y.reshape(-1).to(dtype=self.dtype).cpu(),
                    self._mu0_train,
                    self._std0_train,
                ).item()
            )
        self._seed_offsets_from_residuals()
        logger.info(
            "MidlineSparseMoE Stage-1 core fitted (backend=%s, loss=%.4f, n=%d, "
            "resid_std=%.3f, stage2_std_cap=%.3f, gp_nll=%.4f, offset_type=%s).",
            self.midline_backend,
            float(best_loss),
            int(x.shape[0]),
            resid_std,
            cap,
            float(self._stage1_gp_nll),
            self.offset_type,
        )
        return self

    def _residual_quantile_seeds(self, *, sep_scale: float = 1.0) -> torch.Tensor:
        """Quantile seeds with min spacing ``rec_margin * 2 * σ_cap * sep_scale``."""
        r = (self.train_y.reshape(-1) - self._mu0_train).detach().cpu().numpy()
        qs = np.linspace(0.1, 0.9, num=max(self.k_max, 2))
        candidates = np.quantile(r, qs)
        if self.k_max >= 2:
            seeds = list(np.quantile(r, [0.25, 0.75]))
            for qv in candidates:
                if len(seeds) >= self.k_max:
                    break
                if all(abs(qv - s) >= 1e-8 for s in seeds):
                    seeds.append(float(qv))
            while len(seeds) < self.k_max:
                seeds.append(float(np.quantile(r, 0.5)))
            vals = np.asarray(seeds[: self.k_max], dtype=np.float64)
        else:
            vals = np.asarray([float(np.quantile(r, 0.5))], dtype=np.float64)

        # Enforce minimum spacing relative to Stage-2 σ cap (Behboodian-scale).
        # sep_scale < 1 shrinks forced multi-arm spacing when merge is preferred.
        cap = float(self._std0_stage2.max().item()) if self._core_fitted else float(
            max(np.std(r), 0.05)
        )
        min_sep = float(self.rec_margin) * 2.0 * max(cap, 0.05) * max(float(sep_scale), 0.0)
        order = np.argsort(vals)
        spaced = [float(vals[order[0]])]
        for idx in order[1:]:
            v = float(vals[idx])
            if min_sep <= 0.0 or abs(v - spaced[-1]) >= min_sep:
                spaced.append(v)
            else:
                spaced.append(spaced[-1] + min_sep)
        while len(spaced) < self.k_max:
            spaced.append(spaced[-1] + max(min_sep, 0.0))
        spaced = spaced[: self.k_max]
        arr = np.asarray(spaced, dtype=np.float64)
        arr = arr - arr.mean()
        return torch.as_tensor(arr, dtype=self.dtype)

    def _seed_offsets_from_residuals(self, *, sep_scale: float = 1.0) -> None:
        """Seed constant δ or MLP last-layer bias from residual quantiles."""
        vals = self._residual_quantile_seeds(sep_scale=sep_scale)
        if self.offset_type == "constant":
            assert self.delta is not None
            with torch.no_grad():
                self.delta.copy_(vals.to(device=self.delta.device))
        else:
            assert self.offset_net is not None
            self.offset_net.seed_bias(vals)

    # -------------------------------------------------------------- offsets
    def get_alpha(self, *, deterministic: bool | None = None) -> torch.Tensor:
        return self.alpha_gate(deterministic=deterministic)

    def get_gamma(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Variance carve-out ``γ ∈ (0,1)``.

        Returns ``[K]`` when ``gamma_type='constant'``, else ``[N, K]`` for
        ``γ_k(x)=sigmoid(MLP(x))``.
        """
        if self.gamma_type == "constant":
            assert self.raw_gamma is not None
            return torch.sigmoid(self.raw_gamma)
        assert self.gamma_net is not None
        if x is None:
            x = self.train_x
        return torch.sigmoid(self.gamma_net(x))

    def get_beta(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Variance scale: ``[K]`` ``β_k ≥ 1`` (≡1 unless ``free_expert_scale``).

        With ``var_expand`` and ``x`` given, returns ``[N,K]`` ``β_k · exp(s_k(x))``.
        """
        if not self.free_expert_scale:
            base = torch.ones(self.k_max, dtype=self.dtype, device=self.raw_beta.device)
        else:
            base = 1.0 + torch.nn.functional.softplus(self.raw_beta)
        if x is not None and self.scale_net is not None:
            return base.to(device=x.device).unsqueeze(0) * self.get_var_scale(x)
        return base

    def get_tau(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Deprecated alias: returns ``γ`` (multiplicative carve-out, not additive τ)."""
        return self.get_gamma(x)

    def get_pi(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            x = self.train_x
        return self.gating(x)

    def get_delta_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Unpinned offsets ``[N, K]``."""
        n = x.shape[0]
        if self.offset_type == "constant":
            assert self.delta is not None
            return self.delta.unsqueeze(0).expand(n, -1)
        assert self.offset_net is not None
        return self.offset_net(x)

    def _pinned_delta(
        self,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Pin so ``∑_k π_ik α_k δ_ik = 0`` for each row i (or globally if pi is [K])."""
        # delta: [N,K]; pi: [N,K] or [K]; alpha: [K]
        if pi.ndim == 1:
            w = (pi * alpha).clamp_min(0.0)
            w = w / w.sum().clamp_min(1e-12)
            center = (delta * w.unsqueeze(0)).sum(dim=-1, keepdim=True)
            return delta - center
        w = (pi * alpha.unsqueeze(0)).clamp_min(0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        center = (w * delta).sum(dim=-1, keepdim=True)
        return delta - center

    def get_delta(self, x: torch.Tensor, *, deterministic_alpha: bool = True) -> torch.Tensor:
        alpha = self.get_alpha(deterministic=deterministic_alpha)
        pi = self.get_pi(x)
        return self._pinned_delta(pi, alpha, self.get_delta_raw(x))

    def separation_scores(self, delta: torch.Tensor) -> torch.Tensor:
        """s_k from mean pairwise |δ_·k − δ_·j| over rows (supports [K] or [N,K])."""
        if delta.ndim == 1:
            d = delta.unsqueeze(0)
        else:
            d = delta
        n, k = d.shape
        if k == 1:
            return torch.ones(1, dtype=d.dtype, device=d.device)
        # mean absolute separation between expert curves
        mean_abs = []
        for i in range(k):
            mins = []
            for j in range(k):
                if i == j:
                    continue
                mins.append((d[:, i] - d[:, j]).abs().mean())
            mean_abs.append(torch.stack(mins).min())
        min_sep = torch.stack(mean_abs)
        return torch.tanh(min_sep / max(self.sep_length, 1e-6))

    def soft_usage(
        self,
        alpha: torch.Tensor,
        pi_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Soft used-slot gate ``ã_k = σ((α_k π̄_k - ε) / τ)`` (Option E)."""
        mass = alpha * pi_bar
        return torch.sigmoid((mass - self.keff_eps) / self.keff_tau)

    def compute_k_eff(
        self,
        alpha: torch.Tensor,
        pi: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(K_eff, soft_usage, s)`` with ``K_eff = Σ ã_k s_k``."""
        pi_bar = pi.mean(dim=0) if pi.ndim == 2 else pi
        usage = self.soft_usage(alpha, pi_bar)
        s = self.separation_scores(delta)
        return (usage * s).sum(), usage, s

    def k_eff(self, *, deterministic: bool = True) -> torch.Tensor:
        """Option E: ``K_eff = Σ_k σ((α_k π̄_k - ε)/τ) s_k`` (can exceed 1)."""
        alpha = self.get_alpha(deterministic=deterministic)
        pi = self.get_pi(self.train_x)
        delta = self._pinned_delta(pi, alpha, self.get_delta_raw(self.train_x))
        keff, _, _ = self.compute_k_eff(alpha, pi, delta)
        return keff

    def k_hat(self, eps: float | None = None) -> int:
        """Hard active count ``#{k : α_k π̄_k ≥ ε}`` (ε defaults to ``keff_eps``)."""
        thr = self.keff_eps if eps is None else float(eps)
        with torch.no_grad():
            alpha = self.get_alpha(deterministic=True)
            pi = self.get_pi(self.train_x).mean(dim=0)
            mass = alpha * pi
            return int((mass >= thr).sum().item())

    def _mixture_params(
        self,
        mu0: torch.Tensor,
        std0: torch.Tensor,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expert means/stds ``[N,K]``.

        Mean: ``μ_k = μ₀ + α_k δ_k``.
        Variance::

            Var_k = σ₀²(x) * (1 - α_k γ_k) * β_k + ε

        with ``γ`` shape ``[K]`` or ``[N,K]``, and ``β_k ≥ 1`` (``β≡1`` unless
        free scale is on).
        """
        if delta.ndim == 1:
            delta = delta.unsqueeze(0).expand(mu0.shape[0], -1)
        means = mu0.unsqueeze(-1) + alpha.unsqueeze(0) * delta
        # (1 - α γ) ∈ [0,1] when α,γ ∈ [0,1]
        g = gamma.unsqueeze(0) if gamma.ndim == 1 else gamma
        shrink = (1.0 - alpha.unsqueeze(0) * g).clamp(0.0, 1.0)
        if beta is None:
            beta = self.get_beta()
        scale = beta.unsqueeze(0) if beta.ndim == 1 else beta
        vars_ = std0.unsqueeze(-1).pow(2) * shrink * scale + self.var_floor
        stds = vars_.clamp_min(self.var_floor).sqrt()
        return means, stds

    def _std0_for_mixture(self) -> torch.Tensor:
        """GP predictive std used as variance ceiling in Stage 2."""
        if self.use_full_gp_variance:
            return self._std0_train
        return self._std0_stage2

    def _separation_penalty(
        self,
        delta: torch.Tensor,
        usage: torch.Tensor | None = None,
        margin: float = 2.0,
    ) -> torch.Tensor:
        """Push *used* expert curves apart: ``Σ_{i<j} ã_i ã_j softplus(m-|δ_i-δ_j|)``."""
        if delta.ndim == 1:
            d = delta.unsqueeze(0)
        else:
            d = delta
        k = d.shape[1]
        if k < 2:
            return d.new_zeros(())
        if usage is None:
            u = d.new_ones(k)
        else:
            u = usage.reshape(-1)
        pen = d.new_zeros(())
        for i in range(k):
            for j in range(i + 1, k):
                dist = (d[:, i] - d[:, j]).abs().mean()
                w = u[i] * u[j]
                pen = pen + w * torch.nn.functional.softplus(margin - dist)
        return pen

    def _orphan_penalty(self, alpha: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
        """Bidirectional π–α coupling.

        - ``α softplus(ε − π̄)``: α on with negligible gate mass (classic orphan)
        - ``π̄ softplus(ε − α)``: gate mass parked on α≈0 (dead-mass)
        """
        eps = self.keff_eps
        alpha_on_no_mass = alpha * torch.nn.functional.softplus(eps - pi_bar)
        mass_on_alpha_off = pi_bar * torch.nn.functional.softplus(eps - alpha)
        return alpha_on_no_mass.sum() + mass_on_alpha_off.sum()

    def _overlap_penalty(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        usage: torch.Tensor,
        *,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Push active components apart in standardized mean-gap units.

        For each pair ``i<j`` with usage weights ``ũ_i ũ_j``::

            gap_{ij} = |μ̄_i - μ̄_j| / (σ̄_i + σ̄_j + ε)
            pen += ũ_i ũ_j softplus(margin - gap_{ij})

        Does not prescribe a target ``K``; only discourages merged humps.
        """
        k = means.shape[1]
        if k < 2:
            return means.new_zeros(())
        mu = means.mean(dim=0)
        sig = stds.mean(dim=0).clamp_min(1e-4)
        u = usage.reshape(-1)
        pen = means.new_zeros(())
        for i in range(k):
            for j in range(i + 1, k):
                gap = (mu[i] - mu[j]).abs() / (sig[i] + sig[j])
                pen = pen + u[i] * u[j] * torch.nn.functional.softplus(margin - gap)
        return pen

    def fit_offsets(
        self,
        *,
        num_steps: int = 800,
        lr: float = 5e-2,
        lambda_k: float = 0.08,
        lambda_sep: float = 0.02,
        lambda_alpha: float = 0.05,
        lambda_orphan: float = 0.05,
        lambda_mode: float | None = None,
        lambda_overlap: float = 0.0,
        lambda_beta: float = 0.01,
        lambda_delta: float = 0.0,
        lambda_gamma: float = 0.0,
        lambda_rec: float = 0.0,
        sep_margin: float = 2.0,
        overlap_margin: float = 1.0,
        rec_margin: float = 1.0,
        lambda_k_start: float | None = None,
        seed: int = 0,
        log_every: int = 100,
        adaptive_merge_prune: bool = True,
        merge_eps: float = 0.05,
        merge_ema: float = 0.2,
        merge_ema_up: float | None = None,
        merge_ema_down: float | None = None,
        merge_temp: float = 0.08,
        merge_probe_every: int = 25,
        merge_probe_frac: float = 0.25,
        merge_probe_warmup: int = 50,
        merge_delta_pull: float = 0.25,
        merge_gamma_pull: float = 0.06,
        merge_k_boost: float = 3.0,
        merge_usage_cap: float = 0.15,
        alive_u_min: float = 0.9,
        alive_weight: float = 0.75,
        k_ref: float = 4.0,
        merge_pressure_init: float = 0.0,
    ) -> dict:
        """Stage 2: optimize offsets, Hard-Concrete gates, and mixture weights.

        If ``lambda_k_start`` is set, linearly anneal usage L0 weight from
        ``lambda_k_start`` → ``lambda_k`` (scaled by ``K_max/k_ref`` when
        ``adaptive_merge_prune``) over ``num_steps``.

        With ``adaptive_merge_prune=True``, an online holdout NLL merge probe
        compares the current mixture to midline merge candidates
        (``δ=0``, ``δ=0``+``γ=0``, and the Stage-1 GP) and sets soft
        ``merge_pressure`` ∈ [0,1], fading anti-merge weights
        ``λ_rec, λ_sep, λ_overlap, λ_γ``. Intermediate pressure permits
        partial mix. No mode/expert count targets or problem-id features.

        ``merge_pressure_init`` seeds the fade (e.g. 1.0 forces anti-merge
        weights off from step 1 for ablation).
        """
        if not self._core_fitted:
            raise RuntimeError("Call fit_core() before fit_offsets().")

        if lambda_mode is not None:
            self.mode_loss_weight = float(lambda_mode)

        torch.manual_seed(seed)
        self.train()
        params: list[nn.Parameter] = list(self.alpha_gate.parameters())
        if self.gamma_type == "constant":
            assert self.raw_gamma is not None
            params.append(self.raw_gamma)
        else:
            assert self.gamma_net is not None
            params.extend(list(self.gamma_net.parameters()))
        if self.free_expert_scale:
            params.append(self.raw_beta)
        cal_params: list[nn.Parameter] = []
        if self.scale_net is not None:
            if self.vexp_cal_only:
                cal_params.extend(list(self.scale_net.parameters()))
            else:
                params.extend(list(self.scale_net.parameters()))
        params.extend(list(self.gating.parameters()))
        if self.offset_type == "constant":
            assert self.delta is not None
            params.append(self.delta)
        else:
            assert self.offset_net is not None
            params.extend(list(self.offset_net.parameters()))
        use_cal = self.calib_frac > 0.0
        if use_cal:
            cal_params.append(self.log_temp)
        if self.het_auto:
            (cal_params if use_cal else params).append(self.raw_het_w)
        groups: list[dict] = [{"params": params}]
        if cal_params:
            groups.append({"params": cal_params, "lr": float(lr)})
        opt = torch.optim.Adam(groups, lr=lr)

        mu0 = self._mu0_train
        std0_cap = self._std0_for_mixture()
        std0_full = self._std0_train
        y = self.train_y.reshape(-1)
        x = self.train_x
        history: list[dict] = []
        adaptive = bool(adaptive_merge_prune)
        # Capacity prune: scale end λ_k with model K_max only (not data K*).
        lk_scale = max(1.0, float(self.k_max) / max(float(k_ref), 1e-6)) if adaptive else 1.0
        lk_start = float(lambda_k if lambda_k_start is None else lambda_k_start)
        lk_end = float(lambda_k) * lk_scale
        alive_w = float(alive_weight if adaptive else 0.5)
        u_min = float(alive_u_min if adaptive else 0.8)
        self.merge_pressure = float(np.clip(merge_pressure_init, 0.0, 1.0))
        m_t = float(self.merge_pressure)
        lam_sep0 = float(lambda_sep)
        lam_ov0 = float(lambda_overlap)
        lam_g0 = float(lambda_gamma)
        lam_rec0 = float(lambda_rec)
        # Fixed holdout indices for merge probe (universal; no problem id).
        n_obs = int(y.shape[0])
        frac = float(np.clip(merge_probe_frac, 0.05, 0.5))
        n_hold = max(8, int(round(frac * n_obs)))
        n_hold = min(n_hold, max(1, n_obs // 2))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 9049)
        hold_idx = torch.randperm(n_obs, generator=g)[:n_hold]
        hold_idx = hold_idx.to(device=y.device)
        warmup = max(0, int(merge_probe_warmup))
        late_step = max(warmup + 1, int(0.45 * num_steps))
        cal_gen = torch.Generator(device="cpu")
        cal_gen.manual_seed(int(seed) + 4242)
        n_cal = max(5, int(round(self.calib_frac * n_obs))) if use_cal else 0

        def _draw_cal() -> torch.Tensor:
            m = torch.zeros(n_obs, dtype=torch.bool)
            m[torch.randperm(n_obs, generator=cal_gen)[:n_cal]] = True
            return m.to(device=y.device)

        cal_mask = _draw_cal() if use_cal else None
        # Rows whose NLL never trained the networks (for held-out model choice).
        self._cal_mask = None if cal_mask is None or self.calib_rotate else cal_mask.cpu()
        het_h = self._het_h_train.to(device=y.device) if self.het_auto else None
        # Fixed residual-shape prior from Stage-1 midline (universal).
        resid_all = (y - mu0).detach().cpu().numpy()
        target_uni = _residual_unimodal_score(resid_all, bw_factor=0.55)

        for step in range(1, num_steps + 1):
            if num_steps <= 1 or lambda_k_start is None:
                lk_t = lk_end
            else:
                t = (step - 1) / float(num_steps - 1)
                lk_t = lk_start + t * (lk_end - lk_start)

            # Likelihood-driven fade of anti-merge geometry (no mode count).
            # Smoothstep on merge_pressure so mid-band pressure yields partial
            # mix (some overlap / shallower valleys) rather than binary on/off.
            if adaptive:
                t = float(np.clip(m_t, 0.0, 1.0))
                fade = (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t)  # 1 - smoothstep(t)
                # When merge is preferred, also sparsify harder (data-driven).
                lk_t = lk_t * (1.0 + float(merge_k_boost) * (t * t))
                # Restore full GP predictive width under merge (avoids twin peaks
                # from capped σ₀ with small residual offsets).
                std0 = (1.0 - t * t) * std0_cap + (t * t) * std0_full
            else:
                fade = 1.0
                std0 = std0_cap
            lam_sep_t = lam_sep0 * fade
            lam_ov_t = lam_ov0 * fade
            lam_g_t = lam_g0 * fade
            lam_rec_t = lam_rec0 * fade

            opt.zero_grad()
            alpha = self.get_alpha(deterministic=False)
            pi = self.get_pi(x)
            delta = self._pinned_delta(pi, alpha, self.get_delta_raw(x))
            gamma = self.get_gamma(x)
            beta = self.get_beta(x)
            w_het = torch.sigmoid(self.raw_het_w) if het_h is not None else None
            std0_fit = std0
            if het_h is not None:
                w_fit = w_het.detach() if use_cal else w_het
                std0_fit = std0 * torch.exp(0.5 * w_fit * het_h)
            beta_fit = beta.detach() if (self.vexp_cal_only and beta.ndim == 2) else beta
            means, stds = self._mixture_params(
                mu0, std0_fit, pi, alpha, delta, gamma, beta_fit
            )
            cal_loss = means.new_zeros(())
            if use_cal:
                assert cal_mask is not None
                if self.calib_rotate and step > 1:
                    cal_mask = _draw_cal()
                fm, cm = ~cal_mask, cal_mask
                nll = self._mixture_nll(y[fm], means[fm], stds[fm], pi[fm])
                # Calibration NLL: networks detached, only T / w / expansion learn.
                std0_c = std0[cm]
                if het_h is not None:
                    std0_c = std0_c * torch.exp(0.5 * w_het * het_h[cm])
                d_c = delta.detach()[cm] if delta.ndim == 2 else delta.detach()
                g_c = gamma.detach()[cm] if gamma.ndim == 2 else gamma.detach()
                if beta.ndim == 2:
                    b_c = beta[cm] if self.vexp_cal_only else beta.detach()[cm]
                else:
                    b_c = beta.detach()
                pi_c = pi.detach()[cm]
                m_c, s_c = self._mixture_params(
                    mu0[cm], std0_c, pi_c, alpha.detach(), d_c, g_c, b_c
                )
                s_c = s_c * torch.exp(self.log_temp)
                cal_loss = self._mixture_nll(y[cm], m_c.detach(), s_c, pi_c)
                if w_het is not None:
                    cal_loss = cal_loss + 0.01 * w_het
                if self.vexp_cal_only and beta.ndim == 2:
                    cal_loss = cal_loss + self.lambda_vexp * torch.log(b_c).mean()
            else:
                nll = self._mixture_nll(y, means, stds, pi)

            k_eff, usage, _s = self.compute_k_eff(alpha, pi, delta)
            pi_bar = pi.mean(dim=0)
            # Sparsity on soft *usage* Σã_k. Do not softplus(K_eff - 1):
            # that would punish a correct distinct count.
            usage_pen = usage.sum()
            sep_pen = self._separation_penalty(delta, usage=usage, margin=sep_margin)
            orphan_pen = self._orphan_penalty(alpha, pi_bar)
            alpha_pen = self.alpha_gate.complexity_loss()
            # Alive floor: keep ≥~u_min soft usage (prevents k_hat=0 collapse).
            # Relax alive floor only under strong merge pressure.
            alive_target = u_min * (
                1.0 - 0.85 * (float(m_t) ** 2) if adaptive else 1.0
            )
            alive_pen = torch.nn.functional.softplus(alive_target - usage.sum())
            overlap_pen = means.new_zeros(())
            if lam_ov_t > 0.0:
                overlap_pen = self._overlap_penalty(
                    means, stds, usage, margin=overlap_margin
                )
            beta_pen = means.new_zeros(())
            if self.free_expert_scale and lambda_beta > 0.0:
                beta_pen = (beta - 1.0).pow(2).mean()
            delta_pen = means.new_zeros(())
            if lambda_delta > 0.0:
                delta_pen = delta.pow(2).mean()
            gamma_pen = means.new_zeros(())
            if lam_g_t > 0.0:
                # usage is [K]; reduce γ(x) → mean over x when MLP.
                g_pen = gamma.mean(dim=0) if gamma.ndim == 2 else gamma
                gamma_pen = (usage * (1.0 - g_pen).pow(2)).sum()
            rec_pen = means.new_zeros(())
            if lam_rec_t > 0.0:
                rec_pen = self._overlap_penalty(
                    means, stds, usage, margin=float(rec_margin)
                )
            # Active merge attractor ~ m^3: mild pressure = partial mix only;
            # strong sustained merge collapses extras.
            merge_pull_pen = means.new_zeros(())
            if adaptive and m_t > 0.0:
                mt3 = float(m_t) ** 3
                merge_pull_pen = (
                    float(merge_delta_pull) * mt3 * delta.pow(2).mean()
                    + float(merge_gamma_pull) * mt3 * gamma.pow(2).mean()
                    + float(merge_usage_cap)
                    * mt3
                    * torch.nn.functional.softplus(usage.sum() - 1.05)
                )

            w_usage = lk_t * usage_pen
            w_sep = lam_sep_t * sep_pen
            w_alpha = lambda_alpha * alpha_pen
            w_orphan = lambda_orphan * orphan_pen
            w_alive = alive_w * alive_pen
            w_overlap = lam_ov_t * overlap_pen
            w_beta = lambda_beta * beta_pen
            w_delta = lambda_delta * delta_pen
            w_gamma = lam_g_t * gamma_pen
            w_rec = lam_rec_t * rec_pen
            w_merge_pull = merge_pull_pen

            loss = (
                nll
                + w_usage
                + w_sep
                + w_alpha
                + w_orphan
                + w_alive
                + w_overlap
                + w_beta
                + w_delta
                + w_gamma
                + w_rec
                + w_merge_pull
                + cal_loss
            )

            mode_term = y.new_zeros(())
            w_mode = y.new_zeros(())
            if self.mode_loss_weight > 0.0:
                mode_term = self._mode_peak_count_regularizer(
                    mu0, std0, pi, alpha, delta, gamma
                )
                w_mode = self.mode_loss_weight * mode_term
                loss = loss + w_mode

            loss.backward()
            opt.step()

            do_log = step % log_every == 0 or step == 1 or step == num_steps
            probe_every = max(1, int(merge_probe_every))
            do_probe = adaptive and step > warmup and (
                step == num_steps or step % probe_every == 0 or do_log
            )
            if do_probe:
                with torch.no_grad():
                    a_d = self.get_alpha(deterministic=True)
                    pi_d = self.get_pi(x)
                    d_d = self._pinned_delta(pi_d, a_d, self.get_delta_raw(x))
                    g_d = self.get_gamma(x)
                    b_d = self.get_beta(x)
                    hi = hold_idx
                    y_h = y[hi]
                    mu0_h = mu0[hi]
                    # Probe uses current blended std0 (merge restores full σ₀).
                    std0_probe = (1.0 - float(m_t)) * std0_cap + float(m_t) * std0_full
                    if het_h is not None:
                        std0_probe = std0_probe * torch.exp(
                            0.5 * torch.sigmoid(self.raw_het_w) * het_h
                        )
                    pi_h = pi_d[hi]
                    means_c, stds_c = self._mixture_params(
                        mu0, std0_probe, pi_d, a_d, d_d, g_d, b_d
                    )
                    nll_c = float(
                        self._mixture_nll(y_h, means_c[hi], stds_c[hi], pi_h).item()
                    )
                    zeros = torch.zeros_like(d_d)
                    g_zero = torch.zeros_like(g_d)
                    means_d0, stds_d0 = self._mixture_params(
                        mu0, std0_probe, pi_d, a_d, zeros, g_d, b_d
                    )
                    nll_d0 = float(
                        self._mixture_nll(
                            y_h, means_d0[hi], stds_d0[hi], pi_h
                        ).item()
                    )
                    means_ml, stds_ml = self._mixture_params(
                        mu0, std0_full, pi_d, a_d, zeros, g_zero, b_d
                    )
                    nll_ml = float(
                        self._mixture_nll(
                            y_h, means_ml[hi], stds_ml[hi], pi_h
                        ).item()
                    )
                    # Full Stage-1 predictive on holdout (uncapped σ₀).
                    nll_gp = float(
                        self._gaussian_nll(y_h, mu0_h, std0_full[hi]).item()
                    )
                    nll_m = min(nll_d0, nll_ml, nll_gp)
                    # Holdout NLL gap (soft) × residual unimodality (1- vs 2-GMM).
                    gap = nll_c - nll_m
                    temp = max(float(merge_temp), 1e-6)
                    z = (gap + float(merge_eps)) / temp
                    z = float(np.clip(z, -40.0, 40.0))
                    target_nll = 1.0 / (1.0 + float(np.exp(-z)))
                    # Residual unimodality is the structural prior; holdout NLL
                    # modulates. Multimodal residuals stay near 0.
                    target = float(
                        max(target_uni, 0.0) * (0.9 + 0.1 * max(target_nll, 0.0))
                    )
                    ema_up = float(
                        merge_ema if merge_ema_up is None else merge_ema_up
                    )
                    ema_dn = float(
                        merge_ema if merge_ema_down is None else merge_ema_down
                    )
                    # Rise fast; fall fast on clear reject; sticky while merge
                    # still plausible (keeps A05 merged once established).
                    # Late phase: do not start a brand-new merge.
                    if merge_ema_up is None and merge_ema_down is None:
                        ema_up = min(1.0, float(merge_ema) * 1.6)
                        if target < 0.25:
                            ema_dn = min(1.0, float(merge_ema) * 1.4)
                        elif m_t > 0.55:
                            ema_dn = max(1e-3, float(merge_ema) * 0.25)
                        else:
                            ema_dn = max(1e-3, float(merge_ema) * 0.7)
                    allow_rise = not (step >= late_step and m_t < 0.4)
                    if target >= m_t and allow_rise:
                        m_t = (1.0 - ema_up) * m_t + ema_up * target
                    elif target < m_t:
                        m_t = (1.0 - ema_dn) * m_t + ema_dn * target
                    m_t = float(np.clip(m_t, 0.0, 1.0))
                    self.merge_pressure = float(m_t)

            if do_log:
                with torch.no_grad():
                    d_log = self.get_delta(x, deterministic_alpha=True)
                    row = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "nll": float(nll.detach()),
                        "k_eff": float(k_eff.detach()),
                        "usage_sum": float(usage_pen.detach()),
                        "orphan": float(orphan_pen.detach()),
                        "delta_pen": float(delta_pen.detach()),
                        "gamma_pen": float(gamma_pen.detach()),
                        "rec_pen": float(rec_pen.detach()),
                        "lambda_k_t": float(lk_t),
                        "lambda_rec_t": float(lam_rec_t),
                        "lambda_sep_t": float(lam_sep_t),
                        "lambda_ov_t": float(lam_ov_t),
                        "lambda_g_t": float(lam_g_t),
                        "merge_pressure": float(m_t),
                        "k_hat": self.k_hat(),
                        "gate_type": self.gate_type,
                        "gamma_type": self.gamma_type,
                        "alpha": self.get_alpha(deterministic=True).detach().cpu().tolist(),
                        "gamma": (
                            self.get_gamma(x).mean(dim=0).detach().cpu().tolist()
                            if self.gamma_type == "mlp"
                            else self.get_gamma().detach().cpu().tolist()
                        ),
                        "beta": self.get_beta().detach().cpu().tolist(),
                        "delta_mean": d_log.mean(dim=0).detach().cpu().tolist(),
                        "mode_term": float(mode_term.detach())
                        if self.mode_loss_weight > 0
                        else 0.0,
                        # Weighted contributions (sum with nll ≈ loss)
                        "w_usage": float(w_usage.detach()),
                        "w_sep": float(w_sep.detach()),
                        "w_alpha": float(w_alpha.detach()),
                        "w_orphan": float(w_orphan.detach()),
                        "w_alive": float(w_alive.detach()),
                        "w_overlap": float(w_overlap.detach()),
                        "w_beta": float(w_beta.detach()),
                        "w_delta": float(w_delta.detach()),
                        "w_gamma": float(w_gamma.detach()),
                        "w_rec": float(w_rec.detach()),
                        "w_merge_pull": float(w_merge_pull.detach()),
                        "w_mode": float(w_mode.detach()),
                    }
                history.append(row)
                logger.info(
                    "MidlineSparseMoE Stage-2 step %d: loss=%.4f nll=%.4f "
                    "K_eff=%.3f usage=%.3f merge_p=%.2f K_hat=%d gate=%s",
                    step,
                    row["loss"],
                    row["nll"],
                    row["k_eff"],
                    row["usage_sum"],
                    row["merge_pressure"],
                    row["k_hat"],
                    self.gate_type,
                )

        self.eval()
        self.merge_pressure = float(m_t)
        self._fit_history = history
        final_nll = float(history[-1]["nll"]) if history else float("nan")
        return {
            "history": history,
            "k_eff": float(self.k_eff().detach()),
            "k_hat": self.k_hat(),
            "final_nll": final_nll,
            "merge_pressure": float(self.merge_pressure),
            "gate_type": self.gate_type,
            "lambda_k_end": float(lk_end),
            "stage1_gp_nll": self._stage1_gp_nll,
        }

    def _mode_peak_count_regularizer(
        self,
        mu0: torch.Tensor,
        std0: torch.Tensor,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        *,
        n_probe: int = 4,
        n_grid: int = 96,
    ) -> torch.Tensor:
        n = mu0.numel()
        idx = torch.linspace(0, n - 1, n_probe, device=mu0.device).long()
        means, stds = self._mixture_params(mu0, std0, pi, alpha, delta, gamma)
        d_span = delta.abs().max() if delta.ndim > 0 else delta.abs()
        y_lo = (mu0.min() - d_span - 3.0 * std0.max()).detach()
        y_hi = (mu0.max() + d_span + 3.0 * std0.max()).detach()
        y_grid = torch.linspace(
            float(y_lo), float(y_hi), n_grid, dtype=mu0.dtype, device=mu0.device
        )

        total = mu0.new_zeros(())
        for i in idx:
            m = means[i]
            s = stds[i]
            w = pi[i]
            dens = mu0.new_zeros(n_grid)
            for k in range(self.k_max):
                dens = dens + w[k] * torch.exp(
                    _gaussian_log_prob(y_grid, m[k].expand_as(y_grid), s[k].expand_as(y_grid))
                )
            total = total + _soft_peak_count_torch(dens)
        mean_count = total / float(n_probe)
        return torch.nn.functional.softplus(mean_count - 2.0)

    # -------------------------------------------------------------- predict
    def _core_predict(
        self, x: torch.Tensor, *, blend_merge: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.core is None:
            raise RuntimeError("Core GP not fitted.")
        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import predict_ni_sorf

            mu, std = predict_ni_sorf(self.core, x, dtype=self.dtype)
            mu = mu.reshape(-1).to(device=x.device, dtype=self.dtype)
            std_full = std.reshape(-1).clamp_min(1e-4).to(device=x.device, dtype=self.dtype)
        elif self.stage1 == "loo":
            mu, std = self.core.predict(x.cpu())
            mu = mu.reshape(-1).to(device=x.device, dtype=self.dtype)
            std_full = std.reshape(-1).clamp_min(1e-4).to(device=x.device, dtype=self.dtype)
        else:
            from ..training.eval import evaluate_gp_model

            mu, _lo, _hi, std = evaluate_gp_model(self.core, x)
            mu = mu.reshape(-1)
            std_full = std.reshape(-1).clamp_min(1e-4)
        # Mixture variance uses full GP σ₀ as ceiling unless capped mode requested.
        if self._core_fitted and not self.use_full_gp_variance:
            cap = (
                float(self._std0_cap_value)
                if self._std0_cap_value is not None
                else float(self._std0_stage2.max().item())
            )
            std_cap = std_full.clamp(max=max(cap, 1e-4))
        else:
            std_cap = std_full
        het = self._het_auto_factor(x) if self.het_auto else self._het_factor(x)
        if het is not None:
            het = het.to(device=std_full.device, dtype=std_full.dtype)
            std_full = std_full * het
            std_cap = std_cap * het
        if blend_merge and float(self.merge_pressure) > 0.0:
            t = float(np.clip(self.merge_pressure, 0.0, 1.0))
            return mu, (1.0 - t) * std_cap + t * std_full
        return mu, std_cap

    @torch.no_grad()
    def predict(self, test_x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        mu0, std0 = self._core_predict(test_x, blend_merge=True)
        alpha = self.get_alpha(deterministic=True)
        pi = self.get_pi(test_x)
        delta = self._pinned_delta(pi, alpha, self.get_delta_raw(test_x))
        gamma = self.get_gamma(test_x)
        beta = self.get_beta(test_x)
        means, stds = self._mixture_params(
            mu0, std0, pi, alpha, delta, gamma, beta
        )
        if self.calib_frac > 0.0:
            stds = stds * torch.exp(self.log_temp.detach()).to(device=stds.device)
        mix_mean = (pi * means).sum(dim=-1)
        second = (pi * (stds.pow(2) + means.pow(2))).sum(dim=-1)
        mix_var = (second - mix_mean.pow(2)).clamp_min(1e-12)
        return {
            "mean": mix_mean,
            "stddev": mix_var.sqrt(),
            "expert_means": means,
            "expert_stds": stds,
            "gate_weights": pi,
            "alpha": alpha.expand(test_x.shape[0], -1) if alpha.ndim == 1 else alpha,
            "mu0": mu0,
            "std0": std0,
            "delta": delta,
            "gamma": gamma,
            "beta": beta,
            "tau": gamma,  # deprecated alias
            "k_eff": self.k_eff(deterministic=True),
        }

    def dual_report(
        self,
        x_values: list[float] | np.ndarray,
        **kwargs,
    ) -> dict:
        import sys
        from pathlib import Path

        exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))
        from diagnostics import dual_report

        return dual_report(self, x_values, **kwargs)

    def fit(
        self,
        *,
        num_inits: int = 4,
        num_epochs: int = 1,
        num_steps: int = 800,
        lr: float = 5e-2,
        lambda_k: float = 0.08,
        lambda_sep: float = 0.02,
        lambda_alpha: float = 0.05,
        lambda_orphan: float = 0.05,
        lambda_mode: float = 0.0,
        lambda_overlap: float = 0.0,
        lambda_beta: float = 0.01,
        lambda_delta: float = 0.0,
        lambda_gamma: float = 0.0,
        lambda_rec: float = 0.0,
        sep_margin: float = 2.0,
        overlap_margin: float = 1.0,
        rec_margin: float = 1.0,
        lambda_k_start: float | None = None,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
        adaptive_merge_prune: bool = True,
        num_stage2_inits: int | None = None,
        collapse_nll_margin: float = 0.5,
        collapse_keff_min: float = 0.05,
        try_both_gates: bool = False,
        merge_pressure_init: float = 0.0,
        merge_eps: float = 0.05,
        merge_ema: float = 0.2,
        merge_ema_up: float | None = None,
        merge_ema_down: float | None = None,
        merge_temp: float = 0.08,
        merge_probe_every: int = 25,
        merge_probe_frac: float = 0.25,
        merge_probe_warmup: int = 50,
        merge_delta_pull: float = 0.25,
        merge_gamma_pull: float = 0.06,
        merge_k_boost: float = 3.0,
        merge_usage_cap: float = 0.15,
        alive_u_min: float = 0.9,
        alive_weight: float = 0.75,
        log_every: int = 100,
        stage2_device: str | None = None,
    ) -> dict:
        """Stage 1 then Stage 2 with optional likelihood-driven merge/prune.

        Stage-1 Exact GP uses ``device`` (typically CPU). Stage-2 Adam uses
        ``stage2_device`` (default CUDA if available, else CPU), then Stage-2
        tensors are moved back to CPU for predict parity with CPU ``x_te``.

        When ``adaptive_merge_prune`` is True:
          - Stage-2 runs ``num_stage2_inits`` inits (default **1**), each with
            the requested ``gate_type`` unless ``try_both_gates`` alternates
            constant vs MLP (off by default).
          - Inits with ``k_hat==0``, near-zero ``k_eff`` (total collapse), or
            mixture NLL worse than Stage-1 GP NLL + margin are rejected.
          - Best surviving init by train NLL is kept; if none survive, a
            safety refit with ``λ_k=0`` and elevated alive weight is used.
        """
        self.rec_margin = float(rec_margin)
        self.fit_core(
            num_inits=num_inits,
            num_epochs=num_epochs,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
        )
        adaptive = bool(adaptive_merge_prune)
        n_s2 = int(num_stage2_inits if num_stage2_inits is not None else 1)
        n_s2 = max(1, n_s2)
        if stage2_device is None:
            s2_dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            s2_dev = str(stage2_device)
        self._move_stage2_tensors(s2_dev)
        gp_nll = float(
            self._stage1_gp_nll if self._stage1_gp_nll is not None else float("inf")
        )
        gamma0 = float(self._gamma_init_logit)

        offset_kwargs = dict(
            num_steps=num_steps,
            lr=lr,
            lambda_k=lambda_k,
            lambda_sep=lambda_sep,
            lambda_alpha=lambda_alpha,
            lambda_orphan=lambda_orphan,
            lambda_mode=lambda_mode,
            lambda_overlap=lambda_overlap,
            lambda_beta=lambda_beta,
            lambda_delta=lambda_delta,
            lambda_gamma=lambda_gamma,
            lambda_rec=lambda_rec,
            sep_margin=sep_margin,
            overlap_margin=overlap_margin,
            rec_margin=rec_margin,
            lambda_k_start=lambda_k_start,
            adaptive_merge_prune=adaptive,
            merge_pressure_init=float(merge_pressure_init),
            merge_eps=float(merge_eps),
            merge_ema=float(merge_ema),
            merge_ema_up=merge_ema_up,
            merge_ema_down=merge_ema_down,
            merge_temp=float(merge_temp),
            merge_probe_every=int(merge_probe_every),
            merge_probe_frac=float(merge_probe_frac),
            merge_probe_warmup=int(merge_probe_warmup),
            merge_delta_pull=float(merge_delta_pull),
            merge_gamma_pull=float(merge_gamma_pull),
            merge_k_boost=float(merge_k_boost),
            merge_usage_cap=float(merge_usage_cap),
            alive_u_min=float(alive_u_min),
            alive_weight=float(alive_weight),
            log_every=int(log_every),
        )

        survivors: list[dict] = []
        rejected: list[dict] = []
        preferred_gate = self.gate_type

        for i in range(n_s2):
            if adaptive and try_both_gates:
                # Alternate MLP / constant; start with requested gate_type.
                gt: GateType = (
                    preferred_gate if (i % 2 == 0) else (
                        "constant" if preferred_gate == "mlp" else "mlp"
                    )
                )
            else:
                gt = preferred_gate
            self._reinit_stage2(
                seed=seed + 17 * (i + 1),
                gate_type=gt,
                seed_sep_scale=1.0,
                gamma_init_logit=gamma0,
            )
            # Rebuilt modules land on CPU; keep Stage-2 on ``s2_dev``.
            self._move_stage2_tensors(s2_dev)
            out = self.fit_offsets(seed=seed + 17 * (i + 1), **offset_kwargs)
            k_eff = float(out["k_eff"])
            k_hat = int(out["k_hat"])
            nll = float(out["final_nll"])
            collapsed = (
                k_hat == 0
                or k_eff < float(collapse_keff_min)
                or (np.isfinite(gp_nll) and nll > gp_nll + float(collapse_nll_margin))
            )
            record = {
                "init": i,
                "gate_type": gt,
                "final_nll": nll,
                "k_eff": k_eff,
                "k_hat": k_hat,
                "merge_pressure": float(out.get("merge_pressure", 0.0)),
                "collapsed": collapsed,
                "state": self._stage2_param_state(),
                "history": out.get("history", []),
                "cal_mask": getattr(self, "_cal_mask", None),
            }
            if collapsed and adaptive:
                rejected.append(
                    {k: record[k] for k in record if k not in ("state", "cal_mask")}
                )
                logger.info(
                    "MidlineSparseMoE Stage-2 init %d rejected "
                    "(gate=%s nll=%.4f gp_nll=%.4f k_eff=%.3f k_hat=%d).",
                    i,
                    gt,
                    nll,
                    gp_nll,
                    k_eff,
                    k_hat,
                )
            else:
                survivors.append(record)

        used_safety = False
        if survivors:
            best = min(survivors, key=lambda r: r["final_nll"])
            self._load_stage2_param_state(best["state"])
            self._fit_history = list(best["history"])
            self._cal_mask = best.get("cal_mask")
            final = {
                "history": self._fit_history,
                "k_eff": float(self.k_eff().detach()),
                "k_hat": self.k_hat(),
                "final_nll": float(best["final_nll"]),
                "merge_pressure": float(best["merge_pressure"]),
                "gate_type": best["gate_type"],
                "stage1_gp_nll": gp_nll,
                "num_stage2_inits": n_s2,
                "num_surviving_inits": len(survivors),
                "rejected_inits": rejected,
                "used_safety_refit": False,
            }
        else:
            # Safety net: no prune, strong alive floor, shrink seed spacing.
            used_safety = True
            self._reinit_stage2(
                seed=seed + 991,
                gate_type=preferred_gate,
                seed_sep_scale=0.0,
                gamma_init_logit=gamma0,
            )
            self._move_stage2_tensors(s2_dev)
            safe_kwargs = dict(offset_kwargs)
            safe_kwargs["lambda_k"] = 0.0
            safe_kwargs["lambda_k_start"] = None
            safe_kwargs["alive_weight"] = 2.0
            out = self.fit_offsets(seed=seed + 991, **safe_kwargs)
            final = {
                "history": out.get("history", []),
                "k_eff": float(out["k_eff"]),
                "k_hat": int(out["k_hat"]),
                "final_nll": float(out["final_nll"]),
                "merge_pressure": float(out.get("merge_pressure", 0.0)),
                "gate_type": self.gate_type,
                "stage1_gp_nll": gp_nll,
                "num_stage2_inits": n_s2,
                "num_surviving_inits": 0,
                "rejected_inits": rejected,
                "used_safety_refit": True,
            }
            self._fit_history = list(out.get("history", []))

        # Predict path expects Stage-2 on CPU alongside typical CPU ``x_te``.
        self._move_stage2_tensors("cpu")
        final["stage2_device"] = s2_dev
        final["try_both_gates"] = bool(try_both_gates)
        final["log_every"] = int(log_every)

        self._adaptive_diagnostics = {
            "adaptive_merge_prune": adaptive,
            "stage1_gp_nll": gp_nll,
            "rejected_inits": rejected,
            "num_surviving_inits": int(final["num_surviving_inits"]),
            "used_safety_refit": used_safety,
            "merge_pressure": float(final["merge_pressure"]),
            "gate_type": final["gate_type"],
            "stage2_device": s2_dev,
            "num_stage2_inits": n_s2,
            "try_both_gates": bool(try_both_gates),
        }
        logger.info(
            "MidlineSparseMoE fit done: gate=%s nll=%.4f k_eff=%.3f k_hat=%d "
            "merge_p=%.2f survivors=%d/%d safety=%s stage2_device=%s",
            final["gate_type"],
            final["final_nll"],
            final["k_eff"],
            final["k_hat"],
            final["merge_pressure"],
            final["num_surviving_inits"],
            n_s2,
            used_safety,
            s2_dev,
        )
        return final
