"""MidlineSparseHardXGate: MidlineSparse sibling with hard x-dependent routing.

Isolated from the champion MidlineSparseMoEGPR path (soft constant/MLP ``π``).
Experts are switched by a hard router ``w(x)`` that is always simplex-valued:

    w(x) ∈ Δ^{K-1},   w_k(x) ≥ 0,   ∑_k w_k(x) = 1

so ``p(y|x) = ∑_k w_k(x) N(y; μ_k, Var_k)`` remains a valid conditional density.

Training uses Gumbel-softmax with a straight-through hard one-hot; evaluation
uses deterministic argmax one-hot. Stage-1 midline GP, Hard-Concrete ``α_k``,
and offset experts are reused via subclassing without modifying the champion
gate API.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .midline_sparse_moe import MidlineSparseMoEGPR, OffsetType


class HardXGate(nn.Module):
    """MLP logits → hard simplex weights ``w(x)`` (top-1 one-hot).

    Parameters
    ----------
    gumbel_tau:
        Temperature for Gumbel-softmax during training (straight-through).
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dims: Sequence[int] = (32, 32),
        *,
        gumbel_tau: float = 0.5,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if gumbel_tau <= 0:
            raise ValueError(f"gumbel_tau must be > 0, got {gumbel_tau}")
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_experts)))
        self.network = nn.Sequential(*layers).to(dtype=dtype)
        self.num_experts = int(num_experts)
        self.gumbel_tau = float(gumbel_tau)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def _to_simplex(w: torch.Tensor) -> torch.Tensor:
        """Enforce ``w ≥ 0`` and row-sum 1 (defense in depth after one-hot)."""
        w = w.clamp_min(0.0)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor, *, hard: bool | None = None) -> torch.Tensor:
        """Return ``w(x)`` of shape ``[N, K]`` on the simplex.

        * Train (default): Gumbel-softmax straight-through hard one-hot.
        * Eval (default): deterministic argmax one-hot.
        """
        logits = self.logits(x)
        if hard is None:
            hard = not self.training
        if hard and not self.training:
            idx = logits.argmax(dim=-1)
            w = F.one_hot(idx, self.num_experts).to(dtype=logits.dtype)
            return self._to_simplex(w)
        # Straight-through: forward one-hot, backward through soft Gumbel-softmax.
        w = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
        return self._to_simplex(w)

    def soft_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax relaxation (for diagnostics only; not used in the mixture)."""
        return self._to_simplex(torch.softmax(self.logits(x) / self.gumbel_tau, dim=-1))


class MidlineSparseHardXGateMoEGPR(MidlineSparseMoEGPR):
    """Champion MidlineSparse mechanics + hard ``w(x)`` router (not soft ``π``)."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        k_max: int = 6,
        offset_type: OffsetType = "mlp",
        router_hidden: Sequence[int] = (32, 32),
        offset_hidden: Sequence[int] = (32, 32),
        gumbel_tau: float = 0.5,
        temperature: float = 1.0,  # unused; kept for API symmetry with parent
        tau_max: float = 3.0,
        sep_length: float = 2.0,
        var_floor: float = 1e-6,
        use_full_gp_variance: bool = True,
        free_expert_scale: bool = False,
        gamma_init_logit: float = 1.1,
        keff_eps: float = 0.05,
        keff_tau: float = 0.02,
    ):
        # Parent builds a soft MLP gate; we immediately replace it with HardXGate.
        super().__init__(
            train_x,
            train_y,
            k_max=k_max,
            gate_type="mlp",
            offset_type=offset_type,
            gating_hidden=router_hidden,
            offset_hidden=offset_hidden,
            temperature=temperature,
            tau_max=tau_max,
            sep_length=sep_length,
            var_floor=var_floor,
            use_full_gp_variance=use_full_gp_variance,
            free_expert_scale=free_expert_scale,
            gamma_init_logit=gamma_init_logit,
            keff_eps=keff_eps,
            keff_tau=keff_tau,
        )
        self.gumbel_tau = float(gumbel_tau)
        self.router_type = "hard_x"
        self.gate_type = "hard_x"  # type: ignore[assignment]
        self.gating = self._build_hard_x_gate()

    def _build_hard_x_gate(self) -> HardXGate:
        return HardXGate(
            input_dim=int(self.train_x.shape[-1]),
            num_experts=self.k_max,
            hidden_dims=self._gating_hidden,
            gumbel_tau=self.gumbel_tau,
            dtype=self.dtype,
        )

    def _build_gating(self, gate_type=None) -> nn.Module:  # type: ignore[override]
        """Always HardXGate — never install champion soft constant/MLP gates."""
        return self._build_hard_x_gate()

    def _set_gate_type(self, gate_type) -> None:  # type: ignore[override]
        """Ignore soft gate_type requests from parent multi-init logic."""
        self.gate_type = "hard_x"  # type: ignore[assignment]
        self.router_type = "hard_x"
        self.gating = self._build_hard_x_gate()

    def get_pi(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Hard simplex weights ``w(x)`` (alias kept for MidlineSparse call sites)."""
        if x is None:
            x = self.train_x
        return self.gating(x)

    def get_hard_weights(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            x = self.train_x
        return self.gating(x, hard=True)

    def set_gumbel_tau(self, tau: float) -> None:
        tau = float(tau)
        if tau <= 0:
            raise ValueError(f"gumbel_tau must be > 0, got {tau}")
        self.gumbel_tau = tau
        if isinstance(self.gating, HardXGate):
            self.gating.gumbel_tau = tau

    def fit(self, *args, try_both_gates: bool = False, **kwargs) -> dict:
        """Stage-1/2 fit without soft gate alternation."""
        kwargs.pop("try_both_gates", None)
        out = super().fit(*args, try_both_gates=False, **kwargs)
        out["gate_type"] = "hard_x"
        out["router_type"] = "hard_x"
        if hasattr(self, "_adaptive_diagnostics"):
            self._adaptive_diagnostics["gate_type"] = "hard_x"
            self._adaptive_diagnostics["router_type"] = "hard_x"
        return out

    @torch.no_grad()
    def predict(self, test_x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().predict(test_x)
        # Defense in depth: keep mixture weights on the simplex at every x.
        out["gate_weights"] = HardXGate._to_simplex(out["gate_weights"])
        return out
