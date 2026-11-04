"""Phase 2: Midline + residual MoGP with Bayesian (approximate) joints.

Separate module from Phase 1
(:class:`~gpplus.models.midline_residual_mogp.MidlineResidualMoGP`) and from the
champion MidlineSparse. Implements the foundations §``sec:code-to-posterior``
sketch: approximate a posterior over mixture / residual hyperparameters and
return an MC posterior predictive — including multi-input joints.

Default inference: **Laplace** (diagonal) over gate + Hard-Concrete + γ after a
Phase-1-style residual-GP EM fit, then MC over θ. Residual expert kernels stay
plug-in (EB) unless ``refit_experts_each_draw`` is enabled (expensive).

This is intentionally **not** a subclass that mutates Phase 1 in-place.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from ..config import logger
from .midline_residual_mogp import MidlineResidualMoGP


class MidlineBayesMoGP(nn.Module):
    """Bayesian joints via Laplace / ensemble over Stage~2 parameters θ."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        k_max: int = 5,
        gate_type: Literal["constant", "mlp"] = "constant",
        dtype: torch.dtype = torch.float64,
        std0_cap_frac: float = 0.3,
        midline_backend: Literal["exact", "ni_sorf"] = "exact",
        **ni_sorf_kwargs: Any,
    ):
        super().__init__()
        self.dtype = dtype
        self.base = MidlineResidualMoGP(
            train_x,
            train_y,
            k_max=k_max,
            gate_type=gate_type,
            dtype=dtype,
            std0_cap_frac=std0_cap_frac,
            midline_backend=midline_backend,
            **ni_sorf_kwargs,
        )
        self._laplace_mean: dict[str, torch.Tensor] | None = None
        self._laplace_std: dict[str, torch.Tensor] | None = None
        self._map_state: dict[str, torch.Tensor] | None = None
        self._fitted = False

    @property
    def k_max(self) -> int:
        return self.base.k_max

    def _theta_param_list(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(list(self.base.gate.parameters()))
        params.extend(list(self.base.hc.parameters()))
        params.append(self.base.raw_gamma)
        return params

    def _snapshot_theta(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, p in self.base.named_parameters():
            if name.startswith("residual_experts") or name.startswith("core"):
                continue
            out[name] = p.detach().cpu().clone()
        return out

    def _load_theta(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in self.base.named_parameters():
                if name in state:
                    p.copy_(state[name].to(device=p.device, dtype=p.dtype))

    def fit(
        self,
        *,
        num_inits: int = 2,
        num_em_iters: int = 3,
        num_gate_steps: int = 400,
        lr: float = 5e-2,
        lambda_alpha: float = 0.05,
        laplace_steps: int = 80,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
    ) -> dict[str, Any]:
        """Fit Phase-1 residual MoGP MAP, then diagonal Laplace over θ."""
        hist = self.base.fit(
            num_inits=num_inits,
            num_em_iters=num_em_iters,
            num_gate_steps=num_gate_steps,
            lr=lr,
            lambda_alpha=lambda_alpha,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
        )
        self._map_state = self._snapshot_theta()

        # Diagonal Fisher / Hessian proxy via squared grads of NLL at MAP
        self.base.to(device=device)
        self.base.train()
        x = self.base.train_x.to(device=device)
        y = self.base.train_y.to(device=device)
        sq_grad: dict[str, torch.Tensor] = {}
        for _ in range(int(laplace_steps)):
            self.base.zero_grad(set_to_none=True)
            pi, means, stds, _, _ = self.base._mixture_params(x)
            # Stochastic HC for proper posterior mass on α
            alpha = self.base.get_alpha(deterministic=False)
            # rebuild means with stochastic α (same formula as Phase 1 gate step)
            mu0 = self.base._mu0_train.to(device=device)
            std0 = self.base._std0_stage2.to(device=device)
            assert mu0 is not None and std0 is not None
            # ExactGP residual experts stay on CPU.
            d_mean, d_std = self.base._expert_predict_marginal(x.detach().cpu())
            d_mean = d_mean.to(device=device)
            d_std = d_std.to(device=device)
            gamma = self.base.get_gamma().to(device=device)
            mass = (pi.detach() * alpha.unsqueeze(0)).clamp_min(1e-12)
            mass = mass / mass.sum(dim=-1, keepdim=True)
            d_pin = d_mean - (mass * d_mean).sum(dim=-1, keepdim=True)
            means_s = mu0.unsqueeze(-1) + alpha.unsqueeze(0) * d_pin
            carve = std0.unsqueeze(-1).pow(2) * (
                1.0 - alpha.unsqueeze(0) * gamma.unsqueeze(0)
            )
            stds_s = ((alpha.unsqueeze(0) * d_std).pow(2) + carve + 1e-6).sqrt()
            from .midline_residual_mogp import _gaussian_log_prob

            log_comp = _gaussian_log_prob(y.unsqueeze(-1), means_s, stds_s)
            nll = -(
                torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_comp, dim=-1)
            ).mean()
            loss = nll + float(lambda_alpha) * self.base.hc.complexity_loss()
            loss.backward()
            for name, p in self.base.named_parameters():
                if p.grad is None:
                    continue
                if name.startswith("residual_experts") or name.startswith("core"):
                    continue
                g2 = p.grad.detach().pow(2)
                if name not in sq_grad:
                    sq_grad[name] = g2.clone()
                else:
                    sq_grad[name] = sq_grad[name] + g2

        mean_state = self._snapshot_theta()
        std_state: dict[str, torch.Tensor] = {}
        for name, acc in sq_grad.items():
            fisher = acc / max(float(laplace_steps), 1.0)
            # diag Laplace: σ ≈ (n * Fisher + λ)^{-1/2}; λ floor for stability
            std_state[name] = (fisher + 1e-4).pow(-0.5).cpu()
            mean_state[name] = mean_state[name]

        self._laplace_mean = mean_state
        self._laplace_std = std_state
        self._load_theta(self._map_state)
        self.base.to(device="cpu")
        self.base.eval()
        self._fitted = True
        logger.info(
            "MidlineBayesMoGP: MAP + diagonal Laplace over %d theta tensors "
            "(residual experts remain plug-in EB).",
            len(std_state),
        )
        return {
            "phase1_history": hist,
            "laplace_param_names": sorted(std_state.keys()),
            "note": (
                "Axis A approximation: p(theta|D, residual GPs, mu0). "
                "Not full p(f0,theta|D). Controllers unused."
            ),
        }

    def _draw_theta(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        assert self._laplace_mean is not None and self._laplace_std is not None
        draw: dict[str, torch.Tensor] = {}
        for name, mu in self._laplace_mean.items():
            sd = self._laplace_std.get(name)
            if sd is None:
                draw[name] = mu.clone()
                continue
            eps = torch.randn(
                mu.shape, dtype=mu.dtype, generator=generator
            )
            draw[name] = mu + sd * eps
        return draw

    @torch.no_grad()
    def predict(
        self,
        test_x: torch.Tensor,
        *,
        n_posterior_draws: int = 32,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """MC posterior predictive margins (mixture-of-mixtures → MoG moments)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        g = torch.Generator()
        g.manual_seed(int(seed))
        test_x = test_x.to(dtype=self.dtype)
        if test_x.ndim == 1:
            test_x = test_x.unsqueeze(-1)

        # Accumulate mixture moments across θ draws
        mix_means = []
        mix_second = []
        last_pi = last_means = last_stds = None
        for _ in range(int(n_posterior_draws)):
            self._load_theta(self._draw_theta(generator=g))
            pred = self.base.predict(test_x)
            pi = pred["gate_weights"]
            means = pred["expert_means"]
            stds = pred["expert_stds"]
            m = (pi * means).sum(dim=-1)
            second = (pi * (stds.pow(2) + means.pow(2))).sum(dim=-1)
            mix_means.append(m)
            mix_second.append(second)
            last_pi, last_means, last_stds = pi, means, stds

        assert self._map_state is not None
        self._load_theta(self._map_state)
        mm = torch.stack(mix_means, dim=0).mean(dim=0)
        second = torch.stack(mix_second, dim=0).mean(dim=0)
        var = (second - mm.pow(2)).clamp_min(1e-12)
        # Also return MAP mixture components for ScoringBench density scores
        assert last_pi is not None and last_means is not None and last_stds is not None
        map_pred = self.base.predict(test_x)
        return {
            "mean": mm,
            "stddev": var.sqrt(),
            "expert_means": map_pred["expert_means"],
            "expert_stds": map_pred["expert_stds"],
            "gate_weights": map_pred["gate_weights"],
            "ppd_mean": mm,
            "ppd_stddev": var.sqrt(),
            "n_posterior_draws": torch.tensor(float(n_posterior_draws)),
        }

    @torch.no_grad()
    def predict_joint(
        self,
        test_x: torch.Tensor,
        *,
        n_posterior_draws: int = 16,
        seed: int = 0,
    ) -> dict[str, Any]:
        """MC average of Phase-1 shared-assignment joints (mixture-of-mixtures).

        Returns stacked component weights/means/covs per draw for downstream
        sampling, plus a mean-field moment summary.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_joint().")
        g = torch.Generator()
        g.manual_seed(int(seed))
        draws = []
        for _ in range(int(n_posterior_draws)):
            self._load_theta(self._draw_theta(generator=g))
            draws.append(self.base.predict_joint(test_x))
        assert self._map_state is not None
        self._load_theta(self._map_state)

        # Equal-weight mixture over (draw, expert) pairs
        weights = []
        means = []
        covs = []
        for d in draws:
            w = d["component_weights"] / float(n_posterior_draws)
            for k in range(w.numel()):
                weights.append(w[k])
                means.append(d["means"][k])
                covs.append(d["covs"][k])
        return {
            "component_weights": torch.stack(weights),
            "means": torch.stack(means),
            "covs": torch.stack(covs),
            "n_posterior_draws": int(n_posterior_draws),
            "note": "Laplace-MC over theta; residual GPs and midline frozen (partial Bayes).",
        }

    @torch.no_grad()
    def sample(
        self,
        test_x: torch.Tensor,
        n_samples: int = 1,
        *,
        n_posterior_draws: int = 16,
        seed: int = 0,
    ) -> torch.Tensor:
        joint = self.predict_joint(
            test_x, n_posterior_draws=n_posterior_draws, seed=seed
        )
        w = joint["component_weights"]
        means = joint["means"]
        covs = joint["covs"]
        n = means.shape[-1]
        g = torch.Generator()
        g.manual_seed(int(seed) + 1)
        out = torch.zeros(n_samples, n, dtype=self.dtype)
        cat = torch.distributions.Categorical(probs=w)
        for s in range(int(n_samples)):
            k = int(cat.sample((1,), generator=g).item())
            mvn = torch.distributions.MultivariateNormal(
                means[k],
                covariance_matrix=covs[k]
                + 1e-5 * torch.eye(n, dtype=self.dtype),
            )
            out[s] = mvn.sample((1,), generator=g).reshape(-1)
        return out
