"""Phase 1: Midline + residual MoGP (estimative joints).

Sibling of champion :class:`~gpplus.models.midline_sparse_moe.MidlineSparseMoEGPR`.
Does **not** modify the champion. Stage~1 Exact GP midline is frozen; Stage~2
fits residual GP experts under a sparse gate so multi-input FDDs exist via
expert Gram matrices.

Returned object is still **EBMAP / estimative** (plug-in residual GPs + gate),
not a Bayesian posterior predictive. Use
:class:`~gpplus.models.midline_bayes_mogp.MidlineBayesMoGP` for Phase 2.
"""

from __future__ import annotations

from typing import Any, Literal

import gpytorch
import numpy as np
import torch
from torch import nn

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood
from ..training.eval import evaluate_gp_model
from ..training.trainer import GPTrainer
from .gpr import GPR
from .midline_sparse_moe import HardConcreteGate
from .moe_gpr import ConstantGate, GatingNetwork, _build_weighted_expert_gpr

GateType = Literal["constant", "mlp"]
MidlineBackend = Literal["exact", "ni_sorf"]


def _gaussian_log_prob(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = (std * std).clamp_min(1e-12)
    return -0.5 * (torch.log(2.0 * torch.pi * var) + (y - mean).pow(2) / var)


class MidlineResidualMoGP(nn.Module):
    """Two-stage estimative residual mixture of GP experts on a frozen midline."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        k_max: int = 5,
        gate_type: GateType = "constant",
        dtype: torch.dtype = torch.float64,
        std0_cap_frac: float = 0.3,
        midline_backend: MidlineBackend = "exact",
        ni_sorf_num_rff: int = 2400,
        ni_sorf_num_epochs: int = 1000,
        ni_sorf_residual_num_rff: int = 1024,
        ni_sorf_residual_num_epochs: int = 300,
        ni_sorf_lr: float = 1e-2,
    ):
        super().__init__()
        if midline_backend not in ("exact", "ni_sorf"):
            raise ValueError(
                f"midline_backend must be 'exact' or 'ni_sorf', got {midline_backend!r}"
            )
        self.dtype = dtype
        self.train_x = train_x.to(dtype=dtype)
        if self.train_x.ndim == 1:
            self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = train_y.to(dtype=dtype).reshape(-1)
        self.k_max = int(k_max)
        self.gate_type = gate_type
        self.std0_cap_frac = float(std0_cap_frac)
        self.midline_backend: MidlineBackend = midline_backend
        self.ni_sorf_num_rff = int(ni_sorf_num_rff)
        self.ni_sorf_num_epochs = int(ni_sorf_num_epochs)
        self.ni_sorf_residual_num_rff = int(ni_sorf_residual_num_rff)
        self.ni_sorf_residual_num_epochs = int(ni_sorf_residual_num_epochs)
        self.ni_sorf_lr = float(ni_sorf_lr)

        d = int(self.train_x.shape[-1])
        if gate_type == "constant":
            self.gate: nn.Module = ConstantGate(self.k_max, dtype=dtype)
        else:
            self.gate = GatingNetwork(d, self.k_max, dtype=dtype)
        self.hc = HardConcreteGate(self.k_max, dtype=dtype)
        self.raw_gamma = nn.Parameter(torch.full((self.k_max,), 2.2, dtype=dtype))

        self.core: Any = None
        self.residual_experts: nn.ModuleList = nn.ModuleList()
        self._mu0_train: torch.Tensor | None = None
        self._std0_train: torch.Tensor | None = None
        self._std0_stage2: torch.Tensor | None = None
        self._core_fitted = False
        self._experts_fitted = False

    def get_alpha(self, *, deterministic: bool | None = None) -> torch.Tensor:
        return self.hc(deterministic=deterministic)

    def get_pi(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

    def get_gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma)

    def fit_core(
        self,
        *,
        num_inits: int = 2,
        num_epochs: int = 1,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
    ) -> "MidlineResidualMoGP":
        best_loss = float("nan")
        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import fit_ni_sorf_core, predict_ni_sorf

            core_device = "cuda" if torch.cuda.is_available() else device
            core, info = fit_ni_sorf_core(
                self.train_x,
                self.train_y,
                num_rff=self.ni_sorf_num_rff,
                num_epochs=self.ni_sorf_num_epochs,
                num_inits=max(1, int(num_inits)),
                lr=self.ni_sorf_lr,
                seed=seed,
                device=core_device,
                dtype=self.dtype,
            )
            self.core = core
            best_loss = float(info.get("best_loss", float("nan")))
            mu, std = predict_ni_sorf(core, self.train_x, dtype=self.dtype)
        else:
            x = self.train_x.to(device=device)
            y = self.train_y.to(device=device)
            input_dim = int(x.shape[-1])
            core = GPR(
                x,
                y,
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
                raise RuntimeError("All GPTrainer runs failed for MidlineResidualMoGP core.")
            best = min(valid, key=lambda r: r["loss"])
            core.load_state_dict(best["state_dict"])
            core.eval()
            self.core = core.cpu()
            best_loss = float(best["loss"])
            mu, _lo, _hi, std = evaluate_gp_model(self.core, self.train_x.cpu())
            mu = mu.reshape(-1).detach()
            std = std.reshape(-1).detach()

        self._mu0_train = mu.reshape(-1).detach().to(device="cpu", dtype=self.dtype)
        self._std0_train = (
            std.reshape(-1).detach().clamp_min(1e-4).to(device="cpu", dtype=self.dtype)
        )
        resid = (self.train_y.reshape(-1).cpu() - self._mu0_train).detach()
        resid_std = float(resid.std().clamp_min(1e-3))
        cap = max(self.std0_cap_frac * resid_std, 0.05)
        self._std0_stage2 = self._std0_train.clamp(max=cap)
        self._core_fitted = True
        logger.info(
            "MidlineResidualMoGP Stage-1 fitted (backend=%s, loss=%.4f, n=%d, resid_std=%.3f).",
            self.midline_backend,
            float(best_loss),
            int(self.train_x.shape[0]),
            resid_std,
        )
        return self

    def _fit_residual_experts(
        self,
        responsibilities: torch.Tensor,
        *,
        num_inits: int = 1,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        assert self._mu0_train is not None
        x = self.train_x.to(dtype=self.dtype)
        resid = (
            self.train_y.to(dtype=self.dtype) - self._mu0_train.to(dtype=self.dtype)
        ).reshape(-1)
        resp = responsibilities.detach().to(dtype=self.dtype)
        experts: list[Any] = []

        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import fit_ni_sorf_core

            core_device = "cuda" if torch.cuda.is_available() else device
            n = int(x.shape[0])
            g = torch.Generator()
            for k in range(self.k_max):
                w = resp[:, k].clamp_min(1e-8)
                w = w / w.sum()
                g.manual_seed(int(seed) + 17 * k)
                # Responsibility-weighted bootstrap ≈ weighted NLL for NI-SORF
                idx = torch.multinomial(w, num_samples=n, replacement=True, generator=g)
                expert, _info = fit_ni_sorf_core(
                    x[idx],
                    resid[idx],
                    num_rff=self.ni_sorf_residual_num_rff,
                    num_epochs=self.ni_sorf_residual_num_epochs,
                    num_inits=max(1, int(num_inits)),
                    lr=self.ni_sorf_lr,
                    seed=seed + k,
                    device=core_device,
                    dtype=self.dtype,
                    freeze_epoch_nigp=min(50, self.ni_sorf_residual_num_epochs // 3),
                    nigp_slope_refreshes=5,
                )
                experts.append(expert)
        else:
            expert_device = "cpu"
            x_cpu = x.to(device=expert_device)
            resid_cpu = resid.to(device=expert_device)
            resp_cpu = resp.to(device=expert_device)
            for k in range(self.k_max):
                w = resp_cpu[:, k]
                if float(w.sum().item()) < 1e-3:
                    w = torch.full_like(w, 1.0 / max(self.train_x.shape[0], 1))
                expert = _build_weighted_expert_gpr(x_cpu, resid_cpu, w, ard=True)
                trainer = GPTrainer(
                    expert,
                    num_epochs=1,
                    num_inits=num_inits,
                    seed=seed + k,
                    device=expert_device,
                    dtype=self.dtype,
                    n_jobs=1,
                )
                runs = trainer.train()
                valid = [
                    r
                    for r in runs
                    if r.get("loss") is not None and r.get("state_dict") is not None
                ]
                if valid:
                    best = min(valid, key=lambda r: r["loss"])
                    expert.load_state_dict(best["state_dict"])
                expert.eval()
                experts.append(expert.cpu())

        self.residual_experts = nn.ModuleList(experts)
        self._experts_fitted = True

    def _expert_predict_marginal(
        self, test_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual expert means/stds at test_x: each [N, K]."""
        assert self.core is not None and self._experts_fitted
        n = test_x.shape[0]
        means = torch.zeros(n, self.k_max, dtype=self.dtype, device=test_x.device)
        stds = torch.zeros(n, self.k_max, dtype=self.dtype, device=test_x.device)
        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import predict_ni_sorf

            for k, expert in enumerate(self.residual_experts):
                mu, std = predict_ni_sorf(expert, test_x, dtype=self.dtype)
                means[:, k] = mu.reshape(-1).to(device=test_x.device, dtype=self.dtype)
                stds[:, k] = std.reshape(-1).clamp_min(1e-4).to(
                    device=test_x.device, dtype=self.dtype
                )
        else:
            for k, expert in enumerate(self.residual_experts):
                mu, _lo, _hi, std = evaluate_gp_model(expert, test_x)
                means[:, k] = mu.reshape(-1).to(dtype=self.dtype)
                stds[:, k] = std.reshape(-1).clamp_min(1e-4).to(dtype=self.dtype)
        return means, stds

    def _core_predict(self, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.core is not None
        if self.midline_backend == "ni_sorf":
            import sys
            from pathlib import Path

            exp = Path(__file__).resolve().parents[2] / "experiments_MOEGP" / "feasibility_mode_count"
            if str(exp) not in sys.path:
                sys.path.insert(0, str(exp))
            from ni_sorf_midline import predict_ni_sorf

            mu, std = predict_ni_sorf(self.core, test_x, dtype=self.dtype)
            return mu.reshape(-1).to(dtype=self.dtype), std.reshape(-1).clamp_min(1e-4).to(
                dtype=self.dtype
            )
        mu, _lo, _hi, std = evaluate_gp_model(self.core, test_x)
        return (
            mu.reshape(-1).to(dtype=self.dtype),
            std.reshape(-1).clamp_min(1e-4).to(dtype=self.dtype),
        )

    def _mixture_params(
        self, test_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ExactGP core/experts live on CPU; gate may be on CUDA during Stage-2.
        gate_dev = next(self.gate.parameters()).device
        x_cpu = test_x.to(device="cpu", dtype=self.dtype)
        if x_cpu.ndim == 1:
            x_cpu = x_cpu.unsqueeze(-1)
        x_gate = x_cpu.to(device=gate_dev)
        mu0, std0 = self._core_predict(x_cpu)
        mu0 = mu0.to(device=gate_dev)
        std0 = std0.to(device=gate_dev)
        alpha = self.get_alpha(deterministic=True).to(device=gate_dev)
        pi = self.get_pi(x_gate)
        gamma = self.get_gamma().to(device=gate_dev)
        delta_mean, delta_std = self._expert_predict_marginal(x_cpu)
        delta_mean = delta_mean.to(device=gate_dev)
        delta_std = delta_std.to(device=gate_dev)
        # Pin residual means so mixture mean tracks midline (π α δ = 0)
        mass = (pi * alpha.unsqueeze(0)).clamp_min(1e-12)
        mass = mass / mass.sum(dim=-1, keepdim=True)
        delta_pin = delta_mean - (mass * delta_mean).sum(dim=-1, keepdim=True)
        means = mu0.unsqueeze(-1) + alpha.unsqueeze(0) * delta_pin
        # Variance: residual GP scale blended with midline envelope carve-out
        carve = std0.unsqueeze(-1).pow(2) * (1.0 - alpha.unsqueeze(0) * gamma.unsqueeze(0))
        vars_k = (alpha.unsqueeze(0) * delta_std).pow(2) + carve + 1e-6
        stds = vars_k.sqrt()
        return pi, means, stds, mu0, std0

    def fit(
        self,
        *,
        num_inits: int = 2,
        num_em_iters: int = 3,
        num_gate_steps: int = 400,
        lr: float = 5e-2,
        lambda_alpha: float = 0.05,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
    ) -> dict:
        """Stage~1 freeze + EM-style residual GP experts + gate/α NLL."""
        self.fit_core(
            num_inits=num_inits, seed=seed, device=device, n_jobs=n_jobs
        )
        assert self._mu0_train is not None and self._std0_stage2 is not None
        n = self.train_x.shape[0]
        # Seed responsibilities from residual quantiles (all on CPU).
        mu0_cpu = self._mu0_train.detach().cpu()
        resid = (self.train_y.detach().cpu() - mu0_cpu).numpy()
        qs = np.linspace(0.15, 0.85, num=self.k_max)
        centers = np.quantile(resid, qs)
        r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)
        dist = (r - centers.reshape(1, -1)) ** 2
        soft = np.exp(-dist / (2.0 * max(float(np.var(resid)), 1e-3) + 1e-6))
        soft = soft / soft.sum(axis=1, keepdims=True)
        resp = torch.as_tensor(soft, dtype=self.dtype)

        history: list[dict] = []
        # Gate Adam may use CUDA, but ExactGP experts + midline caches stay on CPU.
        gate_device = device
        x = self.train_x
        y = self.train_y
        for em in range(int(num_em_iters)):
            self._fit_residual_experts(
                resp, num_inits=1, seed=seed + em, device="cpu"
            )
            # Move gate / HC / gamma only (experts stay CPU).
            self.gate.to(device=gate_device)
            self.hc.to(device=gate_device)
            self.raw_gamma.data = self.raw_gamma.data.to(device=gate_device)
            x_d = x.to(device=gate_device)
            y_d = y.to(device=gate_device)
            opt = torch.optim.Adam(
                list(self.gate.parameters())
                + list(self.hc.parameters())
                + [self.raw_gamma],
                lr=lr,
            )
            for step in range(int(num_gate_steps)):
                self.train()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Expert marginals are non-diff w.r.t. gate (EM outer);
                    # evaluate ExactGP experts on CPU, then move to gate_device.
                    x_cpu = x.to(device="cpu")
                    pi_det = self.get_pi(x_d)
                    alpha_det = self.get_alpha(deterministic=True)
                    gamma = self.get_gamma()
                    mu0 = self._mu0_train.to(device=gate_device)
                    std0 = self._std0_stage2.to(device=gate_device)
                    d_mean, d_std = self._expert_predict_marginal(x_cpu)
                    d_mean = d_mean.to(device=gate_device)
                    d_std = d_std.to(device=gate_device)
                    mass = (pi_det * alpha_det.unsqueeze(0)).clamp_min(1e-12)
                    mass = mass / mass.sum(dim=-1, keepdim=True)
                    d_pin = d_mean - (mass * d_mean).sum(dim=-1, keepdim=True)
                    means = mu0.unsqueeze(-1) + alpha_det.unsqueeze(0) * d_pin
                    carve = std0.unsqueeze(-1).pow(2) * (
                        1.0 - alpha_det.unsqueeze(0) * gamma.unsqueeze(0)
                    )
                    stds = ((alpha_det.unsqueeze(0) * d_std).pow(2) + carve + 1e-6).sqrt()

                pi = self.get_pi(x_d)
                alpha = self.get_alpha(deterministic=False)
                # Recompute means with stochastic α for HC path
                mass_s = (pi.detach() * alpha.unsqueeze(0)).clamp_min(1e-12)
                mass_s = mass_s / mass_s.sum(dim=-1, keepdim=True)
                d_pin_s = d_mean - (mass_s * d_mean).sum(dim=-1, keepdim=True)
                means_s = mu0.unsqueeze(-1) + alpha.unsqueeze(0) * d_pin_s
                carve_s = std0.unsqueeze(-1).pow(2) * (
                    1.0 - alpha.unsqueeze(0) * gamma.unsqueeze(0)
                )
                stds_s = ((alpha.unsqueeze(0) * d_std).pow(2) + carve_s + 1e-6).sqrt()

                log_comp = _gaussian_log_prob(y_d.unsqueeze(-1), means_s, stds_s)
                nll = -(
                    torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_comp, dim=-1)
                ).mean()
                loss = nll + float(lambda_alpha) * self.hc.complexity_loss()
                loss.backward()
                opt.step()
                if step % 100 == 0 or step + 1 == num_gate_steps:
                    history.append(
                        {
                            "em": em,
                            "step": step,
                            "loss": float(loss.detach().item()),
                            "nll": float(nll.detach().item()),
                        }
                    )

            # Update responsibilities from current mixture
            self.eval()
            with torch.no_grad():
                pi, means, stds, _, _ = self._mixture_params(x_d)
                log_comp = _gaussian_log_prob(y_d.unsqueeze(-1), means, stds)
                log_joint = torch.log(pi.clamp_min(1e-12)) + log_comp
                resp = torch.softmax(log_joint, dim=-1).cpu()

        self.to(device="cpu")
        self.eval()
        self._fit_history = history
        return {"history": history, "k_max": self.k_max}

    @torch.no_grad()
    def predict(self, test_x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-input MoG margins (ScoringBench-compatible)."""
        self.eval()
        test_x = test_x.to(dtype=self.dtype)
        if test_x.ndim == 1:
            test_x = test_x.unsqueeze(-1)
        if not self._experts_fitted:
            raise RuntimeError("Call fit() before predict().")
        pi, means, stds, mu0, std0 = self._mixture_params(test_x)
        mix_mean = (pi * means).sum(dim=-1)
        second = (pi * (stds.pow(2) + means.pow(2))).sum(dim=-1)
        mix_var = (second - mix_mean.pow(2)).clamp_min(1e-12)
        return {
            "mean": mix_mean,
            "stddev": mix_var.sqrt(),
            "expert_means": means,
            "expert_stds": stds,
            "gate_weights": pi,
            "mu0": mu0,
            "std0": std0,
            "alpha": self.get_alpha(deterministic=True).expand(test_x.shape[0], -1),
            "gamma": self.get_gamma(),
        }

    @torch.no_grad()
    def predict_joint(
        self,
        test_x: torch.Tensor,
        *,
        assignment: Literal["shared"] = "shared",
    ) -> dict[str, torch.Tensor]:
        """Mixture of multivariate Gaussians under shared expert assignment.

        For each expert ``k``, all locations in ``test_x`` are assigned to residual
        GP ``k``. Component weight is the mean gate mass
        ``mean_i π_k(x_i) α_k`` (renormalized). Covariance comes from the residual
        GP posterior joint (plus diagonal midline-envelope carve-out / noise).

        This is an **estimative** FDD, not a Bayesian posterior predictive.
        """
        if assignment != "shared":
            raise ValueError(f"Unsupported assignment={assignment!r}")
        self.eval()
        test_x = test_x.to(dtype=self.dtype)
        if test_x.ndim == 1:
            test_x = test_x.unsqueeze(-1)
        if not self._experts_fitted or self.core is None:
            raise RuntimeError("Call fit() before predict_joint().")

        n = int(test_x.shape[0])
        pi = self.get_pi(test_x)
        alpha = self.get_alpha(deterministic=True)
        gamma = self.get_gamma()
        mu0, std0 = self._core_predict(test_x)
        # Shared-assignment weights from average gate mass
        w = (pi * alpha.unsqueeze(0)).mean(dim=0).clamp_min(1e-12)
        w = w / w.sum()

        means = torch.zeros(self.k_max, n, dtype=self.dtype)
        covs = torch.zeros(self.k_max, n, n, dtype=self.dtype)
        ni_delta_mean = ni_delta_std = None
        if self.midline_backend == "ni_sorf":
            ni_delta_mean, ni_delta_std = self._expert_predict_marginal(test_x)

        for k, expert in enumerate(self.residual_experts):
            expert.eval()
            d_cov = None
            if self.midline_backend == "ni_sorf":
                # Woodbury NI-SORF: diagonal joint from margins (full Gram not exposed).
                assert ni_delta_mean is not None and ni_delta_std is not None
                d_mean = ni_delta_mean[:, k].to(dtype=self.dtype)
                d_cov = torch.diag(ni_delta_std[:, k].to(dtype=self.dtype).pow(2))
                noise = 1e-4
            else:
                with gpytorch.settings.fast_pred_var(False), gpytorch.settings.detach_test_caches(
                    True
                ):
                    post = expert(test_x)
                    if hasattr(post, "mean") and hasattr(post, "covariance_matrix"):
                        d_mean = post.mean.reshape(-1).to(dtype=self.dtype)
                        d_cov = post.covariance_matrix.to(dtype=self.dtype)
                    else:
                        m, _lo, _hi, s = evaluate_gp_model(expert, test_x)
                        d_mean = m.reshape(-1).to(dtype=self.dtype)
                        d_cov = torch.diag(s.reshape(-1).to(dtype=self.dtype).pow(2))
                noise = 1e-4
                lik = expert.likelihood
                if hasattr(lik, "noise"):
                    try:
                        noise = float(lik.noise.mean().clamp_min(1e-6).item())
                    except Exception:  # noqa: BLE001
                        noise = 1e-4

            a_k = float(alpha[k].item())
            g_k = float(gamma[k].item())
            means[k] = mu0 + a_k * d_mean
            carve = (std0.pow(2) * (1.0 - a_k * g_k)).clamp_min(1e-8)
            assert d_cov is not None
            covs[k] = (a_k ** 2) * d_cov + torch.diag(carve) + noise * torch.eye(
                n, dtype=self.dtype
            )

        return {
            "component_weights": w,
            "means": means,
            "covs": covs,
            "assignment": torch.tensor(0),  # marker: shared
            "mu0": mu0,
            "std0": std0,
        }

    @torch.no_grad()
    def sample(
        self,
        test_x: torch.Tensor,
        n_samples: int = 1,
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Draw paths from the shared-assignment mixture-of-MVNs joint."""
        if seed is not None:
            torch.manual_seed(int(seed))
        joint = self.predict_joint(test_x)
        w = joint["component_weights"]
        means = joint["means"]
        covs = joint["covs"]
        n = means.shape[-1]
        out = torch.zeros(n_samples, n, dtype=self.dtype)
        cat = torch.distributions.Categorical(probs=w)
        for s in range(int(n_samples)):
            k = int(cat.sample((1,)).item())
            mvn = torch.distributions.MultivariateNormal(
                means[k],
                covariance_matrix=covs[k]
                + 1e-5 * torch.eye(n, dtype=self.dtype),
            )
            out[s] = mvn.sample((1,)).reshape(-1)
        return out
