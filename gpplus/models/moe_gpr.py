"""Mixture-of-Experts Gaussian Process Regression (exact GPR experts + NN gate)."""

from __future__ import annotations

from typing import Literal, Sequence

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from torch import nn

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood
from .gpr import GPR

GateType = Literal["constant", "mlp"]
FitMode = Literal["hard", "soft"]


class ConstantGate(nn.Module):
    """Input-independent mixture weights ``π = softmax(z / T)`` on the simplex."""

    def __init__(
        self,
        num_experts: int,
        *,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        self.logits_param = nn.Parameter(torch.zeros(num_experts, dtype=dtype))

    def logits(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits shaped ``[N, K]`` (broadcast over batch if ``x`` given)."""
        z = self.logits_param
        if x is None:
            return z.unsqueeze(0)
        n = x.shape[0]
        return z.unsqueeze(0).expand(n, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x) / self.temperature, dim=-1)


class GatingNetwork(nn.Module):
    """MLP gating network with temperature softmax on the probability simplex.

    Emits unconstrained logits then applies ``softmax(z / T)`` so expert weights
    satisfy ``π_k(x) ≥ 0`` and ``∑_k π_k(x) = 1`` at every input.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_experts)))
        self.network = nn.Sequential(*layers).to(dtype=dtype)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate weights ``π(x)`` of shape ``[N, K]`` (non-negative, sum to 1)."""
        return torch.softmax(self.logits(x) / self.temperature, dim=-1)


def _gaussian_log_prob(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density; shapes broadcast to ``[N]``."""
    var = (std * std).clamp_min(1e-12)
    return -0.5 * (torch.log(2.0 * torch.pi * var) + (y - mean).pow(2) / var)


def _kmeans_assignments(xy: np.ndarray, k: int, seed: int, max_iter: int = 50) -> np.ndarray:
    """Simple Lloyd k-means; returns hard labels in ``{0, ..., k-1}``."""
    rng = np.random.default_rng(seed)
    n = xy.shape[0]
    if k >= n:
        return np.arange(n) % k
    centroids = xy[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = ((xy[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = xy[mask].mean(axis=0)
            else:
                centroids[j] = xy[rng.integers(0, n)]
    return labels


def _build_expert_gpr(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    ard: bool = True,
) -> GPR:
    input_dim = int(train_x.shape[-1])
    if ard:
        kernel_module = LogScaleKernel(GaussianKernel(ard_num_dims=input_dim))
    else:
        kernel_module = LogScaleKernel(GaussianKernel())
    return GPR(
        train_x,
        train_y,
        likelihood=LogGaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        kernel_module=kernel_module,
    )


def _build_weighted_expert_gpr(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    weights: torch.Tensor,
    *,
    ard: bool = True,
    base_noise: float = 1.0,
    weight_floor: float = 1e-3,
) -> GPR:
    """Exact GPR on all points with heteroscedastic noise ``σ²_i ∝ 1/r_i``.

    Low-responsibility points get large fixed noise so they barely pull the fit.
    ``learn_additional_noise=True`` lets the model learn a shared nugget used at
    test time (FixedNoise train noises are train-sized only).
    """
    w = weights.reshape(-1).to(dtype=train_x.dtype, device=train_x.device)
    if w.shape[0] != train_x.shape[0]:
        raise ValueError(
            f"weights length {w.shape[0]} must match n_train={train_x.shape[0]}"
        )
    noise = (float(base_noise) / w.clamp_min(float(weight_floor))).detach().clone()
    input_dim = int(train_x.shape[-1])
    if ard:
        kernel_module = LogScaleKernel(GaussianKernel(ard_num_dims=input_dim))
    else:
        kernel_module = LogScaleKernel(GaussianKernel())
    likelihood = FixedNoiseGaussianLikelihood(
        noise=noise,
        learn_additional_noise=True,
    ).to(dtype=train_x.dtype)
    return GPR(
        train_x,
        train_y,
        likelihood=likelihood,
        mean_module=gpytorch.means.ConstantMean(),
        kernel_module=kernel_module,
    )


class MOEGPR(nn.Module):
    """Mixture of exact :class:`~gpplus.models.gpr.GPR` experts with a gate.

    Predictive density::

        p(y | x) = ∑_k π_k(x) N(y; μ_k(x), σ_k²(x))

    where ``π = softmax(z / T)`` (non-negative, sums to 1).

    ``gate_type``:
      - ``"mlp"`` (default): input-dependent MLP logits (baseline ``experiments_MOEGP``)
      - ``"constant"``: learnable global logits (co-located multimodal path)

    ``fit_mode`` (passed to :meth:`fit`):
      - ``"hard"``: hard-assign points, fit each expert on a subset (legacy EM)
      - ``"soft"``: every point soft-weights every expert (``σ² ∝ 1/r``); gate
        trained on mixture NLL + anti-collapse regularizers

    ``assignment_init`` chooses first labels (hard mode) or the soft seed
    (soft mode — still uses all points, only the initial weights differ):
      ``residual`` | ``x`` | ``quadrant`` | ``random``.
    Soft mode must break expert symmetry; uniform ``r=1/K`` collapses both
    experts onto the average of the modes.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_experts: int = 2,
        gating_hidden: Sequence[int] = (64, 64),
        *,
        gate_type: GateType = "mlp",
        temperature: float = 1.0,
        assignment_init: Literal["residual", "x", "quadrant", "random"] = "residual",
        ard: bool = True,
        min_expert_points: int = 3,
        load_balance_weight: float = 0.1,
        entropy_weight: float = 0.01,
        diversity_weight: float = 0.1,
        orthogonality_weight: float = 3.0,
        orth_refit_threshold: float = 0.25,
        responsibility_temperature: float = 0.5,
        soft_init_peak: float = 0.95,
        soft_weight_floor: float = 1e-4,
        soft_base_noise: float = 1.0,
    ):
        super().__init__()
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor instances.")
        if train_y.ndim > 1:
            train_y = train_y.reshape(-1)
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if gate_type not in ("constant", "mlp"):
            raise ValueError(f"gate_type must be 'constant' or 'mlp', got {gate_type!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if assignment_init not in ("residual", "x", "quadrant", "random"):
            raise ValueError(
                f"assignment_init must be 'residual', 'x', 'quadrant', or 'random', "
                f"got {assignment_init!r}"
            )
        if assignment_init == "quadrant" and num_experts != 4:
            raise ValueError("assignment_init='quadrant' requires num_experts=4 (TL/TR/BL/BR).")
        if load_balance_weight < 0 or entropy_weight < 0 or diversity_weight < 0:
            raise ValueError(
                "load_balance_weight, entropy_weight, and diversity_weight must be >= 0."
            )
        if orthogonality_weight < 0:
            raise ValueError(f"orthogonality_weight must be >= 0, got {orthogonality_weight}")
        if not (0.0 <= orth_refit_threshold <= 1.0):
            raise ValueError(
                f"orth_refit_threshold must be in [0, 1], got {orth_refit_threshold}"
            )
        if responsibility_temperature <= 0:
            raise ValueError(
                f"responsibility_temperature must be > 0, got {responsibility_temperature}"
            )
        if not (0.0 < soft_init_peak <= 1.0):
            raise ValueError(f"soft_init_peak must be in (0, 1], got {soft_init_peak}")

        self.register_buffer("train_x", train_x.detach().clone())
        self.register_buffer("train_y", train_y.detach().clone())
        self.num_experts = int(num_experts)
        self.gate_type: GateType = gate_type
        self.temperature = float(temperature)
        self.assignment_init = assignment_init
        self.ard = bool(ard)
        self.min_expert_points = int(min_expert_points)
        self.load_balance_weight = float(load_balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.diversity_weight = float(diversity_weight)
        self.orthogonality_weight = float(orthogonality_weight)
        self.orth_refit_threshold = float(orth_refit_threshold)
        self.responsibility_temperature = float(responsibility_temperature)
        self.soft_init_peak = float(soft_init_peak)
        self.soft_weight_floor = float(soft_weight_floor)
        self.soft_base_noise = float(soft_base_noise)
        self.dtype = train_x.dtype
        input_dim = int(train_x.shape[-1])

        if gate_type == "constant":
            self.gating: nn.Module = ConstantGate(
                num_experts=self.num_experts,
                temperature=self.temperature,
                dtype=self.dtype,
            )
        else:
            self.gating = GatingNetwork(
                input_dim=input_dim,
                num_experts=self.num_experts,
                hidden_dims=gating_hidden,
                temperature=self.temperature,
                dtype=self.dtype,
            )
        self.experts: nn.ModuleList = nn.ModuleList()
        self._assignments: torch.Tensor | None = None
        self._soft_responsibilities_last: torch.Tensor | None = None
        self._is_fitted = False
        self._fit_mode: FitMode | None = None

    def get_expert_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Gate probabilities ``π(x)`` with shape ``[N, K]`` (simplex-valued)."""
        self.gating.eval()
        with torch.no_grad():
            weights = self.gating(x.to(dtype=self.dtype))
        if not torch.all(weights >= -1e-8):
            raise RuntimeError("Gate weights must be non-negative.")
        sums = weights.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-4):
            raise RuntimeError("Gate weights must sum to 1 along the expert axis.")
        return weights.clamp_min(0.0)

    def _ensure_min_assignment(self, labels: torch.Tensor) -> torch.Tensor:
        """Guarantee each expert receives at least ``min_expert_points`` (when feasible)."""
        labels = labels.clone()
        n = labels.numel()
        k = self.num_experts
        min_pts = min(self.min_expert_points, max(1, n // k))
        counts = torch.bincount(labels, minlength=k)
        for j in range(k):
            if counts[j] >= min_pts:
                continue
            donors = (counts > min_pts).nonzero(as_tuple=False).view(-1)
            if donors.numel() == 0:
                donors = (counts == counts.max()).nonzero(as_tuple=False).view(-1)
            need = int(min_pts - counts[j])
            for _ in range(need):
                if donors.numel() == 0:
                    break
                donor = int(donors[0].item())
                idx = (labels == donor).nonzero(as_tuple=False).view(-1)
                if idx.numel() <= min_pts:
                    donors = donors[1:]
                    continue
                move = int(idx[-1].item())
                labels[move] = j
                counts[donor] -= 1
                counts[j] += 1
                if counts[donor] <= min_pts:
                    donors = donors[1:]
        return labels

    def _init_assignments(self, seed: int) -> torch.Tensor:
        """Initialize expert labels.

        - ``residual``: k-means on residuals from a global linear fit (co-located modes).
        - ``x``: k-means on input space (regional / sharp-edge partitions).
        - ``quadrant``: 4 experts as left/right × above/below residual
          (top-left, bottom-left, top-right, bottom-right segments of an X).
        - ``random``: uniform random hard labels in ``{0, ..., K-1}`` (EM must recover).
        """
        x_np = self.train_x.detach().cpu().numpy()
        if self.assignment_init == "quadrant":
            labels_np = self._quadrant_assignments(x_np)
        elif self.assignment_init == "x":
            labels_np = _kmeans_assignments(x_np, self.num_experts, seed=seed)
        elif self.assignment_init == "random":
            rng = np.random.default_rng(seed)
            labels_np = rng.integers(0, self.num_experts, size=int(x_np.shape[0]))
        else:
            y_np = self.train_y.detach().cpu().numpy().ravel()
            x1 = x_np[:, 0]
            a_mat = np.stack([x1, np.ones_like(x1)], axis=1)
            coef, _, _, _ = np.linalg.lstsq(a_mat, y_np, rcond=None)
            features = (y_np - a_mat @ coef).reshape(-1, 1)
            labels_np = _kmeans_assignments(features, self.num_experts, seed=seed)
        labels = torch.as_tensor(labels_np, device=self.train_x.device, dtype=torch.long)
        return self._ensure_min_assignment(labels)

    def _quadrant_assignments(self, x_np: np.ndarray) -> np.ndarray:
        """Label points into 4 X-segments via median-x split and LS residual sign.

        Encoding:
          0 = left + above  (top-left)
          1 = left + below  (bottom-left)
          2 = right + above (top-right)
          3 = right + below (bottom-right)
        """
        y_np = self.train_y.detach().cpu().numpy().ravel()
        x1 = x_np[:, 0]
        a_mat = np.stack([x1, np.ones_like(x1)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(a_mat, y_np, rcond=None)
        resid = y_np - a_mat @ coef
        x_split = float(np.median(x1))
        left = x1 < x_split
        above = resid >= 0.0
        labels = np.zeros(x1.shape[0], dtype=np.int64)
        labels[left & above] = 0
        labels[left & ~above] = 1
        labels[~left & above] = 2
        labels[~left & ~above] = 3
        return labels

    def _fit_experts(
        self,
        labels: torch.Tensor,
        *,
        num_inits: int,
        num_epochs: int,
        seed: int,
        device: str,
        n_jobs: int | None,
    ) -> None:
        from ..training.trainer import GPTrainer

        experts: list[GPR] = []
        for k in range(self.num_experts):
            mask = labels == k
            x_k = self.train_x[mask]
            y_k = self.train_y[mask]
            if x_k.shape[0] < 2:
                raise RuntimeError(f"Expert {k} has fewer than 2 points after assignment.")
            expert = _build_expert_gpr(x_k, y_k, ard=self.ard)
            trainer = GPTrainer(
                expert,
                num_epochs=num_epochs,
                num_inits=num_inits,
                seed=seed + k,
                device=device,
                dtype=self.dtype,
                n_jobs=n_jobs,
            )
            runs = trainer.train()
            valid = [
                r
                for r in runs
                if r.get("loss") is not None and r.get("state_dict") is not None
            ]
            if not valid:
                raise RuntimeError(f"All GPTrainer runs failed for expert {k}.")
            best = min(valid, key=lambda r: r["loss"])
            expert.load_state_dict(best["state_dict"])
            expert.eval()
            experts.append(expert)
            logger.info(
                "MOEGPR expert %d fitted on %d points (loss=%.4f).",
                k,
                int(x_k.shape[0]),
                float(best["loss"]),
            )
        self.experts = nn.ModuleList(experts)
        self._assignments = labels.detach().clone()

    def _fit_experts_soft(
        self,
        responsibilities: torch.Tensor,
        *,
        num_inits: int,
        num_epochs: int,
        seed: int,
        device: str,
        n_jobs: int | None,
    ) -> list[float]:
        """Fit each expert on **all** training points with soft weights ``r_{:,k}``."""
        from ..training.trainer import GPTrainer

        resp = responsibilities.detach()
        if resp.shape != (self.train_x.shape[0], self.num_experts):
            raise ValueError(
                f"responsibilities shape {tuple(resp.shape)} must be "
                f"[N={self.train_x.shape[0]}, K={self.num_experts}]"
            )
        experts: list[GPR] = []
        masses: list[float] = []
        for k in range(self.num_experts):
            w_k = resp[:, k]
            mass = float(w_k.sum().item())
            masses.append(mass)
            expert = _build_weighted_expert_gpr(
                self.train_x,
                self.train_y,
                w_k,
                ard=self.ard,
                base_noise=self.soft_base_noise,
                weight_floor=self.soft_weight_floor,
            )
            trainer = GPTrainer(
                expert,
                num_epochs=num_epochs,
                num_inits=num_inits,
                seed=seed + k,
                device=device,
                dtype=self.dtype,
                n_jobs=n_jobs,
            )
            runs = trainer.train()
            valid = [
                r
                for r in runs
                if r.get("loss") is not None and r.get("state_dict") is not None
            ]
            if not valid:
                raise RuntimeError(f"All GPTrainer runs failed for soft expert {k}.")
            best = min(valid, key=lambda r: r["loss"])
            expert.load_state_dict(best["state_dict"])
            expert.eval()
            experts.append(expert)
            logger.info(
                "MOEGPR soft expert %d fitted on all %d points "
                "(effective_mass=%.2f, loss=%.4f).",
                k,
                int(self.train_x.shape[0]),
                mass,
                float(best["loss"]),
            )
        self.experts = nn.ModuleList(experts)
        # Diagnostic hard labels only (not used for the next soft M-step).
        self._assignments = resp.argmax(dim=-1).detach().clone()
        self._soft_responsibilities_last = resp.detach().clone()
        return masses

    def _expert_predictive(self, expert: GPR, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from ..training.eval import evaluate_gp_model

        mean, _lower, _upper, std = evaluate_gp_model(expert, x)
        return mean.reshape(-1), std.reshape(-1)

    def _detached_expert_log_lik(self) -> torch.Tensor:
        """Per-point expert log-likelihoods ``[N, K]`` with experts frozen."""
        x = self.train_x
        y = self.train_y.reshape(-1)
        with torch.no_grad():
            cols = []
            for expert in self.experts:
                mu, std = self._expert_predictive(expert, x)
                cols.append(_gaussian_log_prob(y, mu, std))
            return torch.stack(cols, dim=-1)

    def _labels_to_soft_responsibilities(self, labels: torch.Tensor) -> torch.Tensor:
        """One-hot-ish soft weights from hard labels (peak on assigned expert)."""
        n = labels.numel()
        k = self.num_experts
        peak = self.soft_init_peak
        off = (1.0 - peak) / max(k - 1, 1)
        resp = torch.full(
            (n, k),
            off,
            dtype=self.dtype,
            device=labels.device,
        )
        resp.scatter_(1, labels.view(-1, 1), peak)
        return resp

    def _init_soft_responsibilities(self, seed: int) -> torch.Tensor:
        """Asymmetric soft seed so experts do not all fit the same average mode."""
        labels = self._init_assignments(seed=seed)
        return self._labels_to_soft_responsibilities(labels)

    @staticmethod
    def _sharpen_responsibilities(
        responsibilities: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Sharpen row-wise soft weights (``temperature < 1`` → peakier)."""
        log_r = torch.log(responsibilities.clamp_min(1e-12)) / float(temperature)
        return torch.softmax(log_r, dim=-1)

    def _soft_responsibilities(self, *, sharpen: bool = False) -> torch.Tensor:
        """E-step soft responsibilities ``r_ik`` of shape ``[N, K]``."""
        with torch.no_grad():
            pi = self.gating(self.train_x)  # [N, K]
            log_lik_mat = self._detached_expert_log_lik()
            log_r = torch.log(pi.clamp_min(1e-12)) + log_lik_mat
            log_r = log_r - torch.logsumexp(log_r, dim=-1, keepdim=True)
            resp = log_r.exp()
            if sharpen and self.responsibility_temperature != 1.0:
                resp = self._sharpen_responsibilities(
                    resp, self.responsibility_temperature
                )
            return resp

    def _centered_expert_means(self) -> torch.Tensor | None:
        """Centered expert predictive means on ``train_x``, shape ``[K, N]``."""
        if len(self.experts) < 1:
            return None
        with torch.no_grad():
            means = []
            for expert in self.experts:
                mu, _std = self._expert_predictive(expert, self.train_x)
                means.append(mu.reshape(-1))
            stacked = torch.stack(means, dim=0)
            return stacked - stacked.mean(dim=1, keepdim=True)

    def _expert_mean_diversity(self) -> float:
        """Mean pairwise squared distance between centered expert means on train_x."""
        stacked = self._centered_expert_means()
        if stacked is None or stacked.shape[0] < 2:
            return 0.0
        with torch.no_grad():
            dists = []
            k = stacked.shape[0]
            for i in range(k):
                for j in range(i + 1, k):
                    dists.append((stacked[i] - stacked[j]).pow(2).mean())
            if not dists:
                return 0.0
            return float(torch.stack(dists).mean().item())

    def _expert_orthogonality_score(self) -> float:
        """Worst-pair absolute cosine similarity of centered expert means.

        Uses ``max_{i<j} |cos(μ_i, μ_j)|`` so any near-parallel pair dominates.
        ``0`` ≈ orthogonal; ``1`` ≈ parallel (identity collapse).
        """
        stacked = self._centered_expert_means()
        if stacked is None or stacked.shape[0] < 2:
            return 0.0
        with torch.no_grad():
            worst = 0.0
            k = stacked.shape[0]
            for i in range(k):
                for j in range(i + 1, k):
                    a = stacked[i]
                    b = stacked[j]
                    denom = a.norm() * b.norm()
                    if float(denom.item()) < 1e-12:
                        return 1.0
                    cos_abs = float(((a * b).sum() / denom).abs().item())
                    if cos_abs > worst:
                        worst = cos_abs
            return float(worst)

    def _apply_orthogonality_to_responsibilities(
        self,
        responsibilities: torch.Tensor,
        seed_responsibilities: torch.Tensor,
        *,
        orth_score: float | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        """Blend EM responsibilities toward a specialized seed when experts align.

        Aggressive blend (not linear in score)::

            blend = 1 - (1 - orth_score) ** orthogonality_weight

        Examples with ``orthogonality_weight=3``:
          - score=0.2 → blend≈0.49
          - score=0.5 → blend≈0.88
          - score=0.8 → blend≈0.99
        """
        if self.orthogonality_weight <= 0:
            return responsibilities, 0.0, float(orth_score or 0.0)
        score = (
            float(orth_score)
            if orth_score is not None
            else self._expert_orthogonality_score()
        )
        score = min(1.0, max(0.0, score))
        # Super-linear pull toward the specialized seed as alignment grows.
        blend = 1.0 - (1.0 - score) ** float(self.orthogonality_weight)
        blend = min(1.0, max(0.0, blend))
        if blend <= 1e-8:
            return responsibilities, 0.0, score
        mixed = (1.0 - blend) * responsibilities + blend * seed_responsibilities
        mixed = mixed.clamp_min(1e-12)
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        return mixed, blend, score

    def _experts_collapsed(self, *, corr_thresh: float = 0.90) -> bool:
        """True if any pair of expert mean curves is nearly identical (corr)."""
        stacked = self._centered_expert_means()
        if stacked is None or stacked.shape[0] < 2:
            return False
        with torch.no_grad():
            k = stacked.shape[0]
            for i in range(k):
                for j in range(i + 1, k):
                    a = stacked[i]
                    b = stacked[j]
                    denom = a.norm() * b.norm()
                    if float(denom.item()) < 1e-12:
                        return True
                    corr = float((a * b).sum().item() / float(denom.item()))
                    if corr >= corr_thresh:
                        return True
        return False

    def _gate_anti_collapse_loss(
        self,
        pi: torch.Tensor,
        *,
        expert_diversity: float | None = None,
        orth_score: float | None = None,
    ) -> torch.Tensor:
        """Load-balance + entropy + diversity bonus + orthogonality penalty."""
        k = self.num_experts
        mean_pi = pi.mean(dim=0).clamp_min(1e-12)
        mean_pi = mean_pi / mean_pi.sum()
        uniform = torch.full_like(mean_pi, 1.0 / k)
        # KL(mean_π || Uniform)
        load_balance = (mean_pi * (mean_pi.log() - uniform.log())).sum()
        # Mean entropy H(π(x)); maximize → minimize negative entropy
        ent = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=-1).mean()
        loss = (
            self.load_balance_weight * load_balance
            - self.entropy_weight * ent
        )
        # Encourage experts to disagree: maximize diversity → minimize -diversity.
        if expert_diversity is not None and self.diversity_weight > 0:
            # Normalize by y-scale so the weight is roughly dimensionless.
            y_scale = self.train_y.float().var().clamp_min(1e-6)
            loss = loss - self.diversity_weight * (
                torch.as_tensor(
                    expert_diversity,
                    dtype=pi.dtype,
                    device=pi.device,
                )
                / y_scale
            )
        # Penalize parallel expert mean curves (identity collapse).
        # Square amplifies mid/high alignment; weight scales the gate term.
        if orth_score is not None and self.orthogonality_weight > 0:
            loss = loss + (self.orthogonality_weight * float(orth_score) ** 2) * torch.as_tensor(
                1.0,
                dtype=pi.dtype,
                device=pi.device,
            )
        return loss

    def _fit_gating(
        self,
        responsibilities: torch.Tensor,
        *,
        num_epochs: int,
        lr: float,
        seed: int,
    ) -> float:
        """Hard-EM gate step: match soft responsibilities (CE)."""
        torch.manual_seed(seed)
        self.gating.train()
        opt = torch.optim.Adam(self.gating.parameters(), lr=lr)
        x = self.train_x
        target = responsibilities.detach()
        last_loss = float("nan")
        t = float(self.temperature)

        if self.gate_type == "constant":
            # Match global π to mean soft responsibilities (no x → r fitting).
            mean_r = target.mean(dim=0).clamp_min(1e-12)
            mean_r = mean_r / mean_r.sum()
            for _ in range(num_epochs):
                opt.zero_grad(set_to_none=True)
                log_pi = torch.log_softmax(self.gating.logits_param / t, dim=-1)
                loss = -(mean_r * log_pi).sum()
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().item())
        else:
            for _ in range(num_epochs):
                opt.zero_grad(set_to_none=True)
                logits = self.gating.logits(x)
                log_pi = torch.log_softmax(logits / t, dim=-1)
                loss = -(target * log_pi).sum(dim=-1).mean()
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().item())

        self.gating.eval()
        return last_loss

    def _fit_gating_mixture_nll(
        self,
        *,
        num_epochs: int,
        lr: float,
        seed: int,
    ) -> dict[str, float]:
        """Soft-EM gate step: mixture NLL + anti-collapse (experts detached)."""
        torch.manual_seed(seed)
        self.gating.train()
        opt = torch.optim.Adam(self.gating.parameters(), lr=lr)
        x = self.train_x
        log_lik_mat = self._detached_expert_log_lik()  # [N, K]
        expert_div = self._expert_mean_diversity()
        orth_score = self._expert_orthogonality_score()
        last_loss = float("nan")
        last_mix = float("nan")
        last_reg = float("nan")
        t = float(self.temperature)

        for _ in range(num_epochs):
            opt.zero_grad(set_to_none=True)
            logits = self.gating.logits(x)
            log_pi = torch.log_softmax(logits / t, dim=-1)
            pi = log_pi.exp()
            mix_nll = -torch.logsumexp(log_pi + log_lik_mat, dim=-1).mean()
            reg = self._gate_anti_collapse_loss(
                pi,
                expert_diversity=expert_div,
                orth_score=orth_score,
            )
            loss = mix_nll + reg
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().item())
            last_mix = float(mix_nll.detach().item())
            last_reg = float(reg.detach().item())

        self.gating.eval()
        with torch.no_grad():
            pi = self.gating(x)
            mean_pi = pi.mean(dim=0)
            ent = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return {
            "gate_loss": last_loss,
            "mixture_nll": last_mix,
            "reg_loss": last_reg,
            "mean_gate_entropy": float(ent.item()),
            "mean_pi": [float(v) for v in mean_pi.detach().cpu().tolist()],
            "expert_diversity": float(expert_div),
            "orthogonality_score": float(orth_score),
        }

    def fit(
        self,
        *,
        fit_mode: FitMode = "hard",
        num_em_iters: int = 5,
        num_inits: int = 8,
        num_epochs: int = 1,
        gate_epochs: int = 200,
        gate_lr: float = 1e-2,
        seed: int = 42,
        device: str = "cpu",
        n_jobs: int | None = 1,
    ) -> dict:
        """Run EM training (``hard`` subset EM or ``soft`` joint weighted EM)."""
        if fit_mode not in ("hard", "soft"):
            raise ValueError(f"fit_mode must be 'hard' or 'soft', got {fit_mode!r}")
        self._fit_mode = fit_mode
        if fit_mode == "soft":
            return self._fit_soft(
                num_em_iters=num_em_iters,
                num_inits=num_inits,
                num_epochs=num_epochs,
                gate_epochs=gate_epochs,
                gate_lr=gate_lr,
                seed=seed,
                device=device,
                n_jobs=n_jobs,
            )

        history: dict = {
            "fit_mode": "hard",
            "em_iters": [],
            "gate_losses": [],
            "assignment_counts": [],
            "gate_type": self.gate_type,
            "temperature": self.temperature,
        }
        labels = self._init_assignments(seed=seed)

        for em_i in range(int(num_em_iters)):
            logger.info(
                "MOEGPR EM iteration %d / %d (gate_type=%s, T=%.4g)",
                em_i + 1,
                num_em_iters,
                self.gate_type,
                self.temperature,
            )
            self._fit_experts(
                labels,
                num_inits=num_inits,
                num_epochs=num_epochs,
                seed=seed + 1000 * em_i,
                device=device,
                n_jobs=n_jobs,
            )
            resp = self._soft_responsibilities()
            gate_loss = self._fit_gating(
                resp,
                num_epochs=gate_epochs,
                lr=gate_lr,
                seed=seed + 10_000 + em_i,
            )
            # Assign from y-aware responsibilities (argmax r), not from gate alone.
            labels = self._ensure_min_assignment(resp.argmax(dim=-1))
            counts = [int((labels == k).sum().item()) for k in range(self.num_experts)]
            history["em_iters"].append(em_i)
            history["gate_losses"].append(gate_loss)
            history["assignment_counts"].append(counts)
            logger.info(
                "MOEGPR EM %d: gate_loss=%.4f assignment_counts=%s",
                em_i + 1,
                gate_loss,
                counts,
            )

        self._is_fitted = True
        return history

    def _fit_soft(
        self,
        *,
        num_em_iters: int,
        num_inits: int,
        num_epochs: int,
        gate_epochs: int,
        gate_lr: float,
        seed: int,
        device: str,
        n_jobs: int | None,
    ) -> dict:
        """Soft joint EM: all data → every expert via ``r_ik``; gate on mixture NLL.

        Starts from an asymmetric soft seed (from ``assignment_init``), not uniform
        ``1/K`` — uniform seeding makes every expert fit the same average of modes.
        """
        seed_resp = self._init_soft_responsibilities(seed=seed)
        resp = seed_resp.clone()
        history: dict = {
            "fit_mode": "soft",
            "em_iters": [],
            "gate_losses": [],
            "mixture_nlls": [],
            "reg_losses": [],
            "mean_pi": [],
            "mean_gate_entropy": [],
            "effective_masses": [],
            "expert_diversity": [],
            "orthogonality_scores": [],
            "orthogonality_blends": [],
            "orth_refits": [],
            "collapsed_resets": [],
            "gate_type": self.gate_type,
            "temperature": self.temperature,
            "load_balance_weight": self.load_balance_weight,
            "entropy_weight": self.entropy_weight,
            "diversity_weight": self.diversity_weight,
            "orthogonality_weight": self.orthogonality_weight,
            "orth_refit_threshold": self.orth_refit_threshold,
            "responsibility_temperature": self.responsibility_temperature,
            "soft_init_peak": self.soft_init_peak,
            "assignment_init": self.assignment_init,
        }

        for em_i in range(int(num_em_iters)):
            logger.info(
                "MOEGPR soft EM %d / %d (gate_type=%s, T=%.4g, r_temp=%.3g, orth_w=%.3g)",
                em_i + 1,
                num_em_iters,
                self.gate_type,
                self.temperature,
                self.responsibility_temperature,
                self.orthogonality_weight,
            )
            masses = self._fit_experts_soft(
                resp,
                num_inits=num_inits,
                num_epochs=num_epochs,
                seed=seed + 1000 * em_i,
                device=device,
                n_jobs=n_jobs,
            )

            # E-step + strong orth blend, then optionally refit in this same iter.
            resp = self._soft_responsibilities(sharpen=True)
            orth_score = self._expert_orthogonality_score()
            resp, orth_blend, orth_score = self._apply_orthogonality_to_responsibilities(
                resp,
                seed_resp,
                orth_score=orth_score,
            )

            collapsed = self._experts_collapsed()
            orth_refit = False
            if collapsed:
                logger.warning(
                    "MOEGPR soft EM %d: experts nearly identical — "
                    "re-seeding soft responsibilities from assignment_init=%s",
                    em_i + 1,
                    self.assignment_init,
                )
                resp = self._init_soft_responsibilities(seed=seed + 17_000 + em_i)
                seed_resp = resp.clone()
                masses = self._fit_experts_soft(
                    resp,
                    num_inits=num_inits,
                    num_epochs=num_epochs,
                    seed=seed + 2000 * em_i + 7,
                    device=device,
                    n_jobs=n_jobs,
                )
                orth_refit = True
                orth_score = self._expert_orthogonality_score()
                orth_blend = 1.0
            elif orth_blend >= self.orth_refit_threshold:
                logger.info(
                    "MOEGPR soft EM %d: orth blend=%.3f >= %.3f — refitting experts",
                    em_i + 1,
                    orth_blend,
                    self.orth_refit_threshold,
                )
                masses = self._fit_experts_soft(
                    resp,
                    num_inits=num_inits,
                    num_epochs=num_epochs,
                    seed=seed + 3000 * em_i + 3,
                    device=device,
                    n_jobs=n_jobs,
                )
                orth_refit = True
                # Refresh r after orth-guided refit.
                resp = self._soft_responsibilities(sharpen=True)
                resp, orth_blend, orth_score = self._apply_orthogonality_to_responsibilities(
                    resp,
                    seed_resp,
                    orth_score=self._expert_orthogonality_score(),
                )

            gate_stats = self._fit_gating_mixture_nll(
                num_epochs=gate_epochs,
                lr=gate_lr,
                seed=seed + 10_000 + em_i,
            )
            # Final r for the next EM iter (and diagnostics).
            resp = self._soft_responsibilities(sharpen=True)
            resp, orth_blend_final, orth_score = self._apply_orthogonality_to_responsibilities(
                resp,
                seed_resp,
                orth_score=self._expert_orthogonality_score(),
            )
            orth_blend = max(orth_blend, orth_blend_final)
            self._soft_responsibilities_last = resp.detach().clone()
            self._assignments = resp.argmax(dim=-1).detach().clone()

            history["em_iters"].append(em_i)
            history["gate_losses"].append(gate_stats["gate_loss"])
            history["mixture_nlls"].append(gate_stats["mixture_nll"])
            history["reg_losses"].append(gate_stats["reg_loss"])
            history["mean_pi"].append(gate_stats["mean_pi"])
            history["mean_gate_entropy"].append(gate_stats["mean_gate_entropy"])
            history["effective_masses"].append(masses)
            history["expert_diversity"].append(gate_stats["expert_diversity"])
            history["orthogonality_scores"].append(orth_score)
            history["orthogonality_blends"].append(orth_blend)
            history["orth_refits"].append(bool(orth_refit))
            history["collapsed_resets"].append(bool(collapsed))
            logger.info(
                "MOEGPR soft EM %d: mix_nll=%.4f reg=%.4f mean_pi=%s H(pi)=%.4f "
                "div=%.4f orth=%.4f blend=%.3f refit=%s masses=%s collapsed_reset=%s",
                em_i + 1,
                gate_stats["mixture_nll"],
                gate_stats["reg_loss"],
                gate_stats["mean_pi"],
                gate_stats["mean_gate_entropy"],
                gate_stats["expert_diversity"],
                orth_score,
                orth_blend,
                orth_refit,
                masses,
                collapsed,
            )

        self._is_fitted = True
        return history

    def predict(self, test_x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mixture and per-expert predictive summaries at ``test_x``."""
        if not self._is_fitted or len(self.experts) != self.num_experts:
            raise RuntimeError("MOEGPR must be fitted before calling predict().")

        test_x = test_x.to(dtype=self.dtype, device=self.train_x.device)
        gate_weights = self.get_expert_weights(test_x)  # [N, K]

        means = []
        stds = []
        for expert in self.experts:
            mu, std = self._expert_predictive(expert, test_x)
            means.append(mu)
            stds.append(std)
        expert_means = torch.stack(means, dim=-1)  # [N, K]
        expert_stds = torch.stack(stds, dim=-1)
        expert_vars = expert_stds.pow(2)

        mixture_mean = (gate_weights * expert_means).sum(dim=-1)
        second_moment = (gate_weights * (expert_vars + expert_means.pow(2))).sum(dim=-1)
        mixture_var = (second_moment - mixture_mean.pow(2)).clamp_min(0.0)
        mixture_std = mixture_var.sqrt()

        return {
            "mixture_mean": mixture_mean,
            "mixture_var": mixture_var,
            "mixture_std": mixture_std,
            "expert_means": expert_means,
            "expert_stds": expert_stds,
            "gate_weights": gate_weights,
        }

    def predictive_log_prob(self, test_x: torch.Tensor, test_y: torch.Tensor) -> torch.Tensor:
        """Mixture log-density ``log ∑_k π_k(x) N(y; μ_k, σ_k²)`` per point."""
        pred = self.predict(test_x)
        y = test_y.reshape(-1).to(dtype=self.dtype, device=pred["mixture_mean"].device)
        pi = pred["gate_weights"]
        means = pred["expert_means"]
        stds = pred["expert_stds"]
        log_comp = []
        for k in range(self.num_experts):
            log_comp.append(_gaussian_log_prob(y, means[:, k], stds[:, k]))
        log_comp_mat = torch.stack(log_comp, dim=-1)
        return torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_comp_mat, dim=-1)
