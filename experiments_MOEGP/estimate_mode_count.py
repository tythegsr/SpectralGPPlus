"""Estimate co-located mode count from a fitted GP via residuals (and optional slopes).

A standard GP predictive is unimodal, so mode count is *not* read from p(y|x).
Instead:

1. Fit one GP on all data → μ(x)
2. Residuals r = y − μ(x) are often multimodal for co-located modes (e.g. ScoringBench X)
3. Fit 1D GMMs on r for K=1..K_max and pick K by BIC (or AIC)

Optional: append a local data slope feature. Do **not** use ∇μ_GP alone for mode
count — that is the slope of the single average function, shared by all points.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_EXP_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _EXP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


Criterion = Literal["bic", "aic"]


@dataclass
class ModeCountResult:
    k_hat: int
    bic: dict[int, float]
    aic: dict[int, float]
    residuals: np.ndarray
    features: np.ndarray
    labels: np.ndarray  # hard labels under k_hat
    method: str


def gp_residuals(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    """r = y − μ_GP(x) using :func:`evaluate_gp_model`."""
    from gpplus.training.eval import evaluate_gp_model

    mean, _lo, _hi, _std = evaluate_gp_model(model, x)
    r = y.reshape(-1) - mean.reshape(-1)
    return r.detach().cpu().numpy().astype(np.float64)


def linear_midline_residuals(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> np.ndarray:
    """Residuals from a global least-squares line (same idea as MoE ``assignment_init=residual``).

    Often sharper for co-located modes than GP residuals, because a GP tends to
    inflate noise and smear the two residual bumps into one wide Gaussian.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float64)
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().ravel()
    else:
        y_np = np.asarray(y, dtype=np.float64).ravel()
    x1 = x_np.reshape(-1, x_np.shape[-1])[:, 0]
    a_mat = np.stack([x1, np.ones_like(x1)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(a_mat, y_np, rcond=None)
    return (y_np - a_mat @ coef).astype(np.float64)


def local_data_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Rough local dy/dx from nearest neighbors in x (noisy when modes mix)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.shape[0]
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    slopes = np.zeros(n, dtype=np.float64)
    half = max(1, n_neighbors // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 2:
            slopes[order[i]] = 0.0
            continue
        xb = xs[lo:hi]
        yb = ys[lo:hi]
        xb_c = xb - xb.mean()
        denom = float(np.dot(xb_c, xb_c))
        slopes[order[i]] = (
            float(np.dot(xb_c, yb - yb.mean()) / denom) if denom > 1e-12 else 0.0
        )
    return slopes


def gp_mean_jacobian_1d(
    model,
    x: torch.Tensor,
) -> np.ndarray:
    """∇_x μ_GP(x) for 1D inputs (same field for all points — not a mode separator)."""
    from gpplus.utils.nigp_utils import exact_posterior_mean_grad_wrt_x

    x = x.detach().requires_grad_(True)
    noise = model.likelihood.noise
    if noise.dim() > 1:
        noise = noise.reshape(-1)
    # Scalar noise broadcast for exact path when FixedNoise not used
    try:
        noise_train = noise
        if noise_train.numel() == 1:
            noise_train = noise_train.expand(x.shape[0])
        grad = exact_posterior_mean_grad_wrt_x(model, x, noise_train, jitter=1e-6)
    except Exception:
        # Fallback: autograd through predictive mean
        from gpplus.training.eval import evaluate_gp_model

        model.eval()
        mean, _, _, _ = evaluate_gp_model(model, x)
        mean = mean.reshape(-1)
        grads = []
        for i in range(mean.numel()):
            if x.grad is not None:
                x.grad.zero_()
            mean[i].backward(retain_graph=True)
            g = x.grad[i].detach().cpu().numpy().ravel()
            grads.append(g[0] if g.size else 0.0)
        return np.asarray(grads, dtype=np.float64)
    return grad.detach().cpu().numpy().reshape(x.shape[0], -1)[:, 0]


def _gmm_bic_aic(
    features: np.ndarray,
    k: int,
    *,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Fit a GMM with ``k`` components; return BIC, AIC, hard labels."""
    from sklearn.mixture import GaussianMixture

    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type="full",
        random_state=seed,
        n_init=5,
        max_iter=200,
    )
    gmm.fit(feats)
    labels = gmm.predict(feats)
    return float(gmm.bic(feats)), float(gmm.aic(feats)), labels.astype(np.int64)


def estimate_mode_count(
    residuals: np.ndarray,
    *,
    k_min: int = 1,
    k_max: int = 5,
    criterion: Criterion = "bic",
    seed: int = 42,
    extra_features: np.ndarray | None = None,
    standardize: bool = True,
) -> ModeCountResult:
    """Choose K by GMM BIC/AIC on residuals (+ optional extra columns)."""
    r = np.asarray(residuals, dtype=np.float64).ravel()
    if extra_features is None:
        feats = r.reshape(-1, 1)
        method = "residual_gmm"
    else:
        extra = np.asarray(extra_features, dtype=np.float64)
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        if extra.shape[0] != r.shape[0]:
            raise ValueError("extra_features length must match residuals.")
        feats = np.column_stack([r, extra])
        method = "residual_plus_extra_gmm"

    if standardize:
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True).clip(min=1e-8)
        feats = (feats - mu) / sd

    bic: dict[int, float] = {}
    aic: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(int(k_min), int(k_max) + 1):
        b, a, lab = _gmm_bic_aic(feats, k, seed=seed + k)
        bic[k] = b
        aic[k] = a
        labels_by_k[k] = lab

    scores = bic if criterion == "bic" else aic
    k_hat = int(min(scores, key=scores.get))
    return ModeCountResult(
        k_hat=k_hat,
        bic=bic,
        aic=aic,
        residuals=r,
        features=feats,
        labels=labels_by_k[k_hat],
        method=method,
    )


def estimate_mode_count_from_gp(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k_min: int = 1,
    k_max: int = 5,
    criterion: Criterion = "bic",
    seed: int = 42,
    residual_source: Literal["gp", "linear"] = "linear",
    feature_mode: Literal["residual", "residual_local_slope", "residual_gp_jac"] = "residual",
    n_neighbors: int = 15,
) -> ModeCountResult:
    """End-to-end: residuals (+ optional features) → GMM mode count.

    Prefer ``residual_source='linear'`` for co-located modes (clearer residual
    bimodality). Use ``'gp'`` when you specifically want r = y − μ_GP.
    Prefer ``criterion='aic'`` if BIC under-selects K on smeared residuals.
    """
    if residual_source == "gp":
        r = gp_residuals(model, x, y)
        base = "gp_residual"
    else:
        r = linear_midline_residuals(x, y)
        base = "linear_residual"

    extra = None
    method_note = base

    if feature_mode == "residual_local_slope":
        extra = local_data_slopes(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            n_neighbors=n_neighbors,
        )
        method_note = f"{base}+local_slope"
    elif feature_mode == "residual_gp_jac":
        # Weak for co-located modes: ∇μ_GP is shared mid-line slope.
        jac = gp_mean_jacobian_1d(model, x)
        extra = jac
        method_note = f"{base}+gp_jacobian(weak)"

    result = estimate_mode_count(
        r,
        k_min=k_min,
        k_max=k_max,
        criterion=criterion,
        seed=seed,
        extra_features=extra,
        standardize=True,
    )
    result.method = method_note
    return result


def fit_base_gp_and_estimate_k(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k_max: int = 5,
    seed: int = 42,
    num_inits: int = 8,
    device: str = "cpu",
    criterion: Criterion = "aic",
    residual_source: Literal["gp", "linear"] = "linear",
    feature_mode: Literal["residual", "residual_local_slope", "residual_gp_jac"] = "residual",
    fit_gp: bool = True,
) -> tuple[object | None, ModeCountResult]:
    """Optionally fit a GPR, then estimate mode count from residuals.

    For co-located modes, ``residual_source='linear'`` does not require a GP at
    all; set ``fit_gp=False`` to skip Stage-1 training.
    """
    model = None
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64).reshape(-1)

    if fit_gp or residual_source == "gp" or feature_mode == "residual_gp_jac":
        import gpytorch

        from gpplus.kernels import GaussianKernel, LogScaleKernel
        from gpplus.likelihoods import LogGaussianLikelihood
        from gpplus.models.gpr import GPR
        from gpplus.training.trainer import GPTrainer

        input_dim = int(x.shape[-1])
        model = GPR(
            x,
            y,
            likelihood=LogGaussianLikelihood(),
            mean_module=gpytorch.means.ConstantMean(),
            kernel_module=LogScaleKernel(GaussianKernel(ard_num_dims=input_dim)),
        )
        trainer = GPTrainer(
            model,
            num_epochs=1,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=torch.float64,
            n_jobs=1,
        )
        runs = trainer.train()
        valid = [
            r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None
        ]
        if not valid:
            raise RuntimeError("Base GP training failed for all inits.")
        best = min(valid, key=lambda r: r["loss"])
        model.load_state_dict(best["state_dict"])
        model.eval()

    if residual_source == "linear" and feature_mode == "residual":
        r = linear_midline_residuals(x, y)
        result = estimate_mode_count(
            r, k_min=1, k_max=k_max, criterion=criterion, seed=seed
        )
        result.method = "linear_residual"
        return model, result

    if model is None:
        raise RuntimeError("GP model required for residual_source='gp' or jacobian features.")

    result = estimate_mode_count_from_gp(
        model,
        x,
        y,
        k_max=k_max,
        seed=seed,
        criterion=criterion,
        residual_source=residual_source,
        feature_mode=feature_mode,
    )
    return model, result


if __name__ == "__main__":
    import argparse

    from load_bimodal_data import train_test_split_bimodal

    parser = argparse.ArgumentParser(description="Estimate mode count via GP residuals + GMM BIC")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--criterion", type=str, default="aic", choices=["aic", "bic"])
    parser.add_argument(
        "--residual-source",
        type=str,
        default="linear",
        choices=["linear", "gp"],
        help="linear mid-line residuals (recommended) or y-μ_GP",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="residual",
        choices=["residual", "residual_local_slope", "residual_gp_jacobian"],
    )
    parser.add_argument("--no-fit-gp", action="store_true", help="Skip GP when using linear residuals")
    args = parser.parse_args()
    feat = args.feature_mode
    if feat == "residual_gp_jacobian":
        feat = "residual_gp_jac"

    data = train_test_split_bimodal(
        n_samples=args.n_samples, test_size=args.test_size, seed=args.seed
    )
    x = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y = torch.as_tensor(data["y_train"], dtype=torch.float64)
    _model, result = fit_base_gp_and_estimate_k(
        x,
        y,
        k_max=args.k_max,
        seed=args.seed,
        criterion=args.criterion,  # type: ignore[arg-type]
        residual_source=args.residual_source,  # type: ignore[arg-type]
        feature_mode=feat,  # type: ignore[arg-type]
        fit_gp=not args.no_fit_gp,
    )
    print(f"method={result.method}")
    print(f"k_hat={result.k_hat}")
    print("BIC:", {k: round(v, 2) for k, v in result.bic.items()})
    print("AIC:", {k: round(v, 2) for k, v in result.aic.items()})
    print("label counts:", np.bincount(result.labels).tolist())
