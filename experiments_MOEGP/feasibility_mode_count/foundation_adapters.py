"""Fit/predict adapters → common Gaussian-mixture predictive for ScoringBench metrics.

Each adapter returns a dict with:
  mixture_mean : [N]
  expert_means : [N, K]
  expert_stds  : [N, K]
  gate_weights : [N, K]
  extras       : optional metadata (k_eff, k_modes, wall_s, …)
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch


def _as_k1_mixture(mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.asarray(std, dtype=np.float64).ravel()
    sd = np.clip(sd, 1e-6, None)
    n = mu.shape[0]
    return {
        "mixture_mean": mu,
        "expert_means": mu.reshape(n, 1),
        "expert_stds": sd.reshape(n, 1),
        "gate_weights": np.ones((n, 1), dtype=np.float64),
    }


def _point_only(mean: np.ndarray, fallback_std: float) -> dict[str, np.ndarray]:
    """Unimodal Gaussian with constant fallback std when UQ is unavailable."""
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.full_like(mu, max(float(fallback_std), 1e-6))
    return _as_k1_mixture(mu, sd)


def _std_from_quantiles(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    *,
    z: float = 1.2815515655446004,  # Φ^{-1}(0.9) for 10–90%
) -> np.ndarray:
    """Approximate Gaussian σ from a central quantile interval."""
    width = np.asarray(q_hi, dtype=np.float64) - np.asarray(q_lo, dtype=np.float64)
    return np.clip(width / (2.0 * z), 1e-6, None)


# ---------------------------------------------------------------------------
# Exact GP+ GPR
# ---------------------------------------------------------------------------
def fit_predict_exact_gp(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    seed: int = 42,
    num_inits: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    import gpytorch

    from gpplus.kernels import GaussianKernel, LogScaleKernel
    from gpplus.likelihoods import LogGaussianLikelihood
    from gpplus.models.gpr import GPR
    from gpplus.training.eval import evaluate_gp_model
    from gpplus.training.trainer import GPTrainer

    t0 = time.perf_counter()
    x_tr = x_tr.to(dtype=torch.float64)
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1)
    x_te = x_te.to(dtype=torch.float64)

    model = GPR(
        x_tr,
        y_tr,
        likelihood=LogGaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        kernel_module=LogScaleKernel(GaussianKernel(ard_num_dims=int(x_tr.shape[-1]))),
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
    valid = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not valid:
        raise RuntimeError("Exact GP training failed for all inits.")
    best = min(valid, key=lambda r: r["loss"])
    model.load_state_dict(best["state_dict"])
    model.eval()

    mean, _lo, _hi, stddev = evaluate_gp_model(model, x_te)
    wall = time.perf_counter() - t0
    out = _as_k1_mixture(
        mean.detach().cpu().numpy(),
        stddev.detach().cpu().numpy(),
    )
    out["extras"] = {"wall_s": wall, "best_loss": float(best["loss"])}
    return out


# ---------------------------------------------------------------------------
# MidlineSparseMoE-GP
# ---------------------------------------------------------------------------
def _midline_mixture_from_predict(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert MidlineSparse ``predict`` tensors → ScoringBench mixture dict."""
    em = pred["expert_means"].detach().cpu().numpy()
    es = pred["expert_stds"].detach().cpu().numpy()
    gw = pred["gate_weights"].detach().cpu().numpy()
    alpha = pred["alpha"].detach().cpu().numpy()
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, gw.shape)
    mass = np.clip(gw * alpha, 0.0, None)
    mass = mass / np.clip(mass.sum(axis=-1, keepdims=True), 1e-12, None)
    mix = (em * mass).sum(axis=-1)
    return {
        "mixture_mean": mix,
        "expert_means": em,
        "expert_stds": es,
        "gate_weights": mass,
    }


def stage1_mixture_from_core(
    model: Any,
    x_te: torch.Tensor,
) -> dict[str, np.ndarray]:
    """K=1 Gaussian from Midline Stage-1 core GP — *uncapped* σ₀ (Exact-GP parity).

    Do not use ``predict()['std0']`` here: that path clamps σ for Stage-2 mixture
    carving and would falsely make Stage-1 look overconfident.
    """
    from gpplus.training.eval import evaluate_gp_model

    if getattr(model, "core", None) is None:
        raise RuntimeError("MidlineSparse core GP not fitted")
    mean, _lo, _hi, stddev = evaluate_gp_model(model.core, x_te)
    return _as_k1_mixture(
        mean.detach().cpu().numpy(),
        stddev.detach().cpu().numpy(),
    )


def fit_predict_midline_sparse(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    k_max: int = 6,
    num_steps: int = 800,
    seed: int = 42,
    champion: bool = True,
    y_min: float = -2.0,
    y_max: float = 30.0,
    x_probe: list[float] | None = None,
    gate_type: str = "mlp",
    gamma_type: str = "constant",
    adaptive_merge_prune: bool = True,
    include_stage1: bool = False,
    merge_pressure_init: float = 0.0,
    lambda_overrides: dict[str, Any] | None = None,
    num_inits: int = 2,
    num_stage2_inits: int = 2,
    midline_backend: str = "exact",
    ni_sorf_num_rff: int = 2400,
    ni_sorf_num_epochs: int = 1000,
    ni_sorf_lr: float = 1e-2,
    std0_cap_frac: float = 0.3,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from gpplus.models import MidlineSparseMoEGPR

    from diagnostics import dual_report

    t0 = time.perf_counter()
    model = MidlineSparseMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        gate_type=gate_type,  # type: ignore[arg-type]
        offset_type="mlp",
        gamma_type=gamma_type,  # type: ignore[arg-type]
        tau_max=2.5,
        sep_length=2.0,
        keff_tau=0.02,
        free_expert_scale=False,
        use_full_gp_variance=False,
        gamma_init_logit=2.2 if champion else 0.0,
        midline_backend=midline_backend,  # type: ignore[arg-type]
        ni_sorf_num_rff=int(ni_sorf_num_rff),
        ni_sorf_num_epochs=int(ni_sorf_num_epochs),
        ni_sorf_lr=float(ni_sorf_lr),
        **dict(model_kwargs or {}),
    )
    model.std0_cap_frac = float(std0_cap_frac)
    stage2_device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if midline_backend == "ni_sorf"
        else "cpu"
    )
    fit_kwargs: dict[str, Any] = dict(
        num_inits=int(num_inits),
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        seed=seed,
        device=stage1_device,
        n_jobs=1,
        # Honor caller adaptive flag even for classic (non-champion) λ stacks —
        # needed so merge_prune_probe can ablate on A01 without forcing champion.
        adaptive_merge_prune=bool(adaptive_merge_prune),
        try_both_gates=False,
        num_stage2_inits=int(num_stage2_inits),
        log_every=100,
        stage2_device=stage2_device,
        merge_pressure_init=float(merge_pressure_init),
    )
    if champion:
        # Post factorial lean stack (loss-term + background ablations):
        # usage + adaptive_merge + alive + delta + alpha; structure/orphan off.
        fit_kwargs.update(
            lambda_k=0.01,
            lambda_sep=0.0,
            sep_margin=2.2,
            lambda_overlap=0.0,
            overlap_margin=1.0,
            lambda_alpha=0.05,
            lambda_orphan=0.0,
            lambda_delta=0.01,
            lambda_gamma=0.0,
            lambda_rec=0.0,
            rec_margin=1.1,
            lambda_k_start=0.0,
            alive_weight=0.75,
        )
    else:
        fit_kwargs.update(
            lambda_k=0.02 if k_max == 2 else 0.08,
            lambda_sep=0.05,
            lambda_alpha=0.05,
            lambda_orphan=0.05,
        )
    if lambda_overrides:
        fit_kwargs.update(lambda_overrides)
    model.fit(**fit_kwargs)
    wall = time.perf_counter() - t0

    pred = model.predict(x_te)
    out = _midline_mixture_from_predict(pred)

    hist = list(getattr(model, "_fit_history", []) or [])
    _W_KEYS = (
        "w_usage",
        "w_sep",
        "w_alpha",
        "w_orphan",
        "w_alive",
        "w_overlap",
        "w_beta",
        "w_delta",
        "w_gamma",
        "w_rec",
        "w_merge_pull",
        "w_mode",
    )
    fit_history = [
        {
            "step": int(r["step"]),
            "loss": float(r["loss"]),
            "nll": float(r["nll"]),
            "k_eff": float(r.get("k_eff", float("nan"))),
            "k_hat": int(r.get("k_hat", -1)),
            "merge_pressure": float(r.get("merge_pressure", 0.0)),
            **{k: float(r[k]) for k in _W_KEYS if k in r},
        }
        for r in hist
        if "step" in r and "loss" in r
    ]

    loss_terms_final: dict[str, float] | None = None
    if hist:
        last = hist[-1]
        loss_terms_final = {
            "loss": float(last.get("loss", float("nan"))),
            "nll": float(last.get("nll", float("nan"))),
            "k_eff": float(last.get("k_eff", float("nan"))),
            "k_hat": float(last.get("k_hat", float("nan"))),
            "merge_pressure": float(last.get("merge_pressure", 0.0)),
            "lambda_k_t": float(last.get("lambda_k_t", float("nan"))),
            "lambda_sep_t": float(last.get("lambda_sep_t", float("nan"))),
            "lambda_rec_t": float(last.get("lambda_rec_t", float("nan"))),
            "lambda_ov_t": float(last.get("lambda_ov_t", float("nan"))),
            "lambda_g_t": float(last.get("lambda_g_t", float("nan"))),
        }
        for k in _W_KEYS:
            loss_terms_final[k] = float(last.get(k, 0.0))

    extras: dict[str, Any] = {
        "wall_s": wall,
        "k_eff": float(model.k_eff().detach()),
        "k_hat": int(model.k_hat()),
        "k_max": k_max,
        "gate_type": getattr(model, "gate_type", gate_type),
        "gamma_type": getattr(model, "gamma_type", gamma_type),
        "merge_pressure": float(getattr(model, "merge_pressure", 0.0)),
        "adaptive_merge_prune": bool(adaptive_merge_prune),
        "champion": bool(champion),
        "merge_pressure_init": float(merge_pressure_init),
        "try_both_gates": False,
        "num_inits": int(num_inits),
        "num_stage2_inits": int(num_stage2_inits),
        "log_every": 100,
        "stage2_device": stage2_device,
        "fit_history": fit_history,
        "lambda_overrides": dict(lambda_overrides or {}),
    }
    if loss_terms_final is not None:
        extras["loss_terms_final"] = loss_terms_final
    diag = getattr(model, "_adaptive_diagnostics", None)
    if isinstance(diag, dict):
        extras["adaptive_diagnostics"] = {
            k: diag[k]
            for k in (
                "stage1_gp_nll",
                "num_surviving_inits",
                "used_safety_refit",
                "merge_pressure",
                "gate_type",
                "stage2_device",
                "num_stage2_inits",
                "try_both_gates",
            )
            if k in diag
        }
        if "stage1_gp_nll" in diag:
            extras["stage1_gp_nll"] = diag["stage1_gp_nll"]
    try:
        if x_probe is None:
            x_probe = np.linspace(float(x_tr.min()), float(x_tr.max()), 9).tolist()
        report = dual_report(model, x_probe, y_min=y_min, y_max=y_max, n_grid=400)
        extras["k_modes_mean"] = float(report["k_modes_mean"])
        extras["k_modes_round"] = int(report["k_modes_round"])
    except Exception as exc:  # noqa: BLE001 — mode report is optional
        extras["k_modes_error"] = str(exc)

    extras["std0_cap_frac"] = float(std0_cap_frac)
    if getattr(model, "calib_frac", 0.0) > 0.0:
        extras["learned_temperature"] = float(torch.exp(model.log_temp.detach()))
    if getattr(model, "het_auto", False):
        extras["learned_het_w"] = float(torch.sigmoid(model.raw_het_w.detach()))
    if getattr(model, "stage1", "gpplus") == "loo":
        core = model.core
        extras["stage1_loo_nll"] = float(core.best_loss)
        if core.mean_terms:
            extras["stage1_lin_var"] = float(torch.exp(core.log_lin.detach()))
        if getattr(core, "use_feat", False):
            extras["stage1_feat_var"] = float(torch.exp(core.log_feat.detach()))
    out["extras"] = extras
    cal = getattr(model, "_cal_mask", None)
    if cal is not None:
        out["_cal_mask"] = np.asarray(cal, dtype=bool)
    if include_stage1:
        out["stage1"] = stage1_mixture_from_core(model, x_te)
    return out


def _mog_log_lik(mix: dict[str, Any], y: np.ndarray) -> np.ndarray:
    """Per-row log density of a K-component Gaussian mixture dict."""
    w = np.clip(np.asarray(mix["gate_weights"], dtype=np.float64), 1e-300, None)
    m = np.asarray(mix["expert_means"], dtype=np.float64)
    s = np.clip(np.asarray(mix["expert_stds"], dtype=np.float64), 1e-9, None)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    lp = np.log(w) - 0.5 * np.log(2.0 * np.pi * s * s) - 0.5 * ((y - m) / s) ** 2
    mx = lp.max(axis=-1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(axis=-1, keepdims=True))).ravel()


def _slice_mog(mix: dict[str, Any], i0: int, i1: int) -> dict[str, Any]:
    out = {k: np.asarray(mix[k])[i0:i1] for k in (
        "mixture_mean", "expert_means", "expert_stds", "gate_weights"
    )}
    out["extras"] = dict(mix.get("extras") or {})
    return out


def fit_predict_midline_gamma_auto(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    calibrate: bool = True,
    calib_rotate: bool = False,
    het: bool = True,
    loo_mean: bool = True,
    multi_midline: bool = True,
    var_expand: bool = False,
    calib_frac: float = 0.15,
    num_inits: int = 3,
    seed: int = 42,
    y_min: float = -2.0,
    y_max: float = 30.0,
    conf_min: float = 0.85,
    accept: str = "gain",
    loo_backend: str = "auto",
    loo_nigp: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Self-configuring gamma-MLP: one call, no problem knobs, no holdout refits.

    Stage 1 is a LOO-trained GP (``stage1='loo'``; padded-SORF Bayesian linear
    core when ``n`` is large) whose LOO mean/std feed Stage 2; temperature and
    het shrinkage are learned on an in-loop calibration mask; x-separated
    regimes get one midline each when the gated mixture wins on LOO likelihood
    (:func:`gpplus.models.gated_midline.fit_gated_midlines`).

    ``accept="calib"`` additionally fits the single-midline model and keeps the
    multi-midline one only if it also wins on the rows each model held out of
    network training (its in-loop calibration mask).
    """
    from gpplus.models.gated_midline import fit_gated_midlines, gate_probs

    t0 = time.perf_counter()
    x_tr = x_tr.to(dtype=torch.float64)
    x_te = x_te.to(dtype=torch.float64)
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1)
    x_mu = x_tr.mean(dim=0, keepdim=True)
    x_sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-8)
    y_mu = float(y_tr.mean())
    y_sd = float(y_tr.std().clamp_min(1e-8))
    zx_tr, zx_te = (x_tr - x_mu) / x_sd, (x_te - x_mu) / x_sd
    zy_tr = (y_tr - y_mu) / y_sd

    model_kwargs = dict(
        stage1="loo",
        loo_mean_terms=bool(loo_mean),
        loo_backend=str(loo_backend),
        loo_nigp=bool(loo_nigp),
        calib_frac=float(calib_frac) if calibrate else 0.0,
        calib_rotate=bool(calib_rotate),
        het_auto=bool(het),
        var_expand=bool(var_expand),
        var_expand_max_log=1.0,
        vexp_cal_only=bool(var_expand and calibrate),
    )

    def _single(xf: torch.Tensor, yf: torch.Tensor, xe: torch.Tensor) -> dict[str, Any]:
        return fit_predict_midline_sparse(
            xf,
            yf,
            xe,
            seed=seed,
            num_inits=num_inits,
            std0_cap_frac=0.3,
            y_min=(float(y_min) - y_mu) / y_sd,
            y_max=(float(y_max) - y_mu) / y_sd,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    if accept not in ("gain", "calib"):
        raise ValueError(f"accept must be 'gain' or 'calib', got {accept!r}")
    sel: dict[str, Any] = {"n_mid": 1}
    if multi_midline:
        sel = fit_gated_midlines(
            zx_tr,
            zy_tr,
            seed=seed,
            mean_terms=bool(loo_mean),
            conf_min=float(conf_min),
            backend=str(loo_backend),
        )
    n_mid = int(sel["n_mid"])
    n_te = int(zx_te.shape[0])
    need_train = n_mid > 1 and accept == "calib"
    x_eval = torch.cat([zx_te, zx_tr], 0) if need_train else zx_te
    calib_check: dict[str, Any] = {}
    if n_mid == 1:
        out = _single(zx_tr, zy_tr, zx_te)
    else:
        g_ev = gate_probs(sel, x_eval)
        assign = np.asarray(sel["assign"])
        parts, part_idx = [], []
        for m in range(n_mid):
            idx = np.flatnonzero(assign == m)
            part_idx.append(idx)
            ti = torch.as_tensor(idx, dtype=torch.long)
            parts.append(_single(zx_tr[ti], zy_tr[ti], x_eval))
        out = dict(parts[0])
        out["gate_weights"] = np.concatenate(
            [g_ev[:, m : m + 1] * np.asarray(p["gate_weights"]) for m, p in enumerate(parts)],
            axis=-1,
        )
        out["expert_means"] = np.concatenate([np.asarray(p["expert_means"]) for p in parts], -1)
        out["expert_stds"] = np.concatenate([np.asarray(p["expert_stds"]) for p in parts], -1)
        out["mixture_mean"] = (out["gate_weights"] * out["expert_means"]).sum(-1)
        out["extras"] = dict(parts[0].get("extras") or {})
        out["extras"]["sub_extras"] = [p.get("extras") for p in parts]
        if need_train:
            y_np = zy_tr.cpu().numpy()
            n_tr = len(y_np)
            single_out = _single(zx_tr, zy_tr, x_eval)
            masks = [p.get("_cal_mask") for p in parts] + [single_out.get("_cal_mask")]
            if all(mk is not None for mk in masks):
                rows_multi = np.concatenate(
                    [idx[np.asarray(mk, dtype=bool)] for idx, mk in zip(part_idx, masks[:-1])]
                )
                rows_one = np.flatnonzero(np.asarray(masks[-1], dtype=bool))
                ll_multi = float(
                    _mog_log_lik(_slice_mog(out, n_te, n_te + n_tr), y_np)[rows_multi].mean()
                )
                ll_one = float(
                    _mog_log_lik(_slice_mog(single_out, n_te, n_te + n_tr), y_np)[
                        rows_one
                    ].mean()
                )
                calib_check = {
                    "calib_ll_multi": ll_multi,
                    "calib_ll_single": ll_one,
                    "calib_accept": bool(ll_multi > ll_one),
                }
                if ll_multi <= ll_one:
                    out, n_mid = single_out, 1
            out = _slice_mog(out, 0, n_te)
    out.pop("_cal_mask", None)
    out["mixture_mean"] = np.asarray(out["mixture_mean"]) * y_sd + y_mu
    out["expert_means"] = np.asarray(out["expert_means"]) * y_sd + y_mu
    out["expert_stds"] = np.asarray(out["expert_stds"]) * y_sd
    ex = out.setdefault("extras", {})
    ex.update(
        {
            "auto": True,
            "n_midlines": n_mid,
            "midline_selection": {
                k: v for k, v in sel.items() if isinstance(v, (int, float, bool))
            },
            "accept_rule": accept,
            "conf_min": float(conf_min),
            "loo_backend": str(loo_backend),
            **calib_check,
            "wall_s": time.perf_counter() - t0,
        }
    )
    return out


def fit_predict_midline_gamma_t1(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    standardize: bool = True,
    std0_cap_frac: float = 0.3,
    calibrate_temperature: bool = False,
    soft_safety: bool = False,
    soft_safety_weight: float = 0.2,
    holdout_frac: float = 0.15,
    temp_grid: Sequence[float] | None = None,
    num_inits: int = 2,
    seed: int = 42,
    y_min: float = -2.0,
    y_max: float = 30.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tier-1 gamma-MLP: x/y standardization, σ₀-cap, held-out temperature, soft safety.

    ``kwargs`` pass through to :func:`fit_predict_midline_sparse` (``k_max``,
    ``num_steps``, ``champion``, ``gamma_type``, ``model_kwargs`` …).
    """
    t0 = time.perf_counter()
    x_tr = x_tr.to(dtype=torch.float64)
    x_te = x_te.to(dtype=torch.float64)
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1)
    n_tr = int(x_tr.shape[0])

    if standardize:
        x_mu = x_tr.mean(dim=0, keepdim=True)
        x_sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-8)
        y_mu = float(y_tr.mean())
        y_sd = float(y_tr.std().clamp_min(1e-8))
    else:
        x_mu = torch.zeros(1, x_tr.shape[-1], dtype=torch.float64)
        x_sd = torch.ones(1, x_tr.shape[-1], dtype=torch.float64)
        y_mu, y_sd = 0.0, 1.0

    def _zx(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mu) / x_sd

    def _fit(xf: torch.Tensor, yf: torch.Tensor, xe: torch.Tensor) -> dict[str, Any]:
        out = fit_predict_midline_sparse(
            _zx(xf),
            (yf - y_mu) / y_sd,
            _zx(xe),
            seed=seed,
            num_inits=num_inits,
            std0_cap_frac=std0_cap_frac,
            y_min=(float(y_min) - y_mu) / y_sd,
            y_max=(float(y_max) - y_mu) / y_sd,
            **kwargs,
        )
        out["mixture_mean"] = np.asarray(out["mixture_mean"]) * y_sd + y_mu
        out["expert_means"] = np.asarray(out["expert_means"]) * y_sd + y_mu
        out["expert_stds"] = np.asarray(out["expert_stds"]) * y_sd
        return out

    temperature = 1.0
    temp_info: dict[str, Any] = {}
    if calibrate_temperature and n_tr >= 20:
        rng = np.random.default_rng(int(seed) + 17)
        perm = rng.permutation(n_tr)
        n_val = max(5, int(round(float(holdout_frac) * n_tr)))
        val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long)
        fit_idx = torch.as_tensor(perm[n_val:], dtype=torch.long)
        val_pred = _fit(x_tr[fit_idx], y_tr[fit_idx], x_tr[val_idx])
        y_val = y_tr[val_idx].cpu().numpy()
        grid = list(temp_grid) if temp_grid is not None else list(
            np.exp(np.linspace(np.log(0.5), np.log(4.0), 31))
        )
        scores = []
        for t in grid:
            m = dict(val_pred)
            m["expert_stds"] = np.asarray(val_pred["expert_stds"]) * float(t)
            scores.append(float(np.mean(_mog_log_lik(m, y_val))))
        best = int(np.argmax(scores))
        temperature = float(grid[best])
        temp_info = {
            "temperature": temperature,
            "temp_val_loglik": scores[best],
            "temp_val_loglik_t1": float(scores[int(np.argmin(np.abs(np.log(grid))))]),
            "temp_n_val": int(n_val),
        }

    x_eval = torch.cat([x_te, x_tr], dim=0) if soft_safety else x_te
    out = _fit(x_tr, y_tr, x_eval)
    n_te = int(x_te.shape[0])
    if soft_safety:
        full = out
        out = _slice_mog(full, 0, n_te)
        tr = _slice_mog(full, n_te, n_te + n_tr)
        resid_sd = float(np.std(y_tr.cpu().numpy() - tr["mixture_mean"]))
        diag = (full.get("extras") or {}).get("adaptive_diagnostics") or {}
        used_safety = bool(diag.get("used_safety_refit", False))
        if used_safety:
            w = float(soft_safety_weight)
            gw = np.asarray(out["gate_weights"]) * (1.0 - w)
            mm = np.asarray(out["mixture_mean"]).reshape(-1, 1)
            broad = np.full_like(mm, max(resid_sd, 1e-6) * 1.5)
            out["gate_weights"] = np.concatenate([gw, np.full_like(mm, w)], axis=-1)
            out["expert_means"] = np.concatenate([out["expert_means"], mm], axis=-1)
            out["expert_stds"] = np.concatenate([out["expert_stds"], broad], axis=-1)
        out["extras"]["soft_safety_applied"] = used_safety
        out["extras"]["train_resid_sd"] = resid_sd

    if temperature != 1.0:
        out["expert_stds"] = np.asarray(out["expert_stds"]) * temperature
    ex = out.setdefault("extras", {})
    ex.update(temp_info)
    ex.update(
        {
            "tier1": True,
            "standardize": bool(standardize),
            "calibrate_temperature": bool(calibrate_temperature),
            "soft_safety": bool(soft_safety),
            "temperature": temperature,
            "wall_s": time.perf_counter() - t0,
        }
    )
    return out


# ---------------------------------------------------------------------------
# TabGP
# ---------------------------------------------------------------------------
def _tabgp_engine_gpplus_dir() -> str | None:
    """Directory containing TabGP's vendored ``gpplus`` package."""
    try:
        import tabgp

        engine_dir = Path(tabgp.__file__).resolve().parent / "_engine"
        if (engine_dir / "gpplus" / "__init__.py").is_file():
            return str(engine_dir)
    except Exception:  # noqa: BLE001
        return None
    return None


@contextlib.contextmanager
def _tabgp_gpplus_shadow():
    """Prefer TabGP's vendored gpplus over SpectralGPPlus during engine runs."""
    import sys

    engine_dir = _tabgp_engine_gpplus_dir()
    if engine_dir is None:
        yield
        return

    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "gpplus" or k.startswith("gpplus.")
    }
    inserted = False
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(engine_dir)
            except ValueError:
                pass
        for k in list(sys.modules):
            if k == "gpplus" or k.startswith("gpplus."):
                sys.modules.pop(k, None)
        sys.modules.update(saved_modules)


def fit_predict_tabgp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    seed: int = 42,
    head: str = "mll",
    device: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    # Prefer math attention: flashdet needs flash-attn (often missing on Windows).
    engine_args = ["--attn_backend", "math"]
    with _tabgp_gpplus_shadow():
        from tabgp import TabGPRegressor

        model = TabGPRegressor(
            head=head,
            seed=seed,
            device=device,
            verbose=False,
            engine_args=engine_args,
        )
        model.fit(
            np.asarray(x_tr, dtype=np.float64),
            np.asarray(y_tr, dtype=np.float64).ravel(),
        )
        mean, std = model.predict(
            np.asarray(x_te, dtype=np.float64), return_std=True
        )
    wall = time.perf_counter() - t0
    out = _as_k1_mixture(mean, std)
    out["extras"] = {"wall_s": wall, "head": head}
    return out


# ---------------------------------------------------------------------------
# TabPFN 3.0 / 3.5
# ---------------------------------------------------------------------------
def _require_tabpfn_auth() -> None:
    """Fail fast when TabPFN 3.x weights need a Prior Labs token.

    Interactive browser login hangs under non-TTY agent runs; require
    ``TABPFN_TOKEN`` or a cached token file instead.
    """
    import os
    from pathlib import Path

    if os.environ.get("TABPFN_TOKEN", "").strip():
        return
    for path in (
        Path.home() / ".cache" / "tabpfn" / "auth_token",
        Path.home() / ".tabpfn" / "token",
    ):
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return
    # Allow forcing the interactive path explicitly
    if os.environ.get("TABPFN_ALLOW_INTERACTIVE_AUTH", "").strip() in (
        "1",
        "true",
        "True",
    ):
        return
    raise RuntimeError(
        "TabPFN 3.x needs a one-time Prior Labs license token. "
        "Set TABPFN_TOKEN (from https://ux.priorlabs.ai/account) or run once "
        "interactively with TABPFN_ALLOW_INTERACTIVE_AUTH=1 so the browser "
        "login can complete. See FOUNDATION_COMPARE.md."
    )


def fit_predict_tabpfn(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    version: str = "v3.5",
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_map = {
        "v3": ModelVersion.V3,
        "3": ModelVersion.V3,
        "v3.0": ModelVersion.V3,
        "3.0": ModelVersion.V3,
        "v3.5": ModelVersion.V3_5,
        "3.5": ModelVersion.V3_5,
        "v3.5-fast": ModelVersion.V3_5_FAST,
    }
    key = str(version).lower()
    if key not in version_map:
        raise ValueError(f"Unknown TabPFN version {version!r}; choose from {sorted(version_map)}")
    mv = version_map[key]

    _require_tabpfn_auth()

    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"random_state": seed}
    if device is not None:
        kwargs["device"] = device
    try:
        model = TabPFNRegressor.create_default_for_version(mv, **kwargs)
    except TypeError:
        model = TabPFNRegressor.create_default_for_version(mv)
        if hasattr(model, "device") and device is not None:
            model.device = device

    Xtr = np.asarray(x_tr, dtype=np.float64)
    ytr = np.asarray(y_tr, dtype=np.float64).ravel()
    Xte = np.asarray(x_te, dtype=np.float64)
    model.fit(Xtr, ytr)

    uq_ok = True
    pred_borders: np.ndarray | None = None
    pred_logits: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None
    quantile_values: np.ndarray | None = None
    try:
        # ``full`` returns bar-distribution criterion + logits (true PPD bins)
        full = model.predict(Xte, output_type="full")
        mean = np.asarray(
            full["mean"] if isinstance(full, dict) else full.mean,
            dtype=np.float64,
        ).ravel()
        quants = full["quantiles"] if isinstance(full, dict) else full.quantiles
        crit = full["criterion"] if isinstance(full, dict) else full.criterion
        logits = full["logits"] if isinstance(full, dict) else full.logits
        if logits is not None and crit is not None:
            pred_borders = (
                crit.borders.detach().cpu().numpy().astype(np.float64)
            )
            pred_logits = (
                logits.detach().cpu().numpy().astype(np.float64)
            )
            # ScoringBench still uses a K=1 Gaussian approx from 10–90%
            if quants is not None and len(quants) >= 2:
                q_list = list(quants)
                q_lo = np.asarray(q_list[0 if len(q_list) < 9 else 0])
                q_hi = np.asarray(q_list[-1 if len(q_list) < 9 else 8])
                std = _std_from_quantiles(q_lo, q_hi)
                # dense quantile table for optional CDF-diff density
                n_q = len(q_list)
                quantile_levels = np.linspace(0.1, 0.9, n_q) if n_q == 9 else np.linspace(
                    0.0, 1.0, n_q + 2
                )[1:-1]
                if len(quantile_levels) == n_q:
                    quantile_values = np.column_stack(
                        [np.asarray(q, dtype=np.float64).ravel() for q in q_list]
                    )
            else:
                # fallback σ from bar second moment
                borders = pred_borders
                widths = np.diff(borders)
                centers = borders[:-1] + 0.5 * widths
                z = pred_logits - pred_logits.max(axis=-1, keepdims=True)
                p = np.exp(z)
                p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-300, None)
                m1 = (p * centers).sum(axis=-1)
                m2 = (p * (centers**2 + (widths**2) / 12.0)).sum(axis=-1)
                std = np.sqrt(np.clip(m2 - m1**2, 1e-12, None))
        else:
            uq_ok = False
            std = np.full_like(mean, max(float(np.std(ytr)), 1e-3))
    except Exception:
        mean = np.asarray(model.predict(Xte, output_type="mean"), dtype=np.float64)
        uq_ok = False
        std = np.full_like(mean, max(float(np.std(ytr)), 1e-3))

    wall = time.perf_counter() - t0
    out = _as_k1_mixture(mean, std)
    if pred_borders is not None and pred_logits is not None:
        out["pred_borders"] = pred_borders
        out["pred_logits"] = pred_logits
    if quantile_levels is not None and quantile_values is not None:
        out["quantile_levels"] = quantile_levels
        out["quantile_values"] = quantile_values
    out["extras"] = {
        "wall_s": wall,
        "version": str(mv.value if hasattr(mv, "value") else mv),
        "uq_from_quantiles": uq_ok,
        "ppd_source": (
            "bar_distribution"
            if pred_borders is not None
            else ("quantiles" if quantile_values is not None else "gaussian_fallback")
        ),
    }
    return out


# ---------------------------------------------------------------------------
# NVIDIA KumoTabular (structured-data-models) — quantile regression PPD
# ---------------------------------------------------------------------------
def fit_predict_kumotabular(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    device: str | None = None,
    num_estimators: int = 8,
    size: str = "medium",
) -> dict[str, Any]:
    """Fit/predict with ``sdm.models.KumoTabular`` (999 quantiles ``q001``–``q999``).

    Native UQ for UQ comparisons. ScoringBench still summarizes to a K=1
    Gaussian via 10–90% quantiles; marginal plots use quantile CDF differences.
    """
    import pandas as pd
    import sdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if size not in ("small", "medium", "large"):
        raise ValueError(f"size must be small|medium|large, got {size!r}")

    t0 = time.perf_counter()
    Xtr = np.asarray(x_tr, dtype=np.float64)
    ytr = np.asarray(y_tr, dtype=np.float64).ravel()
    Xte = np.asarray(x_te, dtype=np.float64)

    n_te = int(Xte.shape[0])
    d = int(Xtr.shape[1])
    feat_cols = [f"x{j}" for j in range(d)]
    df_tr = pd.DataFrame(Xtr, columns=feat_cols)
    df_tr["target"] = ytr
    df_te = pd.DataFrame(Xte, columns=feat_cols)
    df_te["target"] = 0.0

    stypes = {c: "numerical" for c in feat_cols}
    stypes["target"] = "numerical"

    table_tr = sdm.TableTensor.from_pandas(df_tr, stypes=stypes, device=device)
    table_te = sdm.TableTensor.from_pandas(df_te, stypes=stypes, device=device)

    try:
        from sdm import Task

        task = Task.regression
    except Exception:  # noqa: BLE001
        task = "regression"

    model = sdm.models.KumoTabular(
        task=task,
        size=size,  # type: ignore[arg-type]
        pretrained=True,
        device=device,
    )
    x_ctx = table_tr.drop_columns("target")
    y_ctx = table_tr[:, "target"]
    x_query = table_te.drop_columns("target")

    # Default recipe: SortQuantiles + trimmed AverageEstimators → q001..q999.
    model.fit(x=x_ctx, y=y_ctx, num_estimators=num_estimators)
    out_tt = model.predict(x_query)
    model.clear()

    pdf = out_tt.to_pandas()
    q_cols = sorted(
        (c for c in pdf.columns if str(c).startswith("q") and str(c)[1:].isdigit()),
        key=lambda c: int(str(c)[1:]),
    )
    if not q_cols:
        raise RuntimeError(
            "KumoTabular returned no quantile columns; got "
            f"{list(pdf.columns)!r}"
        )

    q_mat = pdf[q_cols].to_numpy(dtype=np.float64)
    levels = np.asarray([int(str(c)[1:]) / 1000.0 for c in q_cols], dtype=np.float64)
    # Approximate predictive mean by averaging quantile levels (Riemann).
    mean = q_mat.mean(axis=1)
    if "q100" in pdf.columns and "q900" in pdf.columns:
        std = _std_from_quantiles(
            pdf["q100"].to_numpy(dtype=np.float64),
            pdf["q900"].to_numpy(dtype=np.float64),
        )
    else:
        # Fallback: nearest available ~10% / ~90% columns
        i_lo = int(np.argmin(np.abs(levels - 0.1)))
        i_hi = int(np.argmin(np.abs(levels - 0.9)))
        std = _std_from_quantiles(q_mat[:, i_lo], q_mat[:, i_hi])

    wall = time.perf_counter() - t0
    out = _as_k1_mixture(mean, std)
    out["quantile_levels"] = levels
    out["quantile_values"] = q_mat
    out["extras"] = {
        "wall_s": wall,
        "device": device,
        "num_estimators": num_estimators,
        "size": size,
        "uq_from_quantiles": True,
        "ppd_source": "quantile_diff",
        "model": "KumoTabular",
        "n_quantiles": int(len(q_cols)),
        "n_te": n_te,
    }
    return out


# ---------------------------------------------------------------------------
# Bishop-style Mixture Density Network (MDN)
# ---------------------------------------------------------------------------
class _BishopMDN(torch.nn.Module):
    """MLP → (π, μ, softplus σ) mixture; optional constant (x-independent) gate."""

    def __init__(
        self,
        input_dim: int,
        k: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        constant_gate: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.k = int(k)
        self.constant_gate = bool(constant_gate)
        layers: list[torch.nn.Module] = []
        prev = int(input_dim)
        for h in hidden:
            layers.append(torch.nn.Linear(prev, int(h)))
            layers.append(torch.nn.Tanh())
            prev = int(h)
        self.trunk = torch.nn.Sequential(*layers).to(dtype=dtype)
        self.mu_head = torch.nn.Linear(prev, self.k).to(dtype=dtype)
        self.log_sigma_head = torch.nn.Linear(prev, self.k).to(dtype=dtype)
        if self.constant_gate:
            self.pi_logits = torch.nn.Parameter(torch.zeros(self.k, dtype=dtype))
            self.pi_head = None
        else:
            self.pi_head = torch.nn.Linear(prev, self.k).to(dtype=dtype)
            self.pi_logits = None
        with torch.no_grad():
            for mod in (self.mu_head, self.log_sigma_head):
                torch.nn.init.normal_(mod.weight, std=0.05)
                torch.nn.init.zeros_(mod.bias)
            if self.pi_head is not None:
                torch.nn.init.zeros_(self.pi_head.weight)
                torch.nn.init.zeros_(self.pi_head.bias)
            self.log_sigma_head.bias.fill_(-0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.mu_head(h)
        std = torch.nn.functional.softplus(self.log_sigma_head(h)) + 1e-4
        if self.constant_gate:
            assert self.pi_logits is not None
            logits = self.pi_logits.expand(x.shape[0], -1)
        else:
            assert self.pi_head is not None
            logits = self.pi_head(h)
        pi = torch.softmax(logits, dim=-1)
        return pi, mu, std


def _mdn_mixture_nll(
    y: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    # y: [N], pi/mu/std: [N, K]
    y = y.reshape(-1, 1)
    var = (std * std).clamp_min(1e-12)
    log_comp = -0.5 * (torch.log(2.0 * torch.pi * var) + (y - mu).pow(2) / var)
    log_mix = torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_comp, dim=-1)
    return (-log_mix).mean()


def fit_predict_mdn(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    k_max: int = 5,
    num_steps: int = 2000,
    lr: float = 1e-2,
    seed: int = 42,
    num_inits: int = 3,
    hidden: tuple[int, ...] = (64, 64),
    constant_gate: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Bishop MDN baseline → same MoG dict as other foundation adapters.

    Free component means/scales (no GP midline). ``constant_gate=True`` makes
    mixture weights x-independent (co-located-friendly ablation ``mdn_const``).
    """
    t0 = time.perf_counter()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    x_tr = x_tr.to(device=device, dtype=dtype)
    y_tr = y_tr.to(device=device, dtype=dtype).reshape(-1)
    x_te = x_te.to(device=device, dtype=dtype)
    if x_tr.ndim == 1:
        x_tr = x_tr.unsqueeze(-1)
    if x_te.ndim == 1:
        x_te = x_te.unsqueeze(-1)
    d = int(x_tr.shape[-1])
    k = int(k_max)

    # Standardize y for stable MDN training; unscale means/stds at predict.
    y_mean = float(y_tr.mean().item())
    y_std = float(y_tr.std(unbiased=False).clamp_min(1e-6).item())
    y_z = (y_tr - y_mean) / y_std

    best_state: dict[str, Any] | None = None
    best_nll = float("inf")
    for init_i in range(int(num_inits)):
        torch.manual_seed(int(seed) + 17 * init_i)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed) + 17 * init_i)
        model = _BishopMDN(
            d, k, hidden=hidden, constant_gate=constant_gate, dtype=dtype
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        model.train()
        for _ in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            pi, mu, std = model(x_tr)
            loss = _mdn_mixture_nll(y_z, pi, mu, std)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pi, mu, std = model(x_tr)
            nll = float(_mdn_mixture_nll(y_z, pi, mu, std).item())
        if nll < best_nll:
            best_nll = nll
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    assert best_state is not None
    model = _BishopMDN(
        d, k, hidden=hidden, constant_gate=constant_gate, dtype=dtype
    ).to(device)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pi, mu_z, std_z = model(x_te)
        # Unscale: y = y_std * z + y_mean
        mu = mu_z * y_std + y_mean
        std = std_z * y_std
        mix_mean = (pi * mu).sum(dim=-1)

    wall = time.perf_counter() - t0
    em = mu.detach().cpu().numpy()
    es = std.detach().cpu().numpy()
    gw = pi.detach().cpu().numpy()
    mm = mix_mean.detach().cpu().numpy()
    return {
        "mixture_mean": mm,
        "expert_means": em,
        "expert_stds": np.clip(es, 1e-6, None),
        "gate_weights": gw,
        "extras": {
            "wall_s": wall,
            "device": device,
            "k_max": k,
            "train_nll_z": best_nll,
            "constant_gate": bool(constant_gate),
            "num_inits": int(num_inits),
            "num_steps": int(num_steps),
            "model": "mdn_const" if constant_gate else "mdn",
        },
    }


def fit_predict_mdn_const(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    **kwargs: Any,
) -> dict[str, Any]:
    """MDN with x-independent mixture weights (constant gate)."""
    kwargs = dict(kwargs)
    kwargs["constant_gate"] = True
    return fit_predict_mdn(x_tr, y_tr, x_te, **kwargs)


# Backward-compatible name: older scripts imported fit_predict_tabfm.
def fit_predict_tabfm(*args, **kwargs) -> dict[str, Any]:
    """Deprecated alias → ``fit_predict_kumotabular`` (TabFM had no regression PPD)."""
    return fit_predict_kumotabular(*args, **kwargs)


# ---------------------------------------------------------------------------
# Histogram GP baselines (TabPFN-style bars, small B)
# ---------------------------------------------------------------------------
def _as_tensors_xy(
    x_tr: Any,
    y_tr: Any,
    x_te: Any,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Normalize adapter inputs to float64 tensors + numpy y."""
    if isinstance(x_tr, torch.Tensor):
        Xtr_t = x_tr.to(dtype=torch.float64)
    else:
        Xtr_t = torch.as_tensor(np.asarray(x_tr, dtype=np.float64), dtype=torch.float64)
    if isinstance(x_te, torch.Tensor):
        Xte_t = x_te.to(dtype=torch.float64)
    else:
        Xte_t = torch.as_tensor(np.asarray(x_te, dtype=np.float64), dtype=torch.float64)
    if isinstance(y_tr, torch.Tensor):
        y_np = y_tr.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        y_np = np.asarray(y_tr, dtype=np.float64).ravel()
    Xtr_np = Xtr_t.detach().cpu().numpy().astype(np.float64)
    return Xtr_t, Xtr_np, Xte_t, y_np


def _pack_bar_predictive(
    borders: np.ndarray,
    probs: np.ndarray,
    *,
    wall_s: float,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mixture summary + bar PPD fields for ScoringBench / marginal plots."""
    from histogram_gp import predictive_moments_from_bars, probs_to_bar_logits

    mean, std = predictive_moments_from_bars(borders, probs)
    out = _as_k1_mixture(mean, std)
    out["pred_borders"] = np.asarray(borders, dtype=np.float64)
    out["pred_logits"] = probs_to_bar_logits(probs)
    ex = {
        "wall_s": float(wall_s),
        "ppd_source": "bar_distribution",
        "n_bins": int(np.asarray(borders).size - 1),
    }
    if extras:
        ex.update(extras)
    out["extras"] = ex
    return out


def _fit_hist_ni_sorf_bins(
    train_x: torch.Tensor,
    onehot: np.ndarray,
    test_x: torch.Tensor,
    *,
    freqs: np.ndarray,
    seed: int,
    num_rff: int = 512,
    num_epochs: int = 150,
    lr: float = 1e-2,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One NI-SORF GP per histogram bin (regression on one-hot), then simplex project."""
    from histogram_gp import project_onto_simplex
    from ni_sorf_midline import fit_ni_sorf_core, predict_ni_sorf

    n, b = onehot.shape
    n_te = int(test_x.shape[0])
    means = np.zeros((b, n_te), dtype=np.float64)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    for j in range(b):
        yj = torch.as_tensor(onehot[:, j], dtype=torch.float64)
        model, _info = fit_ni_sorf_core(
            train_x,
            yj,
            num_rff=int(num_rff),
            num_epochs=int(num_epochs),
            num_inits=1,
            lr=float(lr),
            seed=int(seed) + j,
            device=device,
            freeze_epoch_nigp=min(40, int(num_epochs) // 3),
            nigp_slope_refreshes=4,
        )
        mu, _std = predict_ni_sorf(model, test_x)
        means[j] = mu.detach().cpu().numpy().ravel()
    # Project each test column onto simplex (nonneg + sum=1)
    probs = np.zeros((n_te, b), dtype=np.float64)
    for i in range(n_te):
        probs[i] = project_onto_simplex(means[:, i])
    return probs, {
        "backend": "ni_sorf_bins",
        "num_rff": int(num_rff),
        "num_epochs": int(num_epochs),
        "n_bins": int(b),
        "device": device,
        "freq_prior": freqs.tolist(),
    }


def fit_predict_hist_mogp(
    x_tr: Any,
    y_tr: Any,
    x_te: Any,
    *,
    n_bins: int = 200,
    pad_frac: float = 0.05,
    seed: int = 42,
    num_inits: int = 2,
    num_steps: int = 1000,
    lr: float = 0.03,
    device: str = "cpu",
    sample_project: bool = False,
    n_proj_samples: int = 64,
    use_mtgpr: bool | None = None,
    backend: str = "exact",
    bin_scheme: str = "equal_width",
    tail_frac: float = 0.0,
    floor_eps: float = 0.0,
    ni_sorf_num_rff: int = 512,
    ni_sorf_num_epochs: int = 150,
) -> dict[str, Any]:
    """Idea 1A: one-hot Kronecker multi-output GP + simplex projection.

    Exact backend: separable ``K_XX ⊗ K_HH + σ_n² I`` with ``K_HH`` an RBF on
    bin centers. ``B`` is capped at ``n_train`` (hard one-hots). Borders are
    equal-width by default (faithful ``π/w`` density). MLL/predict use paired
    eigendecomps (``O(N³+B³)``). Default training: 1000 Adam steps at lr=0.03
    (2 inits). Posterior mean → simplex (A); ``sample_project=True`` for B.
    ``backend='ni_sorf'`` keeps the large-n per-bin path. ``use_mtgpr`` ignored.
    """
    from histogram_gp import (
        bin_centers,
        empirical_bin_frequencies,
        floor_mix,
        make_hist_bins,
        smoothed_marginal,
        y_to_onehot,
    )

    del use_mtgpr  # deprecated; Kronecker is the exact multi-output path

    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    Xtr_t, _Xtr_np, Xte_t, y_np = _as_tensors_xy(x_tr, y_tr, x_te)
    borders, bin_info = make_hist_bins(
        y_np,
        n_bins=int(n_bins),
        pad_frac=float(pad_frac),
        scheme=str(bin_scheme),
        tail_frac=float(tail_frac),
    )
    oh = y_to_onehot(y_np, borders)  # [N, B]
    freqs = empirical_bin_frequencies(y_np, borders)
    centers = bin_centers(borders)

    if backend == "ni_sorf":
        probs, train_info = _fit_hist_ni_sorf_bins(
            Xtr_t.to(dtype=torch.float64),
            oh,
            Xte_t.to(dtype=torch.float64),
            freqs=freqs,
            seed=seed,
            num_rff=ni_sorf_num_rff,
            num_epochs=ni_sorf_num_epochs,
            lr=1e-2,
            device=("cuda" if torch.cuda.is_available() else device),
        )
    else:
        Xtr_t = Xtr_t.to(device=device)
        Xte_t = Xte_t.to(device=device)
        probs, train_info = _fit_hist_kronecker_mogp(
            Xtr_t,
            oh,
            Xte_t,
            centers=centers,
            freqs=freqs,
            seed=seed,
            num_inits=num_inits,
            num_steps=num_steps,
            lr=lr,
            device=device,
            sample_project=sample_project,
            n_proj_samples=n_proj_samples,
        )

    if float(floor_eps) > 0.0:
        probs = floor_mix(probs, smoothed_marginal(y_np, borders), float(floor_eps))

    wall = time.perf_counter() - t0
    return _pack_bar_predictive(
        borders,
        probs,
        wall_s=wall,
        extras={
            "model": "hist_mogp",
            "sample_project": bool(sample_project),
            "gp_backend": backend,
            "floor_eps": float(floor_eps),
            **bin_info,
            **train_info,
        },
    )


def _bar_probs(pred: dict[str, Any]) -> np.ndarray:
    z = np.asarray(pred["pred_logits"], dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def fit_predict_hist_t1(
    x_tr: Any,
    y_tr: Any,
    x_te: Any,
    *,
    base_fn: Callable[..., dict[str, Any]] | None = None,
    bin_grid: Sequence[int] | None = (10, 20, 40, 80),
    eps_grid: Sequence[float] | None = (0.0, 1e-3, 1e-2, 0.03, 0.1, 0.2),
    tail_frac: float = 1.0,
    n_bins: int = 200,
    holdout_frac: float = 0.2,
    select_steps: int = 300,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tier-1 hist wrapper: tail bins + held-out (B, ε) selection, then refit.

    ``bin_grid=None`` keeps ``n_bins`` fixed; ``eps_grid=None`` disables the floor.
    ``base_fn`` defaults to :func:`fit_predict_hist_mogp` (``hist_gpd`` also works).
    """
    from histogram_gp import bar_log_prob, floor_mix, probs_to_bar_logits, smoothed_marginal

    fn = base_fn or fit_predict_hist_mogp
    t0 = time.perf_counter()
    Xtr_t, _x, Xte_t, y_np = _as_tensors_xy(x_tr, y_tr, x_te)
    n = int(y_np.size)
    rng = np.random.default_rng(int(seed) + 29)
    perm = rng.permutation(n)
    n_val = max(5, int(round(float(holdout_frac) * n)))
    va, fi = perm[:n_val], perm[n_val:]
    b_cands = [int(n_bins)] if not bin_grid else [
        int(b) for b in bin_grid if int(b) <= max(2, fi.size // 2)
    ] or [max(2, fi.size // 4)]
    e_cands = [0.0] if not eps_grid else [float(e) for e in eps_grid]
    step_kw = {"num_steps": int(select_steps)} if fn is fit_predict_hist_mogp else {}

    sel: list[dict[str, Any]] = []
    need_select = len(b_cands) > 1 or len(e_cands) > 1
    if need_select:
        xf = Xtr_t[torch.as_tensor(fi)]
        xv = Xtr_t[torch.as_tensor(va)]
        for b in b_cands:
            try:
                pv = fn(
                    xf, y_np[fi], xv, n_bins=b, tail_frac=tail_frac, seed=seed,
                    **{**kwargs, **step_kw},
                )
            except RuntimeError as exc:
                sel.append({"n_bins": b, "eps": None, "val_log": None, "error": str(exc)})
                continue
            borders = np.asarray(pv["pred_borders"])
            probs = _bar_probs(pv)
            marg = smoothed_marginal(y_np[fi], borders)
            for e in e_cands:
                p = floor_mix(probs, marg, e) if e > 0 else probs
                lp = bar_log_prob(y_np[va], borders, probs_to_bar_logits(p))
                sel.append({"n_bins": b, "eps": e, "val_log": float(np.mean(lp))})
        ok = [r for r in sel if r["val_log"] is not None]
        if not ok:
            raise RuntimeError(f"hist tier-1 selection: every candidate failed ({sel})")
        best = max(ok, key=lambda r: r["val_log"])
        b_star, e_star = int(best["n_bins"]), float(best["eps"])
    else:
        b_star, e_star = b_cands[0], e_cands[0]

    out = fn(Xtr_t, y_np, Xte_t, n_bins=b_star, tail_frac=tail_frac, seed=seed, **kwargs)
    if e_star > 0:
        borders = np.asarray(out["pred_borders"])
        p = floor_mix(_bar_probs(out), smoothed_marginal(y_np, borders), e_star)
        from histogram_gp import predictive_moments_from_bars

        mean, std = predictive_moments_from_bars(borders, p)
        k1 = _as_k1_mixture(mean, std)
        out.update(k1)
        out["pred_logits"] = probs_to_bar_logits(p)
    ex = out.setdefault("extras", {})
    ex.update(
        {
            "tier1": True,
            "selected_n_bins": b_star,
            "selected_floor_eps": e_star,
            "tail_frac": float(tail_frac),
            "selection": sel,
            "wall_s": time.perf_counter() - t0,
        }
    )
    return out


class _SafeEigh(torch.autograd.Function):
    """Symmetric ``eigh`` whose backward drops near-degenerate eigen-gap terms.

    Exact for losses invariant to rotations inside degenerate eigenspaces
    (log-det / quadratic forms); stock ``eigh`` backward divides by ``λ_j − λ_i``
    and returns inf/NaN when isolated tail bins give repeated eigenvalues.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor):
        evals, evecs = torch.linalg.eigh(a)
        ctx.save_for_backward(evals, evecs)
        return evals, evecs

    @staticmethod
    def backward(ctx, g_evals: torch.Tensor, g_evecs: torch.Tensor):
        evals, evecs = ctx.saved_tensors
        gap = evals.unsqueeze(-2) - evals.unsqueeze(-1)  # [i,j] = λ_j − λ_i
        tol = 1e-9 * evals.abs().max().clamp_min(1.0)
        f = torch.where(gap.abs() > tol, 1.0 / gap, torch.zeros_like(gap))
        inner = torch.zeros_like(gap)
        if g_evecs is not None:
            inner = f * (evecs.transpose(-1, -2) @ g_evecs)
        if g_evals is not None:
            inner = inner + torch.diag_embed(g_evals)
        g_a = evecs @ inner @ evecs.transpose(-1, -2)
        return 0.5 * (g_a + g_a.transpose(-1, -2))


def _fit_hist_kronecker_mogp(
    train_x: torch.Tensor,
    onehot: np.ndarray,
    test_x: torch.Tensor,
    *,
    centers: np.ndarray,
    freqs: np.ndarray,
    seed: int,
    num_inits: int,
    num_steps: int,
    lr: float,
    device: str,
    sample_project: bool,
    n_proj_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Separable Kronecker MOGP via paired eigendecomps (not GPyTorch ExactGP).

    Model: ``Σ = K_XX ⊗ K_HH + σ_n² I`` with ``K_HH`` an RBF on bin centers.
    MLL / solves use ``eigh(K_XX)``, ``eigh(K_HH)`` so cost is ``O(N³+B³)`` per
    step — GPyTorch's ExactGP+MultitaskKernel path does *not* do this once
    isotropic noise is added and can allocate ``O((NB)²)``.
    """
    import math

    import gpytorch
    from torch import nn

    from histogram_gp import project_onto_simplex, project_samples_average

    n, b = onehot.shape
    input_dim = int(train_x.shape[-1])
    train_y = torch.as_tensor(onehot, dtype=torch.float64, device=device)  # [N, B]
    train_x = train_x.to(device=device, dtype=torch.float64)
    test_x = test_x.to(device=device, dtype=torch.float64)
    # Standardize inputs (concrete / fish features are badly scaled for RBF defaults).
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    # Scale bin centers to unit range so ℓ_H is O(1).
    centers_t = torch.as_tensor(centers, dtype=torch.float64, device=device).reshape(-1)
    c_lo = centers_t.min()
    c_hi = centers_t.max()
    c_span = (c_hi - c_lo).clamp_min(1e-6)
    centers_t = ((centers_t - c_lo) / c_span).reshape(-1, 1)
    freq_t = torch.as_tensor(freqs, dtype=torch.float64, device=device)
    if float(freq_t.sum().item()) <= 0:
        freq_t = torch.full((b,), 1.0 / float(b), dtype=torch.float64, device=device)

    class _KroneckerHistEigGP(nn.Module):
        """Dense Kronecker MOGP with structured Exact MLL (Kx⊗Kh + σ²I)."""

        def __init__(self) -> None:
            super().__init__()
            # Bound lengthscales / outputscales: unit-scaled x / bin centers → O(1).
            # Unconstrained Adam + eigh autograd otherwise NaNs when kernels
            # approach rank deficiency (near-constant columns).
            ls_box = gpytorch.constraints.Interval(5e-2, 5.0)
            os_box = gpytorch.constraints.Interval(1e-2, 5.0)
            self.data_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=input_dim, lengthscale_constraint=ls_box
                ),
                outputscale_constraint=os_box,
            )
            self.bin_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=1, lengthscale_constraint=ls_box
                ),
                outputscale_constraint=os_box,
            )
            self.register_buffer("centers", centers_t.detach().clone())
            # Multitask constant mean (one value per bin), init → empirical freqs.
            self.raw_mean = nn.Parameter(freq_t.detach().clone())
            self.raw_noise = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
            self._jitter = 1e-4

        def noise(self) -> torch.Tensor:
            # Floor keeps Λ = λx λh + σ² away from zero when evals are tiny.
            return torch.nn.functional.softplus(self.raw_noise) + 1e-3

        def mean_row(self) -> torch.Tensor:
            return self.raw_mean

        def _kx(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
            k = self.data_kernel(x1, x2).to_dense()
            if x2 is None:
                k = k + self._jitter * torch.eye(
                    k.shape[-1], dtype=k.dtype, device=k.device
                )
            return k

        def _kh(self) -> torch.Tensor:
            k = self.bin_kernel(self.centers).to_dense()
            return k + self._jitter * torch.eye(
                k.shape[-1], dtype=k.dtype, device=k.device
            )

        def neg_mll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Exact NLL under Σ = Kx⊗Kh + σ²I using paired eigh (autograd-safe)."""
            kx = self._kx(x)
            kh = self._kh()
            noise = self.noise()
            residual = y - self.mean_row().unsqueeze(0)
            evals_x, ux = _SafeEigh.apply(kx)
            evals_h, uh = _SafeEigh.apply(kh)
            # Generous eigenvalue floor: eigh VJP is unstable near multiplicity.
            evals_x = evals_x.clamp_min(1e-8)
            evals_h = evals_h.clamp_min(1e-8)
            # Z = Uxᵀ R Uh  →  Λ_ij = λx_i λh_j + σ²
            z = ux.transpose(-1, -2) @ residual @ uh
            lam = evals_x.unsqueeze(-1) * evals_h.unsqueeze(-2) + noise
            lam = lam.clamp_min(1e-8)
            nb = float(y.shape[0] * y.shape[1])
            logdet = lam.log().sum()
            quad = (z.square() / lam).sum()
            return 0.5 * (nb * math.log(2.0 * math.pi) + logdet + quad)

        @torch.no_grad()
        def predict(
            self, x_train: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Posterior mean [M,B] and diag variance [M,B] (structured)."""
            kx = self._kx(x_train)
            kh = self._kh()
            noise = self.noise()
            mu = self.mean_row()
            residual = y - mu.unsqueeze(0)
            evals_x, ux = torch.linalg.eigh(kx)
            evals_h, uh = torch.linalg.eigh(kh)
            evals_x = evals_x.clamp_min(1e-8)
            evals_h = evals_h.clamp_min(1e-8)
            z = ux.transpose(-1, -2) @ residual @ uh
            lam = evals_x.unsqueeze(-1) * evals_h.unsqueeze(-2) + noise
            lam = lam.clamp_min(1e-8)
            alpha = ux @ (z / lam) @ uh.transpose(-1, -2)  # [N, B]

            k_sx = self._kx(x_test, x_train)  # [M, N]
            # (K_sx ⊗ Kh) α = K_sx α Khᵀ  (row-major / task-inner layout)
            mean = mu.unsqueeze(0) + k_sx @ alpha @ kh

            # Diagonal predictive variance via Kronecker quadratic forms.
            # For test i, task a: k(xi,xi) Kh_aa - vᵀ Σ⁻¹ v, v = k_i ⊗ Kh_{:,a}.
            k_ss_diag = self.data_kernel(x_test, diag=True)  # [M]
            if not torch.is_tensor(k_ss_diag):
                k_ss_diag = k_ss_diag.to_dense()
            kh_diag = torch.diagonal(kh)  # [B]
            # Precompute Uhᵀ Kh = evals_h * Uhᵀ  (since Kh = Uh diag(evals) Uhᵀ)
            # For h = Kh e_a = Kh[:,a], Uhᵀ h = evals_h * Uh[a,:]^T elementwise...
            # Uhᵀ Kh[:,a] = evals_h * Uh[a, :].T  yes.
            ux_t_ks = k_sx @ ux  # [M, N]  (rows: Uxᵀ k_i as columns... wait k_sx[i]@Ux)
            # k_sx[i,:] @ Ux = row; we need Uxᵀ k_i = (k_iᵀ Ux)ᵀ = (k_sx[i] @ Ux)ᵀ
            # so ux_t_k[i] = k_sx[i] @ ux  is (Uxᵀ k_i)ᵀ as row — actually k_sx@ux gives
            # rows (k_iᵀ Ux) = (Uxᵀ k_i)ᵀ, so element j is (Uxᵀ k_i)_j. Good.
            # For each task a: (Uhᵀ h)_j = evals_h[j] * Uh[a, j]
            uh_scaled = uh * evals_h.unsqueeze(0)  # [B, B]: uh_scaled[a,j] = Uh[a,j]*λh_j
            # z_ij^{(i,a)} = (Uxᵀ k_i)_i_idx * (Uhᵀ h_a)_j
            # quad_ia = sum_{pq} z_pq^2 / lam_pq
            # = sum_p (Uxᵀk_i)_p^2 * sum_j (Uhᵀh_a)_j^2 / lam_pj
            # Compute inv_lam contributions:
            # For each train-eigen p and task a: sum_j uh_scaled[a,j]^2 / lam[p,j]
            inv = 1.0 / lam  # [N, B_eig]
            # task_factor[a,p] = sum_j uh_scaled[a,j]^2 * inv[p,j]
            task_factor = uh_scaled.square() @ inv.transpose(0, 1)  # [B, N]
            # quad[i,a] = sum_p ux_t_ks[i,p]^2 * task_factor[a,p]
            quad = (ux_t_ks.square() @ task_factor.transpose(0, 1))  # [M, B]
            prior_diag = k_ss_diag.unsqueeze(-1) * kh_diag.unsqueeze(0)  # [M, B]
            var = (prior_diag - quad).clamp_min(1e-12) + noise
            return mean, var

    def _params_finite(m: nn.Module) -> bool:
        return all(torch.isfinite(p).all().item() for p in m.parameters())

    def _clone_state(m: nn.Module) -> dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    steps_used = 0
    last_exc: Exception | None = None
    for init_i in range(max(1, int(num_inits))):
        torch.manual_seed(int(seed) + init_i)
        model = _KroneckerHistEigGP().to(device=device, dtype=torch.float64)
        with torch.no_grad():
            model.raw_mean.copy_(freq_t)
            # Mild random init on noise; lengthscales O(1) after x/bin scaling.
            model.raw_noise.add_(
                0.05 * torch.randn((), dtype=torch.float64, device=device)
            )
            model.data_kernel.base_kernel.lengthscale = 1.0
            model.bin_kernel.base_kernel.lengthscale = 0.2
            model.data_kernel.outputscale = 1.0
            model.bin_kernel.outputscale = 1.0

        cur_lr = float(lr)
        opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
        last_state = _clone_state(model)
        nan_strikes = 0
        for step_i in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            try:
                loss = model.neg_mll(train_x, train_y)
            except RuntimeError as exc:
                last_exc = exc
                loss = torch.tensor(float("nan"))
            if not torch.isfinite(loss):
                model.load_state_dict(
                    {k: v.to(device=device) for k, v in last_state.items()}
                )
                nan_strikes += 1
                cur_lr = max(cur_lr * 0.5, 1e-5)
                opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
                # Keep iterating toward num_steps; skip this update.
                steps_used = max(steps_used, step_i + 1)
                continue
            loss.backward()
            grads_ok = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            )
            if not grads_ok:
                opt.zero_grad(set_to_none=True)
                nan_strikes += 1
                cur_lr = max(cur_lr * 0.5, 1e-5)
                opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
                steps_used = max(steps_used, step_i + 1)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                # One-hot means live near [0,1]; keep them from exploding.
                model.raw_mean.clamp_(-1.0, 2.0)
            # Critical: do not checkpoint a post-step NaN state whose pre-step
            # loss was still finite (that path previously yielded uniform π).
            if not _params_finite(model):
                model.load_state_dict(
                    {k: v.to(device=device) for k, v in last_state.items()}
                )
                nan_strikes += 1
                cur_lr = max(cur_lr * 0.5, 1e-5)
                opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
                steps_used = max(steps_used, step_i + 1)
                continue
            nan_strikes = 0
            last = float(loss.detach().cpu())
            last_state = _clone_state(model)
            steps_used = max(steps_used, step_i + 1)
            if last < best_loss:
                best_loss = last
                best_state = last_state

    if best_state is None:
        raise RuntimeError(
            "hist_mogp Kronecker eigh training produced no finite MLL "
            f"(n={n}, B={b}; last error: {last_exc!r})."
        )
    model = _KroneckerHistEigGP().to(device=device, dtype=torch.float64)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        mean_t, var_t = model.predict(train_x, train_y, test_x)
        mean_nb = mean_t.detach().cpu().numpy()
        var_nb = var_t.detach().cpu().numpy()

    if not np.all(np.isfinite(mean_nb)):
        # Last-resort: empirical frequencies (avoids silent uniform-from-NaN).
        mean_nb = np.broadcast_to(freqs.reshape(1, -1), mean_nb.shape).copy()
        var_nb = np.full_like(mean_nb, 1e-3)

    if sample_project:
        probs = project_samples_average(
            mean_nb,
            var_nb,
            n_samples=int(n_proj_samples),
            rng=np.random.default_rng(int(seed)),
        )
    else:
        probs = project_onto_simplex(mean_nb)

    ell_h = float(
        model.bin_kernel.base_kernel.lengthscale.detach().cpu().reshape(-1)[0]
    )
    sigma_h = float(model.bin_kernel.outputscale.detach().cpu().reshape(-1)[0]) ** 0.5
    noise_v = float(model.noise().detach().cpu())

    return probs, {
        "best_loss": best_loss,
        "num_steps": int(num_steps),
        "steps_used": int(steps_used),
        "num_inits": int(num_inits),
        "backend": "kronecker_bin_rbf_eigh",
        "n_bins": int(b),
        "ell_H": ell_h,
        "sigma_H": sigma_h,
        "noise": noise_v,
    }


def _fit_hist_indep_batch_gp(
    train_x: torch.Tensor,
    onehot: np.ndarray,
    test_x: torch.Tensor,
    *,
    freqs: np.ndarray,
    seed: int,
    num_inits: int,
    num_steps: int,
    lr: float,
    device: str,
    sample_project: bool,
    n_proj_samples: int,
    input_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """B independent Exact GPs via batch_shape; project posterior mean onto simplex."""
    import gpytorch

    from histogram_gp import project_onto_simplex, project_samples_average

    n, b = onehot.shape
    train_y = torch.as_tensor(onehot.T, dtype=torch.float64, device=device)  # [B, N]
    freq_t = torch.as_tensor(freqs, dtype=torch.float64, device=device)
    batch_shape = torch.Size([b])

    class _BatchBinGP(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=input_dim,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for init_i in range(max(1, int(num_inits))):
        torch.manual_seed(int(seed) + init_i)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape).to(
            device=device, dtype=torch.float64
        )
        model = _BatchBinGP(train_x, train_y, likelihood).to(device=device, dtype=torch.float64)
        with torch.no_grad():
            model.mean_module.constant.copy_(freq_t)

        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        last = float("inf")
        for _ in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            out = model(train_x)
            loss = -mll(out, train_y).sum()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        if last < best_loss:
            best_loss = last
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "lik": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }

    assert best_state is not None
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape).to(
        device=device, dtype=torch.float64
    )
    model = _BatchBinGP(train_x, train_y, likelihood).to(device=device, dtype=torch.float64)
    model.load_state_dict(best_state["model"])
    likelihood.load_state_dict(best_state["lik"])
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean_bn = pred.mean.detach().cpu().numpy()  # [B, N_te]
        var_bn = pred.variance.detach().cpu().numpy()

    mean_nb = mean_bn.T  # [N_te, B]
    if sample_project:
        probs = project_samples_average(
            mean_nb,
            var_bn.T,
            n_samples=int(n_proj_samples),
            rng=np.random.default_rng(int(seed)),
        )
    else:
        probs = project_onto_simplex(mean_nb)

    return probs, {
        "best_loss": best_loss,
        "num_steps": int(num_steps),
        "num_inits": int(num_inits),
        "backend": "indep_batch_gp",
    }


def _fit_hist_mtgpr(
    train_x: torch.Tensor,
    onehot: np.ndarray,
    test_x: torch.Tensor,
    *,
    freqs: np.ndarray,
    seed: int,
    num_inits: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ICM multi-output GP via gpplus MTGPR (only when N*B is small)."""
    import gpytorch

    from gpplus.kernels import GaussianKernel, LogScaleKernel
    from gpplus.models.mtgpr import MTGPR
    from gpplus.training.trainer import GPTrainer
    from histogram_gp import project_onto_simplex

    n, b = onehot.shape
    train_y = torch.as_tensor(onehot, dtype=torch.float64, device=device)  # [N, B]
    train_x = train_x.to(device=device)
    test_x = test_x.to(device=device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=b, rank=0
    ).to(dtype=torch.float64)
    base_kernel = LogScaleKernel(
        GaussianKernel(ard_num_dims=int(train_x.shape[-1]))
    )
    # Mild task correlation via low-rank ICM (rank=min(4, B-1)).
    rank = int(min(4, max(b - 1, 1)))
    model = MTGPR(
        train_x,
        train_y,
        num_tasks=b,
        rank_likelihood=0,
        rank_kernel=rank,
        likelihood=likelihood,
        mean_module=gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), b
        ),
        kernel_module=gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=b, rank=rank
        ),
    )
    # Initialize multitask constant means toward empirical frequencies.
    with torch.no_grad():
        base_means = getattr(model.mean_module, "base_means", None)
        if base_means is not None:
            for i, sub in enumerate(list(base_means)):
                if i < len(freqs) and hasattr(sub, "constant"):
                    sub.constant.fill_(float(freqs[i]))

    trainer = GPTrainer(
        model,
        num_epochs=1,
        num_inits=int(num_inits),
        seed=int(seed),
        device=device,
        dtype=torch.float64,
        n_jobs=1,
    )
    runs = trainer.train()
    valid = [
        r
        for r in runs
        if r.get("loss") is not None and r.get("state_dict") is not None
    ]
    if not valid:
        raise RuntimeError("hist_mogp MTGPR training failed for all inits.")
    best = min(valid, key=lambda r: r["loss"])
    model.load_state_dict(best["state_dict"])
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = model.likelihood(model(test_x))
        mean_nb = dist.mean.detach().cpu().numpy()  # [N_te, B]

    probs = project_onto_simplex(mean_nb)
    return probs, {
        "best_loss": float(best["loss"]),
        "num_inits": int(num_inits),
        "backend": "mtgpr",
        "rank_kernel": rank,
    }


def fit_predict_hist_gpd(
    x_tr: Any,
    y_tr: Any,
    x_te: Any,
    *,
    n_bins: int = 32,
    pad_frac: float = 0.05,
    seed: int = 42,
    num_inits: int = 2,
    num_steps: int = 80,
    lr: float = 0.1,
    alpha_epsilon: float = 0.01,
    device: str = "cpu",
    backend: str = "exact",
    tail_frac: float = 0.0,
    standardize_x: bool = False,
    n_mc: int = 0,
    ni_sorf_num_rff: int = 512,
    ni_sorf_num_epochs: int = 150,
) -> dict[str, Any]:
    """Idea 2: Dirichlet-based GP classification on bin indices (Milios et al.).

    On hyperspectral / large-n, ``backend='ni_sorf'`` falls back to NI-SORF
    one-hot bin GPs (same as hist_mogp) because Exact Dirichlet GPs do not scale.
    ``n_mc > 0`` averages ``softmax`` over latent posterior samples (uses the
    marginal variance) instead of ``softmax`` of the latent mean.
    """
    if backend == "ni_sorf":
        out = fit_predict_hist_mogp(
            x_tr,
            y_tr,
            x_te,
            n_bins=n_bins,
            pad_frac=pad_frac,
            seed=seed,
            backend="ni_sorf",
            ni_sorf_num_rff=ni_sorf_num_rff,
            ni_sorf_num_epochs=ni_sorf_num_epochs,
        )
        extras = dict(out.get("extras") or {})
        extras["model"] = "hist_gpd"
        extras["gp_backend"] = "ni_sorf_via_hist_mogp"
        extras["note"] = (
            "Exact Dirichlet GP skipped on NI-SORF backend; used NI-SORF bin MoGP."
        )
        out["extras"] = extras
        return out

    import gpytorch

    from histogram_gp import make_hist_bins, y_to_bin_index

    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    Xtr_t, _Xtr_np, Xte_t, y_np = _as_tensors_xy(x_tr, y_tr, x_te)
    if standardize_x:
        x_mu = Xtr_t.mean(dim=0, keepdim=True)
        x_sd = Xtr_t.std(dim=0, keepdim=True).clamp_min(1e-8)
        Xtr_t = (Xtr_t - x_mu) / x_sd
        Xte_t = (Xte_t - x_mu) / x_sd
    borders, _bin_info = make_hist_bins(
        y_np,
        n_bins=int(n_bins),
        pad_frac=float(pad_frac),
        scheme="equal_width",
        tail_frac=float(tail_frac),
    )
    labels_np = y_to_bin_index(y_np, borders)
    b = int(borders.size - 1)
    d = int(Xtr_t.shape[-1])
    batch_shape = torch.Size([b])

    Xtr_t = Xtr_t.to(device=device)
    Xte_t = Xte_t.to(device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    class _DirichletBinGP(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=d,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class _FixedClassDirichletLik(gpytorch.likelihoods.DirichletClassificationLikelihood):
        # gpytorch infers num_classes from labels.max()+1; empty top bins must still count.
        def _prepare_targets(self, targets, alpha_epsilon=0.01, dtype=torch.float):
            alpha = alpha_epsilon * torch.ones(
                targets.shape[-1], b, device=targets.device, dtype=dtype
            )
            idx = torch.arange(len(targets), device=targets.device)
            alpha[idx, targets] = alpha[idx, targets] + 1.0
            sigma2_i = torch.log(alpha.reciprocal() + 1.0)
            transformed = alpha.log() - 0.5 * sigma2_i
            return sigma2_i.transpose(-2, -1).type(dtype), transformed.type(dtype), b

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_targets: torch.Tensor | None = None

    for init_i in range(max(1, int(num_inits))):
        torch.manual_seed(int(seed) + init_i)
        likelihood = _FixedClassDirichletLik(
            labels,
            alpha_epsilon=float(alpha_epsilon),
            learn_additional_noise=True,
            dtype=torch.float64,
        ).to(device=device)
        targets = likelihood.transformed_targets.to(device=device, dtype=torch.float64)
        model = _DirichletBinGP(Xtr_t, targets, likelihood).to(
            device=device, dtype=torch.float64
        )

        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        last = float("inf")
        for _ in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t)
            loss = -mll(out, targets).sum()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        if last < best_loss:
            best_loss = last
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "lik": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }
            best_targets = targets.detach().cpu().clone()

    assert best_state is not None and best_targets is not None
    likelihood = _FixedClassDirichletLik(
        labels.cpu(),
        alpha_epsilon=float(alpha_epsilon),
        learn_additional_noise=True,
        dtype=torch.float64,
    ).to(device=device)
    targets = best_targets.to(device=device)
    model = _DirichletBinGP(Xtr_t, targets, likelihood).to(
        device=device, dtype=torch.float64
    )
    model.load_state_dict(best_state["model"])
    likelihood.load_state_dict(best_state["lik"])
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = model(Xte_t)
        # Latents are in log-concentration space; softmax of means → simplex probs.
        mean_bn = dist.mean.detach().cpu().numpy()  # [B, N_te]
        var_bn = dist.variance.detach().cpu().numpy() if int(n_mc) > 0 else None
    logits_nb = mean_bn.T

    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-300, None)

    if var_bn is not None:
        sd_nb = np.sqrt(np.clip(var_bn.T, 1e-12, None))
        rng = np.random.default_rng(int(seed) + 7)
        acc = np.zeros_like(logits_nb)
        for _ in range(int(n_mc)):
            acc += _softmax(logits_nb + sd_nb * rng.standard_normal(logits_nb.shape))
        probs = acc / float(n_mc)
    else:
        probs = _softmax(logits_nb)

    wall = time.perf_counter() - t0
    return _pack_bar_predictive(
        borders,
        probs,
        wall_s=wall,
        extras={
            "model": "hist_gpd",
            "best_loss": best_loss,
            "num_steps": int(num_steps),
            "num_inits": int(num_inits),
            "alpha_epsilon": float(alpha_epsilon),
            "backend": "dirichlet_gp",
            "n_mc": int(n_mc),
            "standardize_x": bool(standardize_x),
            "tail_frac": float(tail_frac),
        },
    )


# ---------------------------------------------------------------------------
# Midline residual MoGP / Bayes MoGP siblings (process-level joints)
# ---------------------------------------------------------------------------
def fit_predict_midline_residual_mogp(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    k_max: int = 5,
    seed: int = 42,
    gate_type: str = "mlp",
    num_inits: int = 2,
    num_em_iters: int = 3,
    num_gate_steps: int = 400,
    device: str | None = None,
    midline_backend: str = "exact",
    ni_sorf_num_rff: int = 2400,
    ni_sorf_num_epochs: int = 1000,
    ni_sorf_residual_num_rff: int = 1024,
    ni_sorf_residual_num_epochs: int = 300,
    ni_sorf_lr: float = 1e-2,
) -> dict[str, Any]:
    """Phase-1 residual MoGP (estimative expert-GP joints; frozen Stage-1 midline)."""
    from gpplus.models import MidlineResidualMoGP

    t0 = time.perf_counter()
    # Exact path: CPU ExactGP. NI-SORF path: CUDA Stage-1/experts, CPU caches.
    if midline_backend == "ni_sorf":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    x_tr = x_tr.to(dtype=torch.float64, device="cpu")
    y_tr = y_tr.to(dtype=torch.float64, device="cpu").reshape(-1)
    x_te = x_te.to(dtype=torch.float64, device="cpu")
    if x_tr.ndim == 1:
        x_tr = x_tr.unsqueeze(-1)
    if x_te.ndim == 1:
        x_te = x_te.unsqueeze(-1)

    model = MidlineResidualMoGP(
        x_tr,
        y_tr,
        k_max=int(k_max),
        gate_type=gate_type,  # type: ignore[arg-type]
        dtype=torch.float64,
        midline_backend=midline_backend,  # type: ignore[arg-type]
        ni_sorf_num_rff=int(ni_sorf_num_rff),
        ni_sorf_num_epochs=int(ni_sorf_num_epochs),
        ni_sorf_residual_num_rff=int(ni_sorf_residual_num_rff),
        ni_sorf_residual_num_epochs=int(ni_sorf_residual_num_epochs),
        ni_sorf_lr=float(ni_sorf_lr),
    )
    hist = model.fit(
        num_inits=int(num_inits),
        num_em_iters=int(num_em_iters),
        num_gate_steps=int(num_gate_steps),
        seed=int(seed),
        device=device,
        n_jobs=1,
    )
    pred = model.predict(x_te)
    wall = time.perf_counter() - t0
    out = {
        "mixture_mean": pred["mean"].detach().cpu().numpy(),
        "expert_means": pred["expert_means"].detach().cpu().numpy(),
        "expert_stds": pred["expert_stds"].detach().cpu().numpy(),
        "gate_weights": pred["gate_weights"].detach().cpu().numpy(),
        "extras": {
            "wall_s": wall,
            "model": "midline_residual_mogp",
            "k_max": int(k_max),
            "gate_type": gate_type,
            "num_em_iters": int(num_em_iters),
            "num_gate_steps": int(num_gate_steps),
            "device": device,
            "fit_history_len": len((hist or {}).get("history") or []),
            "ppd_source": "residual_mogp_margins",
        },
    }
    return out


def fit_predict_midline_bayes_mogp(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    k_max: int = 5,
    seed: int = 42,
    gate_type: str = "mlp",
    num_inits: int = 2,
    num_em_iters: int = 3,
    num_gate_steps: int = 400,
    laplace_steps: int = 80,
    n_posterior_draws: int = 32,
    device: str | None = None,
    midline_backend: str = "exact",
    ni_sorf_num_rff: int = 2400,
    ni_sorf_num_epochs: int = 1000,
    ni_sorf_residual_num_rff: int = 1024,
    ni_sorf_residual_num_epochs: int = 300,
    ni_sorf_lr: float = 1e-2,
) -> dict[str, Any]:
    """Phase-2 Bayes MoGP (Laplace over θ + MC margins; MAP components for dens.)."""
    from gpplus.models import MidlineBayesMoGP

    t0 = time.perf_counter()
    if midline_backend == "ni_sorf":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    x_tr = x_tr.to(dtype=torch.float64, device="cpu")
    y_tr = y_tr.to(dtype=torch.float64, device="cpu").reshape(-1)
    x_te = x_te.to(dtype=torch.float64, device="cpu")
    if x_tr.ndim == 1:
        x_tr = x_tr.unsqueeze(-1)
    if x_te.ndim == 1:
        x_te = x_te.unsqueeze(-1)

    model = MidlineBayesMoGP(
        x_tr,
        y_tr,
        k_max=int(k_max),
        gate_type=gate_type,  # type: ignore[arg-type]
        dtype=torch.float64,
        midline_backend=midline_backend,  # type: ignore[arg-type]
        ni_sorf_num_rff=int(ni_sorf_num_rff),
        ni_sorf_num_epochs=int(ni_sorf_num_epochs),
        ni_sorf_residual_num_rff=int(ni_sorf_residual_num_rff),
        ni_sorf_residual_num_epochs=int(ni_sorf_residual_num_epochs),
        ni_sorf_lr=float(ni_sorf_lr),
    )
    hist = model.fit(
        num_inits=int(num_inits),
        num_em_iters=int(num_em_iters),
        num_gate_steps=int(num_gate_steps),
        laplace_steps=int(laplace_steps),
        seed=int(seed),
        device=device,
        n_jobs=1,
    )
    pred = model.predict(
        x_te, n_posterior_draws=int(n_posterior_draws), seed=int(seed)
    )
    wall = time.perf_counter() - t0
    out = {
        "mixture_mean": pred["mean"].detach().cpu().numpy(),
        "expert_means": pred["expert_means"].detach().cpu().numpy(),
        "expert_stds": pred["expert_stds"].detach().cpu().numpy(),
        "gate_weights": pred["gate_weights"].detach().cpu().numpy(),
        "extras": {
            "wall_s": wall,
            "model": "midline_bayes_mogp",
            "k_max": int(k_max),
            "gate_type": gate_type,
            "num_em_iters": int(num_em_iters),
            "num_gate_steps": int(num_gate_steps),
            "laplace_steps": int(laplace_steps),
            "n_posterior_draws": int(n_posterior_draws),
            "device": device,
            "phase1_history_len": len(
                ((hist or {}).get("phase1_history") or {}).get("history") or []
            ),
            "ppd_source": "bayes_mogp_mc_margins",
        },
    }
    return out


def fit_predict_ni_sorf_adapter(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    **kwargs: Any,
) -> dict[str, Any]:
    """Standalone NI-SORF baseline (COSI / hyperspectral Exact-GP replacement)."""
    from ni_sorf_midline import fit_predict_ni_sorf

    return fit_predict_ni_sorf(x_tr, y_tr, x_te, **kwargs)


ADAPTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "exact_gp": fit_predict_exact_gp,
    "ni_sorf": fit_predict_ni_sorf_adapter,
    "midline": fit_predict_midline_sparse,
    "midline_gamma_mlp": lambda *a, **k: fit_predict_midline_sparse(
        *a, gamma_type="mlp", **k
    ),
    "midline_residual_mogp": fit_predict_midline_residual_mogp,
    "midline_bayes_mogp": fit_predict_midline_bayes_mogp,
    "mdn": fit_predict_mdn,
    "mdn_const": fit_predict_mdn_const,
    "tabgp": fit_predict_tabgp,
    "tabpfn3": lambda *a, **k: fit_predict_tabpfn(*a, version="v3", **k),
    "tabpfn35": lambda *a, **k: fit_predict_tabpfn(*a, version="v3.5", **k),
    "kumo": fit_predict_kumotabular,
    "tabfm": fit_predict_kumotabular,  # alias: CLI/docs historically said tabfm
    "hist_mogp": fit_predict_hist_mogp,
    "hist_gpd": fit_predict_hist_gpd,
}
