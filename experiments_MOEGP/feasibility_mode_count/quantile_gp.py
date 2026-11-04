"""Quantile GP: Kumo-style inverse-CDF regression with a joint (x, τ) GP prior.

TabPFN / ``hist_mogp`` predict the CDF at fixed y-locations (a histogram);
this model predicts the inverse CDF at fixed probabilities ``τ_1 < … < τ_J``
and trains on the pinball loss (CRPS = 2∫ pinball dτ)::

    q(x, τ_1)     = f_0(x)
    q(x, τ_{j+1}) = q(x, τ_j) + softplus(f_j(x))      (no crossing)
    f_j(x)        = h(x)ᵀ W_{:, j} + c_j,  h = [RFF_ℓ(x), x]
    vec(W_inc)    ~ N(0, I ⊗ K_τ)                     (Kronecker x/τ prior)

The likelihood is the asymmetric Laplace (Bayesian quantile regression), so
``pinball/σ + log σ`` with a learned ``σ`` sets the data-vs-prior balance.
The RFF lengthscales get gradients only from the pinball loss on a fixed
held-out mask (weights detached), so the fit is single pass with no knobs.

Beyond ``τ_1`` / ``τ_J`` the density continues with generalized-Pareto tails
whose boundary density matches the outer quantile bars. The predictive is the
piecewise-linear CDF through the knots, exported as bars on one shared border
set (the scorer shares borders across rows), with each row's knots included.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def _pinball(y: torch.Tensor, q: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``[N, J]`` pinball loss ``ρ_τ(y − q)``."""
    u = y.unsqueeze(-1) - q
    return torch.maximum(tau * u, (tau - 1.0) * u)


def _inv_softplus(v: torch.Tensor) -> torch.Tensor:
    return v + torch.log(-torch.expm1(-v))


class QuantileGP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        tau: np.ndarray,
        *,
        mode: str = "joint",
        n_rff: int = 256,
        features: str = "rff",
        tau_lengthscale: float = 0.2,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if mode not in ("joint", "indep"):
            raise ValueError(f"mode must be 'joint' or 'indep', got {mode!r}")
        if features not in ("rff", "sorf"):
            raise ValueError(f"features must be 'rff' or 'sorf', got {features!r}")
        self.mode = mode
        self.feature_kind = features
        self.d = int(input_dim)
        gen = torch.Generator().manual_seed(int(seed))
        self.register_buffer("tau", torch.as_tensor(tau, dtype=dtype))
        if features == "sorf":
            from gpplus.utils.sorf_features import sample_padded_sorf

            # ``n_rff`` frequencies, cos/sin pairs: feature dim ``2 n_rff``.
            omega = sample_padded_sorf(self.d, n_rff, generator=gen, dtype=dtype)
            n_feat = 2 * n_rff
        else:
            omega = torch.randn(self.d, n_rff, generator=gen, dtype=dtype)
            n_feat = n_rff
        self.register_buffer("omega", omega)
        self.register_buffer(
            "phase", 2 * math.pi * torch.rand(n_rff, generator=gen, dtype=dtype)
        )
        j = self.tau.numel()
        p = n_feat + self.d
        self.log_ls = torch.nn.Parameter(torch.zeros(self.d, dtype=dtype))
        self.w = torch.nn.Parameter(torch.zeros(p, j, dtype=dtype))
        self.c = torch.nn.Parameter(torch.zeros(j, dtype=dtype))
        self.log_sig = torch.nn.Parameter(torch.tensor(math.log(0.1), dtype=dtype))
        if mode == "joint" and j > 1:
            t = self.tau[1:]
            k = torch.exp(-0.5 * ((t[:, None] - t[None, :]) / tau_lengthscale) ** 2)
            k = k + 1e-4 * torch.eye(j - 1, dtype=dtype)
            self.register_buffer("k_tau_inv", torch.linalg.inv(k))
        else:
            self.register_buffer("k_tau_inv", torch.eye(max(j - 1, 1), dtype=dtype))

    def features(self, x: torch.Tensor, *, detach_ls: bool = False) -> torch.Tensor:
        ls = torch.exp(self.log_ls.detach() if detach_ls else self.log_ls)
        if self.feature_kind == "sorf":
            z = (x / ls) @ self.omega
            s = 1.0 / math.sqrt(self.omega.shape[1])
            return torch.cat([s * torch.cos(z), s * torch.sin(z), x], dim=-1)
        z = (x / ls) @ self.omega + self.phase
        rff = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(z)
        return torch.cat([rff, x], dim=-1)

    def quantiles(
        self,
        x: torch.Tensor,
        *,
        detach_ls: bool = False,
        detach_w: bool = False,
    ) -> torch.Tensor:
        w = self.w.detach() if detach_w else self.w
        c = self.c.detach() if detach_w else self.c
        f = self.features(x, detach_ls=detach_ls) @ w + c
        if self.mode == "indep":
            return f
        inc = F.softplus(f[:, 1:]) + 1e-6
        return torch.cat([f[:, :1], f[:, :1] + torch.cumsum(inc, dim=-1)], dim=-1)

    def prior(self) -> torch.Tensor:
        w0, wi = self.w[:, :1], self.w[:, 1:]
        quad = (wi @ self.k_tau_inv * wi).sum() if wi.numel() else 0.0
        return 0.5 * (w0.pow(2).sum() + quad)

    def init_from(self, y: torch.Tensor) -> None:
        with torch.no_grad():
            qe = torch.quantile(y, self.tau)
            if self.mode == "indep":
                self.c.copy_(qe)
            else:
                gaps = torch.diff(qe).clamp_min(1e-3 * float(y.std()) + 1e-6)
                self.c[0] = qe[0]
                self.c[1:] = _inv_softplus(gaps)
            self.log_sig.fill_(math.log(0.1 * float(y.std()) + 1e-6))


def knot_nll(q: torch.Tensor, tau: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``[N]`` NLL of ``y`` under the knot density (piecewise-uniform + exp tails).

    Differentiable in the knot positions ``q [N, J]``: inside the knots the
    density is ``Δτ_j / (q_{j+1} − q_j)``; beyond them it continues with
    exponential tails whose scale comes from the outer ~20% of mass (``κ = 1``;
    the GPD shape and ``κ`` are still selected after training).
    """
    tn = tau.detach().cpu().numpy()
    k_lo = min(max(int(np.searchsorted(tn, tn[0] + 0.2)), 1), tn.size - 1)
    k_hi = max(min(int(np.searchsorted(tn, tn[-1] - 0.2, side="right")) - 1, tn.size - 2), 0)
    lam_lo = (tau[0] * (q[:, k_lo] - q[:, 0]) / (tau[k_lo] - tau[0])).clamp_min(1e-6)
    lam_hi = ((1 - tau[-1]) * (q[:, -1] - q[:, k_hi]) / (tau[-1] - tau[k_hi])).clamp_min(1e-6)
    j = torch.searchsorted(q.detach().contiguous(), y.unsqueeze(-1)).squeeze(-1) - 1
    j = j.clamp(0, q.shape[1] - 2)
    gap = (q.gather(1, (j + 1).unsqueeze(-1)) - q.gather(1, j.unsqueeze(-1))).squeeze(-1)
    dtau = torch.diff(tau)[j]
    lp_mid = torch.log(dtau) - torch.log(gap.clamp_min(1e-9))
    z_lo = (q[:, 0] - y).clamp_min(0.0)
    z_hi = (y - q[:, -1]).clamp_min(0.0)
    lp_lo = torch.log(tau[0]) - torch.log(lam_lo) - z_lo / lam_lo
    lp_hi = torch.log(1 - tau[-1]) - torch.log(lam_hi) - z_hi / lam_hi
    lp = torch.where(y < q[:, 0], lp_lo, torch.where(y > q[:, -1], lp_hi, lp_mid))
    return -lp


def fit_quantile_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: np.ndarray,
    *,
    mode: str = "joint",
    n_rff: int = 256,
    features: str = "rff",
    lam_nll: float | str = 0.0,
    num_steps: int = 1500,
    lr: float = 0.02,
    calib_frac: float = 0.15,
    ls_inits: tuple[float, ...] = (1.0, 0.3),
    lam_grid: tuple[float, ...] = (0.0, 0.1, 1.0),
    seed: int = 0,
) -> tuple[QuantileGP, dict[str, Any]]:
    """``lam_nll > 0`` adds ``λ · knot_nll`` per fit row (hybrid pinball + density loss).

    ``lam_nll="auto"`` fits every ``λ`` in ``lam_grid`` (as restarts) and keeps
    the one with the best held-out score ``pinball·J + knot_nll`` on the
    calibration mask; the training-row NLL alone overfits knot gaps at small n.
    """
    n, d = x.shape
    gen = torch.Generator().manual_seed(int(seed) + 4242)
    n_cal = max(5, int(round(calib_frac * n))) if calib_frac > 0 else 0
    perm = torch.randperm(n, generator=gen)
    cal = torch.zeros(n, dtype=torch.bool)
    cal[perm[:n_cal]] = True
    fit = ~cal
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    auto = isinstance(lam_nll, str)
    if auto and lam_nll != "auto":
        raise ValueError(f"lam_nll must be a float or 'auto', got {lam_nll!r}")
    lams = tuple(lam_grid) if auto and mode == "joint" else (float(0.0 if auto else lam_nll),)
    best: tuple[float, QuantileGP | None, float, float] = (
        float("inf"), None, float("nan"), float("nan")
    )
    configs = [(ls0, lam) for lam in lams for ls0 in ls_inits]
    for r, (ls0, lam) in enumerate(configs):
        m = QuantileGP(
            d, tau, mode=mode, n_rff=n_rff, features=features,
            seed=seed + 101 * (r % len(ls_inits)),
        )
        m.init_from(y[fit])
        with torch.no_grad():
            m.log_ls.fill_(math.log(ls0 * math.sqrt(d)))
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        n_fit = float(fit.sum())
        for _ in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            sig = torch.exp(m.log_sig)
            qf = m.quantiles(x[fit], detach_ls=n_cal > 0)
            pin = _pinball(y[fit], qf, tau_t).sum()
            loss = (pin / sig + n_fit * tau_t.numel() * m.log_sig + m.prior()) / n_fit
            if lam > 0 and mode == "joint":
                loss = loss + float(lam) * knot_nll(qf, tau_t, y[fit]).mean()
            if n_cal:
                qc = m.quantiles(x[cal], detach_w=True)
                loss = loss + _pinball(y[cal], qc, tau_t).mean() * tau_t.numel()
            loss.backward()
            opt.step()
            with torch.no_grad():
                m.log_ls.clamp_(math.log(1e-2), math.log(1e2))
        with torch.no_grad():
            xs, ys = (x[cal], y[cal]) if n_cal else (x, y)
            qs = m.quantiles(xs)
            score = float(_pinball(ys, qs, tau_t).mean())
            if auto and len(lams) > 1:
                score = score * tau_t.numel() + float(knot_nll(qs, tau_t, ys).mean())
        if score < best[0]:
            best = (score, m, ls0, lam)
    model = best[1]
    assert model is not None
    info = {
        "cal_score": best[0],
        "ls_init": best[2],
        "lam_nll_selected": float(best[3]),
        "lengthscales": torch.exp(model.log_ls).detach().numpy().tolist(),
        "ald_sigma": float(torch.exp(model.log_sig.detach())),
        "n_cal": n_cal,
        "features": features,
        "lam_nll": str(lam_nll),
        "_cal_mask": cal,
    }
    return model, info


TAIL_KAPPAS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
TAIL_XIS = (0.0, 0.2, 0.5, 0.8)


def _tail_scales(
    q: np.ndarray, tau: np.ndarray, kappa: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential tail scales from the mean density over the outer ~20% mass.

    The single outermost spacing can collapse at some x and leave a
    near-zero-width tail; the wider span is robust and ``kappa`` (chosen on
    the held-out mask) sets the tail weight.
    """
    k_lo = int(np.searchsorted(tau, tau[0] + 0.2))
    k_lo = min(max(k_lo, 1), tau.size - 1)
    k_hi = int(np.searchsorted(tau, tau[-1] - 0.2, side="right")) - 1
    k_hi = max(min(k_hi, tau.size - 2), 0)
    lam_lo = tau[0] * (q[:, k_lo] - q[:, 0]) / (tau[k_lo] - tau[0])
    lam_hi = (1.0 - tau[-1]) * (q[:, -1] - q[:, k_hi]) / (tau[-1] - tau[k_hi])
    return (
        np.clip(kappa * lam_lo, 1e-9, None),
        np.clip(kappa * lam_hi, 1e-9, None),
    )


def _gpd_log_sf(z: np.ndarray, lam: np.ndarray, xi: float) -> np.ndarray:
    """Log survival of a GPD tail at excess ``z >= 0`` (``xi = 0``: exponential)."""
    if xi <= 0.0:
        return -z / lam
    return -np.log1p(xi * z / lam) / xi


def _gpd_log_pdf(z: np.ndarray, lam: np.ndarray, xi: float) -> np.ndarray:
    if xi <= 0.0:
        return -np.log(lam) - z / lam
    return -np.log(lam) - (1.0 / xi + 1.0) * np.log1p(xi * z / lam)


def quantile_log_pdf(
    q: np.ndarray, tau: np.ndarray, y: np.ndarray, kappa: float = 1.0, xi: float = 0.0
) -> np.ndarray:
    """Log density at ``y[i]`` of row ``q[i]`` (piecewise-linear CDF + GPD tails)."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    lam_lo, lam_hi = _tail_scales(q, tau, kappa)
    dtau = np.diff(tau)
    out = np.empty(y.size, dtype=np.float64)
    lo = y < q[:, 0]
    hi = y > q[:, -1]
    mid = ~(lo | hi)
    out[lo] = math.log(tau[0]) + _gpd_log_pdf(q[lo, 0] - y[lo], lam_lo[lo], xi)
    out[hi] = math.log(1.0 - tau[-1]) + _gpd_log_pdf(y[hi] - q[hi, -1], lam_hi[hi], xi)
    for i in np.flatnonzero(mid):
        qi = q[i]
        j = min(max(int(np.searchsorted(qi, y[i], side="right")) - 1, 0), qi.size - 2)
        out[i] = math.log(dtau[j] / max(qi[j + 1] - qi[j], 1e-12))
    return out


def quantiles_to_bars(
    q: np.ndarray,
    tau: np.ndarray,
    *,
    y_lo: float,
    y_hi: float,
    kappa: float = 1.0,
    xi: float = 0.0,
    max_borders: int = 4000,
    n_uniform: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared borders ``[B+1]`` and per-row bar probs ``[N, B]`` for knot rows ``q``."""
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    lam_lo, lam_hi = _tail_scales(q, tau, kappa)
    # Excess (in tail scales) where the tail survival falls to 1e-9.
    reach = 9.0 * math.log(10.0) if xi <= 0.0 else (1e9**xi - 1.0) / xi
    span = y_hi - y_lo
    lo = max(min(float(np.min(q[:, 0] - reach * lam_lo)), y_lo), y_lo - 50.0 * span)
    hi = min(max(float(np.max(q[:, -1] + reach * lam_hi)), y_hi), y_hi + 50.0 * span)
    mult = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    tail_pts = np.concatenate(
        [
            (q[:, :1] - lam_lo[:, None] * mult).ravel(),
            (q[:, -1:] + lam_hi[:, None] * mult).ravel(),
        ]
    )
    pts = np.unique(np.concatenate([q.ravel(), tail_pts, np.linspace(lo, hi, n_uniform)]))
    pts = pts[(pts >= lo) & (pts <= hi)]
    if pts.size > max_borders:
        pts = np.unique(np.quantile(pts, np.linspace(0.0, 1.0, max_borders)))
    borders = np.unique(np.concatenate([[lo], pts, [hi]]))
    cdf = np.empty((q.shape[0], borders.size), dtype=np.float64)
    for i in range(q.shape[0]):
        qi = q[i]
        c = np.interp(borders, qi, tau)
        below = borders < qi[0]
        above = borders > qi[-1]
        c[below] = tau[0] * np.exp(_gpd_log_sf(qi[0] - borders[below], lam_lo[i], xi))
        c[above] = 1.0 - (1.0 - tau[-1]) * np.exp(
            _gpd_log_sf(borders[above] - qi[-1], lam_hi[i], xi)
        )
        cdf[i] = c
    probs = np.clip(np.diff(cdf, axis=-1), 0.0, None)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-300, None)
    return borders, probs


def _gpd_tail_excess(u: np.ndarray, lam: np.ndarray, xi: float) -> np.ndarray:
    """Excess ``z`` with tail survival ratio ``u = S(z)`` in (0, 1]."""
    if xi <= 0.0:
        return -lam * np.log(u)
    return lam * (u ** (-xi) - 1.0) / xi


def quantile_crps(
    q: np.ndarray,
    tau: np.ndarray,
    y: np.ndarray,
    *,
    kappa: float = 1.0,
    xi: float = 0.0,
    n_tail: int = 400,
    n_inner: int = 8,
) -> np.ndarray:
    """``[N]`` CRPS ``= 2 ∫₀¹ ρ_τ(y − Q(τ)) dτ`` of the knot + GPD-tail quantile function.

    The inner part is integrated on the knots refined ``n_inner``×; the tails
    on a log-spaced ``τ`` grid down to ``1e-9`` (finite for ``ξ < 1``).
    """
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    lam_lo, lam_hi = _tail_scales(q, tau, kappa)
    t_in = np.unique(
        np.concatenate([np.linspace(tau[j], tau[j + 1], n_inner + 1) for j in range(tau.size - 1)])
    )
    q_in = np.stack([np.interp(t_in, tau, qi) for qi in q])
    u = np.logspace(-9, 0, n_tail)
    t_lo = tau[0] * u
    q_lo = q[:, :1] - _gpd_tail_excess(u[None, :], lam_lo[:, None], xi)
    t_hi = 1.0 - (1.0 - tau[-1]) * u
    q_hi = q[:, -1:] + _gpd_tail_excess(u[None, :], lam_hi[:, None], xi)
    t_all = np.concatenate([t_lo, t_in[1:-1], t_hi[::-1]])
    q_all = np.concatenate([q_lo, q_in[:, 1:-1], q_hi[:, ::-1]], axis=1)
    order = np.argsort(t_all)
    t_all, q_all = t_all[order], q_all[:, order]
    t_all = np.concatenate([[0.0], t_all, [1.0]])
    q_all = np.concatenate([q_all[:, :1], q_all, q_all[:, -1:]], axis=1)
    rho = np.maximum(t_all * (y - q_all), (t_all - 1.0) * (y - q_all))
    return 2.0 * np.trapezoid(rho, t_all, axis=1)


def fit_predict_quantile_gp(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    n_levels: int = 19,
    mode: str = "joint",
    n_rff: int | None = None,
    features: str = "rff",
    lam_nll: float | str = 0.0,
    num_steps: int = 1500,
    calib_frac: float = 0.15,
    seed: int = 0,
    **_: Any,
) -> dict[str, Any]:
    """``features="sorf"``: padded-SORF cos/sin features with a self-configured
    frequency count; ``lam_nll``: weight of the knot-density NLL (hybrid loss)."""
    from foundation_adapters import _pack_bar_predictive

    if n_rff is None:
        if features == "sorf":
            from gpplus.utils.sorf_features import auto_num_freq

            n_rff = auto_num_freq(x_tr.shape[0], x_tr.shape[1], lo=128, hi=1024, woodbury=False)
        else:
            n_rff = 256

    t0 = time.perf_counter()
    x_tr = x_tr.to(dtype=torch.float64).cpu()
    x_te = x_te.to(dtype=torch.float64).cpu()
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1).cpu()
    x_mu, x_sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-8)
    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std().clamp_min(1e-8))
    zx_tr, zx_te = (x_tr - x_mu) / x_sd, (x_te - x_mu) / x_sd
    zy = (y_tr - y_mu) / y_sd
    step = 1.0 / (int(n_levels) + 1)
    tau = np.linspace(step, 1.0 - step, int(n_levels))
    torch.manual_seed(int(seed))
    model, info = fit_quantile_gp(
        zx_tr, zy, tau, mode=mode, n_rff=int(n_rff), features=features, lam_nll=lam_nll,
        num_steps=num_steps, calib_frac=calib_frac, seed=seed,
    )
    cal = info.pop("_cal_mask")

    def _q(xs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            qq = model.quantiles(xs).numpy()
        if mode == "indep":
            qq = np.sort(qq, axis=-1) + 1e-4 * np.arange(qq.shape[1])[None, :]
        return qq

    # Tail weight and shape are two scalars pinball never sees; score them on
    # all training rows (the held-out mask alone has ~1 exceedance per tail).
    del cal
    q_all, y_all = _q(zx_tr), zy.numpy()
    grid = [(k, xi) for xi in TAIL_XIS for k in TAIL_KAPPAS]
    scores = [float(quantile_log_pdf(q_all, tau, y_all, k, xi).mean()) for k, xi in grid]
    kappa, xi = grid[int(np.argmax(scores))]
    q = _q(zx_te) * y_sd + y_mu
    borders, probs = quantiles_to_bars(
        q, tau, y_lo=float(y_tr.min()), y_hi=float(y_tr.max()), kappa=kappa, xi=xi
    )
    extras = {
        "ppd_family": "quantile_gp",
        "quantile_mode": mode,
        "n_levels": int(n_levels),
        "tail_kappa": float(kappa),
        "tail_xi": float(xi),
        "n_rff": int(n_rff),
        **info,
    }
    out = _pack_bar_predictive(
        borders, probs, wall_s=time.perf_counter() - t0, extras=extras
    )
    out["quantile_levels"] = tau
    out["quantile_values"] = q
    return out
