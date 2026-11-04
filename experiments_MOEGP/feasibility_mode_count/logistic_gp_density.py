"""Logistic-GP conditional density on a y grid (lean prototype).

``p(y | x) ∝ exp f(x, y)`` over grid cells, with a Gaussian base around a
learned location/scale and a GP shape term on the residual ``r = y − m(x)``::

    f(x, y) = −½ (r / s(x))² − log s(x) + β(y) + Σ_d w_d cos(ω_xdᵀ x / ℓ_x + ω_yd r / ℓ_y + b_d)
    m(x), log s(x) = RFF-x regressions,     w, v_m, v_s ~ N(0, I)

``β`` is a marginal log-density correction on the cells (random-walk prior);
it carries x-independent structure such as the two levels of a step.

Without the base, a narrow ridge ``y ≈ m(x)`` forces a tiny ``ℓ_y`` and the
density turns wiggly. The shape term still factorizes
(``ω_y r = ω_y y − ω_y m(x)`` folds into the x phase), so ``[N, G, D]`` is
never formed. Weights are MAP-fit on the fit rows; the lengthscales get
gradients only from the held-out-mask NLL (weights detached), as in the auto
gamma model. Two wide outer cells carry the tails, and a learned fraction
``ε`` of the smoothed training marginal keeps the density off zero.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch


class LogisticGPDensity(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        centers: torch.Tensor,
        log_widths: torch.Tensor,
        log_marginal: torch.Tensor,
        *,
        n_rff: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        dt = torch.float64
        gen = torch.Generator().manual_seed(int(seed))
        self.register_buffer("centers", centers.to(dt))
        self.register_buffer("log_widths", log_widths.to(dt))
        self.register_buffer("omega_x", torch.randn(input_dim, n_rff, generator=gen, dtype=dt))
        self.register_buffer("omega_y", torch.randn(n_rff, generator=gen, dtype=dt))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_rff, generator=gen, dtype=dt))
        self.register_buffer("omega_m", torch.randn(input_dim, n_rff, generator=gen, dtype=dt))
        self.register_buffer("phase_m", 2 * math.pi * torch.rand(n_rff, generator=gen, dtype=dt))
        ls0 = math.log(math.sqrt(input_dim))
        self.log_ls_x = torch.nn.Parameter(torch.full((input_dim,), ls0, dtype=dt))
        self.log_ls_m = torch.nn.Parameter(torch.full((input_dim,), ls0, dtype=dt))
        self.log_ls_y = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=dt))
        self.w = torch.nn.Parameter(torch.zeros(n_rff, dtype=dt))
        p = n_rff + input_dim
        self.v_m = torch.nn.Parameter(torch.zeros(p, dtype=dt))
        self.v_s = torch.nn.Parameter(torch.zeros(p, dtype=dt))
        self.c_m = torch.nn.Parameter(torch.zeros((), dtype=dt))
        self.c_s = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=dt))
        self.beta = torch.nn.Parameter(torch.zeros(centers.numel(), dtype=dt))
        self.rw_scale = 0.5
        self.raw_eps = torch.nn.Parameter(torch.tensor(-4.0, dtype=dt))
        self.register_buffer("log_marg", log_marginal.to(dt))
        self.scale = math.sqrt(2.0 / n_rff)

    def _hyper(self, detach: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ls = (self.log_ls_x, self.log_ls_m, self.log_ls_y)
        return tuple(torch.exp(t.detach() if detach else t) for t in ls)  # type: ignore[return-value]

    def log_probs(
        self, x: torch.Tensor, *, detach_ls: bool = False, detach_w: bool = False
    ) -> torch.Tensor:
        """``[N, G]`` log cell probabilities."""
        lx, lm, ly = self._hyper(detach_ls)
        dw = (lambda t: t.detach()) if detach_w else (lambda t: t)
        h = torch.cat(
            [self.scale * torch.cos((x / lm) @ self.omega_m + self.phase_m), x], dim=-1
        )
        m = h @ dw(self.v_m) + dw(self.c_m)
        log_s = (h @ dw(self.v_s) + dw(self.c_s)).clamp(math.log(1e-3), math.log(1e2))
        r = self.centers[None, :] - m[:, None]
        base = -0.5 * (r / torch.exp(log_s)[:, None]).pow(2) - log_s[:, None]
        a = (x / lx) @ self.omega_x + self.phase - (m[:, None] * self.omega_y[None, :]) / ly
        c = self.centers[:, None] * self.omega_y[None, :] / ly
        w = dw(self.w)
        f = self.scale * ((torch.cos(a) * w) @ torch.cos(c).T - (torch.sin(a) * w) @ torch.sin(c).T)
        lp = torch.log_softmax(base + dw(self.beta) + f + self.log_widths, dim=-1)
        # Defensive floor: the Gaussian base has light tails, so mix in the
        # training marginal; ``eps`` is trained on the held-out mask only.
        log_eps = torch.nn.functional.logsigmoid(self.raw_eps)
        log_1m = torch.nn.functional.logsigmoid(-self.raw_eps)
        if not detach_w:
            log_eps, log_1m = log_eps.detach(), log_1m.detach()
        return torch.logaddexp(log_1m + lp, log_eps + self.log_marg[None, :])

    def prior(self) -> torch.Tensor:
        d2 = self.beta[2:] - 2 * self.beta[1:-1] + self.beta[:-2]
        return 0.5 * (
            self.w.pow(2).sum()
            + self.v_m.pow(2).sum()
            + self.v_s.pow(2).sum()
            + (d2 / self.rw_scale).pow(2).sum()
        )


def _grid(y: np.ndarray, n_grid: int) -> np.ndarray:
    lo, hi = float(y.min()), float(y.max())
    sd = float(y.std()) + 1e-8
    inner = np.linspace(lo - 1.0 * sd, hi + 1.0 * sd, n_grid + 1)
    return np.concatenate([[lo - 20.0 * sd], inner, [hi + 20.0 * sd]])


def fit_predict_logistic_gp(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    n_grid: int = 128,
    n_rff: int = 256,
    num_steps: int = 1000,
    lr: float = 0.02,
    calib_frac: float = 0.15,
    seed: int = 0,
    **_: Any,
) -> dict[str, Any]:
    from foundation_adapters import _pack_bar_predictive

    t0 = time.perf_counter()
    x_tr = x_tr.to(dtype=torch.float64).cpu()
    x_te = x_te.to(dtype=torch.float64).cpu()
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1).cpu()
    x_mu, x_sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-8)
    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std().clamp_min(1e-8))
    zx_tr, zx_te = (x_tr - x_mu) / x_sd, (x_te - x_mu) / x_sd
    zy = ((y_tr - y_mu) / y_sd).numpy()

    borders = _grid(zy, int(n_grid))
    widths = np.diff(borders)
    centers = 0.5 * (borders[:-1] + borders[1:])
    centers[0], centers[-1] = borders[1] - 1.0, borders[-2] + 1.0
    idx = np.clip(np.searchsorted(borders, zy, side="right") - 1, 0, widths.size - 1)
    counts = np.bincount(idx, minlength=widths.size).astype(np.float64)
    marg = np.convolve(counts, np.ones(9) / 9.0, mode="same") + 0.5 * widths / widths.sum()
    log_marg = torch.as_tensor(np.log(marg / marg.sum()))

    n = zx_tr.shape[0]
    gen = torch.Generator().manual_seed(int(seed) + 4242)
    n_cal = max(5, int(round(calib_frac * n))) if calib_frac > 0 else 0
    cal = torch.zeros(n, dtype=torch.bool)
    cal[torch.randperm(n, generator=gen)[:n_cal]] = True
    fit = ~cal
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    lw = torch.as_tensor(np.log(widths))

    model = LogisticGPDensity(
        zx_tr.shape[1], torch.as_tensor(centers), lw, log_marg, n_rff=n_rff, seed=seed
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_fit = float(fit.sum())

    def _nll(lp: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return -(lp.gather(1, idx_t[rows, None]).squeeze(1) - lw[idx_t[rows]])

    for _ in range(int(num_steps)):
        opt.zero_grad(set_to_none=True)
        lp = model.log_probs(zx_tr[fit], detach_ls=n_cal > 0)
        loss = (_nll(lp, fit).sum() + model.prior()) / n_fit
        if n_cal:
            loss = loss + _nll(model.log_probs(zx_tr[cal], detach_w=True), cal).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for t in (model.log_ls_x, model.log_ls_m, model.log_ls_y):
                t.clamp_(math.log(1e-2), math.log(1e2))
    with torch.no_grad():
        probs = torch.exp(model.log_probs(zx_te)).numpy()
        cal_nll = float(_nll(model.log_probs(zx_tr[cal]), cal).mean()) if n_cal else float("nan")
    extras = {
        "ppd_family": "logistic_gp",
        "lengthscale_x": torch.exp(model.log_ls_x).detach().numpy().tolist(),
        "lengthscale_y": float(torch.exp(model.log_ls_y.detach())),
        "cal_nll_std_units": cal_nll,
        "floor_eps": float(torch.sigmoid(model.raw_eps.detach())),
        "n_cal": n_cal,
    }
    return _pack_bar_predictive(
        borders * y_sd + y_mu, probs, wall_s=time.perf_counter() - t0, extras=extras
    )


# --------------------------------------------------------------------------
# v2: padded-SORF features, variational weights, Stage-1 LOO midline
# --------------------------------------------------------------------------


class LogisticGPSorf(torch.nn.Module):
    """Logistic GP over ``(x, r = y − m(x))`` with padded-SORF features.

    ``m(x), s₀(x)`` are given (Stage-1 LOO midline: LOO values on training
    rows, predictive on test rows), so the Gaussian base is centred on an
    honest midline. With SORF frequencies ``ω = [ω_x; ω_y]`` and
    ``a = ω_xᵀx/ℓ_x − ω_y m/ℓ_y``, ``c_g = ω_y y_g/ℓ_y``::

        f(x, y_g) = σ_f/√D Σ_d w_c,d cos(a_d + c_gd) + w_s,d sin(a_d + c_gd)

    expands into two ``[N, D] @ [D, G]`` products. ``q(w) = N(μ, diag σ²)``
    is fit by a reparameterized ELBO (``inference="vi"``) or ``w`` by MAP.
    """

    def __init__(
        self,
        input_dim: int,
        centers: torch.Tensor,
        log_widths: torch.Tensor,
        log_marginal: torch.Tensor,
        *,
        num_freq: int,
        inference: str = "vi",
        features: str = "sorf",
        seed: int = 0,
    ) -> None:
        super().__init__()
        from gpplus.utils.sorf_features import sample_padded_sorf

        dt = torch.float64
        gen = torch.Generator().manual_seed(int(seed))

        def _draw(d: int) -> torch.Tensor:
            if features == "rff":
                return torch.randn(d, int(num_freq), generator=gen, dtype=dt)
            return sample_padded_sorf(d, int(num_freq), generator=gen, dtype=dt)

        w = _draw(input_dim + 1)
        self.register_buffer("omega_x", w[:input_dim].contiguous())
        self.register_buffer("omega_y", w[input_dim].contiguous())
        self.register_buffer("omega_m", _draw(input_dim))
        self.register_buffer("centers", centers.to(dt))
        self.register_buffer("log_widths", log_widths.to(dt))
        self.register_buffer("log_marg", log_marginal.to(dt))
        self.D = int(num_freq)
        self.inference = str(inference)
        ls0 = math.log(math.sqrt(input_dim))
        self.log_ls_x = torch.nn.Parameter(torch.full((input_dim,), ls0, dtype=dt))
        self.log_ls_m = torch.nn.Parameter(torch.full((input_dim,), ls0, dtype=dt))
        self.log_ls_y = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=dt))
        self.log_amp = torch.nn.Parameter(torch.zeros((), dtype=dt))
        self.w_mu = torch.nn.Parameter(torch.zeros(2 * self.D, dtype=dt))
        self.w_rho = torch.nn.Parameter(torch.full((2 * self.D,), -3.0, dtype=dt))
        p_ms = 2 * self.D + input_dim
        self.v_s = torch.nn.Parameter(torch.zeros(p_ms, dtype=dt))
        self.c_s = torch.nn.Parameter(torch.zeros((), dtype=dt))
        self.v_m = torch.nn.Parameter(torch.zeros(p_ms, dtype=dt))
        self.c_m = torch.nn.Parameter(torch.zeros((), dtype=dt))
        self.beta = torch.nn.Parameter(torch.zeros(centers.numel(), dtype=dt))
        self.rw_scale = 0.5
        self.raw_eps = torch.nn.Parameter(torch.tensor(-4.0, dtype=dt))

    def sample_w(self, n_samples: int, *, detach: bool = False) -> torch.Tensor:
        mu, rho = (self.w_mu.detach(), self.w_rho.detach()) if detach else (self.w_mu, self.w_rho)
        if self.inference != "vi":
            return mu.unsqueeze(0)
        eps = torch.randn(int(n_samples), mu.numel(), dtype=mu.dtype)
        return mu + torch.exp(rho) * eps

    def log_probs(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        log_s0: torch.Tensor,
        w: torch.Tensor,
        *,
        detach_hyper: bool = False,
        detach_rest: bool = False,
    ) -> torch.Tensor:
        """``[S, N, G]`` log cell probabilities for weight samples ``w [S, 2D]``."""
        hd = (lambda t: t.detach()) if detach_hyper else (lambda t: t)
        rd = (lambda t: t.detach()) if detach_rest else (lambda t: t)
        lx, ly = torch.exp(hd(self.log_ls_x)), torch.exp(hd(self.log_ls_y))
        amp = torch.exp(hd(self.log_amp)) / math.sqrt(self.D)
        p = (x / lx) @ self.omega_x
        pm = (x / torch.exp(hd(self.log_ls_m))) @ self.omega_m
        hx = torch.cat([torch.cos(pm) / math.sqrt(self.D), torch.sin(pm) / math.sqrt(self.D), x], -1)
        log_s = (log_s0 + rd(self.c_s) + hx @ rd(self.v_s)).clamp(math.log(1e-3), math.log(1e2))
        # The LOO midline centres the base; a MAP correction lets it move onto
        # a branch when the conditional is multimodal.
        m = m + rd(self.c_m) + hx @ rd(self.v_m)
        r = self.centers[None, :] - m[:, None]
        base = -0.5 * (r / torch.exp(log_s)[:, None]).pow(2) - log_s[:, None]
        a = p - m[:, None] * self.omega_y[None, :] / ly
        c = self.centers[:, None] * self.omega_y[None, :] / ly
        ca, sa = torch.cos(a)[None], torch.sin(a)[None]
        wc, ws = w[:, None, : self.D], w[:, None, self.D :]
        f = amp * (
            (ca * wc + sa * ws) @ torch.cos(c).T + (ca * ws - sa * wc) @ torch.sin(c).T
        )
        lp = torch.log_softmax(base[None] + rd(self.beta) + f + self.log_widths, dim=-1)
        log_eps = torch.nn.functional.logsigmoid(self.raw_eps)
        log_1m = torch.nn.functional.logsigmoid(-self.raw_eps)
        if not detach_rest:
            log_eps, log_1m = log_eps.detach(), log_1m.detach()
        return torch.logaddexp(log_1m + lp, log_eps + self.log_marg)

    def prior(self) -> torch.Tensor:
        d2 = self.beta[2:] - 2 * self.beta[1:-1] + self.beta[:-2]
        if self.inference == "vi":
            var = torch.exp(2 * self.w_rho)
            kl_w = 0.5 * (var + self.w_mu.pow(2) - 1.0 - 2 * self.w_rho).sum()
        else:
            kl_w = 0.5 * self.w_mu.pow(2).sum()
        return (
            kl_w
            + 0.5 * self.v_s.pow(2).sum()
            + 0.5 * self.v_m.pow(2).sum()
            + 0.5 * (d2 / self.rw_scale).pow(2).sum()
        )


def _loo_midline(zx_tr, zy_tr, zx_te, *, seed: int):
    from gpplus.models.loo_gp_core import make_loo_core

    core = make_loo_core(zx_tr.shape[1], zx_tr.shape[0], seed=seed).fit(
        zx_tr, zy_tr, num_inits=2, seed=seed
    )
    m_tr, s_tr = core.loo_train()
    m_te, s_te = core.predict(zx_te)
    return m_tr, s_tr, m_te, s_te, {"midline_loo_nll": float(core.best_loss)}


def _gamma_midline(zx_tr, zy_tr, zx_te, *, seed: int):
    """Gamma-auto mixture mean/std as the base (training rows are in-sample)."""
    from foundation_adapters import fit_predict_midline_gamma_auto

    n_tr = zx_tr.shape[0]
    out = fit_predict_midline_gamma_auto(
        zx_tr, zy_tr, torch.cat([zx_tr, zx_te], 0), seed=seed, y_min=-10.0, y_max=10.0
    )
    w = np.asarray(out["gate_weights"])
    mu = np.asarray(out["expert_means"])
    sd = np.asarray(out["expert_stds"])
    mean = (w * mu).sum(-1)
    std = np.sqrt(np.clip((w * (sd**2 + mu**2)).sum(-1) - mean**2, 1e-8, None))
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    return t(mean[:n_tr]), t(std[:n_tr]), t(mean[n_tr:]), t(std[n_tr:]), {
        "midline_gamma_n_mid": int((out.get("extras") or {}).get("n_midlines", 1))
    }


def fit_predict_logistic_gp_sorf(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    n_grid: int = 128,
    num_freq: int | None = None,
    num_steps: int = 1000,
    lr: float = 0.02,
    calib_frac: float = 0.15,
    inference: str = "vi",
    midline: str = "loo",
    n_mc: int = 5,
    n_pred_mc: int = 64,
    batch_max: int = 2000,
    base_scale: str = "loo",
    features: str = "sorf",
    learn_amp: bool = False,
    vi_predict: str = "mc",
    seed: int = 0,
    **_: Any,
) -> dict[str, Any]:
    """Logistic-GP bar density with SORF features, VI weights and a LOO midline.

    ``base_scale="loo"`` starts the base width at the LOO std; ``"narrow"``
    starts it at 0.5 (standardized units) and lets ``c_s, v_s`` learn it.
    """
    from foundation_adapters import _pack_bar_predictive
    from gpplus.utils.sorf_features import auto_num_freq

    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    x_tr = x_tr.to(dtype=torch.float64).cpu()
    x_te = x_te.to(dtype=torch.float64).cpu()
    y_tr = y_tr.to(dtype=torch.float64).reshape(-1).cpu()
    x_mu, x_sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-8)
    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std().clamp_min(1e-8))
    zx_tr, zx_te = (x_tr - x_mu) / x_sd, (x_te - x_mu) / x_sd
    zy_t = (y_tr - y_mu) / y_sd
    zy = zy_t.numpy()

    if midline == "none":
        z = lambda k: torch.zeros(k, dtype=torch.float64)  # noqa: E731
        o = lambda k: torch.full((k,), 0.5, dtype=torch.float64)  # noqa: E731
        m_tr, s_tr, m_te, s_te = z(len(zy)), o(len(zy)), z(len(zx_te)), o(len(zx_te))
        mid_info: dict[str, Any] = {}
    else:
        mid_fn = {"loo": _loo_midline, "gamma": _gamma_midline}[midline]
        m_tr, s_tr, m_te, s_te, mid_info = mid_fn(zx_tr, zy_t, zx_te, seed=seed)
    m_tr, m_te = m_tr.reshape(-1).detach(), m_te.reshape(-1).detach()
    ls_tr = s_tr.reshape(-1).clamp_min(1e-3).log().detach()
    ls_te = s_te.reshape(-1).clamp_min(1e-3).log().detach()
    if base_scale == "narrow":
        ls_tr = torch.full_like(ls_tr, math.log(0.5))
        ls_te = torch.full_like(ls_te, math.log(0.5))

    borders = _grid(zy, int(n_grid))
    widths = np.diff(borders)
    centers = 0.5 * (borders[:-1] + borders[1:])
    centers[0], centers[-1] = borders[1] - 1.0, borders[-2] + 1.0
    idx = np.clip(np.searchsorted(borders, zy, side="right") - 1, 0, widths.size - 1)
    counts = np.bincount(idx, minlength=widths.size).astype(np.float64)
    marg = np.convolve(counts, np.ones(9) / 9.0, mode="same") + 0.5 * widths / widths.sum()
    log_marg = torch.as_tensor(np.log(marg / marg.sum()))

    n, d = zx_tr.shape
    D = int(num_freq) if num_freq else auto_num_freq(n, d + 1, woodbury=False)
    gen = torch.Generator().manual_seed(int(seed) + 4242)
    n_cal = max(5, int(round(calib_frac * n))) if calib_frac > 0 else 0
    cal = torch.zeros(n, dtype=torch.bool)
    cal[torch.randperm(n, generator=gen)[:n_cal]] = True
    fit_idx = torch.nonzero(~cal).reshape(-1)
    cal_idx = torch.nonzero(cal).reshape(-1)
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    lw = torch.as_tensor(np.log(widths))

    model = LogisticGPSorf(
        d,
        torch.as_tensor(centers),
        lw,
        log_marg,
        num_freq=D,
        inference=inference,
        features=features,
        seed=seed,
    )
    model.log_amp.requires_grad_(bool(learn_amp))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_fit = float(fit_idx.numel())
    S = int(n_mc) if inference == "vi" else 1

    def _cell_lp(lp: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``[S, B]`` log density of each row's observed cell."""
        g = idx_t[rows]
        return lp.gather(2, g.expand(lp.shape[0], -1).unsqueeze(-1)).squeeze(-1) - lw[g]

    for _ in range(int(num_steps)):
        opt.zero_grad(set_to_none=True)
        rows = fit_idx
        if rows.numel() > int(batch_max):
            rows = rows[torch.randperm(rows.numel())[: int(batch_max)]]
        lp = model.log_probs(
            zx_tr[rows], m_tr[rows], ls_tr[rows], model.sample_w(S), detach_hyper=n_cal > 0
        )
        nll_fit = -_cell_lp(lp, rows).mean(0).sum() * (n_fit / rows.numel())
        loss = (nll_fit + model.prior()) / n_fit
        if n_cal:
            crow = cal_idx
            lp_c = model.log_probs(
                zx_tr[crow], m_tr[crow], ls_tr[crow], model.sample_w(S, detach=True),
                detach_rest=True,
            )
            lp_c = torch.logsumexp(_cell_lp(lp_c, crow), 0) - math.log(lp_c.shape[0])
            loss = loss - lp_c.mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.log_ls_x.clamp_(math.log(1e-2), math.log(1e2))
            model.log_ls_m.clamp_(math.log(1e-2), math.log(1e2))
            model.log_ls_y.clamp_(math.log(1e-2), math.log(1e2))
            model.log_amp.clamp_(math.log(1e-2), math.log(1e2))

    S_pred = int(n_pred_mc) if inference == "vi" else 1
    with torch.no_grad():
        torch.manual_seed(int(seed) + 77)
        w_pred = model.sample_w(S_pred)
        if vi_predict == "mean":
            w_pred = model.w_mu.detach().unsqueeze(0)
        probs = []
        for i0 in range(0, zx_te.shape[0], 512):
            sl = slice(i0, i0 + 512)
            lp = model.log_probs(zx_te[sl], m_te[sl], ls_te[sl], w_pred)
            probs.append(torch.exp(lp).mean(0))
        probs = torch.cat(probs, 0).numpy()
        cal_nll = float("nan")
        if n_cal:
            lp_c = model.log_probs(zx_tr[cal_idx], m_tr[cal_idx], ls_tr[cal_idx], w_pred)
            cal_nll = float(
                -(torch.logsumexp(_cell_lp(lp_c, cal_idx), 0) - math.log(lp_c.shape[0])).mean()
            )
    extras = {
        "ppd_family": "logistic_gp_sorf",
        "inference": inference,
        "midline": midline,
        "num_freq": D,
        "lengthscale_x": torch.exp(model.log_ls_x).detach().numpy().tolist(),
        "lengthscale_y": float(torch.exp(model.log_ls_y.detach())),
        "amplitude": float(torch.exp(model.log_amp.detach())),
        "w_post_sd_mean": float(torch.exp(model.w_rho.detach()).mean()),
        "cal_nll_std_units": cal_nll,
        "floor_eps": float(torch.sigmoid(model.raw_eps.detach())),
        "n_cal": n_cal,
        **mid_info,
    }
    return _pack_bar_predictive(
        borders * y_sd + y_mu, probs, wall_s=time.perf_counter() - t0, extras=extras
    )
