"""Exact GP midline trained on the closed-form leave-one-out predictive likelihood.

Kernel: ARD RBF, optionally plus a linear kernel and a learned-feature linear
kernel ``σ_φ² φ(x)ᵀφ(x')`` (``φ`` a small MLP). The extra terms are the GP form
of a Bayesian mean ``h(x)ᵀβ`` with a Gaussian prior on ``β``, so the LOO
objective holds the learnable mean out together with the RBF part.

The feature kernel is off by default (``n_feat=0``): ``φ`` is fit on every
point, so its LOO residuals are not held out and on small tabular sets it
memorizes ``y`` (noise → 0, test NLL in the hundreds).

LOO identities (Sundararajan & Keerthi 2001), with ``A = (K + Σ_n)^{-1}``::

    μ_i^{LOO} = y_i − [A r]_i / A_ii,   v_i^{LOO} = 1 / A_ii

where ``r = y − c`` and ``Σ_n = diag(σ_n² / w_i)``; ``w`` are optional per-point
weights (EM responsibilities), ``w ≡ 1`` otherwise.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LooGPCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        mean_terms: bool = True,
        n_feat: int = 0,
        hidden: int = 32,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.mean_terms = bool(mean_terms)
        self.use_feat = self.mean_terms and int(n_feat) > 0
        self.dtype = dtype
        self.log_ls = nn.Parameter(torch.zeros(self.input_dim, dtype=dtype))
        self.log_os = nn.Parameter(torch.zeros((), dtype=dtype))
        self.log_noise = nn.Parameter(torch.tensor(math.log(0.1), dtype=dtype))
        self.const = nn.Parameter(torch.zeros((), dtype=dtype))
        if self.mean_terms:
            self.log_lin = nn.Parameter(torch.tensor(math.log(1e-2), dtype=dtype))
        if self.use_feat:
            self.log_feat = nn.Parameter(torch.tensor(math.log(1e-2), dtype=dtype))
            self.feat = nn.Sequential(
                nn.Linear(self.input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_feat),
                nn.Tanh(),
            ).to(dtype=dtype)
        self._jitter = 1e-6
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None
        self.train_w: torch.Tensor | None = None
        self._chol: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self.best_loss: float = float("nan")

    # ----------------------------------------------------------- kernel
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise).clamp_min(1e-6)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ls = torch.exp(self.log_ls)
        d2 = torch.cdist(a / ls, b / ls).pow(2)
        k = torch.exp(self.log_os) * torch.exp(-0.5 * d2)
        if self.mean_terms:
            k = k + torch.exp(self.log_lin) * (a @ b.T)
        if self.use_feat:
            k = k + torch.exp(self.log_feat) * (self.feat(a) @ self.feat(b).T)
        return k

    def kernel_diag(self, a: torch.Tensor) -> torch.Tensor:
        k = torch.exp(self.log_os).expand(a.shape[0])
        if self.mean_terms:
            k = k + torch.exp(self.log_lin) * a.pow(2).sum(-1)
        if self.use_feat:
            k = k + torch.exp(self.log_feat) * self.feat(a).pow(2).sum(-1)
        return k

    def _noise_vec(self, n: int, w: torch.Tensor | None) -> torch.Tensor:
        base = self.noise().expand(n)
        if w is None:
            return base
        return base / w.clamp_min(1e-3)

    # -------------------------------------------------------------- LOO
    def loo(
        self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LOO predictive mean and std (unit-weight noise) at each training point."""
        n = x.shape[0]
        nv = self._noise_vec(n, w)
        k = self.kernel(x, x) + torch.diag(nv + self._jitter)
        chol = torch.linalg.cholesky(k)
        a_inv = torch.cholesky_inverse(chol)
        r = y - self.const
        alpha = a_inv @ r
        d = torch.diagonal(a_inv).clamp_min(1e-12)
        mu = y - alpha / d
        # 1/A_ii is the predictive variance under noise σ²/w_i; report it at
        # the unit-weight noise σ² so weighted and unweighted fits compare.
        var = (1.0 / d - nv + self.noise()).clamp_min(1e-10)
        return mu, var.sqrt()

    def loo_nll(
        self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        mu, sd = self.loo(x, y, w)
        nll = 0.5 * math.log(2 * math.pi) + sd.log() + 0.5 * ((y - mu) / sd).pow(2)
        if w is None:
            return nll.mean()
        return (w * nll).sum() / w.sum().clamp_min(1e-8)

    # -------------------------------------------------------------- fit
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        w: torch.Tensor | None = None,
        num_inits: int = 3,
        steps: int = 300,
        lr: float = 0.05,
        seed: int = 0,
        warm_start: bool = False,
    ) -> "LooGPCore":
        x = x.to(dtype=self.dtype)
        y = y.reshape(-1).to(dtype=self.dtype)
        w = None if w is None else w.reshape(-1).to(dtype=self.dtype)
        ls_inits = [0.0, math.log(0.5), math.log(2.0), math.log(0.25), math.log(4.0)]
        best_loss = float("inf")
        best_state = None
        inits = 1 if warm_start else max(1, int(num_inits))
        for i in range(inits):
            torch.manual_seed(int(seed) + 31 * i)
            if not warm_start:
                self._reset(ls_inits[i % len(ls_inits)], y)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            last = float("inf")
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                try:
                    loss = self.loo_nll(x, y, w)
                except RuntimeError:
                    break
                if self.use_feat:
                    # Weak decay keeps the learned features from memorizing y.
                    loss = loss + 1e-3 * sum(p.pow(2).sum() for p in self.feat.parameters())
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    self.log_ls.clamp_(math.log(1e-2), math.log(1e2))
                    self.log_noise.clamp_(math.log(1e-6), math.log(10.0))
                    self.log_os.clamp_(math.log(1e-4), math.log(1e2))
                last = float(loss.detach())
            with torch.no_grad():
                try:
                    final = float(self.loo_nll(x, y, w))
                except RuntimeError:
                    final = float("inf")
            if final < best_loss:
                best_loss = final
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            del last
        if best_state is None:
            raise RuntimeError("LooGPCore: every init failed.")
        self.load_state_dict(best_state)
        self.best_loss = best_loss
        self._cache(x, y, w)
        return self

    def _reset(self, log_ls0: float, y: torch.Tensor) -> None:
        with torch.no_grad():
            self.log_ls.fill_(float(log_ls0))
            var_y = float(y.var().clamp_min(1e-6))
            self.log_os.fill_(math.log(var_y))
            self.log_noise.fill_(math.log(0.1 * var_y))
            self.const.fill_(float(y.mean()))
            if self.mean_terms:
                self.log_lin.fill_(math.log(1e-2 * var_y))
            if self.use_feat:
                self.log_feat.fill_(math.log(1e-2 * var_y))
                for m in self.feat.modules():
                    if isinstance(m, nn.Linear):
                        m.reset_parameters()

    @torch.no_grad()
    def _cache(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None) -> None:
        n = x.shape[0]
        k = self.kernel(x, x) + torch.diag(self._noise_vec(n, w) + self._jitter)
        self._chol = torch.linalg.cholesky(k)
        self._alpha = torch.cholesky_solve((y - self.const).unsqueeze(-1), self._chol)
        self.train_x, self.train_y, self.train_w = x, y, w

    # ---------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and std (including observation noise)."""
        assert self.train_x is not None and self._chol is not None
        xs = xs.to(device=self.train_x.device, dtype=self.dtype)
        ks = self.kernel(xs, self.train_x)
        mu = self.const + (ks @ self._alpha).reshape(-1)
        v = torch.linalg.solve_triangular(self._chol, ks.T, upper=False)
        var = (self.kernel_diag(xs) - v.pow(2).sum(0)).clamp_min(1e-10) + self.noise()
        return mu, var.sqrt()

    @torch.no_grad()
    def loo_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.train_x is not None and self.train_y is not None
        return self.loo(self.train_x, self.train_y, self.train_w)


class SorfLooCore(nn.Module):
    """Padded-SORF Bayesian linear midline with exact closed-form LOO, ``O(n m²)``.

    Features ``Φ = [σ_f φ_SORF(x), σ_lin x]`` with ``w ~ N(0, I)`` and per-point
    noise ``σ_i² = σ²/w_i (+ NIGP slope term)``. With ``A = ΦᵀΣ⁻¹Φ + I``,
    ``ŵ = A⁻¹ΦᵀΣ⁻¹r``, ``e = r − Φŵ`` and ``h_i = φ_iᵀA⁻¹φ_i / σ_i²``::

        μ_i^{LOO} = y_i − e_i / (1 − h_i),   v_i^{LOO} = σ_i² / (1 − h_i)

    which equals the exact-GP LOO for the kernel ``ΦΦᵀ``. NIGP (McHutchon &
    Rasmussen 2011) adds ``Σ_d σ_x,d² (∂μ/∂x_d)²`` to the noise with slopes held
    fixed between refreshes.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_freq: int = 256,
        mean_terms: bool = True,
        nigp: bool = False,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        from ..utils.sorf_features import SorfFeatures

        self.input_dim = int(input_dim)
        self.mean_terms = bool(mean_terms)
        self.nigp = bool(nigp)
        self.dtype = dtype
        self.feat = SorfFeatures(self.input_dim, int(num_freq), seed=seed, dtype=dtype)
        self.log_os = nn.Parameter(torch.zeros((), dtype=dtype))
        self.log_noise = nn.Parameter(torch.tensor(math.log(0.1), dtype=dtype))
        self.const = nn.Parameter(torch.zeros((), dtype=dtype))
        if self.mean_terms:
            self.log_lin = nn.Parameter(torch.tensor(math.log(1e-2), dtype=dtype))
        if self.nigp:
            self.log_sx = nn.Parameter(torch.full((self.input_dim,), math.log(1e-2), dtype=dtype))
        self._slope2: torch.Tensor | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None
        self.train_w: torch.Tensor | None = None
        self._chol: torch.Tensor | None = None
        self._w_hat: torch.Tensor | None = None
        self.best_loss: float = float("nan")

    @property
    def log_ls(self) -> torch.Tensor:
        return self.feat.log_ls

    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise).clamp_min(1e-6)

    def design(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(0.5 * self.log_os) * self.feat(x)
        if self.mean_terms:
            phi = torch.cat([phi, torch.exp(0.5 * self.log_lin) * x], dim=-1)
        return phi

    def _nigp_var(self, slope2: torch.Tensor | None) -> torch.Tensor | float:
        if not self.nigp or slope2 is None:
            return 0.0
        return (slope2 * torch.exp(2 * self.log_sx)).sum(-1)

    def _noise_vec(self, n: int, w: torch.Tensor | None, slope2=None) -> torch.Tensor:
        base = self.noise().expand(n)
        if w is not None:
            base = base / w.clamp_min(1e-3)
        return base + self._nigp_var(slope2)

    def _solve(self, x, y, w, slope2=None):
        phi = self.design(x)
        nv = self._noise_vec(x.shape[0], w, slope2)
        r = y - self.const
        phis = phi / nv.unsqueeze(-1)
        a = phi.T @ phis + torch.eye(phi.shape[1], dtype=phi.dtype)
        chol = torch.linalg.cholesky(a)
        w_hat = torch.cholesky_solve((phis.T @ r).unsqueeze(-1), chol).reshape(-1)
        v = torch.linalg.solve_triangular(chol, phi.T, upper=False)
        h = (v.pow(2).sum(0) / nv).clamp(max=1.0 - 1e-9)
        e = r - phi @ w_hat
        return phi, nv, chol, w_hat, h, e

    def loo(self, x, y, w=None):
        slope2 = self._slope2 if (self._slope2 is not None and self._slope2.shape[0] == x.shape[0]) else None
        _phi, nv, _c, _w, h, e = self._solve(x, y, w, slope2)
        mu = y - e / (1.0 - h)
        var = (nv / (1.0 - h) - nv + self.noise() + self._nigp_var(slope2)).clamp_min(1e-10)
        return mu, var.sqrt()

    def loo_nll(self, x, y, w=None):
        mu, sd = self.loo(x, y, w)
        nll = 0.5 * math.log(2 * math.pi) + sd.log() + 0.5 * ((y - mu) / sd).pow(2)
        if w is None:
            return nll.mean()
        return (w * nll).sum() / w.sum().clamp_min(1e-8)

    def _mean_slope2(self, x: torch.Tensor, w_hat: torch.Tensor) -> torch.Tensor:
        xg = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            mu = self.design(xg) @ w_hat.detach()
            (g,) = torch.autograd.grad(mu.sum(), xg)
        return g.detach().pow(2)

    def _reset(self, log_ls0: float, y: torch.Tensor) -> None:
        with torch.no_grad():
            self.feat.log_ls.fill_(float(log_ls0))
            var_y = float(y.var().clamp_min(1e-6))
            self.log_os.fill_(math.log(var_y))
            self.log_noise.fill_(math.log(0.1 * var_y))
            self.const.fill_(float(y.mean()))
            if self.mean_terms:
                self.log_lin.fill_(math.log(1e-2 * var_y))
            if self.nigp:
                self.log_sx.fill_(math.log(1e-2))
        self._slope2 = None

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        w: torch.Tensor | None = None,
        num_inits: int = 3,
        steps: int = 300,
        lr: float = 0.05,
        seed: int = 0,
        warm_start: bool = False,
        slope_every: int = 50,
    ) -> "SorfLooCore":
        x = x.to(dtype=self.dtype)
        y = y.reshape(-1).to(dtype=self.dtype)
        w = None if w is None else w.reshape(-1).to(dtype=self.dtype)
        ls_inits = [0.0, math.log(0.5), math.log(2.0), math.log(0.25), math.log(4.0)]
        best_loss, best_state, best_slope = float("inf"), None, None
        inits = 1 if warm_start else max(1, int(num_inits))
        for i in range(inits):
            torch.manual_seed(int(seed) + 31 * i)
            if not warm_start:
                self._reset(ls_inits[i % len(ls_inits)], y)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            for step in range(int(steps)):
                if self.nigp and step > 0 and step % int(slope_every) == 0:
                    with torch.no_grad():
                        *_, w_hat, _h, _e = self._solve(x, y, w, self._slope2)
                    self._slope2 = self._mean_slope2(x, w_hat)
                opt.zero_grad(set_to_none=True)
                try:
                    loss = self.loo_nll(x, y, w)
                except RuntimeError:
                    break
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    self.feat.log_ls.clamp_(math.log(1e-2), math.log(1e2))
                    self.log_noise.clamp_(math.log(1e-6), math.log(10.0))
                    self.log_os.clamp_(math.log(1e-4), math.log(1e2))
                    if self.nigp:
                        self.log_sx.clamp_(math.log(1e-4), math.log(1.0))
            with torch.no_grad():
                try:
                    final = float(self.loo_nll(x, y, w))
                except RuntimeError:
                    final = float("inf")
            if final < best_loss:
                best_loss = final
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                best_slope = self._slope2
        if best_state is None:
            raise RuntimeError("SorfLooCore: every init failed.")
        self.load_state_dict(best_state)
        self._slope2 = best_slope
        self.best_loss = best_loss
        self._cache(x, y, w)
        return self

    @torch.no_grad()
    def _cache(self, x, y, w) -> None:
        _phi, _nv, chol, w_hat, _h, _e = self._solve(x, y, w, self._slope2)
        self._chol, self._w_hat = chol, w_hat
        self.train_x, self.train_y, self.train_w = x, y, w

    @torch.no_grad()
    def predict(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._chol is not None and self._w_hat is not None
        xs = xs.to(dtype=self.dtype)
        phi = self.design(xs)
        mu = self.const + phi @ self._w_hat
        v = torch.linalg.solve_triangular(self._chol, phi.T, upper=False)
        var = v.pow(2).sum(0) + self.noise()
        if self.nigp:
            var = var + self._nigp_var(self._mean_slope2(xs, self._w_hat))
        return mu, var.clamp_min(1e-10).sqrt()

    @torch.no_grad()
    def loo_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.train_x is not None and self.train_y is not None
        return self.loo(self.train_x, self.train_y, self.train_w)


SORF_AUTO_N = 1500


def make_loo_core(
    input_dim: int,
    n: int,
    *,
    backend: str = "auto",
    mean_terms: bool = True,
    num_freq: int | None = None,
    nigp: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> nn.Module:
    """``backend``: ``"exact"`` | ``"sorf"`` | ``"auto"`` (SORF when ``n > SORF_AUTO_N``)."""
    if backend == "auto":
        backend = "sorf" if int(n) > SORF_AUTO_N else "exact"
    if backend == "exact":
        return LooGPCore(input_dim, mean_terms=mean_terms, dtype=dtype)
    if backend != "sorf":
        raise ValueError(f"unknown LOO backend {backend!r}")
    from ..utils.sorf_features import auto_num_freq

    D = int(num_freq) if num_freq else auto_num_freq(n, input_dim)
    return SorfLooCore(
        input_dim, num_freq=D, mean_terms=mean_terms, nigp=nigp, seed=seed, dtype=dtype
    )
