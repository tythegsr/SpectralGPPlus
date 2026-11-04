"""Non-separable histogram GP: sum of Kronecker kernels over (x, bin).

``hist_mogp`` uses one separable kernel ``K_XX ⊗ K_HH``: every bin shares the
same x-lengthscale. Here the one-hot targets get the LMC-style kernel::

    k((x, h), (x', h')) = Σ_q k_x^(q)(x, x') k_h^(q)(h, h')

e.g. one latent with long x / short bin lengthscales and one the other way.
Each factor uses random Fourier features, so the model is Bayesian linear
regression on Kronecker features ``φ_x^(q)(x) ⊗ φ_h^(q)(h)``. The Gram blocks
factor as ``(Φ_x^qᵀ Φ_x^r) ⊗ (Φ_h^qᵀ Φ_h^r)`` and ``Φᵀy = vec(Φ_xᵀ R Φ_h)``,
so the exact marginal likelihood costs ``O(F³)`` with ``F = Σ_q D_x D_h``,
independent of ``N·B``. The posterior mean goes onto the simplex as in
``hist_mogp``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class _KronRFFHist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        centers: torch.Tensor,
        freqs: torch.Tensor,
        *,
        n_latent: int,
        dx: int,
        dh: int,
        seed: int,
    ) -> None:
        super().__init__()
        dt = torch.float64
        gen = torch.Generator().manual_seed(int(seed))
        self.n_latent, self.dx, self.dh = int(n_latent), int(dx), int(dh)
        self.register_buffer("centers", centers.to(dt).reshape(-1, 1))
        self.register_buffer("om_x", torch.randn(n_latent, input_dim, dx, generator=gen, dtype=dt))
        self.register_buffer("ph_x", 2 * math.pi * torch.rand(n_latent, dx, generator=gen, dtype=dt))
        self.register_buffer("om_h", torch.randn(n_latent, 1, dh, generator=gen, dtype=dt))
        self.register_buffer("ph_h", 2 * math.pi * torch.rand(n_latent, dh, generator=gen, dtype=dt))
        # Latent q starts at (ℓ_x, ℓ_h) = (long, short) or (short, long) in turn.
        lx0 = [math.log(2.0), math.log(0.5)]
        lh0 = [math.log(0.05), math.log(0.3)]
        self.log_ls_x = nn.Parameter(
            torch.stack(
                [torch.full((input_dim,), lx0[q % 2], dtype=dt) for q in range(n_latent)]
            )
        )
        self.log_ls_h = nn.Parameter(torch.tensor([lh0[q % 2] for q in range(n_latent)], dtype=dt))
        self.log_os = nn.Parameter(torch.full((n_latent,), math.log(0.05), dtype=dt))
        self.log_noise = nn.Parameter(torch.tensor(math.log(0.05), dtype=dt))
        self.mean = nn.Parameter(freqs.to(dt).clone())

    def clamp_(self) -> None:
        with torch.no_grad():
            self.log_ls_x.clamp_(math.log(5e-2), math.log(2e1))
            self.log_ls_h.clamp_(math.log(1e-2), math.log(5.0))
            self.log_os.clamp_(math.log(1e-4), math.log(5.0))
            self.log_noise.clamp_(math.log(1e-4), math.log(1.0))

    def phi_x(self, x: torch.Tensor) -> list[torch.Tensor]:
        ls = torch.exp(self.log_ls_x)
        s = math.sqrt(2.0 / self.dx)
        return [
            s * torch.cos((x / ls[q]) @ self.om_x[q] + self.ph_x[q]) for q in range(self.n_latent)
        ]

    def phi_h(self) -> list[torch.Tensor]:
        ls = torch.exp(self.log_ls_h)
        s = math.sqrt(2.0 / self.dh)
        return [
            s * torch.cos((self.centers / ls[q]) @ self.om_h[q] + self.ph_h[q])
            for q in range(self.n_latent)
        ]

    def _system(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``chol(A)``, ``b = Φᵀr``, prior variances ``S`` and ``r``."""
        px, ph = self.phi_x(x), self.phi_h()
        r = y - self.mean.unsqueeze(0)
        blocks = [
            [torch.kron(px[q].T @ px[p], ph[q].T @ ph[p]) for p in range(self.n_latent)]
            for q in range(self.n_latent)
        ]
        gram = torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)
        b = torch.cat([(px[q].T @ r @ ph[q]).reshape(-1) for q in range(self.n_latent)])
        s_prior = torch.cat(
            [torch.exp(self.log_os[q]).expand(self.dx * self.dh) for q in range(self.n_latent)]
        )
        noise = torch.exp(self.log_noise)
        a = gram + torch.diag(noise / s_prior)
        a = a + 1e-8 * torch.eye(a.shape[0], dtype=a.dtype)
        return torch.linalg.cholesky(a), b, s_prior, r

    def neg_mll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        chol, b, s_prior, r = self._system(x, y)
        noise = torch.exp(self.log_noise)
        nb = float(r.numel())
        f = float(b.numel())
        ainv_b = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)
        quad = (r.pow(2).sum() - b @ ainv_b) / noise
        logdet = (
            (nb - f) * self.log_noise
            + torch.log(s_prior).sum()
            + 2.0 * torch.log(torch.diagonal(chol)).sum()
        )
        return 0.5 * (quad + logdet + nb * math.log(2 * math.pi)) / nb

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        chol, b, _s, _r = self._system(x, y)
        w = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)
        px, ph = self.phi_x(xs), self.phi_h()
        out = self.mean.unsqueeze(0).expand(xs.shape[0], -1).clone()
        k = self.dx * self.dh
        for q in range(self.n_latent):
            wq = w[q * k : (q + 1) * k].reshape(self.dx, self.dh)
            out = out + px[q] @ wq @ ph[q].T
        return out


def fit_predict_hist_lmc(
    x_tr: Any,
    y_tr: Any,
    x_te: Any,
    *,
    n_bins: int = 40,
    tail_frac: float = 1.0,
    pad_frac: float = 0.05,
    n_latent: int = 2,
    dx: int = 48,
    dh: int = 12,
    num_steps: int = 300,
    lr: float = 0.03,
    seed: int = 42,
    **_: Any,
) -> dict[str, Any]:
    from foundation_adapters import _as_tensors_xy, _pack_bar_predictive
    from histogram_gp import (
        bin_centers,
        empirical_bin_frequencies,
        make_hist_bins,
        project_onto_simplex,
        y_to_onehot,
    )

    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    xtr, _x, xte, y_np = _as_tensors_xy(x_tr, y_tr, x_te)
    xtr = xtr.to(dtype=torch.float64).cpu()
    xte = xte.to(dtype=torch.float64).cpu()
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    borders, bin_info = make_hist_bins(
        y_np, n_bins=int(n_bins), pad_frac=float(pad_frac), tail_frac=float(tail_frac)
    )
    oh = torch.as_tensor(y_to_onehot(y_np, borders))
    freqs = torch.as_tensor(empirical_bin_frequencies(y_np, borders))
    c = bin_centers(borders)
    c = (c - c.min()) / max(float(c.max() - c.min()), 1e-6)
    model = _KronRFFHist(
        xtr.shape[1],
        torch.as_tensor(c),
        freqs,
        n_latent=n_latent,
        dx=dx,
        dh=min(int(dh), oh.shape[1]),
        seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for _ in range(int(num_steps)):
        opt.zero_grad(set_to_none=True)
        try:
            loss = model.neg_mll(xtr, oh)
        except RuntimeError:
            break
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        model.clamp_()
        if float(loss) < best:
            best = float(loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("hist_lmc: optimization failed at the first step")
    model.load_state_dict(best_state)
    probs = project_onto_simplex(model.predict_mean(xtr, oh, xte).numpy())
    extras = {
        "model": "hist_lmc",
        "n_latent": int(n_latent),
        "best_neg_mll_per_cell": best,
        "ls_x": torch.exp(model.log_ls_x).detach().numpy().round(3).tolist(),
        "ls_h": torch.exp(model.log_ls_h).detach().numpy().round(3).tolist(),
        "outputscale": torch.exp(model.log_os).detach().numpy().round(4).tolist(),
        **bin_info,
    }
    return _pack_bar_predictive(borders, probs, wall_s=time.perf_counter() - t0, extras=extras)


def fit_predict_hist_lmc_t1(x_tr: Any, y_tr: Any, x_te: Any, *, seed: int = 42, **kwargs: Any):
    """``hist_lmc`` inside the Tier-1 wrapper (tail bins, held-out B and ε)."""
    from foundation_adapters import fit_predict_hist_t1

    return fit_predict_hist_t1(
        x_tr, y_tr, x_te, base_fn=fit_predict_hist_lmc, tail_frac=1.0, seed=seed, **kwargs
    )
