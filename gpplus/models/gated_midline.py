"""Input-gated mixture of GP midlines, selected automatically from one fit.

EM over ``K_mid`` LOO-GP midlines with an x-gate::

    p(y|x) = Σ_m g_m(x) N(y | μ_m(x), v_m(x))

E step uses each midline's closed-form LOO predictive at the training points
(honest responsibilities). M step refits each midline with responsibility
weights (noise σ²/r_im) and the gate by soft cross-entropy. Midlines whose
responsibility mass falls below ``min_frac·n`` are pruned during EM.

The mixture is kept only if it beats a single LOO midline on LOO
log-likelihood by ``gain_min`` nats/point **and** the gate is confident
(mean max-gate ≥ ``conf_min``), both scored with a cross-fitted gate: this keeps x-separated regimes (one midline
per regime) and leaves co-located multimodality to the Stage-2 offsets.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .loo_gp_core import LooGPCore


class _Gate(nn.Module):
    def __init__(self, d: int, k: int, hidden: int = 32, dtype=torch.float64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, k)).to(
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=-1)


def _fit_gate(
    gate: _Gate, x: torch.Tensor, r: torch.Tensor, alive: list[int], steps: int = 300
) -> None:
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logg = torch.log_softmax(gate.net(x)[:, alive], dim=-1)
        loss = -(r * logg).sum(-1).mean()
        loss.backward()
        opt.step()


def _crossfit_gate(x: torch.Tensor, r: torch.Tensor, *, n_folds: int, seed: int) -> torch.Tensor:
    """Out-of-fold gate log-probabilities ``[N, M]``."""
    n, m = r.shape
    gen = torch.Generator().manual_seed(int(seed) + 911)
    fold = torch.randperm(n, generator=gen) % int(n_folds)
    out = torch.empty(n, m, dtype=torch.float64)
    for f in range(int(n_folds)):
        te = fold == f
        torch.manual_seed(int(seed) + 13 * f)
        g = _Gate(x.shape[1], m)
        _fit_gate(g, x[~te], r[~te], list(range(m)))
        with torch.no_grad():
            out[te] = g(x[te])
    return out


def _kmeans(z: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = z[rng.choice(z.shape[0], size=k, replace=False)]
    lab = np.zeros(z.shape[0], dtype=int)
    for _ in range(iters):
        d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if np.any(lab == j):
                cent[j] = z[lab == j].mean(0)
    return lab


def _log_normal(y: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return -0.5 * math.log(2 * math.pi) - sd.log() - 0.5 * ((y - mu) / sd).pow(2)


def _stratified_subsample(lab: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Indices of ``m`` points with k-means cluster proportions preserved."""
    rng = np.random.default_rng(int(seed) + 5)
    n = lab.shape[0]
    out = []
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        take = max(1, int(round(m * idx.size / n)))
        out.append(rng.choice(idx, size=min(take, idx.size), replace=False))
    return np.sort(np.concatenate(out))


def fit_gated_midlines(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k_mid: int = 3,
    em_iters: int = 4,
    em_tol: float = 0.01,
    em_max_n: int = 1500,
    seed: int = 0,
    min_frac: float = 0.1,
    gain_min: float = 0.1,
    conf_min: float = 0.85,
    mean_terms: bool = True,
    backend: str = "auto",
    single: nn.Module | None = None,
) -> dict[str, Any]:
    """Return the selected midline structure (``n_mid == 1`` → single midline).

    EM, the single-midline reference and the acceptance test all run on a
    cluster-stratified subsample of at most ``em_max_n`` points; the final
    hard assignment of every point uses the gate plus each midline's LOO
    predictive (subsample rows) or plain predictive (held-out rows). EM stops
    after ``em_iters`` passes or once fewer than ``em_tol`` of the hard
    assignments change.
    """
    from .loo_gp_core import make_loo_core

    x_all = x.to(dtype=torch.float64).cpu()
    y_all = y.reshape(-1).to(dtype=torch.float64).cpu()
    n_all, d = x_all.shape

    z = torch.cat([x_all, y_all.unsqueeze(-1)], dim=-1).numpy()
    z = (z - z.mean(0)) / np.clip(z.std(0), 1e-8, None)
    lab_all = _kmeans(z, max(2, int(k_mid)), seed)
    sub = (
        _stratified_subsample(lab_all, int(em_max_n), seed)
        if n_all > int(em_max_n)
        else np.arange(n_all)
    )
    x, y, lab = x_all[sub], y_all[sub], lab_all[sub]
    n = x.shape[0]

    if single is None or n < n_all:
        single = make_loo_core(d, n, backend=backend, mean_terms=mean_terms, seed=seed).fit(
            x, y, num_inits=3, seed=seed
        )
    with torch.no_grad():
        mu1, sd1 = single.loo_train()
        ll_single = float(_log_normal(y, mu1, sd1).mean())

    info: dict[str, Any] = {
        "n_mid": 1,
        "ll_single": ll_single,
        "single": single,
        "em_n": int(n),
    }
    if n < 30 or k_mid < 2:
        return info

    r = torch.full((n, int(k_mid)), 0.05, dtype=torch.float64)
    r[torch.arange(n), torch.as_tensor(lab)] = 0.9
    r = r / r.sum(-1, keepdim=True)

    cores: list[nn.Module] = []
    alive = list(range(int(k_mid)))
    torch.manual_seed(int(seed) + 7)
    gate = _Gate(d, int(k_mid))
    ll_mix = float("-inf")
    prev_hard: torch.Tensor | None = None
    em_done = 0
    for it in range(int(em_iters)):
        em_done = it + 1
        # M step: midlines (warm start after the first pass) and gate.
        for j, m in enumerate(alive):
            if it == 0:
                cores.append(
                    make_loo_core(d, n, backend=backend, mean_terms=False, seed=seed + m).fit(
                        x, y, w=r[:, j], num_inits=2, steps=200, seed=seed + m
                    )
                )
            else:
                cores[j].fit(x, y, w=r[:, j], steps=80, warm_start=True, seed=seed + m)
        _fit_gate(gate, x, r, alive)
        # E step with honest LOO predictives.
        with torch.no_grad():
            logg = gate(x)[:, alive]
            logg = logg - torch.logsumexp(logg, dim=-1, keepdim=True)
            comps = []
            for j in range(len(alive)):
                mu_j, sd_j = cores[j].loo(x, y, r[:, j])
                comps.append(_log_normal(y, mu_j, sd_j))
            lj = logg + torch.stack(comps, dim=-1)
            ll_mix = float(torch.logsumexp(lj, dim=-1).mean())
            r = torch.softmax(lj, dim=-1)
        mass = r.sum(0)
        # Each accepted midline gets its own Stage-2 fit on its partition, so it
        # needs enough points for a d-dimensional GP.
        min_mass = max(5.0 * (d + 1), min_frac * n)
        keep = [j for j in range(len(alive)) if float(mass[j]) >= min_mass]
        pruned = len(keep) < len(alive)
        if pruned:
            alive = [alive[j] for j in keep]
            cores = [cores[j] for j in keep]
            r = r[:, keep]
            r = r / r.sum(-1, keepdim=True).clamp_min(1e-12)
        if len(alive) < 2:
            info.update({"ll_mix": ll_mix, "pruned_to": 1, "em_iters_run": em_done})
            return info
        hard = r.argmax(-1)
        if not pruned and prev_hard is not None:
            if float((hard != prev_hard).double().mean()) < float(em_tol):
                break
        prev_hard = None if pruned else hard

    # The gate saw each point's responsibility, so in-sample it can memorize
    # assignments of interleaved surfaces; score the mixture with a
    # cross-fitted gate instead.
    with torch.no_grad():
        comps = torch.stack(
            [_log_normal(y, *cores[j].loo(x, y, r[:, j])) for j in range(len(alive))], -1
        )
    logg_cv = _crossfit_gate(x, r, n_folds=5, seed=seed)
    ll_mix_cv = float(torch.logsumexp(logg_cv + comps, dim=-1).mean())
    conf = float(torch.exp(logg_cv).max(-1).values.mean())
    accept = (ll_mix_cv - ll_single) > float(gain_min) and conf >= float(conf_min)
    info.update(
        {
            "ll_mix": ll_mix,
            "ll_mix_cv": ll_mix_cv,
            "gate_conf": conf,
            "n_candidates": len(alive),
            "accepted": bool(accept),
            "em_iters_run": em_done,
        }
    )
    if not accept:
        return info
    if n == n_all:
        assign = r.argmax(-1).numpy()
    else:
        # Subsample rows keep their LOO responsibilities; the rest are scored
        # with each midline's plain predictive (they were never fit).
        with torch.no_grad():
            logg = gate(x_all)[:, alive]
            logg = logg - torch.logsumexp(logg, dim=-1, keepdim=True)
            comps = torch.stack(
                [_log_normal(y_all, *cores[j].predict(x_all)) for j in range(len(alive))], -1
            )
            assign = (logg + comps).argmax(-1).numpy()
        assign[sub] = r.argmax(-1).numpy()
    info.update(
        {
            "n_mid": len(alive),
            "gate": gate,
            "alive": alive,
            "assign": assign,
            "cores": cores,
        }
    )
    return info


@torch.no_grad()
def gate_probs(info: dict[str, Any], x: torch.Tensor) -> np.ndarray:
    """Gate probabilities ``[N, n_mid]`` for the accepted midlines."""
    logg = info["gate"](x.to(dtype=torch.float64).cpu())[:, info["alive"]]
    return torch.softmax(logg, dim=-1).numpy()
