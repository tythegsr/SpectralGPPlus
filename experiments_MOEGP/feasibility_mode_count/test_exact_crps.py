"""Check the closed-form CRPS helpers against brute-force integration.

  python test_exact_crps.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from histogram_gp import mixture_crps
from quantile_gp import _tail_scales, quantile_crps


def _crps_numeric(cdf, y, lo, hi, n=400_001):
    z = np.linspace(lo, hi, n)
    return np.trapezoid((cdf(z) - (z >= y)) ** 2, z)


def test_mixture():
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = rng.integers(1, 4)
        mu, sd = rng.normal(size=k) * 2, rng.uniform(0.1, 1.5, size=k)
        w = rng.dirichlet(np.ones(k))
        y = rng.normal() * 2
        cdf = lambda z: sum(w[j] * norm.cdf(z, mu[j], sd[j]) for j in range(k))
        ref = _crps_numeric(cdf, y, -30, 30)
        got = mixture_crps(np.array([y]), mu[None], sd[None], w[None])[0]
        assert abs(ref - got) < 1e-4, (ref, got)


def test_quantile():
    tau = np.linspace(0.05, 0.95, 19)
    q = norm.ppf(tau)[None] * 1.3 + 0.4
    for xi in (0.0, 0.2):
        lam_lo, lam_hi = _tail_scales(q, tau, 1.0)
        # Monte-Carlo-free reference: CRPS = E|Y−y| − ½E|Y−Y'| via the quantile function on a dense τ grid.
        t = np.concatenate([tau[0] * np.logspace(-12, 0, 20000)[:-1], np.linspace(tau[0], tau[-1], 20001),
                            1 - (1 - tau[-1]) * np.logspace(0, -12, 20000)[1:]])
        def qf(tt):
            out = np.interp(tt, tau, q[0])
            lo = tt < tau[0]
            hi = tt > tau[-1]
            u_lo = tt[lo] / tau[0]
            u_hi = (1 - tt[hi]) / (1 - tau[-1])
            ex = (lambda u, l: -l * np.log(u)) if xi == 0 else (lambda u, l: l * (u ** -xi - 1) / xi)
            out[lo] = q[0, 0] - ex(u_lo, lam_lo[0])
            out[hi] = q[0, -1] + ex(u_hi, lam_hi[0])
            return out
        for y in (-2.0, 0.3, 3.5):
            qq = qf(t)
            rho = np.maximum(t * (y - qq), (t - 1) * (y - qq))
            ref = 2 * np.trapezoid(rho, t)
            got = quantile_crps(q, tau, np.array([y]), xi=xi)[0]
            assert abs(ref - got) < 2e-3 * max(1.0, ref), (xi, y, ref, got)


if __name__ == "__main__":
    test_mixture()
    test_quantile()
    print("exact CRPS tests passed")
