"""SorfLooCore: closed-form LOO == brute-force refit; accuracy/time vs exact LooGPCore."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpplus.models.loo_gp_core import LooGPCore, SorfLooCore  # noqa: E402


def _data(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64) * 4 - 2
    y = torch.sin(2 * x[:, 0]) + 0.3 * x[:, 0] + 0.1 * torch.randn(n, generator=g, dtype=torch.float64)
    return x, y


def test_closed_form_matches_bruteforce() -> None:
    x, y = _data(40)
    w = torch.rand(40, dtype=torch.float64) + 0.5
    core = SorfLooCore(1, num_freq=64)
    with torch.no_grad():
        mu, sd = core.loo(x, y, w)
        for i in range(0, 40, 7):
            keep = torch.arange(40) != i
            _p, _nv, chol, w_hat, _h, _e = core._solve(x[keep], y[keep], w[keep])
            phi_i = core.design(x[i : i + 1])
            m_i = core.const + (phi_i @ w_hat).item()
            v = torch.linalg.solve_triangular(chol, phi_i.T, upper=False)
            s_i = (v.pow(2).sum() + core.noise()).sqrt().item()
            assert abs(m_i - mu[i].item()) < 1e-8, (i, m_i, mu[i].item())
            assert abs(s_i - sd[i].item()) < 1e-8, (i, s_i, sd[i].item())


def compare(n: int = 1000) -> None:
    x, y = _data(n)
    xt, yt = _data(1000, seed=1)
    for name, core in (
        ("exact", LooGPCore(1)),
        ("sorf", SorfLooCore(1, num_freq=256)),
        ("sorf_nigp", SorfLooCore(1, num_freq=256, nigp=True)),
    ):
        t0 = time.perf_counter()
        core.fit(x, y, num_inits=2, steps=200)
        mu, sd = core.predict(xt)
        nll = float((0.5 * torch.log(2 * torch.pi * sd**2) + 0.5 * ((yt - mu) / sd) ** 2).mean())
        print(f"n={n} {name:10s} loo_nll={core.best_loss:.4f} test_nll={nll:.4f} "
              f"wall={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    test_closed_form_matches_bruteforce()
    print("closed-form LOO == brute-force refit: OK")
    compare(1000)
