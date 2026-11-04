"""Kernel-approximation error of RFF / ORF / plain SORF / padded SORF vs exact RBF.

Run: ``python test_sorf_features.py`` (prints a table, asserts padded SORF <= RFF).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpplus.utils.rff_utils import init_rbf_weights  # noqa: E402
from gpplus.utils.sorf_features import sample_padded_sorf  # noqa: E402


def _approx_mse(x: torch.Tensor, w: torch.Tensor) -> float:
    p = x @ w
    z = torch.cat([torch.cos(p), torch.sin(p)], -1) / (w.shape[1] ** 0.5)
    k_hat = z @ z.T
    k = torch.exp(-0.5 * torch.cdist(x, x).pow(2))
    return float((k_hat - k).pow(2).mean())


def kernel_error_table(
    dims=(1, 2, 8, 32), freqs=(256, 1024), reps: int = 5, n: int = 300
) -> list[dict]:
    rows = []
    for d in dims:
        for D in freqs:
            acc = {"rff": 0.0, "orf": 0.0, "sorf": 0.0, "psorf": 0.0, "psorf_chi": 0.0}
            for r in range(reps):
                torch.manual_seed(100 * r + d)
                # Unit-lengthscale RBF; scale inputs so typical distances ~ 1.
                x = torch.randn(n, d, dtype=torch.float64) / (d**0.5)
                gen = torch.Generator().manual_seed(7 * r + d)
                ws = {
                    "rff": torch.randn(d, D, dtype=torch.float64),
                    "orf": init_rbf_weights(d, D, dtype=torch.float64, rff_sampling="orf"),
                    "sorf": init_rbf_weights(
                        d, D, dtype=torch.float64, rff_sampling="sorf", correct_sorf=True
                    ),
                    "psorf": sample_padded_sorf(d, D, radial="none", generator=gen),
                    "psorf_chi": sample_padded_sorf(d, D, radial="chi", generator=gen),
                }
                for k, w in ws.items():
                    acc[k] += _approx_mse(x, w) / reps
            rows.append({"d": d, "D": D, **acc})
    return rows


def test_padded_sorf_beats_rff() -> None:
    rows = kernel_error_table()
    for r in rows:
        assert r["psorf"] <= r["rff"], r


if __name__ == "__main__":
    rows = kernel_error_table()
    hdr = f"{'d':>3} {'D':>5} " + " ".join(f"{k:>10}" for k in ("rff", "orf", "sorf", "psorf", "psorf_chi"))
    print(hdr)
    for r in rows:
        print(
            f"{r['d']:>3} {r['D']:>5} "
            + " ".join(f"{r[k]:10.2e}" for k in ("rff", "orf", "sorf", "psorf", "psorf_chi"))
        )
    test_padded_sorf_beats_rff()
    print("padded SORF <= RFF for every (d, D): OK")
