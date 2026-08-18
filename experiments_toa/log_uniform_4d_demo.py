"""
4D demo: space-filling Sobol + linear vs log-uniform decode.

Shows why linear designs starve small log-scale values and how
Sobol u -> log_map fills each decade evenly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from toa_log_data import lin_map, log_map

N = 4096
SEED = 0
OUT_DIR = Path(__file__).resolve().parent / "analysis_log_uniform_4d_demo"

# Wide dynamic-range dims (algae / dust / grain / lwc analogues)
BOUNDS = np.array(
    [
        [1e-2, 1e2],
        [1.0, 1e3],
        [30.0, 1500.0],
        [1e-2, 25.0],
    ],
    dtype=np.float64,
)
DIM_NAMES = ["dim0_algae_like", "dim1_dust_like", "dim2_grain_like", "dim3_lwc_like"]


def decode(u: np.ndarray, map_fn) -> np.ndarray:
    x = np.empty_like(u, dtype=np.float64)
    for j, (lo, hi) in enumerate(BOUNDS):
        x[:, j] = map_fn(u[:, j], float(lo), float(hi))
    return x


def decade_occupancy(x: np.ndarray, lo: float, hi: float) -> list[tuple[str, int, float]]:
    """Count samples in each log10 decade overlapping [lo, hi]."""
    log_x = np.log10(x)
    d0 = int(np.floor(np.log10(lo)))
    d1 = int(np.ceil(np.log10(hi))) - 1
    rows: list[tuple[str, int, float]] = []
    n = len(x)
    for d in range(d0, d1 + 1):
        a, b = float(d), float(d + 1)
        # decade [10^d, 10^{d+1}), clipped to [lo, hi] for the label
        mask = (log_x >= a) & (log_x < b)
        # include right endpoint of final decade that touches hi
        if d == d1:
            mask = (log_x >= a) & (log_x <= np.log10(hi) + 1e-12)
        count = int(mask.sum())
        label = f"[10^{d:d}, 10^{d + 1:d})"
        rows.append((label, count, count / n))
    return rows


def print_decade_stats(design: str, x: np.ndarray) -> None:
    print(f"\n=== Decade occupancy: {design} ===")
    for j, name in enumerate(DIM_NAMES):
        lo, hi = BOUNDS[j]
        print(f"  {name}  physical [{lo:g}, {hi:g}]")
        for label, count, frac in decade_occupancy(x[:, j], lo, hi):
            print(f"    {label:16s}  n={count:5d}  ({100.0 * frac:5.1f}%)")


def plot_histograms(x_lin: np.ndarray, x_log: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), constrained_layout=True)
    for j, name in enumerate(DIM_NAMES):
        ax_lin, ax_log = axes[j, 0], axes[j, 1]
        lo, hi = BOUNDS[j]

        ax_lin.hist(x_lin[:, j], bins=40, alpha=0.55, label="linear decode", color="C0")
        ax_lin.hist(x_log[:, j], bins=40, alpha=0.55, label="log decode", color="C1")
        ax_lin.set_title(f"{name}: physical x")
        ax_lin.set_xlabel("x")
        ax_lin.legend(fontsize=8)

        ax_log.hist(np.log10(x_lin[:, j]), bins=40, alpha=0.55, label="linear decode", color="C0")
        ax_log.hist(np.log10(x_log[:, j]), bins=40, alpha=0.55, label="log decode", color="C1")
        ax_log.set_title(f"{name}: log10(x)")
        ax_log.set_xlabel("log10(x)")
        ax_log.legend(fontsize=8)
        ax_log.axvline(np.log10(lo), color="k", ls=":", lw=0.8)
        ax_log.axvline(np.log10(hi), color="k", ls=":", lw=0.8)

    fig.suptitle(f"Sobol N={N}: linear vs log-uniform decode", fontsize=12)
    fig.savefig(out_dir / "histograms_linear_vs_log.png", dpi=150)
    plt.close(fig)


def plot_scatter_pairs(x_lin: np.ndarray, x_log: np.ndarray, out_dir: Path) -> None:
    pairs = [(0, 1), (2, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for row, (i, j) in enumerate(pairs):
        for col, (x, title) in enumerate(
            [(x_lin, "linear decode"), (x_log, "log decode")]
        ):
            ax = axes[row, col]
            ax.scatter(
                np.log10(x[:, i]),
                np.log10(x[:, j]),
                s=4,
                alpha=0.35,
                c="C0" if col == 0 else "C1",
                rasterized=True,
            )
            ax.set_xlabel(f"log10({DIM_NAMES[i]})")
            ax.set_ylabel(f"log10({DIM_NAMES[j]})")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("2D coverage in log10 space", fontsize=12)
    fig.savefig(out_dir / "scatter_log_pairs.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sampler = qmc.Sobol(d=4, scramble=True, seed=SEED)
    u = sampler.random(n=N)

    x_lin = decode(u, lin_map)
    x_log = decode(u, log_map)

    print_decade_stats("linear", x_lin)
    print_decade_stats("log", x_log)

    plot_histograms(x_lin, x_log, OUT_DIR)
    plot_scatter_pairs(x_lin, x_log, OUT_DIR)

    np.savez_compressed(
        OUT_DIR / "samples.npz",
        u=u,
        x_linear=x_lin,
        x_log=x_log,
        bounds=BOUNDS,
        dim_names=np.array(DIM_NAMES),
    )
    print(f"\nWrote plots and samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
