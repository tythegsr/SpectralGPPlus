"""Plots for MidlineSparseHardXGate (A02 hard w(x) routing).

Prefer ``python run_midline_hard_xgate.py`` (plots are written there).
Standalone (refits A02):

  python plot_midline_hard_xgate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
for p in (_ROOT, _EXP, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpplus.models import MidlineSparseHardXGateMoEGPR
from plot_midline_sparse import save_training_loss_subplots

FIG_DIR = _THIS / "results" / "hard_xgate" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_assignment_strip(
    model: MidlineSparseHardXGateMoEGPR,
    out_path: Path,
    *,
    title: str = r"A02  |  hard $w(x)$ expert assignment",
) -> Path:
    """Color points by hard expert assignment across x."""
    x = model.train_x.detach().cpu().numpy().ravel()
    y = model.train_y.detach().cpu().numpy().ravel()
    w = model.get_hard_weights(model.train_x).detach().cpu().numpy()
    assign = w.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k in range(model.k_max):
        m = assign == k
        if not np.any(m):
            continue
        ax.scatter(x[m], y[m], s=18, alpha=0.75, label=rf"expert $k={k}$")
    xs = np.linspace(float(x.min()), float(x.max()), 200).reshape(-1, 1)
    xt = torch.as_tensor(xs, dtype=model.dtype)
    pred = model.predict(xt)
    ax.plot(xs.ravel(), pred["mean"].cpu().numpy(), "k-", lw=1.8, label="mix mean")
    ax.plot(xs.ravel(), pred["mu0"].cpu().numpy(), "k--", lw=1.0, alpha=0.6, label="midline")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_gate_heatmap(
    model: MidlineSparseHardXGateMoEGPR,
    out_path: Path,
    *,
    title: str = r"Hard $w_k(x)$ (one-hot; $\sum_k w_k=1$)",
) -> Path:
    """Hard one-hot weights vs x (rows sum to 1)."""
    xs = np.linspace(
        float(model.train_x.min()), float(model.train_x.max()), 120
    ).reshape(-1, 1)
    xt = torch.as_tensor(xs, dtype=model.dtype)
    w = model.get_hard_weights(xt).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(
        w.T,
        aspect="auto",
        origin="lower",
        extent=[xs.min(), xs.max(), -0.5, model.k_max - 0.5],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("expert k")
    ax.set_yticks(range(model.k_max))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_hard_xgate_figures(
    model: MidlineSparseHardXGateMoEGPR,
    *,
    stem: str,
    fig_dir: Path | None = None,
    title_prefix: str = "A02",
) -> list[str]:
    """Write assignment + gate heatmap; return paths as strings."""
    out = Path(fig_dir) if fig_dir is not None else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        save_assignment_strip(
            model,
            out / f"{stem}_assignment.png",
            title=rf"{title_prefix}  |  hard $w(x)$ expert assignment",
        ),
        save_gate_heatmap(
            model,
            out / f"{stem}_gate_heatmap.png",
            title=rf"{title_prefix}  |  hard $w_k(x)$ (one-hot; $\sum_k w_k=1$)",
        ),
    ]
    for p in paths:
        print(f"Wrote {p}")
    return [str(p) for p in paths]


def main() -> None:
    from run_midline_hard_xgate import fit_hard_xgate
    from run_midline_sparse import make_step

    xs_tr, ys_tr, xs_te, ys_te = make_step(200, 200, seed=42)
    _row, model = fit_hard_xgate(
        xs_tr,
        ys_tr,
        xs_te,
        ys_te,
        name="plot_A02_hard_x",
        true_k_modes=1,
        k_max=4,
        num_inits=2,
        num_steps=500,
        seed=42,
        return_model=True,
    )
    assert isinstance(model, MidlineSparseHardXGateMoEGPR)
    save_hard_xgate_figures(model, stem="a02_hard_x", title_prefix="A02")


if __name__ == "__main__":
    main()
