"""
1D plots: training points vs GP vs TabPFN predictions on the test grid (per run).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_1d_train_gp_tabpfn_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_pred_gp: np.ndarray | None,
    y_pred_tabpfn: np.ndarray | None,
    y_std_gp: np.ndarray | None,
    y_std_tabpfn: np.ndarray | None,
    out_dir: str | Path,
    *,
    title: str,
    run_index: int,
    y_true_test: np.ndarray | None = None,
    file_suffix: str | None = None,
    interval_z: float = 1.96,
) -> Path:
    """
    Save a single figure with train scatter, optional ground truth, and model mean predictions
    on the test x locations (curves sorted by x).

    Predictions should already be in original y scale (as returned by train_eval_gp / train_eval_PFN).
    x_train / x_test should be in original input scale (not X-standardized).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()

    order = np.argsort(x_test)
    xs = x_test[order]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.scatter(x_train, y_train, s=24, c="0.15", alpha=0.7, label="Train", zorder=5, edgecolors="white", linewidths=0.3)

    if y_true_test is not None:
        yt = np.asarray(y_true_test, dtype=np.float64).ravel()
        ax.plot(xs, yt[order], color="#D95F02", linewidth=2.0, label="True f(x)", zorder=2)

    if y_pred_gp is not None:
        yg = np.asarray(y_pred_gp, dtype=np.float64).ravel()
        if yg.shape[0] != x_test.shape[0]:
            raise ValueError(f"GP pred length {yg.shape[0]} != x_test length {x_test.shape[0]}")
        if y_std_gp is not None:
            sg = np.asarray(y_std_gp, dtype=np.float64).ravel()
            if sg.shape[0] != x_test.shape[0]:
                raise ValueError(f"GP std length {sg.shape[0]} != x_test length {x_test.shape[0]}")
            gp_lower = yg - interval_z * sg
            gp_upper = yg + interval_z * sg
            ax.fill_between(
                xs,
                gp_lower[order],
                gp_upper[order],
                color="#1B9E77",
                alpha=0.18,
                linewidth=0.0,
                label="GP 95% PI",
                zorder=1,
            )
        ax.plot(xs, yg[order], color="#1B9E77", linewidth=1.7, label="GP pred", zorder=3, alpha=0.95)

    if y_pred_tabpfn is not None:
        yp = np.asarray(y_pred_tabpfn, dtype=np.float64).ravel()
        if yp.shape[0] != x_test.shape[0]:
            raise ValueError(f"TabPFN pred length {yp.shape[0]} != x_test length {x_test.shape[0]}")
        if y_std_tabpfn is not None:
            sp = np.asarray(y_std_tabpfn, dtype=np.float64).ravel()
            if sp.shape[0] != x_test.shape[0]:
                raise ValueError(f"TabPFN std length {sp.shape[0]} != x_test length {x_test.shape[0]}")
            pfn_lower = yp - interval_z * sp
            pfn_upper = yp + interval_z * sp
            ax.fill_between(
                xs,
                pfn_lower[order],
                pfn_upper[order],
                color="#7570B3",
                alpha=0.18,
                linewidth=0.0,
                label="TabPFN 95% PI",
                zorder=1,
            )
        ax.plot(xs, yp[order], color="#7570B3", linewidth=1.7, label="TabPFN pred", zorder=4, alpha=0.95)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}\nrun {run_index}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    fp = out_dir / f"run_{run_index:03d}{suffix}_train_gp_tabpfn.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp
