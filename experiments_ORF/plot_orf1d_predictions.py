"""
1D ORF prediction plots: training points, true curve, GP mean, and 95% prediction intervals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_orf1d_prediction_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
    out_dir: str | Path,
    *,
    title: str,
    run_index: int,
    y_true_test: np.ndarray | None = None,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    file_suffix: str | None = None,
    file_tag: str | None = None,
    interval_z: float = 1.96,
    y_pred_tabpfn: np.ndarray | None = None,
    y_std_tabpfn: np.ndarray | None = None,
    gp_label: str = "ORF GP",
) -> Path:
    """Save a 1D ORF prediction figure with optional OOD train-domain markers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    order = np.argsort(x_test)
    xs = x_test[order]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.scatter(
        x_train,
        y_train,
        s=24,
        c="0.15",
        alpha=0.7,
        label="Train",
        zorder=5,
        edgecolors="white",
        linewidths=0.3,
    )

    if x_bounds is not None and test_x_bounds is not None:
        train_lo, train_hi = float(x_bounds[0]), float(x_bounds[1])
        test_lo, test_hi = float(test_x_bounds[0]), float(test_x_bounds[1])
        if test_lo < train_lo or test_hi > train_hi:
            ax.axvline(train_lo, color="0.45", linestyle="--", linewidth=1.0, alpha=0.8, label="Train domain")
            ax.axvline(train_hi, color="0.45", linestyle="--", linewidth=1.0, alpha=0.8)

    if y_true_test is not None:
        yt = np.asarray(y_true_test, dtype=np.float64).ravel()
        ax.plot(xs, yt[order], color="#D95F02", linewidth=2.0, label="True f(x)", zorder=2)

    if y_std is not None:
        sg = np.asarray(y_std, dtype=np.float64).ravel()
        if sg.shape[0] != x_test.shape[0]:
            raise ValueError(f"GP std length {sg.shape[0]} != x_test length {x_test.shape[0]}")
        lower = y_pred - interval_z * sg
        upper = y_pred + interval_z * sg
        ax.fill_between(
            xs,
            lower[order],
            upper[order],
            color="#1B9E77",
            alpha=0.18,
            linewidth=0.0,
            label=f"{gp_label} 95% PI",
            zorder=1,
        )

    ax.plot(xs, y_pred[order], color="#1B9E77", linewidth=1.7, label=f"{gp_label} pred", zorder=3, alpha=0.95)

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
    if file_tag:
        stem = file_tag
    else:
        suffix = f"_{file_suffix}" if file_suffix else ""
        stem = f"run_{run_index:03d}{suffix}"
    has_tabpfn = y_pred_tabpfn is not None
    fp = out_dir / (f"{stem}_1d_gp_tabpfn.png" if has_tabpfn else f"{stem}_rff1d.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_from_npz(npz_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Regenerate a 1D plot from a saved predictions_1d.npz file."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    out_dir = Path(out_dir) if out_dir is not None else npz_path.parent / "plots" / "prediction_runs"

    x_bounds = data["x_bounds"].tolist() if "x_bounds" in data else None
    test_x_bounds = data["test_x_bounds"].tolist() if "test_x_bounds" in data else None
    title = str(data["title"]) if "title" in data else npz_path.stem
    seed = int(data["seed"]) if "seed" in data else 0
    file_tag = str(data["file_tag"]) if "file_tag" in data else None

    return save_orf1d_prediction_plot(
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_pred"],
        data["y_std"] if "y_std" in data else None,
        out_dir,
        title=title,
        run_index=seed,
        y_true_test=data["y_true"] if "y_true" in data else None,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        file_tag=file_tag,
        y_pred_tabpfn=data["y_pred_tabpfn"] if "y_pred_tabpfn" in data else None,
        y_std_tabpfn=data["y_std_tabpfn"] if "y_std_tabpfn" in data else None,
    )


def plot_all_1d(
    results_root: str | Path,
    subdirs: list[str] | None = None,
    plot_output_dir: str | Path | None = None,
) -> list[Path]:
    """Scan results for predictions_1d*.npz and regenerate PNGs."""
    results_root = Path(results_root)
    written: list[Path] = []

    if subdirs is None:
        npz_files = sorted(results_root.glob("**/predictions_1d*.npz"))
    else:
        npz_files = []
        for subdir in subdirs:
            npz_files.extend(results_root.glob(f"{subdir}/**/predictions_1d*.npz"))
        npz_files = sorted(npz_files)

    seen: set[Path] = set()
    for npz_path in npz_files:
        if npz_path in seen:
            continue
        seen.add(npz_path)
        run_dir = npz_path.parent
        out_dir = plot_output_dir if plot_output_dir is not None else run_dir / "plots" / "prediction_runs"
        written.append(plot_from_npz(npz_path, out_dir))

    print(f"Generated {len(written)} 1D prediction plot(s) under {results_root}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate 1D ORF prediction plots from saved npz files")
    parser.add_argument("--results-root", type=str, default="experiments_ORF/results/orf_batch")
    parser.add_argument("--subdirs", nargs="*", default=None, help="Example subdirs to scan (default: all)")
    args = parser.parse_args()
    plot_all_1d(args.results_root, subdirs=args.subdirs)
