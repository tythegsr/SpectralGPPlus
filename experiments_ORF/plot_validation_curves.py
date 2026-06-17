"""
Validation loss curves from saved ORF experiment JSON (validation_metrics_by_init).

For each gp_*.json with validation data, writes two figures:
  - all initializations' val_NLL on one plot
  - best init's train_loss vs val_NLL on one plot
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_multid_slice_predictions import sanitize_plot_subdir

RESULTS_ROOT = Path("experiments_ORF/results/orf_batch")
PLOT_OUTPUT_DIR = Path("experiments_ORF/results/orf_batch/plots")


def _parse_seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _normalize_by_init(by_init: dict) -> dict[str, list[dict]]:
    return {str(k): v for k, v in by_init.items()}


def _extract_validation_block(metrics: dict) -> dict[str, list[dict]] | None:
    """Top-level or nested under orf_metrics (wing vs TabPFN combined results)."""
    block = metrics.get("validation_metrics_by_init")
    if block:
        return _normalize_by_init(block)
    rff = metrics.get("orf_metrics")
    if isinstance(rff, dict):
        block = rff.get("validation_metrics_by_init")
        if block:
            return _normalize_by_init(block)
    return None


def _has_validation_data(metrics: dict) -> bool:
    if metrics.get("monitor_validation") is False:
        return False
    rff = metrics.get("orf_metrics")
    if isinstance(rff, dict) and rff.get("monitor_validation") is False:
        return False
    block = _extract_validation_block(metrics)
    return bool(block)


def _load_gp_jsons(example_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(example_dir.glob("**/gp_*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if not _has_validation_data(data):
                continue
            data["_source_file"] = str(path)
            data["_seed"] = _parse_seed_from_path(path)
            records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {path}: {exc}")
    return records


def _step_axis(records: list[dict]) -> tuple[np.ndarray, str]:
    if not records:
        return np.array([], dtype=np.float64), "Step"
    if any("lbfgs_iter" in r for r in records):
        steps = np.array(
            [float(r.get("lbfgs_iter", i)) for i, r in enumerate(records)],
            dtype=np.float64,
        )
        return steps, "LBFGS iteration"
    if any("epoch" in r for r in records):
        steps = np.array(
            [float(r.get("epoch", i)) for i, r in enumerate(records)],
            dtype=np.float64,
        )
        return steps, "Epoch"
    steps = np.arange(len(records), dtype=np.float64)
    return steps, "Logged step"


def _final_train_loss(records: list[dict]) -> float | None:
    if not records:
        return None
    loss = records[-1].get("train_loss")
    if loss is None:
        return None
    return float(loss)


def _infer_best_init(metrics: dict, by_init: dict[str, list[dict]]) -> int | None:
    if metrics.get("best_init_index") is not None:
        return int(metrics["best_init_index"])
    rff = metrics.get("orf_metrics")
    if isinstance(rff, dict) and rff.get("best_init_index") is not None:
        return int(rff["best_init_index"])

    target = metrics.get("best_train_loss")
    if target is None and isinstance(rff, dict):
        target = rff.get("best_train_loss")
    if target is not None:
        best_key = None
        best_diff = float("inf")
        for key, records in by_init.items():
            final = _final_train_loss(records)
            if final is None:
                continue
            diff = abs(final - float(target))
            if diff < best_diff:
                best_diff = diff
                best_key = key
        if best_key is not None:
            return int(best_key)

    best_key = None
    best_loss = float("inf")
    for key, records in by_init.items():
        final = _final_train_loss(records)
        if final is None:
            continue
        if final < best_loss:
            best_loss = final
            best_key = key
    return int(best_key) if best_key is not None else None


def _sanitize_stem(metrics: dict, source_path: str | None) -> str:
    title = metrics.get("title") or Path(source_path or "run").stem
    stem = sanitize_plot_subdir(str(title))
    seed = metrics.get("_seed")
    if seed is not None:
        stem = f"{stem}_seed{seed}"
    return stem


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    """Floor at tiny positive value so log-y plots stay valid."""
    out = np.asarray(values, dtype=np.float64)
    return np.maximum(out, 1e-12)


def _configure_log_yaxis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_yscale("log")
    ax.set_ylabel(f"{ylabel} (log scale)")
    ax.grid(True, which="both", alpha=0.28)


def _plot_all_inits(
    metrics: dict,
    by_init: dict[str, list[dict]],
    best_init: int | None,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    keys = sorted(by_init.keys(), key=lambda k: int(k))
    cmap = plt.get_cmap("tab20" if len(keys) <= 20 else "viridis")

    x_label = "Step"
    for i, key in enumerate(keys):
        records = by_init[key]
        steps, x_label = _step_axis(records)
        val_nll = _positive_for_log(
            np.array([float(r.get("val_NLL", np.nan)) for r in records], dtype=np.float64)
        )
        init_idx = int(key)
        is_best = best_init is not None and init_idx == best_init
        color = cmap(i % 20 if len(keys) <= 20 else i / max(len(keys) - 1, 1))
        ax.plot(
            steps,
            val_nll,
            color=color,
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0 if is_best else 0.55,
            linestyle="--" if is_best else "-",
            marker="o" if is_best else None,
            markersize=4 if is_best else 0,
            label=f"Init {init_idx + 1}" + (" (best)" if is_best else ""),
            zorder=3 if is_best else 2,
        )

    title = metrics.get("title", "ORF run")
    optimizer = metrics.get("optimizer")
    if optimizer is None:
        rff = metrics.get("orf_metrics")
        if isinstance(rff, dict):
            optimizer = rff.get("optimizer")
    n_val = metrics.get("n_val")
    seed = metrics.get("_seed")
    subtitle_parts = [p for p in (f"optimizer={optimizer}", f"n_val={n_val}", f"seed={seed}") if p.split("=")[-1] not in ("None", "")]
    ax.set_xlabel(x_label)
    _configure_log_yaxis(ax, "Validation NLL (val_NLL)")
    ax.set_title(f"{title}\nValidation loss — all initializations" + (f" ({', '.join(subtitle_parts)})" if subtitle_parts else ""))
    if len(keys) <= 12:
        ax.legend(loc="best", fontsize=8, ncol=2)
    else:
        handles, labels = ax.get_legend_handles_labels()
        best_handles = [h for h, lb in zip(handles, labels) if "(best)" in lb]
        best_labels = [lb for lb in labels if "(best)" in lb]
        if best_handles:
            ax.legend(best_handles, best_labels, loc="best", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_init(
    metrics: dict,
    by_init: dict[str, list[dict]],
    best_init: int,
    save_path: Path,
) -> None:
    key = str(best_init)
    records = by_init.get(key)
    if not records:
        raise ValueError(f"No validation records for best init {best_init}")

    steps, x_label = _step_axis(records)
    train_loss = _positive_for_log(
        np.array([float(r.get("train_loss", np.nan)) for r in records], dtype=np.float64)
    )
    val_nll = _positive_for_log(
        np.array([float(r.get("val_NLL", np.nan)) for r in records], dtype=np.float64)
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(steps, train_loss, color="#1B9E77", linewidth=2.0, marker="o", markersize=4, label="Train loss")
    ax.plot(
        steps,
        val_nll,
        color="#D95F02",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Val NLL",
    )

    best_train = metrics.get("best_train_loss")
    if best_train is None:
        rff = metrics.get("orf_metrics")
        if isinstance(rff, dict):
            best_train = rff.get("best_train_loss")
    title = metrics.get("title", "ORF run")
    train_str = f"{float(best_train):.4f}" if best_train is not None else "?"
    ax.set_xlabel(x_label)
    _configure_log_yaxis(ax, "Loss")
    ax.set_title(f"{title}\nBest init {best_init + 1} (final train loss={train_str})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_run(metrics: dict, save_dir: Path) -> list[Path]:
    """Generate validation curve PNGs for one result JSON dict."""
    by_init = _extract_validation_block(metrics)
    if not by_init:
        return []

    best_init = _infer_best_init(metrics, by_init)
    stem = _sanitize_stem(metrics, metrics.get("_source_file"))
    written: list[Path] = []

    all_inits_path = save_dir / f"{stem}_val_all_inits.png"
    _plot_all_inits(metrics, by_init, best_init, all_inits_path)
    written.append(all_inits_path)

    if best_init is not None and str(best_init) in by_init:
        best_path = save_dir / f"{stem}_val_best_init.png"
        _plot_best_init(metrics, by_init, best_init, best_path)
        written.append(best_path)

    return written


def plot_example(
    subdir: str,
    label: str,
    *,
    results_root: Path,
    plot_output_dir: Path,
) -> int:
    example_dir = results_root / subdir
    if not example_dir.is_dir():
        print(f"[WARN] Missing example dir: {example_dir}")
        return 0

    records = _load_gp_jsons(example_dir)
    if not records:
        print(f"[WARN] No validation metrics under {example_dir}")
        return 0

    save_dir = plot_output_dir / subdir / "validation"
    count = 0
    for metrics in records:
        paths = plot_run(metrics, save_dir)
        count += len(paths)
        for p in paths:
            print(f"  Wrote {p}")

    print(f"Plotted validation curves for {label}: {count} figure(s) from {len(records)} JSON file(s)")
    return count


def plot_all(
    *,
    results_root: Path | None = None,
    plot_output_dir: Path | None = None,
    examples: dict[str, dict[str, Any]] | None = None,
    subdirs: list[str] | None = None,
) -> int:
    results_root = Path(results_root or RESULTS_ROOT)
    plot_output_dir = Path(plot_output_dir or PLOT_OUTPUT_DIR)

    if not results_root.is_dir():
        raise FileNotFoundError(f"RESULTS_ROOT does not exist: {results_root}")

    plot_output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    if subdirs is not None:
        for subdir in subdirs:
            print("=" * 60)
            print(f"Plotting validation curves: {subdir}")
            total += plot_example(subdir, subdir, results_root=results_root, plot_output_dir=plot_output_dir)
    elif examples:
        for name, cfg in examples.items():
            print("=" * 60)
            print(f"Plotting validation curves: {name}")
            total += plot_example(
                cfg["subdir"],
                cfg.get("label", name),
                results_root=results_root,
                plot_output_dir=plot_output_dir,
            )
    else:
        for subdir in sorted(p.name for p in results_root.iterdir() if p.is_dir()):
            print("=" * 60)
            print(f"Plotting validation curves: {subdir}")
            total += plot_example(subdir, subdir, results_root=results_root, plot_output_dir=plot_output_dir)

    print(f"\nGenerated {total} validation plot(s) under {plot_output_dir}")
    return total


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validation loss curves for ORF batch results")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--plot-output-dir", type=str, default=None)
    parser.add_argument(
        "--subdirs",
        nargs="*",
        default=None,
        help="Example subdirs to scan (default: all under results-root)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root) if args.results_root else RESULTS_ROOT
    plot_output = Path(args.plot_output_dir) if args.plot_output_dir else results_root / "plots"

    plot_all(
        results_root=results_root,
        plot_output_dir=plot_output,
        subdirs=args.subdirs,
    )


if __name__ == "__main__":
    main()
