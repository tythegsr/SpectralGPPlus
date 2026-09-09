"""Run ASD field validation + plots after an S4_emit training run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD_PATH = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"

# QoIs present in asd_validation_set.nc / asd_solutions.csv.
ASD_TASK_NAMES: frozenset[str] = frozenset(
    {"grain_size", "cos_i", "dust", "algae", "cwv", "lwc", "aot"}
)

Backend = Literal["sorf", "svgp", "mtgpr"]


def resolve_asd_eval_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("ASD eval: CUDA unavailable; falling back to CPU")
        return "cpu"
    return device


def asd_tasks_from_trained(
    task_names: Sequence[str] | None,
) -> list[str] | None:
    """Keep only trained QoIs that have ASD labels; None → all ASD tasks."""
    if task_names is None:
        return None
    kept = [t for t in task_names if t in ASD_TASK_NAMES]
    return kept or None


def run_s4_asd_eval_and_plots(
    ckpt_dir: str | Path,
    *,
    backend: Backend,
    device: str = "cuda",
    asd_path: str | Path | None = None,
    checkpoint_title: str | None = None,
    tasks: Sequence[str] | None = None,
    plot: bool = True,
    predict_chunk_size: int = 512,
    coszen_override: float | None = None,
    sim_path: str | Path | None = None,
    neighbor_thr: float = 0.04,
    run_neighbors: bool = True,
) -> dict | None:
    """Evaluate checkpoints on ASD validation and optionally write plots.

    Returns the ASD metrics dict, or ``None`` when evaluation is skipped.
    """
    ckpt_dir = Path(ckpt_dir)
    asd_path = Path(asd_path) if asd_path is not None else DEFAULT_ASD_PATH
    if not ckpt_dir.is_dir():
        print(f"ASD eval skipped: checkpoint dir missing: {ckpt_dir}")
        return None
    if not asd_path.is_file():
        print(f"ASD eval skipped: ASD NetCDF missing: {asd_path}")
        return None

    eval_tasks = asd_tasks_from_trained(list(tasks) if tasks is not None else None)
    if tasks is not None and not eval_tasks:
        print(
            "ASD eval skipped: none of the trained QoIs have ASD labels "
            f"(trained={list(tasks)}, asd={sorted(ASD_TASK_NAMES)})"
        )
        return None

    device = resolve_asd_eval_device(device)
    print("\n" + "=" * 72)
    print(f"ASD validation eval  backend={backend}  ckpt_dir={ckpt_dir}")
    print("=" * 72)

    if backend == "sorf":
        from experiments_SORF._eval_s4_asd_validation import (
            evaluate_s4_checkpoints_on_asd,
        )

        results = evaluate_s4_checkpoints_on_asd(
            ckpt_dir,
            asd_path,
            device=device,
            predict_chunk_size=predict_chunk_size,
            tasks=eval_tasks,
            checkpoint_title=checkpoint_title,
            coszen_override=coszen_override,
        )
    elif backend == "svgp":
        from experiments_SVGP._eval_s4_asd_validation import (
            evaluate_s4_svgp_checkpoints_on_asd,
        )

        results = evaluate_s4_svgp_checkpoints_on_asd(
            ckpt_dir,
            asd_path,
            device=device,
            predict_chunk_size=predict_chunk_size,
            tasks=eval_tasks,
            checkpoint_title=checkpoint_title,
            coszen_override=coszen_override,
        )
    elif backend == "mtgpr":
        from experiments_RFFMTGPR._eval_s4_asd_validation import (
            evaluate_s4_mtgpr_checkpoint_on_asd,
        )

        results = evaluate_s4_mtgpr_checkpoint_on_asd(
            ckpt_dir,
            asd_path,
            device=device,
            predict_chunk_size=predict_chunk_size,
            tasks=eval_tasks,
            checkpoint_title=checkpoint_title,
            coszen_override=coszen_override,
        )
    else:
        raise ValueError(f"Unknown ASD eval backend: {backend!r}")

    eval_dir = ckpt_dir / "asd_validation_eval"
    if plot:
        print("\nGenerating ASD validation plots...")
        from experiments_SORF._plot_s4_asd_validation import (
            generate_asd_validation_plots,
        )

        plot_paths = generate_asd_validation_plots(
            eval_dir,
            asd_path=asd_path,
            out_dir=eval_dir / "plots",
        )
        print(f"Saved {len(plot_paths)} ASD plots under {eval_dir / 'plots'}")

    if run_neighbors:
        from experiments_toa.asd_spectral_neighbors import (
            resolve_sim_path_from_ckpt,
            run_asd_spectral_neighbors,
        )

        resolved_sim: Path | None = None
        if sim_path is not None:
            candidate = Path(sim_path)
            if candidate.is_file():
                resolved_sim = candidate
            else:
                print(f"ASD neighbor analysis: sim_path not found: {candidate}")
        if resolved_sim is None:
            resolved_sim = resolve_sim_path_from_ckpt(ckpt_dir)
        if resolved_sim is None:
            print(
                "ASD neighbor analysis skipped: could not resolve synthetic "
                "NetCDF (pass sim_path=training DATA_PATH or ensure gp_S4_*.json "
                "has data_meta.emit_path)"
            )
        else:
            try:
                print("\nRunning ASD spectral-neighbor analysis...")
                run_asd_spectral_neighbors(
                    asd_path,
                    resolved_sim,
                    eval_dir,
                    thr=neighbor_thr,
                    write_excel=True,
                    write_plots=True,
                    write_json=True,
                )
            except Exception as exc:  # noqa: BLE001 — keep ASD eval usable
                print(f"ASD neighbor analysis failed: {exc!r}")

    return results
