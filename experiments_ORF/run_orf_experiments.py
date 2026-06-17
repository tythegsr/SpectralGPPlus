"""
Batch runner for ORF benchmarks (Ackley, Wing, ...).

Edit RESULTS_ROOT, NUM_RUNS, BASE_SEED, NOISE_LEVELS, and EXPERIMENTS below.
Each run uses seed = BASE_SEED + run_index and saves under:

    RESULTS_ROOT / <example subdir> / seed_<seed> / gp_<title>.json

After each example, writes manifest.json in the example directory.
When RUN_PLOTS_AFTER_BATCH is True (default), runs violin plots via
experiments_revisions_april/plot_violin_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
_ORF_DIR = Path(__file__).resolve().parent
_REV_APRIL_DIR = _ROOT / "experiments_revisions_april"
# Insert last-listed dir first on sys.path so experiments_ORF wins over revisions_april
# for shared module names (e.g. load_experimental_data).
for p in (_ROOT, _REV_APRIL_DIR, _ORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _ensure_orf_load_experimental_data() -> None:
    """Bind load_experimental_data to experiments_ORF (both trees define this module)."""
    import importlib.util

    target = (_ORF_DIR / "load_experimental_data.py").resolve()
    cached = sys.modules.get("load_experimental_data")
    if cached is not None:
        cached_path = getattr(cached, "__file__", None)
        if cached_path and Path(cached_path).resolve() == target:
            return
    spec = importlib.util.spec_from_file_location("load_experimental_data", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ORF data module from {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_experimental_data"] = mod
    spec.loader.exec_module(mod)


_ensure_orf_load_experimental_data()

import gpplus
from A1_wing_ORF import run_wing_orf
from A1_wing_ORF_vs_TabPFN import run_wing_orf_vs_tabpfn
from A4_ackley_ORF import run_ackley_40d_orf
from A10_tabpfn1d_abs_x_ORF import run_tabpfn1d_abs_x_orf
from A11_tabpfn1d_sin_freq_x_ORF import run_tabpfn1d_sin_freq_x_orf
from A14_tabpfn1d_step_ORF import run_tabpfn1d_step_orf
from A15_tabpfn1d_x_squared_ORF import run_tabpfn1d_x_squared_orf
from A18_tabpfn1d_linear_homoscedastic_ORF import run_tabpfn1d_linear_homoscedastic_orf
from plot_orf1d_predictions import plot_all_1d
from plot_validation_curves import plot_all as plot_validation_curves
from plot_violin_metrics import plot_all as plot_violin_metrics
from orf_experiment_utils import sin_pi_frequency_problem_name

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
ackley_dimensions = 10

ard = True
num_orf = 100
RESULTS_ROOT = Path(f"experiments_ORF/results_test_pfn_vs_orf/orf_batch_orfNum{num_orf}")
PLOT_OUTPUT_DIR = RESULTS_ROOT / "plots"
BASE_SEED = 42
NUM_RUNS = 10
NOISE_LEVELS = [0.0, 0.005, 0.05]
RUN_PLOTS_AFTER_BATCH = True



EXPERIMENTS: dict[str, dict[str, Any]] = {
    # f"ackley_{ackley_dimensions}Dx_10Dn": {
    #     "run_fn": run_ackley_40d_orf,
    #     "subdir": "ackley",
    #     "label": "Ackley ORF",
    #     "noise_levels": NOISE_LEVELS,
    #     "kwargs": {
    #         "dimensions": ackley_dimensions,
    #         "train_size": 10,
    #         "num_orf": num_orf,
    #         "num_test": 5000,
    #         "num_inits": 16,
    #         "num_epochs": 1,
    #         "device": "cpu",
    #         "ard": ard,
    #         "n_jobs": -1,
    #     },
    # },
    # f"ackley_{ackley_dimensions}Dx_40Dn": {
    #     "run_fn": run_ackley_40d_orf,
    #     "subdir": "ackley",
    #     "label": "Ackley ORF",
    #     "noise_levels": NOISE_LEVELS,
    #     "kwargs": {
    #         "dimensions": ackley_dimensions,
    #         "train_size": 40,
    #         "num_orf": num_orf,
    #         "num_test": 5000,
    #         "num_inits": 16,
    #         "num_epochs": 1,
    #         "device": "cpu",
    #         "ard": ard,
    #         "n_jobs": -1,
    #     },
    # },
    # "wing_s0_10Dx_10Dn": {
    #     "run_fn": run_wing_orf,
    #     "subdir": "wing_s0",
    #     "label": "Wing s0 ORF",
    #     "noise_levels": NOISE_LEVELS,
    #     "kwargs": {
    #         "train_samples_per_source": [100, 0, 0, 0],
    #         "test_samples_per_source": [5000, 0, 0, 0],
    #         "num_orf": num_orf,
    #         "num_inits": 16,
    #         "num_epochs": 1,
    #         "device": "cpu",
    #         "ard": True,
    #         "n_jobs": -1,
    #         "drop_source_column": None,
    #     },
    # },
    # "wing_s0_10Dx_40Dn": {
    #     "run_fn": run_wing_orf,
    #     "subdir": "wing_s0",
    #     "label": "Wing s0 ORF",
    #     "noise_levels": NOISE_LEVELS,
    #     "kwargs": {
    #         "train_samples_per_source": [400, 0, 0, 0],
    #         "test_samples_per_source": [5000, 0, 0, 0],
    #         "num_orf": num_orf,
    #         "num_inits": 16,
    #         "num_epochs": 1,
    #         "device": "cpu",
    #         "ard": True,
    #         "n_jobs": -1,
    #         "drop_source_column": None,
    #     },
    # },
    # "wing_s0_ORF_vs_TabPFN": {
    #     "run_fn": run_wing_orf_vs_tabpfn,
    #     "subdir": "wing_s0_ORF_vs_TabPFN",
    #     "label": "Wing s0 ORF vs TabPFN slices",
    #     "noise_levels": NOISE_LEVELS,
    #     "kwargs": {
    #         "train_samples_per_source": [100, 0, 0, 0],
    #         "test_samples_per_source": [5000, 0, 0, 0],
    #         "num_orf": num_orf,
    #         "num_inits": 16,
    #         "num_epochs": 1,
    #         "device": "cpu",
    #         "pfn_device": "cpu",
    #         "plot_slices": True,
    #         "n_jobs": -1,
    #     },
    # },
}

_TABPFN1D_COMMON_KWARGS = {
    "num_orf": num_orf,
    "num_inits": 16,
    "num_inits": 16,
    "num_epochs": 1,
    "num_test": 5000,
    "device": "cpu",
    "ard": ard,
    "n_jobs": -1,
    "plot_1d": True,
    "monitor_validation": True,
    "val_fraction": 0.2,
    "validation_verbose": False,
    "run_tabpfn": True,
    "pfn_device": "cuda",
}

TABPFN1D_SIN_FREQUENCY = 10.0

_TABPFN1D_PROBLEMS = [
    ("abs_x", run_tabpfn1d_abs_x_orf, 100),
    ("sin_freq_x", run_tabpfn1d_sin_freq_x_orf, 100),
    ("step", run_tabpfn1d_step_orf, 100),
    ("x_squared", run_tabpfn1d_x_squared_orf, 100),
    # ("linear_homoscedastic", run_tabpfn1d_linear_homoscedastic_orf, 100),
]

for _problem, _run_fn, _train_size in _TABPFN1D_PROBLEMS:
    _problem_slug = (
        sin_pi_frequency_problem_name(TABPFN1D_SIN_FREQUENCY)
        if _problem == "sin_freq_x"
        else _problem
    )
    for _ood_suffix, _margin in (
        ("", 0.0),
        ("_ood", 0.5),
    ):
        _subdir = f"tabpfn1d_{_problem_slug}{_ood_suffix}"
        _key = f"tabpfn1d_{_problem_slug}_{_train_size}Dn{_ood_suffix}"
        _extra_kwargs = (
            {"frequency": TABPFN1D_SIN_FREQUENCY} if _problem == "sin_freq_x" else {}
        )
        EXPERIMENTS[_key] = {
            "run_fn": _run_fn,
            "subdir": _subdir,
            "label": f"TabPFN1D {_problem_slug} ORF" + (" OOD" if _margin > 0 else ""),
            "noise_levels": NOISE_LEVELS,
            "kwargs": {
                **_TABPFN1D_COMMON_KWARGS,
                "train_size": _train_size,
                "test_outside_margin": _margin,
                **_extra_kwargs,
            },
        }

def _tabpfn1d_problem_slug(problem: str) -> str:
    if problem == "sin_freq_x":
        return sin_pi_frequency_problem_name(TABPFN1D_SIN_FREQUENCY)
    return problem


TABPFN1D_SUBDIRS = [f"tabpfn1d_{_tabpfn1d_problem_slug(p)}" for p, _, _ in _TABPFN1D_PROBLEMS] + [
    f"tabpfn1d_{_tabpfn1d_problem_slug(p)}_ood" for p, _, _ in _TABPFN1D_PROBLEMS
]


def _plot_examples_config(
    experiments: dict[str, dict[str, Any]],
    default_noise_levels: list[float],
) -> dict[str, dict[str, Any]]:
    """Build plot_violin_metrics EXAMPLES from runner EXPERIMENTS (one entry per subdir)."""
    by_subdir: dict[str, dict[str, Any]] = {}
    for name, cfg in experiments.items():
        subdir = cfg["subdir"]
        noise = cfg.get("noise_levels", default_noise_levels)
        if subdir not in by_subdir:
            by_subdir[subdir] = {
                "subdir": subdir,
                "label": cfg.get("label", name.replace("_", " ").title()),
                "noise_levels": list(noise),
            }
        else:
            merged = set(by_subdir[subdir]["noise_levels"]) | set(noise)
            by_subdir[subdir]["noise_levels"] = sorted(merged)
    return by_subdir


def _resolve_n_jobs(n_jobs: int | None) -> int | None:
    if n_jobs is None or n_jobs >= 0:
        return n_jobs if n_jobs >= 0 else None
    return None


def run_example(
    name: str,
    cfg: dict[str, Any],
    *,
    results_root: Path,
    base_seed: int,
    num_runs: int,
    default_noise_levels: list[float],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all noise levels and seeds for one registered example."""
    run_fn: Callable[..., dict] = cfg["run_fn"]
    subdir = cfg["subdir"]
    noise_levels = cfg.get("noise_levels", default_noise_levels)
    kwargs = dict(cfg.get("kwargs", {}))

    if "n_jobs" in kwargs:
        kwargs["n_jobs"] = _resolve_n_jobs(kwargs["n_jobs"])

    example_dir = results_root / subdir
    example_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "example": name,
        "subdir": subdir,
        "base_seed": base_seed,
        "num_runs": num_runs,
        "noise_levels": noise_levels,
        "kwargs": {k: v for k, v in kwargs.items() if k != "run_fn"},
        "runs": [],
    }

    print("=" * 60)
    print(f"Example: {name} -> {example_dir}")
    print(f"  noise_levels={noise_levels}, num_runs={num_runs}, seeds={base_seed}..{base_seed + num_runs - 1}")
    print("=" * 60)

    t_example = time.time()
    for noise in noise_levels:
        for i in range(num_runs):
            seed = base_seed + i
            save_path = example_dir / f"seed_{seed}"
            run_record = {
                "noise": noise,
                "run_index": i,
                "seed": seed,
                "save_path": str(save_path),
                "status": "pending",
            }
            print(f"\n[{name}] noise={noise} run={i + 1}/{num_runs} seed={seed}")

            if dry_run:
                run_record["status"] = "dry_run"
                manifest["runs"].append(run_record)
                continue

            save_path.mkdir(parents=True, exist_ok=True)
            try:
                metrics = run_fn(
                    seed=seed,
                    save_path=str(save_path),
                    noise_train=noise,
                    noise_test=noise,
                    **kwargs,
                )
                run_record["status"] = "ok"
                run_record["title"] = metrics.get("title")
                run_record["RRMSE"] = metrics.get("RRMSE")
                run_record["NIS"] = metrics.get("NIS")
                run_record["Total_Time"] = metrics.get("Total_Time")
                run_record["noise_std"] = metrics.get("noise_std")
                run_record["raw_noise"] = metrics.get("raw_noise")
                run_record["noise"] = metrics.get("noise")
            except Exception as exc:
                run_record["status"] = "error"
                run_record["error"] = str(exc)
                print(f"  FAILED: {exc}")
                traceback.print_exc()

            manifest["runs"].append(run_record)

    manifest["elapsed_s"] = time.time() - t_example
    manifest_path = example_dir / "manifest.json"
    if not dry_run:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nWrote manifest: {manifest_path}")

    ok = sum(1 for r in manifest["runs"] if r["status"] == "ok")
    print(f"Example {name} done: {ok}/{len(manifest['runs'])} runs succeeded in {manifest['elapsed_s']:.1f}s")
    return manifest


def run_all(
    *,
    results_root: Path | None = None,
    experiments: dict[str, dict[str, Any]] | None = None,
    base_seed: int | None = None,
    num_runs: int | None = None,
    noise_levels: list[float] | None = None,
    dry_run: bool = False,
    run_plots: bool | None = None,
    plot_output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    results_root = results_root or RESULTS_ROOT
    experiments = experiments or EXPERIMENTS
    base_seed = base_seed if base_seed is not None else BASE_SEED
    num_runs = num_runs if num_runs is not None else NUM_RUNS
    default_noise = noise_levels if noise_levels is not None else NOISE_LEVELS
    run_plots = RUN_PLOTS_AFTER_BATCH if run_plots is None else run_plots

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    plot_output_dir = Path(plot_output_dir) if plot_output_dir is not None else results_root / "plots"

    gpplus.config.configure_logger()

    summaries = {}
    for name, cfg in experiments.items():
        summaries[name] = run_example(
            name,
            cfg,
            results_root=results_root,
            base_seed=base_seed,
            num_runs=num_runs,
            default_noise_levels=default_noise,
            dry_run=dry_run,
        )

    if run_plots and not dry_run:
        print("\n" + "=" * 60)
        print("Generating violin plots...")
        print("=" * 60)
        plot_violin_metrics(
            results_root=results_root,
            plot_output_dir=plot_output_dir,
            examples=_plot_examples_config(experiments, default_noise),
        )
        print("\n" + "=" * 60)
        print("Generating validation curve plots...")
        print("=" * 60)
        plot_validation_curves(
            results_root=results_root,
            plot_output_dir=plot_output_dir,
            examples=_plot_examples_config(experiments, default_noise),
        )
        active_1d_subdirs = sorted(
            {
                cfg["subdir"]
                for cfg in experiments.values()
                if cfg.get("subdir", "").startswith("tabpfn1d_")
            }
        )
        if active_1d_subdirs:
            print("\n" + "=" * 60)
            print("Generating 1D prediction plots...")
            print("=" * 60)
            plot_all_1d(results_root, subdirs=active_1d_subdirs)
    elif run_plots and dry_run:
        print("\n[SKIP] Plots skipped (--dry-run).")

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ORF experiment runner")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help=f"Override RESULTS_ROOT (default: {RESULTS_ROOT})",
    )
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument(
        "--examples",
        nargs="*",
        default=None,
        help="Subset of EXPERIMENTS keys to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not train")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip violin, validation, and 1D plots after the batch finishes",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results (no training)",
    )
    parser.add_argument(
        "--plot-1d-only",
        action="store_true",
        help="Only regenerate 1D prediction plots from saved npz files (no training)",
    )
    parser.add_argument(
        "--plot-val-only",
        action="store_true",
        help="Only regenerate validation curve plots from saved JSON (no training)",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.examples:
        unknown = set(args.examples) - set(EXPERIMENTS)
        if unknown:
            raise SystemExit(f"Unknown examples: {unknown}. Available: {list(EXPERIMENTS)}")
        experiments = {k: EXPERIMENTS[k] for k in args.examples}

    results_root = Path(args.results_root) if args.results_root else RESULTS_ROOT
    default_noise = NOISE_LEVELS
    plot_output = results_root / "plots"

    if args.plot_val_only:
        plot_validation_curves(
            results_root=results_root,
            plot_output_dir=plot_output,
            examples=_plot_examples_config(experiments, default_noise),
        )
        return

    if args.plot_1d_only:
        active_1d_subdirs = sorted(
            {
                cfg["subdir"]
                for cfg in experiments.values()
                if cfg.get("subdir", "").startswith("tabpfn1d_")
            }
        )
        plot_all_1d(results_root, subdirs=active_1d_subdirs or TABPFN1D_SUBDIRS)
        return

    if args.plot_only:
        plot_violin_metrics(
            results_root=results_root,
            plot_output_dir=plot_output,
            examples=_plot_examples_config(experiments, default_noise),
        )
        plot_validation_curves(
            results_root=results_root,
            plot_output_dir=plot_output,
            examples=_plot_examples_config(experiments, default_noise),
        )
        active_1d_subdirs = sorted(
            {
                cfg["subdir"]
                for cfg in experiments.values()
                if cfg.get("subdir", "").startswith("tabpfn1d_")
            }
        )
        if active_1d_subdirs:
            plot_all_1d(results_root, subdirs=active_1d_subdirs)
        return

    run_all(
        results_root=results_root,
        experiments=experiments,
        base_seed=args.base_seed,
        num_runs=args.num_runs,
        dry_run=args.dry_run,
        run_plots=not args.no_plot,
        plot_output_dir=plot_output,
    )


if __name__ == "__main__":
    main()
