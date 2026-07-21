"""S2 11-QoI TOA benchmark with TabPFN (independent regressors per task)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# IDE RUN CONFIGURATION
# None trains all 11 QoIs. Example: QOI = ["algae", "fsnow"]
QOI: list[str] | None = None

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _TABPFN_DIR)

from experiments_toa.s2_cli import add_s2_common_args, parse_example_indices, selected_task_names
from toa_s2_tabpfn_base import PFN_MODEL_VERSION_CHOICES, run_s2_toa_tabpfn


def run_s2_toa_tabpfn_entry(**kwargs) -> dict:
    return run_s2_toa_tabpfn(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S2 11-QoI TOA with TabPFN")
    parser.add_argument("--n-train", type=int, default=16000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-posterior", action="store_true", default=True, dest="plot_posterior")
    parser.add_argument("--rel-tolerance", type=float, default=0.01)
    parser.add_argument("--posterior-n-examples", type=int, default=20)
    parser.add_argument("--posterior-example-indices", type=str, default=None)
    add_s2_common_args(parser, default_qoi=QOI)
    parser.add_argument("--pfn-device", type=str, default="cuda")
    parser.add_argument("--pfn-model-version", type=str, default="auto", choices=PFN_MODEL_VERSION_CHOICES)
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    parser.add_argument(
        "--posterior-pdf-mode",
        type=str,
        default="tabpfn_bar",
        choices=("gaussian", "tabpfn_bar"),
    )
    args = parser.parse_args()

    save_path = args.save_path or "experiments_TabPFN/results/s2_toa_tabpfn"
    run_s2_toa_tabpfn(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        save_path=save_path,
        plot_posterior=args.plot_posterior and not args.no_plot,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=parse_example_indices(args.posterior_example_indices),
        data_path=args.data_path,
        pfn_device=args.pfn_device,
        pfn_model_version=args.pfn_model_version,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        posterior_pdf_mode=args.posterior_pdf_mode,  # type: ignore[arg-type]
        input_variable=args.input_variable,
        task_names=selected_task_names(args),
        task_band_config=args.task_band_config,
    )
