"""TOA benchmark with TabPFN (independent regressors per task).

Requires tabpfn >= 6.0 for full-scale runs (n_train=49000, TabPFN v2.5+).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _TABPFN_DIR, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from toa_tabpfn_base import PFN_MODEL_VERSION_CHOICES, run_toa_tabpfn


def run_toa_tabpfn_entry(**kwargs) -> dict:
    return run_toa_tabpfn(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TOA dataset with TabPFN (one regressor per task)")
    parser.add_argument("--n-train", type=int, default=49000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Results directory (default: experiments_TabPFN/results/toa_tabpfn)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip posterior plots after saving JSON/NPZ",
    )
    parser.add_argument(
        "--plot-posterior",
        action="store_true",
        default=True,
        dest="plot_posterior",
        help="Generate posterior diagnostic plots (default: True)",
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=0.01,
        help="Relative error tolerance for pct_within metric (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--posterior-n-examples",
        type=int,
        default=20,
        help="Number of test spectra to plot as 3-panel posterior figures",
    )
    parser.add_argument(
        "--posterior-example-indices",
        type=str,
        default=None,
        help="Comma-separated test row indices to plot (overrides --posterior-n-examples)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to toa_data_flattened.npz (default: repo root)",
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="132,195,196,197,198,199,200,201,202,203,204,205,206,207,208",
        help="Comma-separated 0-based input column indices to remove",
    )
    parser.add_argument(
        "--pfn-device",
        type=str,
        default="cuda",
        help="TabPFN device (default: cuda). Set TABPFN_ALLOW_CPU_LARGE_DATASET=1 for large CPU runs.",
    )
    parser.add_argument(
        "--pfn-model-version",
        type=str,
        default="auto",
        choices=PFN_MODEL_VERSION_CHOICES,
        help="TabPFN checkpoint version (auto uses package default; v2.5/v3.0 when available)",
    )
    parser.add_argument(
        "--ignore-pretraining-limits",
        action="store_true",
        help="Pass ignore_pretraining_limits=True to TabPFNRegressor",
    )
    parser.add_argument(
        "--posterior-pdf-mode",
        type=str,
        default="tabpfn_bar",
        choices=("gaussian", "tabpfn_bar"),
        help="Posterior density for plots (default: tabpfn_bar)",
    )
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None:
        save_path = "experiments_TabPFN/results/toa_tabpfn"

    posterior_example_indices = None
    if args.posterior_example_indices:
        posterior_example_indices = [
            int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()
        ]

    drop_columns = None
    if args.drop_columns:
        drop_columns = [int(x.strip()) for x in args.drop_columns.split(",") if x.strip()]

    run_toa_tabpfn(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        save_path=save_path,
        plot_posterior=args.plot_posterior and not args.no_plot,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=posterior_example_indices,
        data_path=args.data_path,
        drop_columns=drop_columns,
        pfn_device=args.pfn_device,
        pfn_model_version=args.pfn_model_version,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        posterior_pdf_mode=args.posterior_pdf_mode,  # type: ignore[arg-type]
    )
