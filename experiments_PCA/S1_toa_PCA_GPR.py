"""TOA benchmark with PCA + partitioned exact GPR."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _PCA_DIR, _GP_DIR, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from toa_pca_partition_gpr_base import run_toa_pca_partition_gpr


def run_toa_pca_gpr_entry(**kwargs) -> dict:
    return run_toa_pca_partition_gpr(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TOA dataset with PCA + partitioned exact GPR (GPPlus GPR)"
    )
    parser.add_argument("--n-train", type=int, default=49000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-components", type=int, default=8, help="PCA dimension p")
    parser.add_argument("--partition-size", type=int, default=1500, help="Training points per GP partition")
    parser.add_argument("--num-inits", type=int, default=8)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Epochs per init: 1 uses LBFGSScipy; >1 uses torch.optim.Adam",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Reserved for API parity (exact GP uses full test batch in evaluate_gp_model)",
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel hyperparameter inits per partition")
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Results directory (default: experiments_PCA/results/toa_pca)",
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
    )
    parser.add_argument("--rel-tolerance", type=float, default=0.01)
    parser.add_argument("--posterior-n-examples", type=int, default=8)
    parser.add_argument(
        "--posterior-example-indices",
        type=str,
        default=None,
        help="Comma-separated test row indices to plot",
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="132,195,196,197,198,199,200,201,202,203,204,205,206,207,208",
        help="Comma-separated 0-based column indices to remove before PCA; empty string = 285-dim",
    )
    parser.add_argument(
        "--compare-input-dim",
        action="store_true",
        help="Also run PCA+GP on full 285-dim (no drop) and log input_dim_ablation in JSON",
    )
    parser.add_argument(
        "--no-partition-shuffle",
        action="store_true",
        help="Disable shuffle before partitioning (not recommended)",
    )
    parser.add_argument("--top-m-partitions", type=int, default=5)
    parser.add_argument("--single-partition-index", type=int, default=0)
    parser.add_argument(
        "--no-log-grain",
        action="store_true",
        help="Disable log(grain) target transform",
    )
    parser.add_argument(
        "--pca-svd-solver",
        type=str,
        default="randomized",
        choices=("auto", "full", "randomized", "arpack"),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    gpplus.config.logger.setLevel(getattr(logging, args.log_level))

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    drop_columns: list[int] | None
    if args.drop_columns.strip() == "":
        drop_columns = []
    else:
        drop_columns = [int(x.strip()) for x in args.drop_columns.split(",") if x.strip()]

    posterior_example_indices = None
    if args.posterior_example_indices:
        posterior_example_indices = [
            int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()
        ]

    run_toa_pca_gpr_entry(
        n_train=args.n_train,
        n_test=args.n_test,
        n_components=args.n_components,
        partition_size=args.partition_size,
        seed=args.seed,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        device=args.device,
        dtype=dtype,
        save_path=args.save_path,
        ard=args.ard,
        predict_chunk_size=args.predict_chunk_size,
        n_jobs=args.n_jobs,
        plot_posterior=args.plot_posterior and not args.no_plot,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=posterior_example_indices,
        data_path=args.data_path,
        log_grain=not args.no_log_grain,
        drop_columns=drop_columns,
        partition_shuffle=not args.no_partition_shuffle,
        top_m_partitions=args.top_m_partitions,
        single_partition_index=args.single_partition_index,
        compare_input_dim=args.compare_input_dim,
        pca_svd_solver=args.pca_svd_solver,
    )
