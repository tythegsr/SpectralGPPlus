"""S4 independent TabPFN on snow-TOA / EMIT NetCDF → ASD validation.

Fourth-series example: condition TabPFNRegressor (one per QoI) on radiance +
geometry aux (coszen, ele_km) from a snow-TOA or EMIT NetCDF, then predict the
ASD field validation set in-process.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_SORF_DIR = _ROOT / "experiments_SORF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
# ASD-overlapping QoIs by default; None = all 11 S4 tasks.
QOI: list[str] | None = [
    "grain_size",
    "cos_i",
    "dust",
    "algae",
    "cwv",
    "lwc",
    "aot",
]
N_TRAIN = 5000
N_VAL = 0
SEED = 42
SAVE_PATH: str | None = None
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None
EVAL_ASD = True
ASD_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
)
PFN_DEVICE = "cuda"
PFN_MODEL_VERSION = "auto"
IGNORE_PRETRAINING_LIMITS = True
POSTERIOR_PDF_MODE = "tabpfn_bar"  # "gaussian" | "tabpfn_bar"
LOG_SCALE_QOI: list[str] | None = ["dust", "algae"]
LOGIT_SCALE_QOI: list[str] | None = []
LOG_OFFSETS: dict[str, float] | None = {"dust": 100.0, "algae": 100.0}
FILTER_VALID_LABELS = False
TASK_BAND_CONFIG: str | None = None
DATA_PATH: str | None = str(
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
    / "snow_toa_fsnow_90to100_flat_Sep03.nc"
)
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _TABPFN_DIR, _SORF_DIR)

from experiments_toa.s2_cli import parse_example_indices
from experiments_toa.s2_constants import S4_LOGIT_BOUNDS
from emit_s4_tabpfn_base import parse_s4_task_names, run_s4_emit_tabpfn


def run_s4_emit_tabpfn_entry(**kwargs) -> dict:
    return run_s4_emit_tabpfn(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "S4 TabPFN (independent regressors; radiance + geometry aux → ASD)"
        )
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument(
        "--qoi",
        nargs="+",
        default=None,
        metavar="NAME",
        help="QoIs (default: IDE QOI / ASD-overlapping 7)",
    )
    parser.add_argument("--pfn-device", type=str, default=PFN_DEVICE)
    parser.add_argument(
        "--pfn-model-version",
        type=str,
        default=PFN_MODEL_VERSION,
        choices=("auto", "v2.5", "v3.0"),
    )
    parser.add_argument(
        "--ignore-pretraining-limits",
        action=argparse.BooleanOptionalAction,
        default=IGNORE_PRETRAINING_LIMITS,
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--eval-asd",
        action=argparse.BooleanOptionalAction,
        default=EVAL_ASD,
        help="After fit, evaluate on ASD validation and write plots",
    )
    parser.add_argument(
        "--asd-path",
        type=str,
        default=ASD_PATH,
        help="ASD validation NetCDF (default: asd_validation_set.nc)",
    )
    parser.add_argument(
        "--filter-valid-labels",
        action=argparse.BooleanOptionalAction,
        default=FILTER_VALID_LABELS,
    )
    args = parser.parse_args()

    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    save_path = args.save_path or (
        f"experiments_TabPFN/results/Sept03/s4_emit_tabpfn_"
        f"ntrain{args.n_train}_pfn{args.pfn_model_version}{band_str}"
    )
    qoi = parse_s4_task_names(args.qoi if args.qoi is not None else QOI)
    print(
        f"S4 TabPFN  qoi={qoi}  n_train={args.n_train}  n_val={args.n_val}  "
        f"pfn_device={args.pfn_device}  log_qoi={LOG_SCALE_QOI}  "
        f"logit_qoi={LOGIT_SCALE_QOI}  task_bands={TASK_BAND_CONFIG}  "
        f"eval_asd={args.eval_asd}"
    )

    run_s4_emit_tabpfn(
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        save_path=save_path,
        plot_posterior=PLOT_POSTERIOR and PLOT and not args.no_plot,
        plot_asd=PLOT and not args.no_plot,
        rel_tolerance=REL_TOLERANCE,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        posterior_example_indices=parse_example_indices(POSTERIOR_EXAMPLE_INDICES),
        data_path=args.data_path,
        asd_path=args.asd_path,
        eval_asd=args.eval_asd,
        pfn_device=args.pfn_device,
        pfn_model_version=args.pfn_model_version,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        posterior_pdf_mode=POSTERIOR_PDF_MODE,  # type: ignore[arg-type]
        task_names=qoi,
        task_band_config=TASK_BAND_CONFIG,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        logit_bounds=dict(S4_LOGIT_BOUNDS),
        log_offsets=LOG_OFFSETS,
        filter_valid_labels=args.filter_valid_labels,
    )
