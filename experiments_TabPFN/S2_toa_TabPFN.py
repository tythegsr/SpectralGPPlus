"""S2 11-QoI TOA benchmark with TabPFN (independent regressors per task)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TABPFN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = None  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 16000
N_TEST = 5000
SEED = 42
SAVE_PATH: str | None = None
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None  # e.g. "0,3,7" or None
PFN_DEVICE = "cuda"
PFN_MODEL_VERSION = "auto"
IGNORE_PRETRAINING_LIMITS = False
POSTERIOR_PDF_MODE = "tabpfn_bar"  # "gaussian" | "tabpfn_bar"
DATA_PATH: str | None = None  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_radiance"
X_TRANSFORM = "none"  # "none" | "log1p"
LOG_SCALE: bool | None = None
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_from_corr.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _TABPFN_DIR)

from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from toa_s2_tabpfn_base import run_s2_toa_tabpfn


def run_s2_toa_tabpfn_entry(**kwargs) -> dict:
    return run_s2_toa_tabpfn(**kwargs)


if __name__ == "__main__":
    save_path = SAVE_PATH or "experiments_TabPFN/results/s2_toa_tabpfn"
    print(
        f"TabPFN IDE config  qoi={QOI}  pfn_device={PFN_DEVICE}  "
        f"output_log_scale={LOG_SCALE}  x_transform={X_TRANSFORM}"
    )

    run_s2_toa_tabpfn(
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=SEED,
        save_path=save_path,
        plot_posterior=PLOT_POSTERIOR and PLOT,
        rel_tolerance=REL_TOLERANCE,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        posterior_example_indices=parse_example_indices(POSTERIOR_EXAMPLE_INDICES),
        data_path=DATA_PATH,
        pfn_device=PFN_DEVICE,
        pfn_model_version=PFN_MODEL_VERSION,
        ignore_pretraining_limits=IGNORE_PRETRAINING_LIMITS,
        posterior_pdf_mode=POSTERIOR_PDF_MODE,  # type: ignore[arg-type]
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        x_transform=X_TRANSFORM,
        log_scale=LOG_SCALE,
    )
