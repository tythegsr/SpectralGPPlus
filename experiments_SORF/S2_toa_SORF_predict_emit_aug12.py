"""IDE runner: evaluate Aug12 finished S2 SORF checkpoints on EMIT (GPU).

Edit the CONFIG block below, then Run this file in Cursor/VS Code, or:

  python experiments_SORF/S2_toa_SORF_predict_emit_aug12.py

Current target is the softmax snow-fraction 70–100% EMIT subset. Outputs go
to CKPT_DIR/emit_eval_snow_70to100 (does not overwrite the original emit_eval).

Mapped EMIT QoIs with finished Aug12 checkpoints:
  algae, aot, cwv, dust, fNPV, fPV
(cos_i has no EMIT label — state has sinA/cosA aspect params, not cos_i.
 grain_size/liquid_water/fsnow/fsoil are mapped but those checkpoints did
 not finish in the Aug12 run.)

Notes
-----
- X is standardized with each checkpoint's training x_scaler.
- Y stays in physical units; predictions are inverse-transformed before metrics.
- By default, out-of-range EMIT labels are dropped per task before scoring.
- fPV/fNPV are scored against z_pv/z_npv (pre-softmax logits).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SORF_DIR) not in sys.path:
    sys.path.insert(0, str(_SORF_DIR))

import gpplus
from S2_toa_SORF_predict_emit import evaluate_checkpoints_on_emit

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
CKPT_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug12"
    / "s2_toa_sorf_1inits_numrff800_lr0.1_taskbandconfig_rbf_nigp_freezeepochnigp100_sloperefreshes20_dtypefloat64"
)
EMIT_PATH = _ROOT / "split_files" / "emit_data_snow_70to100.nc"
# None = all mapped QoIs that have checkpoints. cos_i excluded (no EMIT label).
TASKS: list[str] | None = ["algae", "aot", "cwv", "dust", "fNPV", "fPV"]
DEVICE = "cuda"  # "cuda" | "cpu"
PREDICT_CHUNK_SIZE = 512
MAX_SAMPLES = 50000  # 0 = no cap after non-default filter
SEED = 42
NONDEFAULT_ONLY = True  # drop grain_radius ≈ 500 default rows
FILTER_VALID_LABELS = True  # drop EMIT labels outside valid ranges
SAVE_DIR: Path | None = CKPT_DIR / "emit_eval_snow_70to100"
LOG_LEVEL = "WARNING"
# ---------------------------------------------------------------------------


def main() -> None:
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL))

    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        print(f"  torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"  torch.version.cuda={torch.version.cuda}")
        device = "cpu"
    elif device.startswith("cuda"):
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    max_samples = None if MAX_SAMPLES == 0 else int(MAX_SAMPLES)
    save_dir = SAVE_DIR if SAVE_DIR is not None else Path(CKPT_DIR) / "emit_eval"

    print(f"ckpt_dir={CKPT_DIR}")
    print(f"emit_path={EMIT_PATH}")
    print(f"tasks={TASKS}")
    print(f"save_dir={save_dir}")

    evaluate_checkpoints_on_emit(
        CKPT_DIR,
        EMIT_PATH,
        device=device,
        predict_chunk_size=PREDICT_CHUNK_SIZE,
        max_samples=max_samples,
        seed=SEED,
        nondefault_only=NONDEFAULT_ONLY,
        filter_valid_labels=FILTER_VALID_LABELS,
        tasks=TASKS,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
