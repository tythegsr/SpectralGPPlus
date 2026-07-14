"""Build Phase 0 matched summary from historical apples-to-apples TOA JSONs (RRMSE)."""

from __future__ import annotations

import json
from pathlib import Path

from common import results_dir, save_json

_ROOT = Path(__file__).resolve().parents[2]

# Best matched large-n cell already on disk (same n, D, Woodbury runner).
HISTORICAL_49000 = {
    "sorf": _ROOT
    / "experiments_SORF/results/toa_sorf/gp_TOA_nTrain49000_nTest5000_sorfD1600.json",
    "orf": _ROOT / "experiments_ORF/results/toa_orf/gp_TOA_nTrain49000_nTest5000_orfD1600.json",
    "rff": _ROOT / "experiments_RFF/results/toa_rff/gp_TOA_nTrain49000_nTest5000_rffD1600.json",
}

# Near-matched mid-n cell (SORF July9 monitoring + ORF archive).
HISTORICAL_16000 = {
    "sorf": _ROOT
    / "experiments_SORF/results/July9_monitoring_maximin/toa_sorf_1inits_no_logit_cos_m1_std1_ls/gp_TOA_nTrain16000_nTest5000_sorfD800.json",
    "orf": _ROOT / "experiments_ORF/results/toa_orf/gp_TOA_nTrain16000_nTest5000_orfD800.json",
}


def _row_from_json(path: Path, mode: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "rff_sampling": mode,
        "n_train": d.get("n_train"),
        "n_test": d.get("n_test"),
        "num_rff": d.get("num_rff"),
        "source_json": str(path),
        "y_cos_RRMSE": d.get("y_cos_RRMSE"),
        "y_grain_RRMSE": d.get("y_grain_RRMSE"),
        "y_cos_RMSE": d.get("y_cos_RMSE"),
        "y_grain_RMSE": d.get("y_grain_RMSE"),
        "y_cos_R2": d.get("y_cos_R2"),
        "y_grain_R2": d.get("y_grain_R2"),
        "y_cos_NLPD": d.get("y_cos_NLPD"),
        "y_grain_NLPD": d.get("y_grain_NLPD"),
        "y_cos_noise": d.get("y_cos_noise"),
        "y_grain_noise": d.get("y_grain_noise"),
        "y_cos_best_train_loss": d.get("y_cos_best_train_loss"),
        "y_grain_best_train_loss": d.get("y_grain_best_train_loss"),
        "Training_Time": d.get("Training_Time"),
        "y_cos_checkpoint_path": d.get("y_cos_checkpoint_path"),
        "y_grain_checkpoint_path": d.get("y_grain_checkpoint_path"),
    }


def main() -> None:
    rows_49k = []
    for mode, path in HISTORICAL_49000.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows_49k.append(_row_from_json(path, mode))
        print(
            f"[49000/{mode}] cos RRMSE={rows_49k[-1]['y_cos_RRMSE']:.6g} "
            f"grain RRMSE={rows_49k[-1]['y_grain_RRMSE']:.6g}"
        )

    payload = {
        "phase": 0,
        "primary_metric": "RRMSE",
        "source": "historical matched STGP JSONs (n_train=49000, D=1600, n_test=5000)",
        "protocol": {
            "n_train": 49000,
            "n_test": 5000,
            "num_rff": 1600,
            "seed": "CLI default 42 (not stored in JSON)",
            "num_inits": "varies historically (SORF best_init suggests multi-init)",
            "train_subset": "mixed historically; shared Woodbury runner",
            "num_epochs": 200,
            "logit_cos": "see per-file (often True historically)",
            "log_grain": True,
        },
        "rows": rows_49k,
        "secondary_n16000_D800": [],
    }
    for mode, path in HISTORICAL_16000.items():
        if path.is_file():
            payload["secondary_n16000_D800"].append(_row_from_json(path, mode))

    out = results_dir() / "phase0_matched_summary.json"
    save_json(out, payload)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
