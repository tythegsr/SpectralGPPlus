"""Shared helpers for SORF vs RFF vs ORF diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[2]
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_SORF_DIR = _ROOT / "experiments_SORF"
_DIAG_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _SORF_DIR / "results" / "why_sorf_wins"

for p in (_ROOT, _RFF_DIR, _MTGPR_DIR, _SORF_DIR, _DIAG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

SAMPLING_MODES = ("rff", "orf", "sorf")
DROP_COLUMNS_DEFAULT = [132] + list(range(195, 209))


def results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def save_json(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return path


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_rbf_kernel(x: torch.Tensor, y: torch.Tensor, raw_lengthscale: torch.Tensor) -> torch.Tensor:
    """
    Exact RBF matching GPPlus RFF featurization.

    With scale s_d = 10^(raw_ls_d / 2):
        k(x, y) = exp(-0.5 * sum_d s_d^2 (x_d - y_d)^2)
    """
    scale = torch.pow(10.0, raw_lengthscale / 2.0)
    if scale.dim() == 0:
        scale = scale.expand(x.shape[-1])
    xs = x * scale
    ys = y * scale
    # (n, m) pairwise squared distances
    xx = (xs * xs).sum(-1, keepdim=True)
    yy = (ys * ys).sum(-1, keepdim=True).transpose(-2, -1)
    xy = xs @ ys.transpose(-2, -1)
    dist2 = (xx + yy - 2.0 * xy).clamp_min(0.0)
    return torch.exp(-0.5 * dist2)
