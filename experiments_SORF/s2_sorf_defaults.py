"""Shared SORF defaults for S2 (baseline + optional ablation recommendations)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Plan baseline until ablation recommendations override.
S2_SORF_DEFAULT_RESPONSE_NOISE_PRIOR = True
S2_SORF_DEFAULT_NOISE_VAR_FRACTION = 0.001
S2_SORF_DEFAULT_NOISE_PRIOR_LOG_SCALE = 0.5

# Empty => STGP uses library defaults + constant-from-prior when prior is on.
S2_SORF_DEFAULT_INITIALIZER_PARAMETER_CONFIGS: dict[str, Any] = {}

_RECOMMENDATIONS_PATH = (
    Path(__file__).resolve().parent / "results" / "ablation_noise_init" / "ablation_recommendations.json"
)


def load_ablation_recommendations(
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    rec_path = Path(path) if path is not None else _RECOMMENDATIONS_PATH
    if not rec_path.is_file():
        return None
    return json.loads(rec_path.read_text(encoding="utf-8"))


def sorf_defaults_from_recommendations(
    recommendations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve SORF noise/init defaults.

    Prefer majority-vote winners from a completed ablation; otherwise return the
    planned baseline (prior on, fraction 0.001, log_scale 0.5, default inits).
    """
    rec = recommendations if recommendations is not None else load_ablation_recommendations()
    out: dict[str, Any] = {
        "response_noise_prior": S2_SORF_DEFAULT_RESPONSE_NOISE_PRIOR,
        "noise_var_fraction": S2_SORF_DEFAULT_NOISE_VAR_FRACTION,
        "noise_prior_log_scale": S2_SORF_DEFAULT_NOISE_PRIOR_LOG_SCALE,
        "initializer_parameter_configs": dict(S2_SORF_DEFAULT_INITIALIZER_PARAMETER_CONFIGS),
        "source": "planned_baseline",
    }
    if not rec or rec.get("status") == "empty":
        return out

    votes = rec.get("global_majority_vote") or {}
    prior_w = votes.get("prior")
    if prior_w is not None:
        out["response_noise_prior"] = bool(prior_w.get("response_noise_prior", True))
        out["noise_var_fraction"] = float(prior_w.get("noise_var_fraction", 0.001))
        out["noise_prior_log_scale"] = float(prior_w.get("noise_prior_log_scale", 0.5))
        out["source"] = "ablation_majority_vote"

    pcs: dict[str, Any] = {}
    for fam, key in (
        ("noise", "raw_noise"),
        ("lengthscale", "raw_lengthscale"),
        ("outputscale", "raw_outputscale"),
    ):
        w = votes.get(fam)
        if not w:
            continue
        hint = w.get("initializer_hint") or {}
        if key in hint:
            pcs[key] = hint[key]
            out["source"] = "ablation_majority_vote"
        # Baseline winners leave hint empty => keep library default for that param.
    out["initializer_parameter_configs"] = pcs
    return out
