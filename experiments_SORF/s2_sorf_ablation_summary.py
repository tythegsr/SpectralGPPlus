"""Summarize SORF noise/prior/init ablation results and recommend ranges.

Reads ``ablation_results.json`` (or row JSON files) under the ablation save root,
ranks by RRMSE within each (task, family), and writes:
  - ``ablation_ranking.csv``
  - ``ablation_recommendations.json``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_SAVE_ROOT = _SORF_DIR / "results" / "July18" / "ablation_noise_init_prior_off"

# Sane noise_std/RMSE band from the plan.
_RATIO_LO = 0.3
_RATIO_HI = 1.5


def load_rows(save_root: Path) -> list[dict[str, Any]]:
    consolidated = save_root / "ablation_results.json"
    if consolidated.is_file():
        return json.loads(consolidated.read_text(encoding="utf-8"))
    rows_dir = save_root / "rows"
    rows: list[dict[str, Any]] = []
    if rows_dir.is_dir():
        for path in sorted(rows_dir.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _score(row: dict[str, Any]) -> tuple[float, float]:
    """Primary: RRMSE; secondary: distance of noise_std/RMSE from 1.0."""
    rrmse = row.get("RRMSE")
    if rrmse is None:
        return (float("inf"), float("inf"))
    ratio = row.get("noise_std_over_RMSE")
    if ratio is None:
        ratio_pen = 10.0
    else:
        ratio_pen = abs(float(ratio) - 1.0)
        if not (_RATIO_LO <= float(ratio) <= _RATIO_HI):
            ratio_pen += 1.0
    return (float(rrmse), float(ratio_pen))


def rank_by_task_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Baseline participates in every family comparison for that task.
        groups[(row["task"], row["family"])].append(row)
        if row["family"] == "baseline":
            for fam in ("prior", "noise", "lengthscale", "outputscale"):
                groups[(row["task"], fam)].append(row)

    for (task, family), group in sorted(groups.items()):
        if family == "baseline":
            continue
        # Deduplicate by trial_id (baseline may appear twice).
        by_id: dict[str, dict[str, Any]] = {}
        for row in group:
            by_id[row["trial_id"]] = row
        unique = list(by_id.values())
        unique.sort(key=_score)
        for rank, row in enumerate(unique, start=1):
            ranked.append(
                {
                    **{k: row.get(k) for k in (
                        "trial_id",
                        "family",
                        "task",
                        "note",
                        "response_noise_prior",
                        "noise_var_fraction",
                        "noise_prior_log_scale",
                        "raw_noise_init",
                        "raw_lengthscale_init",
                        "raw_outputscale_init",
                        "RRMSE",
                        "RMSE",
                        "noise",
                        "noise_std",
                        "noise_std_over_RMSE",
                        "best_train_loss",
                        "wall_time_s",
                    )},
                    "rank_in_family": rank,
                    "is_winner": rank == 1,
                }
            )
    return ranked


def recommend(rows: list[dict[str, Any]], ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-task winners + majority vote across tasks for global defaults."""
    per_task: dict[str, dict[str, Any]] = {}
    votes: dict[str, list[str]] = defaultdict(list)

    winners = [r for r in ranked if r.get("is_winner")]
    for w in winners:
        task = w["task"]
        fam = w["family"]
        per_task.setdefault(task, {})
        per_task[task][fam] = {
            "trial_id": w["trial_id"],
            "RRMSE": w["RRMSE"],
            "noise_std_over_RMSE": w["noise_std_over_RMSE"],
            "response_noise_prior": w["response_noise_prior"],
            "noise_var_fraction": w["noise_var_fraction"],
            "noise_prior_log_scale": w["noise_prior_log_scale"],
            "raw_noise_init": w["raw_noise_init"],
            "raw_lengthscale_init": w["raw_lengthscale_init"],
            "raw_outputscale_init": w["raw_outputscale_init"],
            "initializer_hint": _winner_init_hint(w, rows),
        }
        votes[fam].append(w["trial_id"])

    global_defaults: dict[str, Any] = {}
    for fam, trial_ids in votes.items():
        # Majority vote; ties broken by first sorted trial id.
        counts: dict[str, int] = defaultdict(int)
        for tid in trial_ids:
            counts[tid] += 1
        best_tid = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        sample = next(r for r in winners if r["trial_id"] == best_tid and r["family"] == fam)
        global_defaults[fam] = {
            "trial_id": best_tid,
            "votes": dict(counts),
            "response_noise_prior": sample["response_noise_prior"],
            "noise_var_fraction": sample["noise_var_fraction"],
            "noise_prior_log_scale": sample["noise_prior_log_scale"],
            "raw_noise_init": sample["raw_noise_init"],
            "raw_lengthscale_init": sample["raw_lengthscale_init"],
            "raw_outputscale_init": sample["raw_outputscale_init"],
            "initializer_hint": _winner_init_hint(sample, rows),
        }

    return {
        "per_task": per_task,
        "global_majority_vote": global_defaults,
        "sane_noise_std_over_rmse_band": [_RATIO_LO, _RATIO_HI],
        "n_rows": len(rows),
    }


def _winner_init_hint(winner: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    match = next(
        (
            r
            for r in rows
            if r["trial_id"] == winner["trial_id"] and r["task"] == winner["task"]
        ),
        None,
    )
    if match is None:
        match = next((r for r in rows if r["trial_id"] == winner["trial_id"]), None)
    if match is None:
        return {}
    return dict(match.get("initializer_parameter_configs") or {})


def write_ranking_csv(path: Path, ranked: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ranked:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(ranked[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize S2 SORF noise/init ablation")
    parser.add_argument("--save-root", type=str, default=str(DEFAULT_SAVE_ROOT))
    args = parser.parse_args(argv)

    save_root = Path(args.save_root)
    rows = load_rows(save_root)
    if not rows:
        print(f"No ablation rows found under {save_root}")
        print("Run s2_sorf_noise_init_ablation.py first (or pass --save-root).")
        rec = {
            "per_task": {},
            "global_majority_vote": {},
            "status": "empty",
            "message": "no rows yet; using planned baseline until ablation completes",
            "planned_baseline": {
                "response_noise_prior": True,
                "noise_var_fraction": 0.001,
                "noise_prior_log_scale": 0.5,
                "initializer_parameter_configs": {},
            },
        }
        out = save_root / "ablation_recommendations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"Wrote placeholder recommendations to {out}")
        return 0

    ranked = rank_by_task_family(rows)
    write_ranking_csv(save_root / "ablation_ranking.csv", ranked)
    rec = recommend(rows, ranked)
    rec["status"] = "ok"
    (save_root / "ablation_recommendations.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )

    print(f"Rows: {len(rows)}")
    print(f"Wrote {save_root / 'ablation_ranking.csv'}")
    print(f"Wrote {save_root / 'ablation_recommendations.json'}")
    print("\nGlobal majority-vote winners:")
    for fam, info in sorted(rec.get("global_majority_vote", {}).items()):
        print(
            f"  {fam:12s}  trial={info['trial_id']}  "
            f"prior={info['response_noise_prior']}  "
            f"frac={info['noise_var_fraction']}  "
            f"s={info['noise_prior_log_scale']}  "
            f"noise_init={info['raw_noise_init']}  "
            f"ls_init={info['raw_lengthscale_init']}  "
            f"os_init={info['raw_outputscale_init']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
