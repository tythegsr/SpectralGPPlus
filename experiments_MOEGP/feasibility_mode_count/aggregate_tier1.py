"""Aggregate Tier-1/2/3/4 comparison runs across seeds into a markdown table.

Usage:
    python aggregate_tier1.py ROOT [ROOT ...] [--models a,b,c] [--seeds 0,1,42] [--out file.md]

Every ``*_metrics.json`` under each root is read; the seed comes from the
``protocol.seed`` field, so several out-dirs may contribute the same seed.
Directories named ``rerun_*`` are read last and override earlier entries for
the same (problem, n_train, model, seed). ``CRPS_bar`` (exact CRPS of a bar
PPD, or the closed-form Gaussian-mixture CRPS) and ``wall_s`` come from the
per-model extras when present. Tables are grouped by (problem, n_train).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("RMSE", "CRPS", "CRPS_bar", "log_score", "cov90", "hellinger_mean", "wall_s")
HEADERS = ("RMSE", "CRPS", "CRPS exact", "log score", "cov90", "Hellinger", "wall s")


def _hellinger(row: dict) -> float | None:
    for key in ("hellinger_mean", "Hellinger", "hellinger"):
        if row.get(key) is not None:
            return float(row[key])
    return None


def load(roots: list[Path], seeds: set[int] | None = None) -> dict:
    table: dict = defaultdict(lambda: defaultdict(dict))
    paths = sorted(
        (p for root in roots for p in root.rglob("*_metrics.json")),
        key=lambda p: (p.parent.name.startswith("rerun_"), str(p)),
    )
    for path in paths:
        d = json.loads(path.read_text())
        proto = d.get("protocol", {})
        problem = proto.get("problem", path.stem.removesuffix("_metrics"))
        n_train = proto.get("n_train")
        seed = int(proto.get("seed", -1))
        if seeds is not None and seed not in seeds:
            continue
        results = d.get("results", {})
        for row in d.get("rows", []):
            if row.get("source") == "error" or row.get("RMSE") is None:
                continue
            key = row["model_key"]
            ex = (results.get(key) or {}).get("extras") or {}
            vals = {m: row.get(m) for m in ("RMSE", "CRPS", "log_score", "cov90")}
            vals["CRPS_bar"] = ex.get("CRPS_bar")
            vals["hellinger_mean"] = _hellinger(row)
            vals["wall_s"] = row.get("wall_s", ex.get("wall_s"))
            table[(problem, n_train)][key][seed] = vals
    return table


def fmt(vals: list[float | None]) -> str:
    arr = np.asarray([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return "—"
    if arr.size == 1:
        return f"{arr[0]:.3f}"
    return f"{arr.mean():.3f}±{arr.std(ddof=1):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", default=None, help="Comma list of seeds to include")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    table = load(list(args.roots), seeds)
    wanted = args.models.split(",") if args.models else None
    lines: list[str] = []
    for problem, n_train in sorted(table, key=lambda k: (k[0], k[1] or 0)):
        models = table[(problem, n_train)]
        keys = wanted or sorted(models)
        lines.append(f"\n### {problem} (n_train={n_train})\n")
        lines.append("| model | seeds | " + " | ".join(HEADERS) + " |")
        lines.append("|---|---|" + "---|" * len(HEADERS))
        for key in keys:
            if key not in models:
                continue
            per_seed = models[key]
            cells = [fmt([per_seed[s].get(m) for s in per_seed]) for m in METRICS]
            lines.append(f"| {key} | {len(per_seed)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
