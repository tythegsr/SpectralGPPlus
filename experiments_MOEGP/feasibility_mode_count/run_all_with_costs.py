"""Run feasibility scripts that emit cost tables; aggregate into one master table.

Usage (gpplus2026)::

  python run_all_with_costs.py                  # core MidlineSparse suite
  python run_all_with_costs.py --suite full     # + foundation / UQ / hard_x / kmax
  python run_all_with_costs.py --suite examples # examples only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cost_tables import make_cost_row, write_cost_artifacts

_THIS = Path(__file__).resolve().parent
_PY = sys.executable
OUT = _THIS / "results" / "cost_master"


def _run(cmd: list[str], *, label: str) -> dict:
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_THIS))
    wall = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(
        f"=== [{label}] exit={proc.returncode} wall={wall:.1f}s ===",
        flush=True,
    )
    return {
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": ok,
        "wall_s": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("examples", "core", "full"),
        default="core",
        help="examples: examples only; core: midline modes; full: + FM/UQ/hard_x/kmax",
    )
    args = parser.parse_args()

    jobs: list[tuple[str, list[str]]] = []
    if args.suite in ("examples", "core", "full"):
        jobs.append(("examples", [_PY, "run_midline_sparse.py", "--mode", "examples"]))
    if args.suite in ("core", "full"):
        jobs.extend(
            [
                ("smoke", [_PY, "run_midline_sparse.py", "--mode", "smoke"]),
                ("gate2", [_PY, "run_midline_sparse.py", "--mode", "gate2"]),
                (
                    "merge_prune_probe",
                    [_PY, "run_midline_sparse.py", "--mode", "merge_prune_probe"],
                ),
                (
                    "stage2_curves",
                    [
                        _PY,
                        "run_stage2_training_curves.py",
                        "--problems",
                        "unimodal,A05,A01",
                    ],
                ),
            ]
        )
    if args.suite == "full":
        jobs.extend(
            [
                (
                    "k_max_ablation",
                    [
                        _PY,
                        "k_max_ablation_midline.py",
                        "--problems",
                        "all",
                        "--k-max",
                        "2,4,5,6,8",
                    ],
                ),
                (
                    "compare_foundation",
                    [
                        _PY,
                        "compare_foundation_baselines.py",
                        "--problems",
                        "all",
                        "--models",
                        "midline,exact_gp",
                    ],
                ),
                (
                    "beat_uq",
                    [
                        _PY,
                        "run_beat_foundation_uq.py",
                        "--skip-tabpfn",
                        "--skip-stress",
                        "--skip-memo",
                    ],
                ),
                (
                    "hard_xgate",
                    [_PY, "run_midline_hard_xgate.py", "--problems", "all"],
                ),
            ]
        )

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for label, cmd in jobs:
        results.append(_run(cmd, label=label))

    master_rows = [
        make_cost_row(
            script=r["label"],
            problem="(script total)",
            model="—",
            wall_s=r["wall_s"],
            notes="ok" if r["ok"] else f"exit={r['returncode']}",
        )
        for r in results
    ]
    # Pull per-fit cost CSVs if present
    cost_csv_globs = list((_THIS / "results").rglob("cost.csv")) + list(
        (_THIS / "results").rglob("*_cost.csv")
    )
    art = write_cost_artifacts(
        master_rows,
        out_dir=OUT,
        stem="script_totals",
        title=f"run_all_with_costs ({args.suite}): script wall totals",
    )
    summary = {
        "suite": args.suite,
        "jobs": results,
        "script_totals": art,
        "discovered_cost_csvs": [str(p) for p in sorted(cost_csv_globs)],
    }
    path = OUT / "run_all_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"{len(failed)} job(s) failed: {[r['label'] for r in failed]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
