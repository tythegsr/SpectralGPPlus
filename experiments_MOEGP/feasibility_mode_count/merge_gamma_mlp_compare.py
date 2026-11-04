"""Merge midline_gamma_mlp results into the prior joint_uq suite and summarize."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOINT = ROOT / "results" / "foundation_compare_joint_uq"
GAMMA = ROOT / "results" / "foundation_compare_gamma_mlp"
OUT = ROOT / "results" / "foundation_compare_joint_plus_gamma_mlp"

COMPARE_COLS = ["RMSE", "RRMSE", "NIS", "CRPS", "log_score"]
EXTRA = ["cov50", "cov90", "wall_s", "hellinger_mean", "tv_mean", "abs_k_modes_err"]


def _load_metrics(folder: Path, stem: str) -> dict | None:
    p = folder / f"{stem}_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    stems = sorted(
        {p.name.replace("_metrics.json", "") for p in JOINT.glob("*_metrics.json")}
        | {p.name.replace("_metrics.json", "") for p in GAMMA.glob("*_metrics.json")}
    )
    stems = [s for s in stems if s and not s.endswith("_cost")]

    win_ls: dict[str, int] = defaultdict(int)
    win_crps: dict[str, int] = defaultdict(int)
    vs_midline = {"gamma_better_ls": 0, "const_better_ls": 0, "tie_ls": 0,
                  "gamma_better_crps": 0, "const_better_crps": 0, "tie_crps": 0}
    all_rows_summary: list[dict] = []
    payloads: dict = {}

    for stem in stems:
        joint = _load_metrics(JOINT, stem)
        gamma = _load_metrics(GAMMA, stem)
        if joint is None and gamma is None:
            continue

        rows_by_key: dict[str, dict] = {}
        results: dict = {}
        if joint:
            for r in joint.get("rows") or []:
                rows_by_key[r["model_key"]] = dict(r)
            results.update(joint.get("results") or {})
        if gamma:
            for r in gamma.get("rows") or []:
                if r.get("model_key") == "midline_gamma_mlp":
                    rows_by_key["midline_gamma_mlp"] = dict(r)
            gres = gamma.get("results") or {}
            if "midline_gamma_mlp" in gres:
                results["midline_gamma_mlp"] = gres["midline_gamma_mlp"]

        rows = list(rows_by_key.values())
        # Prefer a stable order
        order = [
            "midline",
            "midline_gamma_mlp",
            "mdn",
            "mdn_const",
            "exact_gp",
            "hist_mogp",
            "hist_gpd",
            "tabgp",
            "tabpfn35",
            "kumo",
            "tabpfn3",
        ]
        rows.sort(
            key=lambda r: order.index(r["model_key"])
            if r.get("model_key") in order
            else 99
        )

        finite_ls = [
            r for r in rows if r.get("log_score") is not None and r.get("source") != "error"
        ]
        if finite_ls:
            best = min(finite_ls, key=lambda r: float(r["log_score"]))
            win_ls[str(best["model_key"])] += 1
        finite_c = [
            r for r in rows if r.get("CRPS") is not None and r.get("source") != "error"
        ]
        if finite_c:
            best_c = min(finite_c, key=lambda r: float(r["CRPS"]))
            win_crps[str(best_c["model_key"])] += 1

        mc = rows_by_key.get("midline")
        mg = rows_by_key.get("midline_gamma_mlp")
        if mc and mg and mc.get("log_score") is not None and mg.get("log_score") is not None:
            d = float(mg["log_score"]) - float(mc["log_score"])
            if abs(d) < 1e-9:
                vs_midline["tie_ls"] += 1
            elif d < 0:
                vs_midline["gamma_better_ls"] += 1
            else:
                vs_midline["const_better_ls"] += 1
            dc = float(mg["CRPS"]) - float(mc["CRPS"])
            if abs(dc) < 1e-9:
                vs_midline["tie_crps"] += 1
            elif dc < 0:
                vs_midline["gamma_better_crps"] += 1
            else:
                vs_midline["const_better_crps"] += 1
            all_rows_summary.append(
                {
                    "setting": stem,
                    "midline_log": float(mc["log_score"]),
                    "gamma_mlp_log": float(mg["log_score"]),
                    "delta_log": d,
                    "midline_CRPS": float(mc["CRPS"]),
                    "gamma_mlp_CRPS": float(mg["CRPS"]),
                    "delta_CRPS": dc,
                    "best_log": min(finite_ls, key=lambda r: float(r["log_score"]))[
                        "model_key"
                    ]
                    if finite_ls
                    else None,
                    "best_CRPS": min(finite_c, key=lambda r: float(r["CRPS"]))[
                        "model_key"
                    ]
                    if finite_c
                    else None,
                }
            )

        # Write merged metrics
        payload = {
            "problem": (joint or gamma or {}).get("problem"),
            "protocol": (joint or gamma or {}).get("protocol"),
            "rows": rows,
            "results": results,
            "merged_from": {
                "joint_uq": str(JOINT),
                "gamma_mlp": str(GAMMA),
            },
        }
        out_json = OUT / f"{stem}_metrics.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_csv = OUT / f"{stem}_metrics.csv"
        fields = ["Model", "model_key", "source"] + COMPARE_COLS + EXTRA
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        payloads[stem] = {"rows": rows}

        # Console table
        print(f"\n==== {stem}")
        print(
            f"{'model':28} {'log':>8} {'CRPS':>8} {'RMSE':>8} {'wall':>8}"
        )
        for r in rows:
            print(
                f"{str(r.get('Model', r.get('model_key'))):28} "
                f"{r.get('log_score') if r.get('log_score') is not None else float('nan'):8.4f} "
                f"{r.get('CRPS') if r.get('CRPS') is not None else float('nan'):8.4f} "
                f"{r.get('RMSE') if r.get('RMSE') is not None else float('nan'):8.4f} "
                f"{r.get('wall_s') if r.get('wall_s') is not None else float('nan'):8.1f}"
            )

    summary = {
        "n_settings": len(payloads),
        "log_score_wins": dict(win_ls),
        "CRPS_wins": dict(win_crps),
        "midline_const_vs_gamma_mlp": vs_midline,
        "pairwise": all_rows_summary,
        "payloads": {
            k: {"rows": [{"model_key": r.get("model_key"), **{c: r.get(c) for c in COMPARE_COLS}} for r in v["rows"]]}
            for k, v in payloads.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pair_csv = OUT / "midline_const_vs_gamma_mlp.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(all_rows_summary[0].keys()) if all_rows_summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows_summary:
            w.writerow(row)

    print("\n==== WIN COUNTS (merged suite)")
    print("log_score:", dict(win_ls))
    print("CRPS:", dict(win_crps))
    print("const vs gamma_mlp:", vs_midline)
    print(f"Wrote {OUT / 'summary.json'}")
    print(f"Wrote {pair_csv}")


if __name__ == "__main__":
    main()
