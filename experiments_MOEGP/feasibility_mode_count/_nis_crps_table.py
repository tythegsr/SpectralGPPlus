"""Build NIS / CRPS / CRPS-exact tables and win counts for Section 10.6.1."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path("results/foundation_compare_tier4")
# Full set for win counts; display tables use DISPLAY.
MODELS = [
    "tabpfn35",
    "exact_gp",
    "midline_gamma_auto",
    "midline_gamma_auto_c85",
    "midline_gamma_auto_calibaccept",
    "midline_gamma_auto_sorf",
    "logistic_gp_sorf_map",
    "logistic_gp_sorf_vi",
    "logistic_gp_sorf_gamma",
    "quantile_gp_sorf",
    "quantile_gp_hybrid_auto",
    "quantile_gp_hybrid",
    "quantile_gp_hybrid_l01",
    "hist_mogp_t1",
]
DISPLAY = [
    "tabpfn35",
    "exact_gp",
    "midline_gamma_auto_c85",
    "midline_gamma_auto_sorf",
    "logistic_gp_sorf_map",
    "logistic_gp_sorf_vi",
    "quantile_gp_sorf",
    "quantile_gp_hybrid_auto",
    "hist_mogp_t1",
]
SHORT = {
    "tabpfn35": "TabPFN",
    "exact_gp": "exact GP",
    "midline_gamma_auto": "γ auto",
    "midline_gamma_auto_c85": "γ c85",
    "midline_gamma_auto_calibaccept": "γ calib",
    "midline_gamma_auto_sorf": "γ SORF",
    "logistic_gp_sorf_map": "log MAP",
    "logistic_gp_sorf_vi": "log VI",
    "logistic_gp_sorf_gamma": "log γ",
    "quantile_gp_sorf": "q-SORF",
    "quantile_gp_hybrid_auto": "q-auto",
    "quantile_gp_hybrid": "q-λ1",
    "quantile_gp_hybrid_l01": "q-λ0.1",
    "hist_mogp_t1": "hist t1",
}
BASE = {
    "A01": 100,
    "unimodal": 100,
    "A02": 100,
    "A04": 100,
    "A05": 100,
    "A06": 100,
    "QSAR_FISH": 726,
    "TABARENA_CONCRETE": 824,
}
PROB_LABEL = {"QSAR_FISH": "fish", "TABARENA_CONCRETE": "concrete"}

raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for path in root.rglob("*_metrics.json"):
    if "fresh" in path.parts:
        continue
    d = json.loads(path.read_text())
    proto = d.get("protocol", {})
    if int(proto.get("seed", -1)) not in (0, 1, 42):
        continue
    problem = proto.get("problem")
    if BASE.get(problem) != proto.get("n_train"):
        continue
    results = d.get("results", {})
    for row in d.get("rows", []):
        if row.get("source") == "error" or row.get("RMSE") is None:
            continue
        key = row["model_key"]
        if key not in MODELS:
            continue
        ex = (results.get(key) or {}).get("extras") or {}
        for mname, val in (
            ("NIS", row.get("NIS")),
            ("CRPS", row.get("CRPS")),
            ("CRPS_bar", ex.get("CRPS_bar")),
        ):
            if val is not None and np.isfinite(float(val)):
                raw[problem][key][mname].append(float(val))

means: dict = {}
for prob in BASE:
    means[prob] = {}
    for mk in MODELS:
        if mk not in raw[prob]:
            continue
        means[prob][mk] = {
            m: float(np.mean(raw[prob][mk][m]))
            for m in ("NIS", "CRPS", "CRPS_bar")
            if raw[prob][mk][m]
        }


def fmt(v: float) -> str:
    return f"{v:.3f}"


def plabel(prob: str) -> str:
    return PROB_LABEL.get(prob, prob)


def table_for(metric: str, models: list[str]) -> list[str]:
    out = []
    out.append("| problem | " + " | ".join(SHORT[m] for m in models) + " |")
    out.append("|---|" + "---|" * len(models))
    for prob in BASE:
        vals = {m: means[prob].get(m, {}).get(metric) for m in models}
        finite = {m: v for m, v in vals.items() if v is not None}
        best = min(finite.values()) if finite else None
        cells = []
        for m in models:
            v = vals[m]
            if v is None:
                cells.append("—")
            elif best is not None and abs(v - best) < 1e-12:
                cells.append(f"**{fmt(v)}**")
            else:
                cells.append(fmt(v))
        out.append(f"| {plabel(prob)} | " + " | ".join(cells) + " |")
    return out


def win_table(models: list[str], *, exclude: set[str] | None = None) -> list[str]:
    pool = [m for m in models if exclude is None or m not in exclude]
    wins = {m: {"NIS": 0, "CRPS": 0, "CRPS_bar": 0} for m in pool}
    shared = {m: {"NIS": 0, "CRPS": 0, "CRPS_bar": 0} for m in pool}
    for met in ("NIS", "CRPS", "CRPS_bar"):
        for prob in BASE:
            vals = {m: means[prob].get(m, {}).get(met) for m in pool}
            finite = {m: v for m, v in vals.items() if v is not None}
            if not finite:
                continue
            best = min(finite.values())
            winners = [m for m, v in finite.items() if abs(v - best) < 1e-12]
            if len(winners) == 1:
                wins[winners[0]][met] += 1
            else:
                for m in winners:
                    shared[m][met] += 1
    out = [
        "| model | NIS sole | CRPS sole | CRPS exact sole | NIS shared | CRPS shared | CRPS exact shared |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in pool:
        out.append(
            f"| {SHORT[m]} | {wins[m]['NIS']} | {wins[m]['CRPS']} | {wins[m]['CRPS_bar']} "
            f"| {shared[m]['NIS']} | {shared[m]['CRPS']} | {shared[m]['CRPS_bar']} |"
        )
    return out


lines: list[str] = []
lines.append("\n#### NIS (lower is better)\n")
lines.extend(table_for("NIS", DISPLAY))
lines.append("\n#### CRPS, ScoringBench / moment-summary (lower is better)\n")
lines.extend(table_for("CRPS", DISPLAY))
lines.append("\n#### CRPS exact (lower is better)\n")
lines.extend(table_for("CRPS_bar", DISPLAY))

lines.append("\n#### Win counts over 8 problems (display models, 3-seed means)\n")
lines.extend(win_table(DISPLAY))
lines.append(
    "\nSole = unique best; shared = tied for best. Bold cells in the tables above are the sole or shared best."
)

lines.append("\n#### Win counts excluding TabPFN (same display set)\n")
lines.extend(win_table(DISPLAY, exclude={"tabpfn35"}))

lines.append("\n#### Win counts, full Tier-4 model list (incl. ablations)\n")
lines.extend(win_table(MODELS))
lines.append("\n#### Full list excluding TabPFN\n")
lines.extend(win_table(MODELS, exclude={"tabpfn35"}))

out = Path("results/foundation_compare_tier4/_nis_crps_tables_gen.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
print("\n".join(lines))
