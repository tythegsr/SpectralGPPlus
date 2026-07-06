"""Build summary tables from Ackley RFF experiment JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ackley_40D_rff"
DEFAULT_METRICS = ("RRMSE", "RMSE", "NIS", "Time")
METRIC_KEYS = {"Time": "Total_Time"}
METRIC_HEADERS_MD = {"Time": "Time (s)"}
METRIC_HEADERS_TEX = {"Time": "Time (s)"}


def load_ackley_records(results_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(results_dir.glob("gp_*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        dims = int(data["dimensions"])
        n_train = int(data["n_train"])
        records.append(
            {
                "dimensions": dims,
                "train_size": n_train // dims,
                "n_train": n_train,
                "num_rff": int(data["num_rff"]),
                "noise_test": float(data.get("noise_test", 0.0)),
                "RRMSE": float(data["RRMSE"]),
                "RMSE": float(data["RMSE"]),
                "NIS": float(data["NIS"]),
                "Total_Time": float(data["Total_Time"]),
                "source_file": str(path),
            }
        )
    records.sort(key=lambda r: (r["dimensions"], r["train_size"], r["num_rff"]))
    return records


def _metric_value(record: dict, metric: str) -> float:
    key = METRIC_KEYS.get(metric, metric)
    return float(record[key])


def format_metric(name: str, value: float) -> str:
    if name == "Time":
        return f"{value:.1f}"
    if name == "NIS":
        return f"{value:.3f}"
    return f"{value:.4f}"


def markdown_table(records: list[dict], metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    config_headers = ["$D_x$", "$N$", "$D$"]
    metric_headers = [METRIC_HEADERS_MD.get(m, m) for m in metrics]
    header = [*config_headers, *metric_headers]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in records:
        cells = [
            str(r["dimensions"]),
            str(r["train_size"]),
            str(r["num_rff"]),
            *(format_metric(m, _metric_value(r, m)) for m in metrics),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(records: list[dict], metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    col_spec = "rrr" + "r" * len(metrics)
    config_headers = ["$D_x$", "$N$", "$D$"]
    metric_headers = [METRIC_HEADERS_TEX.get(m, m) for m in metrics]
    header = [*config_headers, *metric_headers]
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Ackley RFF benchmark results (noise test/train = 0.005).}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in records:
        cells = [
            str(r["dimensions"]),
            str(r["train_size"]),
            str(r["num_rff"]),
            *(format_metric(m, _metric_value(r, m)) for m in metrics),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ackley RFF JSON results as tables")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing gp_Ackley_*.json (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "both"),
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write table output",
    )
    args = parser.parse_args()

    records = load_ackley_records(args.results_dir)
    if not records:
        raise SystemExit(f"No gp_*.json files found under {args.results_dir}")

    parts: list[str] = []
    if args.format in ("markdown", "both"):
        parts.append("## Ackley RFF results\n")
        parts.append(markdown_table(records))
    if args.format in ("latex", "both"):
        if parts:
            parts.append("")
        parts.append(latex_table(records))

    text = "\n".join(parts)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
