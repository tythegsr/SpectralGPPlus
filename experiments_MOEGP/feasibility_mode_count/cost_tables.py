"""Wall-clock cost tables (CSV + PNG) for feasibility runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

COST_COLS = [
    "script",
    "problem",
    "model",
    "n_train",
    "k_max",
    "num_steps",
    "gate_type",
    "num_inits",
    "num_stage2_inits",
    "champion",
    "wall_s",
    "notes",
]


def make_cost_row(
    *,
    script: str,
    problem: str,
    model: str,
    wall_s: float | None,
    n_train: int | None = None,
    k_max: int | None = None,
    num_steps: int | None = None,
    gate_type: str | None = None,
    num_inits: int | None = None,
    num_stage2_inits: int | None = None,
    champion: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "script": script,
        "problem": problem,
        "model": model,
        "n_train": n_train,
        "k_max": k_max,
        "num_steps": num_steps,
        "gate_type": gate_type,
        "num_inits": num_inits,
        "num_stage2_inits": num_stage2_inits,
        "champion": champion,
        "wall_s": None if wall_s is None else float(wall_s),
        "notes": notes or "",
    }


def write_cost_csv(rows: Sequence[dict[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COST_COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c) for c in COST_COLS})
    return out_path


def write_cost_table_png(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Fit cost (wall-clock)",
) -> Path:
    """Render a compact cost table; highlight slowest row."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "no cost rows", ha="center", va="center")
        fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
        plt.close(fig)
        return out_path

    display_cols = [
        "problem",
        "model",
        "n_train",
        "k_max",
        "num_steps",
        "gate_type",
        "num_inits",
        "wall_s",
    ]
    walls = []
    for r in rows:
        v = r.get("wall_s")
        try:
            walls.append(float(v) if v is not None and np.isfinite(float(v)) else -1.0)
        except (TypeError, ValueError):
            walls.append(-1.0)
    slowest = int(np.argmax(walls)) if walls else -1

    cell_text = []
    for r in rows:
        cells = []
        for c in display_cols:
            v = r.get(c)
            if c == "wall_s":
                cells.append(
                    "—"
                    if v is None or not np.isfinite(float(v))
                    else f"{float(v):.1f}"
                )
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        cell_text.append(cells)

    fig_h = 1.6 + 0.36 * max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=10)
    table = ax.table(
        cellText=cell_text,
        colLabels=display_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r - 1 == slowest:
            cell.set_facecolor("#feebc8")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    total = sum(w for w in walls if w >= 0.0)
    fig.text(
        0.5,
        0.02,
        f"total wall ≈ {total:.1f}s  |  yellow = slowest row",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def write_cost_artifacts(
    rows: Sequence[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str = "cost",
    title: str | None = None,
) -> dict[str, str]:
    """Write CSV, PNG, and JSON cost artifacts under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_cost_csv(rows, out_dir / f"{stem}.csv")
    png_path = write_cost_table_png(
        rows,
        out_dir / f"{stem}.png",
        title=title or f"{stem}: fit wall-clock cost",
    )
    json_path = out_dir / f"{stem}.json"
    payload = {
        "rows": list(rows),
        "total_wall_s": float(
            sum(
                float(r["wall_s"])
                for r in rows
                if r.get("wall_s") is not None
                and np.isfinite(float(r["wall_s"]))
            )
        ),
        "n_rows": len(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "json": str(json_path),
        "total_wall_s": payload["total_wall_s"],
    }
