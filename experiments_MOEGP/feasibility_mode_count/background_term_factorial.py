"""Factorial ablation of MidlineSparse background terms on the lean winning stack.

Fixed base (post loss-term factorial winner U....A):
  usage ON, adaptive_merge ON, sep/rec/overlap/gamma OFF.

Factors (2^4 = 16 masks): orphan, alive, delta, alpha.

Example:
  conda activate gpplus2026
  python background_term_factorial.py --problems A05 --factors orphan,alive
  python background_term_factorial.py --problems A01,A04,A05 --k-max 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
for p in (_ROOT, _EXP, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from compare_foundation_baselines import (
    ALL_PROBLEMS,
    COMPARE_COLS,
    load_problem,
)
from density_match import probe_density_match
from foundation_adapters import fit_predict_midline_sparse, fit_predict_tabpfn
from plot_foundation_marginals import probe_x_values, slice_pred
from scoringbench_metrics import compute_scoringbench_style_metrics, print_metrics_table

OUT_DIR = _THIS / "results" / "background_term_factorial"
FIG_DIR = OUT_DIR / "figures"

# Winning lean stack magnitudes (structure terms forced off).
WINNING_BASE: dict[str, float] = {
    "lambda_k": 0.01,
    "lambda_k_start": 0.0,
    "lambda_sep": 0.0,
    "lambda_rec": 0.0,
    "lambda_overlap": 0.0,
    "lambda_gamma": 0.0,
}

# Background champion magnitudes when factor is ON.
BG_CHAMPION: dict[str, float] = {
    "lambda_orphan": 0.15,
    "alive_weight": 0.75,
    "lambda_delta": 0.01,
    "lambda_alpha": 0.05,
}

FACTORS = ("orphan", "alive", "delta", "alpha")

W_KEYS = (
    "w_usage",
    "w_sep",
    "w_alpha",
    "w_orphan",
    "w_alive",
    "w_overlap",
    "w_beta",
    "w_delta",
    "w_gamma",
    "w_rec",
    "w_merge_pull",
    "w_mode",
)


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _safe_savefig(fig, out_path: Path, *, retries: int = 6) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to save {out_path}")


def _parse_problems(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() in ("all", "*"):
        return list(ALL_PROBLEMS)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in ALL_PROBLEMS]
    if bad:
        raise SystemExit(f"Unknown problems {bad}; allowed: {ALL_PROBLEMS}")
    return items


def _parse_kmax(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = int(tok)
        if k < 1:
            raise SystemExit(f"k_max must be >= 1, got {k}")
        vals.append(k)
    if not vals:
        raise SystemExit("Provide at least one k_max")
    seen: set[int] = set()
    out: list[int] = []
    for k in vals:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _parse_factors(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() in ("all", "*", ""):
        return list(FACTORS)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in FACTORS]
    if bad:
        raise SystemExit(f"Unknown factors {bad}; allowed: {FACTORS}")
    return [f for f in FACTORS if f in items]


def factor_masks(factors: list[str]) -> list[dict[str, bool]]:
    masks: list[dict[str, bool]] = []
    for bits in itertools.product([False, True], repeat=len(factors)):
        masks.append(dict(zip(factors, bits, strict=True)))
    return masks


def mask_to_overrides(mask: dict[str, bool]) -> dict[str, Any]:
    """Winning base + background on/off. adaptive_merge always on (not returned)."""
    ov: dict[str, Any] = dict(WINNING_BASE)
    # Factors not in this run stay at champion ON (background always considered)
    ov["lambda_orphan"] = (
        BG_CHAMPION["lambda_orphan"] if mask.get("orphan", True) else 0.0
    )
    ov["alive_weight"] = (
        BG_CHAMPION["alive_weight"] if mask.get("alive", True) else 0.0
    )
    ov["lambda_delta"] = (
        BG_CHAMPION["lambda_delta"] if mask.get("delta", True) else 0.0
    )
    ov["lambda_alpha"] = (
        BG_CHAMPION["lambda_alpha"] if mask.get("alpha", True) else 0.0
    )
    return ov


def mask_label(mask: dict[str, bool], factors: list[str]) -> str:
    on = [f for f in factors if mask.get(f)]
    if not on:
        return "bg_none"
    if all(mask.get(f) for f in factors):
        return "bg_all"
    return "+".join(on)


def _frac(w: float, total: float) -> float:
    if not np.isfinite(total) or abs(total) < 1e-12:
        return float("nan")
    return float(w) / float(total)


def run_midline_mask(
    problem: dict[str, Any],
    *,
    k_max: int,
    mask: dict[str, bool],
    factors: list[str],
    seed: int,
    x_probe: np.ndarray,
    num_inits: int,
    num_stage2_inits: int,
) -> dict[str, Any]:
    x_tr, y_tr = problem["x_tr"], problem["y_tr"]
    x_te, y_te = problem["x_te"], problem["y_te"]
    Xte = x_te.detach().cpu().numpy()
    y_np = y_te.detach().cpu().numpy().ravel()
    n_te = int(Xte.shape[0])
    d = int(Xte.shape[1])
    Xp = np.asarray(x_probe, dtype=np.float64).reshape(-1, d)
    X_eval = np.vstack([Xte, Xp])
    x_eval_t = torch.as_tensor(X_eval, dtype=torch.float64)

    overrides = mask_to_overrides(mask)

    pred_all = fit_predict_midline_sparse(
        x_tr,
        y_tr,
        x_eval_t,
        k_max=int(k_max),
        num_steps=int(problem["num_steps"]),
        seed=seed,
        champion=True,
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
        adaptive_merge_prune=True,
        merge_pressure_init=0.0,
        lambda_overrides=overrides,
        num_inits=int(num_inits),
        num_stage2_inits=int(num_stage2_inits),
    )
    pred = slice_pred(pred_all, 0, n_te)
    pred_probe = slice_pred(pred_all, n_te, n_te + int(Xp.shape[0]))
    metrics = compute_scoringbench_style_metrics(
        y_np,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
    )
    extras = dict(pred_all.get("extras") or {})
    terms = dict(extras.get("loss_terms_final") or {})

    label = mask_label(mask, factors)
    row: dict[str, Any] = {
        "Model": f"Midline[{label}]",
        "model_key": f"midline_{label}",
        "source": "midline",
        "k_max": int(k_max),
        "base_stack": "usage+adaptive_merge",
        **{f"f_{f}": int(bool(mask.get(f))) for f in factors},
        **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
    }
    for k in ("k_eff", "k_modes_mean", "k_modes_round", "k_hat", "wall_s", "merge_pressure"):
        if k in extras:
            row[k] = extras[k]

    final_loss = float(terms.get("loss", float("nan")))
    nll = float(terms.get("nll", float("nan")))
    row["final_loss"] = final_loss
    row["nll"] = nll
    w_sum = 0.0
    for wk in W_KEYS:
        wv = float(terms.get(wk, 0.0))
        row[wk] = wv
        row[f"frac_{wk[2:]}"] = _frac(wv, final_loss)
        w_sum += wv if np.isfinite(wv) else 0.0
    row["frac_nll"] = _frac(nll, final_loss)
    row["w_sum"] = w_sum
    row["loss_residual"] = (
        final_loss - nll - w_sum
        if np.isfinite(final_loss) and np.isfinite(nll)
        else float("nan")
    )

    dm = probe_density_match(
        problem_name=str(problem["name"]),
        x_probes=x_probe,
        pred=pred_probe,
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
        true_k_modes=int(problem["true_k_modes"]),
        k_modes_mean=extras.get("k_modes_mean"),
    )
    if dm:
        for c in ("hellinger_mean", "tv_mean", "abs_k_modes_err"):
            if c in dm:
                row[c] = dm[c]

    return {"row": row, "extras": extras, "terms": terms, "mask": mask}


def run_tabpfn35(
    problem: dict[str, Any],
    *,
    k_max: int,
    factors: list[str],
    seed: int,
    x_probe: np.ndarray,
    device: str | None,
) -> dict[str, Any]:
    x_tr = problem["x_tr"].detach().cpu().numpy()
    y_tr = problem["y_tr"].detach().cpu().numpy().ravel()
    x_te = problem["x_te"].detach().cpu().numpy()
    y_te = problem["y_te"].detach().cpu().numpy().ravel()
    d = int(x_te.shape[1])
    Xp = np.asarray(x_probe, dtype=np.float64).reshape(-1, d)
    X_eval = np.vstack([x_te, Xp])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    pred_all = fit_predict_tabpfn(
        x_tr, y_tr, X_eval, version="v3.5", seed=seed, device=device
    )
    wall = time.perf_counter() - t0
    n_te = int(x_te.shape[0])
    pred = slice_pred(pred_all, 0, n_te)
    pred_probe = slice_pred(pred_all, n_te, n_te + int(Xp.shape[0]))
    metrics = compute_scoringbench_style_metrics(
        y_te,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
    )
    row: dict[str, Any] = {
        "Model": "TabPFN-3.5",
        "model_key": "tabpfn35",
        "source": "tabpfn35",
        "k_max": int(k_max),
        "base_stack": "",
        **{f"f_{f}": "" for f in factors},
        **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
        "wall_s": wall,
        "final_loss": None,
        "nll": None,
    }
    for wk in W_KEYS:
        row[wk] = None
        row[f"frac_{wk[2:]}"] = None
    dm = probe_density_match(
        problem_name=str(problem["name"]),
        x_probes=x_probe,
        pred=pred_probe,
        y_min=float(problem["y_min"]),
        y_max=float(problem["y_max"]),
        true_k_modes=int(problem["true_k_modes"]),
        k_modes_mean=None,
    )
    if dm:
        for c in ("hellinger_mean", "tv_mean"):
            if c in dm:
                row[c] = dm[c]
    return {"row": row}


def save_csv(rows: list[dict], out_path: Path, factors: list[str]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["Model", "model_key", "source", "k_max", "base_stack"]
        + [f"f_{f}" for f in factors]
        + COMPARE_COLS
        + [
            "hellinger_mean",
            "tv_mean",
            "abs_k_modes_err",
            "k_eff",
            "k_modes_mean",
            "k_hat",
            "merge_pressure",
            "wall_s",
            "final_loss",
            "nll",
            "frac_nll",
            "w_sum",
            "loss_residual",
        ]
        + list(W_KEYS)
        + [f"frac_{wk[2:]}" for wk in W_KEYS]
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def save_table_png(
    rows: list[dict],
    out_path: Path,
    *,
    title: str,
    factors: list[str],
) -> Path:
    show_w = ["w_usage", "w_orphan", "w_alive", "w_delta", "w_alpha", "w_merge_pull"]
    col_labels = (
        ["Model"]
        + [f[0].upper() for f in factors]  # O A D A — use first letter; disambiguate
        + ["CRPS", "NIS", "Hell", "k_m", "loss", "nll"]
        + [c.replace("w_", "")[:4] for c in show_w]
    )
    # clearer factor headers
    short = {"orphan": "Orph", "alive": "Aliv", "delta": "Delt", "alpha": "Alph"}
    col_labels = (
        ["Model"]
        + [short.get(f, f[:4]) for f in factors]
        + ["CRPS", "NIS", "Hell", "k_m", "loss", "nll"]
        + [c.replace("w_", "")[:4] for c in show_w]
    )
    cell_text = []
    for row in rows:
        cells = [str(row.get("Model", ""))[:28]]
        for f in factors:
            v = row.get(f"f_{f}")
            cells.append("" if v == "" or v is None else str(int(v)))
        for c in ("CRPS", "NIS", "hellinger_mean", "k_modes_mean", "final_loss", "nll"):
            v = row.get(c)
            cells.append(
                "—"
                if v is None or (isinstance(v, float) and not np.isfinite(float(v)))
                else f"{float(v):.3f}"
            )
        for c in show_w:
            v = row.get(c)
            cells.append(
                "—"
                if v is None or (isinstance(v, float) and not np.isfinite(float(v)))
                else f"{float(v):.3f}"
            )
        cell_text.append(cells)

    fig_h = min(48.0, 1.6 + 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(15.0, fig_h), dpi=110)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
        if r > 0 and rows[r - 1].get("source") == "tabpfn35":
            cell.set_facecolor("#c6f6d5")
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_frac_stacked_bar(
    rows: list[dict],
    *,
    out_path: Path,
    title: str,
) -> Path | None:
    mid = [r for r in rows if r.get("source") == "midline" and r.get("final_loss")]
    if not mid:
        return None
    mid = sorted(
        mid,
        key=lambda r: float(r["CRPS"]) if r.get("CRPS") is not None else 1e9,
    )
    labels = [str(r.get("Model", ""))[:22] for r in mid]
    parts = ["frac_nll"] + [f"frac_{wk[2:]}" for wk in W_KEYS]
    part_labels = ["nll"] + [wk[2:] for wk in W_KEYS]
    data = np.zeros((len(parts), len(mid)))
    for j, r in enumerate(mid):
        for i, p in enumerate(parts):
            v = r.get(p)
            data[i, j] = float(v) if v is not None and np.isfinite(float(v)) else 0.0
    colsum = data.sum(axis=0)
    colsum = np.where(colsum > 1e-12, colsum, 1.0)
    data = data / colsum

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(mid)), 5.5), dpi=120)
    bottom = np.zeros(len(mid))
    cmap = plt.cm.tab20
    for i, name in enumerate(part_labels):
        ax.bar(labels, data[i], bottom=bottom, label=name, color=cmap(i % 20), width=0.85)
        bottom = bottom + data[i]
    ax.set_ylabel("fraction of final_loss")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def run_problem_kmax(
    problem_name: str,
    *,
    k_max: int,
    factors: list[str],
    n_samples: int,
    test_size: float,
    seed: int,
    device: str | None,
    num_inits: int,
    num_stage2_inits: int,
    n_probes: int,
) -> dict[str, Any]:
    problem = load_problem(
        problem_name, n_samples=n_samples, test_size=test_size, seed=seed
    )
    x_tr_np = problem["x_tr"].detach().cpu().numpy()
    x_te_np = problem["x_te"].detach().cpu().numpy()
    x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=int(n_probes))

    masks = factor_masks(factors)
    rows: list[dict] = []
    results: list[dict] = []

    print(
        f"\n=== {problem_name} k_max={k_max}: base=usage+merge, "
        f"{len(masks)} background masks + TabPFN-3.5 ===",
        flush=True,
    )
    for i, mask in enumerate(masks):
        label = mask_label(mask, factors)
        print(f"  [{i + 1}/{len(masks)}] Midline[{label}] ...", flush=True)
        try:
            out = run_midline_mask(
                problem,
                k_max=k_max,
                mask=mask,
                factors=factors,
                seed=seed,
                x_probe=x_probes,
                num_inits=num_inits,
                num_stage2_inits=num_stage2_inits,
            )
            rows.append(out["row"])
            results.append(
                {
                    "mask": mask,
                    "label": label,
                    "row": out["row"],
                    "terms": out["terms"],
                }
            )
            r = out["row"]
            print(
                f"    CRPS={r.get('CRPS'):.4f} NIS={r.get('NIS'):.4f} "
                f"Hell={r.get('hellinger_mean', float('nan')):.4f} "
                f"loss={r.get('final_loss'):.4f} resid={r.get('loss_residual'):.4g} "
                f"k_hat={r.get('k_hat')} merge_p={r.get('merge_pressure')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}", flush=True)
            rows.append(
                {
                    "Model": f"Midline[{label}]",
                    "model_key": f"midline_{label}",
                    "source": "error",
                    "k_max": k_max,
                    "base_stack": "usage+adaptive_merge",
                    **{f"f_{f}": int(bool(mask.get(f))) for f in factors},
                    "error": str(exc),
                    **{c: None for c in COMPARE_COLS},
                }
            )

    print("  TabPFN-3.5 ...", flush=True)
    try:
        tab = run_tabpfn35(
            problem,
            k_max=k_max,
            factors=factors,
            seed=seed,
            x_probe=x_probes,
            device=device,
        )
        rows.append(tab["row"])
        print(
            f"    CRPS={tab['row'].get('CRPS'):.4f} NIS={tab['row'].get('NIS'):.4f} "
            f"Hell={tab['row'].get('hellinger_mean', float('nan')):.4f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    TabPFN FAILED: {exc}", flush=True)
        rows.append(
            {
                "Model": "TabPFN-3.5",
                "model_key": "tabpfn35",
                "source": "error",
                "k_max": k_max,
                "error": str(exc),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{problem_name.lower()}_kmax{k_max}"
    csv_path = OUT_DIR / f"{stem}_metrics.csv"
    json_path = OUT_DIR / f"{stem}_metrics.json"
    png_path = FIG_DIR / f"{stem}_metrics_table.png"
    bar_path = FIG_DIR / f"{stem}_loss_frac_bars.png"

    save_csv(rows, csv_path, factors)
    save_table_png(
        rows,
        png_path,
        title=(
            f"{problem_name} background factorial — base usage+merge, "
            f"k_max={k_max}, n={n_samples}"
        ),
        factors=factors,
    )
    save_frac_stacked_bar(
        rows,
        out_path=bar_path,
        title=f"{problem_name} k_max={k_max}: loss fraction (sorted by CRPS)",
    )

    payload = {
        "protocol": {
            "problem": problem_name,
            "k_max": k_max,
            "factors": factors,
            "n_masks": len(masks),
            "n_samples": n_samples,
            "test_size": test_size,
            "seed": seed,
            "num_inits": num_inits,
            "num_stage2_inits": num_stage2_inits,
            "true_k_modes": problem["true_k_modes"],
            "fixed_base": {
                "usage": True,
                "adaptive_merge": True,
                "sep": False,
                "rec": False,
                "overlap": False,
                "gamma": False,
                "overrides": WINNING_BASE,
            },
            "bg_champion": BG_CHAMPION,
        },
        "rows": rows,
        "results": results,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "table_png": str(png_path),
            "frac_bars_png": str(bar_path),
        },
    }
    _json_dump(json_path, payload)
    print_metrics_table(
        rows,
        cols=[
            "Model",
            "CRPS",
            "NIS",
            "log_score",
            "hellinger_mean",
            "final_loss",
            "nll",
            "k_hat",
        ],
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="A01,A04,A05")
    parser.add_argument("--k-max", default="5", dest="k_max")
    parser.add_argument(
        "--factors",
        default="all",
        help=f"Comma list or all. Allowed: {', '.join(FACTORS)}",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-probes", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=1)
    parser.add_argument("--num-stage2-inits", type=int, default=1)
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    problems = _parse_problems(args.problems)
    kmax_list = _parse_kmax(args.k_max)
    factors = _parse_factors(args.factors)

    all_payloads: dict[str, Any] = {}
    for pname in problems:
        for k_max in kmax_list:
            key = f"{pname}_kmax{k_max}"
            all_payloads[key] = run_problem_kmax(
                pname,
                k_max=k_max,
                factors=factors,
                n_samples=args.n_samples,
                test_size=args.test_size,
                seed=args.seed,
                device=args.device,
                num_inits=args.num_inits,
                num_stage2_inits=args.num_stage2_inits,
                n_probes=args.n_probes,
            )

    summary_path = OUT_DIR / "summary.json"
    _json_dump(
        summary_path,
        {
            "problems": problems,
            "k_max": kmax_list,
            "factors": factors,
            "fixed_base": "usage+adaptive_merge (sep/rec/overlap/gamma off)",
            "n_midline_fits": len(problems)
            * len(kmax_list)
            * (2 ** len(factors)),
            "payload_keys": list(all_payloads.keys()),
            "winning_base": WINNING_BASE,
            "bg_champion": BG_CHAMPION,
        },
    )
    print(f"Wrote {summary_path}")
    return all_payloads


if __name__ == "__main__":
    main()
