"""A01 ScoringBench comparison: MidlineSparse vs live foundation models (+ MoE-GP).

Produces:
  - metrics table (RMSE, RRMSE, NIS, CRPS) as JSON/CSV/PNG
  - CDF comparison at x=5 (theory / empirical / MidlineSparse / MoE-GP)

Live foundation rows (TabPFN 3.0/3.5, TabGP, NVIDIA KumoTabular, exact GP+) come from
``compare_foundation_baselines``. Optional historical TabICL/Crepes website rows
remain available via ``--website-baselines``.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

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
from gpplus.models import MidlineSparseMoEGPR

from load_bimodal_data import (
    BIMODAL_C1,
    BIMODAL_C2,
    BIMODAL_M1,
    BIMODAL_M2,
    BIMODAL_NOISE_STD,
    train_test_split_bimodal,
)
from plot_cdf_ks import (
    empirical_cdf_and_ks,
    ks_distance_to_theory,
    mixture_gaussian_cdf,
    theoretical_bimodal_cdf,
)
from scipy.stats import norm
from scoringbench_metrics import (
    SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
    compute_scoringbench_style_metrics,
    fill_baseline_rrmse,
    print_metrics_table,
)

try:
    from compare_foundation_baselines import (
        ALL_MODELS as _FOUNDATION_MODELS,
        run_problem as _run_foundation_problem,
    )
except ImportError:  # pragma: no cover
    _FOUNDATION_MODELS = ()
    _run_foundation_problem = None

OUT_DIR = _THIS / "results"
FIG_DIR = OUT_DIR / "figures"
MOE_NPZ = (
    _EXP
    / "results"
    / "bimodal_moe_gp"
    / "predictions_1d_n1000_K2_soft_mlp_residual_seed42_em5.npz"
)
MOE_JSON = _EXP / "results" / "bimodal_moe_gp" / "gp_bimodal_moe_K2_soft_seed42.json"

COMPARE_COLS = ["RMSE", "RRMSE", "NIS", "CRPS"]


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

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
            fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to save {out_path}")


def _mixture_mean(em: np.ndarray, gw: np.ndarray) -> np.ndarray:
    return (np.asarray(em) * np.asarray(gw)).sum(axis=-1)


def fit_midline_sparse(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    k_max: int = 2,
    num_steps: int = 900,
    seed: int = 42,
) -> MidlineSparseMoEGPR:
    model = MidlineSparseMoEGPR(
        x_tr,
        y_tr,
        k_max=k_max,
        gate_type="constant",
        offset_type="mlp",
        keff_tau=0.02,
    )
    model.fit(
        num_inits=2,
        num_epochs=1,
        num_steps=num_steps,
        lr=5e-2,
        lambda_k=0.02 if k_max == 2 else 0.08,
        lambda_sep=0.05,
        lambda_alpha=0.05,
        lambda_orphan=0.05,
        seed=seed,
        device="cpu",
        n_jobs=1,
    )
    return model


def metrics_from_predict(
    model: MidlineSparseMoEGPR,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pred = model.predict(x_te)
    em = pred["expert_means"].detach().cpu().numpy()
    es = pred["expert_stds"].detach().cpu().numpy()
    gw = pred["gate_weights"].detach().cpu().numpy()
    # Effective component weights include Hard-Concrete α
    alpha = pred["alpha"].detach().cpu().numpy()
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, gw.shape)
    mass = np.clip(gw * alpha, 0.0, None)
    mass = mass / np.clip(mass.sum(axis=-1, keepdims=True), 1e-12, None)
    y_np = y_te.detach().cpu().numpy().ravel()
    mix = _mixture_mean(em, mass)
    metrics = compute_scoringbench_style_metrics(y_np, mix, em, es, mass)
    arrays = {"expert_means": em, "expert_stds": es, "gate_weights": mass, "mix_mean": mix}
    return metrics, arrays


def load_moe_gp_metrics_and_arrays(
    y_test: np.ndarray,
) -> tuple[dict[str, float] | None, dict[str, np.ndarray] | None]:
    """Load soft MoE-GP metrics; recompute NIS from NPZ if present."""
    metrics: dict[str, float] | None = None
    if MOE_JSON.exists():
        raw = json.loads(MOE_JSON.read_text(encoding="utf-8"))
        metrics = {
            k: float(raw[k])
            for k in ("RMSE", "RRMSE", "CRPS", "MSE", "MAE", "log_score")
            if k in raw and raw[k] is not None
        }
    arrays = None
    if MOE_NPZ.exists():
        data = np.load(MOE_NPZ, allow_pickle=False)
        # Prefer stored test predictions if present; else nearest grid samples
        if "mix_mean_test" in data.files and data["mix_mean_test"].shape[0] == y_test.shape[0]:
            # Reconstruct from saved train/test? NPZ may not store expert test arrays.
            # Fall back to grid interpolation at test x if available.
            pass
        if "x_grid" in data.files and "expert_means" in data.files:
            x_grid = data["x_grid"].ravel()
            em_g = data["expert_means"]
            es_g = data["expert_stds"]
            gw_g = data["gate_weights"]
            # Need test x — regenerate same split
            split = train_test_split_bimodal(n_samples=1000, test_size=0.2, seed=42)
            x_te = split["X_test"].ravel()
            idx = np.array([int(np.argmin(np.abs(x_grid - xv))) for xv in x_te])
            em = em_g[idx]
            es = es_g[idx]
            gw = gw_g[idx]
            mix = _mixture_mean(em, gw)
            recomputed = compute_scoringbench_style_metrics(y_test, mix, em, es, gw)
            if metrics is None:
                metrics = recomputed
            else:
                metrics["NIS"] = recomputed["NIS"]
                # Prefer recomputed CRPS/RRMSE on this split for consistency
                for k in ("RMSE", "RRMSE", "CRPS", "MSE", "MAE"):
                    metrics[k] = recomputed[k]
            arrays = {
                "expert_means": em,
                "expert_stds": es,
                "gate_weights": gw,
                "mix_mean": mix,
                "x_test": x_te,
            }
    return metrics, arrays


def build_comparison_rows(
    *,
    y_test: np.ndarray,
    midline_metrics: dict[str, float],
    moe_metrics: dict[str, float] | None,
    foundation_rows: list[dict] | None = None,
    include_hardcoded_tabpfn: bool = False,
    include_website_baselines: bool = True,
) -> list[dict]:
    """Build table rows.

    Prefer ``foundation_rows`` (live TabPFN/TabGP/Kumo/exact GP+/MidlineSparse).
    When those are present, hardcoded TabPFN is omitted unless
    ``include_hardcoded_tabpfn`` is True. TabICL/Crepes website rows remain
    optional historical reference.
    """
    rows: list[dict] = []
    if foundation_rows:
        for fr in foundation_rows:
            row = {
                "Model": fr.get("Model"),
                "source": fr.get("source", "live"),
            }
            for c in COMPARE_COLS:
                v = fr.get(c)
                row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
            rows.append(row)
        if include_website_baselines:
            hist = fill_baseline_rrmse(
                SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES, y_test=y_test
            )
            for name in ("TabICL", "Crepes(TabICL)"):
                base = hist.get(name)
                if not base:
                    continue
                row = {
                    "Model": name,
                    "source": "ScoringBench website / Comparison.ipynb",
                }
                for c in COMPARE_COLS:
                    v = base.get(c)
                    row[c] = (
                        float(v) if v is not None and np.isfinite(float(v)) else None
                    )
                rows.append(row)
    else:
        baselines = fill_baseline_rrmse(
            SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES, y_test=y_test
        )
        for name, base in baselines.items():
            if name == "TabPFN" and not include_hardcoded_tabpfn:
                continue
            row = {
                "Model": name,
                "source": "ScoringBench website / Comparison.ipynb",
            }
            for c in COMPARE_COLS:
                v = base.get(c)
                row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
            rows.append(row)
        row = {
            "Model": "MidlineSparseMoE-GP",
            "source": "this run (constant gate, MLP offsets)",
        }
        for c in COMPARE_COLS:
            v = midline_metrics.get(c)
            row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
        rows.append(row)

    if moe_metrics is not None:
        row = {
            "Model": "MoE-GP (soft, K=2)",
            "source": "experiments_MOEGP soft baseline",
        }
        for c in COMPARE_COLS:
            v = moe_metrics.get(c)
            row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
        rows.append(row)
    return rows


def save_metrics_table_png(rows: list[dict], out_path: Path) -> Path:
    """Render a clean comparison table figure."""
    col_labels = ["Model"] + COMPARE_COLS
    cell_text = []
    for row in rows:
        cells = [str(row["Model"])]
        for c in COMPARE_COLS:
            v = row.get(c)
            cells.append("—" if v is None or not np.isfinite(float(v)) else f"{float(v):.4f}")
        cell_text.append(cells)

    fig, ax = plt.subplots(figsize=(10.5, 2.2 + 0.35 * len(rows)), dpi=140)
    ax.axis("off")
    ax.set_title(
        "A01 ScoringBench bimodal — RMSE / RRMSE / NIS / CRPS\n"
        "(live foundation models + MidlineSparse; optional website TabICL/Crepes)",
        fontsize=11,
        pad=12,
    )
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.45)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):  # MidlineSparse is last
            cell.set_facecolor("#ebf8ff")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    return _safe_savefig(fig, out_path)


def save_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "source"] + COMPARE_COLS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def components_at_x(
    x_test: np.ndarray,
    em: np.ndarray,
    es: np.ndarray,
    gw: np.ndarray,
    x_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest test point components at ``x_value``."""
    i = int(np.argmin(np.abs(np.asarray(x_test).ravel() - x_value)))
    return em[i], es[i], gw[i]


def _mixture_pdf(y_support: np.ndarray, means, stds, weights) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.asarray(weights, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    dens = np.zeros_like(y_support, dtype=np.float64)
    for j in range(means.shape[0]):
        dens += pi[j] * norm.pdf(y_support, means[j], max(float(stds[j]), 1e-6))
    return dens


def _theoretical_pdf(y_support: np.ndarray, x_value: float) -> np.ndarray:
    m1 = BIMODAL_M1 * x_value + BIMODAL_C1
    m2 = BIMODAL_M2 * x_value + BIMODAL_C2
    return 0.5 * norm.pdf(y_support, m1, BIMODAL_NOISE_STD) + 0.5 * norm.pdf(
        y_support, m2, BIMODAL_NOISE_STD
    )


def save_cdf_comparison(
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    midline_arrays: dict[str, np.ndarray],
    moe_arrays: dict[str, np.ndarray] | None,
    out_path: Path,
    x_value: float = 5.0,
    x_tol: float = 0.5,
) -> tuple[Path, dict[str, float]]:
    y_support = np.linspace(-2.0, 30.0, 600)
    theo = theoretical_bimodal_cdf(y_support, x_value, noise_std=BIMODAL_NOISE_STD)
    mask = np.abs(np.asarray(x_test).ravel() - x_value) <= x_tol
    y_near = y_test[mask] if mask.any() else y_test
    _, ecdf_vals, ecdf_ks = empirical_cdf_and_ks(y_near, y_support, theo)
    ecdf_y = np.sort(np.asarray(y_near, dtype=np.float64).ravel())

    em_m, es_m, gw_m = components_at_x(
        x_test,
        midline_arrays["expert_means"],
        midline_arrays["expert_stds"],
        midline_arrays["gate_weights"],
        x_value,
    )
    mid_cdf = mixture_gaussian_cdf(y_support, em_m, es_m, gw_m)
    mid_ks = ks_distance_to_theory(y_support, mid_cdf, theo)

    ks: dict[str, float] = {"Empirical": ecdf_ks, "MidlineSparseMoE-GP": mid_ks}

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    ax.plot(y_support, theo, "r-", lw=2.5, label="Theoretical CDF", alpha=0.9, zorder=3)
    if ecdf_y.size:
        ax.step(
            ecdf_y,
            ecdf_vals,
            where="post",
            color="0.45",
            lw=1.7,
            label=f"Empirical (KS={ecdf_ks:.3f})",
            alpha=0.8,
            zorder=2,
        )
    ax.plot(
        y_support,
        mid_cdf,
        color="#2b6cb0",
        lw=2.3,
        label=f"MidlineSparseMoE-GP (KS={mid_ks:.3f})",
        zorder=5,
    )

    if moe_arrays is not None:
        x_moe = moe_arrays.get("x_test", x_test)
        em_o, es_o, gw_o = components_at_x(
            x_moe,
            moe_arrays["expert_means"],
            moe_arrays["expert_stds"],
            moe_arrays["gate_weights"],
            x_value,
        )
        moe_cdf = mixture_gaussian_cdf(y_support, em_o, es_o, gw_o)
        moe_ks = ks_distance_to_theory(y_support, moe_cdf, theo)
        ks["MoE-GP (soft)"] = moe_ks
        ax.plot(
            y_support,
            moe_cdf,
            color="#c05621",
            lw=2.0,
            ls="--",
            label=f"MoE-GP soft (KS={moe_ks:.3f})",
            zorder=4,
        )

    ax.set_xlabel("response y")
    ax.set_ylabel("CDF")
    ax.set_title(f"A01 conditional CDF comparison at x = {x_value:g}")
    ax.set_xlim(-2.0, 30.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path, ks


def save_pdf_comparison(
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    midline_arrays: dict[str, np.ndarray],
    moe_arrays: dict[str, np.ndarray] | None,
    out_path: Path,
    x_value: float = 9.0,
    x_tol: float = 0.5,
) -> Path:
    """Conditional densities p(y|x) at ``x_value``: theory vs MidlineSparse vs MoE-GP."""
    y_support = np.linspace(-2.0, 30.0, 600)
    theo = _theoretical_pdf(y_support, x_value)

    em_m, es_m, gw_m = components_at_x(
        x_test,
        midline_arrays["expert_means"],
        midline_arrays["expert_stds"],
        midline_arrays["gate_weights"],
        x_value,
    )
    mid_pdf = _mixture_pdf(y_support, em_m, es_m, gw_m)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    ax.plot(y_support, theo, "r-", lw=2.5, label="Theoretical p(y|x)", alpha=0.9, zorder=3)
    ax.plot(
        y_support,
        mid_pdf,
        color="#2b6cb0",
        lw=2.3,
        label="MidlineSparseMoE-GP",
        zorder=5,
    )

    # Component curves for MidlineSparse
    pi = np.clip(np.asarray(gw_m, dtype=np.float64).ravel(), 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    colors = ("#63b3ed", "#f6ad55")
    for k in range(len(pi)):
        if pi[k] < 0.02:
            continue
        comp = pi[k] * norm.pdf(
            y_support, float(em_m[k]), max(float(es_m[k]), 1e-6)
        )
        ax.plot(
            y_support,
            comp,
            color=colors[k % len(colors)],
            lw=1.4,
            ls=":",
            alpha=0.85,
            label=rf"MidlineSparse $k={k}$ ($\pi$={pi[k]:.2f})",
            zorder=4,
        )

    if moe_arrays is not None:
        x_moe = moe_arrays.get("x_test", x_test)
        em_o, es_o, gw_o = components_at_x(
            x_moe,
            moe_arrays["expert_means"],
            moe_arrays["expert_stds"],
            moe_arrays["gate_weights"],
            x_value,
        )
        moe_pdf = _mixture_pdf(y_support, em_o, es_o, gw_o)
        ax.plot(
            y_support,
            moe_pdf,
            color="#c05621",
            lw=2.0,
            ls="--",
            label="MoE-GP soft",
            zorder=4,
        )

    mask = np.abs(np.asarray(x_test).ravel() - x_value) <= x_tol
    y_near = y_test[mask] if mask.any() else np.array([])
    if y_near.size:
        ax.scatter(
            y_near,
            np.full_like(y_near, -0.002),
            s=18,
            c="0.35",
            alpha=0.45,
            marker="|",
            label=f"test y near x={x_value:g} (n={y_near.size})",
            zorder=2,
        )

    ax.set_xlabel("response y")
    ax.set_ylabel(rf"$p(y \mid x={x_value:g})$")
    ax.set_title(f"A01 conditional density comparison at x = {x_value:g}")
    ax.set_xlim(-2.0, 30.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> dict:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Foundation models to run live (default: all). Pass 'midline' for MidlineSparse-only.",
    )
    parser.add_argument(
        "--website-baselines",
        action="store_true",
        help="Include historical TabICL/Crepes website rows",
    )
    parser.add_argument(
        "--skip-foundation",
        action="store_true",
        help="Skip live foundation runners; fit MidlineSparse only (legacy path)",
    )
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Match A01 / ScoringBench MoE protocol
    data = train_test_split_bimodal(n_samples=1000, test_size=0.2, seed=42)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    x_te = torch.as_tensor(data["X_test"], dtype=torch.float64)
    y_te = torch.as_tensor(data["y_test"], dtype=torch.float64)
    y_test_np = data["y_test"].ravel()

    print("Fitting MidlineSparseMoE-GP on A01 (CDF/PDF + metrics)...")
    t0 = time.perf_counter()
    model = fit_midline_sparse(x_tr, y_tr, k_max=2, num_steps=900, seed=42)
    mid_metrics, mid_arrays = metrics_from_predict(model, x_te, y_te)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(
        f"  MidlineSparse: RMSE={mid_metrics['RMSE']:.4f} "
        f"RRMSE={mid_metrics['RRMSE']:.4f} "
        f"NIS={mid_metrics['NIS']:.4f} "
        f"CRPS={mid_metrics['CRPS']:.4f}"
    )

    foundation_rows: list[dict] | None = None
    foundation_payload: dict | None = None
    if (
        not args.skip_foundation
        and _run_foundation_problem is not None
        and _FOUNDATION_MODELS
    ):
        if args.models.strip().lower() in ("all", "*"):
            model_list = [m for m in _FOUNDATION_MODELS if m != "midline"]
        else:
            model_list = [
                m.strip()
                for m in args.models.split(",")
                if m.strip() and m.strip() != "midline"
            ]
        if model_list:
            print(
                f"Running live foundation baselines on A01 "
                f"(models={','.join(model_list)}; MidlineSparse already fit)..."
            )
            foundation_payload = _run_foundation_problem(
                "A01",
                model_list,
                n_samples=1000,
                test_size=0.2,
                seed=42,
                include_website_baselines=False,
            )
            foundation_rows = list(foundation_payload.get("rows") or [])
            # Prepend MidlineSparse live metrics so it leads the table
            foundation_rows = [
                {
                    "Model": "MidlineSparseMoE-GP",
                    "model_key": "midline",
                    "source": "live",
                    **{c: mid_metrics.get(c) for c in COMPARE_COLS},
                },
                *foundation_rows,
            ]

    moe_metrics, moe_arrays = load_moe_gp_metrics_and_arrays(y_test_np)
    if moe_metrics:
        print(
            f"  MoE-GP soft: RMSE={moe_metrics.get('RMSE', float('nan')):.4f} "
            f"RRMSE={moe_metrics.get('RRMSE', float('nan')):.4f} "
            f"NIS={moe_metrics.get('NIS', float('nan')):.4f} "
            f"CRPS={moe_metrics.get('CRPS', float('nan')):.4f}"
        )

    rows = build_comparison_rows(
        y_test=y_test_np,
        midline_metrics=mid_metrics,
        moe_metrics=moe_metrics,
        foundation_rows=foundation_rows,
        include_website_baselines=args.website_baselines,
    )
    print_metrics_table(rows, cols=["Model"] + COMPARE_COLS)

    table_json = OUT_DIR / "a01_scoringbench_comparison.json"
    table_csv = OUT_DIR / "a01_scoringbench_comparison.csv"
    table_png = FIG_DIR / "a01_scoringbench_metrics_table.png"

    payload = {
        "protocol": {
            "n_samples": 1000,
            "test_size": 0.2,
            "seed": 42,
            "midline": {"k_max": 2, "gate": "constant", "offset": "mlp"},
            "note": (
                "Live foundation models via compare_foundation_baselines; "
                "hardcoded TabPFN website row replaced. Optional TabICL/Crepes "
                "website rows via --website-baselines."
            ),
        },
        "rows": rows,
        "midline_metrics": mid_metrics,
        "moe_metrics": moe_metrics,
        "foundation": foundation_payload,
    }
    _json_dump(table_json, payload)
    save_csv(rows, table_csv)
    save_metrics_table_png(rows, table_png)

    cdf_ks: dict[str, dict[str, float]] = {}
    cdf_paths: dict[str, str] = {}
    pdf_paths: dict[str, str] = {}
    for xv in (5.0, 9.0):
        tag = str(int(xv)) if abs(xv - round(xv)) < 1e-9 else f"{xv:g}"
        cdf_path, ks = save_cdf_comparison(
            x_test=data["X_test"],
            y_test=y_test_np,
            midline_arrays=mid_arrays,
            moe_arrays=moe_arrays,
            out_path=FIG_DIR / f"a01_cdf_x{tag}_comparison.png",
            x_value=xv,
        )
        pdf_path = save_pdf_comparison(
            x_test=data["X_test"],
            y_test=y_test_np,
            midline_arrays=mid_arrays,
            moe_arrays=moe_arrays,
            out_path=FIG_DIR / f"a01_pdf_x{tag}_comparison.png",
            x_value=xv,
        )
        cdf_ks[f"x={tag}"] = ks
        cdf_paths[f"x={tag}"] = str(cdf_path)
        pdf_paths[f"x={tag}"] = str(pdf_path)
        print(f"KS to theory at x={tag}:")
        for k, v in ks.items():
            print(f"  {k}: {v:.4f}")

    payload["cdf_ks"] = cdf_ks
    payload["artifacts"] = {
        "json": str(table_json),
        "csv": str(table_csv),
        "table_png": str(table_png),
        "cdf_pngs": cdf_paths,
        "pdf_pngs": pdf_paths,
    }
    _json_dump(table_json, payload)

    print(f"\nWrote {table_json}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_png}")
    for p in list(cdf_paths.values()) + list(pdf_paths.values()):
        print(f"Wrote {p}")
    return payload


if __name__ == "__main__":
    main()
