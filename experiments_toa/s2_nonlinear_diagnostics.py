"""Nonlinear dependence diagnostics: TOA reflectance bands vs 11 QoIs.

Compares Pearson, Spearman, mutual information, distance correlation, and
multivariate ridge/RF recoverability. Writes summary tables and heatmaps under
``experiments_toa/data 11 QoI/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_bands import default_keep_indices, load_task_band_config
from experiments_toa.s2_constants import (
    S2_DEFAULT_BAND_CONFIG_PATH,
    S2_DEFAULT_DATA_PATH,
    S2_TASK_NAMES,
)
from experiments_toa.s2_data import load_s2_arrays


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Sample distance correlation for 1-D x, y (same length)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 3:
        return float("nan")
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(A * B))
    dvar_x = float(np.mean(A * A))
    dvar_y = float(np.mean(B * B))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_x * dvar_y)))


def _ridge_r2(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    xm = x_tr.mean(0)
    xs = x_tr.std(0) + 1e-8
    z_tr = (x_tr - xm) / xs
    z_te = (x_te - xm) / xs
    lam = 1.0
    a = z_tr.T @ z_tr + lam * np.eye(z_tr.shape[1])
    b = z_tr.T @ (y_tr - y_tr.mean())
    w = np.linalg.solve(a, b)
    pred = z_te @ w + y_tr.mean()
    return float(r2_score(y_te, pred))


def _heatmap(
    mat: np.ndarray,
    *,
    wavelengths: np.ndarray,
    task_names: list[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    tick_idx = np.linspace(0, mat.shape[0] - 1, num=min(12, mat.shape[0]), dtype=int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{wavelengths[i]:.0f}" for i in tick_idx])
    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlabel("QoI")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(
    *,
    data_path: Path = S2_DEFAULT_DATA_PATH,
    band_config: Path = S2_DEFAULT_BAND_CONFIG_PATH,
    out_dir: Path | None = None,
    n_subsample: int = 12000,
    n_dcor: int = 2000,
    seed: int = 0,
) -> dict:
    if out_dir is None:
        out_dir = data_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(S2_TASK_NAMES)
    X_all, Y_all, wl, names, meta = load_s2_arrays(
        data_path,
        input_variable="toa_reflectance",
        task_names=task_names,
    )
    assert names == task_names

    # Shared band set for heatmaps: global default keep ranges.
    shared_bands = default_keep_indices()
    bands_by_task = load_task_band_config(band_config, task_names=task_names)

    rng = np.random.default_rng(seed)
    n = X_all.shape[0]
    idx = rng.choice(n, size=min(n_subsample, n), replace=False)
    idx_dcor = rng.choice(n, size=min(n_dcor, n), replace=False)

    X = X_all[idx][:, shared_bands]
    Y = Y_all[idx]
    X_d = X_all[idx_dcor][:, shared_bands]
    Y_d = Y_all[idx_dcor]
    wl_shared = wl[shared_bands]

    # Standardize bands for MI.
    x_std = (X - X.mean(0)) / (X.std(0) + 1e-8)

    n_bands = X.shape[1]
    n_tasks = len(task_names)
    pearson = np.zeros((n_bands, n_tasks), dtype=np.float64)
    spearman = np.zeros((n_bands, n_tasks), dtype=np.float64)
    mi = np.zeros((n_bands, n_tasks), dtype=np.float64)
    dcor = np.full((n_bands, n_tasks), np.nan, dtype=np.float64)

    # dCor is O(n^2) per pair: evaluate on a wavelength stride, then densify by
    # also scoring the top Pearson/MI bands for the per-QoI summary.
    dcor_stride = max(1, n_bands // 60)
    dcor_grid = list(range(0, n_bands, dcor_stride))

    print(
        f"Shared bands: {n_bands}; subsample={len(idx)}; "
        f"dCor subsample={len(idx_dcor)}; dCor grid={len(dcor_grid)}"
    )
    for t, name in enumerate(task_names):
        print(f"Bandwise stats: {name}")
        y = Y[:, t]
        y_c = y - y.mean()
        y_norm = np.linalg.norm(y_c)
        x_c = X - X.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(x_c, axis=0)
        if y_norm < 1e-12:
            pearson[:, t] = 0.0
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                pearson[:, t] = (x_c.T @ y_c) / (x_norm * y_norm + 1e-12)
            pearson[x_norm < 1e-12, t] = 0.0

        # Spearman via rank transform then Pearson.
        x_rank = np.apply_along_axis(lambda v: v.argsort().argsort().astype(np.float64), 0, X)
        y_rank = y.argsort().argsort().astype(np.float64)
        yr = y_rank - y_rank.mean()
        yr_norm = np.linalg.norm(yr)
        xr = x_rank - x_rank.mean(axis=0, keepdims=True)
        xr_norm = np.linalg.norm(xr, axis=0)
        if yr_norm < 1e-12:
            spearman[:, t] = 0.0
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                spearman[:, t] = (xr.T @ yr) / (xr_norm * yr_norm + 1e-12)
            spearman[xr_norm < 1e-12, t] = 0.0

        mi[:, t] = mutual_info_regression(
            x_std,
            y,
            random_state=seed,
            n_neighbors=5,
        )

        yd = Y_d[:, t]
        for j in dcor_grid:
            dcor[j, t] = _distance_correlation(X_d[:, j], yd)
        # Densify around strongest linear/MI bands for the summary max.
        extra = set(np.argsort(np.abs(pearson[:, t]))[::-1][:5].tolist())
        extra.update(np.argsort(mi[:, t])[::-1][:5].tolist())
        for j in extra:
            if np.isnan(dcor[j, t]):
                dcor[j, t] = _distance_correlation(X_d[:, j], yd)

    rows: list[dict] = []
    for t, name in enumerate(task_names):
        print(f"Multivariate models: {name}")
        task_bands = bands_by_task[name]
        Xt = X_all[idx][:, task_bands]
        yt = Y_all[idx][:, t]
        x_tr, x_te, y_tr, y_te = train_test_split(Xt, yt, test_size=0.25, random_state=seed)

        ridge = _ridge_r2(x_tr, y_tr, x_te, y_te)
        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(x_tr, y_tr)
        rf_r2 = float(r2_score(y_te, rf.predict(x_te)))

        # Permutation importance on a few top impurity features (fast).
        imp = rf.feature_importances_
        top_local = np.argsort(imp)[::-1][:10]
        perm_scores: list[dict] = []
        y_hat_base = rf.predict(x_te)
        base_r2 = float(r2_score(y_te, y_hat_base))
        for loc in top_local:
            x_perm = x_te.copy()
            x_perm[:, loc] = rng.permutation(x_perm[:, loc])
            drop = base_r2 - float(r2_score(y_te, rf.predict(x_perm)))
            band_idx = int(task_bands[int(loc)])
            perm_scores.append(
                {
                    "band_index": band_idx,
                    "wavelength_nm": float(wl[band_idx]),
                    "impurity_importance": float(imp[loc]),
                    "r2_drop": float(drop),
                }
            )

        max_abs_pearson = float(np.nanmax(np.abs(pearson[:, t])))
        max_abs_spearman = float(np.nanmax(np.abs(spearman[:, t])))
        max_mi = float(np.nanmax(mi[:, t]))
        max_dcor = float(np.nanmax(dcor[:, t]))
        i_p = int(np.nanargmax(np.abs(pearson[:, t])))
        i_s = int(np.nanargmax(np.abs(spearman[:, t])))
        i_m = int(np.nanargmax(mi[:, t]))
        i_d = int(np.nanargmax(dcor[:, t]))

        nonlinear_flag = bool(
            (rf_r2 - ridge) > 0.10
            or (max_mi > 0.05 and max_abs_pearson < 0.15)
            or (max_dcor > 0.20 and max_abs_pearson < 0.15)
        )

        row = {
            "qoi": name,
            "n_bands_shared": n_bands,
            "n_bands_task": len(task_bands),
            "max_abs_pearson": max_abs_pearson,
            "max_abs_pearson_band": int(shared_bands[i_p]),
            "max_abs_pearson_wl_nm": float(wl_shared[i_p]),
            "max_abs_spearman": max_abs_spearman,
            "max_abs_spearman_band": int(shared_bands[i_s]),
            "max_abs_spearman_wl_nm": float(wl_shared[i_s]),
            "max_mi": max_mi,
            "max_mi_band": int(shared_bands[i_m]),
            "max_mi_wl_nm": float(wl_shared[i_m]),
            "max_dcor": max_dcor,
            "max_dcor_band": int(shared_bands[i_d]),
            "max_dcor_wl_nm": float(wl_shared[i_d]),
            "ridge_r2": ridge,
            "rf_r2": rf_r2,
            "rf_minus_ridge": float(rf_r2 - ridge),
            "nonlinear_flag": nonlinear_flag,
            "top_permutation_importance": perm_scores,
        }
        rows.append(row)
        print(
            f"  pearson={max_abs_pearson:.3f} spearman={max_abs_spearman:.3f} "
            f"mi={max_mi:.3f} dcor={max_dcor:.3f} ridge={ridge:.3f} rf={rf_r2:.3f} "
            f"flag={nonlinear_flag}"
        )

    # Figures
    _heatmap(
        spearman,
        wavelengths=wl_shared,
        task_names=task_names,
        title=f"Spearman ρ: reflectance vs QoI (n={len(idx)})",
        out_path=out_dir / "band_vs_qoi_spearman.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="Spearman ρ",
    )
    _heatmap(
        mi,
        wavelengths=wl_shared,
        task_names=task_names,
        title=f"Mutual information: reflectance vs QoI (n={len(idx)})",
        out_path=out_dir / "band_vs_qoi_mutual_info.png",
        cmap="magma",
        vmin=0,
        cbar_label="MI (nats approx.)",
    )
    _heatmap(
        dcor,
        wavelengths=wl_shared,
        task_names=task_names,
        title=f"Distance correlation: reflectance vs QoI (n={len(idx_dcor)})",
        out_path=out_dir / "band_vs_qoi_distance_corr.png",
        cmap="magma",
        vmin=0,
        vmax=1,
        cbar_label="dCor",
    )

    # Bar comparison
    labels = [r["qoi"] for r in rows]
    pearson_max = [r["max_abs_pearson"] for r in rows]
    mi_max = [r["max_mi"] for r in rows]
    mi_scale = max(mi_max) if max(mi_max) > 0 else 1.0
    mi_norm = [m / mi_scale for m in mi_max]
    rf_vals = [max(0.0, r["rf_r2"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, pearson_max, width, label="max |Pearson r|")
    ax.bar(x, mi_norm, width, label=f"max MI / {mi_scale:.3f}")
    ax.bar(x + width, rf_vals, width, label="RF R² (clip≥0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Linear vs nonlinear recoverability by QoI")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "qoi_linear_vs_nonlinear_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "data_path": str(data_path),
        "band_config": str(band_config),
        "input_variable": "toa_reflectance",
        "n_samples_total": int(meta["n_samples"]),
        "n_subsample": int(len(idx)),
        "n_dcor_subsample": int(len(idx_dcor)),
        "shared_band_indices": shared_bands,
        "seed": seed,
        "task_names": task_names,
        "per_qoi": rows,
        "nonlinearity_rule": {
            "rf_minus_ridge_gt": 0.10,
            "or_mi_high_pearson_low": {"max_mi_gt": 0.05, "max_abs_pearson_lt": 0.15},
            "or_dcor_high_pearson_low": {"max_dcor_gt": 0.20, "max_abs_pearson_lt": 0.15},
        },
    }

    json_path = out_dir / "nonlinear_dependence_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    csv_path = out_dir / "nonlinear_dependence_summary.csv"
    fieldnames = [
        "qoi",
        "max_abs_pearson",
        "max_abs_spearman",
        "max_mi",
        "max_dcor",
        "ridge_r2",
        "rf_r2",
        "rf_minus_ridge",
        "nonlinear_flag",
        "max_abs_pearson_wl_nm",
        "max_abs_spearman_wl_nm",
        "max_mi_wl_nm",
        "max_dcor_wl_nm",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Nonlinear reflectance–QoI diagnostics")
    parser.add_argument("--data-path", type=str, default=str(S2_DEFAULT_DATA_PATH))
    parser.add_argument("--band-config", type=str, default=str(S2_DEFAULT_BAND_CONFIG_PATH))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-subsample", type=int, default=12000)
    parser.add_argument("--n-dcor", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_diagnostics(
        data_path=Path(args.data_path),
        band_config=Path(args.band_config),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_subsample=args.n_subsample,
        n_dcor=args.n_dcor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
