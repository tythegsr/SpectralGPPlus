"""Per-QoI wavelength keep/drop for S2 TOA (recoverability-preserving).

Two layers
----------
1. **Global absorption / low-SNR** — always drop (band–band std/corr heuristic).
2. **Per-QoI prune** — among remaining bands, drop those that do not help
   recover the QoI under a ridge probe:

   - Rank bands by ``score = max(|Pearson|, |Spearman|)``.
   - Forward-add in that order until ridge \(R^2\) is within ``r2_tol`` of the
     full (non-global-drop) ridge \(R^2\).
   - Everything not selected is a recommended drop for that QoI.

This avoids the failure mode of pure marginal-correlation thresholds (e.g.
``aot`` has weak per-band |r| but strong multivariate ridge \(R^2\)).

Weak-signal QoIs (full ridge \(R^2 < min_ridge_r2``) keep all non-global bands.

Writes under ``--out-dir`` and ``experiments_toa/configs/s2_task_bands_from_corr.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_constants import S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_correlation_analysis import _suggest_drop_indices

QOI_NAMES = list(S2_TASK_NAMES)


def _indices_to_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    sorted_idx = sorted(indices)
    ranges: list[list[int]] = []
    lo = hi = sorted_idx[0]
    for i in sorted_idx[1:]:
        if i == hi + 1:
            hi = i
        else:
            ranges.append([lo, hi])
            lo = hi = i
    ranges.append([lo, hi])
    return ranges


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        wl = np.asarray(f["wl"][:], dtype=np.float64)
        refl = np.asarray(f["toa_reflectance"][:], dtype=np.float64)
        Y = np.column_stack([np.asarray(f[n][:], dtype=np.float64) for n in QOI_NAMES])
    if refl.shape[1] != S2_INPUT_DIM:
        raise ValueError(f"Expected {S2_INPUT_DIM} bands, got {refl.shape[1]}")
    return wl, refl, Y


def _band_scores(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bands = X.shape[1]
    pearson = np.zeros(n_bands, dtype=np.float64)
    spearman = np.zeros(n_bands, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_std = float(np.std(y))
    for j in range(n_bands):
        xj = X[:, j]
        if float(np.std(xj)) < 1e-15 or y_std < 1e-15:
            continue
        pearson[j] = float(np.corrcoef(xj, y)[0, 1])
        spearman[j] = float(spearmanr(xj, y).correlation)
    pearson = np.nan_to_num(pearson, nan=0.0)
    spearman = np.nan_to_num(spearman, nan=0.0)
    score = np.maximum(np.abs(pearson), np.abs(spearman))
    return pearson, spearman, score


def _ridge_r2(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> float:
    if X.shape[1] == 0:
        return 0.0
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    xm = xtr.mean(0)
    xs = xtr.std(0) + 1e-8
    ztr = (xtr - xm) / xs
    zte = (xte - xm) / xs
    lam = 1.0
    w = np.linalg.solve(ztr.T @ ztr + lam * np.eye(ztr.shape[1]), ztr.T @ (ytr - ytr.mean()))
    pred = zte @ w + ytr.mean()
    return float(r2_score(yte, pred))


def _forward_select(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[int],
    order: list[int],
    *,
    target_r2: float,
    r2_tol: float,
    min_bands: int,
    step: int,
    seed: int,
) -> list[int]:
    """Add bands in ``order`` until ridge R2 >= target_r2 - r2_tol."""
    goal = target_r2 - r2_tol
    selected: list[int] = []
    best_r2 = -np.inf
    # Evaluate in chunks for speed.
    for k in range(step, len(order) + step, step):
        selected = order[: min(k, len(order))]
        r2 = _ridge_r2(X[:, selected], y, seed=seed)
        best_r2 = r2
        if len(selected) >= min_bands and r2 >= goal:
            break
    # If still short, keep all candidates.
    if best_r2 < goal:
        selected = list(candidates)
    return selected


def select_bands_per_qoi(
    wl: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r2_tol: float = 0.02,
    min_ridge_r2: float = 0.15,
    min_bands: int = 16,
    step: int = 8,
    subsample: int | None = 20000,
    seed: int = 0,
) -> dict:
    n = X.shape[0]
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample, replace=False)
        X = X[idx]
        Y = Y[idx]

    global_drop = sorted(_suggest_drop_indices(X))
    global_drop_set = set(global_drop)
    candidates = [i for i in range(X.shape[1]) if i not in global_drop_set]

    tasks: dict[str, dict] = {}
    score_rows: list[dict] = []

    for t, name in enumerate(QOI_NAMES):
        pearson, spearman, score = _band_scores(X, Y[:, t])
        y = Y[:, t]
        r2_full = _ridge_r2(X[:, candidates], y, seed=seed)
        max_score = float(score[candidates].max()) if candidates else 0.0
        order = sorted(candidates, key=lambda i: score[i], reverse=True)

        if r2_full < min_ridge_r2:
            keep = list(candidates)
            reason = (
                f"weak full ridge R2={r2_full:.3f} < min_ridge_r2={min_ridge_r2}; "
                "keep all non-global-drop bands"
            )
            r2_sel = r2_full
            weak = True
        else:
            keep = _forward_select(
                X,
                y,
                candidates,
                order,
                target_r2=r2_full,
                r2_tol=r2_tol,
                min_bands=min_bands,
                step=step,
                seed=seed,
            )
            keep = sorted(keep)
            r2_sel = _ridge_r2(X[:, keep], y, seed=seed)
            weak = False
            reason = (
                f"forward-add by score until ridge R2 >= {r2_full:.3f}-{r2_tol} "
                f"(got {r2_sel:.3f} with {len(keep)} bands)"
            )

        keep_set = set(keep)
        per_qoi_drop = [i for i in candidates if i not in keep_set]
        drop_all = sorted(global_drop_set | set(per_qoi_drop))
        best = int(np.argmax(score))

        tasks[name] = {
            "max_score": max_score,
            "ridge_r2_full": r2_full,
            "ridge_r2_selected": r2_sel,
            "ridge_r2_delta": r2_sel - r2_full,
            "weak_signal": weak,
            "reason": reason,
            "n_keep": len(keep),
            "n_drop": len(drop_all),
            "n_global_drop": len(global_drop),
            "n_per_qoi_drop": len(per_qoi_drop),
            "keep_indices": keep,
            "keep_ranges": _indices_to_ranges(keep),
            "keep_wavelengths_nm": [float(wl[i]) for i in keep],
            "drop_indices": drop_all,
            "drop_wavelengths_nm": [float(wl[i]) for i in drop_all],
            "per_qoi_only_drop_indices": per_qoi_drop,
            "per_qoi_only_drop_wavelengths_nm": [float(wl[i]) for i in per_qoi_drop],
            "best_band_index": best,
            "best_wavelength_nm": float(wl[best]),
            "best_pearson": float(pearson[best]),
            "best_spearman": float(spearman[best]),
        }

        for i in range(X.shape[1]):
            score_rows.append(
                {
                    "qoi": name,
                    "band_index": i,
                    "wavelength_nm": float(wl[i]),
                    "pearson": float(pearson[i]),
                    "spearman": float(spearman[i]),
                    "score": float(score[i]),
                    "global_drop": i in global_drop_set,
                    "kept": i in keep_set,
                }
            )

    return {
        "global_drop_indices": global_drop,
        "global_drop_wavelengths_nm": [float(wl[i]) for i in global_drop],
        "params": {
            "r2_tol": r2_tol,
            "min_ridge_r2": min_ridge_r2,
            "min_bands": min_bands,
            "step": step,
            "subsample": subsample,
            "seed": seed,
        },
        "tasks": tasks,
        "score_rows": score_rows,
    }


def _write_band_config(result: dict, out_path: Path, *, input_dim: int = S2_INPUT_DIM) -> None:
    default_keep = [
        i for i in range(input_dim) if i not in set(result["global_drop_indices"])
    ]
    payload = {
        "input_dim": input_dim,
        "default": _indices_to_ranges(default_keep),
        "tasks": {name: info["keep_ranges"] for name, info in result["tasks"].items()},
        "notes": (
            "Auto-generated by s2_per_qoi_band_selection.py. "
            "Always drops global absorption/low-SNR bands. Per-QoI drops use "
            "forward selection by max(|Pearson|,|Spearman|) until ridge R2 is "
            "within r2_tol of the full-spectrum ridge R2."
        ),
        "params": result["params"],
        "global_drop_indices": result["global_drop_indices"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(score_rows: list[dict], out_path: Path) -> None:
    fields = [
        "qoi",
        "band_index",
        "wavelength_nm",
        "pearson",
        "spearman",
        "score",
        "global_drop",
        "kept",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(score_rows)


def _plot_spectra(wl: np.ndarray, result: dict, out_path: Path) -> None:
    n = len(QOI_NAMES)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.35 * n), sharex=True)
    if n == 1:
        axes = [axes]
    global_drop = set(result["global_drop_indices"])
    by_qoi: dict[str, list[dict]] = {name: [] for name in QOI_NAMES}
    for row in result["score_rows"]:
        by_qoi[row["qoi"]].append(row)

    for ax, name in zip(axes, QOI_NAMES):
        rows = sorted(by_qoi[name], key=lambda r: r["band_index"])
        scores = np.array([r["score"] for r in rows])
        kept = np.array([r["kept"] for r in rows])
        ax.plot(wl, scores, color="0.35", lw=0.9)
        ax.fill_between(wl, 0, scores, where=~kept, color="salmon", alpha=0.45, label="drop")
        ax.fill_between(wl, 0, scores, where=kept, color="steelblue", alpha=0.35, label="keep")
        for gi in global_drop:
            ax.axvline(wl[gi], color="cyan", lw=0.3, alpha=0.5)
        info = result["tasks"][name]
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(0, max(float(scores.max()) * 1.05, 0.05))
        ax.tick_params(labelsize=7)
        ax.text(
            0.99,
            0.85,
            f"keep={info['n_keep']}  R2 {info['ridge_r2_full']:.2f}->{info['ridge_r2_selected']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Per-QoI score=max(|Pearson|,|Spearman|); keep via ridge-R2 forward select; cyan=global drops",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_drop_summary_txt(result: dict, out_path: Path) -> None:
    lines = [
        "Per-QoI bands to remove (global absorption + QoI-specific unused bands)",
        f"data: {result.get('data_path', '')}",
        f"params: {result['params']}",
        "",
        "GLOBAL absorption/low-SNR (always remove):",
        f"  indices: {result['global_drop_indices']}",
        f"  nm: {[round(x, 1) for x in result['global_drop_wavelengths_nm']]}",
        "",
    ]
    for name, info in result["tasks"].items():
        lines.append(
            f"{name}: keep {info['n_keep']} / drop {info['n_drop']} "
            f"(R2 {info['ridge_r2_full']:.3f} -> {info['ridge_r2_selected']:.3f})"
        )
        lines.append(f"  keep_ranges: {info['keep_ranges']}")
        lines.append(
            f"  drop_nm (all): {[round(x, 1) for x in info['drop_wavelengths_nm']]}"
        )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    data_path: Path,
    out_dir: Path,
    r2_tol: float = 0.02,
    min_ridge_r2: float = 0.15,
    min_bands: int = 16,
    step: int = 8,
    subsample: int | None = 20000,
    seed: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {data_path}")
    wl, X, Y = _load(data_path)
    print(f"  X={X.shape} Y={Y.shape} wl={wl.min():.1f}-{wl.max():.1f} nm")

    result = select_bands_per_qoi(
        wl,
        X,
        Y,
        r2_tol=r2_tol,
        min_ridge_r2=min_ridge_r2,
        min_bands=min_bands,
        step=step,
        subsample=subsample,
        seed=seed,
    )
    result["data_path"] = str(data_path.resolve())

    report = {k: v for k, v in result.items() if k != "score_rows"}
    (out_dir / "per_qoi_band_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_band_config(result, out_dir / "s2_task_bands_from_corr.json")
    configs_dir = Path(__file__).resolve().parent / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = configs_dir / "s2_task_bands_from_corr.json"
    _write_band_config(result, train_cfg)
    _write_csv(result["score_rows"], out_dir / "per_qoi_band_scores.csv")
    _plot_spectra(wl, result, out_dir / "per_qoi_corr_spectrum.png")
    _write_drop_summary_txt(result, out_dir / "per_qoi_bands_to_drop.txt")

    print(f"Global drops ({len(result['global_drop_indices'])}): {result['global_drop_indices']}")
    print(
        f"{'qoi':14s} {'keep':>5} {'drop':>5} {'R2_full':>8} {'R2_sel':>8} {'dR2':>7}"
    )
    for name, info in result["tasks"].items():
        print(
            f"{name:14s} {info['n_keep']:5d} {info['n_drop']:5d} "
            f"{info['ridge_r2_full']:8.3f} {info['ridge_r2_selected']:8.3f} "
            f"{info['ridge_r2_delta']:7.3f}"
        )
        print(f"  keep_ranges={info['keep_ranges']}")
    print(f"Wrote artifacts under {out_dir}")
    print(f"Training config: {train_cfg}")
    return result


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data 11 QoI"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(data_dir / "snow_toa_simulations_20262107.nc"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "analysis_toa_July21"),
    )
    parser.add_argument("--r2-tol", type=float, default=0.02)
    parser.add_argument("--min-ridge-r2", type=float, default=0.15)
    parser.add_argument("--min-bands", type=int, default=16)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=49000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        data_path=Path(args.data_path),
        out_dir=Path(args.out_dir),
        r2_tol=args.r2_tol,
        min_ridge_r2=args.min_ridge_r2,
        min_bands=args.min_bands,
        step=args.step,
        subsample=None if args.subsample <= 0 else args.subsample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
