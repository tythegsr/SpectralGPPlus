"""Per-QoI PCA dimension diagnostics for S2 TOA (variance + supervised).

Band modes
----------
- ``subset`` — keep/drop from a task band JSON (default: from_corr).
- ``full`` — all 285 wavelengths.

Selection rules
---------------
- ``variance`` — smallest ``p`` with cumulative explained X-variance >= threshold
  (default 0.99).
- ``supervised`` — smallest ``p`` whose ridge R^2 on leading PCA scores is within
  ``r2_tol`` of full-rank ridge R^2 on those scores (same probe style as band
  selection). Weak QoIs fall back to the variance ``p``.

Writes plots/summaries under ``--out-dir`` and JSON configs under
``experiments_toa/configs/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.s2_bands import load_task_band_config
from experiments_toa.s2_constants import (
    S2_DEFAULT_DATA_PATH,
    S2_INPUT_DIM,
    S2_TASK_NAMES,
)
from experiments_toa.s2_data import load_s2_toa_data

BandMode = Literal["subset", "full"]
Selection = Literal["variance", "supervised"]

_DEFAULT_BAND_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "s2_task_bands_from_corr.json"
)
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "data 11 QoI" / "pca_svd"
)


def _ridge_r2(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> float:
    if X.shape[1] == 0:
        return 0.0
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    xm = xtr.mean(0)
    xs = xtr.std(0) + 1e-8
    ztr = (xtr - xm) / xs
    zte = (xte - xm) / xs
    lam = 1.0
    w = np.linalg.solve(
        ztr.T @ ztr + lam * np.eye(ztr.shape[1]),
        ztr.T @ (ytr - ytr.mean()),
    )
    pred = zte @ w + ytr.mean()
    return float(r2_score(yte, pred))


def _p_at_cumvar(cumvar: np.ndarray, threshold: float) -> int:
    if cumvar.size == 0:
        return 1
    hits = np.where(cumvar >= threshold)[0]
    if hits.size == 0:
        return int(cumvar.size)
    return int(hits[0]) + 1


def _apply_x_transform_np(x: np.ndarray, transform: str | None) -> np.ndarray:
    if transform is None or transform == "none":
        return x
    if transform == "log1p":
        return np.log1p(np.maximum(x, 0.0))
    raise ValueError(f"Unknown x_transform {transform!r}; expected 'none' or 'log1p'.")


def _supervised_p(
    z: np.ndarray,
    y: np.ndarray,
    *,
    r2_tol: float,
    min_components: int,
    step: int,
    seed: int,
) -> tuple[int, float, float, list[dict[str, float]]]:
    """Return (p, r2_full, r2_selected, curve)."""
    rank = int(z.shape[1])
    r2_full = _ridge_r2(z, y, seed=seed)
    goal = r2_full - r2_tol
    curve: list[dict[str, float]] = []
    chosen = rank
    r2_sel = r2_full
    for k in range(step, rank + step, step):
        p = min(k, rank)
        r2 = _ridge_r2(z[:, :p], y, seed=seed)
        curve.append({"p": float(p), "r2": float(r2)})
        if p >= min_components and r2 >= goal:
            chosen = p
            r2_sel = r2
            break
        chosen = p
        r2_sel = r2
    # Fine-tune downward within the last step window when possible.
    if chosen > min_components and step > 1:
        lo = max(min_components, chosen - step + 1)
        for p in range(lo, chosen + 1):
            r2 = _ridge_r2(z[:, :p], y, seed=seed)
            if r2 >= goal:
                chosen = p
                r2_sel = r2
                break
    return int(chosen), float(r2_full), float(r2_sel), curve


def _plot_scree(
    *,
    task: str,
    band_mode: str,
    ratio: np.ndarray,
    cumvar: np.ndarray,
    p_99: int,
    p_supervised: int | None,
    out_path: Path,
    variance_threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    idxs = np.arange(1, ratio.size + 1)

    axes[0].plot(idxs, ratio, lw=1.2)
    axes[0].axvline(p_99, color="C1", ls="--", label=f"p_99={p_99}")
    if p_supervised is not None:
        axes[0].axvline(p_supervised, color="C2", ls=":", label=f"p_ridge={p_supervised}")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title(f"{task} ({band_mode}) scree")
    axes[0].legend(fontsize=8)

    axes[1].plot(idxs, cumvar, lw=1.2)
    axes[1].axhline(variance_threshold, color="0.4", ls="--", label=f"{variance_threshold:.0%}")
    axes[1].axvline(p_99, color="C1", ls="--", label=f"p_99={p_99}")
    if p_supervised is not None:
        axes[1].axvline(p_supervised, color="C2", ls=":", label=f"p_ridge={p_supervised}")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("cumulative variance")
    axes[1].set_title(f"{task} ({band_mode}) cumulative")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.0, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_ridge_curve(
    *,
    task: str,
    band_mode: str,
    curve: list[dict[str, float]],
    p_supervised: int,
    r2_full: float,
    r2_tol: float,
    out_path: Path,
) -> None:
    if not curve:
        return
    ps = [c["p"] for c in curve]
    r2s = [c["r2"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ps, r2s, marker="o", ms=3, lw=1.2)
    ax.axhline(r2_full, color="0.4", ls="--", label=f"r2_full={r2_full:.3f}")
    ax.axhline(r2_full - r2_tol, color="0.6", ls=":", label=f"goal (tol={r2_tol})")
    ax.axvline(p_supervised, color="C2", ls=":", label=f"p_ridge={p_supervised}")
    ax.set_xlabel("p (leading PCA components)")
    ax.set_ylabel("ridge R²")
    ax.set_title(f"{task} ({band_mode}) supervised p")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _config_name(selection: Selection, band_mode: BandMode) -> str:
    sel = "var99" if selection == "variance" else "ridge"
    return f"s2_task_pca_components_{sel}_{band_mode}.json"


def _relpath_or_str(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _write_components_config(
    path: Path,
    *,
    selection: Selection,
    band_mode: BandMode,
    tasks: dict[str, int],
    meta: dict[str, Any],
) -> None:
    default_p = int(max(tasks.values())) if tasks else 1
    band_cfg = meta.get("task_band_config")
    payload = {
        "selection": selection,
        "band_mode": band_mode,
        "variance_threshold": meta.get("variance_threshold"),
        "r2_tol": meta.get("r2_tol"),
        "input_variable": meta.get("input_variable"),
        "task_band_config": _relpath_or_str(band_cfg) if band_cfg else None,
        "n_train": meta.get("n_train"),
        "x_transform": meta.get("x_transform"),
        "seed": meta.get("seed"),
        "default": default_p,
        "tasks": {k: int(v) for k, v in tasks.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_pca_svd_analysis(
    *,
    data_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    n_train: int = 16000,
    seed: int = 42,
    input_variable: str = "toa_radiance",
    x_transform: str | None = "none",
    task_band_config: str | Path | None = None,
    band_modes: Sequence[BandMode] = ("subset", "full"),
    selections: Sequence[Selection] = ("variance", "supervised"),
    variance_threshold: float = 0.99,
    r2_tol: float = 0.02,
    min_ridge_r2: float = 0.15,
    min_components: int = 4,
    step: int = 4,
    task_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    out_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    band_cfg_path = Path(task_band_config) if task_band_config is not None else _DEFAULT_BAND_CONFIG
    if data_path is None:
        data_path = S2_DEFAULT_DATA_PATH

    (
        x_train,
        y_train,
        _x_val,
        _y_val,
        _x_test,
        _y_test,
        _train_idx,
        _val_idx,
        _test_idx,
        _wavelengths,
        _meta,
    ) = load_s2_toa_data(
        n_train=n_train,
        n_test=0,
        n_val=0,
        seed=seed,
        data_path=data_path,
        input_variable=input_variable,  # type: ignore[arg-type]
        task_names=names,
    )

    x_np = np.asarray(x_train.detach().cpu().numpy(), dtype=np.float64)
    y_np = np.asarray(y_train.detach().cpu().numpy(), dtype=np.float64)
    if x_np.shape[1] != S2_INPUT_DIM:
        raise ValueError(f"Expected input_dim={S2_INPUT_DIM}, got {x_np.shape[1]}")

    bands_subset = load_task_band_config(
        band_cfg_path, task_names=names, input_dim=S2_INPUT_DIM
    )
    bands_full = {name: list(range(S2_INPUT_DIM)) for name in names}

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"tasks": {}}
    components_out: dict[tuple[Selection, BandMode], dict[str, int]] = {}
    for sel in selections:
        for mode in band_modes:
            components_out[(sel, mode)] = {}

    shared_meta = {
        "variance_threshold": float(variance_threshold),
        "r2_tol": float(r2_tol),
        "input_variable": input_variable,
        "task_band_config": str(band_cfg_path),
        "n_train": int(n_train),
        "x_transform": x_transform or "none",
        "seed": int(seed),
        "min_ridge_r2": float(min_ridge_r2),
        "min_components": int(min_components),
        "step": int(step),
    }

    for task_idx, name in enumerate(names):
        detail["tasks"][name] = {}
        for mode in band_modes:
            band_idx = bands_subset[name] if mode == "subset" else bands_full[name]
            x_task = _apply_x_transform_np(x_np[:, band_idx], x_transform)
            n_bands = int(x_task.shape[1])
            n_comp = min(n_train, n_bands)
            if n_comp < 1:
                raise ValueError(f"{name}/{mode}: empty band selection")

            pca = PCA(n_components=n_comp, svd_solver="full", random_state=seed)
            z = pca.fit_transform(x_task)
            ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
            cumvar = np.cumsum(ratio)
            p_90 = _p_at_cumvar(cumvar, 0.90)
            p_95 = _p_at_cumvar(cumvar, 0.95)
            p_99 = _p_at_cumvar(cumvar, variance_threshold)

            p_supervised: int | None = None
            r2_full = float("nan")
            r2_sel = float("nan")
            weak = False
            reason = "variance only"
            curve: list[dict[str, float]] = []

            if "supervised" in selections:
                p_sup, r2_full, r2_sel, curve = _supervised_p(
                    z,
                    y_np[:, task_idx],
                    r2_tol=r2_tol,
                    min_components=min_components,
                    step=step,
                    seed=seed,
                )
                if r2_full < min_ridge_r2:
                    weak = True
                    p_supervised = p_99
                    reason = (
                        f"weak full ridge R2={r2_full:.3f} < min_ridge_r2={min_ridge_r2}; "
                        f"fallback to p_99={p_99}"
                    )
                else:
                    p_supervised = p_sup
                    reason = (
                        f"ridge on PCA scores until R2 >= {r2_full:.3f}-{r2_tol} "
                        f"(got {r2_sel:.3f} with p={p_supervised})"
                    )

            if "variance" in selections:
                components_out[("variance", mode)][name] = int(p_99)
            if "supervised" in selections and p_supervised is not None:
                components_out[("supervised", mode)][name] = int(p_supervised)

            row = {
                "task": name,
                "band_mode": mode,
                "n_bands": n_bands,
                "rank": n_comp,
                "p_90": p_90,
                "p_95": p_95,
                "p_99": p_99,
                "var_at_p99": float(cumvar[p_99 - 1]),
                "p_supervised": p_supervised,
                "r2_full": r2_full,
                "r2_selected": r2_sel,
                "weak_signal": weak,
                "reason": reason,
            }
            rows.append(row)
            detail["tasks"][name][mode] = {
                **row,
                "explained_variance_ratio": ratio.tolist(),
                "cumulative_variance": cumvar.tolist(),
                "ridge_curve": curve,
            }

            plot_path = out_dir / "plots" / f"{name}_{mode}_scree.png"
            _plot_scree(
                task=name,
                band_mode=mode,
                ratio=ratio,
                cumvar=cumvar,
                p_99=p_99,
                p_supervised=p_supervised if "supervised" in selections else None,
                out_path=plot_path,
                variance_threshold=variance_threshold,
            )
            if "supervised" in selections and curve:
                _plot_ridge_curve(
                    task=name,
                    band_mode=mode,
                    curve=curve,
                    p_supervised=int(p_supervised or p_99),
                    r2_full=float(r2_full),
                    r2_tol=r2_tol,
                    out_path=out_dir / "plots" / f"{name}_{mode}_ridge.png",
                )

            print(
                f"{name:14s} {mode:6s} bands={n_bands:3d}  "
                f"p90={p_90:3d} p95={p_95:3d} p99={p_99:3d}  "
                f"p_ridge={p_supervised}  var@p99={cumvar[p_99 - 1]:.4f}"
            )

    # Summaries
    summary_csv = out_dir / "pca_svd_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)

    summary_json = out_dir / "pca_svd_summary.json"
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump({"meta": shared_meta, "rows": rows}, fh, indent=2)
        fh.write("\n")

    detail_path = out_dir / "pca_svd_detail.json"
    with detail_path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": shared_meta, **detail}, fh, indent=2)
        fh.write("\n")

    written_configs: list[str] = []
    for (sel, mode), tasks in components_out.items():
        if not tasks:
            continue
        cfg_path = config_dir / _config_name(sel, mode)
        band_meta = dict(shared_meta)
        if mode == "full":
            band_meta["task_band_config"] = str(
                config_dir / "s2_task_bands_all.json"
            )
        _write_components_config(
            cfg_path,
            selection=sel,
            band_mode=mode,
            tasks=tasks,
            meta=band_meta,
        )
        written_configs.append(str(cfg_path))
        print(f"Wrote {cfg_path}")

    return {
        "out_dir": str(out_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "detail_json": str(detail_path),
        "configs": written_configs,
        "rows": rows,
    }


def _parse_modes(raw: str) -> list[BandMode]:
    if raw == "both":
        return ["subset", "full"]
    if raw in ("subset", "full"):
        return [raw]  # type: ignore[list-item]
    raise ValueError(f"--band-mode must be subset|full|both, got {raw!r}")


def _parse_selections(raw: str) -> list[Selection]:
    if raw == "both":
        return ["variance", "supervised"]
    if raw in ("variance", "supervised"):
        return [raw]  # type: ignore[list-item]
    raise ValueError(f"--selection must be variance|supervised|both, got {raw!r}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    p.add_argument("--config-dir", type=str, default=str(_DEFAULT_CONFIG_DIR))
    p.add_argument("--n-train", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-variable", type=str, default="toa_radiance")
    p.add_argument("--x-transform", type=str, default="none")
    p.add_argument("--task-band-config", type=str, default=str(_DEFAULT_BAND_CONFIG))
    p.add_argument("--band-mode", type=str, default="both", choices=["subset", "full", "both"])
    p.add_argument(
        "--selection",
        type=str,
        default="both",
        choices=["variance", "supervised", "both"],
    )
    p.add_argument("--variance-threshold", type=float, default=0.99)
    p.add_argument("--r2-tol", type=float, default=0.02)
    p.add_argument("--min-ridge-r2", type=float, default=0.15)
    p.add_argument("--min-components", type=int, default=4)
    p.add_argument("--step", type=int, default=4)
    p.add_argument(
        "--qoi",
        type=str,
        default=None,
        help="Comma-separated QoI names (default: all 11)",
    )
    args = p.parse_args(argv)

    names = None
    if args.qoi:
        names = [s.strip() for s in args.qoi.split(",") if s.strip()]

    run_pca_svd_analysis(
        data_path=args.data_path,
        out_dir=args.out_dir,
        config_dir=args.config_dir,
        n_train=args.n_train,
        seed=args.seed,
        input_variable=args.input_variable,
        x_transform=args.x_transform,
        task_band_config=args.task_band_config,
        band_modes=_parse_modes(args.band_mode),
        selections=_parse_selections(args.selection),
        variance_threshold=args.variance_threshold,
        r2_tol=args.r2_tol,
        min_ridge_r2=args.min_ridge_r2,
        min_components=args.min_components,
        step=args.step,
        task_names=names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
