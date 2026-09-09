"""ASD spectral neighbors in the synthetic training set.

Finds every synthetic spectrum within a mean-normalized RMSE threshold of each
ASD scene, writes an Excel workbook (ASD params as row 1 of each scene sheet),
summary JSON, and companion plots.

Mean-normalized RMSE (not QoI RRMSE)::

    d = RMSE(sim, ASD) / mean(|ASD|)   over spectral bands

Requires ``pandas`` and ``openpyxl`` (``pip install openpyxl`` if missing).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_flat_Sep03.nc"
)
DEFAULT_THR = 0.04

# Unified Excel / JSON column names (ASD NetCDF names mapped in).
QOI_COLS = ("grain_size", "cos_i", "dust", "algae", "lwc", "aot", "cwv", "fsnow")
AUX_COLS = ("sza", "slope", "elevation_km", "coszen")
STD_COLS = ("grain_size_std", "dust_std", "algae_std", "lwc_std")

ASD_QOI_MAP = {
    "grain_size": "grain_radius_mean",
    "cos_i": "cos_i",
    "dust": "dust_conc_mean",
    "algae": "algae_conc_mean",
    "lwc": "lwc_mean",
    "aot": "aod",
    "cwv": "cwv",
}
ASD_AUX_MAP = {
    "sza": "sza",
    "elevation_km": "elevation_km",
}
ASD_STD_MAP = {
    "grain_size_std": "grain_radius_std",
    "dust_std": "dust_conc_std",
    "algae_std": "algae_conc_std",
    "lwc_std": "lwc_std",
}
SIM_QOI_MAP = {
    "grain_size": "grain_size",
    "cos_i": "cos_i",
    "dust": "dust",
    "algae": "algae",
    "lwc": "liquid_water",
    "aot": "aot",
    "cwv": "cwv",
    "fsnow": "fsnow",
}
SIM_AUX_MAP = {
    "sza": "sza",
    "slope": "slope",
    "elevation_km": "ele_km",
    "coszen": "coszen",
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def mean_normalized_rmse(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Row-wise mean-normalized RMSE: RMSE / mean(|query|).

    query: (n_q, n_bands), gallery: (n_g, n_bands) -> (n_q, n_g)
    """
    scale = np.maximum(np.mean(np.abs(query), axis=1, keepdims=True), 1e-8)
    return np.sqrt(((gallery[None, :, :] - query[:, None, :]) ** 2).mean(axis=2)) / scale


def _decode_dates(raw) -> list[str]:
    out = []
    for d in raw:
        if isinstance(d, (bytes, bytearray)):
            out.append(d.decode())
        else:
            out.append(str(d))
    return out


def _read_optional(f: h5py.File, key: str, n: int) -> np.ndarray:
    if key in f:
        return np.asarray(f[key][:], dtype=np.float64)
    return np.full(n, np.nan, dtype=np.float64)


def _label_spread(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "span": float("nan"),
        }
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "span": float(np.max(vals) - np.min(vals)),
    }


def resolve_sim_path_from_ckpt(ckpt_dir: str | Path) -> Path | None:
    """Prefer ``gp_S4_*.json`` → ``data_meta.emit_path`` in the run directory."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("gp_S4*.json"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("gp_*.json"))
    for jp in candidates:
        try:
            with open(jp, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError:
            # Windows MAX_PATH workaround
            try:
                with open("\\\\?\\" + str(jp.resolve()), encoding="utf-8") as f:
                    payload = json.load(f)
            except OSError:
                continue
        meta = payload.get("data_meta") or {}
        emit = meta.get("emit_path") or payload.get("data_path")
        if emit:
            p = Path(str(emit))
            if p.is_file():
                return p
    return None


def find_asd_sim_neighbors(
    asd_path: str | Path,
    sim_path: str | Path,
    thr: float = DEFAULT_THR,
    *,
    chunk: int = 5000,
    plot_topk: int = 8,
) -> dict[str, Any]:
    """Scan the full synthetic set for spectra within ``thr`` of each ASD scene."""
    asd_path = Path(asd_path)
    sim_path = Path(sim_path)
    plot_topk = max(int(plot_topk), 1)

    with h5py.File(asd_path, "r") as fa, h5py.File(sim_path, "r") as fs:
        wl = np.asarray(fa["wl"][:], dtype=np.float64)
        asd_rad = np.asarray(fa["toa_radiance"][:], dtype=np.float64)
        sim_rad = np.asarray(fs["toa_radiance"][:], dtype=np.float64)
        n_asd, n_sim = asd_rad.shape[0], sim_rad.shape[0]
        dates = _decode_dates(fa["date"][:])

        asd_params: dict[str, np.ndarray] = {}
        for col, key in ASD_QOI_MAP.items():
            asd_params[col] = _read_optional(fa, key, n_asd)
        for col in QOI_COLS:
            if col not in asd_params:
                asd_params[col] = np.full(n_asd, np.nan, dtype=np.float64)
        for col, key in ASD_AUX_MAP.items():
            asd_params[col] = _read_optional(fa, key, n_asd)
        for col in AUX_COLS:
            if col not in asd_params:
                asd_params[col] = np.full(n_asd, np.nan, dtype=np.float64)
        for col, key in ASD_STD_MAP.items():
            asd_params[col] = _read_optional(fa, key, n_asd)

        sim_params: dict[str, np.ndarray] = {}
        for col, key in SIM_QOI_MAP.items():
            sim_params[col] = _read_optional(fs, key, n_sim)
        for col, key in SIM_AUX_MAP.items():
            sim_params[col] = _read_optional(fs, key, n_sim)

    # Collect all matches (no top-k cap) and running top-k for spectrum plots.
    match_idx: list[list[int]] = [[] for _ in range(n_asd)]
    match_dist: list[list[float]] = [[] for _ in range(n_asd)]
    top_idx = np.full((n_asd, plot_topk), -1, dtype=np.int64)
    top_dist = np.full((n_asd, plot_topk), np.inf, dtype=np.float64)
    for start in range(0, n_sim, chunk):
        end = min(start + chunk, n_sim)
        d = mean_normalized_rmse(asd_rad, sim_rad[start:end])
        for i in range(n_asd):
            hit = np.flatnonzero(d[i] <= thr)
            if hit.size:
                match_idx[i].extend((start + hit).tolist())
                match_dist[i].extend(d[i, hit].tolist())
            cand_d = np.concatenate([top_dist[i], d[i]])
            cand_i = np.concatenate(
                [top_idx[i], np.arange(start, end, dtype=np.int64)]
            )
            order = np.argsort(cand_d)[:plot_topk]
            top_dist[i] = cand_d[order]
            top_idx[i] = cand_i[order]

    scenes: list[dict[str, Any]] = []
    for i in range(n_asd):
        idx = np.asarray(match_idx[i], dtype=np.int64)
        dist = np.asarray(match_dist[i], dtype=np.float64)
        if idx.size:
            order = np.argsort(dist)
            idx = idx[order]
            dist = dist[order]
        asd_row = {
            "row_type": "ASD",
            "date": dates[i],
            "sim_index": None,
            "mean_normalized_rmse": 0.0,
        }
        for col in (*QOI_COLS, *AUX_COLS, *STD_COLS):
            asd_row[col] = float(asd_params[col][i]) if col in asd_params else float("nan")

        sim_rows: list[dict[str, Any]] = []
        for j, dj in zip(idx.tolist(), dist.tolist()):
            row: dict[str, Any] = {
                "row_type": "SIM",
                "date": dates[i],
                "sim_index": int(j),
                "mean_normalized_rmse": float(dj),
            }
            for col in QOI_COLS:
                row[col] = float(sim_params[col][j])
            for col in AUX_COLS:
                row[col] = float(sim_params[col][j])
            for col in STD_COLS:
                row[col] = float("nan")
            sim_rows.append(row)

        spans: dict[str, dict[str, float]] = {}
        for col in ("grain_size", "dust", "algae", "lwc", "cos_i"):
            if idx.size:
                spans[col] = _label_spread(sim_params[col][idx])
            else:
                spans[col] = _label_spread(np.array([]))

        scenes.append(
            {
                "sample": i + 1,
                "date": dates[i],
                "n_matches": int(idx.size),
                "min_mean_normalized_rmse": float(dist[0]) if dist.size else float(
                    top_dist[i, 0]
                ),
                "median_mean_normalized_rmse": (
                    float(np.median(dist)) if dist.size else float("nan")
                ),
                "max_mean_normalized_rmse": float(dist[-1]) if dist.size else float("nan"),
                "spans": spans,
                "asd_row": asd_row,
                "sim_rows": sim_rows,
                "match_indices": idx.tolist(),
                "match_distances": dist.tolist(),
                "topk_indices": top_idx[i].tolist(),
                "topk_distances": top_dist[i].tolist(),
            }
        )

    return {
        "asd_path": str(asd_path),
        "sim_path": str(sim_path),
        "thr_mean_normalized_rmse": float(thr),
        "n_asd": n_asd,
        "n_sim": n_sim,
        "dates": dates,
        "scenes": scenes,
        # Arrays for plotting (not written to JSON).
        "_plot": {
            "wl": wl,
            "asd_rad": asd_rad,
            "sim_rad": sim_rad,
            "asd_params": asd_params,
            "sim_params": sim_params,
            "match_idx": [np.asarray(s["match_indices"], dtype=np.int64) for s in scenes],
            "match_dist": [
                np.asarray(s["match_distances"], dtype=np.float64) for s in scenes
            ],
            "topk_idx": top_idx,
            "topk_dist": top_dist,
        },
    }


def _sheet_name(sample: int, date: str) -> str:
    raw = f"S{sample}_{date}"
    raw = re.sub(r"[:\\/?*\[\]]", "_", raw)
    return raw[:31]


def write_neighbors_excel(result: dict[str, Any], xlsx_path: str | Path) -> Path:
    """Write summary + one sheet per ASD scene (ASD first row, sims below)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandas is required to write the neighbors Excel file") from exc

    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    id_cols = ["row_type", "date", "sim_index", "mean_normalized_rmse"]
    value_cols = list(QOI_COLS) + list(AUX_COLS) + list(STD_COLS)
    cols = id_cols + value_cols

    summary_rows = []
    for sc in result["scenes"]:
        row = {
            "sample": sc["sample"],
            "date": sc["date"],
            "n_matches": sc["n_matches"],
            "min_mean_normalized_rmse": sc["min_mean_normalized_rmse"],
            "median_mean_normalized_rmse": sc["median_mean_normalized_rmse"],
            "max_mean_normalized_rmse": sc["max_mean_normalized_rmse"],
        }
        for task, sp in sc["spans"].items():
            row[f"{task}_span"] = sp.get("span")
            row[f"{task}_min"] = sp.get("min")
            row[f"{task}_max"] = sp.get("max")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    try:
        writer_kwargs = {"engine": "openpyxl"}
        # Probe openpyxl early for a clear error.
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to write .xlsx files. Install with: pip install openpyxl"
        ) from exc

    with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for sc in result["scenes"]:
            rows = [sc["asd_row"], *sc["sim_rows"]]
            df = pd.DataFrame(rows)
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[cols]
            df.to_excel(writer, sheet_name=_sheet_name(sc["sample"], sc["date"]), index=False)

    return xlsx_path


def summary_for_json(result: dict[str, Any]) -> dict[str, Any]:
    """Strip heavy / non-serializable plot helpers for JSON export."""
    return {
        "asd_path": result["asd_path"],
        "sim_path": result["sim_path"],
        "thr_mean_normalized_rmse": result["thr_mean_normalized_rmse"],
        "n_asd": result["n_asd"],
        "n_sim": result["n_sim"],
        "dates": result["dates"],
        "scenes": [
            {
                "sample": sc["sample"],
                "date": sc["date"],
                "n_matches": sc["n_matches"],
                "min_mean_normalized_rmse": sc["min_mean_normalized_rmse"],
                "median_mean_normalized_rmse": sc["median_mean_normalized_rmse"],
                "max_mean_normalized_rmse": sc["max_mean_normalized_rmse"],
                "spans": sc["spans"],
                "asd": {
                    k: sc["asd_row"][k]
                    for k in (*QOI_COLS, *AUX_COLS, *STD_COLS)
                    if k in sc["asd_row"]
                },
            }
            for sc in result["scenes"]
        ],
    }


def plot_spectra_overlays(
    result: dict[str, Any],
    out_dir: str | Path,
    *,
    max_overlay: int = 8,
) -> list[Path]:
    """ASD vs neighbor spectra overlays (same style as ``asd_sim_nn_compare``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(result["thr_mean_normalized_rmse"])
    scenes = result["scenes"]
    dates = result["dates"]
    n = len(scenes)
    plot = result.get("_plot") or {}
    wl = plot.get("wl")
    asd_rad = plot.get("asd_rad")
    sim_rad = plot.get("sim_rad")
    if wl is None or asd_rad is None or sim_rad is None:
        with h5py.File(result["asd_path"], "r") as fa, h5py.File(result["sim_path"], "r") as fs:
            wl = np.asarray(fa["wl"][:], dtype=np.float64)
            asd_rad = np.asarray(fa["toa_radiance"][:], dtype=np.float64)
            sim_rad = np.asarray(fs["toa_radiance"][:], dtype=np.float64)

    match_idx = plot.get("match_idx") or [
        np.asarray(sc["match_indices"], dtype=np.int64) for sc in scenes
    ]
    match_dist = plot.get("match_dist") or [
        np.asarray(sc["match_distances"], dtype=np.float64) for sc in scenes
    ]
    topk_idx = plot.get("topk_idx")
    topk_dist = plot.get("topk_dist")
    if topk_idx is None:
        topk_idx = np.asarray(
            [sc.get("topk_indices", []) for sc in scenes], dtype=np.int64
        )
        topk_dist = np.asarray(
            [sc.get("topk_distances", []) for sc in scenes], dtype=np.float64
        )

    paths: list[Path] = []
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        ax.plot(wl, asd_rad[i], color="#4C78A8", lw=2.0, label="ASD", zorder=3)
        js = match_idx[i]
        ds = match_dist[i]
        used_fallback = False
        if js.size == 0:
            used_fallback = True
            js = np.asarray(topk_idx[i], dtype=np.int64)
            ds = np.asarray(topk_dist[i], dtype=np.float64)
            ok = js >= 0
            js, ds = js[ok], ds[ok]
        else:
            js = js[:max_overlay]
            ds = ds[:max_overlay]
        for r, (j, dist) in enumerate(zip(js, ds)):
            alpha = 0.85 - 0.12 * r
            ax.plot(
                wl,
                sim_rad[int(j)],
                lw=1.1,
                alpha=max(alpha, 0.35),
                label=f"NN{r + 1} (mnRMSE={float(dist):.3g})",
            )
        n_all = int(scenes[i]["n_matches"])
        if used_fallback:
            title = (
                f"S{i + 1}  {dates[i]}\n"
                f"no matches ≤ {thr:g}; closest {len(js)} shown"
            )
        else:
            title = (
                f"S{i + 1}  {dates[i]}\n"
                f"{n_all} match(es) ≤ {thr:g}"
                + (f"; showing {len(js)}" if n_all > len(js) else "")
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("TOA radiance")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols) : n]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "ASD vs synthetic neighbor spectra (mean-normalized RMSE)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    p_spec = out_dir / "00_spectra_neighbors_mnRMSE.png"
    fig.savefig(p_spec, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_spec)

    # NN1 residual grid (match within thr if any, else closest overall).
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        if match_idx[i].size:
            j = int(match_idx[i][0])
            dist = float(match_dist[i][0])
            tag = "match"
        else:
            j = int(topk_idx[i, 0])
            dist = float(topk_dist[i, 0])
            tag = "closest"
        resid = sim_rad[j] - asd_rad[i]
        ax.axhline(0.0, color="#888888", lw=0.9)
        ax.plot(wl, resid, color="#d62728", lw=1.2)
        ax.set_title(
            f"S{i + 1} {dates[i]}\n{tag} mnRMSE={dist:.3g}",
            fontsize=11,
        )
        ax.set_ylabel("sim − ASD")
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols) : n]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Nearest-neighbor spectral residual (sim − ASD)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    p_res = out_dir / "00b_nn1_spectral_residuals.png"
    fig.savefig(p_res, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_res)
    return paths


def load_asd_model_predictions(
    preds_path: str | Path | None,
) -> dict[str, np.ndarray] | None:
    """Load per-task ASD model predictions from ``asd_validation_predictions.npz``."""
    if preds_path is None:
        return None
    preds_path = Path(preds_path)
    candidates = [preds_path]
    try:
        candidates.append(Path("\\\\?\\" + str(preds_path.resolve())))
    except OSError:
        pass
    z = None
    for cand in candidates:
        try:
            z = np.load(cand, allow_pickle=True)
            break
        except OSError:
            continue
    if z is None:
        return None
    tasks = [str(t) for t in np.asarray(z["task_names"]).tolist()]
    y_pred = np.asarray(z["y_pred"], dtype=np.float64)
    if y_pred.ndim != 2 or y_pred.shape[1] != len(tasks):
        return None
    return {t: y_pred[:, i] for i, t in enumerate(tasks)}


def _log_ticks(lo: float, hi: float, n_target: int = 4) -> np.ndarray:
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.asarray([lo, hi], dtype=float)
    if hi / max(lo, 1e-12) >= 10:
        decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
        ticks = 10.0 ** decades
        ticks = ticks[(ticks >= lo * 0.999) & (ticks <= hi * 1.001)]
        if ticks.size >= 2:
            return ticks
    return np.geomspace(max(lo, 1e-12), hi, n_target)


def _fmt_log_tick(v: float, _pos=None) -> str:
    if v <= 0 or not np.isfinite(v):
        return ""
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _pred_value(model_preds: dict[str, np.ndarray], task: str, i: int) -> float:
    if task not in model_preds or i >= len(model_preds[task]):
        return float("nan")
    v = float(model_preds[task][i])
    return v if np.isfinite(v) else float("nan")


def plot_grain_y_pair_panels(
    *,
    out_path: Path,
    thr: float,
    dates: list[str],
    asd_params: dict[str, np.ndarray],
    sim_params: dict[str, np.ndarray],
    match_idx: list[np.ndarray],
    model_preds: dict[str, np.ndarray],
    y_key: str,
    y_label: str,
    y_std_key: str | None,
    color_key: str,
    color_label: str,
    color_vmin: float | None = None,
    color_vmax: float | None = None,
    title_prefix: str,
) -> Path | None:
    """One multi-panel figure: grain vs ``y_key`` for each ASD scene."""
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import FuncFormatter, NullLocator

    if y_key not in asd_params or y_key not in sim_params:
        return None
    if "grain_size" not in asd_params or "grain_size" not in sim_params:
        return None

    n = len(dates)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        constrained_layout=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )
    axes = np.atleast_1d(axes).ravel()
    sc_last = None
    use_log_color = color_key in {"dust", "algae", "lwc"} and color_key != y_key

    for i in range(n):
        ax = axes[i]
        js = match_idx[i]
        g_asd = float(asd_params["grain_size"][i])
        y_asd = float(asd_params[y_key][i])
        xerr = float(asd_params.get("grain_size_std", np.zeros(n))[i])
        yerr = (
            float(asd_params.get(y_std_key, np.zeros(n))[i])
            if y_std_key and y_std_key in asd_params
            else 0.0
        )

        if js.size:
            g = sim_params["grain_size"][js]
            y = sim_params[y_key][js]
            if color_key in sim_params and color_key != y_key:
                c = sim_params[color_key][js]
                cmin = color_vmin if color_vmin is not None else float(np.nanmin(c))
                cmax = color_vmax if color_vmax is not None else float(np.nanmax(c))
                if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
                    cmin, cmax = 1e-3, 1.0
                norm = (
                    LogNorm(vmin=max(cmin, 1e-3), vmax=max(cmax, max(cmin, 1e-3) * 1.1))
                    if use_log_color
                    else Normalize(vmin=cmin, vmax=cmax)
                )
                sc_last = ax.scatter(
                    g,
                    y,
                    c=c,
                    s=28,
                    cmap="viridis",
                    norm=norm,
                    alpha=0.8,
                    edgecolors="none",
                )
            else:
                sc_last = ax.scatter(
                    g,
                    y,
                    s=28,
                    color="#4C78A8",
                    alpha=0.8,
                    edgecolors="none",
                )
            grain_span = float(np.max(g) - np.min(g))
            g_all = np.concatenate([g, [g_asd]])
            y_all = np.concatenate([y, [y_asd]])
        else:
            grain_span = float("nan")
            g_all = np.array([g_asd])
            y_all = np.array([y_asd])

        g_pred = _pred_value(model_preds, "grain_size", i)
        y_pred = _pred_value(model_preds, y_key, i)
        if np.isfinite(g_pred):
            g_all = np.concatenate([g_all, [g_pred]])
        if np.isfinite(y_pred):
            y_all = np.concatenate([y_all, [y_pred]])

        xerr_plot = min(xerr, 0.45 * g_asd) if g_asd > 0 else xerr
        yerr_plot = min(yerr, 0.45 * abs(y_asd)) if abs(y_asd) > 0 else yerr
        ax.errorbar(
            g_asd,
            y_asd,
            xerr=xerr_plot,
            yerr=yerr_plot if yerr_plot > 0 else None,
            fmt="D",
            ms=9,
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.6,
            capsize=3,
            zorder=5,
            label="ASD ±1σ" if yerr_plot > 0 else "ASD",
            clip_on=True,
        )
        if np.isfinite(g_pred) and np.isfinite(y_pred):
            ax.plot(
                g_pred,
                y_pred,
                marker="*",
                ms=16,
                color="#F58518",
                markeredgecolor="k",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
                label="model pred",
            )
        elif np.isfinite(g_pred):
            ax.axvline(g_pred, color="#F58518", ls="--", lw=1.4, alpha=0.7, zorder=4)
            ax.plot(
                g_pred,
                y_asd,
                marker="*",
                ms=16,
                color="#F58518",
                markeredgecolor="k",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
                label="model grain",
            )
        elif np.isfinite(y_pred):
            ax.axhline(y_pred, color="#F58518", ls="--", lw=1.4, alpha=0.7, zorder=4)
            ax.plot(
                g_asd,
                y_pred,
                marker="*",
                ms=16,
                color="#F58518",
                markeredgecolor="k",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
                label=f"model {y_label}",
            )

        ax.set_xscale("log")
        # CWV/AOT are usually better on linear if range is small; keep log for
        # impurity-like quantities that span orders of magnitude.
        use_ylog = y_key in {"dust", "algae", "lwc", "grain_size"}
        if use_ylog:
            ax.set_yscale("log")
            y_floor = 1e-3
        else:
            y_floor = None

        g_lo = max(np.nanmin(g_all) * 0.7, 1e-3)
        g_hi = max(np.nanmax(g_all) * 1.4, g_lo * 1.5)
        if use_ylog:
            y_lo = max(np.nanmin(y_all) * 0.7, y_floor)
            y_hi = max(np.nanmax(y_all) * 1.4, y_lo * 1.5)
        else:
            y_pad = 0.15 * max(np.nanmax(y_all) - np.nanmin(y_all), abs(y_asd) * 0.2, 1e-3)
            y_lo = np.nanmin(y_all) - y_pad
            y_hi = np.nanmax(y_all) + y_pad
            if y_lo == y_hi:
                y_lo, y_hi = y_asd * 0.5, y_asd * 1.5

        if js.size == 0:
            g_lo = max(g_asd * 0.5, 1e-3)
            g_hi = g_asd * 2.0
            if use_ylog:
                y_lo = max(abs(y_asd) * 0.5, y_floor)
                y_hi = max(abs(y_asd) * 2.0, y_lo * 1.5)
            else:
                y_lo, y_hi = y_asd * 0.5, y_asd * 1.5

        ax.set_xlim(g_lo, g_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(_log_ticks(g_lo, g_hi))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_log_tick))
        ax.xaxis.set_minor_locator(NullLocator())
        if use_ylog:
            ax.set_yticks(_log_ticks(y_lo, y_hi))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log_tick))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.set_xlabel("grain (µm)")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"S{i + 1} {dates[i]}\n"
            f"n={js.size}  grain span={grain_span:.0f}µm"
            if np.isfinite(grain_span)
            else f"S{i + 1} {dates[i]}\nn=0"
        )
        ax.grid(alpha=0.25, which="major")
        if i == 0:
            ax.legend(loc="best", frameon=True, fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    if sc_last is not None and color_key != y_key and color_key in sim_params:
        cbar = fig.colorbar(
            sc_last,
            ax=axes[:n].tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.9,
        )
        cbar.set_label(color_label)
    fig.suptitle(
        f"{title_prefix} among sims with mean-normalized RMSE ≤ {thr:g}",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.08)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_neighbor_overview(
    result: dict[str, Any],
    out_dir: str | Path,
    *,
    model_preds: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Write readable neighbor overview plots; labels use mean-normalized RMSE."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(result["thr_mean_normalized_rmse"])
    scenes = result["scenes"]
    dates = result["dates"]
    n = len(scenes)
    plot = result.get("_plot") or {}
    asd_params = plot.get("asd_params") or {}
    sim_params = plot.get("sim_params") or {}
    match_idx = plot.get("match_idx") or [
        np.asarray(sc["match_indices"], dtype=np.int64) for sc in scenes
    ]
    model_preds = model_preds or {}

    paths: list[Path] = []
    paths.extend(plot_spectra_overlays(result, out_dir))

    # --- 01 match counts ---
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    counts = [sc["n_matches"] for sc in scenes]
    labels = [f"S{sc['sample']}\n{sc['date']}" for sc in scenes]
    bars = ax.bar(np.arange(n), counts, color="#4C78A8", edgecolor="none")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_ylabel("# synthetic matches")
    ax.set_title(
        f"Synthetic spectra with mean-normalized RMSE ≤ {thr:g} of each ASD scene"
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(c),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )
    fig.tight_layout()
    p1 = out_dir / "01_match_counts.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # --- 02 grain–Y pair panels (separate image per Y) ---
    if asd_params and sim_params:
        pair_specs = [
            {
                "filename": "02_grain_dust_pairs.png",
                "y_key": "dust",
                "y_label": "dust",
                "y_std_key": "dust_std",
                "color_key": "algae",
                "color_label": "algae",
                "color_vmin": 0.0,
                "color_vmax": 100.0,
                "title_prefix": "Grain–dust",
            },
            {
                "filename": "02b_grain_algae_pairs.png",
                "y_key": "algae",
                "y_label": "algae",
                "y_std_key": "algae_std",
                "color_key": "dust",
                "color_label": "dust",
                "color_vmin": None,
                "color_vmax": None,
                "title_prefix": "Grain–algae",
            },
            {
                "filename": "02c_grain_lwc_pairs.png",
                "y_key": "lwc",
                "y_label": "LWC",
                "y_std_key": "lwc_std",
                "color_key": "dust",
                "color_label": "dust",
                "color_vmin": None,
                "color_vmax": None,
                "title_prefix": "Grain–LWC",
            },
            {
                "filename": "02d_grain_cwv_pairs.png",
                "y_key": "cwv",
                "y_label": "CWV",
                "y_std_key": None,
                "color_key": "dust",
                "color_label": "dust",
                "color_vmin": None,
                "color_vmax": None,
                "title_prefix": "Grain–CWV",
            },
            {
                "filename": "02e_grain_aot_pairs.png",
                "y_key": "aot",
                "y_label": "AOT",
                "y_std_key": None,
                "color_key": "dust",
                "color_label": "dust",
                "color_vmin": None,
                "color_vmax": None,
                "title_prefix": "Grain–AOT",
            },
        ]
        for spec in pair_specs:
            pth = plot_grain_y_pair_panels(
                out_path=out_dir / spec["filename"],
                thr=thr,
                dates=dates,
                asd_params=asd_params,
                sim_params=sim_params,
                match_idx=match_idx,
                model_preds=model_preds,
                y_key=spec["y_key"],
                y_label=spec["y_label"],
                y_std_key=spec["y_std_key"],
                color_key=spec["color_key"],
                color_label=spec["color_label"],
                color_vmin=spec["color_vmin"],
                color_vmax=spec["color_vmax"],
                title_prefix=spec["title_prefix"],
            )
            if pth is not None:
                paths.append(pth)

        legacy = out_dir / "02_grain_dust_clouds.png"
        if legacy.is_file():
            try:
                legacy.unlink()
            except OSError:
                pass

    # --- 03 QoI spans ---
    tasks = ["grain_size", "dust", "algae", "lwc", "cos_i"]
    titles = {
        "grain_size": "Grain size (µm)",
        "dust": "Dust",
        "algae": "Algae",
        "lwc": "LWC",
        "cos_i": "cos_i",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8))
    axes = axes.ravel()
    x = np.arange(n)
    for ax, task in zip(axes, tasks):
        spans = [sc["spans"][task]["span"] for sc in scenes]
        ax.bar(x, spans, color="#72B7B2", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i + 1}" for i in range(n)])
        ax.set_ylabel("span among matches")
        ax.set_title(titles[task])
        ax.grid(axis="y", alpha=0.3)
        if task == "grain_size":
            ax.set_yscale("log")
    axes[-1].set_visible(False)
    fig.suptitle(
        f"QoI span among synthetic matches (mean-normalized RMSE ≤ {thr:g})",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p3 = out_dir / "03_qoi_spans.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths


def run_asd_spectral_neighbors(
    asd_path: str | Path,
    sim_path: str | Path,
    out_dir: str | Path,
    *,
    thr: float = DEFAULT_THR,
    write_excel: bool = True,
    write_plots: bool = True,
    write_json: bool = True,
    model_preds: dict[str, np.ndarray] | None = None,
    preds_path: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end: find neighbors, write Excel / plots / JSON under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ASD spectral neighbors: thr_mean_normalized_rmse={thr:g}  "
        f"asd={asd_path}  sim={sim_path}"
    )
    result = find_asd_sim_neighbors(asd_path, sim_path, thr=thr)
    for sc in result["scenes"]:
        print(
            f"  S{sc['sample']} {sc['date']}: n_matches={sc['n_matches']}  "
            f"min_mnRMSE={sc['min_mean_normalized_rmse']:.4g}"
        )

    if model_preds is None:
        cand = Path(preds_path) if preds_path is not None else (out_dir / "asd_validation_predictions.npz")
        model_preds = load_asd_model_predictions(cand)
        if model_preds:
            print(f"Loaded model preds for pair plots: {sorted(model_preds)}")

    artifacts: dict[str, Any] = {"out_dir": str(out_dir)}
    thr_tag = f"{thr:g}".replace(".", "p")
    if write_excel:
        xlsx = out_dir / f"asd_sim_neighbors_thr{thr_tag}.xlsx"
        write_neighbors_excel(result, xlsx)
        artifacts["excel"] = str(xlsx)
        print(f"Wrote Excel: {xlsx}")

    if write_plots:
        plot_dir = out_dir / "neighbor_plots"
        paths = plot_neighbor_overview(result, plot_dir, model_preds=model_preds)
        artifacts["plots"] = [str(p) for p in paths]
        print(f"Wrote {len(paths)} neighbor plots under {plot_dir}")

    if write_json:
        summary = summary_for_json(result)
        summary["artifacts"] = artifacts
        if model_preds:
            summary["model_pred_tasks"] = sorted(model_preds)
        jpath = out_dir / "asd_sim_neighbors_summary.json"
        jpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifacts["summary_json"] = str(jpath)
        print(f"Wrote summary JSON: {jpath}")

    result["artifacts"] = artifacts
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asd", type=Path, default=DEFAULT_ASD)
    ap.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--thr",
        type=float,
        default=DEFAULT_THR,
        help="mean-normalized RMSE threshold (default 0.04)",
    )
    ap.add_argument(
        "--preds",
        type=Path,
        default=None,
        help="Optional asd_validation_predictions.npz (default: <out-dir>/...)",
    )
    ap.add_argument("--no-excel", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    if not args.asd.is_file():
        raise SystemExit(f"ASD NetCDF not found: {args.asd}")
    if not args.sim.is_file():
        raise SystemExit(f"Sim NetCDF not found: {args.sim}")

    run_asd_spectral_neighbors(
        args.asd,
        args.sim,
        args.out_dir,
        thr=args.thr,
        write_excel=not args.no_excel,
        write_plots=not args.no_plots,
        write_json=not args.no_json,
        preds_path=args.preds,
    )


if __name__ == "__main__":
    main()
