"""Diagnose ASD field validation vs synthetic snow-TOA for coverage / domain-shift failures.

Compares asd_validation_set.nc against snow_toa_fsnow_90to100_*.nc, quantifies QoI gaps,
spectral nearest-neighbor degeneracy, AOT spectral leverage, and optional ASD eval coverage
linkage. Writes a JSON summary suitable for canvas / report refresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_20260309.nc"
DEFAULT_METRICS = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Aug29"
    / "s4_emit_aotbelow02_sorf_1inits_numrff1000_lr0.1_nigp_freezeepochnigp200_sloperefreshes15_dtypefloat32"
    / "asd_validation_eval"
    / "asd_validation_metrics.json"
)
DEFAULT_OUT = _ROOT / "_asd_vs_sim_analysis.json"

QOI_PAIRS = [
    ("grain_radius_mean", "grain_size", "grain_size"),
    ("cos_i", "cos_i", "cos_i"),
    ("dust_conc_mean", "dust", "dust"),
    ("algae_conc_mean", "algae", "algae"),
    ("cwv", "cwv", "cwv"),
    ("lwc_mean", "liquid_water", "liquid_water"),
    ("aod", "aot", "aot"),
]

GRAIN_FINE_THRESHOLD = 400.0


def _decode_attr(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _decode_attr(v.item())
        if v.dtype.kind in "SU":
            return [_decode_attr(x) for x in v.tolist()]
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _pct_dict(x: np.ndarray, ps=(1, 5, 50, 95, 99)) -> dict[str, float]:
    vals = np.percentile(np.asarray(x, float), ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float).reshape(-1)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        **_pct_dict(x),
    }


def _asd_ranks_in_sim(a: np.ndarray, s: np.ndarray) -> list[float]:
    """Percentile rank of each ASD value within the sim sample (0–100)."""
    s_sorted = np.sort(np.asarray(s, float).reshape(-1))
    n = len(s_sorted)
    ranks: list[float] = []
    for v in np.asarray(a, float).reshape(-1):
        ranks.append(float(100.0 * np.searchsorted(s_sorted, v, side="right") / n))
    return ranks


def _rel_rmse_rows(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Relative RMSE: ||g - q||_rms / mean(q) for each query row vs all gallery rows."""
    q = np.asarray(query, float)
    g = np.asarray(gallery, float)
    scale = np.maximum(q.mean(axis=1, keepdims=True), 1e-6)
    # (n_q, n_g)
    return np.sqrt(((g[None, :, :] - q[:, None, :]) ** 2).mean(axis=2)) / scale


def _load_sorted_subset(
    ds: h5py.Dataset, idx: np.ndarray, dtype=float
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    vals = np.asarray(ds[sorted_idx], dtype)
    out = np.empty_like(vals)
    out[order] = vals
    return out


def _coverage_summary(metrics_path: Path | None) -> dict[str, Any] | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_task: dict[str, Any] = {}
    for task, d in raw.get("per_task", {}).items():
        yt = np.asarray(d["y_true"], float)
        yp = np.asarray(d["y_pred"], float)
        ys = np.asarray(d["y_std"], float)
        err = np.abs(yt - yp)
        r2_key = f"{task}_R2"
        per_task[task] = {
            "coverage_50": float(d.get("coverage_50", np.nan)),
            "coverage_90": float(d.get("coverage_90", np.nan)),
            "coverage_95": float(d.get("coverage_95", np.nan)),
            "RRMSE": float(d["RRMSE"]),
            "R2": float(d.get(r2_key, d.get("R2", np.nan))),
            "mean_y_std": float(ys.mean()),
            "mean_abs_err": float(err.mean()),
            "median_abs_err_over_std": float(np.median(err / np.maximum(ys, 1e-12))),
            "y_true": yt.tolist(),
            "y_pred": yp.tolist(),
            "y_std": ys.tolist(),
        }
    return {
        "metrics_path": str(metrics_path),
        "title": raw.get("title"),
        "aggregate_RRMSE": raw.get("aggregate_RRMSE"),
        "per_task": per_task,
    }


def diagnose(
    asd_path: Path,
    sim_path: Path,
    *,
    metrics_path: Path | None = DEFAULT_METRICS,
    nn_subsample: int = 40000,
    deg_subsample: int = 5000,
    deg_queries: int = 200,
    deg_thresh: float = 0.02,
    aot_subsample: int = 30000,
    nn_k: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "asd_path": str(asd_path),
        "sim_path": str(sim_path),
        "seed": seed,
    }

    with h5py.File(asd_path, "r") as fa, h5py.File(sim_path, "r") as fs:
        out["n_asd"] = int(fa["toa_radiance"].shape[0])
        out["n_sim"] = int(fs["toa_radiance"].shape[0])
        out["sim_attrs"] = {k: _decode_attr(fs.attrs[k]) for k in fs.attrs}
        out["asd_attrs"] = {k: _decode_attr(fa.attrs[k]) for k in fa.attrs}

        dates = [
            d.decode() if isinstance(d, bytes) else str(d) for d in fa["date"][:]
        ]
        wl = np.asarray(fa["wl"][:], float)
        out["dates"] = dates
        out["wl"] = wl.tolist()

        # --- QoI ranges / ranks ---
        qoi_block: dict[str, Any] = {}
        flags: list[str] = []
        for akey, skey, name in QOI_PAIRS:
            a = np.asarray(fa[akey][:], float)
            s = np.asarray(fs[skey][:], float)
            ranks = _asd_ranks_in_sim(a, s)
            entry = {
                "asd_key": akey,
                "sim_key": skey,
                "asd_values": a.tolist(),
                "asd": _stats(a),
                "sim": _stats(s),
                "asd_percentile_ranks_in_sim": ranks,
                "frac_asd_in_sim_range": float(
                    np.mean((a >= s.min()) & (a <= s.max()))
                ),
            }
            qoi_block[name] = entry
            if np.median(a) < np.percentile(s, 5) or np.median(a) > np.percentile(s, 95):
                flags.append(
                    f"{name}: ASD median {float(np.median(a)):.4g} outside sim 5-95% "
                    f"[{float(np.percentile(s, 5)):.4g}, {float(np.percentile(s, 95)):.4g}]"
                )
            if name == "algae" and float(np.max(a)) < float(np.percentile(s, 5)):
                flags.append(
                    f"algae: ASD max {float(np.max(a)):.4g} below sim 5th pct "
                    f"{float(np.percentile(s, 5)):.4g} (sim median {float(np.median(s)):.4g})"
                )
        out["qoi"] = qoi_block

        # --- Geometry ---
        sza_a = np.asarray(fa["sza"][:], float)
        coszen_a = np.cos(np.radians(sza_a))
        cos_i_a = np.asarray(fa["cos_i"][:], float)
        coszen_s = np.asarray(fs["coszen"][:], float)
        cos_i_s = np.asarray(fs["cos_i"][:], float)
        ele_a = np.asarray(fa["elevation_km"][:], float)
        ele_s = np.asarray(fs["ele_km"][:], float)
        slope_s = np.asarray(fs["slope"][:], float) if "slope" in fs else None
        raa_a = np.asarray(fa["RAA"][:], float)
        raa_s = np.asarray(fs["RAA_TRUE"][:], float)

        geom = {
            "asd_flat_cos_i_equals_coszen": bool(np.allclose(cos_i_a, coszen_a)),
            "asd_sza": _stats(sza_a),
            "asd_coszen": _stats(coszen_a),
            "asd_cos_i": _stats(cos_i_a),
            "asd_ele_km": _stats(ele_a),
            "asd_raa": _stats(raa_a),
            "sim_coszen": _stats(coszen_s),
            "sim_cos_i": _stats(cos_i_s),
            "sim_ele_km": _stats(ele_s),
            "sim_raa": _stats(raa_s),
            "sim_frac_abs_cosi_minus_coszen_lt_0p01": float(
                np.mean(np.abs(cos_i_s - coszen_s) < 0.01)
            ),
        }
        if slope_s is not None:
            geom["sim_slope"] = _stats(slope_s)
            geom["sim_flat_slope_lt_2deg_frac"] = float(np.mean(slope_s < 2.0))
            if geom["sim_flat_slope_lt_2deg_frac"] < 0.25:
                flags.append(
                    f"geometry: sim flat-slope fraction {geom['sim_flat_slope_lt_2deg_frac']:.3f} "
                    "(ASD is flat; FLAT_SLOPE_WEIGHT~0.15)"
                )
        if float(np.std(ele_a)) < 1e-6 and (
            float(ele_a[0]) < np.percentile(ele_s, 20)
            or float(ele_a[0]) > np.percentile(ele_s, 80)
        ):
            # still flag fixed elev vs broad sim even if inside mid range
            pass
        if float(np.std(ele_a)) < 1e-6:
            flags.append(
                f"elevation: ASD fixed at {float(ele_a[0]):.2f} km; sim spans "
                f"[{float(ele_s.min()):.2f}, {float(ele_s.max()):.2f}] km"
            )
        asd_rt = out["asd_attrs"].get("RT_atmosphere")
        sim_rt = out["sim_attrs"].get("RT_atmosphere")
        if asd_rt and sim_rt and str(asd_rt) != str(sim_rt):
            flags.append(f"RT mismatch: ASD={asd_rt} vs sim={sim_rt}")
        out["geometry"] = geom

        # --- Algae / dust conditionals ---
        algae_s = np.asarray(fs["algae"][:], float)
        dust_s = np.asarray(fs["dust"][:], float)
        grain_s = np.asarray(fs["grain_size"][:], float)
        fine = grain_s < GRAIN_FINE_THRESHOLD
        algae_a = np.asarray(fa["algae_conc_mean"][:], float)
        dust_a = np.asarray(fa["dust_conc_mean"][:], float)

        joint = (
            (algae_s >= 10)
            & (algae_s <= 30)
            & (dust_s >= 40)
            & (dust_s <= 750)
            & (ele_s >= 2.0)
            & (ele_s <= 3.5)
        )
        if slope_s is not None:
            joint = joint & (slope_s <= 2.0)

        conditional = {
            "grain_fine_threshold": GRAIN_FINE_THRESHOLD,
            "fine_grain_frac": float(fine.mean()),
            "algae_in_10_30_frac": float(np.mean((algae_s >= 10) & (algae_s <= 30))),
            "algae_in_1_100_frac": float(np.mean((algae_s >= 1) & (algae_s <= 100))),
            "algae_fine": _stats(algae_s[fine]) if fine.any() else None,
            "algae_coarse": _stats(algae_s[~fine]) if (~fine).any() else None,
            "dust_fine": _stats(dust_s[fine]) if fine.any() else None,
            "dust_coarse": _stats(dust_s[~fine]) if (~fine).any() else None,
            "joint_asd_like": {
                "definition": "algae[10,30], dust[40,750], elev[2,3.5], slope<=2 if present",
                "frac": float(joint.mean()),
                "n": int(joint.sum()),
            },
            "asd_algae": algae_a.tolist(),
            "asd_dust": dust_a.tolist(),
        }
        if conditional["joint_asd_like"]["frac"] < 0.01:
            flags.append(
                f"joint ASD-like density only {100 * conditional['joint_asd_like']['frac']:.3f}% "
                f"(n={conditional['joint_asd_like']['n']})"
            )
        out["conditional"] = conditional

        # --- Spectral mean shift ---
        rad_a = np.asarray(fa["toa_radiance"][:], float)
        n_spec = min(8000, out["n_sim"])
        idx_spec = np.sort(rng.choice(out["n_sim"], n_spec, replace=False))
        rad_s_mean = np.asarray(fs["toa_radiance"][idx_spec], float).mean(axis=0)
        mean_a = rad_a.mean(axis=0)
        diff = mean_a - rad_s_mean
        ib = int(np.argmax(np.abs(diff)))
        out["spectrum"] = {
            "asd_mean_radiance": float(mean_a.mean()),
            "sim_mean_radiance": float(rad_s_mean.mean()),
            "largest_mean_diff": float(diff[ib]),
            "largest_mean_diff_wl_nm": float(wl[ib]),
            "asd_per_scene_mean_radiance": rad_a.mean(axis=1).tolist(),
            "asd_rmse_mean": np.asarray(fa["rmse_mean"][:], float).tolist(),
            "asd_sample_count": np.asarray(fa["sample_count"][:], float).tolist(),
        }

        # --- Nearest neighbors ---
        n_nn = min(nn_subsample, out["n_sim"])
        idx_nn = np.sort(rng.choice(out["n_sim"], n_nn, replace=False))
        rad_nn = np.asarray(fs["toa_radiance"][idx_nn], float)
        nn_qoi_keys = [
            "grain_size",
            "cos_i",
            "dust",
            "algae",
            "liquid_water",
            "cwv",
            "aot",
            "fsnow",
            "ele_km",
            "sza",
            "slope",
        ]
        nn_qois = {
            k: np.asarray(fs[k][idx_nn], float) for k in nn_qoi_keys if k in fs
        }
        D = _rel_rmse_rows(rad_a, rad_nn)
        asd_qoi = {
            "grain_size": np.asarray(fa["grain_radius_mean"][:], float),
            "cos_i": cos_i_a,
            "dust": dust_a,
            "algae": algae_a,
            "liquid_water": np.asarray(fa["lwc_mean"][:], float),
            "cwv": np.asarray(fa["cwv"][:], float),
            "aot": np.asarray(fa["aod"][:], float),
        }
        nn_scenes: list[dict[str, Any]] = []
        for i, date in enumerate(dates):
            order = np.argsort(D[i])
            nn = order[:nn_k]
            scene: dict[str, Any] = {
                "date": date,
                "best_rel_rmse": float(D[i, nn[0]]),
                "median_rel_rmse_topk": float(np.median(D[i, nn])),
                "asd_truth": {k: float(v[i]) for k, v in asd_qoi.items()},
                "nn_qoi": {},
            }
            for k, arr in nn_qois.items():
                v = arr[nn]
                scene["nn_qoi"][k] = {
                    "median": float(np.median(v)),
                    "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90)),
                    "std": float(np.std(v)),
                }
            nn_scenes.append(scene)
        out["nearest_neighbors"] = {
            "subsample": n_nn,
            "k": nn_k,
            "scenes": nn_scenes,
        }

        # --- Intra-sim degeneracy ---
        n_d = min(deg_subsample, out["n_sim"])
        idx_d = np.sort(rng.choice(out["n_sim"], n_d, replace=False))
        rad_d = np.asarray(fs["toa_radiance"][idx_d], float)
        qd_keys = [
            "grain_size",
            "cos_i",
            "dust",
            "algae",
            "liquid_water",
            "cwv",
            "aot",
        ]
        qd = {k: np.asarray(fs[k][idx_d], float) for k in qd_keys}
        qix = rng.choice(n_d, min(deg_queries, n_d), replace=False)
        deg_rel: dict[str, list[float]] = {k: [] for k in qd}
        deg_abs: dict[str, list[float]] = {k: [] for k in qd}
        dust_delta: list[float] = []
        algae_delta: list[float] = []
        close = 0
        for qi in qix:
            scale = float(rad_d[qi].mean()) + 1e-6
            dd = np.sqrt(((rad_d - rad_d[qi]) ** 2).mean(axis=1)) / scale
            dd[qi] = np.inf
            j = int(np.argmin(dd))
            if dd[j] < deg_thresh:
                close += 1
                for k in qd:
                    abs_d = abs(float(qd[k][j] - qd[k][qi]))
                    rel_d = abs_d / (abs(float(qd[k][qi])) + 1e-6)
                    deg_abs[k].append(abs_d)
                    deg_rel[k].append(rel_d)
                dust_delta.append(float(qd["dust"][j] - qd["dust"][qi]))
                algae_delta.append(float(qd["algae"][j] - qd["algae"][qi]))

        deg_summary: dict[str, Any] = {
            "subsample": n_d,
            "n_queries": int(len(qix)),
            "rel_rmse_thresh": deg_thresh,
            "n_close": close,
            "frac_close": float(close / max(len(qix), 1)),
            "per_qoi": {},
        }
        for k in qd:
            if deg_rel[k]:
                rr = np.asarray(deg_rel[k], float)
                aa = np.asarray(deg_abs[k], float)
                deg_summary["per_qoi"][k] = {
                    "median_rel": float(np.median(rr)),
                    "p90_rel": float(np.percentile(rr, 90)),
                    "median_abs": float(np.median(aa)),
                    "p90_abs": float(np.percentile(aa, 90)),
                    "frac_rel_gt_0p5": float(np.mean(rr > 0.5)),
                }
        if len(dust_delta) >= 3:
            deg_summary["dust_algae_delta_corr"] = float(
                np.corrcoef(dust_delta, algae_delta)[0, 1]
            )
        out["degeneracy"] = deg_summary

        # --- AOT sensitivity ---
        n_a = min(aot_subsample, out["n_sim"])
        idx_a = np.sort(rng.choice(out["n_sim"], n_a, replace=False))
        rad_aot = np.asarray(fs["toa_radiance"][idx_a], float)
        aot_s = np.asarray(fs["aot"][idx_a], float)
        grain_aot = np.asarray(fs["grain_size"][idx_a], float)
        cosi_aot = np.asarray(fs["cos_i"][idx_a], float)
        mask_low = (
            (aot_s < 0.025)
            & (grain_aot > 400)
            & (grain_aot < 1000)
            & (cosi_aot > 0.5)
            & (cosi_aot < 0.8)
        )
        mask_hi = (
            (aot_s > 0.055)
            & (grain_aot > 400)
            & (grain_aot < 1000)
            & (cosi_aot > 0.5)
            & (cosi_aot < 0.8)
        )
        aot_block: dict[str, Any] = {
            "subsample": n_a,
            "bin_definition": "grain 400–1000, cos_i 0.5–0.8; low AOT<0.025 vs high>0.055",
            "n_low": int(mask_low.sum()),
            "n_high": int(mask_hi.sum()),
        }
        if mask_low.sum() > 20 and mask_hi.sum() > 20:
            mlo = rad_aot[mask_low].mean(axis=0)
            mhi = rad_aot[mask_hi].mean(axis=0)
            dspec = mhi - mlo
            top = np.argsort(np.abs(dspec))[-8:][::-1]
            aot_block.update(
                {
                    "mean_rad_low": float(mlo.mean()),
                    "mean_rad_high": float(mhi.mean()),
                    "rel_mean_abs_delta": float(
                        np.mean(np.abs(dspec))
                        / (0.5 * (mlo.mean() + mhi.mean()) + 1e-12)
                    ),
                    "delta_spectrum": dspec.tolist(),
                    "top_bands_nm": wl[top].tolist(),
                    "top_band_deltas": dspec[top].tolist(),
                    "mean_spectrum_low": mlo.tolist(),
                    "mean_spectrum_high": mhi.tolist(),
                }
            )
            if aot_block["rel_mean_abs_delta"] < 0.03:
                flags.append(
                    f"AOT spectral leverage weak: rel mean|delta|="
                    f"{aot_block['rel_mean_abs_delta']:.4f} between low/high AOT bins"
                )

        # Algae distinguishability in same grain/dust bin
        algae_aot = np.asarray(fs["algae"][idx_a], float)
        dust_aot = np.asarray(fs["dust"][idx_a], float)
        low_alg = (
            (algae_aot >= 10)
            & (algae_aot <= 40)
            & (grain_aot > 500)
            & (grain_aot < 1200)
            & (dust_aot > 50)
            & (dust_aot < 500)
        )
        high_alg = (
            (algae_aot > 5000)
            & (grain_aot > 500)
            & (grain_aot < 1200)
            & (dust_aot > 50)
            & (dust_aot < 500)
        )
        algae_spec: dict[str, Any] = {
            "n_low": int(low_alg.sum()),
            "n_high": int(high_alg.sum()),
        }
        if low_alg.sum() > 3 and high_alg.sum() > 20:
            ml = rad_aot[low_alg].mean(axis=0)
            mh = rad_aot[high_alg].mean(axis=0)
            dd = mh - ml
            top = np.argsort(np.abs(dd))[-5:][::-1]
            algae_spec.update(
                {
                    "mean_rad_low": float(ml.mean()),
                    "mean_rad_high": float(mh.mean()),
                    "rel_mean_abs_delta": float(
                        np.mean(np.abs(dd)) / (0.5 * (ml.mean() + mh.mean()) + 1e-12)
                    ),
                    "top_bands_nm": wl[top].tolist(),
                    "top_band_deltas": dd[top].tolist(),
                }
            )
        out["aot_sensitivity"] = aot_block
        out["algae_spectral_contrast"] = algae_spec

        # --- Retrieval meta ---
        out["asd_retrieval_meta"] = {
            "sample_count": np.asarray(fa["sample_count"][:], float).tolist(),
            "rmse_mean": np.asarray(fa["rmse_mean"][:], float).tolist(),
            "rmse_std": np.asarray(fa["rmse_std"][:], float).tolist(),
            "grain_radius_std": np.asarray(fa["grain_radius_std"][:], float).tolist(),
            "algae_conc_std": np.asarray(fa["algae_conc_std"][:], float).tolist(),
            "dust_conc_std": np.asarray(fa["dust_conc_std"][:], float).tolist(),
            "lwc_std": np.asarray(fa["lwc_std"][:], float).tolist(),
        }

        out["flags"] = flags

    cov = _coverage_summary(metrics_path)
    if cov is not None:
        out["asd_eval_coverage"] = cov

    out["recommendations"] = [
        "Apply ASD-like low algae (e.g. logU[1,100] or [5,50]) for coarse grain too, not only grain<400.",
        "Raise FLAT_SLOPE_WEIGHT (0.15 → ~0.4+) and/or add a stratified flat ASD-like subset.",
        "Concentrate elevation near 2.0–3.5 km (Lake Mary–like) instead of uniform [0, 5].",
        "Oversample joint flat + elev~2.7 + algae~20 + dust in observed ASD range.",
        "Keep AOT in [0.01, 0.07] for clean-air ASD but treat it as weakly identifiable; align logit bounds with the sim window.",
        "Reduce dust/algae spectral interchangeability via correlated or physics-informed impurity sampling.",
        "Inject radiance / retrieval-label noise comparable to ASD rmse_* so predictive variance grows off the sim manifold.",
        "Document or reduce MODTRAN6 (field) vs sRTMnet (sim) RT residual.",
    ]
    return out


def _print_summary(result: dict[str, Any]) -> None:
    print(f"ASD n={result['n_asd']}  Sim n={result['n_sim']}")
    print(f"Sim: {Path(result['sim_path']).name}")
    print(f"ASD: {Path(result['asd_path']).name}")
    if "sampling" in result.get("sim_attrs", {}):
        print(f"Sampling: {result['sim_attrs']['sampling']}")

    print("\n=== QoI ASD percentile ranks in sim ===")
    for name, block in result["qoi"].items():
        ranks = block["asd_percentile_ranks_in_sim"]
        print(
            f"  {name:14s} ASD med={block['asd']['median']:.4g}  "
            f"sim med={block['sim']['median']:.4g}  "
            f"ranks%={np.round(ranks, 1)}"
        )

    cond = result["conditional"]
    print("\n=== Conditional / joint ===")
    print(f"  fine-grain frac={cond['fine_grain_frac']:.3f}")
    if cond["algae_fine"] and cond["algae_coarse"]:
        print(
            f"  algae med fine={cond['algae_fine']['median']:.4g}  "
            f"coarse={cond['algae_coarse']['median']:.4g}"
        )
    print(
        f"  joint ASD-like frac={cond['joint_asd_like']['frac']:.4%} "
        f"n={cond['joint_asd_like']['n']}"
    )

    print("\n=== Nearest neighbors (best relRMSE / algae NN spread) ===")
    for sc in result["nearest_neighbors"]["scenes"]:
        alg = sc["nn_qoi"].get("algae", {})
        print(
            f"  {sc['date']}: best={sc['best_rel_rmse']:.4f}  "
            f"algae NN p10-p90=[{alg.get('p10', float('nan')):.4g}, "
            f"{alg.get('p90', float('nan')):.4g}]  "
            f"ASD algae={sc['asd_truth']['algae']:.2f}"
        )

    deg = result["degeneracy"]
    print(
        f"\n=== Degeneracy close={deg['n_close']}/{deg['n_queries']} "
        f"(relRMSE<{deg['rel_rmse_thresh']}) ==="
    )
    for k, v in deg["per_qoi"].items():
        print(
            f"  {k:14s} med_rel={v['median_rel']:.3f} p90_rel={v['p90_rel']:.3f} "
            f"frac_rel>0.5={v['frac_rel_gt_0p5']:.2f}"
        )
    if "dust_algae_delta_corr" in deg:
        print(f"  dust-algae delta corr={deg['dust_algae_delta_corr']:.3f}")

    aot = result["aot_sensitivity"]
    print("\n=== AOT sensitivity ===")
    print(f"  n_low={aot.get('n_low')} n_high={aot.get('n_high')}")
    if "rel_mean_abs_delta" in aot:
        print(f"  rel mean|delta|={aot['rel_mean_abs_delta']:.4f}")
        print(f"  top bands nm={np.round(aot['top_bands_nm'], 1)}")

    cov = result.get("asd_eval_coverage")
    if cov:
        print("\n=== ASD eval coverage (linked) ===")
        for task, d in cov["per_task"].items():
            print(
                f"  {task:12s} cov95={d['coverage_95']:.2f} "
                f"med|err|/std={d['median_abs_err_over_std']:.1f} "
                f"RRMSE={d['RRMSE']:.3g} R2={d['R2']:.3g}"
            )

    if result["flags"]:
        print("\n=== FLAGS ===")
        for f in result["flags"]:
            print(f"  ! {f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asd-path", type=Path, default=DEFAULT_ASD)
    p.add_argument("--sim-path", type=Path, default=DEFAULT_SIM)
    p.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--nn-subsample", type=int, default=40000)
    p.add_argument("--deg-subsample", type=int, default=5000)
    p.add_argument("--deg-queries", type=int, default=200)
    p.add_argument("--aot-subsample", type=int, default=30000)
    p.add_argument("--nn-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-metrics", action="store_true")
    args = p.parse_args()

    metrics = None if args.no_metrics else args.metrics_path
    result = diagnose(
        args.asd_path,
        args.sim_path,
        metrics_path=metrics,
        nn_subsample=args.nn_subsample,
        deg_subsample=args.deg_subsample,
        deg_queries=args.deg_queries,
        aot_subsample=args.aot_subsample,
        nn_k=args.nn_k,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    _print_summary(result)


if __name__ == "__main__":
    main()
