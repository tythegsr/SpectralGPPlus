"""Residual structure check for Aug08 SORF+NIGP predictions (nonstationarity probe)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.linalg import svd

ROOT = Path(
    r"C:\Users\forty\tyler_gpplus\SpectralGPPlus\experiments_SORF\results\Aug08"
    r"\s2_toa_sorf_1inits_numrff1600_lr0.1_taskbandconfig_rbf_nigp_freezeepochnigp100_dtypefloat64"
)


def coverage(zvals: np.ndarray, k: float = 1.96) -> float:
    return float(np.mean(np.abs(zvals) <= k))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-15 or b.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def quintile_bins(values: np.ndarray, resid: np.ndarray, z_task: np.ndarray):
    qs = np.quantile(values, np.linspace(0, 1, 6))
    edges = np.unique(qs)
    if len(edges) < 3:
        return []
    bins = np.digitize(values, edges[1:-1], right=False)
    out = []
    for b in range(len(edges) - 1):
        m = bins == b
        if int(m.sum()) < 30:
            continue
        r = resid[m]
        out.append(
            {
                "bin": int(b),
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "rmse": float(np.sqrt(np.mean(r**2))),
                "mae": float(np.mean(np.abs(r))),
                "mean_resid": float(np.mean(r)),
                "mean_abs_z": float(np.mean(np.abs(z_task[m]))),
                "cov95": coverage(z_task[m]),
            }
        )
    return out


def main() -> None:
    npz = np.load(ROOT / "predictions_S2_TOA_nTrain40000_nTest5000_sorfD1600.npz", allow_pickle=True)
    tasks = [str(t) for t in npz["task_names"]]
    y_true = np.asarray(npz["y_true"], dtype=np.float64)
    y_pred = np.asarray(npz["y_pred"], dtype=np.float64)
    y_std = np.asarray(npz["y_std"], dtype=np.float64)
    x = np.asarray(npz["x_test"], dtype=np.float64)
    wl = np.asarray(npz["wavelengths_nm"], dtype=np.float64)

    resid = y_true - y_pred
    z = resid / np.clip(y_std, 1e-12, None)

    xc = x - x.mean(0, keepdims=True)
    u, s, _vt = svd(xc, full_matrices=False)
    scores = u[:, :3] * s[:3]
    ev = (s**2) / (s**2).sum()

    i_red = int(np.argmin(np.abs(wl - 665.0)))
    i_nir = int(np.argmin(np.abs(wl - 865.0)))
    ndvi_like = (x[:, i_nir] - x[:, i_red]) / (np.abs(x[:, i_nir] + x[:, i_red]) + 1e-8)
    mean_refl = x.mean(1)
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]

    charts = {
        "rrmse_by_task": [],
        "z_coverage_95": [],
        "rmse_ratio_y_quintiles": [],
        "rmse_ratio_pc1_quintiles": [],
    }
    per_task: dict = {}

    for ti, name in enumerate(tasks):
        yt = y_true[:, ti]
        yp = y_pred[:, ti]
        ys = y_std[:, ti]
        r = resid[:, ti]
        z_task = z[:, ti]
        az = np.abs(z_task)

        rmse = float(np.sqrt(np.mean(r**2)))
        mae = float(np.mean(np.abs(r)))
        rrmse = float(rmse / (np.std(yt) + 1e-12))

        y_bins = quintile_bins(yt, r, z_task)
        pc_bins = quintile_bins(pc1, r, z_task)

        y_rmses = [b["rmse"] for b in y_bins] or [rmse]
        pc_rmses = [b["rmse"] for b in pc_bins] or [rmse]
        y_ratio = float(max(y_rmses) / (min(y_rmses) + 1e-12))
        pc_ratio = float(max(pc_rmses) / (min(pc_rmses) + 1e-12))
        bias_span = (
            float(max(b["mean_resid"] for b in y_bins) - min(b["mean_resid"] for b in y_bins))
            if y_bins
            else 0.0
        )

        task_sum = {
            "rmse": rmse,
            "mae": mae,
            "rrmse": rrmse,
            "mean_resid": float(np.mean(r)),
            "std_resid": float(np.std(r)),
            "cov95": coverage(z_task),
            "mean_abs_z": float(np.mean(az)),
            "corr_abs_resid_vs_ytrue": corr(np.abs(r), yt),
            "corr_abs_resid_vs_ypred": corr(np.abs(r), yp),
            "corr_abs_resid_vs_ystd": corr(np.abs(r), ys),
            "corr_abs_resid_vs_pc1": corr(np.abs(r), pc1),
            "corr_abs_resid_vs_pc2": corr(np.abs(r), pc2),
            "corr_abs_resid_vs_mean_refl": corr(np.abs(r), mean_refl),
            "corr_abs_resid_vs_ndvi_like": corr(np.abs(r), ndvi_like),
            "rmse_ratio_y_quintiles": y_ratio,
            "rmse_ratio_pc1_quintiles": pc_ratio,
            "bias_span_y_quintiles": bias_span,
            "y_quintile_bins": y_bins,
            "pc1_quintile_bins": pc_bins,
        }
        per_task[name] = task_sum
        charts["rrmse_by_task"].append({"task": name, "rrmse": rrmse, "rmse": rmse, "cov95": task_sum["cov95"]})
        charts["z_coverage_95"].append(
            {"task": name, "cov95": task_sum["cov95"], "mean_abs_z": task_sum["mean_abs_z"]}
        )
        charts["rmse_ratio_y_quintiles"].append({"task": name, "ratio": y_ratio})
        charts["rmse_ratio_pc1_quintiles"].append({"task": name, "ratio": pc_ratio})

    flags = []
    for name, t in per_task.items():
        reasons = []
        if t["rmse_ratio_y_quintiles"] >= 2.0:
            reasons.append(f"y-quintile RMSE ratio {t['rmse_ratio_y_quintiles']:.2f}")
        if t["rmse_ratio_pc1_quintiles"] >= 1.75:
            reasons.append(f"PC1-quintile RMSE ratio {t['rmse_ratio_pc1_quintiles']:.2f}")
        if abs(t["corr_abs_resid_vs_pc1"]) >= 0.25 or abs(t["corr_abs_resid_vs_pc2"]) >= 0.25:
            reasons.append(
                f"|corr(|r|,PC)| pc1={t['corr_abs_resid_vs_pc1']:.2f} pc2={t['corr_abs_resid_vs_pc2']:.2f}"
            )
        if abs(t["corr_abs_resid_vs_ypred"]) >= 0.35:
            reasons.append(f"corr(|r|,yhat)={t['corr_abs_resid_vs_ypred']:.2f}")
        if t["cov95"] < 0.85 or t["cov95"] > 0.99:
            reasons.append(f"95% z-coverage {t['cov95']:.3f}")
        if reasons:
            flags.append({"task": name, "reasons": reasons, "rrmse": t["rrmse"]})

    out = {
        "meta": {
            "run_dir": str(ROOT),
            "title": "S2_TOA_nTrain40000_nTest5000_sorfD1600",
            "n_test": int(y_true.shape[0]),
            "tasks": tasks,
            "pca_explained_frac_pc1_3": [float(ev[i]) for i in range(3)],
            "wl_red_nm": float(wl[i_red]),
            "wl_nir_nm": float(wl[i_nir]),
        },
        "overall": {
            "n_tasks_flagged": len(flags),
            "flags": flags,
            "mean_rrmse": float(np.mean([per_task[t]["rrmse"] for t in tasks])),
            "mean_cov95": float(np.mean([per_task[t]["cov95"] for t in tasks])),
            "mean_rmse_ratio_y": float(
                np.mean([per_task[t]["rmse_ratio_y_quintiles"] for t in tasks])
            ),
            "mean_rmse_ratio_pc1": float(
                np.mean([per_task[t]["rmse_ratio_pc1_quintiles"] for t in tasks])
            ),
        },
        "charts": charts,
        "tasks": per_task,
    }

    out_path = ROOT / "residual_nonstationarity_check.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)
    print("FLAGGED", json.dumps(flags, indent=2))
    print("mean rrmse", out["overall"]["mean_rrmse"])
    print("mean cov95", out["overall"]["mean_cov95"])
    print("mean rmse ratio y", out["overall"]["mean_rmse_ratio_y"])
    print("mean rmse ratio pc1", out["overall"]["mean_rmse_ratio_pc1"])
    for t in tasks:
        d = per_task[t]
        print(
            f"{t:14s} RRMSE={d['rrmse']:.4f} cov95={d['cov95']:.3f} "
            f"yRatio={d['rmse_ratio_y_quintiles']:.2f} pc1Ratio={d['rmse_ratio_pc1_quintiles']:.2f} "
            f"corr|r|-pc1={d['corr_abs_resid_vs_pc1']:+.3f} corr|r|-yhat={d['corr_abs_resid_vs_ypred']:+.3f}"
        )


if __name__ == "__main__":
    main()
