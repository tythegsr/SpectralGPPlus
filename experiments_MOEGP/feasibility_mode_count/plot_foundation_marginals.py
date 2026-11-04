"""Conditional marginals (per-probe, per-model) and posterior-vs-data plots."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from load_bimodal_data import (
    BIMODAL_C1,
    BIMODAL_C2,
    BIMODAL_M1,
    BIMODAL_M2,
    BIMODAL_NOISE_STD,
    bimodal_mode_means,
)
from load_trimodal_data import TRIMODAL_C, TRIMODAL_M, TRIMODAL_SIGMA, trimodal_mode_means
from load_lognormal_hetero_data import a06_mean
from load_unimodal_hetero_data import (
    A05_EPS_CONTAM,
    A05_KAPPA,
    A05_TAU_MAX,
    A05_TAU_MIN,
    A05_X_HI,
    A05_X_LO,
    a05_mean,
    a05_noise_scale,
)
from matplotlib.colors import LinearSegmentedColormap


def _mixture_pdf(
    y: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    dens = np.zeros_like(y, dtype=np.float64)
    for j in range(means.shape[0]):
        dens += pi[j] * norm.pdf(y, means[j], max(float(stds[j]), 1e-6))
    return dens


def mixture_mean_std(
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture mean and std: Var = Σ π(σ²+μ²) − μ_mix²."""
    em = np.asarray(expert_means, dtype=np.float64)
    es = np.asarray(expert_stds, dtype=np.float64)
    gw = np.clip(np.asarray(gate_weights, dtype=np.float64), 0.0, None)
    gw = gw / np.clip(gw.sum(axis=-1, keepdims=True), 1e-12, None)
    mu = (em * gw).sum(axis=-1)
    second = (gw * (es**2 + em**2)).sum(axis=-1)
    var = np.clip(second - mu**2, 1e-12, None)
    return mu, np.sqrt(var)


def true_pdf_a01(y: np.ndarray, x: float) -> np.ndarray:
    m1 = BIMODAL_M1 * x + BIMODAL_C1
    m2 = BIMODAL_M2 * x + BIMODAL_C2
    return 0.5 * (
        norm.pdf(y, m1, BIMODAL_NOISE_STD) + norm.pdf(y, m2, BIMODAL_NOISE_STD)
    )


def true_pdf_unimodal(y: np.ndarray, x: float) -> np.ndarray:
    return norm.pdf(y, 0.5 * x + 2.0, 0.3)


def true_pdf_a02(y: np.ndarray, x: float, *, noise: float = 0.05) -> np.ndarray:
    mu = 1.0 if x >= 0.5 else 0.0
    return norm.pdf(y, mu, noise)


def true_pdf_a04(y: np.ndarray, x: float) -> np.ndarray:
    dens = np.zeros_like(y, dtype=np.float64)
    for m, c, s in zip(TRIMODAL_M, TRIMODAL_C, TRIMODAL_SIGMA, strict=True):
        dens += (1.0 / 3.0) * norm.pdf(y, m * x + c, s)
    return dens


def true_pdf_a05(y: np.ndarray, x: float) -> np.ndarray:
    mu = float(a05_mean(np.asarray([x]))[0])
    sigma = float(a05_noise_scale(np.asarray([x]))[0])
    w = np.clip((x - A05_X_LO) / max(A05_X_HI - A05_X_LO, 1e-12), 0.0, 1.0)
    tau = float(np.clip(A05_TAU_MIN + (A05_TAU_MAX - A05_TAU_MIN) * w, 1e-3, 1.0 - 1e-3))
    b_pos = sigma / np.sqrt(2.0 * tau)
    b_neg = sigma / np.sqrt(2.0 * (1.0 - tau))
    mean_shift = tau * b_pos - (1.0 - tau) * b_neg
    z = y - mu + mean_shift
    al = np.where(
        z >= 0.0,
        tau * np.exp(-z / b_pos) / b_pos,
        (1.0 - tau) * np.exp(z / b_neg) / b_neg,
    )
    cont = norm.pdf(y, mu, A05_KAPPA * sigma)
    return (1.0 - A05_EPS_CONTAM) * al + A05_EPS_CONTAM * cont


TRUE_PDF: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "A01": true_pdf_a01,
    "unimodal": true_pdf_unimodal,
    "A02": true_pdf_a02,
    "A04": true_pdf_a04,
    "A05": true_pdf_a05,
}


def true_mode_curves(
    problem_name: str, x_grid: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    """Return labeled true mean curves for posterior overlays."""
    x = np.asarray(x_grid, dtype=np.float64).ravel()
    if problem_name == "A01":
        y1, y2 = bimodal_mode_means(x)
        return [("True mode 1", y1), ("True mode 2", y2)]
    if problem_name == "A04":
        y0, y1, y2 = trimodal_mode_means(x)
        return [("True mode 0", y0), ("True mode 1", y1), ("True mode 2", y2)]
    if problem_name == "unimodal":
        return [("True mean", 0.5 * x + 2.0)]
    if problem_name == "A02":
        return [("True mean", (x >= 0.5).astype(np.float64))]
    if problem_name == "A05":
        return [("True mean", a05_mean(x))]
    if problem_name == "A06":
        return [("True mean", a06_mean(x))]
    return []


def empirical_local_pdf(
    y_grid: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_value: float,
    *,
    x_bandwidth: float | None = None,
    min_points: int = 12,
) -> np.ndarray | None:
    x_all = np.asarray(x_all, dtype=np.float64).ravel()
    y_all = np.asarray(y_all, dtype=np.float64).ravel()
    if x_bandwidth is None:
        span = float(x_all.max() - x_all.min()) if x_all.size else 1.0
        x_bandwidth = max(0.08 * span, 1e-3)
    bw = x_bandwidth
    y_near = np.array([])
    for _ in range(6):
        mask = np.abs(x_all - x_value) <= bw
        y_near = y_all[mask]
        if y_near.size >= min_points:
            break
        bw *= 1.6
    if y_near.size < 5:
        return None
    try:
        return np.asarray(gaussian_kde(y_near)(y_grid), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def probe_x_values(
    x_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    n_probes: int = 5,
) -> np.ndarray:
    lo = float(min(np.min(x_tr), np.min(x_te)))
    hi = float(max(np.max(x_tr), np.max(x_te)))
    return np.linspace(lo, hi, n_probes + 2)[1:-1]


MODEL_COLORS = {
    "midline": "#2b6cb0",
    "midline_stage2": "#2b6cb0",
    "midline_stage1": "#63b3ed",
    "exact_gp": "#805ad5",
    "tabgp": "#dd6b20",
    "tabpfn3": "#38a169",
    "tabpfn35": "#276749",
    "kumo": "#c53030",
    "tabfm": "#c53030",  # deprecated alias → KumoTabular
    "champion": "#2b6cb0",
    "nll_only": "#3182ce",
    "force_merge": "#d69e2e",
    "kmax1": "#718096",
}


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
            fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to save {out_path}")


def _probe_tag(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.2f}".replace(".", "p")


def save_marginals_per_probe(
    *,
    problem_name: str,
    x_probes: np.ndarray,
    model_preds: dict[str, dict[str, np.ndarray]],
    model_labels: dict[str, str],
    model_order: list[str] | None,
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_min: float,
    y_max: float,
    out_dir: Path,
    stem: str,
    n_grid: int = 400,
    annotate_experts_for: list[str] | None = None,
) -> list[Path]:
    """One figure per probe ``x``; one subplot per model (true + data KDE + model).

    When ``annotate_experts_for`` lists model keys, those panels also show
    per-component weighted Gaussians (π·N) and μ markers for UQ shape debug.
    """
    from predictive_density import model_predictive_density
    from uq_calibration import annotate_expert_components

    true_fn = TRUE_PDF.get(problem_name)
    y_grid = np.linspace(y_min, y_max, n_grid)
    keys = model_order or list(model_preds.keys())
    keys = [k for k in keys if k in model_preds]
    n_m = len(keys)
    if n_m == 0:
        return []
    annotate_set = set(annotate_experts_for or [])

    ncol = min(3, n_m)
    nrow = int(math.ceil(n_m / ncol))
    paths: list[Path] = []

    for i, xv in enumerate(x_probes):
        true_dens = true_fn(y_grid, float(xv)) if true_fn is not None else None
        emp = empirical_local_pdf(y_grid, x_data, y_data, float(xv))
        # Shared y-limit across model panels at this probe
        ymax = 0.0
        if true_dens is not None:
            ymax = max(ymax, float(np.nanmax(true_dens)))
        if emp is not None:
            ymax = max(ymax, float(np.nanmax(emp)))
        model_dens: dict[str, np.ndarray] = {}
        dens_source: dict[str, str] = {}
        for mk in keys:
            pred = model_preds[mk]
            dens, src = model_predictive_density(y_grid, pred, probe_index=i)
            if dens is None:
                dens = _mixture_pdf(
                    y_grid,
                    pred["expert_means"][i],
                    pred["expert_stds"][i],
                    pred["gate_weights"][i],
                )
                src = "mixture_gaussian"
            model_dens[mk] = dens
            dens_source[mk] = src
            ymax = max(ymax, float(np.nanmax(dens)))
        ymax = max(ymax * 1.12, 1e-6)

        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), dpi=140, sharey=True
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for j, mk in enumerate(keys):
            ax = axes_flat[j]
            color = MODEL_COLORS.get(mk, f"C{j}")
            if true_dens is not None:
                ax.plot(
                    y_grid,
                    true_dens,
                    color="#e53e3e",
                    lw=2.3,
                    label=r"True $p(y\mid x)$",
                    zorder=5,
                )
            if emp is not None:
                ax.plot(
                    y_grid,
                    emp,
                    color="0.35",
                    lw=1.5,
                    ls="--",
                    label="Data KDE (local $x$)",
                    zorder=4,
                )
            src = dens_source.get(mk, "mixture_gaussian")
            # Step plot for discrete PPD (TabPFN bars / Kumo quantile grid)
            if src in ("bar_distribution", "ensemble_samples", "quantile_diff"):
                tag = {
                    "bar_distribution": "PPD bars",
                    "ensemble_samples": "ensemble bins",
                    "quantile_diff": "quantile PPD",
                }[src]
                ax.step(
                    y_grid,
                    model_dens[mk],
                    where="mid",
                    color=color,
                    lw=1.8,
                    label=f"{model_labels.get(mk, mk)} ({tag})",
                    zorder=3,
                )
            else:
                ax.plot(
                    y_grid,
                    model_dens[mk],
                    color=color,
                    lw=2.0,
                    label=model_labels.get(mk, mk),
                    zorder=3,
                )
            if mk in annotate_set:
                annotate_expert_components(
                    ax,
                    y_grid,
                    model_preds[mk]["expert_means"][i],
                    model_preds[mk]["expert_stds"][i],
                    model_preds[mk]["gate_weights"][i],
                )
            ax.set_title(model_labels.get(mk, mk), fontsize=10, color=color)
            ax.set_xlabel(r"$y$")
            if j % ncol == 0:
                ax.set_ylabel(rf"$p(y\mid x={xv:.2f})$")
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(0.0, ymax)
            ax.grid(True, alpha=0.28)
            ax.legend(fontsize=7.5, loc="best", framealpha=0.92)

        for j in range(n_m, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"{problem_name}: conditional marginal at $x={xv:.2f}$ "
            f"(one panel per model)",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()
        out_path = out_dir / f"{stem}_marginal_x{_probe_tag(float(xv))}.png"
        paths.append(_safe_savefig(fig, out_path))
        plt.close(fig)

    return paths


def _predictive_density_matrix(
    pred: dict[str, Any],
    *,
    order: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Build ``dens[iy, ix] = p(y_iy | x_ix)`` on a hard grid (no spatial smooth)."""
    from predictive_density import model_predictive_density

    em = np.asarray(pred["expert_means"], dtype=np.float64)[order]
    es = np.asarray(pred["expert_stds"], dtype=np.float64)[order]
    gw = np.asarray(pred["gate_weights"], dtype=np.float64)[order]
    if em.ndim == 1:
        em = em.reshape(-1, 1)
        es = es.reshape(-1, 1)
        gw = gw.reshape(-1, 1)
    n_x = int(em.shape[0])
    dens = np.zeros((y_grid.shape[0], n_x), dtype=np.float64)

    # Re-index optional PPD fields so probe_index matches ordered x_grid rows.
    pred_ord: dict[str, Any] = {
        "expert_means": em,
        "expert_stds": es,
        "gate_weights": gw,
    }
    for key in ("pred_borders", "pred_logits", "quantile_levels", "quantile_values", "ensemble_samples"):
        if key not in pred or pred[key] is None:
            continue
        val = pred[key]
        if key == "pred_borders" or key == "quantile_levels":
            pred_ord[key] = val
            continue
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == order.shape[0]:
            pred_ord[key] = arr[order]
        else:
            pred_ord[key] = val

    for i in range(n_x):
        dens_i, _src = model_predictive_density(y_grid, pred_ord, probe_index=i)
        if dens_i is None:
            dens_i = _mixture_pdf(y_grid, em[i], es[i], gw[i])
        dens[:, i] = np.asarray(dens_i, dtype=np.float64).ravel()
    return dens


def save_posterior_vs_data(
    *,
    problem_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    x_grid: np.ndarray,
    model_preds_grid: dict[str, dict[str, np.ndarray]],
    model_labels: dict[str, str],
    model_order: list[str] | None,
    out_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    n_y: int = 200,
    # Deprecated kwargs kept for call-site compatibility (ignored).
    interval_z: float = 1.96,
    interval_levels: tuple[float, ...] = (0.5, 0.9),
    use_quantile_intervals: bool = True,
) -> Path:
    """Per-model subplots: hard ``p(y|x)`` density gradient + data + mean curves.

    Density is drawn with ``pcolormesh(..., shading='nearest')`` (no Gaussian
    smoothing of the field). Interval bands are not used.
    """
    del interval_z, interval_levels, use_quantile_intervals  # deprecated

    keys = model_order or list(model_preds_grid.keys())
    keys = [k for k in keys if k in model_preds_grid]
    n_m = len(keys)
    if n_m == 0:
        raise ValueError("no model grid predictions for posterior plot")

    xg = np.asarray(x_grid, dtype=np.float64).ravel()
    order = np.argsort(xg)
    xg = xg[order]
    true_curves = true_mode_curves(problem_name, xg)

    x_tr = np.asarray(x_train, dtype=np.float64).ravel()
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    y_parts = [y_tr]
    if y_test is not None:
        y_parts.append(np.asarray(y_test, dtype=np.float64).ravel())
    y_all = np.concatenate(y_parts)
    if y_min is None:
        y_lo = float(np.min(y_all))
        pad = 0.05 * max(float(np.max(y_all) - y_lo), 1.0)
        y_lo -= pad
    else:
        y_lo = float(y_min)
    if y_max is None:
        y_hi = float(np.max(y_all))
        pad = 0.05 * max(float(y_hi - np.min(y_all)), 1.0)
        y_hi += pad
    else:
        y_hi = float(y_max)
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    y_grid = np.linspace(y_lo, y_hi, int(n_y))

    # Cell edges for nearest shading (hard pixels, no interp blur).
    if xg.size == 1:
        dx = 1.0
        x_edges = np.array([xg[0] - 0.5 * dx, xg[0] + 0.5 * dx])
    else:
        mid = 0.5 * (xg[:-1] + xg[1:])
        x_edges = np.concatenate(
            [[xg[0] - (mid[0] - xg[0])], mid, [xg[-1] + (xg[-1] - mid[-1])]]
        )
    dy = (y_hi - y_lo) / max(n_y, 1)
    y_edges = np.linspace(y_lo - 0.5 * dy, y_hi + 0.5 * dy, n_y + 1)

    ncol = min(3, n_m)
    nrow = int(math.ceil(n_m / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), dpi=140, sharex=True, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for j, mk in enumerate(keys):
        ax = axes_flat[j]
        pred = model_preds_grid[mk]
        em_ord = np.asarray(pred["expert_means"], dtype=np.float64)[order]
        es_ord = np.asarray(pred["expert_stds"], dtype=np.float64)[order]
        gw_ord = np.asarray(pred["gate_weights"], dtype=np.float64)[order]
        mu, _sd = mixture_mean_std(em_ord, es_ord, gw_ord)
        color = MODEL_COLORS.get(mk, f"C{j}")

        dens = _predictive_density_matrix(pred, order=order, y_grid=y_grid)
        # Peak-normalized bar/quantile PPDs (TabPFN/Kumo) are needle-thin; use a
        # robust scale + mild gamma so structure remains visible without smoothing.
        pos = dens[dens > 0.0]
        if pos.size == 0:
            dens_n = dens
        else:
            scale = float(np.percentile(pos, 99.0))
            scale = max(scale, float(np.max(dens)) * 1e-3, 1e-12)
            dens_n = np.clip(dens / scale, 0.0, 1.0) ** 0.45
        if str(color).startswith("C"):
            base = plt.get_cmap("tab10")(int(str(color)[1:]) % 10)
        else:
            base = color
        cmap = LinearSegmentedColormap.from_list(
            f"ppd_{mk}", ["#ffffff", base], N=256
        )

        ax.pcolormesh(
            x_edges,
            y_edges,
            dens_n,
            cmap=cmap,
            shading="flat",
            vmin=0.0,
            vmax=1.0,
            zorder=0,
            rasterized=True,
        )

        ax.scatter(
            x_tr,
            y_tr,
            s=12,
            c="0.25",
            alpha=0.45,
            label="Train",
            zorder=2,
            edgecolors="none",
        )
        if x_test is not None and y_test is not None:
            ax.scatter(
                np.asarray(x_test).ravel(),
                np.asarray(y_test).ravel(),
                s=14,
                c="#1a365d",
                alpha=0.4,
                marker="^",
                label="Test",
                zorder=2,
                edgecolors="none",
            )

        true_ls = (":", "-.", "--")
        for ti, (lab, yt) in enumerate(true_curves):
            ax.plot(
                xg,
                yt,
                color="#2d3748",
                lw=1.5,
                ls=true_ls[ti % len(true_ls)],
                label=lab if j == 0 else None,
                zorder=4,
                alpha=0.95,
            )

        gw = np.clip(gw_ord, 0.0, None)
        gw = gw / np.clip(gw.sum(axis=-1, keepdims=True), 1e-12, None)
        if em_ord.ndim == 2 and em_ord.shape[1] > 1:
            tab = plt.cm.tab10
            for k in range(em_ord.shape[1]):
                if float(gw[:, k].mean()) < 0.05:
                    continue
                ax.plot(
                    xg,
                    em_ord[:, k],
                    color=tab(k % 10),
                    lw=1.2,
                    alpha=0.7,
                    label=rf"comp $k={k}$" if j == 0 else None,
                    zorder=4,
                )

        ax.plot(xg, mu, color="#c53030", lw=2.0, label="Mixture mean", zorder=5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(model_labels.get(mk, mk), fontsize=10)
        ax.set_xlabel(r"$x$")
        if j % ncol == 0:
            ax.set_ylabel(r"$y$")
        ax.grid(True, alpha=0.2, zorder=1)
        if j == 0:
            ax.legend(fontsize=7.0, loc="best", framealpha=0.92, ncol=1)

    for j in range(n_m, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        rf"{problem_name}: predictive density $p(y\mid x)$ vs data",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def slice_pred(pred: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Slice mixture / PPD arrays along the sample axis."""
    out: dict[str, Any] = {
        "mixture_mean": np.asarray(pred["mixture_mean"])[start:stop],
        "expert_means": np.asarray(pred["expert_means"])[start:stop],
        "expert_stds": np.asarray(pred["expert_stds"])[start:stop],
        "gate_weights": np.asarray(pred["gate_weights"])[start:stop],
    }
    # TabPFN bar distribution: borders shared; logits per row
    if "pred_borders" in pred and pred["pred_borders"] is not None:
        out["pred_borders"] = np.asarray(pred["pred_borders"])
    if "pred_logits" in pred and pred["pred_logits"] is not None:
        out["pred_logits"] = np.asarray(pred["pred_logits"])[start:stop]
    if "quantile_levels" in pred and pred["quantile_levels"] is not None:
        out["quantile_levels"] = np.asarray(pred["quantile_levels"])
    if "quantile_values" in pred and pred["quantile_values"] is not None:
        out["quantile_values"] = np.asarray(pred["quantile_values"])[start:stop]
    if "ensemble_samples" in pred and pred["ensemble_samples"] is not None:
        out["ensemble_samples"] = np.asarray(pred["ensemble_samples"])[start:stop]
    return out


def predictive_cdf_matrix(pred: dict[str, Any], y_grid: np.ndarray) -> np.ndarray:
    """``[N, G]`` predictive CDF on ``y_grid`` for any PPD type (bars, quantiles, mixtures)."""
    n = int(np.asarray(pred["expert_means"]).shape[0])
    dens = _predictive_density_matrix(pred, order=np.arange(n), y_grid=y_grid).T
    dens = np.clip(np.nan_to_num(dens, nan=0.0), 0.0, None)
    dy = np.diff(y_grid)
    cdf = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1]) * dy, axis=1)], axis=1
    )
    return cdf / np.clip(cdf[:, -1:], 1e-12, None)


def _cdf_quantiles(cdf: np.ndarray, y_grid: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """``[N, L]`` quantiles by inverting each CDF row."""
    out = np.empty((cdf.shape[0], len(levels)))
    for i in range(cdf.shape[0]):
        c = np.maximum.accumulate(cdf[i])
        out[i] = np.interp(levels, c + 1e-12 * np.arange(c.size), y_grid)
    return out


def _cdf_grid(y_train: np.ndarray, y_test: np.ndarray, n_grid: int = 600) -> np.ndarray:
    lo = float(min(y_train.min(), y_test.min()))
    hi = float(max(y_train.max(), y_test.max()))
    pad = 0.35 * (hi - lo + 1e-9)
    return np.linspace(lo - pad, hi + pad, n_grid)


def save_conditionals_multid(
    *,
    problem_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    model_preds: dict[str, dict[str, Any]],
    model_labels: dict[str, str],
    model_order: list[str] | None,
    out_path: Path,
    ref_key: str | None = None,
    levels: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    k_nn: int | None = None,
) -> tuple[Path, list[int]] | None:
    """Conditionals ``p(y | x_i)`` at test points spanning the predicted median.

    Points are the test rows whose reference-model median is nearest each
    quantile in ``levels`` of the median distribution; each panel overlays
    the histogram of the ``k = max(20, n/30)`` nearest training ``y`` in
    standardized ``x`` (the empirical stand-in for an unknown true density).
    """
    from predictive_density import model_predictive_density

    keys = [k for k in (model_order or list(model_preds)) if k in model_preds]
    if not keys:
        return None
    ref = ref_key if ref_key in model_preds else keys[0]
    x_tr = np.asarray(x_train, dtype=np.float64).reshape(len(y_train), -1)
    x_te = np.asarray(x_test, dtype=np.float64).reshape(-1, x_tr.shape[1])
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    y_grid = _cdf_grid(y_tr, y_tr)
    med = _cdf_quantiles(predictive_cdf_matrix(model_preds[ref], y_grid), y_grid, (0.5,))[:, 0]
    targets = np.quantile(med, levels)
    idx: list[int] = []
    for t in targets:
        order = np.argsort(np.abs(med - t))
        idx.append(int(next(i for i in order if i not in idx)))
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-12
    z_tr, z_te = (x_tr - mu) / sd, (x_te - mu) / sd
    k = int(k_nn or max(20, len(y_tr) // 30))

    nrow, ncol = len(idx), len(keys)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.7 * ncol, 2.1 * nrow), dpi=130, squeeze=False
    )
    for r, (lev, i) in enumerate(zip(levels, idx, strict=True)):
        dist = np.linalg.norm(z_tr - z_te[i], axis=1)
        y_nn = y_tr[np.argsort(dist)[:k]]
        dens_row = {}
        for mk in keys:
            d_i, _ = model_predictive_density(y_grid, model_preds[mk], probe_index=i)
            if d_i is None:
                p = model_preds[mk]
                d_i = _mixture_pdf(
                    y_grid, p["expert_means"][i], p["expert_stds"][i], p["gate_weights"][i]
                )
            dens_row[mk] = np.asarray(d_i, dtype=np.float64)
        lo, hi = np.quantile(y_nn, [0.0, 1.0])
        span = max(hi - lo, 1e-3 * (y_grid[-1] - y_grid[0]))
        x_lo, x_hi = lo - 1.0 * span, hi + 1.0 * span
        for c, mk in enumerate(keys):
            ax = axes[r, c]
            ax.hist(y_nn, bins=max(8, k // 4), density=True, color="0.82", edgecolor="0.6",
                    label=f"{k}-NN train $y$")
            ax.plot(y_grid, dens_row[mk], color=MODEL_COLORS.get(mk, f"C{c}"), lw=1.8)
            ax.set_xlim(x_lo, x_hi)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(model_labels.get(mk, mk), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"q{int(round(100 * lev))} median\n(test #{i})", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(
        f"{problem_name}: $p(y\\mid x_i)$ at test points spanning the "
        f"{model_labels.get(ref, ref)} median",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path, idx


def save_posterior_vs_data_multid(
    *,
    problem_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_preds: dict[str, dict[str, Any]],
    model_labels: dict[str, str],
    model_order: list[str] | None,
    out_path: Path,
) -> Path | None:
    """Per model: test points sorted by predicted median with 50/90% bands vs
    observed ``y`` (left), and the PIT histogram ``F(y_i | x_i)`` (right)."""
    keys = [k for k in (model_order or list(model_preds)) if k in model_preds]
    if not keys:
        return None
    y = np.asarray(y_test, dtype=np.float64).ravel()
    y_grid = _cdf_grid(np.asarray(y_train, dtype=np.float64).ravel(), y)
    fig, axes = plt.subplots(
        len(keys), 2, figsize=(10.0, 2.3 * len(keys)), dpi=130, squeeze=False,
        gridspec_kw={"width_ratios": [3.0, 1.0]},
    )
    n_bins = 10
    for r, mk in enumerate(keys):
        cdf = predictive_cdf_matrix(model_preds[mk], y_grid)
        q = _cdf_quantiles(cdf, y_grid, (0.05, 0.25, 0.5, 0.75, 0.95))
        pit = np.array([np.interp(y[i], y_grid, cdf[i]) for i in range(y.size)])
        order = np.argsort(q[:, 2])
        t = np.arange(y.size)
        color = MODEL_COLORS.get(mk, f"C{r}")
        ax = axes[r, 0]
        ax.fill_between(t, q[order, 0], q[order, 4], color=color, alpha=0.18, lw=0, label="90%")
        ax.fill_between(t, q[order, 1], q[order, 3], color=color, alpha=0.35, lw=0, label="50%")
        ax.plot(t, q[order, 2], color=color, lw=1.4, label="median")
        ax.scatter(t, y[order], s=5, color="k", zorder=4, label="observed $y$")
        cov50 = float(np.mean((y >= q[:, 1]) & (y <= q[:, 3])))
        cov90 = float(np.mean((y >= q[:, 0]) & (y <= q[:, 4])))
        ax.set_title(
            f"{model_labels.get(mk, mk)}  (cov50={cov50:.2f}, cov90={cov90:.2f})", fontsize=9
        )
        ax.set_ylabel("$y$", fontsize=8)
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=4)
        if r == len(keys) - 1:
            ax.set_xlabel("test points sorted by predicted median", fontsize=8)
        axp = axes[r, 1]
        axp.hist(pit, bins=n_bins, range=(0, 1), density=True, color=color, alpha=0.75,
                 edgecolor="white")
        axp.axhline(1.0, color="k", lw=1.0, ls="--")
        axp.set_xlim(0, 1)
        axp.tick_params(labelsize=7)
        if r == 0:
            axp.set_title("PIT histogram", fontsize=9)
    fig.suptitle(f"{problem_name}: posterior predictive vs test data", fontsize=11, y=1.005)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


# Back-compat alias used by older call sites
def save_marginal_panels(**kwargs):  # type: ignore[no-untyped-def]
    out_path = Path(kwargs.pop("out_path"))
    kwargs["out_dir"] = out_path.parent
    kwargs["stem"] = out_path.stem.replace("_marginals_xprobes", "").replace(
        "_xprobes", ""
    )
    kwargs.setdefault("model_order", list(kwargs.get("model_preds", {}).keys()))
    return save_marginals_per_probe(**kwargs)
