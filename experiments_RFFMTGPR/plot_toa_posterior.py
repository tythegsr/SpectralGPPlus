"""TOA 3-panel posterior figures for RFFMTGPR experiments."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from plot_validation_curves import sanitize_plot_subdir

TOA_SPECTRAL_DIM = 285
WAVELENGTH_MIN_NM = 250.0
WAVELENGTH_MAX_NM = 2500.0
TASK_COS = "y_cos"
TASK_GRAIN = "y_grain"
DEFAULT_TASK_NAMES = (TASK_COS, TASK_GRAIN)

# Physical / dataset support for TOA targets (grain size >= 0; cos_i in [0, 1]).
TASK_BOUNDS: dict[str, tuple[float, float | None]] = {
    TASK_COS: (0.0, 1.0),
    TASK_GRAIN: (0.0, None),
}

# Minimum σ for PDF evaluation only (avoids division by zero; not a display fudge).
_PDF_STD_EPS = 1e-9


def wavelength_axis(n_bins: int) -> np.ndarray:
    """Wavelength (nm) for radiance spectrum; 250--2500 nm when n_bins=285."""
    if n_bins == TOA_SPECTRAL_DIM:
        return np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n_bins)
    return np.arange(n_bins, dtype=np.float64)


def _pct_str(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value * 100:.2f}%"


def _format_posterior_value(task_key: str, value: float) -> str:
    if task_key == TASK_COS:
        return f"{value:.4f}"
    return f"{value:.4g}"


def _metrics_text_for_task(
    task_name: str,
    rel_metrics: dict[str, float | int],
    *,
    rel_tolerance: float,
) -> str:
    lines = [
        f"mean_rel: {_pct_str(float(rel_metrics['mean_rel_error']))}",
        f"max_rel: {_pct_str(float(rel_metrics['max_rel_error']))}",
        f"within_{rel_tolerance * 100:g}%: {_pct_str(float(rel_metrics['pct_within_1pct']))}",
    ]
    return "\n".join(lines)


def _normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=np.float64)
    out = 0.5 * (1.0 + erf(z_arr / math.sqrt(2.0)))
    if np.isscalar(z):
        return float(out)
    return out


def _gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), _PDF_STD_EPS)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def _clip_interval(
    lower: float,
    upper: float,
    mean: float,
    std: float,
    *,
    x_min: float,
    x_max: float | None,
) -> tuple[float, float, float]:
    """Clip CI bounds to task support; use model std as-is (no display floor)."""
    plot_std = max(float(std), _PDF_STD_EPS)
    lo = max(x_min, float(lower))
    hi = float(upper)
    if x_max is not None:
        hi = min(x_max, hi)
    if hi < lo:
        lo, hi = mean - 2.0 * plot_std, mean + 2.0 * plot_std
        lo = max(x_min, lo)
        if x_max is not None:
            hi = min(x_max, hi)
    return lo, hi, plot_std


def _truncated_normal_pdf(
    x: np.ndarray,
    mean: float,
    std: float,
    *,
    x_min: float,
    x_max: float | None,
) -> np.ndarray:
    std = max(float(std), _PDF_STD_EPS)
    z_min = (x_min - mean) / std
    z_max = (x_max - mean) / std if x_max is not None else np.inf
    if x_max is None:
        norm = 1.0 - _normal_cdf(z_min)
    else:
        norm = _normal_cdf(z_max) - _normal_cdf(z_min)
    norm = max(float(norm), 1e-12)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    pdf[mask] = _gaussian_pdf(x[mask], mean, std) / norm
    return pdf


def _lognormal_params_from_original(
    y_pred: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Infer log-normal parameters from original-scale (µm) median and CI bounds."""
    pred = max(float(y_pred), 1e-12)
    lo = max(float(lower), 1e-12)
    hi = max(float(upper), lo * (1.0 + 1e-12))
    mu_log = math.log(pred)
    sigma_log = max((math.log(hi) - math.log(lo)) / 4.0, _PDF_STD_EPS)
    return mu_log, sigma_log


def _truncated_lognormal_pdf(
    x: np.ndarray,
    mu_log: float,
    sigma_log: float,
    *,
    x_min: float,
) -> np.ndarray:
    """Log-normal density on original scale (e.g. grain in µm)."""
    sigma_log = max(float(sigma_log), _PDF_STD_EPS)
    x_min = max(float(x_min), 1e-12)
    z_min = (math.log(x_min) - mu_log) / sigma_log
    norm = 1.0 - _normal_cdf(z_min)
    norm = max(float(norm), 1e-12)
    pdf = np.zeros_like(x, dtype=np.float64)
    mask = x >= x_min
    xm = x[mask]
    z = (np.log(xm) - mu_log) / sigma_log
    pdf[mask] = np.exp(-0.5 * z * z) / (xm * sigma_log * math.sqrt(2.0 * math.pi) * norm)
    return pdf


def _validate_original_scale_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    log_grain: bool,
) -> None:
    """Posterior plots expect physical units: cos_i in [0,1], grain in µm (not log µm)."""
    grain_true = np.asarray(y_true, dtype=np.float64).ravel()[1]
    grain_pred = np.asarray(y_pred, dtype=np.float64).ravel()[1]
    cos_true = float(np.asarray(y_true, dtype=np.float64).ravel()[0])
    if not (0.0 <= cos_true <= 1.0):
        raise ValueError(f"cos_i true value {cos_true} outside [0, 1]; expected original scale.")
    if grain_true <= 0.0 or grain_pred <= 0.0:
        raise ValueError("grain size must be positive in µm for plotting.")
    if log_grain and max(grain_true, grain_pred) < 15.0:
        raise ValueError(
            "grain values look like log-scale (<15); pass original µm after inverse_y_predictions."
        )


def _focus_xlim(
    y_true: float,
    y_pred: float,
    ci_lower: float,
    ci_upper: float,
    std: float,
    *,
    x_min: float,
    x_max: float | None,
) -> tuple[float, float]:
    """Zoom x-axis near true/mean; cap using local uncertainty when global σ is huge."""
    local_std = min(
        std,
        abs(y_pred - y_true) + 0.1 * max(abs(y_pred), abs(y_true), 1.0),
    )
    band = max(
        3.0 * local_std,
        2.0 * abs(y_pred - y_true),
        0.08 * max(abs(y_pred), abs(y_true), 1.0),
    )
    x0 = max(x_min, min(y_true, y_pred) - band)
    x1 = max(y_true, y_pred) + band
    if x_max is not None:
        x1 = min(x_max, x1)
    if x1 <= x0:
        x1 = x0 + max(1.0, 0.1 * max(abs(y_pred), 1.0))
    return x0, x1


def _posterior_x_grid(x0: float, x1: float, n_grid: int = 300) -> np.ndarray:
    return np.linspace(x0, x1, n_grid)


def _plot_posterior_density_axis(
    ax: plt.Axes,
    *,
    task_key: str,
    task_label: str,
    y_true: float,
    y_pred: float,
    y_std: float,
    lower: float,
    upper: float,
    rel_metrics: dict[str, float | int] | None,
    rel_tolerance: float,
    log_grain: bool = False,
) -> None:
    x_min, x_max = TASK_BOUNDS.get(task_key, (0.0, None))
    ci_lo, ci_hi, plot_std = _clip_interval(
        lower, upper, y_pred, y_std, x_min=x_min, x_max=x_max
    )
    x0, x1 = _focus_xlim(
        y_true, y_pred, ci_lo, ci_hi, plot_std, x_min=x_min, x_max=x_max
    )
    grid = _posterior_x_grid(x0, x1)
    if log_grain and task_key == TASK_GRAIN:
        mu_log, sigma_log = _lognormal_params_from_original(y_pred, lower, upper)
        pdf = _truncated_lognormal_pdf(grid, mu_log, sigma_log, x_min=x_min)
    else:
        pdf = _truncated_normal_pdf(grid, y_pred, plot_std, x_min=x_min, x_max=x_max)

    fmt = lambda v: _format_posterior_value(task_key, v)
    ci_label = f"95% CI = [{fmt(lower)}, {fmt(upper)}]"
    true_label = f"true = {fmt(y_true)}"
    mean_label = f"mean = {fmt(y_pred)}"

    ax.fill_between(grid, 0.0, pdf, color="C0", alpha=0.25)
    ax.plot(grid, pdf, color="C0", linewidth=1.8, label="posterior")
    ax.axvspan(ci_lo, ci_hi, color="C0", alpha=0.12, label=ci_label)
    ax.axvline(y_true, color="C2", linestyle="--", linewidth=1.5, label=true_label)
    ax.axvline(y_pred, color="C1", linestyle="-", linewidth=1.5, label=mean_label)
    ax.axvline(ci_lo, color="C0", linestyle=":", linewidth=1.0)
    ax.axvline(ci_hi, color="C0", linestyle=":", linewidth=1.0)
    ax.set_xlim(x0, x1)
    ymax = float(np.max(pdf)) if pdf.size else 1.0
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.set_xlabel(task_label)
    ax.set_ylabel("posterior density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    if rel_metrics is not None:
        ax.text(
            0.02,
            0.98,
            _metrics_text_for_task(task_label, rel_metrics, rel_tolerance=rel_tolerance),
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )


def save_toa_posterior_figure(
    out_path: Path,
    *,
    spectrum: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    example_k: int,
    rel_metrics_by_task: dict[str, dict[str, float | int]] | None = None,
    rel_tolerance: float = 0.01,
    wavelength_nm: np.ndarray | None = None,
    log_grain: bool = False,
) -> Path:
    """One figure: spectrum | cos_i posterior | grain posterior (all in original units)."""
    spectrum = np.asarray(spectrum, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_std = np.asarray(y_std, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    _validate_original_scale_targets(y_true, y_pred, log_grain=log_grain)

    wl = wavelength_nm if wavelength_nm is not None else wavelength_axis(spectrum.shape[0])
    cos_true, grain_true = float(y_true[0]), float(y_true[1])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=120)

    axes[0].plot(wl, spectrum, color="C0", linewidth=1.0)
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Radiance")
    axes[0].set_title(f"true cos_i={cos_true:.4f}, true grain={grain_true:.4g}")
    axes[0].grid(True, alpha=0.3)

    rel_cos = rel_metrics_by_task.get(TASK_COS) if rel_metrics_by_task else None
    _plot_posterior_density_axis(
        axes[1],
        task_key=TASK_COS,
        task_label="cos_i",
        y_true=cos_true,
        y_pred=float(y_pred[0]),
        y_std=float(y_std[0]),
        lower=float(lower[0]),
        upper=float(upper[0]),
        rel_metrics=rel_cos,
        rel_tolerance=rel_tolerance,
    )

    rel_grain = rel_metrics_by_task.get(TASK_GRAIN) if rel_metrics_by_task else None
    _plot_posterior_density_axis(
        axes[2],
        task_key=TASK_GRAIN,
        task_label="grain size (µm)",
        y_true=grain_true,
        y_pred=float(y_pred[1]),
        y_std=float(y_std[1]),
        lower=float(lower[1]),
        upper=float(upper[1]),
        rel_metrics=rel_grain,
        rel_tolerance=rel_tolerance,
        log_grain=log_grain,
    )

    fig.suptitle(f"TOA test example {example_k}", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_posterior_example_indices(
    n_test: int,
    n_examples: int,
    *,
    seed: int = 0,
    explicit_indices: list[int] | None = None,
) -> list[int]:
    if explicit_indices is not None:
        return [int(i) for i in explicit_indices if 0 <= int(i) < n_test]
    if n_test <= 0 or n_examples <= 0:
        return []
    n_pick = min(n_examples, n_test)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_test, size=n_pick, replace=False).tolist())


def plot_toa_posterior_figures(
    x_test_orig: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    save_dir: Path,
    *,
    title: str,
    example_indices: list[int],
    rel_metrics_by_task: dict[str, dict[str, float | int]] | None = None,
    rel_tolerance: float = 0.01,
    wavelength_nm: np.ndarray | None = None,
    log_grain: bool = False,
) -> list[Path]:
    save_dir = Path(save_dir)
    written: list[Path] = []
    wl = wavelength_nm

    for k, i in enumerate(example_indices):
        cos_t = float(y_true[i, 0])
        grain_t = float(y_true[i, 1])
        fname = f"example_{k:04d}_cos{cos_t:.3f}_grain{grain_t:.1f}.png"
        out_path = save_dir / fname
        written.append(
            save_toa_posterior_figure(
                out_path,
                spectrum=x_test_orig[i],
                y_true=y_true[i],
                y_pred=y_pred[i],
                y_std=y_std[i],
                lower=lower[i],
                upper=upper[i],
                example_k=i,
                rel_metrics_by_task=rel_metrics_by_task,
                rel_tolerance=rel_tolerance,
                wavelength_nm=wl,
                log_grain=log_grain,
            )
        )
    return written


def save_predictions_npz(
    save_path: str | Path,
    title: str,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x_test_orig: np.ndarray,
    test_idx: np.ndarray,
    task_names: tuple[str, ...] | list[str],
    seed: int,
    rel_tolerance: float,
    example_indices: list[int] | None = None,
    wavelength_nm: np.ndarray | None = None,
    log_grain: bool = False,
) -> str:
    import os

    os.makedirs(save_path, exist_ok=True)
    out_npz = os.path.join(str(save_path), f"predictions_{title}.npz")
    if wavelength_nm is None:
        wavelength_nm = wavelength_axis(x_test_orig.shape[-1])
    np.savez(
        out_npz,
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
        lower=lower,
        upper=upper,
        x_test_orig=x_test_orig,
        test_idx=test_idx,
        task_names=np.array(task_names),
        title=title,
        seed=seed,
        rel_tolerance=rel_tolerance,
        wavelength_nm=wavelength_nm,
        example_indices=np.array(example_indices if example_indices is not None else [], dtype=np.int64),
        log_grain=np.array(log_grain),
    )
    return out_npz


def plot_posterior_from_npz(
    npz_path: str | Path,
    save_dir: str | Path | None = None,
    *,
    rel_tolerance: float | None = None,
    posterior_n_examples: int | None = None,
    posterior_example_indices: list[int] | None = None,
) -> list[Path]:
    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_std = data["y_std"]
    lower = data["lower"]
    upper = data["upper"]
    x_test_orig = data["x_test_orig"]
    title = str(data["title"].item()) if data["title"].shape == () else str(data["title"])
    seed = int(data["seed"].item()) if "seed" in data else 0
    tol = float(rel_tolerance if rel_tolerance is not None else data.get("rel_tolerance", 0.01))
    wl = data["wavelength_nm"] if "wavelength_nm" in data else wavelength_axis(x_test_orig.shape[-1])
    log_grain = bool(data["log_grain"].item()) if "log_grain" in data else False

    from mtgpr_experiment_utils import compute_relative_error_metrics

    rel_by_task: dict[str, dict[str, float | int]] = {}
    for t, name in enumerate(DEFAULT_TASK_NAMES):
        rel_by_task[name] = compute_relative_error_metrics(y_true[:, t], y_pred[:, t], rel_tolerance=tol)

    if posterior_example_indices is not None:
        example_indices = select_posterior_example_indices(
            y_true.shape[0], 0, explicit_indices=posterior_example_indices
        )
    elif "example_indices" in data and data["example_indices"].size > 0:
        example_indices = [int(i) for i in data["example_indices"].tolist()]
    else:
        n_ex = posterior_n_examples if posterior_n_examples is not None else 8
        example_indices = select_posterior_example_indices(y_true.shape[0], n_ex, seed=seed)

    if save_dir is None:
        save_dir = Path(npz_path).parent / "plots" / "posterior" / sanitize_plot_subdir(title)
    return plot_toa_posterior_figures(
        x_test_orig,
        y_true,
        y_pred,
        y_std,
        lower,
        upper,
        Path(save_dir),
        title=title,
        example_indices=example_indices,
        rel_metrics_by_task=rel_by_task,
        rel_tolerance=tol,
        wavelength_nm=wl,
        log_grain=log_grain,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TOA 3-panel posterior plots for RFFMTGPR")
    parser.add_argument("--npz", type=str, required=True, help="predictions_*.npz from a TOA run")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--rel-tolerance", type=float, default=None)
    parser.add_argument("--posterior-n-examples", type=int, default=None)
    parser.add_argument(
        "--posterior-example-indices",
        type=str,
        default=None,
        help="Comma-separated test row indices, e.g. 0,12,99",
    )
    args = parser.parse_args()

    explicit = None
    if args.posterior_example_indices:
        explicit = [int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()]

    paths = plot_posterior_from_npz(
        args.npz,
        args.save_dir,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=explicit,
    )
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
