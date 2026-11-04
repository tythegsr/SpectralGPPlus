"""Charts for MidlineSparseMoE-GP fits and Gate-2 summaries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from diagnostics import mixture_density_on_grid, count_density_peaks

# Glossary used in figure footnotes / legend boxes
_GLOSSARY = {
    "mu0": r"$\mu_0(x)$: Stage-1 GP midline (shared mean)",
    "delta": r"$\delta_k(x)$: expert offset (MLP or constant)",
    "alpha": r"$\alpha_k\in[0,1]$: Hard-Concrete on/off for expert $k$",
    "pi": r"$\pi_k$: mixture / gate weight ($\sum_k\pi_k=1$)",
    "mean_k": r"$\mu_k(x)=\mu_0(x)+\alpha_k\delta_k(x)$: expert mean",
    "gamma": r"$\gamma_k\in(0,1)$: fraction of $\sigma_0^2$ expert $k$ may remove",
    "var": r"$\mathrm{Var}_k=\sigma_0^2(1-\alpha_k\gamma_k)+\varepsilon$ (carve-out $\le\sigma_0^2$)",
    "K_eff": (
        r"$K_{\mathrm{eff}}=\sum_k\tilde\alpha_k s_k$, "
        r"$\tilde\alpha_k=\sigma((\alpha_k\pi_k-\varepsilon)/\tau)$"
    ),
    "K_hat": r"$K_{\mathrm{hat}}=\#\{k:\alpha_k\pi_k>\varepsilon\}$: hard active count",
    "K_modes": r"$K_{\mathrm{modes}}$: #peaks of mixture $p(y|x)$ (≠ experts)",
    "s_k": r"$s_k$: separation score (nearby $\delta$ curves count less)",
    "tilde_alpha": r"$\tilde\alpha_k$: soft used-slot gate (drops low $\alpha\pi$ mass)",
    "sep": r"Sep: usage-weighted pairwise push $\sum_{i<j}\tilde\alpha_i\tilde\alpha_j$",
    "orphan": r"orphan: $\sum_k\alpha_k\mathrm{softplus}(\varepsilon-\bar\pi_k)$",
    "NLL": r"NLL: mixture negative log-likelihood (Stage-2 fit term)",
}


def _add_glossary(fig, lines: list[str], *, fontsize: float = 7.2) -> None:
    """Footer glossary so symbols in the legend are defined."""
    # Two-column-ish wrapped lines under the axes
    mid = (len(lines) + 1) // 2
    left = "\n".join(f"• {t}" for t in lines[:mid])
    right = "\n".join(f"• {t}" for t in lines[mid:])
    fig.text(
        0.02,
        0.02,
        left,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="#2d3748",
        family="sans-serif",
    )
    if right:
        fig.text(
            0.52,
            0.02,
            right,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            color="#2d3748",
            family="sans-serif",
        )
    n_rows = max(mid, len(lines) - mid)
    fig.subplots_adjust(bottom=0.14 + 0.045 * n_rows)


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
    fallback = out_path.parent / f"{out_path.stem}_fallback.png"
    fig.savefig(str(fallback), bbox_inches="tight", dpi=140)
    return fallback


def save_scatter_midline_experts(
    model,
    *,
    out_path: Path,
    title: str = "MidlineSparseMoE",
    n_grid: int = 200,
    true_mode_means: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    true_mode_labels: list[str] | None = None,
) -> Path:
    """Train scatter, GP midline, and active expert mean curves."""
    x = model.train_x.detach().cpu().numpy().ravel()
    y = model.train_y.detach().cpu().numpy().ravel()
    x_grid = np.linspace(float(x.min()), float(x.max()), n_grid).reshape(-1, 1)
    xt = torch.as_tensor(x_grid, dtype=model.dtype, device=model.train_x.device)
    pred = model.predict(xt)
    mu0 = pred["mu0"].detach().cpu().numpy()
    means = pred["expert_means"].detach().cpu().numpy()
    alpha = pred["alpha"].detach().cpu().numpy()
    if alpha.ndim == 2:
        alpha_bar = alpha.mean(axis=0)
    else:
        alpha_bar = alpha
    pi = pred["gate_weights"].detach().cpu().numpy().mean(axis=0)

    with torch.no_grad():
        k_eff = float(model.k_eff().detach())
        k_hat = int(model.k_hat())

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.scatter(
        x,
        y,
        s=10,
        alpha=0.35,
        c="#4a5568",
        label="train data $(x_i,y_i)$",
        zorder=1,
    )
    if true_mode_means is not None:
        true_ls = (":", "-.", "--")
        for i, tm in enumerate(true_mode_means):
            lab = (
                true_mode_labels[i]
                if true_mode_labels is not None and i < len(true_mode_labels)
                else rf"true mode ${i}$"
            )
            ax.plot(
                x_grid[:, 0],
                np.asarray(tm).ravel(),
                color="#718096",
                lw=1.3,
                ls=true_ls[i % len(true_ls)],
                alpha=0.85,
                label=lab,
                zorder=2,
            )
    ax.plot(
        x_grid[:, 0],
        mu0,
        color="#1a365d",
        lw=2.2,
        label=r"GP midline $\mu_0(x)$ (Stage 1)",
        zorder=3,
    )
    cmap = plt.cm.tab10
    for k in range(model.k_max):
        mass = float(alpha_bar[k] * pi[k])
        if mass < 0.03:
            continue
        ax.plot(
            x_grid[:, 0],
            means[:, k],
            color=cmap(k % 10),
            lw=1.6,
            alpha=0.55 + 0.45 * min(mass, 1.0),
            label=(
                rf"expert $k={k}$: $\mu_k=\mu_0+\alpha_k\delta_k(x)$"
                rf"  ($\alpha\pi={mass:.2f}$)"
            ),
            zorder=2,
        )
    ax.set_xlabel("input $x$")
    ax.set_ylabel("response $y$")
    ax.set_title(
        f"{title}\n"
        rf"$K_{{\max}}={model.k_max}$,  $K_{{\mathrm{{hat}}}}={k_hat}$,  "
        rf"$K_{{\mathrm{{eff}}}}={k_eff:.2f}$  (experts ≠ modes)"
    )
    ax.legend(loc="best", fontsize=7.5, frameon=True, fancybox=False, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _add_glossary(
        fig,
        [
            _GLOSSARY["mu0"],
            _GLOSSARY["mean_k"],
            _GLOSSARY["alpha"],
            _GLOSSARY["pi"],
            _GLOSSARY["K_hat"],
            _GLOSSARY["K_eff"],
        ],
    )
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_conditional_pdf(
    model,
    *,
    x_value: float,
    out_path: Path,
    y_min: float,
    y_max: float,
    n_grid: int = 400,
    title: str | None = None,
) -> Path:
    """Mixture p(y|x) at a fixed x, with component Gaussians."""
    xt = torch.tensor([[x_value]], dtype=model.dtype, device=model.train_x.device)
    pred = model.predict(xt)
    means = pred["expert_means"].detach().cpu().numpy().ravel()
    stds = pred["expert_stds"].detach().cpu().numpy().ravel()
    weights = pred["gate_weights"].detach().cpu().numpy().ravel()
    alpha = pred["alpha"].detach().cpu().numpy().ravel()
    if alpha.size == model.k_max:
        # reweight display by alpha for clarity of active experts
        w_disp = weights * alpha
        w_disp = w_disp / max(float(w_disp.sum()), 1e-12)
    else:
        w_disp = weights

    y_grid = np.linspace(y_min, y_max, n_grid)
    dens = mixture_density_on_grid(y_grid, means, stds, weights)
    k_modes = count_density_peaks(y_grid, dens)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        y_grid,
        dens,
        color="#1a365d",
        lw=2.4,
        label=rf"mixture $p(y|x)=\sum_k\pi_k\mathcal{{N}}(\mu_k,\sigma_k^2)$"
        rf"  ($K_{{\mathrm{{modes}}}}={k_modes}$ peaks)",
    )
    cmap = plt.cm.tab10
    for k in range(model.k_max):
        if w_disp[k] < 0.02 and alpha[k] < 0.05:
            continue
        comp = w_disp[k] * norm.pdf(y_grid, means[k], max(stds[k], 1e-6))
        ax.plot(
            y_grid,
            comp,
            color=cmap(k % 10),
            lw=1.3,
            alpha=0.75,
            label=(
                rf"component $k={k}$: weight $\pi={weights[k]:.2f}$, "
                rf"switch $\alpha={alpha[k]:.2f}$, "
                rf"mean $\mu={means[k]:.1f}$"
            ),
        )
    ax.set_xlabel(r"response $y$")
    ax.set_ylabel(rf"density $p(y \mid x={x_value:.2f})$")
    ax.set_title(
        title
        or rf"Conditional mixture at $x={x_value:.2f}$ (co-located multimodality)"
    )
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _add_glossary(
        fig,
        [
            _GLOSSARY["pi"],
            _GLOSSARY["alpha"],
            _GLOSSARY["mean_k"],
            _GLOSSARY["K_modes"],
            "Peaks counted on the mixture density, not on #experts",
        ],
    )
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_alpha_gate_bars(
    model,
    *,
    out_path: Path,
    title: str = "Expert activity",
) -> Path:
    """Bar chart of alpha, pi, and Option-E contribution tilde_alpha * s."""
    with torch.no_grad():
        alpha_t = model.get_alpha(deterministic=True)
        pi_t = model.get_pi(model.train_x)
        delta = model.get_delta(model.train_x, deterministic_alpha=True)
        k_eff_t, usage_t, s_t = model.compute_k_eff(alpha_t, pi_t, delta)
        alpha = alpha_t.cpu().numpy()
        pi = pi_t.mean(dim=0).cpu().numpy()
        usage = usage_t.cpu().numpy()
        s = s_t.cpu().numpy()
        a_k = usage * s
        k_eff = float(k_eff_t.detach())
        k_hat = model.k_hat()

    idx = np.arange(model.k_max)
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(
        idx - width,
        alpha,
        width,
        label=r"$\alpha_k$  (Hard-Concrete on/off)",
        color="#2b6cb0",
    )
    ax.bar(
        idx,
        pi,
        width,
        label=r"$\pi_k$  (mixture / gate weight)",
        color="#38a169",
    )
    ax.bar(
        idx + width,
        a_k,
        width,
        label=r"$\tilde\alpha_k s_k$  (counts toward $K_{\mathrm{eff}}$)",
        color="#c05621",
    )
    ax.set_xticks(idx)
    ax.set_xlabel(r"expert index $k$")
    ax.set_ylabel("value in $[0,1]$")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{title}\n"
        rf"$K_{{\mathrm{{hat}}}}={k_hat}$ hard-active  |  "
        rf"$K_{{\mathrm{{eff}}}}={k_eff:.2f}$ soft-effective"
    )
    ax.legend(frameon=True, fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _add_glossary(
        fig,
        [
            _GLOSSARY["alpha"],
            _GLOSSARY["pi"],
            _GLOSSARY["tilde_alpha"],
            _GLOSSARY["s_k"],
            _GLOSSARY["K_eff"],
            _GLOSSARY["K_hat"],
        ],
    )
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_training_curves(
    history: list[dict],
    *,
    out_path: Path,
    title: str = "Stage-2 training",
) -> Path:
    if not history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
        ax.set_axis_off()
        path = _safe_savefig(fig, out_path)
        plt.close(fig)
        return path

    steps = [h["step"] for h in history]
    nll = [h["nll"] for h in history]
    k_eff = [h["k_eff"] for h in history]
    k_hat = [h["k_hat"] for h in history]
    loss = [h["loss"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    axes[0].plot(steps, nll, color="#1a365d", lw=1.8, label="mixture NLL")
    axes[0].set_title("Fit quality")
    axes[0].set_xlabel("Stage-2 Adam step")
    axes[0].set_ylabel("NLL")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        steps,
        k_eff,
        color="#c05621",
        lw=1.8,
        label=r"$K_{\mathrm{eff}}$ soft count",
    )
    axes[1].plot(
        steps,
        k_hat,
        color="#2b6cb0",
        lw=1.4,
        ls="--",
        label=r"$K_{\mathrm{hat}}$ hard count",
    )
    axes[1].set_title("How many experts stay on")
    axes[1].set_xlabel("Stage-2 Adam step")
    axes[1].legend(frameon=True, fontsize=8, framealpha=0.9)
    axes[2].plot(
        steps,
        loss,
        color="#553c9a",
        lw=1.8,
        label=r"$\mathcal{L}=$ NLL $+ \lambda_K K_{\mathrm{eff}} + \cdots$",
    )
    axes[2].set_title("Total Stage-2 objective")
    axes[2].set_xlabel("Stage-2 Adam step")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    _add_glossary(
        fig,
        [
            _GLOSSARY["NLL"],
            _GLOSSARY["K_eff"],
            _GLOSSARY["K_hat"],
            "No mixture-mean MSE (would collapse co-located modes)",
        ],
    )
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_training_loss_subplots(
    histories: dict[str, list[dict]],
    *,
    out_path: Path,
    model_labels: dict[str, str] | None = None,
    model_order: list[str] | None = None,
    model_colors: dict[str, str] | None = None,
    title: str = "Stage-2 training loss",
) -> Path:
    """One subplot per model/config: composite loss + mixture NLL vs Adam step."""
    labels = model_labels or {}
    colors = model_colors or {}
    keys = [k for k in (model_order or list(histories.keys())) if k in histories]
    if not keys:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.text(0.5, 0.5, "no fit history", ha="center", va="center")
        ax.set_axis_off()
        return _safe_savefig(fig, out_path)

    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), dpi=140, sharex=False
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for j, mk in enumerate(keys):
        ax = axes_flat[j]
        hist = histories.get(mk) or []
        color = colors.get(mk, f"C{j}")
        lab = labels.get(mk, mk)
        if not hist:
            ax.text(
                0.5, 0.5, "no history", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(lab, fontsize=10, color=color)
            ax.set_axis_off()
            continue
        steps = [int(r["step"]) for r in hist]
        loss = [float(r["loss"]) for r in hist]
        nll = [float(r["nll"]) for r in hist if "nll" in r]
        ax.plot(steps, loss, "-", color=color, lw=1.8, label="loss")
        if len(nll) == len(steps):
            ax.plot(steps, nll, "--", color="0.35", lw=1.3, label="NLL")
        ax.set_title(lab, fontsize=10, color=color)
        ax.set_xlabel("Stage-2 step")
        if j % ncol == 0:
            ax.set_ylabel("train objective")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return _safe_savefig(fig, out_path)


def save_gate2_summary_bars(
    gate2_json: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Grouped bars: K_eff / K_hat / K_modes for each Gate-2 row."""
    if isinstance(gate2_json, (str, Path)):
        data = json.loads(Path(gate2_json).read_text(encoding="utf-8"))
    else:
        data = gate2_json
    rows = data["rows"]
    names = []
    keff, khat, kmodes, truek = [], [], [], []
    for key, row in rows.items():
        names.append(row.get("name", key).replace("gate2", "").replace("_", "\n"))
        keff.append(row["k_eff"])
        khat.append(row["k_hat"])
        kmodes.append(row["k_modes_mean"])
        truek.append(row["true_k_modes"])

    x = np.arange(len(names))
    w = 0.22
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(
        x - 1.5 * w,
        truek,
        w,
        label=r"true $K_{\mathrm{modes}}$ (ground truth)",
        color="#718096",
    )
    ax.bar(
        x - 0.5 * w,
        kmodes,
        w,
        label=r"$K_{\mathrm{modes}}$: peaks of $p(y|x)$",
        color="#2b6cb0",
    )
    ax.bar(
        x + 0.5 * w,
        khat,
        w,
        label=r"$K_{\mathrm{hat}}$: #experts with $\alpha\pi>\varepsilon$",
        color="#38a169",
    )
    ax.bar(
        x + 1.5 * w,
        keff,
        w,
        label=r"$K_{\mathrm{eff}}=\sum_k \tilde\alpha_k s_k$ (Option E)",
        color="#c05621",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(
        "Gate-2 dual reporting: modes of $p(y|x)$ vs expert slots\n"
        r"(goal: $K_{\mathrm{modes}}$ tracks truth; $K_{\mathrm{hat}}$ may be larger)"
    )
    ax.legend(frameon=True, fontsize=8, ncol=2, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _add_glossary(
        fig,
        [
            _GLOSSARY["K_modes"],
            _GLOSSARY["K_hat"],
            _GLOSSARY["K_eff"],
            "Experts ≠ modes: overcomplete $K_{\\max}$ can exceed true modes",
        ],
    )
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def _load_limitation_metrics(path: Path | dict) -> dict:
    if isinstance(path, dict):
        return path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_limitation_kmodes_bars(
    metrics: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Mean±std K_modes by problem/cfg, with true-K reference line markers."""
    data = _load_limitation_metrics(metrics)
    agg = data["aggregates"]
    keys = sorted(agg.keys())
    means = [agg[k]["k_modes_mean"]["mean"] for k in keys]
    stds = [agg[k]["k_modes_mean"]["std"] for k in keys]
    # true K from any row in that group
    true_map: dict[str, float] = {}
    for r in data["rows"]:
        key = f"{r['problem']}/{r['cfg']}"
        true_map[key] = float(r["true_k_modes"])
    trues = [true_map.get(k, float("nan")) for k in keys]
    labels = [k.replace("/", "\n") for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color="#2b6cb0",
        alpha=0.9,
        label=r"mean $K_{\mathrm{modes}}$ ± std",
        zorder=2,
    )
    ax.scatter(
        x,
        trues,
        marker="D",
        s=55,
        color="#c53030",
        zorder=3,
        label=r"true $K_{\mathrm{modes}}$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$K_{\mathrm{modes}}$")
    ax.set_title(
        "Limitation stress: mode recovery by problem / config\n"
        f"seeds={data.get('seeds', [])}"
    )
    ax.legend(frameon=True, fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(0, max(3.5, max(means) + max(stds) + 0.4))
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_limitation_pass_rates(
    metrics: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Horizontal bars of pass-rate checks from limitation_stress."""
    data = _load_limitation_metrics(metrics)
    rates = data.get("pass_rates", {})
    labels: list[str] = []
    vals: list[float] = []
    for group, checks in sorted(rates.items()):
        for check, v in checks.items():
            labels.append(f"{group}\n{check}")
            vals.append(float(v))
    if not labels:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no pass_rates", ha="center", va="center")
        path = _safe_savefig(fig, out_path)
        plt.close(fig)
        return path

    y = np.arange(len(labels))
    colors = ["#38a169" if v >= 0.8 else ("#d69e2e" if v >= 0.4 else "#c53030") for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(labels))))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("pass rate across seeds")
    ax.set_title("Limitation stress: pass rates")
    ax.axvline(0.8, color="#718096", ls="--", lw=1, label="0.8 threshold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_limitation_seed_strips(
    metrics: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Per-seed K_modes strips for each problem (champion vs classic)."""
    data = _load_limitation_metrics(metrics)
    problems = sorted({r["problem"] for r in data["rows"]})
    fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 4.2), sharey=True)
    if len(problems) == 1:
        axes = [axes]
    cmap = {"champion_p12": "#2b6cb0", "champion_mlp": "#2b6cb0", "classic": "#718096", "classic_k2": "#718096", "classic_mlp": "#718096"}
    for ax, problem in zip(axes, problems):
        rows = [r for r in data["rows"] if r["problem"] == problem]
        cfgs = sorted({r["cfg"] for r in rows})
        for i, cfg in enumerate(cfgs):
            vals = [r["k_modes_mean"] for r in rows if r["cfg"] == cfg]
            seeds = [r["seed"] for r in rows if r["cfg"] == cfg]
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=42,
                color=cmap.get(cfg, "#4a5568"),
                alpha=0.85,
                zorder=3,
                label=cfg,
            )
            for j, (s, v) in enumerate(zip(seeds, vals)):
                ax.annotate(
                    str(s),
                    (i + jitter[j], v),
                    textcoords="offset points",
                    xytext=(4, 2),
                    fontsize=6,
                    color="#4a5568",
                )
        true_k = rows[0]["true_k_modes"] if rows else None
        if true_k is not None:
            ax.axhline(true_k, color="#c53030", ls="--", lw=1.2, label=f"true={true_k}")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, fontsize=7, rotation=20, ha="right")
        ax.set_title(problem, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r"$K_{\mathrm{modes}}$")
        ax.legend(frameon=False, fontsize=6, loc="best")
    fig.suptitle("Limitation stress: per-seed $K_{\\mathrm{modes}}$", y=1.02)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_limitation_nll_bars(
    metrics: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Mean±std test NLL by problem/cfg."""
    data = _load_limitation_metrics(metrics)
    agg = data["aggregates"]
    keys = sorted(agg.keys())
    means = [agg[k]["test_nll"]["mean"] for k in keys]
    stds = [agg[k]["test_nll"]["std"] for k in keys]
    labels = [k.replace("/", "\n") for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x, means, yerr=stds, capsize=4, color="#805ad5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("test NLL")
    ax.set_title("Limitation stress: test NLL by problem / config")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def save_limitation_keff_khat(
    metrics: Path | dict,
    *,
    out_path: Path,
) -> Path:
    """Grouped mean K_eff / K_hat / K_modes bars."""
    data = _load_limitation_metrics(metrics)
    agg = data["aggregates"]
    keys = sorted(agg.keys())
    labels = [k.replace("/", "\n") for k in keys]
    keff = [agg[k]["k_eff"]["mean"] for k in keys]
    khat = [agg[k]["k_hat"]["mean"] for k in keys]
    kmodes = [agg[k]["k_modes_mean"]["mean"] for k in keys]
    x = np.arange(len(keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(x - w, kmodes, w, label=r"$K_{\mathrm{modes}}$", color="#2b6cb0")
    ax.bar(x, khat, w, label=r"$K_{\mathrm{hat}}$", color="#38a169")
    ax.bar(x + w, keff, w, label=r"$K_{\mathrm{eff}}$", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Limitation stress: dual reporting (mean over seeds)")
    ax.legend(frameon=True, fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def generate_limitation_stress_charts(
    *,
    metrics_path: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Render charts from ``results/limitation_stress/metrics.json``."""
    _THIS = Path(__file__).resolve().parent
    metrics_path = Path(
        metrics_path or (_THIS / "results" / "limitation_stress" / "metrics.json")
    )
    out_dir = Path(out_dir or (_THIS / "results" / "limitation_stress" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run: python run_midline_sparse.py --mode limitation_stress"
        )
    paths = [
        save_limitation_kmodes_bars(
            metrics_path, out_path=out_dir / "limitation_kmodes.png"
        ),
        save_limitation_pass_rates(
            metrics_path, out_path=out_dir / "limitation_pass_rates.png"
        ),
        save_limitation_seed_strips(
            metrics_path, out_path=out_dir / "limitation_seed_strips.png"
        ),
        save_limitation_nll_bars(
            metrics_path, out_path=out_dir / "limitation_nll.png"
        ),
        save_limitation_keff_khat(
            metrics_path, out_path=out_dir / "limitation_dual_counts.png"
        ),
    ]
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} limitation charts under {out_dir}")
    for p in paths:
        print(f"  {p}")
    return paths


def generate_recovery_probe_charts(
    *,
    metrics_path: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Charts for ``results/recovery_probe/metrics.json``."""
    _THIS = Path(__file__).resolve().parent
    metrics_path = Path(
        metrics_path or (_THIS / "results" / "recovery_probe" / "metrics.json")
    )
    out_dir = Path(out_dir or (_THIS / "results" / "recovery_probe" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run: python run_midline_sparse.py --mode recovery_probe"
        )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    keys = sorted(summary.keys())
    labels = [k.replace("/", "\n") for k in keys]

    # K_modes mean ± std
    means = [summary[k]["k_modes_mean"] for k in keys]
    stds = [summary[k]["k_modes_std"] for k in keys]
    true_map = {"A01": 2.0, "A04": 3.0, "A05": 1.0}
    trues = [true_map.get(k.split("/")[0], float("nan")) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x, means, yerr=stds, capsize=4, color="#2b6cb0", alpha=0.9, label=r"$K_{\mathrm{modes}}$")
    ax.scatter(x, trues, marker="D", s=55, color="#c53030", zorder=3, label="true $K$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$K_{\mathrm{modes}}$")
    ax.set_title("Recovery probe: mode recovery (p12 vs p12+rec)")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = _safe_savefig(fig, out_dir / "recovery_kmodes.png")
    plt.close(fig)

    # Pass rates grouped
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    w = 0.35
    km_pass = [summary[k]["pass_k_modes_rate"] for k in keys]
    cert_pass = [summary[k]["certificate_pass_rate"] for k in keys]
    ax.bar(x - w / 2, km_pass, w, label=r"$K_{\mathrm{modes}}$ pass", color="#38a169")
    ax.bar(x + w / 2, cert_pass, w, label="certificate pass", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="#718096", ls="--", lw=1)
    ax.set_ylabel("rate across seeds")
    ax.set_title("Recovery probe: pass rates")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = _safe_savefig(fig, out_dir / "recovery_pass_rates.png")
    plt.close(fig)

    # Certificate heatmap: seed × (problem/tag) of certificate_pass
    tags = sorted({r["tag"] for r in data["rows"]})
    problems = sorted({r["problem"] for r in data["rows"]})
    seeds = data.get("seeds", sorted({r["seed"] for r in data["rows"]}))
    cols = [f"{p}/{t}" for p in problems for t in tags]
    mat = np.full((len(seeds), len(cols)), np.nan)
    for r in data["rows"]:
        i = seeds.index(r["seed"])
        j = cols.index(f"{r['problem']}/{r['tag']}")
        mat[i, j] = 1.0 if r["certificate_pass"] else 0.0
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(cols)), 3.8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([str(s) for s in seeds])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("/", "\n") for c in cols], fontsize=8)
    ax.set_title("Recovery certificate pass by seed")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p3 = _safe_savefig(fig, out_dir / "recovery_cert_heatmap.png")
    plt.close(fig)

    paths = [p1, p2, p3]
    (out_dir / "manifest.txt").write_text(
        "\n".join(str(p) for p in paths) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(paths)} recovery charts under {out_dir}")
    for p in paths:
        print(f"  {p}")
    return paths


def generate_merge_prune_probe_charts(
    *,
    metrics_path: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Tables + line charts for ``results/merge_prune_probe/metrics.json``.

    Compares ``adaptive_off`` vs ``adaptive_on`` across problems and ``K_max``.
    """
    import csv

    _THIS = Path(__file__).resolve().parent
    metrics_path = Path(
        metrics_path or (_THIS / "results" / "merge_prune_probe" / "metrics.json")
    )
    out_dir = Path(out_dir or (_THIS / "results" / "merge_prune_probe" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run: "
            "python run_midline_sparse.py --mode merge_prune_probe"
        )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: list[dict] = list(data.get("rows") or [])
    accept: dict = dict(data.get("accept") or {})
    if not rows:
        raise ValueError(f"No rows in {metrics_path}")

    paths: list[Path] = []
    problems = sorted({str(r["problem"]) for r in rows}, key=lambda p: (
        ["A01", "unimodal", "A02", "A04", "A05"].index(p)
        if p in ("A01", "unimodal", "A02", "A04", "A05")
        else 99
    ))
    metric_specs = [
        ("RMSE", "RMSE"),
        ("CRPS", "CRPS"),
        ("log_score", "log_score (NLL)"),
        ("k_eff", r"$K_{\mathrm{eff}}$"),
        ("k_modes_mean", r"$K_{\mathrm{modes}}$"),
        ("merge_pressure", "merge_pressure"),
        ("k_hat", r"$K_{\mathrm{hat}}$"),
    ]
    colors = {"adaptive_off": "#718096", "adaptive_on": "#2b6cb0"}
    labels = {"adaptive_off": "adaptive off (legacy)", "adaptive_on": "adaptive on"}

    # --- CSV of all rows (easy spreadsheet eval) ---
    csv_path = out_dir.parent / "metrics.csv"
    fieldnames = [
        "problem",
        "true_k_modes",
        "champion",
        "k_max",
        "tag",
        "adaptive_merge_prune",
        "RMSE",
        "CRPS",
        "log_score",
        "k_eff",
        "k_hat",
        "k_modes_mean",
        "k_modes_round",
        "merge_pressure",
        "gate_type",
        "wall_s",
        "used_safety_refit",
        "num_surviving_inits",
        "stage1_gp_nll",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            diag = r.get("adaptive_diagnostics") or {}
            w.writerow(
                {
                    **{k: r.get(k) for k in fieldnames},
                    "used_safety_refit": diag.get("used_safety_refit"),
                    "num_surviving_inits": diag.get("num_surviving_inits"),
                    "stage1_gp_nll": diag.get("stage1_gp_nll"),
                }
            )
    paths.append(csv_path)

    def _series(problem: str, tag: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        sub = sorted(
            [r for r in rows if r["problem"] == problem and r["tag"] == tag],
            key=lambda r: int(r["k_max"]),
        )
        ks = np.asarray([int(r["k_max"]) for r in sub], dtype=int)
        ys = []
        for r in sub:
            v = r.get(key)
            ys.append(float(v) if v is not None and np.isfinite(float(v)) else np.nan)
        return ks, np.asarray(ys, dtype=float)

    # --- Per-problem metrics vs K_max (on vs off) ---
    for problem in problems:
        true_k = next(
            (int(r["true_k_modes"]) for r in rows if r["problem"] == problem),
            None,
        )
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8), dpi=140)
        axes_flat = list(axes.ravel())
        # 7 metrics + leave last for acceptance note
        for ax, (key, ylab) in zip(axes_flat[:7], metric_specs, strict=True):
            for tag in ("adaptive_off", "adaptive_on"):
                ks, ys = _series(problem, tag, key)
                if ks.size == 0:
                    continue
                ax.plot(
                    ks,
                    ys,
                    "o-",
                    color=colors[tag],
                    lw=2.0,
                    ms=6,
                    label=labels[tag],
                )
            if key in ("k_eff", "k_modes_mean") and true_k is not None:
                ax.axhline(
                    true_k,
                    color="#c53030",
                    ls="--",
                    lw=1.3,
                    label=rf"true $K={true_k}$",
                )
            if key == "log_score":
                # Cap display of catastrophic NLL for readability
                ymin, ymax = ax.get_ylim()
                if ymax > 40:
                    ax.set_ylim(min(ymin, -2), 40)
                    ax.text(
                        0.02,
                        0.98,
                        "y capped at 40\n(catastrophe → large NLL)",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="#c53030",
                    )
            ax.set_xlabel(r"$K_{\max}$")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        # Acceptance panel
        ax = axes_flat[7]
        ax.axis("off")
        acc = accept.get(problem, {})
        lines = [f"{problem} acceptance", ""]
        if not acc:
            lines.append("(no accept block)")
        else:
            for k, v in acc.items():
                mark = "PASS" if v is True else ("FAIL" if v is False else str(v))
                lines.append(f"{k}: {mark}")
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f7fafc", edgecolor="#cbd5e0"),
        )
        fig.suptitle(
            f"{problem}: adaptive merge/prune vs $K_{{\\max}}$",
            fontsize=12,
        )
        fig.tight_layout()
        p = _safe_savefig(fig, out_dir / f"{problem.lower()}_metrics_vs_kmax.png")
        plt.close(fig)
        paths.append(p)

        # Per-problem table PNG
        sub = sorted(
            [r for r in rows if r["problem"] == problem],
            key=lambda r: (int(r["k_max"]), 0 if not r["adaptive_merge_prune"] else 1),
        )
        col_labels = [
            "tag",
            "Kmax",
            "RMSE",
            "CRPS",
            "log_score",
            "k_eff",
            "k_modes",
            "k_hat",
            "merge_p",
            "gate",
        ]
        cell_text = []
        for r in sub:
            cell_text.append(
                [
                    str(r.get("tag", "")),
                    str(r.get("k_max", "")),
                    f"{float(r['RMSE']):.4f}" if r.get("RMSE") is not None else "—",
                    f"{float(r['CRPS']):.4f}" if r.get("CRPS") is not None else "—",
                    f"{float(r['log_score']):.3f}"
                    if r.get("log_score") is not None
                    else "—",
                    f"{float(r['k_eff']):.3f}" if r.get("k_eff") is not None else "—",
                    f"{float(r['k_modes_mean']):.2f}"
                    if r.get("k_modes_mean") is not None
                    else "—",
                    str(r.get("k_hat", "—")),
                    f"{float(r.get('merge_pressure') or 0):.2f}",
                    str(r.get("gate_type", "—")),
                ]
            )
        fig, ax = plt.subplots(
            figsize=(12.5, 1.6 + 0.32 * max(len(cell_text), 1)), dpi=140
        )
        ax.axis("off")
        ax.set_title(
            f"{problem}: merge/prune probe table (true $K={true_k}$)",
            fontsize=11,
            pad=10,
        )
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.05, 1.28)
        for (ri, ci), cell in table.get_celld().items():
            if ri == 0:
                cell.set_facecolor("#2b6cb0")
                cell.set_text_props(color="white", weight="bold")
            elif ri > 0 and cell_text[ri - 1][0] == "adaptive_on":
                cell.set_facecolor("#ebf8ff")
            elif ri % 2 == 0:
                cell.set_facecolor("#f7fafc")
        fig.tight_layout()
        p = _safe_savefig(fig, out_dir / f"{problem.lower()}_metrics_table.png")
        plt.close(fig)
        paths.append(p)

    # --- Cross-problem summary at K_max in {4,5,6,10}: on vs off delta RMSE / log ---
    focus_k = [k for k in (4, 5, 6, 10) if any(int(r["k_max"]) == k for r in rows)]
    if focus_k:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=140)
        for ax, key, title in zip(
            axes,
            ("RMSE", "log_score", "k_modes_mean"),
            ("RMSE", "log_score", r"$K_{\mathrm{modes}}$"),
            strict=True,
        ):
            x = np.arange(len(problems))
            w = 0.35
            # average over focus_k for a compact view
            off_vals, on_vals = [], []
            for problem in problems:
                def _avg(tag: str, prob: str = problem) -> float:
                    vals = [
                        float(r[key])
                        for r in rows
                        if r["problem"] == prob
                        and r["tag"] == tag
                        and int(r["k_max"]) in focus_k
                        and r.get(key) is not None
                        and np.isfinite(float(r[key]))
                    ]
                    return float(np.mean(vals)) if vals else np.nan

                off_vals.append(_avg("adaptive_off"))
                on_vals.append(_avg("adaptive_on"))
            ax.bar(x - w / 2, off_vals, w, color=colors["adaptive_off"], label="off")
            ax.bar(x + w / 2, on_vals, w, color=colors["adaptive_on"], label="on")
            ax.set_xticks(x)
            ax.set_xticklabels(problems, fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(f"mean over $K_{{\\max}}\\in{focus_k}$")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            if key == "log_score":
                ymax = np.nanmax(off_vals + on_vals)
                if np.isfinite(ymax) and ymax > 40:
                    ax.set_ylim(0, 40)
        fig.suptitle("Merge/prune: adaptive off vs on (summary)", fontsize=12)
        fig.tight_layout()
        p = _safe_savefig(fig, out_dir / "summary_off_vs_on.png")
        plt.close(fig)
        paths.append(p)

    # --- Acceptance checklist table ---
    if accept:
        fig, ax = plt.subplots(figsize=(9.5, 1.8 + 0.35 * len(accept)), dpi=140)
        ax.axis("off")
        col_labels = ["problem", "checks"]
        cell_text = []
        for problem in problems:
            acc = accept.get(problem, {})
            parts = []
            for k, v in acc.items():
                if isinstance(v, bool):
                    parts.append(f"{'✓' if v else '✗'} {k}")
                else:
                    parts.append(f"{k}={v}")
            cell_text.append([problem, "  |  ".join(parts) if parts else "—"])
        ax.set_title("Merge/prune probe acceptance (evaluation only)", fontsize=11)
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colWidths=[0.15, 0.85],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.05, 1.4)
        for (ri, _ci), cell in table.get_celld().items():
            if ri == 0:
                cell.set_facecolor("#2b6cb0")
                cell.set_text_props(color="white", weight="bold")
            elif ri % 2 == 0:
                cell.set_facecolor("#f7fafc")
        fig.tight_layout()
        p = _safe_savefig(fig, out_dir / "acceptance_table.png")
        plt.close(fig)
        paths.append(p)

    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} merge/prune artifacts under {out_dir}")
    for p in paths:
        print(f"  {p}")
    return paths


def generate_all_charts(
    *,
    out_dir: Path | None = None,
    seed: int = 42,
) -> list[Path]:
    """Fit key settings and write a chart pack."""
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    _EXP = Path(__file__).resolve().parents[1]
    _THIS = Path(__file__).resolve().parent
    for p in (_ROOT, _EXP, _THIS):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import gpplus
    from gpplus.models import MidlineSparseMoEGPR
    from load_bimodal_data import train_test_split_bimodal
    from load_trimodal_data import (
        TRIMODAL_SIGMA,
        TRIMODAL_Y_MAX,
        TRIMODAL_Y_MIN,
        make_trimodal_tensors,
        trimodal_mode_means,
    )
    from load_unimodal_hetero_data import (
        A05_Y_MAX,
        A05_Y_MIN,
        a05_mean,
        make_unimodal_hetero_tensors,
    )
    from run_midline_sparse import make_unimodal, make_step

    gpplus.config.configure_logger()
    out_dir = Path(out_dir or (_THIS / "results" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _fit(
        x,
        y,
        *,
        k_max,
        gate="constant",
        offset="mlp",
        lambda_mode=0.0,
        steps=800,
        # Defaults = champion p12+rec
        lam_k=0.02,
        lam_sep=0.1,
        sep_margin=2.0,
        lam_overlap=0.08,
        overlap_margin=1.0,
        free_expert_scale=False,
        lam_beta=0.0,
        lam_k_start=0.0,
        lam_orphan=0.25,
        lam_delta=0.02,
        lam_gamma=0.1,
        lam_rec=0.15,
        rec_margin=1.0,
        use_full_gp_variance=False,
        gamma_init_logit=2.2,
        s=seed,
    ):
        m = MidlineSparseMoEGPR(
            x,
            y,
            k_max=k_max,
            gate_type=gate,
            offset_type=offset,
            tau_max=2.5,
            keff_tau=0.02,
            free_expert_scale=free_expert_scale,
            use_full_gp_variance=use_full_gp_variance,
            gamma_init_logit=gamma_init_logit,
        )
        m.fit(
            num_inits=2,
            num_epochs=1,
            num_steps=steps,
            lr=5e-2,
            lambda_k=lam_k,
            lambda_sep=lam_sep,
            lambda_alpha=0.05,
            lambda_orphan=lam_orphan,
            lambda_mode=lambda_mode,
            lambda_overlap=lam_overlap,
            lambda_beta=lam_beta,
            lambda_delta=lam_delta,
            lambda_gamma=lam_gamma,
            lambda_rec=lam_rec,
            sep_margin=sep_margin,
            overlap_margin=overlap_margin,
            rec_margin=rec_margin,
            lambda_k_start=lam_k_start,
            seed=s,
            device="cpu",
            n_jobs=1,
        )
        return m

    # --- A01 Kmax=2 (peak recovery) ---
    data = train_test_split_bimodal(n_samples=400, test_size=0.3, seed=seed)
    x_tr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.as_tensor(data["y_train"], dtype=torch.float64)
    m_a01_2 = _fit(x_tr, y_tr, k_max=2, lam_k=0.02, steps=900)
    loss_histories: dict[str, list[dict]] = {
        "a01_k2": list(m_a01_2._fit_history or []),
    }
    loss_labels = {"a01_k2": "A01 K_max=2"}
    paths.append(
        save_scatter_midline_experts(
            m_a01_2,
            out_path=out_dir / "a01_k2_scatter.png",
            title="A01 co-located  |  K_max=2  |  MLP offsets",
        )
    )
    x_cond = float(np.median(x_tr.numpy()))
    paths.append(
        save_conditional_pdf(
            m_a01_2,
            x_value=x_cond,
            out_path=out_dir / "a01_k2_pdf.png",
            y_min=-2.0,
            y_max=30.0,
            title=f"A01 K_max=2 MLP offs  |  p(y|x={x_cond:.1f})",
        )
    )
    paths.append(
        save_alpha_gate_bars(
            m_a01_2,
            out_path=out_dir / "a01_k2_activity.png",
            title="A01 K_max=2 activity",
        )
    )
    paths.append(
        save_training_curves(
            m_a01_2._fit_history,
            out_path=out_dir / "a01_k2_train.png",
            title="A01 K_max=2 Stage-2",
        )
    )

    # --- A01 Kmax=6 ---
    m_a01_6 = _fit(x_tr, y_tr, k_max=6, lambda_mode=0.0, lam_k=0.08, steps=900)
    loss_histories["a01_k6"] = list(m_a01_6._fit_history or [])
    loss_labels["a01_k6"] = "A01 K_max=6"
    paths.append(
        save_scatter_midline_experts(
            m_a01_6,
            out_path=out_dir / "a01_k6_v2_scatter.png",
            title="A01 co-located  |  K_max=6 + mode loss  |  MLP offsets",
        )
    )
    paths.append(
        save_conditional_pdf(
            m_a01_6,
            x_value=x_cond,
            out_path=out_dir / "a01_k6_v2_pdf.png",
            y_min=-2.0,
            y_max=30.0,
            title=f"A01 K_max=6+v2  |  p(y|x={x_cond:.1f})",
        )
    )
    paths.append(
        save_alpha_gate_bars(
            m_a01_6,
            out_path=out_dir / "a01_k6_v2_activity.png",
            title="A01 K_max=6+v2 activity",
        )
    )
    paths.append(
        save_training_curves(
            m_a01_6._fit_history,
            out_path=out_dir / "a01_k6_v2_train.png",
            title="A01 K_max=6+v2 Stage-2",
        )
    )

    # --- Unimodal control (homoscedastic Gaussian) ---
    xu, yu, _, _ = make_unimodal(300, 7)
    m_uni = _fit(xu, yu, k_max=6, lam_k=0.10, steps=800, s=7)
    loss_histories["unimodal"] = list(m_uni._fit_history or [])
    loss_labels["unimodal"] = "Unimodal"
    paths.append(
        save_scatter_midline_experts(
            m_uni,
            out_path=out_dir / "unimodal_scatter.png",
            title="Unimodal control  |  K_max=6",
        )
    )
    paths.append(
        save_conditional_pdf(
            m_uni,
            x_value=float(np.median(xu.numpy())),
            out_path=out_dir / "unimodal_pdf.png",
            y_min=-2.0,
            y_max=12.0,
            title="Unimodal  |  conditional density",
        )
    )
    paths.append(
        save_alpha_gate_bars(
            m_uni,
            out_path=out_dir / "unimodal_activity.png",
            title="Unimodal activity",
        )
    )

    # --- A02 constant vs MLP (NLL / modes contrast) ---
    xs, ys, _, _ = make_step(200, 200, seed=seed)
    m_c = _fit(xs, ys, k_max=4, gate="constant", steps=500)
    m_m = _fit(xs, ys, k_max=4, gate="mlp", steps=500)
    loss_histories["a02_constant"] = list(m_c._fit_history or [])
    loss_histories["a02_mlp"] = list(m_m._fit_history or [])
    loss_labels["a02_constant"] = "A02 constant gate"
    loss_labels["a02_mlp"] = "A02 MLP gate"
    paths.append(
        save_scatter_midline_experts(
            m_c,
            out_path=out_dir / "a02_constant_scatter.png",
            title="A02 step  |  constant gate",
        )
    )
    paths.append(
        save_scatter_midline_experts(
            m_m,
            out_path=out_dir / "a02_mlp_scatter.png",
            title="A02 step  |  MLP gate",
        )
    )

    # --- A04 co-located trimodal (unequal widths); champion p12+rec defaults ---
    x4, y4, _, _ = make_trimodal_tensors(n_samples=400, test_size=0.3, seed=seed)
    m_a04 = _fit(x4, y4, k_max=6, steps=900, s=seed)
    loss_histories["a04"] = list(m_a04._fit_history or [])
    loss_labels["a04"] = "A04 trimodal"
    x4_grid = np.linspace(float(x4.min()), float(x4.max()), 200)
    true_means = trimodal_mode_means(x4_grid)
    true_labels = [
        rf"true mode $k={k}$ ($\sigma={TRIMODAL_SIGMA[k]:g}$)" for k in range(3)
    ]
    paths.append(
        save_scatter_midline_experts(
            m_a04,
            out_path=out_dir / "a04_trimodal_scatter.png",
            title="A04 trimodal (unequal $\\sigma$)  |  K_max=6  |  MLP offsets",
            true_mode_means=true_means,
            true_mode_labels=true_labels,
        )
    )
    x4_cond = float(np.median(x4.numpy()))
    paths.append(
        save_conditional_pdf(
            m_a04,
            x_value=x4_cond,
            out_path=out_dir / "a04_trimodal_pdf.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title=f"A04 trimodal  |  p(y|x={x4_cond:.1f})",
        )
    )
    paths.append(
        save_conditional_pdf(
            m_a04,
            x_value=1.0,
            out_path=out_dir / "a04_trimodal_pdf_x1.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title="A04 trimodal  |  p(y|x=1)",
        )
    )
    paths.append(
        save_conditional_pdf(
            m_a04,
            x_value=7.0,
            out_path=out_dir / "a04_trimodal_pdf_x7.png",
            y_min=TRIMODAL_Y_MIN,
            y_max=TRIMODAL_Y_MAX,
            title="A04 trimodal  |  p(y|x=7)",
        )
    )
    paths.append(
        save_alpha_gate_bars(
            m_a04,
            out_path=out_dir / "a04_trimodal_activity.png",
            title="A04 trimodal activity",
        )
    )
    paths.append(
        save_training_curves(
            m_a04._fit_history,
            out_path=out_dir / "a04_trimodal_train.png",
            title="A04 trimodal Stage-2",
        )
    )

    # --- A05 unimodal skewed contaminated heteroscedastic ---
    x5, y5, _, _ = make_unimodal_hetero_tensors(
        n_samples=400, test_size=0.3, seed=seed
    )
    m_a05 = _fit(x5, y5, k_max=6, steps=900, s=seed)
    loss_histories["a05"] = list(m_a05._fit_history or [])
    loss_labels["a05"] = "A05 hetero"
    x5_grid = np.linspace(float(x5.min()), float(x5.max()), 200)
    paths.append(
        save_scatter_midline_experts(
            m_a05,
            out_path=out_dir / "a05_hetero_scatter.png",
            title="A05 unimodal hetero (skew+contam)  |  K_max=6",
            true_mode_means=(a05_mean(x5_grid),),
            true_mode_labels=[r"true $\mu(x)$"],
        )
    )
    x5_cond = float(np.median(x5.numpy()))
    paths.append(
        save_conditional_pdf(
            m_a05,
            x_value=x5_cond,
            out_path=out_dir / "a05_hetero_pdf.png",
            y_min=A05_Y_MIN,
            y_max=A05_Y_MAX,
            title=f"A05 hetero  |  p(y|x={x5_cond:.1f})",
        )
    )
    paths.append(
        save_alpha_gate_bars(
            m_a05,
            out_path=out_dir / "a05_hetero_activity.png",
            title="A05 hetero activity",
        )
    )
    paths.append(
        save_training_curves(
            m_a05._fit_history,
            out_path=out_dir / "a05_hetero_train.png",
            title="A05 hetero Stage-2",
        )
    )

    # --- Combined Stage-2 loss: one subplot per fitted model ---
    paths.append(
        save_training_loss_subplots(
            loss_histories,
            out_path=out_dir / "training_loss_all.png",
            model_labels=loss_labels,
            model_order=list(loss_histories.keys()),
            title="MidlineSparse Stage-2 training loss (all models)",
        )
    )

    # --- Gate-2 summary from saved JSON if present ---
    gate2_path = _THIS / "results" / "midline_sparse_gate2.json"
    if gate2_path.exists():
        paths.append(
            save_gate2_summary_bars(
                gate2_path,
                out_path=out_dir / "gate2_summary.png",
            )
        )

    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} charts under {out_dir}")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "limitation"),
        default="all",
    )
    args = parser.parse_args()
    if args.mode == "limitation":
        generate_limitation_stress_charts()
    else:
        generate_all_charts()
