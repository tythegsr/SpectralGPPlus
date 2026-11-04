"""Live foundation-model comparison vs 2-Stage Mixture GP and exact GP+.

Problems:
  A01 bimodal, unimodal, A02 step, A04 trimodal, A05 unimodal hetero, A06 cubic lognormal

Protocol (default): shared fixed test of size 1000; nested train sizes
  [100, 200, 400, 800]; seed=42. Pool generated once per problem.

Models:
  midline, mdn, mdn_const, exact_gp, tabgp, tabpfn3, tabpfn35, kumo,
  hist_mogp, hist_gpd

Example:
  conda activate gpplus2026
  python compare_foundation_baselines.py --preview-a06
  python compare_foundation_baselines.py --problems A01,A06 --models midline,exact_gp,tabpfn35
  python compare_foundation_baselines.py --problems all --models midline,mdn,mdn_const
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
for p in (_ROOT, _EXP, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Pin this folder's ``diagnostics`` before the QSAR loader puts
# experiments_SORF (which has a ``diagnostics`` package) on sys.path.
import diagnostics  # noqa: E402,F401
import gpplus
from foundation_adapters import (
    fit_predict_exact_gp,
    fit_predict_hist_gpd,
    fit_predict_hist_mogp,
    fit_predict_hist_t1,
    fit_predict_midline_gamma_auto,
    fit_predict_midline_gamma_t1,
    fit_predict_midline_bayes_mogp,
    fit_predict_midline_residual_mogp,
    fit_predict_midline_sparse,
    fit_predict_mdn,
    fit_predict_mdn_const,
    fit_predict_ni_sorf_adapter,
    fit_predict_kumotabular,
    fit_predict_tabgp,
    fit_predict_tabpfn,
)
from load_bimodal_data import generate_bimodal_data
from load_lognormal_hetero_data import (
    A06_BUMP_A,
    A06_BUMP_C,
    A06_BUMP_W,
    A06_KAPPA,
    A06_NOISE_FLOOR,
    A06_SIGMA_MAX,
    A06_SIGMA_MIN,
    A06_SIGMA_POWER,
    A06_X_HI,
    A06_X_LO,
    A06_Y_MAX,
    A06_Y_MIN,
    a06_mean,
    a06_noise,
    a06_sigma_log,
    generate_lognormal_hetero_data,
)
from load_trimodal_data import (
    TRIMODAL_Y_MAX,
    TRIMODAL_Y_MIN,
    generate_trimodal_data,
)
from load_unimodal_hetero_data import (
    A05_Y_MAX,
    A05_Y_MIN,
    generate_unimodal_hetero_data,
)
from plot_foundation_marginals import (
    probe_x_values,
    save_conditionals_multid,
    save_marginals_per_probe,
    save_posterior_vs_data,
    save_posterior_vs_data_multid,
    slice_pred,
)
from scoringbench_metrics import (
    SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
    compute_scoringbench_style_metrics,
    fill_baseline_rrmse,
    print_metrics_table,
)

# MDN head-to-head artifacts (champion MidlineSparse vs Bishop MDN).
# Prior Oct27 suite lived under results/foundation_compare_Oct27_cap10.
OUT_DIR = _THIS / "results" / "foundation_compare_mdn"
FIG_DIR = OUT_DIR / "figures"

DEFAULT_TRAIN_SIZES = (100, 200, 400, 800)
DEFAULT_N_TEST = 1000

COMPARE_COLS = ["RMSE", "RRMSE", "NIS", "CRPS", "log_score"]
# Extra columns shown on the metrics PNG / console table (cost + coverage).
# Density-match stats (Hellinger / TV / |k_modes| err) stay in CSV/JSON only.
TABLE_EXTRA_COLS = [
    "cov50",
    "cov90",
    "wall_s",
]
DENSITY_COLS = ("hellinger_mean", "tv_mean", "hellinger_max", "tv_max", "abs_k_modes_err")
ALL_PROBLEMS = (
    "A01",
    "unimodal",
    "A02",
    "A04",
    "A05",
    "A06",
    "COSI",
    "TABARENA_CONCRETE",
    "QSAR_FISH",
)
ALL_MODELS = (
    "midline",
    "midline_gamma_mlp",
    "midline_residual_mogp",
    "midline_bayes_mogp",
    "mdn",
    "mdn_const",
    "exact_gp",
    "tabgp",
    "tabpfn3",
    "tabpfn35",
    "kumo",
    "hist_mogp",
    "hist_gpd",
)

# Tier-1/2 ablation variants of the gamma-MLP midline (see
# docs/gamma_mlp_hist_mogp_analysis.md). Values are kwargs for
# ``fit_predict_midline_gamma_t1``.
GAMMA_VARIANTS: dict[str, dict[str, Any]] = {
    "midline_gamma_mlp_std": dict(standardize=True),
    "midline_gamma_mlp_cap05": dict(standardize=True, std0_cap_frac=0.5),
    "midline_gamma_mlp_cap10": dict(standardize=True, std0_cap_frac=1.0),
    "midline_gamma_mlp_temp": dict(standardize=True, calibrate_temperature=True),
    "midline_gamma_mlp_safe": dict(standardize=True, soft_safety=True),
    "midline_gamma_mlp_t1": dict(
        standardize=True,
        std0_cap_frac=0.5,
        calibrate_temperature=True,
        soft_safety=True,
        num_inits=4,
    ),
}
GAMMA_VARIANTS.update(
    {
        "midline_gamma_mlp_vexp": {
            **GAMMA_VARIANTS["midline_gamma_mlp_t1"],
            "model_kwargs": {"var_expand": True},
        },
        "midline_gamma_mlp_het": {
            **GAMMA_VARIANTS["midline_gamma_mlp_t1"],
            "model_kwargs": {"het_stage1": True},
        },
        "midline_gamma_mlp_t2": {
            **GAMMA_VARIANTS["midline_gamma_mlp_t1"],
            "model_kwargs": {"var_expand": True, "het_stage1": True},
        },
    }
)
# Values are kwargs for ``fit_predict_hist_t1`` (base_fn resolved at dispatch).
HIST_VARIANTS: dict[str, dict[str, Any]] = {
    "hist_mogp_tail": dict(bin_grid=None, eps_grid=None, tail_frac=1.0),
    "hist_mogp_floor": dict(bin_grid=None, tail_frac=0.0),
    "hist_mogp_bsel": dict(eps_grid=None, tail_frac=0.0),
    "hist_mogp_t1": dict(tail_frac=1.0),
    # Tier 2: posterior-variance-aware simplex mapping + categorical-aligned GPD.
    "hist_mogp_t1s": dict(
        tail_frac=1.0, base_kwargs={"sample_project": True, "n_proj_samples": 128}
    ),
    "hist_gpd_t1": dict(
        tail_frac=1.0,
        base="hist_gpd",
        base_kwargs={"standardize_x": True, "num_steps": 150},
    ),
    "hist_gpd_t2": dict(
        tail_frac=1.0,
        base="hist_gpd",
        base_kwargs={"standardize_x": True, "num_steps": 150, "n_mc": 128},
    ),
}
# Tier 3: self-configuring gamma (kwargs for ``fit_predict_midline_gamma_auto``).
AUTO_VARIANTS: dict[str, dict[str, Any]] = {
    # Tier-3 acceptance rule (conf ≥ 0.8); EM speedups apply to every key.
    "midline_gamma_auto": dict(conf_min=0.8),
    # Tier 4: conf ≥ 0.85 (default), calib-mask acceptance, forced SORF Stage 1.
    "midline_gamma_auto_c85": {},
    "midline_gamma_auto_calibaccept": dict(accept="calib"),
    "midline_gamma_auto_sorf": dict(loo_backend="sorf"),
    "midline_gamma_auto_sorf_nigp": dict(loo_backend="sorf", loo_nigp=True),
    "midline_gamma_auto_nocal": dict(calibrate=False),
    "midline_gamma_auto_nohet": dict(het=False),
    "midline_gamma_auto_nomean": dict(loo_mean=False),
    "midline_gamma_auto_1mid": dict(multi_midline=False),
    "midline_gamma_auto_rotcal": dict(calib_rotate=True),
    "midline_gamma_auto_vexp": dict(var_expand=True),
}
# Tier 3 density models; values are kwargs for the adapter named by ``fn``.
DENSITY_VARIANTS: dict[str, dict[str, Any]] = {
    "quantile_gp": dict(fn="quantile_gp", mode="joint", n_levels=19),
    "quantile_gp_indep": dict(fn="quantile_gp", mode="indep", n_levels=19),
    "quantile_gp_dense": dict(fn="quantile_gp", mode="joint", n_levels=49),
    "logistic_gp": dict(fn="logistic_gp"),
    "hist_lmc_t1": dict(fn="hist_lmc"),
    # Tier 4: padded-SORF features + Stage-1 LOO midline.
    "logistic_gp_sorf_vi": dict(fn="logistic_gp_sorf", inference="vi"),
    "logistic_gp_sorf_map": dict(fn="logistic_gp_sorf", inference="map"),
    "logistic_gp_sorf_gamma": dict(fn="logistic_gp_sorf", inference="map", midline="gamma"),
    "quantile_gp_hybrid": dict(fn="quantile_gp", mode="joint", n_levels=19, features="sorf", lam_nll=1.0),
    "quantile_gp_hybrid_l01": dict(
        fn="quantile_gp", mode="joint", n_levels=19, features="sorf", lam_nll=0.1
    ),
    "quantile_gp_sorf": dict(fn="quantile_gp", mode="joint", n_levels=19, features="sorf"),
    "quantile_gp_hybrid_auto": dict(
        fn="quantile_gp", mode="joint", n_levels=19, features="sorf", lam_nll="auto"
    ),
}
ALL_MODELS = (
    ALL_MODELS
    + tuple(GAMMA_VARIANTS)
    + tuple(HIST_VARIANTS)
    + tuple(AUTO_VARIANTS)
    + tuple(DENSITY_VARIANTS)
)
# Deprecated (kept runnable by name): sum-of-Kronecker hist GP and the
# variance-expansion gamma variants.
DEPRECATED_MODELS = frozenset(
    m
    for m in ALL_MODELS
    if m.startswith("hist_lmc") or m.endswith("_vexp") or m == "midline_gamma_mlp_t2"
)
DEFAULT_MODELS = tuple(m for m in ALL_MODELS if m not in DEPRECATED_MODELS)

COSI_DATA_PATH = (
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
    / "snow_toa_fsnow_pure_flat_Sep09.nc"
)
COSI_DEFAULT_TRAIN = 5000
COSI_DEFAULT_TEST = 5000

# TabArena-v0.1 concrete_compressive_strength (OpenML 46917): 80/20 of N=1030
TABARENA_CONCRETE_DEFAULT_TRAIN = 824
TABARENA_CONCRETE_DEFAULT_TEST = 206
# UCI QSAR fish toxicity (N=908): ~80/20
QSAR_FISH_DEFAULT_TRAIN = 726
QSAR_FISH_DEFAULT_TEST = 182

MODEL_LABELS = {
    "midline": "2-Stage Mixture GP",
    "midline_gamma_mlp": "2-Stage Mixture GP (gamma MLP)",
    "midline_residual_mogp": "Midline Residual MoGP",
    "midline_bayes_mogp": "Midline Bayes MoGP",
    "mdn": "Bishop MDN (MLP gate)",
    "mdn_const": "Bishop MDN (const gate)",
    "exact_gp": "Exact GP+ (GPR)",
    "ni_sorf": "NI-SORF GP",
    "tabgp": "TabGP-MLL",
    "tabpfn3": "TabPFN-3.0",
    "tabpfn35": "TabPFN-3.5",
    "kumo": "NVIDIA KumoTabular",
    "tabfm": "NVIDIA KumoTabular",  # deprecated alias
    "hist_mogp": "Hist multi-output GP",
    "hist_gpd": "Hist Dirichlet GP",
    **{k: f"Gamma MLP [{k.removeprefix('midline_gamma_mlp_')}]" for k in GAMMA_VARIANTS},
    **{
        k: (
            f"Hist MoGP [{k.removeprefix('hist_mogp_')}]"
            if k.startswith("hist_mogp_")
            else f"Hist Dirichlet GP [{k.removeprefix('hist_gpd_')}]"
        )
        for k in HIST_VARIANTS
    },
    **{
        k: f"Gamma MLP auto [{k.removeprefix('midline_gamma_auto').lstrip('_') or 'full'}]"
        for k in AUTO_VARIANTS
    },
    "quantile_gp": "Quantile GP (pinball, joint τ)",
    "quantile_gp_indep": "Quantile GP (pinball, indep τ)",
    "quantile_gp_dense": "Quantile GP (pinball, J=49)",
    "logistic_gp": "Logistic-GP density (RFF)",
    "hist_lmc_t1": "Hist sum-of-Kronecker GP [t1]",
    "logistic_gp_sorf_vi": "Logistic-GP (SORF, VI, LOO midline)",
    "logistic_gp_sorf_map": "Logistic-GP (SORF, MAP, LOO midline)",
    "logistic_gp_sorf_gamma": "Logistic-GP (SORF, MAP, gamma midline)",
    "quantile_gp_hybrid": "Quantile GP (SORF, pinball+NLL λ=1)",
    "quantile_gp_hybrid_l01": "Quantile GP (SORF, pinball+NLL λ=0.1)",
    "quantile_gp_sorf": "Quantile GP (SORF, pinball)",
    "quantile_gp_hybrid_auto": "Quantile GP (SORF, pinball+NLL, λ auto)",
}

# Static problem metadata (data filled by load_problem_pool / slice_train).
_PROBLEM_META: dict[str, dict[str, Any]] = {
    "A01": {
        "true_k_modes": 2,
        "k_max": 5,
        "champion": True,
        "y_min": -2.0,
        "y_max": 30.0,
        "num_steps": 1000,
    },
    "unimodal": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        "y_min": -2.0,
        "y_max": 12.0,
        "num_steps": 1000,
    },
    "A02": {
        "true_k_modes": 2,
        "k_max": 5,
        "champion": True,
        "y_min": -0.5,
        "y_max": 1.5,
        "num_steps": 1000,
    },
    "A04": {
        "true_k_modes": 3,
        "k_max": 5,
        "champion": True,
        "y_min": TRIMODAL_Y_MIN,
        "y_max": TRIMODAL_Y_MAX,
        "num_steps": 1000,
    },
    "A05": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        "y_min": A05_Y_MIN,
        "y_max": A05_Y_MAX,
        "num_steps": 1000,
    },
    "A06": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        "y_min": A06_Y_MIN,
        "y_max": A06_Y_MAX,
        "num_steps": 1000,
    },
    "COSI": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        "y_min": 0.06,
        "y_max": 1.0,
        "num_steps": 800,
        "midline_backend": "ni_sorf",
        "hist_n_bins": 16,
        "ni_sorf_num_rff": 2400,
        "ni_sorf_num_epochs": 1000,
        "ni_sorf_residual_num_epochs": 300,
        "ni_sorf_residual_num_rff": 1024,
        "ni_sorf_hist_num_rff": 512,
        "ni_sorf_hist_num_epochs": 150,
    },
    "TABARENA_CONCRETE": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        # Wide fixed bounds covering OpenML 46917 response (~2–83 MPa)
        "y_min": 0.0,
        "y_max": 100.0,
        "num_steps": 800,
        "midline_backend": "exact",
        "hist_n_bins": 200,
        "hist_num_steps": 1000,
        "hist_lr": 0.03,
    },
    "QSAR_FISH": {
        "true_k_modes": 1,
        "k_max": 5,
        "champion": True,
        "y_min": 0.0,
        "y_max": 10.0,
        "num_steps": 800,
        "midline_backend": "exact",
        "hist_n_bins": 200,
        "hist_num_steps": 1000,
        "hist_lr": 0.03,
    },
}


def _json_dump(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Data — generate pool once, freeze test, nested train prefixes
# ---------------------------------------------------------------------------
def _generate_pool_xy(
    name: str,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit ``n`` (X, y) rows for ``name`` in one shot (absolute count)."""
    key = name.strip()
    if key == "A01":
        X, y, _ = generate_bimodal_data(n_samples=n, seed=seed)
        return X, y
    if key == "unimodal":
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 10.0, size=(n, 1))
        y = 0.5 * x[:, 0] + 2.0 + 0.3 * rng.normal(size=n)
        return x.astype(np.float64), y.astype(np.float64)
    if key == "A02":
        rng = np.random.default_rng(seed)
        noise = 0.05
        x = rng.uniform(0.0, 1.0, size=(n, 1))
        y = (x[:, 0] >= 0.5).astype(np.float64) + noise * rng.normal(size=n)
        return x.astype(np.float64), y.astype(np.float64)
    if key == "A04":
        X, y, _ = generate_trimodal_data(n_samples=n, seed=seed)
        return X, y
    if key == "A05":
        X, y, _ = generate_unimodal_hetero_data(n_samples=n, seed=seed)
        return X, y
    if key == "A06":
        return generate_lognormal_hetero_data(n_samples=n, seed=seed)
    raise ValueError(f"Unknown problem {name!r}; choose from {ALL_PROBLEMS}")


def _load_cosi_xy(
    *,
    n_train: int,
    n_test: int,
    seed: int = 42,
    data_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """S4 snow-TOA ``cos_i``: return X_tr, y_tr, X_te, y_te (5k/5k style)."""
    import sys

    mtgpr = _ROOT / "experiments_RFFMTGPR"
    if str(mtgpr) not in sys.path:
        sys.path.insert(0, str(mtgpr))
    from emit_s4_mtgpr_base import load_emit_s4_xy, split_emit_s4

    path = Path(data_path) if data_path is not None else COSI_DATA_PATH
    X, Y, _mask, _meta = load_emit_s4_xy(path, task_names=["cos_i"])
    y = np.asarray(Y[:, 0], dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n_train + n_test > n:
        raise ValueError(
            f"COSI pool too small: need n_train+n_test={n_train + n_test}, got n={n}"
        )
    # n_val = leftover after train+test so split_emit_s4 leaves exactly n_test unused
    # as "test", then we take the first n_test of that leftover.
    n_val = n - n_train - n_test
    tr_idx, va_idx, te_rest = split_emit_s4(
        n, n_train=n_train, n_val=n_val, seed=seed
    )
    te_idx = np.asarray(te_rest[:n_test], dtype=np.int64)
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


def _load_tabarena_concrete_xy(
    *,
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OpenML 46917 concrete: return X_tr, y_tr, X_te, y_te (80/20 style)."""
    from load_tabarena_concrete import (
        load_tabarena_concrete_xy,
        split_tabarena_concrete,
    )

    X, y, _meta = load_tabarena_concrete_xy()
    return split_tabarena_concrete(
        X, y, n_train=int(n_train), n_test=int(n_test), seed=int(seed)
    )


def _load_qsar_fish_xy(
    *,
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UCI QSAR fish toxicity: return X_tr, y_tr, X_te, y_te (~80/20)."""
    from load_qsar_fish import load_qsar_fish_xy, split_qsar_fish

    X, y, _meta = load_qsar_fish_xy()
    return split_qsar_fish(
        X, y, n_train=int(n_train), n_test=int(n_test), seed=int(seed)
    )


def load_problem_pool(
    name: str,
    *,
    n_train_max: int,
    n_test: int = DEFAULT_N_TEST,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate pool once; freeze test indices and nested train-pool indices.

    Test set is ``perm[:n_test]``; train pool is ``perm[n_test:]`` of length
    ``n_train_max``. Call ``slice_train`` for each train size — do not regenerate.

    Special cases ``COSI`` / ``TABARENA_CONCRETE`` / ``QSAR_FISH``: fixed
    train/test slices (no nested synthetic generation).
    """
    key = name.strip()
    if key not in _PROBLEM_META:
        raise ValueError(f"Unknown problem {name!r}; choose from {ALL_PROBLEMS}")
    if n_train_max < 1 or n_test < 1:
        raise ValueError("n_train_max and n_test must be >= 1")

    meta = dict(_PROBLEM_META[key])
    if key in ("COSI", "TABARENA_CONCRETE", "QSAR_FISH"):
        if key == "COSI":
            X_tr, y_tr, X_te, y_te = _load_cosi_xy(
                n_train=int(n_train_max), n_test=int(n_test), seed=seed
            )
        elif key == "TABARENA_CONCRETE":
            X_tr, y_tr, X_te, y_te = _load_tabarena_concrete_xy(
                n_train=int(n_train_max), n_test=int(n_test), seed=seed
            )
        else:
            X_tr, y_tr, X_te, y_te = _load_qsar_fish_xy(
                n_train=int(n_train_max), n_test=int(n_test), seed=seed
            )
        # Pack so slice_train(n_train_max) recovers the full train set.
        X = np.vstack([X_te, X_tr])
        y = np.concatenate([y_te, y_tr])
        te_idx = np.arange(int(n_test), dtype=np.int64)
        tr_pool_idx = np.arange(int(n_test), int(n_test) + int(n_train_max), dtype=np.int64)
        return {
            "name": key,
            "X": X,
            "y": y,
            "te_idx": te_idx,
            "tr_pool_idx": tr_pool_idx,
            "x_te": torch.as_tensor(X_te, dtype=torch.float64),
            "y_te": torch.as_tensor(y_te, dtype=torch.float64),
            "n_test": int(n_test),
            "n_train_max": int(n_train_max),
            "seed": int(seed),
            **meta,
        }

    n_total = int(n_train_max) + int(n_test)
    X, y = _generate_pool_xy(key, n_total, seed)
    assert X.shape[0] == n_total and y.shape[0] == n_total

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    te_idx = np.asarray(perm[:n_test], dtype=np.int64)
    tr_pool_idx = np.asarray(perm[n_test:], dtype=np.int64)
    if tr_pool_idx.shape[0] != n_train_max:
        raise RuntimeError(
            f"train pool size {tr_pool_idx.shape[0]} != n_train_max={n_train_max}"
        )

    x_te = torch.as_tensor(X[te_idx], dtype=torch.float64)
    y_te = torch.as_tensor(y[te_idx], dtype=torch.float64)
    return {
        "name": key,
        "X": X,
        "y": y,
        "te_idx": te_idx,
        "tr_pool_idx": tr_pool_idx,
        "x_te": x_te,
        "y_te": y_te,
        "n_test": int(n_test),
        "n_train_max": int(n_train_max),
        "seed": int(seed),
        **meta,
    }


def slice_train(pool: dict[str, Any], n_train: int) -> dict[str, Any]:
    """Nested train prefix of size ``n_train``; same frozen ``x_te`` / ``y_te``."""
    n_train = int(n_train)
    if n_train < 1 or n_train > int(pool["n_train_max"]):
        raise ValueError(
            f"n_train={n_train} out of range [1, {pool['n_train_max']}]"
        )
    tr_idx = pool["tr_pool_idx"][:n_train]
    X, y = pool["X"], pool["y"]
    meta_keys = (
        "true_k_modes",
        "k_max",
        "champion",
        "y_min",
        "y_max",
        "num_steps",
        "midline_backend",
        "hist_n_bins",
        "ni_sorf_num_rff",
        "ni_sorf_num_epochs",
        "ni_sorf_residual_num_epochs",
        "ni_sorf_residual_num_rff",
        "ni_sorf_hist_num_rff",
        "ni_sorf_hist_num_epochs",
    )
    return {
        "name": pool["name"],
        "x_tr": torch.as_tensor(X[tr_idx], dtype=torch.float64),
        "y_tr": torch.as_tensor(y[tr_idx], dtype=torch.float64),
        "x_te": pool["x_te"],
        "y_te": pool["y_te"],
        "n_train": n_train,
        "n_test": int(pool["n_test"]),
        "tr_idx": tr_idx,
        **{k: pool[k] for k in meta_keys if k in pool},
    }


def load_problem(
    name: str,
    *,
    n_train: int = 800,
    n_test: int = DEFAULT_N_TEST,
    seed: int = 42,
    n_samples: int | None = None,
    test_size: float | None = None,
) -> dict[str, Any]:
    """Single (train, test) slice. Prefer ``load_problem_pool`` + ``slice_train``.

    Deprecated ``n_samples`` / ``test_size`` map to one train size and ``n_test``
    when provided (for older callers).
    """
    if n_samples is not None and test_size is not None:
        n_te = max(1, int(test_size * n_samples))
        n_tr = max(1, int(n_samples) - n_te)
        pool = load_problem_pool(
            name, n_train_max=n_tr, n_test=n_te, seed=seed
        )
        return slice_train(pool, n_tr)
    pool = load_problem_pool(
        name, n_train_max=int(n_train), n_test=int(n_test), seed=seed
    )
    return slice_train(pool, int(n_train))


def preview_a06(
    *,
    n_plot: int = 800,
    seed: int = 42,
    out_path: Path | None = None,
) -> Path:
    """Scatter of A06 noisy targets around f(x); no model fits."""
    out_path = out_path or (FIG_DIR / "a06_noise_preview.png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    x = rng.uniform(A06_X_LO, A06_X_HI, size=n_plot)
    noise = a06_noise(rng, x)
    y = a06_mean(x) + noise

    x_grid = np.linspace(A06_X_LO, A06_X_HI, 400)
    f_grid = a06_mean(x_grid)
    sigma = a06_sigma_log(x_grid)
    # Approximate central band via many noise draws at grid (for visual scale).
    band_rng = np.random.default_rng(seed + 1)
    n_mc = 200
    draws = np.empty((n_mc, x_grid.shape[0]), dtype=np.float64)
    for i in range(n_mc):
        draws[i] = a06_noise(band_rng, x_grid)
    q_lo = f_grid + np.quantile(draws, 0.05, axis=0)
    q_hi = f_grid + np.quantile(draws, 0.95, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=140, sharex=True)
    ax = axes[0]
    ax.fill_between(x_grid, q_lo, q_hi, color="#90cdf4", alpha=0.45, label="5–95% noise band")
    ax.scatter(x, y, s=10, alpha=0.45, c="#2c5282", label="noisy samples", zorder=2)
    ax.plot(
        x_grid,
        f_grid,
        color="#c53030",
        lw=2.0,
        label=(
            rf"$f=$ cubic $+\,{A06_BUMP_A:g}"
            rf"\exp(-(x-{A06_BUMP_C:g})^2/(2\cdot{A06_BUMP_W:g}^2))$"
        ),
        zorder=3,
    )
    ax.set_ylabel("y")
    ax.set_title(
        "A06 preview: cubic + steep bump + strong centered lognormal noise"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    scale = A06_KAPPA * np.abs(f_grid) + A06_NOISE_FLOOR
    ax2.plot(x_grid, sigma, color="#2b6cb0", lw=2.0, label=r"$\sigma(x)$ (log-normal scale)")
    ax2.plot(
        x_grid,
        scale,
        color="#dd6b20",
        lw=1.8,
        label=(
            rf"$\kappa|f|+\varepsilon_0$ "
            rf"($\kappa={A06_KAPPA}$, $\varepsilon_0={A06_NOISE_FLOOR}$)"
        ),
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("scale")
    ax2.set_title(
        rf"Heteroscedasticity: $\sigma\in[{A06_SIGMA_MIN},{A06_SIGMA_MAX}]$, "
        rf"power={A06_SIGMA_POWER}"
    )
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
def run_model(
    model_key: str,
    problem: dict[str, Any],
    *,
    seed: int = 42,
    device: str | None = None,
    x_probe: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    x_tr, y_tr = problem["x_tr"], problem["y_tr"]
    x_te, y_te = problem["x_te"], problem["y_te"]
    Xtr = x_tr.detach().cpu().numpy()
    ytr = y_tr.detach().cpu().numpy().ravel()
    Xte = x_te.detach().cpu().numpy()
    y_np = y_te.detach().cpu().numpy().ravel()
    n_te = int(Xte.shape[0])
    d = int(Xte.shape[1])

    # One predict for test + probes + dense grid (TabGP trains on first matrix)
    blocks: list[np.ndarray] = [Xte]
    n_probe = 0
    n_grid = 0
    if x_probe is not None and len(x_probe):
        Xp = np.asarray(x_probe, dtype=np.float64).reshape(-1, d)
        blocks.append(Xp)
        n_probe = int(Xp.shape[0])
    else:
        Xp = None
    if x_grid is not None and len(x_grid):
        Xg = np.asarray(x_grid, dtype=np.float64).reshape(-1, d)
        blocks.append(Xg)
        n_grid = int(Xg.shape[0])
    else:
        Xg = None

    X_eval = np.vstack(blocks) if len(blocks) > 1 else Xte
    x_eval_t = torch.as_tensor(X_eval, dtype=torch.float64)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_ni_sorf = str(problem.get("midline_backend", "exact")) == "ni_sorf"
    # Match S4_emit_SORF: single Adam init for NI-SORF Stage-1 (expensive).
    midline_num_inits = 1 if use_ni_sorf else 2
    ni_kwargs = dict(
        midline_backend="ni_sorf" if use_ni_sorf else "exact",
        ni_sorf_num_rff=int(problem.get("ni_sorf_num_rff", 2400)),
        ni_sorf_num_epochs=int(problem.get("ni_sorf_num_epochs", 1000)),
        ni_sorf_lr=1e-2,
        num_inits=midline_num_inits,
    )
    ni_resid_kwargs = dict(
        **ni_kwargs,
        ni_sorf_residual_num_rff=int(problem.get("ni_sorf_residual_num_rff", 1024)),
        ni_sorf_residual_num_epochs=int(
            problem.get("ni_sorf_residual_num_epochs", 300)
        ),
    )

    if model_key == "midline":
        pred_all = fit_predict_midline_sparse(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            gamma_type="constant",
            **ni_kwargs,
        )
    elif model_key == "midline_gamma_mlp":
        pred_all = fit_predict_midline_sparse(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            gamma_type="mlp",
            **ni_kwargs,
        )
    elif model_key == "midline_residual_mogp":
        pred_all = fit_predict_midline_residual_mogp(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            seed=seed,
            gate_type="mlp",
            device=device,
            **ni_resid_kwargs,
        )
    elif model_key == "midline_bayes_mogp":
        pred_all = fit_predict_midline_bayes_mogp(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            seed=seed,
            gate_type="mlp",
            device=device,
            **ni_resid_kwargs,
        )
    elif model_key == "mdn":
        pred_all = fit_predict_mdn(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=max(2000, int(problem["num_steps"])),
            seed=seed,
            constant_gate=False,
            device=device,
        )
    elif model_key == "mdn_const":
        pred_all = fit_predict_mdn_const(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=max(2000, int(problem["num_steps"])),
            seed=seed,
            device=device,
        )
    elif model_key in ("exact_gp", "ni_sorf"):
        if use_ni_sorf or model_key == "ni_sorf":
            # Hyperspectral / COSI: Exact GP replaced by NI-SORF baseline
            pred_all = fit_predict_ni_sorf_adapter(
                x_tr,
                y_tr,
                x_eval_t,
                seed=seed,
                device=device,
                num_rff=int(problem.get("ni_sorf_num_rff", 2400)),
                num_epochs=int(problem.get("ni_sorf_num_epochs", 1000)),
            )
        else:
            pred_all = fit_predict_exact_gp(
                x_tr, y_tr, x_eval_t, seed=seed, device="cpu"
            )
    elif model_key == "tabgp":
        pred_all = fit_predict_tabgp(Xtr, ytr, X_eval, seed=seed, device=device)
    elif model_key == "tabpfn3":
        pred_all = fit_predict_tabpfn(
            Xtr, ytr, X_eval, version="v3", seed=seed, device=device
        )
    elif model_key == "tabpfn35":
        pred_all = fit_predict_tabpfn(
            Xtr, ytr, X_eval, version="v3.5", seed=seed, device=device
        )
    elif model_key in ("kumo", "tabfm"):
        pred_all = fit_predict_kumotabular(Xtr, ytr, X_eval, device=device)
    elif model_key == "hist_mogp":
        hist_backend = "ni_sorf" if use_ni_sorf else "exact"
        pred_all = fit_predict_hist_mogp(
            x_tr,
            y_tr,
            x_eval_t,
            n_bins=int(problem.get("hist_n_bins", 200)),
            seed=seed,
            device="cpu" if hist_backend == "exact" else device,
            backend=hist_backend,
            num_steps=int(problem.get("hist_num_steps", 1000)),
            lr=float(problem.get("hist_lr", 0.03)),
            num_inits=int(problem.get("hist_num_inits", 2)),
            bin_scheme=str(problem.get("hist_bin_scheme", "equal_width")),
            ni_sorf_num_rff=int(problem.get("ni_sorf_hist_num_rff", 512)),
            ni_sorf_num_epochs=int(problem.get("ni_sorf_hist_num_epochs", 150)),
        )
    elif model_key == "hist_gpd":
        hist_backend = "ni_sorf" if use_ni_sorf else "exact"
        pred_all = fit_predict_hist_gpd(
            x_tr,
            y_tr,
            x_eval_t,
            n_bins=int(problem.get("hist_n_bins", 200)),
            seed=seed,
            device="cpu" if hist_backend == "exact" else device,
            backend=hist_backend,
            ni_sorf_num_rff=int(problem.get("ni_sorf_hist_num_rff", 512)),
            ni_sorf_num_epochs=int(problem.get("ni_sorf_hist_num_epochs", 150)),
        )
    elif model_key in GAMMA_VARIANTS:
        vkw = dict(GAMMA_VARIANTS[model_key])
        vkw.setdefault("num_inits", midline_num_inits)
        nk = {k: v for k, v in ni_kwargs.items() if k != "num_inits"}
        pred_all = fit_predict_midline_gamma_t1(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            gamma_type="mlp",
            **nk,
            **vkw,
        )
    elif model_key in AUTO_VARIANTS:
        pred_all = fit_predict_midline_gamma_auto(
            x_tr,
            y_tr,
            x_eval_t,
            k_max=int(problem["k_max"]),
            num_steps=int(problem["num_steps"]),
            seed=seed,
            champion=bool(problem["champion"]),
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            gamma_type="mlp",
            **AUTO_VARIANTS[model_key],
        )
    elif model_key in DENSITY_VARIANTS:
        vkw = dict(DENSITY_VARIANTS[model_key])
        fn = vkw.pop("fn")
        if fn == "quantile_gp":
            from quantile_gp import fit_predict_quantile_gp as density_fn
        elif fn == "logistic_gp":
            from logistic_gp_density import fit_predict_logistic_gp as density_fn
        elif fn == "logistic_gp_sorf":
            from logistic_gp_density import fit_predict_logistic_gp_sorf as density_fn
        elif fn == "hist_lmc":
            from hist_lmc import fit_predict_hist_lmc_t1 as density_fn
        else:
            raise ValueError(f"Unknown density adapter {fn!r}")
        pred_all = density_fn(x_tr, y_tr, x_eval_t, seed=seed, **vkw)
    elif model_key in HIST_VARIANTS:
        vkw = dict(HIST_VARIANTS[model_key])
        base = vkw.pop("base", "hist_mogp")
        base_kw: dict[str, Any] = dict(device="cpu")
        if base == "hist_mogp":
            base_fn = fit_predict_hist_mogp
            base_kw.update(
                num_steps=int(problem.get("hist_num_steps", 1000)),
                lr=float(problem.get("hist_lr", 0.03)),
                num_inits=int(problem.get("hist_num_inits", 2)),
            )
        else:
            base_fn = fit_predict_hist_gpd
        base_kw.update(vkw.pop("base_kwargs", {}))
        pred_all = fit_predict_hist_t1(
            x_tr,
            y_tr,
            x_eval_t,
            base_fn=base_fn,
            n_bins=int(problem.get("hist_n_bins", 200)),
            seed=seed,
            **vkw,
            **base_kw,
        )
    else:
        raise ValueError(f"Unknown model {model_key!r}")

    i0, i1 = 0, n_te
    pred = slice_pred(pred_all, i0, i1)
    i0 = i1
    i1 = i0 + n_probe
    pred_probe = slice_pred(pred_all, i0, i1) if n_probe else None
    i0 = i1
    i1 = i0 + n_grid
    pred_grid = slice_pred(pred_all, i0, i1) if n_grid else None

    # Native bar NLL for histogram PPDs; keep Gaussian log_score in extras.
    bar_lp = None
    extras_pre = dict(pred_all.get("extras") or {})
    if (
        extras_pre.get("ppd_source") == "bar_distribution"
        and pred.get("pred_borders") is not None
        and pred.get("pred_logits") is not None
    ):
        from histogram_gp import bar_log_prob

        bar_lp = bar_log_prob(y_np, pred["pred_borders"], pred["pred_logits"])

    metrics = compute_scoringbench_style_metrics(
        y_np,
        pred["mixture_mean"],
        pred["expert_means"],
        pred["expert_stds"],
        pred["gate_weights"],
        log_prob=bar_lp,
    )
    extras = extras_pre
    if bar_lp is not None:
        gauss_metrics = compute_scoringbench_style_metrics(
            y_np,
            pred["mixture_mean"],
            pred["expert_means"],
            pred["expert_stds"],
            pred["gate_weights"],
            log_prob=None,
        )
        extras["log_score_gaussian"] = float(gauss_metrics["log_score"])
        extras["log_score_bar"] = float(metrics["log_score"])
        from histogram_gp import bar_crps

        extras["CRPS_bar"] = float(
            np.mean(bar_crps(y_np, pred["pred_borders"], pred["pred_logits"]))
        )
    else:
        # Exact (closed-form) CRPS for Gaussian-mixture PPDs, same column, so
        # every model is ranked on an exact CRPS.
        from histogram_gp import mixture_crps

        extras["CRPS_bar"] = float(
            np.mean(
                mixture_crps(
                    y_np, pred["expert_means"], pred["expert_stds"], pred["gate_weights"]
                )
            )
        )
        extras["CRPS_exact_source"] = "gaussian_mixture"
    if pred.get("quantile_values") is not None and "tail_kappa" in extras:
        from quantile_gp import quantile_crps

        extras["CRPS_quantile"] = float(
            np.mean(
                quantile_crps(
                    pred["quantile_values"],
                    pred["quantile_levels"],
                    y_np,
                    kappa=float(extras["tail_kappa"]),
                    xi=float(extras["tail_xi"]),
                )
            )
        )
    display_key = model_key
    if model_key == "exact_gp" and str(problem.get("midline_backend", "exact")) == "ni_sorf":
        display_key = "ni_sorf"
    row: dict[str, Any] = {
        "Model": MODEL_LABELS.get(display_key, model_key),
        "model_key": model_key,
        "source": "live",
        **{c: float(metrics[c]) for c in COMPARE_COLS if c in metrics},
    }
    for k in ("k_eff", "k_modes_mean", "k_modes_round", "wall_s", "CRPS_bar", "CRPS_quantile"):
        if k in extras:
            row[k] = extras[k]
    return {
        "row": row,
        "metrics": metrics,
        "pred": pred,
        "pred_probe": pred_probe,
        "pred_grid": pred_grid,
        "extras": extras,
    }


def save_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "model_key", "source"] + COMPARE_COLS + [
        "cov50",
        "cov90",
        "CRPS_bar",
        "CRPS_quantile",
        "k_eff",
        "k_modes_mean",
        "k_modes_round",
        "wall_s",
        *DENSITY_COLS,
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def _fmt_table_cell(col: str, v: Any) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "—"
    if col == "wall_s":
        return f"{fv:.1f}"
    if col in ("cov50", "cov90"):
        return f"{fv:.3f}"
    return f"{fv:.4f}"


def save_metrics_table_png(
    rows: list[dict],
    out_path: Path,
    *,
    title: str,
) -> Path:
    metric_cols = COMPARE_COLS + [
        c for c in TABLE_EXTRA_COLS if any(r.get(c) is not None for r in rows)
    ]
    col_labels = ["Model"] + metric_cols
    cell_text = []
    for row in rows:
        cells = [str(row["Model"])]
        for c in metric_cols:
            cells.append(_fmt_table_cell(c, row.get(c)))
        cell_text.append(cells)

    fig_w = max(11.5, 1.35 * len(col_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 2.0 + 0.38 * len(rows)), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.08, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r >= 1 and rows[r - 1].get("model_key") == "midline":
            cell.set_facecolor("#ebf8ff")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.tight_layout()
    path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return path


def run_problem(
    problem_name: str,
    models: list[str],
    *,
    problem: dict[str, Any] | None = None,
    n_train: int | None = None,
    n_test: int = DEFAULT_N_TEST,
    seed: int = 42,
    device: str | None = None,
    include_website_baselines: bool = False,
    n_probes: int = 5,
    marginal_plots: bool = True,
    hist_n_bins: int = 200,
    hist_num_steps: int = 1000,
    # Deprecated aliases (single-split callers)
    n_samples: int | None = None,
    test_size: float | None = None,
) -> dict[str, Any]:
    if problem is None:
        if n_samples is not None and test_size is not None:
            problem = load_problem(
                problem_name,
                n_samples=n_samples,
                test_size=test_size,
                seed=seed,
            )
        else:
            n_tr = int(n_train) if n_train is not None else max(DEFAULT_TRAIN_SIZES)
            problem = load_problem(
                problem_name, n_train=n_tr, n_test=n_test, seed=seed
            )
    problem = dict(problem)
    problem["hist_n_bins"] = int(hist_n_bins)
    problem["hist_num_steps"] = int(hist_num_steps)
    n_train = int(problem.get("n_train") or problem["x_tr"].shape[0])
    n_test = int(problem.get("n_test") or problem["x_te"].shape[0])
    if int(problem["x_te"].shape[0]) != n_test:
        raise RuntimeError(
            f"frozen test size {problem['x_te'].shape[0]} != n_test={n_test}"
        )

    y_test = problem["y_te"].detach().cpu().numpy().ravel()
    x_tr_np = problem["x_tr"].detach().cpu().numpy()
    x_te_np = problem["x_te"].detach().cpu().numpy()
    y_tr_np = problem["y_tr"].detach().cpu().numpy().ravel()
    # 1D problems use x-probes / grids; d>1 plots conditionals at test points
    # chosen by predicted-median quantile instead (see below).
    input_dim = int(x_tr_np.shape[1]) if x_tr_np.ndim == 2 else 1
    if input_dim > 1:
        x_probes = None
        x_grid = None
    else:
        x_probes = probe_x_values(x_tr_np, x_te_np, n_probes=n_probes)
        x_lo = float(min(x_tr_np.min(), x_te_np.min()))
        x_hi = float(max(x_tr_np.max(), x_te_np.max()))
        x_grid = np.linspace(x_lo, x_hi, 200).reshape(-1, 1)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    probe_preds: dict[str, dict[str, np.ndarray]] = {}
    grid_preds: dict[str, dict[str, np.ndarray]] = {}

    if include_website_baselines and problem_name == "A01":
        # Historical TabICL / Crepes only — live TabPFN replaces hardcoded TabPFN
        hist = fill_baseline_rrmse(SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES, y_test=y_test)
        for name in ("TabICL", "Crepes(TabICL)"):
            base = hist.get(name)
            if not base:
                continue
            row = {"Model": name, "model_key": name, "source": "ScoringBench website"}
            for c in COMPARE_COLS:
                v = base.get(c)
                row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
            rows.append(row)

    te_preds: dict[str, dict[str, np.ndarray]] = {}
    for mk in models:
        print(
            f"[{problem_name} n_train={n_train}] fitting "
            f"{MODEL_LABELS.get(mk, mk)} ...",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            out = run_model(
                mk,
                problem,
                seed=seed,
                device=device,
                x_probe=x_probes,
                x_grid=x_grid,
            )
            wall = time.perf_counter() - t0
            row = dict(out["row"])
            # Prefer adapter wall_s; fall back to outer timer so cost always lands.
            if row.get("wall_s") is None:
                row["wall_s"] = float(wall)
            rows.append(row)
            results[mk] = {
                "metrics": out["metrics"],
                "extras": out["extras"],
            }
            if out.get("pred") is not None:
                te_preds[mk] = out["pred"]
            if out.get("pred_probe") is not None:
                probe_preds[mk] = out["pred_probe"]
            if out.get("pred_grid") is not None:
                grid_preds[mk] = out["pred_grid"]
            m = out["metrics"]
            print(
                f"  done in {wall:.1f}s | "
                f"RMSE={m['RMSE']:.4f} CRPS={m['CRPS']:.4f} "
                f"log_score={m['log_score']:.4f} wall_s={row['wall_s']:.1f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", flush=True)
            rows.append(
                {
                    "Model": MODEL_LABELS.get(mk, mk),
                    "model_key": mk,
                    "source": "error",
                    "error": str(exc),
                    **{c: None for c in COMPARE_COLS},
                }
            )
            results[mk] = {"error": str(exc)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{problem_name.lower()}_n{n_train}"
    table_json = OUT_DIR / f"{stem}_metrics.json"
    table_csv = OUT_DIR / f"{stem}_metrics.csv"
    table_png = FIG_DIR / f"{stem}_metrics_table.png"
    marginal_paths: list[str] = []
    posterior_png: Path | None = None
    coverage_bar_path: str | None = None

    # 50%/90% central predictive interval coverage (mixture quantile approx)
    from uq_calibration import interval_coverage, save_coverage_bar_chart

    cov_rows: dict[str, dict[str, float]] = {}
    for mk, pred in te_preds.items():
        try:
            cov = interval_coverage(
                y_test,
                pred["expert_means"],
                pred["expert_stds"],
                pred["gate_weights"],
                levels=(0.5, 0.9),
            )
            cov_rows[mk] = cov
            results[mk]["coverage"] = cov
            for r in rows:
                if r.get("model_key") == mk:
                    r["cov50"] = cov.get("cov50")
                    r["cov90"] = cov.get("cov90")
                    r["width50"] = cov.get("width50")
                    r["width90"] = cov.get("width90")
        except Exception as exc:  # noqa: BLE001
            print(f"  coverage skipped for {mk}: {exc}", flush=True)
    if marginal_plots and cov_rows:
        cb = save_coverage_bar_chart(
            coverage_rows=cov_rows,
            labels=MODEL_LABELS,
            order=[k for k in models if k in cov_rows],
            out_path=FIG_DIR / f"{stem}_coverage_bars.png",
            title=f"{problem_name} (n_train={n_train}): 50%/90% interval coverage (test)",
        )
        print(f"Wrote {cb}")
        coverage_bar_path = str(cb)

    # Eval-only true-density match at probes (Hellinger / TV / |k−true_k|)
    if probe_preds:
        from density_match import probe_density_match

        for mk, pred_p in probe_preds.items():
            ex = (results.get(mk) or {}).get("extras") or {}
            dm = probe_density_match(
                problem_name=problem_name,
                x_probes=x_probes,
                pred=pred_p,
                y_min=float(problem["y_min"]),
                y_max=float(problem["y_max"]),
                true_k_modes=int(problem["true_k_modes"]),
                k_modes_mean=ex.get("k_modes_mean"),
            )
            if dm is None:
                continue
            results[mk]["density_match"] = dm
            for r in rows:
                if r.get("model_key") == mk:
                    for c in DENSITY_COLS:
                        if c in dm:
                            r[c] = dm[c]
            print(
                f"  density[{mk}]: Hellinger={dm['hellinger_mean']:.4f} "
                f"TV={dm['tv_mean']:.4f}"
                + (
                    f" |k_err|={dm['abs_k_modes_err']:.3f}"
                    if "abs_k_modes_err" in dm
                    else ""
                ),
                flush=True,
            )

    if marginal_plots and probe_preds:
        x_all = np.vstack([x_tr_np, x_te_np])
        y_all = np.concatenate([y_tr_np, y_test])
        m_paths = save_marginals_per_probe(
            problem_name=problem_name,
            x_probes=x_probes,
            model_preds=probe_preds,
            model_labels=MODEL_LABELS,
            model_order=models,
            x_data=x_all,
            y_data=y_all,
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
            out_dir=FIG_DIR,
            stem=stem,
        )
        for p in m_paths:
            print(f"Wrote {p}")
            marginal_paths.append(str(p))

    if marginal_plots and grid_preds:
        posterior_png = FIG_DIR / f"{stem}_posterior_vs_data.png"
        save_posterior_vs_data(
            problem_name=problem_name,
            x_train=x_tr_np,
            y_train=y_tr_np,
            x_test=x_te_np,
            y_test=y_test,
            x_grid=x_grid,
            model_preds_grid=grid_preds,
            model_labels=MODEL_LABELS,
            model_order=models,
            out_path=posterior_png,
            y_min=float(problem["y_min"]),
            y_max=float(problem["y_max"]),
        )
        print(f"Wrote {posterior_png}")

    multid_idx: list[int] | None = None
    if marginal_plots and te_preds:
        ref_key = next((k for k in ("tabpfn35", "tabpfn3") if k in te_preds), None)
        try:
            pp = save_posterior_vs_data_multid(
                problem_name=problem_name,
                y_train=y_tr_np,
                y_test=y_test,
                model_preds=te_preds,
                model_labels=MODEL_LABELS,
                model_order=models,
                out_path=FIG_DIR / f"{stem}_bands_pit.png",
            )
            print(f"Wrote {pp}")
            marginal_paths.append(str(pp))
            if input_dim > 1:
                res = save_conditionals_multid(
                    problem_name=problem_name,
                    x_train=x_tr_np,
                    y_train=y_tr_np,
                    x_test=x_te_np,
                    model_preds=te_preds,
                    model_labels=MODEL_LABELS,
                    model_order=models,
                    out_path=FIG_DIR / f"{stem}_conditionals_mq.png",
                    ref_key=ref_key,
                )
                if res is not None:
                    print(f"Wrote {res[0]}")
                    marginal_paths.append(str(res[0]))
                    multid_idx = res[1]
        except Exception as exc:  # noqa: BLE001
            print(f"  multi-d plots skipped: {exc}", flush=True)

    from cost_tables import make_cost_row, write_cost_artifacts

    cost_rows = []
    n_tr = int(problem["x_tr"].shape[0])
    for r in rows:
        if r.get("source") == "error":
            continue
        ex = (results.get(r.get("model_key") or "", {}) or {}).get("extras") or {}
        cost_rows.append(
            make_cost_row(
                script="compare_foundation_baselines",
                problem=problem_name,
                model=str(r.get("Model") or r.get("model_key")),
                wall_s=r.get("wall_s", ex.get("wall_s")),
                n_train=n_tr,
                k_max=int(problem["k_max"]) if r.get("model_key") == "midline" else None,
                num_steps=int(problem["num_steps"])
                if r.get("model_key") == "midline"
                else None,
                gate_type=ex.get("gate_type"),
                num_inits=ex.get("num_inits"),
                num_stage2_inits=ex.get("num_stage2_inits"),
                champion=bool(problem["champion"])
                if r.get("model_key") == "midline"
                else None,
                notes=str(r.get("source") or ""),
            )
        )
    cost_art = write_cost_artifacts(
        cost_rows,
        out_dir=OUT_DIR,
        stem=f"{stem}_cost",
        title=f"{problem_name} n_train={n_train} — fit cost",
    )

    payload = {
        "protocol": {
            "problem": problem_name,
            "n_train": n_train,
            "n_test": n_test,
            "test_shared": True,
            "seed": seed,
            "true_k_modes": problem["true_k_modes"],
            "x_probes": None if x_probes is None else x_probes.tolist(),
            "input_dim": int(input_dim),
            "conditional_test_idx": multid_idx,
            "midline": {
                "k_max": problem["k_max"],
                "champion": problem["champion"],
                "num_steps": problem["num_steps"],
                "midline_backend": problem.get("midline_backend", "exact"),
            },
        },
        "rows": rows,
        "results": results,
        "artifacts": {
            "json": str(table_json),
            "csv": str(table_csv),
            "table_png": str(table_png),
            "marginals_png": marginal_paths,
            "posterior_png": str(posterior_png) if posterior_png else None,
            "coverage_bars_png": coverage_bar_path,
            "cost": cost_art,
        },
    }
    _json_dump(table_json, payload)
    save_csv(rows, table_csv)
    save_metrics_table_png(
        rows,
        table_png,
        title=(
            f"{problem_name} foundation compare — n_train={n_train}, "
            f"n_test={n_test} (shared), seed={seed}"
        ),
    )
    table_cols = ["Model"] + COMPARE_COLS + [
        c for c in TABLE_EXTRA_COLS if any(r.get(c) is not None for r in rows)
    ]
    print_metrics_table(rows, cols=table_cols)
    print(f"Wrote {table_json}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_png}")
    return payload


def run_a01_scoringbench_compat(
    *,
    models: list[str] | None = None,
    include_website_baselines: bool = True,
) -> dict[str, Any]:
    """A01 entry used by ``compare_a01_scoringbench`` (live FMs + MidlineSparse)."""
    models = models or list(ALL_MODELS)
    n_train = max(DEFAULT_TRAIN_SIZES)
    pool = load_problem_pool(
        "A01", n_train_max=n_train, n_test=DEFAULT_N_TEST, seed=42
    )
    problem = slice_train(pool, n_train)
    return run_problem(
        "A01",
        models,
        problem=problem,
        n_train=n_train,
        n_test=DEFAULT_N_TEST,
        seed=42,
        include_website_baselines=include_website_baselines,
    )


def _parse_list(
    raw: str, allowed: tuple[str, ...], default: tuple[str, ...] | None = None
) -> list[str]:
    raw = raw.strip()
    if raw.lower() in ("all", "*"):
        return list(allowed)
    if raw.lower() == "default" and default is not None:
        return list(default)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown items {bad}; allowed: {allowed}")
    return items


def _parse_train_sizes(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return list(DEFAULT_TRAIN_SIZES)
    sizes = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not sizes or any(s < 1 for s in sizes):
        raise SystemExit(f"Invalid --train-sizes {raw!r}; need positive ints")
    return sizes


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="all",
        help=f"Comma list or 'all'. Allowed: {', '.join(ALL_PROBLEMS)}",
    )
    parser.add_argument(
        "--models",
        default="default",
        help=(
            "Comma list, 'default' (excludes deprecated hist_lmc*/vexp keys) or 'all'. "
            f"Allowed: {', '.join(ALL_MODELS)}"
        ),
    )
    parser.add_argument(
        "--train-sizes",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SIZES),
        help="Comma list of nested train sizes (default: 100,200,400,800)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=DEFAULT_N_TEST,
        help="Fixed shared test size per problem (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    parser.add_argument(
        "--website-baselines",
        action="store_true",
        help="On A01, also include historical TabICL/Crepes website rows",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=5,
        help="Number of interior x-probes for conditional marginal panels",
    )
    parser.add_argument(
        "--no-marginal-plots",
        action="store_true",
        help="Skip true/data/model p(y|x) panel figures",
    )
    parser.add_argument(
        "--preview-a06",
        action="store_true",
        help="Write A06 noise preview figure and exit (no model fits)",
    )
    parser.add_argument(
        "--hist-n-bins",
        type=int,
        default=200,
        help=(
            "Requested response bins for hist_mogp / hist_gpd (default: 200; "
            "effective B is capped at n_train for hard one-hots)"
        ),
    )
    parser.add_argument(
        "--hist-num-steps",
        type=int,
        default=1000,
        help="Adam steps for hist_mogp Kronecker training (default: 1000)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Artifact directory (default: results/foundation_compare_mdn). "
            "Relative paths are under this experiment folder."
        ),
    )
    args = parser.parse_args(argv)

    gpplus.config.configure_logger()

    global OUT_DIR, FIG_DIR
    if args.out_dir:
        out = Path(args.out_dir)
        if not out.is_absolute():
            out = _THIS / out
        OUT_DIR = out
        FIG_DIR = OUT_DIR / "figures"

    if args.preview_a06:
        preview_a06(seed=args.seed)
        return {}

    problems = _parse_list(args.problems, ALL_PROBLEMS)
    models = _parse_list(args.models, ALL_MODELS, DEFAULT_MODELS)
    train_sizes = _parse_train_sizes(args.train_sizes)
    n_test = int(args.n_test)

    # COSI defaults: 5k/5k, dedicated out dir, no 1D marginal panels
    if len(problems) == 1 and problems[0] == "COSI":
        if args.train_sizes == ",".join(str(s) for s in DEFAULT_TRAIN_SIZES):
            train_sizes = [COSI_DEFAULT_TRAIN]
        if args.n_test == DEFAULT_N_TEST:
            n_test = COSI_DEFAULT_TEST
        if not args.out_dir:
            OUT_DIR = _THIS / "results" / "foundation_compare_cosi"
            FIG_DIR = OUT_DIR / "figures"
        args.no_marginal_plots = True

    # TabArena concrete: 824/206 Exact midlines, dedicated out dir
    if len(problems) == 1 and problems[0] == "TABARENA_CONCRETE":
        if not args.out_dir:
            OUT_DIR = _THIS / "results" / "foundation_compare_tabarena"
            FIG_DIR = OUT_DIR / "figures"

    # QSAR fish toxicity: 726/182, dedicated out dir unless --out-dir set
    if len(problems) == 1 and problems[0] == "QSAR_FISH":
        if not args.out_dir:
            OUT_DIR = _THIS / "results" / "foundation_compare_qsar_fish"
            FIG_DIR = OUT_DIR / "figures"

    default_train_sizes_cli = args.train_sizes == ",".join(
        str(s) for s in DEFAULT_TRAIN_SIZES
    )
    default_n_test_cli = args.n_test == DEFAULT_N_TEST

    all_payloads: dict[str, Any] = {}
    for pname in problems:
        # Per-problem fixed splits when CLI left train/test at synthetic defaults.
        local_train_sizes = list(train_sizes)
        local_n_test = int(n_test)
        if pname == "TABARENA_CONCRETE" and default_train_sizes_cli:
            local_train_sizes = [TABARENA_CONCRETE_DEFAULT_TRAIN]
        if pname == "TABARENA_CONCRETE" and default_n_test_cli:
            local_n_test = TABARENA_CONCRETE_DEFAULT_TEST
        if pname == "QSAR_FISH" and default_train_sizes_cli:
            local_train_sizes = [QSAR_FISH_DEFAULT_TRAIN]
        if pname == "QSAR_FISH" and default_n_test_cli:
            local_n_test = QSAR_FISH_DEFAULT_TEST
        n_train_max = max(local_train_sizes)
        pool = load_problem_pool(
            pname,
            n_train_max=n_train_max,
            n_test=local_n_test,
            seed=args.seed,
        )
        # Freeze identity of test tensors for nested train sizes.
        x_te_id = id(pool["x_te"])
        y_te_id = id(pool["y_te"])
        x_te_np = pool["x_te"].detach().cpu().numpy().copy()
        y_te_np = pool["y_te"].detach().cpu().numpy().copy()

        for n_tr in local_train_sizes:
            problem = slice_train(pool, n_tr)
            if id(problem["x_te"]) != x_te_id or id(problem["y_te"]) != y_te_id:
                raise RuntimeError(f"{pname}: test tensors rebound for n_train={n_tr}")
            if not np.array_equal(
                problem["x_te"].detach().cpu().numpy(), x_te_np
            ) or not np.array_equal(
                problem["y_te"].detach().cpu().numpy(), y_te_np
            ):
                raise RuntimeError(f"{pname}: test arrays changed for n_train={n_tr}")
            # Nested prefix check vs larger sizes is implicit via tr_pool_idx[:n].
            key = f"{pname}_n{n_tr}"
            all_payloads[key] = run_problem(
                pname,
                models,
                problem=problem,
                n_train=n_tr,
                n_test=local_n_test,
                seed=args.seed,
                device=args.device,
                include_website_baselines=args.website_baselines,
                n_probes=args.n_probes,
                marginal_plots=not args.no_marginal_plots,
                hist_n_bins=int(args.hist_n_bins),
                hist_num_steps=int(args.hist_num_steps),
            )

    summary_path = OUT_DIR / "summary.json"
    _json_dump(
        summary_path,
        {
            "problems": problems,
            "models": models,
            "protocol": {
                "train_sizes": sorted(
                    {int(k.rsplit("_n", 1)[-1]) for k in all_payloads}
                ),
                "n_test_by_problem": {
                    pname: (
                        TABARENA_CONCRETE_DEFAULT_TEST
                        if pname == "TABARENA_CONCRETE" and default_n_test_cli
                        else QSAR_FISH_DEFAULT_TEST
                        if pname == "QSAR_FISH" and default_n_test_cli
                        else int(n_test)
                    )
                    for pname in problems
                },
                "test_shared": True,
                "nested_train": True,
                "seed": args.seed,
            },
            "payloads": {
                k: {"rows": v["rows"], "protocol": v["protocol"]}
                for k, v in all_payloads.items()
            },
        },
    )
    print(f"Wrote {summary_path}")
    return all_payloads


if __name__ == "__main__":
    main()
