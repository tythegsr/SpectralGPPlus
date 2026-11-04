"""Quick ablation of logistic_gp v2 design choices (n=100, n_test=200).

Usage: python _ablate_logistic_v2.py A01,A05 0,1,42 v1,map_loo,vi_loo
"""

from __future__ import annotations

import sys

import numpy as np

from compare_foundation_baselines import load_problem
from histogram_gp import bar_log_prob
from logistic_gp_density import fit_predict_logistic_gp, fit_predict_logistic_gp_sorf

S = fit_predict_logistic_gp_sorf
VARIANTS = {
    "v1": (fit_predict_logistic_gp, {}),
    "map_loo": (S, dict(inference="map")),
    "vi_loo": (S, dict(inference="vi")),
    "map_none": (S, dict(inference="map", midline="none")),
    "vi_none": (S, dict(inference="vi", midline="none")),
    "map_loo_rff": (S, dict(inference="map", features="rff")),
    "map_none_noamp": (S, dict(inference="map", midline="none", learn_amp=False)),
    "map_loo_noamp": (S, dict(inference="map", learn_amp=False)),
    "vi_loo_noamp": (S, dict(inference="vi", learn_amp=False)),
    "vi_loo_mean": (S, dict(inference="vi", vi_predict="mean")),
}
problems = sys.argv[1].split(",")
seeds = [int(s) for s in sys.argv[2].split(",")]
names = sys.argv[3].split(",")
for p in problems:
    res = {k: [] for k in names}
    for sd in seeds:
        prob = load_problem(p, n_train=100, n_test=200, seed=sd)
        y = prob["y_te"].numpy().ravel()
        for name in names:
            fn, kw = VARIANTS[name]
            pred = fn(prob["x_tr"], prob["y_tr"], prob["x_te"], seed=sd, **kw)
            res[name].append(-np.mean(bar_log_prob(y, pred["pred_borders"], pred["pred_logits"])))
    print(p, " ".join(f"{k}={np.mean(v):.3f}{[round(float(a), 2) for a in v]}" for k, v in res.items()), flush=True)
