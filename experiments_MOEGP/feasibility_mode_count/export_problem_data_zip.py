"""Export A01/A04/A05/A06 CSVs (shared-test protocol) into a zip archive."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from compare_foundation_baselines import (
    DEFAULT_N_TEST,
    DEFAULT_TRAIN_SIZES,
    load_problem_pool,
    slice_train,
)

OUT_DIR = Path(__file__).resolve().parent / "results" / "problem_data_export"
ZIP_PATH = Path(__file__).resolve().parent / "results" / "A01_A04_A05_A06_data.zip"
PROBLEMS = ("A01", "A04", "A05", "A06")
SEED = 42
N_TRAIN_MAX = max(DEFAULT_TRAIN_SIZES)
N_TEST = DEFAULT_N_TEST
TRAIN_SIZES = list(DEFAULT_TRAIN_SIZES)

DESCRIPTIONS = {
    "A01": {
        "name": "A01 bimodal",
        "true_k_modes": 2,
        "equation": (
            "Equal mixture of N(2x+5, 2^2) and N(-x+15, 2^2); "
            "x on linspace[0,10]."
        ),
    },
    "A04": {
        "name": "A04 trimodal",
        "true_k_modes": 3,
        "equation": (
            "Equal mixture of N(2x+5, 0.8^2), N(-0.5x+12, 2^2), N(x-2, 4^2); "
            "x on linspace[0,10]."
        ),
    },
    "A05": {
        "name": "A05 unimodal hetero",
        "true_k_modes": 1,
        "equation": (
            "mu=0.5x+2; skewed contaminated heteroscedastic noise "
            "(asymmetric Laplace + rare wide Gaussian); x~U[0,10]."
        ),
    },
    "A06": {
        "name": "A06 cubic+bump lognormal",
        "true_k_modes": 1,
        "equation": (
            "f(x)=0.1x^3-x^2+3x-9 + 18*exp(-(x-5)^2/(2*0.35^2)); "
            "y=f+(kappa*|f|+eps0)*(L-E[L]), L~LogNormal(0,sigma(x)^2); "
            "x~U[0,10]."
        ),
    },
}


def main() -> Path:
    readme = f"""# Synthetic problem data: A01, A04, A05, A06

Protocol (matches `compare_foundation_baselines.py`):
- seed = {SEED}
- n_test = {N_TEST} (shared / frozen within each problem)
- nested train sizes = {TRAIN_SIZES} (prefixes of one train pool of size {N_TRAIN_MAX})
- Pool size N = n_train_max + n_test = {N_TRAIN_MAX + N_TEST}

For each problem folder:
- `test.csv` — frozen test set (`x`, `y`)
- `train_n{{N}}.csv` — nested train subset of size N
- `pool.csv` — full pool with columns `x`, `y`, `role` (test|train_pool), `train_rank`
- `meta.json` — protocol + short equation description

Train nesting: train_n100 ⊂ train_n200 ⊂ train_n400 ⊂ train_n800.
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in PROBLEMS:
        d = OUT_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()

    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "seed": SEED,
        "n_test": N_TEST,
        "n_train_max": N_TRAIN_MAX,
        "train_sizes": TRAIN_SIZES,
        "problems": list(PROBLEMS),
    }

    for name in PROBLEMS:
        pool = load_problem_pool(
            name, n_train_max=N_TRAIN_MAX, n_test=N_TEST, seed=SEED
        )
        X = np.asarray(pool["X"], dtype=np.float64)
        y = np.asarray(pool["y"], dtype=np.float64).ravel()
        te_idx = np.asarray(pool["te_idx"], dtype=np.int64)
        tr_pool_idx = np.asarray(pool["tr_pool_idx"], dtype=np.int64)

        x_te = X[te_idx].reshape(-1)
        y_te = y[te_idx]
        pd.DataFrame({"x": x_te, "y": y_te}).to_csv(
            OUT_DIR / name / "test.csv", index=False
        )

        for n in TRAIN_SIZES:
            problem = slice_train(pool, n)
            x_tr = problem["x_tr"].detach().cpu().numpy().reshape(-1)
            y_tr = problem["y_tr"].detach().cpu().numpy().ravel()
            pd.DataFrame({"x": x_tr, "y": y_tr}).to_csv(
                OUT_DIR / name / f"train_n{n}.csv", index=False
            )

        role = np.array(["unused"] * len(y), dtype=object)
        train_rank = np.full(len(y), -1, dtype=np.int64)
        role[te_idx] = "test"
        for r, idx in enumerate(tr_pool_idx):
            role[idx] = "train_pool"
            train_rank[idx] = r
        pd.DataFrame(
            {
                "x": X.reshape(-1),
                "y": y,
                "role": role,
                "train_rank": train_rank,
                "pool_index": np.arange(len(y)),
            }
        ).to_csv(OUT_DIR / name / "pool.csv", index=False)

        meta = {
            **DESCRIPTIONS[name],
            "problem": name,
            "seed": SEED,
            "n_test": int(N_TEST),
            "n_train_max": int(N_TRAIN_MAX),
            "train_sizes": TRAIN_SIZES,
            "n_pool": int(X.shape[0]),
            "x_dim": int(X.shape[1]),
            "y_min_panel": float(pool["y_min"]),
            "y_max_panel": float(pool["y_max"]),
        }
        (OUT_DIR / name / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(name, "test", len(y_te), "train_max", N_TRAIN_MAX)

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "README.md", arcname="A01_A04_A05_A06_data/README.md")
        zf.write(
            OUT_DIR / "manifest.json", arcname="A01_A04_A05_A06_data/manifest.json"
        )
        for name in PROBLEMS:
            for f in sorted((OUT_DIR / name).iterdir()):
                zf.write(f, arcname=f"A01_A04_A05_A06_data/{name}/{f.name}")

    print("Wrote", ZIP_PATH)
    print("size_bytes", ZIP_PATH.stat().st_size)
    return ZIP_PATH


if __name__ == "__main__":
    main()
