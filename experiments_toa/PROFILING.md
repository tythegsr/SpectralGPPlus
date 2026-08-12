# Training profilers (snakeviz + kernprof)

Two-stage workflow for diagnosing slow S2 TOA training (especially SORF/RFF Woodbury + NIGP).

```text
short config → cProfile (.prof) → snakeviz → identify hot functions
                                         → kernprof -l → line-level detail → speedups
```

## Install

```bash
pip install line_profiler snakeviz
```

Outputs land in `profiles/` (gitignored).

## Short-run knobs (required)

Full 1000-epoch / multi-QoI runs are too long to profile. In the S2 IDE config block, use:

| Knob | Profile value | Why |
|------|---------------|-----|
| `QOI` | `["algae"]` | one task |
| `NUM_EPOCHS` | `30`–`50` | enough Adam steps |
| `NUM_INITS` / `N_JOBS` | `1` | joblib breaks clean profiles |
| `MONITOR_VALIDATION` | `False` | remove eval noise from the profile |
| `PLOT` / `PLOT_POSTERIOR` | `False` | skip I/O |
| `N_TRAIN` | keep real size, or `4000` for a faster smoke | same compute shape as production |

Or use the Aug08-matched helper (freeze vs NIGP, CUDA section timers)::

```bash
# needs CUDA torch env (e.g. gpplus_tydev), not the CPU-only base python
python -m experiments_toa.profile_sorf_10epoch --qoi algae --epochs 10 --phases both
```

Restore full settings after profiling.


## Stage 1 — cProfile → snakeviz

From the repo root:

```bash
python -m experiments_toa.profile_cprofile --script experiments_SORF/S2_toa_SORF.py
snakeviz profiles/S2_toa_SORF.prof
```

Or zero-helper:

```bash
python -m cProfile -o profiles/S2_toa_SORF.prof experiments_SORF/S2_toa_SORF.py
snakeviz profiles/S2_toa_SORF.prof
```

VS Code / Cursor: launch **Profile: SORF (cProfile)** or **Profile: PCA-GPR (cProfile)**, then run `snakeviz` on the written `.prof`.

The helper synchronizes CUDA before/after profiling when a GPU is available and prints a top-40 `pstats` summary.

## Stage 2 — kernprof / line_profiler

Hot paths already carry a no-op-safe `@profile` (active only under `kernprof -l`):

- `woodbury_marginal_log_likelihood` — `gpplus/utils/rff_utils.py`
- `RFFWoodburyMarginalLogLikelihood.forward` — `gpplus/training/rff_mll.py`
- `NIGPWoodburyMarginalLogLikelihood.forward` — `gpplus/training/nigp_mll.py`
- `_negative_mll_loss`, `_train_standard_epoch` — `gpplus/training/training_single_run.py`
- `RFFGPR.featurize`, `RFFGPR.scaled_features` — `gpplus/models/rff_gpr.py`

```bash
kernprof -l -v -o profiles/sorf.lprof experiments_SORF/S2_toa_SORF.py
```

Re-view later:

```bash
python -m line_profiler profiles/sorf.lprof
```

## Hotspot → options

| If hotspot is… | Options |
|----------------|---------|
| Woodbury factor / Cholesky (`m×m`, `m≈2000`) | Lower `NUM_RFF`; try `woodbury_form="dual"` if noise is tiny; prefer float32 if accuracy allows |
| Feature map / SORF (`n×m`, `n≈16000`) | Cache features when lengthscales are frozen; reduce `N_TRAIN`; fewer bands via `TASK_BAND_CONFIG` |
| Autograd / Adam step | Dual Woodbury uses custom analytic autograd by default (see below); fewer epochs; stronger early stopping; longer NIGP freeze or NIGP-off baseline |
| Validation every N epochs | Raise eval interval or set `MONITOR_VALIDATION=False` during train |
| Exact PCA-GPR Cholesky (`partition²`) | Smaller `PARTITION_SIZE`; fewer partitions; lower LBFGS `max_iter`; float32 |
| Multi-init / joblib | Profile with `NUM_INITS=1`, `N_JOBS=1`; more GPUs only help wall clock on real multi-init runs |

Measure first; change hyperparameters only after a profile points at a clear bottleneck.

## Dual Woodbury custom autograd

Dual MLL (`woodbury_marginal_log_likelihood_dual`) uses a fused `torch.autograd.Function`
with closed-form grads (`α`, `v`, `ΦΛ⁻¹`) so training does **not** differentiate through
Cholesky. Primal Woodbury is unchanged. NIGP diag-noise uses ``WoodburyDualRFFDiagMLL``
(fused featurize VJP after ``D^{-1/2}`` row-scaling) when dual + fuse are enabled.

For dual + `RFFGPR`, the MLL also **fuses the RFF featurize VJP** (no cos/sin autograd
graph through `scaled_features`) unless disabled. The fused path streams
`∇_Φ ℓ = α vᵀ − ΦΛ⁻¹` into lengthscale/outputscale **without** allocating a full
`(n, m)` `∇_Φ` buffer (only `W = ΦΛ⁻¹` plus cos/sin halves of width `D`).

**Remaining train-time floor (float64, exact dual MLL):** every Adam step still pays
Gram `ΦᵀΦ` `O(n m²)` + Cholesky `O(m³)` + `ΦΛ⁻¹` `O(n m²)`. Further wins need milder
numeric trades (`m`, dtype) — not more Cholesky-VJP removal. Float32 factors stay
opt-in only.

- Implementation: `gpplus/utils/woodbury_mll_autograd.py`
- Escape hatches / debug:
  - `GPPLUS_WOODBURY_AUTOGRAD=reference` — stock Cholesky autodiff
  - `GPPLUS_WOODBURY_FUSE_FEATURIZE=0` — disable fused featurize VJP
  - `GPPLUS_WOODBURY_LINALG=float32` — optional float32 factors (**off by default**;
    can hurt tiny-noise / NIGP accuracy — use only with metric A/B)
  - `GPPLUS_WOODBURY_CUDA_TIMING=1` — print CUDA event averages for
    featurize / gram / cholesky / fwd_solves / phi_lam_inv / tr_lam_inv / rff_vjp
- Optional early freeze of likelihood noise: `freeze_epoch_noise` on
  `run_s2_toa_stgp` / `FREEZE_EPOCH_NOISE` in SORF S2 scripts (default `0`).
  While frozen, dual backward skips `tr(Λ⁻¹)` (`O(m³)`). Same pattern as
  `freeze_epoch_nigp` / `NIGPInputNoiseFreezeCallback`.
- Paper-style outer-loop slopes: `nigp_slope_refreshes` /
  `NIGP_SLOPE_REFRESHES` (default `None` = recompute `∇μ` every Adam epoch).
  Positive `R` recomputes slopes `R` times after NIGP unlock, spaced over
  `num_epochs - freeze_epoch_nigp`, and reuses the cached `∇μ` in `d` between
  refreshes (canonical NIGP training). Profile with fewer `nigp_grad_mu` calls:

```bash
python -m experiments_toa.profile_sorf_10epoch --phases nigp --slope-refreshes 2 --epochs 10
```

- Micro-bench:

```bash
python -m experiments_RFF.bench_woodbury_dual_autograd
python -m experiments_RFF.bench_woodbury_dual_autograd --n 16000 --m 4000 --dtype float64
python -m experiments_RFF.bench_woodbury_dual_autograd --n 4096 --m 800 --cuda-timing
```

### Re-profile checklist (after custom autograd / fused VJP)

1. Short SORF config: `NUM_EPOCHS=50`, one QoI, `MONITOR_VALIDATION=False`, plots off.
2. `python -m experiments_toa.profile_cprofile --script experiments_SORF/S2_toa_SORF.py`
3. Confirm `run_backward` / Cholesky VJP / featurize share drops; train loss still decreases.
4. Optional A/B: `set GPPLUS_WOODBURY_AUTOGRAD=reference` and re-bench / re-profile.
5. Optional section timing: `set GPPLUS_WOODBURY_CUDA_TIMING=1` (Gram vs chol vs `ΦΛ⁻¹`).
6. Do **not** enable `GPPLUS_WOODBURY_LINALG=float32` for production float64 NIGP without comparing RMSE / curves.
