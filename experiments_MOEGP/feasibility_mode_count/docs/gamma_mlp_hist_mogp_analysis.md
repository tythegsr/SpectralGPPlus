# Gamma-MLP and hist-MoGP: analysis, fixes, and roadmap

This note covers two models in the foundation-model comparison (`compare_foundation_baselines.py`):

- **Gamma MLP** (`midline_gamma_mlp`): the two-stage `MidlineSparseMoEGPR` champion. Stage 1 is an exact GP midline. Stage 2 is an MLP gate and offsets with Hard-Concrete expert pruning. Component variances are $\sigma_0^2(x)(1-\alpha_k\gamma_k(x))\beta_k$.
- **hist-MoGP** (`hist_mogp`): a Kronecker $K_{XX}\otimes K_{HH}$ multi-output GP on one-hot bin indicators. Its posterior mean is projected onto the simplex to give a bar (histogram) density, matching the TabPFN output interface.

Both are compared against TabPFN-3.5 (`tabpfn35`). The protocol is:

- **Synthetic suite:** A01, unimodal, A02, A04, A05 and A06 at n=100, with 200 test points.
- **Tabular:** concrete (824/206) and fish (726/182).
- **Seeds:** 3 (0, 1, 42).
- **Metrics:** RMSE, CRPS, mean negative log score, empirical 90% interval coverage (cov90), and the mean Hellinger distance to the true conditional density at the probe points (synthetics only).

Raw results live in `results/foundation_compare_tier1/` (Tier 1/2), `results/foundation_compare_tier3/` (Tier 3, Section 9), and `results/foundation_compare_tier4/` (Tier 4, Section 10). To regenerate the tables:

```bash
python aggregate_tier1.py results/foundation_compare_tier1 --out results/foundation_compare_tier1/summary_table.md
```

## 1. Evidence before the fixes

Seed 42, from the ablation run `results/foundation_compare_tier1/ablation_seed42/`. The values are the negative log score; lower is better.

| problem | TabPFN-3.5 | Gamma MLP | hist-MoGP |
|---|---|---|---|
| A01 | 2.66 | 2.77 | **17.93** |
| unimodal | 0.29 | 0.37 | 0.92 |
| A02 | −1.50 | −1.19 | −1.16 |
| A04 | 3.04 | 3.19 | **8.97** |
| A05 | 1.58 | 1.92 | **23.99** |
| A06 | 2.53 | 3.73 | **7.37** |
| fish | 1.05 | 1.55 | **4.60** |
| concrete | 1.88 | 3.92 (RMSE **16.7** vs 3.9) | 5.55 |

What the table shows:

- Gamma is competitive on CRPS and Hellinger for the synthetics. It is under-dispersed: cov90 is 0.80 on A01 and 0.82 on fish.
- On concrete, gamma fails outright. The Stage-1 log shows `resid_std=16.9`, which is about the raw spread of y, and every expert dies (`survivors=0/2 safety=True`).
- hist-MoGP has competitive CRPS but catastrophic log scores.

## 2. Root causes (confirmed by ablation)

### Gamma MLP

1. **No standardization before Stage 1 (confirmed; dominant on tabular data).** Concrete features span roughly 0–1000, while the GP kernel initialization assumes O(1) inputs. Stage 1 learns a flat midline, Stage 2 collapses, and the safety fallback returns that same flat GP. Standardizing x and y inside the adapter (`fit_predict_midline_gamma_t1`, `standardize=True`) brings concrete RMSE from **16.7 to 5.6**. This fix alone is responsible for the tabular recovery.
2. **Variance can only shrink, and the in-sample fit over-shrinks it (confirmed).** Component variance is bounded above by $\sigma_0^2$, and Stage 2 is trained by in-sample NLL, so components end up too tight on test data. This shows up as cov90 of 0.80–0.86 on A01, A04, fish and concrete.
3. **The σ0 cap is not the source of overconfidence (hypothesis refuted).** The plan predicted that raising `std0_cap_frac` from 0.3 to 1.0 would widen intervals. It did the opposite: with standardization held fixed, cov90 got worse on **all 8** problems (for example fish 0.80 → 0.64 and concrete 0.84 → 0.63), and the log score blew up on A05 (8.96), fish (6.30) and concrete (5.64). With a looser cap, Stage 2 has more variance budget to give away through γ, and the training NLL rewards giving it away. The cap acts as a regularizer. A cap of 0.5 used alone is no better than 0.3 and fails on A02 (log score 20.4, from one bad Stage-2 solution) and A04 (cov90 0.74), so it is only safe combined with temperature scaling.
4. **The safety fallback never fires after standardization.** `_safe` matches `_std` exactly on every problem. The collapse on concrete came from bad scaling, not from an expert-death mode that needs a softer fallback. The soft safety mixture is harmless and kept as insurance.
5. **Stage-2 dispersion is miscalibrated in both directions, and post-hoc temperature scaling is the right instrument.** A one-parameter temperature, fit by validation log-likelihood on a 15% holdout, selects T between 0.57 and 2.46 depending on problem and seed:
   - T > 1 (widen) on A04, A05, fish, concrete and unimodal.
   - T < 1 (sharpen) on most seeds of A01 and A02, where the base model is already well covered.

    On its own, temperature gives the largest UQ gains: fish log score goes from 1.72 to 1.32 with cov90 from 0.80 to 0.95, and concrete goes from 3.50 to 3.19.

### hist-MoGP

1. **Exact-zero bins, plus targets outside the borders, cause the blow-ups (confirmed).** Euclidean simplex projection leaves bins with exactly zero mass. Any y outside the training borders, or inside a zero bin, gets $\log p\approx -690$ (the 1e-300 floor). The ablation separates the two mechanisms. The numbers below are seed 42, after the eigh fix, from `rerun_hist_seed42/ablation_table.md`:
   | variant | A01 | A05 | fish | concrete |
    |---|---|---|---|---|
    | base | 17.7 | 24.4 | — | — |
    | ε-floor alone (`hist_mogp_floor`) | 16.7 | 22.4 | — | — |
    | tail bins alone (`hist_mogp_tail`) | 7.7 | 4.4 | — | — |
    | B-selection alone (`hist_mogp_bsel`) | 16.6 | 23.1 | — | — |
    | all three (`hist_mogp_t1`) | 2.93 | 3.06 | 1.33 | 3.47 |
   - The **ε-floor alone** does not help. With B=200 and no tails, the failing test points sit *outside* the bar support, where no floor can reach.
   - **Tail bins alone** fix the out-of-support points but not the interior zero bins.
   - **B-selection alone** helps on A02, unimodal and fish but not on A01 or A05.
   - **All three together** remove every blow-up. The mechanisms are complementary: tails provide support, the floor and a coarser B remove zero mass inside the support.
2. **B=200 is far too fine at n=100.** Validation selection picks B=10 on almost every problem and seed, and B=20 on A02. At small n the bin count, not the GP, is the main smoothing control.
3. **The ε-floor is used sparingly once tails exist.** The selected ε is 0 on most synthetics and 0.03–0.2 on A04, fish and unimodal. With the right B and tails, the floor is a guard, not the main correction.
4. **Training instability from degenerate eigenvalues (found and fixed during this work).** Tail bins are far from every interior bin center, so their rows of $K_{HH}$ are close to identity rows, and $K_{HH}$ gets *repeated eigenvalues*. The stock `torch.linalg.eigh` backward divides by $\lambda_j-\lambda_i$, so every gradient became non-finite and the fit raised "no finite MLL". This hit seed 0 on A05, A06 and unimodal. The fix is `_SafeEigh`, whose backward zeroes near-degenerate gap terms. That is exact for losses invariant to rotations within an eigenspace, such as log-det and quadratic forms, and it matches the stock gradient to 2e-16 on non-degenerate matrices. All hist results were rerun with this fix (`rerun_hist_seed`*).
5. **Separable kernel and misspecified likelihood (not fixed by Tier 1).** One bin-correlation structure is shared across all x, and a Gaussian likelihood on one-hot targets is misspecified. These limit how sharp the density can be. After Tier 1, Hellinger is 0.21–0.35, against 0.11–0.23 for TabPFN. CRPS on concrete is 3.76 against 2.81 for gamma t1 and 1.57 for TabPFN.

## 3. What was implemented

### Tier 1: preprocessing and calibration

| change                                                                                                                           | where                                                      | key                    |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------- |
| x/y standardization around the whole gamma pipeline; outputs unscaled                                                            | `fit_predict_midline_gamma_t1`                             | `_std`                 |
| exposed `std0_cap_frac`, swept 0.3 / 0.5 / 1.0                                                                                   | `fit_predict_midline_sparse(std0_cap_frac=…)`              | `_cap05`, `_cap10`     |
| held-out temperature on expert stds (grid over 0.5–4 in log space, chosen by validation log-likelihood, then refit on full data) | `fit_predict_midline_gamma_t1(calibrate_temperature=True)` | `_temp`                |
| soft safety mixture: broad component (weight 0.2, std 1.5× the train residual) added only if the hard safety refit fired         | `soft_safety=True`                                         | `_safe`                |
| all of the above plus Stage-1 `num_inits=4`                                                                                      |                                                            | `midline_gamma_mlp_t1` |
| outer tail bins (width = span per side)                                                                                          | `histogram_gp.make_hist_bins(tail_frac=…)`                 | `hist_mogp_tail`       |
| ε-floor mixing with a smoothed marginal: $(1-\varepsilon)p + \varepsilonm$                                                       | `histogram_gp.floor_mix`, `smoothed_marginal`              | `hist_mogp_floor`      |
| validation selection of B over {10, 20, 40, 80} and ε over {0, 1e-3, 1e-2, 0.03, 0.1, 0.2}                                       | `fit_predict_hist_t1`                                      | `hist_mogp_bsel`       |
| tails + floor + B-selection                                                                                                      |                                                            | `hist_mogp_t1`         |
| degeneracy-safe `eigh` backward                                                                                                  | `foundation_adapters._SafeEigh`                            | (all hist_mogp)        |

### Tier 2: likelihood and variance structure

| change | where | key |
|---|---|---|
| variance expansion: $\beta_k(x)=\exp(s_k(x))$ with $s_k\in[0,2]$ from an MLP (bias initialized so $s\approx 0$) | `MidlineSparseMoEGPR(var_expand=True)`, `get_var_scale` | `midline_gamma_mlp_vexp` |
| heteroscedastic Stage 1: log-variance MLP on Stage-1 residuals, normalized to mean 1 and clipped to [0.2, 5], multiplying $\sigma_0(x)$ | `MidlineSparseMoEGPR(het_stage1=True)` | `midline_gamma_mlp_het` |
| both | | `midline_gamma_mlp_t2` |
| hist: average projected Monte Carlo samples using the posterior variance (instead of projecting the mean) | `sample_project=True` | `hist_mogp_t1s` |
| Dirichlet-GP histogram (Milios et al.) with the Tier-1 wrapper and standardized x; fixed class count for empty bins | `fit_predict_hist_gpd` | `hist_gpd_t1` |
| the same, plus a softmax average over 128 latent samples | `n_mc=128` | `hist_gpd_t2` |

## 4. Results (3 seeds, mean ± sd)

The full table, with every variant plus RMSE and Hellinger, is in `results/foundation_compare_tier1/summary_table.md`. All hist_mogp rows come from the `rerun_hist_seed*` runs with the `_SafeEigh` fix.

**Negative log score (lower is better)**

| problem | TabPFN-3.5 | gamma | gamma t1 | gamma het | hist | hist t1 | hist_gpd t1 |
|---|---|---|---|---|---|---|---|
| A01 | **2.70±0.04** | 2.78±0.08 | 2.84±0.16 | 3.00±0.11 | 8.33±8.16 | 2.97±0.14 | 3.05±0.09 |
| unimodal | **0.25±0.04** | 0.28±0.12 | 0.30±0.05 | 0.43±0.13 | 3.10±3.96 | 0.57±0.13 | 0.81±0.10 |
| A02 | **−1.54±0.03** | −1.06±0.13 | −1.19±0.18 | −1.25±0.09 | −1.10±0.05 | −1.23±0.19 | −0.87±0.19 |
| A04 | **3.07±0.05** | 3.26±0.10 | 3.21±0.01 | 3.42±0.12 | 13.52±2.20 | 3.35±0.14 | 3.37±0.06 |
| A05 | **1.66±0.07** | 2.03±0.10 | 1.95±0.13 | 1.95±0.09 | 16.43±10.87 | 2.30±0.66 | 1.91±0.07 |
| A06 | **2.60±0.07** | 3.37±0.42 | 3.48±0.25 | 2.93±0.06 | 6.19±2.10 | 2.99±0.06 | 3.13±0.15 |
| fish | **0.98±0.07** | 1.55±0.05 | 1.32±0.03 | 1.37±0.07 | 2.21±0.37 | 1.30±0.03 | 1.45±0.07 |
| concrete | **1.85±0.03** | 5.22±2.49 | 3.09±0.09 | 3.21±0.05 | 3.76±0.19 | 3.42±0.05 | 3.60±0.07 |

**CRPS (lower is better)**

| problem | TabPFN-3.5 | gamma | gamma t1 | gamma het | hist | hist t1 | hist_gpd t1 |
|---|---|---|---|---|---|---|---|
| A01 | 2.91 | **2.85** | 2.86 | 2.95 | 2.96 | 2.90 | 3.33 |
| unimodal | 0.172 | **0.167** | 0.173 | 0.176 | 0.319 | 0.215 | 0.320 |
| A02 | **0.030** | 0.044 | 0.042 | 0.043 | 0.161 | 0.102 | 0.134 |
| A04 | 3.76 | **3.67** | 3.79 | 3.86 | 3.88 | 3.89 | 4.03 |
| A05 | **0.914** | 0.919 | 0.985 | 0.998 | 0.992 | 1.009 | 1.161 |
| A06 | **2.40** | 2.70 | 3.14 | 2.64 | 3.39 | 3.01 | 4.73 |
| fish | **0.400** | 0.458 | 0.474 | 0.498 | 0.502 | 0.501 | 0.668 |
| concrete | **1.57** | 7.00 | 2.81 | 3.05 | 4.43 | 3.76 | 5.34 |

**cov90 (nominal 0.90)**

| problem | TabPFN-3.5 | gamma | gamma t1 | gamma het | hist | hist t1 | hist_gpd t1 |
|---|---|---|---|---|---|---|---|
| A01 | 0.93 | 0.87 | 0.90 | 0.95 | 0.97 | 0.97 | 0.98 |
| unimodal | 0.93 | 0.92 | 0.96 | 0.94 | 1.00 | 0.99 | 1.00 |
| A02 | 0.93 | 0.91 | 0.90 | 0.91 | 1.00 | 0.99 | 1.00 |
| A04 | 0.89 | 0.86 | 0.90 | 0.92 | 0.87 | 0.90 | 0.98 |
| A05 | 0.87 | 0.89 | 0.94 | 0.94 | 0.94 | 0.93 | 0.97 |
| A06 | 0.93 | 0.90 | 0.95 | 0.97 | 0.99 | 0.98 | 0.99 |
| fish | 0.90 | 0.84 | 0.94 | 0.95 | 0.97 | 0.96 | 1.00 |
| concrete | 0.87 | 0.75 | 0.92 | 0.95 | 0.99 | 0.98 | 1.00 |

### Reading the results

- **Tier-1 gamma** fixes concrete. RMSE goes from 12.5±4.6 to 5.42±0.39, and log score from 5.22 to 3.09, with much lower variance across seeds.
  - It improves log score on A02, A04, A05, fish and concrete.
  - It costs about 0.02–0.1 nats on A01, unimodal and A06.
  - Coverage moves to nominal: fish goes from 0.84 to 0.94, concrete from 0.75 to 0.92.
  - **The trade-off is CRPS.** Temperature is fit on log-likelihood and tends to widen (T > 1 on most problems), which costs a little CRPS on the synthetics (A06: 2.70 → 3.14, driven by one seed with T = 2.46).
- **Gap to TabPFN after Tier 1** (gamma t1, in nats):
  | problem | gap |
  |---|---|
  | unimodal | 0.05 |
  | A04 | 0.14 |
  | A01 | 0.15 |
  | A05 | 0.29 |
  | fish | 0.34 |
  | A02 | 0.36 |
  | A06 | 0.88 (0.33 with `het`) |
  | concrete | 1.23 |

  The large gaps that remain are structural: concrete and fish (midline quality), A02 (x-separated regimes) and A06 (variance shape).
- **Tier-1 hist-MoGP** turns the worst model into a competitive one on log score. It is within 0.35 nats of gamma t1 everywhere and *better* than gamma t1 on A02, A06 and fish. It is still worse on CRPS for tabular data and consistently over-covers (cov90 0.96–0.99).
- **Ablation conclusion for gamma:** temperature scaling alone (`_temp`) is about as good as the full t1 stack. The seed-42 log score summed over 8 problems is 14.94 for `_temp` and 14.97 for `_t1`. The cap of 0.5 and the extra Stage-1 restarts add nothing measurable, and the soft safety never triggers. A leaner default is standardize plus temperature with cap 0.3.

## 5. Use cases

**Gamma MLP (Tier-1 configuration,** `midline_gamma_mlp_t1`**), best for:**

- Low-dimensional problems with small to moderate n (about 50–1000) and co-located multimodality, where a smooth midline exists and offsets explain the modes (A01, A04, A05 style). Here it matches or beats TabPFN on CRPS and is close on log score.
- Cases where you need interpretable structure: expert count, offsets $\delta_k(x)$, and gate weights. TabPFN cannot provide this.
- Smooth unimodal regression, where it ties TabPFN.
- After standardization, moderate-n tabular regression as a *calibrated* GP baseline. On concrete and fish, cov90 is 0.92–0.94 and RMSE is within 1.5× of TabPFN.

**Gamma MLP, weak at:**

- **x-separated regimes (A02).** A single midline with offsets is the wrong structure. TabPFN is about 0.4 nats better.
- **Strongly heteroscedastic noise (A06).** Without `het_stage1` the variance profile is set by Stage 1. A one-parameter temperature cannot fix a variance *shape*: t1 is 0.88 nats behind TabPFN, `het` is 0.33 behind.
- **Point accuracy on real tabular data.** The ceiling is the exact ARD-RBF GP: concrete RMSE 5.4 against TabPFN's 3.7. Calibration cannot fix this; the midline itself is the bottleneck.
- **Cost.** It takes 35–110 s per fit at n=100 (8 Stage-1 restarts plus 2 Stage-2 inits plus the temperature refit), against about 1–2 s for TabPFN.

**hist-MoGP (**`hist_mogp_t1`**), best for:**

- A classical, auditable baseline that uses *exactly* the same bar-density interface as TabPFN, so the comparison isolates the prior or model and not the output parameterization.
- Bounded or skewed responses at moderate n with B much smaller than n, where a flexible nonparametric shape matters more than sharpness.
- A conservative, robust log score. After Tier 1 there are no catastrophic failures, cov90 is 0.90–0.99, and it beats Tier-1 gamma on log score on A02 (x-separated), A06 (heteroscedastic) and fish. It handles variance shape and regime switches without having a midline that could be wrong.

**hist-MoGP, weak at:**

- **Sharpness.** It is systematically over-dispersed: cov90 is 0.96–0.99 on 6 of 8 problems, and Hellinger is the worst among the Tier-1 models. A separable kernel plus a mean-only projection smooths away local structure.
- **Point accuracy and CRPS on tabular data.** Concrete CRPS is 3.76 against 2.81 for gamma t1.
- **Small n.** At n=100, B=10 wins selection on most problems, so the density resolution is coarse (B ≈ 10 bins over the data span).
- **Cost.** With validation selection the cost is about 4× a single fit (10 min on concrete). With `sample_project` it is 25 min.

## 6. Tier 2 findings

- **Variance expansion (**`_vexp`**) is unstable and is not recommended as is.**
  - It helps slightly on A02 (−1.29 against −1.19 for t1).
  - It blows up on concrete (13.4±17.7, with one seed at 33.9 and CRPS 16) and on A05 (4.74±4.85).
  - It degrades unimodal and A01 by 0.12–0.2 nats.
  - The mechanism is an extra x-dependent degree of freedom trained by in-sample NLL with no penalty. Stage 2 trades widened "absorber" components against sharpened ones, so the mixture means drift (concrete RMSE 5.6 → 7.7).
  - What it needs before reuse: a penalty $\lambda\overline{s_k(x)}$, or fitting $s$ on held-out residuals.
- **Heteroscedastic Stage 1 (**`_het`**) is a targeted win.** It gives a large improvement on A06 (2.93±0.06 against 3.48±0.25 for Tier 1), which is exactly the heteroscedastic case, and a small one on A02.
  - It costs 0.06–0.21 nats on homoscedastic problems (A01, unimodal, A04, fish, concrete), because a 32×32 MLP on n=100 in-sample residuals overfits noise in the variance profile. Those residuals are also optimistically small, since they are the GP's own training residuals.
  - Use it when heteroscedasticity is expected, or choose it by validation (see Tier 3).
- `_t2` **(both) is the best gamma variant on A06 (2.81) but inherits the vexp instability** (A05: 19.4±30.3), so it is not recommended.
- **Sample-averaged projection (**`hist_mogp_t1s`**) is worse on CRPS and Hellinger on every problem**, with cov90 at or above 0.99, and it costs about 2.5× in time. Averaging projections of high-variance Gaussian samples in indicator space flattens the density toward uniform. The posterior variance in one-hot space is not a useful epistemic signal for this model.
- **The Dirichlet GP (**`hist_gpd_t1` **and** `_t2`**) is worse than** `hist_mogp_t1` **on 6 of 8 problems on log score and on 8 of 8 on CRPS.** The exception is A05 (1.91 against 2.30), where its softmax link avoids hist_mogp's one bad seed. The likelihood correction alone does not close the gap, and Monte Carlo softmax averaging changes at most 0.05 nats. Promoting hist_gpd over hist_mogp is not justified.

## 7. Tier 3: scoped structural upgrades

These are ordered by expected impact per unit of effort, based on the gaps that remain after Tiers 1 and 2.

1. **Gamma: cross-fitted temperature with no refit mismatch.** Small effort, and it fixes cov90 overshoot and the CRPS cost. The current temperature is chosen on an 85% fit and applied to a 100% refit with a different Stage-2 solution. That is why individual seeds overshoot (cov90 0.96–0.99 on unimodal, A05, A06) and A06 CRPS degrades on one seed. Options: average T over K folds, or keep the 85% model plus the selected T (no refit), or ensemble the K fold models. The ensemble also adds epistemic spread.
2. **Gamma: validation-selected variance structure.** Small effort. Use the existing holdout to choose between {homoscedastic, `het_stage1`} and to set a penalty on the expansion net. This captures the A06 win without the homoscedastic cost. The infrastructure (`_mog_log_lik`, the holdout split) already exists in `fit_predict_midline_gamma_t1`.
3. **Gamma: a stronger Stage-1 midline on tabular data.** Medium effort. This targets the largest remaining gap: concrete is 1.23 nats behind TabPFN, with RMSE 5.4 against 3.7. Options:
   - (a) A deep-kernel or feature-learning GP with ARD.
   - (b) A boosted-tree or MLP mean function with the GP on its residuals.
   - (c) An ensemble of Stage-1 restarts.

    Stage 2 only redistributes mass around the midline, so the midline sets the RMSE ceiling.
4. **Gamma: an input-gated mixture of midlines.** Medium to large effort, for x-separated regimes (A02). Use 2–3 Stage-1 GPs with an x-gate, so each regime gets its own smooth midline and the Stage-2 offsets model only within-regime multimodality. It can reuse `MidlineSparseMoEGPR` per midline, with a gating MLP trained on the combined NLL.
5. **hist: a non-separable bin kernel.** Medium effort. Use $\sum_{q=1}^{Q} K^{(q)}_{XX}\otimes K^{(q)}_{HH}$ with Q=2–3 (an LMC-style sum of Kronecker terms), so the shape of the density can change with x. The paired-eigh trick no longer applies directly. This needs a Kronecker-sum solver or an inducing-point SVGP over (x, bin).
6. **hist: a continuous conditional density.** Large effort, and the most promising path to a competitive hist model. Replace bins with a GP-modulated log-density, $p(y\mid x)\propto \exp(f(x,y))$, normalized over a y-grid (logistic GP density), fit by variational inference with a softmax likelihood. This removes B as a hyperparameter, removes zero-mass bins by construction, and allows sharp densities.

What is *not* recommended for Tier 3:

- Further simplex-projection variants (sample averaging made things worse).
- Unregularized variance expansion.
- Raising the σ0 cap.

## 8. Success criteria (from the plan)

| criterion | status |
|---|---|
| Gamma concrete RMSE < 6 (3 seeds) | **met**: 5.42±0.39 (base 12.5±4.6) |
| No hist log-score blow-ups (within 1.0 of gamma) | **met on 3-seed means**: hist t1 minus gamma t1 is at most 0.35 nats, and negative on A02, A06 and fish. One single seed (A05, seed 42) is 1.2 nats behind. |
| Gamma cov90 in [0.85, 0.95] | **met on 7 of 8**: range 0.895–0.950, except unimodal at 0.957 (base: 0.75–0.92) |

## 9. Tier 3 results: self-configuring gamma, logistic-GP density, quantile GP

**Headline.**

- **Logistic-GP density (**`logistic_gp`**):** the clear win of Tier 3. It beats `hist_mogp_t1` on log score on 6 of 8 problems, is the best non-TabPFN model on A02 and A04, is within 0.01 nats of the best on A05, and takes 4–14 s per fit.
- **Self-configuring gamma (**`midline_gamma_auto`**):** one call with no problem knobs and no refits. It matches Tier-1 gamma on 4 problems and beats it on 4 (A02, A05, A06, fish), but misses the plan's per-problem bar on A01 and A06 and does not close the concrete gap.
- **Kumo-style quantile GP (**`quantile_gp`**):** reaches TabPFN-level CRPS on smooth synthetics but has a clearly worse log score. That is the expected signature of a quantile (inverse-CDF) parameterization trained on pinball loss.

Protocol, seeds and problems are as in Sections 1–4. Tier-1/2 references come from `results/foundation_compare_tier1/`. New runs, and a TabPFN rerun that reproduces the Tier-1 log scores exactly and adds `CRPS_bar`, are in `results/foundation_compare_tier3/`.

```bash
python aggregate_tier1.py results/foundation_compare_tier1 results/foundation_compare_tier3 --out results/foundation_compare_tier3/summary_table.md
```

### 9.1 What was built

| piece | file | key(s) |
|---|---|---|
| LOO-trained exact GP midline. The closed-form leave-one-out (LOO) predictive feeds Stage 2 honest residuals; a Bayesian linear mean is a kernel term. | `gpplus/models/loo_gp_core.py` | `stage1="loo"` |
| In-loop calibration: a fixed 15% mask trains only the temperature, the het shrinkage $w$ and (optionally) the expansion net. The networks are detached, so there is no holdout refit. | `midline_sparse_moe.py` (`calib_frac`) | |
| Het variance net fit on standardized LOO residuals, with a learned shrinkage $w=\sigma(\cdot)$ | `midline_sparse_moe.py` (`het_auto`) | |
| Gated multi-midline: EM over LOO-GP midlines with an x-gate. Accepted only if the cross-fitted-gate LOO gain is > 0.1 nats/point **and** the mean max-gate is ≥ 0.8. | `gpplus/models/gated_midline.py` | |
| One-call adapter and ablations | `fit_predict_midline_gamma_auto` | `midline_gamma_auto[_nocal,_nohet,_nomean,_1mid,_rotcal,_vexp]` |
| Logistic-GP density: $p(y\mid x)\propto\exp f(x,y)$ on a 128-cell grid. $f$ is a Gaussian base around a learned $m(x)$ and $s(x)$, plus a marginal $\beta(y)$, plus an RFF GP over $(x, y-m(x))$, plus a learned ε-floor. | `logistic_gp_density.py` | `logistic_gp` |
| Quantile GP: monotone softplus increments; an RFF-x / Kronecker-τ prior; asymmetric-Laplace (pinball) likelihood; GPD tails | `quantile_gp.py` | `quantile_gp`, `_indep`, `_dense` (J=49) |
| Sum-of-Kronecker hist GP: $\sum_{q=1}^2 K_x^{(q)}\otimes K_h^{(q)}$ as exact Bayesian regression on Kronecker RFF features, inside the Tier-1 wrapper | `hist_lmc.py` | `hist_lmc_t1` |
| Exact CRPS of a bar PPD (piecewise-uniform, closed form per bar) | `histogram_gp.bar_crps` | column `CRPS_bar` |

### 9.2 Log score (lower is better, 3-seed means)

| problem | TabPFN | gamma t1 | gamma het | hist t1 | **gamma auto** | **logistic GP** | quantile GP | quantile indep | hist LMC t1 |
|---|---|---|---|---|---|---|---|---|---|
| A01 | **2.70** | 2.84 | 3.00 | 2.97 | 2.96 | 2.90 | 3.07 | 2.88 | 2.96 |
| unimodal | **0.25** | 0.30 | 0.43 | 0.57 | 0.32 | 0.44 | 0.51 | 0.45 | 0.61 |
| A02 | **−1.54** | −1.19 | −1.25 | −1.23 | −1.27 | −1.43 | −0.71 | −0.35 | −1.23 |
| A04 | **3.07** | 3.21 | 3.42 | 3.35 | 3.22 | 3.17 | 3.28 | 3.82 | 3.58 |
| A05 | **1.66** | 1.95 | 1.95 | 2.30 | 1.84 | 1.79 | 1.78 | 1.79 | 1.94 |
| A06 | **2.60** | 3.48 | 2.93 | 2.99 | 3.20 | 3.05 | 3.28 | 3.19 | 3.49 |
| fish | **0.98** | 1.32 | 1.37 | 1.30 | 1.29 | 1.34 | 1.82 | 1.51 | 1.36 |
| concrete | **1.85** | 3.09 | 3.21 | 3.42 | 3.11 | 3.26 | 3.53 | 3.46 | 3.59 |

Standard deviations are in `summary_table.md`. The auto model's A06 entry is 3.20±0.57: two seeds score 2.84 and 2.91, and one seed accepted two midlines and scored 3.85 (Section 9.4).

### 9.3 CRPS (lower is better)

The gamma rows are exact Gaussian-mixture CRPS. For bar models this table uses the exact bar CRPS (`CRPS_bar`). The official `CRPS` column scores bar models through their Gaussian moment summary, which badly misrepresents heavy-tailed bars: on A06 the quantile GP gets 9.9 there against 2.84 exact. `hist_mogp_t1` was not rerun, so its value (marked †) is still the moment summary. For TabPFN the moment summary overstates CRPS by 0.00–0.10.

| problem | TabPFN | gamma t1 | gamma auto | hist t1† | logistic GP | quantile GP | quantile indep | hist LMC t1 |
|---|---|---|---|---|---|---|---|---|
| A01 | 2.81 | 2.86 | 3.01 | 2.90 | 2.87 | 2.95 | **2.78** | 2.86 |
| unimodal | 0.172 | 0.173 | 0.174 | 0.215 | 0.190 | 0.185 | **0.168** | 0.217 |
| A02 | **0.029** | 0.042 | 0.038 | 0.102 | 0.041 | 0.042 | 0.047 | 0.072 |
| A04 | **3.68** | 3.79 | 3.73 | 3.89 | 3.71 | 3.72 | 4.12 | 3.78 |
| A05 | 0.905 | 0.985 | 0.958 | 1.009 | 0.904 | 0.899 | **0.878** | 0.955 |
| A06 | **2.36** | 3.14 | 3.13 | 3.01 | 3.01 | 2.84 | 2.79 | 3.28 |
| fish | **0.399** | 0.474 | 0.459 | 0.501 | 0.499 | 0.539 | 0.482 | 0.515 |
| concrete | **1.55** | 2.81 | 3.04 | 3.76 | 3.59 | 3.57 | 3.53 | 4.50 |

### 9.4 Self-configuring gamma: what each mechanism does

Log score for the full auto model and for one-mechanism-off ablations:

| problem | full | no calibration | no het | no learned mean | single midline | rotating mask | + var. expansion |
|---|---|---|---|---|---|---|---|
| A01 | 2.96 | 2.93 | 2.90 | 2.97 | 2.96 | 2.91 | 2.98 |
| unimodal | 0.32 | 0.41 | 0.30 | 0.33 | 0.32 | 0.51 | 0.35 |
| A02 | −1.27 | −1.09 | −1.34 | −1.26 | −1.02 | −1.04 | −1.26 |
| A04 | 3.22 | 3.30 | 3.31 | 3.25 | 3.22 | 3.88 | 3.24 |
| A05 | 1.84 | 2.29 | 2.03 | 1.97 | 1.84 | 3.04 | 1.83 |
| A06 | 3.20 | 4.60 | 3.49 | 3.15 | 2.90 | 4.68 | 4.20 |
| fish | 1.29 | 2.17 | 1.31 | 1.44 | 1.29 | 2.68 | 1.50 |
| concrete | 3.11 | 3.49 | 3.05 | 3.16 | 3.18 | 3.56 | 3.21 |

- **In-loop calibration is the load-bearing piece.** Without it, log score degrades on 7 of 8 problems: A06 by +1.4, fish by +0.9, A05 by +0.45. The learned temperature ranges from 0.6 to 3.8. Stage 2 is systematically overconfident on held-out points, and a single scalar learned on 15% of the data repairs most of that.
- **The calibration mask must be fixed.** Resampling it every step (`_rotcal`) lets every point leak into the network fit, so the temperature calibrates in-sample error. It is the worst variant on 6 of 8 problems (fish 2.68, A05 3.04). This confirms the deviation from the plan described in Section 9.7.
- **Het with learned shrinkage is a net win, but $w$ does not switch it off cleanly.** It helps on A04, A05, A06 and fish, and costs 0.02–0.07 nats on A01, unimodal, A02 and concrete. The learned $w$ is often 0.8–1.0 even on homoscedastic problems, so the 15% mask is too small to identify "no heteroscedasticity" reliably.
- **The Bayesian linear mean helps on tabular data** (fish 1.44 → 1.29, concrete 3.16 → 3.11, concrete RMSE 7.56 → 6.67). On the synthetics it is within ±0.05 nats except A05, where it helps (1.97 → 1.84).
- **Multi-midline is right for A02 and wrong for one A06 seed.**
  - **A02:** all 3 seeds accept 3 midlines, gaining 0.16–0.32 nats each: −1.27 against −1.02 for a single midline. The gap to TabPFN shrinks from 0.52 to 0.27.
  - **A06:** seed 1 accepts 2 midlines (gate confidence 0.844) and scores 3.85 with RMSE 14.3, against 2.97 and 7.6 for a single midline. The heavy lognormal tail looks like a separate regime to the gate.
  - **Concrete:** all 3 seeds accept 3 midlines. Log score is slightly better (3.11 against 3.18), but RMSE is worse (6.67 against 5.82).
  - **Diagnostic finding:** the cross-fitted LOO gain is 0.2–0.7 nats/point on *every* problem except unimodal, so gate confidence makes the whole decision.
- **Post-hoc sensitivity (not used for the headline).** With a confidence threshold of 0.85 instead of 0.8, A06 would score 2.90 and concrete 3.18; A02 is unchanged. Whenever a midline is rejected, the auto model is exactly the `_1mid` model, so these numbers are exact counterfactuals. The threshold was moved after seeing test results, though, so it needs fresh seeds or problems before it is adopted.
- **Variance expansion is still not usable,** even when trained only on the calibration mask with a log-β penalty: A06 4.20, fish 1.50.
- **A learned-feature (MLP) mean kernel memorizes the data and was dropped.** The features are fit on every point, so their LOO residuals are not honest. On n=100 fish and concrete, the Stage-1 LOO NLL was −1.4 and −2.4, against test NLLs of 97 and 222, with the noise driven to about 0.002.

### 9.5 Quantile (Kumo) vs CDF-histogram (TabPFN) parameterization

The user's framing holds up. The two parameterizations optimize different scores:

- **Quantiles optimize CRPS.** CRPS is $2\int_0^1 \text{pinball}_\taud\tau$, so pinball training targets the evaluation metric directly. The quantile GP beats TabPFN's exact bar CRPS on A01 (2.78 against 2.81, indep), unimodal (0.168 against 0.172) and A05 (0.878 against 0.905). On those three problems the quantile GP has the **best CRPS of every model**, with a 4–14 s fit and no pretraining.
- **The implied density is poor for log score.** The density from quantiles is $\Delta\tau/\Delta q$ between knots, so it spikes wherever adjacent quantiles come close. Everything beyond τ = 0.05 / 0.95 is extrapolated.
  - **Tails.** Exponential tails sized from the last spacing blew up on A06 (4.64). It took GPD tails, with scale κ and shape ξ chosen by likelihood, to reach 3.28.
  - **More quantiles make it worse.** J=49 has about the same CRPS as J=19 but a worse log score (A01: 3.38 against 3.07), because there is more spacing noise per level. Kumo can use 999 quantiles only because a large pretrained model shares strength across levels.
- **Discontinuities break smooth quantile functions.** On A02 every quantile has to jump at the step. A smooth RFF function puts the τ-knots of rows near the step across the gap, leaving mass in the empty region: −0.71 against −1.43 for the logistic GP. Density parameterizations represent the bimodal density at the step directly.
- **Joint versus independent τ.** The Kronecker τ-prior with monotone increments wins on A02, A04 and A05: on A04, log score is 3.28 against 3.82 and the indep variant's CRPS_bar is unstable at 4.12±0.63. Independent per-τ fits with post-sorting win on A01, unimodal, A06, fish and concrete. Neither dominates. The increments prior shrinks spacings toward the marginal, which helps when shape is nearly constant in x and hurts otherwise.
- **The CDF/density side wins on log score.** `logistic_gp` is the continuous version of the TabPFN parameterization: a log-density on a fixed y-grid, but with a GP prior instead of bins plus projection. It beats every quantile variant on log score on 6 of 8 problems, and beats `hist_mogp_t1` on log score on 6 of 8. The exceptions are A06 (3.05 against 2.99) and fish (1.34 against 1.30).

Practical rule: if CRPS is the target, a pinball-trained quantile model is a strong, cheap choice. If log score or calibrated densities are the target, parameterize the density (TabPFN, logistic GP) and not the inverse CDF.

### 9.6 Success criteria (Tier-3 plan)

| criterion | status |
|---|---|
| Auto gamma within 0.1 nats of the best per-problem Tier-1/2 gamma variant on every problem | **not met, 6 of 8.** A01 is 2.96 against 2.78 (base gamma). A06 is 3.20 against 2.81 (`_t2`) or 2.93 (`_het`), driven by one seed accepting 2 midlines. It is *better* than every Tier-1/2 variant on A05 (1.84) and fish (1.29). |
| Closes the concrete gap to TabPFN by ≥ 0.4 nats | **not met.** The gap is 1.26 against 1.23 for t1. The midline is still the bottleneck: RMSE is 6.7 for auto, 5.8 single-midline, 3.7 for TabPFN. |
| Closes the A02 gap by ≥ 0.2 nats | **not met** (0.08: gap 0.27 against 0.36). The `_nohet` ablation closes 0.15 (−1.34). |
| Auto cov90 in [0.85, 0.95] on 8 of 8 | **7 of 8.** The range is 0.861–0.942, except unimodal at 0.957 (the same as t1). |
| Auto no slower than t1 | **not met at n ≥ 700.** Multi-midline EM dominates: concrete takes 710 s against 166 s, fish 367 s against 148 s. Single-midline auto (`_1mid`) is about t1 speed (178 s and 159 s). The Tier-3 runs used 6 concurrent processes × 4 threads, so wall times are pessimistic. |
| `quantile_gp` beats `hist_mogp_t1` on CRPS on both tabular sets, with no log-score blow-ups | **partly.** Exact bar CRPS: concrete is 3.57 against 3.76† (met); fish is 0.539 against 0.501 (not met; `_indep` gets 0.482). There are no blow-ups (worst 3.53 on concrete). The official moment CRPS is not met (concrete 6.6), for the tail reason in 9.3. |
| A hist-side Tier-3 model beats `hist_mogp_t1` on log score on ≥ 5 of 8 problems | **met, 6 of 8**: `logistic_gp`. |

### 9.7 Deviations from the plan

- **Learnable mean:** linear kernel only. The gated-MLP mean was implemented and removed because it memorizes the data (Section 9.4).
- **Multi-midline selection:**
  - Pruning uses responsibility mass, floored at $5(d+1)$ points per midline (a sub-model is fit per partition), and acceptance uses a 5-fold cross-fitted gate, instead of Hard-Concrete.
  - The in-sample gate memorized interleaved assignments on tabular data and accepted 3 midlines on fish.
  - Accepted midlines each get a full auto sub-model on their hard partition, combined by gate probabilities.
- **Calibration mask:** fixed, not rotating; `_rotcal` shows why.
- **Quantile GP:**
  - Asymmetric-Laplace likelihood with a learned σ sets the data/prior balance.
  - GPD tails with (κ, ξ) chosen by training-row likelihood replace pure exponential tails.
  - The bars' shared border set contains every test row's knots, so each row's piecewise-linear CDF is represented exactly.
- **Logistic GP:** Gaussian base plus β plus an ε-floor, fit by MAP rather than variational inference. Without the base, unimodal scored 1.36. Without β, A02 scored 2.8, and without the floor −0.8.
- **hist LMC:** exact Bayesian regression on Kronecker RFF features ($Q=2$, $D_x=48$, $D_h=12$) instead of gpytorch's `LMCVariationalStrategy`.
- **Extras:** an exact `bar_crps` metric, and a fix for an import-shadowing bug (`diagnostics`) when QSAR is the first problem loaded.

### 9.8 Recommendations

1. **Make** `logistic_gp` **the main hist-side model** and retire `hist_mogp_t1`, `hist_lmc_t1` and `hist_gpd`. Next steps:
   - Variational inference over $w$, for epistemic width on tabular data.
   - Using the gamma or LOO midline as $m(x)$; its RMSE on concrete is 6.8 against 5.8 for the midline.
   - More random features in high dimension.
2. **Gamma auto as the one-call default is close but not ready.** Keep in-loop calibration, LOO Stage 1, the linear mean and het. Make multi-midline acceptance more conservative and validate on fresh seeds: either the 0.85 threshold, or require the final sub-models to win on the calibration mask. Speed up EM at large n (fewer EM iterations, or subsample for the selection step).
3. **Keep** `quantile_gp` **/** `_indep` **as the CRPS specialists.** A hybrid loss (pinball plus density NLL) or a density head on the quantile knots would be needed to make one model good on both scores.
4. **Report exact** `CRPS_bar` **for every bar model.** The moment-summary CRPS is misleading for any heavy-tailed or multimodal bar PPD.
5. **Not worth pursuing further:** `hist_lmc_t1`, which is worse than `hist_mogp_t1` on log score on 5 of 8 problems and 4–10× slower on the synthetics; Stage-2 variance expansion; and learned-feature mean kernels under LOO.

## 10. Tier 4: padded SORF, 9.8 recommendations, scale, novelty

Tier 4 implements every Section 9.8 recommendation on a scalable foundation: padded structured orthogonal random features (SORF) with closed-form leave-one-out (LOO) replace random Fourier features (RFF) and the exact GP in Stage 1, in the gated midlines, and in `logistic_gp` / `quantile_gp`. Conditional and posterior-vs-data plots run on every problem, including $d>1$. Large-$n$ synthetics are at $n=1000$ and $n=5000$.

```bash
python aggregate_tier1.py results/foundation_compare_tier4 --seeds 0,1,42 --out results/foundation_compare_tier4/summary_table.md
python aggregate_tier1.py results/foundation_compare_tier4 --seeds 3,4,5 --out results/foundation_compare_tier4/fresh_seeds.md
```

### 10.1 What was built

| piece | file | key(s) |
|---|---|---|
| Padded SORF: blocks of size $d_{\mathrm{pad}}=\max(64, 2^{\lceil\log_2 d\rceil})$, keep the first $d$ rows, scale by $\sqrt{d_{\mathrm{pad}}}$. Default radial law is `"none"` (the $\chi(d)$ redraw is worse at $d=1$). | `gpplus/utils/sorf_features.py` | `SorfFeatures`, `sample_padded_sorf`, `auto_num_freq` |
| Kernel-error table vs RFF / ORF / plain Yu-SORF | `test_sorf_features.py` | `results/sorf_kernel_error.txt` |
| Closed-form hat-matrix LOO for the SORF Bayesian linear model, $O(nD^2)$; optional NIGP slope noise; `backend="auto"` picks SORF when $n>1500$ | `gpplus/models/loo_gp_core.py` (`SorfLooCore`, `make_loo_core`) | |
| Gated midlines: EM on a stratified subsample of at most 1500 points, at most 4 EM iterations with a 1% hard-assignment early stop, `conf_min=0.85`, `accept="calib"` | `gated_midline.py`, `foundation_adapters.py` | `_c85`, `_calibaccept`, `_sorf`, `_sorf_nigp` |
| Logistic-GP v2: joint padded-SORF over $(x, r)$, $\cos(a+c)$ factorization, LOO midline with a MAP correction $m=m_{\mathrm{LOO}}+c_m+h(x)^\top v_m$, frozen amplitude | `logistic_gp_density.py` | `logistic_gp_sorf_{vi,map,gamma}` |
| Quantile-GP v2: padded-SORF $x$-features; hybrid pinball + $\lambda\cdot$knot-NLL; $\lambda$ selected on the calibration mask (`"auto"`) | `quantile_gp.py` | `_sorf`, `_hybrid`, `_hybrid_l01`, `_hybrid_auto` |
| Exact CRPS for bar PPDs, Gaussian mixtures (Grimit et al. 2006) and quantile+GPD tails | `histogram_gp.py`, `quantile_gp.py` | columns `CRPS_bar`, `CRPS_quantile` |
| $d>1$ plots: conditionals at the 5/25/50/75/95% quantiles of the predicted median, with a $k$-NN histogram of training $y$; sorted 50/90% bands + PIT | `plot_foundation_marginals.py` | `save_conditionals_multid`, `save_posterior_vs_data_multid` |

Deprecated (kept runnable by name, dropped from `--models default`): `hist_lmc*`, the `_vexp` variants, and `midline_gamma_mlp_t2`. Learned-feature mean kernels under LOO stay out (Section 9.4).

### 10.2 Deviations from the plan

- **Padded-SORF radial law.** The plan allowed an optional $\chi(d)$ redraw of column radii. At $d=1$, $D=256$ that variant is *worse* than RFF ($1.12\times 10^{-3}$ against $9.88\times 10^{-4}$); `"none"` is at or below RFF in every cell of the test, so it is the default.
- **No final full-$n$ M-step.** The plan called for one weighted M-step on all $n$ after subsampled EM. The cores are not used downstream — only `assign` / `gate` / `alive` are — so that step was skipped.
- **Logistic v2 is MAP, not VI.** Mean-field VI over $w$ is strictly worse than MAP (Section 10.4). The default key is `logistic_gp_sorf_map`; `logistic_gp_sorf_vi` is reported as a negative result.
- **Amplitude is frozen.** Learning $\log\mathrm{amp}$ on the calibration mask hurt every problem. `learn_amp=False` is the default.
- **Midline MAP correction.** A fixed LOO $m(x)$ hurt the multimodal problems. The correction $m = m_{\mathrm{LOO}} + c_m + h(x)^\top v_m$ on a separate SORF bank restored them.
- **Hybrid $\lambda$ is selected, not fixed at 1.** $\lambda=1$ overfits knot gaps at $n=100$ (A01 log score 3.56 against 2.96 for pinball-only). `lam_nll="auto"` fits $0, 0.1, 1$ and keeps the best held-out `pinball·J + knot_nll`.
- **Exact mixture CRPS sits in** `CRPS_bar`**.** Gaussian-mixture models use the Grimit et al. (2006) closed form under the same column as bar CRPS, so every model is ranked on an exact CRPS. The official ScoringBench `CRPS` column remains the 512-point grid mixture integral.

### 10.3 Padded SORF is required in low $d$

Plain Yu-SORF (Yu et al., 2016) draws $D\times D$ Hadamard–diagonal blocks and keeps the first $d$ rows. Every frequency then has Euclidean norm exactly $\sqrt{d}$. At $d=1$ that is a single cosine of frequency 1, not an RBF spectral sample. Yu et al. already flag a bias for $d<32$; our synthetics are 1-D in $x$ and 2-D in $(x,y)$.

Kernel approximation mean squared error against the exact RBF (from `results/sorf_kernel_error.txt`):

| $d$ | $D$ | RFF | ORF | Yu-SORF | padded SORF | padded + $\chi$ |
|---|---|---|---|---|---|---|
| 1 | 256 | $9.88\times 10^{-4}$ | $3.60\times 10^{-4}$ | $1.30\times 10^{-1}$ | $8.29\times 10^{-4}$ | $1.12\times 10^{-3}$ |
| 1 | 1024 | $2.33\times 10^{-4}$ | $1.38\times 10^{-4}$ | $1.30\times 10^{-1}$ | $1.09\times 10^{-4}$ | $5.26\times 10^{-4}$ |
| 2 | 256 | $1.04\times 10^{-3}$ | $9.20\times 10^{-4}$ | $4.43\times 10^{-2}$ | $3.99\times 10^{-4}$ | $1.20\times 10^{-3}$ |
| 8 | 256 | $1.23\times 10^{-3}$ | $3.97\times 10^{-4}$ | $2.47\times 10^{-3}$ | $5.01\times 10^{-4}$ | $7.48\times 10^{-4}$ |
| 32 | 256 | $1.42\times 10^{-3}$ | $4.05\times 10^{-4}$ | $5.05\times 10^{-4}$ | $4.45\times 10^{-4}$ | $4.82\times 10^{-4}$ |

Padded SORF (`d_pad = max(64, next_pow2(d))`, scale $\sqrt{d_{\mathrm{pad}}}$, keep the first $d$ rows) is at or below RFF in every cell. The $\chi(d)$ redraw is not. This is an implementation fix, not a contribution (Section 10.7).

`SorfLooCore` is the same Bayesian linear model with these features. Closed-form LOO through the hat diagonal matches brute-force refits to $10^{-8}$. At $n=1000$ the test NLL matches the exact core ($-0.871$ against $-0.872$) in 9.5 s against 25.7 s.

### 10.4 Logistic v2: MAP beats VI; freeze the amplitude

Ablation, 3-seed means, $n=100$, $n_{\mathrm{test}}=200$ (lower is better). `v1` is the Tier-3 RFF MAP model. All v2 rows freeze the amplitude.

| problem | v1 | MAP + LOO midline | VI + LOO midline | MAP, no midline |
|---|---|---|---|---|
| A01 | 2.90 | 2.97 | 3.06 | 2.92 |
| A02 | −1.36 | −1.43 | −1.24 | −1.18 |
| A05 | 1.82 | 1.86 | 1.85 | 1.83 |
| unimodal | 0.44 | 0.33 | 0.65 | 0.37 |
| A06 | 3.15 | 2.89 | 2.98 | 2.90 |

- **VI underperforms MAP.** Plug-in prediction from the variational mean does not close the gap (unimodal 0.66, A01 3.06). Mean-field $q(w)$ with a reparameterized ELBO is a worse point estimate than MAP on this model, not a better uncertainty estimate. Report this honestly; do not ship `logistic_gp_sorf_vi` as the default.
- **Learning the amplitude on the calibration mask overfit.** The mask is 15 points at $n=100$. Freezing `log_amp` was the single largest v2 fix.
- **The $n/4$ Woodbury cap is wrong for density features.** It gave $D=32$ at $n=100$. Density models use `woodbury=False`.
- **SORF vs RFF features are interchangeable once padded.** The gap is in the midline and the amplitude, not the feature family.

### 10.5 Results

Headline numbers are 3-seed means (seeds 0, 1, 42) unless noted. `CRPS exact` is `CRPS_bar`: piecewise-uniform bar CRPS, Grimit et al. (2006) for Gaussian mixtures. Full mean±sd, fresh seeds 3–5, and $n=1000/5000$ are in `results/foundation_compare_tier4/summary_table.md` and `fresh_seeds.md`. Plots: `seed{0,1,42}/figures/` (1-D probes, posterior vs data, bands+PIT) and `real_*/figures/` ($d>1$ conditionals + bands+PIT).

**Headline.**

- **At $n=5000$ on A01, gamma-auto (c85 or SORF) ties TabPFN on log score (2.639).** Hellinger is 0.035 against TabPFN's 0.046. That is the first time a GP-component model in this suite matches TabPFN on a multimodal synthetic at scale. Wall time is the remaining gap: 16 min (SORF) / 82 min (exact-core) against TabPFN's 10 s.
- `conf_min=0.85` **and** `accept="calib"` **are almost the same model on seeds 0/1/42.** They differ from the 0.80 rule only on fish and concrete. Fresh seeds 3–5 confirm A06 no longer blows up (2.84–2.89 against the 3.85 seed in Section 9.4).
- **MAP vs VI is problem-dependent.** MAP is the default on the synthetics. VI is the best non-TabPFN model on *concrete* (2.89 against MAP 3.03 and gamma-auto 3.18).
- **Hybrid $\lambda=1$ overfits;** `"auto"` **falls back to pinball.** `quantile_gp_sorf` is the quantile default. On A05 at $n=1000$ it *beats* TabPFN on log score (1.559 against 1.574).
- **The 10-minute $n=5000$ target is missed for exact-core gamma-auto, met-ish for SORF Stage 1.** A01 $n=5000$: 4950 s (c85) vs 979 s (sorf) vs 9.5 s (TabPFN). Fish 1025 s (c85) / 476 s (sorf); concrete 1818 s / 708 s. The subsampled EM plus SORF Stage 1 is the speed win, not the 4-iteration cap alone.

#### 10.5.1 Log score, $n=100$ (lower is better)

| problem | TabPFN | γ auto | γ c85 | γ calib | γ SORF | log MAP | log VI | q-SORF | q-auto | hist t1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A01 | **2.70** | 2.93 | 2.93 | 2.93 | 2.85 | 2.97 | 3.05 | 2.93 | 2.95 | 2.97 |
| unimodal | **0.25** | 0.28 | 0.28 | 0.28 | 0.38 | 0.33 | 0.50 | 0.44 | 0.33 | 0.57 |
| A02 | **−1.54** | −1.30 | −1.30 | −1.30 | −0.85 | −1.16 | −1.27 | −0.75 | −0.67 | −1.23 |
| A04 | **3.07** | 3.23 | 3.23 | 3.23 | 3.27 | 3.29 | 3.35 | 3.23 | 3.23 | 3.33 |
| A05 | **1.66** | 1.88 | 1.88 | 1.88 | 1.94 | 1.96 | 1.87 | 1.77 | 1.90 | 2.30 |
| A06 | **2.60** | 3.24 | 3.26 | 3.26 | 2.88 | 2.87 | 2.95 | 3.18 | 3.18 | 3.08 |
| fish | **0.98** | 1.37 | 1.30 | 1.30 | 1.39 | 1.52 | 1.40 | 1.61 | 1.64 | 1.31 |
| concrete | **1.85** | 3.22 | 3.18 | 3.18 | 3.43 | 3.03 | 2.89 | 3.70 | 3.73 | 3.42 |

- **c85 vs the 0.80 rule.** Identical on the six synthetics for seeds 0/1/42 (no extra midlines accepted, or the same ones). On fish, c85 is 1.30 against 1.37; on concrete 3.18 against 3.22. `accept="calib"` matches c85 on every problem in this seed set, so the calibration-mask test did not change a single decision here.
- **Fresh seeds 3–5, A06 log score:** auto 2.84, c85 2.89, calib 3.12, TabPFN 2.66. The 3.85 blow-up of Section 9.4 does not recur. calibaccept is *worse* on this fresh A06 (one seed still greedy). Keep c85 as the default, not calib.
- **SORF Stage 1 vs exact.** Helps A01 (2.85 vs 2.93) and A06 (2.88 vs 3.24); hurts unimodal (0.38 vs 0.28), A02 (−0.85 vs −1.30, high variance) and concrete (3.43 vs 3.18). Use `backend="auto"` ($n>1500$), not forced SORF at $n=100$.
- **logistic MAP vs the Tier-3 RFF model.** Matches on A01 (2.97 vs 2.90) and unimodal (0.33 vs 0.44, actually better); loses the A02 win (−1.16 vs −1.43) because one of three seeds is bad (sd 0.45). VI is worse on 7 of 8 problems and better only on concrete.
- **quantile** `"auto"` **≈ pinball.** $\lambda=1$ is 3.62 on A01 and 0.73 on unimodal. The selector picks $\lambda=0$ or $0.1$. Keep pinball as the CRPS specialist; do not ship $\lambda=1$.

#### 10.5.2 Exact CRPS, $n=100$

`hist_mogp_t1` is rerun, so there is no moment-summary dagger.

| problem | TabPFN | γ c85 | log MAP | q-SORF | hist t1 |
|---|---|---|---|---|---|
| A01 | **2.81** | 2.98 | 2.97 | 2.85 | 2.81 |
| unimodal | 0.172 | 0.171 | 0.176 | 0.180 | 0.192 |
| A02 | **0.029** | 0.043 | 0.052 | 0.044 | 0.062 |
| A04 | 3.68 | 3.74 | 4.03 | **3.64** | 3.86 |
| A05 | 0.905 | 0.965 | 0.999 | **0.900** | 0.952 |
| A06 | **2.36** | 2.64 | 2.80 | 2.66 | 2.75 |
| fish | **0.399** | 0.464 | 0.522 | 0.501 | 0.489 |
| concrete | **1.55** | 3.16 | 2.73 | 3.71 | 3.68 |

The quantile GP still wins CRPS on A04 and A05 among non-TabPFN models, matching Section 9.5. `hist_mogp_t1` exact CRPS is now 2.81 on A01 (was 2.90†).

#### 10.5.3 Large $n$ (seed 0, $n_{\mathrm{test}}=1000$)

Log score / wall seconds:

| problem | $n$ | TabPFN | γ c85 | γ SORF | log MAP | q-SORF | exact GP |
|---|---|---|---|---|---|---|---|
| A01 | 1000 | **2.63** / 6 s | 2.71 / 2441 s | 2.70 / 1106 s | 2.70 / 259 s | 2.74 / 21 s | 3.11 / 7 s |
| A01 | 5000 | **2.64** / 10 s | **2.64** / 4950 s | **2.64** / 979 s | 2.66 / 443 s | 2.70 / 81 s | — |
| unimodal | 1000 | 0.24 / 6 s | **0.23** / 2284 s | 0.32 / 1047 s | 0.38 / 298 s | 0.24 / 17 s | **0.22** / 3 s |
| unimodal | 5000 | **0.22** / 10 s | 0.25 / 2057 s | 0.25 / 1856 s | 0.24 / 753 s | 0.24 / 149 s | — |
| A02 | 1000 | **−1.56** / 6 s | −1.40 / 2480 s | −1.38 / 1553 s | −1.53 / 424 s | −1.11 / 30 s | −1.32 / 16 s |
| A02 | 5000 | **−1.56** / 10 s | −1.35 / 2057 s | −1.35 / 1856 s | −1.56 / 752 s | −0.82 / 158 s | — |
| A05 | 1000 | 1.57 / 6 s | 1.74 / 2608 s | 1.74 / 1160 s | 1.66 / 389 s | **1.56** / 27 s | 2.06 / 9 s |
| A05 | 5000 | **1.59** / 10 s | 1.77 / 5585 s | 1.77 / 1158 s | 1.66 / 386 s | 1.60 / 83 s | — |
| A06 | 1000 | **2.45** / 6 s | 2.74 / 2417 s | 2.65 / 1393 s | 2.65 / 399 s | 3.96 / 25 s | 3.34 / 11 s |
| A06 | 5000 | **2.45** / 11 s | 2.83 / 5533 s | 2.64 / 1259 s | 2.59 / 384 s | 2.49 / 76 s | — |

- **SORF Stage 1 at $n=1000$ is within 0.01 nats of the exact-core gamma-auto on A01, A02, A05**, and *better* on A06 (2.65 vs 2.74), at roughly half the wall time. Exact GP (homoscedastic) is not a density competitor on A01/A05/A06; it is the unimodal reference (0.22).
- **A02 at $n=5000$:** logistic MAP −1.560 vs TabPFN −1.564. The remaining gap is 0.004 nats.
- **A06 quantile at $n=100$–$1000$ still has a tail blow-up** (log score 3.96 at $n=1000$); at $n=5000$ it is 2.49, next to TabPFN. The GPD selector needs more tail observations than $n=1000$ supplies on this lognormal.

cov90 for gamma-auto c85 at $n=100$ is in $[0.915, 0.972]$ on the synthetics (unimodal 0.962, slightly high, same as t1). At $n=5000$ A01 it is 0.894, inside $[0.85, 0.95]$.

### 10.6 Plots

- **1-D synthetics.** `save_marginals_per_probe` and `save_posterior_vs_data` are on for the Tier-4 runs (no `--no-marginal-plots`). True-density overlays (`true_pdf_`*) are drawn where they exist. Figures land in `results/foundation_compare_tier4/seed{0,1,42}/figures/`.
- **$d>1$ (fish, concrete).** Probes are the test rows whose reference-model median (TabPFN, else the first model) is nearest each of the 5/25/50/75/95% quantiles of that median. Each panel overlays $p(y\mid x_i)$ on a histogram of the $k=\max(20, n/30)$ nearest training $y$ in standardized $x$. A second figure sorts the test set by predicted median, draws 50% and 90% bands against observed $y$, and shows the PIT histogram $F(y_i\mid x_i)$. These two figures are also produced for 1-D problems.

Smoke on QSAR fish ($n=726$, $d=6$, seed 0): exact GP coverage was 0.58 / 0.91 at 50/90%; the quantile GP was underdispersed (0.40 / 0.75) with a U-shaped PIT. That is the expected pinball-without-NLL signature from Section 9.5.

### 10.6.1 NIS, CRPS, and CRPS exact

Protocol: 8 base-suite problems, 3-seed means (seeds 0, 1, 42). **NIS** is the ScoringBench negative interval score (lower is better). **CRPS** is the ScoringBench / moment-summary column (Gaussian-mixture grid integral, or the Gaussian moment summary of a bar PPD). **CRPS exact** is `CRPS_bar`: piecewise-uniform bar CRPS, Grimit et al. (2006) closed form for mixtures, or the knot+GPD quantile CRPS for quantile models. Bold = best (sole or tied) in that row among the displayed models. γ auto / γ calib match γ c85 on these seeds for almost every cell (Section 10.5), so only c85 is shown.

#### NIS (lower is better)

| problem | TabPFN | exact GP | γ c85 | γ SORF | log MAP | log VI | q-SORF | q-auto | hist t1 |
|---|---|---|---|---|---|---|---|---|---|
| A01 | **3.385** | 4.425 | 4.440 | 4.049 | 4.455 | 5.009 | 3.563 | 3.563 | 3.602 |
| unimodal | 1.000 | **0.947** | 1.069 | 1.281 | 1.054 | 1.708 | 1.171 | 1.180 | 1.641 |
| A02 | **0.533** | 1.310 | 1.925 | 1.186 | 1.956 | 2.379 | 1.337 | 1.398 | 2.186 |
| A04 | 4.204 | 4.069 | **4.055** | 4.245 | 4.477 | 4.765 | 4.349 | 4.349 | 4.608 |
| A05 | 5.310 | 6.196 | **4.840** | 5.197 | 5.560 | 5.459 | 5.153 | 5.175 | 5.170 |
| A06 | **2.994** | 4.003 | 4.383 | 3.594 | 4.075 | 4.438 | 11.268 | 11.633 | 4.103 |
| fish | **2.578** | 3.214 | 3.227 | 3.608 | 3.821 | 3.356 | 8.064 | 5.025 | 3.480 |
| concrete | **1.129** | 7.605 | 2.507 | 3.184 | 1.997 | 1.851 | 4.824 | 3.117 | 2.442 |

#### CRPS, ScoringBench / moment-summary (lower is better)

| problem | TabPFN | exact GP | γ c85 | γ SORF | log MAP | log VI | q-SORF | q-auto | hist t1 |
|---|---|---|---|---|---|---|---|---|---|
| A01 | 2.907 | 3.162 | 2.980 | 2.956 | 3.050 | 3.152 | 2.919 | 2.920 | **2.898** |
| unimodal | 0.172 | **0.166** | 0.171 | 0.181 | 0.173 | 0.230 | 0.179 | 0.175 | 0.215 |
| A02 | **0.030** | 0.048 | 0.043 | 0.039 | 0.082 | 0.091 | 0.056 | 0.056 | 0.102 |
| A04 | 3.762 | 3.749 | **3.739** | 3.822 | 3.983 | 3.950 | 3.758 | 3.758 | 3.892 |
| A05 | **0.914** | 0.993 | 0.965 | 0.978 | 1.087 | 1.059 | 0.970 | 0.967 | 1.009 |
| A06 | **2.401** | 2.804 | 2.642 | 2.588 | 3.125 | 3.322 | 6.265 | 6.749 | 3.059 |
| fish | **0.400** | 0.456 | 0.464 | 0.522 | 0.544 | 0.537 | 0.832 | 0.627 | 0.503 |
| concrete | **1.568** | 10.573 | 3.159 | 3.695 | 2.836 | 2.683 | 5.710 | 4.261 | 3.755 |

#### CRPS exact (lower is better)

| problem | TabPFN | exact GP | γ c85 | γ SORF | log MAP | log VI | q-SORF | q-auto | hist t1 |
|---|---|---|---|---|---|---|---|---|---|
| A01 | **2.805** | 3.162 | 2.979 | 2.956 | 2.974 | 3.039 | 2.851 | 2.853 | 2.812 |
| unimodal | 0.172 | **0.166** | 0.171 | 0.181 | 0.176 | 0.197 | 0.180 | 0.167 | 0.192 |
| A02 | **0.029** | 0.048 | 0.043 | 0.039 | 0.052 | 0.048 | 0.044 | 0.044 | 0.062 |
| A04 | 3.681 | 3.749 | 3.739 | 3.823 | 4.025 | 3.976 | **3.637** | **3.637** | 3.855 |
| A05 | 0.905 | 0.993 | 0.965 | 0.978 | 0.999 | 0.953 | 0.900 | **0.900** | 0.952 |
| A06 | **2.357** | 2.803 | 2.641 | 2.589 | 2.797 | 2.862 | 2.659 | 2.817 | 2.747 |
| fish | **0.399** | 0.456 | 0.464 | 0.523 | 0.522 | 0.530 | 0.501 | 0.498 | 0.489 |
| concrete | **1.548** | 10.570 | 3.163 | 3.707 | 2.725 | 2.588 | 3.711 | 3.620 | 3.676 |

#### Win counts (8 problems)

| model | NIS sole | CRPS sole | CRPS exact sole | NIS shared | CRPS shared | CRPS exact shared |
|---|---|---|---|---|---|---|
| TabPFN | 5 | 5 | 5 | 0 | 0 | 0 |
| exact GP | 1 | 1 | 1 | 0 | 0 | 0 |
| γ c85 | 2 | 1 | 0 | 0 | 0 | 0 |
| γ SORF | 0 | 0 | 0 | 0 | 0 | 0 |
| log MAP | 0 | 0 | 0 | 0 | 0 | 0 |
| log VI | 0 | 0 | 0 | 0 | 0 | 0 |
| q-SORF | 0 | 0 | 0 | 0 | 0 | 1 |
| q-auto | 0 | 0 | 1 | 0 | 0 | 1 |
| hist t1 | 0 | 1 | 0 | 0 | 0 | 0 |

#### Win counts excluding TabPFN

| model | NIS sole | CRPS sole | CRPS exact sole | NIS shared | CRPS shared | CRPS exact shared |
|---|---|---|---|---|---|---|
| exact GP | 2 | 2 | 2 | 0 | 0 | 0 |
| γ c85 | 2 | 2 | 0 | 0 | 0 | 0 |
| γ SORF | 2 | 2 | 2 | 0 | 0 | 0 |
| log MAP | 0 | 0 | 0 | 0 | 0 | 0 |
| log VI | 1 | 1 | 1 | 0 | 0 | 0 |
| q-SORF | 1 | 0 | 0 | 0 | 0 | 1 |
| q-auto | 0 | 0 | 1 | 0 | 0 | 1 |
| hist t1 | 0 | 1 | 1 | 0 | 0 | 0 |

**Reading.**

- **TabPFN dominates all three metrics when it is in the pool:** 5/8 sole wins on NIS, CRPS, and CRPS exact. The three problems it does *not* win on NIS are unimodal (exact GP), A04 and A05 (γ c85). On CRPS exact it also loses A04 and A05 to the quantile GP and unimodal to exact GP.
- **Moment CRPS and CRPS exact disagree.** hist-MoGP “wins” A01 on the ScoringBench column (2.898 against TabPFN 2.907) but loses on exact CRPS (2.812 against 2.805). Quantile models look catastrophic on A06 moment CRPS (6.3–6.7) while exact CRPS is 2.66–2.82 — the same moment-summary trap as Section 9.3. Rank models on **CRPS exact**.
- **Among non-TabPFN models, the wins split by metric.** Exact GP takes unimodal on all three (and A01/unimodal when TabPFN is excluded for CRPS). γ c85 takes A04/A05 on NIS and A04 on moment CRPS. γ SORF, once TabPFN is removed, takes 2 NIS / 2 CRPS / 2 CRPS-exact sole wins (A02, A06 style). The quantile GP is the CRPS-exact specialist on A04 and A05 (beats TabPFN on both). Logistic MAP never wins any of the three among the display set; VI’s only non-TabPFN win is concrete.
- **NIS and CRPS exact are not the same ranking.** γ c85 is the best non-TabPFN NIS model on A04/A05 but never wins CRPS exact. Quantile wins CRPS exact on those problems while its NIS is mid-pack (and blows up on A06 / fish). Interval sharpness (NIS) and full-distribution CRPS select different models.
- **Full ablation list** (γ auto, γ calib, log γ, q-λ1, q-λ0.1): γ auto / c85 / calib share the same two NIS wins; q-λ0.1 takes the sole A05 CRPS-exact win over the full list (0.899). q-λ1 never wins. Tables: `results/foundation_compare_tier4/_nis_crps_tables_gen.md`.

### 10.7 Critical novelty assessment

Each entry: closest work, what is shared, what is actually new, verdict. No individual model class is new. The defensible contribution is a self-configuring, single-pass recipe that approaches TabPFN with GP components.

**Gamma two-stage** (GP midline, then MLP gate, offsets and $\gamma$, Hard-Concrete pruning).
Closest work: Stirn et al. (2023, AISTATS; [arXiv:2212.09184](https://arxiv.org/abs/2212.09184)), "faithful" heteroscedastic nets that stop the variance head from corrupting the mean; TORF (Madhusudhanan et al., 2026; [arXiv:2608.11114](https://arxiv.org/abs/2608.11114)), a frozen Stage-1 mean plus an odd residual flow that *preserves* that mean; Pati & Dunson (2014, AISM; [doi:10.1007/s10463-013-0415-z](https://doi.org/10.1007/s10463-013-0415-z)), a GP mean with a predictor-dependent mixture residual; Bishop (1994) mixture density networks; Louizos et al. (2018; [arXiv:1712.01312](https://arxiv.org/abs/1712.01312)) Hard-Concrete $L_0$ gates. Shared: the two-stage mean-then-residual split, the MLP gate, the $L_0$ prune. Possibly new: Stage 2 trained on *LOO* residuals (so the residual model does not see overfit Stage-1 fits), the $\gamma$ variance split $\sigma_0^2(1-\alpha_k\gamma_k)\beta_k$, and in-loop mask calibration of a single temperature. **Verdict: moderate-to-low novelty.** A combination of known parts; the LOO-residual training and the in-loop mask are the pieces that are not in Stirn or TORF.

**LOO Stage 1** (`LooGPCore`, `SorfLooCore`).
Closest work: Sundararajan & Keerthi (2001, *Neural Computation*), exact GP LOO; Vehtari et al. (2016, JMLR 17), PSIS-LOO; Allen (1974) PRESS; Cawley & Talbot, hat-matrix LOO for kernel ridge. Shared: $h_i=\phi_i^\top A^{-1}\phi_i/\sigma_i^2$, $\mu_{\mathrm{LOO}}=y-e/(1-h)$. The pairing with padded SORF and optional NIGP (McHutchon & Rasmussen, 2011) is engineering. **Verdict: not novel on its own.**

**Gated multi-midline.**
Closest work: overlapping mixtures of GPs (Lázaro-Gredilla, Van Vaerenbergh & Lawrence, 2012, *Pattern Recognition*), the same "global trajectories" idea without a gate; mixtures of GP experts (Tresp, 2001; Rasmussen & Ghahramani, 2002; Härkönen, Wade, Law & Roininen, 2025, JMLR 26:206, [SMC² MoE](https://jmlr.org/papers/v26/22-0973.html); Etienam, Law, Wade & Zankin, 2024, *Machine Learning*, [arXiv:2006.13309](https://arxiv.org/abs/2006.13309)). Shared: $x$-dependent assignment of points to GP experts. Possibly new: the LOO E-step (responsibilities from closed-form LOO predictives, not in-sample fits) and the cross-fitted-gate plus calibration-mask acceptance test. **Verdict: low-to-moderate novelty.** The model class is standard MoE; the selection rule is the only design that is not already in the MoE-GP literature, and even that is a holdout test.

**quantile_gp.**
Closest work: Abeywardana & Ramos (2015, AAAI), a GP prior with an asymmetric-Laplace likelihood; Boukouvalas, Barillec & Cornford (2012), EP for the same likelihood; SQR (Tagasovska & Lopez-Paz, 2019, NeurIPS), which feeds $\tau$ as an input; Rodrigues & Fan (2017, JCGS; [arXiv:1502.01115](https://arxiv.org/abs/1502.01115)), a *post-hoc* GP over $\tau$ that monotonizes separately fitted quantiles; Xu & Reich (2021, *Biometrics*; [doi:10.1111/biom.13576](https://doi.org/10.1111/biom.13576)) SPQR / I-spline CDFs with neural-net coefficients; Richards' blended-GPD spline networks for tails. Shared: pinball / ALD, a smoothness prior across $\tau$, GPD tails. Possibly new: the Kronecker $x$-by-$\tau$ GP with *monotone softplus increments* (Rodrigues & Fan adjust after the fact; SQR does not constrain crossings), train-selected GPD $(\kappa,\xi)$, and the hybrid pinball + knot-density NLL. **Verdict: low-to-moderate novelty.** Every piece exists; the joint monotone Kronecker parameterization plus the hybrid loss is the only combination that is not a straight reimplementation.

**logistic_gp.**
Closest work: Lenk (1988, 1991), logistic Gaussian process density estimation; Tokdar, Zhu & Ghosh (2010, *Bayesian Analysis*; [doi:10.1214/10-BA605](https://doi.org/10.1214/10-BA605)), a GP on $(x,y)$ with a logistic transform — Tokdar's dissertation also centres on a parametric density, which matches our Gaussian base; Riihimäki & Vehtari (2014, *Bayesian Analysis*; [doi:10.1214/14-BA872](https://doi.org/10.1214/14-BA872)), grid + Laplace + reduced-rank; GP-CDE (Dutordoir, Salimbeni, Deisenroth & Hensman, 2018, NeurIPS; [arXiv:1810.12750](https://arxiv.org/abs/1810.12750)), a latent-variable GP for conditional densities. Shared: $p(y\mid x)\propto\exp f(x,y)$ with a GP $f$. The contribution is midline centring $u=y-m(x)$, padded-SORF features, the $\varepsilon$-floor, and the (failed) VI attempt. **Verdict: low novelty.** This is Tokdar / Riihimäki–Vehtari with random features and a learned Gaussian base.

**hist_mogp, bar distributions, exact bar CRPS.**
Closest work: the TabPFN / PFN Riemann (bar) distribution; Imani & White (2018, ICML) and Imani, Luedemann, Scholnick-Hughes, Elelimy & White (2026, JMLR 27:66, [histogram loss](https://www.jmlr.org/papers/v27/24-0260.html)); Guha, Natarajan, Möllenhoff, Khan & Ndiaye (2024, ICLR; [arXiv:2404.08168](https://arxiv.org/abs/2404.08168)), regression-as-classification conformal prediction. The exact CRPS of a piecewise-uniform distribution is standard algebra (integrate $(F-1z\ge y)^2$ bar by bar). **Verdict: not novel.**

**NI-SORF / padded SORF.**
Yu, Suresh, Choromanski, Holtmann-Rice & Kumar (2016, ICML), orthogonal and structured orthogonal random features; McHutchon & Rasmussen (2011), noisy-input GPs. The pad-to-64 fix for $d<32$ is the correction Yu et al. imply and should be documented as an implementation choice, not a contribution. **Verdict: not novel.**

**Overall.** Do not claim a new model class. Claim a recipe: one call, no problem knobs, no internal refits, GP components, LOO-honest Stage 1, in-loop calibration, and a feature map that is accurate at $d=1$. That recipe is the paper, if there is one. The negative results (VI < MAP, $\lambda=1$ hybrid overfits, variance expansion, learned-feature means, Yu-SORF at low $d$, the skipped final M-step) belong in the paper too.

### 10.8 References

- Abeywardana, S. and Ramos, F. (2015). Variational inference for nonparametric Bayesian quantile regression. *AAAI*.
- Allen, D. M. (1974). The relationship between variable selection and data augmentation and a method for prediction. *Technometrics* 16:125–127.
- Bishop, C. M. (1994). Mixture density networks. Aston University technical report NCRG/94/004.
- Boukouvalas, A., Barillec, R. and Cornford, D. (2012). Gaussian process quantile regression using expectation propagation. *ICML* workshop.
- Dutordoir, V., Salimbeni, H., Deisenroth, M. and Hensman, J. (2018). Gaussian process conditional density estimation. *NeurIPS*. [arXiv:1810.12750](https://arxiv.org/abs/1810.12750).
- Etienam, C., Law, K., Wade, S. and Zankin, V. (2024). Fast deep mixtures of Gaussian process experts. *Machine Learning* 113:1483–1508. [arXiv:2006.13309](https://arxiv.org/abs/2006.13309).
- Grimit, E. P., Gneiting, T., Berrocal, V. J. and Johnson, N. A. (2006). The continuous ranked probability score for circular variables and its application to mesoscale forecast ensemble verification. *Q. J. R. Meteorol. Soc.*
- Guha, E., Natarajan, S., Möllenhoff, T., Khan, M. E. and Ndiaye, E. (2024). Conformal prediction via regression-as-classification. *ICLR*. [arXiv:2404.08168](https://arxiv.org/abs/2404.08168).
- Härkönen, T., Wade, S., Law, K. and Roininen, L. (2025). Mixtures of Gaussian process experts with SMC². *JMLR* 26(206):1–38. <https://jmlr.org/papers/v26/22-0973.html>.
- Imani, E. and White, M. (2018). Improving regression performance with distributional losses. *ICML*.
- Imani, E., Luedemann, K., Scholnick-Hughes, S., Elelimy, E. and White, M. (2026). Investigating the histogram loss in regression. *JMLR* 27(66):1–54. <https://www.jmlr.org/papers/v27/24-0260.html>.
- Lázaro-Gredilla, M., Van Vaerenbergh, S. and Lawrence, N. D. (2012). Overlapping mixtures of Gaussian processes for the data association problem. *Pattern Recognition* 45:1386–1395.
- Lenk, P. J. (1988). The logistic normal distribution for Bayesian, nonparametric, predictive densities. *JASA* 83:509–516.
- Louizos, C., Welling, M. and Kingma, D. P. (2018). Learning sparse neural networks through $L_0$ regularization. *ICLR*. [arXiv:1712.01312](https://arxiv.org/abs/1712.01312).
- Madhusudhanan, K., Klötergens, C., Schmidt-Thieme, L. and Yalavarthi, V. K. (2026). Two-stage odd residual flows for mean-preserving probabilistic time series forecasting. [arXiv:2608.11114](https://arxiv.org/abs/2608.11114).
- McHutchon, A. and Rasmussen, C. E. (2011). Gaussian process training with input noise. *NeurIPS*.
- Pati, D. and Dunson, D. B. (2014). Bayesian nonparametric regression with varying residual density. *Ann. Inst. Stat. Math.* 66:1–31. [doi:10.1007/s10463-013-0415-z](https://doi.org/10.1007/s10463-013-0415-z).
- Rasmussen, C. E. and Ghahramani, Z. (2002). Infinite mixtures of Gaussian process experts. *NeurIPS*.
- Riihimäki, J. and Vehtari, A. (2014). Laplace approximation for logistic Gaussian process density estimation and regression. *Bayesian Analysis* 9:425–448. [doi:10.1214/14-BA872](https://doi.org/10.1214/14-BA872).
- Rodrigues, T. and Fan, Y. (2017). Regression adjustment for noncrossing Bayesian quantile regression. *J. Comput. Graph. Statist.* [arXiv:1502.01115](https://arxiv.org/abs/1502.01115).
- Stirn, A., Wessels, H.-H., Schertzer, M., Pereira, L., Sanjana, N. E. and Knowles, D. A. (2023). Faithful heteroscedastic regression with neural networks. *AISTATS*. [arXiv:2212.09184](https://arxiv.org/abs/2212.09184).
- Sundararajan, S. and Keerthi, S. S. (2001). Predictive approaches for choosing hyperparameters in Gaussian processes. *Neural Computation* 13:1103–1118.
- Tagasovska, N. and Lopez-Paz, D. (2019). Single-model uncertainties for deep learning. *NeurIPS*.
- Tokdar, S. T., Zhu, Y. M. and Ghosh, J. K. (2010). Bayesian density regression with logistic Gaussian process and subspace projection. *Bayesian Analysis* 5:319–344. [doi:10.1214/10-BA605](https://doi.org/10.1214/10-BA605).
- Tresp, V. (2001). Mixtures of Gaussian processes. *NeurIPS*.
- Vehtari, A., Gelman, A. and Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Stat. Comput.* See also Vehtari et al., JMLR 2016, PSIS-LOO.
- Xu, S. G. and Reich, B. J. (2021). Bayesian nonparametric quantile process regression and estimation of marginal quantile effects. *Biometrics*. [doi:10.1111/biom.13576](https://doi.org/10.1111/biom.13576).
- Yu, F. X. X., Suresh, A. T., Choromanski, K. M., Holtmann-Rice, D. N. and Kumar, S. (2016). Orthogonal random features. *NeurIPS*. (SORF is in the same line; the ICML 2016 ORF paper is the usual cite.)
- Yu, K. and Zhang, J. (2005). A three-parameter asymmetric Laplace distribution and its extension. *Communications in Statistics*.
