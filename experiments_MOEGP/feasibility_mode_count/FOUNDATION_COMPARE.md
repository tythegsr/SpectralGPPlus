# Foundation-model comparison (gpplus2026)

Live comparison of **2-Stage Mixture GP** (MidlineSparse) against:

| Key | Model |
|-----|--------|
| `midline` | 2-Stage Mixture GP (lean champion: usage + adaptive_merge + alive + delta + alpha; sep/rec/overlap/gamma/orphan off) |
| `mdn` | Bishop MDN (MLP gate; free \(\pi,\mu,\sigma\); same \(K_{\max}\)) |
| `mdn_const` | Bishop MDN with x-independent mixture weights |
| `exact_gp` | Exact GP+ `GPR` (unimodal Gaussian predictive) |
| `tabgp` | TabGP-MLL (local package from `vendor/tabgp_pkg`) |
| `tabpfn3` | TabPFN 3.0 (`ModelVersion.V3`) |
| `tabpfn35` | TabPFN 3.5 (`ModelVersion.V3_5`) |
| `kumo` | NVIDIA KumoTabular via `structured-data-models` (999 quantiles) |
| `hist_mogp` | Histogram Kronecker MOGP (one-hot bins → \(K_{XX}\otimes K_{HH}\) + simplex project; B=200 req., capped at \(n\); 1000 Adam steps) |
| `hist_gpd` | Histogram Dirichlet GP (Milios GPD / `DirichletClassificationLikelihood` on bin indices) |

Problems: **A01**, **unimodal**, **A02**, **A04**, **A05**, **A06**, **COSI**, **TABARENA_CONCRETE**

### TABARENA_CONCRETE (OpenML 46917, Exact midlines)

Single TabArena-v0.1 regression dataset for a fast real-tabular head-to-head (not full TabArena CV):

| Field | Value |
|-------|--------|
| Dataset | `concrete_compressive_strength` (OpenML **46917**, TabArena suite 457) |
| Size | \(N=1030\), \(d=8\) numeric features, target `ConcreteCompressiveStrength` |
| Split | seeded **80/20**: `n_train=824`, `n_test=206`, seed 42 |
| Midline backend | **`exact`** (no NI-SORF / COSI path) |
| Artifacts | `results/foundation_compare_tabarena/` |

All `ALL_MODELS` keys (`midline`, `midline_gamma_mlp`, `midline_residual_mogp`, `midline_bayes_mogp`, `mdn`, `mdn_const`, `exact_gp`, `tabgp`, `tabpfn3`, `tabpfn35`, `kumo`, `hist_mogp`, `hist_gpd`).

```powershell
conda activate gpplus2026
cd experiments_MOEGP/feasibility_mode_count
python compare_foundation_baselines.py --problems TABARENA_CONCRETE --models all --no-marginal-plots
# defaults to train-sizes 824, n-test 206, out-dir results/foundation_compare_tabarena
```

Requires `openml` (`pip install openml`). ScoringBench metrics: RMSE, RRMSE, NIS, CRPS, `log_score`, coverage, wall.

#### Results (`n_train=824`, `n_test=206`, seed 42, Exact midlines)

| Model | RMSE | CRPS | log_score | wall_s |
|-------|------|------|-----------|--------|
| 2-Stage Mixture GP | 16.85 | 8.23 | 4.23 | 34.2 |
| 2-Stage Mixture GP (gamma MLP) | 5.85 | 3.32 | 3.93 | 44.1 |
| Midline Residual MoGP | 5.85 | 4.19 | 3.30 | 538.6 |
| Midline Bayes MoGP | 5.85 | 4.19 | 3.30 | 586.9 |
| Bishop MDN (MLP gate) | 8.22 | 4.43 | 3.47 | 10.3 |
| Bishop MDN (const gate) | 7.70 | 4.30 | 3.50 | 9.7 |
| Exact GP+ (GPR) | 5.85 | 3.15 | 3.16 | 13.7 |
| TabGP-MLL | 4.29 | 2.02 | 2.75 | 24.6 |
| TabPFN-3.0 | **3.71** | **1.55** | 2.15 | 0.8 |
| TabPFN-3.5 | 3.86 | 1.62 | **1.88** | **0.6** |
| NVIDIA KumoTabular | 3.83 | 1.58 | 12.30† | 1.1 |
| Hist multi-output GP | 13.76 | 7.40 | 3.80 | 360.6 |
| Hist Dirichlet GP | 14.42 | 8.07 | 3.76 | 331.8 |

† Kumo `log_score` is out of family here (quantile → density conversion); prefer RMSE/CRPS for that row.

Artifacts: `results/foundation_compare_tabarena/tabarena_concrete_n824_metrics.{json,csv}`, `figures/tabarena_concrete_n824_metrics_table.png`.

### COSI (hyperspectral `cos_i`, NI-SORF midlines)

S4 snow-TOA setup matching [`experiments_SORF/S4_emit_SORF.py`](../experiments_SORF/S4_emit_SORF.py):

- Data: `experiments_toa/data 11 QoI/snow_toa_fsnow_pure_flat_Sep09.nc`, QoI=`cos_i`, **X 287-D**
- Split: **`n_train=5000`**, **`n_test=5000`**, seed 42
- Artifacts: `results/foundation_compare_cosi/`
- Exact GP paths replaced by **NI-SORF** (`RFFGPR` + SORF + NIGP, `num_rff=2400`, 1000 epochs):
  - `exact_gp` → standalone NI-SORF baseline (table label “NI-SORF GP”)
  - `midline` / `midline_gamma_mlp` → NI-SORF Stage-1 midline
  - `midline_residual_mogp` / `midline_bayes_mogp` → NI-SORF midline + NI-SORF residual experts
  - `hist_mogp` / `hist_gpd` → NI-SORF per-bin GPs (`n_bins=16`)

```powershell
conda activate gpplus2026
cd experiments_MOEGP/feasibility_mode_count
python compare_foundation_baselines.py --problems COSI --models all
# defaults to train-sizes 5000, n-test 5000, out-dir results/foundation_compare_cosi
```

### Protocol (shared test across train sizes)

For each problem, generate a pool of size `max(train_sizes) + n_test` **once**, then:

1. One permutation of indices (seeded).
2. Freeze the **same** test set of size `n_test` (default **1000**).
3. Nested train prefixes of sizes default **`[100, 200, 400, 800]`** from the remaining pool (100 ⊂ 200 ⊂ 400 ⊂ 800).

Artifacts are keyed by train size, e.g. `a01_n100_metrics.json`, `a06_n800_metrics_table.png`.

### A06 (cubic + steep bump + strong centered lognormal noise)

\[
f(x)=0.1x^3-x^2+3x-9 + A\exp\!\left(-\frac{(x-c)^2}{2w^2}\right),\quad x\in[0,10]
\]

with \(A=18\), \(c=5\), \(w=0.35\) (narrow peak → near-discontinuous slopes).

\[
y=f(x)+\bigl(\kappa\,|f(x)|+\varepsilon_0\bigr)\,(L-\mathbb{E}[L]),\quad
L\sim\mathrm{LogNormal}(0,\sigma(x)^2),\quad
\mathbb{E}[L]=\exp(\sigma(x)^2/2)
\]

with \(\sigma(x)=\sigma_{\min}+(\sigma_{\max}-\sigma_{\min})(x/10)^p\)
(defaults \(\sigma_{\min}=0.35\), \(\sigma_{\max}=0.95\), \(p=1\), \(\kappa=0.65\), \(\varepsilon_0=0.8\)).

Preview noise **before** fitting models:

```powershell
python compare_foundation_baselines.py --preview-a06
# → results/foundation_compare/figures/a06_noise_preview.png
```

## Environment

```powershell
conda activate gpplus2026
```

Python: `C:\Users\forty\anaconda3\envs\gpplus2026\python.exe`

### One-time installs (already applied on this machine)

```powershell
# TabPFN 3.x
python -m pip install -U "tabpfn>=9.2.0"

# TabGP from Downloads zip → vendor/
Expand-Archive C:\Users\forty\Downloads\tabgp_pkg.zip `
  -DestinationPath C:\Users\forty\tyler_gpplus\SpectralGPPlus\vendor\tabgp_pkg -Force
python -m pip install -e C:\Users\forty\tyler_gpplus\SpectralGPPlus\vendor\tabgp_pkg\tabgp_pkg

# NVIDIA KumoTabular / TabFM (pulls torch>=2.7; reinstall CUDA wheel if needed)
python -m pip install structured-data-models
python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126"
```

Verify:

```powershell
python -c "import torch; from tabpfn.constants import ModelVersion; from tabgp import TabGPRegressor; import sdm; print(torch.__version__, torch.cuda.is_available(), sdm.models.KumoTabular, list(ModelVersion))"
```

Expect CUDA `True`, `KumoTabular`, and `ModelVersion` including `V3` / `V3_5`.

## Run

```powershell
cd C:\Users\forty\tyler_gpplus\SpectralGPPlus\experiments_MOEGP\feasibility_mode_count

# Inspect A06 noise first
python compare_foundation_baselines.py --preview-a06

# Smoke: A01 + A06, all models, default train sizes + shared n_test=1000
python compare_foundation_baselines.py --problems A01,A06 --models all

# Custom train sizes / test size
python compare_foundation_baselines.py --problems A05,A06 --models midline,exact_gp,tabpfn35 `
  --train-sizes 100,200,400,800 --n-test 1000

# Full suite
python compare_foundation_baselines.py --problems all --models all

# MidlineSparse vs Bishop MDN (writes results/foundation_compare_mdn/)
python compare_foundation_baselines.py --problems all --models midline,mdn,mdn_const
```

See also theoretical positioning: [`MDN_JOINT_POSITIONING.md`](MDN_JOINT_POSITIONING.md).

A01 entrypoint (metrics table + CDF/PDF overlays; MidlineSparse fit once, other FMs live):

```powershell
python compare_a01_scoringbench.py
python compare_a01_scoringbench.py --website-baselines   # + historical TabICL/Crepes
python compare_a01_scoringbench.py --skip-foundation     # MidlineSparse (+ MoE NPZ) only
```

## MidlineSparse vs Bishop MDN (live, 2026-11-02)

Protocol: problems A01 / unimodal / A02 / A04 / A05 / A06; shared test \(n=1000\); nested train sizes \(100,200,400,800\); seed 42; same \(K_{\max}\) as MidlineSparse. Artifacts: `results/foundation_compare_mdn/`.

**Headline (24 settings = 6 problems × 4 train sizes):**

| Criterion | MidlineSparse wins | Best MDN (`mdn` or `mdn_const`) wins |
|-----------|-------------------:|-------------------------------------:|
| `log_score` (lower better) | 10 | 14 |
| CRPS | 12 | 12 |

3-way `log_score` wins: midline 10, `mdn` 8, `mdn_const` 6.  
3-way CRPS wins: midline 12, `mdn_const` 7, `mdn` 5.

**Where MidlineSparse pulls ahead**

- **A01 (co-located bimodal):** MidlineSparse best `log_score` at all four \(n\) (e.g. \(n=100\): 2.68 vs MDN-const 3.64 / MDN 4.28). GP envelope + residual gauge helps small-\(n\) multimodality.
- **A04 (trimodal):** MidlineSparse usually best or near-best NLL; free MDN can spike badly (`mdn` `log_score` 8.31 at \(n=400\)).

**Where MDN pulls ahead**

- **A02 (step / x-partitioned):** free MLP-gate MDN dominates RMSE/CRPS/NLL — input-dependent free means beat a shared midline when modes are *not* co-located.
- **A06 (cubic + lognormal noise):** `mdn_const` often best CRPS/RMSE; MidlineSparse merge pressure can over-collapse.
- **unimodal / A05:** mixed; free MDN often slightly better NLL at larger \(n\), MidlineSparse competitive on CRPS.

**Reading for advisors:** MidlineSparse is empirically **not** “just MDN with a GP logo” on co-located multimodality (A01/A04), but it **is** the same CDE object class and can lose to a free MDN when the midline+carve-out inductive bias is wrong (A02/A06). Constant-gate MDN is the fairer co-located ablation; MLP MDN is the fairer x-regime ablation.

Selected cells (`log_score`):

| problem | n | midline | mdn | mdn_const |
|---------|--:|--------:|----:|----------:|
| A01 | 100 | **2.684** | 4.279 | 3.636 |
| A01 | 800 | **2.625** | 2.836 | 2.861 |
| A02 | 800 | −1.379 | **−1.553** | −1.520 |
| A04 | 200 | **2.832** | 3.399 | 3.433 |
| A06 | 200 | 3.318 | 3.183 | **2.830** |
| unimodal | 800 | 0.390 | **0.312** | 0.398 |

## Outputs

`results/foundation_compare_mdn/` (MDN head-to-head) and prior suites under `results/foundation_compare*`

- `{problem}_n{N}_metrics.csv|json` — ScoringBench metrics plus `cov50`/`cov90` and fit cost `wall_s`
- `figures/{problem}_n{N}_metrics_table.png` — same columns (cost shown as `wall_s` seconds)
- `{problem}_n{N}_cost.csv|png|json` — detailed cost breakdown
- `figures/{problem}_n{N}_marginal_x{tag}.png` — one figure per probe \(x\)
- `figures/{problem}_n{N}_posterior_vs_data.png` — data scatter + mixture mean ± band
- `figures/a06_noise_preview.png` — A06 mean + noise (from `--preview-a06`)
- `summary.json` — records `train_sizes`, `n_test`, `test_shared`, `nested_train`

```powershell
python compare_foundation_baselines.py --problems A01,A05 --models all --n-probes 5
```

A01 overlays still under `results/figures/a01_*`.

## Notes

- TabGP vendors its own `gpplus` under `_engine`. The adapter temporarily shadows SpectralGPPlus's `gpplus` and forces `--attn_backend math` (flash-attn is often unavailable on Windows).
- First TabPFN / Kumo call downloads weights (network + disk; Kumo from `nvidia/Kumo-Tabular`).
- When quantile UQ is unavailable, CRPS/NIS use a fallback Gaussian scale from `std(y_train)` — check `extras.uq_from_quantiles` in the JSON.
- A05/A06 also report `k_eff` / `k_modes_*` for MidlineSparse (FM baselines do not invent a mode count).
- **Conditional marginal plots:** TabPFN uses the native bar posterior (`output_type="full"` → borders + logits → piecewise density). **KumoTabular** uses 999 quantiles (`q001`–`q999`) → CDF finite-difference density (`ppd_source=quantile_diff`). ScoringBench CRPS/NIS still use a K=1 Gaussian summary from the 10–90% quantiles for those models.
- CLI alias: `--models tabfm` still resolves to KumoTabular for older scripts.

### TabPFN bar distribution (details that matter for histogram-GP baselines)

Installed `tabpfn>=9.2.0` implements regression as a GMM-free **FullSupportBarDistribution** (`tabpfn.architectures.shared.bar_distribution`):

- Head emits **B = 5000** logits → **softmax** → bar masses (always on the simplex).
- Borders are **not fit to the user dataset**. For v3/v3.5 they come from checkpoint / `_spline_based_regression_borders` (unequal grid ~`[-128,128]` in z-space; historically prior-quantile). Stored as `znorm_space_bardist_`, then affine-rescaled by train mean/std into `raw_space_bardist_`.
- Interior density: \(p(y)=p_k/w_k\). NLL uses `log_softmax - log(width)`.
- Outer bars are **half-normals** (full support on \(\mathbb{R}\)); ICDF sends \(q=0/1\) to \(\pm\infty\).
- Repo helper [`predictive_density.density_from_bar_distribution`](predictive_density.py) is softmax/width only (no half-normal tails) — fine for plots, not exact TabPFN NLL.

**Histogram GP baselines (`hist_mogp`, `hist_gpd`)** copy the bar *interface* (borders + logits, width-scaled density) with **requested B = 200** (effective \(B=\min(B_{\mathrm{req}},n_{\mathrm{train}})\)), **equal-width** bins on the padded training range (quantile optional; ties merged, not ε-pushed), and no half-normal tails. Training default: **1000** Adam steps at lr **0.03**.

`hist_mogp` (exact backend) is a separable Kronecker multi-output GP
\(K_{XX}\otimes K_{HH}+\sigma_n^2 I\) with \(K_{HH}\) an RBF on bin centers; MLL
uses paired eigendecomps. The posterior mean is Euclidean-projected onto the
simplex (optional MC project of samples). Nugget is noise in bin-indicator
space, not aleatoric \(y\)-noise.

| | TabPFN | `hist_mogp` | `hist_gpd` |
|--|--------|-------------|------------|
| Targets | CE / bar NLL on prior grid | one-hot Kronecker MOGP | Dirichlet pseudo-obs (Milios) |
| Simplex | softmax | Euclidean project of posterior mean | softmax / Dirichlet mean |
| B | 5000 | 200 requested (cap \(n\)) | shared `--hist-n-bins` |
| Borders | frozen prior spline | equal-width + pad | same |
| Steps | — | 1000 Adam (lr 0.03) | — |

**Critical assessment / failure modes:** Gaussian likelihood on one-hots is misspecified; mean-only simplex projection is not Bayesian on the simplex; empty equal-width bins remain underdetermined even with \(K_{HH}\) sharing; \(B\sim 10^2\) cannot match TabPFN resolution; no half-normal tails. A prior bug checkpointed post-step NaN hypers and silently yielded **exact uniform** \(\pi\) after simplex projection — training now keeps only finite states. Prefer \(B\ll n\) when scoring densities. If they lose to `exact_gp` on CRPS/`log_score`, that is expected under these limits.

Smoke / full hist-GP suite (writes a dedicated folder):

```powershell
python compare_foundation_baselines.py --problems A01,unimodal `
  --models hist_mogp,exact_gp --train-sizes 100 --n-test 200 --n-probes 5 `
  --hist-n-bins 200 --hist-num-steps 1000 `
  --out-dir results/foundation_compare_hist_kronecker

# Tabular + fish
python compare_foundation_baselines.py --problems TABARENA_CONCRETE,QSAR_FISH `
  --models hist_mogp,exact_gp --hist-n-bins 200 --hist-num-steps 1000 `
  --out-dir results/foundation_compare_hist_kronecker
```

### TabPFN license (required once)

TabPFN 3.x weights are gated. Before `tabpfn3` / `tabpfn35` rows will populate:

1. Create/login at https://ux.priorlabs.ai/account and accept the license.
2. Copy your API key, then either:

```powershell
$env:TABPFN_TOKEN = "<your-api-key>"
# or persist for the user:
# setx TABPFN_TOKEN "<your-api-key>"
```

3. Or run once interactively so the browser login can finish:

```powershell
$env:TABPFN_ALLOW_INTERACTIVE_AUTH = "1"
python -c "from tabpfn import TabPFNRegressor; from tabpfn.constants import ModelVersion; TabPFNRegressor.create_default_for_version(ModelVersion.V3_5).fit([[0.],[1.]],[0.,1.])"
```

Without a token, the compare script records a clear error row instead of hanging on stdin.
