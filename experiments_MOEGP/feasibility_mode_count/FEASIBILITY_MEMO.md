# Feasibility memo: joint mode-count + soft MoE-GP

**Claim under test:** Jointly train a soft mixture of scalable GP experts and a differentiable estimator of the number of modes of \(p(y|x)\), for co-located multimodality, where input-dependent gating (including BNP MoE) is misspecified because mode identity is not a function of \(x\).

**Date:** 2026-10-17  
**Scope:** Investigation only (diagnostics + analysis). Full A+C regularizer **not** wired into `moe_gpr` (see §5 decision).  
**Artifacts:** `diagnostics.py`, `run_probes.py`, `soft_peakness.py`, `results/probe_summary.json`, `results/peakness_addendum.json`

---

## 1. Novelty assertion (Phase 1)

### 1.1 Prior-art matrix

| Line of work | Joint w/ training? | Targets modes vs experts/components? | Co-located (`mode ⊥ x`)? | Scalable GPs? | Covers claim? |
|---|---|---|---|---|---|
| Finite MoE / MDN + AIC/BIC/ICL over \(K\) | No (post-hoc model select) | Components / experts | Partial (if gate is constant) | Sometimes | No — discrete IC, not mode functional |
| Overfitted sparse mixtures / merge–truncate | Sometimes (birth–death) | Components | Rarely discussed as modes | Rare | No — sparsifies experts ≠ modes of \(p(y\|x)\) |
| DP / dependent-DP / KSBP MoE-GP | Yes (BNP) | Expert count via sticks | **Misspecified** when modes co-located | Sometimes | No — \(K_{\mathrm{experts}}\) ≠ \(K_{\mathrm{modes}}\); x-stick wrong for A01 |
| BayesMultiMode-style peak counting | Usually post-hoc on dens. | **Modes** | Yes (on dens.) | N/A / non-GP | No — not co-trained with soft GP MoE |
| Residual GMM → fixed-\(K\) MoE ([`estimate_mode_count.py`](../estimate_mode_count.py)) | Two-stage | Modes (residual) then experts | Yes (by design) | GP optional | Incremental — not differentiable / not joint |
| Deep MoE expert pruning | Yes | Experts (load / routing) | No | No | No — capacity pruning, not statistical modes |

### 1.2 Novelty pass/fail

**One-sentence claim no row fully covers:**  
*A differentiable mode-of-\(p(y|x)\) objective co-trained with soft GP MoE under co-located misspecification of x-gates.*

| Verdict | Reason |
|---|---|
| **Pass (conditional)** | Closest rows either (i) count experts (DP/KSBP), (ii) count modes post-hoc without GP MoE (BayesMultiMode), or (iii) two-stage residual BIC then freeze \(K\) (our baseline). None jointly optimize a mode functional on the mixture density with soft GP experts under explicit co-located gate misspecification. |
| Fail risk | If the regularizer only mimics load-balance / expert diversity, novelty collapses to soft MoE-GP + residual \(K\) selection (incremental). Probes show residual BIC/AIC can under-select (\(k=1\) on A01 in this run), so a density-mode term is not redundant *if* made reliable. |

### 1.3 Closest competitors (3) and differentiation

1. **Kernel stick-breaking / dependent-DP MoE-GP** — Diff: grows **experts** as a function of \(x\); wrong for co-located modes. We target **modes of the mixture density**, allow \(K_{\mathrm{experts}} \ge K_{\mathrm{modes}}\), prefer constant/hybrid gates.
2. **BayesMultiMode / peak counting on densities** — Diff: post-hoc; not an EM/soft-MoE training signal; not paired with scalable GP experts + orthogonality.
3. **Residual-GMM / IC over fixed-\(K\) MoE (our baseline)** — Diff: two-stage, non-differentiable. Joint soft peakness + \(K_{\mathrm{eff}}\) is the novelty lift *only if* training-time surrogates work; offline probes partially support dual reporting, not yet joint training.

---

## 2. Algorithmic variant scores (Phase 2)

| Idea | Diff? | Cost @ \(K_{\max}\) | Co-located | Separated (A02) | Unimodal | Feasibility | Notes |
|---|---|---|---|---|---|---|---|
| **A. Soft peakness on \(y\)-grid of \(p(y\|x)\)** | Soft peaks / 2nd-diff | \(O(K \cdot n_{\mathrm{grid}} \cdot n_x)\) OK 1D | Strong if \(\sigma_k\) controlled | Local ~1 mode/x | →1 | **3–4** | Raw softpeakness separates multi vs uni; hard peaks merge when \(\sigma\) large |
| **B. Residual-axis multimodality** | Score on residuals | Cheap | Strong in principle | Global residual can look bi when regimes are x-split | →1 | **4** diagnostic | BIC under-selected on A01 in cheap probe; keep as feature not sole objective |
| **C. Differentiable \(K_{\mathrm{eff}}=\sum a_k\)** | Yes if usage × separation | Negligible | Helps dual report | May under-activate | Should shrink | **4** | Must use mean-separation, not mass alone |
| **D. Birth–death / prune** | Discrete | High (refits) | Possible | Possible | Possible | **2** | Poor fit with GPTrainer refit-each-EM; defer |
| **E. Joint vs two-stage** | Joint if A/C online | +overhead | Joint needed when peaks merge mid-EM | Two-stage OK | Two-stage fine | **3** | Joint not yet validated; two-stage residual-\(K\) is the safe paper |

**Priority path remains A+C**, B diagnostic, D deferred — but implementation gate fails (see §5).

### Expected vs observed behavior

| Setting | Expected A/C | Observed (cheap soft MoE) |
|---|---|---|
| (i) A01 co-located | \(K_{\mathrm{modes}}\approx 2\) | Hard peaks: **1** at \(K_{\max}=2,3\); **~2.2** at \(K_{\max}=4\). Softpeakness A01 ≫ unimodal. Active experts track \(K_{\max}\), not modes. |
| (ii) A02 step | ~1 mode/x with MLP; constant invents modes | **Confirmed:** MLP \(K_{\mathrm{modes}}\approx 1.17\); constant \(\approx 2.83\). NLL: constant better in this under-trained soft probe (inconclusive for likelihood). |
| (iii) Unimodal | \(K_{\mathrm{modes}}\approx 1\), experts may overfit | **Confirmed:** \(K_{\mathrm{modes}}=1\), \(K_{\mathrm{active}}=4\), expert corr \(\approx 1.0\). |

---

## 3. Empirical probes (Phase 3)

Settings: soft MoE, \(n{=}400\) (A01) / 300 (uni) / 200 (A02), `num_em_iters=3`, `num_inits=2`, exact GPR experts. Runner: `run_probes.py`.

### 3.1 A01 co-located (true \(K_{\mathrm{modes}}=2\))

| Gate | \(K_{\max}\) | \(K_{\mathrm{modes}}\) (mean) | \(K_{\mathrm{active}}\) | \(\lvert\mathrm{corr}\rvert\) | Test mix NLL | Wall (s) |
|---|---|---|---|---|---|---|
| constant | 2 | **1.0** | 2 | 0.67 | 3.195 | 6.4 |
| constant | 3 | **1.0** | 3 | 0.42 | 3.591 | 7.7 |
| constant | 4 | **2.22** | 4 | 0.56 | 3.606 | 10.3 |
| mlp | 2 | 1.0 | 2 | 0.56 | 3.187 | 5.7 |
| mlp | 3 | 1.0 | 3 | 0.47 | 3.119 | 7.8 |
| mlp | 4 | 2.11 | 4 | 0.56 | 6.056 | 10.7 |

- Residual-GMM BIC/AIC on linear residuals: **\(k_{\mathrm{hat}}=1\)** (under-select vs truth 2).  
- BIC proxy over fixed-\(K\) soft MoE prefers \(K_{\max}=2\) (best NLL), even when peaks are merged.  
- **Pass criterion 1:** at \(K_{\max}=4\), post-hoc \(K_{\mathrm{modes}}\approx 2\) beats \(K_{\mathrm{active}}=4\).

### 3.2 Unimodal control (true \(K_{\mathrm{modes}}=1\), \(K_{\max}=4\))

| \(K_{\mathrm{modes}}\) | \(K_{\mathrm{active}}\) | Residual \(k_{\mathrm{hat}}\) | Expert \(\lvert\mathrm{corr}\rvert\) |
|---|---|---|---|
| **1.0** | 4 | 1 | **0.9999** |

Clear dual-reporting win: experts overfit; modes stay 1.

### 3.3 A02 separated step (true local modes = 1; 2 regimes)

| Gate | \(K_{\mathrm{modes}}\) mean | Test NLL | Interpretation |
|---|---|---|---|
| mlp | **1.17** | 2.40 | Locally unimodal (correct structure) |
| constant | **2.83** | **2.02** | Spurious co-located-looking modes |

**Misspecification claim (mode structure):** supported — constant gate invents modes on separated data; MLP keeps local unimodality.  
**NLL contrast:** *not* supported in this cheap soft probe (constant wins likelihood). Prior hard-assignment A02 MLP results remain the stronger likelihood evidence.

### 3.4 Soft peakness surrogate (addendum)

| Setting | Soft peakness mean | Hard \(K_{\mathrm{modes}}\) |
|---|---|---|
| A01 \(K{=}2\) | **14.8** | 1 (merged) |
| A01 \(K{=}4\) | 8.0 | 2.2 |
| Unimodal \(K{=}4\) | **2.0** | 1 |

Soft peakness distinguishes multimodal vs unimodal even when hard peak count fails at \(K{=}2\). It does **not** monotonically track hard peak count (merged tall peak scores high). A usable training surrogate needs an **expected soft peak count**, not raw concavity mass alone.

### 3.5 Phase-3 pass criteria

| Criterion | Result |
|---|---|
| Post-hoc \(K_{\mathrm{modes}}\) better than active experts on overfit | **PASS** (\(K_{\max}=4\)) |
| x-gate fails A01 / helps A02 | **PARTIAL** — mode structure supports A02 contrast; A01 NLL MLP≈constant; A02 NLL favors constant in this run |
| Stable surrogate correlates with true \(K_{\mathrm{modes}}\) | **PARTIAL** — softpeakness A01≫uni; residual BIC failed; no during-EM training curve yet |
| **Overall (strict plan gate)** | **FAIL** |

---

## 4. Limitations catalog (Phase 4)

| Weakness | Observed / expected | Mitigation / v1 scope |
|---|---|---|
| Peak counting ≠ component counting; \(\sigma_k\) inflation merges peaks | A01 \(K{=}2,3\): hard \(K_{\mathrm{modes}}=1\) despite 2+ active experts | Cap expert noise; persistence / mean-separation in peakness; dual-report always |
| Soft EM + GP refits: expensive; no E2E grads through expert hypers | Wall ~6–11s even with 2 inits / 3 EM | Mode term on frozen experts each E-step; RFF/SVGP before scalability claims |
| Orthogonality / residual seeding can dominate | High corr on uni; orth blend often ~1 | Ablate `orth_weight`; require mode metrics with orth low |
| Constant gate cannot express separated regimes | A02 constant invents ~3 modes | Hybrid gate; detect residual multimodality vs x-partition |
| Scalability claim needs RFF/SVGP | Only exact GPR probed | Repeat A01/A02 with SpectralGPPlus backends |
| Regularizer may mimic load-balance | Risk if \(a_k\) = usage only | \(a_k\) from usage × mean separation / peak contribution; control vs load-balance-only |
| Residual IC under-selects | BIC/AIC \(k{=}1\) on A01 here | Prefer AIC + linear residuals; do not rely on residual alone for novelty |
| Soft peakness ≠ soft peak **count** | Raw score high on merged unimodal-looking peak | Redesign surrogate as expected number of soft maxima |
| Post-hoc ≠ joint training | No EM trajectory of mode loss yet | Block GO on joint A+C until stub shows during-EM correlation |

---

## 5. Push ranking and go/no-go (Phase 5)

### Ranked push directions (novelty × feasibility, post-probe)

1. **Dual reporting \(K_{\mathrm{experts}}\) vs \(K_{\mathrm{modes}}\)** with controlled mismatch (uni + overfit \(K_{\max}\)) — **highest immediate value**; already evidenced
2. **Soft MoE-GP for co-located multimodality + residual/\(K\) selection** — honest incremental paper if joint mode loss stalls
3. **Differentiable mode functional (A)** — only after redesign to soft **count** + \(\sigma\) control; then co-train
4. **\(K_{\mathrm{eff}}\) activity (C)** with separation term — cheap companion to (1)/(3)
5. **Hybrid gate** — motivated by A02 mode-structure contrast
6. **Scalable expert backends** under same diagnostics
7. **Theory sketch** — after empirical story stabilizes

### Decision gate (plan rule)

> Implement A+C joint regularizer in `moe_gpr` soft loop **only if** Phase 3 pass criteria hold.

**Decision: NO-GO** on implementing A+C inside `moe_gpr.fit` at this time.

**Reasons:**

- Strict Phase-3 overall pass = false (gate NLL contrast + no during-EM surrogate validation).  
- Hard peak count fails on the primary well-specified case \(K_{\max}=2\) (peak merge).  
- Novelty risk: wiring a broken peakness term would look like load-balance / orth in disguise.

**Allowed next steps (without violating the gate):**

1. Keep **offline** diagnostics (`diagnostics.py`, `soft_peakness.py`) and dual-reporting in experiment scripts.
2. Pivot paper story toward **soft MoE-GP + dual \(K_{\mathrm{experts}}/K_{\mathrm{modes}}\) reporting + residual \(K\)**, with explicit co-located vs separated gate misspecification (A01/A02).
3. Re-open A+C **only after** a second probe shows: (a) hard or soft **peak count** recovers 2 at \(K_{\max}=2\) under noise caps / orth ablations, (b) soft peak-count surrogate tracks truth across A01 / uni during EM, (c) A02 MLP likelihood win under sharp temperature / hard or longer soft training.

---

## 6. MidlineSparseMoE-GP: mechanism and positioning (pivot)

After the A+C NO-GO on wiring mode-count into `moe_gpr`, the working model is **MidlineSparseMoE-GP** (`gpplus/models/midline_sparse_moe.py`).

### 6.1 What the model does

Two-stage procedure:

1. **Stage-1:** fit one exact GPR midline \(\mu_0(x),\sigma_0(x)\).
2. **Stage-2:** learn overcomplete additive experts
   \(\mu_k=\mu_0+\alpha_k\delta_k(x)\) (MLP or constant offsets), mixture weights \(\pi\) (constant or MLP gate), Hard-Concrete switches \(\alpha\), and multiplicative variance carve-out
   \(\mathrm{Var}_k=\sigma_0^2(1-\alpha_k\gamma_k)+\varepsilon\).

Stage-2 loss = mixture NLL + usage L0 \(\sum\tilde\alpha_k\) + usage-weighted separation + HC complexity + orphan-\(\alpha\) penalty + alive floor. No mixture-mean MSE.

Effective expert count (Option E):
\(\tilde\alpha_k=\sigma((\alpha_k\bar\pi_k-\varepsilon)/\tau)\), \(K_{\mathrm{eff}}=\sum_k\tilde\alpha_k s_k\).
Always dual-report \(K_{\mathrm{eff}}/K_{\mathrm{hat}}\) (experts) vs \(K_{\mathrm{modes}}\) (peaks of \(p(y|x)\)).

**Target problem:** co-located multimodality (mode \(\perp x\)), where x-dependent gating is misspecified; unknown \(K\) via continuous sparsity rather than birth–death or outer IC loops.

### 6.2 Where it can beat nearby SOTA (honest)

| Competitor | What they optimize | MidlineSparse edge | When they still win |
|---|---|---|---|
| Soft/hard MoE-GP or MDN + AIC/BIC over \(K\) | Full experts; discrete \(K\) | Shared midline + sparse offsets; continuous \(K\); dual modes vs experts | Well-specified x-partition; need full expert GPs |
| DP / dependent-DP / KSBP MoE-GP | Expert count via sticks of \(x\) | Constant gate fits co-located modes | Separated regimes where x-sticks are correct |
| Residual-GMM / IC → freeze \(K\) then MoE | Two-stage, non-diff \(K\) | Joint Stage-2 sparsity + NLL; differentiable \(K_{\mathrm{eff}}\) | Residual BIC already recovers \(K\) |
| BayesMultiMode / post-hoc peak count | Modes on a fitted density | Modes co-reported with GP mixture + variance control | Pure density mode counting |
| Deep MoE pruning / load-balance | Capacity / routing | Statistical mixture likelihood + geometric \(s_k\) | Large-scale LLM routing |

**Claimable when Gate-2 tight criteria hold:** better test NLL than naive soft MoE on co-located A01 (historically ~2.7 vs ~3.1) with automatic sparse \(K\) and explicit experts≠modes reporting.

**Not claimable yet:** scalable GP backends (still exact GPR); that \(K_{\mathrm{eff}}=K_{\mathrm{modes}}\); superiority on separated A02 NLL (constant often wins likelihood).

### 6.3 Reproduce MidlineSparse probes

```bash
cd experiments_MOEGP/feasibility_mode_count
python run_midline_sparse.py --mode gate2
python run_midline_sparse.py --mode charts
# artifacts: results/midline_sparse_gate2.json, results/figures/
```

---

## 7. Pathways to distinct components (anti fat-blob)

A04 (co-located trimodal, unequal \(\sigma\)) often collapses two wide arms into one fat mixture component. Target remains **modes of \(p(y|x)\)** at a probe \(x\), not expert count.

| ID | Pathway | Status |
|---|---|---|
| P1 | Free per-expert scale \(\beta_k\ge 1\): \(\mathrm{Var}_k=\sigma_0^2(1-\alpha_k\gamma_k)\beta_k+\varepsilon\) | Explored; **inert** on A04 — **muted** in champion (`free_expert_scale=False`; see §7.1) |
| P2 | Relative mean-gap / overlap penalty on active pairs | Present (`lambda_overlap`); keep |
| P3 | Better \(\delta_k\) init (σ-aware multi-arm seeds) | Partial (residual quantiles exist); strengthen next |
| **P4** | Anneal usage L0 (\(\lambda_k\) low→high) | **Implemented** (`lambda_k_start`→`lambda_k`; A04 uses \(0\to 0.02\)) |
| P5 | Peak-count / true-\(K\) losses | **No-go** for this failure mode |
| P6 | Per-expert GPs / free MDN | Only if P1–P4 fail |

### 7.1 Why \(\beta\approx 1\) (P1 did not unlock widths)

With `use_full_gp_variance=True`, Stage-2 uses full GP \(\sigma_0\), which is already large on multimodal residuals. Experts can already widen via \(\gamma\to 0\) without \(\beta>1\). Multi-seed A04 (`a04_p1_free_scale.json`) ended at \(\beta\in[1.00,1.02]\) from init \(\approx 1.13\) — NLL + \(\lambda_\beta\|\beta-1\|^2\) **pulled toward 1**. Fat-blob failure is **mean merge + early sparsity**, not a missing variance ceiling.

Do **not** invest further in free-scale machinery. Champion / production settings mute \(\beta\) (`free_expert_scale=False`). **Production default is lean post-factorial:** capped \(\sigma_0\) + usage + adaptive_merge + alive + delta + alpha; `sep`/`rec`/`overlap`/`gamma`/`orphan` off (`fit_and_report` / `_CHAMPION` / `foundation_adapters`).

**Ablation** (`a04_beta_prior_ablation.json`): even with \(\lambda_\beta=0\), \(\max\beta\approx 1.03\) — NLL does not push free scale. Recommend P4 annealing instead.

### 7.2 Do not learn the loss lambdas

\(\lambda_k,\lambda_{\mathrm{sep}},\lambda_\beta,\lambda_{\mathrm{overlap}},\ldots\) encode inductive bias. Jointly learning them with model params is underdetermined (NLL can drive \(\lambda\to 0\) and recreate soft-MoE collapse). Prefer **fixed grids and schedules** (anneal \(\lambda_k\)), not meta-learned weights. Keep learning \(\alpha,\pi,\delta,\gamma\) (and \(\beta\) only if the prior is weak and representation actually needs it).

**P1 acceptance (soft):** multi-seed A04 mean \(K_{\mathrm{modes}}\ge 2.0\); some seed with peaks \(\ge 2\) at both \(x=1\) and \(x=5\); Gate-2 A01/uni bands do not regress.

```bash
python run_midline_sparse.py --mode a04_p1
# → results/a04_p1_free_scale.json
python run_midline_sparse.py --mode a04_beta_ablation
# → results/a04_beta_prior_ablation.json
```

---

## 8. Recovery theory and algorithmic enforcement

**Primary claim target:** \(K_{\mathrm{modes}}\) of \(p(y|x)\) at co-located probes — not expert count.

### 8.1 Theory (what is guaranteed vs conjectured)

At fixed \(x\), MidlineSparse is a finite 1D Gaussian mixture. Classical facts:

- Univariate GM with \(k\) components has **at most \(k\) modes**.
- Equal-\(\sigma\), equal-weight pair is unimodal iff \(|\mu_1-\mu_2|\le 2\sigma\) (Eisenberger); Behboodian: \(|\mu_1-\mu_2|\le 2\min(\sigma_i,\sigma_j)\) suffices for unimodality.
- **A04** unequal \(\sigma\): mode merge is expected unless means separate vs the large \(\sigma\).
- Parameter identifiability under overlap **does not** imply \(K_{\mathrm{modes}}=M^\star\).

**Conjecture template:** if true components are \(c\)-separated in standardized mean-gap (\(c\ge 1\)) and Stage-2 matches them in \(L^2\) up to labels, then \(K_{\mathrm{modes}}=M^\star\). MidlineSparse has **no proof** of the fit-consistency step.

Full write-up: `docs/midline_sparse_pipeline.tex` § Recovery theory.

### 8.2 Algorithmic enforcement

| Mechanism | Role |
|---|---|
| Champion **p12+rec** (cap \(\sigma_0\) + \(\lambda_\gamma\) + \(\lambda_{\mathrm{rec}}\)) | Default production path; keep variances out of fat-blob regime and push Behboodian mean gaps |
| `lambda_rec` / `rec_margin` | Softplus penalty when \(g_{ij}<m_{\mathrm{rec}}\) (Behboodian-scale default \(1\); \(\lambda_{\mathrm{rec}}=0.15\)) |
| Min-spacing residual init | Seed \(\delta\) biases ≥ \(m_{\mathrm{rec}}\cdot 2\sigma_{\mathrm{cap}}\) apart |
| `recovery_certificate` | Post-fit: fitted gaps ≥ margin? (not vs truth) |

`recovery_probe` ablates rec on/off (`champion_p12` vs `champion_p12_rec`) on **A01 / A04 / A05**.

```bash
python run_midline_sparse.py --mode recovery_probe
# → results/recovery_probe/metrics.json + figures/
```

### 8.3 A05 — unimodal skewed contaminated heteroscedastic

Loader: `experiments_MOEGP/load_unimodal_hetero_data.py`.

- Mean: \(\mu(x)=0.5x+2\) on \(x\in[0,10]\) (true \(K_{\mathrm{modes}}=1\)).
- Noise (zero-mean → still unimodal): nonlinear \(\sigma(x)\propto (x/10)^{1.5}\); asymmetric Laplace with x-dependent skew \(\tau(x)\); \(\varepsilon\approx 0.12\) wide Gaussian contamination (\(\kappa\approx 4\sigma(x)\)).
- Role: harder negative control than homoscedastic Gaussian unimodal — champion must not invent modes under misspecified heavy/skewed/hetero noise.
- Wired into smoke, gate2 (soft `a05_no_invented_modes`), limitation_stress, recovery_probe, and charts.

### 8.4 Likelihood-driven merge / prune (`adaptive_merge_prune`)

Goal: stable correct \(p(y|x)\) for **any** \(K_{\max}\) without targeting mode/expert counts (no residual-BIC \(K_r\), no oracle \(K^\star\), no problem-name schedules).

| Mechanism | Role |
|---|---|
| Online NLL merge probe | Compare current mixture vs \(\delta=0\) midline merge; EMA `merge_pressure` fades \(\lambda_{\mathrm{rec}},\lambda_{\mathrm{sep}},\lambda_{\mathrm{ov}},\lambda_\gamma\) when merge is competitive |
| \(K_{\max}\)-scaled \(\lambda_k\) | Prune unused capacity only (`\(\lambda_k\cdot\max(1,K_{\max}/4)\)`) — model size, not data \(K\) |
| Alive floor + init veto | `softplus(0.9-Σũ)`; reject inits with `k_hat=0`, tiny \(k_{\mathrm{eff}}\), or NLL ≫ Stage-1 GP (fixes A02 \(K_{\max}=4,5\) collapse) |
| Gate selection | Default MLP; multi-init alternates constant/MLP and picks by train NLL |

**Protocol (aligned with foundation compare):** \(n=1000\), `test_size=0.2`, seed 42; each problem uses its `load_problem` **champion** flag (A01 classic / others p12+rec). Ablates only `adaptive_merge_prune` on/off.

```bash
python run_midline_sparse.py --mode merge_prune_probe
# → results/merge_prune_probe/metrics.json + figures/ + metrics.csv
# charts only (reuse saved JSON):
python run_midline_sparse.py --mode merge_prune_charts
```

Artifacts under `results/merge_prune_probe/`:
- `metrics.csv` — full spreadsheet of off vs on × \(K_{\max}\)
- `figures/*_metrics_vs_kmax.png` — per-problem RMSE/CRPS/log_score/\(K_{\mathrm{eff}}\)/\(K_{\mathrm{modes}}\)/merge_pressure
- `figures/*_metrics_table.png` — per-problem numeric table
- `figures/summary_off_vs_on.png` — cross-problem bars
- `figures/acceptance_table.png` — pass/fail checklist

---

### 8.5 Beat-foundation UQ diagnostics

UQ-first comparison of MidlineSparse Stage-1 / Stage-2 against Exact GP and TabPFN-3.5 (protocol aligned with foundation compare: `n=1000`, `test_size=0.2`, seed 42).

**Primary metrics:** CRPS, NIS, cov50/cov90, PIT. `log_score` is secondary only.

```bash
python run_beat_foundation_uq.py
# → results/beat_foundation_uq/
```

#### Stage-1 vs Stage-2 vs FM (CRPS / cov90)

| Problem | Stage-1 | Stage-2 | ExactGP | TabPFN | Verdict |
|---|---|---|---|---|---|
| A01 | CRPS 3.2397 / cov90 0.87 | CRPS 2.8592 / cov90 0.84 | CRPS 3.2397 / cov90 0.87 | CRPS 3.0352 / cov90 0.95 | A01: Stage-2 UQ competitive with FMs (coverage OK); leave champion λs; do not chase log |
| unimodal | CRPS 0.1805 / cov90 0.88 | CRPS 0.2023 / cov90 0.88 | CRPS 0.1805 / cov90 0.88 | CRPS 0.1792 / cov90 0.91 | unimodal: Stage-2 worsens CRPS vs Stage-1≈ExactGP → loss / anti-merge (γ/rec/sep) hurting UQ |
| A04 | CRPS 3.7024 / cov90 0.93 | CRPS 3.6386 / cov90 0.91 | CRPS 3.7024 / cov90 0.93 | CRPS 3.6797 / cov90 0.95 | A04: Stage-2 UQ competitive with FMs (coverage OK); leave champion λs; do not chase log |
| A05 | CRPS 1.0153 / cov90 0.91 | CRPS 1.6295 / cov90 0.98 | CRPS 1.0153 / cov90 0.91 | CRPS 0.9021 / cov90 0.84 | A05: Stage-2 worsens CRPS vs Stage-1≈ExactGP → loss / anti-merge (γ/rec/sep) hurting UQ |

Full numeric tables (CRPS, NIS, cov50/90, RMSE, PIT, log footnote) live in `results/beat_foundation_uq/*_stage_metrics.csv`.

#### Loss stress (unimodal, A05) — CRPS / cov90

| Problem | champion | nll_only | force_merge | kmax1 | ExactGP | TabPFN | Verdict |
|---|---|---|---|---|---|---|---|
| unimodal | 0.2023 / 0.88 | 0.1887 / 0.89 | 0.2042 / 0.89 | 0.2190 / 0.31 | 0.1805 / 0.88 | 0.1792 / 0.91 | unimodal: nll_only improves CRPS vs champion (still lags FM) → weaken rec/γ when merge signal is low |
| A05 | 1.6295 / 0.98 | 1.0759 / 0.94 | 1.3449 / 0.93 | 0.9964 / 0.67 | 1.0153 / 0.91 | 0.9021 / 0.84 | A05: kmax1 improves CRPS vs champion (still lags FM) → weaken rec/γ when merge signal is low |

Artifacts: `results/beat_foundation_uq/*_stage_*`, `*_loss_stress_*`, PIT/coverage PNGs, expert-annotated marginals.

### 8.6 Stage-2 training curves / step budget

Stage-2 is Adam over `num_steps` (not GPTrainer epochs). Probe:

```bash
python run_stage2_training_curves.py --problems unimodal,A05,A01 --budgets 400,800,1600,2400
# → results/stage2_training_curves/
```

Held-out CRPS vs budget (n=1000 protocol): unimodal best at **800**; A01 nearly flat after 800; **A05 improves through 2400** (1.63 → 1.36). Train NLL often oscillates after ~600–800; do not use train NLL alone to pick the budget.

## Appendix: how to reproduce

```bash
cd experiments_MOEGP/feasibility_mode_count
python run_probes.py
# optional peakness addendum was run inline; results in results/*.json
python run_midline_sparse.py --mode smoke   # includes A05
```

Primary stress: A01; contrast: A02; negative controls: unimodal (Gaussian) + **A05** (skew/contam hetero). Claim language stays **modes of \(p(y|x)\)** + **co-located misspecification of x-gates**.
