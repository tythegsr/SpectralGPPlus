# MidlineSparse vs MDN and joint predictive upgrade

Advisor-facing positioning for the champion two-stage MidlineSparse model
([`gpplus/models/midline_sparse_moe.py`](../../gpplus/models/midline_sparse_moe.py)).
Champion code stays frozen; joint upgrades live in sibling modules.

---

## 1. Verdict

MidlineSparse returns the **same kind of probabilistic object as Bishop’s Mixture
Density Network (MDN, 1994)**: a plug-in / estimative conditional MoG
\(\hat p(y\mid x)=p_{\hat\theta}(y\mid x)\). It is **not** a trivial MDN: means are
residual offsets pinned to a frozen Exact-GP midline, variances carve out the GP
envelope \(\sigma_0^2\), and capacity is Hard-Concrete–sparse with
usage/alive/merge controllers.

It is **not** a mixture of Gaussian processes (MoGP): Stage~2 has no expert kernels
and does not define process-level finite-dimensional distributions (FDDs)
\(p(y(x_1),\ldots,y(x_n))\).

---

## 2. MidlineSparse vs Bishop MDN (1994)

| Axis | Bishop MDN | MidlineSparse (champion) |
|------|------------|---------------------------|
| Predictive object | Plug-in MoG \(p_\theta(y\mid x)\) | Same *kind*: estimative MoG on frozen \((\mu_0,\sigma_0)\) |
| Means | Free NN \(\mu_k(x)\) | \(\mu_k=\mu_0+\alpha_k\tilde\delta_k\), \(\pi\alpha\)-pinned to midline |
| Scales | Free NN \(\sigma_k(x)\) | Carve-out \(\sigma_0^2(1-\alpha_k\gamma_k)\) |
| GP role | None | Stage~1 Exact GP midline / envelope (frozen) |
| Capacity | Dense \(K\) | Hard-Concrete \(\alpha\) + usage/alive/merge |
| Gate | Softmax NN | Constant or MLP (co-located modes) |
| Training | End-to-end mixture NLL | Two-stage: GP ML then residual MoG NLL + penalties |
| Multi-input joints | No expert Gram FDDs | No (coupled only via \(\mu_0\) / MLP weights) |
| Bayes PPD | No (unless params marginalized) | No (EBMAP / controller-modified estimative) |

**Justifiable claims**

- “GP midline + sparse residual mixture of Gaussians”
- “Two-stage MoE density on a frozen GP envelope”
- “Valid estimative conditional density for multimodal UQ”

**Unjustifiable claims**

- “Mixture of GPs / MoGP”
- “Each expert is a GP”
- “Bayesian posterior predictive \(p(y\mid x,\mathcal{D})\) under a joint hierarchy”
- “Not related to MDN” (the returned object *is* MDN-class CDE)

**Bottom line:** sparse residual MDN / MoG CDE on a frozen Exact-GP midline — not
Bishop MDN with a GP logo, and not a process model.

---

## 3. Closest related methods

1. **Bishop MDN / classical MoG CDE** — same predictive family; MidlineSparse adds
   GP envelope, residual gauge, sparsity controllers.
2. **MiDGaP** — GP priors over MDN latent mixture parameters; still primarily
   conditional at each \(x\), closer to Bayesian MDN than two-stage freeze.
3. **GP-CDE (Dutordoir et al.)** — GP map from \((x,\text{latent})\to y\); Bayesian
   CDE, not MoGP experts.
4. **Tresp MoGP / Rasmussen–Ghahramani infinite MoE-GP** — expert process priors
   and Gram-mediated multi-point joints. MidlineSparse Stage~2 is **not** this
   (see foundations Prop. `prop:ms-no-expert-kernel`).
5. **Soft MoE-GP + residual \(K\) selection** — earlier line in
   [`FEASIBILITY_MEMO.md`](FEASIBILITY_MEMO.md); MidlineSparse is the sparse-offset
   specialization.

Novelty that survives an MDN objection: midline gauge + variance carve-out +
Hard-Concrete / merge controllers + dual experts≠modes under co-located
multimodality — **not** “we invented mixture conditionals.”

---

## 4. Two upgrade axes (conditional vs joint)

These are independent; conflating them causes the “MDN vs joint” confusion.

### Axis A — Plug-in vs Bayesian posterior predictive (still \(y\mid x\))

| | Plug-in (champion) | Bayesian PPD |
|--|-------------------|--------------|
| Object | \(p_{\mu_0,\hat\theta}(y\mid x)\) | \(\iint p(y\mid x,f_0,\theta)\,p(f_0,\theta\mid\mathcal{D})\) |
| Status | Valid conditional density | Requires joint likelihood + posterior approx + MC predict |
| Still conditional? | Yes | Yes — still \(p(y\mid x,\mathcal{D})\), not multi-input FDDs |

Foundations roadmap: §`sec:code-to-posterior` in
[`docs/midline_sparse_probabilistic_foundations.tex`](docs/midline_sparse_probabilistic_foundations.tex).

### Axis B — One-point conditionals vs process-level joints

| | Champion MoG | Residual MoGP |
|--|--------------|---------------|
| Object | Family \(\{p(y\mid x)\}_x\) | FDDs \(p(y(X_\star)\mid\ldots)\) via expert kernels |
| Cross-input dependence | Shared \(\mu_0\) / MLP weights only | Expert Gram matrices (mixture of MVNs) |
| MDN kinship | High | Leaves MDN class for MoGP class |

**Default joint upgrade = Axis B** (residual MoGP). Axis A is the Bayesianization
of whichever generative model is chosen.

---

## 5. Joint upgrade file layout (champion unchanged)

| Artifact | Path | Role |
|----------|------|------|
| Champion | [`gpplus/models/midline_sparse_moe.py`](../../gpplus/models/midline_sparse_moe.py) | Estimative MoG conditionals (**do not rewrite**) |
| Phase 1 | [`gpplus/models/midline_residual_mogp.py`](../../gpplus/models/midline_residual_mogp.py) | Two-stage estimative residual MoGP + `predict_joint` / `sample` |
| Phase 2 | [`gpplus/models/midline_bayes_mogp.py`](../../gpplus/models/midline_bayes_mogp.py) | Bayesian joints (VI/Laplace/MC PPD); separate module |

### Phase 1 generative sketch

\[
f_0\sim\mathrm{GP},\quad
\delta_k\sim\mathrm{GP}(0,k_{\psi_k}),\quad
z\sim\mathrm{Cat}(\pi),\quad
y\mid x,z=k\sim\mathcal{N}(f_0(x)+\alpha_k\delta_k(x),v_k).
\]

Stage~1 midline frozen as in champion; residual GPRs under soft responsibilities;
Hard-Concrete \(\alpha\) and \(\pi\alpha\)-pinning retained. Still EBMAP / estimative
FDDs — not full Bayes.

### Phase 2 sketch

One MoG/MoGP likelihood; do not Dirac-freeze \(f_0\); approximate
\(p(f_0,\theta\mid\mathcal{D})\); MC posterior predictive. Controllers
(usage/alive/merge) are diagnostics only, not the Bayesian objective.

---

## 6. Empirical MDN baseline (run 2026-11-02)

Matched Bishop-style MDN (and constant-gate `mdn_const`) is wired into
[`foundation_adapters.py`](foundation_adapters.py) / [`compare_foundation_baselines.py`](compare_foundation_baselines.py).
Artifacts: `results/foundation_compare_mdn/`. Full writeup:
[`FOUNDATION_COMPARE.md`](FOUNDATION_COMPARE.md) § MidlineSparse vs Bishop MDN.

```powershell
conda activate gpplus2026
cd experiments_MOEGP/feasibility_mode_count
python compare_foundation_baselines.py --problems all --models midline,mdn,mdn_const
```

**Observed (24 settings):** MidlineSparse wins 10/24 on `log_score` and 12/24 on CRPS
vs best-of(`mdn`,`mdn_const`). Clear MidlineSparse wins on **A01** (all \(n\)) and
strong NLL on **A04**; free MDN wins on **A02** (x-partitioned) and often **A06**.

| Outcome | Reading |
|---------|---------|
| MidlineSparse ≫ MDN on A01 small-\(n\) NLL | Envelope + residual gauge helps co-located multimodality |
| MDN ≫ MidlineSparse on A02 | Shared midline bias hurts when modes are x-separated |
| Split overall | Same CDE class; inductive bias—not object type—drives gaps |
