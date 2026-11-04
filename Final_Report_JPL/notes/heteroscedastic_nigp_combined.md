# Heteroscedastic Gaussian Likelihoods with Noisy-Input GPs

**Audience:** anyone designing the next EMIT/S4 noise model (free output hetero + NIGP).

**Companion Overleaf file:** `heteroscedastic_nigp_combined.tex`  
**Bibliography:** [`ref.bib`](../ref.bib) at the `Final_Report_JPL/` project root (hetero keys are already mirrored there from `hetero_nigp.bib`).  
Compile: set project root to `Final_Report_JPL/`, then pdfLaTeX → BibTeX → pdfLaTeX ×2.  
For a self-contained upload of `notes/` only, switch the tex to `\bibliography{hetero_nigp}` and upload `hetero_nigp.bib`.

---

## Notation quick-reference

| Role | Symbol | Not to be confused with |
|------|--------|-------------------------|
| Constant output noise (homo) | \(\sigma_y^2\) | — |
| Free hetero output noise | \(\sigma_{y,\mathrm{het}}^2(x)\) | \(\sigma_y^2\), \(\sigma_f^2\) |
| Input noise (per coord.) | \(\sigma_{x,k}^2\), \(\Sigma_x\) | output noises |
| NIGP term | \(\sigma_{\mathrm{nigp}}^2(x)\) | \(\sigma_{x,k}^2\) alone |
| Woodbury diagonal | \(\sigma_{\mathrm{eff}}^2(x)=\sigma_{y,\mathrm{het}}^2(x)+\sigma_{\mathrm{nigp}}^2(x)\) | latent \(\sigma_f^2\) |
| Latent QoI posterior var. | \(\sigma_f^2(x)\) | observation \(\sigma_{y\mid\mathcal D}^2\) |
| Predictive obs. var. of \(y\) | \(\sigma_{y\mid\mathcal D}^2(x)=\sigma_f^2(x)+\sigma_{\mathrm{eff}}^2(x)\) | \(\sigma_y^2\) |
| Mean SORF features | \(D_f\), \(\phi_f\), \(\Phi_f\), \(w_f\) | hetero counterparts |
| Noise SORF features | \(D_{\mathrm{het}}\), \(\phi_{\mathrm{het}}\), \(w_{\mathrm{het}}\) | mean SORF |

---

## Bottom line

SpectralGPPlus already has **homoscedastic** Gaussian likelihoods and **NIGP**, which turns per-band input noise \(\sigma_{x,k}^2\) into *derived* per-point observation noise \(\sigma_{\mathrm{eff}}^2\). That is not a free heteroscedastic likelihood: all of \(\sigma_{\mathrm{eff}}^2-\sigma_y^2\) is forced through \(\nabla_x\mu_f\) and \(\Sigma_x\).

**Recommendation:** Additive Residual Hetero-NIGP

\[
\sigma_{\mathrm{eff}}^2(x_i)
= \sigma_{y,\mathrm{het}}^2(x_i) + \sigma_{\mathrm{nigp}}^2(x_i)
\]

- \(\sigma_{y,\mathrm{het}}^2(x)\): free input-dependent **output** noise (RT misspec, scene-dependent residual), learned e.g. by a Noise SORF.
- \(\sigma_{\mathrm{nigp}}^2(x)=(\nabla_x\mu_f)^\top\Sigma_x\nabla_x\mu_f\): **input** uncertainty.
- Same diagonal-noise Woodbury path: \(C_y=\mathrm{diag}(\sigma_{\mathrm{eff}}^2)+\Phi_f\Phi_f^\top\).

---

## What we already do

### Homoscedastic GP + SORF

\[
y_i = f(x_i)+\varepsilon_i,\quad \varepsilon_i\sim\mathcal N(0,\sigma_y^2)
\]

with mean SORF features \(\Phi_f\) and Woodbury \(C_y=\sigma_y^2 I+\Phi_f\Phi_f^\top\) (`LogGaussianLikelihood`).

### NIGP (derived heteroscedasticity)

McHutchon & Rasmussen (NeurIPS 2011):

\[
\sigma_{\mathrm{eff}}^2(x)=\sigma_y^2+\sigma_{\mathrm{nigp}}^2(x),\qquad
\sigma_{\mathrm{nigp}}^2(x)=(\nabla_x\mu_f)^{\top}\Sigma_x\nabla_x\mu_f
\]

with \(\Sigma_x=\mathrm{diag}(\sigma_{x,1}^2,\ldots,\sigma_{x,D_x}^2)\). Slopes come from a **homoskedastic** dual solve, then are detached when forming \(\sigma_{\mathrm{eff}}^2\). Code: `gpplus/utils/nigp_utils.py`.

### Gap

NIGP cannot represent residual output noise that varies with scene for reasons other than input linearization. That needs a free \(\sigma_{y,\mathrm{het}}^2(x)\).

---

## Notation: QoI SORF, NIGP, and Noise SORF

Two feature maps appear: a **mean** SORF for the QoI latent \(f\), and an optional **noise** SORF for \(\log_{10}\sigma_{y,\mathrm{het}}^2\). They are not interchangeable — only the mean model predicts the QoI.

### Data and tasks

| Symbol | Meaning |
|--------|---------|
| \(n\) | Number of training points |
| \(D_x\) | Input dimension (e.g. \(287\) = 285 bands + coszen + elevation) |
| \(x_i\in\mathbb R^{D_x}\) | Training input |
| \(X\in\mathbb R^{n\times D_x}\) | Training design matrix |
| \(y_i\in\mathbb R\) | Observed scalar QoI label (one task; S4 trains per QoI) |
| \(y\in\mathbb R^{n}\) | Training labels |
| \(x_*\), \(y_*\) | Test input and unobserved test QoI |

### QoI (mean) SORF GP

| Symbol | Meaning |
|--------|---------|
| \(f(x)\) | Latent QoI function (noiseless) |
| \(D_f\) | Number of **mean** SORF features (e.g. 2400) |
| \(\phi_f(x)\in\mathbb R^{D_f}\) | Mean SORF feature map |
| \(\Phi_f\in\mathbb R^{n\times D_f}\) | Training mean features |
| \(w_f\in\mathbb R^{D_f}\) | Dual / feature weights for \(f\) |
| \(m_f(x)\), \(k_f(x,x')\) | Prior mean and kernel |
| \(\mu_f(x)\) | Posterior mean of \(f\) |
| \(\sigma_f^2(x)\) | Posterior variance of \(f\) (latent; **not** observation noise) |
| \(C_y\) | Obs. covariance \(\mathrm{diag}(\sigma_{\mathrm{eff}}^2)+\Phi_f\Phi_f^\top\) |

### Homoscedastic baseline

Constant \(\sigma_y^2\) with \(\varepsilon_i\sim\mathcal N(0,\sigma_y^2)\) and \(\sigma_{\mathrm{eff}}^2(x_i)=\sigma_y^2\) before NIGP.

### NIGP (input-driven term)

| Symbol | Meaning |
|--------|---------|
| \(\sigma_{x,k}^2\) | Input-noise **variance** on coordinate \(k\) |
| \(\Sigma_x=\mathrm{diag}(\sigma_{x,1}^2,\ldots,\sigma_{x,D_x}^2)\) | Input-noise covariance |
| \(\mu_f^{\mathrm{homo}}(x)\) | Homoskedastic dual mean of \(f\) used for slopes |
| \(\nabla_x\mu_f(x_i)\) | Jacobian, **detached** when building \(\sigma_{\mathrm{eff}}^2\) |
| \(\sigma_{\mathrm{nigp}}^2(x_i)\) | Input-propagated extra variance at \(x_i\) |

\[
\sigma_{\mathrm{nigp}}^2(x_i)
= (\nabla_x\mu_f)^{\top}\Sigma_x\nabla_x\mu_f
= \sum_{k=1}^{D_x}\sigma_{x,k}^2\,(\partial_{x_k}\mu_f^{\mathrm{homo}}(x_i))^2
\]

Classic NIGP: \(\sigma_{\mathrm{eff}}^2(x_i)=\sigma_y^2+\sigma_{\mathrm{nigp}}^2(x_i)\).

### Noise SORF GP (heteroscedastic likelihood)

This model does **not** predict the QoI. It predicts a log-noise field that supplies free output variance \(\sigma_{y,\mathrm{het}}^2(x)>0\) (the hetero “nugget”).

| Symbol | Meaning |
|--------|---------|
| \(\sigma_{y,\mathrm{het}}^2(x)>0\) | Free input-dependent **output** noise variance |
| \(\ell_{\mathrm{het}}(x)\) | Latent log-noise: \(\ell_{\mathrm{het}}=\log_{10}\sigma_{y,\mathrm{het}}^2\) |
| \(D_{\mathrm{het}}\) | Number of **noise** SORF features (\(D_{\mathrm{het}}\ll D_f\)) |
| \(\phi_{\mathrm{het}}(x)\) | Noise SORF feature map |
| \(\Phi_{\mathrm{het}}\) | Training noise features |
| \(w_{\mathrm{het}}\), \(b_{\mathrm{het}}\) | Noise-feature weights and bias |
| \(\mu_{\ell,\mathrm{het}}\), \(\sigma_{\ell,\mathrm{het}}^2\) | Posterior of \(\ell_{\mathrm{het}}\) **only if** it is a full second GP (optional) |

**Lightweight MAP / feature-linear form** (recommended for SORF stacks):

\[
\ell_{\mathrm{het}}(x)=b_{\mathrm{het}}+w_{\mathrm{het}}^\top\phi_{\mathrm{het}}(x),\qquad
\sigma_{y,\mathrm{het}}^2(x)=10^{\mathrm{SoftClamp}(\ell_{\mathrm{het}}(x))}
\]

**v1 (S5) rule:** only the MAP **mean** \(\mu_{\ell,\mathrm{het}}(x)\) enters \(\sigma_{\mathrm{eff}}^2\). Predictive variance \(\sigma_{\ell,\mathrm{het}}^2(x)\) of the noise field is **discarded** (not integrated into the Woodbury diagonal). Do not confuse \(\sigma_{\ell,\mathrm{het}}^2\) with latent QoI variance \(\sigma_f^2\) or with \(\sigma_{\mathrm{nigp}}^2\).

A small MLP SoftClamp head is an equivalent parameterization of \(\sigma_{y,\mathrm{het}}^2(x)\).

**Full second GP on \(\ell_{\mathrm{het}}\)** (heavier; not used in S5 v1): \((\mu_{\ell,\mathrm{het}},\sigma_{\ell,\mathrm{het}}^2)\) describe uncertainty about the **noise level**, not the QoI. Never mix \(\mu_{\ell,\mathrm{het}}\) into \(\mu_f\).

### S5 three-phase Adam schedule (0-based)

Let \(E_{\mathrm{hom}}=\texttt{EPOCHS\_HOM}\), \(E_{\mathrm{het}}=\texttt{EPOCHS\_HET}\), \(E=\texttt{NUM\_EPOCHS}\). Require \(E_{\mathrm{hom}}+E_{\mathrm{het}}<E\).

| Phase | Epochs | \(\sigma_{\mathrm{eff}}^2\) | Trainable noise | NIGP |
|-------|--------|-------------------------------|-----------------|------|
| Hom | \([0,\,E_{\mathrm{hom}})\) | \(\sigma_{y,\mathrm{hom}}^2\) (scalar) | `raw_noise` only | off |
| NIGP | \([E_{\mathrm{hom}},\,E_{\mathrm{hom}}+E_{\mathrm{het}})\) | \(\sigma_{y,\mathrm{hom}}^2+\sigma_{\mathrm{nigp}}^2\) | \(\Sigma_x\); freeze `raw_noise` + Noise SORF | on |
| Het refine | \([E_{\mathrm{hom}}+E_{\mathrm{het}},\,E)\) | \(\sigma_{y,\mathrm{het}}^2+\sigma_{\mathrm{nigp}}^2\) | Noise SORF on; \(\Sigma_x\) stays on | on |

\(E_{\mathrm{het}}\) is the **NIGP-only** window before hetero unlock (CLI name kept). Noise SORF is warm-started from scalar \(\sigma_{y,\mathrm{hom}}^2\) at het unlock.

**σ_nigp slope refreshes (exactly two):** at NIGP unlock \(e=E_{\mathrm{hom}}\) and at last epoch \(e=E-1\) (via `HeteroNIGPScheduleCallback` + `nigp_slope_manual`). Between them, reuse cached \(\nabla_x\mu_f\).

Entry: `experiments_SORF/S5_emit_heteroscedastic_SORF.py`.

### Combined effective observation noise \(\sigma_{\mathrm{eff}}^2\)

| Mode | Formula | ID |
|------|---------|-----|
| Homoscedastic | \(\sigma_{\mathrm{eff}}^2=\sigma_y^2\) | H0 |
| NIGP-only (current) | \(\sigma_{\mathrm{eff}}^2=\sigma_y^2+\sigma_{\mathrm{nigp}}^2\) | H1 |
| Hetero-only | \(\sigma_{\mathrm{eff}}^2=\sigma_{y,\mathrm{het}}^2\) | H2 |
| Additive Residual Hetero-NIGP | \(\sigma_{\mathrm{eff}}^2=\sigma_{y,\mathrm{het}}^2+\sigma_{\mathrm{nigp}}^2\) | H3 |
| Multiplicative (idea) | \(\sigma_{\mathrm{eff}}^2=\sigma_{y,\mathrm{het}}^2\,(1+\alpha\,\sigma_{\mathrm{nigp}}^2)\) | — |

When free \(\sigma_{y,\mathrm{het}}^2\) is active, it **replaces** scalar \(\sigma_y^2\).

### Prediction: how the pieces fit the overarching SORF

\[
\begin{aligned}
\mu_y(x_*) &= \mu_f(x_*) \\
\sigma_{\mathrm{eff}}^2(x_*) &= \sigma_{y,\mathrm{het}}^2(x_*)+\sigma_{\mathrm{nigp}}^2(x_*) \\
\sigma_{y\mid\mathcal D}^2(x_*) &= \sigma_f^2(x_*)+\sigma_{\mathrm{eff}}^2(x_*) \\
y_*\mid\mathcal D,x_* &\approx \mathcal N(\mu_y(x_*),\,\sigma_{y\mid\mathcal D}^2(x_*))
\end{aligned}
\]

- QoI **mean**: mean SORF only (\(\mu_f\)).
- Latent uncertainty: mean SORF (\(\sigma_f^2\)).
- Observation noise: \(\sigma_{y,\mathrm{het}}^2\) (+ \(\sigma_{\mathrm{nigp}}^2\) if enabled).
- Observation intervals: \(\mu_y\pm 2\sqrt{\sigma_f^2+\sigma_{\mathrm{eff}}^2}\).

```
x ──► φ_f(x)  ──► QoI SORF ──► μ_f(x), σ_f²(x)
  │
  ├──► ∇_x μ_f, Σ_x ──► σ_nigp²(x) ──┐
  │                                   ├──► σ_eff² = σ_{y,het}² + σ_nigp² ──► Woodbury
  └──► φ_het(x) ──► Noise SORF ──► σ_{y,het}²(x) ─┘

Report:  mean = μ_f(x)
         obs std = sqrt(σ_f²(x) + σ_eff²(x))
```

---

## Literature (pointers)

| Topic | Key refs |
|--------|----------|
| NIGP | McHutchon & Rasmussen (2011) |
| Uncertain inputs | Girard et al. (2003); Quiñonero-Candela et al. (2003); Dallaire et al. (2009) |
| Heteroscedastic GP | Goldberg et al. (1998); Kersting et al. MLHGP (2007); Lázaro-Gredilla & Titsias VHGP (2011) |
| Variable / nonstationary noise | Snelson & Ghahramani (2006); Tolvanen et al. (2014); Heinonen et al. (2016) |
| Scalable features | Rahimi & Recht RFF; Yu et al. ORF/SORF; Cutajar et al. Woodbury DGP |
| Domain | EMIT / ISOFIT / Bohn et al. snow–EMIT |

Full BibTeX: `hetero_nigp.bib`.

---

## Heteroscedastic Gaussian likelihood

\[
p(y\mid f,x)=\mathcal N(y; f(x), \sigma_{y,\mathrm{het}}^2(x)),\qquad
C_y=\mathrm{diag}(\sigma_{y,\mathrm{het}}^2(x_1),\ldots,\sigma_{y,\mathrm{het}}^2(x_n))+\Phi_f\Phi_f^\top
\]

Parameterize \(\ell_{\mathrm{het}}=\log_{10}\sigma_{y,\mathrm{het}}^2\) by flexibility:

1. Constant → homoscedastic (\(\sigma_{y,\mathrm{het}}^2\equiv\sigma_y^2\))  
2. Parametric / small MLP + SoftClamp (recommended start)  
3. Small second SORF (\(D_{\mathrm{het}}\ll D_f\))  
4. Full noise GP (Goldberg / MLHGP / VHGP)

---

## Why this data wants free heteroscedasticity

S4: \(X\in\mathbb R^{n\times 287}\) (285-band TOA + coszen + elevation) → 7 QoIs.

| Source | Model |
|--------|--------|
| Instrument / band SNR, scaling, registration | NIGP \(\Sigma_x\) / \(\sigma_{x,k}^2\) (optionally NEDL-anchored) |
| RT misspec, sim–reality gap, QoI-dependent residuals | Free \(\sigma_{y,\mathrm{het}}^2(x)\) |

```
Instrument NEDL ──► Σ_x / σ_{x,k}² ──┐
                                     ├──► σ_eff² = σ_{y,het}² + σ_nigp² ──► Woodbury
RT / QoI residual ──► σ_{y,het}²(x) ─┘
```

---

## Combination ideas

### Primary: Additive Residual Hetero-NIGP

As above. Keep McHutchon two-step \(\nabla_x\mu_f\); extend `effective_noise_variance` to accept per-point `noise_y` of shape `(n,)`.

**Train / test asymmetry:** train MLL with train-side \(\sigma_{\mathrm{eff}}^2\); at test evaluate \(\sigma_{y,\mathrm{het}}^2(x_*)\) and test-side \(\nabla_x\mu_f\) for observation intervals.

### Other ideas

1. **Instrument-anchored \(\Sigma_x\), free \(\sigma_{y,\mathrm{het}}^2\)** — fix/prior \(\sigma_{x,k}^2\) from NEDL; learn residual only.
2. **Phased unlock (S5)** — Hom \(\sigma_{y,\mathrm{hom}}^2\) → NIGP \(\sigma_{\mathrm{nigp}}^2\) → unlock Noise-SORF \(\sigma_{y,\mathrm{het}}^2\) for refinement; \(\sigma_{\mathrm{nigp}}\) slopes refresh at NIGP unlock and last epoch.
3. **Multiplicative SNR coupling** — \(\sigma_{\mathrm{eff}}^2=\sigma_{y,\mathrm{het}}^2\,(1+\alpha\,\sigma_{\mathrm{nigp}}^2)\).
4. **Shared latent log-noise** — one \(\ell_{\mathrm{het}}(x)\) drives \(\sigma_{y,\mathrm{het}}^2\) and scales atmospheric \(\sigma_{x,k}^2\).
5. **QoI-coupled multitask noise** — ICM on \(\ell_{\mathrm{het},t}(x)\) for MTGP path.
6. **SORF-for-noise** — small SORF for \(\ell_{\mathrm{het}}\); large features for the mean.

---

## Identifiability

\(\sigma_{y,\mathrm{het}}^2\) and \(\sigma_{\mathrm{nigp}}^2\) both inflate diagonal \(C_y\). Mitigate with instrument priors, phased unlock, SoftClamp on \(\ell_{\mathrm{het}}\), optional ASD-std warm-start, and per-QoI \(\sigma_{y,\mathrm{het},t}^2\) before multitask noise.

---

## Experiment sketch (follow-on)

| ID | Model | \(\Sigma_x\) / \(\sigma_{x,k}^2\) | \(\sigma_{y,\mathrm{het}}^2(x)\) |
|----|--------|----------------------------------|----------------------------------|
| H0 | Homoscedastic SORF | off | \(\equiv\sigma_y^2\) (constant) |
| H1 | NIGP-only (current) | learn | \(\equiv\sigma_y^2\) (constant) |
| H2 | Hetero-only | off | free |
| H3 | Additive Residual Hetero-NIGP | learn | free |

**Metrics:** RRMSE; NIS / ASD coverage; wall-clock.  
**Success:** H3 improves calibration vs H1 without hurting RRMSE; ablations show \(\sigma_{y,\mathrm{het}}^2\) and \(\sigma_{\mathrm{nigp}}^2\) are not redundant.

---

## Code pointers

- Likelihood: `gpplus/likelihoods/log_gaussian_likelihood.py`
- NIGP: `gpplus/utils/nigp_utils.py` (`effective_noise_variance`)
- Diag Woodbury / MLL: `gpplus/utils/rff_utils.py`, `gpplus/training/nigp_mll.py`
- Noise SORF + additive combine: `gpplus/utils/hetero_noise.py`
- Schedule: `gpplus/training/callbacks.py` (`HeteroNIGPScheduleCallback`)
- S4: `experiments_SORF/S4_emit_SORF.py`, `emit_s4_sorf_base.py`
- S5: `experiments_SORF/S5_emit_heteroscedastic_SORF.py` (hetero kwargs on `emit_s4_sorf_base`)

**Implemented (S5 v1):** per-point \(\sigma_{y,\mathrm{het}}^2\) from Noise SORF MAP mean; additive \(\sigma_{\mathrm{eff}}^2\); three-phase schedule with endpoint \(\sigma_{\mathrm{nigp}}\) refreshes.

---

## One-sentence talk line

NIGP explains input-driven \(\sigma_{\mathrm{nigp}}^2\); free \(\sigma_{y,\mathrm{het}}^2(x)\) explains residual scene-dependent output noise that slopes cannot—add them as \(\sigma_{\mathrm{eff}}^2=\sigma_{y,\mathrm{het}}^2+\sigma_{\mathrm{nigp}}^2\) on the diagonal and keep the same Woodbury path.
