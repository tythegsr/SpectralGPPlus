# Overleaf: GPPlus documentation notes

## Upload to Overleaf

1. Create a new blank project on [Overleaf](https://www.overleaf.com).
2. Upload the desired `.tex` file (or zip this folder).
3. Set the main document to the file you want to compile.
4. Compile with **pdfLaTeX**.

## Contents

| File | Topic |
|------|--------|
| [`sorf_nigp_st_emulator_overview.tex`](sorf_nigp_st_emulator_overview.tex) | **Start here (ST):** SORF + Woodbury + NIGP wiring through single-task prediction / EMIT eval |
| [`gp_study_guide.tex`](gp_study_guide.tex) | Exact GP study guide: prior, likelihood, posterior, MLL, MLE/MAP, LOO |
| [`rff_woodbury_derivation.tex`](rff_woodbury_derivation.tex) | RFF Woodbury MLL / \(\Sigma^{-1}\) factorization; primal \(M\) vs dual \(\Lambda\) |
| [`rff_predictive_uncertainty.tex`](rff_predictive_uncertainty.tex) | Predictive mean, $f_{\mathrm{var}}$, observation std; TOA log / logit warps |
| [`soft_probabilistic_bounds.tex`](soft_probabilistic_bounds.tex) | Soft $\mu\pm k\sigma$ physical bound penalties (per-QoI floors/ceilings) |
| [`pac_bayes_derivation.tex`](pac_bayes_derivation.tex) | PAC-Bayes primer + Catoni training loss mapped to GPPlus MLL wrapper |
| [`nigp_rff_woodbury.tex`](nigp_rff_woodbury.tex) | Classic NIGP (noisy inputs) on RFF/SORF: diag-noise Woodbury via \(D^{-1/2}\) |
| [`rff_multitask_gp_plan.tex`](rff_multitask_gp_plan.tex) | RFF multitask GP implementation plan |
| [`rff_mt_kronecker_efficiency.tex`](rff_mt_kronecker_efficiency.tex) | RFFMTGPR Woodbury: current path, toy example, storage ladder (Tier 0–4), all speedup options |
| [`rff_mtgpr_cost_performance.tex`](rff_mtgpr_cost_performance.tex) | TOA RFFMTGPR experiment cost vs performance ($n$, $D$, epochs, lr) |
| [`pca_gp_pipeline.tex`](pca_gp_pipeline.tex) | PCA + partitioned exact GP for TOA; SVGP context |
| [`pca_dimension_selection.tex`](pca_dimension_selection.tex) | Choosing per-QoI PCA $p$: 99% variance vs ridge-on-scores |
| [`sorf_fwht_bug.tex`](sorf_fwht_bug.tex) | SORF legacy FWHT aliasing bug (why TOA broke when “fixed”) |
| [`lrnn_dbk_gpplus.tex`](lrnn_dbk_gpplus.tex) | LRNN / Deep Basis Kernel: paper alignment vs GPPlus differences |
| [`second_interim_report.tex`](second_interim_report.tex) | Caltech SFP Second Interim Report (figures in `figures/interim2/`) |

### Second interim report on Overleaf

1. Upload `second_interim_report.tex`.
2. Upload the figure folder so Overleaf has either:
   - `interim2/` at the project root (recommended), **or**
   - `figures/interim2/`
3. Upload [`ref.bib`](ref.bib) next to the `.tex` (same project root).
4. The `.tex` uses `\graphicspath{{interim2/}{figures/interim2/}}` and paths like
   `july16_rff_lr0.01/scatter/y_cos_scatter.png` (no `figures/` prefix in
   `\includegraphics`). Do **not** write `\includegraphics{\figdir/...}` — that
   macro+slash form often fails to resolve files.
5. Set the main document to `second_interim_report.tex` and compile with
   **pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX** (Overleaf does this automatically).

Regenerate interim figures/tables (from existing JSON/npz, no re-train)::

```bash
python experiments_toa/regen_interim2_figures.py
```

### Box legend (Woodbury / uncertainty notes)

- **Orange boxes**: differences in layout or approximation vs. another formula.
- **Green boxes**: identical objects (same math, same cost).
- **Gray boxes**: code / experiment pointers.
- **Red boxes**: practical pitfalls (leakage, wrong uncertainty interpretation).

### PCA pipeline note (`pca_gp_pipeline.tex`)

- PCA on 270-d TOA inputs (after column drop), partitioned exact `GPR` with 1500 points per partition.
- **PCA + RFF/ORF/SORF**: `S1_toa_PCA_RFF.py` — PCA then Woodbury `RFFGPR` on full training set.
- Primary metric: **test RRMSE**.
- Documents all five ensemble aggregation modes logged every run.
- Explains SVGP as a conceptual alternative (not implemented in `experiments_PCA` v1).

### PCA dimension selection (`pca_dimension_selection.tex`)

- S2 per-QoI choice of $p$: cumulative variance threshold (default 99%) vs ridge $R^2$ on leading PC scores.
- Band modes: keep/drop subset vs all 285 wavelengths.
- Current PCA+GPR default: supervised ridge on full spectrum (`s2_task_pca_components_ridge_full.json`).
- Points at `s2_pca_svd_analysis.py` and `s2_task_pca_components_*.json`.
