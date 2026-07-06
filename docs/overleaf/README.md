# Overleaf: GPPlus documentation notes

## Upload to Overleaf

1. Create a new blank project on [Overleaf](https://www.overleaf.com).
2. Upload the desired `.tex` file (or zip this folder).
3. Set the main document to the file you want to compile.
4. Compile with **pdfLaTeX**.

## Contents

| File | Topic |
|------|--------|
| [`rff_woodbury_derivation.tex`](rff_woodbury_derivation.tex) | RFF Woodbury MLL / \(\Sigma^{-1}\) factorization |
| [`rff_predictive_uncertainty.tex`](rff_predictive_uncertainty.tex) | Predictive mean, \(f_{\mathrm{var}}\), observation std |
| [`rff_multitask_gp_plan.tex`](rff_multitask_gp_plan.tex) | RFF multitask GP implementation plan |
| [`pca_gp_pipeline.tex`](pca_gp_pipeline.tex) | PCA + partitioned exact GP for TOA; SVGP context |

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
