# MidlineSparse hard-\(x\) gate (separate branch)

This branch is **not** the champion MidlineSparse path used for co-located
multimodality (A01 / A04) and foundation comparisons.

| | Champion MidlineSparse | Hard-\(x\) branch |
|---|---|---|
| Model | `MidlineSparseMoEGPR` | `MidlineSparseHardXGateMoEGPR` |
| Module | `gpplus/models/midline_sparse_moe.py` | `gpplus/models/midline_sparse_hard_xgate.py` |
| Gate | soft `π` — `constant` or MLP `softmax` | hard `w(x)` — top-1 one-hot (Gumbel-ST train) |
| Expert prune | global Hard-Concrete `α_k` | same (global) |
| Validity | `π(x) ∈ Δ^{K-1}` via softmax | `w(x) ∈ Δ^{K-1}` via one-hot + renormalize |
| Scope | co-located modes; \(x\)-gate misspecified | separated regimes of \(x\) (A02-like) |
| Experiments | `run_midline_sparse.py`, `foundation_adapters.py` | `run_midline_hard_xgate.py`, `plot_midline_hard_xgate.py` |
| Git | other MidlineSparse work | feature branch `midline-sparse-hard-xgate` |

## Validity

Hard on/off in \(x\) only sparsifies the gate. At every \(x\):

\[
p(y\mid x)=\sum_k w_k(x)\,p_k(y\mid x),\qquad
w_k(x)\ge 0,\quad \sum_k w_k(x)=1.
\]

Independent zeros without renormalization are not allowed.

## Run

```bash
conda activate gpplus2026
cd experiments_MOEGP/feasibility_mode_count
python run_midline_hard_xgate.py --smoke
python run_midline_hard_xgate.py                              # all: A01,unimodal,A02,A04,A05
python run_midline_hard_xgate.py --problems A01,A04,A05
python run_midline_hard_xgate.py --n-samples 400 --k-max 4    # faster
python run_midline_hard_xgate.py --no-plots
```

Each problem gets hard-\(x\) + soft MLP (A02 also soft constant). Outputs under
`results/hard_xgate/`:

- Summary: `hard_xgate_summary.json`
- Router: `{stem}_hard_x_assignment.png`, `{stem}_hard_x_gate_heatmap.png`
- ScoringBench: `{stem}_metrics.csv|json`, `{stem}_metrics_table.png`
- Marginals: `{stem}_marginal_x*.png` (per probe, per model)
- Posterior: `{stem}_posterior_vs_data.png`

## Do not

- Add `gate_type="hard_x"` to champion `MidlineSparseMoEGPR`
- Wire this model into `foundation_adapters.py` / A01 scoring without an explicit decision
- Treat hard \(x\)-switching as a fix for co-located multimodality
