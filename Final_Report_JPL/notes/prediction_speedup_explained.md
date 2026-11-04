# Why NI-SORF prediction got much faster

**Audience:** anyone who needs a plain-language account of the inference speedup (NIGP still on; same answers, less wasted work).

**Setup we care about:** 7 independent QoI models, ~10k train / 25k test points, SORF-RFF + NIGP.

---

## Bottom line

Prediction was slow for two separate reasons. We fixed both. **NIGP math did not change**—we still use train-side effective noise for the mean and test-side \(\nabla\mu\) for observation uncertainty.

| What was wrong | What we did | Effect |
|---|---|---|
| Same train-side Woodbury solve repeated ~13× per QoI while walking over test chunks | Solve **once**, reuse for every test batch | Largest win (~4× at matched D=2400: ~18.5 s → ~4.66 s/QoI; full 7 QoIs ~127 s → ~33 s wall) |
| Seven QoIs predicted one after another, and the cost table added those times | Optional **parallel** predict across QoIs; report **wall-clock** for all 7 | Cuts total wait on CPU / multi-GPU; fairer “how long to get all QoIs” metric |

**Not** the Matérn-3/2 kernel: that only changes how frequencies are sampled at init. After that, predict is the same as RBF. Older cost slides also compared Mat3/2 at **D=2400** to RBF at **D=2000**, which alone looks ~40% slower because cost scales roughly like \(D^2\).

---

## Picture of the old bottleneck

NIGP needs the slope of the predictive mean, \(\nabla_x \mu\), so input noise can be turned into extra observation noise.

For RFF/SORF that slope uses **frozen** feature weights \(w\) from a **homoskedastic** train solve (σ² only—not the NIGP \(d\)). That solve is expensive (Gram + Cholesky in feature dimension \(m \approx 2D\)).

**Old path (chunked test, e.g. 2048 points):**

```
once:   train ∇μ → d_train → NIGP Woodbury (mean weights)
then for each test chunk (~13 chunks for 25k points):
        featurize test
        mean + variance
        *** featurize ALL train again + re-solve homo w ***   ← waste
        test ∇μ → d_test → obs std
```

So we paid the heavy train solve about **once per chunk**, not once per QoI. That dominated wall time. Logging and plotting were never in `Prediction_Time`.

**New path:**

```
once:   homo w (cached)
once:   train ∇μ → d_train → NIGP Woodbury
then for each test chunk (or one full batch):
        one featurize of test (shared by mean, var, and ∇μ)
        mean + variance + test ∇μ using cached w
```

Default for S4 is also **one chunk over the full test set** (`PREDICT_CHUNK_SIZE = 0`) when memory allows, so we do not needlessly split work.

---

## What stayed the same (correctness)

- Train NIGP correction \(d_i = \sigma_y^2 + (\nabla\mu_i)^\top \Sigma_x (\nabla\mu_i)\) still builds the predictive **mean**.
- Test NIGP correction still builds **observation** intervals (NIS / coverage).
- Chunked vs full-batch and “cached \(w\)” vs “recompute \(w\)” match to numerical tolerance in tests.

---

## Parallelizing the seven QoIs

Each QoI is its own `RFFGPR`. Training still runs QoI-by-QoI. After all models are trained, we **stash them on CPU** and run a **predict phase**:

- Threads fan out over QoIs (`predict_n_jobs`).
- Default: **1 job on a single GPU** (avoids VRAM fights); **many jobs on CPU**; round-robin if several GPUs.
- Top-level `Prediction_Time` = **wall clock of that phase** (what you feel when waiting).
- `Prediction_Time_Sum` = old “add seven timers” number (kept for comparison).

On one GPU, parallel jobs do not magically make seven models finish in the time of one—the GPU is already busy—but the code path is ready for CPU or multi-GPU, and the metric matches deploy-style “all QoIs ready.”

---

## Numbers (order of magnitude)

Same problem class: NIGP on, 25k test points, 7 QoIs (`cos_i`, `grain_size`, `dust`, `algae`, `cwv`, `lwc`, `aot`). Sept18 headline: RBF SORF, D=2400, amortized full-batch, `predict_n_jobs=1` on GPU.

| Run | D | Notes | Per QoI | All 7 (reported) |
|---|---|---|---|---|
| Sept09 Mat3/2 (old predict) | 2400 | chunked, re-solve every chunk | ~18.5 s | ~127 s (sum) |
| **Sept18 RBF (new)** | **2400** | **amortized, full 7 QoIs** | **~4.66 s** | **~33 s wall** |
| Sept18 Mat3/2 (new) | 2400 | amortized, 1 QoI only | ~4.7 s | — |
| Sept09 Mat3/2 @ D=2000 (old) | 2000 | fairer vs RBF | ~13 s | ~91 s (sum) |
| Sept08 NI-SORF RBF (old) | 2000 | | ~13 s | ~89 s (sum) |
| Sept08 SORF, **no** NIGP | 2000 | no ∇μ path | ~4 s | ~28 s (sum) |
| Sept13 speedup test (new path) | 1000 | GPU, jobs=1, full-batch | ~**0.96 s** | ~**7 s** wall |

### Sept18 per-QoI breakdown (D=2400, RBF, 7 QoIs)

| QoI | Predict (s) |
|---|---|
| cos_i | 4.68 |
| grain_size | 4.63 |
| dust | 4.63 |
| algae | 4.63 |
| cwv | 4.69 |
| lwc | 4.69 |
| aot | 4.69 |
| **Mean / Sum / Wall** | **4.66 / 32.65 / 32.96** |

Takeaway: at matched D=2400, amortization is ~**4×** per QoI and ~**3.9×** for the full 7-QoI suite (~127 s → ~33 s). Smaller \(D\) then lands near **1 s/QoI** on GPU for 25k points.

---

## One-sentence version (for a talk)

“We stopped re-factorizing the training system on every test chunk, reused one featurize for mean and NIGP slopes, optionally predict all QoIs together, and report wall-clock—so NIGP inference is the same math with far less repeated linear algebra.”

---

## Where it lives in code

- Amortized NIGP predict: `gpplus/utils/nigp_utils.py` (`feature_weights`, optional `proj`), `gpplus/training/eval.py`, `gpplus/models/rff_gpr.py`
- Two-phase + parallel QoIs: `experiments_SORF/emit_s4_sorf_base.py`, `S4_emit_SORF.py` (`PREDICT_CHUNK_SIZE`, `PREDICT_N_JOBS`)
- Cost-slide label: `render_icy_rrmse_nis_tables.py` (`wall 7-QoI (parallel)` when `predict_n_jobs > 1`)
