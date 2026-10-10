# II-42 M228 Row-Aware Qrels-Free Geometry Preflight Report

Date: 2026-09-30

## Purpose

M226A/B established the current supervised score-geometry milestone. The
`meanz_pair_meanz_score + m220c_joint,w=3.5` endpoint is stable across student
seeds, with M226B mean MAP 0.360175 / Recall 0.639834 and deltas versus PPLX
dense of +0.058710 / +0.029097, with no dense-unsafe rows.

M227A then showed that simple qrels-free geometry teachers have macro headroom
but are not dense-safe. The best policy, `rankt5_dense_mean_z,w=3.5`, reached
MAP 0.325182 / Recall 0.621012, deltas +0.023717 / +0.010275, but remained
unsafe on FiQA and SciDocs. `dense_z` was most similar to supervised M226
geometry, but had much weaker headroom and the same unsafe rows.

M228 asks whether the qrels-free route can still evolve by changing structure
rather than sweeping scalar weights: use query/window agreement and risk
features to decide when a RankT5+dense geometry is trusted, damped, or skipped.
This is a teacher-side preflight. No new student is trained unless a qrels-free
teacher policy first shows enough row safety and macro headroom.

## M228A Plan

Evaluate row-aware qrels-free teacher policies over the same frozen full7
top256 window:

- `agreement_soft_mean_z`
- `agreement_hard_mean_z`
- `triad_consensus_mean_z`
- `dense_anchor_residual_soft`
- `dense_anchor_residual_hard`
- `rankt5_dense_rrf_guarded`
- `risk_dropout_mean_z`
- `family_damped_mean_z`
- `family_damped_residual`

The policies use only qrels-free signals for teacher construction:

- RankT5 cache scores from M217
- dense scores inside the frozen top256 window
- fixed BM25/SSR baseline scores
- top-k source overlap, source-triad agreement, and score margins

M214B/M226 supervised artifacts may be used only as diagnostic references for
similarity to the known-good mean-z geometry. They are not used to construct any
M228 teacher policy. Qrels are used only for final metrics.

Decision rule:

- If a policy improves over M227A on macro MAP/Recall and reduces unsafe rows
  without catastrophic row loss, prepare a narrow M228B qrels-free geometry
  distillation run.
- If no policy reduces the FiQA/SciDocs failure mode while preserving positive
  macro deltas, stop this qrels-free RankT5/dense/baseline geometry branch.
  The next step should require a genuinely stronger weak-supervision source,
  not more routing or scalar tuning.

## M228A Result

M228A completed the row-aware qrels-free teacher-side preflight. The row-aware
gates preserved the same broad macro pattern as M227A, but did not improve the
best macro result and did not reduce the persistent FiQA/SciDocs unsafe rows.

Best policies by macro MAP:

| Policy | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Top24 overlap vs M226 geometry | Pearson z |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `agreement_soft_mean_z` | 3.50 | 0.324848 | 0.621035 | +0.023383 | +0.010298 | FiQA, SciDocs | 0.555489 | 0.702710 |
| `dense_anchor_residual_soft` | 3.50 | 0.324848 | 0.621035 | +0.023383 | +0.010298 | FiQA, SciDocs | 0.555489 | 0.702710 |
| `triad_consensus_mean_z` | 3.50 | 0.314805 | 0.616837 | +0.013340 | +0.006100 | Arguana, FiQA, SciDocs | 0.563356 | 0.663822 |
| `risk_dropout_mean_z` | 3.50 | 0.312712 | 0.614416 | +0.011248 | +0.003679 | Arguana, FiQA, SciDocs, TREC-COVID | 0.565798 | 0.296654 |

The best M228A teacher remains slightly below M227A
`rankt5_dense_mean_z,w=3.5` on macro MAP (+0.023383 versus +0.023717), with
the same FiQA and SciDocs unsafe rows. No M228A variant produced positive macro
MAP/Recall with zero or one dense-unsafe row, and no positive variant removed
either FiQA or SciDocs from the unsafe set.

The alpha diagnostics explain why the row-aware gates did not help. For the
best soft-agreement policy, mean alpha stayed high on exactly the risky rows:
FiQA 0.824 and SciDocs 0.851. The qrels-free source-agreement features see
RankT5 and dense as aligned on these rows, even though the resulting teacher is
unsafe. More gating based on the same signals is therefore unlikely to recover
the supervised M226 row corrections.

## M228 Decision

Do not launch M228B. M228A did not satisfy the continuation rule: it preserved
macro headroom but did not improve over M227A and did not reduce the
FiQA/SciDocs failure mode.

This stops the current qrels-free RankT5/dense/baseline geometry branch. The
route has now been pushed through global geometry, row-aware runtime routing,
row-risk pseudo-labeling, and row-aware teacher-side geometry gating. The
remaining gap to M226B is not a scalar, router, or compact-head issue. It needs
a stronger weak-supervision source or an objective with information that can
distinguish the FiQA/SciDocs failure cases where current qrels-free source
agreement is confidently wrong.
