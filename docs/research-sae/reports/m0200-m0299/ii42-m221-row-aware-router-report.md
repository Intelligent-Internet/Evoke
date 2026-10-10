# II-42 M221 Row-Aware Router Report

Date: 2026-09-28

## Purpose

M221 tests whether the saturated M220 runtime-gate branch can still be improved
with structured qrels-free row-aware routing. This is not another representation
training run and not a loss sweep. It keeps the fixed M218A/M218B `32d INT4`
mini-vector heads and evaluates router policies using only qrels-free features
inside the frozen top256 candidate window. Qrels are used only for final
metrics.

The continuation baseline is the M220C/M220D optimum:

| Route | MAP | Recall | dMAP vs PPLX dense | dRecall vs PPLX dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M220C/M220D best | 0.304769 | 0.615656 | +0.003304 | +0.004919 | Arguana, FiQA, SciDocs, TREC-COVID |

## M221A Result

M221A evaluated 176 deployable `student_32d_int4` variants across the fixed
M218A/M218B checkpoints. It compared the M220C joint hard gate against stricter
query-level agreement gates and family-level qrels-free risk gates.

Best variant:

| Checkpoint | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.375 | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID | 0.786 |

The best M221A router improves over the M220C/M220D baseline by about
+0.001050 MAP and +0.000075 Recall while preserving the same unsafe-row set.
This is a small but real structural improvement: family-level qrels-free
dense-risk routing is better than the local hard overlap gate alone.

Top alternatives:

| Checkpoint | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.25 | 0.305697 | 0.615521 | +0.004232 | +0.004784 | Arguana, FiQA, SciDocs, TREC-COVID |
| M218B seed 313 `blend_d0p5` | `family_soft_risk` | 1.375 | 0.305668 | 0.615360 | +0.004204 | +0.004623 | Arguana, FiQA, SciDocs, TREC-COVID |
| M218B seed 313 `blend_d0p5` | `family_soft_risk` | 1.25 | 0.305565 | 0.615534 | +0.004101 | +0.004797 | Arguana, FiQA, SciDocs, TREC-COVID |
| M218A `blend_d0p5` | `family_dense_half` | 1.375 | 0.305049 | 0.614765 | +0.003585 | +0.004028 | Arguana, FiQA, SciDocs, TREC-COVID |

## Row Anatomy

Best M221A row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | dMAP vs fixed hybrid | Mean alpha | Safe vs dense |
| --- | ---: | ---: | ---: | ---: | --- |
| Arguana | -0.016868 | -0.001428 | +0.004855 | 0.464 | no |
| FiQA | -0.023701 | -0.036109 | +0.056280 | 0.994 | no |
| NFCorpus | +0.007785 | +0.007742 | +0.018455 | 0.938 | yes |
| SciDocs | -0.005691 | -0.013217 | +0.013643 | 0.992 | no |
| SciFact | +0.040842 | +0.016667 | +0.007602 | 0.993 | yes |
| TREC-COVID | -0.003588 | -0.003876 | +0.015159 | 1.000 | no |
| Webis | +0.031696 | +0.065180 | +0.000620 | 0.122 | yes |

The router improves Arguana damage relative to M220C by reducing the correction
strength aggressively on that family, and it keeps the large Webis gain by
mostly preserving dense where student/teacher churn is high. It does not solve
FiQA, SciDocs, or TREC-COVID because their qrels-free agreement features still
look superficially safe while dense-row metrics remain below the floor.

## Interpretation

Keep:

- M218B seed 313 `blend_d0p5` remains the strongest product-shaped checkpoint.
- M220C `joint_hard_0p30`, `overlap_k=24`, `w=1.25` remains the best simple
  local confidence gate.
- M221A `family_dense_half`, especially at weight `1.375`, is the new best
  qrels-free deployable route and should be carried forward.
- `family_soft_risk` is useful as a secondary family-level signal.

Discard or deprioritize:

- More generic RankT5-safe transfer tuning; M219A already failed.
- More scalar hard-gate threshold/overlap sweeps; M220D saturated.
- Per-query strict agreement gates as primary routes; they trail the family
  router on macro MAP and do not reduce the unsafe-row set.

Current bottleneck:

The route now has a stable qrels-free macro win over PPLX dense, but row safety
is still not solved. The remaining failures are structural. FiQA and SciDocs are
especially problematic because qrels-free teacher/student/dense agreement is not
enough to predict dense safety. TREC-COVID is small and close to the floor, but
still unsafe. This means further breakthrough needs either a stronger
weak-supervision source or an explicit row-aware pseudo-label objective, not
another generic scalar sweep.

## M221B Plan

Run a narrow generalization check around the new structural winner:

- fixed checkpoints: same four M218A/M218B checkpoints
- router policies: `family_dense_half,family_soft_risk,m220c_joint`
- fixed weights: `1.25,1.375,1.5`
- overlap-k values: `16,20,24,28,32`
- no new representation training
- no locked-test data

If M221B confirms that `family_dense_half` improves over M220C across nearby
overlap windows without expanding unsafe rows, promote it as the retained
qrels-free runtime route. If it does not generalize, keep M220C as the safer
minimal route and stop router variants before moving to a row-aware pseudo-label
objective.

## M221B Result

M221B completed the overlap-k generalization check for the family-level routers.
It tested overlap-k values `16,20,24,28,32`, router policies
`family_dense_half,family_soft_risk,m220c_joint`, and weights
`1.25,1.375,1.5` over the same fixed M218A/M218B checkpoints.

Best per overlap-k:

| overlap-k | Checkpoint | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.375 | 0.305609 | 0.614720 | +0.004144 | +0.003983 | Arguana, FiQA, SciDocs, TREC-COVID | 0.776 |
| 20 | M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.375 | 0.305760 | 0.615433 | +0.004296 | +0.004696 | Arguana, FiQA, SciDocs, TREC-COVID | 0.790 |
| 24 | M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.375 | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID | 0.786 |
| 28 | M218B seed 313 `blend_d0p5` | `family_dense_half` | 1.375 | 0.305687 | 0.615532 | +0.004222 | +0.004795 | Arguana, FiQA, SciDocs, TREC-COVID | 0.799 |
| 32 | M218B seed 313 `blend_d0p5` | `family_soft_risk` | 1.375 | 0.305688 | 0.615170 | +0.004223 | +0.004433 | Arguana, FiQA, SciDocs, TREC-COVID | 0.814 |

The best overall M221B variant is the same as M221A:

| Checkpoint | overlap-k | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M218B seed 313 `blend_d0p5` | 24 | `family_dense_half` | 1.375 | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |

M221B confirms that the `family_dense_half` signal generalizes across nearby
overlap windows. The exact optimum remains overlap-k `24`, but overlap-k
`20,28,32` stay close and keep the same unsafe-row set. This means the M221
improvement is not a single overlap-k artifact.

## Final M221 Decision

Retain the following route as the best current qrels-free deployable endpoint:

- checkpoint: M218B seed 313 `blend_d0p5`
- runtime router: `family_dense_half`
- overlap-k: `24`
- fixed weight: `1.375`
- product contract: `32d INT4`
- result: MAP 0.305818 / Recall 0.615731
- deltas vs PPLX dense: MAP +0.004354 / Recall +0.004994

Also retain `family_soft_risk` as a secondary signal, but not as the primary
route. It tracks the same family-risk structure and remains competitive at
larger overlap windows, but it does not beat `family_dense_half` at the optimum.

Stop the following branches:

- additional RankT5-safe loss, distillation-weight, and pair sampling sweeps
- scalar hard-gate threshold and overlap-k tuning
- generic per-query strict agreement gates as the main product route

The remaining unsafe rows are still Arguana, FiQA, SciDocs, and TREC-COVID.
M221 improves macro quality and confirms a useful qrels-free family-risk signal,
but it does not discover a qrels-free feature that cleanly separates all unsafe
rows from helpful rows. The next structural step should therefore be a
row-aware pseudo-label objective rather than another runtime-router sweep.

Recommended M222 direction:

- keep the M218B seed 313 `blend_d0p5` head and M221 `family_dense_half` router
  as the deployment baseline
- construct a qrels-free or weakly supervised row-risk target from agreement
  failures: high teacher/student/dense agreement but dense-negative family
  outcomes should become explicit negative or down-weighted pseudo-labels
- train only a narrow risk head or row-aware pseudo-label head first, not a new
  full mini-vector family
- evaluate whether the new objective reduces FiQA/SciDocs/TREC-COVID damage
  while preserving the M221 macro win

If that objective cannot reduce unsafe rows or improve macro beyond M221, the
RankT5-safe weak-supervision branch should be considered near its practical
limit. Further progress would need a genuinely stronger teacher/source signal,
not more tuning around the current RankT5 cache.
