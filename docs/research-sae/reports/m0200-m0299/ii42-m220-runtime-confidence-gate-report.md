# II-42 M220 Runtime Confidence Gate Report

Date: 2026-09-28

## Purpose

M220 tests whether the seed-stable but thin qrels-free M218B RankT5-safe
mini-vector win can be improved at runtime without training another compact
representation head. The gate is qrels-free: it uses agreement among the
RankT5-safe teacher, the dense top-window teacher, and the student mini-vector
inside the frozen top256 candidate window. Qrels are used only for final
metrics.

The baseline continuation target is M218B mean:

| Route | MAP | Recall | dMAP vs PPLX dense | dRecall vs PPLX dense |
| --- | ---: | ---: | ---: | ---: |
| M218B mean | 0.302231 | 0.613316 | +0.000766 | +0.002579 |

## M220A Result

M220A evaluated 128 deployable `student_32d_int4` variants over the M218A and
M218B fixed checkpoints, gate policies
`fixed,teacher_dense_soft,student_teacher_soft,joint_soft,joint_hard_0p25,joint_hard_0p40,joint_hard_0p55,teacher_dense_hard_0p40`,
and fixed weights `0.75,1.0,1.25,1.5`.

Best variant:

| Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| M218A `blend_d0p5` | `joint_hard_0p40` | 1.25 | 0.304021 | 0.614299 | +0.002556 | +0.003562 | Arguana, FiQA, SciDocs, TREC-COVID | 0.839 |

Best fixed no-gate comparison inside the same M220 run:

| Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M218B seed 313 `blend_d0p5` | `fixed` | 1.25 | 0.302744 | 0.613498 | +0.001279 | +0.002761 | Arguana, FiQA, SciDocs, TREC-COVID |

The M220A best improves over M218B mean by about +0.001790 MAP and +0.000983
Recall while keeping the same unsafe row set. It also improves over the best
fixed no-gate variant in the same run by about +0.001277 MAP and +0.000801
Recall.

## Row Anatomy

Best M220A variant row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | dMAP vs fixed hybrid | Mean alpha | Safe vs dense |
| --- | ---: | ---: | ---: | ---: | --- |
| Arguana | -0.023667 | -0.002141 | -0.001945 | 0.864 | no |
| FiQA | -0.028397 | -0.039469 | +0.051585 | 0.974 | no |
| NFCorpus | +0.007394 | +0.005367 | +0.018065 | 0.907 | yes |
| SciDocs | -0.005639 | -0.014283 | +0.013695 | 0.971 | no |
| SciFact | +0.039406 | +0.016667 | +0.006166 | 0.970 | yes |
| TREC-COVID | -0.004316 | -0.004243 | +0.014432 | 0.960 | no |
| Webis | +0.033111 | +0.063034 | +0.002034 | 0.224 | yes |

The hard gate mainly helps macro quality by suppressing low-confidence Webis
correction heavily while keeping most corrections active on the other rows. It
does not solve row safety: Arguana, FiQA, SciDocs, and TREC-COVID remain below
the dense row floor. The important change is that qrels-free confidence gating
can improve the already seed-stable M218 macro win without expanding the unsafe
set.

## Interpretation

M220A is a positive continuation result. Runtime gating is not merely a safety
no-op; the `joint_hard_0p40` agreement criterion recovers additional macro MAP
and Recall from the same fixed 20-byte payload route. This means the current
RankT5-safe route is not dead yet.

The row failure pattern is still structural. FiQA and Arguana remain hard
because the pseudo-teacher/candidate window favors improvements against fixed
hybrid but not against PPLX dense. SciDocs and TREC-COVID have smaller but
persistent dense deficits. More generic RankT5-safe transfer tuning is unlikely
to fix those rows; M219A already failed there.

## Next Step

Run a narrow M220B threshold/weight refinement around the successful hard gate:

- fixed checkpoints: M218A `blend_d0p5`, M218B seed 313 `blend_d0p5`, M218B seed
  337 `blend_d0p5`, and M218B seed 301 `dense_top128_gate`
- gate policies: `joint_hard_0p30,joint_hard_0p35,joint_hard_0p40,joint_hard_0p45,joint_hard_0p50`
- fixed weights: `1.0,1.125,1.25,1.375,1.5`
- no new representation training
- no locked-test data

If M220B cannot improve on M220A without increasing unsafe rows, the
runtime-gate branch should stop. The next real breakthrough would need either a
stronger weak-supervision source or a row-aware pseudo-label objective rather
than more generic RankT5-safe tuning.

## M220B Result

M220B completed the planned hard-gate threshold/weight refinement with no new
representation training. It evaluated the same fixed M218A/M218B checkpoints
with `joint_hard_0p30,joint_hard_0p35,joint_hard_0p40,joint_hard_0p45,joint_hard_0p50`
and weights `1.0,1.125,1.25,1.375,1.5`.

Best by macro MAP:

| Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| M218B seed 301 `dense_top128_gate` | `joint_hard_0p35` | 1.50 | 0.304588 | 0.612891 | +0.003123 | +0.002154 | Arguana, FiQA, SciDocs, TREC-COVID | 0.769 |

Best variant that improves both primary deltas over M220A:

| Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| M218B seed 313 `blend_d0p5` | `joint_hard_0p35` | 1.25 | 0.304584 | 0.614357 | +0.003119 | +0.003620 | Arguana, FiQA, SciDocs, TREC-COVID | 0.862 |

The second variant is the cleaner continuation point because it improves both
macro MAP and Recall over M220A while preserving the same unsafe-row set. The
best-MAP variant is informative but trades off Recall, so it is not the primary
route choice.

Row deltas for the cleaner M220B continuation point:

| Dataset | dMAP | dRecall | dMAP vs fixed hybrid | Mean alpha | Safe vs dense |
| --- | ---: | ---: | ---: | ---: | --- |
| Arguana | -0.021658 | -0.002141 | +0.000065 | 0.904 | no |
| FiQA | -0.025451 | -0.036278 | +0.054531 | 0.992 | no |
| NFCorpus | +0.007379 | +0.006128 | +0.018049 | 0.957 | yes |
| SciDocs | -0.005437 | -0.012417 | +0.013897 | 0.991 | no |
| SciFact | +0.041771 | +0.016667 | +0.008531 | 0.987 | yes |
| TREC-COVID | -0.004064 | -0.003916 | +0.014683 | 0.960 | no |
| Webis | +0.029293 | +0.057301 | -0.001783 | 0.245 | yes |

M220B confirms that the successful hard gate was not a single-grid accident:
the better threshold is around `0.35`, not `0.40`. The route still does not
solve row safety, but it increases the qrels-free macro margin enough to justify
one more narrow generalization check over the overlap window used by the gate.

## M220C Plan

Run a narrow overlap-k generalization check:

- fixed checkpoints: same four M218A/M218B checkpoints
- gate policies: `joint_hard_0p30,joint_hard_0p35,joint_hard_0p40`
- fixed weights: `1.25,1.5`
- overlap-k values: `16,24,48,64`
- no new representation training
- no locked-test data

If M220C cannot improve on the M220B cleaner continuation point without
expanding unsafe rows, the runtime-gate branch should be considered saturated.

## M220C Result

M220C completed the overlap-k generalization check. It tested overlap windows
`16,24,48,64` with hard-gate thresholds `0.30,0.35,0.40` and weights `1.25,1.5`.

Best per overlap-k:

| overlap-k | Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | M218B seed 313 `blend_d0p5` | `joint_hard_0p30` | 1.25 | 0.304520 | 0.614649 | +0.003055 | +0.003912 | Arguana, FiQA, SciDocs, TREC-COVID | 0.856 |
| 24 | M218B seed 313 `blend_d0p5` | `joint_hard_0p30` | 1.25 | 0.304769 | 0.615656 | +0.003304 | +0.004919 | Arguana, FiQA, SciDocs, TREC-COVID | 0.870 |
| 48 | M218B seed 301 `dense_top128_gate` | `joint_hard_0p40` | 1.50 | 0.304599 | 0.613249 | +0.003135 | +0.002512 | Arguana, FiQA, SciDocs, TREC-COVID | 0.808 |
| 64 | M218B seed 301 `dense_top128_gate` | `joint_hard_0p40` | 1.50 | 0.304575 | 0.612767 | +0.003111 | +0.002030 | Arguana, FiQA, SciDocs, TREC-COVID | 0.917 |

M220C improves both primary deltas over M220B's cleaner continuation point. The
new best is:

| Checkpoint | overlap-k | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M218B seed 313 `blend_d0p5` | 24 | `joint_hard_0p30` | 1.25 | 0.304769 | 0.615656 | +0.003304 | +0.004919 | Arguana, FiQA, SciDocs, TREC-COVID |

This indicates the gate is not an overlap-k=32 artifact. The strongest signal is
top-window-front agreement: `overlap_k=24` improves Recall while preserving the
same unsafe-row set. Larger overlap windows retain MAP but give back Recall.

## M220D Plan

Run one last local refinement around the M220C optimum:

- fixed checkpoints: same four M218A/M218B checkpoints
- gate policies: `joint_hard_0p25,joint_hard_0p30,joint_hard_0p35`
- fixed weights: `1.125,1.25,1.375`
- overlap-k values: `20,24,28`
- no new representation training
- no locked-test data

If M220D does not improve on M220C without expanding unsafe rows, stop the
runtime-gate branch. Further progress should move to a row-aware objective or a
stronger weak-supervision source instead of more scalar gate tuning.

## M220D Result

M220D completed the final local refinement around the M220C optimum. It tested
overlap-k values `20,24,28`, thresholds `0.25,0.30,0.35`, and weights
`1.125,1.25,1.375`.

Best per overlap-k:

| overlap-k | Checkpoint | Gate | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Mean alpha |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 20 | M218B seed 313 `blend_d0p5` | `joint_hard_0p30` | 1.25 | 0.304723 | 0.615470 | +0.003258 | +0.004733 | Arguana, FiQA, SciDocs, TREC-COVID | 0.876 |
| 24 | M218B seed 313 `blend_d0p5` | `joint_hard_0p30` | 1.25 | 0.304769 | 0.615656 | +0.003304 | +0.004919 | Arguana, FiQA, SciDocs, TREC-COVID | 0.870 |
| 28 | M218B seed 313 `blend_d0p5` | `joint_hard_0p30` | 1.25 | 0.304389 | 0.614964 | +0.002924 | +0.004227 | Arguana, FiQA, SciDocs, TREC-COVID | 0.889 |

M220D did not improve over M220C. Its best variant is the same M220C optimum:
overlap-k `24`, M218B seed 313 `blend_d0p5`, `joint_hard_0p30`, weight `1.25`.

Conclusion: runtime hard-gate tuning is saturated. The RankT5-safe weak
supervision route remains a real qrels-free macro win over PPLX dense, but the
remaining unsafe rows are stable and structural. Further scalar gate tuning is
not justified. The next useful work should be a new row-aware pseudo-label
objective or a stronger weak-supervision source, not another RankT5-safe
threshold/weight sweep.
