# II-42 M223 Supervised-Teacher Bridge Head Report

Date: 2026-09-29

## Purpose

M222 stopped the weak row-risk pseudo-label branch: it did not improve over the
retained M221 runtime endpoint and did not reduce the unsafe-row set. M223 is a
bridge diagnostic rather than another RankT5-safe sweep. It uses a successful
qrels-supervised M214 global checkpoint as a teacher artifact and asks whether
the current `32d INT4` product head plus the M221 router can carry that kind of
correction signal.

This is not a qrels-free product benchmark. The M214 teacher checkpoint was
trained with qrels. M223 training does not read qrels directly; qrels are used
only for final metrics.

Retained qrels-free endpoint before M223:

| Route | MAP | Recall | dMAP vs PPLX dense | dRecall vs PPLX dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 `family_dense_half` | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |

## M223A Plan

Distill the M214B seed313 global checkpoint into a fresh student initialized
from M210A, using the same product contract and pairwise training machinery as
M218/M222.

Settings:

- teacher artifact: M214B seed313 `dim32_global.pt`
- train/eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `1.0,1.25,1.375,1.5,2.0`
- overlap-k: `24`

Decision rule:

- If the student recovers a large fraction of M214/M215's MAP and Recall gains,
  the weak-supervision source is the main bottleneck, and the next useful route
  is a stronger qrels-free proxy for supervised behavior.
- If the teacher artifact is strong but the student still collapses near M221,
  the compact-head distillation objective is the bottleneck.
- If the M221 router materially improves the distilled student, keep the router
  as part of the retained runtime structure for future teacher sources.

## M223A Result

M223A completed with M214B seed313 `dim32_global.pt` as the teacher artifact.

Best deployable student:

| Source | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `student_32d_int4` | `fixed` | 2.00 | 0.326363 | 0.635114 | +0.024899 | +0.024377 | none |

Best non-deployable teacher ceilings under the M223 harness:

| Source | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `teacher_raw_ceiling` | 2.00 | 0.389749 | 0.640604 | +0.088284 | +0.029867 | none |
| `supervised_teacher_32d_int4` | 2.00 | 0.388821 | 0.640014 | +0.087356 | +0.029277 | none |

The bridge student is a clear positive diagnostic. It is dense-safe under the
`-0.002` row floor and improves substantially over the retained M221 qrels-free
endpoint:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 retained endpoint | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M223A bridge student | 0.326363 | 0.635114 | +0.024899 | +0.024377 | none |

Best deployable row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | Safe vs dense |
| --- | ---: | ---: | --- |
| Arguana | +0.009372 | -0.000714 | yes |
| FiQA | +0.016813 | +0.003766 | yes |
| NFCorpus | +0.038880 | +0.055923 | yes |
| SciDocs | +0.044934 | +0.060233 | yes |
| SciFact | +0.064481 | +0.026667 | yes |
| TREC-COVID | +0.000335 | -0.000967 | yes |
| Webis-Touche2020 | -0.000525 | +0.025732 | yes |

The student recovers roughly 28.5% of the compact teacher's MAP delta and 83.3%
of its Recall delta. This points away from an absolute `32d INT4` capacity
blocker: the product head can carry much stronger, dense-safe behavior when the
teacher signal is well aligned. The main bottleneck is the weak-supervision
source, not the runtime scalar/router branch.

The M221 router does not improve the bridge student. The best route is fixed
weight 2.00; `m220c_joint,w=1.5` and `family_dense_half,w=1.5` remain strong and
dense-safe, but both are below fixed weight on MAP.

## M223 Decision

M223A satisfies the continuation condition for a narrow bridge generalization
check. The next probe should test whether the result is specific to M214B
seed313 or generalizes across the other M214B supervised global teacher
artifacts, without introducing a new objective or broader branch.

## M223B Plan

Run the same bridge-distillation protocol against the remaining M214B global
teacher artifacts:

- M214B seed301 `dim32_global.pt`
- M214B seed337 `dim32_global.pt`

Keep all M223A settings unchanged: `32d INT4`, 16 epochs, distill weight `0.2`,
pair sampling `positive_k=8`, `negative_start_rank=16`, `negative_k=64`, fixed
weights `1.0,1.25,1.375,1.5,2.0`, router policies `fixed`,
`family_dense_half`, and `m220c_joint`, overlap-k `24`, joint threshold `0.30`.

Decision rule:

- If both additional teacher artifacts produce dense-safe deployable students
  with large positive macro dMAP and dRecall, promote the supervised bridge as
  a stable structural diagnostic and stop this branch before designing a new
  weak-supervision source.
- If the result is unstable across teacher artifacts, inspect which rows lose
  safety and whether the M221 router reduces that instability.

## M223B Result

M223B completed the remaining M214B global teacher artifacts with the unchanged
M223A protocol.

Best dense-safe deployable students:

| Teacher artifact | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M214B seed301 `dim32_global` | `m220c_joint` | 2.00 | 0.327294 | 0.635341 | +0.025830 | +0.024604 | none |
| M214B seed337 `dim32_global` | `fixed` | 2.00 | 0.325773 | 0.634698 | +0.024308 | +0.023961 | none |

M223 bridge seed summary, including M223A seed313:

| Teacher seed | Best dense-safe route | MAP | Recall | dMAP vs dense | dRecall vs dense |
| --- | --- | ---: | ---: | ---: | ---: |
| 301 | `m220c_joint,w=2.0` | 0.327294 | 0.635341 | +0.025830 | +0.024604 |
| 313 | `fixed,w=2.0` | 0.326363 | 0.635114 | +0.024899 | +0.024377 |
| 337 | `fixed,w=2.0` | 0.325773 | 0.634698 | +0.024308 | +0.023961 |
| Mean | best dense-safe per seed | 0.326477 | 0.635051 | +0.025012 | +0.024314 |

Best dense-safe row deltas for M223B seed301:

| Dataset | dMAP | dRecall | Safe vs dense |
| --- | ---: | ---: | --- |
| Arguana | +0.006794 | -0.000714 | yes |
| FiQA | +0.018366 | -0.000359 | yes |
| NFCorpus | +0.033134 | +0.040698 | yes |
| SciDocs | +0.043819 | +0.054833 | yes |
| SciFact | +0.060650 | +0.023333 | yes |
| TREC-COVID | +0.000139 | -0.001849 | yes |
| Webis-Touche2020 | +0.017906 | +0.056288 | yes |

Best dense-safe row deltas for M223B seed337:

| Dataset | dMAP | dRecall | Safe vs dense |
| --- | ---: | ---: | --- |
| Arguana | +0.008629 | -0.000714 | yes |
| FiQA | +0.019894 | +0.003976 | yes |
| NFCorpus | +0.037261 | +0.054249 | yes |
| SciDocs | +0.046635 | +0.060633 | yes |
| SciFact | +0.058022 | +0.026667 | yes |
| TREC-COVID | +0.000810 | -0.001496 | yes |
| Webis-Touche2020 | -0.001097 | +0.024410 | yes |

The bridge result is stable across all three M214B teacher seeds. Every seed
produces a dense-safe deployable student with roughly +0.024 to +0.026 macro
MAP over PPLX dense and roughly +0.024 macro Recall over PPLX dense.

The router signal is secondary in this branch. For seed301, `m220c_joint,w=2.0`
beats fixed `w=2.0` and removes the Webis unsafe row. For seeds313 and 337,
fixed `w=2.0` is the best dense-safe choice. The retained structural lesson is
therefore not a new scalar/router sweep; it is that a successful teacher source
can be compressed into the existing product head with dense-safe behavior.

## Final M223 Decision

Promote M223 as a stable structural diagnostic and stop this branch before
launching a broader new run. The evidence now separates the bottlenecks:

- The `32d INT4` product head is not the limiting factor.
- Runtime routing can help tactically, but it is not the main breakthrough
  mechanism once the teacher signal is strong.
- RankT5/dense/baseline weak-supervision sources are the limiting factor in the
  current qrels-free route.

The next breakthrough should be designed around a stronger weak-supervision
source or an explicit row-aware pseudo-label objective that approximates the
M214/M215 success pattern without direct qrels labels. More RankT5-safe scalar
tuning is not justified by M217-M223.
