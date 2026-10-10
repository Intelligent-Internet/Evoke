# II-42 M225 Action-Distill Bridge Head Report

Date: 2026-09-29

## Purpose

M224 is the current supervised bridge milestone: the M214B seed301/313/337 RRF
ensemble distilled into a deployable `32d INT4` student reached MAP 0.331783 /
Recall 0.636344, with deltas versus PPLX dense of +0.030318 / +0.025607 and no
dense-unsafe rows.

M225 tests whether the remaining student-to-teacher gap is partly caused by the
training target. M223/M224 train on teacher score order inside the frozen
top256 window. M225 instead converts the same RRF teacher ensemble into the
teacher's deployment action: which candidates are promoted into the final
top100 and which baseline candidates are demoted after applying the fixed
rerank operator. The product head, quantization, candidate window, runtime
router, and training budget remain unchanged.

This remains a supervised bridge diagnostic. The teacher artifacts were trained
with qrels; M225 training does not read qrels directly, and qrels are otherwise
used only for final metrics.

## M225A Plan

Train one `32d INT4` student for each action policy:

- `promote_demote`: positives combine the teacher final top frontier with
  candidates promoted into teacher top100; negatives combine baseline-top100
  candidates demoted by the teacher with teacher low-frontier candidates.
- `frontier`: positives are the teacher final top frontier; negatives are the
  teacher final 96+ boundary candidates.

Shared settings:

- teacher: RRF ensemble over M214B seed301, seed313, and seed337
  `dim32_global.pt`
- teacher action operator: `weight=2.5`, `protected_head=0`,
  `correction_window=256`
- train/eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- action pair sampling: `positive_k=16`, `negative_start_rank=96`,
  `negative_k=96`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `2.0,2.25,2.5,2.75,3.0`
- overlap-k: `24`

Decision rule:

- If an action-distilled student improves macro MAP and Recall over M224A while
  staying dense-safe, retain it as the new supervised bridge endpoint and
  prepare seed/objective stability.
- If it improves MAP but creates dense-unsafe rows, inspect row losses before
  deciding whether M221/M220 router structure can recover safety.
- If it does not improve over M224A, keep M224A as the milestone and stop
  supervised bridge objective tuning; the next breakthrough should target a
  stronger qrels-free teacher source.

## M225A Result

M225A completed in 1629.6 seconds. Both action policies trained on the full six
training families, with 3722 query samples each:

| Action policy | Samples | Train families |
| --- | ---: | --- |
| `promote_demote` | 3722 | Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID |
| `frontier` | 3722 | Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID |

Best deployable variant:

| Action | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `promote_demote` | `m220c_joint` | 2.25 | 0.306845 | 0.636399 | +0.005380 | +0.025662 | Arguana, FiQA |

Top deployable alternatives were similar: `frontier,m220c_joint,w=2.5` reached
MAP 0.306640 / Recall 0.636474, with dMAP +0.005176 / dRecall +0.025738 and
the same Arguana/FiQA unsafe rows. No deployable M225A variant was dense-safe.

Comparison to retained milestones:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 retained qrels-free endpoint | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M224A supervised bridge endpoint | 0.331783 | 0.636344 | +0.030318 | +0.025607 | none |
| M225A best action-distill | 0.306845 | 0.636399 | +0.005380 | +0.025662 | Arguana, FiQA |

Best M225A row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | Safe vs dense |
| --- | ---: | ---: | --- |
| Arguana | -0.040645 | -0.000714 | no |
| FiQA | -0.009918 | -0.000184 | no |
| NFCorpus | +0.016753 | +0.043472 | yes |
| SciDocs | +0.010629 | +0.054883 | yes |
| SciFact | +0.037110 | +0.023333 | yes |
| TREC-COVID | +0.000069 | -0.001782 | yes |
| Webis-Touche2020 | +0.023664 | +0.060625 | yes |

Teacher source ceilings remain very high in macro MAP, but are not row-safe at
the stronger weights tested in M225A. The best teacher-source ceiling in this
run was `teacher_source,w=3.0`, MAP 0.392427 / Recall 0.635910, dMAP +0.090963
/ dRecall +0.025173, with Webis-Touche2020 unsafe.

## M225 Decision

Do not retain M225A. The action-distill objective preserves the high Recall band
but collapses MAP relative to M224A and introduces dense-unsafe Arguana/FiQA
rows. This means the M224 gain is not simply bottlenecked by raw score-order
training; the student needs the calibrated teacher score geometry or a richer
training target, not just final rerank action labels.

The retained supervised bridge endpoint remains M224A:

- `rrf` ensemble over M214B seed301/313/337 `dim32_global`
- student `32d INT4`
- router `m220c_joint`
- weight `2.5`
- MAP 0.331783 / Recall 0.636344
- deltas vs PPLX dense: +0.030318 / +0.025607
- no dense-unsafe rows

Stop supervised bridge objective tuning here. The next useful breakthrough
should move toward a stronger qrels-free teacher source or a richer
row-aware pseudo-label objective that preserves M224's score geometry while
approximating cross-seed supervised-teacher consensus.
