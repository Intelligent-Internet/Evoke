# II-42 M224 Teacher-Ensemble Bridge Head Report

Date: 2026-09-29

## Purpose

M223 is now a milestone diagnostic: all three M214B supervised global teacher
artifacts produce dense-safe deployable `32d INT4` students, with mean MAP
0.326477 / Recall 0.635051 and mean deltas versus PPLX dense of +0.025012 /
+0.024314. That result shows the product head can carry much stronger
dense-safe behavior when the teacher signal is well aligned.

M224 asks whether that milestone can be guarded or enlarged by combining the
three stable M214B teacher artifacts into an ensemble teacher before student
distillation. This is a structural probe, not another RankT5 loss or scalar
gate sweep.

This remains a supervised bridge diagnostic. The teacher artifacts were trained
with qrels; M224 training does not read qrels directly, and qrels are otherwise
used only for final metrics.

## M224A Plan

Train one `32d INT4` student per ensemble policy:

- `mean_z`: z-score each teacher's per-query top256 scores, then average them
- `rrf`: convert each teacher's top256 ranking into reciprocal-rank scores,
  then average them

Shared settings:

- teacher artifacts: M214B seed301, seed313, and seed337 `dim32_global.pt`
- train/eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `1.5,1.75,2.0,2.25,2.5`
- overlap-k: `24`

Decision rule:

- If an ensemble student improves over the M223 single-teacher mean while
  staying dense-safe, keep it as the new supervised bridge endpoint.
- If no ensemble student improves, retain M223 single-teacher bridge as the
  milestone and stop before wider supervised-teacher tuning.
- If only unsafe variants improve MAP, inspect row losses before deciding
  whether a safety router can recover them.

## M224A Result

M224A completed both ensemble policies. The best deployable variant is dense
safe and improves over both the M223 single-teacher mean and the best individual
M223 teacher seed.

Best deployable variant:

| Ensemble | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rrf` | `m220c_joint` | 2.50 | 0.331783 | 0.636344 | +0.030318 | +0.025607 | none |

Comparison to the M223 bridge milestone:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M223 mean best dense-safe | 0.326477 | 0.635051 | +0.025012 | +0.024314 | none |
| M223 best seed | 0.327294 | 0.635341 | +0.025830 | +0.024604 | none |
| M224A best | 0.331783 | 0.636344 | +0.030318 | +0.025607 | none |

Best deployable row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | dMAP vs fixed hybrid | Safe vs dense |
| --- | ---: | ---: | ---: | --- |
| Arguana | +0.011652 | -0.000714 | +0.033375 | yes |
| FiQA | +0.029656 | -0.000183 | +0.109638 | yes |
| NFCorpus | +0.041516 | +0.045360 | +0.052187 | yes |
| SciDocs | +0.054513 | +0.056283 | +0.073847 | yes |
| SciFact | +0.057246 | +0.023333 | +0.024005 | yes |
| TREC-COVID | +0.000908 | -0.001611 | +0.019655 | yes |
| Webis-Touche2020 | +0.016734 | +0.056781 | -0.014342 | yes |

Teacher ensemble ceilings:

| Ensemble | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mean_z` | 1.75 | 0.383460 | 0.641452 | +0.081995 | +0.030716 | none |
| `rrf` | 2.50 | 0.385680 | 0.637231 | +0.084215 | +0.026494 | Webis-Touche2020 |

The best deployable student comes from the `rrf` ensemble, even though the
best dense-safe teacher ceiling comes from `mean_z`. This suggests that the
student benefits from rank-consensus positives more than from calibrated
score-shape averaging. The RRF teacher likely supplies cleaner top/bottom
pairwise ordering for the product head, while `mean_z` preserves more teacher
score magnitude but transfers less MAP into the student.

`m220c_joint` matters again in the ensemble setting: `rrf,m220c_joint,w=2.5`
beats `rrf,family_dense_half,w=2.5` by +0.002189 MAP and also beats the fixed
RRF routes. That indicates router structure is useful as a safety-preserving
amplifier once teacher agreement is based on cross-seed consensus rather than a
single teacher.

## M224 Decision

Keep M224A as the new supervised bridge endpoint:

- `rrf` ensemble over M214B seed301/313/337 `dim32_global`
- student `32d INT4`
- router `m220c_joint`
- weight `2.5`
- MAP 0.331783 / Recall 0.636344
- deltas vs PPLX dense: +0.030318 / +0.025607
- no dense-unsafe rows

This improves over the M223 single-teacher mean by +0.005306 MAP and +0.001293
Recall, and over the best individual M223 seed by +0.004488 MAP and +0.001003
Recall. Stop this supervised-teacher branch here. The remaining gap to the
M214/M215 supervised ceilings should be attacked by finding a stronger
qrels-free teacher source or a row-aware pseudo-label objective that mimics
cross-seed supervised-teacher consensus, not by more scalar tuning.

## M224B Plan

After M225A showed that action-only distillation does not retain M224A's MAP or
row safety, M224B guards the retained milestone before any further structural
work. It repeats only the winning M224A teacher construction, not the discarded
M225 objective.

Run student seed stability for the RRF ensemble:

- teacher artifacts: M214B seed301, seed313, and seed337 `dim32_global.pt`
- ensemble policy: `rrf`
- student seeds: `301`, `313`, `337`
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `1.5,1.75,2.0,2.25,2.5`
- overlap-k: `24`

Decision rule:

- If all three student seeds remain dense-safe and stay near or above the M223
  single-teacher mean, promote M224A/B as the stable supervised bridge endpoint
  and continue to a narrow M226 score-geometry objective probe.
- If the result is seed-unstable or loses row safety, retain M223/M224A as
  diagnostics only and stop before building on M224A.

## M224B Result

M224B completed the three student-seed stability runs for the M224A RRF
ensemble objective. All three runs stayed dense-safe and all retained a large
margin over the M223 single-teacher mean.

Best dense-safe deployable variant per student seed:

| Student seed | Ensemble | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 301 | `rrf` | `m220c_joint` | 2.50 | 0.330661 | 0.635988 | +0.029196 | +0.025252 | none |
| 313 | `rrf` | `m220c_joint` | 2.50 | 0.330082 | 0.636138 | +0.028617 | +0.025401 | none |
| 337 | `rrf` | `m220c_joint` | 2.50 | 0.330315 | 0.635846 | +0.028850 | +0.025109 | none |

Mean best dense-safe M224B result:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M223 mean best dense-safe | 0.326477 | 0.635051 | +0.025012 | +0.024314 | none |
| M224A best | 0.331783 | 0.636344 | +0.030318 | +0.025607 | none |
| M224B student-seed mean | 0.330353 | 0.635991 | +0.028888 | +0.025254 | none |

M224B validates the endpoint rather than replacing M224A. The exact seed-224
M224A run remains the best single supervised-bridge checkpoint, but M224B shows
that the RRF ensemble objective reliably produces dense-safe gains near the same
band. This satisfies the continuation criterion for the narrower M226
score-geometry probe.
