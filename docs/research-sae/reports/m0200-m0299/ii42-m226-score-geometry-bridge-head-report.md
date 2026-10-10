# II-42 M226 Score-Geometry Bridge Head Report

Date: 2026-09-29

## Purpose

M224A established a new supervised bridge milestone: an RRF ensemble of M214B
seed301/313/337 global teachers distilled into a deployable `32d INT4` student
reached MAP 0.331783 / Recall 0.636344, with deltas versus PPLX dense of
+0.030318 / +0.025607 and no dense-unsafe rows.

M225A tested whether the remaining student-to-teacher gap could be closed by
distilling the teacher's deployment action. It did not preserve MAP or row
safety, so action-only labels are not a good next target.

M226 tests a narrower structural hypothesis: keep M224's pairwise ordering, but
also preserve the teacher ensemble's continuous score geometry inside the
frozen top256 window. This is meant to retain more of the teacher signal without
turning the experiment into another scalar or loss sweep.

This remains a supervised bridge diagnostic. The teacher artifacts were trained
with qrels; M226 training does not read qrels directly, and qrels are otherwise
used only for final metrics.

## M226A Plan

Train two `32d INT4` students:

- `rrf_pair_meanz_score`: pairwise positives/negatives from the M214B RRF
  ensemble, plus a normalized score-geometry loss against the M214B `mean_z`
  ensemble.
- `meanz_pair_meanz_score`: pairwise positives/negatives and score-geometry
  loss both from the M214B `mean_z` ensemble.

Shared settings:

- teacher artifacts: M214B seed301, seed313, and seed337 `dim32_global.pt`
- train/eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- score-geometry weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `2.0,2.25,2.5,2.75,3.0`
- overlap-k: `24`

Decision rule:

- If M226A improves both MAP and Recall over M224A while remaining dense-safe,
  promote M226 as the new supervised bridge endpoint.
- If M226A fails to improve over M224A or introduces dense-unsafe rows, retain
  M224A/B as the milestone and stop supervised bridge objective tuning.

## M226A Result

M226A completed as a strong positive result. The score-geometry objective
materially improves over M224A while preserving dense safety.

Best deployable variant:

| Objective | Pair | Score | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `meanz_pair_meanz_score` | `mean_z` | `mean_z` | `m220c_joint` | 3.00 | 0.358324 | 0.640065 | +0.056859 | +0.029328 | none |

Comparison to retained bridge endpoints:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M223 mean best dense-safe | 0.326477 | 0.635051 | +0.025012 | +0.024314 | none |
| M224B student-seed mean | 0.330353 | 0.635991 | +0.028888 | +0.025254 | none |
| M224A best | 0.331783 | 0.636344 | +0.030318 | +0.025607 | none |
| M226A best | 0.358324 | 0.640065 | +0.056859 | +0.029328 | none |

M226A improves over M224A by +0.026541 MAP and +0.003721 Recall. The gain is
not a scalar-router artifact: the top variants are all score-geometry students,
and the best result uses the `mean_z` teacher ensemble for both pair direction
and continuous score-shape preservation.

Best row deltas versus PPLX dense:

| Dataset | MAP | Recall | dMAP vs dense | dRecall vs dense | Safe vs dense |
| --- | ---: | ---: | ---: | ---: | --- |
| Arguana | 0.378251 | 0.999286 | +0.073631 | -0.000714 | yes |
| FiQA | 0.521496 | 0.830160 | +0.063847 | +0.001291 | yes |
| NFCorpus | 0.228805 | 0.386046 | +0.060794 | +0.059082 | yes |
| SciDocs | 0.240524 | 0.552917 | +0.077670 | +0.057083 | yes |
| SciFact | 0.806110 | 0.986667 | +0.092487 | +0.023333 | yes |
| TREC-COVID | 0.149436 | 0.172517 | +0.010653 | +0.005163 | yes |
| Webis-Touche2020 | 0.183644 | 0.552861 | +0.018931 | +0.060056 | yes |

The non-deployable mean-z score teacher ceiling remains far above the student
on MAP, reaching MAP 0.411304 / Recall 0.637535 at weight 3.0, but it is
dense-unsafe on Webis-Touche2020. The student appears to transfer a safer
compressed version of that score geometry: it captures a large fraction of the
MAP gain while keeping every row above the dense-safety floor.

## M226 Decision

Promote M226A as the new supervised bridge endpoint:

- objective: `meanz_pair_meanz_score`
- product contract: `32d INT4`
- router: `m220c_joint`
- fixed weight: `3.0`
- MAP 0.358324 / Recall 0.640065
- deltas vs PPLX dense: +0.056859 / +0.029328
- dense-unsafe rows: none

This result changes the diagnosis from M224/M225. The useful target is not the
teacher's final promote/demote action; it is the teacher ensemble's continuous
score geometry. The next structural work should guard M226A with seed/objective
stability and then look for a qrels-free teacher source that can reproduce this
mean-z score geometry, rather than continuing action distillation or scalar
router tuning.

## M226B Plan

M226B guards the new M226A endpoint before using it as a design target for
qrels-free work. It repeats only the winning score-geometry objective and does
not change the architecture, optimizer, candidate protocol, or product
contract.

Run student seed stability for the M214B `mean_z` teacher geometry:

- objective: `meanz_pair_meanz_score`
- teacher artifacts: M214B seed301, seed313, and seed337 `dim32_global.pt`
- student seeds: `301`, `313`, `337`
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- score-geometry weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `2.5,2.75,3.0,3.25,3.5`
- overlap-k: `24`

Decision rule:

- If all three seeds remain dense-safe and stay near the M226A band, promote
  M226A/B as a stable supervised bridge endpoint and continue to a qrels-free
  score-geometry teacher preflight.
- If the result is seed-unstable or higher weights create row-safety failures,
  retain M226A as a strong diagnostic but do not build a broader branch on it.

## M226B Result

M226B completed all three student-seed stability runs. The result is stable and
slightly improves the retained endpoint: every seed remains dense-safe, and all
best variants move from the original M226A `w=3.0` point to `w=3.5`.

Best dense-safe deployable variant per student seed:

| Student seed | Objective | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 301 | `meanz_pair_meanz_score` | `m220c_joint` | 3.50 | 0.360162 | 0.639893 | +0.058697 | +0.029156 | none |
| 313 | `meanz_pair_meanz_score` | `m220c_joint` | 3.50 | 0.359628 | 0.640127 | +0.058163 | +0.029391 | none |
| 337 | `meanz_pair_meanz_score` | `m220c_joint` | 3.50 | 0.360734 | 0.639482 | +0.059269 | +0.028745 | none |

Mean best dense-safe M226B result:

| Route | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M226A best | 0.358324 | 0.640065 | +0.056859 | +0.029328 | none |
| M226B student-seed mean | 0.360175 | 0.639834 | +0.058710 | +0.029097 | none |

M226B confirms that score-geometry distillation is not a single-seed artifact.
The MAP improvement is stable, the recall remains in the M226A band, and the
larger `w=3.5` setting does not create dense-unsafe rows. Use M226B as the
strongest supervised bridge endpoint for downstream diagnosis:

- objective: `meanz_pair_meanz_score`
- router: `m220c_joint`
- fixed weight: `3.5`
- mean MAP 0.360175 / Recall 0.639834
- mean deltas vs PPLX dense: +0.058710 / +0.029097
- dense-unsafe rows: none

The next step should not be more supervised bridge tuning. The right follow-up
is to ask which qrels-free score sources, if any, have a top256 score geometry
that can approximate the M226 supervised mean-z teacher strongly enough to be
worth distilling.
