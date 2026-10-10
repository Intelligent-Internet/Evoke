# II-42 M229 Query-Rich Score-Geometry Bridge

## Status

M229A and M229B complete. M229 is promoted as the current query-performance
endpoint before query-cost control.

## Motivation

M226A/B established the current supervised score-geometry milestone under the
32d INT4 product contract. The best M226B student-seed sweep reached mean best
MAP 0.360175 and Recall 0.639834, with deltas vs PPLX dense of +0.058710 MAP
and +0.029097 Recall, and no dense-unsafe rows.

M229 asks whether query-side capacity, not document-side product storage, is
now limiting quality. It keeps the document representation deployable as 32d
INT4, but allows a richer online query tower while preserving the M226
continuous mean-z score-geometry objective.

## M229A Design

- Teacher source: M214B seed301/313/337 global heads as the same supervised
  mean-z teacher ensemble used by M226.
- Product contract: document-side 32d INT4 mini-vector remains unchanged.
- Query-side variants: `query_rich_512x2` and `query_rich_768x3`.
- Objective: `meanz_pair_meanz_score`.
- Training: 16 epochs, `distill_weight=0.2`,
  `score_geometry_weight=0.2`, `positive_k=16`,
  `negative_start_rank=16`, `negative_k=128`.
- Runtime routers: `fixed`, `family_dense_half`, `m220c_joint`.
- Fixed weights: `2.5,3.0,3.25,3.5,3.75,4.0`.

## Decision Criteria

Promote M229 only if a deployable variant is dense-safe and improves over the
M226B stable endpoint on macro MAP and Recall, or clearly improves MAP without
material Recall regression. If it improves, the next step is M229B seed
stability for the winning query architecture and router/weight choice.

If M229A fails to improve or creates dense-unsafe rows at the improved point,
retain M226B as the query-performance endpoint and move to query-cost control
using the M226B/M229 evidence.

## Current Best Baselines

| Endpoint | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 qrels-free runtime route | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M226A supervised score geometry | 0.358324 | 0.640065 | +0.056859 | +0.029328 | none |
| M226B mean stable endpoint | 0.360175 | 0.639834 | +0.058710 | +0.029097 | none |

## Results

### M229A

M229A completed on lambda2 GPU0 at 2026-10-01 11:29:03 America/Toronto.

Best dense-safe deployable variant:

- Architecture: `query_rich_512x2`
- Runtime router: `m220c_joint`
- Fixed weight: `4.0`
- MAP: 0.364824
- Recall: 0.639911
- Delta vs PPLX dense: MAP +0.063360, Recall +0.029174
- Dense-unsafe rows: none

This improves over the M226B stable mean endpoint:

- MAP gain over M226B mean: +0.004649
- Recall gain over M226B mean: +0.000077

It also improves MAP over the best M226B individual seed337 result by about
+0.004090, while Recall is essentially flat relative to the strongest M226
Recall point. The result supports the hypothesis that query-side capacity was
still limiting MAP after M226, while the document-side 32d INT4 contract can
remain fixed.

Top dense-safe M229A deployable variants:

| Rank | Architecture | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `query_rich_512x2` | `m220c_joint` | 4.0 | 0.364824 | 0.639911 | +0.063360 | +0.029174 |
| 2 | `query_rich_512x2` | `m220c_joint` | 3.75 | 0.363812 | 0.639938 | +0.062347 | +0.029201 |
| 3 | `query_rich_768x3` | `m220c_joint` | 4.0 | 0.363618 | 0.639853 | +0.062153 | +0.029116 |
| 4 | `query_rich_768x3` | `m220c_joint` | 3.75 | 0.362772 | 0.639999 | +0.061307 | +0.029262 |
| 5 | `query_rich_512x2` | `m220c_joint` | 3.5 | 0.362627 | 0.639774 | +0.061162 | +0.029037 |

Training summaries:

| Architecture | Query parameters | Doc parameters | Final pair loss | Final geometry loss |
| --- | ---: | ---: | ---: | ---: |
| `query_rich_512x2` | 974272 | 168352 | 0.004303 | 0.035093 |
| `query_rich_768x3` | 2165952 | 168352 | 0.006102 | 0.035459 |

Interpretation:

The smaller query-rich tower is better than the larger one, which suggests the
gain is not from indiscriminate capacity. The useful move is a moderate
query-side residual adapter plus the existing `m220c_joint` runtime structure.
Because the winning point is at the top edge of the M229A weight grid, M229B
uses a narrow stability and boundary check around this setting.

### M229B

M229B completed seed stability for `query_rich_512x2` with the same mean-z
score-geometry objective, 32d INT4 document-side contract, and the M214B
seed301/313/337 teacher ensemble.

Best dense-safe deployable results:

| Seed | Architecture | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 301 | `query_rich_512x2` | `m220c_joint` | 4.5 | 0.367166 | 0.640347 | +0.065701 | +0.029611 |
| 313 | `query_rich_512x2` | `m220c_joint` | 4.5 | 0.366231 | 0.640258 | +0.064767 | +0.029521 |
| 337 | `query_rich_512x2` | `m220c_joint` | 4.5 | 0.365847 | 0.639380 | +0.064383 | +0.028643 |
| Mean | `query_rich_512x2` | `m220c_joint` | 4.5 | 0.366415 | 0.639995 | +0.064950 | +0.029258 |

All three seeds are dense-safe. The M229B mean improves over the M226B mean by
+0.006240 MAP and +0.000161 Recall. The best M229B seed301 point improves over
the best M226B seed337 point by +0.006432 MAP and +0.000865 Recall.

## Milestone Snapshot

M229B is the saved query-performance milestone before query-cost control.

- Retained endpoint: `query_rich_512x2 + m220c_joint,w=4.5`.
- Stability: student seeds 301, 313, and 337 are all dense-safe.
- Mean result: MAP 0.366415 / Recall 0.639995.
- Mean delta vs PPLX dense: MAP +0.064950 / Recall +0.029258.
- Best single seed: seed301, MAP 0.367166 / Recall 0.640347,
  dMAP +0.065701 / dRecall +0.029611.
- Product constraint preserved: document-side correction remains 32d INT4.
- Query-side cost is intentionally not optimized in this milestone; it is the
  next phase target.

Artifact roots:

- M229A: `/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229a-query-rich-score-geometry-32d-int4-epoch16-gpu0-v1`
- M229B seed301: `/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-301-query-rich-score-geometry-32d-int4-gpu0-v1`
- M229B seed313: `/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-313-query-rich-score-geometry-32d-int4-gpu0-v1`
- M229B seed337: `/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-337-query-rich-score-geometry-32d-int4-gpu0-v1`

## Conclusion

M229 confirms that the strongest remaining query-performance gain before cost
control comes from query-side capacity under the M226 score-geometry target, not
from another qrels-free teacher or scalar router sweep. The retained endpoint is
`query_rich_512x2 + m220c_joint,w=4.5`, with document-side 32d INT4 preserved
and no dense-unsafe rows across the seed check.

The next phase should freeze this as the high-quality reference and start
query-cost control: distill or approximate the query-rich tower back toward a
cheaper online query path while guarding MAP, Recall, and dense-safety against
the M229B mean.
