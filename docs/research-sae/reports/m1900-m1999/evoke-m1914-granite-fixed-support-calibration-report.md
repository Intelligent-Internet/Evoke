# M1914 Granite Fixed-Support Calibration Report

Date: 2026-07-12

Decision: **promote the global-power transform as a fixed-support research
milestone and authorize one fixed-budget support observability audit. Do not
promote per-term calibration or change support without M1915 evidence.**

## Question Answered

M1913 had a useful compact frontier but weaker head ranking than OpenSearch
sparse-v2. M1914 tested whether this gap partly came from global impact shape
rather than missing dimensions. It froze the Granite checkpoint, top-192
document support, top-50 query support, and exact BMP engine. Training used
only the pinned M1518 MS MARCO teacher rows and its query-disjoint validation
split; no BEIR row selected a branch or checkpoint.

The deployable selected transform is:

```text
query impact    = impact ^ 1.851864 * 0.696368
document impact = impact ^ 0.562796
```

No per-term scale survived selection. The transform preserves every active
dimension and can be folded directly into the output head.

## Frozen-Feature Audit

| Split | Rows | Mean overlap | Zero-overlap pairs | Parent positive top1 | Teacher positive top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10,000 | 21.336 | 0.902% | 0.5845 | 0.7882 |
| validation | 1,000 | 21.415 | 0.944% | 0.5830 | 0.7750 |

Most candidate pairs already share many active dimensions. Fixed support is
therefore not grossly disconnected; it has enough overlap for an impact-only
test. The remaining teacher gap is still large enough to justify a later
support audit, but only after calibration is exhausted.

## Query-Disjoint Selection

| Branch | Step | Pairwise | Teacher top1 | Positive top1 | KL | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M1913 baseline | 0 | 0.867125 | 0.629000 | 0.583000 | 0.827788 | anchor |
| global power | **1,000** | **0.876250** | **0.653000** | **0.611000** | **0.605501** | selected |
| diagonal | 200 | 0.869000 | 0.631000 | 0.586000 | 0.686492 | reject |
| combined | 200 | 0.875000 | 0.643000 | 0.602000 | 0.621457 | reject |

Global power improves all three heldout ranking metrics and KL. The diagonal
branch begins overfitting after step 200 and loses eligibility after step 300.
The combined branch peaks early and remains below global power. This is strong
evidence against another 50K-dimensional post-hoc term-weight search.

ClearML task: `72d822802e104fa09a0e555c1a93533a`.

## Complete FiQA Native Closure

| Surface | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 | Index | BMP p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1913 BMP | 0.352112 | 0.294284 | 0.665725 | 0.438845 | 0.861416 | 156.34 MB | 12.555 ms |
| M1914 BMP | **0.357677** | **0.298860** | **0.668468** | **0.443793** | 0.859227 | **156.34 MB** | **11.551 ms** |
| delta | +0.005565 | +0.004575 | +0.002743 | +0.004947 | -0.002189 | 0 | -1.004 ms |

Every support ID and nnz is unchanged. Exact returned scores and strict top100
boundaries remain `1.0`; quantized Recall retains `100.49%` of float Recall.
The same posting structure serializes to exactly the same index size, while
lower document power improves BMP p95 by 8.0%.

The transform improves all four primary ranking metrics over M1913, but it
still does not reach the OpenSearch balanced-quality baseline. It is an
improved Granite parent, not the final product checkpoint.

## Locked Broad3 Transfer

| Dataset | Delta NDCG | Delta MAP | Delta Recall | Delta MRR | Delta CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | -0.001469 | -0.000205 | -0.000714 | -0.000169 | -0.000714 |
| NFCorpus | **+0.015800** | **+0.009224** | **+0.001023** | **+0.031062** | +0.002188 |
| SciFact | +0.002704 | -0.001083 | 0.000000 | -0.000227 | -0.003333 |
| macro | **+0.005679** | **+0.002645** | **+0.000103** | **+0.010222** | -0.000620 |

The transform was applied unchanged to all three datasets. Exact BMP and
native cost gates pass for every parent/calibrated pair. All macro head metrics
improve and macro Recall is flat-positive. ArguAna has one small NDCG loss and
SciFact one small MAP loss, but neither row has the predeclared multi-metric
harm pattern or a Recall-floor violation.

The original M1660 auditor hardcoded the float-reference dataset name to
`fiqa`; the first transfer run stopped before producing a result. M1914 adds a
backward-compatible `--dataset` option, defaults it to `fiqa`, and covers both
default and explicit dataset selection in tests. No scoring or engine behavior
changed.

## Interpretation

M1914 establishes a real, low-capacity generalization result:

1. Granite query impacts were too flat relative to document impacts.
2. A global asymmetric monotonic transform recovers ranking without support
   growth, corpus statistics, BM25, or a reranker.
3. Token-specific calibration adds capacity but generalizes worse.
4. The gain transfers beyond FiQA, although small row-level tradeoffs remain.

This is the first justified local modification of the mature Granite parent.
It narrows the next question to support membership: can dimensions just below
the released 50/192 boundaries improve heldout ordering while keeping those
budgets fixed? M1915 may answer only that observability question first. It may
not train a new backbone or add lexical fusion.

## Artifacts

- Contract: `docs/research-sae/reports/m1900-m1999/ii42-m1914-granite-fixed-support-calibration-contract.md`
- Cache builder: `scripts/prepare_m1914_granite_calibration_cache.py`
- Trainer: `scripts/train_m1914_granite_fixed_support_calibrator.py`
- FiQA surface: `scripts/apply_m1914_granite_calibrator_surface.py`
- Broad3 surface: `scripts/prepare_m1914_granite_transfer_surfaces.py`
- Broad3 native summary: `scripts/summarize_m1914_granite_transfer_native.py`
- Remote FiQA summary: `ii42-m1914-granite-fixed-support-v1/native/summary.json`
- Remote transfer summary:
  `ii42-m1914-granite-fixed-support-v1/transfer-summary/summary.json`
