# II-42 M1933 Semantic Budget Frontier Report

Date: 2026-07-13

Decision: **retain `b1.125` as a fixed research candidate and authorize unseen
transfer; do not promote it as the product default yet.**

## Question Answered

M1932 exposed unequal pruning pressure: the M1931 `1.0x lexical` semantic
budget retains about 44% of full M1914 FiQA support and 55% of ArguAna, but
75-78% of NFCorpus and SciFact. M1933 tested whether the product budget itself
was too tight while freezing allocation, impacts, M1931 calibration, lexical
postings, and the additive one-index score.

## Frontier

| Semantic budget | NDCG@10 | MAP@100 | R@100 | MRR@20 | CUB@1000 | Mean touches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000x | 0.458584 | 0.356845 | 0.722599 | 0.493992 | 0.857603 | 105,870 |
| **1.125x** | **0.460068** | **0.358209** | **0.726169** | **0.495059** | **0.859572** | 114,628 |
| 1.250x | 0.461714 | 0.359224 | 0.724514 | 0.497252 | 0.860160 | 122,777 |
| 1.375x | 0.462506 | 0.360178 | 0.724870 | 0.498955 | 0.860574 | 130,258 |
| full | 0.463582 | 0.360854 | 0.727196 | 0.499312 | 0.861146 | 170,965 |

`b1.125` is the smallest point that improves every macro quality metric. Its
semantic posting count rises by exactly 12.5%, total lexical-plus-semantic
postings rise from `2.0x` to `2.125x` lexical, and mean semantic posting
touches rise by 8.27%. Larger budgets improve head metrics further but Recall
is non-monotonic, so training depth or a larger scalar is not justified.

## Leave-One-Dataset-Out

All four folds independently selected `b1.125` from the other three rows.

| Held out | Delta NDCG | Delta MAP | Delta Recall | Delta MRR | Delta CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiQA | +0.003736 | +0.002820 | +0.005251 | +0.001428 | +0.002405 |
| ArguAna | +0.000018 | -0.000713 | 0.000000 | -0.000792 | +0.000714 |
| NFCorpus | -0.002225 | -0.000163 | +0.004029 | +0.000023 | +0.004759 |
| SciFact | +0.004407 | +0.003513 | +0.005000 | +0.003612 | 0.000000 |
| macro | **+0.001484** | **+0.001364** | **+0.003570** | **+0.001068** | **+0.001970** |

The result is a real, small cross-row capacity signal, not a breakthrough.
NFCorpus trades 0.0022 NDCG for higher Recall/CUB, and ArguAna head metrics are
effectively flat-negative. Unseen-corpus transfer is therefore mandatory.

## Exact Native Canary

Offline and PostgreSQL native metrics match exactly on NFCorpus and SciFact.

| Dataset | Posting delta | Storage delta | Mean latency delta | p95 delta |
| --- | ---: | ---: | ---: | ---: |
| NFCorpus | +6.25% total | +6.18% | +13.13% | +16.48% |
| SciFact | +6.25% total | +6.37% | +5.10% | -0.02% |

The total relation grows by only 6.25% because lexical postings remain fixed.
The canary confirms that quality gains survive the real query path. PostgreSQL
storage is a normalized engineering measure, not the final compact BMP index,
so it cannot by itself authorize product promotion.

## Conclusion

M1933 disproves the narrower claim that M1931's `1.0x` budget is already on
the best quality-cost point. A modest global expansion recovers useful frozen
M1914 support and generalizes across the four selection rows. It does not
authorize neural training: the deterministic publisher implements the policy
exactly, and no learned policy has yet shown additional value.

M1934 must now apply `b1.125` unchanged to unseen corpora. Failure there closes
global scalar budget expansion; success authorizes compact-index/native
validation, not an immediate model retraining campaign.

## Artifacts

- Contract: `docs/research-sae/reports/m1900-m1999/ii42-m1933-semantic-budget-frontier-contract.md`
- Auditor: `scripts/audit_m1933_semantic_budget_frontier.py`
- Matrix: `runs/m1933_semantic_budget_frontier_v1/matrix.json`
- Native outputs: `runs/m1933_selected_native_v1`
