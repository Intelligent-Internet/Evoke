# II-42 M1931 Query-Local Source Calibration Report

## Result

M1931 replaces the fixed M1930B lexical/semantic scale with a qrels-free scalar computed from query postings and corpus column statistics before retrieval. The selected policy is `rms_m4`, constrained by the proven M1930B semantic floor. Retrieval remains one sparse dot product over one physical posting relation.

| Dataset | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | 0.411735 | 0.287776 | 0.987866 | 0.286317 | 0.999286 |
| FiQA | 0.353492 | 0.294069 | 0.656539 | 0.435401 | 0.854011 |
| NFCorpus | 0.349647 | 0.169181 | 0.294991 | 0.569378 | 0.583782 |
| SciFact | 0.719461 | 0.676354 | 0.951000 | 0.684871 | 0.993333 |
| **macro** | **0.458584** | **0.356845** | **0.722599** | **0.493992** | **0.857603** |

Against full M1914, M1931 improves macro NDCG@10, MAP@100, and MRR@20 while giving up only `0.002608` Recall@100 and `0.002688` CUB. The remaining row-level gap is concentrated in FiQA.

## Generalization

All four leave-one-dataset-out folds pass. Heldout macro quality is:

| NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| ---: | ---: | ---: | ---: | ---: |
| 0.458321 | 0.356695 | 0.722095 | 0.493640 | 0.857142 |

The query scale is derived without qrels, dataset-specific thresholds, or post-retrieval evidence. ArguAna receives a much larger semantic scale (median `25.1264`) while FiQA remains at the protected floor (`3.1408`), which explains why query-local calibration fixes the fixed-scale head inversion.

## Native Replay

| Dataset | Offline/native match | Mean latency | P95 latency |
| --- | --- | ---: | ---: |
| NFCorpus | exact | 10.411 ms | 20.798 ms |
| SciFact | exact | 20.820 ms | 33.116 ms |

The relation storage is identical to M1930B because M1931 changes only query weights. This confirms that the improvement is product-compatible and does not add index cost.

## Bottleneck And Next Gate

M1931 closes query-side source calibration. The residual Recall/CUB gap now comes from document-side top-impact pruning. At the fixed lexical-equivalent semantic budget, the selected documents still contain highly concentrated atoms:

- ArguAna maxDF `0.8966`, head-1% posting share `0.4745`;
- FiQA maxDF `0.7874`, head share `0.4714`;
- NFCorpus maxDF `0.7000`, head share `0.4269`;
- SciFact maxDF `0.4586`, head share `0.3855`.

M1932 therefore tests cross-corpus query utility and local document frequency as an allocation teacher at the exact same posting budget. Neural residual training remains blocked until that deterministic teacher passes heldout quality and production-cost gates.
