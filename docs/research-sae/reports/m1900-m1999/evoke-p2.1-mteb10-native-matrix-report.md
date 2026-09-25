# II42 P2.1 Native MTEB10 Retrieval Matrix

This is the established project `MTEB(eng, v2)` ten-task Retrieval surface, not every task currently published by MTEB. All three methods use the same local PostgreSQL query/qrel rows and candidate depth `k=1000`. P2.1 compiles raw query text in the native product path and queries one unified-posting index.

## Macro Matrix

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25s native | 0.383554 | 0.268718 | 0.595367 | 0.461805 | 0.777387 |
| VectorChord dense | 0.544127 | 0.413201 | 0.707213 | 0.629992 | 0.841063 |
| II42 P2.1 | 0.503440 | 0.374100 | 0.703125 | 0.578901 | 0.875910 |

## Comparison

- P2.1 minus BM25s macro: NDCG `+0.119885`, MAP `+0.105382`, Recall `+0.107757`, MRR `+0.117095`, CUB `+0.098524`.
- P2.1 minus VectorChord macro: NDCG `-0.040688`, MAP `-0.039101`, Recall `-0.004088`, MRR `-0.051091`, CUB `+0.034847`.

## Full Per-Dataset Matrix

| Dataset | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | BM25s native | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 |
| ArguAna | VectorChord dense | 0.450704 | 0.319743 | 0.982156 | 0.318966 | 0.982156 |
| ArguAna | II42 P2.1 | 0.415546 | 0.290486 | 0.989293 | 0.289107 | 1.000000 |
| CQADupstackGamingRetrieval | BM25s native | 0.465550 | 0.430617 | 0.746695 | 0.457166 | 0.869162 |
| CQADupstackGamingRetrieval | VectorChord dense | 0.554296 | 0.513284 | 0.770804 | 0.545836 | 0.813355 |
| CQADupstackGamingRetrieval | II42 P2.1 | 0.579577 | 0.535178 | 0.866450 | 0.561797 | 0.955603 |
| CQADupstackUnixRetrieval | BM25s native | 0.282784 | 0.259707 | 0.543694 | 0.284534 | 0.754300 |
| CQADupstackUnixRetrieval | VectorChord dense | 0.449146 | 0.410350 | 0.767735 | 0.440482 | 0.870142 |
| CQADupstackUnixRetrieval | II42 P2.1 | 0.433789 | 0.394026 | 0.777384 | 0.428277 | 0.936138 |
| ClimateFEVERHardNegatives | BM25s native | 0.141582 | 0.107780 | 0.411267 | 0.201400 | 0.712000 |
| ClimateFEVERHardNegatives | VectorChord dense | 0.377672 | 0.298098 | 0.675817 | 0.504430 | 0.801333 |
| ClimateFEVERHardNegatives | II42 P2.1 | 0.247970 | 0.194502 | 0.585350 | 0.338304 | 0.831250 |
| FEVERHardNegatives | BM25s native | 0.501070 | 0.452277 | 0.895438 | 0.466675 | 0.975288 |
| FEVERHardNegatives | VectorChord dense | 0.854417 | 0.829734 | 0.914605 | 0.864621 | 0.921224 |
| FEVERHardNegatives | II42 P2.1 | 0.785359 | 0.737200 | 0.966279 | 0.774984 | 0.988602 |
| FiQA2018 | BM25s native | 0.231073 | 0.183888 | 0.510764 | 0.292591 | 0.725380 |
| FiQA2018 | VectorChord dense | 0.518076 | 0.458882 | 0.833468 | 0.603800 | 0.954942 |
| FiQA2018 | II42 P2.1 | 0.354177 | 0.295080 | 0.658711 | 0.434631 | 0.859093 |
| HotpotQAHardNegatives | BM25s native | 0.549783 | 0.466817 | 0.811000 | 0.719432 | 0.933000 |
| HotpotQAHardNegatives | VectorChord dense | 0.684454 | 0.615105 | 0.832000 | 0.816141 | 0.894000 |
| HotpotQAHardNegatives | II42 P2.1 | 0.676280 | 0.591675 | 0.882000 | 0.832442 | 0.962000 |
| SCIDOCS | BM25s native | 0.150694 | 0.103316 | 0.348367 | 0.277795 | 0.561617 |
| SCIDOCS | VectorChord dense | 0.222704 | 0.157222 | 0.483100 | 0.378192 | 0.750733 |
| SCIDOCS | II42 P2.1 | 0.200611 | 0.142496 | 0.467100 | 0.349516 | 0.734633 |
| TRECCOVID | BM25s native | 0.572088 | 0.066617 | 0.100137 | 0.817222 | 0.363134 |
| TRECCOVID | VectorChord dense | 0.724736 | 0.116822 | 0.151098 | 0.871667 | 0.497221 |
| TRECCOVID | II42 P2.1 | 0.758159 | 0.129599 | 0.160195 | 0.907500 | 0.568840 |
| Touche2020Retrieval.v3 | BM25s native | 0.596805 | 0.378977 | 0.633422 | 0.867347 | 0.889978 |
| Touche2020Retrieval.v3 | VectorChord dense | 0.605068 | 0.412771 | 0.661348 | 0.955782 | 0.925524 |
| Touche2020Retrieval.v3 | II42 P2.1 | 0.582929 | 0.430758 | 0.678484 | 0.872449 | 0.922945 |

## P2.1 Engineering

- Dataset rows: `10`
- Evaluated queries: `8815`
- Generation shards: `18`
- Logical payload bytes: `2946006840`
- Stored generation bytes: `2912242585`
- Mean of per-dataset mean query latency: `36.577 ms`
- Mean of per-dataset p95 query latency: `44.173 ms`

VectorChord was evaluated single-worker because approximate search must remain reproducible on the comparison surface.

## Conclusion

All ten VectorChord rows now have usable candidate coverage; the complete matrix is the primary comparison surface.

P2.1 uses a raw-text compiler and one unified-posting index. Its macro deltas against BM25s and VectorChord are reported above without excluding any dataset rows.
