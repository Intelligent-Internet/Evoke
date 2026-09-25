# II42 P2.1 Full Native BEIR15 Recall Matrix

All rows use the local PostgreSQL product path and candidate depth `k=1000`. BM25 and P2.1 use native inverted-index traversal; dense uses the native VectorChord index. P2.1 compiles raw query text inside PostgreSQL and queries one logical unified-posting index.

## Macro Matrix

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25s native | 0.374297 | 0.274610 | 0.562964 | 0.475641 | 0.735619 |
| VectorChord dense | 0.544873 | 0.412036 | 0.670880 | 0.650711 | 0.810525 |
| II42 P2.1 | 0.490809 | 0.371455 | 0.666885 | 0.595950 | 0.839658 |

## Decision Summary

- P2.1 minus BM25s macro: NDCG@10 `+0.116511`, MAP@100 `+0.096845`, Recall@100 `+0.103922`, MRR@20 `+0.120309`, CUB@1000 `+0.104039`.
- P2.1 minus VectorChord macro: NDCG@10 `-0.054065`, MAP@100 `-0.040581`, Recall@100 `-0.003995`, MRR@20 `-0.054761`, CUB@1000 `+0.029133`.
- P2.1 wins Recall@100 on `14/15` rows against BM25s and `11/15` rows against VectorChord. It wins CUB@1000 on `12/15` rows against VectorChord.
- Dense-relative Recall@100 regressions: `climate-fever` -0.172562, `fiqa` -0.141926, `scidocs` -0.017383, `cqadupstack` -0.016747.

P2.1 validates the unified-posting product shape as a strong candidate generator: it nearly matches dense macro Recall@100 while increasing candidate upper bound. It does not replace dense head ranking at this checkpoint because NDCG@10, MAP@100, and MRR@20 remain lower. The next bottleneck is score calibration and boundary ordering, not broader candidate expansion.

## Per-Dataset Recall@100

| Dataset | BM25s | VectorChord | P2.1 | P2.1-BM25s | P2.1-Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| arguana | 0.952891 | 0.972877 | 0.987866 | +0.034975 | +0.014989 |
| climate-fever | 0.341694 | 0.703507 | 0.530945 | +0.189251 | -0.172562 |
| cqadupstack | 0.524878 | 0.747247 | 0.730500 | +0.205622 | -0.016747 |
| dbpedia-entity | 0.390971 | 0.458798 | 0.528869 | +0.137897 | +0.070071 |
| fever | 0.838526 | 0.905383 | 0.949535 | +0.111009 | +0.044152 |
| fiqa | 0.510764 | 0.804102 | 0.662176 | +0.151412 | -0.141926 |
| hotpotqa | 0.724646 | 0.798447 | 0.816205 | +0.091560 | +0.017758 |
| msmarco | 0.408121 | 0.472836 | 0.487246 | +0.079125 | +0.014410 |
| nfcorpus | 0.233236 | 0.269111 | 0.299101 | +0.065865 | +0.029990 |
| nq | 0.678568 | 0.912732 | 0.919298 | +0.240730 | +0.006566 |
| quora | 0.947661 | 0.984258 | 0.988816 | +0.041154 | +0.004558 |
| scidocs | 0.348567 | 0.484483 | 0.467100 | +0.118533 | -0.017383 |
| scifact | 0.882556 | 0.898222 | 0.956000 | +0.073444 | +0.057778 |
| trec-covid | 0.100074 | 0.157343 | 0.161646 | +0.061572 | +0.004303 |
| webis-touche2020 | 0.561303 | 0.493851 | 0.517975 | -0.043327 | +0.024124 |

## Per-Dataset NDCG@10

| Dataset | BM25s | VectorChord | P2.1 |
| --- | ---: | ---: | ---: |
| arguana | 0.344115 | 0.429182 | 0.411456 |
| climate-fever | 0.127530 | 0.394597 | 0.228576 |
| cqadupstack | 0.291987 | 0.431176 | 0.418136 |
| dbpedia-entity | 0.239804 | 0.387310 | 0.353671 |
| fever | 0.449661 | 0.846450 | 0.766722 |
| fiqa | 0.231073 | 0.510599 | 0.358516 |
| hotpotqa | 0.511627 | 0.688827 | 0.659351 |
| msmarco | 0.366717 | 0.647114 | 0.524892 |
| nfcorpus | 0.306793 | 0.322832 | 0.347391 |
| nq | 0.242799 | 0.584749 | 0.481829 |
| quora | 0.738212 | 0.880790 | 0.844995 |
| scidocs | 0.150534 | 0.225314 | 0.200236 |
| scifact | 0.663931 | 0.714060 | 0.723722 |
| trec-covid | 0.571959 | 0.833468 | 0.752852 |
| webis-touche2020 | 0.377718 | 0.276631 | 0.289785 |

## Per-Dataset MAP@100

| Dataset | BM25s | VectorChord | P2.1 |
| --- | ---: | ---: | ---: |
| arguana | 0.237186 | 0.300000 | 0.286486 |
| climate-fever | 0.097173 | 0.314161 | 0.175821 |
| cqadupstack | 0.265961 | 0.389596 | 0.380379 |
| dbpedia-entity | 0.173662 | 0.250729 | 0.261121 |
| fever | 0.395967 | 0.820341 | 0.713127 |
| fiqa | 0.183892 | 0.451725 | 0.297809 |
| hotpotqa | 0.430958 | 0.619042 | 0.575048 |
| msmarco | 0.264019 | 0.383274 | 0.346804 |
| nfcorpus | 0.137331 | 0.139780 | 0.169076 |
| nq | 0.203360 | 0.517974 | 0.415296 |
| quora | 0.695173 | 0.851136 | 0.807626 |
| scidocs | 0.103194 | 0.159333 | 0.142215 |
| scifact | 0.626833 | 0.685142 | 0.679675 |
| trec-covid | 0.066611 | 0.131466 | 0.131082 |
| webis-touche2020 | 0.237831 | 0.166837 | 0.190253 |

## Per-Dataset MRR@20

| Dataset | BM25s | VectorChord | P2.1 |
| --- | ---: | ---: | ---: |
| arguana | 0.233893 | 0.298787 | 0.284977 |
| climate-fever | 0.181684 | 0.513265 | 0.315448 |
| cqadupstack | 0.295992 | 0.425859 | 0.416496 |
| dbpedia-entity | 0.504366 | 0.733080 | 0.689000 |
| fever | 0.410674 | 0.862403 | 0.756337 |
| fiqa | 0.292597 | 0.599713 | 0.439020 |
| hotpotqa | 0.662780 | 0.834553 | 0.824656 |
| msmarco | 0.743909 | 0.961240 | 0.884367 |
| nfcorpus | 0.515219 | 0.548789 | 0.567236 |
| nq | 0.215921 | 0.541040 | 0.435048 |
| quora | 0.735647 | 0.874570 | 0.838923 |
| scidocs | 0.277201 | 0.385142 | 0.348686 |
| scifact | 0.634838 | 0.694454 | 0.688295 |
| trec-covid | 0.817222 | 0.970000 | 0.930000 |
| webis-touche2020 | 0.612666 | 0.517772 | 0.520760 |

## Per-Dataset Candidate Upper Bound@1000

| Dataset | BM25s | VectorChord | P2.1 |
| --- | ---: | ---: | ---: |
| arguana | 0.990007 | 0.972877 | 1.000000 |
| climate-fever | 0.567427 | 0.836721 | 0.743779 |
| cqadupstack | 0.697639 | 0.870515 | 0.888289 |
| dbpedia-entity | 0.580338 | 0.629907 | 0.733407 |
| fever | 0.933235 | 0.912997 | 0.967121 |
| fiqa | 0.725380 | 0.912474 | 0.855207 |
| hotpotqa | 0.847738 | 0.856516 | 0.901283 |
| msmarco | 0.672591 | 0.700965 | 0.784065 |
| nfcorpus | 0.424066 | 0.541961 | 0.588587 |
| nq | 0.860926 | 0.935762 | 0.980229 |
| quora | 0.986610 | 0.987865 | 0.997925 |
| scidocs | 0.561417 | 0.753817 | 0.734433 |
| scifact | 0.965000 | 0.918222 | 0.993333 |
| trec-covid | 0.362988 | 0.498362 | 0.573787 |
| webis-touche2020 | 0.858916 | 0.828912 | 0.853424 |

## P2.1 Engineering Summary

- Generation shards: `131`
- Logical payload bytes: `46664936827`
- Stored generation bytes: `34320714805`
- Query latency mean across dataset means: `134.803 ms`
- Query latency p95 across dataset p95 values: `196.486 ms`

The VectorChord matrix is intentionally single-worker. Concurrent approximate queries produced schedule-dependent results during the reproducibility audit, so concurrent output was rejected.

The retained serial VectorChord matrix was reused after exact full-query per-query metric matches on `ArguAna` and `Climate-FEVER` in this run. The audit paths and SHA-256 values are recorded in the JSON artifact.
Reuse canary queries: `2936`.
