# II-42 M1934 Fixed-Budget Unseen Transfer Report

Date: 2026-07-13

Decision: **the frozen `b1.125` policy transfers and reproduces through the
native one-index path. Promote it from an in-sample research point to a
validated fixed-quality candidate, but do not make it the product default.**

## Question Answered

M1933 selected a semantic posting budget of `1.125x` lexical postings on
FiQA, ArguAna, NFCorpus, and SciFact. M1934 applied the same parent, support,
query calibration, and posting budget without tuning to three unseen rows:
SciDocs, Quora, and TREC-COVID.

The fixed configuration was:

- IBM Granite 30M sparse revision
  `ad82b1fd09541c998c8d45045d601c51fdb8a9b7`;
- M1914 top-192 document and top-50 query support;
- M1931 `rms_m4` query calibration;
- exact II42 lexical postings in a disjoint namespace;
- one additive sparse dot-product index;
- candidate depth 1,000;
- `b1` versus the frozen `b1.125`, with no row-specific selection.

All model inference and large sparse evaluation ran on `spark-1`. Exact
lexical terms came from the native PostgreSQL tokenizer; sparse matrix
assembly ran remotely through a bounded data stream. The streaming evaluator
was checked against the previous materialized SciDocs result to machine
precision before the larger rows ran.

## Unseen Matrix

| Dataset | Budget | NDCG@10 | MAP@100 | R@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciDocs | b1 | 0.199702 | 0.141874 | 0.466350 | 0.347758 | 0.730383 |
| SciDocs | b1.125 | **0.200666** | **0.142329** | **0.467100** | **0.349276** | **0.734633** |
| Quora | b1 | 0.840398 | 0.802779 | 0.988161 | 0.834944 | **0.998043** |
| Quora | b1.125 | **0.845151** | **0.807671** | **0.988838** | **0.838927** | 0.998033 |
| TREC-COVID | b1 | 0.747926 | 0.130121 | 0.161241 | 0.921667 | 0.570697 |
| TREC-COVID | b1.125 | **0.753781** | **0.131118** | **0.161626** | **0.930000** | **0.573401** |
| macro | b1 | 0.596009 | 0.358258 | 0.538584 | 0.701456 | 0.766375 |
| macro | b1.125 | **0.599866** | **0.360373** | **0.539188** | **0.706068** | **0.768689** |

| Dataset | Delta NDCG | Delta MAP | Delta R | Delta MRR | Delta CUB | Row gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SciDocs | +0.000964 | +0.000456 | +0.000750 | +0.001518 | +0.004250 | pass |
| Quora | +0.004753 | +0.004892 | +0.000678 | +0.003983 | -0.000011 | pass |
| TREC-COVID | +0.005855 | +0.000997 | +0.000384 | +0.008333 | +0.002704 | pass |
| macro | **+0.003857** | **+0.002115** | **+0.000604** | **+0.004611** | **+0.002314** | pass |

The candidate passed all row floors and the full unseen macro gate. Quora's
CUB change is a negligible `-1.08e-5`; its top-100 quality metrics all
improve. The broader result therefore supports a real fixed-capacity signal,
not a gain isolated to the M1933 selection rows.

## Cost Surface

The quality gain is not free.

| Measure | b1 | b1.125 | Delta |
| --- | ---: | ---: | ---: |
| semantic postings, three rows | 24,528,765 | 27,594,861 | +12.50% |
| total lexical + semantic postings | 49,057,530 | 52,123,626 | +6.25% |
| mean matched documents/query | 104,521 | 110,666 | +5.88% |
| mean posting touches/query | 475,258 | 511,427 | +7.61% |
| worst semantic maxDF ratio | 0.7186 | 0.7238 | +0.0052 |

TREC-COVID remains the main engineering risk: its candidate averages about
`1.20M` semantic posting touches per query and has `maxDF=0.724`. Quora is
less DF-heavy, but still matches about 149,479 documents per query. These are
native inverted-index costs, not ordinary document nnz statistics.

## Exact Native Replay

SciDocs was published to one PostgreSQL posting relation and evaluated over
all 1,000 queries. Native and offline quality agree within the required
`1e-5`; the largest error is MAP at `8.8e-8`.

| Measure | b1 native | b1.125 native | Delta |
| --- | ---: | ---: | ---: |
| total postings | 5,337,150 | 5,670,722 | +6.25% |
| normalized storage | 407,830,528 B | 433,020,928 B | +6.18% |
| mean latency | 126.21 ms | 135.55 ms | +7.40% |
| p50 latency | 126.14 ms | 135.97 ms | +7.79% |
| p95 latency | 179.31 ms | 195.00 ms | +8.75% |
| NDCG@10 | 0.199702 | 0.200666 | +0.000964 |
| MAP@100 | 0.141874 | 0.142329 | +0.000456 |
| Recall@100 | 0.466350 | 0.467100 | +0.000750 |
| MRR@20 | 0.347758 | 0.349276 | +0.001518 |

This closes the concern that the offline additive matrix was creating a gain
that the plugin could not reproduce. It also confirms that query latency grows
slightly faster than relation size on this row.

## Conclusion

M1934 validates the central direction: a mature learned sparse semantic parent
can complement exact lexical postings inside one physical inverted index, and
a small global capacity expansion improves quality across selection and unseen
corpora without a dataset-specific gate or reranker.

It does **not** justify another scalar budget sweep or a neural model trained
merely to imitate `b1.125`; the deterministic publisher already implements
that policy exactly. The remaining bottleneck is marginal posting utility:
the extra 12.5% semantic postings yield a small quality gain while increasing
traversal by 7.61%, with high-DF terms dominating TREC-COVID.

The next justified experiment is a diagnostic attribution of the postings
added between `b1` and `b1.125`, grouped by DF, impact, query activity, and
rescued versus harmed relevant documents. Training is warranted only if that
audit exposes an inference-time, qrels-free feature that can retain the unseen
gain at approximately the `b1` cost. Otherwise `b1.125` remains an explicit
quality/cost operating point rather than a new default.

## Subsequent Nested-Support Audit

M1935 found that the independently pruned M1934 Quora candidate replaced 264
of 5,698,832 baseline semantic postings while reaching its larger target. A
canonical nested reconstruction preserved every `b1` posting and retained the
result: nested macro NDCG/MAP/Recall/MRR were respectively `0.600053`,
`0.360607`, `0.539394`, and `0.706295`, all slightly above the original M1934
candidate. CUB changed by only `-0.000056`.

The M1934 transfer conclusion is therefore robust, but the original candidate
should not be described as perfectly nested. M1935 also closed DF-only
selection of the incremental tail; see
`docs/research-sae/reports/m1900-m1999/ii42-m1935-incremental-tail-attribution-report.md`.

## Artifacts

- Contract: `docs/research-sae/reports/m1900-m1999/ii42-m1934-fixed-budget-unseen-transfer-contract.md`
- Evaluator: `scripts/audit_m1934_fixed_budget_transfer.py`
- Full matrix: `runs/m1934_fixed_budget_unseen_v1/full/matrix.json`
- Selected surface: `runs/m1934_fixed_budget_unseen_v1/selected/full`
- Native b1 control: `runs/m1934_fixed_budget_unseen_v1/native-b1`
- Native b1.125 candidate: `runs/m1934_fixed_budget_unseen_v1/native`
