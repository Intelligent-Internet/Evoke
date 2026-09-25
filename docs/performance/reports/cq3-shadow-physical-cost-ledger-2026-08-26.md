# CQ-3 Shadow Physical-Cost Ledger

## Decision

Freeze route ratios, cardinality thresholds, cache sizes, and scorer parameters.
The current Shadow root is exact for the qualified selective-filter case, but
two latency gates remain open because two different physical costs remain:

1. selective filters decode complete document-major forward rows; and
2. broad filters read an approximately fixed 147 MB forward-bound stream.

These are representation and executor costs. They cannot be closed by moving
the boundary between the existing routes.

## Authority

The measurement used the running Shadow PostgreSQL 18.4 product instance on
2026-08-26. The installed extension is 0.2.4 with catalog contract
`evoke_catalog_v1`, ONNX Runtime 1.29.0, and Evoke binary SHA-256:

```text
a415dd9b921586b5499b076d5522d53e00e4f8474455c7cebfedbd2e7091acf0
```

That binary is retained under the source snapshot named
`evoke-cq3-locality-6a5b1df3`. The PubMed root remained generation
`57036813/1/2440`, accelerator policy 7, scope v6, healthy, valid, ready, and
free of semantic debt during the measurement.

The raw artifacts are:

```text
docs/performance/data/raw/commons-query-readiness-2026-08-26/
cq3-current-pubmed-panel-20260826.json
e97990c51a48fc4829efa220cc916cdde47889d6b49393d7e5160438e8d63ab6

cq3-current-pubmed-date-category-20260826.json
4ee87bc1a08aeb687fd1b1b3160c1ce049aa0ccf2c7399173bd9bb8ef5840809

fiqa-term-suffix-ceiling-20260826.json
25affa21f241ef765b9dd98b194ec98e334ec474089bd7b45353bbeacdf6bbfc

shadow-term-suffix-ceiling-20260826.json
c6fb9387cc882a977b09ac648025dc686591dd1731a0f6d97754f07761d1c302
```

## Current Matrix

All cases returned 50 rows and zero predicate violations. The separate
`pubmed_date_category` run compared every result with the exact `tid[]`
subset-ranking oracle and passed with identical membership, rank, and score
bytes.

| Case | Route | Allowed docs | Exact-scored rows | Decoded postings | Physical bytes | Statement memory | Warm p50 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered | semantic accelerator | n/a | 5,499 | 6.15 M residual | 65.37 MB forward | 36.01 MB | 443.4 ms | 300 ms, fail |
| Date + journal | forward rows | 39,614 | 39,614 | 9.55 M | 49.84 MB | 0.72 MB | 272.5 ms | 500 ms, pass |
| Date + category | forward rows | 51,033 | 51,033 | 14.50 M | 64.65 MB | 0.72 MB | 531.4 ms | 500 ms, fail |
| Partial date | forward bound | 839,154 | 7,357 | 2.08 M | 153.66 MB | 11.99 MB | 583.9 ms | 1,000 ms, pass |
| Broad date | forward bound | 2,356,781 | 10,010 | 2.98 M | 156.66 MB | 14.50 MB | 653.8 ms | 1,000 ms, pass |

The date-category result varies around the gate rather than exhibiting a
correctness failure. An independent exact-oracle run measured 500.6 ms warm
p50; the five-case panel measured 531.4 ms. This is not a reason to change the
500 ms gate or route threshold.

## Data-Shape Diagnosis

### Selective filters

Date-journal and date-category both use direct forward rows, but category
decodes 52% more postings and reads 30% more row bytes. The category scope
also examines 370,259 values while journal examines 3,017. The difference is
therefore explained by selected-row density, row payload, and scope work, not
by a hidden fallback or semantic inference.

The current row codec is already a sorted delta-varint term stream with int8
codes. It linearly decodes each selected row and merge-matches the sorted query
terms. A directory-only probe now measures the maximum saving available to any
checkpoint that stops after the largest query term. It loads the fixed current
root term-work table, does not read forward rows, and does not run a scorer.

| Surface | Documents | Fields | Max query term / vocabulary | Suffix work | Suffix bytes | 15% gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIQA current root | 57,638 | 1 | 77,376 / 79,787 | 0.123% | 1.253% | fail |
| PubMed medium root | 500,000 | 2 | 157,163 / 159,574 | 0.114% | 0.746% | fail |
| PubMed full current root | 8,214,026 | 2 | 157,163 / 159,574 | 0.113% | 0.995% | fail |

The full-root probe completed in 48 ms and retained generation
`57036813/1/2440`. The result is decisive: field expansion places the largest
query term near the end of the vocabulary, so a term-id prefix would still
decode more than 99% of the physical bytes and 99.8% of the work. The
max-query-term checkpoint branch is rejected. Selective direct-row work now
requires a different addressability structure with an independently proven
physical saving; no codec-loop or checkpoint micro-tuning is admitted.

### Broad filters

Forward-bound reduces exact row scoring from hundreds of thousands or millions
of allowed documents to 7,357--10,010 rows. Its remaining cost is not those
rows. Both broad cases estimate 147.13 MB of bound metadata and read about
154--157 MB in total. This is the fixed semantic proof stream identified by
the essential-term MaxScore oracle.

The deployable-threshold oracle already proves that a mandatory plus essential
semantic prefix can derive a safe kth-score lower bound without using the
final kth score. On 500K and full PubMed it conservatively projects 23--58%
less posting work with no missed competitive block. The next implementation
may therefore be a forced, test-only executor over the existing authority. It
must demonstrate fewer physical bound pages and postings, not only a lower
estimator value.

### Unfiltered search

Unfiltered search is independently over its 300 ms gate. It consumes 6.15 M
residual postings, 65.37 MB of forward payload, and 35.8 MB of accelerator
query state. A filtered-route change cannot close this gate. Any structural
prefix executor must therefore report whether it also reduces this residual
work; otherwise unfiltered acceleration remains a separate bounded issue.

## Next Bounded Slice

1. Keep the selective max-query-term checkpoint rejected. Admit another
   selective representation only after a read-only oracle proves at least 15%
   fewer decoded postings and physical bytes on medium and full PubMed.
2. Implement the already-proved essential-prefix decomposition only behind a
   forced test route. Keep the current score authority and generation.
3. Compare the forced route with the installed product binary on the same root
   and report exact IDs and score bytes, bound pages, postings, row bytes,
   latency, cancellation, RSS, and concurrency.
4. Promote no automatic policy until both physical work and p50/p95 improve.

The slice stops on any result mismatch, full prepass, corpus-sized second score
array, route-specific constant, or improvement visible only in an estimator.
