# CSG-BETA-COMMONS-QUERY-1 Closure Report

## Decision

`CSG-BETA-COMMONS-QUERY-1` was stopped by product decision on 2026-08-26.
It was not completed and this report does not declare Shadow beta-ready.

The work produced durable query, publication, lifecycle, and measurement
improvements. It also isolated the remaining physical costs. Continuing with
route ratios, cardinality thresholds, cache sizes, scorer parameters, or codec
loop tuning is explicitly rejected. The only retained research candidate is
the exact essential-term MaxScore decomposition described below. It has not
been promoted to a product executor.

## Source Snapshot

The closing source state is branch `sae` at commit `3abe2453`:

```text
3abe2453 audit term prefix physical ceiling
926247bd docs: freeze CQ-3 physical cost ledger
4a7e2266 audit semantic essential posting bounds
6a5b1df3 Trace filtered block locality
4144a0b7 Document current scope v6 format
36c71533 Enforce the pinned ONNX Runtime build
23f45d00 Prefetch cold semantic row payloads
17ea4c71 Document eventual scoring and cold query boundary
d7147107 Prewarm semantic filter value metadata
39f11b72 Prune exact scope values with gram masks
```

Before this closure report, the branch was 290 commits ahead of `origin/sae`.
Two historical 2026-08-21 raw JSON files are retained with the report for
reproducibility. They are archived observations and are not adopted as current
evidence:

```text
docs/performance/data/raw/commons-query-readiness-2026-08-21/
shadow-v4-pubmed-date-category.json
shadow-v5-pubmed-date-category.json
```

## Live Deployment Snapshot

The following read-only inventory was captured at `2026-08-26T04:49:36Z`.
PostgreSQL reported no active rebuild or Evoke maintenance statement on any
host.

| Host | PostgreSQL | Extension | Evoke indexes | Valid / ready | Stored bytes | Installed binary SHA-256 | State |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Shadow | 18.4 | 0.2.4 | 14 | 14 / 14 | 178 GiB | `a415dd9b921586b5499b076d5522d53e00e4f8474455c7cebfedbd2e7091acf0` | Qualification host; PubMed remains `current_next` |
| Elm | 18.6 | 0.2.4 | 12 | 12 / 12 | 203 GiB | `27ad2f346b4c81982c32424d2b024e7c38bdb6af3039da942133974876f71606` | Mixed transition inventory remains |
| macOS | 18.6 | 0.2.4 | 19 | 19 / 19 | 1,143 MiB | `a5dc743f6bd5b4e76d6641d01a9c2f597ff93b76bc19aa2ff22ec7a4155aa32a` | Research indexes only |

Shadow uses `evoke.maintenance_worker_limit=4` and preloads Evoke. Its nine
Commons product-shaped roots are valid and ready, but the PubMed root is still
named `commons.data_pubmed__title_abstract__current_next_idx`. Five additional
roots belong to benchmark/runtime calibration schemas. Valid and ready catalog
bits do not prove the cold, concurrency, RSS, exactness, or package-binding
release gates.

### Post-closure stable-name update

Later on 2026-08-26, after the stopped goal had been preserved, the sole
Shadow PubMed root was renamed without rebuilding from
`commons.data_pubmed__title_abstract__current_next_idx` to
`commons.data_pubmed__title_abstract__field_aware_bm25_idx`. The old and new
top-20 query signatures were both
`84d32f56c97c96857d6e5acfe8e5555e`; the root remained valid and ready, and the
physical standby replayed the stable name. This catalog cleanup does not alter
the performance-stop decision or declare the remaining beta gates complete.

Elm still contains three transition artifacts in addition to its stable-name
roots:

```text
commons.data_arxiv__ta_field_aware_legacy_20260819_idx
commons.data_policy_ca__title_description__current_next_idx
commons.data_pubmed__title_abstract__field_aware_scope_next_idx
```

The macOS `postgres` database contains 19 BEIR/shared research roots and no
Commons product inventory. They remain outside automatic product migration.

The three installed binary hashes differ. Therefore the multi-host deployment
was not converged to one immutable current-only package when the goal stopped.

## Durable Results

### Publication and lifecycle

- `CQ-3E` passed its package-bound full PubMed publication gate. The 43.0 GB
  candidate was valid and ready, used zero swap, stayed below the 32 GiB
  process-family RSS ceiling at about 28.93 GB, and passed the five-query
  runtime transition fixture.
- Mutable-root TID lookup, immutable-root scoring plus bounded L0 merge,
  bounded zero-score completion, cancellation cleanup, physical telemetry,
  visibility retry accounting, and the removal of the unreachable filtered
  resident-fold branch were completed.
- CRUD, aborted transaction, 2PC, VACUUM, restart, replication, fold,
  compaction, reclamation, and eventual semantic completion have focused
  implementation evidence. The final clean-package rerun was not performed.
- Eventual SAE remains lexical-first. New committed rows remain searchable;
  semantic score and rank may temporarily differ until worker completion.

### Filter construction and exactness

- Filter plans preserve compositional native scopes and SQL residuals instead
  of discarding useful scope information when one predicate is unsupported.
- Array, text, Unicode, date-range, category, organization, and journal filter
  construction received focused exactness coverage.
- Scope v6 reuses reserved value-entry bytes for a no-false-negative 32-bit
  gram mask. On full PubMed it reduced exact category value comparisons from
  2,796,310 to 370,259, an 86.8% reduction, without growing the scope relation.
- The qualified PubMed date-category row retained exactly 51,033 allowed
  documents, zero predicate violations, and bit-identical top-50 TIDs and
  scores against the exact `tid[]` subset-ranking oracle.

### Cold path and observability

- Page-native prewarm was extended to accelerator directories, gram filters,
  semantic forward heads, and bounded scope metadata under the existing root
  and worker lifecycle.
- A payload-prefetch admission bug was fixed. After guarded relation-local
  eviction, the first exact PubMed date-category query improved from a
  greater-than-35-second timeout to about 5.55 seconds.
- Route telemetry now reports physical rows, postings, bytes, pages, semantic
  BMP references, and statement memory rather than only logical candidates.
- Query planner metadata was moved to a fixed worker-published term-work table
  under the same root. This avoids repeating a chunk-linear metadata walk.

## Closing Physical-Cost Matrix

All rows below returned 50 results with zero predicate violations. The
date-category row also passed complete membership, rank, and score-byte
comparison with its exact oracle.

| Shape | Exact route | Allowed docs | Exact rows | Decoded postings | Physical bytes | Warm p50 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unfiltered PubMed | semantic accelerator | n/a | 5,499 | 6.15 M residual | 65.37 MB forward | 443.4 ms | 300 ms, fail |
| Date + journal | forward rows | 39,614 | 39,614 | 9.55 M | 49.84 MB | 272.5 ms | 500 ms, pass |
| Date + category | forward rows | 51,033 | 51,033 | 14.50 M | 64.65 MB | 500-531 ms | 500 ms, unstable/fail |
| Partial date | forward bound | 839,154 | 7,357 | 2.08 M | 153.66 MB | 583.9 ms | 1,000 ms, pass |
| Broad date | forward bound | 2,356,781 | 10,010 | 2.98 M | 156.66 MB | 653.8 ms | 1,000 ms, pass |

The matrix identifies three separate costs:

1. selective filters linearly decode complete selected forward rows;
2. broad filters read an approximately fixed 147 MB semantic bound stream;
3. unfiltered search examines 6.15 million residual postings and remains over
   its independent latency gate.

Changing the boundary among the existing routes cannot remove these costs.

## Rejected Work

The following branches are closed and must not be restarted as another tuning
cycle:

- route ratios, cardinality thresholds, cache windows, and scorer parameters;
- `allowed_count` or active-chunk heuristics that ignore physical bytes,
  posting work, and block skipability;
- the largest-query-term row-prefix/checkpoint codec branch;
- filtered transpose on the current full root;
- a persistent parallel filter artifact or second lifecycle;
- application-side post-filtering or a second semantic ranking authority.

The row-prefix decision is quantitative. Field-aware expansion places the
largest query term near the end of the vocabulary:

| Surface | Suffix work avoided | Suffix bytes avoided |
| --- | ---: | ---: |
| FIQA current root | 0.123% | 1.253% |
| PubMed 500K | 0.114% | 0.746% |
| PubMed full | 0.113% | 0.995% |

It cannot meet the required 15% physical-work reduction and was not promoted.

## One Retained Research Candidate

The exact essential-term MaxScore decomposition is the only result with enough
evidence to justify one future bounded experiment. It sorts semantic terms by
a sign-safe global contribution cap, uses mandatory plus essential terms to
derive a conservative kth-score lower bound, represents omitted terms by a
safe suffix cap, and exact-scores every surviving block through the existing
authority.

The oracle projected 23.42% to 48.25% conservative work savings on PubMed 500K
and 36.73% to 58.31% on full PubMed. Every derived lower bound was safe and no
competitive block was missed.

The oracle itself is not deployable. It consumed about 7.6-7.9 GiB backend RSS
and 115-144 seconds per full-root query because it materialized complete score
and absolute-sum evidence and evaluated all prefix points. No automatic route,
prefix count, or product constant was added.

If work is resumed, admit exactly one forced test executor that reuses the
current BMP authority and one bounded score workspace. Stop permanently if it
requires a complete term prepass, a second corpus-sized score array, produces
any result-bit difference, loses the physical page/posting saving, or grows
memory with query postings. Do not substitute another threshold experiment.

## Goal Status by Area

| Area | Status at stop |
| --- | --- |
| CQ-0 benchmark harness | Partially complete; cold/first-admitted/warm separation and an oracle for every filter remain open |
| CQ-1 compositional filter planning | Complete in implementation and focused tests |
| CQ-2 filter candidate construction | Complete in implementation and focused tests |
| CQ-3 exact route and cost work | Partially complete; physical costs isolated, two latency gates open, essential-term executor not implemented |
| CQ-3E full-root publisher | Complete and qualified |
| CQ-4 cold start/residency | Partially complete; cold path improved but 2-second gate and concurrent first-query matrix remain open |
| CQ-5 planner/statistics | Implemented; final stable-root release qualification remains open |
| CQ-6 mutable filter lifecycle | Complete in focused implementation evidence; final clean-package rerun remains open |
| CQ-7 Shadow qualification/promotion | Not complete; full concurrency/cold/warm matrix, package binding, latency gates, and stable-name promotion remain open |
| CQ-8 rollout/compatibility cleanup | Not complete; host convergence, old-reader removal, clean package, Elm/mac rollout, and artifact cleanup remain open |

## Explicitly Unfinished

Stopping this goal leaves the following product work unresolved:

1. Shadow stable-name promotion and rollback finalization.
2. Full CQ-4/CQ-5/CQ-7 cold, warm, concurrency 1/4/8, cancellation, RSS,
   exactness, physical-work, and Commons application matrices.
3. One immutable ORT 1.29 current-only package bound to source, SQL, model,
   installed binary, and catalog on Shadow.
4. Removal of scope v2-v5 readers, accelerator policy 5-6 compatibility, and
   the synthetic fixed-32-extent compatibility contract.
5. Clean-package codec, CRUD/2PC/VACUUM, restart, replication, cancellation,
   RSS, package-binding, and Shadow smoke reruns.
6. Elm transition-root cleanup and rebuild from the current-only package.
7. macOS product-catalog recreation and bounded qualification; the existing
   research indexes remain outside product migration.

These items are frozen, not waived. A future release plan must either complete
them or explicitly redefine the product gate. No completed item in this report
is evidence that the omitted gate passed.

### 2026-08-26 Current-Only Package Follow-Up

Items 3 and the scope/policy portions of item 4 were completed after this
historical stop snapshot. Commit `852e28f1` removes scope v2-v5 readers and
accelerator policy 5-6 query compatibility. A clean ORT 1.29 package built from
that commit was installed on Shadow without changing any of the 14 root
generation identifiers. Package identity, 71/71 isolated lifecycle gates,
runtime API 29, ORT 1.29.0, current scope/policy status, and pre/post query
correctness are recorded in
[Shadow Current-Only ORT 1.29 Closure](csg-beta-current-only-ort129-shadow-2026-08-26.md).

The fixed-32-extent fixture, complete CQ-4/CQ-7 performance matrix, replication,
cancellation, and full-root concurrency RSS remain separate gates. The
follow-up therefore closes the two bounded package/reader tasks but does not
change this report's stopped-performance conclusion.

## Closure Verification

The source snapshot was checked without changing any deployed host:

- `git diff --check`: passed;
- product convergence inventory: passed;
- focused term-suffix projection tests: 6 passed;
- Python syntax checks for the inventory and isolated smoke: passed;
- isolated PostgreSQL 18 convergent-segment smoke: 85/85 gates passed.

The isolated smoke covered page-native exactness, CRUD and transaction
visibility, L0 rotation, compaction, VACUUM, restart, retirement and page reuse,
shared workload folds, bounded top-k, and contract drift. Its local output is
`/tmp/evoke-csg-closure-segment-smoke.json` and reports
`all_gates_passed=true`.

## Evidence Index

- `docs/performance/reports/cq3-shadow-physical-cost-ledger-2026-08-26.md`
- `docs/performance/reports/semantic-essential-term-maxscore-oracle-2026-08-25.md`
- `docs/performance/data/raw/commons-query-readiness-2026-08-26/`
- `docs/performance/data/raw/semantic-essential-maxscore-2026-08-25/`
- `/data/evoke-builds/cq3e-full-ort129-bound-final-qualification-v3.json`
- `/data/evoke-builds/releases/98bb9661/cq8-shadow-inventory-preflight.json`
- `/home/leask/evoke-builds/evoke-release-98bb9661/cq8-elm-inventory-preflight.json`
- `/Volumes/Betty/Tmp/evoke-cq8-macos-inventory-98bb9661.json`
