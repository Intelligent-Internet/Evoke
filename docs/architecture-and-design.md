# Architecture

Evoke is one PostgreSQL access method with one relation-owned lifecycle. It can
store exact BM25 postings or a unified lexical/semantic posting space, but it
does not split those modes into separate services, indexes, or APIs.

This document describes the current Beta 1 implementation, not a deployment
record or a future design. [Query semantics](query-semantics.md) defines the
individual overloads; [maintenance lifecycle](maintenance-lifecycle.md) defines
the detailed scheduling and publication contract. The source map below provides
entry points for checking both against the code.

## Product Contract

| Concern | BM25 | `sae = true` / Sparse Semantic Retrieval (SSR) |
| --- | --- | --- |
| Access method | `USING ii42` | `USING ii42` |
| Application search | `ii42_query(...)` | `ii42_query(...)` |
| Durable authority | Index relation | Index relation |
| Scoring input | Lexical terms | Lexical and semantic atoms |
| Mutation policy | Realtime, eventual, manual | Eventual-only |
| Model runtime | Not required | Shared workers required |
| Maintenance | Common selector and root publication | Common selector and root publication |
| Drop | PostgreSQL `DROP INDEX` | PostgreSQL `DROP INDEX` |

The stored index options select dispatch. Applications do not choose an
internal scorer or publish model output separately.

## Physical Shape

The current format is page-native v3. PostgreSQL relation pages contain:

- a metapage with the checked root and durable counters;
- a COW manifest that names current immutable objects;
- a COW document/version directory for MVCC identity and retirement;
- lexical and semantic posting extents in one atom namespace;
- active and pending linked-L0 frontiers containing transaction-tagged mutation
  records not yet folded into immutable extents;
- optional durable folds, semantic accelerator objects, and their TID and
  `INCLUDE`-value scope directories.

L0 append happens before transaction completion. Physical records can therefore
be committed, unresolved, or aborted; their presence is not proof of query
visibility. Transaction state, document/version identity, and the statement's
heap snapshot determine which rows may be returned.

COW means constructing replacement objects and copying affected directory
paths while reusing unchanged immutable references. Text indexes also maintain
lexical and prefix lookup trees; a new term does not require rewriting the
complete vocabulary. These are relation-owned objects, not independent SQL
indexes. Their codecs and layout are described in
[Convergent segmented index](convergent-segmented-index.md).

```text
checked metapage root
    |
    +-- document/version COW authority
    +-- lexical/prefix lookup and atom directory
    +-- immutable posting extents
    +-- durable folds and semantic accelerator baseline
    |       `-- forward/transpose, TID directory, INCLUDE scope
    `-- active and pending linked L0
```

A checked metapage switch publishes the descendant root only after its
referenced objects have been prepared and validated. Earlier objects remain
readable while protected by readers. Retired-page reuse requires a separate
reader fence; failed or deferred reuse does not invalidate publication.
Relation changes and publication use PostgreSQL WAL for logged indexes.

There is one current checked root per index, but it can reference an older
compatible accelerator baseline. Current root identity, accelerator baseline
identity, and shared-cache identity are deliberately not one freshness counter.
PostgreSQL transaction rollback is also distinct from a COW root switch: aborted
mutation records must remain nonvisible even if their pages were written.

## Unified Scoring

In semantic mode, lexical terms and model atoms share one posting namespace.
The query encoder produces weighted atoms. Exact execution accumulates their
matching impacts in a common document score; bounded-approximate execution uses
accelerator candidate selection and residual scoring over that same posting
space. This is one scoring model, not a promise that every route exhaustively
visits every posting or performs only one sort.

This is not late fusion between a BM25 index and an ANN index. The model
checkout defines atom identity, normalization, and scoring profile before the
posting vector enters the common scorer.

The model checkout, tokenizer, runtime ABI, and precision contract must agree
between encoding and stored postings. Runtime availability alone does not prove
that an index's model contract matches. Exact-BM25 rowset functions and operators
remain installed for owner-run diagnostics and regression isolation. They reject
semantic-enabled indexes and are not a second application route.

## Build Lifecycle

`CREATE INDEX` and explicit `REINDEX` are controlled whole-corpus operations:

1. scan the heap through PostgreSQL's `table_index_build_scan` protocol, using
   the visibility and validation rules of the requested index-build operation;
2. tokenize BM25 input or submit bounded document batches to the shared model
   runtime;
3. build document, lexicon, posting, and fold objects;
4. validate the complete physical closure;
5. publish one checked exact root; optional semantic accelerator construction
   is subsequent maintenance, not a prerequisite for build completion.

Builders use in-memory, compact, spill, or semantic-stream strategies selected
from the input shape and configured memory budget. A completed heap scan is not
completed publication. Assembly, validation, PostgreSQL index validity, and the
SQL command's successful completion still matter. COW publication does not
override PostgreSQL's DDL locking, replacement, or rollback rules; an ordinary
`REINDEX` is not thereby a nonblocking online operation.

## Mutation Lifecycle

Foreground DML never rebuilds the corpus.

### BM25

- `realtime` appends exact lexical state that the exact route reads
  immediately;
- `eventual` also records lexical mutations in linked L0 and converges through
  automatic maintenance;
- `manual` records staleness while serving the last published BM25 state until
  explicit maintenance. That explicit operation can rebuild from the heap and
  take stronger locks; it is not the automatic incremental path.

### Semantic-enabled

SSR is eventual-only. Foreground `INSERT` and indexed-column `UPDATE` append:

- exact lexical posting state;
- document/version identity;
- semantic-pending state.

No foreground document inference occurs. The row is lexical-searchable through
the exact route after commit. The default bounded-approximate accelerator may
continue to serve its older compatible baseline and omit post-baseline rows.
There is no maximum serving age: an overdue or failed refresh does not itself
expire a compatible baseline. Background work is scheduled by debt and periodic
eligibility, not by imposing freshness waits on queries. Shared workers encode
bounded changed-document batches and append semantic completion only after
revalidating the corresponding document versions.

`DELETE` and superseded versions become immediately invisible through heap
MVCC. `VACUUM` supplies exact dead-TID retirement so physical statistics and
space can converge safely.

## Maintenance Lifecycle

An automatic maintenance attempt selects one action, rather than draining all
debt synchronously:

1. complete semantic-pending documents;
2. seal an eligible linked-L0 interval;
3. publish a newer semantic accelerator baseline when its source authority is
   complete enough and no pending seal takes priority;
4. compact selected extents;
5. build or refresh a term-local fold;
6. reclaim objects after the reader fence permits reuse.

The list describes action classes, not an unconditional priority order. Pending
seals and writer headroom take priority; a due accelerator refresh also gets an
opportunity when semantic work remains continuously actionable. Per-index work
locks deduplicate attempts. Expensive staging is outside the publication fence;
the worker revalidates captured authority before switching it. Newer L0 records
remain attached rather than being consumed by an older build.

Most mutation convergence is incremental: changed-document encoding, selected
L0 frontiers, extent ranges, term folds, and bounded retired-page reuse. A
semantic accelerator refresh is an important exception: it can scan the entire
sealed posting baseline and construct corpus-sized forward/transpose data. Its
scope builder also fetches indexed `INCLUDE` values under short MVCC snapshots,
in batches of at most 256 document slots, so external TOAST is read safely
without holding one snapshot for the entire pass. It does not re-encode every
document through the model. One selected action therefore does not imply
constant work, a small RSS footprint, or a strict wall-clock deadline.

The accelerator builder uses a separate build lock and releases the ordinary
maintenance lock during that work. Other immutable-root publications normally
defer to avoid discarding the build, while foreground append, semantic
completion, and safe L0 rotation can continue. If both L0 frontiers need space,
urgent sealing wins and the accelerator retries. The publication check retains
the latest compatible L0 frontiers.

Small debt is eligible for periodic work through
`ii42.maintenance_low_debt_interval_ms` (one hour by default); capacity pressure
and missing required artifacts have immediate paths. This interval is a
scheduling policy, not a baseline TTL. Retry cooldown prevents repeated optional
build failures from monopolizing worker capacity. Explicit maintenance can also
advance low debt without waiting for the timer. Automatic maintenance and
automatic preload are disabled when `ii42.maintenance_worker_limit = 0`.

Query visibility, semantic completeness, accelerator freshness, and shared
residency are separate progress axes. Foreground commits depend only on the
relation authority. Semantic completion adds the configured model signal;
accelerator refresh re-derives no model output and only changes the bounded
query execution shape. A compatible older baseline remains usable with a
bounded candidate revalidation while writers and maintenance continue
independently. It is revalidated for deletions and replacements but may omit a
newer delta until a refresh successfully publishes. This is not an exact merge
of every pending mutation into each foreground query.

## Query Lifecycle

The explicit-hit `ii42_query(..., k, ...)` route:

1. verifies access to the indexed table and rejects unsupported RLS or
   partitioned-parent shapes;
2. reads the index options and selects BM25 or semantic encoding;
3. pins the relation, snapshot, checked root, and visible linked-L0 frontier;
4. for semantic indexes, prefers an eligible immutable accelerator baseline,
   with bounded overfetch and current-row revalidation when stale; otherwise it
   attaches a current exact-root resident fold or opens page-native posting
   cursors; the semantic `tid[]` overload applies statement-local membership before
   accumulation;
5. returns `(ctid, doc_id, score)` in the index's exact or declared
   bounded-approximate scoring profile.

Scalar `ii42_query(...)` markers are different SQL syntax for planner-native SSR
retrieval. A supported `ORDER BY score DESC LIMIT k` shape becomes
`Custom Scan (II42 Search)`; the marker is not an ordinary per-row scorer.

### Filtering And Fallbacks

Both planner-native and structured JSON filtering may use an older compatible
scope baseline. Neither requires all acceleration metadata to become current
before serving. Returned candidates must still satisfy current heap visibility
and predicates. This guarantees membership, not exact-current top-k recall.

```text
query + index options -> lexical/query-model encoding
                                  |
                    capture checked read authority
                                  |
                    choose membership constraint
                    /             |             \
                  none         scope          explicit TIDs
                    \             |             /
                     eligible semantic accelerator?
                         /                 \
                       yes                  no
              bounded candidates     exact resident/page-native
                         \                 /
                    current document/heap recheck
                                  |
                              ranked hits
```

Scope metadata belongs to the published accelerator and indexes eligible
`INCLUDE` values, not model-encoded metadata. Supported comparisons include
equality, overlap, range, and `ILIKE`, with typed comparison and collation rules.
The overloads deliberately have different fallback behavior:

| Route | Scope fast path | If that path cannot satisfy the request |
| --- | --- | --- |
| Planner-native SSR | Fully supported predicates; one probe for up to `4 * k` candidates, then heap-qual recheck | An unavailable/unsupported scope or fewer than `k` surviving hits triggers collection of the full snapshot-visible allowed TID set and filtered scoring. |
| Structured JSON SSR | Fully scope-backed filters; one probe for up to `4 * k`, then current-predicate recheck | A successful scope route may return fewer than `k`; underfill alone does not force full membership materialization. Partial/unavailable scope can require other routes, including SQL membership resolution. |
| Explicit `tid[]` SSR | Caller supplies statement-local membership | The set is a hard membership boundary, but the selected scoring route can still be bounded-approximate. |

Without a usable scope, the JSON implementation first probes SQL membership
with a 65,536-match limit plus one overflow witness. A completed probe supplies
the allowed TIDs directly. An overflowing probe may be followed by a bounded
global-ranked prefix when the index has a usable physical route; otherwise,
or if that prefix is insufficient, it falls back to full SQL membership
resolution. The match limit is not a bound on rows scanned or elapsed time, and
the prefix is not a resumable exact filtered iterator. PostgreSQL may use
metadata B-tree, GIN, or trigram indexes for the resolver; otherwise it may scan
a large part of the source table. Bounded cursor batches do not bound the final
allowed set: with `M` matching TIDs it still retains roughly `8 * M` bytes of
keys plus other
scratch and sorting work. Broad-filter fallback can therefore dominate latency
even when the scoring accelerator is healthy.

Query backends do not load a document model or re-encode the corpus. They can
nevertheless read source rows for filtering and visibility, perform cold page
reads, and wait for query encoding in the shared runtime. See
[Query semantics](query-semantics.md) for route eligibility and approximation.

### Warm Serving Identity

A selected, admitted root may expose a pointer-free exact-root fold in the
shared arena; backends lease that posting image instead of reconstructing a
private copy. Semantic roots with an eligible accelerator instead retain its immutable
baseline across delta publication. Query-metadata warm markers follow that
serving accelerator's object reference and baseline sequence; the exact fallback
uses the manifest and root identity. A changed L0 frontier alone is not a reason
to discard compatible baseline metadata.

`ready_baseline_delta` with `baseline_current=false` can be a valid serving
state. `query_metadata_warm=false` means the relevant metadata is not fully
recorded as warm; `resident_fold_current=false` does not imply that the
accelerator route is unavailable. None of these flags alone measures latency
or proves a fault. Readiness, chosen route, compatibility, debt, and actual warm
query behavior must be inspected together.

Automatic resident replacement and prewarming are worker tasks, not a demand
that each query first finish maintenance. This is behavioral separation, not
physical resource isolation: cold reads, CPU/I/O pressure, runtime queues, and
short publication/reuse fences can still affect foreground latency. Explicit
DDL and repair operations have their own stronger locking requirements.

## Process And Memory Ownership

| Resource | Owner |
| --- | --- |
| Durable postings and roots | PostgreSQL index relation |
| Relation page cache | PostgreSQL shared buffers / OS cache |
| Query scratch | Calling backend; depends on selected route, candidates, and membership size, not only `k` |
| Model sessions and tokenizers | Runtime worker-private memory, reused across shared-runtime requests |
| Runtime request queues | Postmaster-owned shared arena |
| Serving-authority markers, HOT_FOLD, exact resident folds, and metadata projections | Shared arena, disposable and subject to admission |
| Accelerator construction workspace | Maintenance backend; can be corpus-sized and has separate admission checks |

BM25 can run without the semantic runtime. Semantic-enabled indexes require
`shared_preload_libraries = 'ii42'` and a positive
`ii42.shared_runtime_size`; they fail closed rather than loading model
state in an application backend.

Model runtime workers and durable semantic accelerators are different things.
`ii42.runtime_accelerators` configures optional remote inference services for
document build and maintenance throughput. It does not name the on-index
forward/transpose/scope objects used for search. Removing those services after
their batches finish does not remove published query accelerators; local
runtime workers remain necessary for query encoding and later semantic writes.
The default reserved query lane keeps document work from taking every runtime
lane when multiple workers are available, but does not isolate host CPU or I/O.

Shared-arena admission, page warming, and per-backend workspace limits are
separate budgets. Warming a relation does not copy all its pages into Evoke's
shared arena, and increasing a workspace budget does not shrink on-disk data.

## Correctness Boundaries

- Relation pages and the checked root are authoritative. Warm markers and
  projections are reconstructible accelerators.
- WAL recovery of a root publication exposes the old or complete new authority,
  not a root naming partially staged objects. Logical transaction visibility
  remains governed by PostgreSQL even when physical L0 records survive abort.
- Each statement keeps its pinned root and heap snapshot. Later statements see
  the current checked root and snapshot-visible linked-L0 frontier; long
  `REPEATABLE READ` transactions preserve row visibility, not a historical
  ranking root across statements.
- A standby replays relation pages and may prewarm them, but performs no
  primary-side durable maintenance while in recovery.
- Runtime/model contract mismatch fails readiness and search closed.
- A caller-supplied TID set is an exact membership boundary from the same
  indexed relation and statement snapshot: no out-of-set row can be returned.
  Ranking still follows the selected exact or bounded-approximate SSR route, so
  a stale accelerator may omit an allowed post-baseline row. TID membership is
  query scratch, not another durable index or lifecycle.
- Row-level security and globally ranked partitioned-parent search remain
  unsupported. A caller-supplied TID set is not an RLS policy boundary, and
  independently ranked children cannot provide one global corpus top-k.

## Public Authority

The product surface is intentionally narrow:

- scalar `ii42_query(...)` for planner-native semantic table retrieval and
  `ii42_query(..., k, ...)` for explicit hit retrieval;
- `ii42_index_options(...)` and `ii42_index_status(...)` for configuration and
  readiness;
- `ii42_index_maintain(...)`, `ii42_index_try_maintain(...)`, and
  `ii42_index_maintain_due(...)` for convergence;
- PostgreSQL `DROP INDEX` for relation-owned teardown.

Detailed diagnostic functions are privileged and do not create alternate
storage or query authority.

## Implementation Map

These are implementation entry points, not additional public APIs. Orchestration
still lives substantially in `ii42_am.c`; the narrower files contain reusable
codecs, selection, locking, and execution components.

| Concern | Source | Entry point |
| --- | --- | --- |
| Installed overloads and SQL dispatch | [Extension SQL](../sql/ii42--0.2.5.sql) | `ii42_query` |
| PostgreSQL build and mutation callbacks | [AM orchestration](../src/ii42_am.c) | `ii42_ambuild`, `ii42_aminsert` |
| Immutable manifest and accelerator eligibility | [Segment contract](../src/ii42_segments.c) | `ii42_segment_manifest_validate`, `ii42_segment_manifest_semantic_accelerator_eligible` |
| Relation-backed COW object writes | [Page storage](../src/ii42_segment_pages.c) | `ii42_segment_pages_write_lexicon_cow_objects` |
| Transaction-tagged L0 handling | [Mutation primitives](../src/ii42_am_mutation.c) | `ii42_am_delta_record_states` |
| Work hints, fairness, and retry cooldown | [Scheduler](../src/ii42_am_scheduler.c) | `ii42_am_scheduler_work_hint_mark`, `ii42_am_scheduler_work_hint_defer_accelerator` |
| Separate accelerator build locking | [Maintenance locks](../src/ii42_am_maintenance.c) | `ii42_am_try_accelerator_build_lock` |
| Corpus-sized accelerator preparation | [Accelerator builder](../src/ii42_am_accelerator.c) | `ii42_am_prepare_accelerator_baseline` |
| Bounded scope snapshots and TOAST reads | [Scope builder](../src/ii42_scope_pg.c) | `ii42_scope_build_for_index` |
| Planner scope probe and fallback | [Planner integration](../src/ii42_planner.c) | `ii42_planner_build_scope_filter`, `ii42_planner_collect_allowed_tid_keys` |
| Structured scope recheck and warm identity | [AM orchestration](../src/ii42_am.c) | `ii42_am_try_structured_scope_filter`, `ii42_am_read_unified_warm_marker` |
| SQL membership probes and full fallback | [Filter resolver](../src/ii42_filter.c) | `ii42_filter_collect_tid_keys` |
| Retired-page reader fence | [Reclamation](../src/ii42_am_reclamation.c) | `ii42_am_acquire_convergent_reader_fence` |

## Further Reading

- [Convergent segmented index](convergent-segmented-index.md)
- [Index policy](index-policy.md)
- [Query semantics](query-semantics.md)
- [Maintenance lifecycle](maintenance-lifecycle.md)
- [Shared runtime and residency](shared-runtime-and-residency.md)
- [API reference](api-reference.md)
- [Testing and validation](testing-and-validation.md)
- [System technical report](technical-report-evoke-system.md)

Research, performance, and technical-report documents are maintained as
separate evidence surfaces. Product architecture changes only when current
source, installed SQL, lifecycle tests, and deliberately refreshed evidence
agree.
