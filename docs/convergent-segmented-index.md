# Convergent Segmented Index

Status: current Beta 1 implementation design.

Design version: 1.60. Reviewed against the source tree on 2026-09-04.

This document is the current design authority for Evoke storage, mutation,
query, and maintenance. It describes one PostgreSQL index relation containing
BM25 evidence and optional Sparse Semantic Retrieval (SSR) semantic postings,
with one checked publication lineage. It is not a deployment inventory or a
claim that every workload has passed a fresh performance qualification.

[Architecture and design](architecture-and-design.md) provides the system
overview. [Query semantics](query-semantics.md) defines the public retrieval
contract, and [Maintenance lifecycle](maintenance-lifecycle.md) defines
operational scheduling and state interpretation. The
[development record](archive/engineering/convergent-segmented-index-development-record.md)
preserves the completed implementation programme and its dated experiments.
Future changes belong in the [Product roadmap](product-roadmap.md), not in a
second design or release ledger.

## Product Shape

The central separation is between durable posting evidence, compiled read
surfaces, and volatile residency. They share a relation lifecycle but need not
advance together.

### Converged Architecture

```text
PostgreSQL heap + transaction visibility
    |
    +-- CREATE INDEX / REINDEX: heap build + optional document encoding
    |
    `-- automatic DML: tokenize changed value; append txn-tagged L0
                                  |
                    commit controls logical visibility
                                  |
                      active L0 -> pending L0 -> seal
                                  ^                 |
                                  |                 v
                 semantic completion         immutable segments
                 for matching version              |
                 (shared runtime workers)           +-- selected compaction
                                                    +-- term-local folds
                                                    `-- accelerator build
                                                           |
            +----------------------------------------------+
            v
  One relation / checked metapage / immutable manifest lineage
    +-- scorer/model contract and lexical catalog
    +-- term, document/version, hash-lookup, prefix COW roots
    +-- sealed payloads, fold coverage, retirement evidence
    `-- optional semantic accelerator + scope baseline references
            |
            +-- exact read plan: folds + extents + visible linked L0
            `-- eligible bounded SSR plan: compatible baseline candidates
                         |
                 current-snapshot heap/TID recheck
                         |
                       results

Shared preload/residency caches attach to these checked read identities.
They do not own posting facts and do not publish a second index root.
```

The ordinary exact route can read committed lexical changes without waiting
for semantic inference or a seal. The default bounded SSR route may instead
continue using a compatible older accelerator baseline. Its returned rows
must pass current-snapshot validation, but newer matching rows can wait for
background convergence. Exactness of the durable posting representation does
not imply exact-current candidate completeness for every query route.

Queries do not wait for an optional accelerator build to finish and do not
perform document inference to repair missing semantics. This is a dependency
boundary, not a guarantee of zero interference: workers and queries still
share CPU, storage, buffers, memory bandwidth, and PostgreSQL locks.

## Product Invariants

- One `USING ii42` relation owns posting payloads, document versions, linked
  mutation debt, derived durable objects, and reclamation evidence.
- The checked metapage and manifest lineage select the physical read surface.
  PostgreSQL transactions and heap MVCC still decide row visibility; the
  manifest is not an alternative transaction rollback mechanism.
- Automatic foreground DML appends changed lexical evidence and identity. It
  does not compact old segments or infer document semantics. Finite append
  capacity can still cause backpressure under sustained overload.
- Exact folds preserve their covered posting events. A bounded semantic
  accelerator is a different read policy and must not be described as an
  exact fold or silently given exact-current recall guarantees.
- Compatible accelerator and scope baselines have no maximum stale age. A
  delta alone is not grounds to discard previously published acceleration.
- Mutation seals and selected compaction/fold work avoid rewriting unrelated
  posting payloads. Optional accelerator preparation is an important
  exception: it can traverse a corpus-sized sealed baseline.
- PostgreSQL owns creation, locking, WAL, VACUUM, replication, `REINDEX`, and
  `DROP INDEX`. There is no independent SQL object lifecycle for an
  accelerator, scope directory, or fold.

## Four Convergence Clocks

| Clock | What advances it | What lag means |
| --- | --- | --- |
| Lexical visibility | Transaction commit and snapshot-visible L0 replay | The exact route sees committed lexical changes; physical append alone is not commit |
| Semantic completeness | Worker completion or quarantine for a checked document version | Newly changed documents may have lexical evidence without semantic impacts |
| Accelerator freshness | Publication of a new derived baseline | Compatible old candidates remain usable; newer matches and scores may lag |
| Query warmness | Shared metadata, resident image, and hot projection admission | A valid read surface may be cold; eviction is not data loss |

Structural debt is managed underneath these clocks by rotation, sealing,
compaction, folds, VACUUM retirement, and page reclamation. Reducing structural
debt does not prove that an accelerator is current or that a query is warm.

`ready_baseline_delta` is a serviceable baseline-plus-delta state, not an
availability failure. `baseline_current=false` records remaining convergence;
it is not a TTL. `query_metadata_warm`, `resident_fold_current`, and
`hot_fold_current` describe different artifacts and are not latency SLOs.

The default `ii42.maintenance_low_debt_interval_ms` is one hour. It allows
small outstanding debt to receive periodic attention without rebuilding after
every write. It neither expires a serving baseline nor promises completion
within one hour. Hard append pressure, pending-seal work, and missing or
incompatible required artifacts follow their own eligibility rules.

## Physical Architecture

```text
index relation
  metapage
    +-- checked manifest reference and published high-water mark
    +-- active linked-L0 frontier
    `-- optional immutable pending-L0 frontier

  immutable manifest
    +-- exact segment descriptors and ancestor-owned payload references
    +-- scorer/model contract
    +-- COW term directory -> extents / major / minor / impact folds
    +-- COW document directory -> version / TID / semantic / retirement state
    +-- lexical catalog + hash lookup + ordered prefix lookup
    +-- optional accelerator directory -> geometry / forward / transpose
    +-- optional scope metadata attached to that accelerator baseline
    `-- authenticated retired page ranges

  WAL-protected pages
    new unreachable objects -> checked root exchange -> reachable objects
    retired objects -> reader-safe reclamation -> reusable pages
```

### Segment Identity And Manifest

The metapage read root binds an immutable manifest to active and pending L0
frontiers. Foreground append advances bounded frontier metadata rather than
serializing the manifest or all existing mutation records. A seal clears only
the pending frontier it consumed and preserves writes that arrived in the
active frontier while it was building.

Immutable references contain physical page-chain locators, object identity,
owner manifest identity, byte length, and checksums. A descendant can retain
an ancestor-owned object without copying its payload merely to change its
owner. Readers discover objects through checked references, not a scan of
relation pages or a second object-id locator table.

Object writers bind children bottom-up: write a child, obtain and validate its
actual physical reference, then serialize the parent. L0 and immutable-object
pages may interleave. Correctness depends on references and checked root
publication, not on predicting a contiguous allocation for the whole build.

The native metapage family is v3. Any non-v3 root
fails closed at installed query, mutation, maintenance, preload, and status
boundaries. Individual codecs have their own version numbers; for example,
the current manifest writer emits version 14. That is not a claim that every
object uses version 3 or that arbitrary historical beta roots are compatible.
Supported same-family decoding is defined by the checked codec, not by a
historical release label.

### Canonical Segment Payload

Segments carry term-ordered runs with explicit scoring kinds, document-slot
identity, tuple-version records, retirements, and semantic state transitions.
Sparse local-to-global slot maps let a semantic-only segment target existing
owners without duplicating their lexical document records.

Pure lexical neutral runs retain term frequencies for BM25 evaluation.
Semantic-enabled unified runs use the packed posting representation and the
index's immutable precision/alpha contract. It is incorrect to model every
posting as an unconditional four-byte value: packed semantic impacts support
the declared `f32`, `fp16`, and approximate `u8` representations. The generic
defaults remain `f32`, block64, and `semantic_alpha_mass = 1.0`; an explicit
deployment profile such as `u8`/`0.50` does not change those defaults.

Compaction can leave an empty history-barrier descriptor when all events in a
selected interval have been retired. It preserves sequence coverage without
inventing a separate tombstone index. Payloads, barriers, and derived objects
remain subject to the same checksum, ownership, and reachability rules.

### Copy-On-Write Directories

| Structure | Authority or purpose | Incremental change |
| --- | --- | --- |
| Term directory | Stable term id, physical DF, extent chain, fold coverage and references | Path-copy entries for affected terms; inherit unrelated entries |
| Document/version directory | Slot incarnation, heap TID, version/fingerprint, semantic state and retirement | Path-copy affected version records and ancestor summaries |
| Lexical catalog | Normalized bytes for stable numeric term ids | Append a catalog-id range for unseen terms; retain old blobs |
| Lexical hash COW | Exact normalized-byte lookup into that catalog | Rewrite addressed buckets and hash ancestry |
| Ordered prefix COW | Bounded expansion of normalized prefixes | Merge new keys into affected ordered leaves and copy their ancestry |

```text
old root R0                        staged successor R1
  +-- A0                             +-- A1 (new path)
  |    +-- D                         |    +-- D (same immutable object)
  |    `-- E0                        |    `-- E1 (changed leaf)
  `-- B                              `-- B (same immutable object)

The old objects are not overwritten. New child references are bound before
the descendant root is published. Retirement and reuse happen separately.
```

Hash lookup uses a packed 64-way trie and compares full normalized bytes, so
hash collisions do not weaken equality. Its bucket target is 16,000 bytes,
chosen to fit within two payload pages on standard 8 KiB PostgreSQL builds.
This improves utilization over the earlier 4 KiB target. It is not arbitrary
sub-page packing of unrelated COW objects. See
[lexicon COW](../src/ii42_lexicon_cow.h).

Prefix leaves hold at most 128 entries and internal nodes at most 48 child
references. The stable term-id suffix is **not** necessarily a suffix in
lexicographic order. `ii42_prefix_cow_build_external_append_patch` sorts new
keys by bytes and patches every affected subtree, reusing unaffected children;
it does not only append down the rightmost path. Prefix expansion seeks the
first possible leaf and stops at the prefix boundary or expansion budget.

Ordinary text lookup and pending-seal resolution use these checked indexes
rather than reconstructing all vocabulary strings. Explicit full builds,
diagnostics, and derived images can still perform broader reads; "changed-key
COW" is not a blanket claim that all operations are independent of vocabulary
or document count.

### Stable Document Identity

Posting slots identify immutable document-version incarnations, not eternal
heap TIDs. The COW record binds the slot to its version, TID, semantic input
fingerprint, and state. Safe slot reuse must prove that old postings and
snapshots cannot confuse the replacement with its predecessor. A matching
numeric slot alone is insufficient for semantic completion publication.

A HOT successor with unchanged indexed input can still satisfy the original
owner's fingerprint after PostgreSQL follows the visible tuple chain. A
non-HOT update has a new TID and must be recorded even when PostgreSQL supplies
`indexUnchanged`; treating that hint as permission to omit the version would
lose the new visible row.

## Write Lifecycle

### Foreground Transaction

```text
INSERT / indexed UPDATE / non-HOT replacement
    -> tokenize indexed values
    -> append txn-owned lexical record and document identity to active L0
    -> for sae=true, record semantic-pending fingerprint (no document inference)
    -> PostgreSQL COMMIT / ABORT / PREPARE decides visibility
    -> commit hint makes background work discoverable
```

L0 can physically contain unresolved or aborted records. Replay uses
transaction and heap visibility; aborted records are physical cleanup debt,
not searchable facts. Prepared transactions and savepoint rollback follow
PostgreSQL rules.

SSR mutations are eventual-only. Pure BM25 also supports realtime and manual
policies. Manual mode is not the automatic incremental path: it can mark an
index stale and explicit maintenance can rebuild from the heap under stronger
relation locks. Do not apply automatic-DML latency claims to that operation.

### Initial Build

`CREATE INDEX` and `REINDEX` use PostgreSQL's `table_index_build_scan` protocol.
The builder collects lexical evidence and, for SSR, uses shared runtime
workers for document encoding. It writes the initial native payload/COW
closure before publishing a usable root. Heap scan at 100 percent does not
mean serialization, validation, publication, or transaction completion is
finished. Optional query acceleration has a separate preparation step after
the base evidence exists.

### Semantic Completion

The worker discovers pending identities, validates current visible input,
encodes a batch, and rechecks the tuple chain and fingerprint before appending
completion or quarantine. The final compare-and-append checks the complete
source COW version under the append lock, including owner, retirement, and
residency, not only the fingerprint.

Completion is an event for the existing version. It may overlay an L0 owner
or an already sealed lexical owner. Empty output, quarantine, and stale input
can be represented without making a second document owner. Compaction retains
the latest necessary transition or absorbs it into the selected owner when
both are covered by the replacement.

Sealing does not wait for model inference. Changed versions can seal as
`SEMANTIC_PENDING`, and later completion can seal as a sparse semantic-only
payload. Successfully encoded unchanged documents are not re-inferred by
compaction, folds, accelerator preparation, or warming. Query encoding remains
online model work, but missing document semantics are never inferred by the
query backend.

### Rotation And Sealing

Capacity pressure or eligible periodic low debt rotates active L0 into the
pending frontier. Rotation is a short WAL-protected metapage change, not a
payload rewrite. New writes use the fresh active frontier while the worker
seals the immutable pending one.

Seal resolves changed terms, constructs their local runs and version events,
appends unseen catalog entries, and writes affected COW paths. It retains old
payload references and unaffected neutral folds. Concurrent active-L0 writes
are preserved by the final checked root exchange. Finite frontiers and extent
limits mean hard pressure may require compaction or a structural fold before
another seal can publish.

### Global BM25 Statistics

Corpus counts, field lengths, physical DF, and retirement correction belong
to checked root/COW/L0 state. PostgreSQL's insertion callback does not identify
every obsolete predecessor. Heap recheck hides dead rows, but deleted or
superseded versions can contribute to statistics until VACUUM materializes
their retirement. The reported contract is
`retirement_statistics = vacuum_convergent`, not exact-current heap statistics
at every instant.

Once retirement evidence is present, exact scoring corrects it while
compaction later removes consumed physical runs. Statistics drift can disable
an epoch-specialized **exact lexical impact** image. That restriction must not
be copied to the separate bounded semantic baseline, which may legitimately
remain eligible across later writes and statistics changes.

## Exact Read Shapes And Fold Invariants

The following is the exact-route submodel, not a description of default
bounded SSR candidate selection:

```text
read_plan(key, snapshot)
    = inherited_major(key, <= c0)
    + optional_minor(key, (c0, c1])
    + visible_immutable_extents(key, > c1)
    + visible_linked_L0_events(key, snapshot)
    - applicable_retirements(key, snapshot)
```

Coverage intervals must not overlap or leave a hole. A fold replaces only the
events it proves it covers. Later writes and semantic transitions form a tail;
they do not invalidate an unrelated neutral prefix. Segment compaction must
respect fold watermarks, or use the checked current-family boundary proof
before replacing descriptors.

Neutral BM25 scoring uses global statistics, not independently scored segment
result lists. For the Lucene-style variant used in the system report:

$$
S_{lex}(q,d) = \sum_{t \in q} IDF_t
\frac{tf_{t,d}}{tf_{t,d}+k_1(1-b+b|d|/avgdl)}.
$$

Here `tf` is term frequency, `IDF` is the configured inverse document
frequency, and `avgdl` is average document length. This convention omits the
global `(k1 + 1)` multiplier; other supported variants have their own scoring
rules. For an SSR unified representation, query and document weights address
the same lexical/semantic posting namespace:

$$
S(q,d) = \sum_{a \in U} w_q(a)\,w_d(a).
$$

The scorer's normalization and model/precision contract determine those
weights; this is not BM25 plus an independently retrieved semantic top-k with
late fusion. See the [model report](technical-report-evoke-model.md) for derivation.
"Exact" refers to the selected stored representation and visibility/statistics
contract. It does not undo `u8` quantization or alpha-mass pruning.

### Structural And Workload Compaction

Selected segment compaction is index-to-index work: it consumes stored
postings, version state, and retirements without retokenizing unchanged text
or running the model. It prefers bounded ranges of similarly sized segments.
When a whole-segment merge would be too expensive, a term-local structural
fold can reduce the pressured key without rewriting unrelated posting lists.

Optional workload folds use observed posting-key work, not stored query result
lists. Major/minor/tail organization makes small updates accumulate behind a
stable large prefix. A minor carry requires geometrically comparable tail
work; promotion can combine major and minor when justified. Structural
pressure can override optional admission to restore the checked read-surface
bound. Policy heuristics are not additional correctness metadata.

Source segments may still serve other terms after a fold is published. Fold
coverage alone is not permission to free them; reclamation follows manifest
reachability and reader safety.

Current implementation bounds distinguish format capacity from scheduling
targets:

| Boundary | Current value | Meaning |
| --- | --- | --- |
| Active L0 capacity | 131,072 records / 1,024 pages | Finite append capacity, not a serving-baseline TTL |
| Capacity rotation | Half of either L0 capacity | 65,536 records / 512 pages; periodic small-debt rotation is separate |
| Selected segment compaction | At most 8 input segments and 64 MiB input | Bound for that action, not for a complete accelerator preparation |
| Term extent format capacity | At most 64 extents per term | Checked current-format ceiling, not the target steady-state fanout |
| Compaction extent pressure | 6 extents | Scheduling pressure, distinct from the format ceiling |

These values come from [segment limits](../src/ii42_segments.h) and
[AM policy constants](../src/ii42_am.c). Earlier design revisions described
32 as the format ceiling and eight as a hard steady-state boundary; those
historical figures must not be substituted for current codec validation.

### Compiled Views And Residency

There are three exact read-shape layers:

1. Authoritative events: linked L0, immutable runs, versions, and retirements.
2. Persistent compiled views: neutral folds and eligible epoch-specialized
   impacts, checked and referenced by the same manifest/COW lineage.
3. Volatile images: shared resident folds, hot projections, metadata, and
   PostgreSQL-buffer residency attached to compatible read identities.

Exact paths choose eligible compiled runs or ordinary page-native runs.
Conservative block bounds can prune only when they cannot hide a competitive
candidate. Missing optional acceleration selects another valid route;
corrupt authoritative objects remain errors, not approximate answers.

The implementation has several physical scorer strategies, including packed,
block-oriented, materialized-query, resident, and narrow hot-term routes. The
shared logical scoring contract does not mean every path executes one
identical accumulator allocation or top-k loop. Cold exact queries can be
expensive even when extent fanout is bounded.

## Derived Semantic Query Accelerator

### Baseline Construction And Publication

The optional semantic accelerator contains geometry, complete candidate
forward rows, transposed data, and eligible `INCLUDE` scope metadata. It is
derived from existing unified postings, not a separate ANN index or another
document encoder. Current policy 7 uses directory v10, forward/transpose v6,
and scope v6; these are accelerator subformats, not the native metapage
version.

Preparation can traverse a corpus-sized sealed baseline. Temporary transpose
files and bounded working batches control memory but do not turn total work
into an incremental changed-row operation. Scope preparation fetches included
heap values in short snapshots covering at most 256 document slots per batch;
copied values must outlive that snapshot safely, including external TOAST.

```text
maintenance selects eligible baseline preparation
    -> claim per-index accelerator-build lock
    -> release ordinary maintenance ownership/reservation as applicable
    -> capture checked sealed source; build unreachable derived objects
         |
         +-- serving queries keep using the previous eligible baseline
         +-- foreground append and semantic completion can continue
         `-- safe L0 rotation can continue
    -> revalidate source contract and immutable ancestry
    -> publish new references, preserving the latest linked-L0 frontiers
    -> preload adopts the newly serving identity
```

The separate build lock deduplicates preparation without making it a second
publication authority. Other immutable-root work normally defers while that
build is in flight, preventing repeated loss of corpus-sized work. If both
L0 frontiers reach hard pressure, urgent sealing wins; the accelerator may
need to retry rather than block writers. This is coordination, not unlimited
parallelism among all maintenance actions.

### Serving A Compatible Older Baseline

A current manifest may retain a reference to an older compatible accelerator
and scope baseline. The active root, baseline source sequence, and shared
warm identity are distinct values. Appending delta or publishing a compatible
descendant must not be treated as unconditional accelerator invalidation.

The bounded route generates baseline candidates, scores their complete
forward rows, and rechecks returned versions/TIDs under the current snapshot.
It can use bounded overfetch for stale candidates. It does not synchronously
merge the complete delta or rebuild a whole allowed universe merely because
the baseline is not current. New matches and ranking improvements arrive with
later publication; there is no maximum stale age.

Unsupported scoring shapes, incompatible policy/model/precision contracts, or
failed memory admission can select packed exact scoring instead. Missing
optional objects and corrupt checked storage are different cases: corruption
must be surfaced, not hidden as an ordinary stale baseline. Negative-weight
or diagnostic exact requests must follow their supported non-bounded route.

The candidate policy uses bounded seed and residual work; high-DF posting
lists can still be expensive. Some residual paths use a dense per-document
workspace with a 64 MiB admission limit. A memory bound can force a different,
slower exact path. Neither baseline readiness nor warm metadata establishes a
universal query-time bound.

Remote `ii42.runtime_accelerators` are a different feature: optional services
for document-encoding batches. Removing that remote fleet does not remove
relation-owned query accelerator objects. Local runtime workers still serve
online query encoding and subsequent semantic completion.

## Query Lifecycle

```text
scalar planner-native ii42_query        explicit-hit ii42_query(..., k, ...)
       |                                           |
supported ranked SQL -> CustomScan        structured JSON / TID / unfiltered
       +---------------------+---------------------+
                             v
           validate relation + index/model/field contract
                             |
           encode semantic query through shared runtime if needed
                             |
          choose filter membership and eligible serving baseline
                             |
        +--------------------+----------------------+
        |                                           |
bounded semantic candidates                 exact posting read plan
forward / transpose / residual              resident or page-native
        +--------------------+----------------------+
                             v
            current-snapshot TID / heap predicate recheck
                             |
               route-specific underfill/fallback policy
                             |
                           results
```

The scalar ranked-table form is SSR-specific and requires a supported
planner shape. Explicit-hit APIs also serve BM25. Unsupported SQL shapes do
not acquire the CustomScan optimization merely because they mention the
marker. Full public signatures and restrictions are in
[Query semantics](query-semantics.md).

### Filter Membership And Bounded Ranking

| Entry path | Fast membership route | Underfill or unsupported case |
| --- | --- | --- |
| Planner-native SSR SQL | All eligible AND predicates translated to baseline scope; bounded overfetch and current heap recheck | Can resolve the complete visible TID subset and score within it |
| Fully scope-backed structured JSON | Baseline scope, bounded overfetch, current recheck | May return fewer than `k`; no automatic complete-universe expansion just to fill it |
| JSON without a fully eligible scope | Bounded SQL membership probe; if complete, score its TIDs | Overflow may try an admitted global candidate prefix, otherwise use full SQL resolution |
| Explicit statement-local TIDs | Caller supplies membership for the current statement | Membership does not itself guarantee exact global SSR ranking |

Eligible JSON operations include `eq`, `in`, `overlap`, `range`, `ilike`, and
`ilike_any`. Native supported scalar `ILIKE` and JSON pattern predicates can
use typed/collation-aware scope evaluation when the required columns and
operator shape are eligible. Partial scope coverage is not proof that all
predicates have been resolved.

The JSON SQL probe has a 65,536-match limit plus an overflow witness. This
limits collected matches, not heap rows examined or elapsed time. A complete
SQL resolver still costs work proportional to the matching universe and its
plan. The global-prefix alternative is a bounded probe, not a resumable exact
rank iterator or a guarantee that all qualifying top-k rows are present.

Current-snapshot predicate membership and ranking completeness are separate
contracts. A filtered release test must check both membership and full top-k
quality against its oracle; valid predicates alone do not demonstrate recall.

## Worker Scheduling

One database-local maintenance service selects actions using catalog state,
coalesced work hints, retry state, debt, and recent activity. Lost hints are
recoverable by reconciliation; they are not durable posting authority.

| Situation | Scheduling behavior |
| --- | --- |
| Hard append or blocked pending-seal pressure | Rotate/seal or relieve structural pressure before optional work |
| Actionable semantic versions | Admit bounded document-encoding batches and append checked completion |
| Small outstanding mutation debt | Coalesce writes and allow a periodic low-debt pass |
| Missing or stale derived baseline | Prepare when eligible, deduplicated by the accelerator lock and retry state |
| Accelerator preparation in flight | Defer conflicting immutable publication unless write safety requires it |
| Fold, residency, or reclaim debt | Select eligible work subject to action-specific admission and reader safety |

With `ii42.runtime_reserve_query_lane = on`, the query runtime reserves capacity
for online requests when multiple local workers are available. This is queue admission, not preemptive CPU/I/O
isolation. The implementation does not enforce a universal wall-time, I/O,
and CPU slice for every loop inside every maintenance action. Long baseline
scans and cache pressure must therefore be measured under concurrent load.

Automatic maintenance and automatic preload depend on the service being
enabled. Setting `ii42.maintenance_worker_limit=0` is not a normal converging
steady state. Conversely, nonzero workers can run housekeeping, preload,
reconciliation, or derived-state work even when there are no new client
writes; a process title alone does not establish runaway rebuilding.

## MVCC, WAL, Recovery, And Replication

WAL ties immutable page contents and linked records to the checked publication
protocol. New objects are unreachable until the root exchange. A failure can
leave orphan physical pages, but must not expose a partially checked closure.
Prepared transactions, aborts, and heap visibility remain PostgreSQL concerns.

Readers pin an accepted read root for their operation. That is not a promise
that a repeatable-read transaction keeps a historical Evoke physical root for
its entire lifetime. Visible tuple versions and current root compatibility
must still satisfy the supported snapshot contract.

### Reclamation And Reader Fences

Retirement and reuse are separate from logical publication. Routine reuse
consumes bounded retired ranges authenticated by a manifest, not a full-root
reachability scan on each seal. Deep orphan discovery is explicit audit work.

```text
old closure no longer selected
    -> authenticated retirement ranges
    -> conditional AccessExclusive reader fence + root revalidation
          | busy: defer / keep append-only safe route
          `-- acquired: protect required reuse and root publication
    -> release fence; complete the path's checked FSM/reclamation handoff
```

Different publication paths use prepared handoffs or directly fenced reuse.
For a directly reused arena, the fence must cover every reused-page write and
the replacement-root publication. It is incorrect to promise that all paths
release the fence before their physical writes. Conditional acquisition avoids
waiting behind existing readers, but a successfully held fence can briefly
delay new readers. Bounded reclamation and profiling remain necessary.

Orphan pages are not automatically safe free space. Trailing cleanup must
prove the range is outside the published closure; interior reuse needs checked
retirement and a safe handoff. Repeated abnormal publication failure can leave
operator-visible space debt and may justify explicit repair or `REINDEX`.

Physical streaming replication replays the relation's checked pages and root
transitions. Standbys do not publish durable maintenance changes; they may
prepare volatile state. The external model checkout and compatible binary/
runtime contract must be provisioned separately on each node.

## Memory Ownership And Warm Identity

| Owner | Contents and limits |
| --- | --- |
| Relation pages | Durable postings, COW state, folds, accelerator/scope objects, retirement evidence |
| PostgreSQL buffers and OS cache | Cached relation pages; residency is not durable authority |
| Runtime workers | Private tokenizer and ONNX sessions, bounded per-worker session cache |
| Shared runtime arena | Request queues, metadata and admitted resident/hot images; no shared ONNX session |
| Query backend | Encoding requests/results, membership keys, candidate or score workspaces and snapshot-local state |
| Maintenance worker | Selected COW/build scratch, scope batches, temporary transpose files and publication state |

The shared service uses runtime ABI version 26. This is separate from ONNX
Runtime's required API 29 / version 1.29.0 contract; the numbers must not be
conflated. Packaging and dependency requirements are documented in
[contributor dependency guide](../CONTRIBUTING.md).

Application backends do not routinely own a full decoded durable index or a
private document model. Nevertheless, scratch is not universally `O(k)`:
allowed TID keys can grow with the matching set, and some score/residual paths
allocate by document-slot count subject to admission. Explicit audit and
builder work also have different memory profiles from ordinary point lookup.

The query-metadata warm marker binds the serving accelerator reference and
baseline sequence when that route is eligible. The exact fallback uses its
checked manifest/root identity. Unrelated L0 movement must not continually
invalidate a still-serving baseline's metadata. Exact resident folds and
narrow hot projections have their own eligibility rules; they must not be
confused with the persistent semantic accelerator.

`ii42.prewarm_max_bytes` is a per-index relation-page warming work budget,
64 MiB by default. It does not cap shared resident-fold admission; that uses
the global arena, priorities, and materialization headroom. Nor does page
warming require copying every relation byte into a resident image. Successful
status or admission alone is insufficient: observe warm query latency, memory,
artifact identity, and rebuild cadence together. Restart/cache clear discards
volatile state but does not remove compatible durable acceleration.

## Migration And Beta Boundaries

- The historical product migration is `psql_bm25s` to a current Evoke index
  built from source data. Experimental intermediate beta formats are not a
  general compatibility target.
- An SQL/API-only deployment with a compatible native root need not rebuild
  data. A model, scoring, or physical-format incompatibility requires its
  explicit migration/rebuild procedure, not optimistic fallback decoding.
- SSR is eventual-only. Pure BM25 retains realtime, eventual, and manual
  policies; manual rebuild behavior is not the automatic incremental contract.
- Partitioned-parent global ranking and row-level security are unsupported.
- Parallel heap build, AM scan, and VACUUM discovery remain future scale work.
- Exact fallback, wide SQL membership resolution, high-DF posting work, and
  cold attachment can still be slow. The design makes no universal latency
  claim from a ready/warm flag.

See [Upgrading](upgrading.md) and
[Semantic index operations](examples/semantic-index-operations.md) before changing
an installed cluster. A source review or a historical rollout ledger does not
establish its currently loaded binary, catalog, model, or root state.

## Evidence And Verification

### Design Rationale And Research Context

The earlier design's research context remains useful. These references explain
mechanisms considered by Evoke, not adoption of another complete engine or
transfer of its benchmark guarantees:

| Reference | Design connection |
| --- | --- |
| [Fast, Incremental Inverted Indexing](https://arxiv.org/abs/1305.0699) | Grouped posting fragments and locality instead of continual whole-index replacement |
| [Efficient Immediate-Access Dynamic Indexing](https://arxiv.org/abs/2211.06030) | Separate immediate searchable ingestion from later read-shape collation |
| [Immediate-Access Indexing for LSR](https://jmmackenzie.io/pdf/rm26-ecir.pdf) | Dynamic learned-sparse postings and the importance of pruning/read optimization |
| [Block-Max Pruning for Learned Sparse Retrieval](https://arxiv.org/abs/2405.01117) | Representation-aware block bounds; Evoke retains its own visibility and storage contract |
| [Geometric Partitioning](https://doi.org/10.1145/1099554.1099739) | Geometrically sized merge work rather than rewriting a large prefix for every small update |
| [Tantivy](https://github.com/quickwit-oss/tantivy) and [Lucene TieredMergePolicy](https://lucene.apache.org/core/10_4_0/core/org/apache/lucene/index/TieredMergePolicy.html) | Immutable publication and selected segment merging, not per-segment result fusion |
| [dLSM](https://arxiv.org/abs/1606.02015) and [HotRAP](https://www.usenix.org/conference/atc25/presentation/qiu) | Stable read surfaces and finer-grained hotness, distinct from volatile cache residency |
| [LSM Compaction Design Space](https://arxiv.org/abs/2202.04522) | Separate trigger, layout, granularity, and movement policy |
| [SILK](https://www.usenix.org/conference/atc19/presentation/balmau) | Foreground latency depends on background scheduling; its preemption model is not an implemented Evoke resource-isolation guarantee |
| [REMIX](https://www.usenix.org/conference/fast21/presentation/zhong) | Logical order across immutable fragments rather than one physical contiguous array |

### Preserved Design Evidence

The pre-convergent baseline tag was
`ii42-pre-convergent-segments-20260730`; the historical architecture milestone
was `csg-beta-arch-1-qualified`. These names locate experiments, not the current
deployment or final Beta 1 qualification.

The fixed-change metadata experiment recorded the cost of recursively
inventorying the entire old closure before COW reuse:

| Vocabulary terms | With recursive inventory, median | Without it, median |
| ---: | ---: | ---: |
| 2,000 | 5.93 ms | 1.29 ms |
| 20,000 | 64.81 ms | 1.51 ms |
| 100,000 | 272.93 ms | 1.93 ms |

This historical microbenchmark supports keeping full-root inventory out of a
small mutation seal. It is not whole-index build throughput or proof that
today's accelerator construction is incremental. The
[development record](archive/engineering/convergent-segmented-index-development-record.md)
also preserves fixed-live-set reuse, exact-fold, lifecycle, and fragmentation
experiments. Its early 1.05x/1.10x/1.15x fragmentation budgets apply to that
measured exact-path matrix, not arbitrary SSR/filter latency.

The archived [query-first report](archive/engineering/query-first-eventual-background-maintenance.md)
preserves the early motivation for foreground/background separation. Its old
worker mechanism and benchmark commands are historical, not installation
guidance. System and model experiments are summarized independently in the
[system report](technical-report-evoke-system.md) and
[model report](technical-report-evoke-model.md).

### Current Verification Obligations

1. Compare exact fragmented, compacted, and folded results with the same
   stored-representation oracle, including ties and retirement statistics.
2. Test bounded SSR routes for current row/predicate membership, expected
   candidate staleness, and full filtered top-k quality. Do not substitute a
   small membership-only test for a representative ranking oracle.
3. Exercise INSERT, non-HOT/HOT UPDATE, DELETE/VACUUM, abort, savepoint, 2PC,
   semantic completion/quarantine, restart, and physical replication.
4. Interleave queries and writes with seals, accelerator preparation,
   publication, preload, and reclamation. Observe root/baseline/warm identities
   and progress, not only final counters.
5. Record query p50/p95/p99, cold and warm behavior, worker RSS, query scratch,
   temporary disk, WAL, page/slot high-water marks, and bytes touched per
   maintenance action on a fixed host and corpus.
6. Verify that an old compatible baseline stays eligible until replacement,
   low debt eventually receives periodic work, and urgent append safety can
   progress without an endless accelerator discard/rebuild loop.

The [maturity suite](../scripts/run_product_maturity_suite.py),
[testing guide](testing-and-validation.md), and [performance guide](performance/README.md)
define executable qualification entry points. A failure of publication safety,
row visibility, ranking quality, or sustained resource behavior blocks release
even if a document describes the intended design correctly.

## Implementation Map

| Concern | Source |
| --- | --- |
| Checked root and manifest codecs | [segments](../src/ii42_segments.c), [types](../src/ii42_segments.h) |
| Relation-page objects and publication | [segment pages](../src/ii42_segment_pages.c) |
| Term and document COW | [term COW](../src/ii42_term_cow.c), [document COW](../src/ii42_document_cow.c) |
| Exact lexical and ordered-prefix lookup | [lexicon COW](../src/ii42_lexicon_cow.c), [prefix COW](../src/ii42_prefix_cow.c) |
| Build, append, and orchestration | [AM](../src/ii42_am.c), [build](../src/ii42_am_build.c), [mutation](../src/ii42_am_mutation.c) |
| Maintenance selection and deduplication | [scheduler](../src/ii42_am_scheduler.c), [maintenance locks](../src/ii42_am_maintenance.c) |
| Accelerator preparation and compatibility | [accelerator](../src/ii42_am_accelerator.c) |
| Predicate routing and heap scope capture | [planner](../src/ii42_planner.c), [filter](../src/ii42_filter.c), [scope](../src/ii42_scope_pg.c) |
| Warm images and reclamation | [preload](../src/ii42_am_preload.c), [resident fold](../src/ii42_am_resident_fold.c), [hot fold](../src/ii42_am_hot_fold.c), [reclamation](../src/ii42_am_reclamation.c) |
| Installed public API | [extension SQL](../sql/ii42--0.2.5.sql) |
