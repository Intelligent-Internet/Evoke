# Shared Runtime And Residency

Evoke keeps durable index authority in PostgreSQL relation pages. Shared memory
owns bounded execution and acceleration state only.

## Ownership

| State | Owner | Durable |
| --- | --- | --- |
| Postings, document versions, root, linked L0 | Index relation | Yes, WAL-logged |
| Relation page cache | PostgreSQL shared buffers / OS | Reconstructible |
| Model sessions and tokenizers | Shared runtime workers | No |
| Encode request/response queues | Postmaster-owned arena | No |
| Exact-root markers, HOT_FOLD, resident fold | Postmaster-owned arena | No |
| Query scratch | Calling backend | No, bounded |

No application backend loads a semantic model or retains an index-sized
posting image.

## BM25 Deployment

BM25-only indexes do not require the shared runtime. Large or unmarked indexes
remain page-native and use PostgreSQL shared buffers. When shared preload is
configured, a fully converged index marked with `auto_preload > 0` can publish
one immutable exact-root resident fold if the complete pointer-free image fits
the shared arena. All backends attach that one image; none retains an
index-sized private copy.

Only when postmaster shared runtime is entirely unavailable may a pure BM25
index whose physical relation size fits `evoke.workspace_cache_bytes` use the
bounded backend-local fallback. Simultaneously active fallback snapshots share
that one per-backend budget; completed queries reuse one idle entry rather than
retaining a snapshot per index. If shared runtime exists but a fold is absent,
stale, loading, or oversized, BM25 uses page-native execution. A
semantic-enabled index prefers its eligible published semantic accelerator and
revalidates a bounded stale-baseline candidate set; page-native execution
remains the exact fallback. Semantic-enabled indexes never use the backend-local
fallback.

Field-aware lexical retrieval follows the same rule. Field-scoped tokens and
weights are flattened into one native accumulation, so selecting or weighting
fields does not create a second cache or a per-field backend snapshot.

The BM25-only `weight_mask` compatibility argument is intentionally outside the
ordinary dispatcher: the mask itself spans every document slot. Evoke admits
that explicit operation only when the physical index fits the finite positive
`evoke.workspace_cache_bytes` budget together with any other active fallback
snapshot; otherwise it fails instead of silently materializing unbounded
backend state. Semantic mode rejects `weight_mask`.

Shared preload remains useful for proactive page warming, but its absence does
not change BM25 rows or scores.

## Semantic Deployment

Semantic-enabled indexes are eventual-only and require:

```conf
shared_preload_libraries = 'evoke'
evoke.shared_runtime_size = '64MB'
evoke.runtime_worker_count = 2
```

Release packages include the digest-locked milestone checkout. A per-index
`model_path` or server-wide `evoke.sae_model_path` selects a different qualified
checkout. Source installs must provision a checkout separately.

Workers use the standard `postgres` database as their connection anchor by
default. Set `evoke.control_database` only when `postgres` is unavailable or a
different stable, connectable database is preferred. Size the arena and worker
pool from measured model/session RSS, queue pressure, and optional residency.
The common resident-fold format supports BM25 and semantic-enabled roots; it
contains already-derived postings and never owns a model session. An eligible
semantic accelerator takes precedence over that exact fold because it is the
index's selected bounded-approximate execution profile.
Every runtime worker can retain its own bounded session cache.
By default, `evoke.runtime_reserve_query_lane = on` keeps one worker/response
lane available for foreground query execution. Controlled offline rebuilds may
turn it off to let document encoding fill the configured worker pool.
`evoke.runtime_accelerators` accepts a JSON array of optional remote Evoke
runtime services. An empty array uses only the local runtime. Non-empty arrays
declare additive services for document build and maintenance batches: the
existing async build pipeline sends compatible batches to the currently best
available lane. Configured services must prove that they hold the same checkout
signature and runtime precision before returning atoms.

Remote accelerators are throughput capacity, not a durable index or foreground
query dependency. After bulk builds and outstanding semantic maintenance have
converged, an operator may reload `evoke.runtime_accelerators = '[]'` and stop
the remote services. Published accelerator objects and resident folds remain
owned by PostgreSQL; later writes are completed by the bounded local runtime
workers unless compatible remote services are configured again. Remove remote
services only after checking that no build or maintenance batch is still using
them.

Service objects accept optional positive integer `weight` and `max_batch_size`
fields. Weighting is a parallel slot-capacity hint, not a speed priority or a
fixed dispatch ratio. `weight=4` is a starting window for a four-worker service,
not a universal throughput optimum; network latency and batch size affect the
useful outstanding window. Without a `weight`, a remote starts as one schedulable
slot. `max_batch_size` is the service's per-request batch cap; without it the
service uses `evoke.runtime_max_batch_size`. The builder forms variable-size
batches so a smaller service remains eligible when larger lanes are busy. The
local runtime contributes its available document worker slots, subject to the
query lane reservation. All lanes share one
scheduler: each PostgreSQL backend reads the shared runtime control metrics and
estimates the next batch from recent EWMA milliseconds per text, current
in-flight requests, slot capacity, service batch cap, and failure backoff.
Faster lanes naturally receive more work after they prove it through successful
batches.

Build-time batch completion is not FIFO-bound. Runtime batches carry an
internal sequence; completed batches can retire from their runtime handle out of
order and release scheduler capacity, while the builder writes semantic rows to
the index only after all earlier sequences are available. This avoids retaining
completed runtime handles behind a slow request, but does not remove ordered
publication: an earlier slow batch can still cause backpressure when the
bounded reorder buffer fills.

The remote list is failure-tolerant by design. An unreachable service, a
service restarting while a request is in flight, or a service returning an
error does not make the index build fail; the dispatcher uses another
configured service when possible and otherwise falls back to local runtime
workers for that batch. Failed endpoints are not permanently removed. The
shared control plane records a short exponential backoff and later document
batches probe the URL again, allowing recovered services to rejoin
automatically. The local document runtime remains an equal candidate pool
member. Explicit weights are only needed when an operator wants to cap or
pre-size a service differently from the automatic exploration window. Actual
routing can shift when a service proves faster, becomes slow, gets busy,
restarts, or returns errors. The selected local runtime provider also covers
the case where no configured remote accepts a batch.

HTTP 503 from a runtime service is treated as backpressure, not as a service
fault. The dispatcher records a short cooling interval and immediately tries
another eligible remote or the local lane for that batch. Other non-200
responses and transport failures use exponential backoff.

`evoke.runtime_document_pipeline_depth` controls builder-level in-flight
document batches. Remote accelerator requests can use that full window because
they do not occupy local runtime response slots. Local runtime submissions stay
bounded by local document worker slots, with response slots still acting as a
hard safety bound. Increasing the builder pipeline for multi-GPU rebuilds
therefore does not let one backend queue a long tail of slow local CPU batches
ahead of faster remote results.

If preload, arena, model checkout, provider, or runtime contract is invalid,
`evoke_index_status(...)` reports a blocker and `evoke_query(...)` fails closed.
The product never falls back to backend-local inference or BM25-only scoring.

The packaged P2 checkout does not pin a runtime provider. CPU export
validation can be recorded in `semantic_runtime.json`, but it is reference
provenance rather than a deployment policy. The index `runtime_precision`
reloption controls the build/query numeric contract, while provider selection
controls placement. The default is `fp16`; it is used for both document
encoding during build/maintenance and query encoding after the index is
query-ready. Automatic provider selection always prefers acceleration and
falls back to CPU only when no usable GPU execution provider is available. The
`fp16` order is TensorRT, CoreML, CUDA, then CPU. The `fp32` order is CUDA,
TensorRT, CoreML, then CPU.
TensorRT receives `trt_fp16_enable=1` for `fp16`; CoreML receives
`AllowLowPrecisionAccumulationOnGPU=1` for `fp16` and `0` for `fp32`. CoreML may
still leave unsupported or shape-related nodes on ORT's CPU execution path, so
it is an accelerated product path rather than proof that the whole graph ran on
CoreML. CPU execution is a compatibility fallback for the selected precision
contract; deployments that care about provider-to-provider numeric drift must
run the provider matrix and representative build/parity gates on each target
host. The model checkout does not carry provider, batch-size, or provider-option
settings; runtime provider policy is always `auto`, and batching is configured
by the PostgreSQL runtime or the external runtime server.

On macOS, CoreML provider setup disables Apple unified activity logging inside
runtime workers before the first CoreML session is created. CoreML compilation
can otherwise crash a forked PostgreSQL background worker inside
CoreAnalytics/os_log before ONNX Runtime can return an error. CoreML defaults
to `MLComputeUnits=CPUAndGPU` with `RequireStaticInputShapes=1` through ONNX
Runtime provider options. These are runtime placement defaults, not model
checkout fields.

The external runtime daemon is `build/evoke-runtime-server`, built with
`make runtime-server`. It exposes `/health`, `/v1/models`, and `/v1/encode`,
and dynamically combines compatible document encode requests up to the service
batch limit across one or more dispatcher lanes. It runs in direct mode: one
model checkout, no PostgreSQL postmaster, no relation storage, no extension SQL
installation, and no `--dsn`. Because it has no PostgreSQL dependency, the
coordinator must pass the checkout signature for the exact model checkout at
startup with `--checkout-signature` or `EVOKE_CHECKOUT_SIGNATURE`.
Keep-alive sockets are bounded by `--max-connections` and
`--connection-idle-timeout-s` (30 seconds by default), so disconnected or idle
coordinators cannot grow the daemon's thread and file-descriptor usage without
limit.

The direct service currently supports document encoding for build and
maintenance acceleration. The PostgreSQL dispatcher consumes
`evoke.runtime_accelerators` for document build and maintenance batches only;
foreground query encoding remains local so query latency and relation-owned
contract checks stay bounded.

## Query Execution

A semantic query:

1. submits one bounded text request to the shared runtime;
2. receives query atoms and the runtime-contract signature;
3. pins the index relation, snapshot, checked root, and linked-L0 frontier;
4. verifies the root uses the same contract signature;
5. uses the published semantic accelerator as an immutable baseline and, when
   stale, revalidates a bounded overfetched candidate set without opening
   post-baseline postings; when the accelerator is ineligible, an eligible
   unfiltered query may lease the exact-root resident fold and otherwise uses
   page-native scoring;
6. releases backend query scratch at completion or according to the configured
   bounded workspace policy.

The shared fold is admitted only for a fully converged root that has no
preferred semantic accelerator. Any active or pending L0, fixed historical
root, prefix query, missing fold, or allocation pressure excludes that fold.
Predicate/TID filters first resolve to the checked root's slot bitmap, then use
the semantic forward accelerator or page-native route. The resident fold is an
exact unfiltered fallback, not a semantic query authority or correctness
dependency.

Document encoding uses bounded batches in build/rebuild or semantic
maintenance workers. Application queries remain single-text requests and have
an admission lane under document-work saturation.

## Runtime Workers

- Each worker executes one ONNX request at a time.
- Workers run in parallel and own independent bounded model-session LRUs.
- Query requests are not queued behind an unbounded document backlog.
- Cancellation requests cooperative ONNX termination.
- The liveness timeout recycles only a worker whose provider does not return.
- A worker crash fails its owned request; other workers continue.
- Model identity includes the digest-validated checkout signature, so replacing
  artifacts at the same path cannot reuse an incompatible session.

`evoke.onnxruntime_intra_op_threads = 0` uses the product auto policy. With
multiple workers, it limits nested ONNX CPU oversubscription. Explicit positive
values cap each worker.

## Warmup, Resident Fold, And HOT_FOLD

`auto_preload > 0` marks an index for best-effort proactive warmup. The
privileged diagnostic can warm one index explicitly:

```sql
SELECT evoke_index_preload('docs_search_idx'::regclass);
```

`64MB` is a minimal runtime example, not a residency ceiling. The postmaster
GUC accepts GB-scale arenas. A deployment can use `1GB`, `8GB`, or more when
selected resident indexes and host RAM justify it. Shared memory is reserved at
postmaster start, so size from measured resident-fold bytes plus runtime state
and leave explicit PostgreSQL and operating-system headroom. Evoke does not
silently admit every small index: `auto_preload > 0` is the operator's durable
selection and priority policy; an explicit preload is the one-shot equivalent.
When all selected folds do not fit, a relation may evict only a lower-priority
relation. Lower-priority indexes fall back to their compact page-native warm
artifacts instead of displacing preferred folds. Once that fallback publishes
the checked-root warm marker, the worker does not repeatedly materialize the
same rejected fold during the current postmaster lifetime; a new root or
restart makes it eligible for one fresh admission attempt.

On page-native v3 the function validates the checked root and publishes the
compact TID-to-slot projection first. Its warmup and serving decisions are:

Document lengths and the compact TID lookup use stable relation keys and carry
their serving authority in the shared payload. For an accelerator route they
cover the accelerator baseline and survive compatible manifest successors. For
an exact route they cover the exact manifest and root. Stale TID matches are
checked against the current document COW record, so updates and reused slots
cannot inherit stale membership.

The query-metadata warm marker is instead bound to the authority that actually
serves the query. An accelerator marker records its directory object and
baseline sequence, while a non-accelerator marker records the exact manifest
and root ID. Consequently, a compatible `ready_baseline_delta` successor keeps
`query_metadata_warm=true`. An actual accelerator replacement or exact-root
change invalidates the marker and requires a new bounded warmup.

1. when a semantic accelerator is preferred, it publishes the compact
   document-length projection, warms query-critical metadata, records a
   serving-authority warm marker, and does not attempt an exact resident fold;
2. otherwise, when linked L0 is empty and the complete pointer-free scoring
   image fits the priority-aware `evoke.shared_runtime_size` arena and host
   materialization headroom, it publishes that exact fold into the common
   shared arena;
3. when no fold is published, it prepares the document-length projection,
   validates the complete COW closure, and warms exact reachable pages when
   they fit the `evoke.prewarm_max_bytes` page-I/O budget;
4. for a larger index, it validates the sealed manifest and query contract,
   then warms query-critical roots, scope-pruning metadata, one page from every
   semantic forward chunk when the budget permits, and a bounded payload
   prefix without materializing or traversing the complete COW closure;
5. durable same-root scope postings serve eligible scalar
   `eq`/`in`/`range`/`ilike` and array `overlap` predicates on declared
   `INCLUDE` columns; compatible older scope baselines remain usable with
   current-row recheck while delta work converges, and these are relation
   pages, not shared residency;
6. optional term-local HOT_FOLD residency remains independent.

The resident fold is a disposable, pointer-free projection of the exact
relation root. It is not durable authority, does not scan source rows, does not
run document inference, and does not publish mutations. Clearing or evicting
it changes latency only.

Publication writes every readable region directly from the validated root and
then validates the fold's header, region bounds, term-to-extent continuity,
extent descriptors, and vocabulary ordering. It does not clear or hash the
entire shared allocation: relation pages and their checksummed closure remain
the authority, while scanning a multi-gigabyte disposable projection would add
restart latency without adding an attach-time integrity guarantee.

The TID-to-slot directory is a compact child of the durable semantic
accelerator root. It is generated from TIDs already collected by the existing
publication pass, so it adds no source-table scan or semantic inference. That
child is complete only for its accelerator baseline. A query may attach it
directly while that compatible baseline serves. After a seal advances the
manifest, a filtered query checks every baseline match against the current COW
TID and predicate scope. It therefore cannot admit an updated, deleted, or
reused slot, although a new post-baseline document may remain absent until
refresh. An exact page-native route instead uses a complete exact-root
projection. A resident fold carries an equivalent exact projection. Shared
projections are retired only when their embedded serving authority changes;
they cannot evict each other, and resident-fold admission fails rather than
removing the reverse directory.

Bounded preload is a startup latency operation, not a replacement for the
complete health and maintenance diagnostics. COW readers validate each object
they touch, so corruption still fails closed on the query path.

`evoke_index_preload(...)` reports `prewarm_scope=exact|bounded`, resident bytes,
the page budget, and the pages touched. The default 64 MB page-warm budget prevents an index
larger than `shared_buffers` from cycling the entire PostgreSQL cache during
startup. Normal query heat and optional HOT_FOLD maintenance continue to
converge hot pages after this bounded first pass.

## Root Changes

Root identity includes relation identity, root identity, storage version, and
semantic contract where applicable. Publication invalidates stale markers.
Existing readers keep their pinned root; new readers validate the current root.
An in-flight shared projection whose root is retired before publication is
discarded rather than made ready under the obsolete reservation.

A mutation makes the prior exact resident fold ineligible immediately, but it
does not invalidate an authority-compatible semantic accelerator. Queries keep
using that immutable accelerator baseline and reject superseded baseline rows.
Stale-baseline execution is one accelerator pass with query-relative overfetch
capped at 4,096 extra candidates; it does not open post-baseline postings. New
or improved delta documents may be absent until the next refresh.

Active L0 does not prevent the worker from refreshing an older sealed
accelerator. The replacement is derived from the pinned immutable manifest and
published while preserving the latest active frontier. Compatible stale
baselines have no serving-age deadline. They refresh immediately after 65,536
sealed records or 512 MiB of sealed payload debt, or as periodic low-priority
work for smaller coalesced debt. During a build, other immutable-root work
defers, but appends, semantic completion, and safe active-L0 rotation continue.
If both frontiers fill, urgent sealing wins and the build retries. Workspace or
reader-fence deferral likewise cannot block mutation convergence.
Auto-preload continues warming the readable old baseline while active ingress
exists; only a clean root whose replacement can publish immediately is handed
to maintenance before warmup. Cold queries may pay page I/O, but no query
backend constructs a shared projection or synchronously rebuilds a missing
replacement. This does not eliminate short publication/reuse fences or shared
CPU, memory-bandwidth, and I/O contention with background work.
This remains one root lifecycle with reconstructible query projections, not a
second mutation protocol.

Restart and standby replay reconstruct all warm state from relation pages.
Standbys may warm replayed pages but perform no primary-side durable
maintenance while in recovery.

Every node serving semantic queries needs a compatible Evoke package, the
pinned ONNX Runtime contract, and the same effective immutable model checkout.
Verify installed binaries for each platform. Node-local provider and capacity
settings may differ with hardware; the model/format contract must not.
Physical replicas receive relation WAL, not server-local model files.

## Diagnostics

Privileged runtime/residency diagnostics include:

```sql
SELECT evoke_index_runtime_state('docs_search_idx'::regclass);
SELECT evoke_index_runtime_state_json('docs_search_idx'::regclass);
SELECT evoke_runtime_service_status();
SELECT evoke_runtime_cache_clear();
```

Interpret the runtime-state fields as root health, linked-L0 debt, runtime
capacity, worker state, and disposable residency. They do not describe a
second storage or lifecycle authority. The JSON function is emitted directly
from the same C snapshot as the text form; it does not reparse diagnostic text.

`evoke_runtime_cache_clear()` clears bounded backend workspace, root
markers, optional projections, and volatile runtime registry state. It does not
modify relation pages, the checked root, pending semantic work, or query
results.

## Memory Acceptance

Production qualification should measure proportional/shared memory rather than
summing process RSS:

- application backend private anonymous memory remains bounded as corpus size
  and connection count grow;
- model/session RSS belongs only to the configured runtime workers;
- relation growth appears in relation pages, PostgreSQL shared buffers, and OS
  page cache rather than copied backend heaps;
- cache clear, eviction, restart, and cold fallback preserve exact results;
- BM25-only behavior remains valid when the semantic runtime is disabled.

See [Connection Memory](connection-memory.md),
[Semantic Runtime](examples/semantic-runtime.md), and
[Testing And Validation](testing-and-validation.md).
