# Index Parameters

Evoke reloptions are set on `CREATE INDEX ... WITH (...)` or `ALTER INDEX`.
Changing an option that affects physical postings requires `REINDEX` before
`ii42_query(...)` becomes query-ready again.

## Index Type

| Option | Default | Contract |
| --- | --- | --- |
| `sae` | `false` | `false` selects exact BM25. `true` selects Sparse Semantic Retrieval (SSR) unified lexical/semantic postings. |
| `consistency` | `realtime` for BM25; `eventual` for SSR | BM25 accepts `realtime`, `eventual`, or `manual`. SSR is eventual-only. |
| `auto_preload` | `0` | Best-effort warmup and shared-residency priority. A positive value first prepares compact query metadata. BM25 or semantic roots without an eligible semantic accelerator may then publish an exact-root resident fold when the converged image fits the shared arena. `0` disables proactive admission; it does not disable maintenance or explicit `ii42_index_preload(...)`. |

Semantic options require explicit `sae = true`. BM25 scoring and tokenizer
options are rejected with `sae = true`; the model checkout owns those
semantics. `field_aware` is a shared index-layout option.

## BM25 Scoring

| Option | Default | Values |
| --- | --- | --- |
| `method` | `lucene` | `robertson`, `lucene`, `atire`, `bm25l`, `bm25+` |
| `idf_method` | `lucene` | Same family names; selects the IDF variant. |
| `k1` | `1.5` | Finite real value greater than or equal to zero. |
| `b` | `0.75` | Finite real value from zero through one. |
| `delta` | `0.5` | Finite non-negative value used by variants that define delta. |
| `create_empty_token` | `true` | Create the synthetic empty token for `int4[]`. |

These options apply only to `sae = false`.

## BM25 Text Processing

| Option | Default | Meaning |
| --- | --- | --- |
| `text_lowercase` | `true` | Lowercase raw `text`/`varchar`. |
| `text_stopwords` | unset | Comma-separated stopword list. |
| `text_stem_english` | `false` | Apply English Porter stemming. |
| `text_fold_diacritics` | `false` | Fold Latin diacritics. |
| `field_aware` | `false` | Preserve lexical column identity in a supported multicolumn text-like BM25 or SSR index. |

`text[]` and `varchar[]` are already tokenized inputs. `int4[]` is an
application-owned integer token stream. See [Supported Input Types](input-types.md).

`field_aware = true` requires more than one homogeneous text-like column. In
SSR mode it expands both lexical and semantic namespaces per field. The query
weight for a field scales both contributions in the one native scorer. Model
normalization and atom generation still come from the checkout.

## Semantic Checkout

| Option | Default | Meaning |
| --- | --- | --- |
| `model_path` | unset; server override, then bundled milestone | Server-local checkout path override. |
| `runtime_precision` | `fp16` | Runtime precision for both index-time document encoding and query-time text encoding. Values: `fp16`, `fp32`. |
| `semantic_impact_precision` | `f32` | Physical precision of semantic impacts in the packed posting authority. Values: `f32`, `fp16`, `u8`. |
| `semantic_alpha_mass` | `1.0` | Per-field fraction of positive semantic impact mass retained after the normal semantic budget. `1.0` is the exact product default; values below `1.0` select an approximate, smaller posting profile. |
| `model` | Manifest value | Optional model-id assertion. |
| `atom_space` | Manifest value | Optional atom-space assertion. |
| `scoring_profile` | Manifest value | Optional scoring-profile assertion. |

The checkout manifest and referenced artifacts are digest-validated. Overrides
must match the checkout; they do not create independent model registrations.
`semantic_alpha_mass` accepts values from `0.01` through `1.0`. Only `1.0`
and `0.50` currently have product qualification: `1.0` preserves the exact
default, while `0.50` is an explicit fast profile that may change retrieval
quality. Other values are expert tuning points and require workload-specific
quality and latency validation.

The alpha profile is part of the immutable index-generation contract. Set it
when creating the index. Changing it with `ALTER INDEX` makes the current root
not query-ready until `REINDEX` publishes a generation built entirely with the
new value. Initial build and eventual semantic completion use the same
per-index value, including field-aware indexes where pruning is applied
independently to each field.

`semantic_impact_precision` controls stored posting impacts, not model
inference. All three values use the same 64-document block and 1,024-document
superblock geometry. `f32` is the exact storage default and `fp16` is the
conservative approximate profile. `u8` is the aggressive positive-impact
profile: it uses one fixed `[0, 2.0]` scale for the P2 authority, reserves zero
for an absent posting, and saturates larger values. The fixed scale makes the
encoding stable across COW publication, compaction, folds, and a full
`REINDEX`. Custom models with signed impacts or a wider output range must use
`fp16` or `f32`.

All three precisions use the same block64 authority, root, worker, CRUD, and
reclamation lifecycle. Block geometry is fixed rather than a reloption;
changing only precision still invalidates the current generation until
`REINDEX` publishes one root entirely in the selected format.

The exact product default remains `f32` with alpha `1.0`. `fp16` is the
conservative intermediate format and `u8` is an explicit storage-oriented
profile. Alpha pruning is orthogonal to precision and is never enabled merely
because an index uses `u8`; deployments must select and qualify alpha per
index.

`runtime_precision` is part of the semantic runtime signature. Changing it
requires `REINDEX`, and the runtime rejects responses that do not report the
same precision as the index contract. It does not force a specific execution
provider: automatic provider selection prefers available GPU acceleration and
falls back to CPU when no usable accelerator is present.

The model checkout does not carry a provider setting. Runtime provider policy
is always `auto`: the runtime chooses the best available accelerator for the
selected precision and falls back to CPU.

Example:

```sql
CREATE INDEX docs_semantic_idx
ON docs USING ii42 (body)
WITH (
    sae = true,
    runtime_precision = fp16,
    model_path = '/opt/evoke/models/search-v1',
    auto_preload = 10
);
```

An explicitly approximate fast profile can be created as follows:

```sql
CREATE INDEX docs_semantic_fast_idx
ON docs USING ii42 (body)
WITH (
    sae = true,
    semantic_impact_precision = 'u8',
    semantic_alpha_mass = 0.50
);
```

## Server Settings

### Backend query workspace

| GUC | Default | Context | Meaning |
| --- | ---: | --- | --- |
| `ii42.workspace_cache_bytes` | `32MB` | User | Retained per-backend query workspace limit and aggregate physical-size admission bound for simultaneously active no-shared-runtime BM25 fallback and explicit BM25 `weight_mask` snapshots. `0` disables retention and both snapshot paths. `-1` removes the ordinary workspace limit but also disables snapshot admission; it is not recommended for production. |
| `ii42.workspace_idle_timeout` | `60s` | User | Lazy idle release interval. `-1` disables idle release. |

This is query-bounded scratch for page-native execution. Shared exact-root
residency is controlled separately by `ii42.shared_runtime_size` and
`auto_preload`, and is shared by BM25 and SSR indexes. The bounded backend-local
BM25 snapshot is admitted only when postmaster shared runtime is unavailable.
When shared runtime exists, both index types use the common resident-fold or
page-native dispatcher.

### Maintenance

| GUC | Default | Context | Meaning |
| --- | ---: | --- | --- |
| `ii42.maintenance_worker_limit` | `1` | SIGHUP | Extension maintenance worker cap. Set `0` to stop admitting new maintenance workers before an upgrade or clean restart; already-running bounded work drains normally. Restore a positive value afterward. |
| `ii42.maintenance_timer_interval_ms` | `60000` | SIGHUP | Structural/completion reconciliation interval and minimum delay after a failed or concurrent semantic-accelerator attempt. Accelerator retry cooldown never delays semantic completion, L0 rotation, sealing, or foreground queries. Effective minimum is one second. |
| `ii42.maintenance_low_debt_interval_ms` | `3600000` | SIGHUP | Minimum age before coalesced L0 or compatible-accelerator debt below its high-water mark becomes low-priority maintenance. High-water work remains immediate; serving artifacts never expire because of this timer. Effective minimum is one second. |
| `ii42.preload_timer_interval_ms` | `1000` | SIGHUP | Independent warmup interval; effective minimum is one second. |
| `ii42.prewarm_max_bytes` | `64MB` | SIGHUP | Per-index relation-page warming work budget. It does not cap exact-root shared residency; resident folds are admitted against the global shared arena, priority policy, and host materialization headroom. |
| `ii42.maintenance_rebuild_memory_budget` | `32768MB` | SIGHUP | Admission budget for rebuild-like worker activity. `0` disables the guard. |
| `ii42.sae_transaction_mutation_max_bytes` | `64MB` | User | Transaction-wide pending semantic document-copy budget; minimum `16MB`. |

Ordinary maintenance performs bounded semantic completion, seal, compaction,
fold, or reclamation. The rebuild memory budget does not turn routine work into
a whole-corpus scan.

### Shared semantic runtime

These settings are meaningful only when `ii42` is loaded through
`shared_preload_libraries`.

| GUC | Default | Context | Meaning |
| --- | ---: | --- | --- |
| `ii42.shared_runtime_size` | `0` | Postmaster | Global shared runtime/residency arena. A positive value is mandatory for SSR; GB-scale values may also hold exact-root folds for selected converged BM25 or SSR indexes. |
| `ii42.control_database` | `postgres` | Postmaster | Optional override for the stable, connectable database used by cluster-level Evoke workers. |
| `ii42.sae_model_path` | empty | SIGHUP | Optional server-wide override for the bundled milestone checkout. |
| `ii42.runtime_worker_count` | `2` | Postmaster | Shared inference worker count. |
| `ii42.runtime_reserve_query_lane` | `on` | Postmaster | Reserve one runtime worker/response lane for foreground queries; turn off only for controlled offline rebuilds. |
| `ii42.runtime_max_batch_size` | `128` | SIGHUP | Maximum local runtime text batch size, from `1` through `512`. This is also the default remote batch cap for services that do not set their own `max_batch_size`; it is a deployment throughput knob, not part of the model checkout identity. |
| `ii42.runtime_document_pipeline_depth` | `16` | SIGHUP | Maximum builder-level in-flight document runtime batches per PostgreSQL backend, from `1` through `4096`. The effective depth is capped by the sum of the local runtime window and all configured remote accelerator `weight` windows. Raise it for controlled offline rebuilds when local and remote accelerators have enough capacity. |
| `ii42.runtime_accelerators` | `[]` | SIGHUP | JSON array of optional remote Evoke runtime services. Empty means local runtime only. Service objects accept `url`, optional positive integer `weight`, and optional positive integer `max_batch_size`. `weight` is a per-service outstanding request window hint, from `1` through `4096`; a service without `weight` starts as one schedulable slot. `max_batch_size` is that service's per-request batch cap. Without `max_batch_size`, the service uses `ii42.runtime_max_batch_size`. |
| `ii42.onnxruntime_session_cache_size` | `1` | Postmaster | Per-worker bounded model-session LRU, from `0` through `16`. |
| `ii42.runtime_liveness_timeout` | `5min` | Postmaster | Runtime liveness guard for one local ONNX execution or one already-submitted remote accelerator request; `0` disables the execution timer. Remote connect/send I/O still uses short transport timeouts, and an async request without a usable connection retains a separate 60-second failover bound. |
| `ii42.onnxruntime_intra_op_threads` | `0` | Postmaster | Per-worker ONNX CPU thread cap; `0` uses the product auto policy. |
| `ii42.onnxruntime_document_cpu_mem_arena` | `off` | Postmaster | Retain CPU provider document-mode arenas for controlled offline rebuilds on memory-rich hosts. |

`ii42.shared_runtime_size` is a hard postmaster-wide budget for runtime queues,
exact-root markers, HOT_FOLD data, and optional pointer-free resident folds.
It supports values up to `1TB`; `64MB` is only the minimal semantic smoke-test
example. Set `auto_preload > 0` on indexes that should compete for resident
space. The complete exact fold must fit the arena, and admission/eviction uses
the configured priority and measured fold bytes. Durable postings remain in
relation pages and PostgreSQL shared buffers.

Remote runtime services are configured as a JSON array:

```conf
ii42.runtime_accelerators = '[{"url":"http://127.0.0.1:8042"}]'
```

Use `weight` only to describe how many outstanding requests PostgreSQL may keep
open against a remote service. It is a capacity hint, not a speed priority or a
fixed dispatch ratio. For low-latency, large-batch services this may be close to
the worker count; for high-latency CPU services using tiny batches it often must
be much higher than the worker count to keep the endpoint busy. Without a
`weight`, a remote service starts as one schedulable slot. Use
`max_batch_size` for each service's real request-size limit; the builder forms
variable-size batches so a smaller service is still used when larger services
are busy. The local runtime contributes its available document worker slots,
plus its bounded pending window, subject to the query lane reservation. The
effective builder pipeline is capped by the total lane capacity so saturated
lanes stop receiving new batches until completions free slots. All lanes share
one scheduler: each PostgreSQL backend reads the shared runtime control metrics
and estimates the next batch from recent EWMA milliseconds per text, current
in-flight requests, slot capacity, service batch cap, and failure backoff. The
builder retires completed runtime batches out of order into a bounded reorder
buffer, then applies them to the index in original sequence order. This keeps a
slow remote service from blocking unrelated fast completions while preserving
document order in `doc_pairs` and `doc_starts`.

The following configuration lets each Spark service accept up to four
outstanding requests and a 32-document request cap, while actual routing still
shifts to whichever lane has recently finished compatible batches fastest:

```conf
ii42.runtime_accelerators = '[{"url":"http://runtime-1.example.internal:18042","weight":4,"max_batch_size":32},{"url":"http://runtime-2.example.internal:18042","weight":4,"max_batch_size":32}]'
```

`build/ii42-runtime-server` is the direct external runtime service. It loads
one model checkout, validates artifact hashes, selects the ONNX Runtime
provider locally, and serves document `/v1/encode` without starting
PostgreSQL. Pass the coordinator database checkout signature explicitly:

```sh
make runtime-server
build/ii42-runtime-server \
    --model-path /path/to/checkout \
    --checkout-signature "$CHECKOUT_SIGNATURE" \
    --runtime-precision fp16 \
    --host 127.0.0.1 \
    --port 8042 \
    --max-batch-size 128 \
    --worker-count 4 \
    --max-connections 128 \
    --connection-idle-timeout-s 30
```

The server bounds concurrent keep-alive connections independently from its
batch queue and closes idle sockets after the configured timeout. If omitted,
the connection limit is derived from worker and queue capacity with a minimum
of 64; the idle timeout defaults to 30 seconds.
Coordinator backends retain at most 32 reusable remote sockets each and expire
them after 30 seconds, independently of the daemon-side cap.

The direct service is intentionally document-mode only. Foreground query
encoding remains local to the PostgreSQL runtime because relation-owned query
contracts and latency admission stay in the database backend.

Each service must reject requests whose `checkout_signature` does not match
the model it has loaded, or whose `runtime_precision` does not match the
service precision. The service list is an additive acceleration contract for
document build and maintenance batches: remote services must not replace
relation-owned postings or model-contract authority. Foreground query encoding
continues to use the local shared runtime. Remote services are allowed to be
temporarily unreachable: connection failures, restarts, and non-200 responses
make the current batch try another configured service or the local lane.
Failed services are only backed off for a short shared interval; later document
batches probe them again, so a recovered service automatically rejoins the
acceleration pool without rebuilding the index or changing configuration. When
remote services are configured, the local document runtime remains an equal
candidate pool member and also covers the case where all remotes are
unavailable or currently slower than local execution.

Single large-index builds need enough in-flight document batches to keep
multiple accelerators busy. Increase `ii42.runtime_document_pipeline_depth`
for offline multi-GPU rebuilds after verifying shared memory and service
capacity. This is a builder pipeline window, not a local response-slot count:
remote accelerator handles can exceed local response capacity, while local
fallback remains capped by local document worker slots to avoid building a
slow local queue ahead of faster remote results. Local response capacity still
acts as a hard safety bound.

Minimal semantic deployment:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
ii42.runtime_worker_count = 2
```

For a host intended to keep selected small or medium indexes shared-resident,
use a measured GB-scale arena instead, for example `1GB` or `8GB`, and leave
safety headroom above the sum of `resident_fold_bytes` and runtime queues.

The standard `postgres` database is used automatically. Override
`ii42.control_database` only when it is unavailable or another stable database
is preferred.

Release packages install their digest-locked milestone checkout under the
target PostgreSQL shared-data directory. Resolution order is per-index
`model_path`, `ii42.sae_model_path` (or `II42_SAE_MODEL_PATH` at process
startup), then the package checkout. Source builds must provide one of these
paths before creating an SSR index.

Capacity-test worker count and session cache against the selected model. Each
runtime worker can own its own model session; application backends cannot.

## Validation

```sql
SELECT ii42_index_options('docs_semantic_idx'::regclass);
SELECT ii42_index_status('docs_semantic_idx'::regclass);
DROP INDEX docs_semantic_idx;
```

Use `ii42_index_status(...)` after reloption, checkout, binary, or runtime
changes. A contract mismatch fails page-native search closed until corrected
and, where required, rebuilt.
