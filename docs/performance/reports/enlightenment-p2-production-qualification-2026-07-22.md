# Enlightenment P2 Production Qualification

Qualification started: 2026-07-22

Final-v3 closure: 2026-07-23

Historical status (superseded): P2.2 final-v3 was deployed on Enlightenment
with a versioned 0.1.2 to 0.2.0 extension upgrade, package binding,
rollback/reinstall, mutable lifecycle, full-text handling, exact query
correctness, and four-dataset native held-out relevance. The current product
does not ship or support that intermediate catalog, ABI, payload, or binary
rollback route. This report remains only as dated qualification evidence.

The original P2.1 package installed on Enlightenment is qualified for static or
read-mostly corpora on the ordinary PostgreSQL product path. That deployed
package passes its lifecycle, restart, corruption, concurrent-DDL, security,
and soak gates, but accumulates semantic generation debt and silently limits
model input to one 512-token window.

P2.2 closes those two product gaps in the installed candidate package:
semantic writes use one relation-owned unified delta, and the ABI-v2 compiler
processes full text through deterministic windows. The existing arXiv 75k
index remained on a pre-unified intermediate semantic generation during this
historical qualification, while an isolated current unified generation on the
same host passed the complete mutable lifecycle and full-text suite. Current
source no longer reads that intermediate semantic format; such an index must
be rebuilt before installing the current runtime. The final-v3 exact scorer
also closes the 50 ms-class target without changing candidates, ranked
identities, or scores; shared-host p95 remains workload-sensitive.

This report separates three claims that must not be conflated:

1. **Index correctness and lifecycle:** passed.
2. **Operational performance on Enlightenment:** measured; usable for
   read-mostly workloads, but not competitive with the existing engines.
3. **Cross-domain retrieval quality:** positive against BM25 on four full
   native qrels surfaces, while still limited to one NFCorpus-calibrated model
   and not a dense-equivalence claim.

## Product Path Under Test

The qualification uses only the public, converged PostgreSQL path:

```sql
CREATE INDEX ... USING ii42(column) WITH (sae = false);
CREATE INDEX ... USING ii42(column) WITH (sae = true);

SELECT * FROM ii42_query(index_name, query_text, result_limit);
SELECT ii42_index_status(index_name);
SELECT ii42_index_maintain(index_name);
```

`sae = false` and `sae = true` share the same API, relation-owned generation,
mutation tracking, maintenance, restart, drop, and recovery machinery. The P2
index stores lexical and semantic impacts in one unified posting payload; this
is not an external BM25 plus ANN fusion path.

## Installation And Provenance

| Item | Qualified value |
| --- | --- |
| Host | Enlightenment, Debian Linux |
| CPU | 2 x Xeon E5-2630 v2, 24 logical CPUs, no AVX2 |
| Memory | 377 GiB |
| PostgreSQL | 18.4, system cluster `/var/lib/postgresql/18/main` |
| Database | `ii_dev`, 1,281 GiB |
| Extension | `ii42` 0.2.0 in schema `ii42_ext` |
| Preloads | `vchord,psql_bm25s,pg_prewarm,ii42` |
| Runtime provider | ONNX Runtime CPU |
| P2 model | `evoke_p2_m1934_b1125_nfcorpus_full_b2` |
| Baseline package | `evoke-v0.1.2-linux-x86_64-pg18.zip` |
| Baseline package SHA-256 | `47eb3219e0eb4583768d184043379fec25ceb11012193e649dd6b8db6ff86762` |
| P2.2 final package | `0.2.0-20260723-p22-final-v3` |
| Final package ZIP SHA-256 | `39373a3c8920a64656322f5fee56ab1a786da7d337e45d16c39d4dc52b4c478e` |
| Installed P2.2 `ii42.so` SHA-256 | `14f7d948da1de927bef49be4a216d312c61716b4ade2df650222c76b398fdee2` |
| 0.2.0 install SQL SHA-256 | `da5fe33ee24dc6aa69271d0605dfc8b853c9f36ff7612a386a1cfb7fee78e7fe` |
| 0.1.2 to 0.2.0 upgrade SQL SHA-256 | `a1cf020ceab844b2f435c1608b00beafec1d367702445424c91d9cf778b44db9` |
| Package fingerprint | `3b017fa5af5e64a85e0592d407adb2996d34f8261c11343c4c83c8e7f9a2974d` |
| Bundled ONNX Runtime SHA-256 | `5bd5bedf736fc501692435d0ec4f6e8b2bdf48cd30af8e6d00d61b3ddc9a7ab8` |

The final-v1, final-v2 rollback, and qualified P2.2 final-v3 packages are
archived at:

```text
/var/lib/postgresql/18/ii42/packages/0.1.2-20260722-final/
/var/lib/postgresql/18/ii42/packages/0.1.2-20260722-p22-final-v1/
/var/lib/postgresql/18/ii42/packages/0.1.2-20260723-p22-final-v2/
/var/lib/postgresql/18/ii42/packages/0.2.0-20260723-p22-final-v3/
```

The model, package, and exact pre-install backup are confined to
`/var/lib/postgresql/18/ii42`. Runtime libraries are in PostgreSQL's system
extension directories, as required for the system instance. No files were
installed into the application data directories or the `commons` schema.
ONNX Runtime 1.26 is bundled with `$ORIGIN` RUNPATH.

The source checkout was intentionally dirty during product convergence. The
embedded commit, `2b8e6203227ee651f3cacafe118c0bd5aba18c06`, identifies the
base revision but is not a clean-release provenance claim.

## Real Corpus Surface

| Corpus | Full source rows | Qualification surface | Vector coverage in source |
| --- | ---: | ---: | ---: |
| arXiv | 3,020,703 | 75,001 rows | approximately 91% |
| PubMed | 7,914,653 | 75,000 BM25 rows; 1,987 P2 rows | approximately 80% |

The arXiv and PubMed tables in `commons` remained read-only. Qualification
tables and indexes are under `ii42_qual_20260722`. The 75k surfaces are
contiguous operational samples, not random or official evaluation splits.
The PubMed P2 surface combines 1,500 deterministic sampled rows with 500 long
documents to expose cross-domain and truncation behavior.

The P2 namespace contains 79,787 dimensions: 29,522 lexical dimensions and
50,265 Granite sparse dimensions. Documents retain at most 192 semantic
dimensions and queries retain 50. The deployed P2.1 checkout truncates input at
512 tokens; the P2.2 ABI-v2 candidate replaces that behavior with deterministic
overlapping windows. The checkout's calibration data contains only 3,633
NFCorpus documents. These facts limit any quality claim on arXiv or PubMed.

## Closed Product Gaps

The following gaps were fixed and verified on the installed package:

- one public API for BM25 and model-backed indexes;
- one atomic relation-owned generation for identity and unified postings;
- batch model encoding only during build/rebuild; query encoding remains an
  immediate single-text request;
- transaction, savepoint, HOT update, delete, rollback, VACUUM, REINDEX,
  failed-rebuild rollback, cold restart, and crash restart behavior;
- build-option and model-signature drift fail closed until atomic REINDEX;
- corrupted model artifacts fail closed without publishing a partial
  generation;
- concurrent REINDEX and foreground writes no longer enter a relation-lock
  cycle;
- partitioned parents fail closed instead of pretending to own a globally
  ranked generation;
- RLS-enabled sources fail closed instead of leaking hidden CTIDs before an
  outer policy join;
- model signatures avoid rehashing every artifact on each connection. At the
  time of this qualification, explicit status still performed a deep artifact
  audit; current releases use bounded `ii42_index_status(...)` readiness and
  reserve artifact hashing for explicit `ii42_index_audit(...)`;
- runtime service ownership, privilege boundaries, non-`public` install, and
  non-relocatable extension semantics;
- versioned 0.1.2 to 0.2.0 catalog upgrade with preserved relation identities
  and fresh-install catalog parity;
- bounded runtime queue and configurable ONNX intra-op thread count;
- no retired `ii42_model_*`, `ii42_sae_*`, publisher, or split corpus-
  generation product route.

## Lifecycle And Recovery

The original P2.1 package and production checkout pass all **42/42** lifecycle
gates. The raw result is
[lifecycle.json](../data/raw/enlightenment-p2-qualification-2026-07-22/lifecycle.json).

The installed P2.2 package was then exercised with the ABI-v2 smoke checkout
in an isolated PG18 cluster on Enlightenment. It passed **48/48** gates,
including model and artifact drift rejection, unified CRUD/compaction/REINDEX
parity, 20 stable queries, five mixed CRUD cycles, deterministic full-text
windows, cold restart, and immediate crash recovery. The raw result is
[p22-contract-v2-v2-lifecycle-soak.json](../data/raw/enlightenment-p2-qualification-2026-07-22/p22-contract-v2-v2-lifecycle-soak.json).

The exact final-v2 source then passed a second **46/46** mutable-lifecycle
gate before packaging. The final Linux package passed all **17** package-bound
maturity steps in 1,951.3 seconds, including a separate **48/48** production
model lifecycle, the core and shared-resident regressions, replication,
concurrent DDL, corruption, restart/crash, two 1,000-iteration memory soaks,
and a 50,000-document mutable lifecycle. Package fingerprints were identical
before and after the suite.

Final-v3 replaces the previously stale live 0.1.2 catalog with a real
versioned extension transition. An isolated temporary-PG test proved that a
fresh 0.2.0 install and a 0.1.2 to 0.2.0 upgrade expose identical function
catalogs. The live upgrade then preserved all 9 existing index OIDs and
relfilenodes. The fixed arXiv comparison preserved all 1,200 ranked identities
with maximum score delta `0.0`; both batch-encoding API families are present
after the upgrade. The first deliberately failed upgrade attempt also rolled
back transactionally without changing the live catalog.

The final-v3 staged package then passed all **18/18** package-bound maturity
steps in 1,849.45 seconds. This includes the new isolated upgrade smoke,
48/48 production-model lifecycle gates, 8/8 concurrent-DDL gates, replication,
corruption/recovery, two 1,000-iteration memory soaks, and the 50,000-document
unified mutable lifecycle. The package fingerprint remained
`3b017fa5...` before and after the suite.

| Operation | Observed time |
| --- | ---: |
| Isolated P2 CREATE INDEX | 1,941 ms |
| Test-cluster start | 420 ms |
| Cold restart | 605 ms |
| Immediate crash restart | 730 ms |

The P2.2 target-host suite measured a 2,791 ms CREATE INDEX, 7,050 ms
full-text audit, 13,770 ms randomized unified-delta differential, 704 ms cold
restart, and 621 ms crash restart. These are isolated smoke timings, not the
75k performance surface.

The suite includes 100 query iterations and 25 mixed CRUD cycles. Automatic
eventual convergence occurred before explicit maintenance and still produced
an atomic generation with no split delta. An isolated final-package
concurrent writer plus `REINDEX INDEX CONCURRENTLY` test passed all 8 gates in
5.65 seconds.

Controlled restarts of the system PostgreSQL service took 111 to 136 seconds.
Most of each interval was existing `psql_bm25s` preload work rather than Evoke
startup. The PostgreSQL service journal has no warning-or-higher entries after
the final restart. In the accepted final-v2 stable window, the Evoke runtime
completed 1,487 of 1,487 requests with zero failures, zero worker recoveries,
and queue depth no higher than 7.

P2.2 also passed three real offline package rollback cycles. The first restored
the pre-install binary (`510190dfa...`) and the contract-v2 candidate
(`4f0d53dea...`), reproducing all 30 fixed arXiv rows exactly. The final cycle
rolled `89172cd...` back to final-v1 (`8bdede8c...`) and reinstalled final-v2.
Both stages preserved all 1,200 fixed ranked identities and scores exactly.
No extension binary was overwritten while a backend was live. This
offline-only rule is part of the deployment contract.

Final-v3 adds a binary-only compatibility rollback while retaining the 0.2.0
catalog. PostgreSQL was stopped, `14f7d948...` was replaced by final-v2
`89172cd...`, and the service was restarted. The 0.2.0 catalog, four batch
APIs, and three checked legacy/ABI-v2 indexes remained valid and query-ready.
The rollback and subsequent final-v3 restore each preserved all 1,200 fixed
arXiv identities with maximum score delta `0.0`. This proves package binary
rollback compatibility; catalog downgrade remains intentionally unsupported.

During the budget canary build the entire Enlightenment host rebooted
abruptly, with no normal shutdown record. PostgreSQL performed WAL recovery
and returned ready. The completed arXiv and SciFact indexes remained
query-ready, valid, and signature-matched; the interrupted transactional
`CREATE INDEX` left no partial relation. The canary was then restarted once
from a clean state. This is host-level crash-recovery evidence, not an
ii42-triggered crash.

The legacy generation also passed an explicit model-drift test. Pointing the
index at a checkout with a changed model identity produced
`runtime_generation_mismatch` and rejected search without changing generation
`487635189/1/1`. Resetting the reloption restored the environment model path,
signature `676a4f4280e5b8347078a9fa6b79d0db`, and query readiness.

## Build Cost And Storage

| Surface | Engine | Rows | Build time | Index bytes |
| --- | --- | ---: | ---: | ---: |
| arXiv | exact BM25 | 75,000 | 18.79 s | 75,628,544 |
| arXiv | VectorChord `rabitq8_l2` | 74,834 vectors | 8.07 s | 36,831,232 |
| arXiv | P2 unified posting | 75,000 | 3:11:21 | 92,872,704 |
| PubMed smoke | exact BM25 | 1,987 | 0.85 s | 3,694,592 |
| PubMed smoke | P2 unified posting | 1,987 | 7:35 | 5,390,336 |

The VectorChord measurement starts from already materialized vectors and does
not include vector encoding. P2 includes text encoding. The comparison is
therefore an operational index-path comparison, not an encoder-throughput
equivalence claim.

On this non-AVX2 host, P2 built arXiv at approximately 6.5 documents/second
and the long-document PubMed surface at approximately 4.4 documents/second.
Direct full-corpus rebuilds would take days. This is the first hard product
blocker for high-churn use.

## Query Performance

The matched arXiv matrix uses 12 deterministic real title/self queries,
`k = 100`, five warm repetitions, and an eight-client concurrency probe. It
has no qrels.

| Engine | Warm p50 | Warm p95 | 8-client p50 | 8-client p95 |
| --- | ---: | ---: | ---: | ---: |
| exact BM25 | 2.64 ms | 19.23 ms | 21.53 ms | 24.32 ms |
| VectorChord | 5.20 ms | 6.79 ms | 26.34 ms | 29.37 ms |
| P2 unified posting | 99.00 ms | 109.43 ms | 217.18 ms | 250.78 ms |

The P2 model runtime averaged about 15.2 ms per query during the matrix. Most
of the remaining P2 latency is native posting traversal and document-term
reranking, not model inference. Across the 12 queries, P2 inspected an average
of 8,684 candidate documents, 11,478 candidate postings, and 1.09 million
rerank document terms. Reducing this fanout is the primary query-performance
task.

### P2.2 exact scorer closure

Earlier P2.2 work deferred heap visibility checks and prefetched the
document-start table, improving the full query by approximately 23%. Final-v2
adds the remaining exact optimization: each candidate document's contiguous
atom/impact slice is copied once into a reusable scratch buffer instead of
performing two page-backed four-byte reads per atom. Accumulation order,
candidate generation, visibility, and ranking are unchanged.

The final gate uses the same 12 title queries, `k = 100`, ten warm repetitions,
and the original unoptimized binary as baseline:

| Stage | p50 | p95 | p50 improvement | p95 improvement |
| --- | ---: | ---: | ---: | ---: |
| Unoptimized `21db255f...` | 95.90 ms | 112.91 ms | - | - |
| final-v2 C1 | 54.40 ms | 68.27 ms | 43.27% | 39.54% |
| final-v2 C2 | 53.22 ms | 74.54 ms | 44.50% | 33.98% |
| Installed final-v2 C3 | 52.31 ms | 68.15 ms | 45.45% | 39.64% |
| Post-rollback/reinstall C4 | 50.79 ms | 58.83 ms | 47.04% | 47.90% |
| Restored final-v3 | 48.29 ms | 60.77 ms | 49.65% | 46.17% |

All accepted final-v2 rounds pass the required 20% p50 and p95 gate. The
12-sample restored final-v3 check also passes numerically.
Across each 1,200-row comparison, ranked identity mismatches are zero and the
maximum score delta is `0.0`.

Two independent 120-sample final-v3 rounds preserved the same exact rankings
and stable 50 ms-class p50 (`53.13` and `51.87 ms`). Their p95 values
(`105.43` and `134.71 ms`) were collected while unrelated VMs, a text
embedding service, and synchronization traffic were active on the shared
host. They are retained as contention evidence and are not substituted for
the accepted final-v2 steady-state p95 gate.

The immediate post-restart observation measured `84.97/159.36 ms` p50/p95
while PostgreSQL and preload workers were still loading resident state. It is
retained as operational evidence, not accepted as a steady-state performance
gate. Once runtime queue depth was zero and sampled PostgreSQL CPU was below
1%, the same installed SHA produced the C4 result above. Correctness remained
exact in both windows.

The final package therefore closes the target-host exact performance gate and
reaches the intended 50 ms p50 class. This does not make query latency equal
to BM25 or VectorChord and does not justify lossy candidate pruning.

A bounded top-k heap was then tested on the same complete 171,331-document
TREC-COVID volume, with the same index, model, 12 fixed queries, `k = 100`,
and five warm repetitions. It preserved all 1,200 ranked identities with zero
score delta, but the warmed p50/p95 regressed from `33.24/43.39 ms` to
`71.83/77.94 ms`. The candidate was rejected and the baseline binary restored.
This is evidence that the positive small resident microbenchmark did not
generalize to full-corpus execution; no heap selector should be promoted
without profiling the complete accumulation and selection path.

The 1,987-row PubMed cross-domain smoke shows the same shape:

| Engine | Warm p50 | Warm p95 | 8-client p50 | 8-client p95 |
| --- | ---: | ---: | ---: | ---: |
| exact BM25 | 3.50 ms | 54.01 ms | 21.26 ms | 69.25 ms |
| P2 unified posting | 45.64 ms | 55.81 ms | 147.19 ms | 223.40 ms |

The high PubMed BM25 p95 reflects a small and noisy smoke surface. It should
not be generalized to the full corpus.

### Single-variable lossy canary

One deliberately isolated cost experiment reduced only
`document_semantic_budget_ratio` from `1.125` to `0.75`. The semantic
query/document ONNX compilers, tokenizer, lexical vocabulary, calibration
arrays, atom space, max-atoms limits, and scoring logic remained byte
identical. Both indexes used the same complete 5,183-document SciFact table,
300 qrels-bearing queries, `k = 1000`, and native `ii42_query`.

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | p50 ms | p95 ms | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact P2.2 baseline | 0.709902 | 0.670223 | 0.952667 | 0.679660 | 0.990000 | 51.749 | 62.845 | 30,294,016 |
| Budget 0.75 | 0.704454 | 0.665971 | 0.947667 | 0.676055 | 0.990000 | 49.405 | 60.748 | 27,328,512 |
| Delta/improvement | -0.005448 | -0.004252 | -0.005000 | -0.003605 | 0.000000 | 4.53% | 3.34% | 9.79% |

The predeclared gate allowed at most `0.001` loss on every quality metric and
required either 20% storage reduction or simultaneous 20% p50/p95 latency
improvement. The candidate failed quality, storage, and latency gates. The
lossy branch therefore stopped without trying additional max-atoms,
quantization, or candidate-cap variables. This is evidence that semantic
budget reduction alone does not provide a safe P2 cost breakthrough.

## Correctness Proxies And Their Limits

Self-retrieval is only an operational proxy. On arXiv, BM25 and P2 returned
the source document within top 100 for 12/12 title queries. VectorChord and
exact vector scan also returned it within top 100 for 12/12 queries, though
only 7/12 were top 1. VectorChord/exact-vector overlap at 100 was 0.9958.

P2/BM25 overlap at 100 was 0.3233; P2/exact-vector overlap was 0.2400. These
numbers demonstrate distinct retrieval behavior but do **not** measure P2
dense equivalence: the identity of the stored 256-dimensional vector model is
not proven equal to the P2 Granite sparse root.

On PubMed, P2 placed 9/12 title queries at rank 1, 11/12 in top 10, and 12/12
in top 100. A free-form `oxidative stress in human cells` query returned
semantically relevant biomedical titles. This confirms that the model path
operates out of domain, not that relevance quality meets a benchmark.

## Long Documents And Full-Text Handling

Native tokenizer measurements at a diagnostic limit of 4,096 tokens show:

| Sample | Mean tokens | p50 | p95 | Maximum | Above 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| arXiv deterministic 256 | 214.8 | 189.5 | 422 | 772 | 2.34% |
| PubMed deterministic 256 | 250.2 | 276 | 573 | 776 | 8.20% |
| arXiv longest 64 | - | 498 | - | 777 | 23/64 |
| PubMed longest 64 | - | 848.5 | - | 2,030 | 64/64 |

These measurements quantify the content discarded by the legacy P2.1
checkout. P2.2 instead uses deterministic overlapping windows, dimension-wise
maximum semantic aggregation, and full-text lexical compilation. On
Enlightenment, two 1,333-token inputs each produced three semantic windows and
50 semantic atoms; single and batch results matched within the established
`1e-4`/`1e-6` tolerance. Inputs beyond the configured six-window safety limit
failed explicitly with SQLSTATE `54000`. Removing truncation is now qualified
as product behavior, but remains separate from relevance quality.

## Memory And Soak

A 300-query arXiv P2 soak completed 300/300 requests with zero failures:

- mean 97.36 ms, p50 97.11 ms, p95 122.01 ms;
- backend anonymous RSS stabilized near 11.2 MiB after warm-up;
- runtime-worker anonymous RSS remained exactly 869,109,760 bytes from the
  first through final sample;
- backend memory contexts grew during initialization and then stabilized near
  6.8 MiB total.

No query-path memory leak or worker recovery was observed. The approximately
88 MiB `memory_bytes` diagnostic is resident decoded index payload, not
per-query workspace.

Testing `ii42.onnxruntime_intra_op_threads` with the real P2 model showed that
the default value `0` was best on this dual-socket old Xeon. Capping it at 2
or 4 threads increased build time and concurrent latency. The control remains
useful for host QoS, but the qualified default stays at zero.

## Historical P2.1 Mutable-Corpus Limitation

The arXiv P2 index was built concurrently over 75,000 rows. A unique online
canary committed during the build is visible immediately through the realtime
BM25 index. P2 published a valid, atomic 75,000-row generation and correctly
records one `pending_writes` item after the table reached 75,001 rows. There is
no hidden split semantic delta and queries remain safe against the published
generation.

This behavior is coherent P2.1 `consistency = eventual` semantics, but it is not
a complete large-corpus update strategy. P2.2 replaces this counter-only debt
with complete lexical-plus-semantic delta records inside the existing relation
delta. Randomized CRUD replay now matches compaction and full `REINDEX` on both
the local package-bound surface and Enlightenment, while primary/standby replay
preserves ranked results. The target-host update, restart, crash, and rollback
gates are closed for the ABI-v2 product path.

Logical schema dump preserves both BM25 and P2 `CREATE INDEX` definitions,
including the model path. Restore correctly rebuilds the indexes, but the
model checkout must be provisioned separately with a verified signature.

## Historical Candidate Verdict

The following table records the candidate evaluated in July 2026. It is not a
support matrix for the current source package. In particular, the current
package deliberately removed the catalog-upgrade and old-binary rollback
routes listed below.

| Capability | Historical verdict |
| --- | --- |
| Unified public API | ready |
| Atomic index publication and rollback | ready |
| Versioned catalog upgrade and binary-only rollback | ready |
| CRUD, VACUUM, REINDEX, cold/crash restart | ready |
| Concurrent DDL and foreground writes | ready |
| Corruption and model-drift fail-closed behavior | ready |
| Deterministic full-text windowing and short-text parity | ready |
| Static/read-mostly arXiv/PubMed operation | qualified within the documented cost envelope |
| Unified mutable lifecycle | ready on isolated and package-bound qualification surfaces |
| Exact arXiv query optimization | final-v2 steady gate passed 47.0%/47.9%; final-v3 retained 1,200/1,200 exact rows and 50 ms-class p50 |
| High-churn large-corpus throughput | unqualified at production scale |
| Query latency parity with BM25/VectorChord | blocked |
| Native held-out relevance quality | four full rows passed; all five P2.2 quality deltas are positive on every row |
| RLS-aware global ranking | unsupported, fails closed |
| Globally ranked partition-parent index | unsupported, fails closed |

That candidate qualified one native unified posting index with an atomic
mutable lifecycle. Large-corpus sustained write throughput was not measured,
so high-churn deployments still required capacity testing. The existing arXiv
index retained one historical P2.1 pending write because it had not been
rebuilt as an ABI-v2 model generation; this is dated evidence, not a current
compatibility promise. P2 did not replace BM25/VectorChord defaults for
latency-sensitive workloads: the exact optimization gate closed, but absolute
build, storage, and query costs remained substantially higher.

## P2.2 Acceptance Audit

| Requirement | Authoritative evidence | Decision |
| --- | --- | --- |
| One index, base, delta, tombstone stream, manifest, and generation | ABI-v2 48/48 lifecycle, final-v3 50k lifecycle, convergence inventory | passed |
| Atomic lexical-plus-semantic mutation publication | Randomized differential, 25 mixed CRUD cycles, compaction, restart, and replication parity | passed |
| Incremental CRUD equivalent to full `REINDEX` | Ranked identities, scores, document counts, deletes, and generation signatures agree | passed |
| Full-text deterministic windows with no silent 512-token truncation | Full-text audit and ABI-v2 lifecycle; explicit input safety bound | passed |
| Short-text output parity | ABI-v2 short-text parity gate | passed |
| Exact optimization before lossy work | Accepted arXiv exact A/B preserved all 1,200 rows and exceeded the 20% p50/p95 gate | passed |
| Native relevance qualification | Four complete held-out qrels rows through `ii42_query`, all with positive five-metric deltas over BM25 | passed |
| Lossy work isolated behind strict Pareto gates | Budget-0.75 changed one semantic variable, failed quality/storage/latency gates, and was rejected | closed by stop rule |
| RLS and partition behavior is globally correct or fail-closed | Unsupported unsafe shapes reject indexing/search rather than approximate policy-visible ranking | passed |
| Historical public API and staged deployment | Unified SQL API, the then-tested 0.1.2 to 0.2.0 transition, 18/18 package maturity, rollback, restore, and live postflight | passed for that candidate only |
| No second index, delta, external fusion, or benchmark-only product path | Static convergence inventory and package-bound product suite | passed |

The requested historical P2.2 scope was therefore closed. This evidence does
not qualify the current package; use the current-only maturity suite and
release checklist for a new release claim.

## P2.2 Final Package Evidence

The Linux x86-64 PG18 final-v3 package uses ONNX Runtime 1.26.0 and passed all
18 package-bound maturity steps in 1,849.45 seconds. The suite includes the
0.1.2 to 0.2.0 upgrade smoke, exact staged-versus-installed hashes, runtime
ABI/provider checks, restart, crash, corruption, concurrent DDL, physical
replication, 1,000-iteration healthy and failing-session memory soaks, and a
50,000-document mutable lifecycle.

The 50k run built in 484.45 seconds, compiled and atomically maintained the
unified delta in 474.40 seconds, reindexed in 480.79 seconds, and returned a
second warm query in 80.53 ms. The healthy 1,000-iteration soak had zero RSS
growth; the deliberate failure soak grew by 1,156 KiB and then stabilized,
well below its 16 MiB bound. The package fingerprint was
`3b017fa5af5e64a85e0592d407adb2996d34f8261c11343c4c83c8e7f9a2974d`
both before and after the suite.

The same source was built as a Linux PG18 package and installed on
Enlightenment. The target now runs the P2.2 final-v3 0.2.0 binary with its
bundled ONNX Runtime. The production arXiv index deliberately keeps its older
ABI-v1 model checkout and generation; contract-aware signatures allow that
generation to remain queryable without weakening drift rejection. New ABI-v2
generations use the expanded v2 build-option signature.

The final live postflight found PostgreSQL active, the installed binary at the
qualified SHA-256, extension version 0.2.0, all four batch runtime APIs, zero
runtime failures or queued work, and all three checked indexes valid,
query-ready, and signature-matched. No maturity process, temporary cluster, or
qualification tmux session remained.

### Native held-out relevance gates

The NFCorpus-calibrated ABI-v2 candidate was evaluated through `ii42_query`
on the official 5,183-document SciFact corpus with all 300 qrels-bearing test
queries. No SciFact-specific model or dataset-specific tuning was used.

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | p50 ms | p95 ms | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.663931 | 0.626833 | 0.882556 | 0.634838 | 0.965000 | 1.415 | 1.605 | 8,601,600 |
| P2.2 | 0.709942 | 0.670381 | 0.952667 | 0.679533 | 0.990000 | 15.839 | 19.389 | 30,318,592 |
| P2.2 - BM25 | +0.046010 | +0.043548 | +0.070111 | +0.044695 | +0.025000 | - | - | - |

This is a real positive relevance result, but not yet a broader promotion gate.
P2.2 uses `3.52x` the index bytes, took `244.96 s` rather than `0.66 s` to
build, and had `11.19x` BM25 query p50 on this local CPU. At least one more
held-out corpus was required before changing the product verdict.

The same frozen model and native route were then evaluated on the complete
8,674-document Arguana corpus with all 1,401 available qrels-bearing queries:

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | p50 ms | p95 ms | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 | 2.388 | 3.067 | 12,083,200 |
| P2.2 | 0.410565 | 0.286933 | 0.988580 | 0.285425 | 0.999286 | 45.573 | 69.884 | 42,532,864 |
| P2.2 - BM25 | +0.066450 | +0.049747 | +0.035689 | +0.051532 | +0.009279 | - | - | - |

Arguana independently preserves the positive quality direction, but at
`3.52x` index bytes, `226.34x` build time, `19.09x` query p50, and `22.79x`
query p95.

The same frozen model was then built from the official 25,657-document
SciDocs corpus and evaluated on all 1,000 qrels-bearing queries. A 2,000-row
shared-root subset was explicitly rejected before this run; the accepted
corpus row count and SHA-256 were checked before index construction.

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | p50 ms | p95 ms | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.150745 | 0.103333 | 0.348217 | 0.277420 | 0.561217 | 3.406 | 4.058 | 34,455,552 |
| P2.2 | 0.199677 | 0.142023 | 0.471850 | 0.344263 | 0.735017 | 57.024 | 71.433 | 116,875,264 |
| P2.2 - BM25 | +0.048932 | +0.038691 | +0.123633 | +0.066843 | +0.173800 | - | - | - |

SciDocs is the third independent row with positive quality deltas across all
five metrics. It also preserves the product trade-off: P2.2 uses `3.39x`
index bytes and has `16.74x/17.60x` BM25 query p50/p95 on the ARM64
qualification host.

The fourth row uses the complete official 171,331-document TREC-COVID corpus
and all 50 qrels-bearing queries. The strict evaluator verified the document
and query counts, relation-owned payloads, atomic valid generations, frozen
model identity, and native `ii42_query` route before accepting the result.
MAP@100 uses the standard cutoff denominator
`min(relevant_document_count, 100)`.

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | p50 ms | p95 ms | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.573101 | 0.313540 | 0.100236 | 0.817222 | 0.362831 | 4.154 | 5.701 | 202,539,008 |
| P2.2 | 0.745254 | 0.530110 | 0.156385 | 0.920000 | 0.406124 | 48.609 | 55.921 | 631,193,600 |
| P2.2 - BM25 | +0.172153 | +0.216570 | +0.056150 | +0.102778 | +0.043293 | - | - | - |

TREC-COVID closes the broader held-out quality gate in the same positive
direction, but preserves the cost warning: P2.2 used `3.12x` the index bytes,
took `10,042.40 s` instead of `10.82 s` to build, and had approximately
`11.70x/9.81x` BM25 query p50/p95.

The normalized four-row macro is:

| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.432973 | 0.320223 | 0.570975 | 0.490843 | 0.719764 |
| P2.2 | 0.516360 | 0.407362 | 0.642371 | 0.557305 | 0.782607 |
| P2.2 - BM25 | +0.083386 | +0.087139 | +0.071396 | +0.066462 | +0.062843 |

The macro is generated from checked native artifacts rather than copied from
the prose tables. All five P2.2 quality deltas are positive on every dataset
row, with one frozen NFCorpus-calibrated model and no dataset-specific tuning.
This establishes cross-domain improvement over BM25 on the completed
four-dataset surface; it does not establish dense equivalence or erase the
large build, storage, and latency gap. Enlightenment lifecycle, rollback,
full-text, package-bound maturity, native relevance, and exact target-host
performance qualification are complete.

## Remaining Work In Priority Order

1. **P1: sustained mutation throughput.** Measure larger high-churn workloads
   before qualifying mutable P2.2 for write-heavy production use.
2. **P1: further exact traversal work.** Profile full-corpus accumulation and
   selection before changing algorithms. The bounded top-k heap is exact but
   rejected because it more than doubled p50 on TREC-COVID.
3. **P2: optional product breadth.** Add RLS-aware ranking and globally ranked
   partition-parent support only if a real deployment requires them; current
   fail-closed behavior is safer than partial support.
4. **P2: isolated lossy Pareto work.** The exact and native relevance gates
   are now complete. Candidate caps, pruning, quantization, and approximate
   bounds may be tested only as one-variable experiments under the documented
   quality floors; the budget-0.75 canary was rejected and none is part of
   final-v3.

## Raw Evidence

Raw machine-readable outputs and build logs are stored in
[the qualification evidence directory](../data/raw/enlightenment-p2-qualification-2026-07-22/README.md).
