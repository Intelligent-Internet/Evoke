# Evoke: Unified Sparse Retrieval and Convergent Indexing in PostgreSQL

## System Technical Report (Beta 1)

**Edition:** 2026-09-03. **Implementation baseline:** `9a658b63`, extension `0.2.5`, page-native v3, packaged P2.2 model.

[繁體中文版](technical-report-ii42-system-zh.md)

### Abstract

Evoke is a PostgreSQL retrieval engine that extends the lexical foundation of `psql_bm25s` with model-generated sparse semantic evidence. Rather than placing BM25, a vector database, and a fusion service behind separate update pipelines, it represents lexical and semantic atoms in one relation-owned inverted index. PostgreSQL remains responsible for transactions, row visibility, recovery, and index lifecycle. The model compiles text into sparse contributions; the index makes those contributions searchable and maintainable.

The central systems problem is preserving efficient reads as documents, corpus statistics, and derived query structures change. Evoke addresses it through immutable posting objects, copy-on-write (COW) metadata, checked-root publication, bounded mutation frontiers, reusable term folds, and independently refreshed semantic accelerators. A compatible published accelerator remains usable while workers process newer evidence. This separates serving continuity from background convergence without confusing approximate ranking freshness with current-row visibility.

This report presents the architecture, scoring model, storage protocol, execution paths, and experimental evidence as one system. The frozen P2.1 quality evaluation reaches macro Recall@100 of 0.666885 on BEIR15 and 0.703125 on MTEB10, close to the corresponding dense references of 0.670880 and 0.707213. A historical lexical regression study restores the 5,183-document SciFact mean to 0.448 ms, compared with 0.454 ms for the original recorded `psql_bm25s` baseline. These are separate, versioned experiments, not a new benchmark of every feature in the current P2.2 package.

**Keywords:** PostgreSQL, BM25, sparse semantic retrieval, inverted index, copy-on-write, MVCC, background convergence, query acceleration.

## 1. Scope, Lineage, and Contributions

The original [lexical technical report](technical-report-psql_bm25s.md) records the BM25 foundation and earlier mutable-index engineering. The separate [model technical report](technical-report-ii42-model.md) documents model compilation, calibration, and detailed quality results. This new report connects those subjects to the current Evoke system; it replaces neither document.

The BM25 foundation draws on eager sparse scoring: repeated query-time work can be reduced by preparing term contributions and using sparse accumulation. [BM25S](https://arxiv.org/abs/2407.03618) develops this approach for Python sparse matrices. A transactional PostgreSQL engine faces additional requirements: corpus statistics change, tuple versions disappear, writes race with readers, and an index must survive crash recovery. Evoke therefore treats precomputation as a reusable representation of evidence, not as a permanently frozen corpus matrix.

The main contributions of this implementation are:

1. **One retrieval authority.** Lexical and semantic evidence share a posting namespace, physical index, and publication lifecycle; semantic retrieval does not require an independently maintained ANN index.
2. **Incremental immutable storage.** COW directories replace affected paths while sharing unchanged objects, and term folds reduce repeated traversal without losing uncovered evidence.
3. **Asynchronous model integration.** Shared runtime workers own model sessions; ordinary semantic-index writes publish lexical evidence and pending identities without foreground document inference.
4. **Serving/convergence separation.** Compatible baseline accelerators survive small mutations and metadata-only root changes, while background debt scheduling eventually publishes replacements.
5. **Explicit evidence boundaries.** Exact posting evaluation, approximate candidate execution, stale-baseline serving, and relevance against human judgments are evaluated separately.

This report describes Beta 1; `0.2.5` and P2.2 remain engineering identifiers. The report is a synthesis of the cited source and archived experiments. It does not assert that a particular server is currently running this revision. Operational contracts live in [Architecture](architecture-and-design.md), [Query Semantics](query-semantics.md), and [Maintenance Lifecycle](maintenance-lifecycle.md).

## 2. System Architecture and Product Surface

Evoke exposes one PostgreSQL access method, `USING ii42`. The default `sae = false` mode provides exact BM25. With `sae = true`, a qualified model adds semantic atoms to the same index. This report calls the product path Sparse Semantic Retrieval (SSR). The `sae` reloption remains the current SQL/catalog name; SAE is otherwise reserved for encoder or vocabulary mechanisms and historical experiment labels.

```text
                         PostgreSQL application
                                   |
                     SQL / ii42_query / predicates
                                   |
                    +--------------+--------------+
                    |                             |
             exact posting path          derived candidate path
                    |                             |
                    +----------+------------------+
                               |
                  tuple identity + MVCC recheck
                               |
                         ranked table rows

  +------------------- one Evoke index relation -------------------+
  | checked root -> COW manifest / term / document / lexicon trees |
  | lexical + semantic postings | linked L0 | folds | accelerators |
  +--------------------------------------------------------------+
              ^                                ^
              |                                |
      foreground lexical DML           background maintenance
      + pending semantic work          + shared model runtime
```

`ii42_query` has explicit-hit overloads for both modes. SSR additionally supports planner-native scalar markers: the planner turns an eligible ranked table query into a custom scan rather than calling a model once per row. BM25 also retains its native operator and ordered index-scan surfaces. `ctid` and index-local `doc_id` are execution identities, not durable application keys.

For a table `docs(id, title, body)`, after installing the extension and configuring the shared runtime and qualified model checkout:

```sql
CREATE INDEX docs_retrieval_idx
    ON docs USING ii42 (title, body)
    WITH (sae = true, field_aware = true);

SELECT d.id, hit.score
FROM ii42_query(
    'docs_retrieval_idx'::regclass,
    'transaction-safe semantic retrieval',
    ARRAY['title', 'body']::text[],
    ARRAY[2.0, 1.0]::real[],
    20::int4
) AS hit
JOIN docs AS d ON d.ctid = hit.ctid
ORDER BY hit.score DESC, d.id;
```

This is one field-aware index, not two independent top-k lists followed by late fusion. Field namespaces preserve identity and permit query-time weighting. The current field-aware contract is not BM25F-style independent per-field length normalization. Without `field_aware`, multicolumn input is fused into one logical document. See [Field-Aware Indexes](field-aware-indexes.md) and [Getting Started](getting-started.md).

## 3. Mathematical Retrieval Model

### 3.1 Lexical Evidence and Statistics

Let $N$ be document count, $df_t$ term document frequency, $tf_{t,d}$ term frequency, and $|d|$ document length. For the Lucene-style variant used below:

```math
\mathrm{idf}_t = \log\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right),
\qquad
L_t(d)=\mathrm{idf}_t\,
\frac{tf_{t,d}}
{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}.
```

The omitted global $(k_1+1)$ multiplier does not change a lexical ranking with fixed parameters. It does matter when calibrating relative lexical/semantic scale, so a model/index contract must fix the convention. The historical model evaluation uses $k_1=1.5$ and $b=0.75$; other supported BM25 variants are described in the original report.

Evoke distinguishes **neutral evidence**, such as term frequency, from **statistics-specialized impacts**. A neutral fold can survive a corpus-statistics change. A specialized impact fold is reusable only with its matching statistics epoch. This preserves the benefit of eager scoring without forcing a full posting rewrite on every change in $N$, $df_t$, or average length.

### 3.2 Semantic Atoms and Unified Scoring

For encoder logit $z_{ij}$ at token position $i$ and semantic coordinate $j$, the sparse foundation produces nonnegative sequence impacts:

```math
w_j(x)=\max_{i\in x}\log\left(1+\mathrm{ReLU}(z_{ij})\right).
```

The model compiler applies the qualified support, calibration, and text-window policies to produce published query and document impacts. Lexical coordinates $\mathcal{V}_L$ and semantic coordinates $\mathcal{V}_S$ are disjoint. A conceptual single-field score is:

```math
S(q,d)=
\sum_{t\in\mathcal{V}_L}q^L_t L_t(d)
+c(q)\sum_{j\in\mathcal{V}_S}\hat q_j\hat d_j.
```

The query scale $c(q)$ belongs to the model calibration contract. It is not an application-side reciprocal-rank fusion coefficient. Both sums contribute to the same document accumulator, allowing a document to rank through combined moderate lexical and semantic evidence rather than requiring it to enter either independent top-k list first.

For selected fields $f$ with weights $a_f$, the field-aware composition is:

```math
S_{fields}(q,d)=\sum_f a_f S_f(q,d).
```

This notation expresses additive field-scoped scoring, not a claim that the engine implements every BM25F normalization scheme.

### 3.3 Precision, Retention, and Exactness

Three choices operate at different layers:

| Choice | Layer | Meaning |
| --- | --- | --- |
| Runtime precision, default `fp16` | ONNX inference | Execution precision/provider for text compilation |
| Semantic impact storage, default `f32` | Canonical postings | Representation of compiled semantic values; `u8` is an explicit quantized profile |
| `semantic_alpha_mass`, default `1.0` | Per-field publication | Fraction of positive semantic mass retained after the normal model budget |

For impacts sorted by descending mass after that budget, a retention policy can be expressed as selecting the smallest prefix $J_\alpha$ satisfying:

```math
\sum_{j\in J_\alpha}\hat d_j
\geq \alpha\sum_j\hat d_j,
\qquad 0<\alpha\leq 1.
```

For nonnegative query impacts, discarded mass provides the following explanatory bound on the unscaled semantic dot-product loss, before any separate quantization error:

```math
0\leq\Delta S_{sem}(q,d)
\leq\|\hat q\|_\infty
\sum_{j\notin J_\alpha}\hat d_j.
```

This is a mathematical interpretation of retention, not an implemented top-k pruning certificate. Small total mass loss does not guarantee unchanged ordering near a score tie. `f32`/alpha `1.0` means no additional storage/retention approximation at those layers; it does not make a candidate accelerator exhaustive or make pending semantic work immediately visible. Exact execution means exact evaluation of the applicable published representation and visibility contract, not equivalence to an unpruned neural model or human relevance judgments.

## 4. Model Compilation and Inference

### 4.1 A Versioned Compiler, Not an Unversioned Service

The packaged model is bound by [the model lock](../packaging/milestone-model.json):

| Identity | Value |
| --- | --- |
| Bundle | `evoke-p2.2-nfcorpus-v2` |
| Model ID | `evoke_p2_p22_nfcorpus_v2_smoke` |
| Runtime ABI | `evoke_p2_unified_text_atoms_v2` |
| Manifest SHA-256 | `b61060a3958ee56209de47a34ee5cbe08351bfeb3fcbbfdcbf477403210764f7` |
| ONNX Runtime | `1.29.0` |

The identical frozen checkout is available as [Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1). The [download guide](examples/semantic-model-checkout.md#download-the-default-model) pins its revision and archive checksum; distribution does not change the model or the historical evaluation identity.

The upstream checkpoint is `ibm-granite/granite-embedding-30m-sparse`, revision `ad82b1fd09541c998c8d45045d601c51fdb8a9b7`. Its approximately 30.3M parameters provide a compact sparse retrieval foundation. The upstream model family and training approach are described in [Granite Embedding Models](https://arxiv.org/abs/2502.20204). Evoke's local work concerns compilation, calibration, publication, and systems integration; these should not be confused with training the foundation from scratch.

P2.2 uses deterministic ABI-v2 windows for full-text query and document compilation. The packaged lexical vocabulary and calibration are frozen from NFCorpus. This extends the engineering text path beyond the historical P2.1 single-sequence evaluation, but does not itself establish new long-document or cross-domain benchmark results.

```text
                        qualified model checkout
                      tokenizer + ONNX + calibration
                                | signature
             +------------------+------------------+
             |                                     |
       document input                         query input
             |                                     |
   deterministic text windows            deterministic text windows
             |                                     |
      shared runtime encode                 local query runtime lane
             |                                     |
   aggregate / calibrate / budget          aggregate / calibrate
             |                                     |
      lexical + semantic atoms              sparse query atoms
             |                                     |
      canonical index postings  <-------- matching + accumulation
                                                   |
                                            candidate/row recheck
```

The compiler/runtime contract binds the interpretation of atom IDs and values. A matching filename or model dimension alone is insufficient. Incompatible checkout identities fail closed instead of silently mixing postings from one model with queries from another.

### 4.2 Runtime Ownership and Optional Remote Encoding

Model sessions belong to shared runtime workers, not to every SQL backend or a shared ONNX session in the arena. Backends submit bounded requests and receive sparse results. SSR requires the configured shared runtime; it does not silently fall back to a private per-backend model. With query-lane reservation enabled and at least two healthy local workers, a reserved lane protects query encoding admission from document batches.

Optional remote runtime services add **document encoding capacity** for builds and maintenance. They do not own PostgreSQL pages, execute database queries, or replace the local query lane. The dispatcher considers outstanding requests, service batch limits, observed latency, and backoff. Out-of-order batch completion can release runtime slots while publication preserves document ordering.

This is different from a **persistent query accelerator**, which is an index-owned derived object built from stored postings. Removing remote encoding capacity after bulk ingestion does not inherently remove an already published query accelerator. Both mechanisms are described in [Shared Runtime and Residency](shared-runtime-and-residency.md).

### 4.3 What the Model Experiment Contributed

The frozen P2.1 experiment separates representation learning from lightweight retrieval calibration. It keeps the upstream encoder fixed and learns three scalars governing query power, document power, and score scale. Training uses 10,000 MS MARCO teacher rows and 1,000 query-disjoint validation rows, with one positive and eight negatives per row. The resulting monotonic transform is:

```math
\hat q_j=0.696368\,q_j^{1.851864},
\qquad
\hat d_j=d_j^{0.562796}.
```

Support membership remains fixed at query top-50 and document top-192 before the publication policy. The historical b1.125 policy allocates semantic postings relative to lexical posting count; it is not the current per-field alpha-mass option. Query-local RMS calibration then aligns lexical and semantic activation scales without inference-time qrels. This provides a small, inspectable adaptation surface instead of introducing another large model into the serving stack.

The upstream foundation uses both public and non-public training material. Reproducibility here covers the fixed checkpoint and the downstream compiler/evaluation pipeline. Current P2.2 identity and historical P2.1 measurements remain separate; full training objectives, selection caveats, and artifact references are in the [model report](technical-report-ii42-model.md).

## 5. Physical Index and Copy-on-Write

### 5.1 Page-Native Objects

All authoritative index storage resides in a PostgreSQL index relation. The checked root identifies an immutable manifest and the active/pending linked-L0 frontiers. Beneath it are term directories, document/version metadata, lexical lookup structures, canonical posting extents, durable term folds, and optional accelerator references.

```text
  checked root
      |
      +-- immutable manifest
      |      +-- term COW tree ------> extents / neutral / impact folds
      |      +-- document COW tree --> doc slots / versions / TIDs / lengths
      |      +-- lexical lookup ----> full term bytes / stable term IDs
      |      +-- accelerator dir ---> seeds / forward rows / scope data
      |
      +-- pending L0 frontier ------> immutable interval being processed
      +-- active L0 frontier -------> transaction-aware incoming records
```

Canonical runs distinguish lexical neutral evidence from direct semantic impacts. Stable term IDs avoid renumbering the corpus when vocabulary grows. Hash-based lexical lookup still checks full term bytes, so a hash collision cannot become a false lexical match. An ordered prefix structure supports bounded prefix expansion. Document metadata preserves slot incarnation and tuple identity so recycled storage cannot alias an earlier document.

### 5.2 Path Copying and Complete References

The term directory uses a persistent 64-way radix tree with 16-term leaves. A change creates replacement leaves and affected ancestors; untouched branches remain shared. Other COW structures use their own layouts but obey the same immutable-reference principle.

```text
  reader pinned at R0                   newly staged R1
          |                                    |
         A0                                   A1
        /  \                                 /  \
       B    C0                                B    C1
           /  \                                /  \
          D    E0                             D    E1
               |                                   |
          old term record                     changed record

  R0 remains readable. Publish R1 only after its new closure is valid.
  Shared B and D are not rewritten merely to change their owner.
```

Each external reference carries physical location, object identity, ownership, length, and validation information, not merely a logical ordinal. New objects are prepared and written bottom-up. Their actual page locations are bound before a parent is serialized. This matters because concurrent L0 allocation can interleave with COW allocation: assuming consecutive physical pages would corrupt otherwise valid logical trees.

The implementation preserves ancestor-owned references when descendants share an object. It does not recursively copy a complete historical closure simply to assign a new owner. For $m$ changed leaves in a tree of height $h$, newly written metadata follows affected paths, conceptually $O(mh)$ before shared-path deduplication, rather than a mandatory $O(|\mathcal{V}|)$ full-vocabulary rewrite. This is a metadata bound, not a bound on the posting payloads being changed or on an entire accelerator rebuild.

Implementation anchors are [term COW](../src/ii42_term_cow.c), [document COW](../src/ii42_document_cow.c), [lexicon COW](../src/ii42_lexicon_cow.c), and the bottom-up writers in [segment pages](../src/ii42_segment_pages.c). The full storage design is in [Convergent Segmented Index](convergent-segmented-index.md).

### 5.3 Checked Publication and Reclamation

COW construction is preparation, not publication. The publisher validates the staged closure and its source/frontier assumptions, then installs a checked successor with a short WAL-backed root transition. Readers use one coherent authority, not half of the old manifest and half of the new one. The engine follows PostgreSQL's [index access-method](https://www.postgresql.org/docs/18/indexam.html) and [extension WAL](https://www.postgresql.org/docs/18/wal-for-extensions.html) integration model.

```text
  capture source -> prepare new objects -> validate closure/frontiers
                                               |
                                  short checked-root publication
                                               |
                     +-------------------------+----------------+
                     |                                          |
                new readers use R1                    old readers finish R0
                                                                |
                                                  retire / reclaim safely
```

Reclamation is a separate phase within the same relation lifecycle. Superseded objects are not immediately reusable while a reader can still reference them. Once retirement conditions and the reader fence permit, bounded work returns pages to reusable storage. Conditional fence acquisition yields to existing readers, but a successfully held fence can briefly delay new readers; direct reused-page writes remain protected through publication. Failed preparations also need cleanup. Object identity and document incarnation checks remain necessary when pages or slots are reused.

## 6. Mutation, Sealing, and Convergence

### 6.1 Initial Build and Publication

An initial build follows PostgreSQL's `table_index_build_scan` visibility protocol, assigns document identities, and compiles lexical evidence. SSR builds additionally encode documents through bounded runtime batches. The builder emits canonical posting objects and COW metadata, validates their closure, and publishes the checked root. The same representation is the target of `REINDEX`; a healthy current-format index does not need a corpus rebuild merely to refresh a compatible derived accelerator.

```text
  PostgreSQL build protocol -> heap scan -> lexical compilation
                         |
                         +-> SSR document batches -> model runtime
                         |                              |
                         +--------- unified atoms <-----+
                                         |
                          posting objects + COW metadata
                                         |
                           validation + checked root
                                         |
                            query-ready exact storage
                                         |
                      background accelerator / warmup
```

`CREATE INDEX` and `REINDEX` establish the canonical root; optional semantic accelerator construction is a subsequent background task. Finishing the heap scan is therefore not equivalent to publishing the index, and publishing the index is not equivalent to completing every derived performance artifact. Build throughput, time to query readiness, and time to warm accelerated service are separate measurements.

### 6.2 Lexical-First Writes

For an SSR index, foreground DML records lexical changes and a pending semantic identity. It does not run document inference inside the write transaction. Background completion checks that the document version, source text, and model contract still match; stale results are discarded rather than attached to a replacement row. Repeated row-specific failures are surfaced and quarantined instead of silently blocking the queue forever.

```text
  INSERT / UPDATE transaction
             |
     lexical evidence + pending identity
             |
     transaction-aware linked L0
             |
             +--> commit / abort / savepoint / prepared transaction
             |
             +--> rotate / seal lexical evidence without waiting for model
             |
             `--> bounded semantic batch -> encode changed fields
                           |
                 revalidate version + contract
                           |
                 append completion to linked L0 -> later seal

  checked sealed state -> selected fold / compact / accelerator refresh
```

Heap MVCC governs returned rows. Deleted or superseded tuple versions cannot be made visible by an old semantic score. Conversely, a compatible approximate baseline can omit a newly inserted or newly improved document until refresh. Exact row visibility and immediate completeness of semantic ranking are different guarantees. `REPEATABLE READ` preserves the PostgreSQL snapshot; it does not pin a historical ranking root across all statements.

### 6.3 Frontiers and Term Folds

Active L0 receives incoming records; a pending interval is processed independently. Sealing one interval must preserve later active-frontier writes. Canonical segments retain reusable evidence. Compaction reorganizes persisted postings; it does not re-encode unchanged documents.

For a term $t$, let $F_t^{\leq c}$ be a fold covering sequence boundary $c$, and $E_{t,r}$ its later extents. The exact logical term stream can be described as:

```math
P_t = F_t^{\leq c}\;\oplus\!
\bigoplus_{r:\,\mathrm{sequence}(r)>c} E_{t,r}.
```

Here $\oplus$ means visibility/version-aware reconciliation, not blindly adding duplicate postings. Authenticated coverage watermarks prevent omissions and double counting when a minor fold becomes a larger fold. Neutral folds reuse evidence across statistics epochs; specialized folds require a matching epoch. Together they form a virtual sparse column without demanding one contiguous rewritten corpus matrix.

### 6.4 Periodic Work Without a Serving TTL

The scheduler distinguishes urgency from opportunity. High-water L0 debt, pending work, missing required artifacts, and incompatible artifacts demand attention. Small debt is eligible after a periodic interval, so a quiet index can converge without crossing a large threshold. The initial low-debt time anchor is not moved forward by every subsequent write.

In this baseline, L0 high-water checks include 65,536 records or 512 pages; accelerator refresh also considers sealed-record and byte debt, with a one-hour default low-debt interval. These are deployment controls, not fundamental properties of sparse retrieval. The serving baseline has **no maximum age solely because delta exists**: a compatible artifact may remain valid until a replacement is published.

Optional accelerator preparation uses a separate per-index build lock and does not hold the urgent-maintenance/discovery lock across its corpus pass. Queries, writes, and safe frontier work can proceed. Conflicting immutable publications may defer or force a retry; urgent sealing wins when frontiers cannot safely absorb more work. This is deliberate serialization at publication boundaries, not a claim of a universally lock-free system.

## 7. Query Execution and Filter Semantics

### 7.1 Exact and Derived Paths

An eligible query binds the index/model contract, compiles query atoms once, selects a serving representation, scores candidates, and validates returned tuple identities. The exact posting route is the reference for the published representation. The preferred derived route uses compatible candidate and compact scoring artifacts when admitted.

```text
  SQL text + field weights + optional predicate
                       |
            bind index / compile query once
                       |
        compatible accelerator and memory admission?
                    /     \
                  yes      no
                   |        |
           derived candidates    exact posting traversal
           + compact scores      + fold / uncovered evidence
                    \       /
                     \     /
               current-row MVCC / qual recheck
                            |
                       ranked results
```

With a compatible stale baseline, the accelerated route uses bounded overfetch to compensate for changed or invalidated rows, but does not merge an unbounded exact post-baseline posting tail on every query. That earlier coupling would turn background delay into foreground work amplification. New evidence becomes available through later publication. Falling back to exact execution is still a potentially more expensive route, not a latency guarantee.

### 7.2 Filters Are Part of Retrieval, Not Only Output Validation

For an allowed set $A$, filtering a short unfiltered prefix is generally not equivalent to retrieving top-k within $A$:

```math
\mathrm{TopK}\{S(q,d):d\in A\}
\neq
A\cap\mathrm{TopK}\{S(q,d):d\in D\}.
```

The mismatch can persist after overfetch, especially if the scorer or candidate policy depends on the selected scope. Testing only that every returned row satisfies the predicate does not test filtered ranking quality.

Evoke builds same-root scope metadata from eligible `INCLUDE` columns. Supported planner predicates include direct conjunctions of equality, overlap, ranges, and admitted `ILIKE` shapes. Structured JSON filters can also reuse a compatible published scope baseline. Both recheck current-row membership and may omit post-baseline matches under the documented approximate contract.

The fallback contracts are intentionally explicit. Planner-native execution can use complete visible TIDs when a scope-backed path is unavailable or unsuitable. Fully scope-backed structured requests can return fewer than k while converging instead of automatically materializing the complete matching universe. Other structured requests first probe SQL membership with a 65,536-match limit and an overflow witness. A complete probe supplies the TID set; overflow may admit a bounded global rank prefix before full SQL resolution. The match limit does not bound rows scanned or elapsed time. Therefore neither “all filters are exact-current enumeration” nor “all filters avoid enumeration” describes the product. The complete overload-specific rules are in [Query Semantics](query-semantics.md).

## 8. Persistent Semantic Query Accelerators

### 8.1 Derived Representations of Existing Evidence

A query accelerator is built from an immutable posting baseline. It combines selected per-term seed documents, cross-term residual candidate accumulation, compact forward rows for direct scoring, and scope metadata where applicable. Forward and inverted views serve different access patterns: the inverted view finds documents associated with query atoms; the forward view evaluates a selected document without repeatedly traversing every long posting list.

```text
  immutable lexical + completed semantic postings
                       |
             stream one term at a time
                       |
         seeds + residual data + temporary transpose
                       |
          compact forward rows + INCLUDE scope
                       |
           validate source/policy + publish directory
                       |
  query atoms -> candidates -> forward scores -> current-row recheck
```

The present implementation uses policy 7, accelerator directory v10, forward/transpose format v6, and scope format v6. These component versions are not the extension version. The current candidate design includes up to 64 seed documents per selected term and additional cross-term work; “64 seeds” does not mean “only 64 results can ever be examined.” Compact derived rows use a per-row scale, delta-varint term IDs, and signed-int8 contributions. This is distinct from the optional unsigned `u8` codec for canonical semantic postings.

### 8.2 Bounded Memory Is Not Constant Total Work

The builder streams postings and uses a temporary-file-backed transpose. Admission accounts for document metadata, vocabulary state, the largest decoded term, sort workspace, and publication scratch. It avoids holding all corpus postings in RAM. Scope extraction reads heap values under bounded MVCC snapshot batches, including external TOAST data; keeping one old snapshot throughout a long build would unnecessarily retain dead tuples.

A complete refresh can still scan the baseline's full posting population and collect its scope values. It is **not** an incremental O(delta) model-inference job, and it need not finish inside one scheduler tick. Separating its locks from urgent maintenance and applying retry cooldown prevents it from becoming a serializing foreground dependency. CPU, disk bandwidth, page cache, WAL, and temporary space remain shared physical resources and must be measured under load.

### 8.3 Freshness and Warmness Have Different Identities

Four independently observable axes describe a serving index:

| Axis | Question |
| --- | --- |
| Visibility | Are returned row versions valid for this SQL snapshot? |
| Semantic completion | Have pending document versions received model evidence? |
| Accelerator freshness | Which completed baseline does the derived executor represent? |
| Warmness | Are useful metadata and pages available in the intended cache/shared-memory tier? |

`ready_baseline_delta` is serviceable, not equivalent to “no accelerator.” `baseline_current=false` reports outstanding convergence. Warm markers are tied to the serving accelerator/baseline identity rather than being discarded on every unrelated manifest advance. This permits COW publication and small ingress to coexist with a stable warm query path.

Shared-runtime capacity, page-prewarm I/O budgets, and exact resident-fold admission are different budgets. Reading pages into cache does not reserve them permanently, and increasing a warmup budget does not remove posting work or disk amplification. Models, shared representations, backend scratch, and OS cache must be accounted for separately; summing process RSS can double-count shared pages. See [Connection Memory](connection-memory.md) and [Shared Runtime and Residency](shared-runtime-and-residency.md).

## 9. Experimental Evidence

### 9.1 Method and Attribution

The evidence is organized into lexical efficiency, model relevance, derived-executor efficiency, and incremental storage behavior. Each answers a different question. The figures below are copied from the cited checked-in records, not generated by a fresh run while writing this report.

Latency comparisons must specify corpus/tokenizer identity, SQL shape, k, cache state, model/codec, executor policy, hardware, and concurrent work. Server execution time excludes application networking and downstream RAG stages; a direct-SRF microbenchmark is also not automatically equivalent to `EXPLAIN ANALYZE` of a complete business query. Quality is macro-averaged where stated, not pooled across all queries.

### 9.2 Lexical Foundation and Regression Recovery

The [2026-08-18 BM25 regression study](performance/reports/bm25-page-native-regression-2026-08-18.md) uses SciFact: 5,183 documents, 1,109 queries, 26,559 terms, top-1000, fixed tokenization, and Lucene BM25 parameters. It uses two warmup passes and three measured passes for the repaired run. The [machine-readable record](performance/data/diagnostics/bm25-page-native-regression-2026-08-18.json) preserves the fixture and timings.

| Recorded implementation | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Original official `psql_bm25s` baseline | 0.454 | 0.451 | 0.525 |
| Page-native path before repair | 63.430 | 63.370 | 65.896 |
| Repaired page-native path | 0.448 | 0.435 | 0.558 |

The regression was not an intrinsic cost of sparse semantic inference: this was a BM25-only fixture. Repeated per-hit document-COW projection dominated the page path. Grouped block projection, bounded resident scoring, and compact tie-selection state restored the recorded mean. The repaired mean is approximately 1.4% lower than the old record, while p95 is approximately 6.3% higher; “no regression at every percentile” would misstate the result. On 101 sampled queries, resident versus forced-page results had zero identity/order mismatches and zero maximum score delta.

This establishes a useful lexical regression fixture, not a current cross-engine ranking across all PostgreSQL BM25 products. Historical and same-cluster controls in the full study must remain distinct.

### 9.3 Model Milestone: P2.1 Unified Retrieval

The frozen 2026-07-15 P2.1/b1.125 evaluation covers BEIR15 with 46,417 queries and 33,860,494 documents, and MTEB10 with 8,815 queries and 1,096,451 documents. All values below are equal-weighted dataset/task macro averages. CUB@1000 is the reported positive-document candidate coverage at a candidate budget of at most 1,000, not measured reranker performance.

| Suite | Retriever | NDCG@10 | Recall@100 | CUB@1000 |
| --- | --- | ---: | ---: | ---: |
| BEIR15 | BM25 | 0.374297 | 0.562964 | 0.735619 |
| BEIR15 | PPLX dense / VectorChord | 0.544873 | 0.670880 | 0.810525 |
| BEIR15 | Evoke P2.1 | 0.490809 | 0.666885 | 0.839658 |
| MTEB10 | BM25 | 0.383554 | 0.595367 | 0.777387 |
| MTEB10 | PPLX dense / VectorChord | 0.544127 | 0.707213 | 0.841063 |
| MTEB10 | Evoke P2.1 | 0.503440 | 0.703125 | 0.875910 |

Measured relative to the BM25-to-dense Recall@100 improvement, P2.1 recovers about 96.3% on each suite:

```math
G_R=\frac{R_{P2.1}-R_{BM25}}{R_{dense}-R_{BM25}}.
```

The main milestone is semantic candidate coverage close to the dense reference within one sparse posting index, with higher reported candidate upper bounds on these macro aggregates. Improving early ranking remains a clear direction: NDCG@10 is below the dense reference even where Recall@100 is close.

These results use the historical P2.1 support, calibration, publication budget, and text limits. They are not a P2.2 full-text/U8/alpha-0.50 benchmark. Thirteen BEIR dense rows were reused from earlier VectorChord results and two were freshly collected; dense query-encoder latency is not included as a comparable end-to-end cost. Some calibration/policy selection used BEIR evidence, so this is not an entirely untouched zero-shot evaluation. The [model report](technical-report-ii42-model.md) supplies per-dataset values, methodology, and artifact links.

### 9.4 Derived-Executor Evidence and Rejected Shortcuts

The [accelerator execution record](performance/reports/semantic-accelerator-bounded-execution.md) separates prototype and installed-binary evidence. Its policy-3 Shadow ArXiv deployment comparison records:

| Fixed surface | Exact p50/p95 ms | Derived p50/p95 ms | Mean/minimum O@100 |
| --- | ---: | ---: | ---: |
| Cross-domain 150 queries | 215.22 / 272.56 | 138.83 / 163.71 | 0.9826 / 0.92 |
| Independent TREC 50 queries | 218.88 / 272.28 | 134.50 / 153.27 | 0.9834 / 0.92 |

O@100 measures overlap with the exact top-100, not qrels recall. The corresponding artifact rebuilt from immutable postings in 2 minutes 21 seconds without document inference. These results demonstrate the value of compact forward scoring and reusable postings. They belong to historical policy 3, whereas the current design is policy 7; current-policy qualification must bind new measurements to the installed artifact and binary.

The same record rejects superficially attractive shortcuts. On a 5.23-million-document Hotpot surface, a roughly 11 ms geometric search supplied insufficient candidates; restoring exact residual coverage made total execution slower than the exact route. Increasing per-term seed caps also failed to provide a favorable global quality/cost tradeoff. The lesson is to optimize **complete retrieval work at acceptable ranking quality**, not a fast isolated candidate kernel.

### 9.5 COW Update Cost and Storage Convergence

The [COW design record](convergent-segmented-index.md) reports a fixed-change metadata experiment after removing recursive full-root reuse inventory:

| Vocabulary terms | Before median ms | Changed-path median ms |
| ---: | ---: | ---: |
| 2,000 | 5.93 | 1.29 |
| 20,000 | 64.81 | 1.51 |
| 100,000 | 272.93 | 1.93 |

This is a historical metadata microbenchmark, not end-to-end index-build throughput. It supports the specific design claim that a small update should not walk all unchanged vocabulary metadata. Separately, the [engineering development record](archive/engineering/convergent-segmented-index-development-record.md) documents fixed-live-set mutation tests where page and document-slot high-water marks plateau after reclamation and incarnation-safe reuse. Together these tests address both CPU amplification and the disk cost of immutability.

## 10. Engineering Qualification and Reproducibility

The core proof obligations are larger than a scorer test:

| Property | Required observation |
| --- | --- |
| Ranking | Result identity, order, score tolerance, qrels metrics, and full filtered-reference comparisons |
| Transactions | Commit/abort, savepoints, prepared transactions, HOT/non-HOT updates, deletes, and TID reuse |
| Publication | No mixed-root reads; failed or cancelled preparation preserves the old readable root |
| Convergence | Pending work drains; admitted accelerators eventually advance after writes settle |
| Serving continuity | Warm baseline before/during/after maintenance; latency tails and lock waits, not only readiness flags |
| Resources | Private memory/PSS, shared memory, temporary disk, WAL, page growth, and post-drain plateaus |
| Recovery | Cold restart, crash replay, `REINDEX`, reader-safe reclamation, and physical replication |
| Package identity | Extension SQL/binary, PostgreSQL major, ORT ABI, model digest, and accelerator policy |

The [validation guide](testing-and-validation.md) maps these obligations to isolated PostgreSQL lifecycle, writer-concurrency, semantic-fairness, memory, and replication harnesses. Python unit/contract tests are a separate layer and do not establish a deployment-level latency result.

A reproducible mixed-load experiment records a warm read-only baseline, parallel-reader control, sustained writes plus maintenance, and a post-drain phase. Sample query p50/p95/maximum, rows/score quality, source and serving generations, L0/semantic debt, worker actions, processed bytes, I/O, memory, and disk growth together. A healthy system may show bounded jitter and temporarily older rankings; readiness alone cannot prove that workers are making progress or that query work remains stable.

PostgreSQL owns `DROP INDEX` teardown and physical replication of index pages. External model/runtime artifacts still need matching provisioning on standbys. Logical replication transfers rows rather than the physical Evoke index. RLS-backed search and globally ranked partitioned-parent indexes remain outside the current supported surface; parallel heap build, parallel AM scan, and parallel VACUUM discovery are further engineering directions. Current boundaries and installation details are in [the README](../README.md) and [Migration](upgrading.md).

## 11. Improvement Directions

The next stage is to strengthen the same architecture rather than add competing index authorities:

- **Broader model evidence and stronger early ranking.** Repeat the full quality matrix on the packaged P2.2 text path, expand independent holdouts and long-document evaluation, and improve calibration so high candidate recall translates into better early precision.
- **High-document-frequency efficiency.** Reduce total posting and residual work while validating complete top-k quality, especially for broad filters and long queries; assess end-to-end cost rather than geometry-only speed.
- **Predictable concurrent operation.** Extend mixed-load measurements across corpus scales and hardware, improve background resource admission and fairness, and make freshness/latency tradeoffs easier to inspect.
- **Compact durable and derived representations.** Continue measuring bytes per posting, COW metadata overhead, temporary build space, and cache residency alongside retrieval quality and recovery behavior.
- **Deployment breadth and reproducibility.** Expand qualified input/workload coverage and publish tightly bound binary/model/data manifests with current measurements.

Detailed proposals, experiment matrices, and promotion gates belong in [Product Roadmap](product-roadmap.md) and [Model Planning](model-planning.md). They are not implied implemented features of this report.

## 12. Conclusion and Review Map

Evoke's engineering proposition is that lexical and learned sparse retrieval can share not only a score accumulator, but a transactional storage and maintenance design. COW makes immutable evidence reusable; bounded frontiers localize ordinary mutation work; workers compile semantic evidence and prepare derived read structures independently; checked publication makes new structures visible without dismantling the old serving path first.

The existing experiments establish meaningful milestones: preserved lexical efficiency on a controlled regression fixture, strong sparse semantic candidate recall, useful derived-executor latency reductions, and reduced metadata amplification. The next evaluation priority is their joint behavior on the current packaged model under realistic mixed workloads. The appropriate success criterion is sustained useful ranking and predictable resource use throughout convergence, not merely a fast read-only snapshot or an all-green status page.

For code-oriented review, start with the following map:

| Area | Primary references |
| --- | --- |
| Lineage and model results | [Lexical report](technical-report-psql_bm25s.md), [model report](technical-report-ii42-model.md) |
| Index layout and COW | [Storage design](convergent-segmented-index.md), [term COW header](../src/ii42_term_cow.h), [segment pages](../src/ii42_segment_pages.c) |
| Query and filters | [Query contract](query-semantics.md), [page query](../src/ii42_page_query.c), [scope](../src/ii42_scope.c), [filter](../src/ii42_filter.c) |
| Model execution | [P2 runtime](../src/ii42_p2_runtime.c), [runtime contract](shared-runtime-and-residency.md), [model lock](../packaging/milestone-model.json) |
| Acceleration | [Builder](../src/ii42_am_accelerator.c), [directory](../src/ii42_semantic_accelerator_directory.c), [forward format](../src/ii42_semantic_forward.c), [execution evidence](performance/reports/semantic-accelerator-bounded-execution.md) |
| Concurrency and qualification | [Lifecycle](maintenance-lifecycle.md), [scheduler](../src/ii42_am_scheduler.c), [validation](testing-and-validation.md) |

External foundations: [BM25S](https://arxiv.org/abs/2407.03618), [Granite Embedding Models](https://arxiv.org/abs/2502.20204), and the PostgreSQL [index access-method](https://www.postgresql.org/docs/18/indexam.html) and [extension WAL](https://www.postgresql.org/docs/18/wal-for-extensions.html) documentation. External work is credited for its own contributions; the Evoke performance figures above come from the linked repository evidence, not from those papers.
