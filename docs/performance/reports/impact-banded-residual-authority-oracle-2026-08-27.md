# Impact-Banded Residual Authority Oracle

Date: 2026-08-27

## Decision

The representation experiment is positive, but it is not product latency
proof.

The only retained exact product candidate is:

```text
current cap-64 selected-term seed
  -> document-order stream of 64-document residual blocks
  -> impact-descending postings inside each block
  -> exact block-suffix upper bounds
  -> resolve partial candidates every 1,024 blocks
  -> deterministic exact top-k
```

This route requires at most 65,536 pending document slots per batch and does
not require a query-global block sort. Across full FiQA, TREC-COVID and Touche
roots it decoded 5.01% to 25.31% of residual postings on average while
preserving O@100 = 1.0 for every measured query. NFCorpus also remained exact,
but its p95 decode ratio was 1.0; the route is not universally sublinear and
must retain an exact fallback.

The projected exact f32 stream occupies 44.97% to 50.62% of the current packed
semantic authority on the four measured roots. This makes a replacement format
plausible. It does not justify a duplicate sidecar. No installed SQL, root
format, writer, worker, or query route changes in this branch.

The u8 result is not rejected. With a product tolerance of minimum
O@100 >= 0.98, it is a credible explicit aggressive profile. Across the 419
measured queries, its query-weighted mean O@100 is 0.99766. The exact f32
authority remains the default candidate; fp16 is the conservative approximate
candidate; u8 should proceed to native validation as an opt-in profile.

## Why This Is New

The historical Phase 4.6 and M1569 experiments traversed one global
impact-ordered frontier. Many active dimensions kept the global residual bound
high, so those experiments touched most documents and were correctly rejected.

The current experiment differs in four ways:

1. the existing product-shaped high-DF surface first admits at most 64
   documents per selected query term and establishes a useful exact kth score;
2. residual postings are partitioned into disjoint document blocks;
3. each block has its own term-local impact frontier, so a broad term cannot
   keep unrelated document ranges alive;
4. exact forward resolution is delayed and bounded rather than performed for
   every first-seen document.

This also follows the 2026-08-25 essential-term MaxScore result: the remaining
problem is a skip-capable physical authority, not another planner ratio or
route threshold.

## Exactness Contract

For block `b`, residual term `t`, query weight `q_t >= 0`, and the next
unvisited impact `d_t,b,p`, define:

```text
U_b = sum_t q_t * d_t,b,p
```

Postings within each term/block slice are sorted by non-increasing impact.
Therefore every unseen document in `b` has residual contribution no greater
than `U_b`. A block suffix is skipped only when `U_b` is strictly below the
visible kth score, with an explicit floating-point slack. Partial candidate
score plus the remaining `U_b` is also a safe candidate upper bound.

The proof depends on these invariants:

- document and query impacts are nonnegative; signed coordinates are separate
  positive namespaces;
- document blocks are disjoint;
- the cap-64 seed and every resolved candidate use the same exact score
  authority;
- f32 maxima are rounded upward before being used as bounds;
- ties are ordered by document id;
- natural document order skips an uncompetitive block with `continue`, never
  terminates the stream with `break`.

The Python harness has randomized exactness tests plus a fixed regression in
which a low-bound block precedes a later winning block. The C research codec
checks authority identity, block order, local-document range and uniqueness,
finite positive impacts, non-increasing impact order, truncation and
corruption.

## Scale Ladder

All promotion runs model the current product seed with
`seed_per_selected_term=64` and place all unseeded selected and unselected
postings in the residual frontier.

| Surface | Documents | Queries | Purpose |
| --- | ---: | ---: | --- |
| NFCorpus | 3,633 | 323 | all-query integrity and tail fallback |
| FiQA | 57,638 | 32 | block-size, scratch and format Pareto |
| TREC-COVID | 171,332 | 32 | medium-root generalization |
| Touche | 303,732 | 32 | larger, high-posting-count generalization |

No Elm, Shadow, PubMed, installed extension, or active root was modified.

## Exact Streaming Result

`Decoded` and `Candidates` are fractions of the full exact residual union.
`Metadata` is the mean number of query term/block directory entries. FiQA fits
within one 1,024-block batch, so document order and upper-bound order are
identical. TREC-COVID and Touche use the deployable document-order result.

| Surface | Decoded mean | Decoded p95 | Candidates mean | Candidates p95 | Metadata | Min O@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NFCorpus | 37.51% | 100.00% | 32.94% | 80.58% | 1,554 | 1.00 |
| FiQA | 25.31% | 55.07% | 27.54% | 71.82% | 39,094 | 1.00 |
| TREC-COVID | 18.27% | 29.62% | 52.57% | 74.74% | 101,618 | 1.00 |
| Touche | 5.01% | 8.56% | 9.11% | 22.24% | 181,819 | 1.00 |

On TREC-COVID, a query-global upper-bound sort improved decoded work from
18.27% to 17.40% and candidates from 52.57% to 42.41%. On Touche it improved
5.01% to 4.84% and 9.11% to 4.50%. Those gains do not justify index-sized
query-local metadata or a global sort. Natural order remains the product
candidate.

## Block-Size Pareto

FiQA used approximately equal maximum pending-document scratch for every row.
The storage estimate includes all residual postings and the compact term/block
directory, relative to the current packed semantic authority.

| Block | Batch | Decoded | Candidates | Metadata | Exact f32 bytes/current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 31.64% | 32.26% | 10,710 | 45.77% |
| 128 | 512 | 28.78% | 30.25% | 20,677 | 47.21% |
| 64 | 1,024 | 25.31% | 27.54% | 39,094 | 49.02% |
| 32 | 2,048 | 20.96% | 23.59% | 71,613 | 51.14% |
| 16 | 4,096 | 15.11% | 17.35% | 125,277 | 53.41% |
| 8 | 8,192 | 8.39% | 8.87% | 206,197 | 55.62% |
| 4 | 16,384 | 3.22% | 2.82% | 315,543 | 57.62% |

Block 4 is a useful lower bound, not a product choice. A practical directory
must expose at least one block head and its upper bound before skipping the
tail. Below block 64, directory-head work grows faster than the remaining
posting and candidate work falls. Block 64 is the balanced retained point.
Testing block 2 would optimize a component that is no longer dominant, so the
scale ladder stops here.

## Precision and Storage

The source matrices are already float32. The f32 codec is therefore exact for
the current authority. fp16 and per-term u8 were initially evaluated as the
sole scoring authority, not as conservative guidance followed by hidden f32
rescoring. Product qualification subsequently replaced per-term u8 with a
fixed global `[0, 2.0]` scale because per-term scales change when COW segments
are merged and therefore cannot preserve scores across the lifecycle.

| Surface | Current packed | f32 block64 | fp16 block64 | u8 block64 |
| --- | ---: | ---: | ---: | ---: |
| NFCorpus | 7.82 MB | 50.62% | 34.96% | 27.99% |
| FiQA | 129.49 MB | 49.03% | 32.07% | 23.69% |
| TREC-COVID | 314.84 MB | 44.97% | 29.70% | 22.11% |
| Touche | 649.05 MB | 50.23% | 32.50% | 23.65% |

| Surface | fp16 mean/min O@100 | fp16 delta NDCG@10 | u8 mean/min O@100 | u8 delta NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| NFCorpus, 323 queries | 0.99963 / 0.99 | +0.00002 | 0.99786 / 0.98 | -0.00044 |
| FiQA, 32 queries | 1.00000 / 1.00 | +0.00000 | 0.99688 / 0.99 | -0.00009 |
| TREC-COVID, 32 queries | 0.99906 / 0.99 | +0.00000 | 0.99719 / 0.99 | +0.00114 |
| Touche, 32 queries | 0.99938 / 0.99 | +0.00199 | 0.99688 / 0.98 | -0.00085 |

fp16 is the conservative approximate format. u8 is more aggressive, but its
observed error is small enough to be useful: across the four surfaces, macro
delta NDCG@10 is -0.00006, MAP@100 is -0.00023, Recall@100 is +0.00018, and
MRR@20 is +0.00010. The three broader roots have only 32 measured queries, so
these are admission signals rather than release qualification. Neither format
replaces the exact f32 default from this oracle alone.

## Usable Profile Matrix

| Profile | Authority | Size vs current packed | Observed O@100 | Intended use |
| --- | --- | ---: | ---: | --- |
| Exact | f32 block64 | 44.97%-50.62% | 1.00000 | default correctness anchor |
| Balanced | fp16 block64 | 29.70%-34.96% | mean 0.99906-1.00000, min 0.99 | conservative fast profile |
| Aggressive oracle | per-term u8 block64 | 22.11%-27.99% | mean 0.99688-0.99786, min 0.98 | representation evidence only |
| Aggressive product candidate | fixed-range u8 block64 | native lifecycle qualified | mean 0.994375-0.996997, min 0.98 | selected online candidate |

The product control is an immutable per-index reloption:

```sql
CREATE INDEX documents_search_idx
ON documents USING evoke (content)
WITH (
    sae = true,
    semantic_impact_precision = 'u8'
);
```

The available values are `f32`, `fp16`, and `u8`, with `f32` as the default.
The option belongs to the generation-contract digest, just like
`semantic_alpha_mass`; changing it requires `REINDEX`. Initial build, eventual
semantic completion, compaction, fold, accelerator publication, restart and
replication must all use the precision recorded in the root. Query execution
does not accept a precision override: it reads the root authority and cannot
mix generations.

`evoke_index_options(...)` and `evoke_index_status(...)` expose configured and
effective precision. Publication fails closed if a worker, root or forward
projection disagrees. The installed reloption is backed by the sole-authority
writer and reader; it is not an inert query-only override.

Relative to the exact block64 stream, u8 removes approximately 44.7% to 52.9%
of authority bytes. Relative to the current packed authority, it removes
approximately 72.0% to 77.9%. This is potentially valuable on large roots where
page-cache density and memory bandwidth dominate.

Quantization alone does not reduce posting count, candidate count, or the
number of high-DF blocks. Its expected physical gain comes from fewer bytes per
decoded posting and more useful blocks per page. The block64 frontier is the
independent mechanism that reduces decoded blocks and candidates. A native
implementation should combine both rather than attributing frontier work
savings to u8.

The existing `semantic_alpha_mass=0.50` profile is a third, orthogonal control:
it removes posting support and has a larger quality risk. The first native
matrix should compare f32/fp16/u8 at alpha 1.0. Only after that result should a
single 2x2 interaction test combine u8 with alpha 0.50; stacking two
approximations before their errors are separated would make regressions
unattributable.

A conservative u8 bound sidecar was also tested. It increased work by only
about 0.05 to 0.21 percentage points, but it requires a separate exact score
authority. That duplicates the representation and is rejected. If u8 is
continued, the u8 values and every forward projection must be one quantized
authority under the same root.

### Fixed-U8 Lifecycle Qualification

The first per-term implementation was rejected after a settled field-aware
`UPDATE` followed by compaction and `REINDEX` produced different scores. The
same impact had been encoded against different term-local ranges in small COW
segments and the rebuilt full term.

The retained product mapping is value-local and generation-independent:

```text
code = clamp(round(impact / 2.0 * 255), 1, 255)
impact = code / 255 * 2.0
```

Zero remains the absent-posting code. U8 rejects signed/nonpositive terms;
`fp16` and `f32` remain available for custom models that need them. With
`semantic_alpha_mass=0.50`, the fixed mapping first passed all 73 lifecycle
gates on the prior packed contract. The native block64 writer and reader then
passed the same 73/73 gate, including COW publication, compaction, fold, crash
recovery, restart, and settled score identity after `REINDEX`. Native block64
`f32` and `fp16` at alpha 1.0 also passed 73/73, so precision is selectable
without changing block geometry or lifecycle.

Relative to f32, O@100 mean/min was
`0.996997/0.98` on all 323 NFCorpus queries, `0.994375/0.98` on 32 FiQA
queries, `0.995625/0.98` on 32 TREC-COVID queries, and `0.994375/0.98` on 32
Touche queries. This qualifies the selected online representation under the
accepted minimum-overlap and lifecycle contracts. It does not by itself prove
full-root production latency: clean-package smoke and same-binary native
physical-cost evidence remain deployment gates.

### Medium Native 2x2 Qualification

A same-binary 100K-document PubMed canary separated precision from alpha
pruning. Each profile used the current block64 writer, identical documents,
20 fixed queries, and five warm runs per query.

| Profile | Index bytes | p50 | p95 | Mean/min O@10 | Mean/min O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32, alpha 1.0 | 753,836,032 | 376.3 ms | 392.2 ms | 1.000 / 1.00 | 1.000 / 1.00 |
| u8, alpha 1.0 | 721,469,440 | 379.7 ms | 396.8 ms | 0.990 / 0.90 | 0.999 / 0.99 |
| f32, alpha 0.50 | 658,800,640 | 372.9 ms | 389.3 ms | 0.815 / 0.60 | 0.857 / 0.65 |
| u8, alpha 0.50 | 640,647,168 | 370.4 ms | 387.7 ms | 0.815 / 0.60 | 0.857 / 0.65 |

At this scale, u8 reduced index bytes by 4.29% but did not yet produce a
measurable latency gain. It preserved the accepted O@100 contract. Alpha 0.50
reduced bytes by 12.61% with f32 and 15.02% with u8, but failed the retrieval
gate regardless of precision. The loss is therefore caused by support pruning,
not quantization.

The deployment decision is `u8 block64 + semantic_alpha_mass=1.0` for online
indexes. `f32` remains the exact correctness default, `fp16` remains selectable,
and alpha 0.50 remains an explicit corpus-specific experiment rather than an
online default.

The same canary exposed and closed one mutable-reader defect. During active L0,
the accelerator is intentionally bypassed and the exact packed reader is used.
That reader still validated block64 impact counts against the retired
16-document limit, causing immediate post-CRUD queries to fail on dense blocks.
The packed decoder now accepts at most 64 impacts while the legacy decoder
retains its 16-impact contract. Codec tests cover f32, fp16, and u8 at 64 and
reject 65. Existing roots need no rebuild for this reader-only correction.

## Rejected Variants

- Block maxima without within-block impact order decoded almost every posting.
- Global impact order saved postings but concentrated candidates poorly and
  remained weaker than block64 after the cap-64 seed.
- Error ratios from 1% to 5% produced only small additional savings; they are
  not worth a second approximation knob.
- Coarse 256- and 1,024-document superblock directories rejected no FiQA fine
  metadata. They increased metadata from 125,277 to 135,987 and 128,061.
- Block 4 is the oracle minimum but not the physical minimum once directory
  heads are charged.
- Query-global block sorting is rejected because it creates large per-query
  state and is not required for the retained gain.
- A hidden sidecar product route is rejected because it would preserve both
  packed BMP and the new impact stream without proving physical I/O savings.

## Lifecycle Fit

The candidate does not require encoder inference during ordinary accelerator
republication. The current writer emits block64 impact slices from the same
publication pass over quantized semantic rows, with no second corpus prepass
and no index-sized backend state. A full `REINDEX` still performs document
encoding because it creates a new semantic root.

The publication must remain under the existing root checksum, WAL, swap,
retirement, reclamation and worker lifecycle. CRUD writes remain lexical-first;
semantic completion publishes only changed rows, and accelerator maintenance
converges the derived stream. Existing forward projections remain useful for
filtered scoring and exact candidate resolution, but they must be derived from
the same authority and cannot become a second independently versioned index.

## Product Gate

The block64 writer/reader, local lifecycle, and medium 2x2 portions of the
original product gate are complete. The remaining bounded deployment gate is:

1. build one clean current-only package from the qualified source;
2. expand the selected u8/alpha-1.0 profile to one online canary root and record
   current-format publication,
   query readiness, and rollback evidence before rebuilding its peers;
4. record exact ordinals and score bits, directory bytes, decoded tails,
   resolved forward rows, physical pages and bytes, p50/p95, cancellation,
   concurrency and RSS;
5. keep f32 as the correctness anchor; admit fp16 only if mean O@100 remains at
   least 0.999 and minimum O@100 at least 0.99; admit u8 only if mean O@100
   remains at least 0.995 and minimum O@100 at least 0.98 on broader validation;
6. promote an approximate profile only if physical bytes and native p50/p95
   improve materially without lifecycle, cancellation, concurrency, or RSS
   regression.

Stop if the native reader still pages in most impact tails, if directory work
dominates at full scale, if query memory grows with all term/block entries, or
if the improvement exists only in the Python oracle. The current production
route remains unchanged until this gate passes.

## Reproduction

Primary harness:

```text
scripts/benchmark_impact_banded_residual_oracle.py
```

Focused tests:

```bash
python3 -m py_compile \
  scripts/benchmark_impact_banded_residual_oracle.py
PYTHONPATH=scripts pytest -q \
  tests/test_impact_banded_residual_oracle.py
```

Representative product-shaped command:

```bash
PYTHONPATH=scripts python3 \
  scripts/benchmark_impact_banded_residual_oracle.py \
  --dataset-root /path/to/semantic-root \
  --output /tmp/impact-banded.json \
  --query-limit 32 \
  --seed-per-selected-term 64 \
  --frontier-scope all-unseeded \
  --block-shift 6 \
  --error-ratio 0 \
  --lazy-block-batch 1024 \
  --frontier-impact-bits 32 \
  --include-document-order \
  --sole-authority-impact-bits 16 \
  --sole-authority-impact-bits 8
```

Raw outputs are retained outside the repository under:

```text
/Volumes/Betty/Tmp/evoke-impact-banded-*.json
```
