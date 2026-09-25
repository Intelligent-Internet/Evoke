# P2.1 b1.125 Native Product Runtime Closure

## Conclusion

P2.1 closes the missing runtime boundary for the frozen P2 `b1.125` route.
Natural-language queries now run inside PostgreSQL through the existing II-42
model lifecycle:

```text
query text
  -> native Roberta byte-level BPE
  -> frozen Granite sparse ONNX compiler
  -> M1914 power transform
  -> corpus RMS calibration
  -> lexical TF plus semantic atom merge
  -> one native unified-posting generation
  -> sparse dot-product retrieval
```

This is one encoder contract and one physical inverted index. It does not add
VectorChord, ANN, an external BM25 index, or post-query score fusion.

The prior Python compiler remains a release oracle only. It is no longer a
serving dependency.

## Frozen Contract

- Parent: M1934 `b1.125`, selected without dataset-specific tuning.
- Encoder: IBM Granite 30M sparse revision
  `ad82b1fd09541c998c8d45045d601c51fdb8a9b7`.
- Semantic support: top-192 documents and top-50 queries.
- Query calibration: M1931 qrels-free `rms_m4` using corpus RMS.
- Retrieval: candidate depth 1,000 and one additive sparse dot product.
- Native generation head: 1,024 postings per atom.
- Runtime ABI: `ii42_p2_unified_text_atoms_v1`.
- ONNX Runtime: linked `1.27.1`, CPU execution provider in this closure.

The exported semantic graph is
`semantic_query_compiler.onnx`, 198,727,261 bytes, SHA-256
`12daec0053759f4bc2a4106d52ffb40d624372e7580af19c1ec7985f394c4999`.
Dataset-specific lexical vocabularies and corpus RMS values remain immutable
checkout artifacts; semantic atom IDs are offset into the same dataset atom
namespace as lexical atoms.

## Runtime Parity

Fresh Python FP32 output was compared with the native PostgreSQL runtime over
all 1,623 available product queries.

| Dataset | Queries | Exact atom IDs | Max absolute weight delta | Max relative delta |
| --- | ---: | ---: | ---: | ---: |
| nfcorpus | 323 | 323/323 | 0.0001240 | 0.01645 |
| scifact | 300 | 300/300 | 0.0000248 | 0.01849 |
| scidocs | 1,000 | 1,000/1,000 | 0.0000248 | 0.00480 |

All three parity gates pass with absolute tolerance `2e-4` and relative
tolerance `2e-2`. NFCorpus native text retrieval and retrieval from the fresh
FP32 oracle atoms produce exactly the same five aggregate metrics.

The older product matrix used BF16 GPU output. P2.1 intentionally freezes an
FP32 ONNX serving surface. Its small differences from the old BF16 matrix are
precision effects, not a compiler or index mismatch.

## Native Recall Matrix

| Dataset | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| nfcorpus | 0.347391 | 0.169076 | 0.299101 | 0.567236 | 0.588587 |
| scidocs | 0.200236 | 0.142215 | 0.467100 | 0.348686 | 0.734433 |
| scifact | 0.723722 | 0.679675 | 0.956000 | 0.688295 | 0.993333 |
| **macro** | **0.423783** | **0.330322** | **0.574067** | **0.534739** | **0.772118** |

Compared with the frozen research references packaged by M1933/M1934, macro
NDCG@10 changes by `-0.000202`, MAP@100 by `-0.000083`, Recall@100 by
`+0.000027`, MRR@20 by `-0.000981`, and CUB by `-0.000051`. The largest
single-row aggregate delta is NFCorpus MRR@20 at `-0.002165`. These deltas are
small enough to retain P2.1 as the native product candidate, but FP32 is now
the versioned serving contract and must not be mixed with old BF16 query
artifacts in strict parity tests.

## Native Efficiency

The following steady-state measurements include tokenizer, ONNX inference,
RMS calibration, atom merge, and native index retrieval. Ten warmup queries
preceded each measured dataset.

| Dataset | Docs | Queries | Postings | Payload | Build | Mean | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nfcorpus | 3,633 | 323 | 1,009,692 | 17.36 MB | 3.52 s | 7.38 ms | 8.82 ms |
| scidocs | 25,657 | 1,000 | 5,670,722 | 82.72 MB | 17.03 s | 15.48 ms | 17.50 ms |
| scifact | 5,183 | 300 | 1,342,265 | 22.64 MB | 4.46 s | 10.22 ms | 12.44 ms |
| **all** | **34,473** | **1,623** | **8,022,679** | **122.72 MB** | **25.01 s** | **12.90 ms** | **17.16 ms** |

Artifact integrity is fully SHA-256 validated on first use. The backend then
caches the validated immutable model path plus canonical manifest, avoiding a
199 MB rehash on every query. The cache is cleared with the ONNX runtime cache.
Mutating checkout files in place is unsupported; publish a new immutable
checkout or explicitly clear the cache.

The latency table is a steady-state product measurement after ten warmup
queries per dataset. It excludes the one-time immutable artifact validation
and ONNX session load, but includes tokenizer, model inference, calibration,
atom merge, generation lookup, and final ranking.

## Integrated Workflow

The release workflow is now one idempotent command:

```bash
python3 scripts/run_p2_b1125_native_workflow.py --release-validation
```

It performs, in order:

1. Reuse or export the frozen semantic ONNX runtime.
2. Export evaluable query text from the local BEIR tables.
3. Validate and reuse a ready checkout/index/generation without reading the
   build source.
4. Only when deployment is absent or `--recreate-index` is explicit, restore
   an interrupted source surface from its manifest-bound sparse NPZ files.
5. Recompute/package corpus RMS calibration, validate source identity, and
   publish one unified generation per dataset when rebuilding.
6. Query natural-language text through PostgreSQL and compute metrics.
7. Optionally regenerate the Python FP32 oracle and enforce native parity.
8. Produce `runs/p2_product_v1/matrix_native.json` and Markdown output.

The runtime status path loads compact `tokenizer_config.json`, not the full
tokenizer graph. Product evaluation uses
`ii42_model_query_text_with_ids(...)`; normal serving remains available through
the stable `ii42_query(...)` text entrypoint. The detailed model function is
retained for compatibility and evaluation diagnostics.

M1916 research source tables are intentionally `UNLOGGED`; they are build
inputs, not serving dependencies. P2.1 detects a healthy immutable generation
and skips those tables. If a generation must be rebuilt after crash recovery,
the workflow accepts only an entirely empty interrupted surface, verifies the
manifest-bound NPZ/ID artifacts, rebuilds the source, and then repeats native
identity validation before publication. A partial non-empty source is rejected
instead of being destroyed automatically.

The full release command completed after a PostgreSQL crash-recovery test:
all three deployments remained ready, all 1,623 text queries were reevaluated,
and all three fresh FP32 parity gates passed. This confirms that online query
availability depends on the persisted generation and immutable checkout, not
the transient research source relation.

The destructive branch was also exercised explicitly on NFCorpus after its
UNLOGGED source had been truncated by recovery. Manifest-driven source restore
completed in `2.9 s`, republishing the same unified generation/index completed
in `5.6 s`, and the subsequent native evaluation plus fresh parity gate passed.

## Verification

- `python -m py_compile`: all P2 workflow/compiler/evaluator scripts passed.
- `ruff check`: all P2 scripts and focused tests passed.
- Focused checkout, publisher, evaluator, workflow, and recovery tests:
  `7 passed`.
- Native C extension build completed on the macOS scalar path.
- `git diff --check` passed.
- Full isolated `installcheck` executed but retains one pre-existing platform
  fixture difference: three maintenance rows omit Linux/native-side diagnostic
  fields on the macOS scalar/backend-local path. No P2 runtime/query output
  appears in that diff; the fixture was not changed as part of P2.

## Remaining Boundary

- This closure covers the three full native artifacts currently materialized
  locally: `nfcorpus`, `scifact`, and `scidocs`. It is not a BEIR15 claim.
- The current local worker selected CPU. CoreML is visible but provider-level
  correctness and performance are not yet a release gate.
- Each dataset checkout currently carries its own 199 MB ONNX copy. A
  content-addressed immutable artifact store or hard links can remove this
  packaging duplication without changing runtime behavior.
- A future extension release should add a versioned SQL upgrade script rather
  than installing these APIs only in the base `0.1.0` schema.

Decision: retain P2.1 `b1.125` as the product-native candidate. The prior ONNX
ABI blocker is resolved. The next engineering stage is broader native index
materialization and provider/package optimization, not another scoring rewrite.
