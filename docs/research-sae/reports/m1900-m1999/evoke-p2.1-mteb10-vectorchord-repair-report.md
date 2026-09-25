# II42 P2.1 MTEB10 VectorChord Repair

## Scope

This repair keeps the existing 1024-dimensional PPLX embeddings and changes
only the local VectorChord index and dense query path. P2.1 postings, BM25
documents, queries, and qrels were not rebuilt.

The affected matrix rows were `FiQA2018` and `TRECCOVID`. Exact cosine scans
on the stored vectors returned relevant documents near the head, proving that
the embeddings and qrel alignment were healthy. The near-zero matrix values
therefore came from the ANN index path.

## Root Cause

- `TRECCOVID`: the existing partitioned VectorChord index was stale or
  internally inconsistent. `REINDEX` restored useful ANN overlap.
- `FiQA2018`: the 115-list spherical partitioned index continued to return an
  unrelated cluster after `REINDEX`, even at maximum probes. The corpus also
  contains 38 empty-text documents with zero-norm embeddings. A partial index
  excludes cosine-undefined zero vectors, and an unpartitioned VectorChord
  layout avoids unstable partition assignment on this 57,638-document table.
- `TRECCOVID` contains one zero-norm document. Dense queries now consistently
  exclude zero-norm rows for both datasets.

The local runtime is VectorChord `1.1.1` with pgvector `0.8.2`.

## Validation

| Dataset | Repair | Sample ANN O@100 | Sample ANN O@1000 |
| --- | --- | ---: | ---: |
| FiQA2018 | partial, unpartitioned index | 0.996 | 0.998 |
| TRECCOVID | reindexed 343-list index | 0.876 | 0.735 |

The overlap reference is an exact cosine scan over the same stored vectors.
FiQA uses five queries; TRECCOVID uses five queries.

## Corrected Rows

| Dataset | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FiQA2018 | VectorChord | 0.518076 | 0.458882 | 0.833468 | 0.603800 | 0.954942 |
| TRECCOVID | VectorChord | 0.724736 | 0.116822 | 0.151098 | 0.871667 | 0.497221 |

The regenerated matrix has no dense validation warnings. Its full VectorChord
macro is NDCG@10 `0.544127`, MAP@100 `0.413201`, Recall@100 `0.707213`,
MRR@20 `0.629992`, and CUB@1000 `0.841063`.

## Regression Prevention

- Corpora below 100,000 documents use an unpartitioned VectorChord index.
- VectorChord indexes and dense queries exclude zero-norm embeddings.
- Low-CUB rows are reported as ANN/data-path validation failures rather than
  embedding-model identity failures.
- Full baseline reruns were limited to the two affected datasets before the
  ten-row matrix was regenerated.
