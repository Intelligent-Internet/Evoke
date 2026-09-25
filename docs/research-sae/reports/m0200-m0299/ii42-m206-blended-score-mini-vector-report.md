# II-42 M206 Blended-Score Mini-Vector Report

Date: 2026-09-22

## Question

After M205, the product payload target should remain fixed at `32d INT4`.
`INT2` is too lossy, and `64d INT4` is useful as a diagnostic but too large for
the intended fast tracking path. The unresolved problem is not payload choice;
it is transferring the dense-beating M203 operator into the mini-vector head.

M206 tests whether training directly against the final blended score solves the
transfer problem:

```text
final_score = baseline_score + 1.5 * minmax(mini_vector_score)
```

This is different from M202/M204/M205, which primarily matched dense residual
score shape and then applied the blend only at evaluation time.

## Full-Path Position

Current best route:

1. Use BM25 + SSR/SAE postings for candidate generation.
2. Keep local rerank window at top256.
3. Do not protect top20 for the mini-vector stage, because M203 showed that
   `protected_head=5/10/20` cannot beat PPLX dense on macro MAP.
4. Use residual blend weight `1.5`.
5. Keep deployable residual payload at `32d INT4` / 20 bytes per document.
6. Use `64d` and `float32` only as diagnostics for capacity and quantization.

M203 exact ceiling for the selected operator:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M203 exact local top256 | 0.303368 | 0.614099 | 0.484459 | 0.582064 |

## M206A

Run:
`m206a-full7-32-64d-float32-int4-blended-epoch20-gpu0-v1`

Protocol:

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Epochs: `20`
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Objective: blended final-score MSE/KL plus top-rank pairwise loss
- Pairwise config: `top_k=16`, `negative_k=64`, weight `0.2`
- Qrels: final metrics only

Decision rule:

- if `32d INT4` beats PPLX dense on macro MAP and Recall, move to seed
  stability;
- if `32d float32` beats dense but `32d INT4` does not, focus on quantization;
- if only `64d` beats dense, treat this as a capacity bottleneck;
- if none beat dense, the blended-score objective is still insufficient and the
  next move should be a gated or row-adaptive operator, not another broad
  dimension sweep.

## M206A Results

M206A completed in 1565 seconds. No variant beat PPLX dense on both macro MAP
and macro Recall.

| Variant | Bytes/doc | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d float32 | 128 | 0.294461 | 0.608449 | -0.007003 | -0.002287 | +0.005302 | +0.007211 |
| 32d INT4 | 20 | 0.293598 | 0.607395 | -0.007867 | -0.003342 | +0.004439 | +0.006157 |
| 64d float32 | 256 | 0.299086 | 0.610145 | -0.002378 | -0.000591 | +0.009927 | +0.008907 |
| 64d INT4 | 36 | 0.298491 | 0.610404 | -0.002974 | -0.000333 | +0.009332 | +0.009166 |

Unsafe rows versus PPLX dense under the `-0.002` primary-metric floor:

| Variant | Unsafe rows |
| --- | --- |
| 32d float32 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| 32d INT4 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| 64d float32 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| 64d INT4 | Arguana, FiQA, SciDocs, TREC-COVID |

## Interpretation

The blended-score target did not solve the transfer problem. It improves over
the fixed BM25+SSR hybrid, but it is worse than M204B/M205A on the decisive
macro MAP comparison and remains below PPLX dense even in the diagnostic
`64d float32` setting.

This is not primarily an INT4 quantization failure. `32d float32` is already
below dense by `-0.007003` MAP and `-0.002287` Recall, so preserving more
precision at 32 dimensions is insufficient. It is also not a simple 32d-only
capacity issue: `64d float32` still misses dense MAP by `-0.002378`, and
`64d INT4` misses dense MAP by `-0.002974`.

The result supports the earlier diagnosis that the aggressive M203 operator is
hard to distill with a global shared projection head. The next useful
structural step should be a row/query-adaptive gate or operator, not another
dimension or loss-weight sweep.

## Decision

Stop M206 after M206A. Do not run seed stability. Keep the product payload
contract at `32d INT4`, but treat the current global blended-score head as a
failed transfer objective for the M203 `head=0/window256/weight1.5` route.
