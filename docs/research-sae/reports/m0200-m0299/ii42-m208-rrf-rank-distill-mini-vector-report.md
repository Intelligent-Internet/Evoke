# II-42 M208 RRF Rank-Distilled Mini-Vector Report

Date: 2026-09-23

## Question

M207 showed that the mini-vector route has a stronger exact last-mile operator
than the original M203 target. The best exact score blend is
`head=0/window=256/w=2.0`, but the more promising training target is RRF
`head=0/window=256/w=2.0/k=10` because it is rank based and does not require the
compact head to reproduce dense score magnitudes.

M208 tests whether a shared compact head trained directly on dense rank order
can transfer this RRF operator into the deployable mini-vector payload.

## Protocol

Run:
`m208a-full7-32-64d-float32-int4-rrf-rank-epoch30-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Tmux session: `ii42_m208a_rrf_rank_gpu0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Epochs: `30`
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Operator: RRF with `protected_head=0`, `correction_window=256`,
  `weight=2.0`, `k=10`
- Objective: listwise dense-rank distillation plus top-rank pairwise loss
- Qrels: final metrics only

M207 exact RRF reference:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 | dMAP vs dense | dRecall vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 | baseline | baseline |
| M207 exact RRF w2 k10 | 0.304472 | 0.615018 | 0.487441 | 0.586548 | +0.003007 | +0.004282 |

## Decision Rule

- If `32d INT4` beats PPLX dense on macro MAP and Recall, move to seed
  stability before raw-text dual-head encoder work.
- If `32d float32` beats dense but `32d INT4` does not, focus on quantization
  aware training.
- If only `64d` beats dense, report a capacity bottleneck.
- If none beat dense, RRF rank distillation is insufficient and the next route
  should be query/row-adaptive gating or a non-linear head.

## M208A Results

M208A completed in 1938 seconds. No variant beat PPLX dense on both macro MAP
and macro Recall.

| Variant | Bytes/doc | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d float32 | 128 | 0.287785 | 0.603826 | -0.013680 | -0.006910 | -0.001374 | +0.002588 |
| 32d INT4 | 20 | 0.288022 | 0.604208 | -0.013443 | -0.006529 | -0.001138 | +0.002970 |
| 64d float32 | 256 | 0.296659 | 0.610335 | -0.004806 | -0.000401 | +0.007499 | +0.009097 |
| 64d INT4 | 36 | 0.295630 | 0.609926 | -0.005835 | -0.000810 | +0.006471 | +0.008688 |

Unsafe rows versus PPLX dense under the `-0.002` primary-metric floor:

| Variant | Unsafe rows |
| --- | --- |
| 32d float32 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID, Webis-Touche2020 |
| 32d INT4 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID, Webis-Touche2020 |
| 64d float32 | Arguana, FiQA, SciDocs, TREC-COVID |
| 64d INT4 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID, Webis-Touche2020 |

## Interpretation

RRF rank distillation did not solve the mini-vector transfer problem. The exact
M207 RRF operator has strong headroom, but the learned shared linear projection
head loses too much ranking information before deployment evaluation.

This is not a simple INT4 quantization failure: `32d float32` is already worse
than `32d INT4` on macro MAP/Recall and both are far below dense. It is also
not solved by using `64d`: `64d float32` nearly matches dense Recall but still
misses dense MAP by `-0.004806`, and the `64d INT4` export loses more MAP.

The failure mode is now narrower than M204-M206. Score-shape, blended-score,
and RRF rank objectives all fail under the same global shared linear projection
head. The exact operator is viable, but the current mini-vector head class is
not expressive enough for last-mile rank correction at deployable payload.

## Decision

Stop M208 after M208A. Do not run seed stability. The next structural route
should be one of:

1. a query/row-adaptive gate that decides when the mini-vector should be allowed
   to rewrite the top ranks;
2. a non-linear or mixture head trained for last-mile sorting while still
   exporting `32d INT4`;
3. a two-stage contract where the mini-vector only selects a small promotion set
   instead of fully reordering top256.
