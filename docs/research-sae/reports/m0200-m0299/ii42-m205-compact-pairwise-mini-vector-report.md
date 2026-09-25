# II-42 M205 Compact Pairwise Mini-Vector Report

Date: 2026-09-22

## Question

M204 showed that the M203-selected operator has enough exact signal to beat
PPLX dense, but the previous shared projection objective does not transfer that
signal cleanly into a deployable mini-vector head. M205 tests two next-step
questions:

1. Can a top-rank pairwise loss improve transfer for the aggressive
   `head=0/window256/weight1.5` operator?
2. How small can the mini-vector payload get, including `16d INT2`, before
   quality collapses?

This remains a projection-head experiment. It does not yet train the full
raw-text dual-head SSR/SAE encoder.

## Target Operator

- Candidate source: frozen M190/SSR latent postings plus BM25
- Protected head: `0`
- Correction window: `256`
- Dense residual weight: `1.5`
- Top-k: `100`
- Dense baseline: exact full-corpus PPLX/M150 retrieval

M203 exact ceiling for this operator:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M203 exact local top256 | 0.303368 | 0.614099 | 0.484459 | 0.582064 |

## M205A

Run:
`m205a-full7-16-24-32-64d-int2-int4-pairwise-epoch20-gpu0-v1`

Protocol:

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Epochs: `20`
- Dimensions: `16, 24, 32, 64`
- Quantization: `INT2` and `INT4`
- Additional objective: top-rank pairwise loss
- Pairwise config: `top_k=16`, `negative_k=64`, weight `0.2`
- Qrels: final metrics only

Payload sizes:

| Variant | Bytes/doc |
| --- | ---: |
| 16d INT2 | 8 |
| 24d INT2 | 10 |
| 32d INT2 | 12 |
| 64d INT2 | 20 |
| 16d INT4 | 12 |
| 24d INT4 | 16 |
| 32d INT4 | 20 |
| 64d INT4 | 36 |

## M205A Results

M205A completed in 2181 seconds. No variant beat PPLX dense on both macro MAP
and macro Recall.

| Variant | Bytes/doc | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16d INT2 | 8 | -0.029172 | -0.018891 | -0.016867 | -0.009392 |
| 16d INT4 | 12 | -0.011371 | -0.003096 | +0.000935 | +0.006403 |
| 24d INT2 | 10 | -0.028438 | -0.014400 | -0.016132 | -0.004901 |
| 24d INT4 | 16 | -0.006917 | -0.003558 | +0.005388 | +0.005940 |
| 32d INT2 | 12 | -0.033797 | -0.017893 | -0.021491 | -0.008394 |
| 32d INT4 | 20 | -0.005175 | -0.001811 | +0.007130 | +0.007688 |
| 64d INT2 | 20 | -0.047096 | -0.025499 | -0.034791 | -0.016000 |
| 64d INT4 | 36 | -0.002031 | +0.001346 | +0.010275 | +0.010845 |

Unsafe rows versus PPLX dense under the `-0.002` primary-metric floor:

| Variant | Unsafe rows |
| --- | --- |
| 16d INT2 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID, Webis-Touche2020 |
| 16d INT4 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| 24d INT2 | Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID |
| 24d INT4 | Arguana, FiQA, SciDocs, TREC-COVID |
| 32d INT2 | Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID |
| 32d INT4 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| 64d INT2 | Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID |
| 64d INT4 | Arguana, FiQA, SciDocs, TREC-COVID |

## Interpretation

The pairwise loss is insufficient in its current form. It did not improve over
M204B on the decisive macro MAP comparison. The best M205A variant remains
`64d INT4`: it beats PPLX dense Recall by `+0.001346`, but still misses dense
MAP by `-0.002031`.

The ultra-compact `INT2` variants are not viable under this objective. Even
`64d INT2` with the same 20 bytes/doc payload as `32d INT4` collapses badly,
which suggests rowwise `INT2` quantization is too lossy without quantization-
aware training or a different score normalization.

The useful payload frontier after M205A is:

- `16d INT2` / `24d INT2`: not viable;
- `16d INT4`: too weak but directionally better than INT2;
- `24d INT4`: possible diagnostic point, still below dense;
- `32d INT4`: still the smallest plausible target, but not solved;
- `64d INT4`: best quality, but too large for the original 20 bytes/doc product
  target and still below dense MAP.

## Decision

Do not run seed stability and do not keep sweeping dimensions. M205A supports
the M204B diagnosis: this is an objective/operator transfer bottleneck. The
next useful step needs a different target formulation, likely one of:

1. train with the exact blended score after baseline fusion, not only normalized
   dense score shape;
2. use query-level or row-level gates to avoid forcing dense correction where
   the local operator is harmful;
3. use quantization-aware training if `INT2` remains a product goal;
4. test a less aggressive exact ceiling such as `protected_head=5` only if it
   still beats PPLX dense.
