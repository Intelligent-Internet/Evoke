# II-42 M216 Dense-Pairwise Global Head Report

Date: 2026-09-24

## Question

M215A showed that once a qrels-pairwise M214 global mini-vector head exists, the
runtime operator can be very simple: fixed `weight=1.5` over `32d INT4` is
stable and dense-safe across seeds. M216A tests the remaining bottleneck:
whether the same product-side route can be learned without qrels by replacing
M214's qrels-pairwise training with exact-dense-teacher pairwise training.

Qrels are used only for final metrics.

## Protocol

Run:
`m216a-dense-pairwise-global-32d-int4-epoch8-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base initialization: M210A residual MLP checkpoint
- Training target: exact dense top256 teacher ordering
- Training loss: pairwise top positives vs lower-ranked dense-teacher negatives,
  plus M210 distillation regularization
- Product contract: `32d INT4`, score only top256 candidates
- Fixed weights: `0.75, 1.0, 1.25, 1.5, 2.0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Locked-test access: none

## Results

M216A completed in 599 seconds with 3722 dense-teacher pairwise training samples.
The optimizer converged cleanly: final pair loss was 0.004093 and final total
loss was 0.012517. Retrieval quality did not improve enough.

Fixed-weight variants:

| Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Unsafe dense rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.00 | 0.298184 | 0.612232 | -0.003281 | +0.001495 | +0.009025 | +0.010994 | Arguana, FiQA, SciDocs, TREC-COVID |
| 1.25 | 0.297977 | 0.611818 | -0.003488 | +0.001081 | +0.008818 | +0.010580 | Arguana, FiQA, SciDocs, TREC-COVID |
| 0.75 | 0.297974 | 0.612178 | -0.003491 | +0.001442 | +0.008815 | +0.010940 | Arguana, FiQA, SciDocs, TREC-COVID |
| 1.50 | 0.296894 | 0.611250 | -0.004571 | +0.000513 | +0.007735 | +0.010011 | Arguana, FiQA, SciDocs, TREC-COVID |
| 2.00 | 0.296092 | 0.610242 | -0.005373 | -0.000495 | +0.006932 | +0.009004 | Arguana, FiQA, SciDocs, TREC-COVID |

Per-row anatomy for the best variant, fixed `weight=1.0`:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.274758 | 0.998572 | -0.029863 | -0.001428 | -0.008140 | +0.004283 |
| FiQA | 0.419671 | 0.790314 | -0.037978 | -0.038554 | +0.042004 | +0.028525 |
| NFCorpus | 0.170933 | 0.332100 | +0.002922 | +0.005136 | +0.013592 | +0.017036 |
| SciDocs | 0.155745 | 0.481600 | -0.007108 | -0.014233 | +0.012226 | +0.023817 |
| SciFact | 0.748521 | 0.980000 | +0.034898 | +0.016667 | +0.001657 | +0.000000 |
| TREC-COVID | 0.132823 | 0.162624 | -0.005960 | -0.004730 | +0.012787 | +0.010954 |
| Webis-Touche2020 | 0.184836 | 0.540412 | +0.020124 | +0.047607 | -0.010952 | -0.007659 |

## Interpretation

M216A collapses back into the M210/M213 quality band. It beats fixed BM25+SSR
but does not beat PPLX dense on macro MAP, and it keeps the same unsafe rows
that motivated M214.

This is a strong negative control. The optimization path itself works, but the
exact dense teacher is not a sufficient supervision signal for the last-mile
relevance improvements seen in M214/M215. The large M214 gain came from
qrels-shaped relevance supervision, not merely from using a pairwise loss, a
global head, or a fixed-weight runtime operator.

## Decision

Stop this automatic line before launching more variants. The established facts
are now:

- The `32d INT4` mini-vector payload is large enough.
- The runtime operator can be simple: fixed top256 scoring is enough once the
  head is good.
- Dense-teacher training is insufficient.
- Qrels-shaped relevance supervision is currently the missing ingredient.

The next useful direction requires a new weak-supervision source that is closer
to qrels than dense similarity, such as cross-encoder/reranker pseudo-labels,
click-derived labels, LLM pairwise judgments, or task-specific positive/negative
mining. More sweeps over dense-teacher objectives are not justified by M216A.
