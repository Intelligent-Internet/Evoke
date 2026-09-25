# II-42 M210 Non-Linear Mini-Vector Head Report

Date: 2026-09-23

## Question

M209B ruled out the current linear action-distillation branch. The remaining
question is whether the last-mile `32d INT4` mini-vector is fundamentally too
small, or whether the linear query/document head is too weak.

M210A changes the model class while preserving the product payload contract:
train a residual MLP head initialized from the stable PCA/linear projection, but
still export a `32d INT4` document code for score-top256 rerank. This is not a
broad sweep; it is a bottleneck test for non-linear head capacity.

## Protocol

Run:
`m210a-residual-mlp-score-shape-full7-32-64d-int4-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Architecture: residual MLP initialized from PCA basis
- Hidden size: `128`
- Residual scale: `0.25`
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Objective: stable dense score-shape distillation plus small pairwise top-rank
  regularizer
- Operators: score minmax plus correction-margin and baseline-ambiguity gates
- Qrels: final metrics only

## Decision Rule

If `32d INT4` beats PPLX dense on macro MAP and Recall, prepare seed stability.
If `32d INT4` beats M209A but still misses dense, inspect row-level failures and
continue with guarded non-linear/gated correction. If it fails to beat M209A,
treat simple residual MLP capacity as insufficient and move to a query-gated
mixture head rather than sweeping losses on this architecture.

## M210A Results

M210A completed in 2365 seconds. It did not beat PPLX dense and did not cleanly
beat the M209A `32d INT4` point. The best deployable `32d INT4` variant was
`score_minmax, head=0, window=256, w=0.75`, with MAP 0.297471 and Recall
0.611828. Relative to PPLX dense, this is MAP -0.003993 and Recall +0.001091.
Relative to fixed BM25+SSR, this is MAP +0.008312 and Recall +0.010590.

Best `32d INT4` variants:

| Rank | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | score minmax, head0, window256, w0.75 | 0.297471 | 0.611828 | -0.003993 | +0.001091 | +0.008312 | +0.010590 |
| 2 | score minmax, head0, window256, w1.0 | 0.297203 | 0.612312 | -0.004261 | +0.001575 | +0.008044 | +0.011074 |
| 3 | correction margin gate, head0, window256, w0.75, gate0.1 | 0.297099 | 0.611777 | -0.004366 | +0.001040 | +0.007940 | +0.010539 |
| 4 | correction margin gate, head0, window256, w1.0, gate0.1 | 0.296791 | 0.612337 | -0.004673 | +0.001601 | +0.007632 | +0.011099 |
| 5 | score minmax, head0, window256, w1.25 | 0.296276 | 0.612480 | -0.005189 | +0.001744 | +0.007117 | +0.011242 |

Best diagnostic variants:

| Variant | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | score minmax, head0, window256, w0.75 | 0.297471 | 0.611828 | -0.003993 | +0.001091 | +0.008312 | +0.010590 |
| 32d float32 | score minmax, head0, window256, w1.0 | 0.297714 | 0.612166 | -0.003751 | +0.001429 | +0.008554 | +0.010928 |
| 64d INT4 | score minmax, head0, window256, w1.0 | 0.298972 | 0.612274 | -0.002493 | +0.001537 | +0.009813 | +0.011035 |
| 64d float32 | score minmax, head0, window256, w1.0 | 0.300663 | 0.612764 | -0.000801 | +0.002027 | +0.011504 | +0.011526 |

Per-row anatomy for the best `32d INT4` operator:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.276159 | 0.998572 | -0.028461 | -0.001428 | -0.006738 | +0.004283 |
| FiQA | 0.411780 | 0.787667 | -0.045869 | -0.041201 | +0.034113 | +0.025878 |
| NFCorpus | 0.170113 | 0.329494 | +0.002101 | +0.002530 | +0.012772 | +0.014431 |
| SciDocs | 0.155018 | 0.483067 | -0.007835 | -0.012767 | +0.011499 | +0.025283 |
| SciFact | 0.749828 | 0.980000 | +0.036204 | +0.016667 | +0.002964 | +0.000000 |
| TREC-COVID | 0.132103 | 0.162919 | -0.006681 | -0.004435 | +0.012067 | +0.011249 |
| Webis-Touche2020 | 0.187299 | 0.541075 | +0.022587 | +0.048271 | -0.008489 | -0.006996 |

## M210A Interpretation

Residual MLP capacity helped Recall but not MAP enough to cross the dense
frontier. Compared with M209A's best `32d INT4` point, M210A is slightly lower
on MAP and higher on Recall. This is a useful shape change, but it is not a
clean win and it does not justify seed stability.

The diagnostic pattern matters: `64d float32` reaches MAP 0.300663 and Recall
0.612764, close to dense MAP and above dense Recall, while the deployable
`32d INT4` version still misses dense MAP. That means non-linearity alone is
not enough under the current single-head correction contract. The remaining
problem is selective use: the model helps NFCorpus, SciFact, and Webis, but
still hurts Arguana, FiQA, SciDocs, and TREC-COVID.

## M210A Decision

Stop this residual-MLP branch before launching another run. It did not satisfy
the decision gate. The next route should be a query-gated mixture/correction
head that learns when to apply the mini-vector signal, not another loss or
weight sweep over one unconditional head.
