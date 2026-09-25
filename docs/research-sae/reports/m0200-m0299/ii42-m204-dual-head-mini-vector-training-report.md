# II-42 M204 Dual-Head Mini-Vector Training Report

Date: 2026-09-22

## Question

M203 found a product-compatible operator ceiling where BM25 + SSR/SAE
candidates plus exact dense local top256 rerank can beat PPLX dense on macro
MAP and Recall. M204 asks whether the deployable `32d INT4` mini-vector head can
learn that stronger operator.

This is still a projection-head training probe, not a full raw-text dual-head
encoder. The purpose is to choose the correct training target before attaching
the head to the production SSR/SAE encoder.

## Target Operator

The target is the best M203 top256-compatible operator:

- candidate source: frozen M190/SSR latent postings plus BM25;
- protected head: `0`;
- correction window: `256`;
- dense residual weight: `1.5`;
- top-k: `100`;
- output payload target: `32d INT4`, 20 bytes/doc.

M203 exact ceiling for this operator:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M203 exact local top256 | 0.303368 | 0.614099 | 0.484459 | 0.582064 |

The ceiling beats PPLX dense on macro MAP and Recall, but is still slightly
below dense on NDCG and MRR.

## M204A

Run:
`m204a-train-full6-nonwebis-eval-full7-32d-head0-window256-weight1p5-seed202-gpu0-v1`

Protocol:

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Epochs: `10`
- Dimension: `32`
- Quantization: rowwise `INT4`
- Qrels: final metrics only

Macro result:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M204A 32d INT4 | 0.297318 | 0.609497 | 0.480124 | 0.581900 |

M204A delta:

| Comparison | dMAP | dRecall | dNDCG | dMRR |
| --- | ---: | ---: | ---: | ---: |
| vs PPLX dense | -0.004147 | -0.001240 | -0.005100 | -0.000661 |
| vs fixed hybrid | +0.008159 | +0.008259 | +0.018158 | +0.021812 |

Retention against the M203 exact fixed-hybrid gain:

| Metric | Retention |
| --- | ---: |
| MAP@100 | 57.4% |
| Recall@100 | 64.2% |

Per-row M204A result:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.270349 | 0.998572 | -0.034271 | -0.001428 | -0.012549 | +0.004283 |
| FiQA | 0.416978 | 0.787496 | -0.040671 | -0.041372 | +0.039310 | +0.025707 |
| NFCorpus | 0.168157 | 0.322530 | +0.000145 | -0.004434 | +0.010816 | +0.007467 |
| SciDocs | 0.154081 | 0.479600 | -0.008773 | -0.016233 | +0.010561 | +0.021817 |
| SciFact | 0.742799 | 0.983333 | +0.029176 | +0.020000 | -0.004065 | +0.003333 |
| TREC-COVID | 0.135247 | 0.163637 | -0.003536 | -0.003717 | +0.015211 | +0.011967 |
| Webis-Touche2020 | 0.193616 | 0.531311 | +0.028904 | +0.038506 | -0.002172 | -0.016760 |

## Interpretation

M204A improves strongly over the fixed BM25+SSR hybrid, but does not beat PPLX
dense. This is not the same failure as M200. M203 already proved the operator
has enough exact signal; M204A shows that the current `32d/10epoch` shared head
does not transfer enough of that stronger signal.

The failure mode is now a transfer bottleneck:

- the exact top256 operator can beat dense;
- the learned `32d INT4` head captures only about 57% of the available MAP gain;
- removing top20 protection makes the head responsible for high-rank ordering,
  which is much harder than the old protected-tail target;
- Arguana, SciFact, and Webis become unsafe versus the fixed hybrid under the
  stronger operator.

## Active Continuation

M204B is running as a narrow diagnosis, not a broad sweep:

- same operator: `head=0`, `window=256`, `weight=1.5`;
- dimensions: `32d` and `64d`;
- epochs: `30`;
- seed: `202`;
- host/GPU: lambda2 physical GPU0.

Decision rule:

- if `64d` beats dense but `32d` does not, this is mainly an output-capacity
  bottleneck;
- if neither beats dense, this is an objective/operator transfer bottleneck;
- if `32d` beats dense, move to seed stability before raw-text encoder work.

## M204B

Run:
`m204b-train-full6-nonwebis-eval-full7-32-64d-head0-window256-weight1p5-epoch30-seed202-gpu0-v1`

Protocol changes from M204A:

- epochs increased from `10` to `30`;
- dimensions expanded to `32d` and `64d`;
- all other operator settings stayed fixed.

Macro result:

| Variant | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M204B 32d INT4 | 0.296982 | 0.610280 | 0.478263 | 0.580086 |
| M204B 64d INT4 | 0.299910 | 0.611944 | 0.479639 | 0.573266 |

Delta versus PPLX dense:

| Variant | dMAP | dRecall | dNDCG | dMRR |
| --- | ---: | ---: | ---: | ---: |
| 32d INT4 | -0.004483 | -0.000457 | -0.006961 | -0.002475 |
| 64d INT4 | -0.001555 | +0.001207 | -0.005585 | -0.009295 |

Delta versus fixed BM25+SSR hybrid:

| Variant | dMAP | dRecall | dNDCG | dMRR |
| --- | ---: | ---: | ---: | ---: |
| 32d INT4 | +0.007822 | +0.009042 | +0.016298 | +0.019999 |
| 64d INT4 | +0.010751 | +0.010706 | +0.017673 | +0.013178 |

Retention against the M203 exact fixed-hybrid gain:

| Variant | MAP retention | Recall retention |
| --- | ---: | ---: |
| 32d INT4 | 55.1% | 70.3% |
| 64d INT4 | 75.7% | 83.2% |

Unsafe rows versus PPLX dense under the `-0.002` primary-metric floor:

| Variant | Unsafe rows |
| --- | --- |
| 32d INT4 | Arguana, FiQA, SciDocs, TREC-COVID |
| 64d INT4 | Arguana, FiQA, SciDocs, TREC-COVID |

Unsafe rows versus the fixed BM25+SSR hybrid:

| Variant | Unsafe rows |
| --- | --- |
| 32d INT4 | Arguana, SciFact, Webis-Touche2020 |
| 64d INT4 | SciFact, Webis-Touche2020 |

## M204B Decision

M204B does not justify seed stability. Longer training did not rescue `32d`,
and `64d` still does not beat PPLX dense on macro MAP. `64d` does cross dense
Recall, but it remains below dense on MAP, NDCG, and MRR, with large row-level
deficits on Arguana, FiQA, SciDocs, and TREC-COVID.

This points to an objective/operator transfer bottleneck, not a simple
optimization-depth issue. Capacity helps, but not enough. The exact M203
operator is too aggressive for the current projection-head objective because
`head=0` requires the mini-vector to rewrite the highest-rank ordering, where
small score-shape errors are expensive.

The next useful step should be a structural target change, not more epochs or a
wide dimension sweep:

1. train the mini-vector against a score-calibrated or rank-pair objective that
   explicitly preserves the dense top ranks under `head=0`;
2. test an intermediate operator such as `protected_head=5` or a learned
   no-op/gate only if exact ceiling remains dense-beating;
3. keep the deployment payload target at `32d INT4`, but use `64d` as the
   capacity diagnostic until the objective transfers cleanly.
