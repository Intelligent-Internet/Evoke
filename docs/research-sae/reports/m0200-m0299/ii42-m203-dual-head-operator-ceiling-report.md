# II-42 M203 Dual-Head Operator Ceiling Report

Date: 2026-09-22

## Question

M202 proved that a learned shared `32d INT4` residual head can safely preserve
most of the top256 dense-teacher gain. It did not prove that the overall route
can beat the PPLX dense baseline, because M202 used a conservative operator:
protect top20 and apply dense residual weight `0.5`.

M203 asks the next structural question:

Can the BM25 + SSR/SAE candidate source, plus a local mini-vector rerank, beat
the full-corpus PPLX dense baseline if the operator is changed but the candidate
source remains fixed?

This is a no-training ceiling replay. It uses exact dense scores as the
mini-vector oracle inside the local candidate window. Qrels are used only for
final metrics.

## Protocol

- Host: `lambda2`
- Candidate source: frozen M190/SSR latent postings plus BM25
- Dense baseline: exact full-corpus PPLX/M150 dense retrieval
- Rerank signal: exact dense score over the local candidate window
- Top-k: `100`
- Source depth: `1000`
- Swept protected heads: `0, 5, 10, 20`
- Swept correction windows: `100, 256, 512, 1000`
- Swept dense weights: `0.25, 0.5, 0.75, 1.0, 1.5`
- Output root:
  `/home/huoju/leask/runs/ii42-m203-dual-head-operator-ceiling-v1`

Added `scripts/research_sae_m203_dual_head_operator_ceiling.py`.

## Dense Baseline

Macro full7 baselines:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| Fixed BM25+SSR hybrid | 0.289159 | 0.601238 | 0.461965 | 0.560087 |
| M202 exact top256, head20, w0.5 | 0.291571 | 0.611626 | 0.461965 | 0.560087 |

The old M202 operator was recall-competitive but could not close the dense MAP,
NDCG, or MRR gap because it protected the top20 and used too little residual
pressure.

## M203A Result

Best product-compatible top256 operator:

| Protected head | Window | Weight | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 256 | 1.5 | +0.001903 | +0.003362 | +0.014209 | +0.012861 |

Full metrics for this operator:

| Route | MAP@100 | Recall@100 | NDCG@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| PPLX dense full-corpus | 0.301465 | 0.610737 | 0.485224 | 0.582561 |
| M203A exact local top256 | 0.303368 | 0.614099 | 0.484459 | 0.582064 |

The larger `window=1000` variant gives slightly more recall:

| Protected head | Window | Weight | dMAP vs dense | dRecall vs dense |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 1.5 | +0.002071 | +0.007849 |

However, `window=256` is the better first deployment target because it preserves
the fast top256 scoring contract while already beating PPLX dense on the two
primary macro metrics.

## Row Anatomy

For `head=0, window=256, weight=1.5`:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense |
| --- | ---: | ---: | ---: | ---: |
| Arguana | 0.301573 | 0.999286 | -0.003047 | -0.000714 |
| FiQA | 0.433263 | 0.800172 | -0.024385 | -0.028696 |
| NFCorpus | 0.172155 | 0.329299 | +0.004144 | +0.002335 |
| SciDocs | 0.157476 | 0.486617 | -0.005378 | -0.009217 |
| SciFact | 0.735331 | 0.983333 | +0.021708 | +0.020000 |
| TREC-COVID | 0.136032 | 0.165747 | -0.002751 | -0.001607 |
| Webis-Touche2020 | 0.187744 | 0.534238 | +0.023031 | +0.041433 |

This is not row-universal dominance over dense. It is macro dominance. FiQA and
SciDocs remain the main deficits; SciFact and Webis supply the largest wins.

## Decision

The route has a real path to beating PPLX dense, but only after changing the
operator target:

- remove top20 protection for the mini-vector stage;
- keep the correction window at top256 for the first product target;
- train/evaluate `32d INT4` against `dense_weight=1.5`;
- compare directly to PPLX dense, not only to fixed hybrid.

M204 should now train the same shared `32d INT4` residual head as M202, but with
the M203-selected operator: `protected_head=0`, `correction_window=256`, and
`dense_weight=1.5`.
