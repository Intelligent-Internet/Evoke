# II-42 M215 Global Expert Operator Probe Report

Date: 2026-09-24

## Question

M214B proved that a qrels-pairwise global mini-vector head can produce a stable
dense-safe `32d INT4` win when paired with the M213 qrels-safe selector. M215A
asks whether that selector is still needed once the global head is strong, or
whether a fixed qrels-free score weight is enough.

The probed M214 checkpoints are qrels-supervised artifacts. M215A itself does
not train a new representation. It evaluates fixed weights and dense-teacher
selectors; qrels are used only for final metrics and oracle diagnostics.

## Protocol

Run:
`m215a-m214b-global-fixed-and-dense-selector-full7-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Checkpoints: M214B global experts from seeds `301`, `313`, and `337`
- Product contract: `32d INT4`, score only top256 candidates
- Fixed weights: `0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`
- Selector probe: dense-teacher agreement selector with `train_onehot`
- Eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- Locked-test access: none

## Results

M215A completed in 912 seconds. Fixed score weights are enough. The strongest
stable dense-safe fixed setting is `weight=1.5`, which beats PPLX dense on every
seed with no unsafe dense rows under the -0.002 row floor.

Fixed-weight means across the three M214B global checkpoints:

| Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | All seeds beat dense | All seeds dense-safe |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.25 | 0.310336 | 0.627699 | +0.008871 | +0.016962 | yes | no |
| 0.50 | 0.328063 | 0.640509 | +0.026599 | +0.029772 | yes | no |
| 0.75 | 0.342478 | 0.641945 | +0.041013 | +0.031209 | yes | yes |
| 1.00 | 0.354492 | 0.641947 | +0.053027 | +0.031210 | yes | yes |
| 1.25 | 0.364271 | 0.641426 | +0.062806 | +0.030690 | yes | yes |
| 1.50 | 0.373315 | 0.641031 | +0.071850 | +0.030294 | yes | yes |
| 2.00 | 0.388662 | 0.639981 | +0.087198 | +0.029244 | yes | no |

Best dense-safe deployable variant per checkpoint:

| Checkpoint | Control | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M214B seed 301 | fixed 1.5 | 0.372717 | 0.641495 | +0.071252 | +0.030758 | +0.083558 | +0.040256 |
| M214B seed 313 | fixed 2.0 | 0.388821 | 0.640014 | +0.087356 | +0.029277 | +0.099662 | +0.038776 |
| M214B seed 337 | fixed 2.0 | 0.389706 | 0.639397 | +0.088241 | +0.028660 | +0.100547 | +0.038159 |

Dense-teacher selectors also beat PPLX dense, but they are weaker than fixed
weights. `dense_mode_soft` is dense-safe across all three checkpoints with mean
MAP 0.344063 and Recall 0.638805. That is useful as a diagnostic but not the
best deployment path.

## Interpretation

M215A removes the selector as the immediate product bottleneck for this
supervised endpoint. Once the M214 global qrels-pairwise head exists, a fixed
weight is not only simpler but stronger than the dense-teacher selector.

The best stable product-shaped setting from this probe is `32d INT4` with fixed
`weight=1.5`: it has no dense-unsafe rows across the three checkpoints and mean
deltas vs PPLX dense of +0.071850 MAP and +0.030294 Recall.

The remaining blocker is therefore not the runtime operator. It is supervision.
M214/M215 prove that the 20-byte mini-vector route has enough capacity and that
the final scoring path can be simple. They do not yet prove a qrels-free or
portable training recipe, because the M214 heads were trained directly from
qrels.

## Decision

Continue with M216. Keep the product-side route fixed as `32d INT4`, score
top256, fixed `weight=1.5`. Replace qrels-pairwise head training with a
qrels-free dense-teacher pairwise objective initialized from M210A. If M216
keeps a meaningful dense-safe win, the route becomes much closer to deployable.
If it collapses back toward M210/M213, the major remaining challenge is finding
a weak-supervision source that carries qrels-like relevance signal without
requiring per-row qrels.
