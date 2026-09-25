# II-42 M214 Row-Family Expert Mini-Vector Head Report

Date: 2026-09-24

## Question

M213C established that the fixed M210A `32d INT4` mini-vector can beat PPLX
dense under a supervised qrels-safe selector, but the same four rows remained
unsafe vs dense: Arguana, FiQA, SciDocs, and TREC-COVID. M214 asks whether the
row-family signal should move from the selector layer into the mini-vector
representation itself.

This remains a supervised diagnostic, not a qrels-free benchmark.

## Protocol

Run:
`m214a-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base initialization: M210A residual MLP checkpoint
- Product contract: `32d INT4`, score only top256 candidates
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Expert training: qrels pairwise positives vs negatives, 8 epochs
- Expert routes: `base`, `global`, `expert_or_global`, `expert_or_base`
- Selector: M213 qrels-safe mode selector with `train_onehot`
- Selector modes: `0.0, 0.5, 0.75, 1.0, 1.25, 1.5`
- Locked-test access: none

## M214A Results

M214A completed in 1463 seconds. The best dense-safe deployable route is
`global + mode_soft`: MAP 0.355242 and Recall 0.642298. This beats PPLX dense
by +0.053778 MAP and +0.031561 Recall, and it has no unsafe row vs dense under
the -0.002 row floor.

Top deployable `32d INT4` variants:

| Rank | Route | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Unsafe dense rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | global | mode soft | 0.355242 | 0.642298 | +0.053778 | +0.031561 | +0.066083 | +0.041059 | none |
| 2 | expert_or_global | mode hard | 0.352506 | 0.636926 | +0.051042 | +0.026189 | +0.063347 | +0.035688 | none |
| 3 | expert_or_base | mode soft | 0.349116 | 0.640420 | +0.047651 | +0.029683 | +0.059957 | +0.039182 | none |
| 4 | expert_or_global | mode soft | 0.348851 | 0.639435 | +0.047386 | +0.028698 | +0.059692 | +0.038196 | none |
| 5 | global | mode hard | 0.359142 | 0.638880 | +0.057677 | +0.028144 | +0.069983 | +0.037642 | FiQA |
| 6 | expert_or_global | mode confident 0.4 | 0.342447 | 0.632003 | +0.040982 | +0.021266 | +0.053288 | +0.030764 | FiQA |

Per-row anatomy for `global + mode_soft`:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Mean alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.386223 | 0.999286 | +0.081603 | -0.000714 | +0.103326 | +0.004996 | 0.704011 |
| FiQA | 0.485191 | 0.832834 | +0.027542 | +0.003966 | +0.107524 | +0.071045 | 0.841965 |
| NFCorpus | 0.227425 | 0.402454 | +0.059414 | +0.075490 | +0.070085 | +0.087390 | 1.125958 |
| SciDocs | 0.230160 | 0.558067 | +0.067307 | +0.062233 | +0.086641 | +0.100283 | 1.195224 |
| SciFact | 0.820304 | 0.990000 | +0.106681 | +0.026667 | +0.073441 | +0.010000 | 0.362304 |
| TREC-COVID | 0.156635 | 0.178864 | +0.017851 | +0.011510 | +0.036598 | +0.027194 | 1.269094 |
| Webis-Touche2020 | 0.180757 | 0.534578 | +0.016045 | +0.041773 | -0.015031 | -0.013494 | 1.099256 |

The best macro-MAP variant is `global + mode_hard`, with MAP 0.359142 and
Recall 0.638880. It is not the selected safe variant because FiQA remains unsafe
vs dense under the row floor.

## Interpretation

M214A is a large structural win over M213C. The core lesson is that the M210A
mini-vector payload size was not the immediate blocker. A qrels-pairwise
mini-vector head can create a much stronger local rerank signal while keeping
the 20-byte/doc deployed contract.

The strongest result comes from the global expert, not the per-row expert route.
That is important: row-family conditioning at selector time helped M213, but
full per-row expert routing is not yet the main source of the M214 gain. The
global qrels-pairwise fine-tune is enough to remove all dense-unsafe rows for
the selected soft mode on this supervised diagnostic.

The caveat is also large. M214A trains directly on qrels for six of the seven
eval rows, so it is not a product benchmark and not comparable to qrels-free
training. It proves endpoint capacity and the value of qrels-shaped supervision;
it does not prove that we can deploy the same gains without supervised row
labels.

## Decision

Continue with M214B seed stability. Keep the same protocol and `32d INT4`
contract, then repeat the full global/per-row expert training over multiple
seeds. If the dense-safe macro win remains stable, the next structural step is
to distill this qrels-pairwise behavior into a qrels-free or weakly-supervised
training target. If stability fails, the M214A result should be treated as an
overfit endpoint probe rather than a usable route.

## M214B Seed Stability

Run family:
`m214b-seed-<seed>-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1`

M214B repeats the M214A protocol across seeds `301`, `313`, and `337`. The large
macro win is stable, and every seed has a deployable route with no unsafe row vs
PPLX dense under the -0.002 row floor.

Best dense-safe deployable variant per seed:

| Seed | Route | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 301 | global | mode soft | 0.354864 | 0.642034 | +0.053399 | +0.031297 | +0.065704 | +0.040796 |
| 313 | global | mode hard | 0.359270 | 0.639067 | +0.057805 | +0.028330 | +0.070111 | +0.037829 |
| 337 | global | mode soft | 0.356743 | 0.641805 | +0.055278 | +0.031068 | +0.067584 | +0.040567 |
| mean | global | mixed | 0.356959 | 0.640969 | +0.055494 | +0.030232 | +0.067799 | +0.039730 |

The best macro-MAP variant is also stable and always comes from the global
expert:

| Seed | Route | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 301 | global | mode hard | 0.359798 | 0.638441 | +0.058333 | +0.027704 | Arguana, FiQA |
| 313 | global | mode hard | 0.359270 | 0.639067 | +0.057805 | +0.028330 | none |
| 337 | global | mode hard | 0.359638 | 0.639454 | +0.058173 | +0.028717 | FiQA |

M214B confirms that M214A was not a lucky seed. The per-row expert routes remain
strong, but the global qrels-pairwise mini-vector head is still the best route.
The important structural result is that the 20-byte/doc `32d INT4` payload can
support a dense-safe macro gain well above PPLX dense when the training target is
qrels-shaped rather than dense-score-shaped.

## M214B Decision

Continue to M215. The next question is no longer payload capacity; it is whether
the qrels-pairwise behavior can be simplified into a less supervised deployment
path. First test the M214 global expert with fixed, qrels-free score weights and
lightweight teacher-agreement selectors. If a fixed operator keeps the dense-safe
win, the selector can be removed from the product route. If fixed operators fail
but teacher-agreement selectors work, selector training remains the product
bottleneck. If neither works, the route remains a supervised endpoint proof and
needs a new weak-supervision objective before it can be considered deployable.
