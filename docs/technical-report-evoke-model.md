# Evoke Model Technical Report (Beta 1)

- Report edition: 2026-09-03
- Evaluation snapshot: 2026-07-15
- Status: Beta 1
- Current package contract: Evoke v0.2.5 / P2.2 / ABI-v2
- Evaluation form: Native single unified posting index
- Comparisons: BM25 and PPLX dense (VectorChord)
- Frozen evaluation version: P2.1 / b1.125
- Language: English | [Traditional Chinese](technical-report-evoke-model-zh.md)

## Executive Summary

Evoke Model Beta 1 builds on the P2.1 learned sparse
retrieval route and its packaged P2.2 successor.
It does not combine BM25 and ANN results through RRF or late fusion. Instead,
an approximately 30.3M-parameter sparse encoder generates semantic postings,
while lexical and semantic postings occupy disjoint namespaces. Candidate
retrieval is then performed by one physical inverted index and one sparse dot
product.

This report preserves two native engineering evaluation surfaces measured with
the frozen P2.1 route:

1. Full BEIR15: 15/15 datasets, 46,417 queries, and 33,860,494 documents.
2. Fixed MTEB10 retrieval surface: 10/10 tasks, 8,815 queries, and 1,096,451
   documents.

The principal results are:

| Surface | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | BM25 | 0.374297 | 0.274610 | 0.562964 | 0.475641 | 0.735619 |
| BEIR15 | PPLX dense / VectorChord | **0.544873** | **0.412036** | **0.670880** | **0.650711** | 0.810525 |
| BEIR15 | P2.1 | 0.490809 | 0.371455 | 0.666885 | 0.595950 | **0.839658** |
| MTEB10 | BM25 | 0.383554 | 0.268718 | 0.595367 | 0.461805 | 0.777387 |
| MTEB10 | PPLX dense / VectorChord | **0.544127** | **0.413201** | **0.707213** | **0.629992** | 0.841063 |
| MTEB10 | P2.1 | 0.503440 | 0.374100 | 0.703125 | 0.578901 | **0.875910** |

On both surfaces, P2.1 recovers approximately 96.3% of the Recall@100 gain
that dense retrieval provides over BM25. Its absolute Recall@100 deficits
relative to dense are only 0.0040 and 0.0041, while its CUB@1000 exceeds dense
by 0.0291 and 0.0348, respectively. These results demonstrate that P2.1 is a
strong candidate generator for RAG first-stage retrieval. Building on this
recall foundation, the next quality direction is to improve NDCG, MAP, and MRR
toward dense-level head ranking.

This technical report describes Evoke Model Beta 1. It documents the model
design, measured retrieval quality, and evaluation scope. The P2.1 matrices
provide the model family's frozen quality
and latency baseline; current-P2.2 full-matrix evaluation and workload-specific
performance targets are follow-up directions. Detailed work is maintained in
[Model Planning](model-planning.md).

## 1. Report Scope and Evaluation Protocol

### 1.1 Three Engineering Systems

| Name | Engineering system | Primary role |
| --- | --- | --- |
| BM25 | Lexical inverted index | Exact-term retrieval and lexical baseline |
| PPLX dense | Stored 1,024-dimensional embeddings from `perplexity-ai/pplx-embed-v1-0.6B`, queried through VectorChord | Dense baseline |
| P2.1 | Granite sparse compiler, lexical/semantic unified postings, and one native inverted index | Frozen evaluation baseline for this release |

The dense baseline uses 1,024-dimensional PPLX vectors and VectorChord search.
The matrix measures the quality of embeddings already stored in the database
and of the VectorChord index; it does not include end-to-end PPLX encoder
inference time. This report therefore does not directly compare VectorChord
lookup latency with P2.1's combined tokenizer, ONNX, calibration, merge, and
lookup latency.

### 1.2 Metric Definitions

- `NDCG@10`: Ranking quality over the first ten results using graded relevance.
- `MAP@100`: Average precision over the first 100 results, using all positive
  documents for the query as the denominator.
- `Recall@100`: Fraction of all positive documents that enter the first 100
  results.
- `MRR@20`: Reciprocal rank of the first positive document within the first 20
  results.
- `CUB@1000`: Candidate upper bound; fraction of all positive documents that
  enter the candidate set of at most 1,000 documents.
- `Macro`: Equal-weighted mean across datasets or tasks, not a
  query-weighted mean.

Quality tables display six digits after the decimal point. Reproducibility JSON
artifacts retain full floating-point precision.

### 1.3 Release Identity and Evidence Boundary

Beta 1 retains its P2.2 engineering identity and
extension version v0.2.5. The current package is bound by
[the model lock](../packaging/milestone-model.json):

| Item | Beta 1 package contract |
| --- | --- |
| Bundle | `evoke-p2.2-nfcorpus-v2` |
| Model ID | `evoke_p2_p22_nfcorpus_v2_smoke` |
| Manifest SHA-256 | `dd0993f0638a3f683fbb6475a8509c228d8f6547aab90a0b762b1ff6fd7d0041` |
| Runtime ABI | `evoke_p2_unified_text_atoms_v2` |
| ONNX Runtime | 1.29.0, pinned by [the dependency lock](../packaging/onnxruntime.version) |

The identical frozen checkout is published as
[Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
on Hugging Face. See the [pinned download and validation instructions](examples/semantic-model-checkout.md#download-the-default-model);
publishing this distribution does not change the model identity or evidence boundary.

P2.2 retains the Granite sparse foundation, with ABI-v2 full-text query and
document compilation through deterministic windows. The packaged checkout's
lexical vocabulary and calibration are frozen from NFCorpus. Broader
cross-domain calibration is a future improvement direction. Qualified
checkouts use the same [model checkout contract](examples/semantic-model-checkout.md).

Sections 2 through 8 describe the frozen P2.1 evaluation pipeline, including
its b1.125 publication budget, ABI-v1 artifact, and historical runtime. The
table above separately identifies the current package contract.
The current engine defaults to `f32` semantic impacts and
`semantic_alpha_mass = 1.0`; `u8` and alpha 0.50 are explicit approximate
index choices, each evaluated separately from the frozen benchmark configuration.
Current storage, serving, and lifecycle semantics are defined in
[Architecture](architecture-and-design.md) and [Query Semantics](query-semantics.md).

## 2. P2.1 Architecture

```text
raw text
   |
   v
RoBERTa byte-level BPE
   |
   v
frozen Granite 30M sparse encoder (ONNX, CPU)
   |
   +--> fixed-support semantic impacts
   |       query top-50 / document top-192
   |       M1914 monotonic power calibration
   |       RMS query-local scale
   |
   +--> lexical TF/BM25-equivalent impacts
           |
           v
disjoint lexical + semantic atom namespaces
           |
           v
one unified posting map / one physical inverted index
           |
           v
one additive sparse dot-product lookup
```

The P2.1 product boundary has three important constraints:

1. There is no external ANN index.
2. BM25 and semantic rankings are not combined by RRF or post-hoc score fusion.
3. Lexical and semantic atoms occupy different namespaces, but they are
   written into one posting index and accumulated during one query.

## 3. Model and Training Design

### 3.1 Mature Sparse Foundation

P2.1 does not relearn language and retrieval geometry from scratch. It uses
`ibm-granite/granite-embedding-30m-sparse`:

- Revision: `ad82b1fd09541c998c8d45045d601c51fdb8a9b7`
- Parameter count: approximately 30.3M
- Transformer layers: 6
- Hidden / embedding size: 384
- Intermediate size: 1,536
- Vocabulary: 50,265
- Maximum sequence length: 512

The Granite sparse model uses retrieval-oriented pretraining, dense-teacher
distillation, and sparse regularization. Its design and training details are
described in [Granite Embedding Models](https://arxiv.org/abs/2502.20204).
The upstream checkpoint is licensed under Apache-2.0, but its complete training
mixture includes non-public data. Consequently, P2.1's local reproducibility
claim covers the compiler and index pipeline after fixing the upstream
checkpoint; it does not claim that the upstream foundation can be retrained
entirely from public data.

### 3.2 Sparse Impact Generation

For logit $z_{ij}$ at token position $i$ and vocabulary/latent coordinate
$j$, the sequence-level sparse impact is:

```math
w_j(x)=\max_{i\in x}\log\left(1+\mathrm{ReLU}(z_{ij})\right).
```

The semantic score is:

```math
S_{sem}(q,d)=\sum_j w_j(q)w_j(d).
```

Max pooling allows each coordinate to be published directly as an inverted
posting. It also makes corpus-wide document frequency and posting traversal
important product-cost control surfaces.

### 3.3 Upstream Training Objective

Let the teacher and student score distributions over the same candidate set
$D_q$ be:

```math
P_T(d\mid q)=\frac{\exp(s_T(q,d)/\tau_T)}
{\sum_{d'\in D_q}\exp(s_T(q,d')/\tau_T)},
```

```math
P_S(d\mid q)=\frac{\exp(s_S(q,d)/\tau_S)}
{\sum_{d'\in D_q}\exp(s_S(q,d')/\tau_S)}.
```

The score-distribution distillation loss is:

```math
\mathcal{L}_{KD}=-\sum_{d\in D_q}P_T(d\mid q)\log P_S(d\mid q).
```

Standard sparse regularization can be written as:

```math
\mathcal{L}_{FLOPS}=
\sum_j\left(\frac{1}{B}\sum_{i=1}^{B}w_j(x_i)\right)^2,
```

```math
\mathcal{L}_{NORM}=\sum_j w_j(x).
```

An abstract form of the upstream objective is:

```math
\mathcal{L}=
\mathcal{L}_{KD}
+\lambda_q\mathcal{L}^{q}_{FLOPS}
+\lambda_d\mathcal{L}^{d}_{FLOPS}
+\sigma_q\mathcal{L}^{q}_{NORM}
+\sigma_d\mathcal{L}^{d}_{NORM}.
```

KD preserves retrieval-score geometry; FLOPS constrains batch-level coordinate
usage; and NORM constrains per-sample impact mass. Together, they control
quality, posting sparsity, and inverted-index cost.

### 3.4 M1914 Fixed-Support Calibration

P2.1's local training does not change sparse-support membership. It calibrates
only impact geometry:

```math
\hat q_j=0.696368\,q_j^{1.851864},
\qquad
\hat d_j=d_j^{0.562796}.
```

Queries retain a fixed top-50, and documents retain a fixed top-192. Training
uses a fixed MS MARCO teacher surface: 10,000 training rows and 1,000
query-disjoint validation rows, with one positive and eight negatives per row.
Only three scalars are learned: global query power, document power, and score
scale. There is no dataset-specific branch.

Let $m$ be the vector of positive-negative margins. The local calibration
loss is:

```math
\mathcal{L}_{cal}=
D_{KL}(P_T\Vert P_\theta)
+0.05\,\mathrm{MSE}(m_\theta,m_T)
+10^{-3}\left[(\gamma_q-1)^2+(\gamma_d-1)^2\right].
```

Training uses AdamW for 1,000 steps with batch size 128, learning rate
`1e-2`, and seed 1914. Power values are constrained to `[0.5, 2.0]`. The
central design principle is monotonic calibration: improve the score
distribution without reselecting atoms.

### 3.5 Lexical Impacts and Single-Index Scoring

Document lexical impact uses the BM25-equivalent term contribution:

```math
\mathrm{idf}_t=
\log\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right),
```

```math
d^{L}_t=\mathrm{idf}_t
\frac{tf_{t,d}}
{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)},
```

where $k_1=1.5$ and $b=0.75$. Query lexical impact is query-term frequency.
This differs from standard BM25 only by a global $(k_1+1)$ factor that does
not affect ranking.

Lexical and semantic atoms occupy disjoint namespaces, so no cross terms
exist:

```math
S_{P2}(q,d)=
\sum_{t\in\mathcal{V}_L}q^L_td^L_t
+c(q)\sum_{j\in\mathcal{V}_S}\hat q_j\hat d_j.
```

### 3.6 Query-Local RMS Calibration

For each atom, the corpus RMS is:

```math
r_j=\sqrt{\frac{1}{N}\sum_d d_j^2}.
```

The lexical and semantic activation proxies for a query are:

```math
A_L(q)=\sum_t q^L_t r_t,
\qquad
A_S(q)=\sum_j \hat q_j r_j.
```

With the fixed global scale $g=6.281606583836263$, P2.1 uses:

```math
c(q)=\mathrm{clip}
\left(4\frac{A_L(q)}{A_S(q)},\,0.5g,\,4g\right).
```

If the proxy is invalid, the implementation falls back to $g$. Inference
does not require qrels. However, the multiplier and policy family were selected
globally using four BEIR rows under leave-one-dataset-out evaluation. The
complete P2.1 route therefore cannot be described as entirely untouched BEIR
zero-shot evaluation.

### 3.7 b1.125 Publication Budget

`b1.125` is not another neural checkpoint. It is a fixed deterministic
publisher budget:

```math
|P_{semantic}|=1.125\,|P_{lexical}|,
```

and therefore, across the two disjoint namespaces:

```math
|P_{total}|=2.125\,|P_{lexical}|.
```

It is the smallest point on the previous budget frontier that improved every
aggregate metric on the selection surface. All four LODO folds selected
b1.125, after which it passed transfer validation on the unseen SciDocs,
Quora, and TREC-COVID rows.

## 4. Model Footprint and Engineering Scale

| Item | P2.1 |
| --- | ---: |
| Sparse encoder parameters | Approximately 30.3M |
| PPLX dense comparison parameters | Nominally approximately 0.6B |
| Parameter ratio | P2.1 is approximately 1/19.8 the size of PPLX |
| ONNX artifact | 198,727,261 bytes (189.52 MiB) |
| ONNX SHA-256 | `12daec0053759f4bc2a4106d52ffb40d624372e7580af19c1ec7985f394c4999` |
| Evaluation runtime | ONNX Runtime 1.27.1 / CPU EP (historical) |
| Query support | Top-50 semantic atoms |
| Document support | Top-192 semantic atoms |
| Candidate depth | 1,000 |
| Native ABI | `evoke_p2_unified_text_atoms_v1` |

Native P2.1 runtime parity has been verified across 1,623 queries: atom IDs
match exactly, and impacts use an absolute tolerance of `2e-4` and a relative
tolerance of `2e-2`. The model artifact, tokenizer, compiler, and generation
all have fixed hashes or manifests.

The small product smoke surface contains 34,473 documents and 1,623 queries:

- 8,022,679 postings, or approximately 232.7 postings per document.
- 122.72 MB of posting payload.
- Index build time of 25.01 seconds.
- Crash recovery and immutable-artifact validation both passed.

## 5. Aggregate Results

### 5.1 P2.1 Versus BM25

| Surface | ΔNDCG@10 | ΔMAP@100 | ΔRecall@100 | ΔMRR@20 | ΔCUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | +0.116511 | +0.096845 | +0.103922 | +0.120309 | +0.104039 |
| MTEB10 | +0.119885 | +0.105382 | +0.107757 | +0.117095 | +0.098524 |

P2.1 row wins over BM25:

- BEIR15: 14/15 on all five metrics.
- MTEB10: 9/10 on NDCG and 10/10 on each of the other four metrics.

The improvements are therefore not carried by one large dataset; they are
broadly consistent at the row level.

### 5.2 P2.1 Versus PPLX Dense / VectorChord

| Surface | ΔNDCG@10 | ΔMAP@100 | ΔRecall@100 | ΔMRR@20 | ΔCUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | -0.054065 | -0.040581 | -0.003995 | -0.054761 | **+0.029133** |
| MTEB10 | -0.040688 | -0.039101 | -0.004088 | -0.051091 | **+0.034847** |

P2.1 row wins over dense:

| Surface | NDCG | MAP | Recall | MRR | CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | 3/15 | 3/15 | 11/15 | 2/15 | 12/15 |
| MTEB10 | 2/10 | 3/10 | 7/10 | 3/10 | 7/10 |

Using the dense-over-BM25 gain as the denominator, P2.1's gain-retention rates
are:

| Surface | NDCG | MAP | Recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| BEIR15 | 68.30% | 70.47% | **96.30%** | 68.72% |
| MTEB10 | 74.66% | 72.94% | **96.34%** | 69.62% |

This is the mathematical basis for P2.1's product positioning: candidate
membership nearly matches dense retrieval, while top-rank geometry has not yet
been fully recovered.

## 6. Full BEIR15 Matrix

Evaluation scale: 15 datasets, 46,417 queries, 33,860,494 documents, and
161,708 qrels.

| Dataset | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | BM25 | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 |
|  | PPLX dense | 0.429182 | 0.300000 | 0.972877 | 0.298787 | 0.972877 |
|  | **P2.1** | 0.411456 | 0.286486 | **0.987866** | 0.284977 | **1.000000** |
| Climate-FEVER | BM25 | 0.127530 | 0.097173 | 0.341694 | 0.181684 | 0.567427 |
|  | **PPLX dense** | **0.394597** | **0.314161** | **0.703507** | **0.513265** | **0.836721** |
|  | P2.1 | 0.228576 | 0.175821 | 0.530945 | 0.315448 | 0.743779 |
| CQADupStack | BM25 | 0.291987 | 0.265961 | 0.524878 | 0.295992 | 0.697639 |
|  | **PPLX dense** | **0.431176** | **0.389596** | **0.747247** | **0.425859** | 0.870515 |
|  | P2.1 | 0.418136 | 0.380379 | 0.730500 | 0.416496 | **0.888289** |
| DBPedia | BM25 | 0.239804 | 0.173662 | 0.390971 | 0.504366 | 0.580338 |
|  | PPLX dense | **0.387310** | 0.250729 | 0.458798 | **0.733080** | 0.629907 |
|  | **P2.1** | 0.353671 | **0.261121** | **0.528869** | 0.689000 | **0.733407** |
| FEVER | BM25 | 0.449661 | 0.395967 | 0.838526 | 0.410674 | 0.933235 |
|  | **PPLX dense** | **0.846450** | **0.820341** | 0.905383 | **0.862403** | 0.912997 |
|  | P2.1 | 0.766722 | 0.713127 | **0.949535** | 0.756337 | **0.967121** |
| FiQA | BM25 | 0.231073 | 0.183892 | 0.510764 | 0.292597 | 0.725380 |
|  | **PPLX dense** | **0.510599** | **0.451725** | **0.804102** | **0.599713** | **0.912474** |
|  | P2.1 | 0.358516 | 0.297809 | 0.662176 | 0.439020 | 0.855207 |
| HotpotQA | BM25 | 0.511627 | 0.430958 | 0.724646 | 0.662780 | 0.847738 |
|  | **PPLX dense** | **0.688827** | **0.619042** | 0.798447 | **0.834553** | 0.856516 |
|  | P2.1 | 0.659351 | 0.575048 | **0.816205** | 0.824656 | **0.901283** |
| MS MARCO | BM25 | 0.366717 | 0.264019 | 0.408121 | 0.743909 | 0.672591 |
|  | **PPLX dense** | **0.647114** | **0.383274** | 0.472836 | **0.961240** | 0.700965 |
|  | P2.1 | 0.524892 | 0.346804 | **0.487246** | 0.884367 | **0.784065** |
| NFCorpus | BM25 | 0.306793 | 0.137331 | 0.233236 | 0.515219 | 0.424066 |
|  | PPLX dense | 0.322832 | 0.139780 | 0.269111 | 0.548789 | 0.541961 |
|  | **P2.1** | **0.347391** | **0.169076** | **0.299101** | **0.567236** | **0.588587** |
| NQ | BM25 | 0.242799 | 0.203360 | 0.678568 | 0.215921 | 0.860926 |
|  | **PPLX dense** | **0.584749** | **0.517974** | 0.912732 | **0.541040** | 0.935762 |
|  | P2.1 | 0.481829 | 0.415296 | **0.919298** | 0.435048 | **0.980229** |
| Quora | BM25 | 0.738212 | 0.695173 | 0.947661 | 0.735647 | 0.986610 |
|  | **PPLX dense** | **0.880790** | **0.851136** | 0.984258 | **0.874570** | 0.987865 |
|  | P2.1 | 0.844995 | 0.807626 | **0.988816** | 0.838923 | **0.997925** |
| SciDocs | BM25 | 0.150534 | 0.103194 | 0.348567 | 0.277201 | 0.561417 |
|  | **PPLX dense** | **0.225314** | **0.159333** | **0.484483** | **0.385142** | **0.753817** |
|  | P2.1 | 0.200236 | 0.142215 | 0.467100 | 0.348686 | 0.734433 |
| SciFact | BM25 | 0.663931 | 0.626833 | 0.882556 | 0.634838 | 0.965000 |
|  | PPLX dense | 0.714060 | **0.685142** | 0.898222 | **0.694454** | 0.918222 |
|  | **P2.1** | **0.723722** | 0.679675 | **0.956000** | 0.688295 | **0.993333** |
| TREC-COVID | BM25 | 0.571959 | 0.066611 | 0.100074 | 0.817222 | 0.362988 |
|  | **PPLX dense** | **0.833468** | **0.131466** | 0.157343 | **0.970000** | 0.498362 |
|  | P2.1 | 0.752852 | 0.131082 | **0.161646** | 0.930000 | **0.573787** |
| Webis-Touche2020 | **BM25** | **0.377718** | **0.237831** | **0.561303** | **0.612666** | **0.858916** |
|  | PPLX dense | 0.276631 | 0.166837 | 0.493851 | 0.517772 | 0.828912 |
|  | P2.1 | 0.289785 | 0.190253 | 0.517975 | 0.520760 | 0.853424 |
| **Macro** | BM25 | 0.374297 | 0.274610 | 0.562964 | 0.475641 | 0.735619 |
|  | **PPLX dense** | **0.544873** | **0.412036** | **0.670880** | **0.650711** | 0.810525 |
|  | **P2.1** | 0.490809 | 0.371455 | 0.666885 | 0.595950 | **0.839658** |

## 7. Full MTEB10 Retrieval Matrix

MTEB10 here denotes the project's fixed set of ten English retrieval tasks. It
does not represent the complete MTEB task suite. The evaluation includes 8,815
queries, 1,096,451 documents, and 44,595 qrels.

| Task | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | BM25 | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 |
|  | **PPLX dense** | **0.450704** | **0.319743** | 0.982156 | **0.318966** | 0.982156 |
|  | P2.1 | 0.415546 | 0.290486 | **0.989293** | 0.289107 | **1.000000** |
| CQADupStack Gaming | BM25 | 0.465550 | 0.430617 | 0.746695 | 0.457166 | 0.869162 |
|  | PPLX dense | 0.554296 | 0.513284 | 0.770804 | 0.545836 | 0.813355 |
|  | **P2.1** | **0.579577** | **0.535178** | **0.866450** | **0.561797** | **0.955603** |
| CQADupStack Unix | BM25 | 0.282784 | 0.259707 | 0.543694 | 0.284534 | 0.754300 |
|  | **PPLX dense** | **0.449146** | **0.410350** | 0.767735 | **0.440482** | 0.870142 |
|  | P2.1 | 0.433789 | 0.394026 | **0.777384** | 0.428277 | **0.936138** |
| ClimateFEVER Hard Negatives | BM25 | 0.141582 | 0.107780 | 0.411267 | 0.201400 | 0.712000 |
|  | **PPLX dense** | **0.377672** | **0.298098** | **0.675817** | **0.504430** | 0.801333 |
|  | P2.1 | 0.247970 | 0.194502 | 0.585350 | 0.338304 | **0.831250** |
| FEVER Hard Negatives | BM25 | 0.501070 | 0.452277 | 0.895438 | 0.466675 | 0.975288 |
|  | **PPLX dense** | **0.854417** | **0.829734** | 0.914605 | **0.864621** | 0.921224 |
|  | P2.1 | 0.785359 | 0.737200 | **0.966279** | 0.774984 | **0.988602** |
| FiQA2018 | BM25 | 0.231073 | 0.183888 | 0.510764 | 0.292591 | 0.725380 |
|  | **PPLX dense** | **0.518076** | **0.458882** | **0.833468** | **0.603800** | **0.954942** |
|  | P2.1 | 0.354177 | 0.295080 | 0.658711 | 0.434631 | 0.859093 |
| HotpotQA Hard Negatives | BM25 | 0.549783 | 0.466817 | 0.811000 | 0.719432 | 0.933000 |
|  | PPLX dense | **0.684454** | **0.615105** | 0.832000 | 0.816141 | 0.894000 |
|  | **P2.1** | 0.676280 | 0.591675 | **0.882000** | **0.832442** | **0.962000** |
| SCIDOCS | BM25 | 0.150694 | 0.103316 | 0.348367 | 0.277795 | 0.561617 |
|  | **PPLX dense** | **0.222704** | **0.157222** | **0.483100** | **0.378192** | **0.750733** |
|  | P2.1 | 0.200611 | 0.142496 | 0.467100 | 0.349516 | 0.734633 |
| TREC-COVID | BM25 | 0.572088 | 0.066617 | 0.100137 | 0.817222 | 0.363134 |
|  | PPLX dense | 0.724736 | 0.116822 | 0.151098 | 0.871667 | 0.497221 |
|  | **P2.1** | **0.758159** | **0.129599** | **0.160195** | **0.907500** | **0.568840** |
| Touche2020 v3 | BM25 | 0.596805 | 0.378977 | 0.633422 | 0.867347 | 0.889978 |
|  | **PPLX dense** | **0.605068** | 0.412771 | 0.661348 | **0.955782** | **0.925524** |
|  | P2.1 | 0.582929 | **0.430758** | **0.678484** | 0.872449 | 0.922945 |
| **Macro** | BM25 | 0.383554 | 0.268718 | 0.595367 | 0.461805 | 0.777387 |
|  | **PPLX dense** | **0.544127** | **0.413201** | **0.707213** | **0.629992** | 0.841063 |
|  | **P2.1** | 0.503440 | 0.374100 | 0.703125 | 0.578901 | **0.875910** |

## 8. Measured Inference and Query Latency

### 8.1 Small End-to-End Native Smoke Surface

The following timings include tokenization, ONNX inference, M1914 calibration,
RMS scaling, lexical/semantic merging, and native index lookup. They exclude
one-time model loading and artifact SHA verification. Each dataset receives
ten warmup queries.

| Dataset | Queries | Mean | P95 |
| --- | ---: | ---: | ---: |
| NFCorpus | 323 | 7.38 ms | 8.82 ms |
| SciFact | 300 | 10.22 ms | 12.44 ms |
| SciDocs | 1,000 | 15.48 ms | 17.50 ms |
| **Combined** | **1,623** | **12.90 ms** | **17.16 ms** |

These frozen end-to-end CPU measurements demonstrate low-tens-of-milliseconds
execution on the small evaluation surface. They cannot be extrapolated
directly to large-corpus lookup or the current package under mixed traffic.

### 8.2 Full-Matrix Query Path

| Surface | Compile mean | Lookup mean | Total mean | Dataset-P95 mean |
| --- | ---: | ---: | ---: | ---: |
| BEIR15, dataset-equal | 9.598 ms | 125.205 ms | 134.803 ms | 196.486 ms |
| BEIR15, query-weighted | 13.058 ms | 121.169 ms | 134.227 ms | Not applicable |
| MTEB10, task-equal | 8.543 ms | 28.034 ms | 36.577 ms | 44.173 ms |
| MTEB10, query-weighted | 9.755 ms | 17.548 ms | 27.304 ms | Not applicable |

`Dataset-P95 mean` is the mean of per-dataset or per-task P95 values, not a
global P95 computed after pooling all queries. On BEIR15, NFCorpus is fastest
at 9.304 ms mean and 11.152 ms P95, while MS MARCO is slowest at 397.130 ms
mean and 471.326 ms P95. On MTEB10, SCIDOCS is fastest at 20.272 ms mean and
23.699 ms P95, while Touche is slowest at 104.653 ms mean and 117.685 ms P95.

BEIR15 P2.1 uses 131 shards, with 46,664,936,827 logical posting bytes and
34,320,714,805 stored bytes (31.96 GiB). MTEB10 uses 18 shards and
2,912,242,585 stored bytes (2.71 GiB). Lookup, rather than encoder compilation,
dominates latency on the complete surfaces. High-document-frequency (high-DF)
semantic atoms can touch many postings even when query support is small;
traversal cost is therefore a focus for scaling improvements. These
single-worker measurements provide a baseline for workload-specific profiling
of the current engine.

## 9. Release Positioning

### 9.1 Evidence Summary

1. **Candidate recall is close to dense.** On both surfaces, Recall@100 trails
   dense by only about 0.004 and recovers approximately 96.3% of dense's gain
   over BM25.
2. **The candidate-pool ceiling exceeds dense.** CUB@1000 exceeds dense by
   0.0291 on BEIR15 and 0.0348 on MTEB10, demonstrating real complementarity
   between unified lexical and semantic postings.
3. **The signal is not confined to one row.** P2.1 beats dense on Recall for
   11/15 and 7/10 rows, respectively, and improves almost every row relative
   to BM25.
4. **The engineering product form is complete.** One CPU encoder, one posting
   map, and one physical inverted index are used, with no external ANN or RRF
   dependency.
5. **Artifacts are auditable.** The model revision, ONNX SHA, ABI, atom parity,
   generation manifest, and crash-recovery evidence are fixed.
6. **The model is compact.** At 30.3M parameters, it is
   substantially smaller than the nominal 0.6B PPLX dense encoder and is a
   CPU retrieval compiler without requiring a large dense encoder.

### 9.2 Intended Use

The release is positioned as:

- A RAG first-stage candidate retriever.
- A semantic extension of BM25 in the same PostgreSQL index.
- A single-index option where deploying a large dense encoder and a separate
  ANN index is undesirable.

When the best top-10 or head-ranking quality is required, an independent
reranker may be applied after candidate retrieval. Any reranker gains must,
however, be reported separately from P2.1's single-index first-stage claim.

## 10. Evaluation Scope and Provenance

The reported matrices are primarily single-worker quality and latency
evaluations. Their configurations and measured values remain frozen, providing
a reference for subsequent model and engine improvements. The evidence covers:

1. BEIR15 uses the complete 15/15 local corpora. MTEB10 represents only this
   project's fixed retrieval surface, not all MTEB tasks.
2. Thirteen of the fifteen rows in the BEIR dense matrix reuse prior serial
   VectorChord results; ArguAna and Climate-FEVER were generated directly on
   the current surface. The final JSON labels each source row. The quality
   comparison preserves row-level provenance; completing a unified PPLX
   revision, embedding hash, and VectorChord build manifest is an evidence
   improvement in the follow-up plan. Stored-vector timings measure lookup;
   end-to-end comparisons additionally require PPLX encoder inference.
3. Missing VectorChord rows for MTEB10 have been repaired. All 10/10 rows now
   use single-worker results from that evaluation snapshot.
4. Five qrels are consistently excluded from ArguAna because their documents
   are missing from the corpus; this is not P2.1-specific handling.
5. Ten of 323 BEIR NFCorpus queries return fewer than 1,000 candidates, with a
   minimum of 467. This is candidate exhaustion, not missing queries.
6. The MTEB preparation path truncates text at 6,000 characters, while the
   evaluated P2.1 model uses a 512-token limit. Those results do not measure
   P2.2's full-text window policy; long-document evaluation is a follow-up
   coverage improvement.
7. The b1.125 and RMS policies used a limited number of BEIR rows for global
   selection. Existing unseen-transfer and MTEB evidence support
   generalization on those surfaces. Independent holdouts and broader
   calibration extend this evidence beyond the current package's
   NFCorpus-bound vocabulary and calibration.

## 11. Future Directions

Future work builds on the current candidate-recall results in three directions:

- **Ranking quality:** improve NDCG, MAP, and MRR toward dense-level head
  ranking, using the measured aggregate gap of approximately 0.04 to 0.055 to
  guide evaluation while preserving candidate recall.
- **Scale and serving efficiency:** reduce high-DF posting traversal cost and
  improve warm-query stability under concurrent reads, writes, and background
  maintenance. Establish workload-specific latency SLOs and resource profiles.
- **Generalization and evidence:** broaden cross-domain calibration,
  long-document evaluation, and current-P2.2 full-matrix coverage, with complete
  comparison provenance.

These are improvement objectives for subsequent iterations. The single-index
design remains the foundation; detailed experiments and acceptance gates are
maintained in [Model Planning](model-planning.md).

## 12. Conclusion

Evoke Model Beta 1 provides a native, single-index
first-stage retriever. In the frozen P2.1 evaluations, an approximately
30.3M-parameter CPU sparse encoder nearly preserves PPLX dense Recall@100
across two complete engineering evaluation surfaces while achieving higher
CUB@1000. Its quality improvements over BM25 are also broad and consistent.
The packaged P2.2 successor supplies the current model contract, with the
P2.1 results retained as its documented evaluation baseline.

The release combines lexical and semantic retrieval in PostgreSQL without a
separate ANN index. It establishes a compact, auditable foundation for RAG
retrieval and for the ranking, scaling, and evaluation improvements described
in the future directions above.

## 13. Reproducibility Artifacts

- [Current package model lock](../packaging/milestone-model.json)
- [P2 b1.125 productization report](research-sae/reports/m1900-m1999/evoke-p2-b1125-productization-report.md)
- [P2.1 full BEIR15 matrix](research-sae/reports/m1900-m1999/evoke-p2.1-beir15-native-full-matrix-report.md)
- [P2.1 full MTEB10 matrix](research-sae/reports/m1900-m1999/evoke-p2.1-mteb10-native-matrix-report.md)
- [MTEB10 VectorChord repair report](research-sae/reports/m1900-m1999/evoke-p2.1-mteb10-vectorchord-repair-report.md)
- [M1914 fixed-support calibration report](research-sae/reports/m1900-m1999/evoke-m1914-granite-fixed-support-calibration-report.md)
- [M1930b single-index additive closure](research-sae/reports/m1900-m1999/evoke-m1930b-one-index-additive-closure-report.md)
- [M1931 query-local calibration](research-sae/reports/m1900-m1999/evoke-m1931-query-local-source-calibration-report.md)
- [M1933 semantic budget frontier](research-sae/reports/m1900-m1999/evoke-m1933-semantic-budget-frontier-report.md)
- [M1934 unseen transfer](research-sae/reports/m1900-m1999/evoke-m1934-fixed-budget-unseen-transfer-report.md)
- BEIR15 JSON: `runs/evoke-p2-beir15-native-full-v1/evoke_p2_1_beir15_native_full_matrix.json`
- MTEB10 JSON: `runs/evoke-p2-mteb10-native-v1/evoke_p2_1_mteb10_native_matrix.json`
- ClearML M1914 task: `72d822802e104fa09a0e555c1a93533a`
