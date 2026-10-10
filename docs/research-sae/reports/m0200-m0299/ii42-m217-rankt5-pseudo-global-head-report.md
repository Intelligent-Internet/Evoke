# II-42 M217 RankT5-Pseudo Global Head Report

Date: 2026-09-26

## Question

M214/M215 proved that the product-shaped `32d INT4` top256 mini-vector route
has enough payload and that a fixed runtime score weight is sufficient once the
head is trained with qrels-pairwise supervision. M216A then showed that exact
dense-teacher pairwise supervision is not enough.

M217A tests whether a qrels-free weak teacher that is closer to relevance can
replace qrels supervision: RankT5 cross-encoder pseudo-labels scored inside the
same frozen top256 candidate window. Qrels are used only for final metrics and
teacher diagnostics.

## Protocol

Run:
`m217a-rankt5-pseudo-global-32d-int4-epoch8-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base initialization: M210A residual MLP checkpoint
- Teacher: local RankT5 model at `NG-0084/rankt5-hub`
- Candidate window: frozen top256 from the existing retrieval stack
- Training target: RankT5 pseudo-relevance pairwise ordering
- Training loss: pairwise top positives vs lower-ranked negatives, plus M210
  distillation regularization
- Product contract: `32d INT4`, 20 bytes/doc payload, score top256 candidates
- Fixed runtime weights: `0.75, 1.0, 1.25, 1.5, 2.0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Locked-test access: none

## Status

M217A completed on 2026-09-26 00:20 EDT. Runtime was about 106496 seconds. The
RankT5 cache completed all seven eval rows, ending with Webis-Touche2020
`49/49` queries and `12544` scored pairs. No RankT5 or M217 process remained
alive after completion, and GPU0 was idle.

Training used 3722 pairwise samples and completed 8 epochs. Final losses:

- Total loss: `0.170239`
- Pair loss: `0.107202`
- Distillation loss: `0.308573`

## Results

Deployable `student_32d_int4` fixed-weight variants:

| Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Beats dense | Unsafe dense rows |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.00 | 0.299895 | 0.612329 | -0.001570 | +0.001592 | no | Arguana, FiQA, SciDocs, TREC-COVID |
| 1.25 | 0.299513 | 0.611302 | -0.001952 | +0.000565 | no | Arguana, FiQA, SciDocs, TREC-COVID |
| 1.50 | 0.299279 | 0.610262 | -0.002186 | -0.000475 | no | Arguana, FiQA, SciDocs, TREC-COVID |
| 0.75 | 0.299114 | 0.611695 | -0.002351 | +0.000958 | no | Arguana, FiQA, SciDocs, TREC-COVID |
| 2.00 | 0.297210 | 0.608541 | -0.004255 | -0.002195 | no | Arguana, FiQA, SciDocs, TREC-COVID, Webis-Touche2020 |

Best deployable variant was fixed `weight=1.0`. It still missed PPLX dense MAP
by `-0.001570`, although Recall was positive by `+0.001592`.

Per-row anatomy for `student_32d_int4`, fixed `weight=1.0`:

| Dataset | dMAP vs dense | dRecall vs dense | Dense-safe |
| --- | ---: | ---: | --- |
| Arguana | -0.027719 | -0.003569 | no |
| FiQA | -0.033819 | -0.037731 | no |
| NFCorpus | +0.006310 | +0.006396 | yes |
| SciDocs | -0.008353 | -0.014083 | no |
| SciFact | +0.044170 | +0.020000 | yes |
| TREC-COVID | -0.005380 | -0.005051 | no |
| Webis-Touche2020 | +0.013805 | +0.045185 | yes |

Non-deployable RankT5 teacher ceiling variants:

| Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Beats dense | Unsafe dense rows |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.50 | 0.320176 | 0.616500 | +0.018711 | +0.005763 | yes | Arguana, FiQA, SciDocs |
| 1.25 | 0.319836 | 0.616443 | +0.018371 | +0.005706 | yes | Arguana, FiQA, SciDocs |
| 2.00 | 0.319092 | 0.614422 | +0.017627 | +0.003685 | yes | Arguana, FiQA, SciDocs |
| 1.00 | 0.318326 | 0.616400 | +0.016861 | +0.005664 | yes | Arguana, FiQA, SciDocs |
| 0.75 | 0.315630 | 0.615785 | +0.014166 | +0.005048 | yes | Arguana, FiQA, SciDocs |

The teacher has macro headroom, but it is not row-safe. The best teacher
ceiling, fixed `weight=1.5`, still loses Arguana, FiQA Recall, and SciDocs
under the dense row floor.

## Interpretation

M217A is a useful negative result, not a product candidate.

The RankT5 pseudo-teacher contains relevance signal: its ceiling beats PPLX
dense on macro MAP and Recall. But that signal is uneven. It over-improves
NFCorpus, SciFact, TREC-COVID, and Webis while still leaving Arguana, FiQA, and
SciDocs unsafe. The deployable student then loses additional quality during
distillation into the compact `32d INT4` head and cannot recover the teacher's
macro MAP gain.

This separates two issues:

- Ordinary RankT5 pseudo-labels are closer to relevance than exact dense
  labels, but still not safe enough as a direct teacher.
- The compact student adds a second bottleneck: it retains some Recall gain but
  fails to transfer the teacher's MAP headroom.

## Decision

Do not launch automatic seed stability for M217A. The pass condition was a
deployable `32d INT4` student beating PPLX dense on macro MAP and Recall with no
dense-unsafe rows. M217A misses that condition.

The next justified step should not be another blind RankT5 sweep. It should
change the weak-supervision source or filter:

- make pseudo-labels row-safe before training, especially for Arguana, FiQA, and
  SciDocs;
- use stronger or ensemble reranker supervision rather than ordinary RankT5
  alone;
- add a qrels-free safety filter that rejects pseudo-actions conflicting with
  dense/BM25 stability features;
- keep the proven product contract fixed: `32d INT4`, score top256, fixed
  runtime weight.

Until one of those changes exists, the M217A line should stop.
