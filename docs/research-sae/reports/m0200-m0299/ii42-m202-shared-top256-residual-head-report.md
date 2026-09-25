# II-42 M202 Shared Top256 Residual Head Report

Date: 2026-09-22

## Question

M201 showed that `32d INT4` and `64d INT4` learned residual projections can
recover most of the top256 dense-teacher gain on the full7 rows, but it trained
one row at a time. M202 asks the next structural question:

Can one shared residual head be trained across source rows, while still
exporting only the small deployable document/query payload?

This is still a projection-head probe, not a complete dual-head text encoder.
The goal is to validate the output contract before spending budget on encoder
training.

## Protocol

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Source candidates: frozen M190 latent postings + BM25
- Correction scope: baseline ranks `21..256`
- Protected head: top20
- Teacher: exact dense score inside the same top256 window
- Training supervision: dense teacher only; no qrels for training
- Metrics: qrels used only for final evaluation
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Holdout/eval row: Webis-Touche2020, plus the six source rows
- Epochs: 10
- Output dimensions: `32d` and `64d`
- Export quantization: rowwise `INT4`
- Output root:
  `/home/huoju/leask/runs/ii42-m202-shared-top256-residual-head-v1`

Added `scripts/research_sae_m202_shared_top256_residual_head.py`.

## M202A Full7 INT4 Results

Run:
`m202a-train-full6-nonwebis-eval-full7-32-64d-epoch10-gpu0-v1`

| Dataset | 32d INT4 dMAP | 32d INT4 dRecall | 64d INT4 dMAP | 64d INT4 dRecall |
| --- | ---: | ---: | ---: | ---: |
| Arguana | +0.000273 | +0.004283 | +0.000295 | +0.003569 |
| FiQA | +0.002076 | +0.015109 | +0.002462 | +0.022296 |
| NFCorpus | +0.002640 | +0.010093 | +0.002757 | +0.008709 |
| SciDocs | +0.001989 | +0.023467 | +0.002294 | +0.020217 |
| SciFact | +0.000189 | +0.003333 | +0.000100 | +0.003333 |
| TREC-COVID | +0.009757 | +0.010455 | +0.008550 | +0.009164 |
| Webis-Touche2020 | +0.000705 | +0.002155 | -0.000972 | -0.004503 |

Macro over full7:

| Variant | Bytes/doc | dMAP | dRecall | Unsafe rows |
| --- | ---: | ---: | ---: | --- |
| M200B exact top256 dense | full dense | +0.002412 | +0.010388 | Webis is negative |
| M201A 32d INT4 row-trained | 20 | +0.002161 | +0.009468 | Webis |
| M201A 64d INT4 row-trained | 36 | +0.002372 | +0.010091 | Webis |
| M202A 32d INT4 shared | 20 | +0.002518 | +0.009842 | none |
| M202A 64d INT4 shared | 36 | +0.002212 | +0.008969 | Webis |

## Interpretation

M202A is the first result that satisfies the aggressive product shape:
`score top256 + 32d INT4`, with no full7 unsafe row under the current `-0.002`
row-safety gate.

The important change versus M200 is not the dimension. It is target alignment.
The learned shared head preserves the scientific-domain rows that raw PCA lost,
while staying at the same 20 bytes/doc payload.

The 64d result is not automatically safer. It improves FiQA but regresses
Webis. Because exact dense top256 correction is itself negative on Webis, a
larger residual head can preserve a harmful teacher action more faithfully.

This supports the user's product constraint:

- training internals may use larger signals or dense teachers;
- the tracked/deployed residual payload does not need to be large;
- the next deployable candidate should remain `32d INT4`, not 256d.

## Decision

Continue the route, but do not broaden into a blind sweep.

The next step is a stability and deployment-readiness stage around the selected
design, not another general dense-tail search:

1. Repeat the exact `32d INT4` shared-head protocol across several seeds.
2. Require no unsafe full7 row and stable macro retention across seeds.
3. Keep 64d only as a diagnostic/reference, not as the primary product target.
4. If the 32d seed check passes, move to the actual dual-head encoder plan:
   emit SAE/SSR postings plus one tiny `32d INT4` residual vector, then score
   only the top256 local candidate window.
5. If seed stability fails, add an activation/no-op gate before encoder
   training, because Webis-like rows prove that dense correction can be harmful.

## Active Continuation

M202B should run on lambda2 GPU0 only, with the same source rows, holdout row,
candidate source, protected head, correction window and quantization. It should
change only the random seed and keep output dimension fixed at `32d`.

## M202B Seed Stability Results

M202B repeated the selected `32d INT4` shared-head protocol with three new
seeds: `211`, `223`, and `239`. The original M202A seed was `202`.

| Seed | Macro dMAP | Macro dRecall | Unsafe rows |
| ---: | ---: | ---: | --- |
| 202 | +0.002518 | +0.009842 | none |
| 211 | +0.002416 | +0.008216 | none |
| 223 | +0.002472 | +0.009829 | none |
| 239 | +0.002441 | +0.010196 | none |

Across the four 32d runs:

| Metric | Mean | Min | Max | Population stdev |
| --- | ---: | ---: | ---: | ---: |
| dMAP | +0.002462 | +0.002416 | +0.002518 | 0.000038 |
| dRecall | +0.009521 | +0.008216 | +0.010196 | 0.000767 |

Per-dataset mean and worst observed deltas across the four 32d runs:

| Dataset | Mean dMAP | Min dMAP | Mean dRecall | Min dRecall |
| --- | ---: | ---: | ---: | ---: |
| Arguana | +0.000285 | +0.000273 | +0.004283 | +0.004283 |
| FiQA | +0.002082 | +0.002034 | +0.015841 | +0.014356 |
| NFCorpus | +0.002539 | +0.002341 | +0.009180 | +0.006018 |
| SciDocs | +0.001961 | +0.001880 | +0.020883 | +0.018967 |
| SciFact | +0.000158 | +0.000071 | +0.003000 | +0.002667 |
| TREC-COVID | +0.009533 | +0.009386 | +0.010221 | +0.009926 |
| Webis-Touche2020 | +0.000675 | +0.000460 | +0.003239 | -0.001252 |

M202B passes the stability gate. The result is not a one-seed artifact:
`32d INT4` kept all rows above the `-0.002` unsafe threshold across four seeds,
with very low macro MAP variance.

## Next Structural Step

Move from projection-head feasibility to encoder feasibility. The next run
should not change the retrieval protocol. It should test whether the residual
head can be attached to the deployment encoder path while preserving the proven
contract:

- first-stage source remains SSR/M190 latent postings plus BM25;
- rerank only the local top256 candidate window;
- protect the top20;
- emit only a `32d INT4` residual vector for tracking/deployment;
- keep qrels out of training and use them only for final metrics.

The key acceptance question is now implementation transfer, not ranking
capacity: can the encoder emit this tiny residual head without breaking the
existing sparse-posting path or increasing tracked payload beyond 20 bytes/doc?
