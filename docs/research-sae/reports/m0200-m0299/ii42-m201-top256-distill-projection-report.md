# II-42 M201 Top256 Distill Projection Report

Date: 2026-09-22

## Question

M200B showed that a raw PCA-style `32d INT4` residual code is too weak and has
unsafe rows, while `256d INT4` is stable. M201 asks a narrower question:

Can a trained small residual projection recover the top256 dense-teacher signal
while keeping the tracked/deployed payload small?

This is a learnability probe, not a deployable encoder. It trains directly from
frozen dense embeddings to a small residual projection and evaluates only the
exported `32d INT4` and `64d INT4` document payloads.

## Protocol

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Source candidates: frozen M190 latent postings + BM25
- Correction scope: baseline ranks `21..256`
- Protected head: top20
- Teacher: exact dense score inside the same top256 window
- Training supervision: dense teacher only; no qrels for training
- Metrics: qrels used only for final evaluation
- Epochs: 20
- Output dimensions: `32d` and `64d`
- Export quantization: rowwise `INT4`
- Output root:
  `/home/huoju/leask/runs/ii42-m201-top256-distill-projection-v1`

The model is a small learned query/document projection initialized from the
M1970 shared PCA basis. It is intentionally not a full text encoder. The purpose
is to test whether the small output dimension can represent the desired
top256-local teacher signal at all.

## Full7 INT4 Results

| Dataset | Exact dMAP | Exact dRecall | 32d INT4 dMAP | 32d INT4 dRecall | 64d INT4 dMAP | 64d INT4 dRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | +0.000348 | +0.004996 | +0.000351 | +0.004283 | +0.000337 | +0.004283 |
| FiQA | +0.003060 | +0.024603 | +0.002740 | +0.024897 | +0.002926 | +0.025694 |
| NFCorpus | +0.003306 | +0.009822 | +0.003210 | +0.012529 | +0.003347 | +0.010884 |
| SciDocs | +0.002452 | +0.021550 | +0.002233 | +0.021667 | +0.002494 | +0.022233 |
| SciFact | +0.000175 | +0.003333 | +0.000031 | +0.000000 | +0.000196 | +0.000000 |
| TREC-COVID | +0.009286 | +0.010116 | +0.008875 | +0.009269 | +0.009337 | +0.010154 |
| Webis-Touche2020 | -0.001743 | -0.001706 | -0.002312 | -0.006367 | -0.002033 | -0.002613 |

Macro over full7:

| Variant | Bytes/doc | dMAP | dRecall | Unsafe rows |
| --- | ---: | ---: | ---: | --- |
| Exact top256 dense | full dense | +0.002412 | +0.010388 | Webis is negative |
| M200B 32d INT4 PCA floor | 20 | +0.000424 | +0.003080 | NFCorpus, SciFact |
| M200B 64d INT4 PCA floor | 36 | +0.000907 | +0.004858 | NFCorpus, SciFact |
| M201A 32d INT4 learned | 20 | +0.002161 | +0.009468 | Webis |
| M201A 64d INT4 learned | 36 | +0.002372 | +0.010091 | Webis |

## Interpretation

M201A changes the diagnosis:

- `32d INT4` is not intrinsically too small. With top256-local supervision it
  recovers most of the teacher gain on six of seven rows.
- The M200B failures on NFCorpus and SciFact are mainly projection/target
  alignment failures, not hard capacity failures.
- `64d INT4` is effectively at the exact top256 teacher ceiling on macro full7.
- Webis remains the hard case because the exact dense top256 teacher itself is
  negative. Training the residual head harder cannot solve a bad teacher/action
  policy; this needs a gate or no-op decision.

This supports the product constraint: train with larger internal supervision if
useful, but keep the tracked deployment payload small. The viable outputs remain:

- `32d INT4`: 20 bytes/doc, aggressive target;
- `64d INT4`: 36 bytes/doc, safer first target;
- no 256d deployment payload unless used as an offline teacher/reference.

## Decision

Continue the route. The next experiment should not be a blind dimension sweep.

M202 should train a shared residual head under the same M201 target:

1. Use 256d/exact top256 correction as teacher.
2. Export only `32d INT4` and `64d INT4` heads.
3. Train on source rows, then evaluate holdout rows without fitting to them.
4. Add a query-level correction gate, because Webis shows that dense correction
   can be harmful even when represented accurately.
5. Keep the product cost contract fixed: top256 scoring, no ANN, no output
   larger than `64d INT4` for the first deployable route.

Pass condition for the next stage:

- `64d INT4`: no row below `-0.002` on full7 and at least 80% macro MAP/Recall
  retention against exact top256;
- `32d INT4`: no scientific-domain sign failure and at least 70% macro retention;
- gate: must learn to suppress or reduce Webis-like harmful correction.
