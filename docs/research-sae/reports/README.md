# SAE / II-42 Research Report Archive

Updated: 2026-09-20

This directory is the canonical archive for research plans, experiment reports,
design notes, milestone summaries, and route-closure documents. Research
Markdown files no longer live in the repository root.

Historical files whose names contain `plan` or `roadmap` are experiment
records, not active engineering TODO authorities. Active NG protocols are
explicitly linked from [Model Planning](../../model-planning.md); they govern
research only, not production changes. Current storage and
lifecycle behavior is defined by the
[Convergent Segmented Index](../../convergent-segmented-index.md).

The bilingual model technical reports have been promoted out of this archive:
[English](../../technical-report-evoke-model.md) and
[Traditional Chinese](../../technical-report-evoke-model-zh.md). Historical evaluation
reports stay below; detailed active follow-up is in
[Model Planning](../../model-planning.md), subordinate to the product roadmap.

The archive is organized by experiment number so that each directory remains
small enough to browse on GitHub. Files without an II-42 experiment number use
the `designs/` or `milestones/` collections. The older pre-numbered SAE archive
remains directly in this directory to preserve its established links.

Machine-readable evidence is not part of the product source tree. The frozen
archive, checksums, and path mapping are documented in
[Research Artifacts](../artifacts/README.md). Small fixtures required
by tests remain under `tests/fixtures/research/`; active experiment output
belongs under the ignored `runs/` directory.

## Numbered Stages

Latest completed work: [NG87 terminal review](ng0001-ng0099/ng0087-terminal-review.zh.md).
Both clean-base, full-std arms completed their frozen 8192-update endpoints;
candidate quality improved, but document NNZ grew 5.44x/2.79x and both cost
screens failed. The [original protocol](ng0001-ng0099/ng0087-clean-base-teacher-ab.zh.md)
and [execution record](ng0001-ng0099/ng0087-progress.zh.md) remain preserved.
The next [NG88 quality-retention/cost design](ng0001-ng0099/ng0088-quality-retention-cost-plan.zh.md)
now has a frozen [first-stage protocol](ng0001-ng0099/ng0088-execution.zh.md)
and [running full-background evaluation](ng0001-ng0099/ng0088-progress.zh.md),
on Lambda2 GPU 0 only. The equal-budget training comparison remains conditional;
no optimizer updates or product qualification are implied. Earlier records
below are preserved history.

The newly authorized [NG85 student A/B protocol](ng0001-ng0099/ng0085-student-ab.zh.md)
fixes four-domain16k TRAIN supervision, matched visible text, two teachers and
two training-order seeds per arm. It is a separate bounded research decision,
not a revision of NG84's inconclusive gate. The prior-exposure and missing
explicit-negative limitations are disclosed; full-background and native-cost
qualification are not implied by training completion. Old polling stays paused.
See [NG85 execution progress](ng0001-ng0099/ng0085-progress.zh.md) for verified
tests, immutable attempts and the canonical FiQA empty-evidence quarantine.
The authorized v4 continuation reuses verified v2 preparation, dense codes and
candidate bank. Its bounded queue no longer needs four simultaneously idle
cards; FP32 length-grouped teacher batches passed score parity and the unchanged
runtime budget after v3's canary stop. Unsorted batch8 was slower and rejected.
Teacher scoring has now completed; student B stopped at numerical normalization
after 2,839 updates, and A was stopped by the matched-stage guard. The
[bounded loss diagnosis](ng0001-ng0099/ng0085-loss-diagnosis.zh.md) preserves the
attempt and separates exact replay, score-space gradients and candidate-panel
quality from full-background evaluation. No completed endpoint exists yet.
The fixed candidate panel is complete and shows degradation. Strict replay
stopped on trace drift at update 2789. Observational replay then captured a
finite, nonempty query with all 180 active coordinates outside positive-RMS
support at update 2840. This is not bit-exact reproduction or a tested correction.
The newly authorized [NG86 prospective intervention](ng0001-ng0099/ng0086-std-gradient-intervention.zh.md)
isolates std differentiation with paired starts, fixed checkpoints and a larger
candidate panel; no full-corpus or production improvement is claimed.
Its [execution record](ng0001-ng0099/ng0086-progress.zh.md) records two real GPU
workers, verified starting states and gradient parity. Both training phases
are now terminal: full-std completed the window while control captured another
zero-support failure. Full-std evaluation is complete, but control evaluation
stopped at the initial ETA gate, so the primary paired quality result remains
unavailable in the original run. Preserved checkpoints support the subsequently
authorized [evaluation-only continuation](ng0001-ng0099/ng0086-evaluation-continuation.zh.md),
which changes initialization accounting but not the time cap or scientific
protocol; no additional training is launched.
The [completed paired result](ng0001-ng0099/ng0086-evaluation-results.zh.md)
shows primary full-std/control nDCG 0.656405/0.599128, but only an uncertain
small gain over the 0.647473 base and 2.75x base document NNZ. The original
failure is preserved; this is mechanism improvement, not a dense-beating model.

The latest [NG direction review](designs/ng-research-route-review-20260914.zh.md)
reassesses the historical structural, mature-basis, supervision and cost evidence
through NG83. It withdraws another coverage audit as the main research direction
and distinguishes bounded mechanism failures from an impossibility claim. Its
proposed task-ranking comparison is now authorized as the staged
[NG84 execution protocol](ng0001-ng0099/ng0084-task-ranking-execution.zh.md).
The first stage is frozen TRAIN-only teacher admission, not new model training
or independent qualification. Full two-arm training remains conditional on the
teacher and multi-source data gates; old polling remains paused.
See [NG84 progress](ng0001-ng0099/ng0084-progress.zh.md) for frozen hashes,
resource checks, actual tracking mode and execution status. All384 admission
queries are now complete and independently reimplemented metrics agree; the
macro gain+0.007595 has a95% paired interval crossing zero. The frozen decision
is inconclusive, so no two-arm training has started. Canonical FiQA TRAIN
preparation is complete, but is not independent validation or a trained result.

The next-generation `NG-xxxx` sequence is independent of the historical M
sequence. [NG-0066 training-duration review](ng0001-ng0099/ng0066-training-duration-review.zh.md)
records three completed training-order seeds, exposed-DEV limitations,
all-positive rank harm, and the proposed data-breadth comparison. It does not
qualify a new product model or authorize scoring the locked test.

[NG-0068 supervision review](ng0001-ng0099/ng0068-supervision-and-ranking-review.zh.md)
records TRAIN-only teacher/label conflicts and blinded human-review preparation.
[NG-0069 breadth protocol](ng0001-ng0099/ng0069-matched-exposure-breadth-plan.zh.md)
fixes the next matched-query-exposure comparison; teacher preparation is not
training completion or model qualification. The
[NG-0069 execution contract](ng0001-ng0099/ng0069-execution-contract.zh.md)
records verified preparation and the bounded three-seed training/evaluation pipeline.
[NG-0069 dense evaluation diagnosis](ng0001-ng0099/ng0069-dense-evaluation-cost-review.zh.md)
records the cost-canary stop and TRAIN-only CPU profiling, not a model-quality result.
[NG-0069 final breadth review](ng0001-ng0099/ng0069-final-breadth-review.zh.md)
passes the three-seed breadth gate but records the remaining dense quality gap
and consistent representation densification. Native cost is not qualified.

[NG-0071 global-boundary ranking design](ng0001-ng0099/ng0071-global-boundary-ranking-plan.zh.md)
synthesizes the historical geometry, routing, mature-model, masking, cost and
supervision evidence into a four-arm mature-base ranking experiment. Its
configuration and tested objective reference do not constitute a launched
trainer or a qualified model. It leaves the frozen NG69 protocol unchanged.
[NG-0071 preparation review](ng0001-ng0099/ng0071-preparation-review.zh.md)
records TRAIN-only supervision coverage, provenance-only selection repair,
teacher-target saturation, real-base CPU VJP/update/reload validation and the
remaining global-snapshot/four-arm execution gates.
[NG-0071 global preflight contract](ng0001-ng0099/ng0071-global-preflight-contract.zh.md)
specifies the zero-update TRAIN-only full-corpus witness and supervision audit.
[NG-0071 final pilot review](ng0001-ng0099/ng0071-final-pilot-review.zh.md)
records the failed NQ floor despite macro improvement and lower cost proxies.
[NG-0072 cross-code diagnosis](ng0001-ng0099/ng0072-retention-diagnosis-review.zh.md)
and [NG-0073 actual-gradient replay](ng0001-ng0099/ng0073-gradient-replay-review.zh.md)
separate representation drift from direct score-gradient requests.
[NG-0074 uniform control](ng0001-ng0099/ng0074-uniform-control-review.zh.md)
does not repair NQ generalization; its completion and mirror status are explicit.
[NG-0075 parameter-update diagnosis](ng0001-ng0099/ng0075-parameter-update-diagnosis-review.zh.md)
qualifies local displacement measurements but identifies missing rival coverage.
[NG-0076 authentic midpoint](ng0001-ng0099/ng0076-authentic-midpoint-review.zh.md)
uses original checkpoints/cached documents and independent full-corpus ranking
to locate early NQ front-rank loss despite perfect Recall@100 on8 TRAIN queries.
The [NG-0077 trusted-margin plan](ng0001-ng0099/ng0077-trusted-margin-retention-plan.zh.md)
has a [completed CPU anchor and numerical review](ng0001-ng0099/ng0077-trusted-margin-preparation-review.zh.md):
45,175 anchors retain all660 positive denominators, with complete trusted
initial-top10 coverage in the existing pools. No new training run or qualified
model is implied. The [no-optimizer CUDA canary](ng0001-ng0099/ng0077-cuda-canary-protocol.zh.md)
stopped at its first document-cache value comparison, before loss/VJP or
training. The completed [batch-context diagnosis](ng0001-ng0099/ng0077-batch-context-diagnostic.zh.md)
recovers exact cached values inoriginal corpus batches and exact failed values
incandidate-pool batches. It preserves the failed gate; the separately frozen
[context-correct score/noise/VJP canary](ng0001-ng0099/ng0077-context-qualified-canary.zh.md)
has now passed all12 queries and shared-VJP fixtures, with closed/mirrored
receipts and an independent CPU raw-code/score review. The next
[matched96-update study](ng0001-ng0099/ng0077-matched-retention-pilot.zh.md)
fixes lambda0/1 and a768-query TRAIN-only endpoint before outcomes; its first
execution stage is training-only, not a new quality or native-cost result.
Both training arms have now closed. The separate
[TRAIN-only endpoint protocol](ng0001-ng0099/ng0077-endpoint-protocol.zh.md)
requires full mirrored training evidence, independent actual-loss/gradient
review, fresh768-query encodings and unchanged full-corpus quality gates.
The [NG-0077 final retention review](ng0001-ng0099/ng0077-final-retention-review.zh.md)
records completed execution but failed quality gates: K-Z sentinel nDCG
+0.001364 with an interval crossing zero, continued HotpotQA/NQ harm, and
strong pilot fitting without reliable transfer. Lower semantic NNZ/query-DF
are not a total native-cost result. The next step is a TRAIN-only boundary
coverage audit, not lambda tuning or automatic expansion of the failed pilot.
The [NG-0078 boundary-coverage protocol](ng0001-ng0099/ng0078-boundary-coverage-plan.zh.md)
fixes all768 TRAIN identities and complete endpoint heads before new reductions;
it distinguishes actual positive DCG loss from noncausal pair/coverage counts.
No new model inference, training, DEV or locked-test scoring is performed.
The [completed NG-0078 review](ng0001-ng0099/ng0078-boundary-coverage-review.zh.md)
finds that most measured NQ Pilot front-rank harm already involves trusted
in-pool pairs, while cross-query original-positive harm remains substantial.
It prioritizes controlled query-supervision breadth, not a larger candidate K
or another retention-weight sweep. Full CPU closure and Mac mirroring passed.
The [NG-0079 breadth protocol](ng0001-ng0099/ng0079-query-breadth-plan.zh.md)
fixes 384 queries repeated four times versus 1,536 distinct queries once,
with the original D objective and static initial witnesses. Both arms start
fresh from common NG3; query exposure and per-update domain sequences match,
but actual token/document work must still be measured. CPU preparation is
not a training result, and exposed TRAIN_SENTINEL is not an independent holdout.
The [NG-0079 execution contract](ng0001-ng0099/ng0079-training-execution.zh.md)
records closed witness preparation, independent scalar review and the frozen
eight-phase training graph. Both fresh96 prefixes must reproduce the historical
model and optimizer state exactly before either arm continues to384 updates.
The separate [NG79 terminal evaluation contract](ng0001-ng0099/ng0079-endpoint-protocol.zh.md)
fixes1920 TRAIN-only queries, fresh384 encodings, independent actual-loss and
endpoint review, and the unchanged79079-bootstrap/domain/cost gates.
NG79 terminal execution closed on September13; the
[post-NG79 synthesis](designs/ng-teacher-geometry-supervision-reset.zh.md)
records the failed quality/cost gate, not a qualified model.
The active [NG80 bottleneck campaign](ng0001-ng0099/ng0080-bottleneck-campaign.zh.md)
separates frozen-trunk information, continued adaptation, sparse scoring,
supervision diversity, input visibility and DF utility. Its first wave is a
TRAIN-only readout diagnostic, not completed gradient training or independent
generalization evidence. The [execution ledger](ng0001-ng0099/ng0080-execution-ledger.zh.md)
tracks closed and pending branches separately.
The [cached supervision audit](ng0001-ng0099/ng0080-supervision-audit.zh.md)
measures teacher query geometry, independent margin connectivity and input
window exposure without training or treating unjudged documents as negatives.
The [matched frozen/full-trunk control](ng0001-ng0099/ng0080-dense-control.zh.md)
fixes the same readout initialization, score-field objective and 12,288-query
exposure schedule, with real-base VJP/reload gates before training.
Both F/D training graphs closed on September13. The separate
[terminal full-background protocol](ng0001-ng0099/ng0080-dense-endpoints.zh.md)
audits all updates before fresh initial/F/D encodings and PPLX comparison on
the fixed TRAIN_DIAGNOSTIC. This dense-readout control is not complete hybrid
qualification; training closure alone is not a retrieval improvement.
The [completed dense-control review](ng0001-ng0099/ng0080-dense-control-review.zh.md)
reports D-F nDCG+0.165622 across the full background, with gains in all three
domains, but D still below PPLX. This is not a sparse/hybrid product result.
The [full S/P training protocol](ng0001-ng0099/ng0080-sparse-training.zh.md)
fixes 3,072 updates per arm and complete optimizer continuation from the original
parent, without reusing canary updates or selecting intermediate results.
Its separate [terminal training audit](ng0001-ng0099/ng0080-sparse-terminal-review.zh.md)
checks scalar gradients, full exposures and encoder/optimizer serialization;
it does not establish retrieval quality or replay every parameter update.
The [full-corpus S/P endpoint](ng0001-ng0099/ng0080-sparse-endpoints.zh.md)
separates student semantic/hybrid results from TRAIN-calibrated dense fusion,
with unchanged diagnostic identities, corpus background and serving RMS.
The [completed S/P endpoint review](ng0001-ng0099/ng0080-sparse-endpoint-review.zh.md)
reports S hybrid nDCG+0.015363 versus the original, below dense and with
HotpotQA/NQ regressions. Document NNZ grows4.02x and literal combined DF-work
7.57x; these are proxies, not native latency. Neither arm is promoted.
The [actual sparse preflight](ng0001-ng0099/ng0080-sparse-preflight.zh.md)
separately fixes TRAIN-only score scaling, original RMS and real S/P
gradient/reload checks. Its discarded canaries are not scientific training.

The proposed [NG81 overall hybrid quality/cost study](ng0001-ng0099/ng0081-overall-hybrid-quality-cost-plan.zh.md)
implements the user's forward target of overall BM25-plus-semantic superiority
to dense, without requiring every domain to win. It separates fair TRAIN-only
fusion calibration, matched geometry/boundary/cost training and joint-mask
diagnostics. Its historical review explicitly covers previous fusion successes
and failures, rather than reopening a broad scalar/RRF/gate search. The user
authorized the [fixed-endpoint NG81-A execution](ng0001-ng0099/ng0081-fusion-execution.zh.md),
which has [completed all eight phases with verified Mac mirroring](ng0001-ng0099/ng0081-fusion-progress.zh.md).
All models select lexical alpha0.1; S z-score nDCG0.787832 remains below
dense0.822175. S semantic support touches99.9992% of the corpus on average,
with no exactly universal feature. This is a completed exposed diagnostic,
not independent quality or native-cost qualification. New gradient training
remains conditional; polling remains paused and historical results are unchanged.

[NG82's Potion review and frozen diagnostics](ng0001-ng0099/ng0082-potion-review-and-execution.zh.md)
separate teacher score geometry from high-DF joint-mask utility. Its
[completed results and revised next plan](ng0001-ng0099/ng0082-progress.zh.md)
record two bounded CPU lanes,47 passing tests and26,112 independently checked
rank records. No arm passes the joint gates: utility masks underperform DF
controls despite better TRAIN joint surrogate; query-weighted256 preserves
94.2% of its quadratic objective but loses teacher ranking quality. No gradient
updates or locked-test access occurred. This is not a static-model replacement,
a new fusion sweep, or a native performance claim.

[NG83's causal protocol](ng0001-ng0099/ng0083-ranking-causality-protocol.zh.md)
has [completed same-TRAIN mask replay](ng0001-ng0099/ng0083-progress.zh.md):
the utility surrogate improves while TRAIN nDCG loses0.078509, so the failure is
not only unseen-query generalization. The
[matched score-operator study](ng0001-ng0099/ng0083-operator-execution.zh.md)
isolates relation weighting at fixed capacity and has closed training/ranking/review.
Boundary hybrid0.805811 versus field0.804562 is not significantly better;
the transfer gate fails. Both pure operators improve over initialization but
neither recovers dense quality. New source/artifacts and24,576 rank metrics
are verified; no NG83 worker remains active.
The [supervision-readiness inventory](ng0001-ng0099/ng0083-supervision-readiness.zh.md)
records547 shared positive documents and preserves the exposed diagnostic boundary.

| Stage | Documents |
| --- | ---: |
| [NG0001-NG0099](ng0001-ng0099/) | 62 |
| [M0100-M0199](m0100-m0199/) | 51 |
| [M0200-M0299](m0200-m0299/) | 19 |
| [M0300-M0399](m0300-m0399/) | 76 |
| [M0400-M0499](m0400-m0499/) | 30 |
| [M0500-M0599](m0500-m0599/) | 77 |
| [M0600-M0699](m0600-m0699/) | 236 |
| [M0700-M0799](m0700-m0799/) | 137 |
| [M0800-M0899](m0800-m0899/) | 50 |
| [M1000-M1099](m1000-m1099/) | 8 |
| [M1100-M1199](m1100-m1199/) | 191 |
| [M1200-M1299](m1200-m1299/) | 100 |
| [M1300-M1399](m1300-m1399/) | 36 |
| [M1400-M1499](m1400-m1499/) | 3 |
| [M1500-M1599](m1500-m1599/) | 74 |
| [M1600-M1699](m1600-m1699/) | 37 |
| [M1700-M1799](m1700-m1799/) | 30 |
| [M1800-M1899](m1800-m1899/) | 14 |
| [M1900-M1999](m1900-m1999/) | 93 |

Numbers absent from this table currently have no root-level research Markdown
documents to archive. New reports should be written directly to the matching
stage directory rather than added to the repository root.

## Curated Collections

- [Design documents](designs/)
- [Teacher geometry and supervision reset](designs/ng-teacher-geometry-supervision-reset.zh.md)
  revisits the post-NG79 research direction, separating task diversity,
  teacher fidelity, relevance judgments and posting cost. It is a discussion
  proposal, not an executable protocol or permission to access locked tests.
- [Historical hybrid Vector/BM25 deployment design](designs/evoke-hybrid-vector-bm25-use-case-design.md)
- [Historical online-maintenance design](designs/ii42-online-maintenance-future-plan.md)
- [Milestone and current-state summaries](milestones/)
- [M1900-M1951 learned-sparse milestone](m1900-m1999/ii42-m1900-m1951-learned-sparse-one-index-milestone.md)
- [M1934 unseen-transfer report](m1900-m1999/evoke-m1934-fixed-budget-unseen-transfer-report.md)

## Legacy Archive

The early SAE archive remains directly in this directory. It uses pre-M100,
phase, milestone, or descriptive naming and covers the original SAE, SPLADE,
block-max, candidate-budget, PostgreSQL, and unified-payload work. Keeping
those stable avoids rewriting a large historical link graph while numbered
M100-and-later reports stay in their matching stage directories.

Historical bare Markdown filenames should be interpreted relative to this
archive. Bare JSON, JSONL, checksum, checkpoint, or log filenames should be
looked up by `original_path` in the
[research artifact manifest](../artifacts/manifest.jsonl). For
numbered II-42 documents, use the experiment number to select the matching
report stage directory.

## Archive Rules

- Do not delete experiment reports when a route closes; move them here.
- Keep reports, plans, and conclusions as Markdown.
- Keep raw evidence under ignored `runs/` while active, then freeze it in the
  external artifact archive.
- Write new numbered reports directly into their stage directory.
- Update links and runner output defaults when moving a document.
- Create a new stage directory only when the first report in that range exists.
