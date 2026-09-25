# Model Planning

Status: active model follow-up plan subordinate to the
[Product Roadmap](product-roadmap.md), the current engineering planning authority.

Updated: 2026-09-20

This document owns the detailed follow-up work removed from the Beta 1 model
technical reports ([English](technical-report-ii42-model.md),
[Traditional Chinese](technical-report-ii42-model-zh.md)). The reports retain model
design, frozen evaluation results, limitations, and broad future directions.
This plan is neither a list of shipped improvements nor a new model release
contract. It does not authorize a model replacement, index rebuild, deployment,
or change to default approximation policy.

## Active NG Research

September 20: [NG87 terminal review](research-sae/reports/ng0001-ng0099/ng0087-terminal-review.zh.md)
confirms both 8192-update arms and all candidate profiles are complete.
A/PPLX improves exposed-validation hybrid nDCG from 0.670314 to 0.693741,
B/RankT5 to 0.684901, while candidate document NNZ grows 5.44x and 2.79x.
Both fail the frozen cost screen; these are not full-background results.
The requested [NG88 follow-up design](research-sae/reports/ng0001-ng0099/ng0088-quality-retention-cost-plan.zh.md)
first checks full-background transfer, then conditionally compares frozen-A
retention with PPLX continuation at the same actual output budget. Historical
compression and DF-mask failures remain explicit. Its
[first-stage execution](research-sae/reports/ng0001-ng0099/ng0088-execution.zh.md)
is now frozen and [running](research-sae/reports/ng0001-ng0099/ng0088-progress.zh.md)
sequentially on Lambda2 physical GPU 0 only. No new optimizer updates,
locked-test access, deployment or polling have been started. Earlier launch
records below are history; their GPU assignments do not override GPU 0 only.

September 18 follow-up: the user authorized
[NG87 clean-base full-std teacher A/B](research-sae/reports/ng0001-ng0099/ng0087-clean-base-teacher-ab.zh.md).
Both arms restart from the mature base with empty optimizer state, reuse sealed
NG85 inputs/targets, and train two complete epochs with the corrected derivative.
One matched order seed is admitted first. The terminal update 8192 is fixed;
all 2048 exposed validation queries are evaluated, with earlier checkpoints
restricted to a nested monitoring panel. No alpha/cost sweep, failure retry,
second seed, full-corpus inference or product change is automatically launched.
See [NG87 execution](research-sae/reports/ng0001-ng0099/ng0087-progress.zh.md).
The earlier NG85/86 records below remain historical, not current launch instructions.

September 18: [NG86 prospective std-gradient intervention](research-sae/reports/ng0001-ng0099/ng0086-std-gradient-intervention.zh.md)
is now authorized as a separate, bounded two-arm protocol. NG85 observational
replay completed with a finite, nonempty query activating only zero-RMS
coordinates at update 2840. Its drift prevents a bit-exact replay claim; the old
intervention gate is not relaxed. NG86 freezes the same B/2048 model/optimizer,
RankT5 targets, order and readout in both arms; only student-std differentiation
changes. Fixed checkpoints 2560/2816/3072 and a 640-query candidate panel
(128 fit, 512 exposed validation) separate ranking, support and stability.
Step 2816 is the primary matched comparison; no endpoint or automatic scale-up
is claimed before execution and sealed review.
The [NG86 execution record](research-sae/reports/ng0001-ng0099/ng0086-progress.zh.md)
now records both terminal training phases: the control captured the same kind
of finite zero-support failure at update 2847, whereas full-std completed 3072.
Full-std's three 640-query profiles finished, but the control profile stopped
at its conservative ETA gate after 16 base/fit rows. Its 2560/2816 checkpoints
exist; the primary paired ranking comparison and 512-query base/start anchors
were initially missing. Same-order training telemetry shows substantially less support
and scale drift; historical 64-query overlap shows partial ranking recovery,
not restoration beyond the base or dense. Next recommended work is a separate
bounded read-only evaluation continuation, not retraining, changing the frozen
attempt or selecting the best checkpoint after seeing results. The user has
now authorized the [evaluation-only continuation](research-sae/reports/ng0001-ng0099/ng0086-evaluation-continuation.zh.md).
It runs in a new immutable unit, retains the 1800-second worker cap, separates
model-load and per-row timing, and checks raw-score parity against existing
outputs before completing the original panel. The [completed paired review](research-sae/reports/ng0001-ng0099/ng0086-evaluation-results.zh.md)
now reports primary-step nDCG 0.656405 versus control 0.599128, a descriptive
paired delta +0.057277 with interval [+0.037804,+0.076601]. The base is 0.647473;
the smaller gain over base has an interval crossing zero, and document NNZ is
still 2.75x base. Dense's matched-candidate 0.745786 remains higher. The next
recommended design returns to clean-base PPLX/RankT5 training with the corrected
derivative shared by both arms, rather than another rescue of the damaged
checkpoint or alpha/cost sweep. It requires a new bounded protocol; no training
or second seed was started. Recurring polling remains paused.

September16 update: the user has separately authorized
[NG85 fixed-budget student A/B](research-sae/reports/ng0001-ng0099/ng0085-student-ab.zh.md).
This does not revise NG84's inconclusive gate. NG85 freezes16,384 four-domain
TRAIN queries, identical student-visible teacher inputs and candidate banks,
two complete epochs and two order seeds per arm. The objective compares PPLX
and RankT5 soft supervision of the same complete hybrid score. Explicit judged
negatives are absent; reliable-negative loss and cost regularization are both
zero in this quality contrast, not silently fabricated or coefficient-swept.
Validation remains previously exposed research data; locked test and production
remain untouched. The bounded campaign stops after training, before independent
full-background endpoint review and native-cost qualification. Old polling stays
paused. The NG84 history below is retained rather than retroactively rewritten.
The [NG85 execution record](research-sae/reports/ng0001-ng0099/ng0085-progress.zh.md)
records the first data-integrity stop and the new immutable attempt. FiQA empty
evidence is quarantined before model scoring, not relabeled or silently dropped
from an already evaluated score table.
After v2 stopped at GPU admission, the user authorized the v3 resource queue:
verified prepare/dense/bank outputs are reused without scientific changes.
Independent jobs use available single GPUs, at most two concurrently; pre-start
resource contention waits, while real computation failures still stop the run.
The original teacher/student budgets and endpoint gates remain unchanged.
The first real teacher canary then exceeded its frozen time budget. The v4
continuation additionally uses FP32 length-grouped teacher batches, admitted
only after raw-score parity and a 22.58% canary time reduction. Unsorted batch8
was slower and rejected. No model/loss/data/exposure or budget gate was relaxed;
teacher scoring subsequently completed all four shards (993,661 pairs).
Student B/85001 stopped after update 2,839 at query RMS normalization;
A/85001 was stopped by the matched-stage guard. No endpoint is complete.
The authorized [NG85 bounded loss diagnosis](research-sae/reports/ng0001-ng0099/ng0085-loss-diagnosis.zh.md)
replays B from its exact 2,048-step checkpoint and profiles five predetermined
checkpoints on a fixed, previously exposed candidate panel. It changes neither
the frozen objective nor the original attempt. Zero-support RMS dimensions,
scale-gradient interactions and actual ranking changes must be distinguished
before any new loss treatment or larger campaign is admitted.
The candidate panel completed and shows quality/support drift, not improvement.
Strict replay matched 740 updates exactly in its recorded statistics, then
stopped at update 2789 on a small gradient-norm mismatch before the original
failure. Its failed receipt is preserved. The separate observational replay
subsequently captured the same-step zero-support failure without changing the
objective. Its complete artifacts are verified; the new NG86 intervention is
not a relabeling of that non-bit-exact run as an exact control.
Matched-context teacher ablation is not the product dense baseline: endpoint
review must also compare normal original-text model inputs and full-text BM25,
without handicapping dense through the shared-prefix preparation.

Latest direction review, September14:
[NG research route review](research-sae/reports/designs/ng-research-route-review-20260914.zh.md).
The user first requested a historical reassessment, then explicitly authorized
the next exploration. [NG84 execution protocol](research-sae/reports/ng0001-ng0099/ng0084-task-ranking-execution.zh.md)
now freezes a bounded TRAIN-only ranking-teacher admission on Lambda2. The
matched training arms remain conditional on teacher signal and a frozen
multi-source data/training manifest; preparation is not a trained model result.
The [NG84 terminal review](research-sae/reports/ng0001-ng0099/ng0084-progress.zh.md)
has now closed all384 TRAIN-panel queries and replayed raw scores with a second
metric implementation. RankT5 macro nDCG0.825959 versus PPLX0.818364 gives
delta+0.007595, but the paired95% interval[-0.011516,+0.026883] crosses zero.
The frozen decision is `inconclusive_do_not_auto_launch`; no two-arm training,
teacher sweep, or adaptive sample extension was launched. HotpotQA/NQ gains
and FEVER losses are disclosed, not used for post-hoc teacher routing.
Canonical FiQA TRAIN preparation has separately completed:5,500 queries,
57,638 documents and14,166 known-positive memberships, with no test-qrel access
or RLHN relabels promoted to human truth. Historical exposure, matched-context
targets and independent validation are still pending. No NG84 GPU job remains
active; earlier polling stays paused. This stops at the admission decision,
not at a completed sparse-training or product-quality conclusion.
NG75-NG78 already investigated actual front-rank competitor coverage; withdraw
the repeated coverage audit as the next main research direction. A new operator
bank may need a narrow differential check, but this is not a new mechanism or
an automatic prerequisite to further research. The review recommends one bounded,
adequately trained task-ranking-supervision comparison on a mature sparse base,
with explicit generalization and native-cost decisions. This recommendation is
now instantiated by the NG84 staged protocol, not permission to reopen failed
weight/mask/projection sweeps, relax historical gates, or resume old polling.
The chronological records below retain their original evidence;
their superseded next-step proposals do not override this review.

The next-generation research target is the complete BM25-plus-semantic hybrid,
not the semantic component alone. A quality gain must survive held-out ranking
and all-positive recall checks; a cost gain needs actual index/traversal/latency
evidence, not only fewer nonzeros or smaller compressed artifacts.

On September14 the user clarified that the quality target is **overall hybrid
superiority to dense**, not superiority in every individual domain. Fix task
weights before evaluation; disclose domain losses without imposing an
every-domain-win veto. Report dense-plus-BM25 as a separate strong comparator.
This changes forward research selection, not frozen historical results.
The proposed next stage is
[NG81 overall hybrid quality/cost](research-sae/reports/ng0001-ng0099/ng0081-overall-hybrid-quality-cost-plan.zh.md):
fair TRAIN-only fusion calibration first, then matched geometry/boundary/cost
training and a separate joint-mask diagnostic. Historical M550/M1951 fusion
gains, M1329 rank-fusion failure, M1518 DF-FLOPS failure and NG71's lower-cost
direction are explicitly reviewed, not proposed as new discoveries. The user
then authorized new exploration. The bounded
[NG81-A execution contract](research-sae/reports/ng0001-ng0099/ng0081-fusion-execution.zh.md)
starts with fixed-endpoint calibration/evaluation only; new gradient training
remains conditional and polling stays paused. Broader supervision and
independent qualification follow measured evidence.
The [NG81-A execution progress](research-sae/reports/ng0001-ng0099/ng0081-fusion-progress.zh.md)
now records all eight phases closed, complete new-run mirroring and verified
inventories/hashes. All models select lexical alpha0.1. S z-score nDCG0.787832
remains below dense0.822175; its gain over S's historical profile has a paired
interval crossing zero. P calibration improves its own profile but still trails
dense. S semantic support touches99.9992% of the corpus on average despite no
exactly universal feature. This closes the bounded calibration control, not the
model-quality target. Do not reopen a finer fusion sweep: the next conditional
diagnostic is joint-mask ranking utility versus matched-work DF masking, before
freezing new training. No NG81 job or new gradient training remains active;
polling remains paused. These are exposed diagnostic results, not independent
qualification or native latency measurements.

The user then requested a careful Potion/Tokenlearn review and new parallel
exploration. [NG82's frozen review and protocol](research-sae/reports/ng0001-ng0099/ng0082-potion-review-and-execution.zh.md)
separate static-encoder speed from posting selectivity and distinguish the
historical MSE recipe from current cosine-loss code. Two bounded CPU diagnostics
have [completed and been reviewed](research-sae/reports/ng0001-ng0099/ng0082-progress.zh.md): joint
teacher-margin utility masks versus DF controls, and ordinary PCA versus
query-weighted asymmetric teacher score compression. Neither updates model
weights or accesses locked test data. No arm passes the predeclared joint gates:
DF90 masking saves only10.1% logical work with stable quality; utility-half
improves its TRAIN joint surrogate but loses0.041762 diagnostic nDCG. The
query-weighted256 target retains94.213% of its second-order objective but
pure nDCG falls to0.781213 from dense0.822175. Weighted-versus-PCA intervals
cross zero. New training remains conditional; polling remains paused.
The next plan separates same-TRAIN full-corpus proxy diagnosis, matched-capacity
boundary-weighting controls, and multi-source supervision preparation before
testing passage-level warm-up versus direct ranking on the mature base.
Do not expand individual-utility masks or a PCA/whitening grid. M401/M409 and
NG53's nonadditive-mask failure remain explicit negative controls.

The user authorized [NG83](research-sae/reports/ng0001-ng0099/ng0083-ranking-causality-protocol.zh.md).
Its [same-TRAIN replay has completed](research-sae/reports/ng0001-ng0099/ng0083-progress.zh.md):
utility-half improves joint pair loss but reduces TRAIN nDCG from0.762049 to0.683540,
with paired interval entirely below zero. The surrogate mismatch exists in-sample;
it is not only a query-generalization failure. Do not expand this pruning recipe.
The [matched score-operator control](research-sae/reports/ng0001-ng0099/ng0083-operator-execution.zh.md)
has closed its common12,288-query bank, both3,072-update fits and independently
reviewed full-corpus ranking. Boundary hybrid0.805811 versus field0.804562 has
a paired interval crossing zero; the predeclared SAE-transfer gate fails.
Both pure operators improve over PCA initialization, while global score MSE
worsens; candidate-conditioned ranking and global reconstruction are different
objectives. Boundary's TRAIN margin gain does not transfer to diagnostic margins.
No lambda sweep or immediate SAE transfer is authorized by these results. The
original next-step proposal asked whether actual top-rank competitors were
represented in the supervision bank. The latest direction review above withdraws
that as a new mainline because NG75-NG78 already addressed this failure family;
it does not retroactively change NG83's frozen result or transfer gate.
The [supervision inventory](research-sae/reports/ng0001-ng0099/ng0083-supervision-readiness.zh.md)
separates same-domain scale, source/intent diversity, positive-only labels and
publication-use restrictions. No new data download or locked-test scoring occurred.
All NG83 workers are closed, new artifacts mirrored/verified, and polling paused.

The completed first-wave execution record is the
[NG80 bottleneck campaign](research-sae/reports/ng0001-ng0099/ng0080-bottleneck-campaign.zh.md)
and its [execution ledger](research-sae/reports/ng0001-ng0099/ng0080-execution-ledger.zh.md).
The user authorized parallel small experiments on September13. Readout,
full-trunk adaptation, sparse score-field training, supervision quality/breadth,
input visibility and high-DF utility are distinct questions. Do not infer a
capacity limit from failed linear probing or infer native speed from NNZ.
The first wave uses 12,288 TRAIN queries and 1,536 gradient-excluded diagnostic
queries from the historically exposed three-domain TRAIN corpus. This is not
an independent holdout or a new multi-domain generalization claim.
The F/D training graphs have now closed. Their separately frozen
[terminal diagnostic](research-sae/reports/ng0001-ng0099/ng0080-dense-endpoints.zh.md)
must pass the exposure/serialized-moment audit before fresh full-background
ranking. Do not translate a training-loss change into a retrieval-quality gain.
The [completed F/D review](research-sae/reports/ng0001-ng0099/ng0080-dense-control-review.zh.md)
finds full-trunk nDCG0.666245 versus frozen-trunk0.500623 on the fixed
TRAIN_DIAGNOSTIC; PPLX remains0.822175. This supports adaptation in this
dense-readout setting, not sparse transfer or independent product qualification.
The [actual sparse preflight](research-sae/reports/ng0001-ng0099/ng0080-sparse-preflight.zh.md)
pins TRAIN-only scale, unchanged serving RMS, fresh document VJP and discarded
S/P optimizer canaries before a separate full-training launch. Runtime v4 has
closed calibration and both 16-update canaries, including real gradient and
reload checks; this is execution readiness, not a retrieval-quality result.
The separately frozen [full S/P training](research-sae/reports/ng0001-ng0099/ng0080-sparse-training.zh.md)
has closed all 3,072 updates per arm from the common original parent, with fresh
document VJP and complete optimizer continuation. Training is not terminal
retrieval qualification; the execution ledger tracks verified phase closure.
The separate [S/P terminal review](research-sae/reports/ng0001-ng0099/ng0080-sparse-terminal-review.zh.md)
checks all scalar derivatives, identities, exposures, checkpoint parameters and
serialized moments before endpoint evaluation. It is not an independent full
parameter-gradient replay, and does not change either frozen training run.
The [full-corpus S/P endpoint protocol](research-sae/reports/ng0001-ng0099/ng0080-sparse-endpoints.zh.md)
requires that audit before fresh initial/S/P encoding, and compares fixed
student hybrids against PPLX and a separately TRAIN-calibrated dense hybrid.
Available GPUs, not an assumption that earlier slots remain free, govern
execution.
The first actual endpoint audit failed because it reconstructed a different
FP64 target rather than the frozen FP32 CUDA target. The failed unit is
preserved. Three scalar-only CUDA probes reproduce recorded loss/gradients
exactly; the revised audit preserves the original tolerance and training.
Its 285 CPU tests pass; the full two-arm scalar/exposure/checkpoint audit has
now closed with the original tolerances. Fresh full-corpus initial/S/P encoding
also closed. Lexical preparation aborted at Python shutdown after a ClearML
repository-discovery thread error; its failed exit is not accepted despite
written results. A new V3 continuation explicitly disables that discovery and
reuses only the four sealed audit/encoding phases. It does not rerun training,
relax closure gates, or change any scoring/calibration method. Its fresh lexical
preparation, TRAIN calibration, four full-corpus ranking phases and independent
terminal review have all closed successfully. The
[completed S/P endpoint review](research-sae/reports/ng0001-ng0099/ng0080-sparse-endpoint-review.zh.md)
finds S hybrid nDCG0.786079 versus original0.770716 and dense0.822175.
The gain is concentrated in FEVER, with HotpotQA/NQ point-estimate regressions;
P has no macro gain whose paired interval excludes zero. S document NNZ is4.02x
and total lexical-plus-semantic literal DF-work is7.57x the original, not a
measured native latency ratio. No model is promoted. Independent transfer,
joint-mask utility, diverse supervision, context controls and native total cost
remain unqualified. The user paused polling; no new gradient training is active.

[NG66](research-sae/reports/ng0001-ng0099/ng0066-training-duration-review.zh.md)
found large TRAIN gains but weak exposed-DEV improvement from repeated training.
[NG68](research-sae/reports/ng0001-ng0099/ng0068-supervision-and-ranking-review.zh.md)
separates teacher/label disagreement, missing front-rank competitors, and mixed
original/LLM-derived supervision. Its blinded human-review sample has not yet
received human judgments. Source-negative does not mean judged irrelevant.

The completed controlled comparison is
[NG69 matched-exposure breadth](research-sae/reports/ng0001-ng0099/ng0069-matched-exposure-breadth-plan.zh.md):
6,144 distinct TRAIN queries once versus 1,536 queries four times, keeping the
model and training objective fixed. Teacher preparation comes first; neither
preparation nor one seed qualifies the model. New human supervision, better
ranking teachers, and bounded current-model negative mining are separate
controlled branches, not simultaneous changes. NG67 locked-test queries remain
unencoded and unscored. Consult the numbered protocol for selection, replication,
cost reporting and advancement gates before launching downstream phases.
The [execution contract](research-sae/reports/ng0001-ng0099/ng0069-execution-contract.zh.md)
records independently verified teacher targets, the fixed lexical extension,
and the bounded three-seed pipeline. A completed preparation stage is not a
quality result; all paired ranking and native cost gates still apply.
After all phases complete, [`analyze_ng69_breadth.py`](../scripts/analyze_ng69_breadth.py)
independently audits exposure/optimizer chains, all-positive rankings and sparse
counts, and applies the predeclared new-DEV breadth gate. It reports comparisons
against dense separately; passing the breadth gate is not product qualification.
All three breadth seeds have completed training. Full-background evaluation
stopped at the dense ranking cost canary, not a model correctness failure.
The [bounded CPU diagnosis](research-sae/reports/ng0001-ng0099/ng0069-dense-evaluation-cost-review.zh.md)
isolates allocation-heavy independent score verification and validates a smaller
equivalent reduction. After renewed user authorization, the separate
[evaluation-only continuation](research-sae/reports/ng0001-ng0099/ng0069-evaluation-continuation.zh.md)
reuses all 14 sealed successes and keeps the original quality/resource gates.
The original failed attempt remains unchanged and fully mirrored with SHA checks.
The [final three-seed review](research-sae/reports/ng0001-ng0099/ng0069-final-breadth-review.zh.md)
passes the breadth gate: new-DEV B-A nDCG +0.012425, paired95% interval
[0.007169,0.017956], but B remains0.039682 below dense. Document NNZ grows
2.34-2.52x versus initialization and new-DEV query-DF1.23-1.79x. Do not
directly scale or promote this CE recipe; preserve both quality and cost findings.
[NG70 provenance audit](research-sae/reports/ng0001-ng0099/ng0070-supervision-provenance-plan.zh.md)
separately resolves original versus added positive lineage on TRAIN only; it
does not create human judgments or alter the ongoing A/B evaluation.
Its [completed source/target review](research-sae/reports/ng0001-ng0099/ng0070-supervision-provenance-review.zh.md)
finds that NQ has a much larger added-positive share than HotpotQA and that
uniform multi-positive target mass reduces the original-positive label floor.
These are lineage/optimization facts, not proven label errors. Prioritize
trusted pair-level evidence and token visibility before arbitrary reweighting.

The earlier completed global-boundary pilot is
[NG71 global-boundary ranking](research-sae/reports/ng0001-ng0099/ng0071-global-boundary-ranking-plan.zh.md).
It keeps the mature NG3 initialization and complete fixed hybrid scoring,
separates corpus-witness coverage from a balanced rank-sensitive objective,
and makes cost growth and all-positive boundary harm explicit scale gates.
Its four-arm pilot has completed; conditional breadth expansion and the later
constrained-cost branch remain unexecuted. The checked-in
preparation configuration remains disabled. A separately frozen execution run
may enable the unchanged science only after source/input and accepted preparation
proofs verify; NG69's final independent review has passed. Do not modify the
frozen NG69 comparison or inspect locked-test queries to prepare NG71.
The [NG71 preparation review](research-sae/reports/ng0001-ng0099/ng0071-preparation-review.zh.md)
records completed TRAIN provenance/eligibility checks and real-base CPU/CUDA VJP,
update and reload fixtures. Configuration v2 excludes unresolved source lineage
before fixed-hash selection, not difficult or teacher-disagreeing queries.
High easy-pair saturation makes boundary-witness supervision coverage a required
diagnostic. These checks do not constitute scientific training or quality gains.
The [global step-zero preflight](research-sae/reports/ng0001-ng0099/ng0071-global-preflight-contract.zh.md)
reuses verified complete-corpus encodings on TRAIN only and independently checks
scores, head/boundary/positive ranks and uncertain witness supervision. The full
384-query step-zero manifest, all192fixed batches and four score-space objectives
passed independent replay. Head/boundary witnesses carry55-69%of absolute pair
derivative mass, but teacher conflicts still mask57-65%of head cross100priority
mass. Neither quantity is a parameter-gradient or quality guarantee. Training
chunk and A96 reference primitives, the bounded four-arm scheduler and terminal
observation/paired reviewer are implemented and locally tested. Actual scientific
execution remains distinct from these checks: require its phase receipts, not
just a successful test suite. All eight training chunks must seal before any
sentinel or exposed-DEV observation; locked test remains untouched.
The frozen `pilot-v1` completed all 29 phases on lambda2 at 00:44 UTC on September 13.
Each arm completed 192 updates / 768 exposures. The
[final four-arm review](research-sae/reports/ng0001-ng0099/ng0071-final-pilot-review.zh.md)
records full-background rankings, paired metrics, optimizer/loss audits and
cost proxies. D-A exposed-DEV nDCG is +0.008494 (95% [0.004117, 0.012889]) and
Recall +0.006994, but NQ nDCG -0.007544 violates the predeclared -0.005 domain
floor. The pilot quality gate failed; do not proceed to P2 or drop that domain.
D still trails dense by 0.045361 nDCG and 0.012668 Recall. Document NNZ is 0.9604x
initialization and query-DF 0.3706x, not a native latency or total-cost result.
Changing the loss alone hurt; the witness/objective interaction helped, while
NQ TRAIN gains did not transfer to sentinel/DEV. Next, use only frozen TRAIN
codes to separate query-side, document-side and interaction margin drift, with
NG70 provenance and unchanged full-corpus scoring. Do not launch another
training sweep before that diagnosis. Uniform-pair ablation remains necessary
before attributing future gains to rank weights. Locked test is untouched.
Tracking finalization sometimes added about five minutes and closed naturally;
worker wall time must not be confused with optimizer-loop or serving latency.
The next bounded step is the frozen
[NG72 TRAIN cross-code diagnosis](research-sae/reports/ng0001-ng0099/ng0072-retention-diagnosis-plan.zh.md).
It reuses initial/D192 encodings for four query/document combinations, requires
exact existing TRAIN rank parity, and decomposes fixed-pair margin changes.
It performs no model inference, optimizer updates, DEV or locked-test scoring;
its counterfactual attribution is not a deployable mixed-version model or a
causal training result. Runtime evidence must be recorded separately from this
pre-execution protocol.
The [completed NG72 review](research-sae/reports/ng0001-ng0099/ng0072-retention-diagnosis-review.zh.md)
records successful CPU-only closure, independent margin/metric audit and a full
19-file mirror. HotpotQA sentinel query drift offsets useful document updates;
NQ sentinel loses front-rank quality from either side. Around 70% of negative
margin change in the fixed lost-pair groups comes from weights on surviving
matching support, not disappearance of that support. Average margins and even
NQ Recall@100 improve while nDCG drops. Do not freeze one role globally or treat
NNZ control as the main remedy. Next audit existing TRAIN pair-gradient direction
and supervision coverage before selecting a minimal one-sided margin-retention
control, with a uniform-pair ablation. No new model is qualified or deployed;
the diagnostic mixed codes are not a deployment policy.

The [NG73 actual-gradient replay](research-sae/reports/ng0001-ng0099/ng0073-gradient-replay-review.zh.md)
is now complete. In NQ pilot lost pairs, only 4 of 238 exposures directly
requested margin shrink, versus 143 expansion, 61 ineligible and 30 absent;
shrink contributes only 0.00915% of supervised absolute pair-derivative mass.
This does not explain untrained sentinel drift or establish parameter-level
causality. Do not install a one-sided soft-target floor as the presumed fix.
The next controlled test is the previously required uniform-pair ablation:
same mature base, witness snapshots, teacher targets/confidence, positive
balancing, 384 TRAIN queries and 192 updates, changing only rank weights to 1.
Keep the failed NQ floor, untouched locked test and actual-total-cost goal.
This control now has a frozen [NG74 protocol](research-sae/reports/ng0001-ng0099/ng0074-uniform-control-plan.zh.md)
and a separate [execution status](research-sae/reports/ng0001-ng0099/ng0074-uniform-control-status.zh.md).
The bounded lambda2 GPU0 controller has completed all six phases, including
192 U updates and full-corpus terminal comparison. The
[NG74 review](research-sae/reports/ng0001-ng0099/ng0074-uniform-control-review.zh.md)
finds U-D exposed-DEV nDCG +0.000414 (95% [-0.002499, 0.003420]), but NQ
-0.007319; U-A NQ -0.014862 still fails the unchanged -0.005 domain floor.
Uniform weighting improves TRAIN more than sentinel/DEV and does not repair
generalization. Full 109-file Mac mirroring and independent terminal reduction
passed, with maximum numeric difference 1.39e-17 from the sealed remote review.
Do not launch a weight sweep, expand the failed pilot or claim a
native total-cost win from U's 0.957938x doc NNZ / 0.350651x query-DF proxies.
The next [NG75 design](research-sae/reports/ng0001-ng0099/ng0075-parameter-update-diagnosis-plan.zh.md)
measures actual AdamW displacement against fixed TRAIN pair-margin Jacobians
and finite changes, rather than treating raw score gradients as parameter
causality. The [completed NG75 review](research-sae/reports/ng0001-ng0099/ng0075-parameter-update-diagnosis-review.zh.md)
confirms all 388 local direction predictions and independent numerical
reduction, but the two initial rivals cover only 1 of 60 known endpoint lost
comparisons on the same 24 TRAIN sentinel queries. Local NQ mean margins
improve while those queries' endpoint front-rank quality worsens. Do not
interpret this as either solved interference or evidence to sweep retention
weights. The next [NG76 protocol](research-sae/reports/ng0001-ng0099/ng0076-endpoint-union-trajectory-plan.zh.md)
keeps the cohort, includes both complete endpoint top100 sets and all gold,
and plans bounded observation of the original D trajectory. Intermediate
union ranks are not full-corpus quality. Its [preparation status](research-sae/reports/ng0001-ng0099/ng0076-endpoint-union-trajectory-status.zh.md)
now records verified endpoint top100/gold-cutoff parity for 24 queries,
2,974 unique documents and 5,226 gold/rival pairs. The read-only observer and
historical replay controller are now implemented and separately frozen;
318 local tests and 26 remote frozen fixtures passed. The lambda2 controller
stopped its first bounded replay at step80: one document-NNZ count differs
by one, despite all first80 numeric comparisons remaining within the frozen
tolerance. The full failed run is mirrored and hash verified; no D192 phase
started. The [independent no-observer control](research-sae/reports/ng0001-ng0099/ng0076-replay-control-plan.zh.md)
tests historical reproducibility without changing loss, data or backend
settings. It completed all96 updates with every non-timing trace value,
model SHA and optimizer fingerprint exactly matching original D96; 333 local
tests and 41 remote frozen fixtures pass. The observer-related execution path
is now the main suspect, not an identified kernel cause or universal
determinism claim. The complete54-file/600,682,490-byte result mirror,
609dependencies/30source entries and full non-timing raw JSON equality are
verified onMac; source retained, not a disaster-restore test.
The next [authentic-midpoint diagnostic](research-sae/reports/ng0001-ng0099/ng0076-authentic-midpoint-plan.zh.md)
reuses original D96's complete document CSR and adds only the missing24
sentinel query encodings plus2 fixed parity controls. Its [completed review](research-sae/reports/ng0001-ng0099/ng0076-authentic-midpoint-review.zh.md)
records 26 query forwards, zero document forwards/optimizer updates,
full-corpus rankings and independent CSC/count verification of all5,592,216
scores with zero difference. The complete73-file/51,434,330-byte run/fixture
mirror and678dependencies/36source entries are verified. These8 NQ queries
already lose0.058349 nDCG byD96, partially recover0.007658 byD192, while
Recall@100 stays1 throughout. This rules out a solely second-half onset,
not identifies a causal batch/domain. Across all24 queries,307 middle top100
positions are absent from the endpoint union; pair counts are not nDCG weights.
Do not repeatedly replay or weaken exact gates when the original checkpoint
and document outputs already exist. The13-point historical trajectory is
still incomplete; this independent midpoint diagnosis is a smaller route
to the scientific question. The next [NG77 preparation plan](research-sae/reports/ng0001-ng0099/ng0077-trusted-margin-retention-plan.zh.md)
tests one-sided trusted baseline-margin retention fromstep0 on the same mature
base. Its [completed CPU preparation and independent review](research-sae/reports/ng0001-ng0099/ng0077-trusted-margin-preparation-review.zh.md)
verify 45,175 trusted anchors across all384 pilot queries/660 positives. All
trusted initial top10 comparisons already occur in D's forward pools;
rank11–100 coverage is only37–42%, and future outside-pool rivals remain a
limitation. Thus the first intervention can leave pools/data unchanged.
Original D directly shrinks892 trusted NQ margins atinitial score level,
including40 top10-rival pairs; this is not actual-update causal attribution.
Stable one-sided loss, lambda0 FP32 parity and independent scalar audit pass;
the full32-file/72,828,023-byte preparation is verified onMac andlambda2.
The fixed scientific choice remains lambda0 versuslambda1 withT1. The first
12-query no-optimizer CUDA canary stopped on query1: document support was
identical but558 values across34 documents exceeded the frozen tolerance.
All34 had different dynamic padding from their original corpus-cache groups.
The completed [batch-context diagnostic](research-sae/reports/ng0001-ng0099/ng0077-batch-context-diagnostic.zh.md)
recovers bit-exact cache values for228 documents inoriginal corpus groups
and bit-exact failed values for78 documents inpool groups. Bothcontexts match
the original forward formula; parameters/RNG are unchanged. This identifies
an invalid cross-context equivalence premise inthis canary, notthe historical
NQ quality-regression cause. Bothunits'75files/1,211,947bytes and211dependencies
are verified onMac/lambda2. The separately frozen
[context-qualified canary](research-sae/reports/ng0001-ng0099/ng0077-context-qualified-canary.zh.md)
now passes all12 unchanged D pools, same-context raw-code parity, original
score/noise limits and three-domain shared-VJP fixtures. Mean keep/D gradient
noise is2.9393e-7, below the unchanged1e-3 ceiling. All111files/7,309,388bytes,
process and ClearML closure, parents/source, and independent CPU raw-code,
scalar-score and noise review pass. Cross-context value equivalence still
fails inall12diagnostics and the original canary remains failed.
The [matched96-update protocol](research-sae/reports/ng0001-ng0099/ng0077-matched-retention-pilot.zh.md)
fixes lambda0/1, unchanged384 TRAIN exposures/all660positives, and the
768TRAIN-only endpoint identities/gates before either new outcome.
Execution is staged: close both training processes first, then freeze a
separate full-corpus endpoint controller before any new quality calculation.
Do not call the old terminal selector with its default DEV inclusion.
No GPU observer, sentinel anchors, weight sweep, automatic192/DEV, native-cost
win or held-out qualification is implied. ClearML auto repository detection
added substantial closure overhead: canary body24.74s versus phase373.40s;
budget and polling must follow full phase time, not only model forwards.
Both96-update training arms have now closed in361.06/362.07seconds, with
identical query/token/document exposure counts. Z recovers the historicalD96
parameter hash; K produces a different model. The
[separate endpoint protocol](research-sae/reports/ng0001-ng0099/ng0077-endpoint-protocol.zh.md)
adds independent Decimal/NumPy actual-loss review before fresh768-query,
full233,009-document ranking. Identical Z document cache may be reused only
with complete pinned provenance; no old384-query endpoint cache is reused.
The [final NG77 review](research-sae/reports/ng0001-ng0099/ng0077-final-retention-review.zh.md)
now records all nine endpoint phases closed, with failed predeclared quality
gates. K-Z TRAIN_SENTINEL nDCG is +0.001364 (95% [-0.000187, +0.003210]),
below the +0.005 minimum, with no Recall change. K-initial NQ nDCG is
-0.038196 and HotpotQA -0.014044; HotpotQA Recall -0.011719 also fails its
floor. Strong pilot fitting (+0.042852 macro nDCG versus initial) did not
transfer; K remains 0.064044 below dense on sentinel nDCG. The old trusted
anchor nonpositive-margin count falls only in net from 217 to 201, while
the anchor top10 displacement count stays 28. Document NNZ 0.803210x and
query-DF 0.298127x are only semantic cost proxies, not a native total-cost win.
Do not scale K, sweep lambda, drop NQ or repeat NG72's crossed-code diagnosis.
Next reuse the sealed TRAIN endpoints to contrast actual repaired/new/persistent
front-rank displacements with old pool, trusted-anchor and existing supervision
coverage. This is a bounded read-only decision audit, not new training or a
held-out claim. Use it to choose one controlled refresh, supervision-breadth or
human-judgment branch, rather than adding all three. Any new training protocol
must be frozen separately; no automatic192-step or DEV continuation is allowed.
The new [NG78 boundary-coverage protocol](research-sae/reports/ng0001-ng0099/ng0078-boundary-coverage-plan.zh.md)
freezes that read-only audit before new reductions. It uses all768 TRAIN
identities, all1,371 golds, and the complete three endpoint heads plus original
pools. Per-positive full-corpus DCG/Recall accounting must reproduce NG77;
pair transitions and supervision coverage are descriptive, not additive causal
credit. CPU-only bounded execution and independent artifact verification remain
separate from protocol readiness. No new training is authorized by this audit.
The [completed NG78 review](research-sae/reports/ng0001-ng0099/ng0078-boundary-coverage-review.zh.md)
finds17 strict-anchor repairs,1new loss and200persistent losses. On harmed NQ
Pilot positives,34of40 persistent top10 reversals were already trusted in-pool
pairs; only1was outside. Golds whose reversals were all trusted account for
78.53%of gross positive DCG loss, versus1.65%with any outside-pool reversal.
These are descriptive partitions, not causal percentages. NQ/HotpotQA Sentinel
original golds account for80.66%/97.16%of gross loss, so do not simply erase
added evaluation positives. Prioritize a separately frozen matched-exposure
query-supervision breadth comparison under the same ranking objective, not
candidate-K expansion or a retention-weight sweep. Preserve exposed sentinel
exclusion from training, all domains/golds, exact cost reporting and unchanged
quality floors. NG78 is closed and mirrored; it did not itself launch training.
The [NG79 query-breadth protocol](research-sae/reports/ng0001-ng0099/ng0079-query-breadth-plan.zh.md)
now specifies 384 distinct queries repeated four times versus 1,536 distinct
queries once, both 384 updates from fresh common NG3 initialization. Preserve
the first384 exposure prefix and every update's domain sequence; use static
initial witnesses and the original D objective. Preparation is CPU-only cached
full-corpus scoring, not new encoder inference or training. The already-exposed
TRAIN_SENTINEL stays ID/text-disjoint from gradients; no DEV/LOCKED access,
retention sweep or automatic scaling. Actual token/document work is reported,
not falsely equated by matched query exposure. Training and endpoint evaluation
have separate frozen execution contracts.
Preparation source/protocol commit is `9514f58525dd116e608a8dafba8339c387806aca`;
the immutable `NG-0079/static-witnesses-v1` input manifest is
`f6e0430ea6de939f48c76c153f2f45362d4b7a07761a33a1550e2a647d2fe87c`.
Preparation closed at2026-09-13 13:29:40UTC, with exit0, closed offline tracking
and a verified48-file mirror. The independent CPU reviewer checks selection,
all2,680 positives and scalar D supervision; original384 witness records match
exactly. A Mac mathematical replay stopped safely at the24GiB RAM floor, so
lambda2 performed that review; Mac verified every file and dependency SHA.
The [NG79 training execution contract](research-sae/reports/ng0001-ng0099/ng0079-training-execution.zh.md)
uses the unchanged forward/update/reload engine behind a separately guarded
fixed-reference schedule. Committed source regression616tests and68frozen
fixtures passed on lambda2. The training-only eight-phase graph ran on GPU0
from 14:08:06 UTC to 15:14:17 UTC on 2026-09-13. All eight phases closed with exit0,
closed process groups and closed actual-start offline tracking, not remote
ClearML synchronization. Both fresh96 prefixes exactly reproduced the historical
model and complete optimizer state before same-arm continuation. The entire
179-file, 4,779,845,422-byte output was mirrored and SHA-verified on Mac;
inventory `e60d294e5f36ea1443eabc4edb1b6bf0d2ae895d654d43cc6eaa6373c0daccff`.
This is execution evidence, not a quality gate. Input manifest for
`NG-0079/matched-breadth-384-v1` is
`fab3122d21c270dda659bef84e9e35bcaebe5d5a2ada1b8bc448437f97ef7068`.
No terminal encoding/ranking, DEV or locked-test phase is in that training graph.
The separately frozen [terminal384 endpoint contract](research-sae/reports/ng0001-ng0099/ng0079-endpoint-protocol.zh.md)
uses unchanged shared encoding/ranking engines behind narrow 384-step guards, an
independent actual-loss/optimizer-chain audit before inference, and independent
all-positive endpoint/cost-proxy review. Source `fd885dd9` passed 659 related tests;
the actual frozen endpoint unit passed 111 fixtures on lambda2. Its input SHA is
`b95b74c0e7cedd4343c7b870c2b92e8ff32a39157fba6e6df590172e84158a94`.
Only terminal384 is observed, on the original 384 Pilot/384 Sentinel plus 1,152
added TRAIN queries against all 233,009 documents. Preserve the original
768-query batch prefix, all frozen domain/Recall/NNZ/DF gates and seed79079.
The endpoint controller started at 15:25:57 UTC on lambda2. Its independent
actual-loss/exposure/optimizer-chain audit closed successfully in 245.47 seconds;
all 3,072 query exposures passed. All nine endpoint phases subsequently closed
at 17:10:49 UTC. The terminal review reports B-R Sentinel macro nDCG +0.008210
with an interval crossing zero; B remains 0.051403 below dense and NQ remains
below initialization. B document NNZ is 1.584837x initialization. Thus execution
passed, but the predeclared quality/cost gate did not. Native total cost remains
unqualified. The [post-NG79 synthesis](research-sae/reports/designs/ng-teacher-geometry-supervision-reset.zh.md)
keeps these results distinct from the newly authorized NG80 hypotheses.

## Baseline and Scope

The current package binds P2.2 ABI-v2 through
[`packaging/milestone-model.json`](../packaging/milestone-model.json) and pins
ONNX Runtime through [`packaging/onnxruntime.version`](../packaging/onnxruntime.version).
The full BEIR15/MTEB10 matrices in the reports remain P2.1 / b1.125 evidence;
they must not be relabeled as a current-package rerun. NFCorpus-bound lexical
vocabulary and calibration limit transfer claims for the bundled checkout.

Existing full-text windows, compact postings, explicit impact quantization,
semantic accelerators, and background convergence are the starting point,
not features to reimplement from the old report's checklist. Before opening a
work item, compare it with the current
[storage design](convergent-segmented-index.md),
[query semantics](query-semantics.md), and
[validation workflow](testing-and-validation.md).

## 1. Bind the Evidence Before Further Optimization

- Bind each comparison to source commit, package fingerprint, model manifest,
  tokenizer, calibration, dataset/qrels identity, index options, and query
  shape. Record exact and approximate serving modes separately.
- Complete the dense baseline's PPLX revision, embedding SHA-256, and
  VectorChord build/index manifest. Preserve provenance for the 13 reused
  BEIR rows rather than presenting them as one fresh run.
- Produce a separately labeled current-P2.2 evaluation. Begin with bounded
  differential checks, then expand to the full matrices before making new
  full-surface quality claims. Preserve the P2.1 baseline unchanged.
- Report real pooled P50/P95/P99 alongside per-dataset values. Keep encoder,
  lookup, total query, cold-start, and warm-query timings separate; include
  the dense query encoder in any end-to-end speed comparison.
- Measure 1/4/8/16 concurrency where host capacity permits, with QPS, CPU,
  RSS/shared residency, disk/I/O, cache state, and tail latency. Include
  read-only and mixed-read/write runs with maintenance active.

Deliverable: an identity-bound baseline with explicit measurement coverage and
gaps, not an inferred SLO from historical single-worker timings.

## 2. Reduce High-DF Posting Cost Without Regressing Serving

High DF is a hypothesis to measure, not an explanation for every slow query.
Start with representative high-DF and ordinary queries on MS MARCO and
qualified product corpora. Isolate compiler, filtering, traversal, cache misses,
I/O, and background publication before changing an algorithm.

- Profile query-atom DF, posting/block touches, skip and upper-bound
  effectiveness, candidate counts, shard fanout where applicable, and cache
  behavior. Compare BM25-only and unified BM25-plus-semantic paths.
- Evaluate incremental improvements to existing compact posting layouts,
  block skipping/upper-bound pruning, locality, and artifact sharing or
  deduplication. Measure memory and disk as well as query latency.
- One candidate is a derived term-local exact-bound directory under the existing
  accelerator generation: merge signed bounds into global block order and
  intersect allowed blocks before opening posting payloads. Validate it against
  current packed-reference page reuse and forward-bound structures before
  adding a new representation. Canonical postings remain score authority and
  the existing exact path remains the fallback. This is a proposal, not shipped
  functionality.
- Treat impact quantization and semantic-support/budget changes as explicit
  quality/cost experiments. Compare with existing `f32`/`u8` and alpha modes;
  do not silently turn approximate profiles into generic defaults.
- Start locally on small indexes and increase scale only after behavior is
  understood. Record root, accelerator, and serving-baseline identities,
  publication intervals, bytes rewritten, and resource use under continuous
  writes. Reuse compatible serving artifacts while bounded background work
  converges; do not introduce a serving-baseline expiry or force foreground
  rebuilds because delta exists.
- Prefer changes that reuse current roots. Require evidence and a separate
  migration decision before any full rebuild or incompatible format change.

Deliverable: a narrow measured change with an explained mechanism and
before/after behavior, not a stack of threshold adjustments. Current
maintenance and accelerator designs remain authoritative; historical research
plans are not reopened automatically.

## 3. Improve Quality and Generalization

- Evaluate independent corpora and authorized real-query samples excluded
  from policy selection. Keep a held-out set separate from calibration and
  label any reused selection surface.
- Quantify the bundled NFCorpus vocabulary/calibration boundary before
  proposing broader calibration or a replacement checkpoint. Keep one global
  policy unless a separately reviewed product contract requires otherwise.
- Measure P2.2 full-text windows on long documents against the historical
  6,000-character preparation and 512-token P2.1 limits. Record chunk/window
  aggregation, input length, inference cost, and relevance separately.
- Target the remaining NDCG/MAP/MRR gap without hiding losses in Recall@100
  or CUB@1000. Report downstream reranker gains separately from single-index
  first-stage retrieval; do not add an unreported ANN or late-fusion route.

Deliverable: independent quality evidence tied to the exact model and input
policy, with any model change versioned and qualified separately.

## Acceptance and Stop Conditions

- Exact execution optimizations preserve ranked identities and scores within
  the declared numerical contract. Filtered queries must be compared against
  the complete filtered top-k reference, not just predicate membership.
- Approximate experiments declare their trade-off in advance. Track all five
  metrics and per-dataset changes; preserve Recall@100 within a predeclared
  statistical tolerance and the CUB@1000 advantage, or document any departure
  for explicit review. Do not trade away head ranking silently.
- Set workload-specific latency/resource limits before measurement. Require
  warm-query stability while maintenance converges, no major foreground
  blocking, and no unbounded rewrite, memory, or disk growth. Stop and retain
  evidence if quality, visibility, or latency regresses materially.
- Reproduce locked model/package artifacts from their manifests. Validate
  rebuilt indexes by logical identities, scores, options, and integrity;
  identical physical index bytes are not assumed across hosts or build orders.
- A new default, model contract, or physical format needs its own review and
  qualification. Passing a local microbenchmark does not close a full-matrix
  or deployment gate.

No implementation or new benchmark is recorded as completed by this document.
As work lands, link its evidence here and update the release reports only when
new measurements justify a new report edition.
