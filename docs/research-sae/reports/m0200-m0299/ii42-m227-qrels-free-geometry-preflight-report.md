# II-42 M227 Qrels-Free Geometry Teacher Preflight Report

Date: 2026-09-29

## Purpose

M226A/B established that the `32d INT4` product head can transfer a strong
continuous teacher score geometry. The retained supervised endpoint is
`meanz_pair_meanz_score + m220c_joint,w=3.5`, with M226B mean MAP 0.360175 /
Recall 0.639834 and mean deltas versus PPLX dense of +0.058710 / +0.029097,
with no dense-unsafe rows.

M227 asks whether existing qrels-free signals inside the frozen top256 window
have enough teacher-side headroom and shape similarity to justify a later
qrels-free score-geometry distillation run. This is a preflight, not a product
head training run.

Teacher construction is qrels-free: policies use the completed RankT5 cache,
dense scores, and fixed baseline scores. M214B supervised teacher artifacts may
be used only as a diagnostic reference for score-geometry similarity, not to
construct any M227 policy. Qrels are used only for final metrics.

## M227A Plan

Evaluate qrels-free teacher policies:

- `rankt5_z`
- `dense_z`
- `baseline_z`
- `rankt5_dense_mean_z`
- `rankt5_dense_baseline_mean_z`
- `rankt5_dense_rrf`
- `dense_baseline_rrf`
- `rankt5_dense_intersection_128`
- `rankt5_dense_union_128`

Shared settings:

- frozen candidate window: top256
- RankT5 scores: completed M217 cache
- eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- fixed weights: `1.0,1.5,2.0,2.5,3.0,3.5`
- diagnostic similarity reference: M214B seed301/313/337 mean-z teacher
  geometry

Decision rule:

- If a qrels-free teacher policy has positive macro MAP/Recall, acceptable row
  safety, and meaningful similarity to the supervised mean-z geometry, prepare
  a narrow M227B distillation run.
- If no policy has sufficient headroom or every high-MAP policy is row-unsafe,
  stop before training a qrels-free student and report that a stronger
  weak-supervision source is needed.

## M227A Result

M227A completed the qrels-free score-geometry teacher preflight. The best
qrels-free policies have positive macro headroom, but no evaluated policy is
dense-safe. Therefore there is no justified M227B student-distillation run from
these sources.

Best policies by macro MAP:

| Policy | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows | Top24 overlap vs M226 geometry | Pearson z |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `rankt5_dense_mean_z` | 3.50 | 0.325182 | 0.621012 | +0.023717 | +0.010275 | FiQA, SciDocs | 0.555 | 0.711 |
| `rankt5_dense_union_128` | 3.50 | 0.322757 | 0.620060 | +0.021293 | +0.009323 | FiQA, SciDocs | 0.565 | 0.623 |
| `rankt5_dense_baseline_mean_z` | 3.50 | 0.316840 | 0.617716 | +0.015375 | +0.006979 | FiQA, SciDocs | 0.569 | 0.699 |
| `dense_z` | 3.50 | 0.305675 | 0.613481 | +0.004210 | +0.002744 | FiQA, SciDocs | 0.633 | 0.818 |

The most similar qrels-free source to supervised M226 geometry is `dense_z`
(Pearson z 0.818, top24 overlap 0.633), but it has very little macro headroom
and is still unsafe on FiQA and SciDocs. The best macro teacher,
`rankt5_dense_mean_z`, has better headroom but lower geometry similarity and
the same unsafe rows. This is the same failure pattern seen in earlier
RankT5-safe work: RankT5+dense blending creates useful macro movement, but it
does not reproduce the supervised row corrections that make M226 dense-safe.

No policy satisfied the continuation rule:

- positive macro MAP/Recall: yes for several policies
- meaningful similarity to M226 geometry: partial
- acceptable row safety: no

## M227 Decision

Do not launch M227B from these qrels-free sources. The preflight indicates that
the currently available qrels-free RankT5/dense/baseline score geometry is not
a sufficient replacement for the M226 supervised mean-z teacher geometry.

The retained endpoints are:

- supervised diagnostic endpoint: M226B `meanz_pair_meanz_score +
  m220c_joint,w=3.5`, mean MAP 0.360175 / Recall 0.639834, dense-safe
- best qrels-free deployable endpoint so far: M221 family-router route, still
  useful but far below the supervised geometry bridge and not row-safe

The next breakthrough should target a genuinely stronger weak-supervision
source or an explicit row-aware pseudo-label objective that can fix FiQA and
SciDocs while preserving the RankT5+dense macro gains. More blending of the
current RankT5/dense/baseline geometry is unlikely to close the gap.
