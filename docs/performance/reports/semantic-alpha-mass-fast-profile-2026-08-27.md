# Semantic Alpha-Mass Fast Profile

## Decision

Evoke retains `semantic_alpha_mass = 1.0` as the exact default and exposes
values below `1.0` only as an explicit approximate index profile. The current
qualified fast candidate is `0.50`.

The policy retains lexical postings unchanged. For each document field, it
sorts positive semantic impacts in descending order and keeps the shortest
prefix whose cumulative impact reaches the configured fraction of the field's
semantic mass. The existing checkout-defined semantic posting budget remains
an upper bound.

## Native Evidence

On the 500K-document PubMed native index, `semantic_alpha_mass = 0.50` versus
the exact `1.0` generation produced:

| Measure | Change |
| --- | ---: |
| Index bytes | -20.03% |
| Query p50 | -15.03% |
| Query p95 | -14.68% |
| Semantic cluster references | -42.01% |
| Scored documents | -9.17% |
| Forward bytes | -16.64% |

Initial build, eventual semantic completion, CRUD, crash recovery, restart,
REINDEX, accelerator invalidation/republication, and storage/RSS plateau gates
passed under the `0.50` profile.

## Quality Boundary

The broader retrieval matrix was macro-positive, but NFCorpus regressed by
approximately `0.01091` NDCG@10, `0.02053` Recall@100, and `0.01898` MRR@20.
Therefore `0.50` cannot replace the exact default. Values other than `1.0` and
`0.50` are exposed for controlled workload experiments but are not currently
qualified product profiles.

## Lifecycle Contract

The alpha value is a per-index reloption used by both initial publication and
eventual completion. A non-default value participates in the immutable
generation-contract digest. Changing the value on an existing index fails
closed until `REINDEX` publishes one coherent generation; semantic postings
from different alpha profiles cannot be mixed.
