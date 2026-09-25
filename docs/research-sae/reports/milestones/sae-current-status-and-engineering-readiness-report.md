# II-42 Current Status And Engineering Readiness

Updated: 2026-07-26

> Historical milestone. The candidate, API examples, and outstanding release
> tasks below describe the recorded 2026-07-26 state, not current Beta 1.
> Current contracts are linked in the document table below.

## Executive Decision

The product architecture is converged on one PostgreSQL access method and one
relation-owned lifecycle:

```text
text
  -> optional shared model encoder
  -> one lexical or unified sparse posting generation
  -> ii42_query(...)
```

`sae = false` stores an exact BM25 index. `sae = true` stores one merged
lexical/model posting map. Both use the same create, query, status,
maintenance, rebuild, recovery, replication, and drop contracts. The model
encoder is an additional build/query step, not another index lifecycle.

The current release candidate is P2.2 on extension version `0.2.0`.
Engineering lifecycle readiness is strong. Public release readiness is not yet
declared because the repository archive and documentation pass is still in
progress, and the final package-bound suite must be repeated from the final
clean commit.

## Product Contract

Applications use:

```sql
CREATE INDEX docs_search_idx
ON docs USING ii42 (body);

CREATE INDEX docs_semantic_idx
ON docs USING ii42 (body)
WITH (sae = true);

SELECT *
FROM ii42_query(
    'docs_search_idx'::regclass,
    'postgresql sparse search',
    20
);

SELECT ii42_index_status('docs_search_idx'::regclass);
SELECT ii42_index_maintain('docs_search_idx'::regclass);
DROP INDEX docs_search_idx;
```

The named relation decides whether search is BM25-only or model-backed.
Applications do not select a scorer, publish a corpus generation, manage a
semantic sidecar, or call model-/SAE-prefixed lifecycle functions.

## Implemented Lifecycle

One physical index relation owns:

- one active base generation;
- one relation-owned delta;
- tombstones inside the same mutation stream;
- one identity map;
- one manifest, generation number, lock, scheduler, and compaction path.

For `sae = true`, each visible delta record contains the complete
lexical-plus-semantic atom map. A write cannot commit a lexical-only partial
record. Query evaluates:

```text
active unified base + visible unified delta - tombstones
```

Maintenance compacts that logical surface and publishes one replacement
generation. Failure before the metapage switch leaves the previous readable
generation intact.

The verified lifecycle covers:

- empty and populated builds;
- realtime, eventual, and manual consistency;
- `INSERT`, indexed and HOT `UPDATE`, `DELETE`, rollback, and savepoints;
- `VACUUM`, compaction, `REINDEX`, and concurrent DDL;
- failed rebuild and corrupt-model fail-closed behavior;
- cold restart and immediate crash recovery;
- logged and unlogged relations;
- primary/standby physical replication;
- relation drop without sidecar cleanup;
- deterministic full-text windows with explicit oversized-input failure;
- batch encoding for build/rebuild while query encoding remains immediate.

## Current Evidence

The P2.2 target-host qualification is recorded in
[Enlightenment P2 Production Qualification](../../../performance/reports/enlightenment-p2-production-qualification-2026-07-22.md).
That work established:

- a real `0.1.2 -> 0.2.0` catalog upgrade with fresh-install parity;
- 48/48 model-backed lifecycle gates;
- 18/18 package-bound maturity gates;
- exact ranking identity through rollback and reinstall;
- physical replication through build, mutation, maintenance, and `REINDEX`;
- zero runtime failures in the accepted stability window;
- full native qrels rows on SciFact, Arguana, SciDocs, and TREC-COVID with
  positive P2.2 deltas over BM25 on all tracked quality metrics.

The current cleanup candidate additionally passed:

- the complete isolated PostgreSQL regression;
- fresh and versioned non-`public` schema installation;
- the `0.1.2 -> 0.2.0` upgrade smoke;
- 1,056 Python tests;
- all 18 package-bound maturity steps, including resource soaks and the
  50,000-document model-backed lifecycle;
- byte-identical staged, current, and installed extension binaries.

These results are engineering evidence for the current source state. They are
not a substitute for the final clean-commit package rerun.

## Quality And Performance Boundary

P2.2 proves that a unified posting index can improve over BM25 on the qualified
native held-out rows. It does not prove dense equivalence or universal
cross-domain superiority. The current model is calibrated from NFCorpus and
still has material build, storage, and posting-traversal costs on large
corpora.

The canonical quality and product reports are:

- [Model technical report, including frozen P2.1 evidence](../../../technical-report-evoke-model-zh.md);
- [P2.1 BEIR15 native full matrix](../m1900-m1999/evoke-p2.1-beir15-native-full-matrix-report.md);
- [P2.1 MTEB10 native matrix](../m1900-m1999/evoke-p2.1-mteb10-native-matrix-report.md);
- [P2.1 end-to-end product health audit](../m1900-m1999/ii42-p2.1-end-to-end-product-health-audit-report.md);
- [P2.2 target-host qualification](../../../performance/reports/enlightenment-p2-production-qualification-2026-07-22.md).

Later M1900-M1973 reports remain research evidence. They do not replace the
P2.2 product contract unless they pass the same native lifecycle, quality,
cost, and package gates.

## Release Work Remaining

The remaining work is bounded:

1. archive root-level research reports and machine evidence under `docs/`,
   repair tracked links, and enforce the layout;
2. finish public API, quickstart, upgrade, security, changelog, and release
   documentation;
3. rerun static, Python, shell, PostgreSQL, upgrade, lifecycle, replication,
   and package-bound maturity gates;
4. build PostgreSQL 17/18 ZIP artifacts and the PostgreSQL 18 image from the
   final clean commit;
5. verify package fingerprints remain identical before and after maturity
   testing.

The historical gate dispositions are preserved in the
[engineering archive](../../../archive/engineering/README.md).
Current release checks are maintained in
[Testing and validation](../../../testing-and-validation.md).

## Canonical Engineering Documents

| Document | Purpose |
| --- | --- |
| [Convergent Segmented Index](../../../convergent-segmented-index.md) | Current storage and publication design |
| [Maintenance lifecycle](../../../maintenance-lifecycle.md) | Current linked-L0, semantic, accelerator, and reclamation behavior |
| [Semantic index quickstart](../../../examples/semantic-index-quickstart.md) | Minimal semantic setup |
| [Operations guide](../../../examples/semantic-index-operations.md) | CRUD, maintenance, recovery, and diagnostics |
| [API reference](../../../api-reference.md) | Supported SQL surfaces and privilege boundaries |
| [Testing and validation](../../../testing-and-validation.md) | Current qualification commands |

Historical status documents and closed experiment routes are archive context,
not active implementation guidance.
