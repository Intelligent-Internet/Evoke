# Product Roadmap

Status: current engineering planning authority.

Evoke v0.2.5 has completed the bounded current-only qualification on Shadow,
Elm, and Mac. The package, catalog, runtime, index-root, query, restart,
resource, and replication evidence is recorded in the
[three-environment rollout report](performance/reports/evoke-v0.2.5-current-only-three-environment-rollout-2026-08-30.md).
That report is dated qualification evidence, not a live deployment inventory.
Closed implementation and rollout ledgers are indexed separately in the
[engineering archive](archive/engineering/README.md).

The current product design authority remains the
[Convergent Segmented Index](convergent-segmented-index.md). The generic
semantic defaults are `f32`, block64, and `semantic_alpha_mass = 1.0`. The
Commons rollout used the explicit approximate `u8` and
`semantic_alpha_mass = 0.50` profile; that deployment choice is not a hidden
product default.

## Release Preparation

- Keep the public SQL surface, privileges, documentation, and examples aligned
  with the installed extension contract.
- Keep source, dependency, model, package, and generated notice fingerprints
  reproducible in release automation.
- Keep active source, documentation, and automation free of maintainer-local
  paths, generated outputs, credentials, and unsupported operational claims.
  Archived raw evidence may retain its original execution provenance.
- Require fresh-install, upgrade, regression, privilege, lifecycle, restart,
  and package checks for the final v0.2.5 artifact.

Release preparation must not change a physical index format, ranking contract,
or default approximation policy without a separately reviewed design and new
qualification evidence.

## Deferred Post-Beta Work

### Model Evidence and Efficiency

[Model Planning](model-planning.md) is the subordinate work plan for the
Beta 1 model reports. It holds evidence binding, high-DF
posting efficiency, quality/generalization experiments, and acceptance gates.
The roadmap remains the planning authority; the plan does not authorize an
unreviewed model, format, default-policy, or deployment change.

### QDIR-RSS-1: measured query-directory memory

Measure query-directory residency on representative BM25 and semantic roots
before changing admission limits or layout. Disk-size reductions alone are not
query-latency or memory evidence.

### CSG-I114: access-method orchestration boundaries

`src/ii42_am.c` still owns a large but cohesive orchestration closure. Further
extraction is optional and must preserve typed, one-way authority boundaries.
Do not add a second root, scheduler, scorer, lifecycle, or broad private API to
reduce file size.

### CSG-I132: measured parallel scale work

Consider parallel build, VACUUM discovery, or scan work only after profiling
shows a current product gate is not met. Any implementation must preserve one
checked publication authority and exact page-native scoring.

## Invariants And Non-Goals

- PostgreSQL owns relation lifecycle; ordinary `DROP INDEX` removes every Evoke
  payload owned by that relation.
- Sparse Semantic Retrieval (SSR) remains lexical-first and eventual-only, with
  one unified posting index and no sidecar mutation lifecycle.
- Approximate semantic precision is an immutable per-index choice and must not
  weaken lexical exactness, filtering membership, MVCC visibility, or result
  completeness silently.
- Historical reports and unchecked archive items are evidence, not active work
  queues.
