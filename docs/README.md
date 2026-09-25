# Evoke Documentation

Evoke exposes one PostgreSQL access method and one product lifecycle. The index
contract is selected at `CREATE INDEX` time:

- `sae = false` (default): exact BM25;
- `sae = true`: Sparse Semantic Retrieval (SSR), with lexical and semantic
  atoms in one unified posting index.

The `sae` reloption is the current SQL/catalog name. In prose, the current
product retrieval path is SSR; SAE is reserved for encoder/vocabulary
mechanisms, historical experiment names, and literal API or GUC names.

Both modes share `evoke_index_status(...)` and common maintenance functions.
PostgreSQL `DROP INDEX` owns teardown for both modes. BM25 keeps its ordinary
index-scan surface. Semantic ranking uses one overloaded `evoke_query(...)`
family for both ordinary ranked table SQL and explicit hit rows. The current
implementation is page-native v3.
SSR is eventual-only and lexical-first; shared workers complete semantic
postings without creating a second index or mutation lifecycle.

## Start

- [Getting started](getting-started.md): canonical first BM25 index, search,
  optional semantic step, readiness, rebuild, and teardown.
- [Main README](../README.md): installation and minimal BM25/semantic examples.
- [Semantic quickstart](examples/semantic-index-quickstart.md): focused
  unified BM25-plus-semantic setup after the basic BM25 flow.
- [Supported input types](input-types.md): scalar, token-array, and multicolumn
  choices.
- [Model checkout](examples/semantic-model-checkout.md): pinned
  [Beta 1 model download](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1),
  integrity checks, source installation, and release packaging.

## Understand

- [System technical report](technical-report-evoke-system.md)
  ([Traditional Chinese](technical-report-evoke-system-zh.md)): overall Beta 1
  design, COW storage, model integration, accelerators, lifecycle, and evidence.
- [Model technical report](technical-report-evoke-model.md)
  ([Traditional Chinese](technical-report-evoke-model-zh.md)): Beta 1 model design,
  current package identity, frozen evaluation evidence, and limits.
- [Architecture](architecture-and-design.md): product boundaries, storage,
  query, mutation, maintenance, and process ownership.
- [Convergent segmented index](convergent-segmented-index.md): current product
  design and flow diagrams for COW mutation, exact and bounded-baseline query
  routes, semantic completion, accelerator publication, and reclamation.
- [Query semantics](query-semantics.md): ranking, operators, result identity,
  and filtering boundaries.
- [Field-aware indexes](field-aware-indexes.md): one-index multicolumn field
  identity and weighting.
- [Multicolumn indexes](multicolumn-indexes.md): supported combined
  column shapes.

## Use

- [API reference](api-reference.md): authoritative public, owner, and
  diagnostic SQL surfaces.
- [Function index](functions.md): compact function lookup.
- [Index parameters](index-parameters.md): reloptions and server settings.
- [Index policy](index-policy.md): BM25 consistency and eventual-only SSR.
- [Semantic query API](examples/semantic-query-api.md): how the
  unified application route dispatches.
- [Multi-index fusion](multi-index-fusion.md): public composition of separate
  Evoke indexes.
- [Hybrid search](hybrid-search.md): public composition of Evoke and external
  vector candidates.
- [Hybrid fusion engine](hybrid-fusion-engine.md): normalization, fusion, and
  execution contract.
- [Scoring profile](examples/scoring-profile-reference.md): the immutable model
  scoring contract, distinct from application fusion weights.

## Operate

- [Semantic index operations](examples/semantic-index-operations.md): CRUD,
  completion, health, rebuild, backup, restart, and replication.
- [Semantic runtime setup](examples/semantic-runtime.md): inference workers,
  query-lane reservation, and model-contract handoff.
- [Shared runtime and residency](shared-runtime-and-residency.md): model-session,
  shared-memory, backend-memory, and warmup ownership.
- [Maintenance lifecycle](maintenance-lifecycle.md): linked
  L0, COW publication, compaction, fold, and reclamation.
- [Connection memory](connection-memory.md): build, query, worker, and backend
  memory controls.
- [Migration](upgrading.md): `psql_bm25s` to Evoke and BM25 to semantic mode.

## Develop

- [Contributing](../CONTRIBUTING.md): build, test, and release workflow.
- [Testing and validation](testing-and-validation.md): qualification layers and
  exactness boundaries.
- [Product roadmap](product-roadmap.md): current release state and deferred
  post-beta work.
- [Model planning](model-planning.md): detailed model-evidence and efficiency
  follow-up under the product roadmap, separate from the release reports.

## Evidence And Archives

- [SAE research archive](research-sae/README.md)
- [Performance guide and evidence](performance/README.md): current benchmark
  entrypoints and separately dated historical measurements.
- [Competitive analyses](analysis/README.md)
- [Engineering archive](archive/engineering/README.md): query-first evidence,
  closed readiness and development ledgers, and the historical v0.2.5 roadmap.
- [Lexical foundation technical report](technical-report-psql_bm25s.md)
- [Project narrative](archive/engineering/psql-bm25s-project-narrative.md)

These are evidence surfaces, not substitutes for the current API and
operations documents. Performance and technical conclusions remain frozen
until a new experiment design and current datasets are deliberately validated.
