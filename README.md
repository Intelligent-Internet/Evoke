# Evoke

**The fastest PostgreSQL BM25 engine, now with Sparse Semantic Retrieval
(SSR) postings in the same index.**

<img width="1500" height="600" alt="commons-banner-github" src="docs/banner.png" />

Evoke leads the measured PostgreSQL BM25 engines in the project's published,
like-for-like PostgreSQL 18 BEIR comparison. Its default path is an exact,
high-throughput, PostgreSQL-native BM25 access method. See the
[PG18 15 x 5 benchmark](docs/performance/README.md) for the measured engines,
datasets, query shapes, and reproducibility boundary behind this claim.

On top of that BM25 foundation, Evoke fuses Sparse Semantic Retrieval (SSR)
postings with lexical evidence inside one page-native PostgreSQL index. For
AI-era RAG, this provides a more convenient, precise, and integrated retrieval
solution: exact lexical ranking and semantic recall through one SQL interface,
one transactional relation, and one maintenance and replication lifecycle.

One `USING ii42` index can run in either of two modes:

- **Exact BM25** is the default and the performance foundation. It provides
  corpus-statistics-based lexical ranking without model inference.
- **BM25 + SSR** is enabled with `sae = true`. A model emits semantic atoms
  that share one posting space, scorer, relation, and lifecycle with lexical
  evidence.

The SQL reloption remains `sae` for the current ABI and catalog contract. In
prose, this documentation uses SSR for the product retrieval path and reserves
SAE for model/encoder vocabulary mechanisms, historical experiment names, and
literal API or GUC names.

Both modes use the same index and lifecycle APIs. BM25 keeps its ordinary
PostgreSQL operator/index-scan surface. RAG and semantic applications use one
`ii42_query(...)` family: two- and four-argument scalar overloads mark natural
ranked SQL, while overloads with an explicit `k` return hit rows. Both modes use
`ii42_index_status(...)` and the `ii42_index_maintain*` functions. Index removal
follows the ordinary PostgreSQL lifecycle.

## Product Shape

```text
table rows
    |
    v
CREATE INDEX ... USING ii42
    |
    +-- BM25: lexical postings
    |
    `-- sae=true: lexical + semantic atoms
                 in one posting namespace
    |
    v
one relation-owned page-native index
    |
    v
ii42_query(...): natural SQL or explicit hit rows
```

The current storage path is page-native v3:

- one checked root owns document versions, postings, mutation state, and
  maintenance state;
- transactions append changes to relation-owned linked L0; commit and
  PostgreSQL MVCC decide their visibility;
- background work seals changed batches, compacts selected segments, folds
  terms, completes semantic rows, and reclaims retired pages;
- query backends normally read relation pages through PostgreSQL shared buffers
  and keep only query-bounded scratch; selected converged indexes may expose one
  exact-root fold shared by all backends;
- `CREATE INDEX` and explicit `REINDEX` derive postings from source rows;
  optional query-accelerator refresh can traverse a complete stored baseline
  without re-encoding unchanged documents.

Exact reads include visible lexical L0. Bounded SSR queries may keep using a
compatible older accelerator and scope baseline, with current-row recheck,
until background refresh publishes a replacement. Warmness, baseline freshness,
and semantic completeness are separate states.

There is no semantic side table, external ANN index, application-managed
publish step, or second mutation lifecycle.

The detailed storage, mutation, convergence, reclamation, and performance
contract is defined in the
[Convergent Segmented Index design](docs/convergent-segmented-index.md).

Project support, contribution, conduct, and security policies are defined in
[SUPPORT.md](SUPPORT.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## Capabilities

- PostgreSQL 17 and 18 extension packages; PostgreSQL 18 is the primary
  development target.
- `text`, `varchar`, `text[]`, `varchar[]`, and `int4[]` index inputs.
- Single-column and supported multicolumn text-like indexes.
- Optional field-aware multicolumn lexical and semantic posting namespaces,
  including public whole-field weighting.
- Lucene, Robertson, ATIRE, BM25L, and BM25+ scoring variants in BM25 mode.
- Built-in token normalization, stopwords, English stemming, and diacritic
  folding for raw-text BM25 indexes.
- Planner-visible natural SQL for semantic ranking, including ordinary
  PostgreSQL predicates and field-aware weights.
- One explicit `ii42_query(...)` hit API for BM25, semantic, filtered, and
  field-aware retrieval.
- Transactional `INSERT`, `UPDATE`, and `DELETE` with MVCC-correct reads.
- WAL durability, restart/crash recovery, `VACUUM`, `REINDEX`, and physical
  replication support.
- Lexical-first, eventual-only semantic mutation so foreground writes do not
  run document inference.
- Shared-runtime model execution; application backends do not own model
  sessions or index-sized semantic state.
- Public multi-index fusion and Evoke/vector hybrid composition above the
  single-index product path.

## Semantic Retrieval Model

With `sae = true`, a sparse encoder converts query and document text into
weighted semantic atoms. Those atoms and lexical terms occupy one posting
namespace and are accumulated by one page-native scorer. Foreground writes
remain fast because lexical evidence is published first; shared workers encode
only changed document versions and complete semantic evidence later.

Release packages ship the digest-locked P2.2 ABI-v2 successor to the P2.1
milestone as their default checkout. It uses the same 30.3M-parameter Granite
sparse route and is lifecycle-qualified for full-text document and query
encoding. Its lexical vocabulary and calibration are frozen from NFCorpus, so
deployments may override it with their own qualified checkout. The Beta 1
model report separates this current package contract from the frozen
P2.1 BEIR15/MTEB10 single-index evidence and documents the known limitations:
[English](docs/technical-report-evoke-model.md) and
[Traditional Chinese](docs/technical-report-evoke-model-zh.md).

The frozen [Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
is publicly available on Hugging Face, including the ONNX artifacts, model
card, license, and checksum-locked build archive. Model weights are not stored
in this Git repository. Follow the
[model download instructions](docs/examples/semantic-model-checkout.md#download-the-default-model)
before building a complete release ZIP or Docker image; no Hugging Face token
is required.

## Install

### Release package

Download the [v0.2.5 Beta release](https://github.com/Intelligent-Internet/Evoke/releases/tag/v0.2.5).
Linux x86-64 ZIPs are available for PostgreSQL 17 and 18, each with a SHA-256
checksum file. They include ONNX Runtime 1.29.0 and the frozen default model;
no separate model checkout is needed to install a release package.

Install only a package matching the target operating system, architecture,
PostgreSQL major, and dependency ABI. Verify the checksum and compare
`BUILD-INFO.txt` with the target `pg_config` directories before copying files.
For a fresh installation, copy the package before starting PostgreSQL. When
replacing an installation loaded through `shared_preload_libraries`, first
quiesce Evoke maintenance and stop PostgreSQL; never overwrite `ii42` or its
bundled ONNX Runtime beneath a running postmaster. Follow the complete
[deployment boundary](docs/upgrading.md#deployment-boundary), including the
required restart and installed-package validation.

```bash
shasum -a 256 -c evoke-*.zip.sha256
unzip evoke-*.zip
sudo rsync -a evoke-*/ /
```

On Linux, use `sha256sum -c`. Then create the extension in each database that
will own Evoke indexes:

```sql
CREATE EXTENSION ii42;
```

This ordinary installation is sufficient. A separate extension schema is
optional and is mainly useful for custom schema placement or a side-by-side
[`psql_bm25s` migration](docs/upgrading.md). Evoke is intentionally not
relocatable after creation.

### Docker

The public PostgreSQL 18 image includes Evoke, ONNX Runtime 1.29.0, and the
frozen default model. No registry login or separate model download is required.
The published platform is `linux/amd64`; other architectures need emulation
or a source build. Pin the versioned tag for reproducible deployment;
`ghcr.io/intelligent-internet/evoke:pg18` and
`ghcr.io/intelligent-internet/evoke:latest` are moving aliases for the same
published PostgreSQL 18 image. GitHub Latest points to this release as well;
the Beta 1 designation describes its product maturity, not a separate GitHub
pre-release channel.

```bash
docker pull ghcr.io/intelligent-internet/evoke:pg18-v0.2.5

read -r -s -p 'PostgreSQL password: ' POSTGRES_PASSWORD
printf '\n'
export POSTGRES_PASSWORD
docker run -d \
    --name evoke-pg18 \
    -e POSTGRES_PASSWORD \
    -p 127.0.0.1:5432:5432 \
    -v evoke-pg18-data:/var/lib/postgresql \
    ghcr.io/intelligent-internet/evoke:pg18-v0.2.5
unset POSTGRES_PASSWORD
```

On a fresh data volume, initialization creates the extension and enables its
shared runtime. Existing database volumes are not upgraded by initialization
scripts; follow [Upgrading](docs/upgrading.md) before changing an existing
deployment. The release also provides a checksummed Docker archive for
offline loading with `docker load`.

To build the image from source instead, first
[download the default model](docs/examples/semantic-model-checkout.md#download-the-default-model)
to `.artifacts/evoke-milestone-model`, or pass `--model-checkout` with a validated
checkout path:

```bash
scripts/build_release_docker_image.sh \
    --version 0.2.5 \
    --image-tag evoke:local-pg18
```

### Source build

```bash
make PG_CONFIG=/path/to/pg_config
make PG_CONFIG=/path/to/pg_config install
make PG_CONFIG=/path/to/pg_config installcheck
```

Source builds require ONNX Runtime by default so a query-serving installation
cannot silently omit semantic inference. A lexical-only diagnostic build must opt
out explicitly with `II42_ENABLE_ONNXRUNTIME=0`; do not install that build on a
PostgreSQL server that serves `sae = true` indexes. See
[Contributing](CONTRIBUTING.md) for build and validation details.
Source installs do not download a model implicitly; install a checkout at the
compiled shared-data path or configure an override before using SSR. The
weights are not required to compile the extension or use exact BM25; see
[source-install model setup](docs/examples/semantic-model-checkout.md#source-installation).

## BM25 Quickstart

For single-column, multicolumn, field-aware, and semantic variants in one
beginner flow, see [Getting Started](docs/getting-started.md).

```sql
CREATE EXTENSION IF NOT EXISTS ii42;

CREATE TABLE docs (
    id bigint PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL
);

INSERT INTO docs (id, title, body) VALUES
    (1, 'Red apple', 'fresh red apple fruit'),
    (2, 'Green apple', 'green apple slices'),
    (3, 'Orange', 'orange citrus fruit'),
    (4, 'Cat guide', 'small cat animal care');

CREATE INDEX docs_body_idx ON docs USING ii42 (body);

SELECT d.id, d.title, hit.score
FROM ii42_query(
    'docs_body_idx'::regclass,
    'apple fruit',
    10
) AS hit
JOIN docs AS d ON d.ctid = hit.ctid
ORDER BY hit.score DESC, d.id;
```

The default index uses Lucene-style BM25 and `realtime` consistency.

## Semantic Quickstart

Release packages include the locked milestone checkout. Configure the shared
runtime before starting PostgreSQL:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
```

Restart PostgreSQL, then create the index:

```sql
CREATE INDEX docs_semantic_idx
ON docs USING ii42 (body)
WITH (sae = true);

SELECT d.id,
       d.title,
       ii42_query(
           'docs_semantic_idx'::regclass,
           'database search architecture'
       ) AS score
FROM docs AS d
ORDER BY score DESC
LIMIT 10;
```

SSR is eventual-only. A foreground write publishes lexical evidence and a
semantic-pending document version without running model inference. Shared
workers later add semantic atoms in bounded batches. Both states are read by
the same page-native scorer.

See the [SSR quickstart](docs/examples/semantic-index-quickstart.md),
[index parameters](docs/index-parameters.md), and
[operations guide](docs/examples/semantic-index-operations.md) for advanced
runtime, model, field, filter, preload, and lifecycle configuration.

## Product API

| Task | API |
| --- | --- |
| Create | `CREATE INDEX ... USING ii42` |
| Natural semantic search | `ORDER BY ii42_query(index, query, ...) DESC LIMIT k` |
| Explicit hit search | `ii42_query(index, query, k, ...)` |
| Options | `ii42_index_options(index)` |
| Status | `ii42_index_status(index)` |
| Details | `ii42_index_details(index)` |
| Maintain | `ii42_index_maintain(index)` |
| Maintain if not busy | `ii42_index_try_maintain(index)` |
| Maintain due indexes | `ii42_index_maintain_due(max_indexes)` |
| Compose Evoke indexes | `ii42_fusion_query(...)` |
| Compose Evoke and vector candidates | `ii42_hybrid_fuse_candidates(...)` |

Exact-BM25 rowset and token-level diagnostic helpers remain extension-owner
surfaces. Runtime inspection and control have separate privileges described in
the [API reference](docs/api-reference.md#privilege-summary).
Prefer ordinary SQL plus `ii42_query(...)`
for one semantic index when its supported query shape applies. Fusion and
hybrid APIs compose independently retrieved sources without changing any
source index.

## Consistency And Maintenance

BM25 supports:

- `realtime`: committed lexical changes are query-visible immediately;
- `eventual`: lower foreground cost with automatic convergence;
- `manual`: explicitly maintained static or externally scheduled indexes.

Semantic-enabled indexes accept only `eventual`. They are still lexical-first:
the exact route can search new lexical postings before semantic completion.
A compatible bounded accelerator may temporarily omit post-baseline rows while
background work converges. Maintenance always operates on the same index root
and mutation stream; it does not build a side index.

Use:

```sql
SELECT ii42_index_status('docs_body_idx'::regclass);
SELECT ii42_index_maintain('docs_body_idx'::regclass);
DROP INDEX docs_body_idx;
```

## Current Boundaries

- The planner-native semantic path accepts one base table, ordinary `WHERE`
  predicates, descending rank, and a bounded `LIMIT`. Joins, RLS, row locking,
  secondary ordering, and partitioned-parent global ranking fail closed. Simple
  AND predicates on `INCLUDE` columns may use the published scope baseline for
  one filtered probe. PostgreSQL rechecks every returned row; unavailable,
  unsupported, or insufficient scope probes fall back to complete-subset
  scoring. Newly matching post-baseline rows may wait for background
  convergence under the approximate semantic contract.
- `ii42_query(...)` supports structured
  `eq`/`in`/`overlap`/`ilike`/`ilike_any`/`range` predicates for explicit
  subset top-k. Fully scope-backed requests use the same compatible published
  baseline and current-row recheck as planner-native search; they may omit
  post-baseline matches or return fewer than `k` while maintenance converges.
  Statement-local `tid[]` remains a low-level exact-set route.
- A partitioned parent does not provide one globally ranked corpus. Query child
  indexes independently or use one unpartitioned search relation.
- Parallel heap build, parallel AM scan, and parallel VACUUM discovery are not
  implemented in the current release.
- Physical standbys need the same extension binary, ONNX Runtime, and model
  checkout. Logical replication copies table rows, not index relations.

## Documentation

- [Getting started](docs/getting-started.md)
- [Documentation map](docs/README.md)
- [System technical report](docs/technical-report-evoke-system.md)
  ([繁體中文](docs/technical-report-evoke-system-zh.md))
- [Architecture](docs/architecture-and-design.md)
- [Convergent segmented index](docs/convergent-segmented-index.md)
- [API reference](docs/api-reference.md)
- [Index parameters](docs/index-parameters.md)
- [Query semantics](docs/query-semantics.md)
- [Index policy](docs/index-policy.md)
- [Shared runtime and residency](docs/shared-runtime-and-residency.md)
- [Maintenance lifecycle](docs/maintenance-lifecycle.md)
- [Migration](docs/upgrading.md)
- [Testing](docs/testing-and-validation.md)
- [Contributing](CONTRIBUTING.md)

The system and model technical reports describe the Beta 1 architecture
and distinguish current contracts from versioned experimental evidence.
Earlier research, performance, and lexical technical reports remain preserved
as evidence archives. New performance claims require reproducible experiments
and current data, not a documentation-only refresh.

## License

Evoke is licensed under the [Apache License 2.0](LICENSE). Release packages
also include required third-party notices under `LICENSES/`.
