# Migrate From `psql_bm25s`

II42 supports one historical product migration: preserve the source table and
rebuild a `psql_bm25s` index as a current II42 index.

II42 does not ship beta-to-beta `ALTER EXTENSION` scripts, retired physical
format readers, or a chain of intermediate catalog transitions. A current
package installs one current catalog. An unsupported II42 beta deployment must
preserve its source tables, remove its old indexes and extension catalog, then
create current indexes from the current package.

## Migration Shape

Migrate side by side so the old index remains available until the new index is
validated:

1. Keep the source table, `psql_bm25s` extension, and old index online.
2. Install the current II42 package on the primary and every physical standby.
3. Create II42 in a separate schema. The two extensions define some
   same-signature operators and cannot coexist in one schema.
4. Build a new `USING ii42` index over the same source columns.
5. Compare representative results, CRUD behavior, and operational readiness.
6. Switch application queries to II42.
7. Drop `psql_bm25s` only after the rollback window closes.

## Install Beside `psql_bm25s`

```sql
CREATE SCHEMA ii42_ext;
CREATE EXTENSION ii42 WITH SCHEMA ii42_ext;
```

The access-method name is not schema-qualified, so index creation still uses
`USING ii42`:

```sql
CREATE INDEX CONCURRENTLY docs_body_ii42_idx
ON docs USING ii42 (body);
```

For a unified lexical and semantic index, configure the shared runtime first
and create the new index with only the mode switch:

```sql
CREATE INDEX CONCURRENTLY docs_body_semantic_idx
ON docs USING ii42 (body)
WITH (sae = true);
```

Sparse Semantic Retrieval (SSR) is eventual-only. Foreground writes become
lexically searchable first and shared workers complete semantic postings through
the same index lifecycle.

Do not mutate the old index in place or reinterpret its relation pages. The
new index is derived from the source table by the current writer.

## Inventory And Rebuild Tool

The operator tool can inventory both access methods and generate a reviewable
plan. It is not a neutral BM25-only migration command: its generated
`migrate_psql_bm25s_text_like` action selects `sae = true`, eventual consistency,
and runtime precision `fp16`. Integer-token migration remains lexical-only.
For a BM25-preserving text migration, use the side-by-side SQL above and retain
the original supported scoring, text, and maintenance options explicitly.

Inventory and preview without executing replacement:

```bash
python3 scripts/rebuild_ii42_indexes.py \
    --inventory-output /secure/path/ii42-migration-plan.json \
    --database application_database

python3 scripts/rebuild_ii42_indexes.py \
    --plan /secure/path/ii42-migration-plan.json \
    --dry-run \
    --jobs 1
```

The plan records source tables, indexed keys, `INCLUDE` columns, partial
predicates, reloptions, and relation sizes. Review the generated action and
actual target SQL, not only the captured source definition: the runner
preserves only supported option allowlists and may supply defaults. In
particular, `--mode lexical-only` filters actions; it does not convert a
semantic text migration into BM25.

Execution validates a replacement, then drops the old index and renames the
replacement transactionally. It does not retain an independent old index for
an application rollback window. Use the manual side-by-side procedure when
that retention is required. Remove `--dry-run` only after reviewing these
cutover semantics and installing a coherent package/runtime on every node.

## Validate Before Cutover

Check the new index through the II42 schema:

```sql
SELECT ii42_ext.ii42_index_status(
    'docs_body_ii42_idx'::regclass
);

SELECT source.id, hit.score
FROM ii42_ext.ii42_query(
    'docs_body_ii42_idx'::regclass,
    'migration verification query',
    20
) AS hit
JOIN docs AS source ON source.ctid = hit.ctid
ORDER BY hit.score DESC, source.id;
```

The migration gate must cover:

- representative result and score parity for BM25-only migration;
- valid, ready, and query-ready current roots;
- `INSERT`, indexed-column `UPDATE`, `DELETE`, and rollback behavior;
- maintenance convergence, restart, and physical-standby replay;
- package, ONNX Runtime, and model identity for `sae = true`; and
- application queries under their actual role and `search_path`.

After cutover, remove the old index and extension only after dependency checks
and rollback approval:

```sql
DROP INDEX old_docs_body_psql_bm25s_idx;
DROP EXTENSION psql_bm25s;
```

Dropping the old extension must not alter source rows or the current II42
index.

When removing a current II42 index directly, use the ordinary PostgreSQL
lifecycle:

```sql
DROP INDEX docs_body_ii42_idx;
```

## Current-Only II42 Boundary

There is no supported in-place migration between experimental II42 catalogs or
physical generations. Matching `extversion` values do not prove that a binary,
catalog, and root belong to one package. For an unsupported II42 beta state:

1. inventory the live source/index topology;
2. retain the source tables and reviewed creation options;
3. remove the unsupported II42 indexes and catalog;
4. install one coherent current package; and
5. recreate and qualify the indexes with the current writer.

Retired roots fail closed. Do not add a compatibility reader, hand-written SQL
patch, or intermediate `ALTER EXTENSION` script as a general way to avoid a
rebuild.

The current-format COW fold-boundary repair is narrower than a beta-format
upgrade. An earlier 0.2.5 writer could compact across a term-fold watermark
after both sides had already been classified. The current reader accepts that
root only when the manifest is explicitly COW and its exact tail directory
proves each replacement run's covered-versus-tail ownership. The current
writer prevents new straddles. Missing or ambiguous proof, non-COW roots, and
different storage versions still fail closed and require a rebuild from source.

A deployment-specific catalog convergence is narrower than a supported
upgrade path. It is admissible only when release evidence proves that the
installed binary, access method, storage format, and current roots are already
the target release, and the only difference is an exact SQL function mapping.
Apply that mapping transactionally after installing the coherent package with
PostgreSQL stopped, then prove unchanged index relfilenodes, identical query
results, current catalog identity, and primary/standby health. This exception
does not authorize compatibility aliases, physical-format reinterpretation,
or a reusable beta migration script. If any of those invariants cannot be
proved, preserve the source tables and rebuild.

## Deployment Boundary

Package files, ONNX Runtime, model checkout, and PostgreSQL major must match on
the primary and every physical standby. WAL replicates relation and catalog
changes; it does not install host files.

When replacing an II42 binary loaded through `shared_preload_libraries`:

1. stop new maintenance admission and allow active work to drain;
2. stop PostgreSQL;
3. install the complete fingerprinted package atomically;
4. restart and verify binary/runtime identity; and
5. restore the positive maintenance-worker limit after smoke tests.

Never overwrite a shared-preloaded library beneath a running postmaster.
`sae = true` also requires a positive `ii42.shared_runtime_size`; model
sessions and tokenizers have no backend-local fallback.

The package-bound migration smoke is documented in
[Testing And Validation](testing-and-validation.md). Advanced current-index
configuration belongs in [Index Parameters](index-parameters.md), not in the
migration contract.
