#!/usr/bin/env python3
"""Build a strict local PostgreSQL BEIR15 benchmark database.

The script targets the existing local PostgreSQL instance and the `postgres`
database.  It keeps each BEIR dataset in its own document table so that
VectorChord can use dataset-specific list counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg import sql


DATASETS: tuple[str, ...] = (
    'nfcorpus',
    'scifact',
    'arguana',
    'scidocs',
    'fiqa',
    'trec-covid',
    'webis-touche2020',
    'cqadupstack',
    'quora',
    'nq',
    'dbpedia-entity',
    'hotpotqa',
    'fever',
    'climate-fever',
    'msmarco',
)


EXPECTED_DOCS: dict[str, int] = {
    'nfcorpus': 3_633,
    'scifact': 5_183,
    'arguana': 8_674,
    'scidocs': 25_657,
    'fiqa': 57_638,
    'trec-covid': 171_331,
    'webis-touche2020': 382_545,
    'cqadupstack': 457_199,
    'quora': 522_931,
    'nq': 2_681_468,
    'dbpedia-entity': 4_635_922,
    'hotpotqa': 5_233_329,
    'fever': 5_416_568,
    'climate-fever': 5_416_593,
    'msmarco': 8_841_823,
}


DEFAULT_REMOTE_ROOT = os.environ.get('EVOKE_BEIR15_SOURCE', '')
DEFAULT_STAGING = Path(tempfile.gettempdir()) / 'evoke-beir15-pg-staging'
DEFAULT_SCHEMA = 'evoke_beir15'
DEFAULT_DSN = 'dbname=postgres'
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    table_name: str
    expected_docs: int
    vector_lists: int | None
    vector_probes: int | None


def run(
    argv: list[str],
    *,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print('+', ' '.join(argv), flush=True)
    return subprocess.run(
        argv,
        cwd=cwd,
        check=check,
        text=True,
    )


def safe_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def vector_lists(row_count: int) -> int | None:
    """Return the VectorChord partition count for product evaluation."""
    if row_count < 100_000:
        return None
    if row_count < 2_000_000:
        return max(1, round(row_count / 500))
    return max(1, math.ceil(4.0 * math.sqrt(row_count)))


def vector_probes(list_count: int | None) -> int | None:
    if list_count is None:
        return None
    return max(1, round(math.sqrt(list_count)))


def dataset_plan(
    dataset: str,
    *,
    expected_docs: int | None = None,
) -> DatasetPlan:
    if expected_docs is None and dataset not in EXPECTED_DOCS:
        raise ValueError(f'unknown dataset: {dataset}')
    row_count = EXPECTED_DOCS[dataset] if expected_docs is None else expected_docs
    if row_count <= 0:
        raise ValueError(f'{dataset}: expected docs must be positive')
    lists = vector_lists(row_count)
    return DatasetPlan(
        name=dataset,
        table_name='docs_' + safe_name(dataset),
        expected_docs=row_count,
        vector_lists=lists,
        vector_probes=vector_probes(lists),
    )


def dataset_plan_from_files(staging: Path, dataset: str) -> DatasetPlan:
    root = local_dataset_dir(staging, dataset)
    path = root / 'documents.jsonl'
    if not path.exists():
        raise RuntimeError(f'{dataset}: missing documents file: {path}')
    return dataset_plan(dataset, expected_docs=count_lines(path))


def selected_datasets(
    values: list[str],
    *,
    allow_custom: bool = False,
) -> list[str]:
    if not values or values == ['all']:
        return list(DATASETS)
    unknown = [value for value in values if value not in DATASETS]
    if unknown and not allow_custom:
        raise SystemExit(f'unknown dataset(s): {unknown}')
    return values


def connect(dsn: str) -> psycopg.Connection[Any]:
    conn = psycopg.connect(dsn)
    conn.execute(
        'SELECT set_config(%s, %s, false)',
        ('application_name', 'evoke-beir15-build'),
    )
    conn.execute('SELECT set_config(%s, %s, false)', ('lock_timeout', '30s'))
    conn.execute('SELECT set_config(%s, %s, false)', ('statement_timeout', '0'))
    conn.execute(
        'SELECT set_config(%s, %s, false)',
        ('synchronous_commit', 'off'),
    )
    return conn


def qident(name: str) -> sql.Identifier:
    return sql.Identifier(name)


def local_dataset_dir(staging: Path, dataset: str) -> Path:
    return staging / dataset


def required_files(root: Path) -> list[Path]:
    return [
        root / 'documents.jsonl',
        root / 'queries.jsonl',
        root / 'quality_qrels.json',
    ]


def sync_dataset(source: str, staging: Path, dataset: str) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    dest = local_dataset_dir(staging, dataset)
    dest.mkdir(parents=True, exist_ok=True)
    source_path = source.rstrip('/') + f'/{dataset}/'
    run([
        'rsync',
        '-a',
        '-z',
        '--copy-links',
        '--partial',
        '--append-verify',
        '--info=progress2',
        '--include=documents.jsonl',
        '--include=queries.jsonl',
        '--include=quality_qrels.json',
        '--include=official_prepare_summary.json',
        '--include=mteb_retrieval_prepare_summary.json',
        '--include=missing_queries.json',
        '--exclude=*',
        source_path,
        str(dest) + '/',
    ])
    missing = [str(path) for path in required_files(dest) if not path.exists()]
    if missing:
        raise RuntimeError(f'{dataset}: missing required files: {missing}')


def count_lines(path: Path) -> int:
    output = subprocess.check_output(['wc', '-l', str(path)], text=True)
    return int(output.split()[0])


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected object row')
            yield row


def get_text(row: dict[str, Any]) -> str:
    for key in ('text', 'text_content', 'contents', 'body'):
        value = row.get(key)
        if isinstance(value, str):
            return value
    title = row.get('title')
    if isinstance(title, str):
        return title
    return ''


def get_embedding(row: dict[str, Any], path: Path, row_no: int) -> str:
    for key in ('embedding', 'vector', 'pplx_embedding'):
        value = row.get(key)
        if isinstance(value, list):
            if len(value) != EMBEDDING_DIM:
                raise ValueError(
                    f'{path}:{row_no}: embedding dim {len(value)} != '
                    f'{EMBEDDING_DIM}'
                )
            return '[' + ','.join(f'{float(item):.8g}' for item in value) + ']'
    raise ValueError(f'{path}:{row_no}: missing embedding list')


def get_row_id(row: dict[str, Any], path: Path, row_no: int) -> str:
    for key in ('id', 'doc_id', 'query_id', '_id'):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f'{path}:{row_no}: missing id')


def metadata_json(row: dict[str, Any]) -> str:
    value = row.get('metadata')
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def create_base_schema(conn: psycopg.Connection[Any], schema: str) -> None:
    with conn.transaction():
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(
            qident(schema)
        ))
        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.dataset_manifest (
                dataset text PRIMARY KEY,
                table_name text NOT NULL,
                expected_docs bigint NOT NULL,
                expected_queries bigint,
                expected_qrels bigint,
                loaded_docs bigint,
                loaded_queries bigint,
                loaded_qrels bigint,
                vector_lists integer,
                vector_probes integer,
                source_path text,
                status text NOT NULL DEFAULT 'new',
                updated_at timestamptz NOT NULL DEFAULT now()
            )
        """).format(qident(schema)))
        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.queries (
                dataset text NOT NULL,
                query_id text NOT NULL,
                text_content text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding {} NOT NULL,
                PRIMARY KEY (dataset, query_id)
            )
        """).format(qident(schema), sql.SQL(f'halfvec({EMBEDDING_DIM})')))
        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.qrels (
                dataset text NOT NULL,
                query_id text NOT NULL,
                doc_id text NOT NULL,
                score real NOT NULL,
                PRIMARY KEY (dataset, query_id, doc_id)
            )
        """).format(qident(schema)))


def create_docs_table(
    conn: psycopg.Connection[Any],
    schema: str,
    plan: DatasetPlan,
) -> None:
    with conn.transaction():
        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.{} (
                doc_id text NOT NULL,
                text_content text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding {} NOT NULL
            )
        """).format(
            qident(schema),
            qident(plan.table_name),
            sql.SQL(f'halfvec({EMBEDDING_DIM})'),
        ))
        conn.execute(sql.SQL("""
            INSERT INTO {}.dataset_manifest (
                dataset,
                table_name,
                expected_docs,
                vector_lists,
                vector_probes,
                status
            )
            VALUES (%s, %s, %s, %s, %s, 'initialized')
            ON CONFLICT (dataset) DO UPDATE SET
                table_name = EXCLUDED.table_name,
                expected_docs = EXCLUDED.expected_docs,
                vector_lists = EXCLUDED.vector_lists,
                vector_probes = EXCLUDED.vector_probes,
                updated_at = now()
        """).format(qident(schema)), (
            plan.name,
            plan.table_name,
            plan.expected_docs,
            plan.vector_lists,
            plan.vector_probes,
        ))


def load_documents(
    conn: psycopg.Connection[Any],
    schema: str,
    plan: DatasetPlan,
    root: Path,
) -> int:
    path = root / 'documents.jsonl'
    line_count = count_lines(path)
    if line_count != plan.expected_docs:
        raise RuntimeError(
            f'{plan.name}: documents line count {line_count} != '
            f'{plan.expected_docs}'
        )
    table = sql.SQL('{}.{}').format(qident(schema), qident(plan.table_name))
    start = time.monotonic()
    with conn.transaction():
        conn.execute(sql.SQL('TRUNCATE {}').format(table))
        with conn.cursor() as cur:
            copy_sql = sql.SQL(
                'COPY {} (doc_id, text_content, metadata, embedding) '
                'FROM STDIN'
            ).format(table)
            with cur.copy(copy_sql) as copy:
                for row_no, row in enumerate(read_jsonl(path), 1):
                    copy.write_row((
                        get_row_id(row, path, row_no),
                        get_text(row),
                        metadata_json(row),
                        get_embedding(row, path, row_no),
                    ))
    elapsed = time.monotonic() - start
    print(f'{plan.name}: loaded {line_count} docs in {elapsed:.1f}s')
    return line_count


def load_queries(
    conn: psycopg.Connection[Any],
    schema: str,
    dataset: str,
    root: Path,
) -> int:
    path = root / 'queries.jsonl'
    line_count = count_lines(path)
    with conn.transaction():
        conn.execute(sql.SQL(
            'DELETE FROM {}.queries WHERE dataset = %s'
        ).format(qident(schema)), (dataset,))
        with conn.cursor() as cur:
            copy_sql = sql.SQL(
                'COPY {}.queries '
                '(dataset, query_id, text_content, metadata, embedding) '
                'FROM STDIN'
            ).format(qident(schema))
            with cur.copy(copy_sql) as copy:
                for row_no, row in enumerate(read_jsonl(path), 1):
                    copy.write_row((
                        dataset,
                        get_row_id(row, path, row_no),
                        get_text(row),
                        metadata_json(row),
                        get_embedding(row, path, row_no),
                    ))
    print(f'{dataset}: loaded {line_count} queries')
    return line_count


def iter_qrels(path: Path) -> Iterable[tuple[str, str, float]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected qrels object')
    for query_id, docs in data.items():
        if isinstance(docs, dict):
            for doc_id, score in docs.items():
                yield str(query_id), str(doc_id), float(score)
        elif isinstance(docs, list):
            for item in docs:
                if isinstance(item, dict):
                    doc_id = item.get('doc_id') or item.get('id')
                    score = item.get('score', item.get('relevance', 1.0))
                    if doc_id is not None:
                        yield str(query_id), str(doc_id), float(score)
                else:
                    yield str(query_id), str(item), 1.0
        else:
            raise ValueError(f'{path}: invalid qrels value for {query_id}')


def load_qrels(
    conn: psycopg.Connection[Any],
    schema: str,
    dataset: str,
    root: Path,
) -> int:
    path = root / 'quality_qrels.json'
    rows = list(iter_qrels(path))
    with conn.transaction():
        conn.execute(sql.SQL(
            'DELETE FROM {}.qrels WHERE dataset = %s'
        ).format(qident(schema)), (dataset,))
        with conn.cursor() as cur:
            copy_sql = sql.SQL(
                'COPY {}.qrels (dataset, query_id, doc_id, score) '
                'FROM STDIN'
            ).format(qident(schema))
            with cur.copy(copy_sql) as copy:
                for query_id, doc_id, score in rows:
                    copy.write_row((dataset, query_id, doc_id, score))
    print(f'{dataset}: loaded {len(rows)} qrels')
    return len(rows)


def load_dataset(
    conn: psycopg.Connection[Any],
    schema: str,
    staging: Path,
    dataset: str,
    plan: DatasetPlan,
) -> None:
    root = local_dataset_dir(staging, dataset)
    missing = [str(path) for path in required_files(root) if not path.exists()]
    if missing:
        raise RuntimeError(f'{dataset}: missing required files: {missing}')
    create_docs_table(conn, schema, plan)
    docs = load_documents(conn, schema, plan, root)
    queries = load_queries(conn, schema, dataset, root)
    qrels = load_qrels(conn, schema, dataset, root)
    with conn.transaction():
        conn.execute(sql.SQL("""
            UPDATE {}.dataset_manifest
            SET loaded_docs = %s,
                loaded_queries = %s,
                loaded_qrels = %s,
                expected_queries = %s,
                expected_qrels = %s,
                source_path = %s,
                status = 'loaded',
                updated_at = now()
            WHERE dataset = %s
        """).format(qident(schema)), (
            docs,
            queries,
            qrels,
            queries,
            qrels,
            str(root),
            dataset,
        ))


def vchord_options(plan: DatasetPlan) -> str:
    lines = [
        'residual_quantization = true',
        '[build.internal]',
        'spherical_centroids = true',
        'build_threads = 8',
    ]
    if plan.vector_lists is not None:
        lines.insert(2, f'lists = [{plan.vector_lists}]')
    return '\n'.join(lines) + '\n'


def create_indexes(
    conn: psycopg.Connection[Any],
    schema: str,
    dataset: str,
    *,
    plan: DatasetPlan,
    rebuild: bool,
) -> None:
    table = sql.SQL('{}.{}').format(qident(schema), qident(plan.table_name))
    doc_idx = f'{plan.table_name}_doc_id_idx'
    bm25_idx = f'{plan.table_name}_bm25_idx'
    vector_idx = f'{plan.table_name}_embedding_vchord_idx'
    with conn.transaction():
        if rebuild:
            for index_name in (vector_idx, bm25_idx, doc_idx):
                conn.execute(sql.SQL('DROP INDEX IF EXISTS {}.{}').format(
                    qident(schema),
                    qident(index_name),
                ))
        conn.execute(sql.SQL(
            'CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} (doc_id)'
        ).format(qident(doc_idx), table))
    start = time.monotonic()
    with conn.transaction():
        conn.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS {}
            ON {} USING evoke (text_content)
            WITH (
                consistency = 'eventual'
            )
        """).format(qident(bm25_idx), table))
    print(f'{dataset}: bm25 index ready in {time.monotonic() - start:.1f}s')
    start = time.monotonic()
    vector_options = sql.SQL('')
    if plan.vector_lists is not None:
        vector_options = sql.SQL(' WITH (options = {})').format(
            sql.Literal(vchord_options(plan)),
        )
    with conn.transaction():
        conn.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS {}
            ON {} USING vchordrq (embedding halfvec_cosine_ops)
            {}
            WHERE vector_norm(embedding::vector) > 0
        """).format(
            qident(vector_idx),
            table,
            vector_options,
        ))
    print(
        f'{dataset}: vector index ready in {time.monotonic() - start:.1f}s '
        f'lists={plan.vector_lists} probes={plan.vector_probes}'
    )
    with conn.transaction():
        conn.execute(sql.SQL("""
            UPDATE {}.dataset_manifest
            SET status = 'indexed',
                vector_lists = %s,
                vector_probes = %s,
                updated_at = now()
            WHERE dataset = %s
        """).format(qident(schema)), (
            plan.vector_lists,
            plan.vector_probes,
            dataset,
        ))


def verify_dataset(
    conn: psycopg.Connection[Any],
    schema: str,
    dataset: str,
    plan: DatasetPlan,
) -> None:
    table = sql.SQL('{}.{}').format(qident(schema), qident(plan.table_name))
    bm25_regclass = f'{schema}.{plan.table_name}_bm25_idx'
    vector_index_name = f'{plan.table_name}_embedding_vchord_idx'
    with conn.cursor() as cur:
        cur.execute('SET enable_indexscan = off')
        cur.execute('SET enable_bitmapscan = off')
        cur.execute(sql.SQL('SELECT count(*) FROM {}').format(table))
        docs = cur.fetchone()[0]
        if docs != plan.expected_docs:
            raise RuntimeError(
                f'{dataset}: loaded docs {docs} != {plan.expected_docs}'
            )
        cur.execute(
            sql.SQL('SELECT count(*) FROM {}.queries WHERE dataset = %s')
            .format(qident(schema)),
            (dataset,),
        )
        queries = cur.fetchone()[0]
        cur.execute(
            sql.SQL('SELECT count(*) FROM {}.qrels WHERE dataset = %s')
            .format(qident(schema)),
            (dataset,),
        )
        qrels = cur.fetchone()[0]
        cur.execute(
            sql.SQL('SELECT text_content, embedding FROM {}.queries '
                    'WHERE dataset = %s LIMIT 1').format(qident(schema)),
            (dataset,),
        )
        query_row = cur.fetchone()
        if not query_row:
            raise RuntimeError(f'{dataset}: no query rows')
        query_text, query_embedding = query_row
        cur.execute('RESET enable_indexscan')
        cur.execute('RESET enable_bitmapscan')
        cur.execute(
            sql.SQL("""
                SELECT count(*)
                FROM public.evoke_query(%s::regclass, %s, 5, NULL) h
                JOIN {} d ON d.ctid = h.ctid
            """).format(table),
            (bm25_regclass, query_text),
        )
        bm25_hits = cur.fetchone()[0]
        if bm25_hits <= 0:
            raise RuntimeError(f'{dataset}: evoke query returned no hits')
        cur.execute('SET enable_seqscan = off')
        if plan.vector_probes is not None:
            cur.execute(
                'SELECT set_config(%s, %s, true)',
                ('vchordrq.probes', str(plan.vector_probes)),
            )
        else:
            cur.execute('RESET vchordrq.probes')
        cur.execute(sql.SQL("""
            EXPLAIN (COSTS OFF)
            SELECT doc_id
            FROM {}
            WHERE vector_norm(embedding::vector) > 0
            ORDER BY embedding <=> %s::halfvec
            LIMIT 5
        """).format(table), (str(query_embedding),))
        plan_rows = [row[0] for row in cur.fetchall()]
        plan_text = '\n'.join(plan_rows)
        if vector_index_name not in plan_text:
            raise RuntimeError(
                f'{dataset}: vector index not used in plan:\n{plan_text}'
            )
    print(
        f'{dataset}: verify ok docs={docs} queries={queries} '
        f'qrels={qrels} bm25_hits={bm25_hits}'
    )


def show_manifest(conn: psycopg.Connection[Any], schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT dataset,
                   status,
                   loaded_docs,
                   expected_docs,
                   loaded_queries,
                   loaded_qrels,
                   vector_lists,
                   vector_probes
            FROM {}.dataset_manifest
            ORDER BY COALESCE(array_position(%s::text[], dataset), 2147483647),
                     dataset
        """).format(qident(schema)), (list(DATASETS),))
        for row in cur.fetchall():
            print('\t'.join('' if value is None else str(value)
                            for value in row))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action',
        choices=[
            'init',
            'sync',
            'load',
            'index',
            'verify',
            'manifest',
            'build',
        ],
    )
    parser.add_argument('--datasets', nargs='*', default=['all'])
    parser.add_argument('--dsn', default=DEFAULT_DSN)
    parser.add_argument('--schema', default=DEFAULT_SCHEMA)
    parser.add_argument('--source', default=DEFAULT_REMOTE_ROOT)
    parser.add_argument('--staging', type=Path, default=DEFAULT_STAGING)
    parser.add_argument('--reindex', action='store_true')
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument(
        '--expected-from-files',
        action='store_true',
        help=(
            'Use documents.jsonl line counts from --staging instead of the '
            'official BEIR15 row-count manifest. This keeps the same native '
            'DB/index lifecycle for sampled shared15-style roots.'
        ),
    )
    parser.add_argument(
        '--allow-custom-datasets',
        action='store_true',
        help=(
            'Allow dataset names outside the BEIR15 manifest. This is intended '
            'for MTEB-style roots and requires --expected-from-files.'
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    global EMBEDDING_DIM
    args = parse_args(argv)
    if args.embedding_dim <= 0:
        raise SystemExit('--embedding-dim must be positive')
    EMBEDDING_DIM = args.embedding_dim
    if args.allow_custom_datasets and not args.expected_from_files:
        raise SystemExit(
            '--allow-custom-datasets requires --expected-from-files'
        )
    datasets = selected_datasets(
        args.datasets,
        allow_custom=args.allow_custom_datasets,
    )
    if args.action == 'sync':
        if not args.source:
            raise SystemExit(
                'sync requires --source or EVOKE_BEIR15_SOURCE'
            )
        for dataset in datasets:
            sync_dataset(args.source, args.staging, dataset)
        return 0
    plans = {
        dataset: (
            dataset_plan_from_files(args.staging, dataset)
            if args.expected_from_files
            else dataset_plan(dataset)
        )
        for dataset in datasets
    }
    with connect(args.dsn) as conn:
        if args.action in ('init', 'build'):
            create_base_schema(conn, args.schema)
            for dataset in datasets:
                create_docs_table(conn, args.schema, plans[dataset])
        if args.action in ('load', 'build'):
            for dataset in datasets:
                load_dataset(
                    conn,
                    args.schema,
                    args.staging,
                    dataset,
                    plans[dataset],
                )
        if args.action in ('index', 'build'):
            for dataset in datasets:
                create_indexes(
                    conn,
                    args.schema,
                    dataset,
                    plan=plans[dataset],
                    rebuild=args.reindex,
                )
        if args.action in ('verify', 'build'):
            for dataset in datasets:
                verify_dataset(conn, args.schema, dataset, plans[dataset])
        if args.action == 'manifest':
            show_manifest(conn, args.schema)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
