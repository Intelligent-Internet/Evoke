#!/usr/bin/env python3
"""Load a JSONL corpus into a plain PostgreSQL qrels staging table."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Load id/text JSONL for native evoke qrels evaluation.',
    )
    parser.add_argument('--dsn', default='dbname=postgres')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--table', required=True)
    parser.add_argument('--create-schema', action='store_true')
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument('--progress-every', type=int, default=10000)
    parser.add_argument('--expected-rows', type=int)
    parser.add_argument('--output-json', type=Path)
    return parser.parse_args()


def corpus_rows(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'{path}:{line_number}: invalid JSON'
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected object')
            doc_id = row.get('id', row.get('doc_id'))
            text = row.get('text', row.get('text_content'))
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f'{path}:{line_number}: missing id')
            if not isinstance(text, str):
                raise ValueError(f'{path}:{line_number}: missing text')
            yield doc_id, text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_table(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    table: str,
    create_schema: bool,
    truncate: bool,
) -> None:
    with conn.transaction():
        if create_schema:
            conn.execute(
                sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(
                    sql.Identifier(schema),
                )
            )
        conn.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.{} (
                    doc_id text PRIMARY KEY,
                    text_content text NOT NULL
                )
            """).format(
                sql.Identifier(schema),
                sql.Identifier(table),
            )
        )
        existing = int(conn.execute(
            sql.SQL('SELECT count(*) FROM {}.{}').format(
                sql.Identifier(schema),
                sql.Identifier(table),
            )
        ).fetchone()[0])
        if existing and not truncate:
            raise RuntimeError(
                f'{schema}.{table} already contains {existing} rows; '
                'pass --truncate to replace it explicitly'
            )
        if truncate:
            conn.execute(
                sql.SQL('TRUNCATE {}.{}').format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )


def load_corpus(
    conn: psycopg.Connection[Any],
    *,
    path: Path,
    schema: str,
    table: str,
    progress_every: int,
) -> int:
    copy_statement = sql.SQL(
        'COPY {}.{} (doc_id, text_content) FROM STDIN'
    ).format(sql.Identifier(schema), sql.Identifier(table))
    loaded = 0
    started = time.monotonic()
    with conn.transaction():
        with conn.cursor().copy(copy_statement) as copy:
            for row in corpus_rows(path):
                copy.write_row(row)
                loaded += 1
                if progress_every > 0 and loaded % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 1.0e-9)
                    print(
                        f'[native-corpus] loaded={loaded} '
                        f'rows_per_s={loaded / elapsed:.1f}',
                        flush=True,
                    )
    return loaded


def validate_row_count(
    *,
    loaded: int,
    table_count: int,
    expected_rows: int | None,
) -> None:
    if loaded != table_count:
        raise RuntimeError(
            f'loaded row count mismatch: copied={loaded}, '
            f'table={table_count}'
        )
    if expected_rows is not None and loaded != expected_rows:
        raise RuntimeError(
            f'expected row count mismatch: expected={expected_rows}, '
            f'loaded={loaded}'
        )


def main() -> int:
    args = parse_args()
    if args.expected_rows is not None and args.expected_rows < 0:
        raise ValueError('--expected-rows cannot be negative')
    started = time.monotonic()
    with psycopg.connect(args.dsn, prepare_threshold=0) as conn:
        prepare_table(
            conn,
            schema=args.schema,
            table=args.table,
            create_schema=args.create_schema,
            truncate=args.truncate,
        )
        loaded = load_corpus(
            conn,
            path=args.input,
            schema=args.schema,
            table=args.table,
            progress_every=args.progress_every,
        )
        count = int(conn.execute(
            sql.SQL('SELECT count(*) FROM {}.{}').format(
                sql.Identifier(args.schema),
                sql.Identifier(args.table),
            )
        ).fetchone()[0])
    validate_row_count(
        loaded=loaded,
        table_count=count,
        expected_rows=args.expected_rows,
    )
    payload = {
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'expected_rows': args.expected_rows,
        'input': str(args.input),
        'input_bytes': args.input.stat().st_size,
        'input_sha256': sha256(args.input),
        'rows': loaded,
        'schema': args.schema,
        'table': args.table,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
