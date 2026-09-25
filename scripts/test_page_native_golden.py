#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
)
from test_convergent_segment_read_smoke import (
    ID_ROWS,
    PREDICATE_ROWS,
    TEXT_ROWS,
    acquire_maintenance_guard,
    collect_surfaces,
    configure_cluster,
    connect,
    fetch_ids,
    fetch_ids_ordered,
    fetch_ordered,
    fetch_predicate_ids,
    fetch_search,
    pg_config_value,
    release_maintenance_guard,
    reserve_port,
    run,
    start_cluster,
    stop_cluster,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
GOLDEN_PATH = REPO_ROOT / 'tests/fixtures/page_native_golden.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate the independent page-native v3 golden surfaces.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path, required=True)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        required=True,
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_golden() -> dict[str, Any]:
    payload = json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('unsupported page-native golden schema')
    return payload


def assert_shared_runtime_config(config_path: Path) -> None:
    config = config_path.read_text(encoding='utf-8')
    required = (
        "shared_preload_libraries = 'evoke'",
        'evoke.shared_runtime_size',
    )
    missing = [setting for setting in required if setting not in config]
    if missing:
        raise AssertionError(
            f'golden cluster lacks shared runtime settings: {missing}'
        )


def assert_rows_close(
    actual: list[list[Any]],
    expected: list[list[Any]],
    *,
    label: str,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            f'{label}: row count differs: {len(actual)} != {len(expected)}'
        )
    for rank, (actual_row, expected_row) in enumerate(
        zip(actual, expected)
    ):
        if actual_row[0] != expected_row[0]:
            raise AssertionError(
                f'{label}: rank {rank} id differs: '
                f'{actual_row[0]!r} != {expected_row[0]!r}'
            )
        if not math.isclose(
            float(actual_row[1]),
            float(expected_row[1]),
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise AssertionError(
                f'{label}: rank {rank} score differs: '
                f'{actual_row[1]} != {expected_row[1]}'
            )


def assert_surface_map(
    actual: dict[str, list[list[Any]]],
    expected: dict[str, list[list[Any]]],
    *,
    label: str,
) -> None:
    for name, expected_rows in expected.items():
        if name not in actual:
            raise AssertionError(f'{label}: missing surface {name!r}')
        assert_rows_close(
            actual[name],
            expected_rows,
            label=f'{label}:{name}',
        )


def setup(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute('CREATE EXTENSION evoke')
        cursor.execute('CREATE SCHEMA parity')
        cursor.execute(
            'CREATE TABLE parity.docs_v3 ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.docs_v3 VALUES (%s, %s)',
            TEXT_ROWS,
        )
        cursor.execute(
            'CREATE INDEX docs_v3_idx ON parity.docs_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.predicate_v3 ('
            'id int PRIMARY KEY, body text[] NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.predicate_v3 VALUES (%s, %s)',
            PREDICATE_ROWS,
        )
        cursor.execute(
            'CREATE INDEX predicate_v3_idx ON parity.predicate_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.ids_v3 ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.ids_v3 VALUES (%s, %s)',
            ID_ROWS,
        )
        cursor.execute(
            'CREATE INDEX ids_v3_idx ON parity.ids_v3 '
            'USING evoke (tokens) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.semantic_docs ('
            'id text PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.semantic_docs VALUES (%s, %s)',
            (
                ('doc-a', 'alpha cuda'),
                ('doc-b', 'semantic gpu'),
                ('doc-c', 'optimization graph'),
            ),
        )
        cursor.execute(
            'CREATE INDEX semantic_docs_idx ON parity.semantic_docs '
            'USING evoke (body) WITH (sae=true, consistency=eventual)'
        )


def collect_semantic_surfaces(
    connection: psycopg.Connection[Any],
    queries: list[str],
) -> dict[str, list[list[Any]]]:
    surfaces: dict[str, list[list[Any]]] = {}
    with connection.cursor() as cursor:
        for query in queries:
            cursor.execute(
                'SELECT source.id, hit.score::float8 '
                'FROM evoke_query('
                "'parity.semantic_docs_idx'::regclass, %s, 3) AS hit "
                'JOIN parity.semantic_docs AS source '
                'ON source.ctid = hit.ctid '
                'ORDER BY hit.score DESC, source.id',
                (query,),
            )
            surfaces[query] = [
                [str(document_id), float(score)]
                for document_id, score in cursor.fetchall()
            ]
    return surfaces


def collect_and_validate(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> dict[str, Any]:
    golden = load_golden()
    bm25 = golden['bm25']
    semantic = golden['semantic']

    manifest_digest = hashlib.sha256(
        (model_path / 'manifest.json').read_bytes()
    ).hexdigest()
    if manifest_digest != semantic['manifest_sha256']:
        raise AssertionError(
            'semantic golden model differs: '
            f'{manifest_digest} != {semantic["manifest_sha256"]}'
        )

    initial = collect_surfaces(connection, version='v3')
    initial['raw:evoke_zero_score_missing'] = [
        [document_id, score]
        for document_id, score in fetch_search(
            connection,
            'parity.docs_v3_idx',
            'evoke_zero_score_missing',
        )
    ]
    assert_surface_map(initial, bm25['initial'], label='initial')

    predicates = {
        query: fetch_predicate_ids(
            connection,
            'parity.predicate_v3',
            query,
        )
        for query in bm25['predicate']
    }
    if predicates != bm25['predicate']:
        raise AssertionError(
            f'predicate golden differs: {predicates} != {bm25["predicate"]}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT evoke_index_runtime_signature_internal("
            "'parity.semantic_docs_idx'::regclass)"
        )
        runtime_signature = str(cursor.fetchone()[0])
    if runtime_signature != semantic['runtime_signature']:
        raise AssertionError(
            'semantic runtime signature differs: '
            f'{runtime_signature} != {semantic["runtime_signature"]}'
        )
    semantic_surfaces = collect_semantic_surfaces(
        connection,
        list(semantic['surfaces']),
    )
    assert_surface_map(
        semantic_surfaces,
        semantic['surfaces'],
        label='semantic',
    )

    acquire_maintenance_guard(connection, 'parity.docs_v3_idx')
    acquire_maintenance_guard(connection, 'parity.ids_v3_idx')
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(7, 'new incremental lexical document')"
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 '
                "SELECT 8, string_agg('l0_unique_' || term_id::text, ' ') "
                'FROM generate_series(1, 2000) AS terms(term_id)'
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(9, 'small inline tail record')"
            )
            cursor.execute(
                'INSERT INTO parity.ids_v3 VALUES (5, ARRAY[4, 4, 8])'
            )

        linked_l0 = {
            'ids:[4, 8]': [
                [document_id, score]
                for document_id, score in fetch_ids(
                    connection,
                    'parity.ids_v3_idx',
                    [4, 8],
                )
            ],
            'raw:"small inline"': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    '"small inline"',
                )
            ],
            'raw:+l0_unique_1999 -missing': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    '+l0_unique_1999 -missing',
                )
            ],
            'raw:evoke_zero_score_missing': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    'evoke_zero_score_missing',
                )
            ],
            'search:incremental lexical': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    'incremental lexical',
                )
            ],
            'search:l0_unique_1999': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    'l0_unique_1999',
                )
            ],
            'search:small inline tail': [
                [document_id, score]
                for document_id, score in fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    'small inline tail',
                )
            ],
        }
        assert_surface_map(
            linked_l0,
            bm25['linked_l0'],
            label='linked-l0',
        )

        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE parity.docs_v3 SET body = "
                "'replacement lexical generation' WHERE id = 7"
            )
            cursor.execute('DELETE FROM parity.docs_v3 WHERE id = 8')
            cursor.execute(
                'UPDATE parity.ids_v3 SET tokens = ARRAY[7, 7, 8] '
                'WHERE id = 5'
            )
            cursor.execute('DELETE FROM parity.ids_v3 WHERE id = 2')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.docs_v3')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.ids_v3')

        mvcc = {
            'ids:[0, 2]': [
                [document_id, score]
                for document_id, score in fetch_ids_ordered(
                    connection,
                    'parity.ids_v3',
                    [0, 2],
                )
            ],
            'ids:[7, 8]': [
                [document_id, score]
                for document_id, score in fetch_ids_ordered(
                    connection,
                    'parity.ids_v3',
                    [7, 8],
                )
            ],
            'text:blue bird river': [
                [document_id, score]
                for document_id, score in fetch_ordered(
                    connection,
                    'parity.docs_v3',
                    'parity.docs_v3_idx',
                    'blue bird river',
                )
            ],
            'text:replacement lexical': [
                [document_id, score]
                for document_id, score in fetch_ordered(
                    connection,
                    'parity.docs_v3',
                    'parity.docs_v3_idx',
                    'replacement lexical',
                )
            ],
        }
        assert_surface_map(mvcc, bm25['mvcc'], label='mvcc')
    finally:
        release_maintenance_guard(connection, 'parity.ids_v3_idx')
        release_maintenance_guard(connection, 'parity.docs_v3_idx')

    return {
        'schema_version': 1,
        'fixture': str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        'manifest_sha256': manifest_digest,
        'runtime_signature': runtime_signature,
        'surfaces': {
            'initial': len(bm25['initial']),
            'predicate': len(predicates),
            'linked_l0': len(linked_l0),
            'mvcc': len(mvcc),
            'semantic': len(semantic_surfaces),
        },
        'passed': True,
    }


def run_golden(args: argparse.Namespace) -> dict[str, Any]:
    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / 'manifest.json').is_file():
        raise FileNotFoundError(f'model manifest is missing: {model_path}')

    pg_bin = args.pg_bin.expanduser().resolve()
    root = create_short_socket_root('evoke-v3-golden-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    port = reserve_port()
    socket_dir.mkdir()
    started = False
    connection: psycopg.Connection[Any] | None = None
    try:
        run(
            [
                str(pg_bin / 'initdb'),
                '-D',
                str(data_dir),
                '-A',
                'trust',
                '-U',
                'postgres',
            ]
        )
        configure_cluster(
            data_dir,
            socket_dir,
            port,
            extension_libdir=extension_libdir,
            system_libdir=pg_config_value(pg_bin, '--pkglibdir'),
            extension_control_dir=extension_control_dir,
            system_sharedir=pg_config_value(pg_bin, '--sharedir'),
        )
        assert_shared_runtime_config(data_dir / 'postgresql.conf')
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as handle:
            escaped_model = str(model_path).replace("'", "''")
            handle.write(f"evoke.sae_model_path = '{escaped_model}'\n")
            handle.write("evoke.control_database = 'template1'\n")
            handle.write('evoke.runtime_worker_count = 2\n')

        start_cluster(pg_bin / 'pg_ctl', data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        setup(connection)
        return collect_and_validate(connection, model_path)
    except Exception:
        if log_path.is_file():
            print(log_path.read_text(encoding='utf-8'))
        raise
    finally:
        if connection is not None:
            connection.close()
        if started:
            stop_cluster(pg_bin / 'pg_ctl', data_dir)
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    report = run_golden(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
