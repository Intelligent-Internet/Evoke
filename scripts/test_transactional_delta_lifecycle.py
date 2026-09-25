#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
    vacuum_with_session_maintenance_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
ROLLBACK_GID = 'evoke_sae_rollback_prepared'
COMMIT_GID = 'evoke_sae_commit_prepared'
BM25_ROLLBACK_GID = 'evoke_bm25_rollback_prepared'
BM25_COMMIT_GID = 'evoke_bm25_commit_prepared'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate relation-owned delta behavior across PostgreSQL 2PC.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def run(command: list[str], *, check: bool = True) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()


def configure_cluster(
    data_dir: Path,
    socket_dir: Path,
    port: int,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'evoke'\n")
        if extension_libdir is not None:
            libdir = str(extension_libdir).replace("'", "''")
            handle.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if extension_control_dir is not None:
            control_dir = str(extension_control_dir).replace("'", "''")
            handle.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
        handle.write("evoke.shared_runtime_size = '64MB'\n")
        # This suite advances maintenance one action at a time. Disable
        # hint-driven workers so they cannot rotate L0 before the test owns
        # the per-index gate; automatic scheduling is covered separately.
        handle.write("evoke.maintenance_worker_limit = 0\n")
        handle.write("evoke.maintenance_timer_interval_ms = '1h'\n")
        handle.write("evoke.maintenance_low_debt_interval_ms = '1000ms'\n")
        handle.write('max_prepared_transactions = 10\n')
        handle.write('max_worker_processes = 16\n')
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{socket_dir}'\n")
        handle.write(f'port = {port}\n')


def connect(
    socket_dir: Path,
    port: int,
    *,
    autocommit: bool = True,
) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=autocommit,
    )


def fetch_json(
    connection: psycopg.Connection[Any],
    query: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(query)
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid JSON result: {row}')
    return dict(row[0])


def index_status(
    connection: psycopg.Connection[Any],
    index_name: str = 'docs_body_idx',
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT evoke_index_status({}::regclass)').format(
                sql.Literal(index_name)
            )
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid index status: {row}')
    return dict(row[0])


def generation_id(status: dict[str, Any]) -> str:
    return str(status['generation']['generation_id'])


def semantic_pending(status: dict[str, Any]) -> int:
    completion = status['generation']['delta']['semantic_completion']
    return int(completion['pending'])


def uses_convergent_segments(status: dict[str, Any]) -> bool:
    return (
        status['generation']['layout'].get('storage')
        == 'convergent_segments'
    )


def semantic_completion(status: dict[str, Any]) -> dict[str, Any]:
    completion = status['generation']['delta']['semantic_completion']
    if not isinstance(completion, dict):
        raise AssertionError(f'invalid semantic completion status: {status}')
    return completion


def query_ids(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    index_name: str = 'docs_body_idx',
    table_name: str = 'docs',
    positive_only: bool = False,
) -> list[str]:
    score_filter = (
        sql.SQL('WHERE hit.score > 0')
        if positive_only
        else sql.SQL('')
    )
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                SELECT source.id
                FROM evoke_query(
                    {}::regclass,
                    %s,
                    20
                ) AS hit
                JOIN {} AS source ON source.ctid = hit.ctid
                {}
                ORDER BY hit.score DESC, source.id
                """
            ).format(
                sql.Literal(index_name),
                sql.Identifier(table_name),
                score_filter,
            ),
            (query,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def acquire_maintenance_lock(
    socket_dir: Path,
    port: int,
    index_name: str = 'docs_body_idx',
) -> psycopg.Connection[Any]:
    connection = connect(socket_dir, port)
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                'SELECT evoke_index_try_maintenance_lock({}::regclass)'
            ).format(sql.Literal(index_name))
        )
        row = cursor.fetchone()
    if row is None or row[0] is not True:
        connection.close()
        raise AssertionError('could not acquire the maintenance gate')
    return connection


def wait_for_maintenance_lock(
    socket_dir: Path,
    port: int,
    index_name: str,
    *,
    max_attempts: int = 100,
    delay_seconds: float = 0.05,
) -> psycopg.Connection[Any]:
    for attempt in range(max_attempts):
        try:
            return acquire_maintenance_lock(
                socket_dir,
                port,
                index_name,
            )
        except AssertionError:
            if attempt + 1 == max_attempts:
                raise
            time.sleep(delay_seconds)
    raise AssertionError('maintenance lock retry loop exhausted')


def release_maintenance_lock(
    connection: psycopg.Connection[Any],
    index_name: str = 'docs_body_idx',
) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    'SELECT evoke_index_maintenance_unlock({}::regclass)'
                ).format(sql.Literal(index_name))
            )
        connection.commit()
    finally:
        connection.close()


def maintain_until(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected_reason: str,
    *,
    max_attempts: int = 3,
    force_checkpoint: bool = False,
) -> list[str]:
    results: list[str] = []
    local_setting = not connection.autocommit

    if force_checkpoint:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SET {'LOCAL ' if local_setting else ''}"
                "evoke.test_convergent_l0_rotation_records = '1'"
            )
    try:
        for _ in range(max_attempts):
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        'SELECT evoke_index_try_maintain({}::regclass)'
                    ).format(sql.Literal(index_name))
                )
                row = cursor.fetchone()
            if row is None:
                raise AssertionError('maintenance returned no result')
            result = str(row[0])
            results.append(result)
            if expected_reason in result:
                break
    finally:
        if force_checkpoint and not local_setting:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )
    return results


def maintain_due_for_index(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT {}::regclass::oid').format(
                sql.Literal(index_name)
            )
        )
        target_row = cursor.fetchone()
        if target_row is None:
            raise AssertionError('maintenance target returned no OID')
        target_oid = int(target_row[0])
        cursor.execute(
            'SELECT index_oid::oid, result '
            'FROM evoke_index_maintain_due(256)'
        )
        rows = cursor.fetchall()
    for index_oid, result in rows:
        if int(index_oid) == target_oid:
            return str(result)
    return None


def maintain_until_converged(
    connection: psycopg.Connection[Any],
    index_name: str,
    *,
    max_attempts: int = 100,
) -> tuple[list[str], list[dict[str, Any]]]:
    results: list[str] = []
    statuses: list[dict[str, Any]] = []
    local_setting = not connection.autocommit

    with connection.cursor() as cursor:
        cursor.execute(
            f"SET {'LOCAL ' if local_setting else ''}"
            "evoke.test_convergent_l0_rotation_records = '1'"
        )
    try:
        for _ in range(max_attempts):
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        'SELECT evoke_index_maintain({}::regclass)'
                    ).format(sql.Literal(index_name))
                )
                row = cursor.fetchone()
            if row is None:
                raise AssertionError('maintenance returned no result')
            results.append(str(row[0]))
            status = index_status(connection, index_name)
            statuses.append(status)
            if (
                semantic_pending(status) == 0
                and int(status['details']['delta_records']) == 0
            ):
                break
            if 'reason=xid_horizon' in results[-1]:
                time.sleep(0.05)
        else:
            raise AssertionError(
                f'{index_name} did not converge after {max_attempts} '
                f'attempts: {results}'
            )
    finally:
        if not local_setting:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

    return results, statuses


def prepared_gids(connection: psycopg.Connection[Any]) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT gid
            FROM pg_catalog.pg_prepared_xacts
            WHERE gid LIKE 'evoke_%'
            ORDER BY gid
            """
        )
        return [str(row[0]) for row in cursor.fetchall()]


def prepare_insert(
    connection: psycopg.Connection[Any],
    *,
    gid: str,
    doc_id: str,
    body: str,
    table_name: str = 'docs',
    aux: int | None = None,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        if aux is None:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO {} (id, body) VALUES (%s, %s)'
                ).format(sql.Identifier(table_name)),
                (doc_id, body),
            )
        else:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO {} (id, aux, body) VALUES (%s, %s, %s)'
                ).format(sql.Identifier(table_name)),
                (doc_id, aux, body),
            )
        cursor.execute(
            sql.SQL('PREPARE TRANSACTION {}').format(sql.Literal(gid))
        )


def main() -> None:
    args = parse_args()
    pg_bin = args.pg_bin.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.expanduser().resolve()
        libraries = [
            args.extension_libdir / name
            for name in ('evoke.so', 'evoke.dylib')
        ]
        if not any(path.is_file() for path in libraries):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    for executable in (initdb, pg_ctl, pg_bin / 'postgres'):
        if not executable.is_file():
            raise FileNotFoundError(
                f'missing PostgreSQL executable: {executable}'
            )
    if not (model_path / 'manifest.json').is_file():
        raise FileNotFoundError(f'missing model checkout: {model_path}')

    socket_root = create_short_socket_root('evoke-2pc-')
    evidence: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    try:
        with tempfile.TemporaryDirectory(prefix='evoke-2pc-data-') as temp:
            root = Path(temp)
            data_dir = root / 'data'
            socket_dir = socket_root / 's'
            log_path = root / 'postgres.log'
            socket_dir.mkdir()
            port = reserve_port()
            started = False
            lock_connection: psycopg.Connection[Any] | None = None
            try:
                run([
                    str(initdb),
                    '-D',
                    str(data_dir),
                    '-A',
                    'trust',
                    '-U',
                    'postgres',
                ])
                configure_cluster(
                    data_dir,
                    socket_dir,
                    port,
                    args.extension_libdir,
                    args.extension_control_dir,
                )
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    'start',
                    '-w',
                ])
                started = True
                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute('CREATE EXTENSION evoke')
                        cursor.execute(
                            """
                            CREATE TABLE docs (
                                id text PRIMARY KEY,
                                body text NOT NULL
                            );
                            INSERT INTO docs VALUES
                                ('base-a', 'stable baseline alpha'),
                                ('base-b', 'stable baseline beta');
                            """
                        )
                        cursor.execute(
                            sql.SQL(
                                """
                                CREATE INDEX docs_body_idx
                                ON docs USING evoke (body)
                                WITH (
                                    sae = true,
                                    model_path = {}
                                )
                                """
                            ).format(sql.Literal(str(model_path)))
                        )
                    initial_status = index_status(connection)
                initial_generation = generation_id(initial_status)
                sae_convergent = uses_convergent_segments(initial_status)

                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'docs_body_idx',
                )
                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE bm25_periodic_docs (
                                id text PRIMARY KEY,
                                body text NOT NULL
                            );
                            INSERT INTO bm25_periodic_docs VALUES
                                ('periodic-base', 'periodic baseline token');
                            CREATE INDEX bm25_periodic_docs_body_idx
                                ON bm25_periodic_docs USING evoke (body)
                                WITH (consistency = realtime);
                            INSERT INTO bm25_periodic_docs VALUES
                                ('periodic-new', 'periodicuniquesentinel');
                            """
                        )
                        cursor.execute(
                            "SELECT evoke_index_try_maintain("
                            "'bm25_periodic_docs_body_idx'::regclass)"
                        )
                        before_interval_result = str(cursor.fetchone()[0])
                    before_interval_status = index_status(
                        connection,
                        'bm25_periodic_docs_body_idx',
                    )
                    before_interval_ids = query_ids(
                        connection,
                        'periodicuniquesentinel',
                        index_name='bm25_periodic_docs_body_idx',
                        table_name='bm25_periodic_docs',
                        positive_only=True,
                    )
                    time.sleep(1.2)
                    periodic_rotation = maintain_due_for_index(
                        connection,
                        'bm25_periodic_docs_body_idx',
                    )
                    after_rotation_status = index_status(
                        connection,
                        'bm25_periodic_docs_body_idx',
                    )
                    periodic_seal = maintain_due_for_index(
                        connection,
                        'bm25_periodic_docs_body_idx',
                    )
                    after_periodic_status = index_status(
                        connection,
                        'bm25_periodic_docs_body_idx',
                    )
                    after_periodic_ids = query_ids(
                        connection,
                        'periodicuniquesentinel',
                        index_name='bm25_periodic_docs_body_idx',
                        table_name='bm25_periodic_docs',
                        positive_only=True,
                    )
                    with connection.cursor() as cursor:
                        cursor.execute('DROP TABLE bm25_periodic_docs')
                release_maintenance_lock(
                    lock_connection,
                    'docs_body_idx',
                )
                lock_connection = None
                evidence['bm25_periodic_low_debt'] = {
                    'before_interval_result': before_interval_result,
                    'before_interval_status': before_interval_status,
                    'before_interval_ids': before_interval_ids,
                    'periodic_rotation': periodic_rotation,
                    'after_rotation_status': after_rotation_status,
                    'periodic_seal': periodic_seal,
                    'after_periodic_status': after_periodic_status,
                    'after_periodic_ids': after_periodic_ids,
                }
                gates['bm25_sub_threshold_delta_periodically_converges'] = (
                    'reason=no_pending' in before_interval_result
                    and int(
                        before_interval_status['generation']['delta'][
                            'active'
                        ]['records']
                    ) == 1
                    and before_interval_ids == ['periodic-new']
                    and periodic_rotation is not None
                    and 'reason=active_l0_rotated' in periodic_rotation
                    and int(
                        after_rotation_status['generation']['delta'][
                            'pending'
                        ]['records']
                    ) == 1
                    and periodic_seal is not None
                    and 'reason=pending_l0_sealed' in periodic_seal
                    and int(
                        after_periodic_status['details']['delta_records']
                    ) == 0
                    and after_periodic_ids == ['periodic-new']
                )

                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'docs_body_idx',
                )
                with connect(socket_dir, port) as connection:
                    prepare_insert(
                        connection,
                        gid=ROLLBACK_GID,
                        doc_id='rollback-prepared',
                        body='rollback prepared unique semantic sentinel',
                    )
                with connect(socket_dir, port) as observer:
                    rollback_pending_status = index_status(observer)
                    rollback_pending_ids = query_ids(
                        observer,
                        'rollback prepared unique semantic sentinel',
                    )
                    rollback_pending_gids = prepared_gids(observer)
                evidence['rollback_pending_status'] = rollback_pending_status
                evidence['rollback_pending_ids'] = rollback_pending_ids
                evidence['rollback_pending_gids'] = rollback_pending_gids
                if sae_convergent:
                    rollback_completion = semantic_completion(
                        rollback_pending_status
                    )
                    gates['prepare_does_not_publish_generation'] = (
                        generation_id(rollback_pending_status)
                            == initial_generation
                        and int(
                            rollback_pending_status['details'][
                                'delta_records'
                            ]
                        ) == 1
                        and int(
                            rollback_pending_status['details'][
                                'pending_writes'
                            ]
                        ) == 1
                        and ROLLBACK_GID in rollback_pending_gids
                        and 'rollback-prepared' not in rollback_pending_ids
                        and int(rollback_completion['pending']) == 1
                        and int(
                            rollback_completion['unsealed_upserts']
                        ) == 1
                        and rollback_completion['actionable'] is False
                    )
                else:
                    gates['prepare_does_not_publish_generation'] = (
                        generation_id(rollback_pending_status)
                            == initial_generation
                        and int(
                            rollback_pending_status['details'][
                                'delta_records'
                            ]
                        ) == 1
                        and int(
                            rollback_pending_status['details'][
                                'pending_writes'
                            ]
                        ) == 1
                        and ROLLBACK_GID in rollback_pending_gids
                        and 'rollback-prepared' not in rollback_pending_ids
                        and semantic_pending(rollback_pending_status) == 0
                    )

                release_maintenance_lock(lock_connection)
                lock_connection = None
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    'restart',
                    '-m',
                    'fast',
                    '-w',
                ])
                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'docs_body_idx',
                )
                with connect(socket_dir, port) as observer:
                    restart_pending_status = index_status(observer)
                    restart_pending_ids = query_ids(
                        observer,
                        'rollback prepared unique semantic sentinel',
                    )
                    restart_pending_gids = prepared_gids(observer)
                    with observer.cursor() as cursor:
                        cursor.execute(
                            sql.SQL('ROLLBACK PREPARED {}').format(
                                sql.Literal(ROLLBACK_GID)
                            )
                        )
                    rollback_status = index_status(observer)
                    rollback_ids = query_ids(
                        observer,
                        'rollback prepared unique semantic sentinel',
                    )
                    rollback_after_gids = prepared_gids(observer)
                evidence['restart_pending_status'] = restart_pending_status
                evidence['restart_pending_ids'] = restart_pending_ids
                evidence['restart_pending_gids'] = restart_pending_gids
                evidence['rollback_status'] = rollback_status
                evidence['rollback_ids'] = rollback_ids
                evidence['rollback_after_gids'] = rollback_after_gids
                if sae_convergent:
                    restart_completion = semantic_completion(
                        restart_pending_status
                    )
                    rollback_completion = semantic_completion(
                        rollback_status
                    )
                    gates[
                        'prepared_rollback_survives_restart_and_stays_hidden'
                    ] = (
                        generation_id(restart_pending_status)
                            == initial_generation
                        and ROLLBACK_GID in restart_pending_gids
                        and 'rollback-prepared' not in restart_pending_ids
                        and generation_id(rollback_status)
                            == initial_generation
                        and int(
                            rollback_status['details']['delta_records']
                        ) == 1
                        and int(
                            rollback_status['details']['pending_writes']
                        ) == 1
                        and 'rollback-prepared' not in rollback_ids
                        and ROLLBACK_GID not in rollback_after_gids
                        and int(restart_completion['pending']) == 1
                        and int(rollback_completion['pending']) == 1
                        and restart_completion['actionable'] is False
                        and rollback_completion['actionable'] is False
                    )
                else:
                    gates[
                        'prepared_rollback_survives_restart_and_stays_hidden'
                    ] = (
                        generation_id(restart_pending_status)
                            == initial_generation
                        and ROLLBACK_GID in restart_pending_gids
                        and 'rollback-prepared' not in restart_pending_ids
                        and generation_id(rollback_status)
                            == initial_generation
                        and int(
                            rollback_status['details']['delta_records']
                        ) == 1
                        and int(
                            rollback_status['details']['pending_writes']
                        ) == 1
                        and 'rollback-prepared' not in rollback_ids
                        and semantic_pending(restart_pending_status) == 0
                        and semantic_pending(rollback_status) == 0
                    )

                with connect(socket_dir, port) as connection:
                    prepare_insert(
                        connection,
                        gid=COMMIT_GID,
                        doc_id='commit-prepared',
                        body='commit prepared unique semantic sentinel',
                    )
                with connect(socket_dir, port) as observer:
                    commit_pending_status = index_status(observer)
                    commit_pending_ids = query_ids(
                        observer,
                        'commit prepared unique semantic sentinel',
                    )
                    with observer.cursor() as cursor:
                        cursor.execute(
                            sql.SQL('COMMIT PREPARED {}').format(
                                sql.Literal(COMMIT_GID)
                            )
                        )
                    committed_status = index_status(observer)
                    committed_ids = query_ids(
                        observer,
                        'commit prepared unique semantic sentinel',
                    )
                evidence['commit_pending_status'] = commit_pending_status
                evidence['commit_pending_ids'] = commit_pending_ids
                evidence['committed_status'] = committed_status
                evidence['committed_ids'] = committed_ids
                if sae_convergent:
                    commit_pending_completion = semantic_completion(
                        commit_pending_status
                    )
                    committed_completion = semantic_completion(
                        committed_status
                    )
                    gates['prepared_commit_becomes_visible_through_delta'] = (
                        generation_id(commit_pending_status)
                            == initial_generation
                        and int(
                            commit_pending_status['details']['delta_records']
                        ) == 2
                        and 'commit-prepared' not in commit_pending_ids
                        and generation_id(committed_status)
                            == initial_generation
                        and int(
                            committed_status['details']['delta_records']
                        ) == 2
                        and int(
                            committed_status['details']['pending_writes']
                        ) == 2
                        and int(commit_pending_completion['pending']) == 2
                        and int(committed_completion['pending']) == 2
                        and commit_pending_completion['actionable'] is False
                        and committed_completion['actionable'] is False
                        and bool(committed_ids)
                        and committed_ids[0] == 'commit-prepared'
                    )
                else:
                    gates['prepared_commit_becomes_visible_through_delta'] = (
                        generation_id(commit_pending_status)
                            == initial_generation
                        and int(
                            commit_pending_status['details']['delta_records']
                        ) == 2
                        and 'commit-prepared' not in commit_pending_ids
                        and generation_id(committed_status)
                            == initial_generation
                        and int(
                            committed_status['details']['delta_records']
                        ) == 2
                        and int(
                            committed_status['details']['pending_writes']
                        ) == 2
                        and semantic_pending(commit_pending_status) == 0
                        and semantic_pending(committed_status) == 1
                        and bool(committed_ids)
                        and committed_ids[0] == 'commit-prepared'
                    )

                maintain_results, maintain_statuses = (
                    maintain_until_converged(
                        lock_connection,
                        'docs_body_idx',
                    )
                )
                compacted_status = index_status(lock_connection)
                compacted_commit_ids = query_ids(
                    lock_connection,
                    'commit prepared unique semantic sentinel',
                )
                compacted_rollback_ids = query_ids(
                    lock_connection,
                    'rollback prepared unique semantic sentinel',
                )
                release_maintenance_lock(lock_connection)
                lock_connection = None
                evidence['maintain_results'] = maintain_results
                evidence['maintain_statuses'] = maintain_statuses
                evidence['compacted_status'] = compacted_status
                evidence['compacted_commit_ids'] = compacted_commit_ids
                evidence['compacted_rollback_ids'] = compacted_rollback_ids
                completion_steps = [
                    index
                    for index, result in enumerate(maintain_results)
                    if 'mode=semantic_completion' in result
                ]
                clean_steps = [
                    index
                    for index, status in enumerate(maintain_statuses)
                    if (
                        semantic_pending(status) == 0
                        and int(status['details']['delta_records']) == 0
                    )
                ]
                if sae_convergent:
                    prerequisite_seal_steps = [
                        index
                        for index, result in enumerate(maintain_results)
                        if 'mode=segment_seal' in result
                    ]
                    optional_steps = [
                        index
                        for index, result in enumerate(maintain_results)
                        if (
                            'mode=segment_compaction' in result
                            or 'mode=term_fold' in result
                            or 'mode=workload_fold' in result
                        )
                    ]
                    gates[
                        'eventual_2pc_completion_precedes_compaction'
                    ] = (
                        bool(completion_steps)
                        and bool(clean_steps)
                        and bool(prerequisite_seal_steps)
                        and prerequisite_seal_steps[0]
                            < completion_steps[0]
                        and completion_steps[0] < clean_steps[0]
                        and not any(
                            step < completion_steps[0]
                            for step in optional_steps
                        )
                        and semantic_pending(
                            maintain_statuses[completion_steps[0]]
                        ) == 1
                        and int(
                            maintain_statuses[completion_steps[0]][
                                'details'
                            ]['delta_records']
                        ) > 0
                        and semantic_pending(compacted_status) == 0
                        and int(
                            compacted_status['details']['delta_records']
                        ) == 0
                    )
                else:
                    gates[
                        'eventual_2pc_completion_precedes_compaction'
                    ] = (
                        bool(completion_steps)
                        and bool(clean_steps)
                        and completion_steps[0] < clean_steps[0]
                        and semantic_pending(
                            maintain_statuses[completion_steps[0]]
                        ) == 0
                        and int(
                            maintain_statuses[completion_steps[0]][
                                'details'
                            ]['delta_records']
                        ) > 0
                        and generation_id(
                            maintain_statuses[completion_steps[0]]
                        ) == initial_generation
                    )
                gates['maintenance_reclaims_prepared_transaction_debt'] = (
                    generation_id(compacted_status) != initial_generation
                    and int(compacted_status['details']['delta_records']) == 0
                    and int(compacted_status['details']['pending_writes']) == 0
                    and int(compacted_status['details']['pending_deletes']) == 0
                    and int(compacted_status['generation']['docs']) == 3
                    and bool(compacted_commit_ids)
                    and compacted_commit_ids[0] == 'commit-prepared'
                    and 'rollback-prepared' not in compacted_rollback_ids
                )

                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'docs_body_idx',
                )
                with connect(socket_dir, port) as writer:
                    with writer.cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO docs (id, body)
                            VALUES (
                                'crash-pending',
                                'crash pending lexical sentinel'
                            )
                            """
                        )
                with connect(socket_dir, port) as observer:
                    crash_pending_before = index_status(observer)
                    crash_pending_before_ids = query_ids(
                        observer,
                        'crash pending lexical sentinel',
                    )
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    'stop',
                    '-m',
                    'immediate',
                    '-w',
                ])
                started = False
                lock_connection.close()
                lock_connection = None
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    'start',
                    '-w',
                ])
                started = True
                with connect(socket_dir, port) as observer:
                    crash_pending_restart = index_status(observer)
                    crash_pending_restart_ids = query_ids(
                        observer,
                        'crash pending lexical sentinel',
                    )
                    (
                        crash_pending_maintain_results,
                        crash_pending_maintain_statuses,
                    ) = maintain_until_converged(
                        observer,
                        'docs_body_idx',
                    )
                    crash_pending_compacted = index_status(observer)
                    crash_pending_compacted_ids = query_ids(
                        observer,
                        'crash pending lexical sentinel',
                    )
                evidence['crash_pending_before'] = crash_pending_before
                evidence['crash_pending_before_ids'] = (
                    crash_pending_before_ids
                )
                evidence['crash_pending_restart'] = crash_pending_restart
                evidence['crash_pending_restart_ids'] = (
                    crash_pending_restart_ids
                )
                evidence['crash_pending_maintain_results'] = (
                    crash_pending_maintain_results
                )
                evidence['crash_pending_maintain_statuses'] = (
                    crash_pending_maintain_statuses
                )
                evidence['crash_pending_compacted'] = crash_pending_compacted
                evidence['crash_pending_compacted_ids'] = (
                    crash_pending_compacted_ids
                )
                gates['eventual_pending_survives_immediate_crash'] = (
                    generation_id(crash_pending_before)
                        == generation_id(compacted_status)
                    and semantic_pending(crash_pending_before) == 1
                    and 'crash-pending' in crash_pending_before_ids
                    and generation_id(crash_pending_restart)
                        == generation_id(crash_pending_before)
                    and semantic_pending(crash_pending_restart) == 1
                    and 'crash-pending' in crash_pending_restart_ids
                    and semantic_pending(crash_pending_compacted) == 0
                    and int(
                        crash_pending_compacted['details']['delta_records']
                    ) == 0
                    and generation_id(crash_pending_compacted)
                        != generation_id(crash_pending_restart)
                    and 'crash-pending' in crash_pending_compacted_ids
                )
                sae_final_status = crash_pending_compacted

                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE bm25_docs (
                                id text PRIMARY KEY,
                                aux integer NOT NULL,
                                body text
                            );
                            INSERT INTO bm25_docs VALUES
                                (
                                    'base-a',
                                    1,
                                    'stablealphaunique'
                                ),
                                (
                                    'base-b',
                                    2,
                                    'stablebetaunique'
                                );
                            CREATE UNIQUE INDEX bm25_docs_aux_idx
                                ON bm25_docs (aux);
                            CREATE INDEX bm25_docs_body_idx
                                ON bm25_docs USING evoke (body)
                                WITH (consistency = realtime);
                            """
                        )
                    bm25_initial = index_status(
                        connection,
                        'bm25_docs_body_idx',
                    )
                bm25_initial_generation = generation_id(bm25_initial)
                bm25_convergent = uses_convergent_segments(bm25_initial)

                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE mvcc_docs (
                                id text PRIMARY KEY,
                                body text NOT NULL
                            );
                            INSERT INTO mvcc_docs VALUES
                                ('mvcc-row', 'mvccoldtoken');
                            CREATE INDEX mvcc_docs_body_idx
                                ON mvcc_docs USING evoke (body)
                                WITH (consistency = realtime);
                            """
                        )
                    mvcc_initial = index_status(
                        connection,
                        'mvcc_docs_body_idx',
                    )
                mvcc_initial_generation = generation_id(mvcc_initial)
                mvcc_convergent = uses_convergent_segments(mvcc_initial)
                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'mvcc_docs_body_idx',
                )
                snapshot_reader = connect(
                    socket_dir,
                    port,
                    autocommit=False,
                )
                try:
                    with snapshot_reader.cursor() as cursor:
                        cursor.execute(
                            'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
                        )
                    mvcc_old_before_update = query_ids(
                        snapshot_reader,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    with connect(socket_dir, port) as writer:
                        with writer.cursor() as cursor:
                            cursor.execute(
                                """
                                UPDATE mvcc_docs
                                SET body = 'mvccnewtoken'
                                WHERE id = 'mvcc-row'
                                """
                            )
                    with connect(socket_dir, port) as observer:
                        mvcc_before_fold = index_status(
                            observer,
                            'mvcc_docs_body_idx',
                        )
                    mvcc_fold_results = maintain_until(
                        lock_connection,
                        'mvcc_docs_body_idx',
                        (
                            'reason=xid_horizon'
                            if mvcc_convergent
                            else 'lightweight_delta_fold'
                        ),
                        force_checkpoint=mvcc_convergent,
                    )
                    mvcc_fold_result = mvcc_fold_results[-1]
                    mvcc_after_fold = index_status(
                        lock_connection,
                        'mvcc_docs_body_idx',
                    )
                    mvcc_fresh_old_after_fold = query_ids(
                        lock_connection,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_fresh_new_after_fold = query_ids(
                        lock_connection,
                        'mvccnewtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_old_after_fold = query_ids(
                        snapshot_reader,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_new_in_old_snapshot = query_ids(
                        snapshot_reader,
                        'mvccnewtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    snapshot_reader.rollback()
                finally:
                    snapshot_reader.close()
                evidence['mvcc_initial'] = mvcc_initial
                evidence['mvcc_old_before_update'] = mvcc_old_before_update
                evidence['mvcc_before_fold'] = mvcc_before_fold
                evidence['mvcc_fold_results'] = mvcc_fold_results
                evidence['mvcc_fold_result'] = mvcc_fold_result
                evidence['mvcc_after_fold'] = mvcc_after_fold
                evidence['mvcc_old_after_fold'] = mvcc_old_after_fold
                evidence['mvcc_new_in_old_snapshot'] = (
                    mvcc_new_in_old_snapshot
                )
                evidence['mvcc_fresh_old_after_fold'] = (
                    mvcc_fresh_old_after_fold
                )
                evidence['mvcc_fresh_new_after_fold'] = (
                    mvcc_fresh_new_after_fold
                )
                if mvcc_convergent:
                    gates['bm25_fold_preserves_long_snapshot_versions'] = (
                        mvcc_old_before_update == ['mvcc-row']
                        and generation_id(mvcc_before_fold)
                            == mvcc_initial_generation
                        and int(
                            mvcc_before_fold['details']['delta_records']
                        ) == 1
                        and any(
                            'mode=segment_rotation' in item
                            for item in mvcc_fold_results
                        )
                        and 'reason=xid_horizon' in mvcc_fold_result
                        and generation_id(mvcc_after_fold)
                            == mvcc_initial_generation
                        and int(mvcc_after_fold['generation']['docs']) == 1
                        and int(
                            mvcc_after_fold['details']['delta_records']
                        ) == 1
                        and mvcc_old_after_fold == ['mvcc-row']
                        and not mvcc_new_in_old_snapshot
                        and not mvcc_fresh_old_after_fold
                        and mvcc_fresh_new_after_fold == ['mvcc-row']
                    )
                else:
                    gates['bm25_fold_preserves_long_snapshot_versions'] = (
                        mvcc_old_before_update == ['mvcc-row']
                        and generation_id(mvcc_before_fold)
                            == mvcc_initial_generation
                        and int(
                            mvcc_before_fold['details']['delta_records']
                        ) == 1
                        and any(
                            'reason=shared_delta' in item
                            for item in mvcc_fold_results[:-1]
                        )
                        and 'lightweight_delta_fold' in mvcc_fold_result
                        and generation_id(mvcc_after_fold)
                            != mvcc_initial_generation
                        and int(mvcc_after_fold['generation']['docs']) == 2
                        and int(
                            mvcc_after_fold['details']['delta_records']
                        ) == 0
                        and mvcc_old_after_fold == ['mvcc-row']
                        and not mvcc_new_in_old_snapshot
                        and not mvcc_fresh_old_after_fold
                        and mvcc_fresh_new_after_fold == ['mvcc-row']
                    )

                with connect(socket_dir, port) as connection:
                    vacuum_with_session_maintenance_lock(
                        lock_connection,
                        'mvcc_docs',
                    )
                    mvcc_after_vacuum = index_status(
                        lock_connection,
                        'mvcc_docs_body_idx',
                    )
                    if mvcc_convergent:
                        (
                            mvcc_reclaim_results,
                            mvcc_reclaim_statuses,
                        ) = maintain_until_converged(
                            lock_connection,
                            'mvcc_docs_body_idx',
                        )
                    else:
                        mvcc_reclaim_results = maintain_until(
                            lock_connection,
                            'mvcc_docs_body_idx',
                            'lightweight_delta_fold',
                        )
                        mvcc_reclaim_statuses = []
                    mvcc_reclaim_result = mvcc_reclaim_results[-1]
                    mvcc_after_reclaim = index_status(
                        lock_connection,
                        'mvcc_docs_body_idx',
                    )
                    mvcc_old_after_reclaim = query_ids(
                        lock_connection,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_new_after_reclaim = query_ids(
                        lock_connection,
                        'mvccnewtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    with lock_connection.cursor() as cursor:
                        cursor.execute('REINDEX INDEX mvcc_docs_body_idx')
                    mvcc_after_reindex = index_status(
                        lock_connection,
                        'mvcc_docs_body_idx',
                    )
                    mvcc_old_after_reindex = query_ids(
                        lock_connection,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_new_after_reindex = query_ids(
                        lock_connection,
                        'mvccnewtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                evidence['mvcc_after_vacuum'] = mvcc_after_vacuum
                evidence['mvcc_reclaim_results'] = mvcc_reclaim_results
                evidence['mvcc_reclaim_statuses'] = mvcc_reclaim_statuses
                evidence['mvcc_reclaim_result'] = mvcc_reclaim_result
                evidence['mvcc_after_reclaim'] = mvcc_after_reclaim
                evidence['mvcc_old_after_reclaim'] = mvcc_old_after_reclaim
                evidence['mvcc_new_after_reclaim'] = mvcc_new_after_reclaim
                evidence['mvcc_after_reindex'] = mvcc_after_reindex
                evidence['mvcc_old_after_reindex'] = mvcc_old_after_reindex
                evidence['mvcc_new_after_reindex'] = mvcc_new_after_reindex
                if mvcc_convergent:
                    gates['bm25_vacuum_controls_physical_reclamation'] = (
                        int(
                            mvcc_after_vacuum['details']['pending_deletes']
                        ) >= 1
                        and any(
                            'mode=segment_seal' in item
                            for item in mvcc_reclaim_results
                        )
                        and int(
                            mvcc_after_reclaim['generation']['docs']
                        ) == 1
                        and int(
                            mvcc_after_reclaim['details']['delta_records']
                        ) == 0
                        and int(
                            mvcc_after_reclaim['details']['pending_writes']
                        ) == 0
                        and int(
                            mvcc_after_reclaim['details']['pending_deletes']
                        ) == 0
                        and not mvcc_old_after_reclaim
                        and mvcc_new_after_reclaim == ['mvcc-row']
                        and int(
                            mvcc_after_reindex['generation']['docs']
                        ) == 1
                        and not mvcc_old_after_reindex
                        and mvcc_new_after_reindex == ['mvcc-row']
                    )
                else:
                    gates['bm25_vacuum_controls_physical_reclamation'] = (
                        int(
                            mvcc_after_vacuum['details']['pending_deletes']
                        ) >= 1
                        and any(
                            'reason=shared_delta' in item
                            for item in mvcc_reclaim_results[:-1]
                        )
                        and 'lightweight_delta_fold' in mvcc_reclaim_result
                        and int(
                            mvcc_after_reclaim['generation']['docs']
                        ) == 1
                        and int(
                            mvcc_after_reclaim['details']['delta_records']
                        ) == 0
                        and not mvcc_old_after_reclaim
                        and mvcc_new_after_reclaim == ['mvcc-row']
                        and int(
                            mvcc_after_reindex['generation']['docs']
                        ) == 1
                        and not mvcc_old_after_reindex
                        and mvcc_new_after_reindex == ['mvcc-row']
                    )
                release_maintenance_lock(
                    lock_connection,
                    'mvcc_docs_body_idx',
                )
                lock_connection = None

                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE null_docs (
                                id text PRIMARY KEY,
                                body text
                            );
                            INSERT INTO null_docs VALUES
                                ('null-row', 'nulltransitiontoken');
                            CREATE INDEX null_docs_body_idx
                                ON null_docs USING evoke (body)
                                WITH (consistency = realtime);
                            """
                        )
                    null_initial = index_status(
                        connection,
                        'null_docs_body_idx',
                    )
                null_initial_generation = generation_id(null_initial)
                null_convergent = uses_convergent_segments(null_initial)
                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'null_docs_body_idx',
                )
                null_writer = connect(
                    socket_dir,
                    port,
                    autocommit=False,
                )
                try:
                    with null_writer.cursor() as cursor:
                        cursor.execute(
                            """
                            UPDATE null_docs
                            SET body = NULL
                            WHERE id = 'null-row'
                            """
                        )
                    null_in_xact_status = index_status(
                        null_writer,
                        'null_docs_body_idx',
                    )
                    null_in_xact_ids = query_ids(
                        null_writer,
                        'nulltransitiontoken',
                        index_name='null_docs_body_idx',
                        table_name='null_docs',
                        positive_only=True,
                    )
                    null_writer.commit()
                finally:
                    null_writer.close()
                null_after_commit = index_status(
                    lock_connection,
                    'null_docs_body_idx',
                )
                null_after_commit_ids = query_ids(
                    lock_connection,
                    'nulltransitiontoken',
                    index_name='null_docs_body_idx',
                    table_name='null_docs',
                    positive_only=True,
                )
                null_after_vacuum: dict[str, Any] | None = None
                null_maintain_results: list[str]
                null_maintain_statuses: list[dict[str, Any]]
                if null_convergent:
                    vacuum_with_session_maintenance_lock(
                        lock_connection,
                        'null_docs',
                    )
                    null_after_vacuum = index_status(
                        lock_connection,
                        'null_docs_body_idx',
                    )
                    (
                        null_maintain_results,
                        null_maintain_statuses,
                    ) = maintain_until_converged(
                        lock_connection,
                        'null_docs_body_idx',
                    )
                    null_maintain_result = null_maintain_results[-1]
                else:
                    with lock_connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT evoke_index_try_maintain("
                            "'null_docs_body_idx'::regclass)"
                        )
                        null_maintain_result = str(cursor.fetchone()[0])
                    null_maintain_results = [null_maintain_result]
                    null_maintain_statuses = []
                null_after_maintain = index_status(
                    lock_connection,
                    'null_docs_body_idx',
                )
                with lock_connection.cursor() as cursor:
                    cursor.execute('REINDEX INDEX null_docs_body_idx')
                null_after_reindex = index_status(
                    lock_connection,
                    'null_docs_body_idx',
                )
                null_after_reindex_ids = query_ids(
                    lock_connection,
                    'nulltransitiontoken',
                    index_name='null_docs_body_idx',
                    table_name='null_docs',
                    positive_only=True,
                )
                evidence['null_initial'] = null_initial
                evidence['null_in_xact_status'] = null_in_xact_status
                evidence['null_in_xact_ids'] = null_in_xact_ids
                evidence['null_after_commit'] = null_after_commit
                evidence['null_after_commit_ids'] = null_after_commit_ids
                evidence['null_after_vacuum'] = null_after_vacuum
                evidence['null_maintain_results'] = null_maintain_results
                evidence['null_maintain_statuses'] = (
                    null_maintain_statuses
                )
                evidence['null_maintain_result'] = null_maintain_result
                evidence['null_after_maintain'] = null_after_maintain
                evidence['null_after_reindex'] = null_after_reindex
                evidence['null_after_reindex_ids'] = null_after_reindex_ids
                if null_convergent:
                    assert null_after_vacuum is not None
                    gates[
                        'bm25_null_transition_uses_exact_zero_record_overlay'
                    ] = (
                        generation_id(null_in_xact_status)
                            == null_initial_generation
                        and int(
                            null_in_xact_status['details']['delta_records']
                        ) == 0
                        and int(
                            null_in_xact_status['details']['pending_writes']
                        ) == 0
                        and not bool(
                            null_in_xact_status['details']['stale']
                        )
                        and not null_in_xact_ids
                        and generation_id(null_after_commit)
                            == null_initial_generation
                        and int(
                            null_after_commit['details']['delta_records']
                        ) == 0
                        and int(
                            null_after_commit['details']['pending_writes']
                        ) == 0
                        and not null_after_commit_ids
                        and int(
                            null_after_vacuum['details']['pending_deletes']
                        ) >= 1
                        and int(
                            null_after_vacuum['details']['delta_records']
                        ) >= 1
                        and any(
                            'mode=segment_seal' in item
                            for item in null_maintain_results
                        )
                        and int(
                            null_after_maintain['details']['delta_records']
                        ) == 0
                        and int(
                            null_after_maintain['details']['pending_writes']
                        ) == 0
                        and int(
                            null_after_maintain['details']['pending_deletes']
                        ) == 0
                        and int(
                            null_after_maintain['generation']['docs']
                        ) == 0
                        and int(
                            null_after_reindex['generation']['docs']
                        ) == 0
                        and not null_after_reindex_ids
                    )
                else:
                    gates[
                        'bm25_null_transition_uses_exact_zero_record_overlay'
                    ] = (
                        generation_id(null_in_xact_status)
                            == null_initial_generation
                        and int(
                            null_in_xact_status['details']['delta_records']
                        ) == 0
                        and int(
                            null_in_xact_status['details']['pending_writes']
                        ) == 1
                        and not bool(
                            null_in_xact_status['details']['stale']
                        )
                        and not null_in_xact_ids
                        and generation_id(null_after_commit)
                            == null_initial_generation
                        and int(
                            null_after_commit['details']['delta_records']
                        ) == 0
                        and int(
                            null_after_commit['details']['pending_writes']
                        ) == 1
                        and not null_after_commit_ids
                        and int(
                            null_after_maintain['details']['delta_records']
                        ) == 0
                        and int(
                            null_after_maintain['details']['pending_writes']
                        ) == 0
                        and int(
                            null_after_maintain['generation']['docs']
                        ) == 0
                        and int(
                            null_after_reindex['generation']['docs']
                        ) == 0
                        and not null_after_reindex_ids
                    )
                release_maintenance_lock(
                    lock_connection,
                    'null_docs_body_idx',
                )
                lock_connection = None

                oversized_body = ' '.join(
                    f'oversizedrecoverabletoken{ordinal:04d}'
                    for ordinal in range(1200)
                )
                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE oversized_docs (
                                id text PRIMARY KEY,
                                body text NOT NULL
                            );
                            INSERT INTO oversized_docs VALUES
                                ('oversized-row', 'oversizedbaselinetoken');
                            CREATE INDEX oversized_docs_body_idx
                                ON oversized_docs USING evoke (body)
                                WITH (consistency = realtime);
                            """
                        )
                    oversized_initial = index_status(
                        connection,
                        'oversized_docs_body_idx',
                    )
                oversized_initial_generation = generation_id(
                    oversized_initial
                )
                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'oversized_docs_body_idx',
                )
                oversized_convergent = (
                    oversized_initial['generation']['layout'].get('storage')
                    == 'convergent_segments'
                )
                oversized_query_error = ''
                oversized_in_xact_ids: list[str] = []
                oversized_writer = connect(
                    socket_dir,
                    port,
                    autocommit=False,
                )
                try:
                    with oversized_writer.cursor() as cursor:
                        cursor.execute(
                            """
                            UPDATE oversized_docs
                            SET body = %s
                            WHERE id = 'oversized-row'
                            """,
                            (oversized_body,),
                        )
                    oversized_in_xact_status = index_status(
                        oversized_writer,
                        'oversized_docs_body_idx',
                    )
                    with oversized_writer.cursor() as cursor:
                        cursor.execute('SAVEPOINT oversized_query')
                    try:
                        oversized_in_xact_ids = query_ids(
                            oversized_writer,
                            'oversizedrecoverabletoken0000',
                            index_name='oversized_docs_body_idx',
                            table_name='oversized_docs',
                            positive_only=True,
                        )
                    except psycopg.Error as error:
                        oversized_query_error = str(error).splitlines()[0]
                        with oversized_writer.cursor() as cursor:
                            cursor.execute(
                                'ROLLBACK TO SAVEPOINT oversized_query'
                            )
                            cursor.execute(
                                'RELEASE SAVEPOINT oversized_query'
                            )
                    else:
                        with oversized_writer.cursor() as cursor:
                            cursor.execute(
                                'RELEASE SAVEPOINT oversized_query'
                            )
                    oversized_after_error = index_status(
                        oversized_writer,
                        'oversized_docs_body_idx',
                    )
                    oversized_writer.commit()
                finally:
                    oversized_writer.close()
                oversized_after_commit = index_status(
                    lock_connection,
                    'oversized_docs_body_idx',
                )
                with lock_connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT evoke_index_try_maintain("
                        "'oversized_docs_body_idx'::regclass)"
                    )
                    oversized_maintain_result = str(cursor.fetchone()[0])
                oversized_after_maintain = index_status(
                    lock_connection,
                    'oversized_docs_body_idx',
                )
                oversized_after_maintain_ids = query_ids(
                    lock_connection,
                    'oversizedrecoverabletoken0000',
                    index_name='oversized_docs_body_idx',
                    table_name='oversized_docs',
                    positive_only=True,
                )
                with lock_connection.cursor() as cursor:
                    cursor.execute(
                        'REINDEX INDEX oversized_docs_body_idx'
                    )
                oversized_after_reindex = index_status(
                    lock_connection,
                    'oversized_docs_body_idx',
                )
                oversized_after_reindex_ids = query_ids(
                    lock_connection,
                    'oversizedrecoverabletoken0000',
                    index_name='oversized_docs_body_idx',
                    table_name='oversized_docs',
                    positive_only=True,
                )
                evidence['oversized_body_bytes'] = len(
                    oversized_body.encode('utf-8')
                )
                evidence['oversized_initial'] = oversized_initial
                evidence['oversized_query_error'] = oversized_query_error
                evidence['oversized_in_xact_ids'] = oversized_in_xact_ids
                evidence['oversized_in_xact_status'] = (
                    oversized_in_xact_status
                )
                evidence['oversized_after_error'] = oversized_after_error
                evidence['oversized_after_commit'] = oversized_after_commit
                evidence['oversized_maintain_result'] = (
                    oversized_maintain_result
                )
                evidence['oversized_after_maintain'] = (
                    oversized_after_maintain
                )
                evidence['oversized_after_maintain_ids'] = (
                    oversized_after_maintain_ids
                )
                evidence['oversized_after_reindex'] = (
                    oversized_after_reindex
                )
                evidence['oversized_after_reindex_ids'] = (
                    oversized_after_reindex_ids
                )
                if oversized_convergent:
                    gates[
                        'bm25_oversized_l0_is_immediately_visible_and_recovers'
                    ] = (
                        len(oversized_body.encode('utf-8')) > 8192
                        and not oversized_query_error
                        and oversized_in_xact_ids == ['oversized-row']
                        and int(
                            oversized_in_xact_status['details'][
                                'delta_records'
                            ]
                        ) > 0
                        and int(
                            oversized_in_xact_status['details'][
                                'pending_writes'
                            ]
                        ) > 0
                        and not bool(
                            oversized_in_xact_status['details']['stale']
                        )
                        and int(
                            oversized_after_commit['details']['delta_records']
                        ) > 0
                        and not bool(
                            oversized_after_commit['details']['stale']
                        )
                        and oversized_after_maintain['query_ready'] is True
                        and oversized_after_maintain_ids == ['oversized-row']
                        and oversized_after_reindex_ids == ['oversized-row']
                    )
                else:
                    gates[
                        'bm25_oversized_delta_fails_closed_then_recovers'
                    ] = (
                        len(oversized_body.encode('utf-8')) > 8192
                        and 'cannot rebuild an evoke index from its writing '
                            'transaction' in oversized_query_error
                        and generation_id(oversized_in_xact_status)
                            == oversized_initial_generation
                        and int(
                            oversized_in_xact_status['details'][
                                'delta_records'
                            ]
                        ) == 0
                        and int(
                            oversized_in_xact_status['details'][
                                'pending_writes'
                            ]
                        ) == 1
                        and bool(
                            oversized_in_xact_status['details']['stale']
                        )
                        and generation_id(oversized_after_error)
                            == oversized_initial_generation
                        and generation_id(oversized_after_commit)
                            == oversized_initial_generation
                        and int(
                            oversized_after_commit['details']['delta_records']
                        ) == 0
                        and int(
                            oversized_after_commit['details']['pending_writes']
                        ) == 1
                        and bool(oversized_after_commit['details']['stale'])
                        and int(
                            oversized_after_maintain['details'][
                                'delta_records'
                            ]
                        ) == 0
                        and int(
                            oversized_after_maintain['details'][
                                'pending_writes'
                            ]
                        ) == 0
                        and not bool(
                            oversized_after_maintain['details']['stale']
                        )
                        and oversized_after_maintain_ids == ['oversized-row']
                        and oversized_after_reindex_ids == ['oversized-row']
                    )
                release_maintenance_lock(
                    lock_connection,
                    'oversized_docs_body_idx',
                )
                lock_connection = None

                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'bm25_docs_body_idx',
                )
                late_precommit_error = ''
                with connect(
                    socket_dir,
                    port,
                    autocommit=False,
                ) as writer:
                    try:
                        with writer.cursor() as cursor:
                            cursor.execute(
                                'SET LOCAL '
                                'evoke.test_precommit_error_after_flush = on'
                            )
                            cursor.execute(
                                """
                                UPDATE bm25_docs
                                SET body =
                                    'abortedprecommitxylophone'
                                WHERE id = 'base-a'
                                """
                            )
                        writer.commit()
                    except Exception as exc:
                        late_precommit_error = str(exc)
                        writer.rollback()
                with connect(socket_dir, port) as observer:
                    bm25_after_late_abort = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    bm25_old_after_abort = query_ids(
                        observer,
                        'stablealphaunique',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    bm25_new_after_abort = query_ids(
                        observer,
                        'abortedprecommitxylophone',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                evidence['bm25_late_precommit_error'] = late_precommit_error
                evidence['bm25_after_late_abort'] = bm25_after_late_abort
                evidence['bm25_old_after_abort'] = bm25_old_after_abort
                evidence['bm25_new_after_abort'] = bm25_new_after_abort
                gates['bm25_late_precommit_abort_keeps_generation_exact'] = (
                    'injected evoke error' in late_precommit_error
                    and generation_id(bm25_after_late_abort)
                        == bm25_initial_generation
                    and int(
                        bm25_after_late_abort['details']['delta_records']
                    ) == 1
                    and int(
                        bm25_after_late_abort['details']['pending_writes']
                    ) == 1
                    and bm25_old_after_abort == ['base-a']
                    and not bm25_new_after_abort
                )

                snapshot_reader = connect(
                    socket_dir,
                    port,
                    autocommit=False,
                )
                try:
                    with snapshot_reader.cursor() as cursor:
                        cursor.execute(
                            'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
                        )
                        cursor.execute('SELECT count(*) FROM bm25_docs')
                        cursor.fetchone()
                    with connect(socket_dir, port) as writer:
                        with writer.cursor() as cursor:
                            cursor.execute(
                                """
                                UPDATE bm25_docs
                                SET body =
                                    'committedsnapshotquasar'
                                WHERE id = 'base-b'
                                """
                            )
                    old_snapshot_ids = query_ids(
                        snapshot_reader,
                        'committedsnapshotquasar',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    snapshot_reader.rollback()
                    fresh_snapshot_ids = query_ids(
                        snapshot_reader,
                        'committedsnapshotquasar',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    snapshot_reader.rollback()
                finally:
                    snapshot_reader.close()
                with connect(socket_dir, port) as observer:
                    bm25_after_snapshot_update = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                evidence['bm25_old_snapshot_ids'] = old_snapshot_ids
                evidence['bm25_fresh_snapshot_ids'] = fresh_snapshot_ids
                evidence['bm25_after_snapshot_update'] = (
                    bm25_after_snapshot_update
                )
                gates['bm25_overlay_cache_is_statement_snapshot_scoped'] = (
                    not old_snapshot_ids
                    and fresh_snapshot_ids == ['base-b']
                    and generation_id(bm25_after_snapshot_update)
                        == bm25_initial_generation
                    and int(
                        bm25_after_snapshot_update['details']['delta_records']
                    ) == 2
                )

                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT ctid::text
                            FROM bm25_docs
                            WHERE id = 'base-b'
                            """
                        )
                        unchanged_tid_before = str(cursor.fetchone()[0])
                        cursor.execute(
                            """
                            UPDATE bm25_docs
                            SET aux = aux + 100
                            WHERE id = 'base-b'
                            """
                        )
                        cursor.execute(
                            """
                            SELECT ctid::text
                            FROM bm25_docs
                            WHERE id = 'base-b'
                            """
                        )
                        unchanged_tid_after = str(cursor.fetchone()[0])
                    unchanged_ids = query_ids(
                        connection,
                        'committedsnapshotquasar',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    bm25_after_unchanged = index_status(
                        connection,
                        'bm25_docs_body_idx',
                    )
                evidence['bm25_unchanged_tid_before'] = unchanged_tid_before
                evidence['bm25_unchanged_tid_after'] = unchanged_tid_after
                evidence['bm25_unchanged_ids'] = unchanged_ids
                evidence['bm25_after_unchanged'] = bm25_after_unchanged
                gates['bm25_nonhot_index_unchanged_carries_new_tid'] = (
                    unchanged_tid_before != unchanged_tid_after
                    and unchanged_ids == ['base-b']
                    and int(
                        bm25_after_unchanged['details']['delta_records']
                    ) == 3
                    and int(
                        bm25_after_unchanged['details']['pending_writes']
                    ) == 3
                )

                with connect(socket_dir, port) as connection:
                    prepare_insert(
                        connection,
                        gid=BM25_ROLLBACK_GID,
                        doc_id='bm25-rollback-prepared',
                        body='rollbackpreparedzephyr',
                        table_name='bm25_docs',
                        aux=1001,
                    )
                with connect(socket_dir, port) as observer:
                    bm25_rollback_pending = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    bm25_pending_gids = prepared_gids(observer)
                    with observer.cursor() as cursor:
                        cursor.execute(
                            sql.SQL('ROLLBACK PREPARED {}').format(
                                sql.Literal(BM25_ROLLBACK_GID)
                            )
                        )
                    bm25_rollback_status = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    bm25_rollback_ids = query_ids(
                        observer,
                        'rollbackpreparedzephyr',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                evidence['bm25_rollback_pending'] = bm25_rollback_pending
                evidence['bm25_rollback_status'] = bm25_rollback_status
                evidence['bm25_rollback_ids'] = bm25_rollback_ids
                gates['bm25_rollback_prepared_stays_hidden'] = (
                    BM25_ROLLBACK_GID in bm25_pending_gids
                    and generation_id(bm25_rollback_pending)
                        == bm25_initial_generation
                    and generation_id(bm25_rollback_status)
                        == bm25_initial_generation
                    and int(
                        bm25_rollback_status['details']['delta_records']
                    ) == 4
                    and not bm25_rollback_ids
                )

                with connect(socket_dir, port) as connection:
                    prepare_insert(
                        connection,
                        gid=BM25_COMMIT_GID,
                        doc_id='bm25-commit-prepared',
                        body='commitpreparednebula',
                        table_name='bm25_docs',
                        aux=1002,
                    )
                with connect(socket_dir, port) as observer:
                    bm25_commit_pending = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    with observer.cursor() as cursor:
                        cursor.execute(
                            sql.SQL('COMMIT PREPARED {}').format(
                                sql.Literal(BM25_COMMIT_GID)
                            )
                        )
                    bm25_committed_status = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    bm25_committed_ids = query_ids(
                        observer,
                        'commitpreparednebula',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                evidence['bm25_commit_pending'] = bm25_commit_pending
                evidence['bm25_committed_status'] = bm25_committed_status
                evidence['bm25_committed_ids'] = bm25_committed_ids
                gates['bm25_commit_prepared_is_immediately_visible'] = (
                    generation_id(bm25_commit_pending)
                        == bm25_initial_generation
                    and generation_id(bm25_committed_status)
                        == bm25_initial_generation
                    and int(
                        bm25_committed_status['details']['delta_records']
                    ) == 5
                    and int(
                        bm25_committed_status['details']['pending_writes']
                    ) == 5
                    and bm25_committed_ids == ['bm25-commit-prepared']
                )

                with connect(socket_dir, port) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM bm25_docs WHERE id = 'base-a'"
                        )
                    vacuum_with_session_maintenance_lock(
                        lock_connection,
                        'bm25_docs',
                    )
                    bm25_after_vacuum = index_status(
                        connection,
                        'bm25_docs_body_idx',
                    )
                    bm25_deleted_ids = query_ids(
                        connection,
                        'stablealphaunique',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                evidence['bm25_after_vacuum'] = bm25_after_vacuum
                evidence['bm25_deleted_ids'] = bm25_deleted_ids
                gates['bm25_realtime_vacuum_records_exact_tombstone'] = (
                    generation_id(bm25_after_vacuum)
                        != bm25_initial_generation
                    and int(
                        bm25_after_vacuum['details']['delta_records']
                    ) == (
                        int(
                            bm25_after_vacuum['details']['pending_writes']
                        )
                        + int(
                            bm25_after_vacuum['details']['pending_deletes']
                        )
                    )
                    and int(
                        bm25_after_vacuum['details']['pending_deletes']
                    ) >= 1
                    and not bm25_deleted_ids
                )

                release_maintenance_lock(
                    lock_connection,
                    'bm25_docs_body_idx',
                )
                lock_connection = None
                with connect(socket_dir, port) as observer:
                    if bm25_convergent:
                        (
                            bm25_maintain_results,
                            bm25_maintain_statuses,
                        ) = maintain_until_converged(
                            observer,
                            'bm25_docs_body_idx',
                        )
                        bm25_maintain_result = bm25_maintain_results[-1]
                    else:
                        with observer.cursor() as cursor:
                            cursor.execute(
                                "SELECT evoke_index_maintain("
                                "'bm25_docs_body_idx'::regclass)"
                            )
                            bm25_maintain_result = str(cursor.fetchone()[0])
                        bm25_maintain_results = [bm25_maintain_result]
                        bm25_maintain_statuses = []
                # Wait for any worker launched before manual convergence, then
                # hold the same per-index fence while observing and crashing.
                # This tests a committed root rather than a concurrent dirty
                # maintenance buffer without forcing a checkpoint.
                lock_connection = wait_for_maintenance_lock(
                    socket_dir,
                    port,
                    'bm25_docs_body_idx',
                )
                bm25_compacted = index_status(
                    lock_connection,
                    'bm25_docs_body_idx',
                )
                bm25_compacted_commit_ids = query_ids(
                    lock_connection,
                    'commitpreparednebula',
                    index_name='bm25_docs_body_idx',
                    table_name='bm25_docs',
                    positive_only=True,
                )
                bm25_compacted_update_ids = query_ids(
                    lock_connection,
                    'committedsnapshotquasar',
                    index_name='bm25_docs_body_idx',
                    table_name='bm25_docs',
                    positive_only=True,
                )
                bm25_compacted_rollback_ids = query_ids(
                    lock_connection,
                    'rollbackpreparedzephyr',
                    index_name='bm25_docs_body_idx',
                    table_name='bm25_docs',
                    positive_only=True,
                )
                evidence['bm25_maintain_result'] = bm25_maintain_result
                evidence['bm25_maintain_results'] = bm25_maintain_results
                evidence['bm25_maintain_statuses'] = (
                    bm25_maintain_statuses
                )
                evidence['bm25_compacted'] = bm25_compacted
                evidence['bm25_compacted_commit_ids'] = (
                    bm25_compacted_commit_ids
                )
                evidence['bm25_compacted_update_ids'] = (
                    bm25_compacted_update_ids
                )
                evidence['bm25_compacted_rollback_ids'] = (
                    bm25_compacted_rollback_ids
                )
                gates['bm25_transaction_debt_compacts_exactly'] = (
                    generation_id(bm25_compacted)
                        != bm25_initial_generation
                    and int(
                        bm25_compacted['details']['delta_records']
                    ) == 0
                    and int(
                        bm25_compacted['details']['pending_writes']
                    ) == 0
                    and int(
                        bm25_compacted['details']['pending_deletes']
                    ) == 0
                    and int(bm25_compacted['generation']['docs']) == 2
                    and bm25_compacted_commit_ids
                        == ['bm25-commit-prepared']
                    and bm25_compacted_update_ids == ['base-b']
                    and not bm25_compacted_rollback_ids
                )

                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    'stop',
                    '-m',
                    'immediate',
                    '-w',
                ])
                started = False
                lock_connection.close()
                lock_connection = None
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    'start',
                    '-w',
                ])
                started = True
                with connect(socket_dir, port) as observer:
                    crash_status = index_status(observer)
                    crash_commit_ids = query_ids(
                        observer,
                        'commit prepared unique semantic sentinel',
                    )
                    crash_rollback_ids = query_ids(
                        observer,
                        'rollback prepared unique semantic sentinel',
                    )
                    bm25_crash_status = index_status(
                        observer,
                        'bm25_docs_body_idx',
                    )
                    bm25_crash_commit_ids = query_ids(
                        observer,
                        'commitpreparednebula',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    bm25_crash_update_ids = query_ids(
                        observer,
                        'committedsnapshotquasar',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    bm25_crash_rollback_ids = query_ids(
                        observer,
                        'rollbackpreparedzephyr',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    bm25_crash_deleted_ids = query_ids(
                        observer,
                        'stablealphaunique',
                        index_name='bm25_docs_body_idx',
                        table_name='bm25_docs',
                        positive_only=True,
                    )
                    mvcc_crash_status = index_status(
                        observer,
                        'mvcc_docs_body_idx',
                    )
                    mvcc_crash_old_ids = query_ids(
                        observer,
                        'mvccoldtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    mvcc_crash_new_ids = query_ids(
                        observer,
                        'mvccnewtoken',
                        index_name='mvcc_docs_body_idx',
                        table_name='mvcc_docs',
                        positive_only=True,
                    )
                    null_crash_status = index_status(
                        observer,
                        'null_docs_body_idx',
                    )
                    null_crash_ids = query_ids(
                        observer,
                        'nulltransitiontoken',
                        index_name='null_docs_body_idx',
                        table_name='null_docs',
                        positive_only=True,
                    )
                    oversized_crash_status = index_status(
                        observer,
                        'oversized_docs_body_idx',
                    )
                    oversized_crash_ids = query_ids(
                        observer,
                        'oversizedrecoverabletoken0000',
                        index_name='oversized_docs_body_idx',
                        table_name='oversized_docs',
                        positive_only=True,
                    )
                    crash_prepared_gids = prepared_gids(observer)
                evidence['crash_status'] = crash_status
                evidence['crash_commit_ids'] = crash_commit_ids
                evidence['crash_rollback_ids'] = crash_rollback_ids
                evidence['bm25_crash_status'] = bm25_crash_status
                evidence['bm25_crash_commit_ids'] = bm25_crash_commit_ids
                evidence['bm25_crash_update_ids'] = bm25_crash_update_ids
                evidence['bm25_crash_rollback_ids'] = (
                    bm25_crash_rollback_ids
                )
                evidence['bm25_crash_deleted_ids'] = bm25_crash_deleted_ids
                evidence['mvcc_crash_status'] = mvcc_crash_status
                evidence['mvcc_crash_old_ids'] = mvcc_crash_old_ids
                evidence['mvcc_crash_new_ids'] = mvcc_crash_new_ids
                evidence['null_crash_status'] = null_crash_status
                evidence['null_crash_ids'] = null_crash_ids
                evidence['oversized_crash_status'] = oversized_crash_status
                evidence['oversized_crash_ids'] = oversized_crash_ids
                evidence['crash_prepared_gids'] = crash_prepared_gids
                gates['compacted_2pc_state_survives_crash_restart'] = (
                    generation_id(crash_status)
                    == generation_id(sae_final_status)
                    and int(crash_status['details']['delta_records']) == 0
                    and bool(crash_commit_ids)
                    and crash_commit_ids[0] == 'commit-prepared'
                    and 'rollback-prepared' not in crash_rollback_ids
                    and not crash_prepared_gids
                )
                gates['bm25_compacted_state_survives_crash_restart'] = (
                    generation_id(bm25_crash_status)
                        == generation_id(bm25_compacted)
                    and int(
                        bm25_crash_status['details']['delta_records']
                    ) == 0
                    and int(
                        bm25_crash_status['details']['pending_writes']
                    ) == 0
                    and int(
                        bm25_crash_status['details']['pending_deletes']
                    ) == 0
                    and bm25_crash_commit_ids
                        == ['bm25-commit-prepared']
                    and bm25_crash_update_ids == ['base-b']
                    and not bm25_crash_rollback_ids
                    and not bm25_crash_deleted_ids
                )
                gates['bm25_mvcc_reclamation_survives_crash_restart'] = (
                    int(mvcc_crash_status['generation']['docs']) == 1
                    and int(
                        mvcc_crash_status['details']['delta_records']
                    ) == 0
                    and not mvcc_crash_old_ids
                    and mvcc_crash_new_ids == ['mvcc-row']
                )
                gates['bm25_counter_only_recovery_survives_crash_restart'] = (
                    int(null_crash_status['generation']['docs']) == 0
                    and int(
                        null_crash_status['details']['delta_records']
                    ) == 0
                    and int(
                        null_crash_status['details']['pending_writes']
                    ) == 0
                    and not null_crash_ids
                    and int(
                        oversized_crash_status['generation']['docs']
                    ) == 1
                    and int(
                        oversized_crash_status['details']['delta_records']
                    ) == 0
                    and int(
                        oversized_crash_status['details']['pending_writes']
                    ) == 0
                    and oversized_crash_ids == ['oversized-row']
                )
            except Exception:
                if log_path.is_file():
                    print(
                        log_path.read_text(
                            encoding='utf-8',
                            errors='replace',
                        ),
                        file=sys.stderr,
                    )
                raise
            finally:
                if lock_connection is not None:
                    lock_connection.close()
                if started:
                    run([
                        str(pg_ctl),
                        '-D',
                        str(data_dir),
                        'stop',
                        '-m',
                        'fast',
                    ], check=False)
    finally:
        shutil.rmtree(socket_root, ignore_errors=True)

    result = {
        'suite': 'transactional_delta_lifecycle',
        'api_version': 'evoke_index_v1',
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'evidence': evidence,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    if not result['all_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
