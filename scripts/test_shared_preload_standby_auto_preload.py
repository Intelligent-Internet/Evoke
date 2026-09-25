#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import psycopg

from evoke_test_support import (
    REPO_ROOT,
    create_short_socket_root,
    ensure_temp_root,
    extension_control_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify convergent standby replay and exact-root auto preload.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--temp-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--keep', action='store_true')
    return parser.parse_args()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return result.stdout


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def append_lines(path: Path, lines: list[str]) -> None:
    content = path.read_text(encoding='utf-8')
    if not content.endswith('\n'):
        content += '\n'
    path.write_text(content + '\n'.join(lines) + '\n', encoding='utf-8')


def wait_ready(dsn: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn) as conn:
                conn.execute('SELECT 1')
                return
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(f'database did not become ready: {last_error}')


def wait_until(
    description: str,
    check: Callable[[], Any],
    timeout_s: float,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f'timeout waiting for {description}: {last_error}')


def write_cluster_config(
    data_dir: Path,
    socket_dir: Path,
    port: int,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    lines = [
        "listen_addresses = ''",
        f"unix_socket_directories = '{socket_dir}'",
        f'port = {port}',
        "shared_preload_libraries = 'evoke'",
        'max_worker_processes = 16',
        "wal_level = 'replica'",
        'max_wal_senders = 5',
        'max_replication_slots = 5',
        'hot_standby = on',
        "evoke.shared_runtime_size = '64MB'",
        "evoke.preload_timer_interval_ms = '1000ms'",
        "evoke.maintenance_timer_interval_ms = '1h'",
        'evoke.maintenance_worker_limit = 1',
    ]
    if extension_libdir is not None:
        libdir = str(extension_libdir).replace("'", "''")
        lines.append(f"dynamic_library_path = '{libdir}:$libdir'")
    if extension_control_dir is not None:
        control_dir = str(extension_control_dir).replace("'", "''")
        lines.append(f"extension_control_path = '{control_dir}:$system'")
    append_lines(data_dir / 'postgresql.conf', lines)


def index_status(
    conn: psycopg.Connection[Any],
    index_name: str = 'docs_bm25_idx',
) -> dict[str, Any]:
    row = conn.execute(
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    ).fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError(f'index status unavailable: {index_name}')
    status = row[0]
    generation = status['generation']
    if (
        not status['index_ready']
        or not status['query_ready']
        or not generation['valid']
        or generation['layout']['storage'] != 'convergent_segments'
        or generation['primary']['storage'] != 'convergent_segments'
    ):
        raise RuntimeError(f'invalid convergent status: {status}')
    return status


def cache_state(conn: psycopg.Connection[Any]) -> str:
    row = conn.execute(
        "SELECT evoke_index_runtime_state('docs_bm25_idx')",
    ).fetchone()
    if row is None:
        raise RuntimeError('runtime state is unavailable')
    return str(row[0])


def assert_page_native_cache_state(state: str) -> None:
    required = (
        'storage=convergent_segments',
        'shared_preload_unified_warm_entries=',
        'shared_preload_hot_fold_entries=',
    )
    retired = (
        'shared_lexical_delta_cache_current',
        'shared_tombstone_current',
        'shared_preload_lexical_delta_cache_entries',
        'shared_preload_tombstone_entries',
    )
    if any(value not in state for value in required):
        raise RuntimeError(f'page-native cache state is incomplete: {state}')
    if any(value in state for value in retired):
        raise RuntimeError(f'legacy mutation cache remains visible: {state}')


def cached_relation_blocks(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute(
        '''
        SELECT count(DISTINCT relblocknumber)
        FROM pg_buffercache
        WHERE reldatabase = (
            SELECT oid FROM pg_database WHERE datname = current_database()
        )
        AND relfilenode = pg_relation_filenode('docs_bm25_idx')
        AND relforknumber = 0
        AND relblocknumber IS NOT NULL
        ''',
    ).fetchone()
    return 0 if row is None else int(row[0])


def query_ids(conn: psycopg.Connection[Any], query: str) -> list[int]:
    rows = conn.execute(
        '''
        SELECT docs.id
        FROM evoke_query('docs_bm25_idx'::regclass, %s, 20) AS hit
        JOIN docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        ''',
        (query,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def generation_id(status: dict[str, Any]) -> int:
    return int(status['generation']['generation'])


def reachable_blocks(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute(
        "SELECT evoke_index_generation_audit_internal("
        "'docs_bm25_idx'::regclass)",
    ).fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError('deep generation audit is unavailable')
    return int(row[0]['primary']['reachable_blocks'])


def wait_for_replay(
    dsn: str,
    expected_generation: int,
    timeout_s: float,
) -> dict[str, Any]:
    def check() -> dict[str, Any] | None:
        with psycopg.connect(dsn, autocommit=True) as conn:
            status = index_status(conn)
            if generation_id(status) < expected_generation:
                return None
            return status

    return wait_until('standby generation replay', check, timeout_s)


def wait_for_auto_preload(
    dsn: str,
    expected_generation: int,
    timeout_s: float,
) -> dict[str, Any]:
    def check() -> dict[str, Any] | None:
        with psycopg.connect(dsn, autocommit=True) as conn:
            status = index_status(conn)
            if generation_id(status) < expected_generation:
                return None
            row = conn.execute(
                "SELECT evoke_index_shared_preload_resident("
                "'docs_bm25_idx')",
            ).fetchone()
            if row is None or row[0] is not True:
                return None
            state = cache_state(conn)
            assert_page_native_cache_state(state)
            cached = cached_relation_blocks(conn)
            wanted = reachable_blocks(conn)
            if cached < wanted:
                return None
            return {
                'generation': generation_id(status),
                'reachable_blocks': wanted,
                'cached_blocks': cached,
                'cache_state': state,
                'status': status,
            }

    return wait_until('standby exact-root auto preload', check, timeout_s)


def verify_standby_maintenance_noops(dsn: str) -> list[str]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            '''
            SELECT pg_is_in_recovery()::text
            UNION ALL
            SELECT evoke_index_try_maintain('docs_bm25_idx')::text
            UNION ALL
            SELECT evoke_index_maintain('docs_bm25_idx')::text
            UNION ALL
            SELECT evoke_index_refresh('docs_bm25_idx')::text
            UNION ALL
            SELECT count(*)::text FROM evoke_index_maintain_due(10)
            ''',
        ).fetchall()
    values = [str(row[0]) for row in rows]
    if values[0] != 'true':
        raise RuntimeError(f'standby is not in recovery: {values}')
    if any('reason=recovery' not in value for value in values[1:4]):
        raise RuntimeError(f'standby maintenance did not no-op: {values}')
    if values[4] != '0':
        raise RuntimeError(f'standby maintain_due returned work: {values}')
    return values


def maintain_primary(
    dsn: str,
    timeout_s: float,
) -> dict[str, Any]:
    results: list[str] = []
    deadline = time.monotonic() + timeout_s
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO docs VALUES (3001, 'standby updated replacement')",
        )
        while time.monotonic() < deadline:
            status = index_status(conn)
            delta = status['generation']['delta']
            if (
                int(delta['records']) == 0
                and int(delta['active']['records']) == 0
                and int(delta['pending']['records']) == 0
            ):
                conn.execute('CHECKPOINT')
                return {'results': results, 'status': status}
            row = conn.execute(
                "SELECT evoke_index_maintain('docs_bm25_idx')",
            ).fetchone()
            if row is None:
                raise RuntimeError('primary maintenance returned no result')
            results.append(str(row[0]))
            if 'reason=lock_busy' in results[-1]:
                time.sleep(0.1)
    raise RuntimeError(f'primary did not converge: {results}')


def explicit_exact_preload(dsn: str) -> dict[str, Any]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        status = index_status(conn)
        wanted = reachable_blocks(conn)
        lock_row = conn.execute(
            "SELECT evoke_index_try_maintenance_lock("
            "'docs_bm25_idx')",
        ).fetchone()
        if lock_row is None or lock_row[0] is not True:
            raise RuntimeError('could not lock standby preload probe')
        try:
            conn.execute('SELECT evoke_runtime_cache_clear()')
            evicted = conn.execute(
                "SELECT * FROM pg_buffercache_evict_relation("
                "'docs_bm25_idx')",
            ).fetchone()
            preload_row = conn.execute(
                "SELECT evoke_index_preload("
                "'docs_bm25_idx')",
            ).fetchone()
            if preload_row is None:
                raise RuntimeError('explicit preload returned no result')
            preload_state = str(preload_row[0])
            warmed_match = re.search(
                r'pages_warmed=([0-9]+)',
                preload_state,
            )
            if warmed_match is None:
                raise RuntimeError(
                    f'explicit preload omitted page count: {preload_state}'
                )
            pages_warmed = int(warmed_match.group(1))
            resident_fold = 'tier=shared_resident_fold' in preload_state
            cached = cached_relation_blocks(conn)
            resident_row = conn.execute(
                "SELECT evoke_index_shared_preload_resident("
                "'docs_bm25_idx')",
            ).fetchone()
            state = cache_state(conn)
            assert_page_native_cache_state(state)
            if resident_fold:
                exact_preload = pages_warmed == 0
            else:
                exact_preload = pages_warmed == wanted and cached == wanted
            if not exact_preload:
                raise RuntimeError(
                    'explicit preload did not publish an exact root path: '
                    f'wanted={wanted}, warmed={pages_warmed}, cached={cached}, '
                    f'evicted={evicted}, result={preload_state}'
                )
            if resident_row is None or resident_row[0] is not True:
                raise RuntimeError('explicit preload did not publish marker')
            return {
                'reachable_blocks': wanted,
                'pages_warmed': pages_warmed,
                'cached_blocks': cached,
                'preload_path': (
                    'shared_resident_fold'
                    if resident_fold
                    else 'exact_page_warm'
                ),
                'eviction': None if evicted is None else list(evicted),
                'result': preload_state,
                'cache_state': state,
            }
        finally:
            conn.execute(
                "SELECT evoke_index_maintenance_unlock("
                "'docs_bm25_idx')",
            )


def stop_cluster(pg_ctl: Path, data_dir: Path) -> None:
    subprocess.run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            '-m',
            'fast',
            '-w',
            'stop',
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def main() -> None:
    args = parse_args()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )

    pg_bin = args.pg_bin.resolve()
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    pg_basebackup = pg_bin / 'pg_basebackup'
    role = os.environ.get('USER') or 'postgres'
    env = os.environ.copy()
    env.setdefault('LC_ALL', 'C')
    temp_root = ensure_temp_root(args.temp_root)
    root = Path(
        tempfile.mkdtemp(prefix='evoke_v3_preload_', dir=temp_root)
    )
    socket_root = create_short_socket_root('evoke_v3_preload_socket_')
    primary_data = root / 'primary'
    standby_data = root / 'standby'
    primary_socket = socket_root / 'p'
    standby_socket = socket_root / 's'
    primary_socket.mkdir()
    standby_socket.mkdir()
    primary_port = free_port()
    standby_port = free_port()
    primary_dsn = (
        f'dbname=postgres user={role} host={primary_socket} '
        f'port={primary_port}'
    )
    standby_dsn = (
        f'dbname=postgres user={role} host={standby_socket} '
        f'port={standby_port}'
    )
    primary_started = False
    standby_started = False
    summary: dict[str, Any] = {
        'api_version': 'evoke_index_v1',
        'suite': 'convergent_standby_auto_preload',
        'checks': {},
    }

    try:
        run(
            [
                str(initdb),
                '-D',
                str(primary_data),
                '-A',
                'trust',
                '-U',
                role,
            ],
            env=env,
        )
        write_cluster_config(
            primary_data,
            primary_socket,
            primary_port,
            args.extension_libdir,
            args.extension_control_dir,
        )
        run(
            [
                str(pg_ctl),
                '-D',
                str(primary_data),
                '-l',
                str(root / 'primary.log'),
                '-w',
                'start',
            ],
            env=env,
        )
        primary_started = True
        wait_ready(primary_dsn, args.timeout)

        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute('CREATE EXTENSION evoke')
            conn.execute('CREATE EXTENSION pg_buffercache')
            conn.execute(
                'CREATE TABLE docs (id int PRIMARY KEY, body text NOT NULL)'
            )
            conn.execute(
                '''
                INSERT INTO docs
                SELECT value, 'standby preload document ' || value::text
                FROM generate_series(1, 2000) AS value
                ''',
            )
            conn.execute(
                '''
                CREATE INDEX docs_bm25_idx
                ON docs USING evoke (body)
                WITH (
                    consistency = 'eventual',
                    auto_preload = 1
                )
                ''',
            )
            conn.execute(
                "INSERT INTO docs VALUES "
                "(2001, 'standby freshdelta replay'), "
                "(2002, 'standby doomeddelta replay')",
            )
            conn.execute('DELETE FROM docs WHERE id = 2002')
            conn.execute('VACUUM docs')
            conn.execute('CHECKPOINT')
            initial_primary_status = index_status(conn)
            initial_generation = generation_id(initial_primary_status)

        run(
            [
                str(pg_basebackup),
                '-D',
                str(standby_data),
                '-h',
                str(primary_socket),
                '-p',
                str(primary_port),
                '-U',
                role,
                '-X',
                'stream',
                '-R',
            ],
            env=env,
        )
        append_lines(
            standby_data / 'postgresql.conf',
            [
                f"unix_socket_directories = '{standby_socket}'",
                f'port = {standby_port}',
                'hot_standby = on',
            ],
        )
        run(
            [
                str(pg_ctl),
                '-D',
                str(standby_data),
                '-l',
                str(root / 'standby.log'),
                '-w',
                'start',
            ],
            env=env,
        )
        standby_started = True
        wait_ready(standby_dsn, args.timeout)

        initial_warm = wait_for_auto_preload(
            standby_dsn,
            initial_generation,
            args.timeout,
        )
        with psycopg.connect(standby_dsn, autocommit=True) as conn:
            fresh_ids = query_ids(conn, 'freshdelta')
            doomed_ids = query_ids(conn, 'doomeddelta')
        if 2001 not in fresh_ids or 2002 in doomed_ids:
            raise RuntimeError(
                'standby active L0 replay is not query-correct: '
                f'fresh={fresh_ids}, doomed={doomed_ids}'
            )
        summary['checks']['initial_replay_and_auto_preload'] = {
            'primary_generation': initial_generation,
            'fresh_ids': fresh_ids,
            'doomed_ids': doomed_ids,
            'warm': initial_warm,
        }
        summary['checks']['standby_maintenance_noops'] = (
            verify_standby_maintenance_noops(standby_dsn)
        )

        primary_maintenance = maintain_primary(primary_dsn, args.timeout)
        updated_status = primary_maintenance['status']
        updated_generation = generation_id(updated_status)
        if updated_generation <= initial_generation:
            raise RuntimeError(
                'primary maintenance did not replace the checked root: '
                f'{initial_generation} -> {updated_generation}'
            )
        wait_for_replay(standby_dsn, updated_generation, args.timeout)
        updated_warm = wait_for_auto_preload(
            standby_dsn,
            updated_generation,
            args.timeout,
        )
        with psycopg.connect(standby_dsn, autocommit=True) as conn:
            updated_ids = query_ids(conn, 'updated replacement')
        if 3001 not in updated_ids:
            raise RuntimeError(
                f'updated root query missed replayed row: {updated_ids}'
            )
        summary['checks']['replacement_replay_and_auto_preload'] = {
            'maintenance': primary_maintenance['results'],
            'generation': updated_generation,
            'updated_ids': updated_ids,
            'warm': updated_warm,
        }
        summary['checks']['explicit_exact_preload'] = explicit_exact_preload(
            standby_dsn
        )
        summary['passed'] = True

        encoded = json.dumps(summary, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + '\n', encoding='utf-8')
        print(encoded)
        print('convergent standby auto preload smoke passed')
    finally:
        if standby_started:
            stop_cluster(pg_ctl, standby_data)
        if primary_started:
            stop_cluster(pg_ctl, primary_data)
        if args.keep:
            print(f'kept temporary clusters at {root}')
            print(f'kept socket root at {socket_root}')
        else:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=os.sys.stderr)
        raise
