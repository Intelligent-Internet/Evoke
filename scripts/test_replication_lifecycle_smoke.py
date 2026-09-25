#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg import sql

from ii42_test_support import (
    REPO_ROOT,
    create_short_socket_root,
    ensure_temp_root,
    extension_control_root,
)


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return result.stdout


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def append_lines(path: Path, lines: list[str]) -> None:
    content = path.read_text(encoding='utf-8')
    if not content.endswith('\n'):
        content += '\n'
    path.write_text(content + '\n'.join(lines) + '\n', encoding='utf-8')


def wait_ready(dsn: str, timeout_s: float = 30.0) -> None:
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
    raise RuntimeError(f'database not ready: {last_error}')


def wait_until(
    description: str,
    fn: Callable[[], Any],
    timeout_s: float = 45.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            value = fn()
            if value:
                return value
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f'timeout waiting for {description}: {last_error}')


def wait_until_timed(
    summary: dict[str, Any],
    key: str,
    description: str,
    fn: Callable[[], Any],
    *,
    started_at: float,
    timeout_s: float = 45.0,
) -> Any:
    value = wait_until(description, fn, timeout_s)
    summary.setdefault('replay_latency_ms', {})[key] = round(
        (time.monotonic() - started_at) * 1000.0,
        3,
    )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify physical replication of unified ii42 index generations.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument(
        '--temp-root',
        type=Path,
        default=Path('/tmp'),
        help=(
            'Root for temporary cluster data and logs. PostgreSQL sockets '
            'use a separate short-lived path so long roots remain valid.'
        ),
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/ii42.control, '
            'or the extension directory itself.'
        ),
    )
    return parser.parse_args()


def relation_exists(conn: psycopg.Connection[Any], name: str) -> bool:
    row = conn.execute('SELECT to_regclass(%s)', (name,)).fetchone()
    return row is not None and row[0] is not None


def relation_filenode(conn: psycopg.Connection[Any], name: str) -> int:
    row = conn.execute(
        'SELECT relfilenode FROM pg_class WHERE oid = %s::regclass',
        (name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f'relation not found: {name}')
    return int(row[0])


def index_status(conn: psycopg.Connection[Any], name: str) -> dict[str, Any]:
    row = conn.execute(
        'SELECT ii42_index_status(%s::regclass)',
        (name,),
    ).fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError(f'index status unavailable: {name}')
    return row[0]


def semantic_pending(status: dict[str, Any]) -> int:
    completion = status['generation']['delta']['semantic_completion']
    return int(completion['pending'])


def acquire_maintenance_lock(
    dsn: str,
    index_name: str,
    *,
    timeout_seconds: float = 30.0,
) -> psycopg.Connection[Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        conn = psycopg.connect(dsn, autocommit=False)
        row = conn.execute(
            'SELECT ii42_index_try_maintenance_lock(%s::regclass)',
            (index_name,),
        ).fetchone()
        if row is not None and row[0] is True:
            return conn
        conn.close()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'could not acquire maintenance lock within '
                f'{timeout_seconds:.1f}s for {index_name}'
            )
        time.sleep(0.1)


def release_maintenance_lock(
    conn: psycopg.Connection[Any],
    index_name: str,
) -> None:
    try:
        conn.execute(
            'SELECT ii42_index_maintenance_unlock(%s::regclass)',
            (index_name,),
        )
        conn.commit()
    finally:
        conn.close()


def maintain_until_converged(
    dsn: str,
    index_name: str,
    *,
    max_attempts: int = 100,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    results: list[str] = []
    statuses: list[dict[str, Any]] = []
    attempts = 0
    deadline = time.monotonic() + timeout_seconds

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "SET ii42.test_convergent_l0_rotation_records = '1'"
        )
        try:
            while attempts < max_attempts and time.monotonic() < deadline:
                status = index_status(conn, index_name)
                if (
                    semantic_pending(status) == 0
                    and int(status['details']['delta_records']) == 0
                ):
                    return {
                        'results': results,
                        'statuses': statuses,
                        'status': status,
                    }
                row = conn.execute(
                    'SELECT ii42_index_maintain(%s::regclass)',
                    (index_name,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        f'maintenance returned no result for {index_name}'
                    )
                result = str(row[0])
                results.append(result)
                statuses.append(index_status(conn, index_name))
                if (
                    'reason=lock_busy' in result
                    or 'reason=accelerator_root_checkpoint' in result
                ):
                    time.sleep(0.1)
                    continue
                attempts += 1
                if 'reason=xid_horizon' in result:
                    time.sleep(0.05)
        finally:
            conn.execute(
                'RESET ii42.test_convergent_l0_rotation_records'
            )

    raise RuntimeError(
        f'{index_name} did not converge after {attempts} attempts '
        f'within {timeout_seconds:.1f}s: '
        f'{results}'
    )


def generation_cache_state(
    conn: psycopg.Connection[Any],
    name: str,
) -> dict[str, Any]:
    row = conn.execute(
        'SELECT ii42_index_runtime_state_json(%s::regclass)',
        (name,),
    ).fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError(f'runtime state unavailable: {name}')
    return row[0]


def generation_storage_ready(
    generation: dict[str, Any],
    primary: dict[str, Any],
    posting: dict[str, Any],
    *,
    unified_posting: bool,
) -> bool:
    if int(primary.get('physical_blocks') or 0) <= 0:
        return False
    if generation.get('diagnostics_complete') is True:
        storage_ready = (
            int(primary.get('payload_bytes') or 0) > 0
            and int(primary.get('reachable_blocks') or 0) > 0
        )
        if not unified_posting:
            return storage_ready
        return (
            storage_ready
            and int(posting.get('bytes') or -1)
                == int(primary.get('payload_bytes') or -2)
        )
    return (
        generation.get('diagnostics_scope') == 'readiness'
        and primary.get('payload_bytes') is None
        and primary.get('reachable_blocks') is None
        and posting.get('bytes') is None
    )


def generation_ready(
    conn: psycopg.Connection[Any],
    name: str,
    *,
    docs: int,
    semantic: bool,
) -> dict[str, Any] | None:
    if not relation_exists(conn, name):
        return None
    status = index_status(conn, name)
    generation = status.get('generation')
    if not isinstance(generation, dict):
        return None
    posting_state = generation.get('posting')
    primary_state = generation.get('primary')
    layout = generation.get('layout')
    if not isinstance(posting_state, dict) or not isinstance(primary_state, dict):
        return None
    if (
        status.get('query_ready') is not True
        or status.get('blocker') != 'none'
        or generation.get('atomic') is not True
        or generation.get('valid') is not True
        or int(generation.get('docs', -1)) != docs
        or not isinstance(layout, dict)
        or layout.get('storage') != 'convergent_segments'
    ):
        return None
    if semantic:
        if (
            primary_state.get('role') != 'unified_posting'
            or primary_state.get('identity') != 'segment_versions'
            or not generation_storage_ready(
                generation,
                primary_state,
                posting_state,
                unified_posting=True,
            )
            or posting_state.get('role') != 'unified_posting'
            or posting_state.get('active') is not True
            or posting_state.get('valid') is not True
            or int(posting_state.get('record_count', -1)) != docs
        ):
            return None
    elif (
        primary_state.get('role') != 'bm25'
        or not generation_storage_ready(
            generation,
            primary_state,
            posting_state,
            unified_posting=False,
        )
        or posting_state.get('present') is not False
    ):
        return None
    return status


def bm25_top_hit(conn: psycopg.Connection[Any]) -> int | None:
    row = conn.execute(
        '''
        SELECT docs.id
        FROM ii42_query_ids(
            'docs_bm25_idx'::regclass,
            ARRAY[0, 4]::int4[],
            1,
            NULL
        ) AS hit
        JOIN docs_bm25 AS docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        LIMIT 1
        ''',
    ).fetchone()
    return None if row is None else int(row[0])


def semantic_hits(
    conn: psycopg.Connection[Any],
    query: str,
) -> list[tuple[int, float]]:
    return text_hits(
        conn,
        index_name='docs_semantic_idx',
        table_name='docs_semantic',
        query=query,
    )


def semantic_filtered_hits(
    conn: psycopg.Connection[Any],
    query: str,
    scope: str,
) -> list[tuple[int, float]]:
    rows = conn.execute(
        '''
        SELECT docs.id, hit.score
        FROM ii42_query(
            'docs_semantic_idx'::regclass,
            %s,
            jsonb_build_object(
                'scope',
                jsonb_build_object('eq', %s::text)
            ),
            100
        ) AS hit
        JOIN docs_semantic AS docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        ''',
        (query, scope),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def text_hits(
    conn: psycopg.Connection[Any],
    *,
    index_name: str,
    table_name: str,
    query: str,
) -> list[tuple[int, float]]:
    statement = sql.SQL(
        '''
        SELECT docs.id, hit.score
        FROM ii42_query(
            {index_name}::regclass,
            %s,
            100
        ) AS hit
        JOIN {table_name} AS docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        '''
    ).format(
        index_name=sql.Literal(index_name),
        table_name=sql.Identifier(table_name),
    )
    rows = conn.execute(statement, (query,)).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def write_cluster_config(
    data_dir: Path,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    lines = [
        "shared_preload_libraries = 'ii42'",
        'max_worker_processes = 16',
        "wal_level = 'replica'",
        'max_wal_senders = 5',
        'max_prepared_transactions = 10',
        'hot_standby = on',
        "ii42.shared_runtime_size = '64MB'",
    ]
    if extension_libdir is not None:
        libdir = str(extension_libdir).replace("'", "''")
        lines.append(f"dynamic_library_path = '{libdir}:$libdir'")
    if extension_control_dir is not None:
        control_dir = str(extension_control_dir).replace("'", "''")
        lines.append(f"extension_control_path = '{control_dir}:$system'")
    append_lines(
        data_dir / 'postgresql.conf',
        lines,
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
    pg_bin = args.pg_bin
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    pg_basebackup = pg_bin / 'pg_basebackup'
    role = os.environ.get('USER') or 'postgres'
    env = os.environ.copy()
    env.setdefault('LC_ALL', 'C')
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / 'manifest.json').is_file():
        raise FileNotFoundError(f'model manifest was not found: {model_path}')
    manifest = json.loads(
        (model_path / 'manifest.json').read_text(encoding='utf-8')
    )
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            '--model-path must use the current Evoke model contract'
        )
    temp_root = ensure_temp_root(args.temp_root)
    root = Path(
        tempfile.mkdtemp(
            prefix='ii42_replication_',
            dir=temp_root,
        )
    )
    socket_root = create_short_socket_root('ii42_replication_socket_')
    primary_data = root / 'primary'
    standby_data = root / 'standby'
    primary_socket = socket_root / 'p'
    standby_socket = socket_root / 's'
    primary_log = root / 'primary.log'
    standby_log = root / 'standby.log'
    primary_socket.mkdir()
    standby_socket.mkdir()
    primary_port = find_free_port()
    standby_port = find_free_port()
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
    semantic_maintenance_lock: psycopg.Connection[Any] | None = None
    summary: dict[str, Any] = {
        'api_version': 'ii42_index_v1',
        'suite': 'physical_replication_lifecycle',
        'checks': {},
    }

    try:
        run(
            [
                str(initdb),
                '-D',
                str(primary_data),
                '-U',
                role,
                '-A',
                'trust',
                '--no-locale',
            ],
            env=env,
        )
        append_lines(
            primary_data / 'pg_hba.conf',
            [
                'host replication all 127.0.0.1/32 trust',
                'host all all 127.0.0.1/32 trust',
            ],
        )
        write_cluster_config(
            primary_data,
            args.extension_libdir,
            args.extension_control_dir,
        )
        run(
            [
                str(pg_ctl),
                '-D',
                str(primary_data),
                '-l',
                str(primary_log),
                '-o',
                (
                    f"-k '{primary_socket}' -p {primary_port} "
                    "-c listen_addresses='127.0.0.1'"
                ),
                'start',
                '-w',
            ],
            env=env,
        )
        primary_started = True
        wait_ready(primary_dsn)

        with psycopg.connect(primary_dsn) as conn:
            conn.execute('CREATE EXTENSION ii42')
            conn.execute(
                '''
                CREATE TABLE docs_bm25 (
                    id int PRIMARY KEY,
                    token_ids int4[] NOT NULL
                )
                ''',
            )
            with conn.cursor() as cur:
                cur.executemany(
                    'INSERT INTO docs_bm25 (id, token_ids) VALUES (%s, %s)',
                    [(1, [0, 4]), (2, [1, 2]), (3, [0])],
                )
            conn.execute(
                '''
                CREATE TABLE docs_semantic (
                    id int PRIMARY KEY,
                    body text NOT NULL,
                    scope text NOT NULL
                )
                ''',
            )
            with conn.cursor() as cur:
                cur.executemany(
                    'INSERT INTO docs_semantic (id, body, scope) '
                    'VALUES (%s, %s, %s)',
                    [
                        (1, 'alpha runtime', 'alpha'),
                        (2, 'semantic replication', 'replication'),
                        (3, 'unified posting', 'unified'),
                    ],
                )
            conn.execute(
                '''
                CREATE TABLE docs_convert (
                    id int PRIMARY KEY,
                    body text NOT NULL
                )
                ''',
            )
            with conn.cursor() as cur:
                cur.executemany(
                    'INSERT INTO docs_convert (id, body) VALUES (%s, %s)',
                    [
                        (1, 'conversion alpha target'),
                        (2, 'conversion semantic target'),
                        (3, 'unified lifecycle target'),
                    ],
                )
            conn.commit()

        run(
            [
                str(pg_basebackup),
                '-D',
                str(standby_data),
                '-R',
                '-X',
                'stream',
                '-h',
                '127.0.0.1',
                '-p',
                str(primary_port),
                '-U',
                role,
            ],
            env=env,
        )
        run(
            [
                str(pg_ctl),
                '-D',
                str(standby_data),
                '-l',
                str(standby_log),
                '-o',
                (
                    f"-k '{standby_socket}' -p {standby_port} "
                    "-c listen_addresses='127.0.0.1'"
                ),
                'start',
                '-w',
            ],
            env=env,
        )
        standby_started = True
        wait_ready(standby_dsn)

        escaped_model_path = str(model_path).replace("'", "''")
        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                '''
                CREATE INDEX docs_bm25_idx
                ON docs_bm25 USING ii42 (token_ids)
                WITH (sae = false)
                ''',
            )
            conn.execute(
                f'''
                CREATE INDEX docs_semantic_idx
                ON docs_semantic USING ii42 (body) INCLUDE (scope)
                WITH (
                    sae = true,
                    model_path = '{escaped_model_path}'
                )
                ''',
            )
            conn.execute(
                '''
                CREATE INDEX docs_convert_idx
                ON docs_convert USING ii42 (body)
                WITH (sae = false)
                ''',
            )
            conn.commit()
        semantic_maintenance_lock = acquire_maintenance_lock(
            primary_dsn,
            'docs_semantic_idx',
        )
        with psycopg.connect(primary_dsn) as conn:
            initial_semantic_generation = str(
                index_status(conn, 'docs_semantic_idx')[
                    'generation'
                ]['generation_id']
            )
        initial_replay_started = time.monotonic()

        initial_semantic_query = 'alpha semantic replication'
        conversion_query = 'conversion alpha target'
        with psycopg.connect(primary_dsn) as conn:
            primary_initial_semantic_hits = semantic_hits(
                conn,
                initial_semantic_query,
            )
            primary_initial_filtered_hits = semantic_filtered_hits(
                conn,
                initial_semantic_query,
                'alpha',
            )
            primary_initial_conversion_hits = text_hits(
                conn,
                index_name='docs_convert_idx',
                table_name='docs_convert',
                query=conversion_query,
            )
        if not any(
            doc_id == 1 for doc_id, _score in primary_initial_semantic_hits
        ):
            raise RuntimeError(
                'primary semantic query did not retrieve its lexical anchor'
            )
        if not any(
            doc_id == 1
            for doc_id, _score in primary_initial_filtered_hits
        ):
            raise RuntimeError(
                'primary filtered semantic query did not retrieve its scope'
            )
        if not any(
            doc_id == 1 for doc_id, _score in primary_initial_conversion_hits
        ):
            raise RuntimeError(
                'primary conversion query did not retrieve its lexical anchor'
            )

        def initial_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                bm25 = generation_ready(
                    conn,
                    'docs_bm25_idx',
                    docs=3,
                    semantic=False,
                )
                semantic = generation_ready(
                    conn,
                    'docs_semantic_idx',
                    docs=3,
                    semantic=True,
                )
                conversion = generation_ready(
                    conn,
                    'docs_convert_idx',
                    docs=3,
                    semantic=False,
                )
                bm25_hit = bm25_top_hit(conn)
                standby_semantic_hits = semantic_hits(
                    conn,
                    initial_semantic_query,
                )
                standby_filtered_hits = semantic_filtered_hits(
                    conn,
                    initial_semantic_query,
                    'alpha',
                )
                standby_conversion_hits = text_hits(
                    conn,
                    index_name='docs_convert_idx',
                    table_name='docs_convert',
                    query=conversion_query,
                )
                mismatches = {
                    'bm25_generation_ready': bm25 is not None,
                    'semantic_generation_ready': semantic is not None,
                    'semantic_generation_match': (
                        semantic is not None
                        and str(
                            semantic['generation']['generation_id']
                        ) == initial_semantic_generation
                    ),
                    'conversion_generation_ready': conversion is not None,
                    'bm25_top_hit': bm25_hit,
                    'semantic_hits_match': (
                        standby_semantic_hits
                        == primary_initial_semantic_hits
                    ),
                    'semantic_filtered_hits_match': (
                        standby_filtered_hits
                        == primary_initial_filtered_hits
                    ),
                    'conversion_hits_match': (
                        standby_conversion_hits
                        == primary_initial_conversion_hits
                    ),
                }
                if not all(
                    value is True
                    for key, value in mismatches.items()
                    if key != 'bm25_top_hit'
                ) or bm25_hit != 1:
                    observations = {
                        'checks': mismatches,
                        'bm25_status': index_status(conn, 'docs_bm25_idx'),
                        'semantic_status': index_status(
                            conn,
                            'docs_semantic_idx',
                        ),
                        'conversion_status': index_status(
                            conn,
                            'docs_convert_idx',
                        ),
                        'primary_semantic_hits': (
                            primary_initial_semantic_hits
                        ),
                        'standby_semantic_hits': standby_semantic_hits,
                        'primary_filtered_hits': (
                            primary_initial_filtered_hits
                        ),
                        'standby_filtered_hits': standby_filtered_hits,
                        'primary_conversion_hits': (
                            primary_initial_conversion_hits
                        ),
                        'standby_conversion_hits': standby_conversion_hits,
                    }
                    raise RuntimeError(
                        json.dumps(observations, sort_keys=True)
                    )
                return {
                    'bm25': bm25,
                    'semantic': semantic,
                    'conversion': conversion,
                    'conversion_hits': primary_initial_conversion_hits,
                    'semantic_hits': primary_initial_semantic_hits,
                    'semantic_filtered_hits': (
                        primary_initial_filtered_hits
                    ),
                }

        initial = wait_until_timed(
            summary,
            'initial_generations',
            'initial unified generations',
            initial_replay,
            started_at=initial_replay_started,
        )
        summary['checks']['initial_replay'] = {
            'bm25_generation': initial['bm25']['generation']['generation_id'],
            'semantic_generation': (
                initial['semantic']['generation']['generation_id']
            ),
            'conversion_generation': (
                initial['conversion']['generation']['generation_id']
            ),
            'conversion_hits': initial['conversion_hits'],
            'semantic_hits': initial['semantic_hits'],
            'semantic_filtered_hits': initial['semantic_filtered_hits'],
        }

        def stable_generation_replay() -> bool | None:
            with psycopg.connect(standby_dsn) as conn:
                observed = str(
                    index_status(conn, 'docs_semantic_idx')[
                        'generation'
                    ]['generation_id']
                )
                return True if observed == initial_semantic_generation else None

        stable_generation_started = time.monotonic()
        wait_until_timed(
            summary,
            'stable_semantic_generation',
            'stable semantic generation replay before 2PC',
            stable_generation_replay,
            started_at=stable_generation_started,
        )
        rollback_gid = 'ii42_replication_rollback_prepared'
        commit_gid = 'ii42_replication_commit_prepared'
        rollback_query = 'rollback prepared replication sentinel'
        commit_query = 'commit prepared replication sentinel'

        def primary_flush_lsn(conn: psycopg.Connection[Any]) -> str:
            row = conn.execute('SELECT pg_current_wal_flush_lsn()').fetchone()
            if row is None or row[0] is None:
                raise RuntimeError('primary WAL flush LSN is unavailable')
            value = row[0]
            if isinstance(value, memoryview):
                value = value.tobytes()
            if isinstance(value, (bytes, bytearray)):
                return bytes(value).decode('ascii')
            return str(value)

        def standby_has_replayed(
            conn: psycopg.Connection[Any],
            target_lsn: str,
        ) -> bool:
            row = conn.execute(
                'SELECT pg_last_wal_replay_lsn() >= %s::pg_lsn',
                (target_lsn,),
            ).fetchone()
            return row is not None and row[0] is True

        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute('BEGIN')
            conn.execute(
                'INSERT INTO docs_semantic VALUES (%s, %s, %s)',
                (90, rollback_query, 'rollback-prepared'),
            )
            conn.execute(
                sql.SQL('PREPARE TRANSACTION {}').format(
                    sql.Literal(rollback_gid)
                )
            )
            rollback_prepare_lsn = primary_flush_lsn(conn)
        rollback_prepare_started = time.monotonic()
        rollback_prepare_observation: dict[str, Any] = {}

        def rollback_prepare_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                replayed = standby_has_replayed(conn, rollback_prepare_lsn)
                if not replayed:
                    rollback_prepare_observation.clear()
                    rollback_prepare_observation['wal_replayed'] = False
                    return None
                status = index_status(conn, 'docs_semantic_idx')
                hits = semantic_hits(conn, rollback_query)
                filtered_hits = semantic_filtered_hits(
                    conn,
                    rollback_query,
                    'rollback-prepared',
                )
                heap_visible = bool(
                    conn.execute(
                        'SELECT EXISTS ('
                        'SELECT 1 FROM docs_semantic WHERE id = 90'
                        ')',
                    ).fetchone()[0]
                )
                rollback_prepare_observation.clear()
                rollback_prepare_observation.update(
                    {
                        'wal_replayed': True,
                        'heap_visible': heap_visible,
                        'generation_id': (
                            status['generation']['generation_id']
                        ),
                        'delta_records': (
                            status['details']['delta_records']
                        ),
                        'hits': hits,
                        'filtered_hits': filtered_hits,
                    }
                )
                if (
                    status['generation']['generation_id']
                    != initial_semantic_generation
                    or int(status['details']['delta_records']) < 1
                    or any(doc_id == 90 for doc_id, _score in hits)
                    or filtered_hits
                ):
                    return None
                return {
                    'status': status,
                    'hits': hits,
                    'filtered_hits': filtered_hits,
                }

        try:
            rollback_prepared = wait_until_timed(
                summary,
                'rollback_prepare',
                'prepared rollback delta replay',
                rollback_prepare_replay,
                started_at=rollback_prepare_started,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f'{exc}; last_observation='
                f'{json.dumps(rollback_prepare_observation, sort_keys=True)}'
            ) from exc
        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL('ROLLBACK PREPARED {}').format(
                    sql.Literal(rollback_gid)
                )
            )
            rollback_lsn = primary_flush_lsn(conn)
        rollback_started = time.monotonic()

        def rollback_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                if not standby_has_replayed(conn, rollback_lsn):
                    return None
                status = index_status(conn, 'docs_semantic_idx')
                hits = semantic_hits(conn, rollback_query)
                filtered_hits = semantic_filtered_hits(
                    conn,
                    rollback_query,
                    'rollback-prepared',
                )
                if (
                    status['generation']['generation_id']
                    != initial_semantic_generation
                    or int(status['details']['delta_records']) < 1
                    or any(doc_id == 90 for doc_id, _score in hits)
                    or filtered_hits
                ):
                    return None
                return {
                    'status': status,
                    'hits': hits,
                    'filtered_hits': filtered_hits,
                }

        rolled_back = wait_until_timed(
            summary,
            'rollback_prepared',
            'prepared rollback visibility replay',
            rollback_replay,
            started_at=rollback_started,
        )

        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute('BEGIN')
            conn.execute(
                'INSERT INTO docs_semantic VALUES (%s, %s, %s)',
                (91, commit_query, 'commit-prepared'),
            )
            conn.execute(
                sql.SQL('PREPARE TRANSACTION {}').format(
                    sql.Literal(commit_gid)
                )
            )
            commit_prepare_lsn = primary_flush_lsn(conn)
        commit_prepare_started = time.monotonic()

        def commit_prepare_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                if not standby_has_replayed(conn, commit_prepare_lsn):
                    return None
                status = index_status(conn, 'docs_semantic_idx')
                hits = semantic_hits(conn, commit_query)
                filtered_hits = semantic_filtered_hits(
                    conn,
                    commit_query,
                    'commit-prepared',
                )
                if (
                    status['generation']['generation_id']
                    != initial_semantic_generation
                    or int(status['details']['delta_records']) < 2
                    or any(doc_id == 91 for doc_id, _score in hits)
                    or filtered_hits
                ):
                    return None
                return {
                    'status': status,
                    'hits': hits,
                    'filtered_hits': filtered_hits,
                }

        commit_prepared = wait_until_timed(
            summary,
            'commit_prepare',
            'prepared commit delta replay',
            commit_prepare_replay,
            started_at=commit_prepare_started,
        )
        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL('COMMIT PREPARED {}').format(
                    sql.Literal(commit_gid)
                )
            )
            commit_lsn = primary_flush_lsn(conn)
        with psycopg.connect(primary_dsn) as conn:
            primary_committed_filtered_hits = semantic_filtered_hits(
                conn,
                commit_query,
                'commit-prepared',
            )
        if not any(
            doc_id == 91
            for doc_id, _score in primary_committed_filtered_hits
        ):
            raise RuntimeError(
                'committed prepared scope was not visible on the primary'
            )
        commit_started = time.monotonic()
        commit_replay_observation: dict[str, Any] = {}

        def commit_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                replayed = standby_has_replayed(conn, commit_lsn)
                if not replayed:
                    commit_replay_observation.clear()
                    commit_replay_observation['wal_replayed'] = False
                    return None
                status = index_status(conn, 'docs_semantic_idx')
                hits = semantic_hits(conn, commit_query)
                filtered_hits = semantic_filtered_hits(
                    conn,
                    commit_query,
                    'commit-prepared',
                )
                heap_visible = bool(
                    conn.execute(
                        'SELECT EXISTS ('
                        'SELECT 1 FROM docs_semantic WHERE id = 91'
                        ')',
                    ).fetchone()[0]
                )
                commit_replay_observation.clear()
                commit_replay_observation.update(
                    {
                        'wal_replayed': True,
                        'heap_visible': heap_visible,
                        'generation_id': (
                            status['generation']['generation_id']
                        ),
                        'delta_records': (
                            status['details']['delta_records']
                        ),
                        'hits': hits,
                        'filtered_hits': filtered_hits,
                    }
                )
                if (
                    status['generation']['generation_id']
                    != initial_semantic_generation
                    or not any(doc_id == 91 for doc_id, _score in hits)
                    or filtered_hits != primary_committed_filtered_hits
                ):
                    return None
                return {
                    'status': status,
                    'hits': hits,
                    'filtered_hits': filtered_hits,
                }

        try:
            committed = wait_until_timed(
                summary,
                'commit_prepared',
                'prepared commit visibility replay',
                commit_replay,
                started_at=commit_started,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f'{exc}; last_observation='
                f'{json.dumps(commit_replay_observation, sort_keys=True)}'
            ) from exc
        summary['checks']['prepared_transaction_replay'] = {
            'generation': initial_semantic_generation,
            'rollback_prepare': rollback_prepared,
            'rollback': rolled_back,
            'commit_prepare': commit_prepared,
            'commit': committed,
            'primary_committed_filtered_hits': (
                primary_committed_filtered_hits
            ),
        }
        release_maintenance_lock(
            semantic_maintenance_lock,
            'docs_semantic_idx',
        )
        semantic_maintenance_lock = None

        with psycopg.connect(primary_dsn) as conn:
            conn.execute('DELETE FROM docs_semantic WHERE id = 91')
            conn.commit()
        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            conn.execute('VACUUM docs_semantic')
        prepared_cleanup_maintenance = maintain_until_converged(
            primary_dsn,
            'docs_semantic_idx',
        )
        with psycopg.connect(primary_dsn, autocommit=True) as conn:
            prepared_cleanup_lsn = primary_flush_lsn(conn)
        prepared_cleanup_started = time.monotonic()

        def prepared_cleanup_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                if not standby_has_replayed(conn, prepared_cleanup_lsn):
                    return None
                status = generation_ready(
                    conn,
                    'docs_semantic_idx',
                    docs=3,
                    semantic=True,
                )
                if (
                    status is None
                    or int(status['details']['delta_records']) != 0
                    or int(status['details']['pending_writes']) != 0
                    or int(status['details']['pending_deletes']) != 0
                    or any(
                        doc_id in (90, 91)
                        for doc_id, _score in semantic_hits(
                            conn,
                            f'{rollback_query} {commit_query}',
                        )
                    )
                    or semantic_filtered_hits(
                        conn,
                        commit_query,
                        'commit-prepared',
                    )
                ):
                    return None
                return status

        prepared_cleanup = wait_until_timed(
            summary,
            'prepared_cleanup',
            'prepared transaction debt compaction replay',
            prepared_cleanup_replay,
            started_at=prepared_cleanup_started,
        )
        summary['checks']['prepared_transaction_debt_reclaimed'] = {
            'generation': (
                prepared_cleanup['generation']['generation_id']
            ),
            'delta_records': (
                prepared_cleanup['details']['delta_records']
            ),
            'maintenance': prepared_cleanup_maintenance,
        }

        semantic_maintenance_lock = acquire_maintenance_lock(
            primary_dsn,
            'docs_semantic_idx',
        )
        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                "INSERT INTO docs_semantic VALUES "
                "(4, 'new semantic replication row', 'pending')",
            )
            conn.commit()
        unified_delta_replay_started = time.monotonic()
        updated_semantic_query = 'new semantic replication'
        with psycopg.connect(primary_dsn) as conn:
            primary_pending_status = index_status(
                conn,
                'docs_semantic_idx',
            )
            primary_delta_cache_before = generation_cache_state(
                conn,
                'docs_semantic_idx',
            )
            primary_pending_semantic_hits = semantic_hits(
                conn,
                updated_semantic_query,
            )
            primary_pending_filtered_hits = semantic_filtered_hits(
                conn,
                updated_semantic_query,
                'pending',
            )
            primary_delta_cache_after = generation_cache_state(
                conn,
                'docs_semantic_idx',
            )
        if not any(
            doc_id == 4 for doc_id, _score in primary_pending_semantic_hits
        ) or not any(
            doc_id == 4 for doc_id, _score in primary_pending_filtered_hits
        ) or semantic_pending(primary_pending_status) != 1:
            raise RuntimeError(
                'primary lexical-pending delta was not visible before '
                'maintenance'
            )

        def standby_delta_replayed() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                if not relation_exists(conn, 'docs_semantic_idx'):
                    return None
                status = index_status(conn, 'docs_semantic_idx')
                details = status.get('details')
                if (
                    not isinstance(details, dict)
                    or int(details.get('delta_records', 0)) <= 0
                    or semantic_pending(status) != 1
                ):
                    return None
                return generation_cache_state(conn, 'docs_semantic_idx')

        standby_delta_cache_before = wait_until_timed(
            summary,
            'linked_l0',
            'standby linked-L0 replay',
            standby_delta_replayed,
            started_at=unified_delta_replay_started,
        )
        with psycopg.connect(standby_dsn) as conn:
            standby_delta_cold_hits = semantic_hits(
                conn,
                updated_semantic_query,
            )
            standby_delta_cache_after_cold = generation_cache_state(
                conn,
                'docs_semantic_idx',
            )
            standby_delta_warm_hits = semantic_hits(
                conn,
                updated_semantic_query,
            )
            standby_delta_filtered_hits = semantic_filtered_hits(
                conn,
                updated_semantic_query,
                'pending',
            )
            standby_delta_cache_after_warm = generation_cache_state(
                conn,
                'docs_semantic_idx',
            )
        standby_before_shared = standby_delta_cache_before['shared_preload']
        standby_cold_shared = (
            standby_delta_cache_after_cold['shared_preload']
        )
        standby_warm_shared = (
            standby_delta_cache_after_warm['shared_preload']
        )

        def decoded_cache_contract_absent(state: dict[str, Any]) -> bool:
            retired_keys = {
                'unified_delta_cache_entries',
                'unified_delta_cache_current',
                'unified_delta_cache_loading',
            }
            return retired_keys.isdisjoint(state)

        primary_before_shared = primary_delta_cache_before['shared_preload']
        primary_after_shared = primary_delta_cache_after['shared_preload']
        if (
            standby_delta_cold_hits != primary_pending_semantic_hits
            or standby_delta_warm_hits != primary_pending_semantic_hits
            or standby_delta_filtered_hits != primary_pending_filtered_hits
            or not decoded_cache_contract_absent(primary_before_shared)
            or not decoded_cache_contract_absent(primary_after_shared)
            or not decoded_cache_contract_absent(standby_before_shared)
            or not decoded_cache_contract_absent(standby_cold_shared)
            or not decoded_cache_contract_absent(standby_warm_shared)
        ):
            raise RuntimeError(
                'convergent linked-L0 replay diverged or activated the '
                'legacy unified delta cache: '
                f'primary_hits={primary_pending_semantic_hits}, '
                f'standby_cold_hits={standby_delta_cold_hits}, '
                f'standby_warm_hits={standby_delta_warm_hits}, '
                f'standby_filtered_hits={standby_delta_filtered_hits}, '
                f'primary_before={primary_before_shared}, '
                f'primary_after={primary_after_shared}, '
                f'standby_before={standby_before_shared}, '
                f'standby_cold={standby_cold_shared}, '
                f'standby_warm={standby_warm_shared}'
            )
        release_maintenance_lock(
            semantic_maintenance_lock,
            'docs_semantic_idx',
        )
        semantic_maintenance_lock = None
        summary['checks']['eventual_linked_l0_replay'] = {
            'primary_pending_status': primary_pending_status,
            'primary_before': primary_delta_cache_before,
            'primary_after': primary_delta_cache_after,
            'standby_before': standby_delta_cache_before,
            'standby_after_cold': standby_delta_cache_after_cold,
            'standby_after_warm': standby_delta_cache_after_warm,
            'pending_semantic_hits': standby_delta_warm_hits,
            'pending_filtered_hits': standby_delta_filtered_hits,
        }

        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                'UPDATE docs_bm25 SET token_ids = %s WHERE id = 3',
                ([0, 4, 4, 4],),
            )
            conn.commit()
        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                "SELECT ii42_index_maintain('docs_bm25_idx'::regclass)",
            )
            conn.commit()
        semantic_insert_maintenance = maintain_until_converged(
            primary_dsn,
            'docs_semantic_idx',
        )
        with psycopg.connect(primary_dsn) as conn:
            primary_updated_bm25_status = index_status(
                conn,
                'docs_bm25_idx',
            )
            primary_updated_semantic_status = index_status(
                conn,
                'docs_semantic_idx',
            )
            primary_updated_semantic_hits = semantic_hits(
                conn,
                updated_semantic_query,
            )
        # Generation docs count physical TIDs. UPDATE/DELETE versions remain
        # until VACUUM supplies exact tombstones, so replication must match the
        # primary generation rather than the current live heap-row count.
        primary_updated_bm25_docs = int(
            primary_updated_bm25_status['generation']['docs']
        )
        primary_updated_semantic_docs = int(
            primary_updated_semantic_status['generation']['docs']
        )
        update_replay_started = time.monotonic()

        def update_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                bm25_status = index_status(conn, 'docs_bm25_idx')
                semantic_status = index_status(conn, 'docs_semantic_idx')
                bm25 = generation_ready(
                    conn,
                    'docs_bm25_idx',
                    docs=primary_updated_bm25_docs,
                    semantic=False,
                )
                semantic = generation_ready(
                    conn,
                    'docs_semantic_idx',
                    docs=primary_updated_semantic_docs,
                    semantic=True,
                )
                hits = semantic_hits(conn, updated_semantic_query)
                bm25_top = bm25_top_hit(conn)
                if (
                    bm25 is None
                    or semantic is None
                    or bm25_top != 3
                    or hits != primary_updated_semantic_hits
                ):
                    raise RuntimeError(
                        'standby update replay is not converged: '
                        + json.dumps(
                            {
                                'bm25_docs': bm25_status.get(
                                    'generation',
                                    {},
                                ).get('docs'),
                                'bm25_top': bm25_top,
                                'semantic_docs': semantic_status.get(
                                    'generation',
                                    {},
                                ).get('docs'),
                                'semantic_hits': hits,
                                'expected_semantic_hits': (
                                    primary_updated_semantic_hits
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                return {
                    'bm25': bm25,
                    'semantic': semantic,
                    'semantic_hits': hits,
                }

        updated = wait_until_timed(
            summary,
            'maintained_update_and_insert',
            'updated unified generations',
            update_replay,
            started_at=update_replay_started,
        )
        summary['checks']['update_and_insert_replay'] = {
            'bm25_physical_docs': primary_updated_bm25_docs,
            'semantic_physical_docs': primary_updated_semantic_docs,
            'semantic_hits': updated['semantic_hits'],
            'semantic_maintenance': semantic_insert_maintenance,
        }

        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                'INSERT INTO docs_bm25 (id, token_ids) VALUES (%s, %s)',
                (4, [0, 4, 4, 4, 4]),
            )
            conn.execute('DELETE FROM docs_bm25 WHERE id = 3')
            conn.execute(
                '''
                UPDATE docs_semantic
                SET body = 'updated semantic replication target',
                    scope = 'updated'
                WHERE id = 2
                ''',
            )
            conn.execute('DELETE FROM docs_semantic WHERE id = 4')
            conn.commit()
        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                "SELECT ii42_index_maintain('docs_bm25_idx'::regclass)",
            )
            conn.commit()
        semantic_update_maintenance = maintain_until_converged(
            primary_dsn,
            'docs_semantic_idx',
        )
        delete_update_replay_started = time.monotonic()

        final_semantic_query = 'updated semantic replication target'
        with psycopg.connect(primary_dsn) as conn:
            primary_final_semantic_hits = semantic_hits(
                conn,
                final_semantic_query,
            )
            primary_final_filtered_hits = semantic_filtered_hits(
                conn,
                final_semantic_query,
                'updated',
            )
            primary_final_bm25_status = index_status(
                conn,
                'docs_bm25_idx',
            )
            primary_final_semantic_status = index_status(
                conn,
                'docs_semantic_idx',
            )
        primary_final_bm25_docs = int(
            primary_final_bm25_status['generation']['docs']
        )
        primary_final_semantic_docs = int(
            primary_final_semantic_status['generation']['docs']
        )
        final_semantic_ids = {
            doc_id for doc_id, _score in primary_final_semantic_hits
        }
        if 2 not in final_semantic_ids or 4 in final_semantic_ids:
            raise RuntimeError(
                'primary semantic update/delete was not visible to native '
                'search'
            )
        final_filtered_ids = {
            doc_id for doc_id, _score in primary_final_filtered_hits
        }
        if final_filtered_ids != {2}:
            raise RuntimeError(
                'primary semantic scope update/delete was not exact: '
                f'{primary_final_filtered_hits}'
            )

        def delete_and_update_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                bm25_status = index_status(conn, 'docs_bm25_idx')
                semantic_status = index_status(conn, 'docs_semantic_idx')
                bm25 = generation_ready(
                    conn,
                    'docs_bm25_idx',
                    docs=primary_final_bm25_docs,
                    semantic=False,
                )
                semantic = generation_ready(
                    conn,
                    'docs_semantic_idx',
                    docs=primary_final_semantic_docs,
                    semantic=True,
                )
                hits = semantic_hits(conn, final_semantic_query)
                filtered_hits = semantic_filtered_hits(
                    conn,
                    final_semantic_query,
                    'updated',
                )
                bm25_top = bm25_top_hit(conn)
                if (
                    bm25 is None
                    or semantic is None
                    or bm25_top != 4
                    or hits != primary_final_semantic_hits
                    or filtered_hits != primary_final_filtered_hits
                ):
                    raise RuntimeError(
                        'standby delete/update replay is not converged: '
                        + json.dumps(
                            {
                                'bm25_docs': bm25_status.get(
                                    'generation',
                                    {},
                                ).get('docs'),
                                'bm25_top': bm25_top,
                                'semantic_docs': semantic_status.get(
                                    'generation',
                                    {},
                                ).get('docs'),
                                'semantic_hits': hits,
                                'semantic_filtered_hits': filtered_hits,
                                'expected_semantic_hits': (
                                    primary_final_semantic_hits
                                ),
                                'expected_semantic_filtered_hits': (
                                    primary_final_filtered_hits
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                return {
                    'bm25': bm25,
                    'semantic': semantic,
                    'semantic_hits': hits,
                    'semantic_filtered_hits': filtered_hits,
                }

        final = wait_until_timed(
            summary,
            'maintained_insert_update_delete',
            'insert, update, and delete replay',
            delete_and_update_replay,
            started_at=delete_update_replay_started,
        )
        summary['checks']['delete_and_update_replay'] = {
            'bm25_generation': final['bm25']['generation']['generation_id'],
            'bm25_physical_docs': primary_final_bm25_docs,
            'semantic_generation': (
                final['semantic']['generation']['generation_id']
            ),
            'semantic_physical_docs': primary_final_semantic_docs,
            'semantic_hits': final['semantic_hits'],
            'semantic_filtered_hits': final['semantic_filtered_hits'],
            'semantic_maintenance': semantic_update_maintenance,
        }

        with psycopg.connect(standby_dsn) as conn:
            before = {
                'bm25': relation_filenode(conn, 'docs_bm25_idx'),
                'semantic': relation_filenode(conn, 'docs_semantic_idx'),
            }
        with psycopg.connect(primary_dsn) as conn:
            conn.execute('REINDEX INDEX docs_bm25_idx')
            conn.execute('REINDEX INDEX docs_semantic_idx')
            conn.commit()
        reindex_replay_started = time.monotonic()
        with psycopg.connect(primary_dsn) as conn:
            primary_reindexed_semantic_hits = semantic_hits(
                conn,
                final_semantic_query,
            )
            primary_reindexed_filtered_hits = semantic_filtered_hits(
                conn,
                final_semantic_query,
                'updated',
            )

        def reindex_replay() -> dict[str, int] | None:
            with psycopg.connect(standby_dsn) as conn:
                after = {
                    'bm25': relation_filenode(conn, 'docs_bm25_idx'),
                    'semantic': relation_filenode(conn, 'docs_semantic_idx'),
                }
                if after == before:
                    return None
                if (
                    after['bm25'] == before['bm25']
                    or after['semantic'] == before['semantic']
                    or generation_ready(
                        conn,
                        'docs_bm25_idx',
                        docs=3,
                        semantic=False,
                    ) is None
                    or generation_ready(
                        conn,
                        'docs_semantic_idx',
                        docs=3,
                        semantic=True,
                    ) is None
                    or bm25_top_hit(conn) != 4
                    or semantic_hits(conn, final_semantic_query)
                    != primary_reindexed_semantic_hits
                    or semantic_filtered_hits(
                        conn,
                        final_semantic_query,
                        'updated',
                    ) != primary_reindexed_filtered_hits
                ):
                    return None
                return after

        after = wait_until_timed(
            summary,
            'reindex',
            'reindexed unified generations',
            reindex_replay,
            started_at=reindex_replay_started,
        )
        summary['checks']['reindex'] = {
            'before': before,
            'after': after,
            'semantic_hits': primary_reindexed_semantic_hits,
            'semantic_filtered_hits': primary_reindexed_filtered_hits,
        }

        with psycopg.connect(standby_dsn) as conn:
            conversion_bm25_filenode = relation_filenode(
                conn,
                'docs_convert_idx',
            )
        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                f'''
                ALTER INDEX docs_convert_idx SET (
                    sae = true,
                    model_path = '{escaped_model_path}'
                )
                '''
            )
            conn.commit()
        bm25_to_sae_mismatch_started = time.monotonic()

        def conversion_mismatch(
            expected_layout: dict[str, Any],
        ) -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                if not relation_exists(conn, 'docs_convert_idx'):
                    return None
                status = index_status(conn, 'docs_convert_idx')
                generation = status.get('generation')
                if (
                    status.get('query_ready') is not False
                    or not isinstance(generation, dict)
                    or generation.get('layout') != expected_layout
                ):
                    raise RuntimeError(
                        json.dumps(
                            {
                                'expected_layout': expected_layout,
                                'observed_status': status,
                            },
                            sort_keys=True,
                        )
                    )
                try:
                    text_hits(
                        conn,
                        index_name='docs_convert_idx',
                        table_name='docs_convert',
                        query=conversion_query,
                    )
                except psycopg.Error as error:
                    conn.rollback()
                    return {
                        'error': str(error),
                        'status': status,
                    }
                return None

        bm25_to_sae_pending = wait_until_timed(
            summary,
            'bm25_to_sae_layout_mismatch',
            'BM25-to-SAE mismatch replay',
            lambda: conversion_mismatch(
                {
                    'matches': False,
                    'physical': 'bm25',
                    'requested': 'semantic',
                    'storage': 'convergent_segments',
                }
            ),
            started_at=bm25_to_sae_mismatch_started,
        )

        with psycopg.connect(primary_dsn) as conn:
            conn.execute('REINDEX INDEX docs_convert_idx')
            conn.commit()
        bm25_to_sae_reindex_started = time.monotonic()
        with psycopg.connect(primary_dsn) as conn:
            primary_converted_semantic_hits = text_hits(
                conn,
                index_name='docs_convert_idx',
                table_name='docs_convert',
                query=conversion_query,
            )
        if not primary_converted_semantic_hits:
            raise RuntimeError(
                'converted semantic index returned no primary hits'
            )

        def bm25_to_sae_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                status = generation_ready(
                    conn,
                    'docs_convert_idx',
                    docs=3,
                    semantic=True,
                )
                filenode = relation_filenode(conn, 'docs_convert_idx')
                hits = text_hits(
                    conn,
                    index_name='docs_convert_idx',
                    table_name='docs_convert',
                    query=conversion_query,
                )
                if (
                    status is None
                    or filenode == conversion_bm25_filenode
                    or hits != primary_converted_semantic_hits
                ):
                    return None
                return {
                    'filenode': filenode,
                    'hits': hits,
                    'status': status,
                }

        converted_semantic = wait_until_timed(
            summary,
            'bm25_to_sae_reindex',
            'BM25-to-SAE reindex replay',
            bm25_to_sae_replay,
            started_at=bm25_to_sae_reindex_started,
        )

        with psycopg.connect(primary_dsn) as conn:
            conn.execute(
                'ALTER INDEX docs_convert_idx RESET (model_path)'
            )
            conn.execute(
                'ALTER INDEX docs_convert_idx SET (sae = false)'
            )
            conn.commit()
        sae_to_bm25_mismatch_started = time.monotonic()

        sae_to_bm25_pending = wait_until_timed(
            summary,
            'sae_to_bm25_layout_mismatch',
            'SAE-to-BM25 mismatch replay',
            lambda: conversion_mismatch(
                {
                    'matches': False,
                    'physical': 'semantic',
                    'requested': 'bm25',
                    'storage': 'convergent_segments',
                }
            ),
            started_at=sae_to_bm25_mismatch_started,
        )

        conversion_semantic_filenode = int(
            converted_semantic['filenode']
        )
        with psycopg.connect(primary_dsn) as conn:
            conn.execute('REINDEX INDEX docs_convert_idx')
            conn.commit()
        sae_to_bm25_reindex_started = time.monotonic()
        with psycopg.connect(primary_dsn) as conn:
            primary_converted_bm25_hits = text_hits(
                conn,
                index_name='docs_convert_idx',
                table_name='docs_convert',
                query=conversion_query,
            )
        if not primary_converted_bm25_hits:
            raise RuntimeError(
                'converted BM25 index returned no primary hits'
            )

        def sae_to_bm25_replay() -> dict[str, Any] | None:
            with psycopg.connect(standby_dsn) as conn:
                status = generation_ready(
                    conn,
                    'docs_convert_idx',
                    docs=3,
                    semantic=False,
                )
                filenode = relation_filenode(conn, 'docs_convert_idx')
                hits = text_hits(
                    conn,
                    index_name='docs_convert_idx',
                    table_name='docs_convert',
                    query=conversion_query,
                )
                if (
                    status is None
                    or filenode == conversion_semantic_filenode
                    or hits != primary_converted_bm25_hits
                ):
                    return None
                return {
                    'filenode': filenode,
                    'hits': hits,
                    'status': status,
                }

        converted_bm25 = wait_until_timed(
            summary,
            'sae_to_bm25_reindex',
            'SAE-to-BM25 reindex replay',
            sae_to_bm25_replay,
            started_at=sae_to_bm25_reindex_started,
        )
        summary['checks']['same_relation_type_conversion_replay'] = {
            'bm25_to_sae': {
                'before_filenode': conversion_bm25_filenode,
                'after_filenode': converted_semantic['filenode'],
                'hits': converted_semantic['hits'],
                'pending_error': bm25_to_sae_pending['error'],
                'pending_layout': (
                    bm25_to_sae_pending['status']['generation']['layout']
                ),
            },
            'sae_to_bm25': {
                'before_filenode': conversion_semantic_filenode,
                'after_filenode': converted_bm25['filenode'],
                'hits': converted_bm25['hits'],
                'pending_error': sae_to_bm25_pending['error'],
                'pending_layout': (
                    sae_to_bm25_pending['status']['generation']['layout']
                ),
            },
        }

        with psycopg.connect(primary_dsn) as conn:
            conn.execute('DROP INDEX docs_bm25_idx')
            conn.execute('DROP INDEX docs_semantic_idx')
            conn.execute('DROP INDEX docs_convert_idx')
            conn.commit()
        drop_replay_started = time.monotonic()

        def drop_replay() -> bool:
            with psycopg.connect(standby_dsn) as conn:
                return not relation_exists(
                    conn,
                    'docs_bm25_idx',
                ) and not relation_exists(
                    conn,
                    'docs_semantic_idx',
                ) and not relation_exists(conn, 'docs_convert_idx')

        wait_until_timed(
            summary,
            'drop',
            'unified index drops',
            drop_replay,
            started_at=drop_replay_started,
        )
        summary['checks']['drop'] = 'ok'
        summary['passed'] = True
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception:
        for log_path in (primary_log, standby_log):
            if log_path.exists():
                print(log_path.read_text(encoding='utf-8'))
        raise
    finally:
        if semantic_maintenance_lock is not None:
            semantic_maintenance_lock.close()
        if standby_started:
            subprocess.run(
                [str(pg_ctl), '-D', str(standby_data), 'stop', '-m', 'fast'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        if primary_started:
            subprocess.run(
                [str(pg_ctl), '-D', str(primary_data), 'stop', '-m', 'fast'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == '__main__':
    main()
