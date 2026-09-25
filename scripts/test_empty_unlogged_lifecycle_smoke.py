#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import psycopg

from ii42_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY = 'obsidian hummingbird quantum relay exact sentinel'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate empty and UNLOGGED ii42 BM25/SAE index lifecycle.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument(
        '--extension-libdir',
        type=Path,
        help='Directory containing the staged ii42 shared library.',
    )
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share root containing extension/ii42.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--port', type=int, default=55443)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


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


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def pg_config_value(pg_bin: Path, option: str) -> Path:
    result = subprocess.run(
        [str(pg_bin / 'pg_config'), option],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def quote_config(value: Path) -> str:
    return str(value).replace("'", "''")


def configure_cluster(
    data_dir: Path,
    socket_dir: Path,
    port: int,
    *,
    extension_libdir: Path,
    system_libdir: Path,
    extension_control_dir: Path,
    system_sharedir: Path,
) -> None:
    config = data_dir / 'postgresql.conf'
    with config.open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'ii42'\n")
        handle.write("ii42.shared_runtime_size = '64MB'\n")
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{socket_dir}'\n")
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 16\n')
        handle.write("ii42.maintenance_timer_interval_ms = '100ms'\n")
        handle.write(
            "dynamic_library_path = '"
            f'{quote_config(extension_libdir)}:'
            f"{quote_config(system_libdir)}'\n"
        )
        handle.write(
            "extension_control_path = '"
            f'{quote_config(extension_control_dir)}:'
            f"{quote_config(system_sharedir)}'\n"
        )


def start_cluster(pg_ctl: Path, data_dir: Path, log_path: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            '-l',
            str(log_path),
            'start',
            '-w',
        ]
    )


def stop_cluster(pg_ctl: Path, data_dir: Path, mode: str) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            'stop',
            '-m',
            mode,
            '-w',
        ],
        check=False,
    )


def connect(socket_dir: Path, port: int) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=True,
    )


def index_status(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT ii42_index_status(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f'missing status for {index_name}')
    value = row[0]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f'invalid status for {index_name}')
    return value


def generation_ready(
    status: dict[str, Any],
    *,
    expected_docs: int,
    sae: bool,
) -> bool:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        return False
    layout = generation.get('layout')
    primary = generation.get('primary')
    posting = generation.get('posting')
    if (
        not isinstance(layout, dict)
        or not isinstance(primary, dict)
        or not isinstance(posting, dict)
    ):
        return False
    base_ready = (
        status.get('query_ready') is True
        and status.get('sae_enabled') is sae
        and generation.get('atomic') is True
        and generation.get('valid') is True
        and int(generation.get('docs', -1)) == expected_docs
        and layout.get('storage') == 'convergent_segments'
        and layout.get('matches') is True
        and primary.get('storage') == 'convergent_segments'
    )
    if not base_ready:
        return False
    if not sae:
        return (
            primary.get('role') == 'bm25'
            and posting.get('role') == 'none'
            and posting.get('present') is False
            and posting.get('active') is False
            and int(posting.get('record_count', -1)) == 0
        )
    return (
        primary.get('role') == 'unified_posting'
        and posting.get('role') == 'unified_posting'
        and posting.get('present') is True
        and posting.get('valid') is True
        and posting.get('active') is True
        and int(posting.get('record_count', -1)) == expected_docs
    )


def pending_insert_ready(
    status: dict[str, Any],
    *,
    expected_base_docs: int,
    sae: bool,
) -> bool:
    generation = status.get('generation')
    details = status.get('details')
    if not isinstance(generation, dict) or not isinstance(details, dict):
        return False
    delta = generation.get('delta')
    if not isinstance(delta, dict):
        return False
    return (
        generation_ready(
            status,
            expected_docs=expected_base_docs,
            sae=sae,
        )
        and int(delta.get('records', -1)) == 1
        and int(details.get('delta_records', -1)) == 1
        and int(details.get('pending_writes', -1)) == 1
        and int(details.get('pending_deletes', -1)) == 0
    )


def search_count(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT count(*) FROM ii42_query(%s::regclass, %s, 10)',
            (index_name, QUERY),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f'missing search result for {index_name}')
    return int(row[0])


def table_count(
    connection: psycopg.Connection[Any],
    table_name: str,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT count(*) FROM {table_name}')
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f'missing table count for {table_name}')
    return int(row[0])


def setup(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> None:
    ddl = f"""
        CREATE EXTENSION ii42;
        CREATE SCHEMA lifecycle;

        CREATE TABLE lifecycle.empty_bm25 (id int, body text NOT NULL);
        CREATE INDEX empty_bm25_idx
        ON lifecycle.empty_bm25 USING ii42 (body)
        WITH (sae = false, consistency = realtime);

        CREATE TABLE lifecycle.empty_sae (id int, body text NOT NULL);
        CREATE INDEX empty_sae_idx
        ON lifecycle.empty_sae USING ii42 (body)
        WITH (
            sae = true,
            model_path = {sql_literal(str(model_path))},
            consistency = eventual
        );

        CREATE UNLOGGED TABLE lifecycle.unlogged_bm25 (
            id int,
            body text NOT NULL
        );
        INSERT INTO lifecycle.unlogged_bm25 VALUES
            (1, '{QUERY}'),
            (2, 'postgresql inverted index lifecycle');
        CREATE INDEX unlogged_bm25_idx
        ON lifecycle.unlogged_bm25 USING ii42 (body)
        WITH (sae = false, consistency = realtime);

        CREATE UNLOGGED TABLE lifecycle.unlogged_sae (
            id int,
            body text NOT NULL
        );
        INSERT INTO lifecycle.unlogged_sae VALUES
            (1, '{QUERY}'),
            (2, 'semantic posting recovery lifecycle');
        CREATE INDEX unlogged_sae_idx
        ON lifecycle.unlogged_sae USING ii42 (body)
        WITH (
            sae = true,
            model_path = {sql_literal(str(model_path))},
            consistency = eventual
        );
    """
    with connection.cursor() as cursor:
        cursor.execute(ddl)


def collect_statuses(
    connection: psycopg.Connection[Any],
) -> dict[str, dict[str, Any]]:
    return {
        name: index_status(connection, f'lifecycle.{name}_idx')
        for name in (
            'empty_bm25',
            'empty_sae',
            'unlogged_bm25',
            'unlogged_sae',
        )
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    gates: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix='ii42_empty_unlogged_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()
        model_path = args.model_path.expanduser().resolve()
        manifest_path = model_path / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'missing model manifest: {manifest_path}'
            )
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                'empty/unlogged lifecycle smoke requires the current '
                'Evoke model contract: '
                f'{manifest_path}'
            )
        system_libdir = pg_config_value(args.pg_bin, '--pkglibdir')
        system_sharedir = pg_config_value(args.pg_bin, '--sharedir')
        extension_libdir = (
            args.extension_libdir.expanduser().resolve()
            if args.extension_libdir is not None
            else system_libdir
        )
        extension_control_dir = extension_control_root(
            args.extension_control_dir
            if args.extension_control_dir is not None
            else system_sharedir
        )
        run([str(initdb), '-D', str(data_dir), '-A', 'trust', '-U', 'postgres'])
        configure_cluster(
            data_dir,
            socket_dir,
            args.port,
            extension_libdir=extension_libdir,
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        started = False
        connection: psycopg.Connection[Any] | None = None
        try:
            start_cluster(pg_ctl, data_dir, log_path)
            started = True
            connection = connect(socket_dir, args.port)
            setup(connection, model_path)

            initial = collect_statuses(connection)
            evidence['initial'] = initial
            gates['logged_empty_indexes_are_query_ready'] = (
                generation_ready(
                    initial['empty_bm25'],
                    expected_docs=0,
                    sae=False,
                )
                and generation_ready(
                    initial['empty_sae'],
                    expected_docs=0,
                    sae=True,
                )
                and search_count(connection, 'lifecycle.empty_bm25_idx') == 0
                and search_count(connection, 'lifecycle.empty_sae_idx') == 0
            )
            gates['unlogged_build_preserves_main_generation'] = (
                generation_ready(
                    initial['unlogged_bm25'],
                    expected_docs=2,
                    sae=False,
                )
                and generation_ready(
                    initial['unlogged_sae'],
                    expected_docs=2,
                    sae=True,
                )
                and search_count(
                    connection,
                    'lifecycle.unlogged_bm25_idx',
                ) > 0
                and search_count(
                    connection,
                    'lifecycle.unlogged_sae_idx',
                ) > 0
            )

            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO lifecycle.empty_bm25 VALUES (1, %s)',
                    (QUERY,),
                )
                cursor.execute(
                    'INSERT INTO lifecycle.empty_sae VALUES (1, %s)',
                    (QUERY,),
                )
            after_first_insert = collect_statuses(connection)
            evidence['after_first_insert'] = after_first_insert
            gates['first_insert_is_visible_via_unified_delta'] = (
                pending_insert_ready(
                    after_first_insert['empty_bm25'],
                    expected_base_docs=0,
                    sae=False,
                )
                and pending_insert_ready(
                    after_first_insert['empty_sae'],
                    expected_base_docs=0,
                    sae=True,
                )
                and search_count(connection, 'lifecycle.empty_bm25_idx') == 1
                and search_count(connection, 'lifecycle.empty_sae_idx') == 1
            )

            connection.close()
            connection = None
            stop_cluster(pg_ctl, data_dir, 'fast')
            started = False
            start_cluster(pg_ctl, data_dir, log_path)
            started = True
            connection = connect(socket_dir, args.port)
            after_restart = collect_statuses(connection)
            evidence['after_fast_restart'] = after_restart
            gates['fast_restart_preserves_base_and_delta'] = (
                pending_insert_ready(
                    after_restart['empty_bm25'],
                    expected_base_docs=0,
                    sae=False,
                )
                and pending_insert_ready(
                    after_restart['empty_sae'],
                    expected_base_docs=0,
                    sae=True,
                )
                and generation_ready(
                    after_restart['unlogged_bm25'],
                    expected_docs=2,
                    sae=False,
                )
                and generation_ready(
                    after_restart['unlogged_sae'],
                    expected_docs=2,
                    sae=True,
                )
                and search_count(
                    connection,
                    'lifecycle.empty_bm25_idx',
                ) == 1
                and search_count(
                    connection,
                    'lifecycle.empty_sae_idx',
                ) == 1
                and search_count(
                    connection,
                    'lifecycle.unlogged_bm25_idx',
                ) > 0
                and search_count(
                    connection,
                    'lifecycle.unlogged_sae_idx',
                ) > 0
            )

            connection.close()
            connection = None
            stop_cluster(pg_ctl, data_dir, 'immediate')
            started = False
            start_cluster(pg_ctl, data_dir, log_path)
            started = True
            connection = connect(socket_dir, args.port)
            after_crash = collect_statuses(connection)
            evidence['after_crash'] = after_crash
            evidence['unlogged_rows_after_crash'] = {
                name: table_count(connection, f'lifecycle.{name}')
                for name in ('unlogged_bm25', 'unlogged_sae')
            }
            gates['crash_restores_valid_empty_unlogged_generations'] = (
                evidence['unlogged_rows_after_crash']['unlogged_bm25'] == 0
                and evidence['unlogged_rows_after_crash']['unlogged_sae'] == 0
                and generation_ready(
                    after_crash['unlogged_bm25'],
                    expected_docs=0,
                    sae=False,
                )
                and generation_ready(
                    after_crash['unlogged_sae'],
                    expected_docs=0,
                    sae=True,
                )
                and search_count(
                    connection,
                    'lifecycle.unlogged_bm25_idx',
                ) == 0
                and search_count(
                    connection,
                    'lifecycle.unlogged_sae_idx',
                ) == 0
            )

            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO lifecycle.unlogged_bm25 VALUES (3, %s)',
                    (QUERY,),
                )
                cursor.execute(
                    'INSERT INTO lifecycle.unlogged_sae VALUES (3, %s)',
                    (QUERY,),
                )
            after_crash_insert = collect_statuses(connection)
            evidence['after_crash_insert'] = after_crash_insert
            gates['first_insert_after_crash_uses_unified_delta'] = (
                pending_insert_ready(
                    after_crash_insert['unlogged_bm25'],
                    expected_base_docs=0,
                    sae=False,
                )
                and pending_insert_ready(
                    after_crash_insert['unlogged_sae'],
                    expected_base_docs=0,
                    sae=True,
                )
                and search_count(
                    connection,
                    'lifecycle.unlogged_bm25_idx',
                ) == 1
                and search_count(
                    connection,
                    'lifecycle.unlogged_sae_idx',
                ) == 1
            )
            current_signature = initial['unlogged_sae']['generation'][
                'contract_signature'
            ]
            crash_signature = after_crash['unlogged_sae']['generation'][
                'contract_signature'
            ]
            insert_signature = after_crash_insert['unlogged_sae'][
                'generation'
            ]['contract_signature']
            evidence['sae_contract_stability'] = {
                'initial': current_signature,
                'after_crash': crash_signature,
                'after_insert': insert_signature,
            }
            gates['unlogged_contract_remains_current'] = (
                crash_signature == current_signature
                and insert_signature == current_signature
                and after_crash['unlogged_sae'].get(
                    'runtime_signature_matches'
                ) is True
                and after_crash_insert['unlogged_sae'].get(
                    'runtime_signature_matches'
                ) is True
            )
        except Exception:
            if log_path.exists():
                print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
            raise
        finally:
            if connection is not None:
                connection.close()
            if started:
                stop_cluster(pg_ctl, data_dir, 'fast')

    return {
        'api_version': 'ii42_index_v1',
        'route': 'empty and UNLOGGED unified index lifecycle',
        'model_path': str(model_path),
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    report = run_smoke(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not report['all_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
