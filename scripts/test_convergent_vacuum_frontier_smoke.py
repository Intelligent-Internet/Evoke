#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
)
from test_convergent_segment_read_smoke import (
    DEFAULT_PG_BIN,
    configure_cluster,
    connect,
    fetch_ids,
    fetch_status,
    maintenance_result_fields,
    pg_config_value,
    reserve_port,
    run,
    start_cluster,
    stop_cluster,
    try_maintain,
)


ACTIVE_L0_MAX_RECORDS = 131_072
VACUUM_COW_BATCH_RECORDS = 65_536
DEFAULT_ROW_COUNT = 140_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate bounded convergent VACUUM retirement above the L0 '
            'record frontier in isolated PostgreSQL 18.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path, required=True)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--row-count',
        type=int,
        default=DEFAULT_ROW_COUNT,
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    generation = status['generation']
    return {
        'query_ready': status['query_ready'],
        'blocker': status['blocker'],
        'generation_id': generation['generation_id'],
        'docs': generation['docs'],
        'sealed_docs': generation['sealed_docs'],
        'document_slot_high_watermark': (
            generation['document_slot_high_watermark']
        ),
        'delta': generation['delta'],
        'retirement_statistics': generation['retirement_statistics'],
    }


def assert_empty_index(
    connection: psycopg.Connection[Any],
    *,
    label: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT count(*) FROM frontier.docs')
        heap_count = int(cursor.fetchone()[0])
    hits = fetch_ids(connection, 'frontier.docs_idx', [0])
    if heap_count != 0 or hits:
        raise AssertionError(
            f'{label} retained deleted documents: '
            f'heap_count={heap_count}, hits={hits[:5]}'
        )


def converge_frontier(
    connection: psycopg.Connection[Any],
) -> tuple[list[str], dict[str, Any]]:
    maintenance: list[str] = []
    for _ in range(32):
        status = fetch_status(connection, 'frontier.docs_idx')
        delta = status['generation']['delta']
        if (
            delta['active']['records'] == 0
            and delta['pending']['records'] == 0
        ):
            return maintenance, status

        result = try_maintain(connection, 'frontier.docs_idx')
        maintenance.append(result)
        fields = maintenance_result_fields(result)
        if fields.get('reason') in {
            'lock_busy',
            'no_pending',
            'xid_horizon',
        }:
            time.sleep(0.01)
            continue
        if fields.get('maintained') != 'true':
            raise AssertionError(
                'maintenance stopped before VACUUM frontier convergence: '
                f'result={result}, status={compact_status(status)}'
            )

    status = fetch_status(connection, 'frontier.docs_idx')
    raise AssertionError(
        'VACUUM frontier did not converge after 32 maintenance attempts: '
        f'status={compact_status(status)}, maintenance={maintenance}'
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.row_count <= ACTIVE_L0_MAX_RECORDS:
        raise ValueError(
            f'--row-count must exceed {ACTIVE_L0_MAX_RECORDS}'
        )

    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
    extension_libraries = [
        extension_libdir / name for name in ('evoke.so', 'evoke.dylib')
    ]
    if not any(path.is_file() for path in extension_libraries):
        raise FileNotFoundError(
            f'evoke library is missing from {extension_libdir}'
        )

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    system_libdir = pg_config_value(args.pg_bin, '--pkglibdir')
    system_sharedir = pg_config_value(args.pg_bin, '--sharedir')
    root = create_short_socket_root('evoke-vacuum-frontier-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    port = reserve_port()
    socket_dir.mkdir()
    gates: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    timings: dict[str, float] = {}
    smoke_started = time.perf_counter()
    started = False
    connection: psycopg.Connection[Any] | None = None

    try:
        run(
            [
                str(initdb),
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
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)

        fixture_started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION evoke')
            cursor.execute('CREATE SCHEMA frontier')
            cursor.execute(
                'CREATE TABLE frontier.docs ('
                'id int PRIMARY KEY, tokens int4[] NOT NULL)'
            )
            cursor.execute(
                'INSERT INTO frontier.docs '
                'SELECT gs, ARRAY[0, 1 + (gs %% 257)]::int4[] '
                'FROM generate_series(1, %s) gs',
                (args.row_count,),
            )
            cursor.execute(
                'CREATE INDEX docs_idx ON frontier.docs '
                'USING evoke (tokens) '
                'WITH (sae=false, consistency=realtime)'
            )
        timings['fixture_build_seconds'] = round(
            time.perf_counter() - fixture_started,
            6,
        )

        initial_status = fetch_status(connection, 'frontier.docs_idx')
        if (
            int(initial_status['generation']['docs']) != args.row_count
            or initial_status['generation']['delta']['records'] != 0
        ):
            raise AssertionError(
                'large frontier fixture did not build as one sealed root: '
                f'{compact_status(initial_status)}'
            )
        gates['large_fixture_exceeds_active_l0_frontier'] = True

        failure_started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute('DELETE FROM frontier.docs')
            cursor.execute(
                "SET evoke.test_convergent_vacuum_error_after_batch = 'on'"
            )
            try:
                cursor.execute('VACUUM (INDEX_CLEANUP ON) frontier.docs')
            except psycopg.Error as error:
                if (
                    'injected evoke convergent VACUUM document-COW batch '
                    'error' not in str(error)
                ):
                    raise
            else:
                raise AssertionError(
                    'convergent VACUUM failure injection did not fire'
                )
            finally:
                cursor.execute(
                    'RESET evoke.test_convergent_vacuum_error_after_batch'
                )
        timings['injected_vacuum_seconds'] = round(
            time.perf_counter() - failure_started,
            6,
        )

        partial_status = fetch_status(connection, 'frontier.docs_idx')
        partial_delta = partial_status['generation']['delta']
        assert_empty_index(connection, label='partial post-VACUUM failure')
        gates['partial_vacuum_batch_is_failure_safe'] = (
            int(partial_status['generation']['docs'])
            == args.row_count - VACUUM_COW_BATCH_RECORDS
            and partial_delta['records'] == 1
            and partial_delta['active']['records'] == 1
            and partial_delta['retirements'] == 1
        )
        if not gates['partial_vacuum_batch_is_failure_safe']:
            raise AssertionError(
                'failed VACUUM did not retain a bounded recoverable state: '
                f'{compact_status(partial_status)}'
            )

        partial_maintenance_started = time.perf_counter()
        partial_maintenance, partial_converged_status = converge_frontier(
            connection
        )
        timings['partial_maintenance_seconds'] = round(
            time.perf_counter() - partial_maintenance_started,
            6,
        )
        gates['failed_vacuum_fence_is_sealable'] = (
            partial_converged_status['generation']['delta']['records'] == 0
            and int(partial_converged_status['generation']['docs'])
            == args.row_count - VACUUM_COW_BATCH_RECORDS
        )
        if not gates['failed_vacuum_fence_is_sealable']:
            raise AssertionError(
                'failed VACUUM left an unsealable sequence fence: '
                f'{compact_status(partial_converged_status)}'
            )

        resumed_vacuum_started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute('VACUUM (INDEX_CLEANUP ON) frontier.docs')
        timings['resumed_vacuum_seconds'] = round(
            time.perf_counter() - resumed_vacuum_started,
            6,
        )

        retired_status = fetch_status(connection, 'frontier.docs_idx')
        retired_delta = retired_status['generation']['delta']
        assert_empty_index(connection, label='post-VACUUM')
        gates['vacuum_retires_all_documents'] = (
            retired_status['query_ready'] is True
            and int(retired_status['generation']['docs']) == 0
            and int(retired_status['generation']['sealed_docs']) == 0
        )
        gates['vacuum_keeps_l0_retirement_bounded'] = (
            retired_delta['records'] <= 1
            and retired_delta['active']['records']
            + retired_delta['pending']['records']
            == retired_delta['records']
            and retired_delta['retirements']
            == retired_delta['records']
        )
        if not (
            gates['vacuum_retires_all_documents']
            and gates['vacuum_keeps_l0_retirement_bounded']
        ):
            raise AssertionError(
                'large VACUUM did not publish bounded retirement state: '
                f'{compact_status(retired_status)}'
            )

        with connection.cursor() as cursor:
            cursor.execute('VACUUM (INDEX_CLEANUP ON) frontier.docs')
        repeated_status = fetch_status(connection, 'frontier.docs_idx')
        repeated_delta = repeated_status['generation']['delta']
        gates['repeat_vacuum_is_idempotent'] = (
            int(repeated_status['generation']['docs']) == 0
            and repeated_delta['records'] <= retired_delta['records']
            and repeated_delta['retirements']
            == repeated_delta['records']
        )
        if not gates['repeat_vacuum_is_idempotent']:
            raise AssertionError(
                'repeat VACUUM changed retirement state: '
                f'before={compact_status(retired_status)}, '
                f'after={compact_status(repeated_status)}'
            )

        maintenance_started = time.perf_counter()
        maintenance, converged_status = converge_frontier(connection)
        timings['final_maintenance_seconds'] = round(
            time.perf_counter() - maintenance_started,
            6,
        )
        assert_empty_index(connection, label='post-maintenance')
        gates['maintenance_seals_vacuum_frontier'] = (
            converged_status['generation']['delta']['records'] == 0
            and int(converged_status['generation']['docs']) == 0
            and int(converged_status['generation']['sealed_docs']) == 0
        )
        if not gates['maintenance_seals_vacuum_frontier']:
            raise AssertionError(
                'sealed VACUUM frontier retained live state: '
                f'{compact_status(converged_status)}'
            )

        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_status = fetch_status(connection, 'frontier.docs_idx')
        assert_empty_index(connection, label='post-restart')
        gates['restart_preserves_converged_retirement'] = (
            restarted_status['generation']['delta']['records'] == 0
            and int(restarted_status['generation']['docs']) == 0
        )

        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO frontier.docs VALUES '
                '(%s, ARRAY[0, 999]::int4[])',
                (args.row_count + 1,),
            )
        appended_hits = fetch_ids(connection, 'frontier.docs_idx', [999])
        gates['post_frontier_insert_is_searchable'] = len(appended_hits) == 1
        if not gates['post_frontier_insert_is_searchable']:
            raise AssertionError(
                'index rejected or hid a write after large retirement: '
                f'hits={appended_hits}'
            )

        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_hits = fetch_ids(connection, 'frontier.docs_idx', [999])
        gates['post_frontier_insert_survives_restart'] = (
            restarted_hits == appended_hits
        )
        if not gates['post_frontier_insert_survives_restart']:
            raise AssertionError(
                'post-frontier write changed across restart: '
                f'before={appended_hits}, after={restarted_hits}'
            )

        evidence = {
            'row_count': args.row_count,
            'active_l0_max_records': ACTIVE_L0_MAX_RECORDS,
            'vacuum_cow_batch_records': VACUUM_COW_BATCH_RECORDS,
            'initial_status': compact_status(initial_status),
            'partial_status': compact_status(partial_status),
            'partial_maintenance': partial_maintenance,
            'partial_converged_status': compact_status(
                partial_converged_status
            ),
            'retired_status': compact_status(retired_status),
            'repeat_vacuum_status': compact_status(repeated_status),
            'maintenance': maintenance,
            'converged_status': compact_status(converged_status),
            'restarted_status': compact_status(restarted_status),
            'post_frontier_hits': appended_hits,
            'restarted_post_frontier_hits': restarted_hits,
            'timings_seconds': timings,
        }
    except Exception:
        if log_path.is_file():
            print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
        raise
    finally:
        if connection is not None:
            connection.close()
        if started:
            stop_cluster(pg_ctl, data_dir)
        shutil.rmtree(root, ignore_errors=True)

    timings['total_seconds'] = round(
        time.perf_counter() - smoke_started,
        6,
    )
    return {
        'api_version': 'evoke_index_v1',
        'route': 'bounded convergent VACUUM retirement frontier',
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.output.unlink(missing_ok=True)
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
