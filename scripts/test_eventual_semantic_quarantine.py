#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import psycopg

from test_unified_index_lifecycle_smoke import (
    configure_cluster,
    connect,
    restart_cluster,
    run,
    sql_literal,
    start_cluster,
    stop_cluster,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'eventual_semantic_quarantine'
INDEX_NAME = f'{SCHEMA}.docs_idx'
LEXICAL_MARKERS = {
    'global': ('avocado', 'broccoli', 'cinnamon'),
    'head': ('harbor', 'jasmine', 'walnut'),
    'middle': ('pumpkin', 'rabbit', 'sunset'),
    'tail': ('winter', 'yellow', 'oxygen'),
    'retry': ('river', 'silver', 'music'),
    'restart': ('teacher', 'orange', 'banana'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate durable row-local semantic completion quarantine.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--port', type=int, default=55591)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def fetch_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_status(%s::regclass)',
            (INDEX_NAME,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError('evoke_index_status did not return JSON')
    return dict(row[0])


def completion_state(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    status = fetch_status(connection)
    return dict(
        status['generation']['delta']['semantic_completion']
    )


def quarantine_details(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_semantic_quarantine_internal('
            '%s::regclass)',
            (INDEX_NAME,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError('quarantine diagnostics did not return JSON')
    return dict(row[0])


def set_test_setting(
    connection: psycopg.Connection[Any],
    name: str,
    value: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT set_config(%s, %s, false)',
            (name, value),
        )


def acquire_guard(
    connection: psycopg.Connection[Any],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_try_maintenance_lock(%s::regclass)',
            (INDEX_NAME,),
        )
        acquired = bool(cursor.fetchone()[0])
    if not acquired:
        raise RuntimeError('could not acquire maintenance guard')


def release_guard(
    connection: psycopg.Connection[Any],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintenance_unlock(%s::regclass)',
            (INDEX_NAME,),
        )
        cursor.execute(
            'SELECT evoke_index_maintenance_lock_held(%s::regclass)',
            (INDEX_NAME,),
        )
        row = cursor.fetchone()
    if row is None or bool(row[0]):
        raise RuntimeError('maintenance guard remained held after release')


def maintain(connection: psycopg.Connection[Any]) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintain(%s::regclass)',
            (INDEX_NAME,),
        )
        row = cursor.fetchone()
    return str(row[0])


def vacuum_documents(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f'VACUUM {SCHEMA}.docs')


def maintain_until(
    connection: psycopg.Connection[Any],
    condition: Callable[[], bool],
    description: str,
    *,
    max_actions: int = 16,
    diagnostics: Callable[[], Any] | None = None,
) -> list[str]:
    actions: list[str] = []

    for _ in range(max_actions):
        if condition():
            return actions
        actions.append(maintain(connection))
    if condition():
        return actions
    raise AssertionError(
        f'maintenance did not reach {description}: actions={actions}, '
        f'state={completion_state(connection)}, '
        f'diagnostics={diagnostics() if diagnostics else None}'
    )


def maintain_until_error(
    connection: psycopg.Connection[Any],
    *,
    max_actions: int = 16,
) -> tuple[list[str], dict[str, str]]:
    actions: list[str] = []

    for _ in range(max_actions):
        try:
            actions.append(maintain(connection))
        except psycopg.Error as error:
            return actions, {
                'sqlstate': str(error.sqlstate),
                'message': str(error).splitlines()[0],
            }
    raise AssertionError(
        'maintenance did not surface the expected global error: '
        f'actions={actions}, state={completion_state(connection)}'
    )


def quarantine_reached(
    connection: psycopg.Connection[Any],
    failure_count: int,
) -> bool:
    state = completion_state(connection)

    if (
        int(state['pending']) != 1
        or state.get('pending_exact') is not True
        or state.get('converged') is True
    ):
        return False
    details = quarantine_details(connection)
    return (
        details.get('exact') is True
        and int(details.get('count', 0)) == 1
        and int(details['rows'][0]['failure_count']) == failure_count
    )


def convergence_reached(
    connection: psycopg.Connection[Any],
) -> bool:
    state = completion_state(connection)

    return (
        state.get('converged') is True
        and int(state['pending']) == 0
        and int(state.get('quarantined', 0)) == 0
    )


def setup(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE EXTENSION evoke;
            CREATE SCHEMA {SCHEMA};
            CREATE TABLE {SCHEMA}.docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            CREATE ROLE evoke_quarantine_observer NOLOGIN;
            INSERT INTO {SCHEMA}.docs VALUES (
                'base',
                'durable semantic quarantine baseline'
            );
            CREATE INDEX docs_idx
            ON {SCHEMA}.docs
            USING evoke (body)
            WITH (
                sae = true,
                model_path = {sql_literal(str(model_path))},
                consistency = eventual,
                auto_preload = 0
            );
            GRANT USAGE ON SCHEMA {SCHEMA}
            TO evoke_quarantine_observer
            """
        )


def insert_batch(
    connection: psycopg.Connection[Any],
    prefix: str,
    poison_position: int | None,
) -> str | None:
    rows: list[tuple[str, str]] = []
    poison_id: str | None = None

    for position in range(3):
        doc_id = f'{prefix}-{position}'
        marker = LEXICAL_MARKERS[prefix][position]
        body = f'{marker} ordinary semantic completion document'
        if position == poison_position:
            poison_id = doc_id
            body = (
                f'{marker} POISON_{prefix} semantic completion document'
            )
        rows.append((doc_id, body))
    with connection.cursor() as cursor:
        cursor.executemany(
            f'INSERT INTO {SCHEMA}.docs (id, body) VALUES (%s, %s)',
            rows,
        )
    return poison_id


def query_rows(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    oracle: bool,
) -> list[tuple[str, float]]:
    set_test_setting(
        connection,
        'evoke.test_unified_overlay_oracle',
        'on' if oracle else 'off',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                '{INDEX_NAME}'::regclass,
                %s,
                20
            ) AS hit
            JOIN {SCHEMA}.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (query,),
        )
        return [
            (str(row[0]), round(float(row[1]), 5))
            for row in cursor.fetchall()
        ]


def assert_quarantine(
    connection: psycopg.Connection[Any],
    poison_id: str,
    marker: str,
    *,
    failure_count: int,
) -> dict[str, Any]:
    completion = completion_state(connection)
    details = quarantine_details(connection)
    normal = query_rows(connection, marker, oracle=False)
    oracle = query_rows(connection, marker, oracle=True)

    if int(completion['pending']) != 1:
        raise AssertionError(f'quarantine debt is not one: {completion}')
    if completion['pending_exact'] is not True:
        raise AssertionError(f'pending state is not exact: {completion}')
    if completion['converged'] is not False:
        raise AssertionError(f'quarantine reported convergence: {completion}')
    if int(details['count']) != 1 or details['exact'] is not True:
        raise AssertionError(f'invalid owner diagnostics: {details}')
    if int(details['rows'][0]['failure_count']) != failure_count:
        raise AssertionError(f'invalid failure count: {details}')
    if poison_id not in {row[0] for row in normal}:
        raise AssertionError(
            'quarantined row lost lexical visibility: '
            f'normal={normal}, oracle={oracle}, details={details}'
        )
    if normal != oracle:
        raise AssertionError('quarantine query differs from overlay oracle')
    return {
        'completion': completion,
        'diagnostics': details,
        'query_rows': normal,
    }


def assert_converged(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    completion = completion_state(connection)
    details = quarantine_details(connection)

    if completion['converged'] is not True:
        raise AssertionError(f'index did not converge: {completion}')
    if int(completion['pending']) != 0:
        raise AssertionError(f'pending rows remain: {completion}')
    if int(details['count']) != 0 or details['exact'] is not True:
        raise AssertionError(f'quarantine diagnostics remain: {details}')
    return completion


def assert_restart_preserved_quarantine(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if (
        before.get('count') != after.get('count')
        or before.get('exact') is not True
        or after.get('exact') is not True
        or len(before.get('rows', [])) != 1
        or len(after.get('rows', [])) != 1
    ):
        raise AssertionError(
            'quarantine summary changed across restart: '
            f'before={before}, after={after}'
        )
    before_row = before['rows'][0]
    after_row = after['rows'][0]
    durable_keys = (
        'ctid',
        'sqlstate',
        'error_hash',
        'failure_count',
    )
    if any(before_row[key] != after_row[key] for key in durable_keys):
        raise AssertionError(
            'quarantine identity changed across restart: '
            f'before={before}, after={after}'
        )
    if int(after_row['pending_age_ms']) < int(before_row['pending_age_ms']):
        raise AssertionError(
            'quarantine age moved backwards across restart: '
            f'before={before}, after={after}'
        )
    if int(after_row['retry_in_ms']) > int(before_row['retry_in_ms']):
        raise AssertionError(
            'quarantine retry moved backwards across restart: '
            f'before={before}, after={after}'
        )


def assert_owner_only_diagnostics(
    connection: psycopg.Connection[Any],
) -> dict[str, str]:
    denied: dict[str, str] = {}

    with connection.cursor() as cursor:
        cursor.execute('SET ROLE evoke_quarantine_observer')
        try:
            cursor.execute(
                'SELECT evoke_index_semantic_quarantine_internal('
                '%s::regclass)',
                (INDEX_NAME,),
            )
        except psycopg.Error as error:
            denied = {
                'sqlstate': str(error.sqlstate),
                'message': str(error).splitlines()[0],
            }
        finally:
            cursor.execute('RESET ROLE')
    if denied.get('sqlstate') != '42501':
        raise AssertionError(
            f'quarantine diagnostics were not owner-only: {denied}'
        )
    if 'evoke_index_semantic_quarantine_internal' not in denied['message']:
        raise AssertionError(
            f'quarantine ACL failed at the wrong boundary: {denied}'
        )
    return denied


def assert_unresolved_identity_hidden(
    writer: psycopg.Connection[Any],
    unresolved: psycopg.Connection[Any],
) -> dict[str, Any]:
    unresolved.autocommit = False
    with unresolved.cursor() as cursor:
        cursor.execute(
            f'INSERT INTO {SCHEMA}.docs (id, body) VALUES (%s, %s)',
            ('unresolved', 'uncommitted semantic identity'),
        )
    during = quarantine_details(writer)
    if during.get('exact') is not False or during.get('rows') != []:
        raise AssertionError(
            f'unresolved semantic identity was exposed: {during}'
        )
    unresolved.rollback()
    unresolved.autocommit = True
    after = quarantine_details(writer)
    if after.get('exact') is not True or after.get('rows') != []:
        raise AssertionError(
            f'aborted semantic identity remained visible: {after}'
        )
    return {'during': during, 'after': after}


def run_retry_scenario(
    writer: psycopg.Connection[Any],
    maintenance: psycopg.Connection[Any],
) -> dict[str, Any]:
    poison_position = 1
    poison_id = insert_batch(writer, 'retry', poison_position)

    if poison_id is None:
        raise AssertionError('retry scenario has no poison row')
    set_test_setting(
        maintenance,
        'evoke.test_semantic_quarantine_retry_ms',
        '0',
    )
    set_test_setting(
        maintenance,
        'evoke.test_semantic_completion_fail_pattern',
        'POISON_retry',
    )
    first_result = maintain_until(
        maintenance,
        lambda: quarantine_reached(writer, 1),
        'first row-local quarantine',
    )
    first = assert_quarantine(
        writer,
        poison_id,
        LEXICAL_MARKERS['retry'][poison_position],
        failure_count=1,
    )
    second_result = maintain_until(
        maintenance,
        lambda: quarantine_reached(writer, 2),
        'second row-local quarantine',
    )
    second = assert_quarantine(
        writer,
        poison_id,
        LEXICAL_MARKERS['retry'][poison_position],
        failure_count=2,
    )
    set_test_setting(
        maintenance,
        'evoke.test_semantic_completion_fail_pattern',
        '',
    )
    recovery_result = maintain_until(
        maintenance,
        lambda: convergence_reached(writer),
        'retry recovery convergence',
    )
    final = assert_converged(writer)
    set_test_setting(
        maintenance,
        'evoke.test_semantic_quarantine_retry_ms',
        '',
    )
    return {
        'first_result': first_result,
        'first': first,
        'second_result': second_result,
        'second': second,
        'recovery_result': recovery_result,
        'final': final,
    }


def run_scenario(
    writer: psycopg.Connection[Any],
    maintenance: psycopg.Connection[Any],
    prefix: str,
    poison_position: int,
    recovery: str,
) -> dict[str, Any]:
    poison_id = insert_batch(writer, prefix, poison_position)
    if poison_id is None:
        raise AssertionError('scenario has no poison row')
    set_test_setting(
        maintenance,
        'evoke.test_semantic_completion_fail_pattern',
        f'POISON_{prefix}',
    )
    result = maintain_until(
        maintenance,
        lambda: quarantine_reached(writer, 1),
        f'{prefix} row-local quarantine',
    )
    observation = assert_quarantine(
        writer,
        poison_id,
        LEXICAL_MARKERS[prefix][poison_position],
        failure_count=1,
    )
    observation['initial_result'] = result

    if recovery == 'retry':
        set_test_setting(
            maintenance,
            'evoke.test_semantic_quarantine_retry_ms',
            '0',
        )
        with writer.cursor() as cursor:
            cursor.execute(
                f'UPDATE {SCHEMA}.docs SET body = %s WHERE id = %s',
                (f'fixed_{prefix} semantic completion document', poison_id),
            )
        vacuum_documents(maintenance)
        observation['recovery_actions'] = maintain_until(
            maintenance,
            lambda: convergence_reached(writer),
            f'{prefix} update recovery',
            diagnostics=lambda: quarantine_details(writer),
        )
    elif recovery == 'delete':
        with writer.cursor() as cursor:
            cursor.execute(
                f'DELETE FROM {SCHEMA}.docs WHERE id = %s',
                (poison_id,),
            )
        vacuum_documents(maintenance)
        observation['delete_result'] = maintain_until(
            maintenance,
            lambda: convergence_reached(writer),
            f'{prefix} delete recovery',
            diagnostics=lambda: quarantine_details(writer),
        )
    elif recovery == 'reindex':
        set_test_setting(
            maintenance,
            'evoke.test_semantic_completion_fail_pattern',
            '',
        )
        with maintenance.cursor() as cursor:
            cursor.execute(f'REINDEX INDEX {INDEX_NAME}')
        observation['reindex_result'] = 'reindexed'
    else:
        raise ValueError(f'unknown recovery mode: {recovery}')

    observation['final'] = assert_converged(writer)
    set_test_setting(
        maintenance,
        'evoke.test_semantic_completion_fail_pattern',
        '',
    )
    set_test_setting(
        maintenance,
        'evoke.test_semantic_quarantine_retry_ms',
        '',
    )
    return observation


def main() -> None:
    args = parse_args()
    if not (args.model_path / 'manifest.json').is_file():
        raise FileNotFoundError(
            f'model checkout is missing: {args.model_path}'
        )

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    data_dir = root / 'data'
    socket_dir = root / 'socket'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
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
        args.port,
        args.extension_libdir,
        args.extension_control_dir,
    )
    with (data_dir / 'postgresql.conf').open(
        'a',
        encoding='utf-8',
    ) as handle:
        handle.write(
            "evoke.maintenance_timer_interval_ms = '3600000ms'\n"
        )
    start_cluster(pg_ctl, data_dir, log_path)

    writer = connect(socket_dir, args.port)
    maintenance = connect(socket_dir, args.port)
    report: dict[str, Any] = {}
    try:
        setup(writer, args.model_path)
        acquire_guard(maintenance)
        unresolved = connect(socket_dir, args.port)
        try:
            report['unresolved_identity'] = (
                assert_unresolved_identity_hidden(writer, unresolved)
            )
        finally:
            unresolved.close()

        insert_batch(writer, 'global', None)
        set_test_setting(
            maintenance,
            'evoke.test_semantic_completion_fail_global',
            'on',
        )
        global_actions, global_error = maintain_until_error(maintenance)
        global_state = completion_state(writer)
        if not global_error.get('sqlstate', '').startswith('08'):
            raise AssertionError(f'global error was not propagated: {global_error}')
        if int(global_state['pending']) < 1:
            raise AssertionError(
                f'global failure lost pending rows: {global_state}'
            )
        if global_state['pending_exact'] is not True:
            raise AssertionError(
                f'global failure left an inexact frontier: {global_state}'
            )
        if int(global_state['sealed_pending']) < 1:
            raise AssertionError(
                f'global failure did not preserve sealed work: {global_state}'
            )
        if int(global_state.get('quarantined', 0)) != 0:
            raise AssertionError(
                f'global failure quarantined rows: {global_state}'
            )
        set_test_setting(
            maintenance,
            'evoke.test_semantic_completion_fail_global',
            'off',
        )
        report['global_failure'] = {
            'error': global_error,
            'actions': global_actions,
            'state': global_state,
            'recovery_result': maintain_until(
                maintenance,
                lambda: convergence_reached(writer),
                'global failure recovery',
            ),
            'final': assert_converged(writer),
        }
        report['owner_only_diagnostics'] = (
            assert_owner_only_diagnostics(writer)
        )
        report['retry'] = run_retry_scenario(
            writer,
            maintenance,
        )

        report['head'] = run_scenario(
            writer,
            maintenance,
            'head',
            0,
            'retry',
        )
        report['middle'] = run_scenario(
            writer,
            maintenance,
            'middle',
            1,
            'reindex',
        )
        report['tail'] = run_scenario(
            writer,
            maintenance,
            'tail',
            2,
            'delete',
        )

        insert_batch(writer, 'restart', 1)
        set_test_setting(
            maintenance,
            'evoke.test_semantic_completion_fail_pattern',
            'POISON_restart',
        )
        restart_actions = maintain_until(
            maintenance,
            lambda: quarantine_reached(writer, 1),
            'restart quarantine',
        )
        before_restart = quarantine_details(writer)
        release_guard(maintenance)
        writer.close()
        maintenance.close()
        restart_cluster(pg_ctl, data_dir, log_path)
        writer = connect(socket_dir, args.port)
        maintenance = connect(socket_dir, args.port)
        acquire_guard(maintenance)
        after_restart = quarantine_details(writer)
        assert_restart_preserved_quarantine(
            before_restart,
            after_restart,
        )
        with writer.cursor() as cursor:
            cursor.execute(
                f'UPDATE {SCHEMA}.docs SET body = %s WHERE id = %s',
                ('restart fixed semantic completion document', 'restart-1'),
            )
        vacuum_documents(maintenance)
        recovery_actions = maintain_until(
            maintenance,
            lambda: convergence_reached(writer),
            'restart recovery',
        )
        report['restart'] = {
            'quarantine_actions': restart_actions,
            'before': before_restart,
            'after': after_restart,
            'recovery_actions': recovery_actions,
            'final': assert_converged(writer),
        }
        report['passed'] = True
    finally:
        try:
            if not maintenance.closed:
                try:
                    release_guard(maintenance)
                except psycopg.Error:
                    pass
            if not writer.closed:
                with writer.cursor() as cursor:
                    cursor.execute(
                        f'DROP SCHEMA IF EXISTS {SCHEMA} CASCADE'
                    )
        finally:
            if not writer.closed:
                writer.close()
            if not maintenance.closed:
                maintenance.close()
            stop_cluster(pg_ctl, data_dir)
            temporary.cleanup()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
