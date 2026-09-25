#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root
from test_convergent_segment_read_smoke import (
    configure_cluster,
    pg_config_value,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BLCKSZ = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run payload-health corruption and rebuild smoke checks.'
    )
    parser.add_argument(
        '--bindir',
        type=Path,
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='PostgreSQL bin directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--json-output', type=Path)
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep the temporary cluster for debugging.',
    )
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def run(
    cmd: list[str],
    *,
    input_sql: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_sql,
        text=True,
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
    )


def psql_cmd(args: argparse.Namespace, pgdata: Path, port: int) -> list[str]:
    return [
        str(Path(args.bindir) / 'psql'),
        '-X',
        '-Atq',
        '-h',
        str(pgdata),
        '-p',
        str(port),
        '-d',
        'postgres',
        '-v',
        'ON_ERROR_STOP=1',
    ]


def psql(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    sql: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run(
        psql_cmd(args, pgdata, port),
        input_sql=sql,
        check=check,
    )


def init_cluster(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    *,
    extension_libdir: Path,
    system_libdir: Path,
    extension_control_dir: Path,
    system_sharedir: Path,
) -> None:
    bindir = args.bindir

    run([
        str(bindir / 'initdb'),
        '-D',
        str(pgdata),
        '-A',
        'trust',
        '-U',
        os.environ.get('USER', 'postgres'),
    ])
    configure_cluster(
        pgdata,
        pgdata,
        port,
        extension_libdir=extension_libdir,
        system_libdir=system_libdir,
        extension_control_dir=extension_control_dir,
        system_sharedir=system_sharedir,
    )


def start_cluster(args: argparse.Namespace, pgdata: Path) -> None:
    bindir = args.bindir

    run([
        str(bindir / 'pg_ctl'),
        '-D',
        str(pgdata),
        '-l',
        str(pgdata / 'postgres.log'),
        '-w',
        'start',
    ])


def stop_cluster(args: argparse.Namespace, pgdata: Path) -> None:
    bindir = args.bindir

    subprocess.run(
        [
            str(bindir / 'pg_ctl'),
            '-D',
            str(pgdata),
            '-m',
            'fast',
            '-w',
            'stop',
        ],
        text=True,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )


def psql_json(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    sql: str,
) -> dict[str, object]:
    output = psql(args, pgdata, port, sql).stdout.strip()
    return json.loads(output)


def run_smoke(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> dict[str, object]:
    setup = psql(
        args,
        pgdata,
        port,
        '''
        CREATE EXTENSION evoke;
        CREATE TABLE docs (
            id int primary key,
            body text not null
        );
        INSERT INTO docs
        SELECT gs, pg_catalog.format('health payload doc %s', gs)
        FROM generate_series(1, 2000) gs;
        CREATE INDEX docs_bm25_idx
            ON docs USING evoke (body)
            WITH (sae = false, consistency = 'eventual');
        SELECT pg_relation_filepath('docs_bm25_idx'::regclass);
        CHECKPOINT;
        ''',
    )
    relpath = setup.stdout.strip().splitlines()[-1]
    healthy_generation = psql_json(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_generation_audit_internal("
        "'docs_bm25_idx')::text;",
    )
    if healthy_generation.get('valid') is not True:
        raise AssertionError(
            f'healthy generation is not valid: {healthy_generation}'
        )
    if healthy_generation.get('diagnostics_complete', True) is not True:
        raise AssertionError(
            'healthy generation unexpectedly used bounded diagnostics: '
            f'{healthy_generation}'
        )

    stop_cluster(args, pgdata)
    index_path = pgdata / relpath
    with index_path.open('r+b') as index_file:
        index_file.truncate(BLCKSZ)
        index_file.flush()
        os.fsync(index_file.fileno())
    start_cluster(args, pgdata)

    state = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state('docs_bm25_idx');",
    ).stdout.strip()
    if 'payload_health=corrupt' not in state:
        raise AssertionError(f'corrupt payload was not detected: {state}')
    if 'rebuild_required=true' not in state:
        raise AssertionError(f'rebuild_required was not reported: {state}')
    if 'diagnostics_complete=false' not in state:
        raise AssertionError(f'bounded diagnostic marker is missing: {state}')

    cache_state = psql_json(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state_json("
        "'docs_bm25_idx')::text;",
    )
    cache_generation = cache_state['generation']
    if cache_generation['payload_health'] != 'corrupt':
        raise AssertionError(f'cache JSON lost corrupt health: {cache_state}')
    if cache_generation['payload_health_reason'] != (
        'segment_root_out_of_bounds'
    ):
        raise AssertionError(f'cache JSON lost health reason: {cache_state}')

    details = psql_json(
        args,
        pgdata,
        port,
        "SELECT to_jsonb(detail) FROM public.evoke_index_details("
        "'docs_bm25_idx') AS detail;",
    )
    if details['pages'] != 1 or details['index_bytes'] != BLCKSZ:
        raise AssertionError(f'physical corruption bounds are wrong: {details}')
    unknown_debt = (
        details['pending_writes'],
        details['pending_deletes'],
        details['delta_records'],
        details['delta_bytes'],
    )
    if unknown_debt != (None, None, None, None):
        raise AssertionError(f'corrupt debt was fabricated: {details}')

    generation = psql_json(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_generation_audit_internal("
        "'docs_bm25_idx')::text;",
    )
    if generation.get('diagnostics_complete') is not False:
        raise AssertionError(f'generation is not bounded: {generation}')
    if generation.get('valid') is not False:
        raise AssertionError(f'corrupt generation is valid: {generation}')
    if generation.get('health_reason') != 'segment_root_out_of_bounds':
        raise AssertionError(f'generation lost health reason: {generation}')
    if generation['delta']['records'] is not None:
        raise AssertionError(f'generation fabricated delta debt: {generation}')

    status = psql_json(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_status('docs_bm25_idx')::text;",
    )
    if status.get('query_ready') is not False:
        raise AssertionError(f'corrupt index is query-ready: {status}')
    if status.get('blocker') != 'generation_invalid':
        raise AssertionError(f'corrupt blocker is not explicit: {status}')
    if status['generation'].get('diagnostics_complete') is not False:
        raise AssertionError(f'index status expanded corrupt root: {status}')
    if status['generation'].get('docs_scope') != 'metapage_snapshot':
        raise AssertionError(f'index status fabricated sealed docs: {status}')
    if status.get('runtime_signature_matches') is not False:
        raise AssertionError(f'corrupt signature match is not false: {status}')

    preload = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_preload('docs_bm25_idx');",
        check=False,
    )
    if preload.returncode == 0:
        raise AssertionError('corrupt payload preload unexpectedly succeeded')
    if 'cannot preload corrupt evoke generation' not in preload.stderr or (
        'segment_root_out_of_bounds' not in preload.stderr
    ):
        raise AssertionError(
            'corrupt preload did not fail with bounded health: '
            f'{preload.stderr}'
        )

    query = psql(
        args,
        pgdata,
        port,
        '''
        SELECT count(*)
        FROM public.evoke_query(
            'docs_bm25_idx'::regclass,
            'health',
            10
        );
        ''',
        check=False,
    )
    if query.returncode == 0:
        raise AssertionError('corrupt payload query unexpectedly succeeded')
    if 'invalid evoke convergent segment payload' not in query.stderr:
        raise AssertionError(
            'corrupt payload did not fail fast with the expected error: '
            f'{query.stderr}'
        )
    if 'segment_root_out_of_bounds' not in query.stderr:
        raise AssertionError(
            'corrupt payload did not report the expected health reason: '
            f'{query.stderr}'
        )

    repair = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_refresh('docs_bm25_idx');",
    ).stdout.strip()
    if 'payload_health=ok' not in repair:
        raise AssertionError(
            f'corrupt payload was not explicitly rebuilt: {repair}'
        )

    repaired_state = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state('docs_bm25_idx');",
    ).stdout.strip()
    if 'payload_health=ok' not in repaired_state:
        raise AssertionError(f'repaired payload is still unhealthy: {repaired_state}')

    hits = psql(
        args,
        pgdata,
        port,
        '''
        SELECT count(*)
        FROM public.evoke_query(
            'docs_bm25_idx'::regclass,
            'health',
            10
        );
        ''',
    ).stdout.strip()
    if hits != '10':
        raise AssertionError(f'unexpected hits after repair: {hits}')

    repaired_generation = psql_json(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_generation_audit_internal("
        "'docs_bm25_idx')::text;",
    )
    if repaired_generation.get('valid') is not True:
        raise AssertionError(
            f'repaired generation is not valid: {repaired_generation}'
        )
    if repaired_generation.get('diagnostics_complete', True) is not True:
        raise AssertionError(
            'repaired generation retained bounded diagnostics: '
            f'{repaired_generation}'
        )

    return {
        'passed': True,
        'relation_path': relpath,
        'corrupt_relation_bytes': BLCKSZ,
        'health': generation['health'],
        'health_reason': generation['health_reason'],
        'unknown_debt_fields': 4,
        'query_failed_closed': True,
        'preload_failed_closed': True,
        'repair_description': repair,
        'repaired_hits': int(hits),
    }


def main() -> None:
    args = parse_args()
    args.bindir = args.bindir.expanduser().resolve()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    system_libdir = pg_config_value(args.bindir, '--pkglibdir')
    system_sharedir = pg_config_value(args.bindir, '--sharedir')
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
    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_payload_health_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(
            args,
            pgdata,
            port,
            extension_libdir=extension_libdir,
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        start_cluster(args, pgdata)
        result = run_smoke(args, pgdata, port)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        print('payload health corruption smoke passed')
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'kept temporary cluster at {workdir}')


if __name__ == '__main__':
    main()
