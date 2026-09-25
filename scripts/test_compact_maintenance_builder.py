#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parent.parent
MB = 1024 * 1024
STANDARD_HEADROOM_NUM = 3
STANDARD_HEADROOM_DEN = 5
COMPACT_HEADROOM_NUM = 3
COMPACT_HEADROOM_DEN = 4


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def headroom_budget_bytes(estimate: int, numerator: int, denominator: int) -> int:
    return ceil_div(estimate * denominator, numerator)


def headroom_admitted(
    estimate: int,
    budget_bytes: int,
    numerator: int,
    denominator: int,
) -> bool:
    return estimate <= (budget_bytes * numerator) // denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run low-memory maintenance builder smoke checks.'
    )
    parser.add_argument(
        '--bindir',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='PostgreSQL bin directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep the temporary cluster for debugging.',
    )
    parser.add_argument(
        '--builder',
        choices=('compact', 'spill'),
        default='compact',
        help='Low-memory builder path to force with the memory budget.',
    )
    parser.add_argument(
        '--extension-libdir',
        type=Path,
        help='Directory containing the staged evoke shared library.',
    )
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share root containing extension/evoke.control, or '
            'the extension directory itself.'
        ),
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
) -> str:
    return run(psql_cmd(args, pgdata, port), input_sql=sql).stdout.strip()


def init_cluster(args: argparse.Namespace, pgdata: Path, port: int) -> None:
    bindir = Path(args.bindir)

    run([
        str(bindir / 'initdb'),
        '-D',
        str(pgdata),
        '-A',
        'trust',
        '-U',
        os.environ.get('USER', 'postgres'),
    ])
    with (pgdata / 'postgresql.conf').open('a', encoding='utf-8') as conf:
        conf.write("\nlisten_addresses = ''\n")
        conf.write(f"unix_socket_directories = '{pgdata}'\n")
        conf.write(f'port = {port}\n')
        conf.write("shared_preload_libraries = 'evoke'\n")
        if args.extension_libdir is not None:
            libdir = str(args.extension_libdir).replace("'", "''")
            conf.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if args.extension_control_dir is not None:
            control_dir = str(args.extension_control_dir).replace("'", "''")
            conf.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
        conf.write("evoke.shared_runtime_size = '16MB'\n")
        conf.write("evoke.preload_timer_interval_ms = '1h'\n")
        conf.write("evoke.maintenance_timer_interval_ms = '1h'\n")


def start_cluster(args: argparse.Namespace, pgdata: Path) -> None:
    bindir = Path(args.bindir)

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
    bindir = Path(args.bindir)

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


def estimates_from_state(state: str) -> tuple[int, int, int]:
    values = []
    for name in (
        'standard_estimated_bytes',
        'compact_estimated_bytes',
        'spill_estimated_bytes',
    ):
        match = re.search(rf'{name}=([0-9]+)', state)
        if match is None:
            raise AssertionError(
                f'could not parse {name} from generation state: {state}'
            )
        values.append(int(match.group(1)))
    return values[0], values[1], values[2]


def run_smoke(args: argparse.Namespace, pgdata: Path, port: int) -> None:
    psql(
        args,
        pgdata,
        port,
        '''
        CREATE EXTENSION evoke;
        CREATE TABLE docs (
            id int primary key,
            tokens text[] not null
        );
        INSERT INTO docs
        SELECT gs, ARRAY[
            'compact',
            'maintenance',
            'token-' || (gs % 1000)::text,
            'doc-' || gs::text
        ]
        FROM generate_series(1, 30000) gs;
        CREATE INDEX docs_bm25_idx
            ON docs USING evoke (tokens)
            WITH (
                consistency = 'eventual'
            );
        ''',
    )
    state = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state('docs_bm25_idx');",
    )
    standard_estimate, compact_estimate, spill_estimate = (
        estimates_from_state(state)
    )
    if (
        standard_estimate <= 0
        or compact_estimate <= 0
        or spill_estimate <= 0
        or standard_estimate != spill_estimate * 3
        or compact_estimate != spill_estimate * 2
    ):
        raise AssertionError(
            'v3 rebuild estimates do not describe the live payload: '
            f'standard={standard_estimate} compact={compact_estimate} '
            f'spill={spill_estimate}; state={state}'
        )
    if args.builder == 'compact':
        target_budget_bytes = headroom_budget_bytes(
            compact_estimate,
            COMPACT_HEADROOM_NUM,
            COMPACT_HEADROOM_DEN,
        )
    else:
        target_budget_bytes = spill_estimate
    budget_mb = math.ceil(target_budget_bytes / MB)
    budget_bytes = budget_mb * MB
    standard_admitted = headroom_admitted(
        standard_estimate,
        budget_bytes,
        STANDARD_HEADROOM_NUM,
        STANDARD_HEADROOM_DEN,
    )
    compact_admitted = headroom_admitted(
        compact_estimate,
        budget_bytes,
        COMPACT_HEADROOM_NUM,
        COMPACT_HEADROOM_DEN,
    )
    spill_admitted = spill_estimate <= budget_bytes
    if args.builder == 'compact':
        budget_matches = compact_admitted and not standard_admitted
    else:
        budget_matches = spill_admitted and not compact_admitted
    if not budget_matches:
        raise AssertionError(
            'test corpus did not produce the requested builder budget window: '
            f'builder={args.builder} '
            f'spill={spill_estimate} compact={compact_estimate} '
            f'standard={standard_estimate} budget={budget_bytes} '
            f'standard_admitted={standard_admitted} '
            f'compact_admitted={compact_admitted} '
            f'spill_admitted={spill_admitted}'
        )

    psql(
        args,
        pgdata,
        port,
        f'''
        ALTER SYSTEM SET evoke.maintenance_rebuild_memory_budget =
            '{budget_mb}MB';
        SELECT pg_reload_conf();
        INSERT INTO docs VALUES
            (30001, ARRAY['compact', 'maintenance', 'fresh']);
        ''',
    )
    budget_state = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state('docs_bm25_idx');",
    )
    if f'rebuild_builder={args.builder}' not in budget_state:
        raise AssertionError(
            f'v3 status did not select {args.builder}: {budget_state}'
        )
    psql(
        args,
        pgdata,
        port,
        'REINDEX INDEX docs_bm25_idx;',
    )
    rebuilt_state = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state('docs_bm25_idx');",
    )
    if f'rebuild_builder={args.builder}' not in rebuilt_state:
        raise AssertionError(
            f'v3 REINDEX did not preserve the {args.builder} admission '
            f'window: {rebuilt_state}; budget_mb={budget_mb}'
        )

    psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_preload('docs_bm25_idx');",
    )
    rows = psql(
        args,
        pgdata,
        port,
        '''
        SELECT count(*)
        FROM public.evoke_query(
            'docs_bm25_idx',
            'fresh',
            10,
            NULL
        ) h
        JOIN docs d ON d.ctid = h.ctid
        WHERE d.id = 30001;
        ''',
    )
    if rows != '1':
        raise AssertionError(
            f'{args.builder} rebuild did not publish fresh row: {rows}'
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
        args.extension_libdir = args.extension_libdir.expanduser().resolve()
        libraries = (
            args.extension_libdir / 'evoke.so',
            args.extension_libdir / 'evoke.dylib',
        )
        if not any(path.is_file() for path in libraries):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    tmpdir = Path(tempfile.mkdtemp(
        prefix='evoke-compact-builder-',
        dir='/tmp',
    ))
    pgdata = tmpdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(args, pgdata, port)
        start_cluster(args, pgdata)
        run_smoke(args, pgdata, port)
    finally:
        stop_cluster(args, pgdata)
        if args.keep:
            print(f'kept temporary cluster at {tmpdir}')
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
