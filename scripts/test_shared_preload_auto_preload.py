#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parent.parent
MANY_INDEX_COUNT = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run shared-preload auto_preload reloption smoke checks.'
    )
    parser.add_argument(
        '--bindir',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='PostgreSQL bin directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--cache-mb',
        type=int,
        default=64,
        help='Main shared runtime arena size for the temporary cluster.',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Seconds to wait for background auto preload.',
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep the temporary cluster for debugging.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
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
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_sql,
        text=True,
        cwd=cwd,
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


def psql(args: argparse.Namespace, pgdata: Path, port: int, sql: str) -> str:
    result = run(psql_cmd(args, pgdata, port), input_sql=sql)
    return result.stdout.strip()


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
        if args.extension_libdir is not None:
            libdir = str(args.extension_libdir).replace("'", "''")
            conf.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if args.extension_control_dir is not None:
            control_dir = str(
                args.extension_control_dir
            ).replace("'", "''")
            conf.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
        conf.write(f"unix_socket_directories = '{pgdata}'\n")
        conf.write(f'port = {port}\n')
        conf.write("shared_preload_libraries = 'evoke'\n")
        conf.write(
            f"evoke.shared_runtime_size = '{args.cache_mb}MB'\n"
        )
        conf.write('evoke.maintenance_worker_limit = 1\n')
        conf.write("evoke.preload_timer_interval_ms = '1000ms'\n")
        conf.write("evoke.maintenance_timer_interval_ms = '1000ms'\n")

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
    subprocess.run(
        [
            str(Path(args.bindir) / 'pg_ctl'),
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


def require_invalid_auto_preload(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> None:
    result = run(
        psql_cmd(args, pgdata, port),
        input_sql='''
        CREATE TABLE docs_invalid (
            id int primary key,
            tokens text[] not null
        );
        CREATE INDEX docs_invalid_bm25_idx
            ON docs_invalid USING evoke (tokens)
            WITH (auto_preload = -1);
        ''',
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError('auto_preload accepted a negative value')
    if 'auto_preload' not in result.stderr + result.stdout:
        raise AssertionError(
            'negative auto_preload failed without naming the option: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
    psql(args, pgdata, port, 'DROP TABLE IF EXISTS docs_invalid;')


def setup(args: argparse.Namespace, pgdata: Path, port: int) -> None:
    psql(
        args,
        pgdata,
        port,
        '''
        CREATE EXTENSION evoke;
        ''',
    )
    require_invalid_auto_preload(args, pgdata, port)
    psql(
        args,
        pgdata,
        port,
        '''
        CREATE TABLE docs_high (
            id int primary key,
            tokens text[] not null
        );
        CREATE TABLE docs_low (LIKE docs_high INCLUDING ALL);
        CREATE TABLE docs_stale (LIKE docs_high INCLUDING ALL);
        CREATE TABLE docs_cold (LIKE docs_high INCLUDING ALL);

        INSERT INTO docs_high
        SELECT gs, ARRAY['auto', 'preload', 'high', gs::text]
        FROM generate_series(1, 2000) gs;
        INSERT INTO docs_low
        SELECT gs, ARRAY['auto', 'preload', 'low', gs::text]
        FROM generate_series(1, 2000) gs;
        INSERT INTO docs_stale
        SELECT gs, ARRAY['auto', 'preload', 'stale', gs::text]
        FROM generate_series(1, 2000) gs;
        INSERT INTO docs_cold
        SELECT gs, ARRAY['auto', 'preload', 'cold', gs::text]
        FROM generate_series(1, 2000) gs;

        CREATE INDEX docs_high_bm25_idx
            ON docs_high USING evoke (tokens)
            WITH (auto_preload = 10);
        CREATE INDEX docs_low_bm25_idx
            ON docs_low USING evoke (tokens)
            WITH (auto_preload = 1);
        CREATE INDEX docs_stale_bm25_idx
            ON docs_stale USING evoke (tokens)
            WITH (consistency = 'manual', auto_preload = 5);
        CREATE INDEX docs_cold_bm25_idx
            ON docs_cold USING evoke (tokens);

        INSERT INTO docs_stale VALUES
            (2001, ARRAY['auto', 'preload', 'stale', 'delta']);

        CREATE TABLE docs_many (
            id int primary key,
            tokens text[] not null
        );
        INSERT INTO docs_many
        SELECT gs, ARRAY['auto', 'preload', 'many', gs::text]
        FROM generate_series(1, 20) gs;
        ''',
    )
    psql(
        args,
        pgdata,
        port,
        '\n'.join(
            f'''
            CREATE INDEX docs_many_bm25_{i:03d}_idx
                ON docs_many USING evoke (tokens)
                WITH (auto_preload = 1);
            '''
            for i in range(MANY_INDEX_COUNT)
        ),
    )


def state(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    index_name: str,
) -> str:
    return psql(
        args,
        pgdata,
        port,
        f"""
        SELECT public.evoke_index_runtime_state(
            '{index_name}'::regclass
        );
        """,
    )


def state_value(cache_state: str, key: str) -> str:
    match = re.search(rf'{re.escape(key)}=([^,)]*)', cache_state)
    if match is None:
        raise AssertionError(f'missing {key} in state: {cache_state}')
    return match.group(1)


def resident(cache_state: str) -> bool:
    return state_value(cache_state, 'shared_preload_resident') == 'true'


def assert_worker_phase_counters_visible(cache_state: str) -> None:
    state_value(cache_state, 'active_background_workers')
    state_value(cache_state, 'active_preload_workers')
    state_value(cache_state, 'active_index_maintenance_workers')


def wait_for_auto_preload(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> tuple[str, str, str, str]:
    deadline = time.monotonic() + args.timeout
    high = low = stale = cold = ''
    saw_first_resident = False

    while time.monotonic() < deadline:
        high = state(args, pgdata, port, 'docs_high_bm25_idx')
        low = state(args, pgdata, port, 'docs_low_bm25_idx')
        stale = state(args, pgdata, port, 'docs_stale_bm25_idx')
        cold = state(args, pgdata, port, 'docs_cold_bm25_idx')

        if resident(high) or resident(low) or resident(stale) or resident(cold):
            if not resident(high):
                high = state(args, pgdata, port, 'docs_high_bm25_idx')
            if not resident(high):
                raise AssertionError(
                    'auto_preload priority was not respected: '
                    f'high={high} stale={stale} low={low} cold={cold}'
                )
            if resident(cold):
                raise AssertionError(f'unmarked index was preloaded: {cold}')
            saw_first_resident = True

        if saw_first_resident and resident(low):
            if resident(cold):
                raise AssertionError(f'unmarked index was preloaded: {cold}')
            if resident(stale):
                raise AssertionError(
                    'manual stale generation was preloaded before maintenance: '
                    f'{stale}'
                )
            if state_value(stale, 'payload_health') != 'stale':
                raise AssertionError(
                    'manual stale generation did not report stale health: '
                    f'{stale}'
                )
            return high, low, stale, cold

        time.sleep(0.1)

    raise AssertionError(
        'auto_preload did not warm marked indexes before timeout: '
        f'high={high} stale={stale} low={low} cold={cold}'
    )


def wait_for_many_preload(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> str:
    deadline = time.monotonic() + args.timeout
    high = ''
    resident_count = 0

    while time.monotonic() < deadline:
        high = state(args, pgdata, port, 'docs_high_bm25_idx')
        resident_count = int(psql(
            args,
            pgdata,
            port,
            r'''
            SELECT count(*)
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname ~ '^docs_many_bm25_[0-9]{3}_idx$'
              AND public.evoke_index_shared_preload_resident(
                    relation.oid
                  );
            ''',
        ))
        if resident_count == MANY_INDEX_COUNT:
            assert_worker_phase_counters_visible(high)
            return high
        time.sleep(0.1)

    raise AssertionError(
        'auto_preload registry did not admit all marked indexes: '
        f'resident={resident_count}/{MANY_INDEX_COUNT} state={high}'
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

    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_shared_auto_preload_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(args, pgdata, port)
        setup(args, pgdata, port)
        high, low, stale, cold = wait_for_auto_preload(args, pgdata, port)
        many = wait_for_many_preload(args, pgdata, port)
        print('high_state=' + high)
        print('low_state=' + low)
        print('stale_state=' + stale)
        print('cold_state=' + cold)
        print('many_state=' + many)
        print('shared preload auto_preload smoke passed')
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'kept temporary cluster at {workdir}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
