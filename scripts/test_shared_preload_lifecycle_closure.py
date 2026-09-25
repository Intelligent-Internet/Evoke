#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parent.parent
RETIRED_GENERATION_FIELDS = frozenset({
    'share_eligible',
    'share_required',
    'descriptor_present',
    'descriptor_valid',
    'mapped_size',
})
RETIRED_SHARED_PRELOAD_FIELDS = frozenset({
    'admission_misses',
    'admission_miss_bytes',
    'last_admission_miss',
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run shared-preload cache lifecycle closure checks.'
    )
    parser.add_argument(
        '--bindir',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='PostgreSQL bin directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=45,
        help='Seconds to wait for background preload transitions.',
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep temporary clusters for debugging.',
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
    result = run(psql_cmd(args, pgdata, port), input_sql=sql)
    return result.stdout.strip()


def init_cluster(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    cache_mb: int,
    prewarm_max_mb: int = 1,
) -> None:
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
        conf.write(f"evoke.shared_runtime_size = '{cache_mb}MB'\n")
        conf.write(
            f"evoke.prewarm_max_bytes = '{prewarm_max_mb}MB'\n"
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


def read_log(pgdata: Path) -> str:
    log_path = pgdata / 'postgres.log'

    if not log_path.exists():
        return ''
    return log_path.read_text(encoding='utf-8', errors='replace')


def json_state(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    index_name: str,
) -> dict[str, Any]:
    payload = psql(
        args,
        pgdata,
        port,
        "SELECT public.evoke_index_runtime_state_json("
        f"'{index_name}'::regclass);",
    )
    return json.loads(payload)


def wait_for(
    timeout: int,
    producer: Callable[[], Any],
    predicate: Callable[[Any], bool],
    description: str,
) -> Any:
    deadline = time.monotonic() + timeout
    last_value: Any = None

    while time.monotonic() < deadline:
        last_value = producer()
        if predicate(last_value):
            return last_value
        time.sleep(0.2)
    raise AssertionError(f'timed out waiting for {description}: {last_value}')


def query_index(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    index_name: str,
    table_name: str,
    term: str,
) -> None:
    hits = psql(
        args,
        pgdata,
        port,
        f'''
        SELECT count(*)
        FROM public.evoke_query(
            '{index_name}'::regclass,
            '{term}',
            5,
            NULL,
            true,
            NULL,
            false,
            false
        ) h
        JOIN {table_name} d ON d.ctid = h.ctid
        WHERE h.score > 0;
        ''',
    )
    if hits != '5':
        raise AssertionError(
            f'unexpected hits for {index_name}: hits={hits}'
        )


def assert_plain_count_avoids_evoke_full_scan(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    table_name: str,
    index_name: str,
    expected_rows: int,
) -> None:
    plan = psql(
        args,
        pgdata,
        port,
        f'EXPLAIN (COSTS OFF) SELECT count(*) FROM {table_name};',
    )
    if index_name in plan:
        raise AssertionError(
            'plain count(*) should not use evoke as a full-index scan: '
            f'index={index_name}, plan={plan}'
        )

    count = psql(args, pgdata, port, f'SELECT count(*) FROM {table_name};')
    if count != str(expected_rows):
        raise AssertionError(
            f'unexpected count for {table_name}: {count} != {expected_rows}'
        )


def assert_ordered_retrieval_still_uses_evoke(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    table_name: str,
    index_name: str,
) -> None:
    plan = psql(
        args,
        pgdata,
        port,
        f'''
        EXPLAIN (COSTS OFF)
        SELECT id
        FROM {table_name}
        ORDER BY tokens <=> ARRAY['oversized']::text[]
        LIMIT 5;
        ''',
    )
    if index_name not in plan:
        raise AssertionError(
            'ordered retrieval should still use evoke index scan: '
            f'index={index_name}, plan={plan}'
        )


def require_extension(args: argparse.Namespace, pgdata: Path, port: int) -> None:
    psql(args, pgdata, port, 'CREATE EXTENSION evoke;')


def assert_maintenance_worker_quiesce(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> None:
    psql(
        args,
        pgdata,
        port,
        '''
        ALTER SYSTEM SET evoke.maintenance_worker_limit = 0;
        SELECT pg_reload_conf();
        ''',
    )
    wait_for(
        args.timeout,
        lambda: psql(
            args,
            pgdata,
            port,
            'SHOW evoke.maintenance_worker_limit;',
        ),
        lambda value: value == '0',
        'maintenance worker quiesce reload',
    )
    psql(
        args,
        pgdata,
        port,
        'SELECT public.evoke_index_touch_maintenance();',
    )
    for _ in range(10):
        active_workers = psql(
            args,
            pgdata,
            port,
            '''
            SELECT count(*)
            FROM pg_stat_activity
            WHERE backend_type = 'evoke background';
            ''',
        )
        if active_workers != '0':
            raise AssertionError(
                'maintenance worker launched while admission was quiesced: '
                f'active={active_workers}'
            )
        time.sleep(0.1)

    psql(
        args,
        pgdata,
        port,
        '''
        ALTER SYSTEM SET evoke.maintenance_worker_limit = 1;
        SELECT pg_reload_conf();
        ''',
    )
    wait_for(
        args.timeout,
        lambda: psql(
            args,
            pgdata,
            port,
            'SHOW evoke.maintenance_worker_limit;',
        ),
        lambda value: value == '1',
        'maintenance worker admission restore',
    )


def set_maintenance_worker_limit(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    limit: int,
) -> None:
    psql(
        args,
        pgdata,
        port,
        f'''
        ALTER SYSTEM SET evoke.maintenance_worker_limit = {limit};
        SELECT pg_reload_conf();
        ''',
    )
    wait_for(
        args.timeout,
        lambda: psql(
            args,
            pgdata,
            port,
            'SHOW evoke.maintenance_worker_limit;',
        ),
        lambda value: value == str(limit),
        f'maintenance worker limit {limit}',
    )


def maintain_until_converged(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    index_name: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {}

    for _ in range(24):
        psql(
            args,
            pgdata,
            port,
            "SET evoke.test_convergent_l0_rotation_records = '1';\n"
            'SELECT public.evoke_index_maintain('
            f"'{index_name}'::regclass);\n"
            'RESET evoke.test_convergent_l0_rotation_records;',
        )
        state = json_state(args, pgdata, port, index_name)
        if int(state['debt']['delta_records']) == 0:
            return state
    raise AssertionError(
        f'index did not converge after direct maintenance: {state}'
    )


def run_oversized_exact_root_case(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_shared_lifecycle_miss_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(args, pgdata, port, cache_mb=1)
        require_extension(args, pgdata, port)
        assert_maintenance_worker_quiesce(args, pgdata, port)
        psql(
            args,
            pgdata,
            port,
            '''
            CREATE TABLE docs_oversized (
                id int primary key,
                tokens text[] not null
            );
            INSERT INTO docs_oversized
            SELECT gs, ARRAY[
                'oversized', 'auto', 'preload', 'token_' || gs::text,
                'x_' || (gs % 1000)::text,
                'y_' || (gs % 2000)::text,
                'z_' || (gs % 3000)::text,
                'w_' || (gs % 4000)::text
            ]
            FROM generate_series(1, 80000) gs;
            CREATE INDEX docs_oversized_bm25_idx
                ON docs_oversized USING evoke (tokens)
                WITH (auto_preload = 12);
            ''',
        )
        assert_plain_count_avoids_evoke_full_scan(
            args,
            pgdata,
            port,
            'docs_oversized',
            'docs_oversized_bm25_idx',
            80000,
        )
        assert_ordered_retrieval_still_uses_evoke(
            args,
            pgdata,
            port,
            'docs_oversized',
            'docs_oversized_bm25_idx',
        )
        state = wait_for(
            args.timeout,
            lambda: json_state(
                args,
                pgdata,
                port,
                'docs_oversized_bm25_idx',
            ),
            lambda value: (
                value['shared_preload']['resident'] and
                not value['shared_preload']['loading']
            ),
            'oversized convergent exact-root preload',
        )
        generation = state['generation']
        shared = state['shared_preload']
        if (
            generation['storage'] != 'convergent_segments'
            or RETIRED_GENERATION_FIELDS.intersection(generation)
        ):
            raise AssertionError(
                'convergent status retained legacy generation authority: '
                f'{state}'
            )
        if generation['payload_expected_bytes'] <= shared['arena_size']:
            raise AssertionError(
                'exact-root fixture does not exceed the DSM arena: '
                f'{state}'
            )
        if shared['unified_warm_entries'] < 1:
            raise AssertionError(
                'convergent exact root has no unified warm entry: '
                f'{state}'
            )
        if (
            shared['resident_fold_current']
            or int(shared['resident_fold_bytes']) != 0
        ):
            raise AssertionError(
                'oversized root unexpectedly admitted a resident fold: '
                f'{state}'
            )
        if RETIRED_SHARED_PRELOAD_FIELDS.intersection(shared):
            raise AssertionError(
                'buffer-cache prewarm exposed retired admission history: '
                f'{state}'
            )

        psql(
            args,
            pgdata,
            port,
            '''
            CREATE INDEX docs_oversized_manual_idx
                ON docs_oversized USING evoke (tokens)
                WITH (auto_preload = 0);
            ''',
        )
        preload_result = psql(
            args,
            pgdata,
            port,
            "SELECT public.evoke_index_preload("
            "'docs_oversized_manual_idx'::regclass);",
        )
        pages_match = re.search(r'pages_warmed=([0-9]+)', preload_result)
        if (
            'storage=convergent_segments' not in preload_result or
            'tier=postgres_buffer_cache' not in preload_result or
            'prewarm_scope=bounded' not in preload_result or
            'page_budget=128' not in preload_result or
            pages_match is None
        ):
            raise AssertionError(
                'convergent oversized root used the wrong preload policy: '
                f'{preload_result}'
            )
        if int(pages_match.group(1)) != 128:
            raise AssertionError(
                'convergent oversized prewarm did not consume its page budget: '
                f'{preload_result}'
            )
        retry_result = psql(
            args,
            pgdata,
            port,
            "SELECT public.evoke_index_preload("
            "'docs_oversized_manual_idx'::regclass);",
        )
        retry_pages = re.search(r'pages_warmed=([0-9]+)', retry_result)
        if retry_pages is None or int(retry_pages.group(1)) != 128:
            raise AssertionError(
                'explicit preload did not re-admit an incomplete marker: '
                f'{retry_result}'
            )
        query_index(
            args,
            pgdata,
            port,
            'docs_oversized_bm25_idx',
            'docs_oversized',
            'oversized',
        )
        state = json_state(
            args,
            pgdata,
            port,
            'docs_oversized_bm25_idx',
        )
        if state['shared_preload']['validated_page_entries'] < 1:
            raise AssertionError(
                'foreground query did not publish a validated-page bitmap: '
                f'{state}'
            )
        if (
            'evoke auto_preload index could not be admitted to '
            'the shared runtime arena'
        ) in read_log(pgdata):
            raise AssertionError(
                'convergent exact-root prewarm emitted a DSM admission miss'
            )
        print('exact_root_state=' + json.dumps(state, sort_keys=True))
        print('exact_root_preload=' + preload_result)
        print('exact_root_preload_retry=' + retry_result)
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'kept admission-miss cluster at {workdir}')


def run_registry_isolation_case(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_shared_lifecycle_churn_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(
            args,
            pgdata,
            port,
            cache_mb=8,
            prewarm_max_mb=4,
        )
        require_extension(args, pgdata, port)
        psql(
            args,
            pgdata,
            port,
            '''
            CREATE TABLE hot_marked (
                id int primary key,
                tokens text[] not null
            );
            CREATE TABLE warm_marked (LIKE hot_marked INCLUDING ALL);
            CREATE TABLE pressure_a (LIKE hot_marked INCLUDING ALL);
            CREATE TABLE pressure_b (LIKE hot_marked INCLUDING ALL);
            CREATE TABLE pressure_c (LIKE hot_marked INCLUDING ALL);

            INSERT INTO hot_marked
            SELECT gs, ARRAY['hot', 'marked', 'shared', gs::text]
            FROM generate_series(1, 1000) gs;
            INSERT INTO warm_marked
            SELECT gs, ARRAY['warm', 'marked', 'shared', gs::text]
            FROM generate_series(1, 1000) gs;

            INSERT INTO pressure_a
            SELECT gs, ARRAY[
                'pressure', 'alpha', 'token_' || gs::text,
                'x_' || (gs % 1000)::text,
                'y_' || (gs % 2000)::text,
                'z_' || (gs % 3000)::text,
                'w_' || (gs % 4000)::text
            ]
            FROM generate_series(1, 30000) gs;
            INSERT INTO pressure_b
            SELECT gs, ARRAY[
                'pressure', 'beta', 'token_' || gs::text,
                'x_' || (gs % 1000)::text,
                'y_' || (gs % 2000)::text,
                'z_' || (gs % 3000)::text,
                'w_' || (gs % 4000)::text
            ]
            FROM generate_series(1, 30000) gs;
            INSERT INTO pressure_c
            SELECT gs, ARRAY[
                'pressure', 'gamma', 'token_' || gs::text,
                'x_' || (gs % 1000)::text,
                'y_' || (gs % 2000)::text,
                'z_' || (gs % 3000)::text,
                'w_' || (gs % 4000)::text
            ]
            FROM generate_series(1, 30000) gs;

            CREATE INDEX hot_marked_bm25_idx
                ON hot_marked USING evoke (tokens)
                WITH (auto_preload = 20);
            CREATE INDEX warm_marked_bm25_idx
                ON warm_marked USING evoke (tokens)
                WITH (auto_preload = 4);
            CREATE INDEX pressure_a_bm25_idx
                ON pressure_a USING evoke (tokens);
            CREATE INDEX pressure_b_bm25_idx
                ON pressure_b USING evoke (tokens);
            CREATE INDEX pressure_c_bm25_idx
                ON pressure_c USING evoke (tokens);
            ''',
        )

        hot = wait_for(
            args.timeout,
            lambda: json_state(args, pgdata, port, 'hot_marked_bm25_idx'),
            lambda value: value['shared_preload']['resident'],
            'high-priority marked preload residency',
        )
        warm = wait_for(
            args.timeout,
            lambda: json_state(args, pgdata, port, 'warm_marked_bm25_idx'),
            lambda value: value['shared_preload']['resident'],
            'lower-priority marked preload residency',
        )
        if hot['generation']['auto_preload_priority'] != 20:
            raise AssertionError(f'hot priority mismatch: {hot}')
        if warm['generation']['auto_preload_priority'] != 4:
            raise AssertionError(f'warm priority mismatch: {warm}')
        if (
            not hot['shared_preload']['resident_fold_current']
            or not warm['shared_preload']['resident_fold_current']
            or int(hot['shared_preload']['resident_fold_entries']) < 2
            or int(hot['shared_preload']['resident_fold_bytes']) <= 0
            or int(warm['shared_preload']['resident_fold_bytes']) <= 0
        ):
            raise AssertionError(
                'marked small indexes did not publish exact resident folds: '
                f'hot={hot}, warm={warm}'
            )

        for index_name, table_name, term in (
            ('pressure_a_bm25_idx', 'pressure_a', 'alpha'),
            ('pressure_b_bm25_idx', 'pressure_b', 'beta'),
            ('pressure_c_bm25_idx', 'pressure_c', 'gamma'),
        ):
            query_index(args, pgdata, port, index_name, table_name, term)
            pressure_state = json_state(args, pgdata, port, index_name)
            if pressure_state['shared_preload']['resident']:
                raise AssertionError(
                    'unmarked convergent query polluted auto-preload '
                    'residency: '
                    f'{pressure_state}'
                )
            if (
                pressure_state['generation']['storage'] !=
                'convergent_segments'
                or RETIRED_GENERATION_FIELDS.intersection(
                    pressure_state['generation']
                )
            ):
                raise AssertionError(
                    'unmarked convergent query exposed legacy status: '
                    f'{pressure_state}'
                )

        after = json_state(args, pgdata, port, 'hot_marked_bm25_idx')
        if not after['shared_preload']['resident']:
            raise AssertionError(
                'foreground pressure evicted the high-priority marked index: '
                f'{after}'
            )
        warm_after = json_state(
            args,
            pgdata,
            port,
            'warm_marked_bm25_idx',
        )
        if not warm_after['shared_preload']['resident']:
            raise AssertionError(
                'unmarked foreground queries evicted the marked warm root: '
                f'{warm_after}'
            )
        if after['shared_preload']['access_clock'] <= 0:
            raise AssertionError(
                'foreground churn did not advance access clock: '
                f'{after}'
            )
        if RETIRED_SHARED_PRELOAD_FIELDS.intersection(
            after['shared_preload']
        ):
            raise AssertionError(
                'convergent status exposed retired admission history: '
                f'{after}'
            )
        print('registry_state=' + json.dumps(after, sort_keys=True))
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'kept churn cluster at {workdir}')


def run_resident_root_replacement_case(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_shared_resident_root_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()

    try:
        init_cluster(
            args,
            pgdata,
            port,
            cache_mb=8,
            prewarm_max_mb=4,
        )
        require_extension(args, pgdata, port)
        set_maintenance_worker_limit(args, pgdata, port, 0)
        psql(
            args,
            pgdata,
            port,
            '''
            CREATE TABLE resident_docs (
                id int primary key,
                tokens text[] not null
            );
            INSERT INTO resident_docs
            SELECT gs, ARRAY['resident', 'stable', gs::text]
            FROM generate_series(1, 1000) gs;
            CREATE INDEX resident_docs_idx
                ON resident_docs USING evoke (tokens)
                WITH (auto_preload = 0);
            ''',
        )
        initial_preload = psql(
            args,
            pgdata,
            port,
            "SELECT public.evoke_index_preload("
            "'resident_docs_idx'::regclass);",
        )
        initial = json_state(args, pgdata, port, 'resident_docs_idx')
        if (
            'tier=shared_resident_fold' not in initial_preload
            or 'prewarm_scope=exact' not in initial_preload
            or re.search(r'resident_bytes=([1-9][0-9]*)', initial_preload)
                is None
            or not initial['shared_preload']['resident_fold_current']
            or int(initial['shared_preload']['resident_fold_bytes']) <= 0
        ):
            raise AssertionError(
                'explicit preload did not publish an exact resident fold: '
                f'preload={initial_preload}, state={initial}'
            )
        query_index(
            args,
            pgdata,
            port,
            'resident_docs_idx',
            'resident_docs',
            'resident',
        )

        psql(
            args,
            pgdata,
            port,
            '''
            INSERT INTO resident_docs
            SELECT gs, ARRAY['resident', 'replacement', gs::text]
            FROM generate_series(1001, 1005) gs;
            ''',
        )
        mutated = json_state(args, pgdata, port, 'resident_docs_idx')
        if (
            mutated['shared_preload']['resident_fold_current']
            or int(mutated['debt']['delta_records']) == 0
        ):
            raise AssertionError(
                'root mutation did not retire exact resident state: '
                f'{mutated}'
            )
        query_index(
            args,
            pgdata,
            port,
            'resident_docs_idx',
            'resident_docs',
            'resident',
        )

        converged = maintain_until_converged(
            args,
            pgdata,
            port,
            'resident_docs_idx',
        )
        if converged['shared_preload']['resident_fold_current']:
            raise AssertionError(
                'maintenance unexpectedly retained an old exact-root fold: '
                f'{converged}'
            )
        replacement_preload = psql(
            args,
            pgdata,
            port,
            "SELECT public.evoke_index_preload("
            "'resident_docs_idx'::regclass);",
        )
        replacement = json_state(args, pgdata, port, 'resident_docs_idx')
        if (
            'tier=shared_resident_fold' not in replacement_preload
            or not replacement['shared_preload']['resident_fold_current']
            or int(replacement['shared_preload']['resident_fold_bytes']) <= 0
            or int(replacement['debt']['delta_records']) != 0
        ):
            raise AssertionError(
                'converged replacement root did not republish resident fold: '
                f'preload={replacement_preload}, state={replacement}'
            )
        query_index(
            args,
            pgdata,
            port,
            'resident_docs_idx',
            'resident_docs',
            'replacement',
        )
        print('resident_root_initial=' + initial_preload)
        print('resident_root_replacement=' + replacement_preload)
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'kept resident-root cluster at {workdir}')


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

    run_oversized_exact_root_case(args)
    run_registry_isolation_case(args)
    run_resident_root_replacement_case(args)
    print('shared preload lifecycle closure passed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
