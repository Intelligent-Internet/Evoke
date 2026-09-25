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
from typing import Any

import psycopg


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify that convergent query state is shared and '
            'backend-private '
            'retention does not scale with the complete index.'
        ),
    )
    parser.add_argument(
        '--bindir',
        default='/opt/homebrew/opt/postgresql@18/bin',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--small-rows', type=int, default=5000)
    parser.add_argument('--large-rows', type=int, default=80000)
    parser.add_argument('--query-repeats', type=int, default=20)
    parser.add_argument(
        '--query-path',
        choices=(
            'direct',
            'direct_complex',
            'direct_prefix_miss',
            'ordered',
            'filtered_ordered',
            'predicate',
            'predicate_prefix',
        ),
        default='direct',
    )
    parser.add_argument(
        '--max-index-scaled-private-bytes',
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        rendered_command = ' '.join(command)
        raise RuntimeError(
            f'command failed ({result.returncode}): {rendered_command}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}',
        )
    return result


def psql(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
    sql: str,
) -> str:
    result = run(
        [
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
        ],
        input_text=sql,
    )
    return result.stdout.strip()


def init_cluster(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
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
        conf.write(f"unix_socket_directories = '{pgdata}'\n")
        conf.write(f'port = {port}\n')
        conf.write("shared_preload_libraries = 'evoke'\n")
        conf.write("evoke.shared_runtime_size = '1MB'\n")
        if args.extension_libdir is not None:
            libdir = str(args.extension_libdir.resolve()).replace("'", "''")
            conf.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if args.extension_control_dir is not None:
            control_dir = str(
                args.extension_control_dir.resolve(),
            ).replace("'", "''")
            conf.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
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
            'immediate',
            '-w',
            'stop',
        ],
        text=True,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )


def build_fixture(
    args: argparse.Namespace,
    pgdata: Path,
    port: int,
) -> None:
    psql(
        args,
        pgdata,
        port,
        f'''
        CREATE EXTENSION evoke;

        CREATE TABLE memory_small (
            id int PRIMARY KEY,
            tokens text[] NOT NULL
        );
        INSERT INTO memory_small
        SELECT gs, ARRAY[
            'shared', 'memory', 'small', 'token_' || gs::text,
            'x_' || (gs % 1000)::text,
            'y_' || (gs % 2000)::text
        ]
        FROM generate_series(1, {args.small_rows}) gs;
        CREATE INDEX memory_small_idx
            ON memory_small USING evoke (tokens)
            WITH (auto_preload = 10);

        CREATE TABLE memory_large (
            id int PRIMARY KEY,
            tokens text[] NOT NULL
        );
        INSERT INTO memory_large
        SELECT gs, ARRAY[
            'shared', 'memory', 'large', 'token_' || gs::text,
            'x_' || (gs % 1000)::text,
            'y_' || (gs % 2000)::text,
            'z_' || (gs % 3000)::text,
            'w_' || (gs % 4000)::text
        ]
        FROM generate_series(1, {args.large_rows}) gs;
        CREATE INDEX memory_large_idx
            ON memory_large USING evoke (tokens)
            WITH (auto_preload = 20);

        SELECT evoke_index_preload('memory_small_idx'::regclass);
        SELECT evoke_index_preload('memory_large_idx'::regclass);
        ''',
    )


def process_rss_bytes(pid: int) -> int:
    result = run(['ps', '-o', 'rss=', '-p', str(pid)])
    return int(result.stdout.strip()) * 1024


def parse_vmmap_bytes(value: str) -> int:
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGTP]?)', value)
    if match is None:
        raise ValueError(f'unexpected vmmap byte value: {value!r}')
    multipliers = {
        '': 1,
        'K': 1024,
        'M': 1024**2,
        'G': 1024**3,
        'T': 1024**4,
        'P': 1024**5,
    }
    return int(float(match.group(1)) * multipliers[match.group(2)])


def parse_vmmap_private_memory(text: str) -> dict[str, Any]:
    footprint_match = re.search(
        r'^Physical footprint:\s+([^\s]+)$',
        text,
        re.MULTILINE,
    )
    if footprint_match is None:
        raise RuntimeError('vmmap did not report a physical footprint')

    malloc_section = text.rfind('\nMALLOC ZONE')
    if malloc_section < 0:
        raise RuntimeError('vmmap did not report malloc-zone ownership')
    malloc_allocated = None
    for line in text[malloc_section:].splitlines():
        fields = line.split()
        if fields and fields[0] == 'TOTAL' and len(fields) >= 7:
            malloc_allocated = parse_vmmap_bytes(fields[6])
            break
    if malloc_allocated is None:
        raise RuntimeError('vmmap did not report malloc allocated bytes')

    private_writable = 0
    private_live_writable = 0
    allocator_empty = 0
    private_writable_by_region: dict[str, int] = {}
    region_pattern = re.compile(
        r'^(.*?)\s+[0-9a-f]+-[0-9a-f]+\s+'
        r'\[\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+'
        r'([^\s]+)\]\s+([rwx-]+)/[rwx-]+\s+SM=([A-Z/]+)',
    )
    for line in text.splitlines():
        match = region_pattern.match(line)
        if match is None:
            continue
        region_name = match.group(1).strip()
        current_permissions = match.group(6)
        sharing_mode = match.group(7)
        if not current_permissions.startswith('rw'):
            continue
        if sharing_mode not in {'PRV', 'COW'}:
            continue
        private_bytes = parse_vmmap_bytes(match.group(4))
        private_bytes += parse_vmmap_bytes(match.group(5))
        private_writable += private_bytes
        if '(empty)' in region_name:
            allocator_empty += private_bytes
        else:
            private_live_writable += private_bytes
        private_writable_by_region[region_name] = (
            private_writable_by_region.get(region_name, 0) +
            private_bytes
        )

    return {
        'observer': 'darwin_vmmap',
        'capabilities': {
            'allocator_empty': True,
            'malloc_allocated': True,
            'physical_footprint': True,
            'private_writable': True,
        },
        'physical_footprint_bytes': parse_vmmap_bytes(
            footprint_match.group(1),
        ),
        'malloc_allocated_bytes': malloc_allocated,
        'private_writable_bytes': private_writable,
        'private_live_writable_bytes': private_live_writable,
        'allocator_empty_bytes': allocator_empty,
        'private_writable_by_region_bytes': private_writable_by_region,
    }


def parse_proc_kb_fields(text: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z_]+):\s+([0-9]+)\s+kB', line)
        if match is not None:
            fields[match.group(1)] = int(match.group(2)) * 1024
    return fields


def parse_proc_private_memory(
    smaps_text: str,
    rollup_text: str | None,
) -> dict[str, Any]:
    header_pattern = re.compile(
        r'^([0-9a-f]+)-([0-9a-f]+)\s+'
        r'([rwxps-]{4})\s+[0-9a-f]+\s+\S+\s+\d+\s*(.*)$',
        re.IGNORECASE,
    )
    private_writable = 0
    private_writable_by_region: dict[str, int] = {}
    pss_from_mappings = 0
    current_permissions: str | None = None
    current_region = '[anonymous]'
    current_fields: dict[str, int] = {}

    def finish_mapping() -> None:
        nonlocal private_writable, pss_from_mappings
        if current_permissions is None:
            return
        pss_from_mappings += current_fields.get('Pss', 0)
        if (
            'w' not in current_permissions or
            not current_permissions.endswith('p')
        ):
            return
        private_bytes = sum(
            current_fields.get(name, 0)
            for name in (
                'Private_Clean',
                'Private_Dirty',
                'Private_Hugetlb',
            )
        )
        private_writable += private_bytes
        if private_bytes > 0:
            private_writable_by_region[current_region] = (
                private_writable_by_region.get(current_region, 0) +
                private_bytes
            )

    for line in smaps_text.splitlines():
        header = header_pattern.fullmatch(line)
        if header is not None:
            finish_mapping()
            current_permissions = header.group(3)
            current_region = header.group(4).strip() or '[anonymous]'
            current_fields = {}
            continue
        field = re.fullmatch(
            r'([A-Za-z_]+):\s+([0-9]+)\s+kB',
            line,
        )
        if field is not None and current_permissions is not None:
            current_fields[field.group(1)] = int(field.group(2)) * 1024
    finish_mapping()

    rollup_fields = (
        parse_proc_kb_fields(rollup_text)
        if rollup_text is not None
        else {}
    )
    physical_footprint = rollup_fields.get('Pss', pss_from_mappings)
    return {
        'observer': 'linux_proc_smaps',
        'capabilities': {
            'allocator_empty': False,
            'malloc_allocated': False,
            'physical_footprint': True,
            'private_writable': True,
        },
        # PSS is the process's proportional resident ownership on Linux.
        'physical_footprint_bytes': physical_footprint,
        'malloc_allocated_bytes': None,
        'private_writable_bytes': private_writable,
        'private_live_writable_bytes': private_writable,
        'allocator_empty_bytes': None,
        'private_writable_by_region_bytes': private_writable_by_region,
    }


def process_private_memory(pid: int) -> dict[str, Any]:
    if sys.platform == 'darwin':
        result = run(['vmmap', '-wide', str(pid)])
        return parse_vmmap_private_memory(result.stdout)
    if sys.platform.startswith('linux'):
        proc_root = Path('/proc') / str(pid)
        smaps_text = (proc_root / 'smaps').read_text(encoding='utf-8')
        rollup_path = proc_root / 'smaps_rollup'
        rollup_text = (
            rollup_path.read_text(encoding='utf-8')
            if rollup_path.exists()
            else None
        )
        return parse_proc_private_memory(smaps_text, rollup_text)
    raise RuntimeError(
        f'unsupported private-memory observer platform: {sys.platform}',
    )


def nonnegative_growth(
    after: int | None,
    before: int | None,
) -> int | None:
    if after is None or before is None:
        return None
    return max(0, after - before)


def scaled_growth(
    large: int | None,
    small: int | None,
) -> int | None:
    if large is None or small is None:
        return None
    return max(0, large - small)


def backend_context_bytes(
    connection: psycopg.Connection[Any],
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COALESCE(sum(total_bytes), 0)::int8 "
            "FROM pg_backend_memory_contexts "
            "WHERE name LIKE 'evoke%cache entry'",
        )
        return int(cursor.fetchone()[0])


def execute_observed_query(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    table_name: str,
    term: str,
    query_path: str,
) -> int:
    if query_path == 'direct_prefix_miss':
        cursor.execute(
            f'''
            SELECT count(*)
            FROM evoke_query(
                %s::regclass,
                %s,
                5,
                NULL,
                true,
                NULL,
                false,
                false
            ) h
            JOIN {table_name} d ON d.ctid = h.ctid
            ''',
            (index_name, term),
        )
    elif query_path in {
        'direct',
        'direct_complex',
    }:
        cursor.execute(
            f'''
            SELECT count(*)
            FROM evoke_query(
                %s::regclass,
                %s,
                5,
                NULL,
                true,
                NULL,
                false,
                false
            ) h
            JOIN {table_name} d ON d.ctid = h.ctid
            WHERE h.score > 0
            ''',
            (index_name, term),
        )
    elif query_path == 'ordered':
        cursor.execute(
            f'''
            SELECT count(*)
            FROM (
                SELECT id
                FROM {table_name}
                ORDER BY tokens <=> ARRAY[%s]::text[] ASC
                LIMIT 5
            ) hits
            ''',
            (term,),
        )
    elif query_path == 'filtered_ordered':
        cursor.execute(
            f'''
            SELECT count(*)
            FROM (
                SELECT id
                FROM {table_name}
                WHERE tokens @@ %s
                ORDER BY tokens <=> ARRAY[%s]::text[] ASC
                LIMIT 5
            ) hits
            ''',
            (term, term),
        )
    elif query_path in {'predicate', 'predicate_prefix'}:
        predicate = term
        if query_path == 'predicate_prefix':
            predicate += '*'
        cursor.execute(
            f'''
            SELECT count(*)
            FROM (
                SELECT id
                FROM {table_name}
                WHERE tokens @@ %s
                LIMIT 5
            ) hits
            ''',
            (predicate,),
        )
    else:
        raise ValueError(f'unsupported query path: {query_path}')
    return int(cursor.fetchone()[0])


def observe_index(
    dsn: str,
    index_name: str,
    table_name: str,
    term: str,
    query_repeats: int,
    query_path: str,
) -> dict[str, Any]:
    if query_repeats < 1:
        raise ValueError('query_repeats must be positive')
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute('SET enable_seqscan = false')
            cursor.execute('SET enable_bitmapscan = false')
            cursor.execute('SELECT pg_backend_pid()')
            pid = int(cursor.fetchone()[0])
        rss_before = process_rss_bytes(pid)
        private_before = process_private_memory(pid)
        context_before = backend_context_bytes(connection)
        with connection.cursor() as cursor:
            first_started = time.perf_counter_ns()
            hits = execute_observed_query(
                cursor,
                index_name,
                table_name,
                term,
                query_path,
            )
            first_elapsed_ns = time.perf_counter_ns() - first_started
        if hits != 5:
            raise AssertionError(
                f'unexpected {index_name} hit count: {hits}',
            )
        context_first = backend_context_bytes(connection)
        private_first = process_private_memory(pid)
        rss_first = process_rss_bytes(pid)

        repeat_elapsed_ns = 0
        with connection.cursor() as cursor:
            repeat_started = time.perf_counter_ns()
            for _ in range(query_repeats - 1):
                hits = execute_observed_query(
                    cursor,
                    index_name,
                    table_name,
                    term,
                    query_path,
                )
                if hits != 5:
                    raise AssertionError(
                        f'unexpected {index_name} hit count: {hits}',
                    )
            repeat_elapsed_ns = time.perf_counter_ns() - repeat_started
            cursor.execute(
                'SELECT evoke_index_runtime_state_json(%s::regclass)',
                (index_name,),
            )
            state = cursor.fetchone()[0]
        context_after = backend_context_bytes(connection)
        private_after = process_private_memory(pid)
        rss_after = process_rss_bytes(pid)
        shared = state['shared_preload']
        if not shared['resident'] or shared['unified_warm_entries'] < 1:
            raise AssertionError(
                f'{index_name} has no exact shared root marker: {state}',
            )
        observers = {
            private_before['observer'],
            private_first['observer'],
            private_after['observer'],
        }
        capabilities = {
            tuple(sorted(private_before['capabilities'].items())),
            tuple(sorted(private_first['capabilities'].items())),
            tuple(sorted(private_after['capabilities'].items())),
        }
        if len(observers) != 1 or len(capabilities) != 1:
            raise AssertionError(
                'private-memory observer changed during measurement',
            )
        return {
            'index': index_name,
            'query_path': query_path,
            'private_memory_observer': observers.pop(),
            'private_memory_capabilities': dict(capabilities.pop()),
            'rows': state['generation']['docs'],
            'relation_bytes': state['generation']['payload_capacity_bytes'],
            'shared_expected_bytes': (
                state['generation']['payload_expected_bytes']
            ),
            'shared_capacity_bytes': (
                state['generation']['payload_capacity_bytes']
            ),
            'shared_root_marker': shared['resident'],
            'rss_before_bytes': rss_before,
            'rss_first_bytes': rss_first,
            'rss_after_bytes': rss_after,
            'rss_growth_bytes': max(0, rss_after - rss_before),
            'physical_footprint_before_bytes': (
                private_before['physical_footprint_bytes']
            ),
            'physical_footprint_first_bytes': (
                private_first['physical_footprint_bytes']
            ),
            'physical_footprint_after_bytes': (
                private_after['physical_footprint_bytes']
            ),
            'physical_footprint_growth_bytes': nonnegative_growth(
                private_after['physical_footprint_bytes'],
                private_before['physical_footprint_bytes'],
            ),
            'malloc_allocated_before_bytes': (
                private_before['malloc_allocated_bytes']
            ),
            'malloc_allocated_first_bytes': (
                private_first['malloc_allocated_bytes']
            ),
            'malloc_allocated_after_bytes': (
                private_after['malloc_allocated_bytes']
            ),
            'malloc_allocated_growth_bytes': nonnegative_growth(
                private_after['malloc_allocated_bytes'],
                private_before['malloc_allocated_bytes'],
            ),
            'private_writable_before_bytes': (
                private_before['private_writable_bytes']
            ),
            'private_writable_first_bytes': (
                private_first['private_writable_bytes']
            ),
            'private_writable_after_bytes': (
                private_after['private_writable_bytes']
            ),
            'private_writable_growth_bytes': nonnegative_growth(
                private_after['private_writable_bytes'],
                private_before['private_writable_bytes'],
            ),
            'private_live_writable_before_bytes': (
                private_before['private_live_writable_bytes']
            ),
            'private_live_writable_first_bytes': (
                private_first['private_live_writable_bytes']
            ),
            'private_live_writable_after_bytes': (
                private_after['private_live_writable_bytes']
            ),
            'private_live_writable_growth_bytes': nonnegative_growth(
                private_after['private_live_writable_bytes'],
                private_before['private_live_writable_bytes'],
            ),
            'allocator_empty_before_bytes': (
                private_before['allocator_empty_bytes']
            ),
            'allocator_empty_first_bytes': (
                private_first['allocator_empty_bytes']
            ),
            'allocator_empty_after_bytes': (
                private_after['allocator_empty_bytes']
            ),
            'allocator_empty_growth_bytes': nonnegative_growth(
                private_after['allocator_empty_bytes'],
                private_before['allocator_empty_bytes'],
            ),
            'private_writable_before_by_region_bytes': (
                private_before['private_writable_by_region_bytes']
            ),
            'private_writable_after_by_region_bytes': (
                private_after['private_writable_by_region_bytes']
            ),
            'evoke_context_before_bytes': context_before,
            'evoke_context_first_bytes': context_first,
            'evoke_context_after_bytes': context_after,
            'evoke_context_growth_bytes': max(
                0,
                context_after - context_before,
            ),
            'repeat_private_live_growth_bytes': nonnegative_growth(
                private_after['private_live_writable_bytes'],
                private_first['private_live_writable_bytes'],
            ),
            'repeat_malloc_growth_bytes': nonnegative_growth(
                private_after['malloc_allocated_bytes'],
                private_first['malloc_allocated_bytes'],
            ),
            'repeat_context_growth_bytes': max(
                0,
                context_after - context_first,
            ),
            'query_repeats': query_repeats,
            'first_query_ms': first_elapsed_ns / 1_000_000,
            'repeat_query_average_ms': (
                repeat_elapsed_ns / max(1, query_repeats - 1) / 1_000_000
            ),
        }


def main() -> int:
    args = parse_args()
    workdir = Path(tempfile.mkdtemp(
        prefix='evoke_v3_backend_memory_',
        dir='/tmp',
    ))
    pgdata = workdir / 'pgdata'
    port = free_port()
    output: dict[str, Any] = {
        'route': 'convergent backend memory ownership',
        'query_path': args.query_path,
        'workdir': str(workdir),
    }
    failure: str | None = None

    try:
        init_cluster(args, pgdata, port)
        build_fixture(args, pgdata, port)
        dsn = (
            f'host={pgdata} port={port} dbname=postgres '
            f'user={os.environ.get("USER", "postgres")}'
        )
        if args.query_path == 'predicate_prefix':
            observed_term = 'x_1'
        elif args.query_path == 'direct_complex':
            observed_term = (
                '("shared memory" OR x_1*) AND NOT absent'
            )
        elif args.query_path == 'direct_prefix_miss':
            observed_term = 'definitely_missing_prefix*'
        else:
            observed_term = None
        small = observe_index(
            dsn,
            'memory_small_idx',
            'memory_small',
            observed_term or 'small',
            args.query_repeats,
            args.query_path,
        )
        large = observe_index(
            dsn,
            'memory_large_idx',
            'memory_large',
            observed_term or 'large',
            args.query_repeats,
            args.query_path,
        )
        scaled_rss = max(
            0,
            large['rss_growth_bytes'] - small['rss_growth_bytes'],
        )
        scaled_context = max(
            0,
            large['evoke_context_growth_bytes'] -
            small['evoke_context_growth_bytes'],
        )
        scaled_footprint = scaled_growth(
            large['physical_footprint_growth_bytes'],
            small['physical_footprint_growth_bytes'],
        )
        scaled_malloc = scaled_growth(
            large['malloc_allocated_growth_bytes'],
            small['malloc_allocated_growth_bytes'],
        )
        scaled_private_writable = scaled_growth(
            large['private_writable_growth_bytes'],
            small['private_writable_growth_bytes'],
        )
        scaled_private_live_writable = scaled_growth(
            large['private_live_writable_growth_bytes'],
            small['private_live_writable_growth_bytes'],
        )
        scaled_allocator_empty = scaled_growth(
            large['allocator_empty_growth_bytes'],
            small['allocator_empty_growth_bytes'],
        )
        private_gate_available = (
            scaled_private_live_writable is not None and
            small['repeat_private_live_growth_bytes'] is not None and
            large['repeat_private_live_growth_bytes'] is not None
        )
        malloc_gate_available = (
            scaled_malloc is not None and
            small['repeat_malloc_growth_bytes'] is not None and
            large['repeat_malloc_growth_bytes'] is not None
        )
        passed = (
            scaled_context <= args.max_index_scaled_private_bytes and
            private_gate_available and
            scaled_private_live_writable <= (
                args.max_index_scaled_private_bytes
            ) and
            small['repeat_private_live_growth_bytes'] <= (
                args.max_index_scaled_private_bytes
            ) and
            large['repeat_private_live_growth_bytes'] <= (
                args.max_index_scaled_private_bytes
            ) and (
                not malloc_gate_available or (
                    scaled_malloc <= args.max_index_scaled_private_bytes and
                    small['repeat_malloc_growth_bytes'] <= (
                        args.max_index_scaled_private_bytes
                    ) and
                    large['repeat_malloc_growth_bytes'] <= (
                        args.max_index_scaled_private_bytes
                    )
                )
            )
        )
        output.update({
            'small': small,
            'large': large,
            'index_scaled_rss_bytes': scaled_rss,
            'index_scaled_context_bytes': scaled_context,
            'index_scaled_physical_footprint_bytes': scaled_footprint,
            'index_scaled_malloc_allocated_bytes': scaled_malloc,
            'index_scaled_private_writable_bytes': (
                scaled_private_writable
            ),
            'index_scaled_private_live_writable_bytes': (
                scaled_private_live_writable
            ),
            'index_scaled_allocator_empty_bytes': scaled_allocator_empty,
            'rss_is_observational_only': True,
            'physical_footprint_is_observational_only': True,
            'allocator_empty_is_observational_only': True,
            'private_writable_gate_applied': private_gate_available,
            'malloc_allocated_gate_applied': malloc_gate_available,
            'max_index_scaled_private_bytes': (
                args.max_index_scaled_private_bytes
            ),
            'passed': passed,
            'observe_only': args.observe_only,
        })
        if not passed and not args.observe_only:
            failure = (
                'convergent backend-private memory scales with the index: '
                f'{json.dumps(output, sort_keys=True)}'
            )
    finally:
        stop_cluster(args, pgdata)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    if failure is not None:
        raise AssertionError(failure)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
