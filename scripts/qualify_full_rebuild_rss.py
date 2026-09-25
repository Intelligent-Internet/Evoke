#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg


PROC_STATUS_KEYS = {
    'VmRSS',
    'VmHWM',
    'RssAnon',
    'RssFile',
    'RssShmem',
    'VmSwap',
}
SMAPS_KEYS = {
    'Rss',
    'Pss',
    'Shared_Clean',
    'Shared_Dirty',
    'Private_Clean',
    'Private_Dirty',
    'Swap',
}
DEFAULT_MAX_RSS_MIB = 32 * 1024
DEFAULT_MAX_SWAP_MIB = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run one PostgreSQL rebuild statement and record backend and '
            'PostgreSQL process-family memory peaks by Evoke build phase.'
        ),
    )
    parser.add_argument('--psql', type=Path, default=Path('psql'))
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user', default=os.environ.get('USER', 'postgres'))
    parser.add_argument('--database', default='postgres')
    parser.add_argument('--sql-file', type=Path, required=True)
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--index')
    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--record-interval', type=float, default=30.0)
    parser.add_argument(
        '--max-rss-mib',
        type=float,
        default=DEFAULT_MAX_RSS_MIB,
    )
    parser.add_argument(
        '--max-swap-mib',
        type=float,
        default=DEFAULT_MAX_SWAP_MIB,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def process_mapped_files(pid: int) -> list[Path]:
    maps_path = Path('/proc') / str(pid) / 'maps'
    try:
        lines = maps_path.read_text(encoding='utf-8').splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    mapped: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = fields[5]
        if path.endswith(' (deleted)'):
            continue
        mapped.add(Path(path))
    return sorted(mapped)


def binary_binding(
    pid: int,
    expected_binary: Path,
) -> dict[str, Any]:
    expected_path = expected_binary.resolve(strict=True)
    expected_sha256 = sha256(expected_path)
    candidates = [
        path
        for path in process_mapped_files(pid)
        if path.name == expected_path.name
    ]
    mapped: list[dict[str, str]] = []
    for path in candidates:
        try:
            mapped.append({
                'path': str(path),
                'sha256': sha256(path),
            })
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    matching = [
        item['path']
        for item in mapped
        if item['sha256'] == expected_sha256
    ]
    return {
        'expected_path': str(expected_path),
        'expected_sha256': expected_sha256,
        'mapped_files': mapped,
        'matching_paths': matching,
        'qualified': bool(matching),
    }


def binary_binding_errors(binding: dict[str, Any] | None) -> list[str]:
    if not isinstance(binding, dict):
        return ['loaded extension binary binding is missing']
    if binding.get('qualified') is not True:
        return ['loaded extension binary differs from qualified artifact']
    return []


def read_kib_values(path: Path, keys: set[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        name, separator, raw_value = line.partition(':')
        if not separator or name not in keys:
            continue
        fields = raw_value.split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0]) * 1024
    return values


def process_memory(pid: int) -> dict[str, int]:
    root = Path('/proc') / str(pid)
    values = {
        f'status_{key}_bytes': value
        for key, value in read_kib_values(
            root / 'status',
            PROC_STATUS_KEYS,
        ).items()
    }
    values.update({
        f'smaps_{key}_bytes': value
        for key, value in read_kib_values(
            root / 'smaps_rollup',
            SMAPS_KEYS,
        ).items()
    })
    return values


def process_parent_pid(pid: int) -> int | None:
    try:
        lines = (Path('/proc') / str(pid) / 'status').read_text(
            encoding='utf-8'
        ).splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    for line in lines:
        name, separator, raw_value = line.partition(':')
        if separator and name == 'PPid':
            fields = raw_value.split()
            if fields and fields[0].isdigit():
                return int(fields[0])
    return None


def process_label(pid: int) -> str:
    root = Path('/proc') / str(pid)
    try:
        raw = (root / 'cmdline').read_bytes().split(b'\0', 1)[0]
        if raw:
            return raw.decode('utf-8', errors='replace')
        return (root / 'comm').read_text(encoding='utf-8').strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ''


def process_family_pids(anchor_pid: int) -> list[int]:
    postmaster_pid = process_parent_pid(anchor_pid)
    if postmaster_pid is None or postmaster_pid <= 1:
        return [anchor_pid]
    children_path = (
        Path('/proc')
        / str(postmaster_pid)
        / 'task'
        / str(postmaster_pid)
        / 'children'
    )
    try:
        child_fields = children_path.read_text(encoding='utf-8').split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        child_fields = []
    child_pids = [
        int(value)
        for value in child_fields
        if value.isdigit()
    ]
    return sorted({postmaster_pid, anchor_pid, *child_pids})


def aggregate_process_family_memory(
    processes: list[tuple[str, dict[str, int]]],
) -> dict[str, int]:
    aggregate = {
        'family_MaxProcessHwm_bytes': 0,
        'family_Private_bytes': 0,
        'family_Pss_bytes': 0,
        'family_Rss_bytes': 0,
        'family_Swap_bytes': 0,
        'family_process_count': len(processes),
        'runtime_worker_Hwm_bytes': 0,
        'runtime_worker_Pss_bytes': 0,
        'runtime_worker_Rss_bytes': 0,
        'runtime_worker_Swap_bytes': 0,
    }
    for label, memory in processes:
        hwm = memory.get('status_VmHWM_bytes', 0)
        rss = memory.get(
            'status_VmRSS_bytes',
            memory.get('smaps_Rss_bytes', 0),
        )
        pss = memory.get('smaps_Pss_bytes', 0)
        private = memory.get('smaps_Private_Clean_bytes', 0) + memory.get(
            'smaps_Private_Dirty_bytes',
            0,
        )
        swap = max(
            memory.get('status_VmSwap_bytes', 0),
            memory.get('smaps_Swap_bytes', 0),
        )
        aggregate['family_MaxProcessHwm_bytes'] = max(
            aggregate['family_MaxProcessHwm_bytes'],
            hwm,
        )
        aggregate['family_Rss_bytes'] += rss
        aggregate['family_Pss_bytes'] += pss
        aggregate['family_Private_bytes'] += private
        aggregate['family_Swap_bytes'] += swap
        if 'evoke runtime service' in label:
            aggregate['runtime_worker_Hwm_bytes'] = max(
                aggregate['runtime_worker_Hwm_bytes'],
                hwm,
            )
            aggregate['runtime_worker_Rss_bytes'] += rss
            aggregate['runtime_worker_Pss_bytes'] += pss
            aggregate['runtime_worker_Swap_bytes'] += swap
    return aggregate


def process_family_memory(anchor_pid: int) -> dict[str, int]:
    processes = [
        (process_label(pid), process_memory(pid))
        for pid in process_family_pids(anchor_pid)
    ]
    return aggregate_process_family_memory(processes)


def phase_name(activity: dict[str, Any]) -> str:
    query = str(activity.get('query') or '')
    if query.startswith('evoke build:'):
        return query
    progress = activity.get('progress_phase')
    if progress:
        return f'postgres: {progress}'
    return 'sql'


def update_peaks(
    peaks: dict[str, dict[str, int]],
    phase: str,
    memory: dict[str, int],
) -> None:
    phase_peak = peaks.setdefault(phase, {})
    for key, value in memory.items():
        phase_peak[key] = max(phase_peak.get(key, 0), value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def observer_dsn(args: argparse.Namespace) -> str:
    return ' '.join([
        f'host={args.host}',
        f'port={args.port}',
        f'user={args.user}',
        f'dbname={args.database}',
        "application_name=evoke_full_rebuild_rss_observer",
    ])


def connection_backend_pid(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_catalog.pg_backend_pid()')
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('observer backend PID is unavailable')
    return int(row[0])


def fetch_activity(
    connection: psycopg.Connection[Any],
    application_name: str,
) -> dict[str, Any] | None:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                activity.pid,
                EXTRACT(EPOCH FROM activity.backend_start)::float8,
                activity.state,
                activity.query,
                activity.wait_event_type,
                activity.wait_event,
                progress.phase,
                progress.blocks_total,
                progress.blocks_done,
                progress.tuples_total,
                progress.tuples_done
            FROM pg_catalog.pg_stat_activity AS activity
            LEFT JOIN pg_catalog.pg_stat_progress_create_index AS progress
              ON progress.pid = activity.pid
            WHERE activity.application_name = %s
              AND activity.backend_type = 'client backend'
            ORDER BY activity.backend_start DESC
            LIMIT 1
            ''',
            (application_name,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    keys = (
        'pid',
        'backend_started_unix',
        'state',
        'query',
        'wait_event_type',
        'wait_event',
        'progress_phase',
        'blocks_total',
        'blocks_done',
        'tuples_total',
        'tuples_done',
    )
    return dict(zip(keys, row, strict=True))


def index_catalog(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                index_relation.oid,
                index_state.indisvalid,
                index_state.indisready,
                pg_catalog.pg_relation_size(index_relation.oid),
                pg_catalog.pg_get_indexdef(index_relation.oid)
            FROM pg_catalog.pg_class AS index_relation
            JOIN pg_catalog.pg_index AS index_state
              ON index_state.indexrelid = index_relation.oid
            WHERE index_relation.oid = pg_catalog.to_regclass(%s)
            ''',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None:
        return {'present': False}
    return {
        'present': True,
        'oid': row[0],
        'valid': row[1],
        'ready': row[2],
        'bytes': row[3],
        'definition': row[4],
    }


def qualification_errors(
    process_returncode: int,
    index_name: str | None,
    catalog: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if process_returncode != 0:
        errors.append(f'psql exited with status {process_returncode}')
    if index_name is None:
        return errors
    if catalog is None or not catalog.get('present'):
        errors.append(f'qualified index is absent: {index_name}')
        return errors
    if not catalog.get('valid'):
        errors.append(f'qualified index is not valid: {index_name}')
    if not catalog.get('ready'):
        errors.append(f'qualified index is not ready: {index_name}')
    return errors


def peak_memory_metric(
    peaks: dict[str, dict[str, int]],
    keys: tuple[str, ...],
) -> int | None:
    values = [
        memory[key]
        for memory in peaks.values()
        for key in keys
        if key in memory
    ]
    return max(values) if values else None


def memory_qualification(
    peaks: dict[str, dict[str, int]],
    max_rss_bytes: int,
    max_swap_bytes: int,
    family_peaks: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, int | None], list[str]]:
    peak_backend_rss = peak_memory_metric(
        peaks,
        ('status_VmHWM_bytes', 'smaps_Rss_bytes'),
    )
    peak_backend_swap = peak_memory_metric(
        peaks,
        ('status_VmSwap_bytes', 'smaps_Swap_bytes'),
    )
    peak_family_pss = None
    peak_family_private = None
    peak_family_hwm = None
    peak_runtime_worker_hwm = None
    peak_family_swap = None
    if family_peaks is not None:
        peak_family_pss = peak_memory_metric(
            family_peaks,
            ('family_Pss_bytes',),
        )
        peak_family_private = peak_memory_metric(
            family_peaks,
            ('family_Private_bytes',),
        )
        peak_family_hwm = peak_memory_metric(
            family_peaks,
            ('family_MaxProcessHwm_bytes',),
        )
        peak_runtime_worker_hwm = peak_memory_metric(
            family_peaks,
            ('runtime_worker_Hwm_bytes',),
        )
        peak_family_swap = peak_memory_metric(
            family_peaks,
            ('family_Swap_bytes',),
        )
    rss_values = [
        value
        for value in (
            peak_backend_rss,
            peak_family_pss,
            peak_family_private,
            peak_family_hwm,
            peak_runtime_worker_hwm,
        )
        if value is not None
    ]
    swap_values = [
        value
        for value in (peak_backend_swap, peak_family_swap)
        if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    peak_swap = max(swap_values) if swap_values else None
    observed = {
        'peak_backend_rss_bytes': peak_backend_rss,
        'peak_backend_swap_bytes': peak_backend_swap,
        'peak_family_hwm_bytes': peak_family_hwm,
        'peak_family_private_bytes': peak_family_private,
        'peak_family_pss_bytes': peak_family_pss,
        'peak_family_swap_bytes': peak_family_swap,
        'peak_rss_bytes': peak_rss,
        'peak_runtime_worker_hwm_bytes': peak_runtime_worker_hwm,
        'peak_swap_bytes': peak_swap,
    }
    errors: list[str] = []
    if family_peaks is not None and peak_family_pss is None:
        errors.append('full rebuild process-family PSS telemetry is missing')
    if peak_rss is None:
        errors.append('full rebuild RSS telemetry is missing')
    elif peak_rss > max_rss_bytes:
        errors.append(
            'full rebuild peak RSS exceeds limit: '
            f'{peak_rss} > {max_rss_bytes} bytes'
        )
    if peak_swap is None:
        errors.append('full rebuild swap telemetry is missing')
    elif peak_swap > max_swap_bytes:
        errors.append(
            'full rebuild peak swap exceeds limit: '
            f'{peak_swap} > {max_swap_bytes} bytes'
        )
    return observed, errors


def psql_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.psql),
        '-X',
        '-v',
        'ON_ERROR_STOP=1',
        '-h',
        args.host,
        '-p',
        str(args.port),
        '-U',
        args.user,
        '-d',
        args.database,
        '-f',
        str(args.sql_file),
    ]


def main() -> int:
    args = parse_args()
    if args.sample_interval <= 0 or args.record_interval <= 0:
        raise ValueError('sampling intervals must be positive')
    if args.max_rss_mib < 0 or args.max_swap_mib < 0:
        raise ValueError('memory limits must be nonnegative')
    if not args.sql_file.is_file():
        raise FileNotFoundError(args.sql_file)

    max_rss_bytes = int(args.max_rss_mib * 1024 * 1024)
    max_swap_bytes = int(args.max_swap_mib * 1024 * 1024)

    stdout_path = args.output.with_suffix(f'{args.output.suffix}.stdout')
    stderr_path = args.output.with_suffix(f'{args.output.suffix}.stderr')
    result: dict[str, Any] = {
        'application_name': args.application_name,
        'binary': None if args.binary is None else {
            'path': str(args.binary),
            'sha256': sha256(args.binary),
        },
        'command': psql_command(args),
        'database': args.database,
        'host': args.host,
        'index': args.index,
        'memory_limits': {
            'max_rss_bytes': max_rss_bytes,
            'max_swap_bytes': max_swap_bytes,
        },
        'family_phase_peaks': {},
        'phase_peaks': {},
        'port': args.port,
        'sample_count': 0,
        'samples': [],
        'sql_file': str(args.sql_file),
        'started_unix': time.time(),
        'status': 'running',
        'stderr_log': str(stderr_path),
        'stdout_log': str(stdout_path),
        'user': args.user,
    }
    write_json(args.output, result)

    environment = dict(os.environ)
    environment['PGAPPNAME'] = args.application_name
    with stdout_path.open('w', encoding='utf-8') as stdout_handle, \
            stderr_path.open('w', encoding='utf-8') as stderr_handle, \
            psycopg.connect(observer_dsn(args), autocommit=True) as observer:
        if args.binary is not None:
            binding = binary_binding(
                connection_backend_pid(observer),
                args.binary,
            )
            result['binary']['binding'] = binding
            binding_errors = binary_binding_errors(binding)
            if binding_errors:
                result['completed_unix'] = time.time()
                result['elapsed_seconds'] = round(
                    result['completed_unix'] - result['started_unix'],
                    3,
                )
                result['psql_returncode'] = None
                result['memory_observed'] = None
                result['qualification_errors'] = binding_errors
                result['qualified'] = False
                result['returncode'] = 1
                result['status'] = 'failed'
                write_json(args.output, result)
                return 1
        process = subprocess.Popen(
            psql_command(args),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        last_phase: str | None = None
        last_record = 0.0
        while process.poll() is None:
            activity = fetch_activity(observer, args.application_name)
            now = time.time()
            if activity is not None:
                pid = int(activity['pid'])
                phase = phase_name(activity)
                memory = process_memory(pid)
                family_memory = process_family_memory(pid)
                update_peaks(result['phase_peaks'], phase, memory)
                update_peaks(
                    result['family_phase_peaks'],
                    phase,
                    family_memory,
                )
                result['sample_count'] += 1
                if phase != last_phase or now - last_record >= (
                    args.record_interval
                ):
                    result['samples'].append({
                        'activity': activity,
                        'elapsed_seconds': round(
                            now - result['started_unix'],
                            3,
                        ),
                        'family_memory': family_memory,
                        'memory': memory,
                        'phase': phase,
                    })
                    last_phase = phase
                    last_record = now
                    write_json(args.output, result)
            time.sleep(args.sample_interval)

        process_returncode = process.wait()
        result['completed_unix'] = time.time()
        result['elapsed_seconds'] = round(
            result['completed_unix'] - result['started_unix'],
            3,
        )
        result['psql_returncode'] = process_returncode
        catalog = None
        if args.index is not None:
            catalog = index_catalog(observer, args.index)
            result['index_catalog'] = catalog
        errors = qualification_errors(
            process_returncode,
            args.index,
            catalog,
        )
        memory_observed, memory_errors = memory_qualification(
            result['phase_peaks'],
            max_rss_bytes,
            max_swap_bytes,
            result['family_phase_peaks'],
        )
        result['memory_observed'] = memory_observed
        errors.extend(memory_errors)
        result['qualification_errors'] = errors
        result['qualified'] = not errors
        result['returncode'] = process_returncode if process_returncode else (
            0 if not errors else 1
        )
        result['status'] = 'completed' if not errors else 'failed'
        write_json(args.output, result)
    return result['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
