#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg

from qualify_full_rebuild_rss import (
    binary_binding_errors,
    index_catalog,
    memory_qualification,
    sha256,
    write_json,
)


EXPECTED_EQUIVALENCE_SQL_SHA256 = (
    '89182d8143f4346246b94b905d9fdc19f424386fbfacc82de8b86db067dcfae7'
)
MAX_FAMILY_COVERAGE_DELAY_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Finalize one CQ-3E full rebuild from build, process-family, '
            'catalog, and runtime-migration stability evidence.'
        ),
    )
    parser.add_argument('--psql', type=Path, default=Path('psql'))
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user', default=os.environ.get('USER', 'postgres'))
    parser.add_argument('--database', default='postgres')
    parser.add_argument('--build-report', type=Path, required=True)
    parser.add_argument('--family-report', type=Path, required=True)
    parser.add_argument('--equivalence-sql', type=Path, required=True)
    parser.add_argument('--baseline-index', required=True)
    parser.add_argument('--candidate-index', required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def report_memory_limits(
    report: dict[str, Any],
) -> tuple[int | None, int | None]:
    limits = report.get('memory_limits')
    if not isinstance(limits, dict):
        return None, None
    max_rss = limits.get('max_rss_bytes')
    max_swap = limits.get('max_swap_bytes')
    return (
        max_rss if isinstance(max_rss, int) else None,
        max_swap if isinstance(max_swap, int) else None,
    )


def report_backend_identity(
    report: dict[str, Any],
) -> tuple[int, float] | None:
    samples = report.get('samples')
    if not isinstance(samples, list) or not samples:
        return None
    identities: set[tuple[int, float]] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        activity = sample.get('activity')
        if not isinstance(activity, dict):
            return None
        pid = activity.get('pid')
        started = activity.get('backend_started_unix')
        if not isinstance(pid, int) or not isinstance(started, (int, float)):
            return None
        identities.add((pid, float(started)))
    return next(iter(identities)) if len(identities) == 1 else None


def report_last_sample_unix(report: dict[str, Any]) -> float | None:
    started = report.get('started_unix')
    samples = report.get('samples')
    if not isinstance(started, (int, float)) or not isinstance(samples, list):
        return None
    observed: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        elapsed = sample.get('elapsed_seconds')
        if not isinstance(elapsed, (int, float)):
            return None
        observed.append(float(started) + float(elapsed))
    return max(observed) if observed else None


def equivalence_sql_errors(path: Path) -> list[str]:
    actual = sha256(path)
    if actual == EXPECTED_EQUIVALENCE_SQL_SHA256:
        return []
    return [
        'postbuild equivalence SQL differs from the tracked fixture: '
        f'{actual}',
    ]


def build_report_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get('status') != 'completed':
        errors.append('full rebuild report is not completed')
    if report.get('psql_returncode') != 0:
        errors.append('full rebuild psql did not exit successfully')
    if report.get('sample_count', 0) <= 0:
        errors.append('full rebuild report has no samples')
    if report_backend_identity(report) is None:
        errors.append('full rebuild backend identity is missing or unstable')
    max_rss, max_swap = report_memory_limits(report)
    if max_rss is None or max_swap is None:
        errors.append('full rebuild report has no complete memory limits')
        return sorted(set(errors))
    family_peaks = report.get('family_phase_peaks')
    observed, memory_errors = memory_qualification(
        report.get('phase_peaks', {}),
        max_rss,
        max_swap,
        family_peaks if isinstance(family_peaks, dict) else None,
    )
    errors.extend(memory_errors)
    recorded = report.get('memory_observed')
    if recorded is not None and recorded != observed:
        errors.append('full rebuild recorded memory summary is inconsistent')
    errors.extend(
        str(error)
        for error in report.get('qualification_errors', [])
        if str(error)
    )
    binary = report.get('binary')
    binding = binary.get('binding') if isinstance(binary, dict) else None
    errors.extend(binary_binding_errors(binding))
    return sorted(set(errors))


def family_report_errors(
    report: dict[str, Any],
    application_name: str | None,
) -> list[str]:
    errors: list[str] = []
    if report.get('status') != 'completed':
        errors.append('process-family report is not completed')
    if application_name is not None and report.get(
        'application_name'
    ) != application_name:
        errors.append('process-family application identity differs')
    if report.get('sample_count', 0) <= 0:
        errors.append('process-family report has no samples')
    identity = report_backend_identity(report)
    if identity is None:
        errors.append(
            'process-family backend identity is missing or unstable'
        )
    elif report.get('backend_started_unix') != identity[1]:
        errors.append('process-family backend start identity differs')
    coverage_delay = report.get('coverage_started_seconds_after_backend')
    if not isinstance(coverage_delay, (int, float)):
        errors.append('process-family coverage delay is missing')
    elif not 0.0 <= float(coverage_delay) <= (
        MAX_FAMILY_COVERAGE_DELAY_SECONDS
    ):
        errors.append(
            'process-family observation began too late: '
            f'{coverage_delay} seconds'
        )
    max_rss, max_swap = report_memory_limits(report)
    if max_rss is None or max_swap is None:
        errors.append('process-family report has no complete memory limits')
        return sorted(set(errors))
    observed, memory_errors = memory_qualification(
        {},
        max_rss,
        max_swap,
        report.get('family_phase_peaks', {}),
    )
    errors.extend(memory_errors)
    if report.get('memory_observed') != observed:
        errors.append('process-family recorded memory summary is inconsistent')
    errors.extend(
        str(error)
        for error in report.get('qualification_errors', [])
        if str(error)
    )
    if report.get('qualified') is not True:
        errors.append('process-family report is not qualified')
    return sorted(set(errors))


def cross_report_errors(
    build_report: dict[str, Any],
    family_report: dict[str, Any],
    candidate_index: str,
) -> list[str]:
    errors: list[str] = []
    if build_report.get('index') != candidate_index:
        errors.append('full rebuild candidate index identity differs')
    for field in ('host', 'port', 'user', 'database'):
        if build_report.get(field) != family_report.get(field):
            errors.append(f'process-family {field} identity differs')
    if report_memory_limits(build_report) != report_memory_limits(
        family_report
    ):
        errors.append('process-family memory limits differ')
    build_identity = report_backend_identity(build_report)
    family_identity = report_backend_identity(family_report)
    if (
        build_identity is None
        or family_identity is None
        or build_identity != family_identity
    ):
        errors.append('process-family backend identity differs')
    build_phases = build_report.get('phase_peaks')
    family_phases = family_report.get('family_phase_peaks')
    if not isinstance(build_phases, dict) or not isinstance(
        family_phases,
        dict,
    ):
        errors.append('full rebuild phase coverage is missing')
    elif not set(build_phases).issubset(family_phases):
        errors.append('process-family report does not cover every build phase')
    build_last_sample = report_last_sample_unix(build_report)
    family_completed = family_report.get('completed_unix')
    if build_last_sample is None or not isinstance(
        family_completed,
        (int, float),
    ):
        errors.append('full rebuild observation timestamps are missing')
    elif family_completed < build_last_sample:
        errors.append('process-family observation ended before the rebuild')
    return sorted(set(errors))


def connection_dsn(args: argparse.Namespace) -> str:
    return ' '.join([
        f'host={args.host}',
        f'port={args.port}',
        f'user={args.user}',
        f'dbname={args.database}',
        'application_name=evoke_cq3e_finalizer',
    ])


def catalog_errors(
    name: str,
    catalog: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not catalog.get('present'):
        return [f'index is absent: {name}']
    if catalog.get('valid') is not True:
        errors.append(f'index is not valid: {name}')
    if catalog.get('ready') is not True:
        errors.append(f'index is not ready: {name}')
    if not isinstance(catalog.get('bytes'), int) or catalog['bytes'] <= 0:
        errors.append(f'index has no physical bytes: {name}')
    return errors


def psql_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.psql),
        '-X',
        '-v',
        'ON_ERROR_STOP=1',
        '-v',
        f'baseline_index={args.baseline_index}',
        '-v',
        f'candidate_index={args.candidate_index}',
        '-h',
        args.host,
        '-p',
        str(args.port),
        '-U',
        args.user,
        '-d',
        args.database,
        '-f',
        str(args.equivalence_sql),
    ]


def main() -> int:
    args = parse_args()
    for path in (
        args.build_report,
        args.family_report,
        args.equivalence_sql,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    build_report = read_json(args.build_report)
    family_report = read_json(args.family_report)
    result: dict[str, Any] = {
        'baseline_index': args.baseline_index,
        'build_report': {
            'path': str(args.build_report),
            'sha256': sha256(args.build_report),
        },
        'candidate_index': args.candidate_index,
        'database': args.database,
        'equivalence_sql': {
            'path': str(args.equivalence_sql),
            'sha256': sha256(args.equivalence_sql),
        },
        'family_report': {
            'path': str(args.family_report),
            'sha256': sha256(args.family_report),
        },
        'host': args.host,
        'port': args.port,
        'started_unix': time.time(),
        'status': 'checking',
        'user': args.user,
    }
    errors = equivalence_sql_errors(args.equivalence_sql)
    errors.extend(build_report_errors(build_report))
    errors.extend(family_report_errors(
        family_report,
        build_report.get('application_name'),
    ))
    errors.extend(cross_report_errors(
        build_report,
        family_report,
        args.candidate_index,
    ))

    with psycopg.connect(connection_dsn(args), autocommit=True) as connection:
        baseline_catalog = index_catalog(connection, args.baseline_index)
        candidate_catalog = index_catalog(connection, args.candidate_index)
        result['baseline_catalog'] = baseline_catalog
        result['candidate_catalog'] = candidate_catalog
        errors.extend(catalog_errors(args.baseline_index, baseline_catalog))
        errors.extend(catalog_errors(args.candidate_index, candidate_catalog))

    errors = sorted(set(errors))
    result['precheck_errors'] = errors
    if errors:
        result['qualified'] = False
        result['status'] = 'failed'
        result['completed_unix'] = time.time()
        write_json(args.output, result)
        return 1

    stdout_path = args.output.with_suffix(f'{args.output.suffix}.stdout')
    stderr_path = args.output.with_suffix(f'{args.output.suffix}.stderr')
    command = psql_command(args)
    environment = {
        **os.environ,
        'PGAPPNAME': 'evoke_cq3e_postbuild_equivalence',
    }
    process = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    stdout_path.write_text(process.stdout, encoding='utf-8')
    stderr_path.write_text(process.stderr, encoding='utf-8')
    result['equivalence'] = {
        'command': command,
        'returncode': process.returncode,
        'stderr_log': str(stderr_path),
        'stdout_log': str(stdout_path),
    }
    result['completed_unix'] = time.time()
    result['elapsed_seconds'] = round(
        result['completed_unix'] - result['started_unix'],
        3,
    )
    result['qualification_errors'] = (
        []
        if process.returncode == 0
        else ['postbuild runtime-migration stability failed']
    )
    result['qualified'] = process.returncode == 0
    result['status'] = 'completed' if result['qualified'] else 'failed'
    write_json(args.output, result)
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
