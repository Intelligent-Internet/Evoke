#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import psycopg

from qualify_full_rebuild_rss import (
    fetch_activity,
    memory_qualification,
    phase_name,
    process_family_memory,
    update_peaks,
    write_json,
)


DEFAULT_MAX_RSS_MIB = 32 * 1024
DEFAULT_MAX_SWAP_MIB = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Attach to a PostgreSQL application and record process-family '
            'PSS, private memory, high-water marks, and swap.'
        ),
    )
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user', default=os.environ.get('USER', 'postgres'))
    parser.add_argument('--database', default='postgres')
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--record-interval', type=float, default=30.0)
    parser.add_argument('--wait-timeout', type=float, default=300.0)
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


def observer_dsn(args: argparse.Namespace) -> str:
    return ' '.join([
        f'host={args.host}',
        f'port={args.port}',
        f'user={args.user}',
        f'dbname={args.database}',
        'application_name=evoke_process_family_observer',
    ])


def qualification_report(
    family_peaks: dict[str, dict[str, int]],
    max_rss_bytes: int,
    max_swap_bytes: int,
) -> dict[str, Any]:
    observed, errors = memory_qualification(
        {},
        max_rss_bytes,
        max_swap_bytes,
        family_peaks,
    )
    return {
        'memory_observed': observed,
        'qualification_errors': errors,
        'qualified': not errors,
    }


def main() -> int:
    args = parse_args()
    if args.sample_interval <= 0 or args.record_interval <= 0:
        raise ValueError('sampling intervals must be positive')
    if args.wait_timeout < 0:
        raise ValueError('wait timeout must be nonnegative')
    if args.max_rss_mib < 0 or args.max_swap_mib < 0:
        raise ValueError('memory limits must be nonnegative')

    max_rss_bytes = int(args.max_rss_mib * 1024 * 1024)
    max_swap_bytes = int(args.max_swap_mib * 1024 * 1024)
    result: dict[str, Any] = {
        'application_name': args.application_name,
        'database': args.database,
        'family_phase_peaks': {},
        'host': args.host,
        'memory_limits': {
            'max_rss_bytes': max_rss_bytes,
            'max_swap_bytes': max_swap_bytes,
        },
        'port': args.port,
        'sample_count': 0,
        'samples': [],
        'started_unix': time.time(),
        'status': 'waiting',
        'user': args.user,
    }
    write_json(args.output, result)

    seen = False
    last_phase: str | None = None
    last_record = 0.0
    with psycopg.connect(observer_dsn(args), autocommit=True) as observer:
        while True:
            activity = fetch_activity(observer, args.application_name)
            now = time.time()
            if activity is None:
                if seen:
                    break
                if now - result['started_unix'] >= args.wait_timeout:
                    result['qualification_errors'] = [
                        'target PostgreSQL application was not observed',
                    ]
                    result['qualified'] = False
                    result['status'] = 'failed'
                    write_json(args.output, result)
                    return 1
                time.sleep(args.sample_interval)
                continue

            if not seen:
                seen = True
                result['backend_started_unix'] = activity.get(
                    'backend_started_unix'
                )
                result['coverage_started_seconds_after_backend'] = round(
                    max(
                        0.0,
                        now - float(activity['backend_started_unix']),
                    ),
                    3,
                )
                result['observation_started_unix'] = now
                result['status'] = 'running'

            phase = phase_name(activity)
            family_memory = process_family_memory(int(activity['pid']))
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
                    'phase': phase,
                })
                last_phase = phase
                last_record = now
                write_json(args.output, result)
            time.sleep(args.sample_interval)

    result['completed_unix'] = time.time()
    result['elapsed_seconds'] = round(
        result['completed_unix'] - result['started_unix'],
        3,
    )
    result.update(qualification_report(
        result['family_phase_peaks'],
        max_rss_bytes,
        max_swap_bytes,
    ))
    result['status'] = 'completed' if result['qualified'] else 'failed'
    write_json(args.output, result)
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
