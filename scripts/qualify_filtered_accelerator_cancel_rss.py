#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import time

import psycopg
from psycopg import errors


RSS_PATTERN = re.compile(r'^VmRSS:\s+(\d+)\s+kB$', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Cancel a filtered Evoke query repeatedly in the same backend '
            'and verify that its RSS reaches a bounded plateau.'
        ),
    )
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--query-sql', required=True)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--cancel-delay-ms', type=float, default=10.0)
    parser.add_argument('--rss-slack-mib', type=int, default=64)
    parser.add_argument(
        '--setup-sql',
        action='append',
        default=[],
        help=(
            'session SQL executed on the victim backend before measuring; '
            'repeat to force a specific scorer route'
        ),
    )
    return parser.parse_args()


def backend_rss_bytes(
    control: psycopg.Connection[tuple[object, ...]],
    pid: int,
) -> int:
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT pg_read_file('/proc/' || %s || '/status')",
            (pid,),
        )
        status = str(cursor.fetchone()[0])
    match = RSS_PATTERN.search(status)
    if match is None:
        raise AssertionError(f'could not read VmRSS for backend {pid}')
    return int(match.group(1)) * 1024


def run_query(
    victim: psycopg.Connection[tuple[object, ...]],
    query_sql: str,
) -> None:
    with victim.cursor() as cursor:
        cursor.execute(query_sql)
        cursor.fetchall()


def wait_until_active(
    control: psycopg.Connection[tuple[object, ...]],
    pid: int,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with control.cursor() as cursor:
            cursor.execute(
                'SELECT state = \'active\' '
                'FROM pg_stat_activity WHERE pid = %s',
                (pid,),
            )
            row = cursor.fetchone()
        if row is not None and bool(row[0]):
            return
        time.sleep(0.001)
    raise AssertionError(f'backend {pid} did not enter active query state')


def main() -> None:
    args = parse_args()
    if args.iterations < 6:
        raise SystemExit('--iterations must be at least 6')
    if args.cancel_delay_ms <= 0:
        raise SystemExit('--cancel-delay-ms must be positive')
    if args.rss_slack_mib < 0:
        raise SystemExit('--rss-slack-mib must be non-negative')

    with (
        psycopg.connect(args.dsn, autocommit=True) as victim,
        psycopg.connect(args.dsn, autocommit=True) as control,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
    ):
        pid = victim.info.backend_pid
        with victim.cursor() as cursor:
            for setup_sql in args.setup_sql:
                cursor.execute(setup_sql)
        baseline = backend_rss_bytes(control, pid)
        samples: list[int] = []

        for iteration in range(args.iterations):
            future = executor.submit(run_query, victim, args.query_sql)
            wait_until_active(control, pid)
            time.sleep(args.cancel_delay_ms / 1000.0)
            with control.cursor() as cursor:
                cursor.execute('SELECT pg_cancel_backend(%s)', (pid,))
                if not bool(cursor.fetchone()[0]):
                    raise AssertionError(
                        f'pg_cancel_backend({pid}) failed at iteration '
                        f'{iteration}'
                    )
            try:
                future.result(timeout=30)
            except errors.QueryCanceled:
                pass
            else:
                raise AssertionError(
                    'filtered query completed before cancellation; increase '
                    'its work or lower --cancel-delay-ms'
                )

            with victim.cursor() as cursor:
                cursor.execute('SELECT 1')
                if cursor.fetchone()[0] != 1:
                    raise AssertionError('canceled backend is not reusable')
            samples.append(backend_rss_bytes(control, pid))

    split = args.iterations // 2
    middle = samples[split // 2:split]
    tail = samples[split:]
    slack = args.rss_slack_mib * 1024 * 1024
    ceiling = max(middle) + slack
    if max(tail) > ceiling:
        raise AssertionError(
            'filtered cancellation RSS did not plateau: '
            f'baseline={baseline} samples={samples} ceiling={ceiling}'
        )
    print(
        {
            'backend_pid': pid,
            'baseline_rss_bytes': baseline,
            'samples_rss_bytes': samples,
            'tail_ceiling_bytes': ceiling,
            'setup_sql': args.setup_sql,
            'backend_reusable': True,
            'rss_plateau': True,
        }
    )


if __name__ == '__main__':
    main()
