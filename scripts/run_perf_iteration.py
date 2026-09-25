#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_TEMP_ROOT = Path(tempfile.gettempdir())
DEFAULT_CACHED_DATASETS_DIR = (
    DEFAULT_TEMP_ROOT / 'evoke_dataset_cache/beir_official'
)
DEFAULT_OUTPUT_ROOT = DEFAULT_TEMP_ROOT / 'evoke_perf_iterations'
DEFAULT_PG_CONFIG = Path(
    os.environ.get(
        'PG_CONFIG',
        '/opt/homebrew/opt/postgresql@17/bin/pg_config',
    )
)
HARNESS_SCRIPT = 'scripts/benchmark_filtered_ordered_must_query_only.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run one local performance iteration: build baseline and '
            'candidate, execute a focused benchmark harness, compare the '
            'results, and emit a recommendation.'
        )
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        '--baseline-ref',
        default='main',
        help='Git ref used for the baseline build.',
    )
    parser.add_argument(
        '--candidate-label',
        default='candidate',
        help='Short label for the current worktree build.',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='Datasets to benchmark with the focused harness.',
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=DEFAULT_CACHED_DATASETS_DIR,
        help='Dataset cache directory.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='Directory where artifacts will be written.',
    )
    parser.add_argument(
        '--pg-config',
        type=Path,
        default=DEFAULT_PG_CONFIG,
        help='pg_config used for make install.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Top-k used by the focused harness.',
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=250,
        help='How many query cases to keep per dataset.',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help='How many query-loop repeats to run per dataset.',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help=(
            'How many independent query-loop measurements to run per '
            'dataset and label. The median-QPS iteration is used for the '
            'primary comparison, and all iterations are preserved.'
        ),
    )
    parser.add_argument(
        '--keep-median-qps-pct',
        type=float,
        default=5.0,
        help='Median QPS gain needed for an automatic keep recommendation.',
    )
    parser.add_argument(
        '--max-single-regression-pct',
        type=float,
        default=-5.0,
        help='Worst allowed single-dataset regression for keep.',
    )
    parser.add_argument(
        '--reject-median-qps-pct',
        type=float,
        default=-2.0,
        help=(
            'Median QPS delta at or below this recommends reject. The '
            'default leaves a small noise band for local benchmark jitter.'
        ),
    )
    parser.add_argument(
        '--no-worktree-cleanup',
        action='store_true',
        help='Keep the temporary baseline worktree for debugging.',
    )
    parser.add_argument(
        '--keep-on-failure',
        action='store_true',
        help='Keep benchmark databases/state files when a harness step fails.',
    )
    parser.add_argument(
        '--query-retries',
        type=int,
        default=5,
        help=(
            'Retries for transient query-harness failures caused by local '
            'fresh-backend relation visibility races.'
        ),
    )
    return parser.parse_args()


def run(
    cmd: list[str],
    cwd: Path,
    capture: bool = False,
    failure_artifact: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=capture,
    )
    if proc.returncode == 0:
        return proc

    if failure_artifact is not None:
        write_json(
            failure_artifact,
            {
                'cmd': cmd,
                'cwd': str(cwd),
                'returncode': proc.returncode,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
            },
        )
    raise subprocess.CalledProcessError(
        proc.returncode,
        cmd,
        output=proc.stdout,
        stderr=proc.stderr,
    )


def git_output(repo_root: Path, args: list[str]) -> str:
    proc = run(
        ['git', '-C', str(repo_root)] + args,
        cwd=repo_root,
        capture=True,
    )
    return proc.stdout.strip()


def sanitize_name(value: str) -> str:
    return value.replace('-', '_').replace('/', '_')


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def is_transient_query_failure(stderr: str | None) -> bool:
    if not stderr:
        return False
    return (
        'invalid evoke index metapage' in stderr or
        'truncated evoke index payload' in stderr or
        (
            'evoke index relation' in stderr and
            ' is empty' in stderr
        )
    )


def median_query_payload(
    payloads: list[dict[str, object]],
) -> dict[str, object]:
    if not payloads:
        raise ValueError('at least one query payload is required')
    if len(payloads) == 1:
        payload = dict(payloads[0])
        payload['query_iterations'] = payloads
        payload['query_summary'] = {
            'iteration_count': 1,
            'selected_iteration': int(payload.get('iteration', 1)),
            'qps_mean': float(payload['query']['qps']),
            'qps_median': float(payload['query']['qps']),
            'qps_min': float(payload['query']['qps']),
            'qps_max': float(payload['query']['qps']),
            'qps_stdev': 0.0,
        }
        return payload

    qps_values = [float(payload['query']['qps']) for payload in payloads]
    median_qps = statistics.median(qps_values)
    selected_index = min(
        range(len(payloads)),
        key=lambda idx: (abs(qps_values[idx] - median_qps), idx),
    )
    payload = dict(payloads[selected_index])
    payload['query_iterations'] = payloads
    payload['query_summary'] = {
        'iteration_count': len(payloads),
        'selected_iteration': int(payload.get('iteration', selected_index + 1)),
        'qps_mean': statistics.mean(qps_values),
        'qps_median': median_qps,
        'qps_min': min(qps_values),
        'qps_max': max(qps_values),
        'qps_stdev': statistics.stdev(qps_values),
    }
    return payload


def run_make_install(repo_root: Path, pg_config: Path) -> None:
    run(
        ['make', f'PG_CONFIG={pg_config}', 'install'],
        cwd=repo_root,
    )


def prepare_worktree(
    repo_root: Path,
    baseline_ref: str,
    output_dir: Path,
) -> Path:
    worktree_dir = output_dir / 'baseline-worktree'
    if worktree_dir.exists():
        remove_worktree(repo_root, worktree_dir)
    run(
        [
            'git',
            '-C',
            str(repo_root),
            'worktree',
            'prune',
        ],
        cwd=repo_root,
    )
    run(
        [
            'git',
            '-C',
            str(repo_root),
            'worktree',
            'add',
            '--detach',
            str(worktree_dir),
            baseline_ref,
        ],
        cwd=repo_root,
    )
    return worktree_dir


def remove_worktree(repo_root: Path, worktree_dir: Path) -> None:
    if not worktree_dir.exists():
        return
    run(
        [
            'git',
            '-C',
            str(repo_root),
            'worktree',
            'remove',
            '--force',
            str(worktree_dir),
        ],
        cwd=repo_root,
    )


def run_harness_command(
    repo_root: Path,
    datasets_dir: Path,
    dataset: str,
    state_file: Path,
    prepare_path: Path,
    query_path: Path,
    failure_dir: Path,
    top_k: int,
    max_cases: int,
    repeats: int,
    iterations: int,
    query_retries: int,
    application_name: str,
    keep_on_failure: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    harness = Path(__file__).resolve().with_name(
        'benchmark_filtered_ordered_must_query_only.py'
    )
    prepare_cmd = [
        'python3',
        str(harness),
        '--repo-root',
        str(repo_root),
        '--datasets-dir',
        str(datasets_dir),
        '--dataset',
        dataset,
        '--top-k',
        str(top_k),
        '--max-cases',
        str(max_cases),
        '--state-file',
        str(state_file),
        '--mode',
        'prepare',
    ]
    query_cmd = [
        'python3',
        str(harness),
        '--repo-root',
        str(repo_root),
        '--state-file',
        str(state_file),
        '--application-name',
        application_name,
        '--repeats',
        str(repeats),
        '--mode',
        'query',
    ]
    cleanup_cmd = [
        'python3',
        str(harness),
        '--repo-root',
        str(repo_root),
        '--state-file',
        str(state_file),
        '--mode',
        'cleanup',
    ]

    if iterations < 1:
        iterations = 1

    prepare_proc = run(
        prepare_cmd,
        cwd=repo_root,
        capture=True,
        failure_artifact=failure_dir / f'{sanitize_name(dataset)}.prepare.failure.json',
    )
    prepare_payload = json.loads(prepare_proc.stdout)
    write_json(prepare_path, prepare_payload)
    query_payloads = []
    try:
        for iteration in range(iterations):
            iteration_cmd = list(query_cmd)
            app_name_idx = iteration_cmd.index('--application-name') + 1
            iteration_cmd[app_name_idx] = (
                f'{application_name}_iter_{iteration + 1}'
            )
            query_proc = None
            for attempt in range(query_retries + 1):
                failure_path = (
                    failure_dir /
                    f'{sanitize_name(dataset)}.query_{iteration + 1}'
                    f'.attempt_{attempt + 1}.failure.json'
                )
                try:
                    query_proc = run(
                        iteration_cmd,
                        cwd=repo_root,
                        capture=True,
                        failure_artifact=failure_path,
                    )
                    break
                except subprocess.CalledProcessError as exc:
                    if (
                        attempt >= query_retries or
                        not is_transient_query_failure(exc.stderr)
                    ):
                        raise
                    time.sleep(0.25 * (attempt + 1))
            if query_proc is None:
                raise RuntimeError('query harness did not produce a result')
            query_payload = json.loads(query_proc.stdout)
            query_payload['iteration'] = iteration + 1
            query_payloads.append(query_payload)
            write_json(
                query_path.with_suffix(f'.query_{iteration + 1}.json'),
                query_payload,
            )
        query_payload = median_query_payload(query_payloads)
        write_json(query_path, query_payload)
    finally:
        if not keep_on_failure:
            cleanup_error = None
            try:
                run(
                    cleanup_cmd,
                    cwd=repo_root,
                    capture=True,
                    failure_artifact=(
                        failure_dir /
                        f'{sanitize_name(dataset)}.cleanup.failure.json'
                    ),
                )
            except subprocess.CalledProcessError as exc:
                cleanup_error = exc
            if state_file.exists():
                state_file.unlink()
            if cleanup_error is not None and sys.exc_info()[0] is None:
                raise cleanup_error
    return prepare_payload, query_payload


def run_label(
    label: str,
    repo_root: Path,
    datasets_dir: Path,
    output_dir: Path,
    datasets: list[str],
    pg_config: Path,
    top_k: int,
    max_cases: int,
    repeats: int,
    iterations: int,
    query_retries: int,
    keep_on_failure: bool,
) -> dict[str, dict[str, object]]:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    run_make_install(repo_root, pg_config)
    results: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        state_file = label_dir / f'{sanitize_name(dataset)}.state.json'
        app_name = (
            f'evoke_iter_{sanitize_name(label)}_'
            f'{sanitize_name(dataset)}'
        )
        prepare_path = label_dir / f'{sanitize_name(dataset)}.prepare.json'
        query_path = label_dir / f'{sanitize_name(dataset)}.query.json'
        failure_dir = label_dir / 'failures'
        _, query_payload = run_harness_command(
            repo_root=repo_root,
            datasets_dir=datasets_dir,
            dataset=dataset,
            state_file=state_file,
            prepare_path=prepare_path,
            query_path=query_path,
            failure_dir=failure_dir,
            top_k=top_k,
            max_cases=max_cases,
            repeats=repeats,
            iterations=iterations,
            query_retries=query_retries,
            application_name=app_name,
            keep_on_failure=keep_on_failure,
        )
        results[dataset] = query_payload
    return results


def pct_delta(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def build_comparison(
    baseline_label: str,
    candidate_label: str,
    baseline_results: dict[str, dict[str, object]],
    candidate_results: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    rows = []
    qps_deltas = []
    build_deltas = []
    for dataset in args.datasets:
        baseline = baseline_results[dataset]
        candidate = candidate_results[dataset]
        old_qps = float(baseline['query']['qps'])
        new_qps = float(candidate['query']['qps'])
        old_build = float(baseline['build_ms'])
        new_build = float(candidate['build_ms'])
        row = {
            'dataset': dataset,
            'documents': int(baseline['stats']['documents']),
            'queries': int(baseline['stats']['queries']),
            f'{baseline_label}_qps': old_qps,
            f'{candidate_label}_qps': new_qps,
            'qps_delta_pct': pct_delta(old_qps, new_qps),
            f'{baseline_label}_build_ms': old_build,
            f'{candidate_label}_build_ms': new_build,
            'build_delta_pct': pct_delta(old_build, new_build),
        }
        rows.append(row)
        qps_deltas.append(row['qps_delta_pct'])
        build_deltas.append(row['build_delta_pct'])

    median_qps = statistics.median(qps_deltas)
    median_build = statistics.median(build_deltas)
    worst_qps = min(qps_deltas)
    if (
        median_qps >= args.keep_median_qps_pct
        and worst_qps >= args.max_single_regression_pct
    ):
        recommendation = 'keep'
    elif (
        median_qps <= args.reject_median_qps_pct
        or worst_qps < args.max_single_regression_pct
    ):
        recommendation = 'reject'
    else:
        recommendation = 'inconclusive'

    return {
        'baseline_label': baseline_label,
        'candidate_label': candidate_label,
        'datasets': args.datasets,
        'rows': rows,
        'summary': {
            'median_qps_delta_pct': median_qps,
            'median_build_delta_pct': median_build,
            'best_qps_delta_pct': max(qps_deltas),
            'worst_qps_delta_pct': worst_qps,
            'recommendation': recommendation,
            'thresholds': {
                'keep_median_qps_pct': args.keep_median_qps_pct,
                'reject_median_qps_pct': args.reject_median_qps_pct,
                'max_single_regression_pct': (
                    args.max_single_regression_pct
                ),
            },
        },
    }


def format_pct(value: float) -> str:
    return f'{value:+.2f}%'


def write_markdown_report(path: Path, comparison: dict[str, object]) -> None:
    rows = comparison['rows']
    summary = comparison['summary']
    baseline_label = comparison['baseline_label']
    candidate_label = comparison['candidate_label']
    lines = [
        '# Performance Iteration Report',
        '',
        f'- baseline: `{baseline_label}`',
        f'- candidate: `{candidate_label}`',
        (
            '- recommendation: '
            f"`{summary['recommendation']}`"
        ),
        (
            '- median QPS delta: '
            f"{format_pct(summary['median_qps_delta_pct'])}"
        ),
        (
            '- median build delta: '
            f"{format_pct(summary['median_build_delta_pct'])}"
        ),
        '',
        '| Dataset | Docs | Queries | '
        f'{baseline_label} qps | {candidate_label} qps | '
        'QPS delta | '
        f'{baseline_label} build | {candidate_label} build | '
        'build delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| '
            f"{row['dataset']} | "
            f"{row['documents']} | "
            f"{row['queries']} | "
            f"{row[f'{baseline_label}_qps']:.2f} | "
            f"{row[f'{candidate_label}_qps']:.2f} | "
            f"{format_pct(row['qps_delta_pct'])} | "
            f"{row[f'{baseline_label}_build_ms']:.2f} | "
            f"{row[f'{candidate_label}_build_ms']:.2f} | "
            f"{format_pct(row['build_delta_pct'])} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    datasets_dir = args.datasets_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_label = sanitize_name(args.baseline_ref)
    candidate_label = sanitize_name(args.candidate_label)
    baseline_sha = git_output(repo_root, ['rev-parse', args.baseline_ref])
    candidate_sha = git_output(repo_root, ['rev-parse', 'HEAD'])

    baseline_worktree = prepare_worktree(
        repo_root,
        args.baseline_ref,
        output_dir,
    )
    try:
        baseline_results = run_label(
            label=baseline_label,
            repo_root=baseline_worktree,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            datasets=args.datasets,
            pg_config=args.pg_config,
            top_k=args.top_k,
            max_cases=args.max_cases,
            repeats=args.repeats,
            iterations=args.iterations,
            query_retries=args.query_retries,
            keep_on_failure=args.keep_on_failure,
        )
        candidate_results = run_label(
            label=candidate_label,
            repo_root=repo_root,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            datasets=args.datasets,
            pg_config=args.pg_config,
            top_k=args.top_k,
            max_cases=args.max_cases,
            repeats=args.repeats,
            iterations=args.iterations,
            query_retries=args.query_retries,
            keep_on_failure=args.keep_on_failure,
        )
    finally:
        run_make_install(repo_root, args.pg_config)
        if not args.no_worktree_cleanup:
            remove_worktree(repo_root, baseline_worktree)

    comparison = build_comparison(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        args=args,
    )
    comparison['git'] = {
        'baseline_ref': args.baseline_ref,
        'baseline_sha': baseline_sha,
        'candidate_sha': candidate_sha,
        'candidate_branch': git_output(
            repo_root,
            ['rev-parse', '--abbrev-ref', 'HEAD'],
        ),
        'candidate_dirty': bool(
            git_output(repo_root, ['status', '--porcelain'])
        ),
        'candidate_status': git_output(
            repo_root,
            ['status', '--porcelain'],
        ).splitlines(),
    }
    comparison['harness'] = {
        'script': HARNESS_SCRIPT,
        'top_k': args.top_k,
        'max_cases': args.max_cases,
        'repeats': args.repeats,
        'iterations': args.iterations,
        'query_retries': args.query_retries,
    }
    comparison_path = output_dir / 'comparison.json'
    report_path = output_dir / 'comparison.md'
    write_json(comparison_path, comparison)
    write_markdown_report(report_path, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
