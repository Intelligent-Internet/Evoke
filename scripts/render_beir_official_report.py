#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PATH_SPECS = (
    ('evoke_ids', 'ids'),
    ('evoke_text', 'text'),
)


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def fmt_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator in (None, 0):
        return 'n/a'
    return f'{numerator / denominator:.2f}x'


def fmt_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return 'n/a'
    delta = value - baseline
    return f'{delta:+.2f}'


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value:,}'


def active_path_specs(
    payload: dict,
    results: dict[str, dict],
) -> list[tuple[str, str]]:
    requested = set(payload.get('paths', []))
    specs = []
    for key, label in PATH_SPECS:
        if label in requested or any(key in entry for entry in results.values()):
            specs.append((key, label))
    return specs


def query_qps(entry: dict, path_key: str) -> float | None:
    return entry.get(path_key, {}).get('query', {}).get('qps')


def build_overview(
    results: dict[str, dict],
    path_key: str,
) -> dict[str, int | float | None]:
    measured = 0
    passed = 0
    failed = 0
    ratios = []

    for entry in results.values():
        if 'error' in entry:
            continue

        official = entry.get('official_qps')
        qps = query_qps(entry, path_key)
        if qps is None:
            continue

        measured += 1
        if official not in (None, 0):
            ratios.append(qps / official)
        if official is not None and qps >= official:
            passed += 1
        else:
            failed += 1

    return {
        'measured': measured,
        'passed': passed,
        'failed': failed,
        'ratio_min': min(ratios) if ratios else None,
        'ratio_median': statistics.median(ratios) if ratios else None,
        'ratio_max': max(ratios) if ratios else None,
    }


def build_query_header(path_specs: list[tuple[str, str]]) -> list[str]:
    columns = [
        'Dataset',
        'Official BM25S QPS',
        'Local upstream QPS',
    ]
    for _path_key, label in path_specs:
        columns.extend([
            f'Local evoke {label} QPS',
            f'{label} vs official',
            'gap vs official',
            f'{label} vs local upstream',
            f'{label} status',
        ])
    return columns


def build_query_separator(path_specs: list[tuple[str, str]]) -> list[str]:
    separator = ['---', '---:', '---:']
    for _path_key, _label in path_specs:
        separator.extend(['---:', '---:', '---:', '---:', '---'])
    return separator


def build_summary_rows(
    results: dict[str, dict],
    path_specs: list[tuple[str, str]],
) -> list[str]:
    rows = []
    for dataset, entry in results.items():
        if 'error' in entry:
            columns = [dataset, fmt_float(entry.get('official_qps')), 'error']
            for _path_key, _label in path_specs:
                columns.extend(['error'] * 5)
            rows.append(f'| {" | ".join(columns)} |')
            continue

        official = entry.get('official_qps')
        upstream = entry.get('upstream_bm25s', {}).get('query', {}).get('qps')
        columns = [dataset, fmt_float(official), fmt_float(upstream)]

        for path_key, _label in path_specs:
            qps = query_qps(entry, path_key)
            if qps is None:
                status = 'n/a'
            elif official is not None and qps >= official:
                status = 'pass'
            else:
                status = 'fail'
            columns.extend([
                fmt_float(qps),
                fmt_ratio(qps, official),
                fmt_delta(qps, official),
                fmt_ratio(qps, upstream),
                status,
            ])

        rows.append(f'| {" | ".join(columns)} |')
    return rows


def build_build_header(path_specs: list[tuple[str, str]]) -> list[str]:
    columns = ['Dataset', 'Local upstream build ms']
    for _path_key, label in path_specs:
        columns.extend([
            f'evoke {label} build ms',
            f'evoke {label} bytes',
        ])
    columns.append('Wall time s')
    return columns


def build_build_separator(path_specs: list[tuple[str, str]]) -> list[str]:
    separator = ['---', '---:']
    for _path_key, _label in path_specs:
        separator.extend(['---:', '---:'])
    separator.append('---:')
    return separator


def build_build_rows(
    results: dict[str, dict],
    path_specs: list[tuple[str, str]],
) -> list[str]:
    rows = []
    for dataset, entry in results.items():
        if 'error' in entry:
            continue

        columns = [
            dataset,
            fmt_float(entry.get('upstream_bm25s', {}).get('build_ms'), 3),
        ]
        for path_key, _label in path_specs:
            path_entry = entry.get(path_key, {})
            columns.extend([
                fmt_float(path_entry.get('build_ms'), 3),
                fmt_bytes(path_entry.get('build_bytes')),
            ])
        columns.append(fmt_float(entry.get('wall_time_s'), 3))
        rows.append(f'| {" | ".join(columns)} |')
    return rows


def build_error_rows(results: dict[str, dict]) -> list[str]:
    rows = []
    for dataset, entry in results.items():
        if 'error' not in entry:
            continue

        err = entry['error'].replace('\n', ' ')
        rows.append(f'| {dataset} | `{err}` |')
    return rows


def render_report(payload: dict, source_path: Path) -> str:
    results = payload.get('results', {})
    path_specs = active_path_specs(payload, results)
    query_header = build_query_header(path_specs)
    query_separator = build_query_separator(path_specs)
    build_header = build_build_header(path_specs)
    build_separator = build_build_separator(path_specs)
    error_count = sum(1 for entry in results.values() if 'error' in entry)

    lines = [
        '# Official BEIR Performance Comparison',
        '',
        'Source results JSON:',
        '',
        f'- `{source_path}`',
        '',
        'Official reference source:',
        '',
        f'- {payload.get("official_source", "n/a")}',
        '',
        'Paths:',
        '',
        f'- `{", ".join(payload.get("paths", [])) or "n/a"}`',
        '',
        'Top-k:',
        '',
        f'- `{payload.get("top_k", "n/a")}`',
        '',
        'Coverage summary:',
        '',
        f'- datasets covered: `{len(results)}`',
        f'- benchmark errors: `{error_count}`',
    ]

    for path_key, label in path_specs:
        overview = build_overview(results, path_key)
        lines.extend([
            f'- {label} measured datasets: `{overview["measured"]}`',
            f'- {label} target passed: `{overview["passed"]}`',
            f'- {label} below official: `{overview["failed"]}`',
            f'- minimum {label}/official ratio: '
            f'`{fmt_ratio(overview["ratio_min"], 1.0)}`',
            f'- median {label}/official ratio: '
            f'`{fmt_ratio(overview["ratio_median"], 1.0)}`',
            f'- maximum {label}/official ratio: '
            f'`{fmt_ratio(overview["ratio_max"], 1.0)}`',
        ])

    lines.extend([
        '',
        '## Query Summary',
        '',
        f'| {" | ".join(query_header)} |',
        f'| {" | ".join(query_separator)} |',
    ])
    lines.extend(build_summary_rows(results, path_specs))

    lines.extend([
        '',
        '## Build Summary',
        '',
        f'| {" | ".join(build_header)} |',
        f'| {" | ".join(build_separator)} |',
    ])
    lines.extend(build_build_rows(results, path_specs))

    error_rows = build_error_rows(results)
    if error_rows:
        lines.extend([
            '',
            '## Errors',
            '',
            '| Dataset | Error |',
            '| --- | --- |',
        ])
        lines.extend(error_rows)

    return '\n'.join(lines) + '\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render a Markdown report from official BEIR benchmark JSON.'
    )
    parser.add_argument('input', type=Path, help='Benchmark JSON file.')
    parser.add_argument('output', type=Path, help='Markdown output file.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    report = render_report(payload, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
