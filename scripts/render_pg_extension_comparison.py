#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ORDER = [
    'arguana',
    'climate-fever',
    'cqadupstack',
    'dbpedia-entity',
    'fever',
    'fiqa',
    'hotpotqa',
    'msmarco',
    'nfcorpus',
    'nq',
    'quora',
    'scidocs',
    'scifact',
    'trec-covid',
    'webis-touche2020',
]


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def fmt_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator in (None, 0):
        return 'n/a'
    return f'{numerator / denominator:.2f}x'


def query_qps(entry: dict[str, Any], key: str) -> float | None:
    return entry.get(key, {}).get('query', {}).get('qps')


def build_ms(entry: dict[str, Any], key: str) -> float | None:
    return entry.get(key, {}).get('build_ms')


def ratio_stats(
    results: dict[str, dict[str, Any]],
    key: str,
) -> tuple[int, int, float | None, float | None, float | None]:
    at_or_above = 0
    measured = 0
    ratios: list[float] = []

    for entry in results.values():
        qps = query_qps(entry, key)
        upstream = query_qps(entry, 'upstream_bm25s')
        if qps is None or upstream in (None, 0):
            continue
        measured += 1
        ratio = qps / upstream
        ratios.append(ratio)
        if ratio >= 1.0:
            at_or_above += 1

    if not ratios:
        return measured, at_or_above, None, None, None

    return (
        measured,
        at_or_above,
        min(ratios),
        statistics.median(ratios),
        max(ratios),
    )


def total_build_ratio(
    results: dict[str, dict[str, Any]],
    key: str,
) -> float | None:
    total = 0.0
    upstream_total = 0.0

    for entry in results.values():
        value = build_ms(entry, key)
        upstream = build_ms(entry, 'upstream_bm25s')
        if value is None or upstream is None:
            continue
        total += value
        upstream_total += upstream

    if upstream_total == 0:
        return None
    return total / upstream_total


def render_report(payload: dict[str, Any], source_path: Path) -> str:
    results = payload.get('results', {})
    ordered_results = {
        name: results[name]
        for name in ORDER
        if name in results and 'pg_bm25s' in results[name]
    }
    omitted = [
        name for name in ORDER
        if name in results and 'pg_bm25s' not in results[name]
    ]

    ids_stats = ratio_stats(ordered_results, 'evoke_ids')
    text_stats = ratio_stats(ordered_results, 'evoke_text')
    pg_stats = ratio_stats(ordered_results, 'pg_bm25s')

    lines = [
        '# PostgreSQL BM25 Extension Comparison',
        '',
        'Source results JSON:',
        '',
        f'- `{source_path}`',
        '',
        'This report compares the two PostgreSQL extension implementations',
        'on the same localhost machine and the same tokenized BEIR corpus.',
        'The public upstream `bm25s` numbers are listed only as context.',
        '',
        '## Scope',
        '',
        '- host: `localhost`',
        '- benchmark set: focused same-machine subset with completed',
        '  `pg_bm25s` PostgreSQL-native measurements',
        '- top-k: `1000`',
        '- local upstream path: Python `bm25s` on the same token stream',
        '- `evoke ids`: `evoke_query_ids(...)`',
        '- `evoke text[]`: `evoke_query_tokens(...)`',
        '- `pg_bm25s`: `bm25s_search(...)` after `CREATE INDEX USING bm25s`',
        '- `pg_bm25s` corpus/query text is pretokenized and space-joined so',
        '  both PostgreSQL extensions receive the same normalized tokens',
        '',
        'Completed comparison datasets:',
        '',
        f'- `{", ".join(ordered_results)}`',
        '',
        '## Query Summary',
        '',
        '`min`, `median`, and `max` below are per-dataset ratios against',
        'same-machine local upstream `bm25s`.',
        '',
        '| Path | At or above local upstream | Min vs local upstream | '
        'Median vs local upstream | Max vs local upstream |',
        '| --- | ---: | ---: | ---: | ---: |',
        f'| `evoke ids` | `{ids_stats[1]}/{ids_stats[0]}` | '
        f'`{fmt_ratio(ids_stats[2], 1.0)}` | '
        f'`{fmt_ratio(ids_stats[3], 1.0)}` | '
        f'`{fmt_ratio(ids_stats[4], 1.0)}` |',
        f'| `evoke text[]` | `{text_stats[1]}/{text_stats[0]}` | '
        f'`{fmt_ratio(text_stats[2], 1.0)}` | '
        f'`{fmt_ratio(text_stats[3], 1.0)}` | '
        f'`{fmt_ratio(text_stats[4], 1.0)}` |',
        f'| `pg_bm25s` | `{pg_stats[1]}/{pg_stats[0]}` | '
        f'`{fmt_ratio(pg_stats[2], 1.0)}` | '
        f'`{fmt_ratio(pg_stats[3], 1.0)}` | '
        f'`{fmt_ratio(pg_stats[4], 1.0)}` |',
        '',
        '## Index Build Summary',
        '',
        '`build` means index construction time only.',
        '',
        '| Path | Total build vs local upstream |',
        '| --- | ---: |',
        f'| `evoke ids` | '
        f'`{fmt_ratio(total_build_ratio(ordered_results, "evoke_ids"), 1.0)}` |',
        f'| `evoke text[]` | '
        f'`{fmt_ratio(total_build_ratio(ordered_results, "evoke_text"), 1.0)}` |',
        f'| `pg_bm25s` | '
        f'`{fmt_ratio(total_build_ratio(ordered_results, "pg_bm25s"), 1.0)}` |',
        '',
        '## Dataset Table',
        '',
        '| Dataset | Docs | Queries | `bm25s` official QPS | '
        '`bm25s` local QPS | `evoke ids` QPS | `ids / local` | '
        '`evoke text[]` QPS | `text[] / local` | '
        '`pg_bm25s` QPS | `pg_bm25s / local` |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
        '---: | ---: | ---: |',
    ]

    for dataset, entry in ordered_results.items():
        stats = entry.get('stats', {})
        lines.append(
            '| '
            + ' | '.join(
                [
                    f'`{dataset}`',
                    f'{stats.get("documents", "n/a"):,}',
                    f'{stats.get("queries", "n/a"):,}',
                    fmt_float(entry.get('official_qps')),
                    fmt_float(query_qps(entry, 'upstream_bm25s')),
                    fmt_float(query_qps(entry, 'evoke_ids')),
                    fmt_ratio(
                        query_qps(entry, 'evoke_ids'),
                        query_qps(entry, 'upstream_bm25s'),
                    ),
                    fmt_float(query_qps(entry, 'evoke_text')),
                    fmt_ratio(
                        query_qps(entry, 'evoke_text'),
                        query_qps(entry, 'upstream_bm25s'),
                    ),
                    fmt_float(query_qps(entry, 'pg_bm25s')),
                    fmt_ratio(
                        query_qps(entry, 'pg_bm25s'),
                        query_qps(entry, 'upstream_bm25s'),
                    ),
                ]
            )
            + ' |'
        )

    lines.extend([
        '',
        '## Readout',
        '',
        '- Use this table for the implementation comparison. The only',
        '  ratios that matter here are against same-machine local',
        '  upstream `bm25s`.',
        '- The public upstream QPS column is context only. It is not used',
        '  for the speedup ratios in this report.',
        '- `evoke ids` remains the strongest path when raw throughput',
        '  matters.',
        '- `evoke text[]` is the closer apples-to-apples comparison',
        '  against `pg_bm25s`, because both operate on token strings inside',
        '  PostgreSQL rather than integer token IDs.',
        '- `pg_bm25s` here is measured through its database-native path,',
        '  not as a Python library. Any remaining gap is therefore part of',
        '  the real extension behavior seen by PostgreSQL users.',
    ])

    if omitted:
        lines.extend([
            '',
            '## Large-Set Note',
            '',
            'The following official BEIR datasets are omitted from the direct',
            'comparison table because `pg_bm25s` native build was not',
            'practically comparable on localhost:',
            '',
            f'- `{", ".join(omitted)}`',
            '- The clearest example was `climate-fever`, where native',
            '  `CREATE INDEX USING bm25s` remained active for roughly',
            '  15 minutes before it was stopped.',
        ])

    return '\n'.join(lines) + '\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render the PostgreSQL extension comparison report.'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    report = render_report(payload, args.input)
    args.output.write_text(report, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
