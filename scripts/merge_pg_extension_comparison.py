#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def merge_entry(
    merged: dict[str, Any],
    source: dict[str, Any],
    dataset: str,
) -> None:
    entry = source['results'].get(dataset)
    if entry is None:
        return

    target = merged['results'].setdefault(
        dataset,
        {
            'dataset': dataset,
            'official_qps': entry.get('official_qps'),
        },
    )

    if target.get('official_qps') is None:
        target['official_qps'] = entry.get('official_qps')
    if 'stats' in entry and 'stats' not in target:
        target['stats'] = entry['stats']
    if 'wall_time_s' in entry:
        target['wall_time_s'] = entry['wall_time_s']

    for key in (
        'upstream_bm25s',
        'evoke_ids',
        'evoke_text',
        'pg_bm25s',
    ):
        if key in entry:
            target[key] = entry[key]

    if 'error' in entry:
        target['error'] = entry['error']
    if 'traceback' in entry:
        target['traceback'] = entry['traceback']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge local upstream, evoke, and pg_bm25s results.'
    )
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--pg-bm25s', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ids = load_payload(args.ids)
    text = load_payload(args.text)
    upstream = load_payload(args.upstream)
    pg_bm25s = load_payload(args.pg_bm25s)

    datasets = sorted(
        set(ids.get('results', {}))
        | set(text.get('results', {}))
        | set(upstream.get('results', {}))
        | set(pg_bm25s.get('results', {}))
    )

    merged: dict[str, Any] = {
        'created_at': (
            pg_bm25s.get('created_at')
            or upstream.get('created_at')
            or text.get('created_at')
            or ids.get('created_at')
        ),
        'official_source': (
            pg_bm25s.get('official_source')
            or upstream.get('official_source')
            or text.get('official_source')
            or ids.get('official_source')
        ),
        'top_k': (
            pg_bm25s.get('top_k')
            or upstream.get('top_k')
            or text.get('top_k')
            or ids.get('top_k')
        ),
        'paths': ['upstream', 'evoke_ids', 'evoke_text',
                  'pg_bm25s'],
        'results': {},
    }

    for dataset in datasets:
        merge_entry(merged, upstream, dataset)
        merge_entry(merged, ids, dataset)
        merge_entry(merged, text, dataset)
        merge_entry(merged, pg_bm25s, dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
