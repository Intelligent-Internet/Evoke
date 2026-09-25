#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


OFFICIAL_ORDER = [
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


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def merge_results(primary: dict, overlays: list[dict]) -> dict:
    merged = dict(primary)
    results = dict(merged.get('results', {}))

    for overlay in overlays:
        for dataset, entry in overlay.get('results', {}).items():
            results[dataset] = entry

    for entry in results.values():
        query = (
            entry.get('evoke_ids', {})
            .get('query', {})
        )
        if query.get('qps') is not None:
            entry.pop('error', None)
            entry.pop('traceback', None)

    ordered = OrderedDict()
    for dataset in OFFICIAL_ORDER:
        if dataset in results:
            ordered[dataset] = results[dataset]
    for dataset in sorted(results):
        if dataset not in ordered:
            ordered[dataset] = results[dataset]

    merged['results'] = ordered
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Finalize official BEIR benchmark results.'
    )
    parser.add_argument(
        'primary',
        type=Path,
        help='Primary benchmark JSON file.',
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Output path for the finalized JSON file.',
    )
    parser.add_argument(
        '--overlay',
        action='append',
        default=[],
        type=Path,
        help='Overlay result JSON files whose dataset entries override primary.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary = load_payload(args.primary)
    overlays = [load_payload(path) for path in args.overlay]
    merged = merge_results(primary, overlays)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2),
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
