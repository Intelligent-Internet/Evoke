#!/usr/bin/env python3
"""Normalize native evoke qrels results into one auditable matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    'ndcg_at_10',
    'map_at_100',
    'recall_at_100',
    'mrr_at_20',
    'candidate_upper_bound',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a cross-dataset native evoke qrels matrix.',
    )
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--build-json', action='append', type=Path, default=[])
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    return parser.parse_args()


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'bytes': path.stat().st_size,
        'path': str(path),
        'sha256': digest.hexdigest(),
    }


def method_name(label: str) -> str:
    normalized = label.lower().replace('.', '').replace('-', '')
    if normalized == 'bm25':
        return 'BM25'
    if normalized in {'p2', 'p22'}:
        return 'P2.2'
    raise ValueError(f'unsupported method label: {label}')


def metric_values(source: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in METRICS if key not in source]
    if missing:
        raise ValueError(f'method summary is missing metrics: {missing}')
    return {key: float(source[key]) for key in METRICS}


def normalize_method(
    label: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    summary = source.get('summary', source)
    metadata = source.get('metadata', {})
    options = metadata.get('index_options', {})
    return {
        'build_seconds': source.get('build_seconds'),
        'index_bytes': source.get(
            'index_bytes',
            metadata.get('index_size_bytes'),
        ),
        'latency_ms_p50': summary.get('latency_ms_p50'),
        'latency_ms_p95': summary.get('latency_ms_p95'),
        'metrics': metric_values(summary),
        'model_id': source.get('model_id', options.get('model')),
        'name': method_name(label),
    }


def normalize_methods(raw: Any) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = (
            (str(item.get('label', '')), item)
            for item in raw
            if isinstance(item, dict)
        )
    else:
        raise ValueError('methods must be an object or array')
    for label, source in iterator:
        if not isinstance(source, dict):
            raise ValueError(f'invalid method payload: {label}')
        method = normalize_method(str(label), source)
        name = str(method['name'])
        if name in methods:
            raise ValueError(f'duplicate normalized method: {name}')
        methods[name] = method
    if set(methods) != {'BM25', 'P2.2'}:
        raise ValueError('each result must contain BM25 and P2.2')
    return methods


def load_build_times(paths: list[Path]) -> dict[str, dict[str, float]]:
    builds: dict[str, dict[str, float]] = {}
    for path in paths:
        raw = json.loads(path.read_text(encoding='utf-8'))
        dataset = str(raw.get('dataset', ''))
        values = raw.get('build_seconds')
        if not dataset or not isinstance(values, dict):
            raise ValueError(f'invalid build metadata: {path}')
        if dataset in builds:
            raise ValueError(f'duplicate build metadata: {dataset}')
        builds[dataset] = {
            method_name(str(label)): float(value)
            for label, value in values.items()
        }
    return builds


def normalize_result(
    path: Path,
    *,
    build_times: dict[str, dict[str, float]],
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.get('route') != 'evoke_query':
        raise ValueError(f'{path}: result does not use evoke_query')
    dataset = str(raw.get('dataset', ''))
    if not dataset:
        raise ValueError(f'{path}: missing dataset')
    methods = normalize_methods(raw.get('methods'))
    build_dataset = dataset
    if build_dataset not in build_times and dataset.endswith('-official'):
        build_dataset = dataset.removesuffix('-official')
    for name, seconds in build_times.get(build_dataset, {}).items():
        methods[name]['build_seconds'] = seconds
    bm25 = methods['BM25']['metrics']
    p2 = methods['P2.2']['metrics']
    return {
        'candidate_k': int(raw['candidate_k']),
        'dataset': dataset,
        'document_count': int(raw['document_count']),
        'methods': methods,
        'p2_minus_bm25': {
            key: p2[key] - bm25[key]
            for key in METRICS
        },
        'provenance': file_metadata(path),
        'query_count': int(
            raw.get('query_count', raw.get('qrels_query_count', 0))
        ),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_matrix(
    paths: list[Path],
    *,
    build_paths: list[Path],
) -> dict[str, Any]:
    builds = load_build_times(build_paths)
    datasets = [
        normalize_result(path, build_times=builds)
        for path in paths
    ]
    names = [row['dataset'] for row in datasets]
    if len(names) != len(set(names)):
        raise ValueError('dataset results must be unique')
    candidate_ks = {row['candidate_k'] for row in datasets}
    if len(candidate_ks) != 1:
        raise ValueError('candidate_k must match across datasets')
    model_ids = {
        row['methods']['P2.2']['model_id']
        for row in datasets
    }
    if None in model_ids or len(model_ids) != 1:
        raise ValueError('all P2.2 rows must use one explicit model')

    macro_methods = {
        name: {
            key: mean([
                row['methods'][name]['metrics'][key]
                for row in datasets
            ])
            for key in METRICS
        }
        for name in ('BM25', 'P2.2')
    }
    macro_delta = {
        key: macro_methods['P2.2'][key] - macro_methods['BM25'][key]
        for key in METRICS
    }
    return {
        'all_quality_deltas_positive': all(
            row['p2_minus_bm25'][key] > 0.0
            for row in datasets
            for key in METRICS
        ),
        'candidate_k': candidate_ks.pop(),
        'datasets': datasets,
        'macro': {
            'methods': macro_methods,
            'p2_minus_bm25': macro_delta,
        },
        'model_id': model_ids.pop(),
        'route': 'evoke_query',
        'surface': 'evoke_native_qrels_matrix_v1',
    }


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return '-'
    return f'{float(value):.{digits}f}'


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Evoke Native Qrels Matrix',
        '',
        f'- Route: `{payload["route"]}`',
        f'- Model: `{payload["model_id"]}`',
        f'- Candidate k: `{payload["candidate_k"]}`',
        '',
        '| Dataset | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | '
        'CUB | p50 ms | p95 ms | Index bytes | Build s |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
        '---: | ---: |',
    ]
    for row in payload['datasets']:
        for name in ('BM25', 'P2.2'):
            method = row['methods'][name]
            metrics = method['metrics']
            lines.append(
                f'| {row["dataset"]} | {name} | '
                f'{format_value(metrics["ndcg_at_10"])} | '
                f'{format_value(metrics["map_at_100"])} | '
                f'{format_value(metrics["recall_at_100"])} | '
                f'{format_value(metrics["mrr_at_20"])} | '
                f'{format_value(metrics["candidate_upper_bound"])} | '
                f'{format_value(method["latency_ms_p50"], 3)} | '
                f'{format_value(method["latency_ms_p95"], 3)} | '
                f'{method["index_bytes"] or "-"} | '
                f'{format_value(method["build_seconds"], 3)} |'
            )
        delta = row['p2_minus_bm25']
        lines.append(
            f'| {row["dataset"]} | P2.2 - BM25 | '
            f'{delta["ndcg_at_10"]:+.6f} | '
            f'{delta["map_at_100"]:+.6f} | '
            f'{delta["recall_at_100"]:+.6f} | '
            f'{delta["mrr_at_20"]:+.6f} | '
            f'{delta["candidate_upper_bound"]:+.6f} | - | - | - | - |'
        )

    lines.extend(['', '## Macro', ''])
    lines.append('| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name in ('BM25', 'P2.2'):
        metrics = payload['macro']['methods'][name]
        lines.append(
            f'| {name} | {metrics["ndcg_at_10"]:.6f} | '
            f'{metrics["map_at_100"]:.6f} | '
            f'{metrics["recall_at_100"]:.6f} | '
            f'{metrics["mrr_at_20"]:.6f} | '
            f'{metrics["candidate_upper_bound"]:.6f} |'
        )
    delta = payload['macro']['p2_minus_bm25']
    lines.append(
        f'| P2.2 - BM25 | {delta["ndcg_at_10"]:+.6f} | '
        f'{delta["map_at_100"]:+.6f} | '
        f'{delta["recall_at_100"]:+.6f} | '
        f'{delta["mrr_at_20"]:+.6f} | '
        f'{delta["candidate_upper_bound"]:+.6f} |'
    )
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    payload = build_matrix(args.input, build_paths=args.build_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'all_quality_deltas_positive': payload[
            'all_quality_deltas_positive'
        ],
        'datasets': [row['dataset'] for row in payload['datasets']],
        'macro': payload['macro'],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
