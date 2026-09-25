#!/usr/bin/env python3
"""Gate one P2 runtime variant against its native exact baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


QUALITY_METRICS = (
    'ndcg_at_10',
    'map_at_100',
    'recall_at_100',
    'mrr_at_20',
    'candidate_upper_bound',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--variant', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-md', required=True, type=Path)
    parser.add_argument('--max-metric-drop', type=float, default=0.001)
    parser.add_argument(
        '--required-resource-improvement',
        type=float,
        default=20.0,
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        'bytes': path.stat().st_size,
        'path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def one_method(payload: dict[str, Any], *, source: Path) -> dict[str, Any]:
    methods = payload.get('methods')
    if not isinstance(methods, list) or len(methods) != 1:
        raise ValueError(f'{source}: expected exactly one evaluated method')
    method = methods[0]
    if not isinstance(method, dict):
        raise ValueError(f'{source}: method is not an object')
    summary = method.get('summary')
    metadata = method.get('metadata')
    if not isinstance(summary, dict) or not isinstance(metadata, dict):
        raise ValueError(f'{source}: method summary or metadata is missing')
    return method


def input_signature(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('native result has no input metadata')
    return {
        'candidate_k': payload.get('candidate_k'),
        'document_count': payload.get('document_count'),
        'qrels_sha256': (inputs.get('qrels') or {}).get('sha256'),
        'queries_sha256': (inputs.get('queries') or {}).get('sha256'),
        'qrels_query_count': payload.get('qrels_query_count'),
        'route': payload.get('route'),
        'schema': payload.get('schema'),
        'table': payload.get('table'),
    }


def improvement_percent(baseline: float, variant: float) -> float:
    if baseline <= 0.0:
        raise ValueError('baseline resource measurement must be positive')
    return (baseline - variant) / baseline * 100.0


def compare(
    baseline_path: Path,
    variant_path: Path,
    *,
    max_metric_drop: float,
    required_resource_improvement: float,
) -> dict[str, Any]:
    if max_metric_drop < 0.0:
        raise ValueError('max metric drop cannot be negative')
    if required_resource_improvement < 0.0:
        raise ValueError('required resource improvement cannot be negative')

    baseline = read_json(baseline_path)
    variant = read_json(variant_path)
    signature = input_signature(baseline)
    if signature != input_signature(variant):
        raise ValueError('baseline and variant evaluation surfaces differ')
    if signature['route'] != 'evoke_query':
        raise ValueError('canary must use the native evoke_query route')

    baseline_method = one_method(baseline, source=baseline_path)
    variant_method = one_method(variant, source=variant_path)
    baseline_summary = baseline_method['summary']
    variant_summary = variant_method['summary']
    quality_delta = {
        key: float(variant_summary[key]) - float(baseline_summary[key])
        for key in QUALITY_METRICS
    }
    quality_passed = all(
        delta >= -max_metric_drop
        for delta in quality_delta.values()
    )

    baseline_metadata = baseline_method['metadata']
    variant_metadata = variant_method['metadata']
    resource_improvement = {
        'index_bytes': improvement_percent(
            float(baseline_metadata['index_size_bytes']),
            float(variant_metadata['index_size_bytes']),
        ),
        'latency_ms_p50': improvement_percent(
            float(baseline_summary['latency_ms_p50']),
            float(variant_summary['latency_ms_p50']),
        ),
        'latency_ms_p95': improvement_percent(
            float(baseline_summary['latency_ms_p95']),
            float(variant_summary['latency_ms_p95']),
        ),
    }
    storage_passed = (
        resource_improvement['index_bytes']
        >= required_resource_improvement
    )
    latency_passed = all(
        resource_improvement[key] >= required_resource_improvement
        for key in ('latency_ms_p50', 'latency_ms_p95')
    )
    resource_passed = storage_passed or latency_passed

    return {
        'baseline': {
            'label': baseline_method.get('label'),
            'metadata': file_metadata(baseline_path),
            'model': baseline_metadata.get(
                'index_options',
                {},
            ).get('model'),
            'summary': baseline_summary,
        },
        'gate': {
            'latency_passed': latency_passed,
            'max_metric_drop': max_metric_drop,
            'passed': quality_passed and resource_passed,
            'quality_passed': quality_passed,
            'required_resource_improvement_percent': (
                required_resource_improvement
            ),
            'resource_passed': resource_passed,
            'storage_passed': storage_passed,
        },
        'quality_delta': quality_delta,
        'resource_improvement_percent': resource_improvement,
        'route': 'evoke_query',
        'surface_signature': signature,
        'variant': {
            'label': variant_method.get('label'),
            'metadata': file_metadata(variant_path),
            'model': variant_metadata.get(
                'index_options',
                {},
            ).get('model'),
            'summary': variant_summary,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload['baseline']['summary']
    variant = payload['variant']['summary']
    delta = payload['quality_delta']
    resource = payload['resource_improvement_percent']
    gate = payload['gate']
    lines = [
        '# P2 Runtime Variant Canary',
        '',
        f'- Native route: `{payload["route"]}`',
        f'- Passed: `{str(gate["passed"]).lower()}`',
        f'- Maximum metric drop: `{gate["max_metric_drop"]:.6f}`',
        '- Required resource improvement: '
        f'`{gate["required_resource_improvement_percent"]:.1f}%`',
        '',
        '| Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB | '
        'p50 ms | p95 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        f'| Baseline | {baseline["ndcg_at_10"]:.6f} | '
        f'{baseline["map_at_100"]:.6f} | '
        f'{baseline["recall_at_100"]:.6f} | '
        f'{baseline["mrr_at_20"]:.6f} | '
        f'{baseline["candidate_upper_bound"]:.6f} | '
        f'{baseline["latency_ms_p50"]:.3f} | '
        f'{baseline["latency_ms_p95"]:.3f} |',
        f'| Variant | {variant["ndcg_at_10"]:.6f} | '
        f'{variant["map_at_100"]:.6f} | '
        f'{variant["recall_at_100"]:.6f} | '
        f'{variant["mrr_at_20"]:.6f} | '
        f'{variant["candidate_upper_bound"]:.6f} | '
        f'{variant["latency_ms_p50"]:.3f} | '
        f'{variant["latency_ms_p95"]:.3f} |',
        f'| Delta | {delta["ndcg_at_10"]:+.6f} | '
        f'{delta["map_at_100"]:+.6f} | '
        f'{delta["recall_at_100"]:+.6f} | '
        f'{delta["mrr_at_20"]:+.6f} | '
        f'{delta["candidate_upper_bound"]:+.6f} | - | - |',
        '',
        '| Resource | Improvement | Gate |',
        '| --- | ---: | --- |',
        f'| Index bytes | {resource["index_bytes"]:.3f}% | '
        f'`{str(gate["storage_passed"]).lower()}` |',
        f'| Latency p50 | {resource["latency_ms_p50"]:.3f}% | - |',
        f'| Latency p95 | {resource["latency_ms_p95"]:.3f}% | '
        f'`{str(gate["latency_passed"]).lower()}` |',
    ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    payload = compare(
        args.baseline,
        args.variant,
        max_metric_drop=args.max_metric_drop,
        required_resource_improvement=args.required_resource_improvement,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(payload),
        encoding='utf-8',
    )
    print(json.dumps({
        'gate': payload['gate'],
        'quality_delta': payload['quality_delta'],
        'resource_improvement_percent': payload[
            'resource_improvement_percent'
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload['gate']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
