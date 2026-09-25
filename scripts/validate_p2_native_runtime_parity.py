#!/usr/bin/env python3
"""Validate native P2 text compilation against the frozen Python oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', default='dbname=postgres')
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--expected-jsonl', required=True, type=Path)
    parser.add_argument('--absolute-tolerance', type=float, default=2.0e-4)
    parser.add_argument('--relative-tolerance', type=float, default=2.0e-2)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected object')
            if not isinstance(row.get('text'), str) or not row['text']:
                raise ValueError(f'{path}:{line_no}: invalid text')
            atom_ids = row.get('atom_ids')
            weights = row.get('atom_impacts')
            if not isinstance(atom_ids, list) or not isinstance(weights, list):
                raise ValueError(f'{path}:{line_no}: missing atom arrays')
            if len(atom_ids) != len(weights):
                raise ValueError(f'{path}:{line_no}: atom length mismatch')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no rows')
    return rows


def compare_weights(
    actual: list[float],
    expected: list[float],
) -> tuple[float, float]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for actual_value, expected_value in zip(actual, expected, strict=True):
        absolute = abs(float(actual_value) - float(expected_value))
        relative = absolute / max(abs(float(expected_value)), 1.0e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    manifest = json.loads(
        (args.model_path / 'manifest.json').read_text(encoding='utf-8')
    )
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            '--model-path must use the current Evoke model contract'
        )
    rows = read_rows(args.expected_jsonl)
    atom_mismatches: list[dict[str, Any]] = []
    weight_mismatches: list[dict[str, Any]] = []
    native_latencies: list[float] = []
    wall_latencies: list[float] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0

    with psycopg.connect(args.dsn) as conn:
        conn.execute(
            'SELECT set_config(%s, %s, false)',
            ('statement_timeout', '0'),
        )
        for row in rows:
            started = time.perf_counter()
            payload = conn.execute(
                'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                (str(args.model_path), row['text']),
            ).fetchone()[0]
            wall_latencies.append((time.perf_counter() - started) * 1000.0)
            native_latencies.append(float(payload['latency_ms']))
            if payload['atoms'] != row['atom_ids']:
                atom_mismatches.append({
                    'actual_atoms': payload['atoms'],
                    'expected_atoms': row['atom_ids'],
                    'id': row.get('id'),
                    'text': row['text'],
                })
                if len(atom_mismatches) >= 10:
                    break
                continue
            absolute, relative = compare_weights(
                payload['weights'],
                row['atom_impacts'],
            )
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if (
                absolute > args.absolute_tolerance
                and relative > args.relative_tolerance
            ):
                weight_mismatches.append({
                    'id': row.get('id'),
                    'maximum_absolute': absolute,
                    'maximum_relative': relative,
                    'text': row['text'],
                })
                if len(weight_mismatches) >= 10:
                    break

    passed = not atom_mismatches and not weight_mismatches
    result = {
        'absolute_tolerance': args.absolute_tolerance,
        'atom_id_mismatches': atom_mismatches,
        'exact_atom_ids': not atom_mismatches,
        'maximum_absolute_weight_delta': maximum_absolute,
        'maximum_relative_weight_delta': maximum_relative,
        'model_path': str(args.model_path.resolve()),
        'native_latency_ms': {
            'maximum': max(native_latencies),
            'mean': statistics.fmean(native_latencies),
            'minimum': min(native_latencies),
            'p95': sorted(native_latencies)[
                min(len(native_latencies) - 1, int(len(native_latencies) * 0.95))
            ],
        },
        'passed': passed,
        'query_count': len(native_latencies),
        'relative_tolerance': args.relative_tolerance,
        'source': str(args.expected_jsonl.resolve()),
        'wall_latency_ms': {
            'maximum': max(wall_latencies),
            'mean': statistics.fmean(wall_latencies),
            'minimum': min(wall_latencies),
            'p95': sorted(wall_latencies)[
                min(len(wall_latencies) - 1, int(len(wall_latencies) * 0.95))
            ],
        },
        'weight_mismatches': weight_mismatches,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
