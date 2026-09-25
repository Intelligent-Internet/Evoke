from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_p2_runtime_variant import compare


def native_result(
    *,
    model: str,
    index_bytes: int,
    quality: float,
    p50: float,
    p95: float,
) -> dict[str, object]:
    summary = {
        'candidate_upper_bound': quality,
        'latency_ms_p50': p50,
        'latency_ms_p95': p95,
        'map_at_100': quality,
        'mrr_at_20': quality,
        'ndcg_at_10': quality,
        'recall_at_100': quality,
    }
    return {
        'candidate_k': 1000,
        'document_count': 100,
        'inputs': {
            'qrels': {'sha256': 'qrels'},
            'queries': {'sha256': 'queries'},
        },
        'methods': [{
            'label': model,
            'metadata': {
                'index_options': {'model': model},
                'index_size_bytes': index_bytes,
            },
            'summary': summary,
        }],
        'qrels_query_count': 10,
        'route': 'evoke_query',
        'schema': 'bench',
        'table': 'docs',
    }


def write_result(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


def test_storage_gain_and_quality_floor_pass(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline.json'
    variant = tmp_path / 'variant.json'
    write_result(
        baseline,
        native_result(
            model='baseline',
            index_bytes=100,
            quality=0.8,
            p50=10.0,
            p95=20.0,
        ),
    )
    write_result(
        variant,
        native_result(
            model='variant',
            index_bytes=75,
            quality=0.7995,
            p50=9.5,
            p95=19.0,
        ),
    )

    result = compare(
        baseline,
        variant,
        max_metric_drop=0.001,
        required_resource_improvement=20.0,
    )

    assert result['gate']['passed'] is True
    assert result['gate']['storage_passed'] is True
    assert result['gate']['latency_passed'] is False


def test_quality_drop_rejects_resource_gain(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline.json'
    variant = tmp_path / 'variant.json'
    write_result(
        baseline,
        native_result(
            model='baseline',
            index_bytes=100,
            quality=0.8,
            p50=10.0,
            p95=20.0,
        ),
    )
    write_result(
        variant,
        native_result(
            model='variant',
            index_bytes=50,
            quality=0.79,
            p50=5.0,
            p95=10.0,
        ),
    )

    result = compare(
        baseline,
        variant,
        max_metric_drop=0.001,
        required_resource_improvement=20.0,
    )

    assert result['gate']['resource_passed'] is True
    assert result['gate']['quality_passed'] is False
    assert result['gate']['passed'] is False


def test_surface_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline.json'
    variant = tmp_path / 'variant.json'
    baseline_value = native_result(
        model='baseline',
        index_bytes=100,
        quality=0.8,
        p50=10.0,
        p95=20.0,
    )
    variant_value = native_result(
        model='variant',
        index_bytes=75,
        quality=0.8,
        p50=8.0,
        p95=15.0,
    )
    variant_value['document_count'] = 99
    write_result(baseline, baseline_value)
    write_result(variant, variant_value)

    with pytest.raises(ValueError, match='surfaces differ'):
        compare(
            baseline,
            variant,
            max_metric_drop=0.001,
            required_resource_improvement=20.0,
        )
