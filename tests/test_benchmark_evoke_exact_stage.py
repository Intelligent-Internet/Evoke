from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    'benchmark_evoke_exact_stage',
    SCRIPTS / 'benchmark_evoke_exact_stage.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_queries_reads_matched_surface(tmp_path: Path) -> None:
    path = tmp_path / 'queries.json'
    path.write_text(json.dumps({
        'rows': [
            {'source_id': 'd1', 'title': 'first title'},
            {'source_id': 'd2', 'title': 'second title'},
        ],
    }), encoding='utf-8')

    assert MODULE.load_queries(path) == [
        {'query_id': 'd1', 'text': 'first title'},
        {'query_id': 'd2', 'text': 'second title'},
    ]


def test_latency_summary_uses_inclusive_percentile() -> None:
    summary = MODULE.latency_summary([float(value) for value in range(1, 21)])

    assert summary['count'] == 20
    assert summary['p50_ms'] == 10.5
    assert summary['p95_ms'] == 19.05


def stage(
    *,
    p50: float,
    p95: float,
    doc_id: str = 'd1',
    score: float = 1.0,
) -> dict[str, object]:
    return {
        'latency': {'p50_ms': p50, 'p95_ms': p95},
        'ranked_queries': [{
            'doc_ids': [doc_id],
            'query_id': 'q1',
            'scores': [score],
        }],
    }


def test_compare_rankings_requires_quality_and_latency_gate() -> None:
    result = MODULE.compare_rankings(
        stage(p50=100.0, p95=120.0),
        stage(p50=70.0, p95=80.0, score=1.00005),
        score_tolerance=0.0001,
        required_improvement=20.0,
    )

    assert result['passed'] is True
    assert result['identity_mismatches'] == 0
    assert result['ranked_rows'] == 1

    harmed = MODULE.compare_rankings(
        stage(p50=100.0, p95=120.0),
        stage(p50=90.0, p95=100.0, doc_id='other'),
        score_tolerance=0.0001,
        required_improvement=20.0,
    )
    assert harmed['passed'] is False
    assert harmed['identity_mismatches'] == 1
