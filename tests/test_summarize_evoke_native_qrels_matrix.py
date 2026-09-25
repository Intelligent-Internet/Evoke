from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SPEC = importlib.util.spec_from_file_location(
    'summarize_evoke_native_qrels_matrix',
    SCRIPTS / 'summarize_evoke_native_qrels_matrix.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def metrics(value: float) -> dict[str, float]:
    return {key: value for key in MODULE.METRICS}


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
    return path


def old_result(path: Path, dataset: str, p2_value: float) -> Path:
    return write_json(path, {
        'candidate_k': 1000,
        'dataset': dataset,
        'document_count': 10,
        'methods': {
            'bm25': {
                **metrics(0.5),
                'build_seconds': 1.0,
                'index_bytes': 100,
                'latency_ms_p50': 2.0,
                'latency_ms_p95': 3.0,
            },
            'p2': {
                **metrics(p2_value),
                'build_seconds': 4.0,
                'index_bytes': 200,
                'latency_ms_p50': 5.0,
                'latency_ms_p95': 6.0,
                'model_id': 'frozen-p2',
            },
        },
        'query_count': 2,
        'route': 'evoke_query',
    })


def new_result(path: Path, dataset: str, model: str) -> Path:
    methods = []
    for label, value, size in (
        ('BM25', 0.4, 300),
        ('P2.2', 0.8, 400),
    ):
        methods.append({
            'label': label,
            'metadata': {
                'index_options': {
                    'model': model if label == 'P2.2' else None,
                },
                'index_size_bytes': size,
            },
            'summary': {
                **metrics(value),
                'latency_ms_p50': 7.0,
                'latency_ms_p95': 8.0,
            },
        })
    return write_json(path, {
        'candidate_k': 1000,
        'dataset': dataset,
        'document_count': 20,
        'methods': methods,
        'qrels_query_count': 3,
        'route': 'evoke_query',
    })


def test_build_matrix_normalizes_both_result_formats(tmp_path: Path) -> None:
    first = old_result(tmp_path / 'first.json', 'first', 0.6)
    second = new_result(tmp_path / 'second.json', 'second', 'frozen-p2')
    build = write_json(tmp_path / 'build.json', {
        'build_seconds': {'BM25': 9.0, 'P2.2': 10.0},
        'dataset': 'second',
    })

    payload = MODULE.build_matrix(
        [first, second],
        build_paths=[build],
    )

    assert payload['model_id'] == 'frozen-p2'
    assert payload['all_quality_deltas_positive'] is True
    assert payload['datasets'][1]['methods']['P2.2']['build_seconds'] == 10.0
    assert payload['macro']['methods']['BM25']['recall_at_100'] == 0.45
    assert payload['macro']['methods']['P2.2']['recall_at_100'] == 0.7
    assert '| P2.2 - BM25 |' in MODULE.render_markdown(payload)


def test_build_matrix_matches_official_result_to_base_build(
    tmp_path: Path,
) -> None:
    result = old_result(
        tmp_path / 'result.json',
        'trec-covid-official',
        0.6,
    )
    build = write_json(tmp_path / 'build.json', {
        'build_seconds': {'BM25': 9.0, 'P2.2': 10.0},
        'dataset': 'trec-covid',
    })

    payload = MODULE.build_matrix([result], build_paths=[build])

    methods = payload['datasets'][0]['methods']
    assert methods['BM25']['build_seconds'] == 9.0
    assert methods['P2.2']['build_seconds'] == 10.0


def test_build_matrix_rejects_model_drift(tmp_path: Path) -> None:
    first = old_result(tmp_path / 'first.json', 'first', 0.6)
    second = new_result(tmp_path / 'second.json', 'second', 'other-model')

    try:
        MODULE.build_matrix([first, second], build_paths=[])
    except ValueError as exc:
        assert 'one explicit model' in str(exc)
    else:
        raise AssertionError('cross-row model drift must fail closed')
