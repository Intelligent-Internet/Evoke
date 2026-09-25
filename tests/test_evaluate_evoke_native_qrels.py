from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    'evaluate_evoke_native_qrels',
    SCRIPTS / 'evaluate_evoke_native_qrels.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_index_specs_rejects_duplicates() -> None:
    assert MODULE.parse_index_specs([
        'bm25=bench.docs_bm25_idx',
        'p2=bench.docs_p2_idx',
    ]) == [
        ('bm25', 'bench.docs_bm25_idx'),
        ('p2', 'bench.docs_p2_idx'),
    ]

    try:
        MODULE.parse_index_specs(['p2=a', 'p2=b'])
    except ValueError as exc:
        assert 'duplicate index label' in str(exc)
    else:
        raise AssertionError('duplicate labels must fail')


def test_parse_args_accepts_explicit_id_column(monkeypatch: object) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'evaluate_evoke_native_qrels.py',
            '--dataset',
            'scifact',
            '--schema',
            'bench',
            '--table',
            'docs',
            '--id-column',
            'id',
            '--index',
            'p2=bench.docs_idx',
            '--queries-jsonl',
            'queries.jsonl',
            '--qrels-json',
            'qrels.json',
        ],
    )
    assert MODULE.parse_args().id_column == 'id'


def test_load_queries_filters_to_qrels_without_reading_embeddings(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'queries.jsonl'
    path.write_text(
        '\n'.join([
            json.dumps({'id': 'q1', 'text': 'one', 'embedding': [1, 2]}),
            json.dumps({'id': 'q2', 'text': 'two', 'embedding': [3, 4]}),
        ]) + '\n',
        encoding='utf-8',
    )
    assert MODULE.load_queries(
        path,
        qrels={'q2': {'d2': 1.0}},
        limit=0,
    ) == [{'query_id': 'q2', 'text': 'two'}]


def test_metrics_match_existing_beir_contract() -> None:
    metrics = MODULE.metrics_for_query(
        ['d0', 'd1', 'd2', 'd3'],
        {'d1': 1.0, 'd3': 1.0},
        candidate_k=3,
    )
    assert metrics['candidate_upper_bound'] == 0.5
    assert metrics['recall_at_100'] == 1.0
    assert metrics['mrr_at_20'] == 0.5
    assert metrics['map_at_100'] == 0.5


def test_map_at_100_uses_cutoff_denominator() -> None:
    qrels = {f'doc-{offset}': 1.0 for offset in range(150)}
    ranked_docs = [f'doc-{offset}' for offset in range(100)]

    metrics = MODULE.metrics_for_query(
        ranked_docs,
        qrels,
        candidate_k=100,
    )

    assert metrics['map_at_100'] == 1.0
    assert metrics['recall_at_100'] == 100 / 150


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert MODULE.percentile(values, 0.50) == 3.0
    assert MODULE.percentile(values, 0.95) == 100.0


def test_validate_expected_count_fails_closed() -> None:
    MODULE.validate_expected_count(
        label='documents',
        actual=10,
        expected=10,
    )

    try:
        MODULE.validate_expected_count(
            label='queries',
            actual=9,
            expected=10,
        )
    except RuntimeError as exc:
        assert 'expected queries count mismatch' in str(exc)
    else:
        raise AssertionError('partial query evaluation must fail closed')

    try:
        MODULE.validate_expected_count(
            label='documents',
            actual=0,
            expected=0,
        )
    except ValueError as exc:
        assert '--expected-documents must be positive' in str(exc)
    else:
        raise AssertionError('non-positive expected counts must fail')


def test_validate_index_metadata_requires_product_owned_generation() -> None:
    valid = {
        'index_name': 'bench.docs_idx',
        'index_options': {'payload_owner': 'index_relation'},
        'index_status': {
            'generation': {'atomic': True, 'valid': True},
            'query_ready': True,
        },
    }
    MODULE.validate_index_metadata(valid)

    invalid = {
        **valid,
        'index_options': {'payload_owner': 'external_sidecar'},
    }
    try:
        MODULE.validate_index_metadata(invalid)
    except RuntimeError as exc:
        assert 'payload_owner is not index_relation' in str(exc)
    else:
        raise AssertionError('sidecar-owned index must fail the product gate')
