from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SPEC = importlib.util.spec_from_file_location(
    'load_evoke_native_qrels_corpus',
    SCRIPTS / 'load_evoke_native_qrels_corpus.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_corpus_rows_accepts_product_staging_shape(tmp_path: Path) -> None:
    path = tmp_path / 'documents.jsonl'
    path.write_text(
        '\n'.join([
            json.dumps({'id': 'd1', 'text': 'first', 'embedding': [1]}),
            json.dumps({'doc_id': 'd2', 'text_content': 'second'}),
        ]) + '\n',
        encoding='utf-8',
    )
    assert list(MODULE.corpus_rows(path)) == [
        ('d1', 'first'),
        ('d2', 'second'),
    ]


def test_corpus_rows_rejects_missing_text(tmp_path: Path) -> None:
    path = tmp_path / 'documents.jsonl'
    path.write_text('{"id":"d1"}\n', encoding='utf-8')
    try:
        list(MODULE.corpus_rows(path))
    except ValueError as exc:
        assert 'missing text' in str(exc)
    else:
        raise AssertionError('missing text must fail')


def test_validate_row_count_accepts_expected_count() -> None:
    MODULE.validate_row_count(
        loaded=25_657,
        table_count=25_657,
        expected_rows=25_657,
    )


def test_validate_row_count_rejects_partial_corpus() -> None:
    try:
        MODULE.validate_row_count(
            loaded=2_000,
            table_count=2_000,
            expected_rows=25_657,
        )
    except RuntimeError as exc:
        assert 'expected=25657' in str(exc)
        assert 'loaded=2000' in str(exc)
    else:
        raise AssertionError('partial corpus must fail')


def test_validate_row_count_rejects_copy_table_mismatch() -> None:
    try:
        MODULE.validate_row_count(
            loaded=99,
            table_count=98,
            expected_rows=None,
        )
    except RuntimeError as exc:
        assert 'copied=99' in str(exc)
        assert 'table=98' in str(exc)
    else:
        raise AssertionError('copy/table mismatch must fail')
