from pathlib import Path

import pytest

from scripts.build_evoke_beir15_pg import (
    dataset_plan,
    dataset_plan_from_files,
    selected_datasets,
    vector_lists,
)


def test_dataset_plan_rejects_unknown_without_file_count() -> None:
    with pytest.raises(ValueError, match='unknown dataset'):
        dataset_plan('Touche2020Retrieval.v3')


def test_dataset_plan_allows_custom_dataset_with_file_count() -> None:
    plan = dataset_plan('Touche2020Retrieval.v3', expected_docs=303_732)

    assert plan.name == 'Touche2020Retrieval.v3'
    assert plan.table_name == 'docs_touche2020retrieval_v3'
    assert plan.expected_docs == 303_732
    assert plan.vector_lists is not None
    assert plan.vector_probes is not None


def test_dataset_plan_from_files_supports_custom_dataset(tmp_path: Path) -> None:
    root = tmp_path / 'ArguAna'
    root.mkdir()
    (root / 'documents.jsonl').write_text('{}\n{}\n{}\n', encoding='utf-8')

    plan = dataset_plan_from_files(tmp_path, 'ArguAna')

    assert plan.name == 'ArguAna'
    assert plan.table_name == 'docs_arguana'
    assert plan.expected_docs == 3
    assert plan.vector_lists is None
    assert plan.vector_probes is None


def test_vector_lists_keeps_small_corpora_unpartitioned() -> None:
    assert vector_lists(99_999) is None
    assert vector_lists(100_000) == 200


def test_selected_datasets_keeps_beir_guard_by_default() -> None:
    with pytest.raises(SystemExit, match='unknown dataset'):
        selected_datasets(['ArguAna'])


def test_selected_datasets_allows_custom_when_requested() -> None:
    assert selected_datasets(
        ['ArguAna', 'Touche2020Retrieval.v3'],
        allow_custom=True,
    ) == ['ArguAna', 'Touche2020Retrieval.v3']
