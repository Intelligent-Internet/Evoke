"""CPU-only tests for NG84 selection, complete positives and admission rules."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'ng84_teacher', Path(__file__).resolve().parents[1] / 'scripts/ng84_teacher.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_train_only_selection_is_balanced_stable_and_disjoint():
    queries = [dict(query_id=f'{d}-{i}', subset=d, query=str(i))
               for d in MODULE.DOMAINS for i in range(5)]
    split = dict(train=[i for i in range(15) if i % 5 != 0], diagnostic=[0, 5, 10])
    selected = MODULE.select(queries, split, 3)
    assert selected == MODULE.select(queries, split, 3)
    assert len(selected) == 9 and not set(selected) & set(split['diagnostic'])
    assert [queries[i]['subset'] for i in selected] == list(MODULE.DOMAINS) * 3


def test_selection_rejects_overlap_and_duplicates():
    with pytest.raises(ValueError):
        MODULE.select([], dict(train=[0], diagnostic=[0]), 1)
    with pytest.raises(ValueError):
        MODULE.select([], dict(train=[0, 0], diagnostic=[]), 1)


def test_metrics_retains_multiple_positives_and_stable_ties():
    result = MODULE.metrics([9, 4, 2], [1., 1., 1.], [9, 2])
    assert result['positive_ranks'] == [1, 3]
    assert result['recall10'] == 1
    assert result['ndcg10'] < 1
    assert MODULE.metrics([2, 9, 4], [1., 1., 1.], [9, 2]) == result


@pytest.mark.parametrize('ids,scores,gold', [
    ([1, 1], [1., 2.], [1]), ([1, 2], [float('nan'), 1.], [1]),
    ([1, 2], [1., 2.], [3]), ([1, 2], [1., 2.], []),
])
def test_metrics_rejects_invalid_surface(ids, scores, gold):
    with pytest.raises(ValueError):
        MODULE.metrics(ids, scores, gold)


def results(deltas):
    return [dict(query_id=f'{d}-{i}', domain=d,
                 pplx=dict(ndcg10=.5), rankt5=dict(ndcg10=.5 + delta))
            for d, delta in zip(MODULE.DOMAINS, deltas, strict=True) for i in range(128)]


def test_admission_is_overall_not_every_domain_win():
    report = MODULE.summarize(results([.05, .05, -.02]))
    assert report['decision'] == 'admit_conditional_training'
    assert report['macro_delta'] == pytest.approx(.08 / 3)
    assert report['independent_holdout'] is False


def test_no_signal_and_incomplete_results_do_not_advance():
    assert MODULE.summarize(results([-.02] * 3))['decision'] == 'no_useful_signal_on_this_surface'
    assert MODULE.summarize(results([.001] * 3))['decision'] == 'inconclusive_do_not_auto_launch'
    with pytest.raises(ValueError):
        MODULE.summarize(results([.05] * 3)[:-1])
    with pytest.raises(ValueError):
        MODULE.summarize(results([.05] * 3) + [dict(query_id='extra', domain='unknown')])


def test_verify_rejects_tampered_config_before_model_access(tmp_path):
    MODULE.write(tmp_path / 'inputs.json', {})
    MODULE.write(tmp_path / 'config.json', dict(MODULE.config(), optimizer_updates=1))
    with pytest.raises(ValueError, match='config changed'):
        MODULE.verify(tmp_path)


def test_supervisor_rejects_occupied_gpu_without_output(tmp_path, monkeypatch):
    run = tmp_path / 'NG-0084' / 'admission-v1'
    run.mkdir(parents=True)
    monkeypatch.setattr(MODULE, 'verify', lambda _: None)
    monkeypatch.setattr(MODULE, 'gpu_state', lambda _: (1000, 10))
    with pytest.raises(ValueError, match='GPU occupied'):
        MODULE.supervise(run, 3)
    assert not (run / 'teacher').exists()


def test_runtime_contract_does_not_train_or_open_test():
    spec = MODULE.config()
    assert spec['optimizer_updates'] == 0
    assert spec['locked_test_access'] is False
    assert spec['training_auto_launch'] is False
    assert spec['precision'] == 'float32'
    assert spec['max_seconds'] == 10800


def test_loading_report_accepts_transformers_sets_and_stays_json_serializable(tmp_path):
    report = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=[], error_msgs=[])
    receipt = MODULE.loading_receipt(report)
    MODULE.write(tmp_path / 'runtime.json', receipt)
    assert MODULE.read(tmp_path / 'runtime.json') == receipt
    for key in report:
        with pytest.raises(ValueError, match='incomplete'):
            MODULE.loading_receipt(dict(report, **{key: ['broken']}))
