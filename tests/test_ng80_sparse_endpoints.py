from pathlib import Path
import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy import sparse


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_sparse_endpoints as endpoint


def test_sparse_dual_reductions_include_full_background_and_empty_query():
    rng = np.random.default_rng(80)
    matrix = sparse.random(150, 20, density=.2, random_state=rng).tocsc()
    query = sparse.csr_matrix(([.2, .5], ([0, 0], [3, 9])), shape=(1, 20))
    score, alternate, work = endpoint.sparse_scores(matrix, query)
    np.testing.assert_allclose(score, matrix.toarray() @ query.toarray().ravel())
    np.testing.assert_allclose(score, alternate, rtol=0, atol=1e-12)
    assert work == np.diff(matrix.indptr)[[3, 9]].sum()
    empty = endpoint.sparse_scores(matrix, sparse.csr_matrix((1, 20)))
    np.testing.assert_array_equal(empty[0], np.zeros(150))
    assert empty[2] == 0
    with pytest.raises(ValueError):
        endpoint.sparse_scores(matrix.tocsr(), query)


def test_full_query_normalization_and_head_replay():
    values = np.asarray([-.1, .3, 1., -.5])
    normalized = endpoint.normalize(values)
    np.testing.assert_allclose(normalized.mean(), 0, atol=1e-15)
    np.testing.assert_allclose(normalized.std(), 1)
    np.testing.assert_allclose(normalized[[3, 0]],
        endpoint.normalize_head(values[[3, 0]], endpoint.normalization_stats(values)))
    np.testing.assert_array_equal(endpoint.normalize(np.ones(5)), np.zeros(5))
    for bad in ([np.nan, 1.], [np.inf, 0.], [1.]):
        with pytest.raises(ValueError):
            endpoint.normalize(bad)


def test_calibration_ties_choose_less_lexical_without_diagnostic():
    rows = [dict(alpha=a, ndcg10=.4, recall100=.6) for a in endpoint.ALPHAS]
    assert endpoint.choose_alpha(rows) == 0
    rows[3]['ndcg10'] = .5
    assert endpoint.choose_alpha(rows) == .5
    rows[3]['ndcg10'] = .4
    rows[3]['recall100'] = .8
    assert endpoint.choose_alpha(rows) == .5
    with pytest.raises(ValueError):
        endpoint.choose_alpha(rows[::-1])


def test_selection_is_balanced_train_only_and_disjoint():
    queries, train, diagnostic = [], [], []
    for domain in endpoint.DOMAINS:
        for n in range(4608):
            index = len(queries)
            queries.append(dict(query_id=f'{domain}-{n}', subset=domain))
            (train if n < 4096 else diagnostic).append(index)
    result = endpoint.selection(queries, dict(train=train, diagnostic=diagnostic))
    assert len(result['calibration']) == 384
    assert not set(result['calibration']) & set(result['diagnostic'])
    with pytest.raises(ValueError):
        endpoint.selection(queries, dict(train=train, diagnostic=diagnostic[:-1] + [train[0]]))


def test_scientific_controls_and_no_optimizer_phase():
    assert endpoint.MODELS == ('initial', 'S', 'P')
    assert len(endpoint.PHASES) == len(set(endpoint.PHASES)) == 11
    assert endpoint.LANES['audit'] == ('audit-training',)
    assert endpoint.LANES['ranking'][-1] == 'review'
    assert not any(p.startswith('train-') for p in endpoint.PHASES)


def test_reused_phase_path_is_pinned_and_only_reads_successful_inputs(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    manifest = dict(reused_phases={phase: endpoint.REUSE_PARENT + '/' + phase
                                  for phase in endpoint.REUSE_PHASES})
    (run / 'inputs.json').write_text(json.dumps(manifest))
    for phase in endpoint.REUSE_PHASES:
        assert endpoint.phase_root(tmp_path, run, phase) == (
            tmp_path / endpoint.REUSE_PARENT / phase)
    assert endpoint.phase_root(tmp_path, run, 'prepare-lexical') == run / 'prepare-lexical'
    manifest['reused_phases']['encode-P'] = '../escape'
    (run / 'inputs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid reused'):
        endpoint.phase_root(tmp_path, run, 'encode-P')
    with pytest.raises(ValueError, match='unknown endpoint'):
        endpoint.phase_root(tmp_path, run, 'other')


def test_reuse_parent_rejects_changed_failed_evidence(tmp_path, monkeypatch):
    parent = tmp_path / endpoint.REUSE_PARENT
    parent.mkdir(parents=True)
    (parent / 'inputs.json').write_text('{}')
    monkeypatch.setattr(endpoint, 'REUSE_INPUT', endpoint.campaign.io.sha(parent / 'inputs.json'))
    monkeypatch.setattr(endpoint, 'verify', lambda *_: dict(dependencies={}))
    seen = []
    monkeypatch.setattr(endpoint.audit, 'sealed', lambda path: seen.append(path.name))
    for lane in ('audit', 'first', 'second', 'prepare'):
        value = (dict(status='failed_preserve_attempt', completed=[]) if lane == 'prepare'
                 else dict(status='all_phases_complete', completed=list(endpoint.LANES[lane])))
        (parent / f'{lane}-controller-exit.json').write_text(json.dumps(value))
    failed = parent / 'prepare-lexical'
    failed.mkdir()
    (failed / 'exit.json').write_text(json.dumps(dict(
        exit_code=-6, error=None, owned_group_closed=True)))
    (failed / 'complete.json').write_text(json.dumps(dict(passed=False, files={
        'exit.json': endpoint.campaign.io.sha(failed / 'exit.json')})))
    deps = endpoint.reuse_dependencies(tmp_path)
    assert set(seen) == set(endpoint.REUSE_PHASES)
    assert endpoint.REUSE_PARENT + '/prepare-lexical/complete.json' in deps
    (failed / 'exit.json').write_text(json.dumps(dict(
        exit_code=-6, error=None, owned_group_closed=True, altered=True)))
    with pytest.raises(ValueError, match='failed evidence changed'):
        endpoint.reuse_dependencies(tmp_path)
    (parent / 'ranking-controller-start.json').write_text('{}')
    with pytest.raises(ValueError, match='no ranking'):
        endpoint.reuse_dependencies(tmp_path)


def test_tracking_disables_repository_and_every_installed_framework():
    from clearml import Task

    tree = ast.parse(textwrap.dedent(inspect.getsource(Task.init)))
    keys = {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'should_connect' and node.args
            and isinstance(node.args[0], ast.Constant)}
    assert keys and keys <= set(endpoint.TRACKING_FRAMEWORKS)
    assert endpoint.TRACKING_FRAMEWORKS['detect_repository'] is False
    assert not any(endpoint.TRACKING_FRAMEWORKS.values())


@pytest.mark.parametrize('attempt', range(3))
def test_actual_offline_tracking_exits_without_repository_thread(tmp_path, attempt):
    code = '''
import json
import torch
from clearml import Task
from ng80_sparse_endpoints import TRACKING_FRAMEWORKS
Task.set_offline(True)
task = Task.init(project_name='II42-NG-validation', task_name='tracking-shutdown',
    reuse_last_task_id=False, auto_connect_frameworks=TRACKING_FRAMEWORKS.copy(),
    auto_connect_arg_parser=False, auto_connect_streams=False,
    auto_resource_monitoring=False)
assert task._detect_repo_async_thread is None
task.connect(dict(synthetic=True, model_inference=False))
task.close()
print('NG80_TRACKING_CLOSED', task.id, flush=True)
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CLEARML_TASK_', 'TRAINS_TASK_', 'CLEARML_PROC_', 'TRAINS_PROC_'))}
    env.update(CLEARML_OFFLINE_MODE='1', CUDA_VISIBLE_DEVICES='',
               CLEARML_CACHE_DIR=str(tmp_path / str(attempt)),
               PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'NG80_TRACKING_CLOSED offline-' in result.stdout
    assert 'Fatal Python error' not in result.stderr
    assert list((tmp_path / str(attempt)).rglob('offline-*.zip'))


def test_freeze_is_new_and_binds_closed_parent_and_sources(tmp_path, monkeypatch):
    inputs = {}
    for arm in ('S', 'P'):
        root = tmp_path / f'NG-0080/sparse-{arm}-v1'
        root.mkdir(parents=True)
        (root / 'config.json').write_text(json.dumps(dict(arm=arm)))
        (root / 'inputs.json').write_text(json.dumps(dict(dependencies={}, source={})))
        (root / 'controller-exit.json').write_text(json.dumps(dict(
            status='all_phases_complete', completed=list(endpoint.fit.PHASES))))
        inputs[arm] = endpoint.campaign.io.sha(root / 'inputs.json')
    monkeypatch.setattr(endpoint.audit, 'INPUTS', inputs)
    monkeypatch.setattr(endpoint.fit, 'verify', lambda base, root:
                        endpoint.campaign.read(root / 'inputs.json'))
    monkeypatch.setattr(endpoint.audit, 'sealed', lambda *_: {})
    monkeypatch.setattr(endpoint, 'selection', lambda *_: dict(diagnostic=[0], calibration=[1]))
    parent = tmp_path / endpoint.dense_fit.PARENT / 'prepare'
    parent.mkdir(parents=True)
    (parent / 'queries.json').write_text('[]')
    (parent / 'selection.json').write_text('{}')
    run = tmp_path / 'NG-0080/new-endpoint'
    endpoint.freeze(tmp_path, run)
    endpoint.verify(tmp_path, run)
    with pytest.raises(ValueError, match='fresh endpoint'):
        endpoint.freeze(tmp_path, run)
    (run / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='changed input'):
        endpoint.verify(tmp_path, run)
    unfinished = tmp_path / 'NG-0080/sparse-P-v1/controller-exit.json'
    unfinished.write_text(json.dumps(dict(status='running', completed=[])))
    with pytest.raises(ValueError, match='unfinished'):
        endpoint.freeze(tmp_path, tmp_path / 'NG-0080/unfinished-endpoint')
