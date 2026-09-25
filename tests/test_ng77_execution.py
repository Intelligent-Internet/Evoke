"""Synthetic full-chunk hooks and explicit TRAIN-only endpoint selection."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution
import ng71_observation as observation
from ng77_training import RetainedObjective
from test_ng77_cuda import Encoder, fixture


class SavedModel(torch.nn.Linear):
    def save_pretrained(self, path):
        path.mkdir()
        torch.save(self.state_dict(), path / 'model.pt')


class ChunkEncoder(Encoder):
    def __init__(self, base=None, config=None, device=None, checkpoint=None):
        super().__init__()
        model = SavedModel(3, 4, dtype=torch.float64)
        model.load_state_dict(self.model.state_dict())
        self.model = model
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint / 'model.pt', weights_only=True))
        self.tokenizer = lambda texts, **kwargs: dict(input_ids=(
            [1, 2, 3] if isinstance(texts[0], float) else [[1, 2, 3] for _ in texts]))


def inputs():
    ids = [f'q{i}' for i in range(384)]
    records = {q: dict(reference_update=0) for q in ids}
    config = dict(training_enabled=True, pilot=dict(train_queries=384),
                  base=dict(state_sha256=execution.training.parameter_hash(ChunkEncoder().model)))
    queries = {q: dict(query=[1., 2., 3.]) for q in ids}
    return config, records, ids * 2, queries


def run_chunk(root, coefficient):
    config, records, order, queries = inputs()
    sample, anchors = fixture()
    transforms = None if coefficient is None else {
        q: RetainedObjective(deepcopy(anchors), coefficient) for q in records}
    root.mkdir()
    with patch.object(execution.training, 'Encoder', ChunkEncoder), \
            patch.object(execution, 'example', side_effect=lambda *args: deepcopy(sample)), \
            patch.object(execution.training, 'make_optimizer', side_effect=lambda e:
                         torch.optim.AdamW(e.model.parameters(), lr=.0001)), \
            patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
            patch.object(torch.cuda, 'empty_cache'):
        result = execution.train_chunk(None, config, root, 'D', 0, 96,
            records, order, queries, None, SimpleNamespace(get_logger=lambda: Mock()),
            loss_transforms=transforms, stage_limit_seconds=1800)
    rows = [json.loads(line) for line in (root / 'progress.jsonl').read_text().splitlines()]
    return result, rows


def test_full_96_update_zero_control_keeps_model_moments_and_original_trace(tmp_path):
    a, x = run_chunk(tmp_path / 'original', None)
    b, y = run_chunk(tmp_path / 'zero', 0.)
    c, z = run_chunk(tmp_path / 'keep', 1.)
    assert a['model_sha256'] == b['model_sha256'] != c['model_sha256']
    assert a['optimizer_fingerprint'] == b['optimizer_fingerprint']
    assert len(x) == len(y) == len(z) == 96
    for left, right in zip(x, y, strict=True):
        left.pop('seconds')
        right.pop('seconds')
        for item in right['examples']:
            receipt = item.pop('objective_components')
            assert receipt['retention_coefficient'] == 0.
            assert receipt['original_loss'] == item['loss']
            assert receipt['keep_loss'] >= 0
        assert left == right
    assert len({r['query_id'] for step in z for r in step['examples']}) == 384
    assert all(r['objective_components']['retention_coefficient'] == 1.
               and r['vjp_replay_exact'] for step in z for r in step['examples'])


@pytest.mark.parametrize('bad', ['missing', 'extra', 'noncallable', 'observer', 'time'])
def test_invalid_hooks_fail_before_gpu_initialization(bad):
    config, records, order, queries = inputs()
    _, anchors = fixture()
    hooks = {q: RetainedObjective(anchors, 0.) for q in records}
    observer, limit = None, 1800
    if bad == 'missing':
        del hooks[order[0]]
    elif bad == 'extra':
        hooks['sentinel'] = RetainedObjective(anchors, 0.)
    elif bad == 'noncallable':
        hooks[order[0]] = SimpleNamespace(last=None)
    elif bad == 'observer':
        observer = lambda *args: None
    else:
        limit = 5401
    with patch.object(execution.training, 'Encoder') as constructor:
        with pytest.raises(ValueError):
            execution.train_chunk(None, config, None, 'D', 0, 96, records,
                order, queries, None, None, observer=observer,
                loss_transforms=hooks, stage_limit_seconds=limit)
        constructor.assert_not_called()


def surface():
    rows = [dict(query_id=f'q{i}', split=('TRAIN' if i < 6144 else
            'DEV_NEW' if i < 7680 else 'LOCKED_TEST')) for i in range(7872)]
    selection = dict(pilot=[r['query_id'] for r in rows[:384]],
                     sentinel=[r['query_id'] for r in rows[384:768]])
    return rows, selection


def test_train_only_endpoint_is_768_never_dev_or_locked():
    rows, selection = surface()
    with patch.object(execution, 'read', return_value=rows):
        result = observation.surface_queries(Path('/base'), selection, True, include_dev=False)
        assert [r['query_id'] for r in result] == selection['pilot'] + selection['sentinel']
        assert {r['split'] for r in result} == {'TRAIN'}
        assert {r['surface'] for r in result} == {'TRAIN_PILOT', 'TRAIN_SENTINEL'}
        assert len(observation.surface_queries(Path('/base'), selection, True)) == 2304
        assert len(observation.surface_queries(Path('/base'), selection, False)) == 384
        rows[384]['split'] = 'DEV_NEW'
        with pytest.raises(ValueError, match='forbidden query split'):
            observation.surface_queries(Path('/base'), selection, True, include_dev=False)


def test_train_only_rejects_overlap_and_nonboolean_flags():
    rows, selection = surface()
    with patch.object(execution, 'read', return_value=rows):
        with pytest.raises(ValueError, match='booleans'):
            observation.surface_queries(Path('/base'), selection, True, include_dev=0)
        selection['sentinel'][0] = selection['pilot'][0]
        with pytest.raises(ValueError, match='overlap'):
            observation.surface_queries(Path('/base'), selection, True, include_dev=False)
