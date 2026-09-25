"""Fixed-reference schedule guards and bit-exact shared update engine checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution
from test_ng77_execution import ChunkEncoder, run_chunk
from test_ng77_cuda import fixture


def inputs(broad=False):
    count, epochs = (1536, 1) if broad else (384, 4)
    ids = [f'q{i}' for i in range(count)]
    state = execution.training.parameter_hash(ChunkEncoder().model)
    config = dict(training_enabled=True, pilot=dict(train_queries=count, epochs=epochs, updates=384),
                  optimizer=dict(queries_per_update=4), base=dict(state_sha256=state))
    records = {q: dict(query_id=q, split='TRAIN', reference_update=0,
                       checkpoint_state_sha256=state, total_positives=1, positive_ids=[0],
                       pool=[0, 1], positive_mask=[True, False]) for q in ids}
    queries = {q: dict(query_id=q, split='TRAIN', query=[1., 2., 3.]) for q in ids}
    return config, records, ids * epochs, queries


@pytest.mark.parametrize('broad', [False, True])
@pytest.mark.parametrize('start', [0, 96, 192, 288])
def test_both_fixed_schedules_route_exactly_one_96_step_chunk(broad, start):
    c, r, o, q = inputs(broad)
    previous = Path('/previous') if start else None
    with patch.object(execution, '_train_validated_chunk', return_value='ok') as backend:
        result = execution.train_fixed_reference_chunk(None, c, None, 'R', start, start + 96,
                                                       r, o, q, None, None, previous)
    assert result == 'ok'
    assert backend.call_args.args[4:6] == (start, start + 96)
    assert backend.call_args.kwargs == {'stage_limit_seconds': 1800}


@pytest.mark.parametrize('bad', ['disabled', 'count', 'order', 'reference', 'state',
                               'locked', 'gold', 'predecessor', 'too_far', 'identity'])
def test_invalid_breadth_inputs_fail_before_loading_gpu(bad):
    c, r, o, q = inputs()
    start, end, previous = 0, 96, None
    if bad == 'disabled': c['training_enabled'] = False
    elif bad == 'count': c['pilot']['train_queries'] = 385
    elif bad == 'order': o[0] = o[1]
    elif bad == 'reference': r[o[0]]['reference_update'] = 96
    elif bad == 'state': r[o[0]]['checkpoint_state_sha256'] = 'other'
    elif bad == 'locked': q[o[0]]['split'] = 'LOCKED_TEST'
    elif bad == 'gold': r[o[0]]['positive_ids'] = [1]
    elif bad == 'predecessor': start, end = 96, 192
    elif bad == 'too_far': start, end, previous = 384, 480, Path('/previous')
    else: r[o[0]]['query_id'] = 'other'
    with patch.object(execution.training, 'Encoder') as encoder:
        with pytest.raises(ValueError):
            execution.train_fixed_reference_chunk(None, c, None, 'R', start, end,
                                                   r, o, q, None, None, previous)
        encoder.assert_not_called()


def test_static_first96_is_bit_exact_with_historical_engine(tmp_path):
    old, old_trace = run_chunk(tmp_path / 'old', None)
    c, r, o, q = inputs()
    sample, _ = fixture()
    root = tmp_path / 'static'
    root.mkdir()
    with patch.object(execution.training, 'Encoder', ChunkEncoder), \
            patch.object(execution, 'example', side_effect=lambda *args: deepcopy(sample)), \
            patch.object(execution.training, 'make_optimizer', side_effect=lambda e:
                         torch.optim.AdamW(e.model.parameters(), lr=.0001)), \
            patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
            patch.object(torch.cuda, 'empty_cache'):
        new = execution.train_fixed_reference_chunk(None, c, root, 'R', 0, 96, r, o, q,
            None, SimpleNamespace(get_logger=lambda: Mock()))
    assert old['model_sha256'] == new['model_sha256']
    assert old['optimizer_fingerprint'] == new['optimizer_fingerprint']
    new_trace = [json.loads(line) for line in (root / 'progress.jsonl').read_text().splitlines()]
    for a, b in zip(old_trace, new_trace, strict=True):
        a.pop('seconds')
        b.pop('seconds')
        assert a == b


def test_old_refresh_entrypoint_still_rejects_new_schedule():
    c, r, o, q = inputs()
    with patch.object(execution.training, 'Encoder') as encoder:
        with pytest.raises(ValueError, match='manifest incomplete'):
            execution.train_chunk(None, c, None, 'R', 0, 96, r, o, q, None, None)
        with pytest.raises(ValueError, match='unplanned'):
            execution.train_chunk(None, c, None, 'R', 192, 288, r, o, q, None, None, Path('/previous'))
        encoder.assert_not_called()
