"""CPU contract and real optimizer serialization tests; no research data."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_sparse_fit as fit
import ng71_training as training
import ng71_execution as execution
from ng80_training import Readout


def canaries():
    return {a: dict(arm=a, discard_updates=True, canary_updates=16,
        initial_model_sha256=fit.preflight.INITIAL, exact_reload=True,
        initial_full_batch_score_parity=True, compute_seconds=25. if a == 'S' else 32.,
        real_vjp_relative_l2=dict.fromkeys(fit.dense_fit.data.DOMAINS, 0.))
        for a in ('P', 'S')}


def test_schedule_covers_exactly_3072_and_discards_canaries():
    spans = [fit.bounds(p) for p in fit.PHASES]
    assert spans == [(0, 1024), (1024, 2048), (2048, 3072)]
    assert [s for a, b in spans for s in range(a, b)] == list(range(3072))
    for phase in ('train-canary', 'train-16', 'train-4096', None):
        with pytest.raises(ValueError):
            fit.bounds(phase)
    budget = fit.common_budget(canaries())
    assert budget['S']['predicted_chunk_seconds'][1024] < budget['P']['predicted_chunk_seconds'][1024]


@pytest.mark.parametrize('field,value', [
    ('discard_updates', False), ('canary_updates', 15), ('exact_reload', False),
    ('initial_model_sha256', 'canary-checkpoint'),
    ('initial_full_batch_score_parity', False), ('compute_seconds', 1000.),
    ('real_vjp_relative_l2', {'fever': 0., 'hotpotqa': 0., 'nq': .0001}),
    ('real_vjp_relative_l2', {'fever': 0., 'hotpotqa': 0., 'nq': float('nan')}),
])
def test_common_budget_cannot_relax_qualified_contract(field, value):
    data = canaries()
    data['P'][field] = value
    with pytest.raises(ValueError):
        fit.common_budget(data)


def test_freeze_creates_new_config_without_overwriting_parent(tmp_path, monkeypatch):
    parent = tmp_path / fit.PREFLIGHT
    parent.mkdir(parents=True)
    config = dict(training_enabled=True, canary_only=True, base={}, optimizer={})
    (parent / 'config.json').write_text(json.dumps(config))
    (parent / 'order.json').write_text('[0, 1, 2]')
    for name in ('ng80_training.py', 'ng71_training.py', 'ng71_execution.py'):
        (parent / name).write_bytes((SOURCE / name).read_bytes())
    original = dict(dependencies={}, source={p.name: fit.campaign.io.sha(p)
        for p in parent.iterdir() if p.is_file()})
    (parent / 'inputs.json').write_text(json.dumps(original))
    before = fit.campaign.io.sha(parent / 'config.json')
    monkeypatch.setattr(fit, 'PREFLIGHT_SHA', fit.campaign.io.sha(parent / 'inputs.json'))
    monkeypatch.setattr(fit.preflight, 'verify', lambda *_: original)
    monkeypatch.setattr(fit, 'closed', lambda *_: {})
    for phase in fit.preflight.CONTINUATION_PHASES:
        folder = parent / phase
        folder.mkdir()
        result = canaries()[phase.split('-')[1]] if phase.startswith('train-') else {}
        (folder / 'results.json').write_text(json.dumps(result))
        (folder / 'complete.json').write_text(json.dumps(dict(
            files={'results.json': fit.campaign.io.sha(folder / 'results.json')})))
    run = tmp_path / 'NG-0080/new-S'
    fit.freeze(tmp_path, run, 'S')
    assert fit.campaign.read(run / 'config.json') == dict(config, canary_only=False, arm='S')
    assert fit.campaign.io.sha(parent / 'config.json') == before
    fit.verify(tmp_path, run)
    with pytest.raises(FileExistsError):
        fit.freeze(tmp_path, run, 'S')


def example_data():
    queries = [dict(query_id=f'q{i}') for i in range(12288)]
    records = [dict(query_id=q['query_id'], query_index=i, pool=[0, 1],
        positive=[True, False], teacher=[.8, .2], lexical=[.2, .1],
        initial_semantic=[.4, .3]) for i, q in enumerate(queries)]
    return records, list(range(12288)), queries, [[0, 1]] * 12288, [[0]] * 12288


def test_example_identity_order_positives_and_score_alignment():
    data = example_data()
    fit.validate_examples(*data)
    for key, value in (('query_index', 1), ('query_id', 'other'), ('pool', [1, 0]),
                       ('positive', [False, False]), ('teacher', [.8]),
                       ('lexical', [float('nan'), .2])):
        changed = deepcopy(data[0][0])
        data[0][0][key] = value
        with pytest.raises(ValueError):
            fit.validate_examples(*data)
        data[0][0] = changed
    data[1][1] = data[1][0]
    with pytest.raises(ValueError, match='exposure'):
        fit.validate_examples(*data)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = torch.nn.Linear(2, 2)
        self.lm_head = torch.nn.Linear(2, 2)

    def save_pretrained(self, path):
        path.mkdir(exist_ok=False)
        torch.save(self.state_dict(), path / 'model.pt')


class Encoder:
    checkpoints = []

    def __init__(self, base, config, device, checkpoint=None):
        self.config, self.device = config, 'cpu'
        with torch.random.fork_rng():
            torch.manual_seed(12)
            self.model = Model()
        # Serialization/exposure fixtures need active, nonconstant postings.
        with torch.no_grad():
            for layer in (self.model.roberta, self.model.lm_head):
                layer.weight.copy_(torch.tensor([[.2, .4], [.5, .3]]))
                layer.bias.fill_(.1)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint / 'model.pt', weights_only=True))
        self.checkpoints.append(checkpoint)

    def encode(self, rows, role):
        values = torch.tensor(rows, dtype=torch.float32)
        return self.model.lm_head(self.model.roberta(values)).relu().log1p()

    def tokenizer(self, rows, **kwargs):
        return dict(input_ids=[list(range(min(len(r), kwargs['max_length']))) for r in rows])


@pytest.mark.parametrize('arm', ['S', 'P'])
def test_sparse_complete_optimizer_and_both_role_reload(tmp_path, monkeypatch, arm):
    config = dict(optimizer=dict(trunk_lr=5e-6, head_lr=2e-5,
        betas=[.9, .999], eps=1e-8, weight_decay=.01))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(training, 'Encoder', Encoder)
    initial = training.parameter_hash(Encoder(tmp_path, config, 'cpu').model)
    monkeypatch.setattr(fit.preflight, 'INITIAL', initial)
    Encoder.checkpoints.clear()
    model, optimizer = fit.load_model(tmp_path, tmp_path, arm, 0)
    assert Encoder.checkpoints == [None]
    assert model.projection is None
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    output = tmp_path / 'train-1'
    output.mkdir()
    sample = dict(query=[[.1, .2]], document=[[.1, .2], [.5, .6]])
    result = fit.save_and_reload(tmp_path, tmp_path, output, model, optimizer, 1, sample)
    result['arm'] = arm
    assert result['real_reload_exact'] and result['model_sha256'] != initial
    assert not (output / 'projection.pt').exists()
    (output / 'results.json').write_text(json.dumps(result))
    monkeypatch.setattr(fit, 'closed', lambda run, phase:
                        json.loads((run / phase / 'results.json').read_text()))
    loaded, restored = fit.load_model(tmp_path, tmp_path, arm, 1)
    assert Encoder.checkpoints[-1] == output / 'checkpoint'
    assert execution.optimizer_fingerprint(loaded.encoder, restored, 1) == result['optimizer_sha256']
    assert training.parameter_hash(loaded.encoder.model) == result['model_sha256']
    result['arm'] = 'P' if arm == 'S' else 'S'
    (output / 'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='chain parent'):
        fit.load_model(tmp_path, tmp_path, arm, 1)


def test_closed_phase_rejects_unlisted_nested_receipt(tmp_path, monkeypatch):
    folder = tmp_path / 'phase'
    folder.mkdir()
    files = ['results.json', 'exit.json', 'clearml.json']
    for name in files:
        (folder / name).write_text('{}')
    (folder / 'complete.json').write_text(json.dumps(dict(files=dict.fromkeys(files, 'unused'))))
    monkeypatch.setattr(fit.campaign, 'closed', lambda *_: None)
    assert fit.closed(tmp_path, 'phase') == {}
    (folder / 'nested').mkdir()
    (folder / 'nested/complete.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory'):
        fit.closed(tmp_path, 'phase')


@pytest.mark.parametrize('arm', ['S', 'P'])
@pytest.mark.parametrize('dead', [False, True])
def test_executor_records_exact_chunk_exposures_and_updates(tmp_path, monkeypatch, arm, dead):
    if dead:
        original_init = Encoder.__init__

        def initialize_dead(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            with torch.no_grad():
                self.model.lm_head.bias.fill_(-100.)

        monkeypatch.setattr(Encoder, '__init__', initialize_dead)
    config = dict(base=dict(query_input_tokens=2, document_input_tokens=2),
        optimizer=dict(trunk_lr=5e-6, head_lr=2e-5, betas=[.9, .999], eps=1e-8, weight_decay=.01))
    data = example_data()
    records, indices, queries, panels, labels = data
    for query in queries:
        query['query'] = [.2, .8]
    docs = [dict(text=[.1, .2]), dict(text=[.5, .6])]
    model = Readout(Encoder(tmp_path, config, 'cpu'), arm)
    with torch.no_grad():
        reference = (model.encode([d['text'] for d in docs], 'document')
                     @ model.encode([queries[0]['query']], 'query').T).ravel().tolist()
    for record in records:
        record['initial_semantic'] = reference
    state = {'config.json': config, 'inputs.json': dict(arm=arm),
        'order.json': indices, 'queries.json': queries, 'panels.json': panels,
        'labels.json': labels, 'scale.json': dict(scale=2.)}
    original_read = fit.campaign.read
    monkeypatch.setattr(fit.campaign, 'read', lambda p: state[p.name]
                        if p.name in state else original_read(p))
    monkeypatch.setattr(fit.campaign, 'rows', lambda p:
                        records if p.name == 'examples.jsonl' else docs)
    monkeypatch.setattr(training, 'Encoder', Encoder)
    monkeypatch.setattr(fit.preflight, 'INITIAL', training.parameter_hash(model.encoder.model))
    monkeypatch.setattr(fit, 'bounds', lambda _: (0, 2))
    output = tmp_path / 'out'
    output.mkdir()
    if dead:
        assert reference == [0., 0.]
        with pytest.raises(ValueError, match='zero complete-batch gradient'):
            fit.execute(tmp_path, tmp_path, output, 'test-only-short-span')
        assert not (output / 'optimizer.pt').exists()
        return
    result = fit.execute(tmp_path, tmp_path, output, 'test-only-short-span')
    updates = [json.loads(s) for s in (output / 'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in updates] == [1, 2]
    assert [q['query_id'] for r in updates for q in r['examples']] == [f'q{i}' for i in range(8)]
    assert result['query_exposures'] == 8 and result['candidate_exposures'] == 16
    assert result['token_exposures'] == dict(query=16, document=32)
    assert result['updates'] == 2 and result['real_reload_exact']
    assert not result['canary_updates_reused'] and not result['quality_evaluated']
    assert all(r['update_l2'] > 0 and len(r['examples']) == 4 for r in updates)
