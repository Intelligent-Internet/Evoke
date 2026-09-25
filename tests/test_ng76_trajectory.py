"""Synthetic replay/observer checks; never load research corpora or models."""

import hashlib
from pathlib import Path
import random
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution
import ng71_training as training
import ng76_trajectory as controller
import ng76_trajectory_observer as observer


class Toy:
    def __init__(self):
        self.model = torch.nn.Linear(3, 5).eval()
        with torch.no_grad():
            self.model.weight.fill_(.2)
            self.model.bias.copy_(torch.arange(1., 6.))
        self.config = {'optimizer': {'queries_per_update': 4, 'gradient_clip_norm': 1.}}

    def encode(self, texts, role):
        texts = [[len(text), 1., 2.] if isinstance(text, str) else text for text in texts]
        values = self.model(torch.tensor(texts, dtype=torch.float32)).relu().log1p()
        return .9 * values / values.sum(1, keepdim=True) if role == 'query' else values


def example():
    return dict(query=[1., 2., 3.], documents=[[0., 1., 1.], [1., 0., 2.]],
                lexical=[.2, .1], target=[.9, .1])


def test_no_grad_observer_preserves_exact_multistep_optimizer_trajectory():
    original, observed = Toy(), Toy()
    left = torch.optim.AdamW(original.model.parameters(), lr=.01)
    right = torch.optim.AdamW(observed.model.parameters(), lr=.01)
    for step in range(8):
        if step % 2 == 0:
            (q, d), proof = observer.read_only(observed, right, lambda: (
                observer.encode_selected(observed, [example()['query']], 'query'),
                observer.encode_selected(observed, example()['documents'], 'document')))
            assert q.nnz > 0 and d.nnz > 0 and proof['grad_enabled']
        a = training.optimizer_step(original, left, [example()] * 4)
        b = training.optimizer_step(observed, right, [example()] * 4)
        assert a == b
        assert training.parameter_hash(original.model) == training.parameter_hash(observed.model)
        assert execution.optimizer_fingerprint(original, left, step + 1) == execution.optimizer_fingerprint(
            observed, right, step + 1)


@pytest.mark.parametrize('kind', ['model', 'optimizer', 'gradient', 'mode', 'requires_grad', 'rng', 'numpy', 'python'])
def test_observer_mutation_fails_closed(kind):
    enc = Toy()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=.01)
    training.optimizer_step(enc, opt, [example()] * 4)

    def mutate():
        p = next(enc.model.parameters())
        if kind == 'model':
            p.add_(1)
        elif kind == 'optimizer':
            opt.state[p]['exp_avg'].add_(1)
        elif kind == 'gradient':
            p.grad.zero_()
        elif kind == 'mode':
            enc.model.train()
        elif kind == 'requires_grad':
            p.requires_grad_(False)
        elif kind == 'rng':
            torch.rand(1)
        elif kind == 'numpy':
            np.random.random()
        else:
            random.random()

    with pytest.raises(ValueError, match='mutated'):
        observer.read_only(enc, opt, mutate)


def test_hash_binds_shape_dtype_key_and_content():
    initial = observer.tree_hash({'a': np.zeros((2, 2), dtype=np.float32)})
    for value in ({'b': np.zeros((2, 2), dtype=np.float32)},
                  {'a': np.zeros((4,), dtype=np.float32)},
                  {'a': np.zeros((2, 2), dtype=np.float64)},
                  {'a': np.ones((2, 2), dtype=np.float32)}):
        assert observer.tree_hash(value) != initial


def test_pool_scores_bind_global_doc_ids_not_matrix_offsets_and_add_lexical_once():
    data = dict(query_ids=['q'], documents={'9': {}, '2': {}, '5': {}},
                pools={'q': [2, 9]}, lexical_scores={'q': [.1, .4]})
    q = sparse.csr_matrix(np.array([[1., 2.]], dtype=np.float32))
    d = sparse.csr_matrix(np.array([[2., 3.], [8., 9.], [4., 5.]], dtype=np.float32))
    assert observer.pool_scores(data, q, d) == {'q': [8.1, 14.4]}


def test_forecast_counts_observation_and_training_and_closure_not_just_gpu():
    assert observer.forecast(20, [10.], [], 6, 96) == 1009.25
    assert observer.forecast(50, [10., 12.], [1., 2.], 5, 80) == 740.
    for args in ((0, [], [], 6, 96), (-1, [1.], [], 6, 96), (1, [float('nan')], [], 6, 96)):
        with pytest.raises(ValueError):
            observer.forecast(*args)


def test_all_pairs_telescope_and_repeated_crossings_preserve_controls():
    data = dict(query_ids=['q'], queries={'q': {'subset': 'nq'}},
                pools={'q': [1, 3, 5]}, gold={'q': [3]})
    scores = {s: {'q': [0., 1., 2.]} for s in range(0, 193, 16)}
    scores[16]['q'] = [2., 1., 0.]
    scores[192]['q'] = [1., 1., 0.]
    pairs, domains = controller.reduce_path(data, scores)
    assert len(pairs) == 2
    assert domains['nq']['lost'] == domains['nq']['gained'] == 1
    assert pairs[0]['lost_intervals'] == [16, 192]
    assert pairs[0]['gained_intervals'] == [32]
    assert pairs[0]['ahead'][-1] is False  # Global ID 1 wins score tie with gold ID 3.
    assert all(p['telescoping_residual'] == 0 for p in pairs)
    del scores[32]
    with pytest.raises(ValueError, match='missing'):
        controller.reduce_path(data, scores)


def test_original_training_hook_runs_after_init_before_first_update():
    enc = Toy()
    config = dict(training_enabled=True, pilot={'train_queries': 384},
                  base={'state_sha256': training.parameter_hash(enc.model)})
    records = {str(i): {'reference_update': 0} for i in range(384)}
    order = list(records) * 2
    called = []

    def stop(step, encoder, optimizer, actual):
        called.append((step, encoder, actual))
        raise ValueError('intentional observer stop')

    with patch.object(training, 'Encoder', return_value=enc), patch.object(
            training, 'make_optimizer', return_value=torch.optim.AdamW(enc.model.parameters())), patch.object(
                training, 'optimizer_step') as update:
        with pytest.raises(ValueError, match='intentional'):
            execution.train_chunk(None, config, None, 'D', 0, 96, records, order,
                                  {}, {}, None, observer=stop)
        update.assert_not_called()
    assert called == [(0, enc, None)]


def test_trajectory_cannot_finish_with_wrong_history_or_coverage():
    obj = object.__new__(observer.Observer)
    obj.start, obj.end = 0, 96
    obj.seen, obj.durations = list(range(0, 97, 16)), [1.] * 96
    original = dict(arm='D', cumulative_steps=96, model_sha256='model', optimizer_fingerprint='opt',
                    initial_model_sha256='base', initial_optimizer_fingerprint=None,
                    tokens={}, candidate_pairs=123, query_exposures=384)
    assert obj.finish(original, original)['historical_endpoint_exact']
    with pytest.raises(ValueError, match='endpoint mismatch'):
        obj.finish(dict(original, optimizer_fingerprint='changed'), original)
    obj.seen.pop()
    with pytest.raises(ValueError, match='coverage'):
        obj.finish(original, original)


def payload():
    ids = [f'{domain}-{i}' for domain in ('fever', 'hotpotqa', 'nq') for i in range(8)]
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return dict(query_ids=ids, queries={q: dict(query_id=q, query=q, split='TRAIN',
        subset=q.split('-')[0], text_sha256=sha(q)) for q in ids},
        pools={q: [1, 3] for q in ids}, gold={q: [3] for q in ids},
        lexical_scores={q: [.1, .2] for q in ids}, documents={'1': {'text': 'a'}, '3': {'text': 'bcd'}},
        document_text_sha256={'1': sha('a'), '3': sha('bcd')},
        scientific_training_enabled=False, trajectory_replay_enabled=False,
        locked_test_scored=False, observation_steps=list(range(0, 193, 16)))


def test_actual_observer_writes_codes_and_proof_without_mutating_state(tmp_path):
    enc = Toy()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=.01)
    data = payload()
    obj = observer.Observer(data, tmp_path, [dict(step=s) for s in range(97, 193)], 96, 192, time.time())
    before = observer.state(enc, opt)
    obj(96, enc, opt, None)
    assert before == observer.state(enc, opt)
    saved = execution.read(tmp_path / 'observe-096/observation.json')
    assert saved['state_unchanged'] and saved['step'] == 96 and saved['endpoint_parity'] is None
    assert sparse.load_npz(tmp_path / 'observe-096/query.npz').shape == (24, 5)
    with pytest.raises(ValueError, match='duplicate'):
        obj(96, enc, opt, None)


@pytest.mark.parametrize('kind', ['split', 'query', 'doc', 'gold', 'replay', 'steps'])
def test_payload_rejects_test_inference_text_and_frozen_scope_changes(kind):
    data = payload()
    if kind == 'split':
        data['queries']['nq-0']['split'] = 'LOCKED_TEST'
    elif kind == 'query':
        data['queries']['nq-0']['query'] = 'different'
    elif kind == 'doc':
        data['documents']['1']['text'] = 'different'
    elif kind == 'gold':
        data['gold']['nq-0'] = [7]
    elif kind == 'replay':
        data['trajectory_replay_enabled'] = True
    else:
        data['observation_steps'].pop()
    with pytest.raises(ValueError):
        observer.validate_payload(data)


def test_execution_requires_frozen_source_and_has_only_two_replays():
    assert controller.PHASES == ('replay-D-96', 'replay-D-192', 'review')
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        (p / 'source.py').write_text('before')
        hashes = {'source.py': execution.io.sha(p / 'source.py')}
        controller.verify_files(p, hashes)
        (p / 'source.py').write_text('after')
        with pytest.raises(ValueError, match='changed'):
            controller.verify_files(p, hashes)
