"""Read-only fixed-TRAIN observations, never a training objective."""

import hashlib
import random
import time

import numpy as np
from scipy import sparse
import torch

import ng71_execution as execution
import ng71_preflight as io
import ng71_training as training
from ng75_update_diagnosis import compare_replay
from prepare_ng76_trajectory import endpoint_rank


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tree_hash(value):
    """Hash tensor content and layout, not serialization object addresses."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            digest.update(repr((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f'{type(item).__name__}:{len(item)}:'.encode())
            for part in item:
                visit(part)
        else:
            require(item is None or isinstance(item, (str, int, float, bool)),
                    'unsupported state fingerprint value')
            digest.update(repr((type(item).__name__, item)).encode())
        digest.update(b'\x00')

    visit(value)
    return digest.hexdigest()


def state(encoder, optimizer):
    return dict(
        model_sha256=training.parameter_hash(encoder.model),
        optimizer_sha256=tree_hash(optimizer.state_dict()),
        gradients_sha256=tree_hash({n: p.grad for n, p in encoder.model.named_parameters()}),
        training_modes={n: m.training for n, m in encoder.model.named_modules()},
        requires_grad={n: p.requires_grad for n, p in encoder.model.named_parameters()},
        grad_enabled=torch.is_grad_enabled(),
        rng_sha256=tree_hash((random.getstate(), np.random.get_state(), torch.get_rng_state(),
                             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])))


def read_only(encoder, optimizer, callback):
    before = state(encoder, optimizer)
    with torch.no_grad():
        result = callback()
    after = state(encoder, optimizer)
    require(before == after, 'observer mutated model/optimizer/gradients/mode/RNG')
    return result, before


def validate_payload(data):
    ids = data['query_ids']
    require(len(ids) == len(set(ids)) == 24, 'sentinel cohort changed')
    require(data['scientific_training_enabled'] is False
            and data['trajectory_replay_enabled'] is False
            and data['locked_test_scored'] is False, 'prepared boundary changed')
    require(data['observation_steps'] == list(range(0, 193, 16)), 'observation schedule changed')
    for q in ids:
        row = data['queries'][q]
        require(row['split'] == 'TRAIN' and row['query_id'] == q
                and hashlib.sha256(row['query'].encode()).hexdigest() == row['text_sha256'],
                'forbidden or changed query')
        pool, gold = data['pools'][q], data['gold'][q]
        require(pool == sorted(set(pool)) and set(gold) <= set(pool)
                and len(gold) == len(set(gold)) > 0, 'invalid fixed pool/gold')
    wanted = {d for q in ids for d in data['pools'][q]}
    require(set(data['documents']) == {str(d) for d in wanted}, 'document coverage changed')
    for key, row in data['documents'].items():
        require(hashlib.sha256(row['text'].encode()).hexdigest() == data['document_text_sha256'][key],
                'document text changed')


def encode_selected(encoder, rows, role):
    parts = []
    outer = 96 if role == 'document' else 8
    for offset in range(0, len(rows), outer):
        values = encoder.encode(rows[offset:offset + outer], role)
        require(not values.requires_grad and values.dtype == torch.float32,
                'observer must use no-grad float32 readout')
        parts.append(sparse.csr_matrix(values.cpu().numpy()))
        require(torch.cuda.max_memory_allocated() <= 20 * 1024 ** 3, 'observer GPU bound exceeded')
    matrix = sparse.vstack(parts, format='csr')
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
            and np.all(matrix.getnnz(axis=1) > 0), 'invalid observed codes')
    return matrix


def pool_scores(data, queries, documents):
    doc_ids = sorted(int(d) for d in data['documents'])
    positions = {d: i for i, d in enumerate(doc_ids)}
    require(queries.shape[0] == len(data['query_ids']) and documents.shape[0] == len(doc_ids),
            'observed row identity count changed')
    result = {}
    for i, q in enumerate(data['query_ids']):
        qv = queries.getrow(i).astype(np.float64)
        dv = documents[[positions[d] for d in data['pools'][q]]].astype(np.float64)
        values = (qv @ dv.T).toarray().ravel() + np.asarray(data['lexical_scores'][q])
        require(np.isfinite(values).all(), 'non-finite fixed-pool scores')
        result[q] = values.tolist()
    return result


def margins(data, scores):
    return {(q, p, n): scores[q][data['pools'][q].index(p)] - scores[q][j]
            for q in data['query_ids'] for p in data['gold'][q]
            for j, n in enumerate(data['pools'][q]) if n not in data['gold'][q]}


def forecast(elapsed, observations, durations, remaining_observations, remaining_steps):
    require(elapsed >= 0 and observations and min(observations) > 0,
            'invalid observation timing')
    training_seconds = max(durations) * remaining_steps if durations else 359.50
    require(not durations or min(durations) > 0, 'invalid update timing')
    require(np.isfinite([elapsed, *observations, training_seconds]).all(), 'non-finite timing')
    return elapsed + 1.5 * (max(observations) * remaining_observations + training_seconds) + 360


class Observer:
    def __init__(self, data, output, expected, start, end, started_unix):
        validate_payload(data)
        self.data, self.output, self.expected = data, output, expected
        self.start, self.end, self.started = start, end, started_unix
        self.durations, self.observations, self.seen = [], [], []
        require([r['step'] for r in expected] == list(range(start + 1, end + 1)),
                'historical trace coverage changed')

    def __call__(self, step, encoder, optimizer, actual):
        if actual is not None:
            expected = self.expected[step - self.start - 1]
            compare_replay(actual, expected)
            require(actual['step'] == expected['step']
                    and actual['reference_update'] == expected['reference_update'], 'step/reference changed')
            for a, b in zip(actual['examples'], expected['examples'], strict=True):
                require(all(a[k] == b[k] for k in ('query_id', 'query_tokens', 'document_tokens',
                                                  'query_nnz', 'document_nnz')), 'replay identity/readout changed')
            self.durations.append(actual['seconds'])
        if step % 16:
            return
        require(step not in self.seen and self.start <= step <= self.end, 'duplicate/out-of-range observation')
        target = self.output / f'observe-{step:03d}'
        target.mkdir(exist_ok=False)
        tick = time.monotonic()

        def collect():
            docs = [self.data['documents'][k]['text'] for k in sorted(self.data['documents'], key=int)]
            texts = [self.data['queries'][q]['query'] for q in self.data['query_ids']]
            d = encode_selected(encoder, docs, 'document')
            q = encode_selected(encoder, texts, 'query')
            sparse.save_npz(target / 'query.npz', q)
            sparse.save_npz(target / 'document.npz', d)
            return pool_scores(self.data, q, d)

        scores, fingerprint = read_only(encoder, optimizer, collect)
        endpoint = 'initial' if step == 0 else 'D-192' if step == 192 else None
        if endpoint:
            for q in self.data['query_ids']:
                np.testing.assert_allclose(scores[q], self.data['endpoint_scores'][endpoint][q],
                                           rtol=2e-5, atol=2e-7)
                endpoint_rank(self.data['pools'][q], scores[q], self.data['endpoint_rankings'][endpoint][q])
        io.write(target / 'scores.json', scores)
        duration = time.monotonic() - tick
        io.write(target / 'observation.json', dict(step=step, seconds=duration,
            state=fingerprint, state_unchanged=True, endpoint_parity=endpoint,
            intermediate_full_corpus_quality=False, training_sentinel_in_loss=False,
            files={p.name: io.sha(p) for p in target.iterdir() if p.is_file()}))
        self.observations.append(duration)
        self.seen.append(step)
        predicted = forecast(time.time() - self.started, self.observations, self.durations,
                             (self.end - step) // 16, self.end - step)
        io.write(self.output / f'observer-canary-{step}.json', dict(
            passed=predicted < 1740, predicted_seconds=predicted,
            actual_observations=len(self.seen), includes_closure_reserve_seconds=360))
        print('NG76_OBSERVE', step, duration, 'predicted_phase_seconds', predicted, flush=True)
        require(predicted < 1740, 'trajectory forecast exceeds unchanged phase bound')

    def finish(self, result, original):
        require(self.seen == list(range(self.start, self.end + 1, 16))
                and len(self.durations) == self.end - self.start, 'trajectory coverage incomplete')
        for key in ('arm', 'cumulative_steps', 'model_sha256', 'optimizer_fingerprint',
                    'initial_model_sha256', 'initial_optimizer_fingerprint', 'tokens',
                    'candidate_pairs', 'query_exposures'):
            require(result[key] == original[key], 'historical endpoint mismatch: ' + key)
        return dict(result, passed=True, observer_steps=self.seen, historical_trace_matches=True,
                    historical_endpoint_exact=True, scientific_training_updates=0,
                    historical_optimizer_updates=self.end - self.start, locked_test_scored=False,
                    quality_evaluation=False, overall_goal_qualified=False)
