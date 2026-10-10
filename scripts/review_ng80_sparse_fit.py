"""Independent scalar, exposure and serialized-state review of NG80 S/P.

This reads closed training only. It neither performs model inference nor
replays the parameter VJP or AdamW updates.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

import ng80_campaign as campaign
import ng80_fit as dense_fit
import ng80_sparse_fit as fit
from ng80_data import require
from review_ng80_fit import sealed


INPUTS = {
    'S': '35d02db27758f7dc9409639efa051882d54fc20b0d7e6c38ac46e00d6d04d109',
    'P': '23c6d49d74cd83ba34f70e193379225a74cdcd3a0fcdfc02e7b2e306e6a794f7',
}


def sigmoid(values):
    return np.exp(-np.logaddexp(0., -np.asarray(values, dtype=np.float64)))


def scalar(record, teacher, positive, arm, scale, probability_device='cpu'):
    scores = np.asarray(record['scores'], dtype=np.float64)
    target = np.asarray(teacher, dtype=np.float32)
    mask = np.asarray(positive, dtype=bool)
    require(arm in INPUTS and np.isfinite(scale) and scale > 0
            and scores.shape == target.shape == mask.shape
            == (record['documents'],) and len(scores) > 1 and mask.any()
            and np.isfinite(scores).all() and np.isfinite(target).all()
            and record['positive'] == positive
            and record['vjp_replay_exact'] is True, 'scalar example identity')
    np.testing.assert_allclose(record['teacher'], teacher, rtol=0, atol=1e-12)
    if arm == 'S':
        error = scores / scale - target.astype(np.float64)
        error -= error.mean()
        loss = float(np.mean(error ** 2))
        gradient = error / (2 * len(error) * scale)
    else:
        # Eligibility is defined on the actual FP32 teacher differences.
        margins = target[mask, None] - target[None, ~mask]
        # Replay the frozen target's precision/backend, not a new FP64 target.
        # Loss and score derivatives below remain independent NumPy formulas.
        import torch
        probability = torch.sigmoid(torch.as_tensor(
            margins, device=probability_device) / .04).cpu().numpy().astype(np.float64)
        eligible = margins > 0
        weight = np.maximum(2 * probability - 1, 0) * eligible
        weight /= np.maximum(eligible.sum(1), 1)[:, None] * mask.sum()
        difference = scores[mask, None] - scores[None, ~mask]
        loss = float(np.sum(weight * (
            np.logaddexp(0., -difference) + (1 - probability) * difference)))
        derivative = weight * (sigmoid(difference) - probability) / 4
        gradient = np.zeros_like(scores)
        gradient[mask] = derivative.sum(1)
        gradient[~mask] = -derivative.sum(0)
    np.testing.assert_allclose(record['loss'], loss, rtol=3e-5, atol=2e-8)
    np.testing.assert_allclose(record['score_gradient'], gradient,
                               rtol=3e-5, atol=2e-8)
    return float(np.max(np.abs(record['score_gradient'] - gradient)))


def optimizer_audit(path, step, expected, model, config=None):
    import torch

    saved = torch.load(path, weights_only=True, map_location='cpu')
    names, state = saved['names'], saved['state']
    parameters = dict(model.named_parameters())
    require(saved['step'] == step and saved['fingerprint'] == expected
            and len(names) == len(state['param_groups']), 'optimizer header')
    slots = {}
    for group_names, group in zip(names, state['param_groups'], strict=True):
        require(len(group_names) == len(group['params']), 'optimizer layout')
        for name, index in zip(group_names, group['params'], strict=True):
            require(name not in slots, 'duplicate optimizer parameter')
            slots[name] = index
    require(set(slots) == set(parameters)
            and len(set(slots.values())) == len(slots)
            and set(slots.values()) == set(state['state']), 'optimizer coverage')
    if config is not None:
        trunk = [n for n in parameters if n.startswith('roberta.')]
        head = [n for n in parameters if not n.startswith('roberta.')]
        require(names == [trunk, head], 'trunk/head parameter partition')
        for group, lr in zip(state['param_groups'],
                             (config['trunk_lr'], config['head_lr']), strict=True):
            require(group['lr'] == lr and group['betas'] == tuple(config['betas'])
                    and group['eps'] == config['eps']
                    and group['weight_decay'] == config['weight_decay']
                    and not group['amsgrad'] and not group['maximize'],
                    'optimizer configuration')
    digest = hashlib.sha256(json.dumps(
        dict(names=names, groups=state['param_groups']), sort_keys=True).encode())
    # Unlike the dense ParameterDict, encoder registration order is not sorted.
    for name, parameter in parameters.items():
        values = state['state'][slots[name]]
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}
                and values['step'].numel() == 1
                and float(values['step']) == step, 'moment step/inventory')
        digest.update(name.encode())
        for key, raw in sorted(values.items()):
            value = raw.detach().cpu().contiguous()
            require(torch.isfinite(value).all(), 'nonfinite moment')
            if key != 'step':
                require(value.shape == parameter.shape
                        and value.dtype == parameter.dtype == torch.float32,
                        'moment shape/precision')
            if key == 'exp_avg_sq':
                require((value >= 0).all(), 'negative second moment')
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    require(digest.hexdigest() == expected, 'serialized moment fingerprint')
    return len(slots)


def checkpoint_audit(base, run, root, result):
    from ng71_training import Encoder

    config = campaign.read(run / 'config.json')
    encoder = Encoder(base, config, 'cpu', checkpoint=root / 'checkpoint')
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.model.state_dict().items()):
        require(np.isfinite(tensor.detach().numpy()).all(), 'nonfinite checkpoint')
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    require(digest.hexdigest() == result['model_sha256'], 'checkpoint fingerprint')
    count = optimizer_audit(root / 'optimizer.pt', result['updates'],
                            result['optimizer_sha256'], encoder.model,
                            config['optimizer'])
    del encoder
    gc.collect()
    return count


def review(base, probability_device='cuda'):
    require(probability_device == 'cuda', 'scientific P audit requires frozen CUDA target arithmetic')
    parent = base / dense_fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels = campaign.read(parent / 'prepare/panels.json')
    labels = campaign.read(parent / 'prepare/labels.json')
    order = dense_fit.order(queries, campaign.read(parent / 'prepare/selection.json'))
    examples = campaign.rows(base / fit.PREFLIGHT / 'calibrate/examples.jsonl')
    fit.validate_examples(examples, order, queries, panels, labels)
    scale = campaign.read(base / fit.PREFLIGHT / 'calibrate/scale.json')['scale']
    q = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    d = np.load(base / campaign.DENSE, mmap_mode='r')
    lengths = {role: np.minimum(np.load(parent / 'encode-hidden' /
               f'{role}-token-lengths.npy'), limit)
               for role, limit in (('query', 64), ('document', 256))}
    output = {}
    for arm, expected in INPUTS.items():
        run = base / f'NG-0080/sparse-{arm}-v1'
        require(campaign.io.sha(run / 'inputs.json') == expected, 'training input')
        fit.verify(base, run)
        controller = campaign.read(run / 'controller-exit.json')
        require(controller['status'] == 'all_phases_complete'
                and controller['completed'] == list(fit.PHASES)
                and campaign.read(run / 'order.json') == order, 'training graph')
        previous, total, error, moments = None, 0, 0., {}
        tokens, candidates = dict(query=0, document=0), 0
        for end in (1024, 2048, 3072):
            root = run / f'train-{end}'
            result = sealed(root)
            initial = previous['model_sha256'] if previous else fit.preflight.INITIAL
            require(result['arm'] == arm and result['updates'] == end
                    and result['start_update'] == end - 1024
                    and result['initial_model_sha256'] == initial
                    and result['model_sha256'] != initial
                    and result['real_reload_exact'] and result['scale'] == scale
                    and result['query_exposures'] == 4096
                    and not any(result[k] for k in ('canary_updates_reused',
                        'locked_test_access', 'quality_evaluated',
                        'native_cost_evaluated', 'token_counts_include_replay')),
                    'chunk contract/continuation')
            moments[str(end)] = checkpoint_audit(base, run, root, result)
            observed, losses = [], []
            chunk_tokens, chunk_nnz = dict(query=0, document=0), dict(query=0, document=0)
            chunk_candidates = 0
            with (root / 'updates.jsonl').open() as stream:
                for offset, line in enumerate(stream):
                    row = json.loads(line)
                    step = end - 1024 + offset
                    indices = order[4 * step:4 * (step + 1)]
                    require(offset < 1024 and row['update'] == step + 1
                            and len(row['examples']) == len(indices) == 4
                            and np.isfinite([row[k] for k in (
                                'gradient_norm', 'update_l2', 'seconds')]).all()
                            and min(row[k] for k in (
                                'gradient_norm', 'update_l2', 'seconds')) > 0
                            and row['clipped'] == (row['gradient_norm'] > 1),
                            'update identity/count/gradient')
                    for index, record, baseline in zip(indices, row['examples'],
                            examples[4 * step:4 * (step + 1)], strict=True):
                        pool = panels[index]
                        require(record['query_index'] == index
                                and record['query_id'] == queries[index]['query_id']
                                and record['pool'] == pool
                                and record['lexical'] == baseline['lexical'],
                                'query/panel/lexical identity')
                        teacher = d[pool].astype(np.float64) @ q[index].astype(np.float64)
                        error = max(error, scalar(record, teacher,
                            [i in labels[index] for i in pool], arm, scale, probability_device))
                        work = dict(query=int(lengths['query'][index]),
                                    document=int(lengths['document'][pool].sum()))
                        require(record['token_exposures'] == work, 'token exposure')
                        for role in work:
                            value = record[role + '_nnz']
                            require(type(value) is int and value > 0, 'invalid NNZ exposure')
                            chunk_tokens[role] += work[role]
                            chunk_nnz[role] += value
                        chunk_candidates += len(pool)
                    np.testing.assert_allclose(row['loss'], np.mean([
                        r['loss'] for r in row['examples']]), rtol=0, atol=1e-12)
                    losses.append(row['loss'])
                    observed.extend(indices)
            require(observed == order[4 * (end - 1024):4 * end]
                    and chunk_tokens == result['token_exposures']
                    and chunk_nnz == result['encoded_nnz_exposures']
                    and chunk_candidates == result['candidate_exposures'], 'exposure totals')
            np.testing.assert_allclose(result['mean_loss'], np.mean(losses), rtol=0, atol=1e-12)
            for role in tokens:
                tokens[role] += chunk_tokens[role]
            candidates += chunk_candidates
            total += len(observed)
            previous = result
        output[arm] = dict(updates=3072, query_exposures=total,
            unique_queries=len(set(order)), candidate_exposures=candidates,
            token_exposures=tokens, max_scalar_gradient_error=error,
            serialized_optimizer_parameters=moments,
            terminal_model_sha256=previous['model_sha256'])
    require(output['S']['token_exposures'] == output['P']['token_exposures']
            and output['S']['candidate_exposures'] == output['P']['candidate_exposures'],
            'matched exposure failed')
    return dict(passed=True, arms=output, model_parameters_independently_hashed=True,
        optimizer_tensors_independently_hashed=True,
        checkpoint_continuation_worker_reload_verified=True,
        actual_parameter_vjp_recomputed=False, independent_optimizer_replay=False,
        inference_performed=False, independent_holdout=False, locked_test_access=False,
        target_arithmetic='fp32_cuda', scalar_arithmetic='independent_numpy_fp64',
        numeric_tolerances_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'never overwrite an audit')
    campaign.io.write(args.output, review(args.research_root))
