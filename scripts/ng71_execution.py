"""NG71 phase primitives; callers still need a frozen, bounded controller.

This module launches no job. TRAIN references and terminal evaluation must be
separated by the execution layer; a completed phase is not a complete study.
"""

import gc
from collections import Counter
import hashlib
import json
import shutil
import time

import numpy as np
import torch
from scipy import sparse

import ng71_diagnostics as diagnostics
import ng71_preflight as io
import ng71_training as training


def read(path):
    return json.loads(path.read_text())


def sealed(path):
    complete = read(path / 'complete.json')
    exited, tracking = read(path / 'exit.json'), read(path / 'clearml.json')
    if (complete['passed'] is not True or exited['exit_code'] != 0
            or exited.get('error') is not None or exited['owned_group_closed'] is not True
            or tracking['closed'] is not True or tracking['actual_start'] is not True):
        raise ValueError('phase is not sealed successfully')
    if not {'results.json', 'exit.json', 'clearml.json'} <= set(complete['files']):
        raise ValueError('phase receipts are not covered by completion hashes')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual != set(complete['files']) | {'complete.json'}:
        raise ValueError('sealed phase file inventory changed')
    for name, expected in complete['files'].items():
        candidate = path / name
        if not candidate.resolve().is_relative_to(path.resolve()):
            raise ValueError('phase manifest escapes its own directory')
        if io.sha(candidate) != expected:
            raise ValueError('sealed phase content changed: ' + name)
    return read(path / 'results.json')


def verify_text(record, query, documents):
    if query['query_id'] != record['query_id'] or query['split'] != 'TRAIN':
        raise ValueError('only the pinned TRAIN query can supply gradients')
    if hashlib.sha256(query['query'].encode()).hexdigest() != record['query_sha256']:
        raise ValueError('TRAIN query text changed')
    if len(record['pool']) != len(record['document_text_sha256']):
        raise ValueError('candidate text hash inventory changed')
    for doc, expected in zip(record['pool'], record['document_text_sha256'], strict=True):
        if hashlib.sha256(documents[doc]['text'].encode()).hexdigest() != expected:
            raise ValueError('TRAIN canonical document text changed')


def example(record, query, documents, config, arm):
    verify_text(record, query, documents)
    spec = config['arms'][arm]
    if (spec['pool'] not in ('original', 'witness')
            or spec['objective'] not in ('legacy_CE', 'balanced_soft_pair', 'uniform_soft_pair')):
        raise ValueError('unrecognized frozen arm definition')
    selected = diagnostics.original_record(record) if spec['pool'] == 'original' else record
    result = dict(query=query['query'],
                  documents=[documents[doc]['text'] for doc in selected['pool']],
                  lexical=selected['lexical_scores'])
    if spec['objective'] == 'legacy_CE':
        result['target'] = diagnostics.legacy_target(selected, config['ranking']).tolist()
    else:
        result['pairs'] = training.prepare_pairs(
            selected, config['ranking'], uniform=spec['objective'] == 'uniform_soft_pair')
    return result


def optimizer_names(model, optimizer):
    names = {id(value): name for name, value in model.named_parameters()}
    groups = [[names[id(value)] for value in group['params']]
              for group in optimizer.param_groups]
    flattened = [name for group in groups for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(names.values()):
        raise ValueError('optimizer must cover every parameter exactly once')
    return groups


def optimizer_fingerprint(encoder, optimizer, step):
    """Preserve NG65's name-bound, complete-moment continuation checks."""
    layout = dict(names=optimizer_names(encoder.model, optimizer),
                  groups=optimizer.state_dict()['param_groups'])
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode())
    for name, parameter in encoder.model.named_parameters():
        state = optimizer.state[parameter]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != step:
            raise ValueError('optimizer moment inventory or actual step changed')
        digest.update(name.encode())
        for key, raw in sorted(state.items()):
            value = raw.detach().cpu().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError('non-finite optimizer moment')
            if key != 'step' and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                raise ValueError('optimizer moment shape or precision changed')
            if key == 'exp_avg_sq' and (value < 0).any():
                raise ValueError('negative second optimizer moment')
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_optimizer(path, encoder, optimizer, step):
    fingerprint = optimizer_fingerprint(encoder, optimizer, step)
    with path.open('xb') as stream:
        torch.save(dict(step=step, names=optimizer_names(encoder.model, optimizer),
                        state=optimizer.state_dict(), fingerprint=fingerprint), stream)
    return fingerprint


def restore_optimizer(path, encoder, optimizer, step):
    # Keep non-capturable AdamW step scalars on CPU; moments follow parameters.
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if saved['step'] != step or saved['names'] != optimizer_names(encoder.model, optimizer):
        raise ValueError('optimizer update count or parameter identity changed')
    if saved['state']['param_groups'] != optimizer.state_dict()['param_groups']:
        raise ValueError('optimizer configuration differs from frozen continuation')
    optimizer.load_state_dict(saved['state'])
    actual = optimizer.state_dict()
    if actual['param_groups'] != saved['state']['param_groups']:
        raise ValueError('optimizer group settings changed on reload')
    if actual['state'].keys() != saved['state']['state'].keys():
        raise ValueError('optimizer state inventory changed')
    for key, fields in saved['state']['state'].items():
        if fields.keys() != actual['state'][key].keys():
            raise ValueError('optimizer slot inventory changed')
        for name, value in fields.items():
            if not torch.equal(value, actual['state'][key][name].detach().cpu()):
                raise ValueError('optimizer tensor changed on reload')
    fingerprint = optimizer_fingerprint(encoder, optimizer, step)
    if fingerprint != saved['fingerprint']:
        raise ValueError('optimizer fingerprint changed on reload')
    return fingerprint


def projected_seconds(durations, total_steps):
    if not durations or min(durations) <= 0 or not np.isfinite(durations).all():
        raise ValueError('invalid stage timing samples')
    return max(durations) * total_steps * 1.5 + 180


def train_chunk(base, config, output, arm, start, end, records, order,
                queries, documents, task, previous=None, *, observer=None,
                loss_transforms=None, stage_limit_seconds=5400):
    """A sealed 96-update chunk with fresh query and document forwards."""
    if not config['training_enabled']:
        raise ValueError('scientific training remains disabled')
    if (start, end) not in ((0, 96), (96, 192)):
        raise ValueError('unplanned training chunk')
    if (previous is None) != (start == 0):
        raise ValueError('optimizer continuation predecessor missing or unexpected')
    if (len(records) != config['pilot']['train_queries']
            or len(order) != 768
            or Counter(order) != Counter({q: 2 for q in records})
            or any(r['reference_update'] != start for r in records.values())):
        raise ValueError('fixed pilot order or reference manifest incomplete')
    return _train_validated_chunk(base, config, output, arm, start, end, records, order,
        queries, documents, task, previous, observer=observer,
        loss_transforms=loss_transforms, stage_limit_seconds=stage_limit_seconds)


def train_fixed_reference_chunk(base, config, output, arm, start, end, records, order,
                                queries, documents, task, previous=None, *, stage_limit_seconds=1800):
    """A frozen TRAIN schedule with a static initial reference, not NG71 refresh."""
    pilot = config['pilot']
    if not config['training_enabled']:
        raise ValueError('scientific training remains disabled')
    if (type(start) is not int or type(end) is not int or start < 0 or start % 96
            or end != start + 96 or end > pilot['updates']
            or pilot['updates'] != 384 or config['optimizer']['queries_per_update'] != 4):
        raise ValueError('unplanned fixed-reference chunk')
    if (previous is None) != (start == 0):
        raise ValueError('optimizer continuation predecessor missing or unexpected')
    if ((pilot['train_queries'], pilot['epochs']) not in ((384, 4), (1536, 1))
            or len(records) != pilot['train_queries'] or len(order) != 1536
            or Counter(order) != Counter({q: pilot['epochs'] for q in records})
            or set(queries) != set(records)):
        raise ValueError('fixed TRAIN breadth exposure manifest changed')
    for identity, record in records.items():
        if (record['query_id'] != identity or queries[identity]['query_id'] != identity
                or record['split'] != 'TRAIN' or queries[identity]['split'] != 'TRAIN'
                or record['reference_update'] != 0
                or record['checkpoint_state_sha256'] != config['base']['state_sha256']
                or type(record['total_positives']) is not int or record['total_positives'] <= 0
                or record['total_positives'] != len(record['positive_ids'])
                or len(set(record['positive_ids'])) != record['total_positives']
                or len(record['pool']) != len(set(record['pool']))
                or any(type(yes) is not bool for yes in record['positive_mask'])
                or set(record['positive_ids']) != {
                    d for d, yes in zip(record['pool'], record['positive_mask'], strict=True) if yes}):
            raise ValueError('TRAIN identity, all-positive or static reference changed')
    return _train_validated_chunk(base, config, output, arm, start, end, records, order,
        queries, documents, task, previous, stage_limit_seconds=stage_limit_seconds)


def _train_validated_chunk(base, config, output, arm, start, end, records, order,
                           queries, documents, task, previous, *, observer=None,
                           loss_transforms=None, stage_limit_seconds=5400):
    """Shared unchanged forward/update/reload engine behind validated schedules."""
    if (type(stage_limit_seconds) is not int
            or not 300 < stage_limit_seconds <= 5400):
        raise ValueError('invalid bounded training stage duration')
    if loss_transforms is not None:
        if (observer is not None or not isinstance(loss_transforms, dict)
                or set(loss_transforms) != set(records)
                or any(not callable(t) or not hasattr(t, 'last')
                       for t in loss_transforms.values())):
            raise ValueError('loss transforms need complete identity-bound traces and no observer')
    prior = sealed(previous) if previous else None
    if prior and (prior['arm'] != arm or prior['cumulative_steps'] != start):
        raise ValueError('wrong arm or update predecessor')
    encoder = training.Encoder(base, config, device='cuda',
                               checkpoint=previous / 'checkpoint' if previous else None)
    initial = training.parameter_hash(encoder.model)
    if initial != (prior['model_sha256'] if prior else config['base']['state_sha256']):
        raise ValueError('training initialization changed')
    optimizer = training.make_optimizer(encoder)
    initial_optimizer = None
    if previous:
        if io.sha(previous / 'optimizer.pt') != prior['optimizer_sha256']:
            raise ValueError('predecessor optimizer file disagrees with receipt')
        initial_optimizer = restore_optimizer(previous / 'optimizer.pt', encoder, optimizer, start)
        if initial_optimizer != prior['optimizer_fingerprint']:
            raise ValueError('predecessor optimizer moments disagree with receipt')
    durations, tokens, candidate_count = [], dict(query=0, document=0), 0
    if observer is not None:
        observer(start, encoder, optimizer, None)
    with (output / 'progress.jsonl').open('x') as stream:
        for step in range(start, end):
            tick = time.monotonic()
            ids = order[step * 4:(step + 1) * 4]
            examples = [example(records[q], queries[q], documents, config, arm) for q in ids]
            if loss_transforms is not None:
                for identity, item in zip(ids, examples, strict=True):
                    item['loss_transform'] = loss_transforms[identity]
            sizes = []
            for item in examples:
                qt = len(encoder.tokenizer(item['query'], truncation=True,
                                           max_length=64)['input_ids'])
                dt = sum(len(ids) for ids in encoder.tokenizer(
                    item['documents'], truncation=True, max_length=256)['input_ids'])
                tokens['query'] += qt
                tokens['document'] += dt
                candidate_count += len(item['documents'])
                sizes.append((qt, dt))
            result = training.optimizer_step(encoder, optimizer, examples)
            for item, identity, size in zip(result['examples'], ids, sizes, strict=True):
                item.update(query_id=identity, query_tokens=size[0], document_tokens=size[1])
                if loss_transforms is not None:
                    # Read scalar receipts only; no extra model forward between updates.
                    item['objective_components'] = dict(loss_transforms[identity].last)
            durations.append(time.monotonic() - tick)
            result.update(step=step + 1, seconds=durations[-1],
                          reference_update=records[ids[0]]['reference_update'])
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if observer is not None:
                observer(step + 1, encoder, optimizer, result)
            if torch.cuda.max_memory_allocated() > 20 * 1024 ** 3:
                raise ValueError('training exceeded GPU allocation limit')
            task.get_logger().report_scalar('training', 'loss',
                float(np.mean([r['loss'] for r in result['examples']])), step + 1)
            completed = step + 1 - start
            if completed in (1, 24):
                prediction = projected_seconds(durations, end - start)
                threshold = stage_limit_seconds - 300
                io.write(output / f'canary-{completed}.json', dict(
                    passed=prediction < threshold, predicted_seconds=prediction,
                    observed_updates=completed, total_updates=end - start))
                if prediction >= threshold:
                    raise ValueError('training canary exceeds unchanged stage budget')
            if completed == 1 or completed % 8 == 0:
                print('NG71_TRAIN', arm, step + 1, end, sum(durations), flush=True)
    final = training.parameter_hash(encoder.model)
    if final == initial:
        raise ValueError('training checkpoint did not change')
    final_optimizer = save_optimizer(output / 'optimizer.pt', encoder, optimizer, end)
    encoder.model.save_pretrained(output / 'checkpoint')
    probes = [queries[q]['query'] for q in order[:2]]
    with torch.no_grad():
        query_probe = encoder.encode(probes, 'query').cpu()
        doc_probe = encoder.encode(examples[0]['documents'][:4], 'document').cpu()
    del encoder, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    encoder = training.Encoder(base, config, device='cuda', checkpoint=output / 'checkpoint')
    optimizer = training.make_optimizer(encoder)
    if training.parameter_hash(encoder.model) != final:
        raise ValueError('checkpoint state changed on reload')
    restore_optimizer(output / 'optimizer.pt', encoder, optimizer, end)
    with torch.no_grad():
        if (not torch.equal(query_probe, encoder.encode(probes, 'query').cpu())
                or not torch.equal(doc_probe, encoder.encode(
                    examples[0]['documents'][:4], 'document').cpu())):
            raise ValueError('query/document readout changed after reload')
    return dict(arm=arm, cumulative_steps=end, steps=end - start,
                query_exposures=(end - start) * 4, model_sha256=final,
                initial_model_sha256=initial, optimizer_sha256=io.sha(output / 'optimizer.pt'),
                optimizer_fingerprint=final_optimizer,
                initial_optimizer_fingerprint=initial_optimizer,
                previous_complete_sha256=io.sha(previous / 'complete.json') if previous else None,
                checkpoint_optimizer_replay_exact=True, tokens=tokens,
                candidate_pairs=candidate_count,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())


def encode_reference(base, config, output, checkpoint_phase, queries, documents):
    """Full-corpus A96 reference with TRAIN pilot queries only, never DEV."""
    prior = sealed(checkpoint_phase)
    if prior['arm'] != 'A' or prior['cumulative_steps'] != 96:
        raise ValueError('only the shared A96 checkpoint can refresh references')
    if (len(documents) != config['pilot']['corpus_documents']
            or len(queries) != config['pilot']['train_queries']
            or len({q['query_id'] for q in queries}) != len(queries)
            or any(q['split'] != 'TRAIN' for q in queries)):
        raise ValueError('reference corpus or TRAIN-only query surface changed')
    result = encode_checkpoint(base, config, output, checkpoint_phase, queries, documents)
    return dict(result, reference_update=96, reference_arm='A', new_dev_inference=False)


def encode_checkpoint(base, config, output, checkpoint_phase, queries, documents,
                      *, document_cache=None, stage_limit_seconds=5400):
    """Encode an approved surface; the controller enforces terminal barriers."""
    prior = sealed(checkpoint_phase)
    if type(stage_limit_seconds) is not int or not 600 < stage_limit_seconds <= 5400:
        raise ValueError('encoding stage budget must not relax existing bounds')
    if (prior['arm'] not in config['arms'] or prior['cumulative_steps'] not in (96, 192)
            or len(documents) != config['pilot']['corpus_documents']
            or len({q['query_id'] for q in queries}) != len(queries)
            or any(q['split'] not in ('TRAIN', 'DEV_NEW') for q in queries)):
        raise ValueError('unapproved checkpoint or query surface')
    if prior['cumulative_steps'] == 96 and any(q['split'] != 'TRAIN' for q in queries):
        raise ValueError('intermediate checkpoint may only encode TRAIN')
    return _encode_validated_checkpoint(base, config, output, checkpoint_phase, queries,
        documents, prior, document_cache=document_cache, stage_limit_seconds=stage_limit_seconds)


def validate_fixed_terminal_surface(queries):
    """Only the predeclared NG79 TRAIN-only1920-query observation surface."""
    expected = {'TRAIN_PILOT': 384, 'TRAIN_SENTINEL': 384, 'TRAIN_ADDED': 1152}
    if (len(queries) != 1920 or len({q['query_id'] for q in queries}) != 1920
            or any(q['split'] != 'TRAIN' for q in queries)
            or Counter(q['surface'] for q in queries) != Counter(expected)):
        raise ValueError('fixed terminal TRAIN observation surface changed')
    for surface, count in expected.items():
        if Counter(q['subset'] for q in queries if q['surface'] == surface) != Counter(
                {d: count // 3 for d in ('fever', 'hotpotqa', 'nq')}):
            raise ValueError('fixed terminal domain coverage changed')


def encode_fixed_terminal_checkpoint(base, config, output, checkpoint_phase, queries,
                                     documents, *, stage_limit_seconds=1800):
    """A fresh384-step encoding; old96/192 endpoint guards remain unchanged."""
    validate_fixed_terminal_surface(queries)
    prior = sealed(checkpoint_phase)
    if (stage_limit_seconds != 1800 or type(stage_limit_seconds) is not int
            or config['pilot']['updates'] != 384 or set(config['arms']) != {'R', 'B'}
            or prior['arm'] not in config['arms'] or prior['cumulative_steps'] != 384
            or prior['steps'] != 96 or prior['query_exposures'] != 384
            or len(documents) != config['pilot']['corpus_documents']
            or len(documents) != 233009):
        raise ValueError('unapproved fixed terminal checkpoint or corpus')
    return _encode_validated_checkpoint(base, config, output, checkpoint_phase, queries,
        documents, prior, document_cache=None, stage_limit_seconds=stage_limit_seconds)


def _encode_validated_checkpoint(base, config, output, checkpoint_phase, queries,
                                  documents, prior, *, document_cache, stage_limit_seconds):
    """Shared unchanged canonical batching, encoding, canary and save engine."""
    cached = None
    if document_cache is not None:
        cached = sealed(document_cache)
        if (cached['model_sha256'] != prior['model_sha256']
                or cached['counts']['document']['rows'] != len(documents)):
            raise ValueError('document cache does not match checkpoint or corpus')
    encoder = training.Encoder(base, config, device='cuda',
                               checkpoint=checkpoint_phase / 'checkpoint')
    if training.parameter_hash(encoder.model) != prior['model_sha256']:
        raise ValueError('reference checkpoint state changed')
    counts = {}
    for role, rows, key in (('document', documents, 'text'), ('query', queries, 'query')):
        if role == 'document' and cached is not None:
            # The caller additionally pins corpus/tokenizer/source provenance.
            shutil.copyfile(document_cache / 'document.npz', output / 'document.npz')
            if io.sha(output / 'document.npz') != io.sha(document_cache / 'document.npz'):
                raise ValueError('cached document copy changed')
            counts[role] = dict(cached['counts'][role])
            continue
        parts, tick = [], time.monotonic()
        batch = 96 if role == 'document' else 8
        for start in range(0, len(rows), batch):
            with torch.no_grad():
                values = encoder.encode([r[key] for r in rows[start:start + batch]], role)
            parts.append(sparse.csr_matrix(values.cpu().numpy()))
            if torch.cuda.max_memory_allocated() > 20 * 1024 ** 3:
                raise ValueError('reference encoding exceeded GPU limit')
            if start == batch * 7:
                prediction = (time.monotonic() - tick) * len(rows) / (batch * 8) * 1.5
                io.write(output / (role + '-canary.json'), dict(
                    passed=prediction < stage_limit_seconds - 600, predicted_seconds=prediction))
                if prediction >= stage_limit_seconds - 600:
                    raise ValueError('reference encoding canary exceeds stage budget')
            if start % (batch * 32) == 0:
                print('NG71_REFERENCE', role, min(start + batch, len(rows)),
                      len(rows), time.monotonic() - tick, flush=True)
        matrix = sparse.vstack(parts, format='csr')
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if (matrix.dtype != np.float32 or not matrix.has_canonical_format
                or not (matrix.data > 0).all() or not (np.diff(matrix.indptr) > 0).all()):
            raise ValueError('reference sparse encoding is not canonical')
        sparse.save_npz(output / (role + '.npz'), matrix)
        counts[role] = dict(nnz=matrix.nnz, rows=matrix.shape[0],
                            csr_bytes=matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        del matrix, parts
        gc.collect()
    if training.parameter_hash(encoder.model) != prior['model_sha256']:
        raise ValueError('encoding changed model parameters')
    io.write(output / 'query-ids.json', [q['query_id'] for q in queries])
    return dict(arm=prior['arm'], update=prior['cumulative_steps'],
                model_sha256=prior['model_sha256'], counts=counts,
                checkpoint_complete_sha256=io.sha(checkpoint_phase / 'complete.json'),
                reused_document_complete_sha256=(io.sha(document_cache / 'complete.json')
                                                  if document_cache is not None else None),
                locked_test_access=False,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
