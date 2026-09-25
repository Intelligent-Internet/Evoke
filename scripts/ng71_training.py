"""NG71 differentiable objective and bounded-microbatch training primitives.

No job is launched by importing this module. The execution layer must verify
frozen input/model manifests, corpus ranks and label provenance before use.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import ng71_ranking as reference


def prepare_pairs(record: dict, config: dict, *, uniform: bool = False) -> dict:
    """Freeze pair targets and coefficients independently of student scores."""
    positives = np.asarray(record['positive_mask'], dtype=bool)
    negatives = np.asarray(record['judged_negative_mask'], dtype=bool)
    if not all(type(x) is bool for x in record['positive_mask']
               + record['judged_negative_mask']):
        raise ValueError('labels must be boolean')
    ranks = np.asarray(record['global_ranks'])
    teacher = np.asarray(record['teacher_scores'], dtype=np.float64)
    params = {name: config[name] for name in (
        'ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor',
        'teacher_temperature', 'student_temperature',
    )}
    audit = reference.loss_and_gradient(
        np.zeros(len(positives)), positives, ranks, teacher, negatives,
        total_positives=record['total_positives'],
        corpus_size=record['corpus_size'], rank_scope=record['rank_scope'],
        metric_weights=not uniform, **params,
    )
    pair_indices, targets, coefficients = [], [], []
    for p in np.flatnonzero(positives):
        pending = []
        for n in np.flatnonzero(~positives):
            if negatives[n]:
                target, confidence = 1.0, 1.0
            else:
                margin = (teacher[p] - teacher[n]) / params['teacher_temperature']
                target = float(np.exp(-np.logaddexp(0.0, -margin)))
                confidence = 2 * target - 1
                if confidence <= 0:
                    continue
            weight = 1.0 if uniform else reference.pair_weight(
                int(ranks[p]), int(ranks[n]), record['total_positives'],
                **{k: params[k] for k in (
                    'ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor',
                )},
            )
            if weight > 0:
                pending.append((int(n), target, confidence, weight))
        denominator = sum(row[3] for row in pending)
        for n, target, confidence, weight in pending:
            pair_indices.append([int(p), n])
            targets.append(target)
            coefficients.append(
                confidence * weight / denominator / record['total_positives'])
    return {
        'indices': pair_indices, 'targets': targets, 'coefficients': coefficients,
        'candidate_count': len(positives),
        'student_temperature': params['student_temperature'],
        'total_positives': record['total_positives'],
        'supervised_positives': audit['supervised_positives'],
        'unresolved_pairs': audit['unresolved_pairs'],
        'uniform': uniform,
    }


def pair_loss(scores: torch.Tensor, pairs: dict) -> torch.Tensor:
    if scores.ndim != 1 or len(scores) != pairs['candidate_count']:
        raise ValueError('candidate score shape changed')
    if not torch.isfinite(scores).all():
        raise ValueError('non-finite scores')
    temperature = pairs['student_temperature']
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid student temperature')
    if not pairs['indices']:
        if pairs['targets'] or pairs['coefficients']:
            raise ValueError('empty indices have nonempty pair data')
        return scores.sum() * 0
    raw_indices = np.asarray(pairs['indices'])
    if raw_indices.dtype.kind not in 'iu':
        raise ValueError('pair indices must be integers')
    indices = torch.as_tensor(pairs['indices'], dtype=torch.long, device=scores.device)
    targets = scores.new_tensor(pairs['targets'])
    coefficients = scores.new_tensor(pairs['coefficients'])
    if indices.shape != (len(targets), 2) or coefficients.shape != targets.shape:
        raise ValueError('pair manifest shape changed')
    if not torch.isfinite(targets).all() or not torch.isfinite(coefficients).all():
        raise ValueError('non-finite pair manifest')
    if (targets <= .5).any() or (targets > 1).any() or (coefficients <= 0).any():
        raise ValueError('invalid preference or confidence')
    if (indices < 0).any() or (indices >= len(scores)).any():
        raise ValueError('pair index out of bounds')
    if (indices[:, 0] == indices[:, 1]).any():
        raise ValueError('self pair')
    margin = (scores[indices[:, 0]] - scores[indices[:, 1]]) / temperature
    return (F.binary_cross_entropy_with_logits(margin, targets, reduction='none')
            * coefficients).sum()


def legacy_loss(scores: torch.Tensor, target) -> torch.Tensor:
    target = torch.as_tensor(target, dtype=scores.dtype, device=scores.device)
    if (target.shape != scores.shape or not torch.isfinite(target).all()
            or (target < 0).any() or not torch.isfinite(scores).all()
            or not torch.isclose(target.sum(), target.new_tensor(1.))):
        raise ValueError('invalid legacy CE distribution')
    return -(target * scores.log_softmax(-1)).sum()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Encoder:
    """Device-explicit reproduction of frozen NG59's full sparse readout."""

    def __init__(self, base: Path, config: dict, device: str = 'cpu',
                 checkpoint: Path | None = None):
        from transformers import AutoTokenizer

        source = base / 'NG-0002/checkpoint_io.py'
        spec = importlib.util.spec_from_file_location('ng71_checkpoint_io', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source_model = checkpoint or base / config['base']['checkpoint']
        self.model = module.load_model(source_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base / config['base']['tokenizer'], local_files_only=True)
        self.rms = torch.as_tensor(
            np.load(base / config['base']['rms']), dtype=torch.float32, device=device)
        self.device = device
        self.config = config
        if checkpoint is None:
            if parameter_hash(self.model) != config['base']['state_sha256']:
                raise ValueError('initial model state changed')
        if self.model.training:
            raise ValueError('VJP replay requires the frozen eval-mode contract')

    def encode(self, texts: list[str], role: str) -> torch.Tensor:
        if role not in ('query', 'document') or not texts:
            raise ValueError('invalid role or empty text batch')
        query = role == 'query'
        size = self.config['optimizer'][role + '_microbatch']
        limit = self.config['base'][role + '_input_tokens']
        outputs = []
        for start in range(0, len(texts), size):
            tokens = self.tokenizer(
                texts[start:start + size], truncation=True, padding=True,
                max_length=limit, return_tensors='pt').to(self.device)
            hidden = self.model.base_model(**tokens).last_hidden_state
            values = self.model.lm_head(hidden).relu().log1p()
            values = values.masked_fill(
                ~tokens['attention_mask'].bool()[..., None], 0).amax(1)
            if query:
                denominator = values @ self.rms
                if not torch.isfinite(denominator).all() or (denominator <= 0).any():
                    raise ValueError('invalid fixed RMS normalization')
                # Preserve NG59's float32 operation order, not just its algebra.
                values = (self.config['base']['semantic_weight'] * values
                          / denominator[:, None])
            if not torch.isfinite(values).all() or (values.sum(1) <= 0).any():
                raise ValueError('invalid sparse readout')
            outputs.append(values)
        return torch.cat(outputs)


def make_optimizer(encoder: Encoder) -> torch.optim.Optimizer:
    config = encoder.config['optimizer']
    return torch.optim.AdamW([
        {'params': encoder.model.roberta.parameters(), 'lr': config['trunk_lr']},
        {'params': encoder.model.lm_head.parameters(), 'lr': config['head_lr']},
    ], betas=tuple(config['betas']), eps=config['eps'],
        weight_decay=config['weight_decay'])


def backward_query(encoder, query, documents, lexical, pairs=None, target=None,
                   *, accumulation=4, replay=True, loss_transform=None):
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('invalid accumulation')
    if (pairs is None) == (target is None):
        raise ValueError('exactly one of pair supervision and CE target is required')
    with torch.set_grad_enabled(not replay):
        codes = encoder.encode(documents, 'document')
    leaf = codes.detach().requires_grad_(True) if replay else codes
    query_code = encoder.encode([query], 'query')
    lexical = torch.as_tensor(lexical, dtype=leaf.dtype, device=leaf.device)
    if lexical.shape != (len(documents),) or not torch.isfinite(lexical).all():
        raise ValueError('invalid already-weighted lexical scores')
    scores = (leaf @ query_code.T).ravel() + lexical
    scores.retain_grad()
    loss = pair_loss(scores, pairs) if pairs is not None else legacy_loss(scores, target)
    if loss_transform is not None:
        loss = loss_transform(scores, loss)
        if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                or not loss.requires_grad or not torch.isfinite(loss)):
            raise ValueError('loss transform must return a finite differentiable scalar')
    (loss / accumulation).backward()
    if replay:
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise ValueError('missing document-output gradient')
        for start in range(0, len(documents), 4):
            fresh = encoder.encode(documents[start:start + 4], 'document')
            if not torch.equal(fresh.detach(), codes[start:start + 4]):
                raise ValueError('VJP forward replay changed')
            fresh.backward(leaf.grad[start:start + 4])
    return {
        'loss': float(loss.detach()), 'scores': scores.detach().cpu().tolist(),
        'score_gradient': scores.grad.detach().cpu().tolist(),
        'documents': len(documents),
        'query_nnz': int((query_code > 0).sum()),
        'document_nnz': int((codes > 0).sum()),
        'objective': ('uniform_soft_pair' if pairs.get('uniform', False)
                      else 'balanced_soft_pair') if pairs is not None else 'legacy_CE',
        'eligible_pairs': len(pairs['indices']) if pairs is not None else None,
        'supervised_positives': pairs['supervised_positives'] if pairs is not None else None,
        'vjp_replay_exact': replay,
    }


def optimizer_step(encoder, optimizer, examples):
    """Four actual exposures; fail rather than manufacture an empty update."""
    if len(examples) != encoder.config['optimizer']['queries_per_update']:
        raise ValueError('query accumulation changed')
    if not any(example.get('target') is not None or example.get('pairs', {}).get('indices')
               for example in examples):
        raise ValueError('entire batch has no eligible supervision')
    optimizer.zero_grad(set_to_none=True)
    records = [backward_query(encoder, **example, accumulation=len(examples))
               for example in examples]
    norm = torch.nn.utils.clip_grad_norm_(
        encoder.model.parameters(), encoder.config['optimizer']['gradient_clip_norm'],
        error_if_nonfinite=True)
    if not norm > 0:
        raise ValueError('zero batch gradient; no optimizer step executed')
    before = [p.detach().clone() for p in encoder.model.parameters()]
    optimizer.step()
    delta = sum(float((p.detach() - old).double().square().sum())
                for p, old in zip(encoder.model.parameters(), before)) ** .5
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('optimizer did not produce a finite parameter change')
    return {'examples': records, 'gradient_before_clip': float(norm),
            'update_l2': delta}
