"""NG80 matched score-field controls, separated from resource orchestration."""

import numpy as np
import torch
from torch.nn import functional as F


def field_loss(scores, teacher, scale=1.):
    if (scores.ndim != 1 or scores.shape != teacher.shape or len(scores) < 2
            or not torch.isfinite(scores).all() or not torch.isfinite(teacher).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('invalid field supervision')
    error = scores / scale - teacher
    return (error - error.mean()).square().mean()


def protective_loss(scores, teacher, positive):
    """Uniform per-positive preference, not the old rank-weighted D replay."""
    if scores.ndim != 1 or scores.shape != teacher.shape or positive.shape != scores.shape:
        raise ValueError('protective shape')
    if positive.dtype != torch.bool or not positive.any():
        raise ValueError('positive mask required')
    if not torch.isfinite(scores).all() or not torch.isfinite(teacher).all():
        raise ValueError('nonfinite preference')
    margins = teacher[positive, None] - teacher[None, ~positive]
    target = torch.sigmoid(margins / .04)
    eligible = margins > 0
    confidence = (2 * target - 1).clamp_min(0)
    student = scores[positive, None] - scores[None, ~positive]
    loss = F.binary_cross_entropy_with_logits(student, target, reduction='none')
    per_positive = (loss * confidence * eligible).sum(1) / eligible.sum(1).clamp_min(1)
    return per_positive.mean()


class Readout:
    """The same eval-mode mature parent, with a diagnostic dense projection."""

    def __init__(self, encoder, kind, weights=None):
        if kind not in ('F', 'D', 'S', 'P'):
            raise ValueError('unknown arm')
        self.encoder, self.kind = encoder, kind
        self.projection = None
        if kind in ('F', 'D'):
            if weights is None or weights.shape != (encoder.model.config.hidden_size + 1, 1024):
                raise ValueError('TRAIN-fitted mean projection required')
            self.projection = torch.nn.Linear(weights.shape[0] - 1, 1024).to(encoder.device)
            with torch.no_grad():
                self.projection.weight.copy_(torch.as_tensor(weights[:-1].T))
                self.projection.bias.copy_(torch.as_tensor(weights[-1]))
            for p in encoder.model.lm_head.parameters():
                p.requires_grad_(False)
            if kind == 'F':
                for p in encoder.model.parameters():
                    p.requires_grad_(False)
            else:
                # Dense controls must also adapt any tied input embeddings.
                for p in encoder.model.base_model.parameters():
                    p.requires_grad_(True)

    def parameters(self):
        result = [p for p in self.encoder.model.parameters() if p.requires_grad]
        if self.projection is not None:
            result += list(self.projection.parameters())
        return result

    def encode(self, texts, role):
        if self.kind in ('S', 'P'):
            return self.encoder.encode(texts, role)
        output = []
        limit = self.encoder.config['base'][role + '_input_tokens']
        for start in range(0, len(texts), 4):
            tokens = self.encoder.tokenizer(texts[start:start + 4], truncation=True,
                padding=True, max_length=limit, return_tensors='pt').to(self.encoder.device)
            hidden = self.encoder.model.base_model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            values = self.projection(pooled)
            if not torch.isfinite(values).all() or (values.norm(dim=1) <= 1e-12).any():
                raise ValueError('invalid dense readout')
            output.append(F.normalize(values, dim=1))
        return torch.cat(output)

    def optimizer(self):
        config = self.encoder.config['optimizer']
        groups = []
        trunk = [p for p in self.encoder.model.roberta.parameters() if p.requires_grad]
        if trunk:
            groups.append({'params': trunk, 'lr': config['trunk_lr']})
        head = self.projection if self.projection is not None else self.encoder.model.lm_head
        trunk_ids = {id(p) for p in trunk}
        groups.append({'params': [p for p in head.parameters()
                                  if p.requires_grad and id(p) not in trunk_ids],
                       'lr': config['head_lr']})
        return torch.optim.AdamW(groups, betas=tuple(config['betas']), eps=config['eps'],
                                weight_decay=config['weight_decay'])


def backward(model, query, documents, teacher, positive, lexical, scale,
             accumulation=4, replay=True):
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('invalid accumulation')
    if len(documents) < 2 or len(teacher) != len(documents):
        raise ValueError('invalid candidate alignment')
    with torch.set_grad_enabled(not replay):
        codes = model.encode(documents, 'document')
    leaf = codes.detach().requires_grad_(True) if replay else codes
    q = model.encode([query], 'query')
    semantic = (leaf @ q.T).ravel()
    target = semantic.new_tensor(teacher)
    if model.kind == 'P':
        lexical = semantic.new_tensor(lexical)
        if lexical.shape != semantic.shape or not torch.isfinite(lexical).all():
            raise ValueError('invalid already-weighted lexical scores')
        scores = semantic + lexical
        loss = protective_loss(scores, target,
                               torch.as_tensor(positive, dtype=torch.bool, device=scores.device))
    else:
        scores = semantic
        loss = field_loss(scores, target, scale)
    scores.retain_grad()
    (loss / accumulation).backward()
    if replay:
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise ValueError('missing document VJP')
        for start in range(0, len(documents), 4):
            fresh = model.encode(documents[start:start + 4], 'document')
            if not torch.equal(fresh.detach(), codes[start:start + 4]):
                raise ValueError('fresh replay differs from current output')
            fresh.backward(leaf.grad[start:start + 4])
    return {'loss': float(loss.detach()), 'scores': scores.detach().cpu().tolist(),
            'score_gradient': scores.grad.detach().cpu().tolist(),
            'teacher': list(map(float, teacher)), 'positive': list(map(bool, positive)),
            'documents': len(documents), 'vjp_replay_exact': replay,
            'query_nnz': int((q != 0).sum()), 'document_nnz': int((codes != 0).sum())}
