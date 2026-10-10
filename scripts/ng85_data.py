"""Frozen TRAIN-only data and shared score mathematics for NG85."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


DOMAINS = ('fever', 'hotpotqa', 'nq', 'fiqa')
ANCHORS = {
    'NG-0080/readout-v1/prepare/queries.json':
        'bd20960ea03f8f7121ba836dd226a405d32d78254c5e0551688dbe448049f0b4',
    'NG-0080/readout-v1/prepare/selection.json':
        'f445a4746edb117decbf39be921436589008f6b30a4fca916796576d3bf33372',
    'NG-0080/readout-v1/prepare/labels.json':
        'ec681fe479c4f551445bd26d5474d3c0404406df0c0ccfdf47f1579cca1fb0c1',
    'NG-0067/data/documents.jsonl':
        '16a008e102028b2cd04872dc24bfc60e0f2f9e508a563748020df81499a5b961',
    'NG-0083/operator-v1/prepare/bank.npz':
        '296d885e72f3145478ed5c8d847d3c1c9613a6379cee23164d4cdb837d93fa32',
    'NG-0084/fiqa-train-v1/manifest.json':
        '6d1da51fbc148930b97907abdb79c3f8324b5a07e1eb679cec15e3c7a73dc499',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def hash_order(ids, salt):
    return sorted(ids, key=lambda x: hashlib.sha256(
        (salt + '/' + str(x)).encode()).hexdigest())


def stable_top(scores, k):
    scores = np.asarray(scores)
    require(scores.ndim == 1 and np.isfinite(scores).all(), 'invalid scores')
    require(0 < k <= len(scores), 'invalid top k')
    # Partition, then explicitly resolve the threshold tie by document ID.
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    ids = np.flatnonzero(scores > threshold).tolist()
    ids += np.flatnonzero(scores == threshold)[:k - len(ids)].tolist()
    return sorted(ids, key=lambda i: (-float(scores[i]), i))


def target_distribution(scores):
    """Identical per-query affine calibration; no teacher-specific tuning."""
    value = np.asarray(scores, dtype=np.float64)
    require(value.ndim == 1 and len(value) > 1 and np.isfinite(value).all(),
            'invalid teacher vector')
    value = (value - value.mean()) / max(float(value.std()), 1e-6)
    value = np.exp(value - value.max())
    return value / value.sum()


def training_order(queries, seed, epochs=2):
    groups = {d: [i for i, q in enumerate(queries)
                  if q['split'] == 'fit' and q['subset'] == d] for d in DOMAINS}
    require(len(set(map(len, groups.values()))) == 1 and groups['fiqa'],
            'four balanced TRAIN domains required')
    output = []
    for epoch in range(epochs):
        ordered = [hash_order(groups[d], f'NG85/{seed}/{epoch}/{d}')
                   for d in DOMAINS]
        output.extend(i for group in zip(*ordered, strict=True) for i in group)
    return output


def prefix_visible(text, tokenizers, limits):
    """A literal source prefix that fits every encoder, never decode/rewrite."""
    require(len(tokenizers) == len(limits) and text.strip(), 'empty text')
    original = text
    for _ in range(32):
        lengths = [len(t(text, truncation=False, verbose=False)['input_ids'])
                   for t in tokenizers]
        if all(n <= limit for n, limit in zip(lengths, limits, strict=True)):
            require(original.startswith(text), 'visible text is not source prefix')
            return text, lengths
        ratio = min(limit / n for n, limit in zip(lengths, limits, strict=True))
        end = min(len(text) - 1, max(1, int(len(text) * ratio * .95)))
        text = text[:end]
    raise ValueError('cannot establish common visible prefix')


def fiqa_integrity(documents, queries):
    """Quarantine absent evidence before sampling; never alter source qrels."""
    all_ids = {str(d['_id']) for d in documents}
    require(len(all_ids) == len(documents), 'duplicate canonical document ID')
    empty = {str(d['_id']) for d in documents
             if not (d['title'] + '\n' + d['text']).strip()}
    excluded, eligible = [], set()
    for i, query in enumerate(queries):
        positives = {str(p) for p in query['positive_ids']}
        require(positives and positives <= all_ids, 'missing canonical positive ID')
        missing_text = sorted(positives & empty)
        if missing_text:
            excluded.append(dict(query_id=query['query_id'],
                                 positive_ids_without_text=missing_text))
        else:
            eligible.add(i)
    return ([d for d in documents if str(d['_id']) not in empty], eligible,
            dict(empty_document_ids=sorted(empty), excluded_queries=excluded,
                 selection_before_any_model_scoring=True,
                 source_qrels_modified=False))


def prepare(base, out):
    from transformers import AutoTokenizer, T5Tokenizer

    for name, digest in ANCHORS.items():
        require(sha(base / name) == digest, 'changed input: ' + name)
    old = base / 'NG-0080/readout-v1/prepare'
    original, split, labels = [read(old / name) for name in
                              ('queries.json', 'selection.json', 'labels.json')]
    bank = np.load(base / 'NG-0083/operator-v1/prepare/bank.npz')
    bank_rows = {}
    for j, index in enumerate(bank['query_indices']):
        start, end = bank['offsets'][j:j + 2]
        bank_rows[int(index)] = bank['document_ids'][start:end].tolist()
    require(set(bank_rows) == set(split['train']), 'bank identity mismatch')
    queries = []
    for i, record in enumerate(original):
        queries.append(dict(record, split='fit' if i in bank_rows else 'validation',
            positives=labels[i], document_ids=bank_rows.get(i),
            label_provenance='NG67 inherited RLHN known-positive; not new human judgments',
            judged_negative_ids=[]))
    documents = rows(base / 'NG-0067/data/documents.jsonl')
    documents = [dict(source='ng67', source_id=d['key'], text=d['text'])
                 for d in documents]
    require(len(documents) == 233009, 'old corpus count')
    fiqa = base / 'NG-0084/fiqa-train-v1'
    manifest = read(fiqa / 'manifest.json')
    # Pin the actual canonical payloads through the already audited manifest.
    expected = manifest['outputs']
    for name in ('train.jsonl', 'documents.jsonl'):
        require(sha(fiqa / name) == expected[name], 'FiQA payload mismatch')
    offset = len(documents)
    extra = rows(fiqa / 'documents.jsonl')
    fiqa_queries = rows(fiqa / 'train.jsonl')
    require(len(extra) == 57638 and len(fiqa_queries) == 5500, 'FiQA source count')
    extra, eligible, exclusion = fiqa_integrity(extra, fiqa_queries)
    require(len(extra) == 57600 and len(eligible) == 5466, 'FiQA integrity inventory changed')
    write(out / 'fiqa-integrity.json', exclusion)
    ids = {str(row['_id']): offset + i for i, row in enumerate(extra)}
    require(len(ids) == 57600, 'FiQA searchable corpus IDs')
    documents.extend(dict(source='fiqa', source_id=str(d['_id']),
        text=(d['title'] + '\n' + d['text']).strip()) for d in extra)
    order = [i for i in hash_order(range(len(fiqa_queries)), 'NG85/fiqa-v1')
             if i in eligible]
    for j, i in enumerate(order[:4608]):
        q = fiqa_queries[i]
        queries.append(dict(query_id=q['query_id'], query=q['query'], subset='fiqa',
            split='fit' if j < 4096 else 'validation',
            positives=[ids[str(d)] for d in q['positive_ids']], document_ids=None,
            label_provenance='BeIR fiqa-qrels canonical TRAIN positive',
            judged_negative_ids=[]))
    require(Counter(q['subset'] for q in queries if q['split'] == 'fit')
            == dict.fromkeys(DOMAINS, 4096), 'TRAIN count changed')
    require(len({q['query_id'] for q in queries}) == len(queries), 'duplicate query ID')
    tokenizers = [AutoTokenizer.from_pretrained(base / folder, local_files_only=True,
                   trust_remote_code=True) for folder in
                  ('NG-0002/model', 'NG-0014/dense-model')]
    tokenizers.append(T5Tokenizer.from_pretrained(base / 'NG-0084/rankt5-hub',
                                                 local_files_only=True))
    for role, values, field, limits in (
            ('query', queries, 'query', (64, 64, 96)),
            ('document', documents, 'text', (256, 256, 384))):
        filename = 'queries.jsonl' if role == 'query' else 'documents.jsonl'
        with (out / filename).open('x') as stream:
            for i, row in enumerate(values):
                text, lengths = prefix_visible(row[field], tokenizers, limits)
                result = dict(row, original_text_sha256=hashlib.sha256(
                    row[field].encode()).hexdigest(), token_lengths=lengths,
                    source_characters=len(row[field]), visible_characters=len(text))
                result[field] = text
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if i % 2048 == 0:
                    write(out / 'progress.json', dict(role=role, completed=i,
                        total=len(values)))
                    print('NG85_CONTEXT', role, i, len(values), flush=True)
    visible = rows(out / 'queries.jsonl')
    fit = {normalized(q['query']) for q in visible if q['split'] == 'fit'}
    require(len(fit) == 16384, 'duplicate visible TRAIN queries; stop, do not pad')
    collisions = [q['query_id'] for q in visible
                  if q['split'] == 'validation' and normalized(q['query']) in fit]
    require(not collisions, 'visible query crosses split: ' + str(collisions[:5]))
    write(out / 'results.json', dict(fit_queries=16384, validation_queries=2048,
        documents=len(documents), context_contract='literal_source_prefix_no_further_truncation',
        validation_independent=False, locked_test_access=False,
        global_prior_exposure_audit_complete=False, source_anchors=ANCHORS,
        fiqa_empty_documents_excluded=38, fiqa_queries_with_unavailable_gold_excluded=34,
        source_qrels_modified=False))
