#!/usr/bin/env python3
"""Extend frozen lexical surfaces without fitting on new DEV or TEST."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np
import psutil
from scipy import sparse


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def weight_documents(counts, full_lengths, idf, average):
    result = counts.astype(np.float64, copy=True)
    row = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    tf = result.data.copy()
    result.data = idf[result.indices] * tf / (
        tf + 1.5 * (.25 + .75 * full_lengths[row] / average))
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base, out = args.research_root.resolve(), args.output.resolve()
    assert not out.exists(), 'Never overwrite a lexical attempt'
    started = time.monotonic()
    data, reference = base / 'NG-0067/data', base / 'NG-0059/data'
    selected = read(base / 'NG-0069/teacher-v1/selection.json')['indices']
    paths = [data / name for name in ('documents.jsonl', 'train.jsonl',
             'train-pools.json', 'train-labels.json', 'dev.jsonl', 'dev-labels.json')]
    manifest = read(base / 'NG-0067/complete.json')['files']
    for path in paths:
        assert sha(path) == manifest['data/' + path.name], path
    dependencies = [reference / name for name in ('lexical-query.npz',
        'lexical-document.npz', 'train-lexical.json', 'sealed.jsonl',
        'positive-labels.json')]
    dependencies += [base / 'NG-0008/ng8_lexical.py',
                     base / 'NG-0008/native/libtext.dylib',
                     base / 'NG-0009/surfaces/vocabulary.json',
                     base / 'NG-0009/surfaces/train-rms.npy',
                     base / 'NG-0069/teacher-v1/selection.json']
    source = base / 'NG-0008/ng8_lexical.py'
    spec = importlib.util.spec_from_file_location('frozen_lexical', source)
    lex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lex)
    tokenizer = lex.Tokenizer()
    vocab = read(base / 'NG-0009/surfaces/vocabulary.json')
    rms = np.load(base / 'NG-0009/surfaces/train-rms.npy')[:len(vocab)]
    docs = rows(data / 'documents.jsonl')
    train = rows(data / 'train.jsonl')
    queries = [train[i] for i in selected]
    queries += rows(data / 'dev.jsonl') + rows(reference / 'sealed.jsonl')
    train_labels = read(data / 'train-labels.json')
    labels = [train_labels[i] for i in selected]
    labels += read(data / 'dev-labels.json')
    labels += read(reference / 'positive-labels.json')[1536:]
    assert len(docs) == 233009 and len(queries) == len(labels) == 7872
    assert len({r['query_id'] for r in queries}) == len(queries)
    out.mkdir(parents=True)
    frozen = {str(p.relative_to(base)): sha(p) for p in paths + dependencies}
    write(out / 'inputs.json', {'files': frozen, 'source_sha256': sha(Path(__file__))})
    fit, _, unknown = lex.count_matrix([r['text'] for r in docs[:14539]],
                                       tokenizer, vocab)
    assert unknown == 0
    average = np.asarray(fit.astype(np.float64).sum(1)).mean()
    df = np.bincount(fit.indices, minlength=len(vocab))
    idf = np.log1p((14539 - df + .5) / (df + .5))
    cached = sparse.load_npz(reference / 'lexical-document.npz')
    assert cached.shape == (59111, len(vocab))
    probe_ids = np.linspace(34104, 59110, 64, dtype=int)
    probe_text = [docs[i]['text'] for i in probe_ids]
    probe, _, _ = lex.count_matrix(probe_text, tokenizer, vocab)
    weighted = weight_documents(probe, np.array([len(tokenizer(t)) for t in probe_text]),
                                idf, average)
    assert (weighted != cached[probe_ids]).nnz == 0, 'Old BM25 transform changed'
    parts, oov_d, total_tokens = [cached], 0, 0
    for start in range(59111, len(docs), 4096):
        texts = [r['text'] for r in docs[start:start + 4096]]
        counts, _, unknown = lex.count_matrix(texts, tokenizer, vocab)
        lengths = np.array([len(tokenizer(t)) for t in texts])
        parts.append(weight_documents(counts, lengths, idf, average))
        oov_d += unknown
        total_tokens += int(lengths.sum())
        assert time.monotonic() - started < 1800, 'Lexical CPU time bound'
        assert psutil.Process().memory_info().rss < 16 * 1024 ** 3, 'Host RSS bound'
        print('lexical documents', start + len(texts), '/', len(docs), flush=True)
    ld = sparse.vstack(parts, format='csr')
    qc, _, oov_q = lex.count_matrix([r['query'] for r in queries], tokenizer, vocab)
    lq = qc.multiply((.1 / lex.scale(qc, rms))[:, None]).tocsr()
    old_q = sparse.load_npz(reference / 'lexical-query.npz')
    assert (lq[:1536] != old_q[:1536]).nnz == 0
    assert (lq[-192:] != old_q[1536:]).nnz == 0
    all_pools = read(data / 'train-pools.json')
    pools = [all_pools[i] for i in selected]
    lexical = [(lq.getrow(i).astype(np.float64)
                @ ld[pool].astype(np.float64).T).toarray().ravel().tolist()
               for i, pool in enumerate(pools)]
    assert lexical[:1536] == read(reference / 'train-lexical.json')
    metadata = [{'query_id': r['query_id'], 'query': r['query'],
                 'subset': r['subset'], 'split': 'TRAIN' if i < 6144 else
                 'DEV_NEW' if i < 7680 else 'DEV_EXPOSED'}
                for i, r in enumerate(queries)]
    sparse.save_npz(out / 'lexical-document.npz', ld)
    sparse.save_npz(out / 'lexical-query.npz', lq)
    for name, value in (('queries.json', metadata), ('labels.json', labels),
                        ('train-pools.json', pools), ('train-lexical.json', lexical)):
        write(out / name, value)
    for name, digest in frozen.items():
        assert sha(base / name) == digest, name
    write(out / 'results.json', {
        'passed': True, 'documents': len(docs), 'train_queries': 6144,
        'dev_new_queries': 1536, 'dev_exposed_queries': 192,
        'locked_test_queries_accessed': 0, 'fit_document_count': 14539,
        'vocabulary_size': len(vocab), 'idf_refitted': False,
        'old_prefix_and_train_scores_exact': True,
        'new_document_oov_tokens': oov_d, 'new_document_tokens': total_tokens,
        'query_oov_tokens': oov_q,
        'empty_lexical_queries': int(np.count_nonzero(np.diff(lq.indptr) == 0)),
        'seconds': time.monotonic() - started,
        'maxrss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    write(out / 'complete.json', {'passed': True,
          'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
