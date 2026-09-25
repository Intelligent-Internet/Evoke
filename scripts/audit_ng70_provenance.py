#!/usr/bin/env python3
"""TRAIN-only original-positive provenance; no inference or qrel modification."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import unicodedata

import psutil
import requests


REPO = 'cfli/bge-full-data'
REVISION = '78f5c99b534a52824ab26bd24edda592eaed4c7a'
DOMAINS = ('fever', 'hotpotqa', 'nq')


def normalized(text):
    # No case folding, fuzzy matching, title stripping or semantic equivalence.
    return ' '.join(unicodedata.normalize('NFC', text).split())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def classify(query, originals):
    positives = [c for c in query['candidates'] if c['label'] == 'positive']
    assert positives, 'No current positives'
    assert len({c['key'] for c in positives}) == len(positives)
    positive_texts = {normalized(c['text']) for c in positives}
    variants = {frozenset(normalized(p) for p in row['pos']) for row in originals}
    if not variants:
        status, original = 'query_not_found', set()
    elif len(variants) != 1:
        status, original = 'ambiguous_original_query', set()
    else:
        original = set(next(iter(variants)))
        status = 'resolved' if original and original <= positive_texts else 'original_positive_missing'
    pairs = []
    for candidate in positives:
        text = normalized(candidate['text'])
        kind = 'unresolved'
        if status == 'resolved':
            kind = 'original_positive' if text in original else 'added_vs_original'
        pairs.append({'document_key': candidate['key'], 'source_doc_id': candidate.get('source_doc_id'),
                      'text_sha256': hashlib.sha256(candidate['text'].encode()).hexdigest(),
                      'provenance': kind})
    return {'query_id': query['query_id'], 'domain': query['subset'], 'status': status,
            'upstream_rows': len(originals), 'distinct_positive_variants': len(variants),
            'pairs': pairs}


def merge_train(selected, row):
    identity = row['query_id']
    if identity in selected:
        assert selected[identity] == row, 'Conflicting duplicate TRAIN query'
    else:
        selected[identity] = row


def selected_train(base):
    teacher = base / 'NG-0069/teacher-v1'
    complete = json.loads((teacher / 'complete.json').read_text())
    assert complete['passed'] and sha(teacher / 'targets.json') == complete['files']['targets.json']
    targets = json.loads((teacher / 'targets.json').read_text())
    wanted = {t['query_id'] for t in targets}
    assert len(wanted) == len(targets) == 6144
    selected = {}
    files = [base / 'NG-0059/data/train.jsonl', base / 'NG-0067/data/train.jsonl']
    for path in files:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['query_id'] not in wanted:
                    continue
                assert row['subset'] in DOMAINS
                merge_train(selected, row)
    assert set(selected) == wanted
    return [selected[t['query_id']] for t in targets], {
        str(p.relative_to(base)): sha(p) for p in files + [teacher / 'targets.json',
                                                         teacher / 'complete.json']}


def collect(queries, out):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data'
    response = requests.get(url, params={'limit': 1000}, timeout=30)
    response.raise_for_status()
    assert 'next' not in response.links, 'Unexpected metadata pagination'
    shards = [r for r in response.json() if r['type'] == 'file'
              and any(r['path'].startswith('data/' + d + '-') for d in DOMAINS)]
    assert len(shards) == 12
    save(out / 'upstream-shards.json', shards)
    wanted = {(q['subset'], normalized(q['query'])) for q in queries}
    found, counters, access = defaultdict(list), Counter(), []
    fs = HfFileSystem(token=False)
    for item in shards:
        domain = item['path'].split('/')[1].split('-')[0]
        remote = f'datasets/{REPO}@{REVISION}/' + item['path']
        with fs.open(remote, 'rb', block_size=1024 * 1024) as handle:
            parquet = pq.ParquetFile(handle)
            assert {'query', 'pos'} <= set(parquet.schema_arrow.names)
            row_number = 0
            # Column projection avoids downloading the large mined-negative lists.
            for batch in parquet.iter_batches(batch_size=256, columns=['query', 'pos'],
                                               use_threads=False):
                for source in batch.to_pylist():
                    key = (domain, normalized(source['query']))
                    if key in wanted:
                        found[key].append(dict(source, shard=item['path'], row=row_number))
                    row_number += 1
                assert psutil.Process().memory_info().rss < 2 * 1024 ** 3
            counters[domain] += row_number
            access.append({'path': item['path'], 'rows': row_number,
                           'full_file_sha_verified': False, 'columns': ['query', 'pos']})
        print('PROJECTED', item['path'], row_number, flush=True)
    save(out / 'matched-original-rows.json', [
        {'domain': domain, 'normalized_query': query, 'sources': sources}
        for (domain, query), sources in sorted(found.items())])
    save(out / 'column-access.json', access)
    return found, dict(counters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base, out = args.research_root.resolve(), args.output.resolve()
    assert not out.exists(), 'Never overwrite provenance attempts'
    queries, inputs = selected_train(base)
    out.mkdir(parents=True)
    save(out / 'inputs.json', {'files': inputs, 'source_sha256': sha(Path(__file__)),
         'upstream_repo': REPO, 'upstream_revision': REVISION,
         'selected_train_queries': 6144, 'model_inference': False,
         'wall_limit_seconds': 900, 'rss_limit_bytes': 2 * 1024 ** 3})
    os.environ.update(CLEARML_OFFLINE_MODE='1', CUDA_VISIBLE_DEVICES='',
                      CLEARML_CACHE_DIR=str(out / 'tracking-cache'))
    from clearml import Task
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0070/positive-provenance',
                     task_type=Task.TaskTypes.data_processing, reuse_last_task_id=False,
                     auto_connect_frameworks=False, auto_connect_arg_parser=False,
                     auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = {'task_id': task.id, 'actual_start': True, 'closed': False,
               'offline': True, 'remote_synced': False}
    save(out / 'clearml-start.json', receipt)
    task.connect({'inputs_sha256': sha(out / 'inputs.json')})
    started, failure = time.monotonic(), None

    def expired(signum, frame):
        raise TimeoutError('Provenance wall bound')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        originals, scanned = collect(queries, out)
        records = [classify(q, originals[(q['subset'], normalized(q['query']))]) for q in queries]
        domains = {}
        for domain in DOMAINS:
            part = [r for r in records if r['domain'] == domain]
            domains[domain] = {'queries': len(part),
                'query_status': dict(Counter(r['status'] for r in part)),
                'positive_provenance': dict(Counter(p['provenance'] for r in part for p in r['pairs']))}
        save(out / 'pair-provenance.json', records)
        save(out / 'results.json', {'domains': domains, 'source_rows_scanned': scanned,
             'upstream_revision': REVISION, 'source_full_file_sha_verified': False,
             'new_human_labels': 0, 'model_inference': False,
             'dev_queries_analyzed': 0, 'locked_test_scored': False,
             'qrels_modified': False, 'overall_goal_qualified': False})
        for name, digest in inputs.items():
            assert sha(base / name) == digest
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        signal.alarm(0)
        task.close()
        save(out / 'clearml.json', dict(receipt, closed=True))
        save(out / 'exit.json', {'exit_code': 1 if failure else 0, 'error': failure,
                                 'seconds': time.monotonic() - started})
    save(out / 'complete.json', {'passed': True, 'files': {
        str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}})
    print('NG70_PROVENANCE_COMPLETE', json.dumps(domains), flush=True)


if __name__ == '__main__':
    main()
