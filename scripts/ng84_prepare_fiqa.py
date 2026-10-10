"""Prepare canonical FiQA TRAIN positives, without scoring or test qrels."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

from ng84_teacher import require, sha, write


QREL_SHA = 'e46a99529aa61b12e086a5c057e4e0ecaeda9502e53d5fe855efce778f1ec59a'
QUERY_SHA = 'e9b3863724ffd538d8efff19684ef8ae495bdbddd96b35e81bdc39e3e22860d6'
CORPUS_SHA = 'b36bb07dab3bfad488190818828e749a52faf5ba391c1d51dac8b06064576f99'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def canonical_rows(qrels, queries, corpus_ids):
    positives = defaultdict(set)
    for row in qrels:
        require(float(row['score']) > 0, 'this source has only positive qrels')
        require(row['corpus-id'] in corpus_ids, 'positive absent from canonical corpus')
        positives[row['query-id']].add(row['corpus-id'])
    query_map = {row['_id']: row['text'] for row in queries}
    require(len(query_map) == len(queries) and set(query_map) == set(positives),
            'query rows must be exactly original TRAIN IDs')
    output = []
    for key in sorted(positives):
        text = query_map[key]
        require(bool(normalized(text)), 'empty training query')
        output.append(dict(query_id='fiqa:' + key, original_query_id=key, query=text,
            subset='fiqa', public_split='train', positive_ids=sorted(positives[key]),
            judged_negative_ids=[], unjudged_policy='unknown_not_hard_negative',
            label_source='BeIR/fiqa-qrels/train.tsv',
            exposure_status='prior_project_exposure_not_yet_audited',
            normalized_query_sha256=hashlib.sha256(normalized(text).encode()).hexdigest()))
    return output


def prepare(source, rlhn, output):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    require(not output.exists(), 'fresh data unit required')
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    qrel = source / 'fiqa-qrels/train.tsv'
    query = source / 'fiqa/queries/queries-00000-of-00001.parquet'
    corpus = source / 'fiqa/corpus/corpus-00000-of-00001.parquet'
    require(sha(qrel) == QREL_SHA and sha(query) == QUERY_SHA
            and sha(corpus) == CORPUS_SHA, 'canonical TRAIN source changed')
    with qrel.open() as stream:
        qrels = list(csv.DictReader(stream, delimiter='\t'))
    ids = sorted({row['query-id'] for row in qrels})
    # Only TRAIN query rows are exposed to Python, although the source is shared.
    queries = pq.read_table(query, filters=[('_id', 'in', ids)], columns=['_id', 'text']).to_pylist()
    documents = pq.read_table(corpus, columns=['_id', 'title', 'text']).to_pylist()
    corpus_ids = {r['_id'] for r in documents}
    require(len(corpus_ids) == len(documents), 'duplicate canonical document ID')
    rows = canonical_rows(qrels, queries, corpus_ids)
    originals = Counter(normalized(q['text']) for q in queries)
    counts, lineage, shards = Counter(), Counter(), []
    for path in sorted(rlhn.rglob('*.parquet')):
        shards.append(dict(path=str(path), sha256=sha(path)))
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=65536, columns=['subset', 'query']):
            counts.update(batch.column(0).to_pylist())
            selected = batch.filter(pc.equal(batch.column(0), 'fiqa'))
            for text in selected.column(1).to_pylist():
                count = originals[normalized(text)]
                key = ('unique_original_train_match' if count == 1 else
                       'ambiguous_original_train_match' if count > 1 else 'not_in_original_train')
                lineage[key] += 1
    require(counts['fiqa'] == 5496 and lineage['unique_original_train_match'] == 5496,
            'unexpected RLHN FiQA lineage; do not silently admit changed sources')
    output.mkdir(parents=True)
    for name, values in (('train.jsonl', rows), ('documents.jsonl', documents)):
        with (output / name).open('x') as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    manifest = dict(schema='NG84_FiQA_canonical_TRAIN_v1',
        qrel_revision='252958f2d646e22cab6d0c72dd3f0d5de6d0655a',
        corpus_query_revision='979c07a7cb5ccc6ca009792241fa1250b98055dd',
        inputs={str(p.relative_to(source)): sha(p) for p in (qrel, query, corpus)},
        rlhn_shards=shards, rlhn_subset_counts=dict(counts), rlhn_fiqa_lineage=dict(lineage),
        train_queries=len(rows), canonical_documents=len(documents),
        known_positive_memberships=sum(len(r['positive_ids']) for r in rows),
        source_qrel_rows=len(qrels), source_code_sha256=sha(Path(__file__)),
        used_rlhn_relabels_as_human_qrels=False, test_qrels_loaded=False,
        retrieval_inference_executed=False, training_split_frozen=False,
        historical_exposure_audit_complete=False, independent_holdout=False,
        upstream_declared_license='CC-BY-SA-4.0; full derivative release review pending',
        outputs={n: sha(output / n) for n in ('train.jsonl', 'documents.jsonl')})
    write(output / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in (
        'train_queries', 'canonical_documents', 'known_positive_memberships', 'rlhn_fiqa_lineage')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--rlhn-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.rlhn_root, args.output)
