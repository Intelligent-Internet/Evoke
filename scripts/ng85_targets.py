"""Shared visible-text encoding, candidate bank and sharded RankT5 targets."""

from array import array
from collections import Counter
import inspect
from pathlib import Path
import time

import numpy as np

from ng85_data import hash_order, read, require, rows, sha, stable_top, write


def dense(base, run, out, task):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ng71_training import parameter_hash

    folder = base / 'NG-0014/dense-model'
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True,
                                              local_files_only=True)
    model, info = AutoModel.from_pretrained(folder, trust_remote_code=True,
        local_files_only=True, dtype=torch.float32, attn_implementation='sdpa',
        output_loading_info=True)
    require(not any(info.get(k) for k in ('missing_keys', 'unexpected_keys',
        'mismatched_keys', 'error_msgs')), 'PPLX loading mismatch')
    require(sha(Path(inspect.getfile(type(model)))) == sha(folder / 'modeling.py'),
            'PPLX executable differs from checkpoint')
    expected = read(base / 'NG-0059/recovery/teacher/complete.json')['model_state_sha256']
    require(parameter_hash(model) == expected, 'PPLX parameter hash mismatch')
    model.eval().cuda()
    started = time.monotonic()
    with torch.inference_mode():
        for role, name, field, limit in (
                ('query', 'queries', 'query', 64),
                ('document', 'documents', 'text', 256)):
            records = rows(run / 'prepare' / (name + '.jsonl'))
            values = np.lib.format.open_memmap(out / (role + '.npy'), mode='w+',
                dtype=np.float32, shape=(len(records), 1024))
            tick = time.monotonic()
            for i in range(0, len(records), 8):
                texts = [r[field] for r in records[i:i + 8]]
                tokens = tokenizer(texts, truncation=False, padding=True,
                                   return_tensors='pt').to('cuda')
                require(tokens.input_ids.shape[1] <= limit, 'PPLX extra truncation required')
                hidden = model(**tokens, use_cache=False).last_hidden_state
                mask = tokens.attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                require(torch.isfinite(pooled).all() and (pooled.norm(dim=1) > 0).all(),
                        'invalid dense codes')
                values[i:i + len(texts)] = torch.nn.functional.normalize(
                    pooled, dim=1).cpu().numpy()
                if i % 2048 == 0:
                    values.flush()
                    write(out / 'progress.json', dict(role=role, completed=i + len(texts),
                        total=len(records), role_seconds=time.monotonic() - tick))
                    task.get_logger().report_scalar('encode', role, i + len(texts), i)
                    print('NG85_DENSE', role, i + len(texts), len(records), flush=True)
                if i == 504:
                    estimate = 1.5 * (time.monotonic() - tick) * len(records) / 512 + 600
                    require(estimate < 8 * 3600, 'dense role exceeds predeclared budget')
            values.flush()
    require(parameter_hash(model) == expected, 'PPLX mutated')
    write(out / 'results.json', dict(model_state_sha256=expected,
        seconds=time.monotonic() - started, max_vram_bytes=torch.cuda.max_memory_allocated(),
        no_further_truncation=True, dimensions=1024, locked_test_access=False))


def lexical_corpus(documents, tokenizer):
    from scipy import sparse

    vocabulary, indices, counts, indptr = {}, array('i'), array('f'), array('q', [0])
    for i, document in enumerate(documents):
        terms = Counter(tokenizer(document['text']))
        for term, count in terms.items():
            column = vocabulary.setdefault(term, len(vocabulary))
            indices.append(column)
            counts.append(count)
        indptr.append(len(indices))
        if i % 8192 == 0:
            print('NG85_LEXICAL', i, len(documents), flush=True)
    matrix = sparse.csr_matrix((np.asarray(counts), np.asarray(indices), np.asarray(indptr)),
                              shape=(len(documents), len(vocabulary)), dtype=np.float32)
    matrix.sort_indices()
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    df = np.bincount(matrix.indices, minlength=matrix.shape[1])
    idf = np.log1p((len(documents) - df + .5) / (df + .5))
    rows_id = np.repeat(np.arange(len(documents)), np.diff(matrix.indptr))
    denominator = matrix.data + 1.5 * (.25 + .75 * lengths[rows_id] / lengths.mean())
    matrix.data = (idf[matrix.indices] * matrix.data / denominator).astype(np.float32)
    rms = np.sqrt(np.asarray(matrix.power(2).mean(axis=0)).ravel())
    return matrix, vocabulary, rms


def bank(base, run, out, task):
    from scipy import sparse
    from ng80_lexical import Tokenizer

    queries = rows(run / 'prepare/queries.jsonl')
    documents = rows(run / 'prepare/documents.jsonl')
    library = base / 'NG-0080/lexical-runtime-v2/libtext.so'
    old = read(library.parent / 'complete.json')
    require(old['passed'] and sha(library) == old['files']['libtext.so'], 'native library changed')
    tokenizer = Tokenizer(library)
    matrix, vocabulary, rms = lexical_corpus(documents, tokenizer)
    sparse.save_npz(out / 'bm25.npz', matrix)
    np.save(out / 'bm25-rms.npy', rms)
    write(out / 'vocabulary.json', vocabulary)
    q = np.load(run / 'dense/query.npy', mmap_mode='r')
    d = np.load(run / 'dense/document.npy', mmap_mode='r')
    require(q.shape == (len(queries), 1024) and d.shape == (len(documents), 1024),
            'dense identity shapes')
    background = hash_order(range(len(documents)), 'NG85/background-v1')[:16]
    offsets, ids_flat, lexical_flat, pplx_flat = [0], [], [], []
    q_indices, q_values, q_indptr = [], [], [0]
    started = time.monotonic()
    for i, row in enumerate(queries):
        terms = Counter(tokenizer(row['query']))
        columns = [vocabulary[t] for t in terms if t in vocabulary]
        weights = np.asarray([terms[t] for t in terms if t in vocabulary], dtype=np.float32)
        denominator = max(float(weights @ rms[columns]), 1e-12)
        weights = .1 * weights / denominator
        query = sparse.csr_matrix((weights, ([0] * len(columns), columns)),
                                  shape=(1, matrix.shape[1]))
        q_indices.extend(columns)
        q_values.extend(weights.tolist())
        q_indptr.append(len(q_indices))
        candidates = row['document_ids']
        if candidates is None:
            ds = np.asarray(d @ q[i]).ravel()
            bs = (matrix @ query.T).toarray().ravel()
            candidates = sorted(set(row['positives']) | set(stable_top(ds, 32))
                                | set(stable_top(bs, 32)) | set(background))
        require(len(candidates) == len(set(candidates))
                and set(row['positives']) <= set(candidates)
                and all(0 <= j < len(documents) for j in candidates), 'candidate contract')
        ids_flat.extend(candidates)
        offsets.append(len(ids_flat))
        pplx_flat.extend((d[candidates] @ q[i]).astype(np.float64).tolist())
        lexical_flat.extend((matrix[candidates] @ query.T).toarray().ravel().tolist())
        if i % 256 == 0:
            write(out / 'progress.json', dict(completed=i, total=len(queries),
                                             seconds=time.monotonic() - started))
            task.get_logger().report_scalar('bank', 'queries', i, i)
            print('NG85_BANK', i, len(queries), flush=True)
    sparse.save_npz(out / 'lexical-queries.npz', sparse.csr_matrix(
        (q_values, q_indices, q_indptr), shape=(len(queries), matrix.shape[1]), dtype=np.float32))
    np.savez(out / 'bank.npz', offsets=np.asarray(offsets, dtype=np.int64),
        document_ids=np.asarray(ids_flat, dtype=np.int32),
        lexical=np.asarray(lexical_flat, dtype=np.float32),
        pplx=np.asarray(pplx_flat, dtype=np.float64))
    write(out / 'results.json', dict(queries=len(queries), documents=len(documents),
        pairs=len(ids_flat), vocabulary=len(vocabulary), native_tokenizer_sha256=sha(library),
        lexical_document_nnz=matrix.nnz, known_positives_preserved=True,
        full_background_is_complete_beir=False, locked_test_access=False))


def teacher_batches(texts, tokenizer, batch, length_sorted=False):
    """Group within one query; carry original positions for score restoration."""
    require(batch > 0 and len(texts) > 0, 'invalid teacher batch input')
    order = list(range(len(texts)))
    if length_sorted:
        lengths = [len(x) for x in tokenizer(texts, truncation=False)['input_ids']]
        require(len(lengths) == len(texts), 'teacher tokenizer row mismatch')
        order.sort(key=lambda i: (lengths[i], i))
    for start in range(0, len(order), batch):
        positions = order[start:start + batch]
        yield positions, [texts[i] for i in positions]


def rankt5(base, run, out, shard, task):
    import sentencepiece
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    folder = base / 'NG-0084/rankt5-hub'
    manifest = read(run / 'ng84-rankt5-sha256.json')
    for name, digest in manifest.items():
        require(sha(folder / name) == digest, 'RankT5 checkpoint file changed')
    tokenizer = T5Tokenizer.from_pretrained(folder, local_files_only=True)
    sp = sentencepiece.SentencePieceProcessor(model_file=str(folder / 'spiece.model'))
    probe = 'Query: evidence? Document: A study.'
    require(tokenizer(probe)['input_ids'] == sp.encode(probe) + [tokenizer.eos_token_id],
            'RankT5 tokenizer differs from SentencePiece')
    model, loading = T5ForConditionalGeneration.from_pretrained(folder,
        local_files_only=True, dtype=torch.float32, output_loading_info=True)
    require(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys',
        'mismatched_keys', 'error_msgs')), 'RankT5 loading mismatch')
    model.cuda().eval()
    token_id = tokenizer.convert_tokens_to_ids('<extra_id_10>')
    require(token_id == 32089, 'RankT5 relevance token changed')
    queries = rows(run / 'prepare/queries.jsonl')
    documents = rows(run / 'prepare/documents.jsonl')
    bank_values = np.load(run / 'bank/bank.npz')
    execution = run / 'execution.json'
    settings = read(execution) if execution.exists() else {}
    batch = settings.get('rankt5_batch_size', 2)
    length_sorted = settings.get('rankt5_length_sorted', False)
    require(batch in (2, 8), 'unregistered teacher batch')
    indices = [i for i, q in enumerate(queries) if q['split'] == 'fit'][shard::4]
    pair_total = sum(int(bank_values['offsets'][i + 1] - bank_values['offsets'][i])
                     for i in indices)
    outputs, offsets = [], [0]
    start = time.monotonic()
    with torch.inference_mode():
        for number, i in enumerate(indices):
            left, right = bank_values['offsets'][i:i + 2]
            ids = bank_values['document_ids'][left:right]
            prompts = ['Query: ' + queries[i]['query'] + ' Document: ' + documents[j]['text']
                       for j in ids]
            query_scores = np.full(len(ids), np.nan, dtype=np.float64)
            batches = teacher_batches(prompts, tokenizer, batch, length_sorted)
            for batch_number, (positions, texts) in enumerate(batches):
                tokens = tokenizer(texts, truncation=False, padding=True,
                                   return_tensors='pt').to('cuda')
                require(tokens.input_ids.shape[1] <= 512, 'RankT5 extra truncation required')
                decoder = torch.full((len(texts), 1), model.config.decoder_start_token_id,
                                     dtype=torch.long, device='cuda')
                values = model(**tokens, decoder_input_ids=decoder).logits[:, 0, token_id]
                require(torch.isfinite(values).all(), 'nonfinite RankT5 targets')
                if number == 0 and batch_number == 0:
                    generated = model.generate(**tokens, max_new_tokens=1, do_sample=False,
                        return_dict_in_generate=True, output_scores=True)
                    torch.testing.assert_close(values, generated.scores[0][:, token_id],
                                               rtol=1e-5, atol=1e-5)
                    write(out / 'parity.json', dict(passed=True, original_sentencepiece=True,
                        forward_generate_max_error=float(
                            (values - generated.scores[0][:, token_id]).abs().max())))
                query_scores[positions] = values.cpu().double().numpy()
            require(np.isfinite(query_scores).all(), 'missing restored teacher score')
            outputs.extend(query_scores.tolist())
            offsets.append(len(outputs))
            elapsed = time.monotonic() - start
            if number % 32 == 0:
                write(out / 'progress.json', dict(completed_queries=number + 1,
                    total_queries=len(indices), completed_pairs=len(outputs), total_pairs=pair_total,
                    seconds=elapsed, estimated_total_seconds=elapsed * pair_total / len(outputs)))
                print('NG85_RANKT5', shard, number + 1, len(indices), round(elapsed), flush=True)
                task.get_logger().report_scalar('teacher', 'queries', number + 1, number)
            if number == 31:
                projected = 1.5 * elapsed * pair_total / len(outputs) + 600
                write(out / 'canary.json', dict(queries=32, pairs=len(outputs),
                    seconds=elapsed, guarded_seconds=projected,
                    batch_size=batch, passed=projected < 8 * 3600,
                    budget_seconds=8 * 3600))
                require(projected < 8 * 3600,
                        'RankT5 shard cannot fit predeclared eight-hour budget')
    np.savez(out / 'scores.npz', query_indices=np.asarray(indices, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64), scores=np.asarray(outputs, dtype=np.float64))
    write(out / 'results.json', dict(shard=shard, queries=len(indices), pairs=len(outputs),
        seconds=time.monotonic() - start, no_further_truncation=True, batch_size=batch,
        length_sorted=length_sorted,
        max_vram_bytes=torch.cuda.max_memory_allocated(), locked_test_access=False,
        bank_sha256=sha(run / 'bank/bank.npz')))


def execute(base, run, phase, task):
    from ng85_run import closed

    closed(run, 'prepare')
    out = run / phase
    if phase == 'bank':
        closed(run, 'dense')
        return bank(base, run, out, task)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.device_count() == 1, 'exactly one GPU required')
    if phase == 'dense':
        return dense(base, run, out, task)
    closed(run, 'bank')
    return rankt5(base, run, out, int(phase.split('-')[1]), task)
