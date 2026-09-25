#!/usr/bin/env python3
"""Bounded NG71 preparation, not a scientific four-arm training launch.

Data mode audits TRAIN without inference. Model mode exercises the real base
on one TRAIN text fixture, including one disposable optimizer update. It does
not estimate retrieval quality. Witness mode reuses verified full-corpus NG69
encodings for the zero-update TRAIN P0 diagnostic after NG69 independent review.
Pilot-witness mode extends that diagnostic to the fixed full TRAIN pilot and
independently checks four objectives in score space, without encoder inference.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

import psutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def model_check(base, config, output, device='cpu'):
    import numpy as np
    import torch
    import transformers

    import ng71_training as training
    from ng71_data import ANCHORS

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in ('NG-0059/data/train.jsonl',):
        if sha(base / name) != ANCHORS[name]:
            raise ValueError('TRAIN fixture changed')
    with (base / 'NG-0059/data/train.jsonl').open() as stream:
        row = json.loads(next(stream))
    documents = [next(c['text'] for c in row['candidates'] if c['label'] == label)
                 for label in ('positive', 'negative')]
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('explicit CUDA preflight requires a visible GPU')
    encoder = training.Encoder(base, config, device=device)
    initial = training.parameter_hash(encoder.model)
    # Fixed preference is a gradient fixture, not a fabricated human judgment.
    pairs = {'indices': [[0, 1]], 'targets': [.8], 'coefficients': [.6],
             'candidate_count': 2, 'student_temperature': 1.,
             'supervised_positives': 1}
    example = dict(query=row['query'], documents=documents,
                   lexical=[.01, .02], pairs=pairs)

    def legacy_forward(texts, role):
        tokens = encoder.tokenizer(
            texts, truncation=True, padding=True,
            max_length=64 if role == 'query' else 256,
            return_tensors='pt').to(device)
        hidden = encoder.model.base_model(**tokens).last_hidden_state
        value = encoder.model.lm_head(hidden).relu().log1p()
        value = value.masked_fill(~tokens['attention_mask'].bool()[..., None], 0).amax(1)
        return .9 * value / (value @ encoder.rms)[:, None] if role == 'query' else value

    with torch.no_grad():
        for role, texts in (('query', [row['query']]), ('document', documents)):
            if not torch.equal(encoder.encode(texts, role), legacy_forward(texts, role)):
                raise ValueError('real base forward differs from NG59 formula')
    encoder.model.zero_grad(set_to_none=True)
    direct = training.backward_query(encoder, **example, accumulation=1, replay=False)
    gradients = {n: p.grad.clone() for n, p in encoder.model.named_parameters() if p.grad is not None}
    encoder.model.zero_grad(set_to_none=True)
    replay = training.backward_query(encoder, **example, accumulation=1, replay=True)
    if direct['scores'] != replay['scores'] or direct['loss'] != replay['loss']:
        raise ValueError('real base objective changed during replay')
    actual = {n: p.grad for n, p in encoder.model.named_parameters() if p.grad is not None}
    if actual.keys() != gradients.keys():
        raise ValueError('real base gradient coverage changed')
    errors = []
    error_squared = reference_squared = 0.
    for name, gradient in gradients.items():
        delta = (gradient - actual[name]).double()
        errors.append(float(delta.abs().max()))
        error_squared += float(delta.square().sum())
        reference_squared += float(gradient.double().square().sum())
    relative = (error_squared / reference_squared) ** .5
    if not np.isfinite(relative) or relative > 1e-5:
        raise ValueError('real base VJP gradient error exceeds frozen 1e-5 relative L2')
    del gradients, actual
    ce_example = {k: v for k, v in example.items() if k != 'pairs'}
    ce_example['target'] = [.9, .1]
    encoder.model.zero_grad(set_to_none=True)
    ce = training.backward_query(encoder, **ce_example, accumulation=1)
    ce_scores = torch.tensor(ce['scores'], dtype=torch.float32)
    expected_ce_gradient = ce_scores.softmax(-1) - torch.tensor(ce_example['target'])
    observed_ce_gradient = torch.tensor(ce['score_gradient'])
    if not torch.allclose(observed_ce_gradient, expected_ce_gradient, rtol=1e-6, atol=1e-8):
        raise ValueError('legacy CE score gradient changed')
    optimizer = training.make_optimizer(encoder)
    updated = training.optimizer_step(encoder, optimizer, [example] * 4)
    with torch.no_grad():
        probe = encoder.encode([row['query']], 'query').clone()
    after = training.parameter_hash(encoder.model)
    if initial == after:
        raise ValueError('canary weights did not change')
    checkpoint = output / 'disposable-engineering-checkpoint'
    encoder.model.save_pretrained(checkpoint)
    optimizer_state = optimizer.state_dict()
    torch.save(optimizer_state, output / 'optimizer.pt')
    reloaded = training.Encoder(base, config, device=device, checkpoint=checkpoint)
    restored = training.make_optimizer(reloaded)
    restored.load_state_dict(torch.load(output / 'optimizer.pt', weights_only=True))
    if training.parameter_hash(reloaded.model) != after:
        raise ValueError('checkpoint state changed on reload')
    with torch.no_grad():
        if not torch.equal(probe, reloaded.encode([row['query']], 'query')):
            raise ValueError('checkpoint readout changed on reload')
    restored_state = restored.state_dict()
    if restored_state['param_groups'] != optimizer_state['param_groups']:
        raise ValueError('optimizer groups changed on reload')
    for key, fields in optimizer_state['state'].items():
        for name, value in fields.items():
            if not torch.equal(value, restored_state['state'][key][name]):
                raise ValueError('optimizer tensor changed on reload')
    peak_gpu = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
    if peak_gpu > 20 * 1024 ** 3:
        raise ValueError('CUDA allocation exceeded 20 GiB')
    return {
        'passed': True, 'purpose': 'engineering_fixture_not_scientific_training',
        'query_id': row['query_id'], 'documents': 2, 'actual_optimizer_updates': 1,
        'unique_train_queries': 1, 'repeated_fixture_exposures': 4,
        'artificial_fixed_pair_target': .8, 'quality_claim_allowed': False,
        'initial_state_sha256': initial, 'disposable_state_sha256': after,
        'trainable_parameters': sum(p.numel() for p in encoder.model.parameters()),
        'legacy_forward_bit_exact': True, 'vjp_forward_bit_exact': True,
        'vjp_gradient_relative_l2': relative,
        'vjp_gradient_max_abs': max(errors), 'gradient_tensors_checked': len(errors),
        'gradient_tolerance_predeclared': 1e-5,
        'checkpoint_reload_bit_exact': True, 'optimizer_reload_bit_exact': True,
        'legacy_ce_score_gradient_max_abs': float(
            (expected_ce_gradient - observed_ce_gradient).abs().max()),
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'numpy': np.__version__, 'transformers': transformers.__version__,
                    'platform': platform.platform(), 'device': device, 'tf32': False,
                    'gpu': torch.cuda.get_device_name() if device == 'cuda' else None},
        'peak_gpu_allocated_bytes': peak_gpu,
        'update': updated, 'locked_test_access': False,
        'scientific_training_enabled': False,
    }


def worker(args):
    # ClearML's import hooks must not observe a half-initialized torch package.
    if args.mode in ('model', 'cuda-model', 'witness', 'pilot-witness'):
        __import__('torch')
    from clearml import Task

    source = json.loads((args.output / 'source-frozen.json').read_text())
    for name, expected in source['files'].items():
        if sha(args.output / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    config = json.loads((args.output / 'config.json').read_text())
    Task.set_offline(True)
    task = Task.init(
        project_name='Evoke-NG', task_name='NG-0071/' + args.output.name,
        task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, post_hoc=False,
                   offline=True, remote_synced=False, closed=False,
                   reason='online auth endpoint returned HTTP 401 at preflight')
    write(args.output / 'clearml-start.json', receipt)
    task.connect({'mode': args.mode, 'source_sha256': sha(args.output / 'source-frozen.json'),
                  'scientific_training_enabled': False})
    try:
        if args.mode == 'data':
            from ng71_data import audit
            result = audit(args.research_root, config)
        elif args.mode in ('model', 'cuda-model'):
            result = model_check(args.research_root, config, args.output,
                                 device='cuda' if args.mode == 'cuda-model' else 'cpu')
        else:
            from ng71_snapshot import build
            result = build(args.research_root, config, args.output,
                           surface='pilot' if args.mode == 'pilot-witness' else 'canary')
        write(args.output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__,
                         status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(args.output / 'clearml.json', receipt)


def run_controller(args):
    available = psutil.virtual_memory().available / 1024 ** 3
    witness_mode = args.mode in ('witness', 'pilot-witness')
    cuda_mode = args.mode == 'cuda-model'
    rss_gib = 16 if witness_mode or cuda_mode else 8
    seconds = 5400 if witness_mode else 600
    if available <= (24 if witness_mode or cuda_mode else 8) or shutil.disk_usage(args.output.parent).free <= 40 * 1024 ** 3:
        raise ValueError('insufficient resources for bounded CPU preparation')
    args.output.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parent
    for name in ('ng71_preflight.py', 'ng71_training.py', 'ng71_ranking.py',
                 'ng71_data.py', 'audit_ng70_provenance.py',
                 'ng71_snapshot.py', 'ng71_witness.py', 'ng71_diagnostics.py',
                 'ng71_execution.py'):
        shutil.copy2(source / name, args.output / name)
    shutil.copy2(args.config, args.output / 'config.json')
    if witness_mode:
        for name in ('ng0071-global-preflight-contract.zh.md',
                     'ng0071-global-boundary-ranking-plan.zh.md'):
            shutil.copy2(args.config.parent / name, args.output / name)
    dependencies = {}
    if witness_mode:
        from ng71_snapshot import dependencies as snapshot_dependencies
        dependencies = snapshot_dependencies(
            args.research_root, json.loads(args.config.read_text()))
    if args.mode in ('model', 'cuda-model'):
        config = json.loads(args.config.read_text())
        for name in (config['base']['checkpoint'], config['base']['tokenizer']):
            for path in sorted((args.research_root / name).glob('*')):
                if path.is_file():
                    dependencies[str(path.relative_to(args.research_root))] = sha(path)
        for name in (config['base']['rms'], 'NG-0002/checkpoint_io.py'):
            dependencies[name] = sha(args.research_root / name)
        legacy = args.research_root / config['base']['legacy_dependencies']
        if sha(legacy) != config['base']['legacy_dependencies_sha256']:
            raise ValueError('historical dependency manifest changed')
        expected = {str((legacy.parent / p).resolve().relative_to(args.research_root)): digest
                    for p, digest in json.loads(legacy.read_text())['files'].items()}
        for name, digest in dependencies.items():
            if name in expected and digest != expected[name]:
                raise ValueError('historical model/tokenizer/RMS dependency changed: ' + name)
        required = [config['base']['rms'], 'NG-0002/checkpoint_io.py',
                    config['base']['checkpoint'] + '/model.safetensors',
                    config['base']['tokenizer'] + '/tokenizer.json',
                    config['base']['tokenizer'] + '/tokenizer_config.json']
        if any(name not in expected or name not in dependencies for name in required):
            raise ValueError('required historical model dependency missing')
        dependencies[config['base']['legacy_dependencies']] = sha(legacy)
    write(args.output / 'dependencies.json', dependencies)
    write(args.output / 'source-frozen.json', {
        'mode': args.mode, 'scientific_training_enabled': False,
        'files': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}})
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('CLEARML_TASK_', 'TRAINS_TASK_', 'CLEARML_PROC_', 'TRAINS_PROC_')):
            env.pop(key)
    env.update(CLEARML_OFFLINE_MODE='1', CLEARML_CACHE_DIR=str(args.output / 'tracking-cache'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               CUDA_VISIBLE_DEVICES=str(args.gpu) if cuda_mode else '')
    started = time.time()
    peak = 0
    error = None
    with (args.output / 'stdout.log').open('x') as log:
        process = subprocess.Popen([
            sys.executable, str(args.output / 'ng71_preflight.py'), '--worker',
            '--mode', args.mode, '--research-root', str(args.research_root),
            '--output', str(args.output)], env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        write(args.output / 'started.json', {
            'controller_pid': os.getpid(), 'worker_pid': process.pid,
            'started_unix': started, 'limit_seconds': seconds,
            'physical_gpu': args.gpu if cuda_mode else None})
        interrupted = []

        def request_stop(signum, frame):
            interrupted.append(signum)

        previous = {sig: signal.signal(sig, request_stop)
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        while process.poll() is None:
            try:
                if interrupted:
                    error = f'preparation interrupted by signal {interrupted[0]}'
                    break
                parent = psutil.Process(process.pid)
                rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
                peak = max(peak, rss)
                if rss > rss_gib * 1024 ** 3 or time.time() - started > seconds:
                    error = f'CPU preparation exceeded {rss_gib} GiB RSS or {seconds} seconds'
                    break
                if cuda_mode:
                    used = subprocess.check_output([
                        'nvidia-smi', f'--id={args.gpu}',
                        '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                        text=True, timeout=5).strip()
                    if int(used) > 20 * 1024:
                        error = 'selected GPU total used memory exceeded 20 GiB'
                        break
            except psutil.NoSuchProcess:
                pass
            except BaseException as exc:
                error = f'preparation monitor failed: {type(exc).__name__}: {exc}'
                break
            time.sleep(.5)
        if error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        code = process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    deadline = time.monotonic() + 10
    closed = False
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            closed = True
            break
        time.sleep(.1)
    receipt = dict(exit_code=code, error=error, owned_group_closed=closed,
                   elapsed_seconds=time.time() - started, peak_tree_rss_bytes=peak,
                   initial_host_available_gib=available,
                   preparation_bounds={'cpu_threads': 4, 'tree_rss_gib': rss_gib,
                                       'wall_seconds': seconds, 'gpu_used': cuda_mode},
                   scientific_training_bounds_relaxed=False)
    write(args.output / 'exit.json', receipt)
    tracking = json.loads((args.output / 'clearml.json').read_text()) if (args.output / 'clearml.json').exists() else {}
    passed = code == 0 and error is None and closed and tracking.get('closed') is True
    write(args.output / 'complete.json', {
        'passed': passed,
        'files': {str(p.relative_to(args.output)): sha(p)
                  for p in sorted(args.output.rglob('*')) if p.is_file()}})
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise SystemExit('Preparation failed; preserve this attempt')


def controller(args):
    if args.mode != 'cuda-model':
        return run_controller(args)
    if args.gpu is None:
        raise ValueError('CUDA model preflight requires an explicit physical GPU')
    import fcntl

    # The same cooperative reservation is used by the preceding NG69 pipeline.
    with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = subprocess.check_output([
            'nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(',')
        if int(info[0]) >= 100 or int(info[1]) != 0:
            raise ValueError('requested GPU is occupied; do not evict other work')
        return run_controller(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True,
                        choices=('data', 'model', 'cuda-model', 'witness', 'pilot-witness'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.research_root = args.research_root.resolve(strict=True)
    if args.worker:
        for name, expected in json.loads((args.output / 'dependencies.json').read_text()).items():
            if sha(args.research_root / name) != expected:
                raise ValueError('model dependency changed: ' + name)
        worker(args)
    else:
        if args.config is None:
            parser.error('--config required for the controller')
        controller(args)


if __name__ == '__main__':
    main()
