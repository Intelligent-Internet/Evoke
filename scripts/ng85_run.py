"""NG85 immutable phase orchestration, cooperative locks and owned watchdogs."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ng85_data import ANCHORS, read, require, sha, write


PHASES = ('prepare', 'dense', 'bank', *(f'rankt5-{i}' for i in range(4)),
          *(f'train-{a}-{s}' for a in ('A', 'B') for s in (85001, 85002)))


def freeze(base, run):
    require(run.parent == base / 'NG-0085' and not run.exists(), 'new NG85 run only')
    for name, digest in ANCHORS.items():
        require(sha(base / name) == digest, 'changed source: ' + name)
    run.mkdir(parents=True)
    source = Path(__file__).parent
    names = ['ng85_run.py', 'ng85_queue.py', 'ng85_data.py', 'ng85_targets.py', 'ng85_train.py',
             'ng80_training.py', 'ng80_lexical.py', 'ng84-rankt5-sha256.json',
             'ng71_training.py', 'ng71_ranking.py', 'ng71_execution.py',
             'ng71_diagnostics.py', 'ng71_preflight.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    reports = source.parent / 'docs/research-sae/reports/ng0001-ng0099'
    shutil.copy2(reports / 'ng0071-ranking-config.json', run / 'config.json')
    shutil.copy2(reports / 'ng0085-student-ab.zh.md', run / 'protocol.md')
    dependencies = dict(ANCHORS)
    for name in ('NG-0002/checkpoint_io.py', 'NG-0031/encoded/mlm-raw-rms.npy',
                 'NG-0059/recovery/teacher/complete.json',
                 'NG-0080/lexical-runtime-v2/libtext.so'):
        # The Linux library is verified in the remote phase, not on the Mac.
        if (base / name).exists():
            dependencies[name] = sha(base / name)
    write(run / 'inputs.json', dict(protocol='NG85_teacher_only_AB_v1',
        dependencies=dependencies,
        source={p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()},
        fit_queries=16384, epochs=2, seeds=[85001, 85002],
        preprocessing='quarantine_empty_FiQA_evidence_before_scoring_v2',
        searchable_documents=290609,
        lexical_weight=.1, semantic_weight=.9, teacher_temperature='per_query_std',
        student_temperature='detached_per_query_std',
        explicit_judged_negative_count=0, reliability_loss_weight=0.,
        cost_regularizer_weight=0., cost_promotion_max_ratio=1.5,
        validation_independent=False, locked_test_access=False,
        production_deployment=False, automatic_research_expansion=False))


def verify(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG85_teacher_only_AB_v1'
            and not manifest['locked_test_access'], 'NG85 protocol mismatch')
    for root, entries in ((run, manifest['source']),
                          (base, manifest['dependencies'])):
        for name, digest in entries.items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and sha(path) == digest, 'changed frozen input: ' + name)
    return manifest


def closed(run, phase):
    folder = run / phase
    record = read(folder / 'complete.json')
    require(record['passed'] and read(folder / 'exit.json')['exit_code'] == 0,
            'unclosed prerequisite: ' + phase)
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder / 'complete.json'}
    require(actual == set(record['files']), 'phase inventory changed: ' + phase)
    for name, digest in record['files'].items():
        path = folder / name
        require(path.resolve().is_relative_to(folder.resolve()) and sha(path) == digest,
                'changed phase payload: ' + phase + '/' + name)
    return read(folder / 'results.json')


def gpu_state(gpu):
    value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return tuple(int(x.strip()) for x in value.strip().split(','))


def continue_frozen(base, run, parent, batch_probe=None):
    """Copy verified completed inputs, never overwrite a failed attempt."""
    old = verify(base, parent)
    require(parent.parent == base / 'NG-0085', 'NG85 parent required')
    require(run.parent == parent.parent and not run.exists(), 'new NG85 run only')
    phases = ('prepare', 'dense', 'bank')
    for phase in phases:
        closed(parent, phase)
    evidence = None
    if batch_probe is not None:
        import numpy as np

        require(batch_probe.parent == base / 'NG-0085', 'NG85 probe required')
        evidence = read(batch_probe / 'results.json')
        probe_inputs = read(batch_probe / 'inputs.json')
        require(probe_inputs['inputs_sha256'] == sha(parent / 'inputs.json')
                and probe_inputs['batches'] == [2, 8]
                and probe_inputs['queries'] == 32 and probe_inputs['dtype'] == 'float32'
                and probe_inputs['length_sorted']
                and evidence['passed'] and evidence['equivalent']
                and evidence['metrics']['8']['guarded_seconds'] < 8 * 3600,
                'teacher batch probe did not pass frozen budget/parity')
        reference = np.load(batch_probe / 'batch-2.npy')
        candidate = np.load(batch_probe / 'batch-8.npy')
        require(probe_inputs['source_sha256'] == sha(
                    batch_probe / 'ng85_teacher_batch_probe.py')
                and probe_inputs['targets_source_sha256'] == sha(
                    Path(__file__).parent / 'ng85_targets.py')
                and probe_inputs['targets_source_sha256'] == sha(
                    batch_probe / 'ng85_targets.py')
                and reference.ndim == 1 and reference.size > 0
                and reference.shape == candidate.shape
                and np.isfinite(reference).all() and np.isfinite(candidate).all()
                and np.allclose(reference, candidate, rtol=1e-5, atol=1e-5),
                'teacher batch probe payload mismatch')
    run.mkdir()
    for name in old['source']:
        require(Path(name).name == name, 'continuation expects flat source inventory')
        shutil.copy2(parent / name, run / name)
    new = dict(old, source=dict(old['source']))
    for name in ('ng85_run.py', 'ng85_queue.py'):
        shutil.copy2(Path(__file__).parent / name, run / name)
        new['source'][name] = sha(run / name)
    if evidence is not None:
        name = 'ng85_targets.py'
        shutil.copy2(Path(__file__).parent / name, run / name)
        new['source'][name] = sha(run / name)
        probe_files = ('inputs.json', 'results.json', 'batch-2.npy', 'batch-8.npy',
                       'ng85_teacher_batch_probe.py', 'ng85_targets.py')
        dependencies = dict(new['dependencies'])
        for name in probe_files:
            path = batch_probe / name
            dependencies[str(path.relative_to(base))] = sha(path)
        new['dependencies'] = dependencies
        write(run / 'execution.json', dict(rankt5_batch_size=8, rankt5_length_sorted=True,
            teacher_precision='float32', teacher_wall_budget_seconds=8 * 3600,
            batch_probe=batch_probe.name, probe_results_sha256=sha(
                batch_probe / 'results.json'), training_settings_unchanged=True))
        new['source']['execution.json'] = sha(run / 'execution.json')
    reused = {}
    for phase in phases:
        shutil.copytree(parent / phase, run / phase)
        closed(run, phase)
        reused[phase] = sha(parent / phase / 'complete.json')
    from ng85_queue import (CAMPAIGN_SECONDS, GPU_ORDER, MAX_PARALLEL,
                            POLL_SECONDS, QUEUE_WAIT_SECONDS)
    write(run / 'continuation.json', dict(parent=parent.name,
        parent_inputs_sha256=sha(parent / 'inputs.json'), reused_phases=reused,
        gpu_order=GPU_ORDER, max_parallel=MAX_PARALLEL, poll_seconds=POLL_SECONDS,
        queue_wait_seconds=QUEUE_WAIT_SECONDS, campaign_seconds=CAMPAIGN_SECONDS,
        scientific_settings_unchanged=True, teacher_batch_probe=(
            batch_probe.name if batch_probe is not None else None), compute_retries=False,
        queue_only_prestart_busy=True))
    new['source']['continuation.json'] = sha(run / 'continuation.json')
    write(run / 'inputs.json', new)
    verify(base, run)


def worker(base, run, phase):
    from clearml import Task

    verify(base, run)
    out = run / phase
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0085/' + phase,
        reuse_last_task_id=False, auto_connect_frameworks=False,
        auto_connect_arg_parser=False, auto_connect_streams=False,
        auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, offline=True,
                   remote_synced=False, closed=False)
    write(out / 'clearml-start.json', receipt)
    try:
        task.connect(dict(phase=phase, inputs_sha256=sha(run / 'inputs.json')))
        if phase == 'prepare':
            from ng85_data import prepare
            prepare(base, out)
        elif phase in ('dense', 'bank') or phase.startswith('rankt5-'):
            from ng85_targets import execute
            execute(base, run, phase, task)
        else:
            from ng85_train import execute
            execute(base, run, phase, task)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(out / 'clearml.json', receipt)


def supervise(base, run, phase, gpu):
    import fcntl
    import psutil
    from ng85_queue import ResourceBusy, check_resources, log_admission

    verify(base, run)
    cpu = phase in ('prepare', 'bank')
    require((gpu is None) == cpu, 'CPU phase or exactly one explicit GPU')
    require(cpu or gpu in range(4), 'invalid GPU')
    lock_name = '.ng85-cpu.lock' if cpu else f'.ng-gpu-{gpu}.lock'
    with (base / lock_name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            log_admission(run, phase, admitted=False, gpu=gpu, reason='cooperative_lock')
            raise ResourceBusy('cooperative GPU/CPU lock busy') from exc
        check_resources(run, phase, gpu)
        out = run / phase
        out.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', CLEARML_OFFLINE_MODE='1',
            CLEARML_CACHE_DIR=str(out / 'tracking-cache'))
        cap = 12 * 3600 if phase.startswith('train-') else 8 * 3600
        rss_cap = 48 if cpu else 24
        process, error, peak, group_closed = None, None, 0, False
        stopped = []
        handlers = {s: signal.signal(s, lambda v, frame: stopped.append(v))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        start = time.monotonic()
        try:
            with (out / 'stdout.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-B', str(run / 'ng85_run.py'),
                    'worker', '--base', str(base), '--run', str(run), '--phase', phase],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(out / 'started.json', dict(pid=process.pid, gpu=gpu,
                    started_unix=time.time(), wall_cap_seconds=cap, rss_cap_gib=rss_cap,
                    inputs_sha256=sha(run / 'inputs.json')))
                while process.poll() is None:
                    require(not stopped, 'stop requested')
                    try:
                        tree = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in
                                  [tree, *tree.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(rss < rss_cap * 1024 ** 3, 'owned RSS cap')
                    require(time.monotonic() - start < cap, 'phase wall limit')
                    require(psutil.virtual_memory().available > 20 * 1024 ** 3,
                            'host available memory floor')
                    require(shutil.disk_usage(run).free > 40 * 1024 ** 3, 'disk floor')
                    require(cpu or gpu_state(gpu)[0] < 22 * 1024, 'GPU memory cap')
                    time.sleep(5)
                require(process.returncode == 0, 'phase worker failed')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    group_closed = True
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            receipt = read(out / 'clearml.json') if (out / 'clearml.json').exists() else {}
            passed = bool(error is None and group_closed and receipt.get('closed')
                          and receipt.get('outcome') == 'passed')
            write(out / 'exit.json', dict(exit_code=process.returncode if process else None,
                error=error, owned_group_closed=group_closed, peak_tree_rss_bytes=peak,
                elapsed_seconds=time.monotonic() - start, finished_unix=time.time()))
            write(out / 'complete.json', dict(passed=passed,
                files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                       if p.is_file()}))
        require(passed, 'NG85 phase failed; preserve attempt and review')


def campaign(base, run):
    """Execute only this frozen campaign; no retries or new research rounds."""
    import fcntl
    from ng85_queue import CAMPAIGN_SECONDS, run_stage

    verify(base, run)
    with (run / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda value, frame: stopped.append(value))
        stages = [('prepare',), ('dense',), ('bank',),
                  tuple(f'rankt5-{i}' for i in range(4)),
                  tuple(f'train-{a}-{s}' for s in (85001, 85002) for a in ('A', 'B'))]
        deadline = time.monotonic() + CAMPAIGN_SECONDS
        try:
            for stage in stages:
                run_stage(base, run, stage, stopped, deadline, closed)
        except BaseException as exc:
            write(run / 'campaign-progress.json', dict(state='stopped_on_failure',
                error=f'{type(exc).__name__}: {exc}',
                finished_unix=time.time(), locked_test_access=False))
            raise
        write(run / 'campaign-progress.json', dict(
            state='training_complete_full_corpus_evaluation_pending',
            finished_unix=time.time(), production_qualified=False,
            locked_test_access=False, automatic_new_round=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'continue', 'verify',
                                        'supervise', 'worker', 'campaign'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--batch-probe', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.base, args.run)
    elif args.mode == 'continue':
        require(args.parent is not None, 'parent attempt required')
        continue_frozen(args.base, args.run, args.parent, args.batch_probe)
    elif args.mode == 'verify':
        verify(args.base, args.run)
    elif args.mode == 'worker':
        worker(args.base, args.run, args.phase)
    elif args.mode == 'campaign':
        campaign(args.base, args.run)
    else:
        from ng85_queue import BUSY_EXIT, ResourceBusy
        try:
            supervise(args.base, args.run, args.phase, args.gpu)
        except ResourceBusy as exc:
            print(str(exc), flush=True)
            sys.exit(BUSY_EXIT)


if __name__ == '__main__':
    main()
