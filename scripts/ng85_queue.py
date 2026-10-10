"""Bounded, non-preemptive GPU queue for the frozen NG85 comparison."""

import json
import subprocess
import sys
import time

from ng85_data import require, write


BUSY_EXIT = 75
POLL_SECONDS = 60
QUEUE_WAIT_SECONDS = 12 * 3600
CAMPAIGN_SECONDS = 96 * 3600
GPU_ORDER = (3, 0, 1, 2)
MAX_PARALLEL = 2


class ResourceBusy(RuntimeError):
    """No worker was started; the same pending phase may wait for resources."""


def gpu_snapshot(gpu):
    args = ['nvidia-smi', '-i', str(gpu)]
    line = subprocess.check_output(args + [
        '--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    uuid, memory, utilization = (v.strip() for v in line.split(','))
    apps = subprocess.check_output(args + [
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    return dict(gpu=gpu, uuid=uuid, memory_mib=int(memory),
                utilization=int(utilization),
                compute_pids=[int(v) for v in apps.splitlines() if v.strip()])


def idle(snapshot):
    return (snapshot['memory_mib'] < 100 and snapshot['utilization'] == 0
            and not snapshot['compute_pids'])


def log_admission(run, phase, **record):
    line = dict(phase=phase, unix=time.time(), **record)
    # One small append per admission; keep failed pre-start attempts outside
    # immutable phase directories so they cannot resemble failed computation.
    with (run / 'admission.jsonl').open('a') as stream:
        stream.write(json.dumps(line, allow_nan=False) + '\n')


def check_resources(run, phase, gpu):
    import shutil
    import psutil

    snapshot = None if gpu is None else gpu_snapshot(gpu)
    memory = psutil.virtual_memory().available
    disk = shutil.disk_usage(run).free
    admitted = ((snapshot is None or idle(snapshot))
                and memory > 28 * 1024 ** 3 and disk > 60 * 1024 ** 3)
    log_admission(run, phase, admitted=admitted, snapshot=snapshot,
                  host_available_bytes=memory, disk_free_bytes=disk)
    if not admitted:
        raise ResourceBusy('resources busy; no worker started')


def finish_phase(run, phase, code, pending, closed):
    if code == BUSY_EXIT:
        require(not (run / phase).exists(), 'busy phase has computation output')
        pending.append(phase)
    else:
        require(code == 0, 'failed phase: ' + phase)
        closed(run, phase)


def run_stage(base, run, phases, stopped, deadline, closed):
    pending = []
    for phase in phases:
        if (run / phase).exists():
            closed(run, phase)
        else:
            pending.append(phase)
    active, waited = {}, 0.
    previous = time.monotonic()
    try:
        while pending or active:
            now = time.monotonic()
            require(not stopped, 'campaign stop requested')
            require(now < deadline, 'campaign wall limit')
            finished = []
            for phase, (process, gpu) in list(active.items()):
                code = process.poll()
                if code is not None:
                    del active[phase]
                    finish_phase(run, phase, code, pending, closed)
                    finished.append(code)
            if pending and not active and all(code == BUSY_EXIT for code in finished):
                waited += now - previous
            previous = now
            require(waited < QUEUE_WAIT_SECONDS, 'resource queue wait limit')
            used = {gpu for _, gpu in active.values()}
            snapshots = {}
            if pending and pending[0] not in ('prepare', 'bank'):
                snapshots = {g: gpu_snapshot(g) for g in GPU_ORDER if g not in used}
                slots = [g for g in GPU_ORDER if g in snapshots and idle(snapshots[g])]
            else:
                slots = [None] if not active else []
            for gpu in slots[:max(0, MAX_PARALLEL - len(active))]:
                if not pending:
                    break
                phase = pending.pop(0)
                command = [sys.executable, '-B', str(run / 'ng85_run.py'),
                    'supervise', '--base', str(base), '--run', str(run),
                    '--phase', phase]
                if gpu is not None:
                    command += ['--gpu', str(gpu)]
                active[phase] = (subprocess.Popen(command), gpu)
            write(run / 'campaign-progress.json', dict(
                state='running' if active else 'waiting_for_resources',
                active={p: dict(pid=proc.pid, gpu=g) for p, (proc, g) in active.items()},
                pending=pending, resource_snapshots=snapshots,
                idle_wait_seconds=waited, updated_unix=time.time(),
                locked_test_access=False))
            if pending or active:
                time.sleep(POLL_SECONDS)
    except BaseException:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            process.wait(timeout=60)
        raise
