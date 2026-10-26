"""Host-owned four-GPU reservations retained across staged research requests.

The keeper owns both GPU flocks and scheduler admission. Borrowers own neither.
These are same-UID CORAL coordination locks, not protection from external CUDA users.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import yaml
from greedylore_autoresearch_grader.contract import EvalContract
from greedylore_autoresearch_grader.gpu import (
    GpuDevice,
    GpuInventory,
    GpuReservation,
    IdleGpuPolicy,
    _belongs_to_launcher,
    reserve_idle_gpus,
)
from greedylore_autoresearch_grader.resources import ResourceOptions
from greedylore_autoresearch_grader.runtime import InfraFailure, ProcessIdentity

from coral.grader.trusted_writes import _atomic_json

from .runtime_artifacts import safe_path, settings

_CONTEXT = ('agent_id', 'agent_root_pid', 'agent_start_time', 'host_pid', 'host_start_time')


def _identity(pid, start):
    if type(pid) is not int or type(start) is not int:
        return None
    found = ProcessIdentity.read(pid)
    return found if found and found.uid == os.getuid() and found.start_time == start else None


def _spawn(coral_dir, node_id, token):
    # Isolated Python ignores candidate PYTHONPATH, site customization and cwd.
    # Source roots come from already imported trusted packages, never a request.
    import greedylore_autoresearch_grader

    import coral

    roots = [str(Path(__file__).resolve().parents[1]),
             str(Path(greedylore_autoresearch_grader.__file__).resolve().parents[1]),
             str(Path(coral.__file__).resolve().parents[1])]
    bootstrap = ('import sys,json; sys.path[:0]=' + repr(roots) + '; '
                 'from autoresearch.lease import _keeper; _keeper(*json.load(sys.stdin))')
    environment = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(Path.home()),
                   'LANG': 'C.UTF-8'}
    log = Path(coral_dir) / 'private/research_leases/keeper.err'
    with log.open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-I', '-c', bootstrap],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=stream, cwd=Path(coral_dir) / 'private',
                                   env=environment, start_new_session=True)
    try:
        process.stdin.write(json.dumps([str(coral_dir), node_id, token]).encode())
        process.stdin.close()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    # Do not retain pipes or reservation descriptors in request workers.
    return process.pid


class ResearchLease:
    def __init__(self, coral_dir, node_id):
        if not isinstance(node_id, str) or not re.fullmatch(r'N[0-9]{3}', node_id):
            raise ValueError('invalid research lease node')
        self.coral_dir = Path(coral_dir).resolve()
        self.node_id = node_id
        self.path = safe_path(self.coral_dir / 'private/research_leases' / f'{node_id}.json')

    @contextmanager
    def _locked(self):
        safe_path(self.path.parent).mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = safe_path(self.path.with_suffix('.lock'))
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, 'a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _read(self):
        return json.loads(safe_path(self.path).read_text()) if self.path.exists() else None

    def _active(self, context):
        if settings(self.coral_dir).get('staged_research') is not True:
            return False
        if not _identity(context.get('agent_root_pid'), context.get('agent_start_time')):
            return False
        if not _identity(context.get('host_pid'), context.get('host_start_time')):
            return False
        try:
            state = json.loads(safe_path(self.coral_dir / 'private/autoresearch_state.json').read_text())
            owner = state['nodes'][self.node_id]['research']
            staged = safe_path(self.coral_dir / 'private/research_staged' / f'{self.node_id}.json')
            return (owner.get('agent_id') == context.get('agent_id')
                    and owner.get('status') == 'active'
                    and not (self.coral_dir / 'private/workflow_stopping').exists()
                    and not (staged.exists() and json.loads(staged.read_text()).get('transfer_terminal')))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _live(self, row):
        keeper = (row or {}).get('keeper') or {}
        return bool(_identity(keeper.get('pid'), keeper.get('start_time')))

    def ensure(self, context):
        context = {key: context.get(key) for key in _CONTEXT}
        if not isinstance(context['agent_id'], str) or not self._active(context):
            raise PermissionError('research lease requires live authenticated owner and host')
        with self._locked():
            row = self._read()
            if self._live(row):
                if row['context'] != context:
                    raise PermissionError('research lease belongs to another process binding')
                if row.get('release_requested'):
                    return False
                return row['status'] == 'ready'
            token = uuid.uuid4().hex
            row = {'version': 1, 'token': token, 'context': context, 'status': 'waiting',
                   'keeper': None, 'job': None, 'release_requested': False}
            _atomic_json(self.path, row)
            pid = _spawn(self.coral_dir, self.node_id, token)
            keeper = ProcessIdentity.read(pid)
            if keeper is None or keeper.uid != os.getuid():
                raise InfraFailure('research GPU keeper exited during initialization')
            row['keeper'] = asdict(keeper)
            _atomic_json(self.path, row)
        return False

    def check(self):
        row = self._read()
        return bool(row and row['status'] == 'ready' and not row.get('release_requested')
                    and self._live(row) and self._active(row['context']))

    def borrow(self):
        with self._locked():
            if not self.check():
                raise InfraFailure('research GPU lease is not ready')
            row = self._read()
            devices = []
            for value in row['devices']:
                value = dict(value)
                value['compute_pids'] = tuple(value['compute_pids'])
                value['compute_processes'] = tuple(ProcessIdentity(**p)
                                                    for p in value['compute_processes'])
                devices.append(GpuDevice(**value))
            return GpuReservation(tuple(devices), [], row['topology'])

    def register_job(self, pid):
        with self._locked():
            if not self.check():
                raise InfraFailure('research GPU lease is not ready')
            identity = ProcessIdentity.read(pid)
            if (identity is None or identity.uid != os.getuid()
                    or identity.pgid != pid or identity.session_id != pid):
                raise PermissionError('research job must be a live same-UID new session')
            row = self._read()
            if row.get('job') and row['job'] != asdict(identity):
                raise InfraFailure('research GPU lease already has a registered job')
            row['job'] = asdict(identity)
            _atomic_json(self.path, row)

    def clear_job(self, pid):
        with self._locked():
            row = self._read()
            if row and (row.get('job') or {}).get('pid') == pid:
                row['job'] = None
                _atomic_json(self.path, row)

    def release(self):
        with self._locked():
            row = self._read()
            if row:
                row['release_requested'] = True
                _atomic_json(self.path, row)


def _job_members(job):
    if not job:
        return set()
    root = ProcessIdentity(**job)
    current = ProcessIdentity.read(root.pid)
    if current is not None and current != root:
        return set()
    # A live leader, or previously captured members, anchors the session identity.
    if current != root:
        return set()
    return {identity for path in Path('/proc').iterdir() if path.name.isdigit()
            if (identity := ProcessIdentity.read(int(path.name))) is not None
            and _belongs_to_launcher(identity, root.pid)}


def _stop_owned(identities):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for identity in identities:
            if ProcessIdentity.read(identity.pid) == identity:
                try:
                    os.kill(identity.pid, signum)
                except ProcessLookupError:
                    pass
        if signum == signal.SIGTERM and identities:
            time.sleep(.2)


def _keeper(coral_dir, node_id, token):
    lease = ResearchLease(coral_dir, node_id)
    reservation = None
    owned = set()
    cancelled = False

    def cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancel)

    def check():
        row = lease._read()
        if row and row['token'] == token:
            owned.update(_job_members(row.get('job')))
        if (cancelled or not row or row['token'] != token or row.get('release_requested')
                or not lease._active(row['context'])):
            raise InfraFailure('research GPU lease cancelled')
        return row

    try:
        # ensure() publishes the process identity while holding the same lock.
        with lease._locked():
            check()
        options = ResourceOptions.from_args(settings(lease.coral_dir))
        contract = EvalContract.load(lease.coral_dir / 'private/eval_contract.yaml')
        if contract.hardware.gpu_count != 4:
            raise InfraFailure('staged research requires exactly four GPUs')
        config = yaml.safe_load((lease.coral_dir / 'config.yaml').read_text())
        limit = config.get('grader', {}).get('parallel', {}).get('max_workers', 1)
        if type(limit) is not int or limit < 1:
            raise InfraFailure('invalid staged research execution limit')
        policy = IdleGpuPolicy(**contract.raw['hardware']['idle_gpu'])
        reservation = reserve_idle_gpus(
            4, options.lock_dir, policy, inventory_provider=GpuInventory.from_nvidia_smi,
            allow_other_users=False, wait_timeout=options.wait_timeout,
            poll_interval=options.poll_interval, check_cancelled=check,
            source_role='researcher', execution_group=str(lease.coral_dir), execution_limit=limit)
        with lease._locked():
            row = check()
            row.update(status='ready', devices=[asdict(d) for d in reservation.devices],
                       topology=reservation.topology)
            _atomic_json(lease.path, row)
        next_monitor = 0.0
        while True:
            row = check()
            if time.monotonic() >= next_monitor:
                job = row.get('job')
                reservation.check_exclusive(job['pid'] if job else None)
                next_monitor = time.monotonic() + options.monitor_interval
            time.sleep(min(.2, options.monitor_interval))
    except (InfraFailure, OSError, ValueError, KeyError, TypeError) as exc:
        error = str(exc)
    finally:
        _stop_owned(owned)
        if reservation is not None:
            reservation.release()
        with lease._locked():
            row = lease._read()
            if row and row['token'] == token:
                row.update(status='released', error=locals().get('error'))
                _atomic_json(lease.path, row)
