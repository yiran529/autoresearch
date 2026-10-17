"""Host-owned identities and bounded evidence records for Analyzer diagnostics."""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

from coral.grader.trusted_writes import (
    TrustedWriteError,
    _atomic_json,
    _proc_stat,
    process_start_time,
)

_TASK = re.compile(r'N[0-9]{3}-(research|engineering)-v[1-9][0-9]*')
_REQUEST = re.compile(r'[a-zA-Z0-9_-]{1,64}')


def _live_descendant(peer: int, root: dict | None) -> bool:
    if not root:
        return False
    try:
        if process_start_time(root['pid']) != root['start_time']:
            return False
        visited = set()
        while peer > 1 and peer not in visited:
            visited.add(peer)
            parent, start = _proc_stat(peer)
            if peer == root['pid']:
                return start == root['start_time']
            peer = parent
    except (TrustedWriteError, KeyError):
        return False
    return False


def _validate_request(payload: dict) -> None:
    if not isinstance(payload, dict) or set(payload) != {'request_id', 'argv', 'gpus', 'timeout'}:
        raise ValueError('invalid diagnostic request fields')
    if not isinstance(payload['request_id'], str) or not _REQUEST.fullmatch(payload['request_id']):
        raise ValueError('invalid diagnostic request ID')
    argv = payload['argv']
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(arg, str) or '\x00' in arg for arg in argv)
            or not argv[0]):
        raise ValueError('diagnostic argv must be a nonempty string list without NUL')
    if type(payload['gpus']) is not int or not 1 <= payload['gpus'] <= 4:
        raise ValueError('diagnostic requires 1–4 GPUs')
    timeout = payload['timeout']
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 300):
        raise ValueError('diagnostic timeout must be within 300 seconds')


class DiagnosticTask:
    """A private per-analysis ledger; not a formal Attempt or Agent assignment."""

    def __init__(self, coral_dir: Path, task_id: str):
        if not _TASK.fullmatch(task_id):
            raise ValueError('invalid analysis task identity')
        self.coral_dir = Path(coral_dir).resolve()
        self.task_id = task_id
        self.private = self.coral_dir / 'private/diagnostics' / task_id
        self.path = self.private / 'task.json'
        self.root = self.coral_dir / 'public/diagnostics' / task_id
        self.workspace = self.root / 'workspace'
        self.records = self.root / 'records'

    @contextmanager
    def locked(self):
        self.private.mkdir(parents=True, mode=0o700, exist_ok=True)
        with (self.private / 'task.lock').open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def read(self):
        return json.loads(self.path.read_text())

    @contextmanager
    def session(self):
        """Do not let concurrent analyzers stage into the same immutable snapshot."""
        with (self.private / 'session.lock').open('a+b') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PermissionError('analysis session is already running') from exc
            with self.locked():
                root = self.read().get('process')
                if root and _live_descendant(root['pid'], root):
                    raise PermissionError('analysis session is already active')
            try:
                yield
            finally:
                self.close()

    def bind(self, pid: int) -> None:
        with self.locked():
            state = self.read()
            root = state.get('process')
            if root and _live_descendant(root['pid'], root):
                raise PermissionError('analysis task is already active')
            state['process'] = {'pid': pid, 'start_time': process_start_time(pid)}
            _atomic_json(self.path, state)

    def close(self) -> None:
        with self.locked():
            state = self.read()
            state['process'] = None
            _atomic_json(self.path, state)

    def run(self, payload: dict, peer_pid: int, executor) -> dict:
        _validate_request(payload)
        # This task-local lock covers one bounded command, never the workflow lock.
        with self.locked():
            state = self.read()
            root = state.get('process')
            if not _live_descendant(peer_pid, root):
                raise PermissionError('diagnostic caller is not this active Analyzer')
            request_id = payload['request_id']
            previous = state['requests'].get(request_id)
            if previous:
                if previous['request'] != payload:
                    raise ValueError('diagnostic request identity reused with different arguments')
                return previous['result']
            if state['executions'] >= 3:
                raise ValueError('analysis diagnostic execution budget exhausted')
            diagnostic_id = f'{self.task_id}/{request_id}'
            record = {**payload, 'diagnostic_id': diagnostic_id,
                      'node_id': state['node_id'], 'mode': state['mode'],
                      'analysis_version': state['version'], 'research_sha': state['research_sha'],
                      'evidence_hash': state.get('evidence_hash'),
                      'cwd': str(self.workspace), 'requested_at': time.time()}
            interrupted = {'status': 'interrupted', 'started': True,
                           'diagnostic_id': diagnostic_id,
                           'reason': 'worker interrupted before completion; request is not rerun'}
            # Conservatively charge before execution; a crash cannot reset the budget.
            state['executions'] += 1
            state['requests'][request_id] = {'request': payload, 'result': interrupted}
            _atomic_json(self.path, state)
            record_dir = self.records / request_id
            record_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(record_dir / 'record.json', {**record, **interrupted}, mode=0o644)

            def cancelled():
                if not _live_descendant(root['pid'], root):
                    raise PermissionError('Analyzer exited or changed identity')

            try:
                result = executor(self, payload, cancelled)
            except Exception as exc:
                result = {'status': 'failed', 'started': True, 'reason': str(exc)}
            result = {**result, 'diagnostic_id': diagnostic_id,
                      'record_path': str(record_dir / 'record.json')}
            if result.get('started') is False:
                state['executions'] -= 1
            state['requests'][request_id]['result'] = result
            _atomic_json(record_dir / 'record.json', {
                **record, **result, 'finished_at': time.time(),
            }, mode=0o644)
            _atomic_json(self.path, state)
            return result


def create_task(coral_dir: Path, node_id: str, mode: str, version: int,
                research_sha: str, *, evidence_hash: str | None = None) -> DiagnosticTask:
    if type(version) is not int or not re.fullmatch(r'[0-9a-f]{40}', research_sha):
        raise ValueError('invalid analysis task identity')
    task = DiagnosticTask(coral_dir, f'{node_id}-{mode}-v{version}')
    identity = {'node_id': node_id, 'mode': mode, 'version': version,
                'research_sha': research_sha, 'evidence_hash': evidence_hash}
    with task.locked():
        if task.path.exists():
            state = task.read()
            if any(state.get(key) != value for key, value in identity.items()):
                raise ValueError('analysis task identity conflicts with stored evidence')
        else:
            for directory in ('experiments', 'results'):
                (task.workspace / directory).mkdir(parents=True, exist_ok=True)
            _atomic_json(task.path, {**identity, 'process': None,
                                    'executions': 0, 'requests': {}})
    return task


def handle_diagnostic(payload: dict, context: dict, *, executor=None) -> dict:
    """Fixed trusted operation: identify the task from host PID ancestry only."""
    if executor is None:
        from .diagnostic_sandbox import execute_diagnostic

        executor = execute_diagnostic
    coral_dir = Path(context['coral_dir'])
    peer = context['peer_pid']
    for path in (coral_dir / 'private/diagnostics').glob('*/task.json'):
        task = DiagnosticTask(coral_dir, path.parent.name)
        if _live_descendant(peer, task.read().get('process')):
            return task.run(payload, peer, executor)
    raise PermissionError('peer does not belong to an active Analyzer diagnostic task')
