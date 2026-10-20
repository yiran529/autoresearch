"""Host-owned identities and bounded evidence records for Analyzer diagnostics."""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

from coral.grader.timeouts import DIAGNOSTIC_EXECUTION_TIMEOUT
from coral.grader.trusted_writes import (
    TrustedWriteError,
    _atomic_json,
    _proc_stat,
    process_start_time,
)

_TASK = re.compile(r'N[0-9]{3}-(?:(research|engineering)-v[1-9][0-9]*|engineer-E[0-9]{4,})')
_REQUEST = re.compile(r'[a-zA-Z0-9_-]{1,64}')


def remaining_budget(state: dict) -> tuple[int, int]:
    """Return total and successful runs remaining, including persisted retries."""
    completed = sum(item['result'].get('status') == 'completed'
                    for item in state['requests'].values())
    return max(0, 10 - state['executions']), max(0, 3 - completed)


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
    fields = {'request_id', 'argv', 'gpus', 'timeout'}
    if not isinstance(payload, dict) or set(payload) not in (fields, fields | {'scripts_dir'}):
        raise ValueError('invalid diagnostic request fields')
    if 'scripts_dir' in payload and (not isinstance(payload['scripts_dir'], str)
                                   or not Path(payload['scripts_dir']).is_absolute()):
        raise ValueError('diagnostic scripts_dir must be absolute')
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
            or not 0 < timeout <= DIAGNOSTIC_EXECUTION_TIMEOUT):
        raise ValueError(f'diagnostic timeout must be within {DIAGNOSTIC_EXECUTION_TIMEOUT} seconds')


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
    def locked(self, timeout=None):
        self.private.mkdir(parents=True, mode=0o700, exist_ok=True)
        with (self.private / 'task.lock').open('a+b') as stream:
            if timeout is None:
                fcntl.flock(stream, fcntl.LOCK_EX)
            else:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('diagnostic task is still draining') from None
                        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
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
            with self.locked(timeout=5):
                root = self.read().get('process')
                if root and _live_descendant(root['pid'], root):
                    raise PermissionError('analysis session is already active')
            try:
                yield
            finally:
                self.close(timeout=5)

    def bind(self, pid: int, timeout=None) -> None:
        with self.locked(timeout=timeout):
            if (self.coral_dir / 'private/workflow_stopping').exists():
                raise PermissionError('workflow is stopping')
            state = self.read()
            root = state.get('process')
            if root and _live_descendant(root['pid'], root):
                raise PermissionError('analysis task is already active')
            state['process'] = {'pid': pid, 'start_time': process_start_time(pid)}
            _atomic_json(self.path, state)

    def close(self, timeout=None) -> None:
        with self.locked(timeout=timeout):
            state = self.read()
            state['process'] = None
            _atomic_json(self.path, state)

    def run(self, payload: dict, peer_pid: int, executor, *, prepare=None, authorize=None) -> dict:
        _validate_request(payload)
        # This task-local lock covers one bounded command, never the workflow lock.
        with self.locked():
            state = self.read()
            if authorize:
                authorize()
            root = state.get('process')
            if not _live_descendant(peer_pid, root):
                raise PermissionError('diagnostic caller is not this active Analyzer')
            request_id = payload['request_id']
            previous = state['requests'].get(request_id)
            if previous:
                if previous['request'] != payload:
                    raise ValueError('diagnostic request identity reused with different arguments')
                return previous['result']
            if 0 in remaining_budget(state):
                raise ValueError('analysis diagnostic execution budget exhausted')
            if prepare:
                prepare(state)
            diagnostic_id = f'{self.task_id}/{request_id}'
            record = {**payload, 'diagnostic_id': diagnostic_id,
                      'node_id': state['node_id'], 'mode': state['mode'],
                      'analysis_version': state['version'], 'research_sha': state['research_sha'],
                      'evidence_hash': state.get('evidence_hash'),
                      'cwd': str(self.workspace), 'requested_at': time.time()}
            if state.get('source_role') == 'engineer':
                record.update(owner_agent_id=state['owner_agent_id'], candidate_sha=state['candidate_sha'],
                              code_path=state['code_path'])
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
                if authorize:
                    authorize()
                if (self.coral_dir / 'private/workflow_stopping').exists():
                    raise PermissionError('workflow is stopping')
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
    with task.locked(timeout=5):
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
        if task.read().get('source_role') != 'engineer' and _live_descendant(peer, task.read().get('process')):
            if 'scripts_dir' in payload:
                raise ValueError('Analyzer scripts belong in its supplied workspace')
            return task.run(payload, peer, executor)
    from .engineer_diagnostics import run_engineer_diagnostic

    return run_engineer_diagnostic(coral_dir, peer, payload, executor)
