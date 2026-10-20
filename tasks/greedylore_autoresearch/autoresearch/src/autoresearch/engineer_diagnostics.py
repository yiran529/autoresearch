"""Bind bounded GPU diagnostics to a live Engineer and committed code."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from coral.grader.trusted_writes import (
    _atomic_json,
    resolve_peer_binding,
)

from .diagnostic_sandbox import snapshot_files
from .diagnostic_session import settings
from .diagnostics import DiagnosticTask, _validate_request
from .state import WorkflowStore, _owned_active


def _authorize(coral, binding):
    store = WorkflowStore(coral, repo_path=Path(binding.worktree))
    state = store._load_unlocked()
    worker = state['engineering_workers'].get(binding.agent_id, {})
    node = worker.get('node_id')
    if (binding.island_id is not None or worker.get('status') != 'active'
            or worker.get('finish_requested')
            or not _owned_active(state, binding.agent_id, node, 'engineering')):
        raise PermissionError('diagnostic requires this Engineer\'s active, non-finalizing claim')
    store._reject_changed_anchor_unlocked(state, worker, node)
    return worker


def _snapshot_commit(repo, sha, destination, private):
    """Extract only bounded regular Git contents; never follow archive links."""
    if destination.exists():
        return
    with tempfile.TemporaryDirectory(dir=private, prefix='code-') as temporary:
        root = Path(temporary)
        process = subprocess.Popen(['git', '-C', str(repo), 'archive', sha], stdout=subprocess.PIPE)
        try:
            size = entries = 0
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    entries += 1
                    relative = PurePosixPath(member.name)
                    if (relative.is_absolute() or '..' in relative.parts
                            or not (member.isdir() or member.isfile())):
                        raise ValueError('diagnostic code archive contains unsafe entries')
                    size += member.size
                    if entries > 4096 or size > 64 * 1024 * 1024:
                        raise ValueError('diagnostic code archive budget exceeded')
                    path = root / relative
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, path.open('wb') as target:
                            shutil.copyfileobj(source, target)
                        path.chmod(member.mode & 0o700)
            if process.wait(timeout=30):
                raise ValueError('cannot snapshot diagnostic candidate')
            # A failed preparation must not leave a reusable partial snapshot.
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(root, destination)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def run_engineer_diagnostic(coral: Path, peer: int, payload: dict, executor):
    _validate_request(payload)
    binding = resolve_peer_binding(coral, peer)
    if not re.fullmatch(r'engineer-E[0-9]{4,}', binding.agent_id):
        raise PermissionError('diagnostic requires an active Engineer or Analyzer')
    worker = _authorize(coral, binding)
    config = settings(coral)
    if not config.get('engineer_diagnostics', True):
        raise PermissionError('Engineer diagnostics are disabled')
    scripts = payload.get('scripts_dir')
    if not scripts:
        raise ValueError('Engineer diagnostic requires --scripts-dir')
    scripts = Path(scripts)
    repo = Path(binding.worktree)
    scratch = Path('/tmp') / binding.agent_id
    # snapshot_files separately rejects symlinks in every path component.
    if not (scripts.is_relative_to(repo) or scripts.is_relative_to(scratch)) or '..' in scripts.parts:
        raise PermissionError('diagnostic scripts must belong to this Worktree or /tmp/<agent-id>/')
    task = DiagnosticTask(coral, f"{worker['node_id']}-{binding.agent_id}")
    identity = {'node_id': worker['node_id'], 'mode': 'engineering', 'version': None,
                'research_sha': worker['research_sha'], 'owner_agent_id': binding.agent_id,
                'source_role': 'engineer'}
    with task.locked(timeout=5):
        if not task.path.exists():
            for directory in ('experiments', 'results'):
                (task.workspace / directory).mkdir(parents=True, exist_ok=True)
            _atomic_json(task.path, {**identity, 'process': None, 'executions': 0, 'requests': {}})
        state = task.read()
        if any(state.get(key) != value for key, value in identity.items()):
            raise PermissionError('Engineer diagnostic identity changed')
        # Rebind the resumed process while retaining every request and budget.
        state['process'] = {'pid': binding.root_pid, 'start_time': binding.start_time}
        _atomic_json(task.path, state)

    def prepare(state):
        if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                    '--untracked-files=no']).strip():
            raise ValueError('commit tracked candidate changes before GPU diagnostics')
        sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('invalid diagnostic candidate SHA')
        code = task.workspace / 'code' / sha
        _snapshot_commit(repo, sha, code, task.private)
        snapshot_files(code, task.records / payload['request_id'] / 'code' / sha)
        # Validate/freeze inputs before replacing the previous staged scripts.
        with tempfile.TemporaryDirectory(dir=task.private, prefix='scripts-') as temporary:
            staged = Path(temporary) / 'experiments'
            snapshot_files(scripts, staged)
            if (task.workspace / 'experiments').exists():
                shutil.rmtree(task.workspace / 'experiments')
            snapshot_files(staged, task.workspace / 'experiments')
        state.update(candidate_sha=sha, code_path=str(code), runtime={
            'readonly_paths': config.get('diagnostic_runtime_roots', []),
            'path': config.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin'),
            'gpu_lock_dir': config.get('gpu_lock_dir')})

    return task.run(payload, peer, executor, prepare=prepare,
                    authorize=lambda: _authorize(coral, binding))


@contextmanager
def publication_guard(store, agent_id):
    """Exclude running/staging diagnostics while freezing the final Analysis."""
    from .state import WorkflowError

    worker = store._load_unlocked()['engineering_workers'].get(agent_id, {})
    node = worker.get('node_id')
    if not node:
        raise WorkflowError('Engineering summary owner was not found')
    task = DiagnosticTask(store.coral_dir, f'{node}-{agent_id}')
    try:
        with task.locked(timeout=0):
            yield
    except TimeoutError as exc:
        raise WorkflowError('wait for the running Engineering diagnostic before finishing') from exc


def evidence_references(coral, node_id, research_sha):
    """Only host-owned durable records of matching Engineering claims are sealed."""
    references = []
    state = WorkflowStore(coral)._load_unlocked()
    for agent_id, worker in state['engineering_workers'].items():
        if worker.get('node_id') != node_id or worker.get('research_sha') != research_sha:
            continue
        task = DiagnosticTask(coral, f'{node_id}-{agent_id}')
        if not task.path.exists():
            continue
        ledger = task.read()
        if ledger.get('owner_agent_id') != agent_id or ledger.get('research_sha') != research_sha:
            raise ValueError('Engineering diagnostic evidence identity mismatch')
        for request_id in ledger['requests']:
            references.append(str((task.records / request_id).relative_to(coral / 'public')))
    return references
