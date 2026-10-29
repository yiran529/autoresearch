"""Researcher mechanism probes reuse its lease inside the diagnostic sandbox."""
from pathlib import Path
from types import SimpleNamespace

from coral.grader.trusted_writes import _atomic_json

from .diagnostic_sandbox import execute_diagnostic, snapshot_files
from .runtime_artifacts import safe_path
from .runtime_resources import configured_resources


def execute(execution, job, code, work, row):
    scripts = safe_path(Path(job['scripts_dir']))
    worktree = Path(job['context']['worktree'])
    scratch = Path('/tmp') / execution.host.agent_id
    if not scripts.is_absolute() or '..' in scripts.parts or not (
            scripts.is_relative_to(worktree) or scripts.is_relative_to(scratch)):
        raise PermissionError('diagnostic scripts must belong to this Worktree or agent scratch')
    workspace = work / 'workspace'
    (workspace / 'results').mkdir(parents=True, exist_ok=True)
    snapshot_files(scripts, workspace / 'experiments')
    checkpoints = []
    for observation in row['observations']:
        if observation.get('evidence_path'):
            public = execution.coral / 'public' / observation['evidence_path'] / 'checkpoints/llama130m_muon/manifest.json'
            if public.exists():
                checkpoint = execution.artifacts.resolve_checkpoint(public)
                if checkpoint is not None:
                    checkpoints.append(checkpoint)
    state = {'source_role': 'researcher', 'code_path': str(code), 'checkpoint_paths': list(map(str, checkpoints)),
             'runtime': {'readonly_paths': [str(code), *map(str, checkpoints)],
                 'resources': configured_resources(execution.options),
                 'path': execution.options.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin')}}
    task = SimpleNamespace(coral_dir=execution.coral, workspace=workspace, records=work / 'records',
        read=lambda: state, register_job=execution.lease.register_job, clear_job=execution.lease.clear_job)
    payload = {'request_id': job['request_id'], 'argv': job['argv'], 'timeout': job['timeout'], 'gpus': 4}
    result = execute_diagnostic(task, payload, execution._check, borrowed=execution.lease.borrow())
    _atomic_json(work / 'diagnostic-result.json', result, mode=0o644)
    # Execution's canonical archive is persisted before the separate public mirror.
    # A mirror failure must not overwrite this actual execution outcome.
    return result
