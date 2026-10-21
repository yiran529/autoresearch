"""Host-owned, crash-recoverable tensor retention and completed-Node cleanup.

The private ledger authorizes removal before unlinking payloads. Public manifests
and all evaluation logs remain immutable evidence, including after a crash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coral.hub._island import island_root

from .atomic import atomic_write_text
from .validation import ArtifactValidationError


def _safe_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or '..' in relative.parts:
        raise ArtifactValidationError('checkpoint path escapes fixed evaluation logs')
    path = root / relative
    if root.is_symlink() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ArtifactValidationError('checkpoint path contains symlink')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ArtifactValidationError('checkpoint path escapes fixed evaluation logs')
    return path


def _ledger(coral_dir: Path) -> tuple[Path, dict]:
    path = _safe_path(coral_dir, Path('private/checkpoint_retention.json'))
    if not path.exists():
        return path, {'version': 1, 'pruned': {}}
    try:
        value = json.loads(path.read_text())
        if value['version'] != 1 or not isinstance(value['pruned'], dict):
            raise ValueError('invalid ledger')
        return path, value
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactValidationError('invalid checkpoint retention ledger') from exc


def checkpoint_is_pruned(coral_dir: Path, manifest: Path) -> bool:
    """Recognize explicit cleanup only when the original manifest still matches."""
    coral_dir = Path(coral_dir)
    _, ledger = _ledger(coral_dir)
    if not ledger['pruned']:
        return False
    try:
        key = str(manifest.relative_to(coral_dir))
    except ValueError as exc:
        raise ArtifactValidationError('checkpoint manifest outside CORAL run') from exc
    record = ledger['pruned'].get(key)
    if record is None:
        return False
    if record.get('manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ArtifactValidationError('pruned checkpoint manifest integrity check failed')
    return True


def _prune_manifest(coral_dir: Path, manifest: Path, ledger_path: Path, ledger: dict,
                    node_id: str) -> None:
    key = str(manifest.relative_to(coral_dir))
    data = manifest.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    existing = ledger['pruned'].get(key)
    if existing is not None:
        if existing['manifest_sha256'] != digest:
            raise ArtifactValidationError('pruned checkpoint manifest integrity check failed')
        names = existing['payloads']
    else:
        value = json.loads(data)
        names = []
        for child in sorted(manifest.parent.iterdir()):
            _safe_path(coral_dir, child.relative_to(coral_dir))
            if not (child.name.endswith('.pt')
                    or ('.pt.' in child.name and child.name.endswith('.tmp'))):
                continue
            if not child.is_file():
                raise ArtifactValidationError('checkpoint payload must be a regular file')
            names.append(child.name)
        # Never turn preexisting corruption into an authorized prune.
        if value.get('status') == 'complete':
            world = value.get('world_size')
            if (type(world) is not int or not 1 <= world <= 4
                    or set(value.get('files', {})) != {'model.pt', *(f'rank-{r}.pt' for r in range(world))}):
                raise ArtifactValidationError('checkpoint integrity check failed: invalid manifest files')
            for name, info in value['files'].items():
                payload = _safe_path(manifest.parent, Path(name))
                if payload.parent != manifest.parent or not payload.is_file():
                    raise ArtifactValidationError('checkpoint integrity check failed: missing payload')
                with payload.open('rb') as stream:
                    payload_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
                if (payload.stat().st_size != info['size_bytes']
                        or payload_hash != info['sha256']):
                    raise ArtifactValidationError('checkpoint integrity check failed: changed payload')
        ledger['pruned'][key] = {'node_id': node_id, 'manifest_sha256': digest,
                                'payloads': names}
        atomic_write_text(ledger_path, json.dumps(ledger, sort_keys=True) + '\n')
    for name in names:
        payload = _safe_path(manifest.parent, Path(name))
        if payload.parent != manifest.parent:
            raise ArtifactValidationError('checkpoint payload path escapes manifest directory')
        payload.unlink(missing_ok=True)



def _prune_incomplete_checkpoint(coral_dir: Path, root: Path, ledger_path: Path,
                                 ledger: dict, node_id: str) -> None:
    """Clean an interrupted writer without authorizing damage to a known manifest."""
    key = str((root / 'manifest.json').relative_to(coral_dir))
    existing = ledger['pruned'].get(key)
    if existing is not None:
        if existing.get('manifest_missing') is not True:
            raise ArtifactValidationError('checkpoint integrity check failed: manifest disappeared')
        names = existing['payloads']
    else:
        names = []
        for child in sorted(root.iterdir()):
            _safe_path(coral_dir, child.relative_to(coral_dir))
            if not (child.name.endswith('.pt')
                    or ('.pt.' in child.name and child.name.endswith('.tmp'))):
                continue
            if not child.is_file():
                raise ArtifactValidationError('checkpoint payload must be a regular file')
            names.append(child.name)
        if not names:
            return
        ledger['pruned'][key] = {'node_id': node_id, 'manifest_sha256': None,
                                'manifest_missing': True, 'payloads': names}
        atomic_write_text(ledger_path, json.dumps(ledger, sort_keys=True) + '\n')
    for name in names:
        payload = _safe_path(root, Path(name))
        if payload.parent != root:
            raise ArtifactValidationError('checkpoint payload path escapes manifest directory')
        payload.unlink(missing_ok=True)


def prune_research_checkpoints(coral_dir, node_id, attempts, selected_sha=None, *, island_id=None):
    """Keep the selected terminal run's tensors, leaving pending runs untouched.

    The caller restricts use to active multi-attempt research and its final freeze;
    historical nodes and the original singleton Root are never cleanup targets.
    Call while holding trusted_writes_lock to serialize private ledger updates.
    """
    from .research import select_research_attempt

    coral_dir = Path(coral_dir)
    candidates = [a for a in attempts if a.metadata.get('node_id') == node_id
                  and a.metadata.get('mode') == 'research' and a.budget_class != 'tune']
    if not any(a.status != 'pending' for a in candidates):
        return None
    selected = (next((a for a in candidates if a.commit_hash == selected_sha), None)
                if selected_sha is not None else select_research_attempt(candidates))
    if selected_sha is not None and selected is None:
        raise ArtifactValidationError('selected checkpoint attempt is missing')
    retained = selected.commit_hash if selected else None
    if node_id == 'N000':
        return retained
    _prune_attempt_checkpoints(coral_dir, node_id, candidates, selected, island_id=island_id)
    return retained


def prune_completed_node_checkpoints(coral_dir, node_id, attempts, *, island_id=None):
    """Remove all of a completed Node's tensors; return False while graders drain.

    The caller validates completion and excludes diagnostics while holding
    trusted_writes_lock. Include archived Attempts and all grader retries.
    Root N000 retains its existing exemption.
    """
    if node_id == 'N000':
        return False
    candidates = [a for a in attempts if a.metadata.get('node_id') == node_id]
    if any(a.status in {'pending', 'running'} or any(
            run.get('status') in {'pending', 'running'}
            for run in a.metadata.get('grader_runs') or []) for a in candidates):
        return False
    _prune_attempt_checkpoints(Path(coral_dir), node_id, candidates, island_id=island_id)
    return True


def _prune_attempt_checkpoints(coral_dir, node_id, candidates, selected=None, *, island_id=None):
    logs = island_root(coral_dir, island_id) / 'eval_logs'
    ledger_path, ledger = _ledger(coral_dir)
    for attempt in candidates:
        if attempt.status == 'pending':
            continue
        commit = _safe_path(logs, Path(attempt.commit_hash))
        if commit.parent != logs:
            raise ArtifactValidationError('invalid checkpoint commit path')
        runs = attempt.metadata.get('grader_runs') or []
        terminal = [run for run in runs if run.get('status') not in {'pending', 'running'}]
        keep_run = terminal[-1].get('run_id') if terminal else None
        # Only visit declared run roots, plus the legacy pre-run-id layout.
        roots = [(commit, None)]
        for run in terminal:
            run_id = run.get('run_id')
            if not isinstance(run_id, str) or not run_id:
                raise ArtifactValidationError('invalid checkpoint grader run ID')
            root = _safe_path(commit, Path(run_id))
            if root.parent != commit:
                raise ArtifactValidationError('invalid checkpoint grader run path')
            expected_log = f'eval_logs/{attempt.commit_hash}/{run_id}'
            if run.get('log_dir', expected_log) != expected_log:
                raise ArtifactValidationError('checkpoint grader log path mismatch')
            roots.append((root, run_id))
        for root, run_id in roots:
            if attempt is selected and run_id == keep_run:
                continue
            checkpoints = _safe_path(root, Path('checkpoints'))
            if not checkpoints.exists():
                continue
            for checkpoint in sorted(checkpoints.iterdir()):
                _safe_path(checkpoints, Path(checkpoint.name))
                if not checkpoint.is_dir():
                    continue
                manifest = _safe_path(checkpoint, Path('manifest.json'))
                if manifest.is_file():
                    _prune_manifest(coral_dir, manifest, ledger_path, ledger, node_id)
                else:
                    _prune_incomplete_checkpoint(coral_dir, checkpoint, ledger_path, ledger, node_id)
