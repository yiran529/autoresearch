"""Reclaim completed Nodes' tensors without changing their published evidence."""

from __future__ import annotations

import fcntl
from contextlib import ExitStack, contextmanager

from coral.hub.attempts import read_attempts

from .checkpoint_retention import _safe_path, prune_completed_node_checkpoints
from .diagnostics import DiagnosticTask, _live_descendant


@contextmanager
def _idle_diagnostics(coral_dir, node_id):
    # Task locks exclude commands; session locks also exclude Analyzer staging.
    # Never wait under trusted_writes_lock: publication takes these in reverse.
    with ExitStack() as stack:
        try:
            for path in sorted((coral_dir / 'private/diagnostics').glob(f'{node_id}-*/task.json')):
                _safe_path(coral_dir, path.relative_to(coral_dir))
                task = DiagnosticTask(coral_dir, path.parent.name)
                stack.enter_context(task.locked(timeout=0))
                session = stack.enter_context((task.private / 'session.lock').open('a+b'))
                fcntl.flock(session, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = task.read()
                if state.get('node_id') != node_id:
                    raise ValueError('checkpoint cleanup diagnostic Node mismatch')
                process = state.get('process')
                if (state.get('source_role') != 'engineer' and process
                        and _live_descendant(process['pid'], process)):
                    raise TimeoutError('Analyzer is still draining')
        except (TimeoutError, BlockingIOError):
            yield False
            return
        yield True


def cleanup_completed_nodes(store, state):
    """Called with the workflow lock after completion has been saved to disk.

    Cleanup failure is independently retryable and cannot revoke completion.
    Missing cleanup fields on existing version-2 Nodes are handled on the next
    reconciliation; N000 keeps its baseline tensors.
    """
    changed = False
    attempts = None
    for node_id, record in state['nodes'].items():
        cancelled = (record['research'].get('protocol') == 'staged-v2'
                     and record['research'].get('status') == 'cancelled')
        if (node_id == 'N000' or (record.get('cleanup') or {}).get('status') == 'done'
                or not (cancelled or store._node_completed(record))):
            continue
        cleanup = {'status': 'pending', 'last_error': None}
        try:
            if cancelled:
                from .lease import ResearchLease
                from .runtime_artifacts import RuntimeArtifacts
                from .staged_research import node_idle
                lease = ResearchLease(store.coral_dir, node_id)
                if node_idle(store.coral_dir, node_id) and not lease._live(lease._read()):
                    artifacts = RuntimeArtifacts(store.coral_dir)
                    root = artifacts.namespace / 'nodes' / node_id
                    if root.exists():
                        artifacts.preserve_small(node_id, root, store.coral_dir / 'public/cancelled' / node_id)
                    artifacts.cleanup_node(node_id, terminal=True, _already_locked=True)
                    cleanup['status'] = 'done'
            elif store._node_completed_unlocked(state, node_id):
                if attempts is None:
                    attempts = read_attempts(store.coral_dir, island_id=store.island_id,
                                             include_archived=True)
                with _idle_diagnostics(store.coral_dir, node_id) as idle:
                    if idle and prune_completed_node_checkpoints(
                            store.coral_dir, node_id, attempts, island_id=store.island_id):
                        if record['research'].get('protocol') == 'staged-v2':
                            from .runtime_artifacts import RuntimeArtifacts
                            from .staged_research import node_idle
                            if not node_idle(store.coral_dir, node_id):
                                continue
                            RuntimeArtifacts(store.coral_dir).cleanup_node(node_id, terminal=True, _already_locked=True)
                        cleanup['status'] = 'done'
        except (ValueError, OSError, KeyError, TypeError) as exc:
            cleanup['last_error'] = str(exc)
        if record.get('cleanup') != cleanup:
            record['cleanup'] = cleanup
            changed = True
    return changed
