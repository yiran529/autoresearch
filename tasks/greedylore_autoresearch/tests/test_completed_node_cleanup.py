"""Completed-Node tensor reclamation through the real workflow lifecycle."""

# ruff: noqa: S101, F811
import fcntl
import json
import os
from pathlib import Path

import pytest
from autoresearch.diagnostics import create_task
from test_checkpoint_retention import checkpoint
from test_diagnostics import request
from test_workflow_state import (
    _active_research,
    _engineer_with_attempts,
    _publish_analysis,
    _publish_research_evidence,
    _state,
    _store,
)
from test_workflow_state import engineering_ready as engineering_ready
from test_workflow_state import root_run as root_run

from coral.hub.attempts import read_attempts, write_attempt


def _skipped_engineering_node(coral):
    store, assignment = _active_research(coral)
    candidate = _publish_research_evidence(coral, assignment.node_id, quality=0.0)
    (coral / 'public/notes/nodes/N001/research_analysis_v1.md').unlink()
    candidate.metadata['grader_runs'] = [{'run_id': 'quality-run', 'status': 'regressed',
        'log_dir': f'eval_logs/{candidate.commit_hash}/quality-run'}]
    write_attempt(coral, candidate)
    root = checkpoint(coral, candidate, 'quality-run')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['contract_hash'] = 'fixed-contract'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    store.request_finish(assignment.agent_id, 'research')
    store.advance()
    return store, candidate, root


def test_research_finish_keeps_selected_until_valid_analysis_completes(root_run):
    store, _, root = _skipped_engineering_node(root_run)
    assert (root / 'model.pt').exists()
    node = root_run / 'public/notes/nodes/N001'
    original = (root / 'manifest.json').read_bytes()
    _publish_analysis(root_run, 'N001')
    analysis = (node / 'research_analysis_v1.md').read_bytes()
    store.reconcile(0, 0)
    assert not (root / 'model.pt').exists()
    assert (root / 'manifest.json').read_bytes() == original
    assert (node / 'research_analysis_v1.md').read_bytes() == analysis
    assert _state(root_run)['nodes']['N001']['cleanup']['status'] == 'done'
    assert store.is_node_completed('N001')
    # A reopened store must recognize deletion and still accept this Parent.
    restored = _store(root_run)
    assert restored.is_node_completed('N001')
    revision = _state(root_run)['revision']
    restored.advance()
    assert _state(root_run)['revision'] == revision


@pytest.mark.parametrize('last', ['analysis', 'engineering'])
def test_cleanup_waits_for_both_branches_in_either_order(engineering_ready, last):
    store, _, body = _engineer_with_attempts(engineering_ready, analysis_pending=True)
    research = next(a for a in read_attempts(store.coral_dir) if a.metadata.get('node_id') == 'N001'
                    and a.metadata['mode'] == 'research')
    root = checkpoint(store.coral_dir, research)
    store.advance()
    if last == 'analysis':
        store.submit_engineering_analysis('engineer-E0001', body)
        store.advance()
    else:
        _publish_analysis(store.coral_dir, 'N001')
        store.advance()
    assert (root / 'model.pt').exists()
    if last == 'analysis':
        _publish_analysis(store.coral_dir, 'N001')
    else:
        store.submit_engineering_analysis('engineer-E0001', body)
    store.reconcile(0, 1)
    assert not (root / 'model.pt').exists()
    assert store.is_node_completed('N001')


@pytest.mark.parametrize('busy', ['task_lock', 'session_lock', 'live_analyzer'])
def test_cleanup_waits_for_diagnostic_to_drain_without_blocking(root_run, busy):
    store, candidate, root = _skipped_engineering_node(root_run)
    task = create_task(root_run, 'N001', 'research', 1, candidate.commit_hash)
    if busy == 'live_analyzer':
        task.bind(os.getpid())
    _publish_analysis(root_run, 'N001')
    lock = None
    if busy != 'live_analyzer':
        lock = (task.private / ('task.lock' if busy == 'task_lock' else 'session.lock')).open('a+b')
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        store.advance()
        assert (root / 'model.pt').exists()
        assert _state(root_run)['nodes']['N001']['cleanup']['status'] == 'pending'
        assert store.is_node_completed('N001')
    finally:
        if lock:
            lock.close()
        else:
            task.close()
    store.advance()
    assert not (root / 'model.pt').exists()
    assert _state(root_run)['nodes']['N001']['cleanup']['status'] == 'done'


@pytest.mark.parametrize('operation', ['bind', 'run', 'session'])
def test_completed_node_rejects_new_or_queued_diagnostic(root_run, operation):
    store, candidate, _ = _skipped_engineering_node(root_run)
    task = create_task(root_run, 'N001', 'research', 1, candidate.commit_hash)
    task.bind(os.getpid())
    _publish_analysis(root_run, 'N001')
    store.advance()
    if operation != 'run':
        task.close()
    with pytest.raises(PermissionError, match='completed Node'):
        if operation == 'bind':
            task.bind(os.getpid())
        elif operation == 'session':
            with task.session():
                pytest.fail('Analyzer staging started after completion')
        else:
            task.run(request(), os.getpid(), lambda *_: pytest.fail('diagnostic ran after completion'))


def test_cleanup_failure_preserves_completion_and_retries_after_restart(root_run, monkeypatch):
    store, _, root = _skipped_engineering_node(root_run)
    _publish_analysis(root_run, 'N001')
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path == root / 'rank-0.pt':
            persisted = _state(root_run)['nodes']['N001']
            assert persisted['research']['status'] == 'completed'
            assert persisted['analysis']['status'] == 'completed'
            raise OSError('simulated cleanup interruption')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', interrupted)
    store.advance()
    record = _state(root_run)['nodes']['N001']
    assert record['cleanup']['status'] == 'pending'
    assert 'simulated cleanup interruption' in record['cleanup']['last_error']
    assert not record.get('error')
    assert store.is_node_completed('N001')
    monkeypatch.setattr(Path, 'unlink', unlink)
    _store(root_run).advance()
    assert not (root / 'rank-0.pt').exists()
    assert _state(root_run)['nodes']['N001']['cleanup'] == {'status': 'done', 'last_error': None}


def test_completed_flags_without_valid_analysis_cannot_authorize_cleanup(root_run):
    store, _, root = _skipped_engineering_node(root_run)
    state = _state(root_run)
    state['nodes']['N001']['analysis'].update(
        status='completed', analysis_ref='nodes/N001/research_analysis_v1.md')
    store.state_path.write_text(json.dumps(state))
    store.advance()
    assert (root / 'model.pt').exists()
    assert _state(root_run)['nodes']['N001'].get('cleanup', {}).get('status') != 'done'
