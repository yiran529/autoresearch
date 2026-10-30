"""Routine admission trusts publication; transition boundaries still verify evidence."""

# ruff: noqa: S101

import json

import pytest
import yaml
from autoresearch.state import WorkflowError, WorkflowStore
from test_workflow_state import _research_request
from test_workflow_state import root_run as root_run


def ready_store(root):
    (root / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'max_active_nodes': 2}},
    }))
    store = WorkflowStore(root)
    store.reconcile(0, 0)
    return store


def test_routine_reconcile_does_not_revalidate_published_history(root_run, monkeypatch):
    store = ready_store(root_run)
    original = store.reconcile(2, 0)

    def unexpected(*args, **kwargs):
        raise AssertionError('published history must not rebuild evidence during scheduling')

    monkeypatch.setattr(store, '_validated_analysis_ref_unlocked', unexpected)
    monkeypatch.setattr(store, '_adopt_analysis_unlocked', unexpected)
    terminal = store._terminal_research_unlocked

    def active_only(node_id, **kwargs):
        assert node_id != 'N000', 'published Research must not reload historical Attempts'
        return terminal(node_id, **kwargs)

    monkeypatch.setattr(store, '_terminal_research_unlocked', active_only)
    assert store.reconcile(2, 0) == original
    assert WorkflowStore(root_run).reconcile(2, 0) == original


def test_reconcile_reuses_one_unfinished_scan_when_allocating(root_run, monkeypatch):
    store = ready_store(root_run)
    store.cancel_assignment('research-N001')
    scans = []
    original = store._unfinished_node_ids_unlocked

    def counted(state):
        scans.append(tuple(state['nodes']))
        return original(state)

    monkeypatch.setattr(store, '_unfinished_node_ids_unlocked', counted)
    roster = store.reconcile(2, 0)
    assert {a.node_id for a in roster} == {'N002', 'N003'}
    assert len(scans) == 1


@pytest.mark.parametrize('damage', ['missing', 'symlink', 'wrong_ref'])
def test_missing_or_noncanonical_publication_keeps_slot_after_restart(root_run, damage):
    store = ready_store(root_run)
    report = root_run / 'public/notes/nodes/N000/research_analysis_v1.md'
    if damage == 'wrong_ref':
        state = json.loads(store.state_path.read_text())
        state['nodes']['N000']['analysis']['analysis_ref'] = 'nodes/N001/research_analysis_v1.md'
        store.state_path.write_text(json.dumps(state))
    else:
        content = report.read_bytes()
        report.unlink()
        if damage == 'symlink':
            external = root_run.parent / 'external.md'
            external.write_bytes(content)
            report.symlink_to(external)
    resumed = WorkflowStore(root_run)
    assert 'N000' in resumed._unfinished_node_ids_unlocked(resumed._load_unlocked())


def test_content_corruption_is_still_rejected_at_explicit_and_parent_boundaries(root_run):
    store = ready_store(root_run)
    original = store.reconcile(2, 0)
    report = root_run / 'public/notes/nodes/N000/research_analysis_v1.md'
    document = json.loads(report.read_text())
    document['evidence_manifest_hash'] = 'corrupted'
    report.write_text(json.dumps(document))
    assert store.reconcile(2, 0) == original
    with pytest.raises(WorkflowError, match='valid Research Analysis'):
        store.is_node_completed('N000')
    with pytest.raises(WorkflowError, match='valid Research Analysis'):
        store.initialize_research('research-N001', 'N001', _research_request())


@pytest.mark.parametrize('damage', ['missing', 'symlink'])
def test_older_engineering_report_cannot_replace_the_frozen_publication(root_run, damage):
    store = ready_store(root_run)
    state = store._load_unlocked()
    node = state['nodes']['N000']
    node['engineering_claims'] = [{'agent_id': 'engineer-E0000', 'status': 'completed'}]
    node['engineering_status'] = 'completed'
    state['engineering_workers']['engineer-E0000'] = {
        'status': 'completed', 'node_id': 'N000', 'analysis_submission': {'version': 2},
    }
    root = store.notes_root / 'nodes/N000'
    (root / 'engineering_analysis_v1.md').write_text('{}')
    report = root / 'engineering_analysis_v2.md'
    report.write_text('{}')
    assert 'N000' not in store._unfinished_node_ids_unlocked(state)
    report.unlink()
    if damage == 'symlink':
        report.symlink_to(root / 'engineering_analysis_v1.md')
    assert 'N000' in store._unfinished_node_ids_unlocked(state)


@pytest.mark.parametrize('mode', ['research', 'engineering'])
@pytest.mark.parametrize('damage', ['missing', 'symlink'])
def test_modern_publication_requires_its_canonical_json_record(root_run, mode, damage):
    from coral.grader.analysis_artifacts import publish_analysis

    store = ready_store(root_run)
    state = store._load_unlocked()
    root = store.notes_root / 'nodes/N000'
    document = json.loads((root / 'research_analysis_v1.md').read_text())
    document['mode'] = mode
    report = root / f'{mode}_analysis_v1.md'
    report.unlink(missing_ok=True)
    publish_analysis(report, document)
    if mode == 'engineering':
        state['nodes']['N000']['engineering_claims'] = [
            {'agent_id': 'engineer-E0000', 'status': 'completed'},
        ]
        state['engineering_workers']['engineer-E0000'] = {
            'status': 'completed', 'node_id': 'N000', 'analysis_submission': {'version': 1},
        }
    assert 'N000' not in store._unfinished_node_ids_unlocked(state)
    record = report.with_suffix('.json')
    content = record.read_bytes()
    record.unlink()
    if damage == 'symlink':
        external = root_run.parent / 'external.json'
        external.write_bytes(content)
        record.symlink_to(external)
    assert 'N000' in store._unfinished_node_ids_unlocked(state)
