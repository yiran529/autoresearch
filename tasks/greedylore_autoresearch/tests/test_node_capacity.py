"""Node slots cover the whole lifecycle, including recovery of an older backlog."""

# ruff: noqa: S101

import hashlib
import json

import pytest
import yaml
from autoresearch.state import WorkflowStore
from test_analysis_recovery import _fake_code_snapshot
from test_workflow_state import (
    _git,
    _publish_research_evidence,
    _research_request,
)
from test_workflow_state import (
    engineering_ready as engineering_ready,
)
from test_workflow_state import (
    root_run as root_run,
)

from coral.hub.attempts import read_attempt, write_attempt
from coral.types import Attempt


def backlog(root):
    store = WorkflowStore(root)
    for item in store.reconcile(4, 0):
        store.initialize_research(item.agent_id, item.node_id, _research_request())
    (root / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'max_active_nodes': 2}},
    }))
    return WorkflowStore(root)


def runnable(roster):
    return [a.node_id for a in roster if a.role == 'research' and a.status != 'waiting']


def test_resume_defers_existing_nodes_without_rewriting_specs(root_run):
    store = backlog(root_run)
    path = root_run / 'public/notes/nodes/N003/spec.md'
    original = path.read_bytes()
    roster = store.reconcile(2, 0)
    assert runnable(roster) == ['N001', 'N002']
    assert {a.node_id for a in roster if a.status == 'waiting'} == {'N003', 'N004'}
    assert path.read_bytes() == original
    assert runnable(WorkflowStore(root_run).reconcile(2, 0)) == ['N001', 'N002']


def test_research_end_keeps_slot_until_analysis_published_then_releases_one(root_run):
    store = backlog(root_run)
    _publish_research_evidence(root_run, 'N001', quality=0.0)
    path = root_run / 'public/notes/nodes/N001/research_analysis_v1.md'
    report = path.read_bytes()
    path.unlink()
    store.request_finish('research-N001', 'research')
    assert runnable(store.reconcile(2, 0)) == ['N002']
    path.write_bytes(report)
    assert runnable(store.reconcile(2, 0)) == ['N002', 'N003']
    state = json.loads(store.state_path.read_text())
    assert set(state['nodes']) == {'N000', 'N001', 'N002', 'N003', 'N004'}


def test_unfinished_engineering_keeps_node_slot(root_run):
    store = backlog(root_run)
    _publish_research_evidence(root_run, 'N001', quality=0.0)
    store.request_finish('research-N001', 'research')
    store.advance()
    state = json.loads(store.state_path.read_text())
    state['nodes']['N001']['engineering_status'] = 'pending'
    state['nodes']['N001'].pop('engineering_skip_reason', None)
    store.state_path.write_text(json.dumps(state))
    assert runnable(store.reconcile(2, 0)) == ['N002']


@pytest.mark.parametrize('value', [-1, True, '2', 1.5])
def test_invalid_node_capacity_is_rejected(root_run, value):
    (root_run / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'max_active_nodes': value}},
    }))
    with pytest.raises(ValueError, match='max_active_nodes'):
        WorkflowStore(root_run)


def test_waiting_node_analyzer_is_not_launched(root_run):
    from autoresearch.lifecycle import run_worker

    store = backlog(root_run)
    # This report was produced before the limit was enabled, but is not published.
    (root_run / 'config.yaml').unlink()
    old_store = WorkflowStore(root_run)
    _publish_research_evidence(root_run, 'N003', quality=0.0)
    old_store.request_finish('research-N003', 'research')
    (root_run / 'public/notes/nodes/N003/research_analysis_v1.md').unlink()
    old_store.advance()
    (root_run / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'max_active_nodes': 2}},
    }))
    before = json.loads(store.state_path.read_text())['nodes']['N003']['analysis']
    calls = []

    def runner(*args):
        calls.append(args)
        raise AssertionError('waiting Analyzer must not run')

    def snapshot(evidence, context, scratch):
        target = scratch / 'code'
        target.mkdir()
        return target

    result = run_worker(str(root_run), only_node='N003', runner=runner,
                        code_snapshotter=snapshot)
    after = json.loads(store.state_path.read_text())['nodes']['N003']['analysis']
    assert result['errors'] == {}
    assert calls == []
    assert after == before


def test_task_dispatch_uses_node_slots_and_preserves_interrupted_runs(root_run):
    from autoresearch.state import filter_dispatch

    store = backlog(root_run)
    candidates = []
    for number in [1, 3]:
        sha = f'{number:040x}'
        write_attempt(root_run, Attempt(
            commit_hash=sha, agent_id=f'research-N{number:03d}',
            title='interrupted', status='pending', score=None, parent_hash=None,
            timestamp='2026-01-01T00:00:00+00:00',
            metadata={'node_id': f'N{number:03d}', 'mode': 'research',
                      'grader_runs': [{'run_id': 'previous', 'status': 'running'}]}))
        candidates.append({'commit_hash': sha, 'island_id': None})
    before = read_attempt(root_run, f'{3:040x}').to_dict()
    result = filter_dispatch(candidates=candidates, context={'coral_dir': str(root_run)})
    assert result == {'eligible_hashes': [f'{1:040x}']}
    assert read_attempt(root_run, f'{3:040x}').to_dict() == before
    # Model a historical operator cancellation in the persisted resume backlog.
    state = json.loads(store.state_path.read_text())
    state['nodes']['N001']['research']['status'] = 'cancelled'
    store.state_path.write_text(json.dumps(state))
    assert filter_dispatch(candidates=candidates, context={'coral_dir': str(root_run)}) == {
        'eligible_hashes': [f'{3:040x}']}


def test_waiting_node_cannot_submit_another_candidate(root_run):
    from autoresearch.state import authorize_attempt

    backlog(root_run)
    with pytest.raises(ValueError, match='Node slot'):
        authorize_attempt({'node_id': 'N003', 'mode': 'research'}, {
            'coral_dir': str(root_run), 'agent_id': 'research-N003',
        })


def test_deferred_engineer_does_not_block_admitted_node(engineering_ready):
    root_run, repo, _ = engineering_ready
    store = WorkflowStore(root_run, repo_path=repo)
    store.reconcile(4, 0)
    (root_run / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'max_active_nodes': 2}},
    }))
    store = WorkflowStore(root_run, repo_path=repo)
    state = json.loads(store.state_path.read_text())
    # The earlier run had already bound an Engineer to a now-deferred Node.
    state['engineering_workers']['engineer-E0001'] = {
        'status': 'active', 'node_id': 'N003', 'research_sha': 'e' * 40,
    }
    state['nodes']['N003']['research']['status'] = 'completed'
    state['nodes']['N003']['engineering_status'] = 'active'
    state['nodes']['N003']['engineering_claims'] = [{
        'agent_id': 'engineer-E0001', 'status': 'active', 'research_sha': 'e' * 40,
    }]
    state['next_engineer_seq'] = 2
    store.state_path.write_text(json.dumps(state))
    engineers = {a.node_id: a.status for a in store.reconcile(0, 0)
                 if a.role == 'engineering'}
    assert engineers == {'N003': 'waiting', 'N001': 'active'}
    store.cancel_assignment('research-N002')
    engineers = {a.node_id: a.status for a in store.reconcile(0, 0)
                 if a.role == 'engineering'}
    assert engineers == {'N003': 'active', 'N001': 'active'}


def test_node_mode_ignores_legacy_role_caps_at_manager_entry(root_run):
    from autoresearch.state import handle_workflow

    backlog(root_run)
    result = handle_workflow('reconcile_workflow', {}, {
        'coral_dir': str(root_run), 'is_manager': True,
        'max_researchers': 0, 'max_engineers': 0,
    })
    assert [a['node_id'] for a in result['assignments']
            if a['role'] == 'research' and a['status'] != 'waiting'] == ['N001', 'N002']


@pytest.mark.parametrize('layout', ['configured', 'discovered', 'custom', 'source_config'])
@pytest.mark.parametrize('entrypoint', ['completion', 'authorize', 'dispatch'])
def test_all_entrypoints_release_slots_after_pinned_code_analysis(engineering_ready, layout, entrypoint):
    from autoresearch.lifecycle import run_worker
    from autoresearch.state import authorize_attempt, filter_dispatch

    root, repo, sha = engineering_ready
    store = WorkflowStore(root, repo_path=repo)
    attempt = read_attempt(root, sha)
    attempt.score = 0.0
    attempt.status = 'regressed'
    attempt.metadata['scores']['quality_gate']['value'] = 0.0
    write_attempt(root, attempt)
    for item in store.reconcile(2, 0):
        if item.role == 'research':
            store.initialize_research(item.agent_id, item.node_id, _research_request())

    def runner(_prompt, _roots, _schema, scratch):
        out = scratch / 'output.json'
        out.write_text(json.dumps({
            'summary': 'Fixed code was inspected.', 'findings': ['Recorded failure.'],
            'limitations': ['One trial.'], 'handover': {
                'mechanism_verdict': 'inconclusive', 'practical_verdict': 'not_solved',
                'evidence_refs': [f'code/{sha}/model.txt'],
                'new_information': ['A fixed source file.'], 'follow_up_questions': [],
            },
        }))
        return out

    result = run_worker(str(root), checkout_root=repo, only_node='N001',
                        runner=runner, code_snapshotter=_fake_code_snapshot)
    assert result['errors'] == {}
    assert store.is_node_completed('N001')
    if layout == 'custom':
        renamed = repo.with_name('custom-repository')
        repo.rename(renamed)
        repo = renamed
    config = {'grader': {'args': {'max_active_nodes': 2}}}
    if layout != 'discovered':
        config['workspace'] = {'repo_path': str(repo)}
    if layout == 'source_config':
        # create_project saves the source path before cloning run/repo. Later
        # Agent commits exist only in the run clone, not this source repository.
        source = repo.with_name('source-repository')
        source.mkdir()
        _git(source, 'init', '-q')
        _git(source, 'config', 'user.name', 'Test')
        _git(source, 'config', 'user.email', 'test@example.invalid')
        (source / 'model.txt').write_text('source baseline\n')
        _git(source, 'add', '.')
        _git(source, 'commit', '-qm', 'original source')
        config['workspace'] = {'repo_path': str(source)}
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    manager = WorkflowStore(root, repo_path=repo)
    assert manager._admitted_node_ids_unlocked(manager._load_unlocked()) == {'N002', 'N003'}
    before = hashlib.sha256(store.state_path.read_bytes()).hexdigest()
    if entrypoint == 'completion':
        assert WorkflowStore(root).is_node_completed('N001')
    elif entrypoint == 'authorize':
        assert authorize_attempt({'node_id': 'N003', 'mode': 'research'}, {
            'coral_dir': str(root), 'agent_id': 'research-N003',
        }) == {'allow_multiple_research': True}
    else:
        write_attempt(root, Attempt(
            commit_hash='f' * 40, agent_id='research-N003', title='queued',
            status='pending', score=None, parent_hash=None,
            timestamp='2026-01-01T00:00:00+00:00',
            metadata={'node_id': 'N003', 'mode': 'research'}))
        assert filter_dispatch(candidates=[{'commit_hash': 'f' * 40, 'island_id': None}],
                               context={'coral_dir': str(root)}) == {'eligible_hashes': ['f' * 40]}
    assert hashlib.sha256(store.state_path.read_bytes()).hexdigest() == before
