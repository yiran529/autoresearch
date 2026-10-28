"""Independent Nodes must progress without waiting on another Node's journal lock."""
# ruff: noqa: S101

import fcntl
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import test_workflow_state
import yaml
from test_research_record import design
from test_staged_research import Backend

root_run = test_workflow_state.root_run


@pytest.fixture
def nodes(root_run, monkeypatch):
    from autoresearch.research_record import ResearchRecord
    from autoresearch.staged_research import StagedResearch
    from autoresearch.state import WorkflowStore

    (root_run / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'max_active_nodes': 2}}}))
    store = WorkflowStore(root_run)
    assignments = store.reconcile(2, 2)
    state = store._load_unlocked()
    for assignment in assignments:
        state['nodes'][assignment.node_id]['research']['status'] = 'active'
    store.state_path.write_text(json.dumps(state))
    hosts, backends = {}, {}
    for assignment in assignments:
        record = ResearchRecord(store, assignment.agent_id, assignment.node_id)
        record.initialize({'hypothesis': 'independent scheduling'}, design(), 'init')
        backend = Backend()
        host = StagedResearch(store, assignment.agent_id, assignment.node_id, backend=backend)
        host.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': 'fixed'}
        host.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
        hosts[assignment.node_id], backends[assignment.node_id] = host, backend
    monkeypatch.setattr('autoresearch.staged_execution.Execution',
                        lambda host: backends[host.node_id])
    assert set(hosts) == {'N001', 'N002'}
    return store, hosts, backends


def test_busy_node_drain_returns_without_changing_or_reexecuting_job(nodes):
    _, hosts, backends = nodes
    host = hosts['N001']
    saved = host.path.read_bytes()
    with host.path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(host.drain)
            try:
                assert future.result(timeout=2) is False
                assert host.path.read_bytes() == saved
                assert backends['N001'].calls == []
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def test_targeted_worker_does_not_visit_another_queued_node(nodes):
    from autoresearch.staged_research import run_worker

    store, hosts, _ = nodes
    saved = hosts['N001'].path.read_bytes()
    run_worker(str(store.coral_dir), only_node='N002')
    assert hosts['N002'].wait('train')['status'] == 'completed'
    assert hosts['N001'].path.read_bytes() == saved


def test_recover_action_launches_only_its_own_node(nodes, monkeypatch):
    from autoresearch.staged_research import handle_action, run_worker

    store, hosts, _ = nodes
    monkeypatch.setattr('coral.grader.trusted_worker.start_task_worker',
                        lambda coral, entrypoint, **options: run_worker(coral, **options.get('kwargs', {})))
    handle_action(store, 'research-N002', {'node_id': 'N002', 'action': 'recover',
                  'data': {}, 'request_id': 'recover'}, {})
    assert hosts['N002'].wait('train')['status'] == 'completed'
    assert hosts['N001'].wait('train')['status'] == 'queued'


def test_drain_picks_up_next_job_enqueued_while_publishing_previous_job(nodes, monkeypatch):
    _, hosts, backends = nodes
    host = hosts['N001']
    original_save = host._save
    added = False

    def save(state):
        nonlocal added
        original_save(state)
        if not added and state['jobs']['train']['status'] == 'completed':
            added = True
            host.record.append_design(design('second'), 'design2')
            host.enqueue('train', {'iteration': 2, 'sha': 'c' * 40}, 'train2', {})

    monkeypatch.setattr(host, '_save', save)
    host.drain()
    assert host.wait('train2')['status'] == 'completed'
    assert len(backends['N001'].calls) == 2


def test_recovery_starts_second_node_before_first_node_finishes(nodes):
    from autoresearch.staged_research import run_worker

    store, hosts, backends = nodes
    first_started, second_started, release = threading.Event(), threading.Event(), threading.Event()
    first_run, second_run = backends['N001'].run, backends['N002'].run

    def first(host, job):
        first_started.set()
        assert release.wait(5)
        return first_run(host, job)

    def second(host, job):
        second_started.set()
        return second_run(host, job)

    backends['N001'].run, backends['N002'].run = first, second
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_worker, str(store.coral_dir))
        try:
            assert first_started.wait(2)
            assert second_started.wait(2), 'N002 waited for N001 to finish'
        finally:
            release.set()
        future.result(timeout=3)
    assert all(host.wait('train')['status'] == 'completed' for host in hosts.values())


def test_duplicate_node_worker_does_not_interrupt_or_repeat_training(nodes):
    _, hosts, backends = nodes
    host, backend = hosts['N001'], backends['N001']
    started, release = threading.Event(), threading.Event()
    original = backend.run

    def run(host, job):
        started.set()
        assert release.wait(5)
        return original(host, job)

    backend.run = run
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(host.drain)
        assert started.wait(2)
        second = pool.submit(host.drain)
        try:
            assert second.result(timeout=2) is False
            assert host.wait('train')['status'] == 'running'
        finally:
            release.set()
        first.result(timeout=3)
    assert len(backend.calls) == 1
    assert len(host.record.read()['iterations'][0]['runs']) == 1


@pytest.mark.parametrize('only_node', [None, 'N002'])
def test_worker_respects_node_admission_when_capacity_is_reduced(nodes, only_node):
    from autoresearch.staged_research import run_worker

    store, hosts, _ = nodes
    (store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'max_active_nodes': 1}}}))
    run_worker(str(store.coral_dir), only_node=only_node)
    assert hosts['N002'].wait('train')['status'] == 'queued'


@pytest.mark.parametrize('node_id', ['../N001', 'N0001', '', 1])
def test_targeted_worker_rejects_invalid_node_id(nodes, node_id):
    from autoresearch.staged_research import run_worker

    store, hosts, _ = nodes
    with pytest.raises(ValueError, match='Node'):
        run_worker(str(store.coral_dir), only_node=node_id)
    assert all(host.wait('train')['status'] == 'queued' for host in hosts.values())
