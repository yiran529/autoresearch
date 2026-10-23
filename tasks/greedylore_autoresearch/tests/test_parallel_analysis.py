"""Independent Node Analyses make progress without duplicate sessions."""

# ruff: noqa: S101

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from autoresearch.lifecycle import _run_node, run_worker
from autoresearch.state import WorkflowStore
from test_analysis_recovery import _fake_code_snapshot
from test_workflow_state import _research_request
from test_workflow_state import root_run as root_run

from coral.hub.attempts import write_attempt
from coral.types import Attempt


def ready_nodes(root):
    store = WorkflowStore(root)
    for item in store.reconcile(2, 0):
        store.initialize_research(item.agent_id, item.node_id, _research_request())
        write_attempt(root, Attempt(
            commit_hash=item.node_id[-1] * 40, agent_id=item.agent_id,
            title='finished research', score=0.0, status='regressed', parent_hash=None,
            timestamp='2026-01-01T00:00:00+00:00',
            metadata={'mode': 'research', 'node_id': item.node_id,
                      'contract_hash': 'fixed-contract', 'grader_runs': [],
                      'scores': {'quality_gate': {'value': 0.0}}}))
        store.request_finish(item.agent_id, 'research')
    store.advance()
    return store


def response(scratch):
    out = scratch / 'output.json'
    out.write_text(json.dumps({'summary': 'Independent diagnosis.',
                              'findings': ['Recorded outcome.'],
                              'limitations': ['No broader inference.']}))
    return out


def node_id(scratch):
    return yaml.safe_load((scratch / 'evidence/spec.md').read_text())['node_id']


def test_two_node_analyzers_run_concurrently(root_run):
    store = ready_nodes(root_run)
    barrier = threading.Barrier(2)

    def runner(_prompt, _roots, _schema, scratch):
        barrier.wait(timeout=3)
        return response(scratch)

    result = run_worker(str(root_run), runner=runner, code_snapshotter=_fake_code_snapshot)
    assert result['errors'] == {}
    assert set(result['completed']) == {'research-N001', 'research-N002'}
    assert store.is_node_completed('N001') and store.is_node_completed('N002')


@pytest.mark.parametrize('error', [
    'HTTP 503 Service Unavailable', 'HTTP 500 Internal Server Error',
    'Selected model is at capacity',
])
def test_analyzer_server_errors_back_off_without_blocking_or_charging_attempts(root_run, error):
    store = ready_nodes(root_run)
    attempts = {p: p.read_bytes() for p in (root_run / 'public/attempts').glob('*.json')}

    def failed_runner(*_):
        raise RuntimeError(error)

    clock = 1000
    for delay in [60, 120, 240, 480]:
        result = _run_node(store, 'N001', failed_runner, _fake_code_snapshot, lambda: clock)
        assert result['errors']['research-N001'] == error
        job = json.loads(store.state_path.read_text())['nodes']['N001']['analysis']
        assert job['status'] == 'pending'
        assert job.get('validation_failures', 0) == 0
        assert job['next_retry_at'] == clock + delay
        # No model call is made before the persisted retry deadline.
        deferred = _run_node(store, 'N001',
            lambda *_: pytest.fail('must honor retry deadline'), _fake_code_snapshot, lambda: clock)
        assert deferred['status'] == 'deferred'
        clock += delay
    recovered = _run_node(store, 'N001', lambda *args: response(args[-1]),
                          _fake_code_snapshot, lambda: clock)
    assert recovered['completed'] == ['research-N001']
    assert store.is_node_completed('N001')
    assert {p: p.read_bytes() for p in (root_run / 'public/attempts').glob('*.json')} == attempts


def test_retry_can_finish_while_other_node_analyzer_is_still_busy(root_run):
    store = ready_nodes(root_run)
    release = threading.Event()
    retried = threading.Event()
    clock = [1000]
    calls = []

    def runner(_prompt, _roots, _schema, scratch):
        node = node_id(scratch)
        calls.append(node)
        if node == 'N001':
            if calls.count(node) == 1:
                raise RuntimeError('transient Analyzer failure')
            retried.set()
        else:
            assert release.wait(5)
        return response(scratch)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_worker, str(root_run), runner=runner,
                             code_snapshotter=_fake_code_snapshot, now=lambda: clock[0])
        try:
            deadline = time.monotonic() + 3
            while True:
                job = json.loads(store.state_path.read_text())['nodes']['N001']['analysis']
                if job['failures'] == 1:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            clock[0] = 2000
            assert retried.wait(3), 'retry was blocked behind another Node Analyzer'
        finally:
            release.set()
        result = future.result(timeout=5)
    assert result['errors'] == {}
    assert calls.count('N001') == 2 and calls.count('N002') == 1
    assert store.is_node_completed('N001') and store.is_node_completed('N002')


def test_node_lock_defers_only_that_node_without_charging_an_attempt(root_run):
    import fcntl

    store = ready_nodes(root_run)
    folder = root_run / 'private/analysis-locks'
    folder.mkdir()
    with (folder / 'N001.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_worker(str(root_run), runner=lambda *args: response(args[-1]),
                            code_snapshotter=_fake_code_snapshot)
    state = json.loads(store.state_path.read_text())
    assert state['nodes']['N001']['analysis']['generation'] == 0
    assert state['nodes']['N001']['analysis']['failures'] == 0
    assert result['completed'] == ['research-N002']


def test_invalid_existing_report_returns_error_without_spinning(root_run):
    store = ready_nodes(root_run)
    report = root_run / 'public/notes/nodes/N001/research_analysis_v1.md'
    report.write_text('invalid stored report')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_worker, str(root_run), only_node='N001',
                             runner=lambda *_: pytest.fail('must not overwrite existing report'))
        try:
            result = future.result(timeout=2)
        finally:
            (root_run / 'private/workflow_stopping').touch()
    assert 'research-N001' in result['errors']
    assert report.read_text() == 'invalid stored report'
    assert json.loads(store.state_path.read_text())['nodes']['N001']['analysis']['generation'] == 0
