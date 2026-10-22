"""Exclusive method selection through the real workflow and trusted boundary."""

# ruff: noqa: S101

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import yaml
from autoresearch.state import WorkflowStore, handle_workflow
from test_workflow_state import _research_request
from test_workflow_state import root_run as root_run

from coral.grader.trusted_writes import AgentBinding, TrustedWriteService, process_start_time


@pytest.fixture
def planning(root_run, monkeypatch):
    (root_run / 'config.yaml').write_text(yaml.safe_dump({
        'grader': {'args': {'research_planning_lease_seconds': 1800}}}))
    store = WorkflowStore(root_run)
    assignments = store.reconcile(2, 0)
    clock = [1000.0]
    monkeypatch.setattr('time.time', lambda: clock[0])
    def binding(assignment):
        return AgentBinding(assignment.agent_id, os.getpid(), process_start_time(os.getpid()),
                            str(root_run.parent), None)
    service = TrustedWriteService(root_run, SimpleNamespace(grader=SimpleNamespace(args={})),
                                  workflow_handler=handle_workflow)
    return store, assignments, clock, service, binding


def acquire(fixture, index=0):
    _, assignments, _, service, binding = fixture
    return service.handle_authenticated('acquire_research_planning', {}, binding(assignments[index]))


def initialize(fixture, index, token, **changes):
    _, assignments, _, service, binding = fixture
    return service.handle_authenticated('initialize_research', {
        'node_id': assignments[index].node_id, **_research_request(),
        'planning_token': token, 'overlap_review': 'Different feedback state; checked existing Specs.',
        **changes}, binding(assignments[index]))


def test_only_one_researcher_can_hold_the_selection_lock(planning):
    first = acquire(planning)
    assert first['status'] == 'acquired'
    assert first['registry_ref'] == 'tree.md'
    assert acquire(planning)['token'] == first['token']
    other = acquire(planning, 1)
    assert other['status'] == 'busy' and other['owner_node'] == 'N001'
    assert 'token' not in other
    with pytest.raises(ValueError, match='planning'):
        initialize(planning, 1, first['token'])


def test_registration_is_immutable_and_releases_lock_for_the_next_reader(planning):
    store, assignments, _, _, _ = planning
    token = acquire(planning)['token']
    result = initialize(planning, 0, token)
    path = store.notes_root / result['spec_ref']
    before = path.read_bytes()
    spec = yaml.safe_load(before)
    assert spec['method_registration']['overlap_review'].startswith('Different feedback')
    second = acquire(planning, 1)
    assert second['status'] == 'acquired'
    assert 'N001' in (store.notes_root / second['registry_ref']).read_text()
    assert path.is_file()
    assert initialize(planning, 0, token) == result
    assert path.read_bytes() == before
    assert acquire(planning, 1)['token'] == second['token']
    assert json.loads(store.state_path.read_text())['nodes'][assignments[0].node_id]['research']['status'] == 'active'


def test_new_spec_requires_lock_and_a_nonempty_overlap_review(planning):
    _, assignments, _, service, binding = planning
    payload = {'node_id': 'N001', **_research_request()}
    with pytest.raises(ValueError, match='planning'):
        service.handle_authenticated('initialize_research', payload, binding(assignments[0]))
    token = acquire(planning)['token']
    with pytest.raises(ValueError, match='overlap_review'):
        initialize(planning, 0, token, overlap_review=' ')
    assert acquire(planning, 1)['status'] == 'busy'


def test_timeout_fences_stale_registration_and_requires_fresh_selection(planning):
    _, _, clock, _, _ = planning
    old = acquire(planning)['token']
    clock[0] = 2800
    assert acquire(planning, 1)['status'] == 'acquired'
    with pytest.raises(ValueError, match='planning'):
        initialize(planning, 0, old)


def test_renew_and_release_require_the_current_owner_and_token(planning):
    _, assignments, clock, service, binding = planning
    token = acquire(planning)['token']
    clock[0] = 2000
    renewed = service.handle_authenticated('renew_research_planning', {'token': token}, binding(assignments[0]))
    assert renewed['expires_at'] == 3800
    with pytest.raises(ValueError, match='planning'):
        service.handle_authenticated('release_research_planning', {'token': token}, binding(assignments[1]))
    service.handle_authenticated('release_research_planning', {'token': token}, binding(assignments[0]))
    assert acquire(planning, 1)['status'] == 'acquired'


def test_cancelled_owner_does_not_block_another_method_selection(planning):
    store, assignments, _, _, _ = planning
    acquire(planning)
    store.cancel_assignment(assignments[0].agent_id)
    assert acquire(planning, 1)['status'] == 'acquired'


def test_owner_process_identity_change_reclaims_lock_and_fences_old_token(planning):
    store, _, _, _, _ = planning
    old = acquire(planning)['token']
    state = json.loads(store.state_path.read_text())
    state['research_planning']['root_start_time'] += 1
    store.state_path.write_text(json.dumps(state))
    assert acquire(planning, 1)['status'] == 'acquired'
    with pytest.raises(ValueError, match='planning'):
        initialize(planning, 0, old)


def test_concurrent_acquisition_has_one_winner_and_survives_store_reload(planning):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: acquire(planning, i), [0, 1]))
    assert sorted(r['status'] for r in results) == ['acquired', 'busy']
    index = next(i for i, r in enumerate(results) if r['status'] == 'acquired')
    assert acquire(planning, index)['token'] == results[index]['token']


def test_engineer_cannot_acquire_research_selection_lock(planning):
    store, _, _, service, _ = planning
    engineer = AgentBinding('engineer-E0001', os.getpid(), process_start_time(os.getpid()),
                             str(store.coral_dir.parent), None)
    with pytest.raises(ValueError, match='Research'):
        service.handle_authenticated('acquire_research_planning', {}, engineer)


def test_selection_lock_does_not_block_normal_workflow_reconcile(planning):
    store, _, _, _, _ = planning
    token = acquire(planning)['token']
    assert len(store.reconcile(2, 0)) == 2
    assert acquire(planning)['token'] == token


def test_cli_acquires_registers_and_releases_through_real_trusted_entry(planning, monkeypatch, capsys):
    from autoresearch import cli

    store, assignments, _, service, binding = planning
    monkeypatch.setattr(cli, '_coral_dir', lambda: store.coral_dir)
    monkeypatch.setattr(cli, '_agent_id', lambda: assignments[0].agent_id)
    monkeypatch.setattr(cli, '_require_clean_worktree', lambda: None)
    monkeypatch.setattr(cli, '_checkout_base', lambda _sha: None)
    monkeypatch.setattr(cli, 'request_trusted', lambda _c, op, data:
                        service.handle_authenticated(op, data, binding(assignments[0])))
    cli.main(['node', 'planning', 'acquire', '--wait'])
    token = json.loads(capsys.readouterr().out)['token']
    cli.main(['node', 'initialize', '--parent', 'N000', '--hypothesis', 'new feedback',
              '--selection-rationale', 'different state', '--delta', 'state transition',
              '--risk', 'quality', '--falsification-criteria', 'matched replay',
              '--planning-token', token, '--overlap-review', 'Reviewed existing Specs.'])
    result = json.loads(capsys.readouterr().out)
    assert (store.notes_root / result['spec_ref']).is_file()
    assert acquire(planning, 1)['status'] == 'acquired'


def test_restarted_agent_must_reacquire_with_its_new_process_identity(planning):
    store, assignments, _, service, _ = planning
    old = acquire(planning)['token']
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        replacement = AgentBinding(assignments[0].agent_id, process.pid, process_start_time(process.pid),
                                   str(store.coral_dir.parent), None)
        with pytest.raises(ValueError, match='planning'):
            service.handle_authenticated('initialize_research', {
                'node_id': 'N001', **_research_request(), 'planning_token': old,
                'overlap_review': 'Saved decision from prior process.'}, replacement)
        with pytest.raises(ValueError, match='planning'):
            service.handle_authenticated('renew_research_planning', {'token': old}, replacement)
        fresh = service.handle_authenticated('acquire_research_planning', {}, replacement)
        assert fresh['status'] == 'acquired' and fresh['token'] != old
        with pytest.raises(ValueError, match='planning'):
            initialize(planning, 0, fresh['token'])
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize('recover_by', ['next_acquisition', 'manager_reconcile'])
def test_crash_after_spec_publication_recovers_registration_and_unlocks(planning, monkeypatch, recover_by):
    store, _, _, _, _ = planning
    token = acquire(planning)['token']
    save = WorkflowStore._save_unlocked
    failed = []

    def crash_once(self, state):
        if state['nodes']['N001']['research']['status'] == 'active' and not failed:
            failed.append(True)
            raise OSError('simulated crash after Spec publication')
        return save(self, state)

    monkeypatch.setattr(WorkflowStore, '_save_unlocked', crash_once)
    with pytest.raises(OSError, match='simulated crash'):
        initialize(planning, 0, token)
    assert (store.notes_root / 'nodes/N001/spec.md').is_file()
    assert json.loads(store.state_path.read_text())['nodes']['N001']['research']['status'] == 'reserved'
    if recover_by == 'manager_reconcile':
        store.reconcile(2, 0)
        recovered = json.loads(store.state_path.read_text())
        assert recovered['nodes']['N001']['research']['status'] == 'active'
        assert 'research_planning' not in recovered
    assert acquire(planning, 1)['status'] == 'acquired'
    assert json.loads(store.state_path.read_text())['nodes']['N001']['research']['status'] == 'active'
