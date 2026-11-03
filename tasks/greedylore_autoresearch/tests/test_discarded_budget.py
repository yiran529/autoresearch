"""Deleting invalid outputs must not restore already spent execution budget."""
# ruff: noqa: S101
import json

import pytest
from test_research_record import design, initialize, record
from test_staged_research import host
from test_staged_timing import measure, timing_host


def test_engineering_counts_discarded_runs():
    from autoresearch.state import _engineering_budget

    assert _engineering_budget([], prior={'successful': 2, 'total': 2}) == (2, 2, 0)


@pytest.mark.parametrize('prior', [
    {'successful': -1, 'total': 0}, {'successful': 0, 'total': -1},
    {'successful': True, 'total': 1}, {'successful': 0, 'total': 1.5},
    {'successful': 0, 'total': '2'}, {'successful': 2, 'total': 1}, [],
])
def test_engineering_rejects_invalid_discarded_counts(prior):
    from autoresearch.state import _engineering_budget

    with pytest.raises(ValueError, match='prior budget'):
        _engineering_budget([], prior=prior)


@pytest.mark.parametrize('successes,allowed', [(1, True), (2, False)])
def test_engineering_admission_preserves_budget_after_output_deletion(tmp_path, successes, allowed):
    from autoresearch.state import authorize_attempt

    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    coral = tmp_path / '.coral'
    (coral / 'private').mkdir(parents=True)
    state = {'version': 2, 'revision': 0, 'next_node_seq': 2, 'next_engineer_seq': 2,
        'next_ready_seq': 2, 'nodes': {
            'N000': {'research': {}},
            'N001': {'research': {}, 'engineering_claims': [
                {'agent_id': 'engineer-E0001', 'status': 'active'}]}},
        'engineering_workers': {'engineer-E0001': {'status': 'active',
            'node_id': 'N001', 'prior_budget': {'successful': 2, 'total': 2}}}}
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    for number in range(successes):
        write_attempt(coral, Attempt(commit_hash=f'{number:040x}', agent_id='engineer-E0001',
            title='Fresh candidate', score=1., status='improved', parent_hash='a'*40,
            timestamp='2026-01-01T00:00:00+00:00', metadata={'mode': 'engineering', 'node_id': 'N001'}))
    context = {'coral_dir': str(coral), 'agent_id': 'engineer-E0001', 'commit_hash': 'f'*40}
    if allowed:
        assert authorize_attempt({'node_id': 'N001', 'mode': 'engineering'}, context) == {}
    else:
        with pytest.raises(ValueError, match='Engineering.*limit'):
            authorize_attempt({'node_id': 'N001', 'mode': 'engineering'}, context)


def test_new_research_record_retains_discarded_trajectory_count(tmp_path):
    r = record(tmp_path)
    r.private.parent.mkdir()
    r.private.write_text(json.dumps({'document': None, 'requests': {}, 'prior_trajectories': 1}))
    from autoresearch.staged_research import StagedResearch

    h = StagedResearch(r.store, r.agent_id, r.node_id, backend=object())
    assert h.status()['progress']['development'] == []
    initialize(r)
    assert r.read()['prior_trajectories'] == 1
    for iteration in (1, 2):
        r.begin(iteration, str(iteration) * 40, f'run-{iteration}')
        r.observe(iteration, {'status': 'failed'}, f'observe-{iteration}')
        if iteration == 1:
            r.append_design(design('second'), 'second')
    with pytest.raises(ValueError, match='budget'):
        r.append_design(design('third'), 'third')


def test_new_timing_jobs_include_discarded_measurements(tmp_path):
    h, _, backend = timing_host(tmp_path)
    state = h._read()
    state['prior_budget'] = {'timing_successful': 4, 'timing_total': 4}
    h._save(state)
    measure(h)
    with pytest.raises(ValueError, match='timing budget exhausted'):
        measure(h, 'extra', repeat_reason='Would exceed the original budget')
    assert len(backend.calls) == 1


def test_new_diagnostics_include_discarded_executions(tmp_path):
    h, _, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
    h.drain()
    state = h._read()
    state['prior_budget'] = {'diagnostic_successful': 0, 'diagnostic_total': 20}
    h._save(state)
    with pytest.raises(ValueError, match='diagnostic budget exhausted'):
        h.enqueue('diagnostic', {'iteration': 1, 'argv': ['python', 'probe.py'],
            'timeout': 60, 'scripts_dir': '/tmp/scripts'}, 'new-probe', {})
