"""Bounded multi-candidate Research and deterministic frozen selection."""

# ruff: noqa: S101, F811
import pytest
from test_workflow_state import _active_research, _state
from test_workflow_state import root_run as root_run

from coral.hub.attempts import write_attempt
from coral.types import Attempt


def candidate(sha='d', score=1.0, quality=1, *, status='improved', archived=False, **metadata):
    return Attempt(commit_hash=sha * 40, agent_id='research-N001', title='Research refinement',
        score=score, status=status, parent_hash='a' * 40,
        timestamp=f'2026-01-{ord(sha) - ord("a") + 1:02d}T00:00:00+00:00',
        metadata={'mode': 'research', 'node_id': 'N001', 'contract_hash': 'fixed-contract',
                  'scores': {'quality_gate': {'value': quality}}, 'archived': archived, **metadata})


def test_select_quality_passing_best_score_and_stable_ties():
    from autoresearch.research import select_research_attempt
    a, b, c = candidate('d', .8), candidate('e', .9), candidate('f', .99, 0)
    assert select_research_attempt([a, b, c]).commit_hash == b.commit_hash
    assert select_research_attempt([candidate('f', .9), b]).commit_hash == b.commit_hash
    assert select_research_attempt([candidate('d', .7, 0), candidate('e', None, None,
        status='crashed', failure_class='infra')]).commit_hash == 'e' * 40


def test_select_excludes_archived_nonfinite_pending_and_incomparable():
    from autoresearch.research import select_research_attempt
    best = candidate('d', .8)
    assert select_research_attempt([best, candidate('e', .99, archived=True),
        candidate('f', .99, comparability='incomparable'), candidate('g', float('nan')),
        candidate('h', 2, status='pending')]).commit_hash == best.commit_hash


def test_one_terminal_candidate_keeps_research_active(root_run):
    store, _ = _active_research(root_run)
    write_attempt(root_run, candidate())
    store.advance()
    node = _state(root_run)['nodes']['N001']
    assert node['research']['status'] == 'active'
    assert node['analysis'] is None
    assert node['engineering_status'] == 'blocked'


def test_research_finish_freezes_best_not_last_and_rejects_new_work(root_run):
    from autoresearch.state import authorize_attempt
    store, assignment = _active_research(root_run)
    for attempt in [candidate('d', .9), candidate('e', .8)]:
        write_attempt(root_run, attempt)
    store.request_finish(assignment.agent_id, 'research')
    store.advance()
    node = _state(root_run)['nodes']['N001']
    assert node['research']['selected_sha'] == 'd' * 40
    assert node['research']['status'] == 'completed'
    assert node['analysis']['research_sha'] == 'd' * 40
    with pytest.raises(ValueError):
        authorize_attempt({'node_id': 'N001', 'mode': 'research'},
            {'coral_dir': str(root_run), 'agent_id': assignment.agent_id})


@pytest.mark.parametrize('successes,failures,allowed', [(2, 0, True), (3, 0, False),
    (0, 9, True), (0, 10, False)])
def test_research_authorizer_enforces_both_run_caps(root_run, successes, failures, allowed):
    from autoresearch.state import authorize_attempt
    store, assignment = _active_research(root_run)
    for n in range(successes + failures):
        a = candidate(chr(ord('d') + n), .5 if n < successes else None,
                      status='improved' if n < successes else 'crashed')
        if n >= successes:
            a.metadata['failure_class'] = 'candidate'
        write_attempt(root_run, a)
    context = {'coral_dir': str(root_run), 'agent_id': assignment.agent_id,
               'commit_hash': '0' * 40, 'budget_class': 'real'}
    if allowed:
        authorize_attempt({'node_id': 'N001', 'mode': 'research'}, context)
    else:
        with pytest.raises(ValueError, match='Research run limit'):
            authorize_attempt({'node_id': 'N001', 'mode': 'research'}, context)


def test_third_success_auto_freezes_and_pending_blocks_finish(root_run):
    store, assignment = _active_research(root_run)
    for a in [candidate('d', .7), candidate('e', .9), candidate('f', .8, status='pending')]:
        write_attempt(root_run, a)
    with pytest.raises(ValueError, match='terminal'):
        store.request_finish(assignment.agent_id, 'research')
    store.advance()
    assert _state(root_run)['nodes']['N001']['research']['status'] == 'active'
    write_attempt(root_run, candidate('f', .8))
    store.advance()
    assert _state(root_run)['nodes']['N001']['research']['selected_sha'] == 'e' * 40


def test_archived_retries_count_and_budget_exhaustion_allows_finish(root_run):
    from autoresearch.research import run_budget
    a = candidate('d', None, None, status='crashed', failure_class='infra',
                  infra_retry_count=9, archived=True)
    assert run_budget([a]) == (0, 10, 0)


@pytest.mark.parametrize('finish', ['direct', 'request'])
def test_explicit_finish_prunes_loser_and_preserves_evidence(root_run, finish):
    from test_checkpoint_retention import checkpoint
    store, assignment = _active_research(root_run)
    best, worse = candidate('d', .9), candidate('e', .8)
    for a in [best, worse]:
        write_attempt(root_run, a)
    kept, pruned = checkpoint(root_run, best), checkpoint(root_run, worse)
    original = (pruned / 'manifest.json').read_bytes()
    if finish == 'direct':
        store.complete_assignment(assignment.agent_id, 'research')
    else:
        store.request_finish(assignment.agent_id, 'research')
    store.advance()
    assert (kept / 'model.pt').exists()
    assert not (pruned / 'model.pt').exists()
    assert (pruned / 'manifest.json').read_bytes() == original
    assert _state(root_run)['nodes']['N001']['research']['selected_sha'] == best.commit_hash


def test_scored_quality_failure_counts_as_successful_run():
    from autoresearch.research import run_budget
    attempts = [candidate(sha, 0.0, 0, status='regressed', failure_class='candidate')
                for sha in ['d', 'e', 'f']]
    assert run_budget(attempts) == (3, 3, 0)


def test_explicit_finish_can_abandon_retryable_infra_failure(root_run):
    store, assignment = _active_research(root_run)
    for a in [candidate('d', None, None, status='crashed', failure_class='infra',
                        infra_retry_count=0, max_infra_retries=2), candidate('e', .9)]:
        write_attempt(root_run, a)
    store.request_finish(assignment.agent_id, 'research')
    store.advance()
    assert _state(root_run)['nodes']['N001']['research']['selected_sha'] == 'e' * 40


def test_trusted_submission_accepts_three_candidates_then_freezes_best(root_run):
    import os
    from types import SimpleNamespace

    from autoresearch.state import authorize_attempt
    from test_submission import _git, _repo

    from coral.grader.trusted_writes import AgentBinding, TrustedWriteService, process_start_time

    store, assignment = _active_research(root_run)
    repo, base = _repo(root_run.parent)
    binding = AgentBinding(assignment.agent_id, os.getpid(), process_start_time(os.getpid()), str(repo), None)
    service = TrustedWriteService(root_run, SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda *_: binding, attempt_authorizer=authorize_attempt,
        submission_validator=lambda *_: {'mode': 'research', 'node_id': 'N001', 'node_type': 'research',
            'base_commit': base, 'quality_anchor': None, 'contract_hash': 'fixed-contract',
            'max_infra_retries': 2, 'submission_hash': 'canonical'},
        checkpoint_fn=lambda *_args, **_kwargs: 'shared-state')
    shas = []
    for i, score in enumerate([.9, .8, .7, .6]):
        parent = _git(repo, 'rev-parse', 'HEAD')
        (repo / 'model.py').write_text(f'refresh_period = {i + 1}\n')
        _git(repo, 'add', '.')
        _git(repo, 'commit', '-m', f'refinement {i}')
        sha = _git(repo, 'rev-parse', 'HEAD')
        payload = {'commit_hash': sha, 'agent_id': assignment.agent_id, 'title': 'refinement',
                   'parent_hash': parent, 'budget_class': 'real', 'island_id': None}
        if i == 3:
            with pytest.raises(ValueError, match='Research run limit'):
                service.handle('submit_attempt', payload, os.getpid())
            break
        result = service.handle('submit_attempt', payload, os.getpid())
        a = Attempt.from_dict(result)
        a.status, a.score = 'regressed', score
        a.metadata['scores'] = {'quality_gate': {'value': 1}}
        write_attempt(root_run, a)
        shas.append(sha)
        if i < 2:
            store.advance()
            assert _state(root_run)['nodes']['N001']['analysis'] is None
    store.advance()
    node = _state(root_run)['nodes']['N001']
    assert node['research']['selected_sha'] == shas[0]
    assert node['analysis']['research_sha'] == shas[0]
    assert node['engineering_status'] == 'pending'


def test_cross_topology_candidate_can_be_selected():
    from autoresearch.research import quality_eligible, select_research_attempt

    cross = candidate('e', .9, comparability='cross_topology')
    assert quality_eligible(cross)
    assert select_research_attempt([candidate('d', .8), cross]) == cross
