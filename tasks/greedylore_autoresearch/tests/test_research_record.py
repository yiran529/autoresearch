"""Frozen designs and persistent trajectory accounting are host-owned."""

import json
from types import SimpleNamespace

import pytest
import yaml


def record(tmp_path):
    from autoresearch.research_record import ResearchRecord

    coral = tmp_path / '.coral'
    private = coral / 'private'
    private.mkdir(parents=True)
    (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {'staged_research': True}}}))
    (private / 'autoresearch_state.json').write_text(json.dumps({'nodes': {'N001': {
        'research': {'agent_id': 'research-N001', 'status': 'active'},
    }}}))
    store = SimpleNamespace(coral_dir=coral, notes_root=coral / 'public/notes')
    return ResearchRecord(store, 'research-N001', 'N001')


def design(name='first'):
    return {'method': name, 'mechanism': 'retain useful history',
            'predictions': 'less update distortion and lower loss',
            'tests': 'paired same-state comparison; fail if no loss effect'}


def initialize(r):
    return r.initialize({'parent_node': 'N000', 'base_commit': 'a' * 40,
                         'hypothesis': 'history helps', 'research_mode': 'REFINE'}, design(), 'init')


def test_immutable_designs_append_only_history_and_idempotency(tmp_path):
    r = record(tmp_path)
    initialize(r)
    row = r.begin(1, 'b' * 40, 'run-1')
    assert row['iterations'][0]['status'] == 'running'
    r.observe(1, {'status': 'paused', 'update_step': 1000,
                  'checkpoint_path': '/managed/step-1000'}, 'observe-1')
    r.decide(1, 'continue', 'need later-stage evidence', 'decision-1')
    r.begin(1, 'b' * 40, 'run-2')
    r.observe(1, {'status': 'completed', 'update_step': 2500, 'eligible': True}, 'observe-2')
    value = r.decide(1, 'revise', 'reduce the observed side effect', 'decision-2')
    assert len(value['iterations']) == 1
    second = r.append_design(design('second'), 'append-2')
    assert second['iterations'][0]['design'] == design()
    assert len(second['iterations'][0]['observations']) == 2
    assert r.append_design(design('second'), 'append-2') == second
    with pytest.raises(ValueError, match='request'):
        r.append_design(design('different'), 'append-2')


def test_three_trajectories_include_algorithm_failures_and_survive_reopen(tmp_path):
    r = record(tmp_path)
    initialize(r)
    for i in range(1, 4):
        r.begin(i, str(i) * 40, f'run-{i}')
        r.observe(i, {'status': 'failed', 'reason': 'algorithm failed'}, f'failure-{i}')
        if i < 3:
            r.append_design(design(f'candidate-{i+1}'), f'append-{i+1}')
    from autoresearch.research_record import ResearchRecord

    resumed = ResearchRecord(r.store, r.agent_id, r.node_id)
    with pytest.raises(ValueError, match='3|three|budget'):
        resumed.append_design(design('fourth'), 'append-4')
    assert len(resumed.read()['iterations']) == 3


def test_resume_rejects_changed_sha_and_freeze_selects_earlier_completed_candidate(tmp_path):
    r = record(tmp_path)
    initialize(r)
    r.begin(1, 'b' * 40, 'run-1')
    r.observe(1, {'status': 'paused', 'update_step': 1000}, 'pause')
    with pytest.raises(ValueError, match='SHA|sha|version'):
        r.begin(1, 'c' * 40, 'bad-resume')
    r.begin(1, 'b' * 40, 'resume')
    r.observe(1, {'status': 'completed', 'eligible': True}, 'complete')
    r.append_design(design('second'), 'append')
    r.begin(2, 'c' * 40, 'run-2')
    r.observe(2, {'status': 'failed'}, 'failed')
    frozen = r.freeze(1, 'first completed candidate is better', 'freeze')
    assert frozen['selected_iteration'] == 1 and frozen['frozen'] is True
    with pytest.raises(ValueError, match='frozen'):
        r.decide(1, 'revise', 'posthoc', 'after-freeze')
    assert r.freeze(1, 'first completed candidate is better', 'freeze') == frozen


def test_other_agent_cannot_modify_record(tmp_path):
    r = record(tmp_path)
    initialize(r)
    from autoresearch.research_record import ResearchRecord

    with pytest.raises(PermissionError):
        ResearchRecord(r.store, 'research-N002', 'N001').append_design(design(), 'other')
