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


def checkpoint_observation(r, *, result_status='completed'):
    """The original manifest is already archived before a host observation."""
    manifest = {'version': 1, 'status': 'complete', 'world_size': 4,
                'update_step': 4000, 'contract_hash': 'contract-v1',
                'model_layout': {'weights': ['weight'] * 1000},
                'config': {'optimizer': 'muon', 'layers': list(range(100))},
                'files': {'model.pt': {'sha256': 'a' * 64, 'size_bytes': 123}},
                'limitations': ['No independent replication.'], 'errors': []}
    relative = 'eval_logs/' + 'b' * 40 + '/run1/checkpoints/model/manifest.json'
    path = r.store.coral_dir / 'public' / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    observation = {'status': result_status, 'quality_pass': True,
                   'final_eval_loss': 3.4, 'evidence_path': 'eval_logs/' + 'b' * 40 + '/run1',
                   'diagnostic_checkpoint': manifest}
    return observation, path, relative


def test_observation_compacts_only_exact_archived_checkpoint_and_replay_is_stable(tmp_path):
    import copy
    import hashlib

    r = record(tmp_path)
    initialize(r)
    r.begin(1, 'b' * 40, 'run1')
    observation, path, relative = checkpoint_observation(r)
    original = copy.deepcopy(observation)
    raw = path.read_bytes()
    result = r.observe(1, observation, 'observe1')
    saved = result['iterations'][0]['observations'][0]
    compact = saved['diagnostic_checkpoint']
    assert 'model_layout' not in compact and 'config' not in compact
    assert compact['manifest_ref'] == relative
    assert compact['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
    assert {k: v for k, v in compact.items() if not k.startswith('manifest_')} == {
        k: v for k, v in original['diagnostic_checkpoint'].items() if k not in {'config', 'model_layout'}}
    assert {k: v for k, v in saved.items() if k != 'diagnostic_checkpoint'} == {
        k: v for k, v in original.items() if k != 'diagnostic_checkpoint'}
    assert observation == original and path.read_bytes() == raw
    assert json.loads(r.public.read_text()) == r.read()
    # Idempotent replay must not depend on the manifest still being present.
    path.unlink()
    assert r.observe(1, original, 'observe1') == result
    with pytest.raises(ValueError, match='request ID conflicts'):
        r.observe(1, {**original, 'final_eval_loss': 3.5}, 'observe1')


@pytest.mark.parametrize('case', ['missing', 'different', 'json_type', 'symlink', 'escape'])
def test_checkpoint_details_retained_without_exact_safe_manifest(tmp_path, case):
    r = record(tmp_path)
    initialize(r)
    r.begin(1, 'b' * 40, 'run1')
    observation, path, _ = checkpoint_observation(r)
    if case == 'missing':
        path.unlink()
    elif case == 'different':
        path.write_text(json.dumps({'config': 'different'}))
    elif case == 'json_type':
        manifest = json.loads(path.read_text())
        manifest['config']['enabled'] = True
        path.write_text(json.dumps(manifest))
        observation['diagnostic_checkpoint']['config']['enabled'] = 1
    elif case == 'symlink':
        outside = tmp_path / 'outside.json'
        path.rename(outside)
        path.symlink_to(outside)
    else:
        observation['evidence_path'] = '../private'
    value = r.observe(1, observation, 'observe1')
    assert value['iterations'][0]['observations'][0] == observation


def test_final_observation_compacts_checkpoint_without_changing_frozen_selection(tmp_path):
    r = record(tmp_path)
    initialize(r)
    r.begin(1, 'b' * 40, 'run1')
    observation, path, relative = checkpoint_observation(r)
    r.observe(1, observation, 'observe1')
    # Existing v2 fixture uses the legacy nomination predicate; final publication
    # requires only a frozen selected candidate with the same SHA.
    private = json.loads(r.private.read_text())
    private['document'].update(frozen=True, selected_iteration=1, final_evaluation={})
    r.private.write_text(json.dumps(private))
    entry = {'request_id': 'final1', 'sha': 'b' * 40, 'result': observation}
    result = r.final_observe('vision', entry, 'final1.record')
    saved = result['final_evaluation']['vision'][0]
    assert saved['result']['diagnostic_checkpoint']['manifest_ref'] == relative
    assert 'model_layout' not in saved['result']['diagnostic_checkpoint']
    assert result['selected_iteration'] == 1 and result['frozen'] is True
    assert r.final_observe('vision', entry, 'final1.record') == result
    assert 'model_layout' in json.loads(path.read_text())
