"""A single origin and first design drive host-derived compatibility records."""
# ruff: noqa: S101
from __future__ import annotations

import copy
import json

import pytest
import yaml
from autoresearch.staged_research import handle_action
from test_problem_handover import body
from test_research_record import design
from test_workflow_state import _store
from test_workflow_state import root_run as root_run  # noqa: F401


def origin():
    return {
        'parent_node': 'N000', 'problem_ref': 'N000:Q1',
        'question': 'Can useful history improve training at a fixed budget?',
        'hypothesis': 'Useful history reduces the harmful update error.',
        'selection_rationale': 'Root Analysis identifies a quality discrepancy.',
        'delta': 'Retain useful history rather than dropping it.',
        'risk': 'State may become stale.',
        'falsification_criteria': 'No mechanism or quality benefit under matched tests.',
        'parent_response': 'Investigate the parent quality question.',
        'engineering_review': 'Review parent implementation evidence; no optimizations reused.',
    }


@pytest.fixture
def staged(root_run):
    (root_run / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'p_explore': 0,
    }}}))
    report = root_run / 'public/notes/nodes/N000/research_analysis_v1.md'
    document = json.loads(report.read_text())
    document['handover'] = body()
    report.write_text(json.dumps(document))
    store = _store(root_run)
    store.reconcile(1, 0)
    return store


def register(store, data=None, request_id='register1'):
    return handle_action(store, 'research-N001', {
        'node_id': 'N001', 'action': 'register',
        'data': data if data is not None else {'origin': origin(), 'design': design()},
        'request_id': request_id,
    }, {})


def test_origin_is_the_only_authored_research_start_and_generates_compatibility(staged):
    result = register(staged)
    assert result['version'] == 3
    assert result['question'] == origin()['question']
    assert {k: result['origin'][k] for k in origin()} == origin()
    assert result['origin']['node_id'] == 'N001'
    assert result['origin']['base_commit'] == 'a' * 40
    assert 'research_intent' not in result['origin']
    assert result['iterations'][0]['design'] == design()
    spec = yaml.safe_load((staged.notes_root / 'nodes/N001/spec.md').read_text())
    intent = spec['research_intent']
    assert intent['problem_ref'] == origin()['problem_ref']
    assert intent['question'] == origin()['question']
    assert intent['handover_response'] == origin()['parent_response']
    assert intent['engineering_review'] == origin()['engineering_review']
    for field in ('mechanism_test', 'practical_test'):
        assert 'notes/nodes/N001/research.json: iterations[0].design' in intent[field]
        assert 'predictions' in intent[field] and 'tests' in intent[field]
    assert spec['parent_handover']['snapshot']['node_id'] == 'N000'


def test_register_replay_and_reopen_do_not_duplicate_or_rewrite_history(staged):
    first = register(staged)
    assert register(_store(staged.coral_dir)) == first
    from autoresearch.research_record import ResearchRecord
    record = ResearchRecord(staged, 'research-N001', 'N001')
    record.decide(1, 'stop', 'Revise within the same question', 'stop1')
    record.append_design(design('second'), 'design2')
    replay = register(_store(staged.coral_dir))
    assert len(replay['iterations']) == 2
    assert replay['origin'] == first['origin']
    assert replay['iterations'][0]['design'] == first['iterations'][0]['design']


@pytest.mark.parametrize('change', ['empty_design', 'forged_base', 'extra_intent', 'bad_problem', 'bad_id'])
def test_invalid_registration_has_no_partial_spec_or_record(staged, change):
    data = {'origin': origin(), 'design': design()}
    request_id = 'register1'
    if change == 'empty_design':
        data['design']['tests'] = ''
    elif change == 'forged_base':
        data['origin']['base_commit'] = 'f' * 40
    elif change == 'extra_intent':
        data['origin']['research_intent'] = {'question': 'duplicate'}
    elif change == 'bad_problem':
        data['origin']['problem_ref'] = 'N000:Q9'
    else:
        request_id = '../invalid'
    with pytest.raises(ValueError):
        register(staged, data, request_id)
    assert not (staged.notes_root / 'nodes/N001/spec.md').exists()
    assert not (staged.notes_root / 'nodes/N001/research.json').exists()
    state = json.loads(staged.state_path.read_text())
    assert state['nodes']['N001']['research']['status'] == 'reserved'


def test_request_controls_do_not_become_part_of_origin(staged):
    data = {'origin': origin(), 'design': design(),
            'planning_token': None, 'overlap_review': 'History differs from Root.'}
    result = register(staged, data)
    assert 'planning_token' not in result['origin']
    assert 'overlap_review' not in result['origin']
    changed = copy.deepcopy(data)
    changed['origin']['question'] = 'A different question'
    with pytest.raises(ValueError, match='conflict|immutable'):
        register(staged, changed)


def test_interrupted_registration_freezes_first_design_before_record_publication(staged, monkeypatch):
    from autoresearch.research_record import ResearchRecord
    original = ResearchRecord.initialize
    def fail_publication(*args, **kwargs):
        raise OSError('record publication interrupted')
    monkeypatch.setattr(ResearchRecord, 'initialize', fail_publication)
    with pytest.raises(OSError, match='interrupted'):
        register(staged)
    monkeypatch.setattr(ResearchRecord, 'initialize', original)
    compat_spec = yaml.safe_load((staged.notes_root / 'nodes/N001/spec.md').read_text())
    old_registration = {key: compat_spec[key] for key in (
        'parent_node', 'hypothesis', 'selection_rationale', 'delta', 'risk', 'falsification_criteria', 'research_intent')}
    with pytest.raises(ValueError, match='origin'):
        register(staged, {'registration': old_registration, 'design': design()})
    changed = {'origin': origin(), 'design': design('different first method')}
    with pytest.raises(ValueError, match='conflict|immutable'):
        register(staged, changed)
    result = register(staged)
    assert result['iterations'][0]['design'] == design()


def test_long_initial_predictions_are_preserved_without_compatibility_truncation(staged):
    initial = design()
    initial['predictions'] = 'P' * 8000
    initial['tests'] = 'T' * 8000
    result = register(staged, {'origin': origin(), 'design': initial})
    assert result['iterations'][0]['design'] == initial


def test_old_registration_shape_cannot_bypass_origin_for_a_new_node(staged):
    from test_problem_handover import intent
    registration = {key: origin()[key] for key in (
        'parent_node', 'hypothesis', 'selection_rationale', 'delta', 'risk', 'falsification_criteria')}
    registration['research_intent'] = intent('N000:Q1')
    with pytest.raises(ValueError, match='origin'):
        register(staged, {'registration': registration, 'design': design()})
    assert not (staged.notes_root / 'nodes/N001/spec.md').exists()
    # Pre-upgrade publication recovery remains possible, with its existing IDs.
    staged.initialize_research('research-N001', 'N001', registration)
    result = register(staged, {'registration': registration, 'design': design()}, 'register.v1')
    assert result['version'] == 2
    assert register(staged, {'registration': registration, 'design': design()}, 'register.v1') == result


def test_register_accepts_existing_record_id_contract(staged):
    result = register(staged, request_id='r.' + 'x' * 94)
    assert result['version'] == 3


def test_origin_whitespace_normalization_is_replay_safe(staged):
    value = origin()
    value['hypothesis'] = '  ' + value['hypothesis'] + '  '
    payload = {'origin': value, 'design': design()}
    first = register(staged, payload)
    assert register(staged, payload) == first
    assert first['origin']['hypothesis'] == origin()['hypothesis']


def test_staged_toolkit_delivers_origin_template_and_command_guide(staged):
    toolkit = staged.notes_root / 'toolkit'
    request = json.loads((toolkit / 'templates/research_origin.json').read_text())
    from autoresearch.research_origin import registration_from_origin
    registration, initial_design, authored_origin = registration_from_origin('N001', request)
    assert authored_origin['question'] and initial_design['tests']
    assert registration['research_intent']['question'] == authored_origin['question']
    assert not (toolkit / 'templates/research_intent.json').exists()
    guide = (toolkit / 'references/research_commands.md').read_text()
    for command in ('research register', 'research diagnostic', 'research design', 'research train'):
        # Registration's generic command appears in the role; guide explicitly
        # exercises diagnostic/revision/continuation actions.
        if command != 'research register':
            assert command in guide
    rules = (toolkit / 'workflow_rules.md').read_text()
    assert 'normally use two, at most three candidate trajectories' in rules
    assert (toolkit / 'references/evaluation_tasks.md').is_file()
    assert 'node initialize --intent-file' not in rules


def test_staged_old_initialization_endpoint_cannot_bypass_origin(staged):
    from autoresearch.state import handle_workflow
    with pytest.raises(ValueError, match='origin and design'):
        handle_workflow('initialize_research', {'node_id': 'N001'}, {
            'coral_dir': str(staged.coral_dir), 'agent_id': 'research-N001',
        })
    assert not (staged.notes_root / 'nodes/N001/spec.md').exists()
