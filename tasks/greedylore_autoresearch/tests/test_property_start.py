"""A motivated desired property need not already be a parent handover question."""
# ruff: noqa: S101
import json

import pytest
import yaml
from test_origin_registration import origin, register
from test_origin_registration import staged as staged  # noqa: F401
from test_research_record import design
from test_workflow_state import root_run as root_run  # noqa: F401


@pytest.mark.parametrize('mode', ['REFINE', 'EXPLORE'])
def test_unified_origin_accepts_property_start_without_inventing_parent_question(staged, mode):
    state = json.loads(staged.state_path.read_text())
    state['nodes']['N001']['research']['research_mode'] = mode
    staged.state_path.write_text(json.dumps(state))
    start = origin()
    start.update(problem_ref=None,
        question='Can the method require fewer sequential communication stages?',
        parent_response='Study a desired property of the Parent; no matching handover question exists.')
    result = register(staged, {'origin': start, 'design': design()})
    assert result['origin']['problem_ref'] is None
    assert result['origin']['question'] == start['question']
    spec = yaml.safe_load((staged.notes_root / 'nodes/N001/spec.md').read_text())
    assert spec['research_intent']['problem_ref'] is None
    assert spec['parent_handover']['snapshot']['node_id'] == 'N000'


def test_property_start_still_rejects_an_invented_parent_question(staged):
    start = origin()
    start['problem_ref'] = 'N000:Q9'
    with pytest.raises(ValueError, match='problem_ref'):
        register(staged, {'origin': start, 'design': design()})
    assert not (staged.notes_root / 'nodes/N001/spec.md').exists()
