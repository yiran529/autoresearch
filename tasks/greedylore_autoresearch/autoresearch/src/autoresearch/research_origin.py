"""One authored research origin; compatibility intent is derived by the host."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .research_record import validate_design

_ORIGIN_FIELDS = {
    'parent_node', 'question', 'hypothesis', 'selection_rationale', 'delta', 'risk',
    'falsification_criteria', 'problem_ref', 'parent_response', 'engineering_review',
}
_REGISTRATION_FIELDS = {
    'parent_node', 'hypothesis', 'selection_rationale', 'delta', 'risk', 'falsification_criteria',
}


def registration_from_origin(node_id, data):
    """Validate before publishing Spec or releasing the planning lease."""
    if (not isinstance(data, dict) or not {'origin', 'design'} <= data.keys()
            or data.keys() - {'origin', 'design', 'planning_token', 'overlap_review'}):
        raise ValueError('register requires origin and design; only planning controls are optional')
    origin = data['origin']
    if not isinstance(origin, dict) or set(origin) != _ORIGIN_FIELDS:
        raise ValueError('origin has invalid fields')
    for key in _ORIGIN_FIELDS - {'problem_ref'}:
        if not isinstance(origin[key], str) or not origin[key].strip() or len(origin[key]) > 4000:
            raise ValueError(f'origin {key} must be nonempty text of at most 4000 characters')
    origin = {key: value.strip() if isinstance(value, str) else value for key, value in origin.items()}
    if not re.fullmatch(r'N[0-9]{3}', origin['parent_node']):
        raise ValueError('origin parent_node is invalid')
    if origin['problem_ref'] is not None and not isinstance(origin['problem_ref'], str):
        raise ValueError('origin problem_ref must be a parent question reference or null')
    design = validate_design(data['design'])
    # Point to the immutable initial design rather than asking the agent to
    # duplicate its predictions/tests in another interface or truncating them.
    digest = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    source = f'notes/nodes/{node_id}/research.json: iterations[0].design (sha256 {digest})'
    intent = {
        'problem_ref': origin['problem_ref'], 'question': origin['question'],
        'mechanism_test': f'Judge the mechanism predictions and matched tests in {source} '
                          '(predictions and tests fields). Keep mechanism and practical verdicts separate.',
        'practical_test': f'Judge the quality and complete-cycle cost predictions/tests in {source} '
                          'against the authoritative evaluation contract; interim diagnostics do not replace its gates.',
        'handover_response': origin['parent_response'],
        'engineering_review': origin['engineering_review'],
    }
    registration = {key: origin[key] for key in _REGISTRATION_FIELDS}
    registration['research_intent'] = intent
    registration.update({key: data[key] for key in ('planning_token', 'overlap_review') if key in data})
    return registration, design, copy.deepcopy(origin)
