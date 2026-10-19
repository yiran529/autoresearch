"""Render historical evaluation inputs already captured in fixed evidence."""

import json
from pathlib import Path

from .validation import ArtifactValidationError


def render_context(evidence, snapshot: Path) -> str:
    parts = ['\n\n## Evaluation context (recorded for this Attempt)']
    paths = sorted(snapshot.rglob('evaluation_context.json'))
    if not paths:
        parts.append('Historical evaluation context: unavailable. Do not substitute current '
                     'task configuration or infer missing dense values or gates.')
    for path in paths:
        try:
            value = json.loads(path.read_text())
            valid = (isinstance(value, dict) and value.get('version') == 1
                     and value.get('contract_hash') == evidence.contract_hash)
        except (ValueError, OSError) as exc:
            raise ArtifactValidationError('invalid recorded evaluation context') from exc
        if not valid:
            raise ArtifactValidationError('evaluation context conflicts with Attempt contract')
        parts.extend([f'Source: {path}', json.dumps(value, indent=2, sort_keys=True)])
    return '\n'.join(parts)
