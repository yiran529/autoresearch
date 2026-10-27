"""Host-owned path views shared by sandbox mounts, prompts and responses.

This module deliberately uses only the standard library: the isolated Analyzer
client need not import the host controller or its private environment.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

WORKSPACE_VIEW = Path('/workspace')
_RESULT_PATHS = ('record_path', 'stdout_path', 'stderr_path', 'results_path')


def _absolute(value) -> Path:
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or path == Path('/'):
        raise ValueError('artifact views require dedicated absolute paths without traversal')
    return path


class PathView:
    """An explicit mapping; it grants no access until the host mounts/copies it."""

    def __init__(self, mounts):
        self.mounts = tuple((_absolute(source), _absolute(target)) for source, target in mounts)
        if len({source for source, _ in self.mounts}) != len(self.mounts):
            raise ValueError('duplicate artifact source')
        if len({target for _, target in self.mounts}) != len(self.mounts):
            raise ValueError('duplicate artifact destination')
        self._specific = sorted(self.mounts, key=lambda item: len(item[0].parts), reverse=True)

    def path(self, value) -> Path:
        path = _absolute(value)
        for source, target in self._specific:
            if path.is_relative_to(source):
                return target / path.relative_to(source)
        raise ValueError(f'path outside selected artifact view: {path}')

    def rewrite_text(self, text: str) -> str:
        if not self.mounts:
            return text
        # One pass prevents a destination being interpreted as another source.
        replacements = {str(source): str(target) for source, target in self._specific}
        alternatives = '|'.join(re.escape(source) for source in replacements)
        pattern = r'(?<![\w./-])(?:' + alternatives + r')(?=/|$|[^\w./-])'
        return re.sub(pattern, lambda match: replacements[match[0]], text)

    def project_result(self, document: dict, *, published=False) -> dict:
        result = dict(document)
        for field in _RESULT_PATHS:
            if result.get(field):
                target = self.path(result[field])
                if published and not target.exists():
                    result.pop(field)
                    unavailable = dict(result.get('unavailable_artifacts', {}))
                    unavailable[field] = (
                        'Not preserved in the public archive; see artifact_inventory.json.')
                    result['unavailable_artifacts'] = unavailable
                else:
                    result[field] = str(target)
        return result


def diagnostic_view(workspace, *, checkpoint_paths=()) -> PathView:
    workspace = _absolute(workspace)
    mounts = [(workspace, WORKSPACE_VIEW)]
    records = workspace.parent / 'records'
    if records.is_dir():
        mounts.append((records, WORKSPACE_VIEW / 'records'))
    for path in sorted({_absolute(value) for value in checkpoint_paths}):
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        mounts.append((path, WORKSPACE_VIEW / 'checkpoints' / identity))
    return PathView(mounts)
