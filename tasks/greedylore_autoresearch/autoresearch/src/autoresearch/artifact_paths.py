"""Host-owned path views shared by sandbox mounts, prompts and responses.

This module deliberately uses only the standard library: the isolated Analyzer
client need not import the host controller or its private environment.
"""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path

WORKSPACE_VIEW = Path('/workspace')
_RESULT_PATHS = ('record_path', 'stdout_path', 'stderr_path', 'results_path')


def research_diagnostics_root(coral_dir: Path) -> Path:
    """Keep legacy file mounts intact; use an already protected directory."""
    root = Path(coral_dir) / 'public/research_diagnostics'
    fallback = Path(coral_dir) / 'public/diagnostics/research_probes'
    # bwrap removes its empty file placeholder when the last sandbox exits.
    # Keep this run on its original fallback even after that cleanup.
    if fallback.exists() or fallback.is_symlink():
        return fallback
    if not root.is_symlink():
        try:
            info = root.stat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(info.st_mode) and info.st_size == 0:
                # Persist the choice before bwrap can remove the placeholder.
                if any(path.is_symlink() for path in (fallback, *fallback.parents)):
                    raise ValueError('diagnostic publication path contains a symlink')
                fallback.mkdir(parents=True, exist_ok=True)
                return fallback
    return root


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


def checkpoint_view(path) -> Path:
    """Address only: access still requires host selection and an actual mount."""
    identity = hashlib.sha256(str(_absolute(path)).encode()).hexdigest()[:24]
    return WORKSPACE_VIEW / 'checkpoints' / identity


def diagnostic_view(workspace, *, checkpoint_paths=()) -> PathView:
    workspace = _absolute(workspace)
    mounts = [(workspace, WORKSPACE_VIEW)]
    records = workspace.parent / 'records'
    if records.is_dir():
        mounts.append((records, WORKSPACE_VIEW / 'records'))
    for path in sorted({_absolute(value) for value in checkpoint_paths}):
        mounts.append((path, checkpoint_view(path)))
    return PathView(mounts)
