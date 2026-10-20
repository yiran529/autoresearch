"""Immutable Analysis records and deterministic Markdown reading views.

Writers must hold the run's trusted write lock. JSON is written before the view,
so an interrupted publication can complete without another model invocation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

_NAME = re.compile(r'(research|engineering)_analysis_v([1-9][0-9]*)\.md')


def render_analysis(document: dict, filename: str) -> str:
    match = _NAME.fullmatch(filename)
    if not isinstance(document, dict) or match is None or document.get('mode') != match[1]:
        raise ValueError('invalid Analysis identity')
    node = document.get('node_id')
    if not isinstance(node, str) or not re.fullmatch(r'N[0-9]{3}', node):
        raise ValueError('invalid Analysis identity')
    summary = document.get('summary')
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError('Analysis summary must be nonempty')
    record = Path(filename).with_suffix('.json').name
    lines = [f'# {node} {match[1].title()} Analysis · v{match[2]}', '',
             f'Research SHA: `{document.get("research_sha", "")}`', '',
             f'Full record: [{record}]({record})', '', '## Summary', '', summary]
    for key in ('findings', 'limitations'):
        items = document.get(key)
        if (not isinstance(items, list) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)):
            raise ValueError(f'Analysis {key} must be nonempty text items')
        lines.extend(['', f'## {key.title()}', ''])
        for item in items:
            lines.append('- ' + item.replace('\n', '\n  '))
    if match[1] == 'research':
        root = f'../../../diagnostics/{node}-research-v{match[2]}'
        lines.extend(['', '## Evidence navigation', '',
                      f'- [Diagnostic records (when enabled)]({root}/records/)',
                      '- The full JSON record preserves the frozen evaluation and evidence manifest.',
                      '- Paths beginning `code/`, `evidence/` or `results/` in the text refer to '
                      f'the [Analyzer workspace]({root}/workspace/). For diagnostic results, '
                      'use the cited request ID under records/ for the immutable copy.'])
    return '\n'.join(lines) + '\n'


def _safe(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Analysis path traverses a symlink')


def load_analysis(path: Path) -> dict:
    """Read legacy JSON-in-md or validate a new view against its fixed sibling."""
    path = Path(path)
    _safe(path)
    text = path.read_text(encoding='utf-8')
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        if not _NAME.fullmatch(path.name):
            raise ValueError('invalid Analysis document') from None
        record = path.with_suffix('.json')
        _safe(record)
        document = json.loads(record.read_text(encoding='utf-8'))
        if text != render_analysis(document, path.name):
            raise ValueError('Analysis view does not match its JSON record')
    if not isinstance(document, dict):
        raise ValueError('Artifact must be a JSON object')
    return document


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_analysis(path: Path, document: dict) -> Path:
    """Publish only the fixed JSON sibling and view; never rewrite an old version."""
    path = Path(path)
    text = render_analysis(document, path.name)
    if path.parent.name != document['node_id']:
        raise ValueError('Analysis target identity does not match Node')
    record = path.with_suffix('.json')
    for target in (path, record):
        _safe(target)
        if target.exists() and load_analysis(target) != document:
            raise ValueError('immutable Analysis already exists')
    if not record.exists():
        _atomic_text(record, json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
    if not path.exists():
        _atomic_text(path, text)
    return path


def convert_legacy_analysis(path: Path, backup: Path) -> Path:
    """Explicit host migration only: preserve the exact original before replacing it."""
    path, backup = Path(path), Path(backup)
    _safe(path)
    _safe(backup)
    original = path.read_text(encoding='utf-8')
    try:
        document = json.loads(original)
    except json.JSONDecodeError:
        return publish_analysis(path, load_analysis(path))
    rendered = render_analysis(document, path.name)
    if backup.exists():
        if backup.read_text(encoding='utf-8') != original:
            raise ValueError('migration backup already contains different bytes')
    else:
        _atomic_text(backup, original)
    publish_analysis(path, document)
    _atomic_text(path, rendered)
    return path
