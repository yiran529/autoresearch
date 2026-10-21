"""Derived reading indexes; authoritative records stay in their original locations."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import yaml

from coral.hub.attempts import read_attempts

from .atomic import atomic_write_text


def node_relationship(notes_root: Path, node_id: str, record: dict) -> tuple[str, str]:
    """Project fixed Spec lineage and the host-assigned mode without guessing."""
    if node_id == 'N000':
        return 'Root', '—'
    path = notes_root / 'nodes' / node_id / 'spec.md'
    spec = {}
    if path.is_file() and not path.is_symlink():
        try:
            value = yaml.safe_load(path.read_text())
            if isinstance(value, dict):
                spec = value
        except (OSError, yaml.YAMLError):
            pass  # A derived index must not invent lineage from an unreadable Spec.
    parent = spec.get('parent_node')
    if not isinstance(parent, str) or not re.fullmatch(r'N[0-9]{3}', parent):
        parent = ('Not selected' if not path.exists()
                  and record['research']['status'] in {'reserved', 'cancelled'} else 'Unavailable')
    mode = record['research'].get('research_mode', spec.get('research_mode'))
    return parent, mode if isinstance(mode, str) and mode in {'REFINE', 'EXPLORE'} else 'Unknown'


def rebuild_node_indexes(store, state: dict) -> None:
    attempts = read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True)
    public = store.island_root
    for node_id, record in sorted(state['nodes'].items()):
        node = store.notes_root / 'nodes' / node_id
        parent, research_mode = node_relationship(store.notes_root, node_id, record)
        if parent in state['nodes']:
            parent = f'[{parent}](../{parent}/README.md)'
        lines = [f'# {node_id} artifact index', '',
                 'Start here. Links point to persistent shared files; producer worktrees may have retired.', '',
                 f'Parent: {parent}', '', f'Research mode: {research_mode}', '',
                 '## Reports', '']
        selected_sha = record['research'].get('selected_sha')
        if selected_sha:
            lines.extend([f'Selected Research SHA: `{selected_sha}`', ''])
        for name in ('spec.md', 'evaluation.md', 'handover.json'):
            if (node / name).is_file():
                lines.append(f'- [{name}]({name})')
        for mode in ('research', 'engineering'):
            versions = sorted((p for p in node.glob(f'{mode}_analysis_v*.md')
                               if p.stem.rsplit('_v', 1)[-1].isdigit()),
                              key=lambda p: int(p.stem.rsplit('_v', 1)[-1]))
            if versions:
                latest = versions[-1]
                lines.append(f'- Latest {mode} Analysis: [{latest.name}]({latest.name})')
                if latest.with_suffix('.json').is_file():
                    name = latest.with_suffix('.json').name
                    lines.append(f'  - Full frozen record: [{name}]({name})')
            else:
                lines.append(f'- {mode.title()} Analysis: not published yet.')

        def link(path: Path, label: str | None = None) -> str | None:
            if (not path.exists() or path.is_symlink()
                    or not path.resolve().is_relative_to(public.resolve())):
                return None
            relative = path.relative_to(public).as_posix()
            return f'[{label or relative}](../../../{quote(relative, safe="/")})'

        lines.extend(['', '## Code and evaluation history', '',
                      'Code lives in the run/repo Git repository, not in a retired agent directory. '
                      'From any worktree of this run, use `git show <SHA>:<file>` or '
                      '`git diff <Research-SHA> <Engineering-SHA> -- <file>`. '
                      'The current checked-out HEAD may differ from the candidate below.', '',
                      'The selected Research SHA is the frozen Engineering anchor. All Research candidates, '
                      'including archived attempts and grader errors, remain evidence; tune rows '
                      'do not count toward formal Research selection.', '',
                      '| Mode | Candidate SHA | Status | Budget | Visibility | Attempt | Runs |',
                      '| --- | --- | --- | --- | --- | --- | --- |'])
        selected = sorted((a for a in attempts if a.metadata.get('node_id') == node_id),
                          key=lambda a: (a.timestamp, a.commit_hash))
        for attempt in selected:
            attempt_link = link(public / 'attempts' / f'{attempt.commit_hash}.json', 'record') or 'unavailable'
            runs = []
            for run in attempt.metadata.get('grader_runs', []):
                run_id = run.get('run_id')
                if isinstance(run_id, str):
                    item = link(public / 'eval_logs' / attempt.commit_hash / run_id, run_id)
                    if item:
                        runs.append(item)
            lines.append(f'| {attempt.metadata.get("mode", "unknown")} | `{attempt.commit_hash}` | '
                         f'{attempt.status} | {attempt.budget_class} | '
                         f'{"archived" if attempt.archived else "current"} | '
                         f'{attempt_link} | {", ".join(runs) or "unavailable"} |')
        lines.extend(['', '## Supporting evidence', '',
                      'Attachments are agent-provided files, not host-verified executions. '
                      'Use their manifest for author, code SHA and content hashes.'])
        attachments = record.get('attachments', [])
        for item in attachments:
            manifest = link(public / item['path'] / 'manifest.json')
            if manifest:
                lines.append(f'- {manifest}')
        if not attachments:
            lines.append('- No supporting attachments archived.')
        lines.extend(['', '## Managed diagnostics', ''])
        diagnostics = public / 'diagnostics'
        directories = sorted(diagnostics.glob(f'{node_id}-*')) if diagnostics.exists() else []
        found = False
        for directory in directories:
            for child in ('records', 'workspace'):
                item = link(directory / child)
                if item:
                    found = True
                    lines.append(f'- {item}')
        if not found:
            lines.append('- No managed diagnostic workspace or records.')
        lines.extend(['', 'Workspace code/evidence are fixed session inputs; experiments/results are working files. '
                      'For a completed diagnostic, prefer its immutable records/<request-id>/ copy. '
                      'Do not use historical source_worktree paths as code recovery locations.'])
        atomic_write_text(node / 'README.md', '\n'.join(lines) + '\n', mode=0o644)
