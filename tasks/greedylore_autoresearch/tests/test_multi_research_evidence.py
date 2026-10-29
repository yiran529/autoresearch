from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest
from autoresearch.analysis import (
    _artifact_evidence,
    _prepare_code_snapshot,
    _prepare_evidence_snapshot,
    _prompt,
    resolve_evidence,
)
from autoresearch.evaluation import refresh_node_evaluation
from autoresearch.validation import seal_document, validate_analysis
from test_analysis_recovery import _analysis_payload, _attempt, _context


def _multi_context(tmp_path):
    context = _context(tmp_path)
    first = context.attempts[0]
    second = _attempt('a' * 40, 'research', '2026-01-02T00:00:00+00:00')
    second.metadata['archived'] = True
    second.status = 'crashed'
    second.metadata['budget_class'] = 'grader_error'
    context = replace(context, attempts=(first, second))
    state = context.coral_dir / 'private/autoresearch_state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'nodes': {'N001': {'research': {
        'status': 'analyzing', 'selected_sha': first.commit_hash,
    }}}}))
    for attempt in context.attempts:
        raw = context.log_root / attempt.commit_hash / 'raw.json'
        raw.parent.mkdir(parents=True)
        raw.write_text(json.dumps({'commit': attempt.commit_hash}))
    return context


def test_all_research_candidates_are_fixed_evidence_and_recoverable(tmp_path):
    context = _multi_context(tmp_path)
    evidence = resolve_evidence('N001', 'research', context=context)
    assert evidence.research_sha == 'r' * 40
    candidates = evidence.manifest()['research_attempts']
    assert [item['commit_sha'] for item in candidates] == ['r' * 40, 'a' * 40]
    assert candidates[1]['status'] == 'crashed'
    assert candidates[1]['archived'] is True
    assert candidates[1]['budget_class'] == 'grader_error'
    snapshot = _prepare_evidence_snapshot(evidence, context, tmp_path / 'scratch')
    assert (snapshot / ('eval_logs/' + 'a' * 40 + '/raw.json')).is_file()
    assert 'Research candidate inventory' in _prompt(evidence)
    assert 'a' * 40 in _prompt(evidence)
    document = seal_document(_analysis_payload(None), evidence)
    recovered = _artifact_evidence(document, evidence, context)
    assert validate_analysis(document, recovered) == document
    assert recovered.manifest() == evidence.manifest()


def test_evaluation_shows_frozen_anchor_and_keeps_every_candidate(tmp_path):
    context = _multi_context(tmp_path)
    report = refresh_node_evaluation('N001', context.attempts, context.log_root,
                                     coral_dir=context.coral_dir)
    assert 'Selected Research SHA: `' + 'r' * 40 + '`' in report
    assert report.count('| research |') == 2
    assert 'crashed' in report


def test_multiple_candidates_require_frozen_selection(tmp_path):
    context = _multi_context(tmp_path)
    (context.coral_dir / 'private/autoresearch_state.json').unlink()
    with pytest.raises(ValueError, match='(?i)(select|frozen|Research)'):
        resolve_evidence('N001', 'research', context=context)


def test_legacy_singleton_manifest_stays_unchanged(tmp_path):
    context = _context(tmp_path)
    evidence = resolve_evidence('N001', 'research', context=context)
    assert 'research_attempts' not in evidence.manifest()


def test_legacy_diagnostic_backfill_does_not_invalidate_sealed_analysis(tmp_path, monkeypatch):
    from autoresearch.artifact_paths import research_diagnostics_root
    from autoresearch.research_record import ResearchRecord

    context = _multi_context(tmp_path)
    state_path = context.coral_dir / 'private/autoresearch_state.json'
    state = json.loads(state_path.read_text())
    state['nodes']['N001']['research']['agent_id'] = 'research-N001'
    state_path.write_text(json.dumps(state))
    frozen_record = {'frozen': True}
    (context.notes_root / 'nodes/N001/research.json').write_text(json.dumps(frozen_record))
    monkeypatch.setattr(ResearchRecord, 'read', lambda self: frozen_record)
    placeholder = context.coral_dir / 'public/research_diagnostics'
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()
    before = resolve_evidence('N001', 'research', context=context)
    document = seal_document(_analysis_payload(None), before)
    mirror = research_diagnostics_root(context.coral_dir) / 'N001/old-probe'
    mirror.mkdir(parents=True)
    (mirror / 'diagnostic-result.json').write_text('{"status":"completed"}')
    placeholder.unlink()
    after = resolve_evidence('N001', 'research', context=context)
    assert after.manifest() == before.manifest()
    assert validate_analysis(document, _artifact_evidence(document, after, context)) == document


def test_code_snapshot_contains_all_research_candidates(tmp_path):
    context = _multi_context(tmp_path)
    context.checkout_root.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(context.checkout_root), *args],
                              check=True, capture_output=True, text=True).stdout.strip()
    git('init')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    attempts = []
    for index, attempt in enumerate(context.attempts):
        (context.checkout_root / 'source.txt').write_text(str(index))
        git('add', 'source.txt')
        git('commit', '-m', str(index))
        attempts.append(replace(attempt, commit_hash=git('rev-parse', 'HEAD')))
    context = replace(context, attempts=tuple(attempts))
    state = context.coral_dir / 'private/autoresearch_state.json'
    state.write_text(json.dumps({'nodes': {'N001': {'research': {
        'selected_sha': attempts[0].commit_hash,
    }}}}))
    evidence = resolve_evidence('N001', 'research', context=context)
    root = _prepare_code_snapshot(evidence, context, tmp_path / 'scratch')
    assert [(root / a.commit_hash / 'source.txt').read_text() for a in attempts] == ['0', '1']


def test_evaluation_renders_candidates_before_selection(tmp_path):
    context = _multi_context(tmp_path)
    (context.coral_dir / 'private/autoresearch_state.json').unlink()
    report = refresh_node_evaluation('N001', context.attempts, context.log_root)
    assert 'Selected Research SHA: pending Research finish.' in report
    assert report.count('| research |') == 2
