# ruff: noqa: S101
"""Delivered prompt size must not scale with per-file hashes or repeated guidance."""
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

from autoresearch.analysis import _prepare_evidence_snapshot, _prompt, resolve_evidence
from autoresearch.checkpoints import discover_checkpoints
from test_analysis_recovery import _context
from test_checkpoints import _checkpoint


def test_analyzer_spills_complete_manifest_with_missing_records_and_stable_identity(tmp_path):
    context = _context(tmp_path)
    logs = context.log_root / ('r' * 40) / ('run-' + 'r' * 40)
    logs.mkdir(parents=True)
    for i in range(100):
        (logs / f'log-{i}.txt').write_text(f'evidence {i}')
    evidence = resolve_evidence('N001', 'research', context=context)
    first = _prepare_evidence_snapshot(evidence, context, tmp_path / 'first')
    second = _prepare_evidence_snapshot(evidence, context, tmp_path / 'retry')
    manifest = first / 'evidence_manifest.json'
    assert manifest.is_file()
    assert json.loads(manifest.read_text()) == evidence.to_dict()
    assert manifest.read_bytes() == (second / manifest.name).read_bytes()
    index = (first / 'evidence_index.md').read_text()
    for record in evidence.evidence_records:
        assert record.path in index
        if record.status == 'present':
            assert (first / record.path).is_file()
    assert 'missing' in index
    prompt = _prompt(evidence, first)
    assert str(manifest) in prompt
    assert 'missing' in prompt
    assert evidence.research_sha in prompt
    assert all(record.sha256 not in prompt for record in evidence.evidence_records if record.sha256)
    small = replace(evidence, evidence_records=evidence.evidence_records[:1])
    assert len(prompt) - len(_prompt(small, first)) < 300


def test_checkpoint_guidance_once_and_all_checkpoint_metadata_retained(tmp_path):
    context = _context(tmp_path)
    checkpoint = _checkpoint(context)
    other = checkpoint.with_name('second-workload')
    shutil.copytree(checkpoint, other)
    evidence = resolve_evidence('N001', 'research', context=context)
    roots, prompt = discover_checkpoints(evidence, context)
    assert set(roots) == {checkpoint, other}
    assert str(checkpoint) in prompt and str(other) in prompt
    assert prompt.count('diagnostic_checkpoint.load_checkpoint') == 1
    assert 'restore_rng=False' in prompt and 'register_checkpoint_hook' in prompt


def test_memory_spills_coverage_without_losing_legacy_required_reading(tmp_path):
    from autoresearch.memory import MemoryContext, _consolidate_node
    from test_memory import _analysis, _summary

    context = MemoryContext(tmp_path / '.coral')
    refs = [_analysis(context.notes_root, f'N{i:03}', 'research', 1) for i in range(20)]
    # Legacy summaries have coverage, but wisdom does not: all prior refs are mandatory.
    for name in ('research_summary.md', 'engineering_summary.md'):
        (context.notes_root / name).write_text(_summary(refs[:-1]))
    captured = []

    def runner(prompt, roots, schema, scratch):
        task_file = roots[0] / 'memory_task.json'
        assert task_file.is_file()
        task = json.loads(task_file.read_text())
        assert task['covered_refs'] == refs
        assert task['required_refs'] == refs
        assert str(task_file) in prompt
        assert refs[0] not in prompt  # long required list is reached through mandatory file
        assert 'every Analysis in its required_refs' in prompt
        captured.append(task)
        output = scratch / 'output.json'
        output.write_text(json.dumps({name: 'Evidence retained.' for name in
            ('research_summary', 'engineering_summary', 'wisdom')}))
        return output

    _consolidate_node(context, SimpleNamespace(pending_nodes=('N019',),
        trigger_refs=frozenset({refs[-1]})), runner, None)
    assert captured


def test_spilled_analyzer_inputs_are_readonly_in_actual_sandbox(tmp_path):
    import subprocess

    import pytest
    from autoresearch.artifact_paths import diagnostic_view
    from autoresearch.diagnostic_sandbox import build_sandbox, isolation_available

    if not isolation_available():
        pytest.skip('real bubblewrap namespaces unavailable')
    context = _context(tmp_path)
    evidence = resolve_evidence('N001', 'research', context=context)
    workspace = tmp_path / 'workspace'
    snapshot = _prepare_evidence_snapshot(evidence, context, workspace)
    for name in ('experiments', 'results', 'records'):
        (workspace / name).mkdir()
    prompt = diagnostic_view(workspace).rewrite_text(_prompt(evidence, snapshot))
    assert '/workspace/evidence/evidence_manifest.json' in prompt
    script = '''
import errno, json
from pathlib import Path
root = Path('/workspace/evidence')
assert json.loads((root / 'evidence_manifest.json').read_text())['manifest']['node_id'] == 'N001'
assert 'missing' in (root / 'evidence_index.md').read_text()
for name in ('evidence_manifest.json', 'evidence_index.md'):
    try:
        (root / name).write_text('changed')
    except OSError as exc:
        assert exc.errno in (errno.EROFS, errno.EACCES)
    else:
        raise AssertionError('prompt evidence is writable')
'''
    command = build_sandbox(workspace, [snapshot], [], ['/usr/bin/python3', '-c', script])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
