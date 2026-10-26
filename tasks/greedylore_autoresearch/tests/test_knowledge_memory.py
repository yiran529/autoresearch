"""Knowledge publication, cumulative provenance and incremental public evidence."""
# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json

import pytest
from autoresearch.memory import (
    MemoryContext,
    consolidate_memory,
    memory_due,
    read_processed_analyses,
)
from test_memory import _analysis, _summary


def response(scratch, *, wisdom='# Wisdom\nNo reusable finding yet.'):
    path = scratch / 'response.json'
    path.write_text(json.dumps({'research_summary': '# Research\nSee the new Analysis.',
        'engineering_summary': '# Engineering\nNo measured runtime change.', 'wisdom': wisdom}))
    return path


def test_host_publishes_three_documents_and_owns_cumulative_coverage(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    first = _analysis(context.notes_root, 'N001', 'research', 1)
    consolidate_memory(if_due=True, context=context, runner=lambda p, r, s, t: response(t))
    for name in ('research_summary.md', 'engineering_summary.md', 'wisdom.md'):
        assert read_processed_analyses(context.notes_root / name) == {first}
    assert not memory_due(context.notes_root).due
    second = _analysis(context.notes_root, 'N002', 'research', 1)
    calls = []
    def run(prompt, roots, schema, scratch):
        metadata = json.loads(prompt.rsplit('\n\n', 1)[1])
        calls.append(metadata)
        assert (roots[0] / 'wisdom.md').is_file()
        # Historical reports remain available; they are not all required reading.
        assert (roots[0] / first).is_file()
        return response(scratch)
    consolidate_memory(if_due=True, context=context, runner=run)
    assert calls[0]['required_refs'] == [second]
    for name in ('research_summary.md', 'engineering_summary.md', 'wisdom.md'):
        assert read_processed_analyses(context.notes_root / name) == {first, second}


def test_wisdom_revision_keeps_counterexample_and_both_analysis_sources(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    first = _analysis(context.notes_root, 'N001', 'research', 1)
    old = f'# Wisdom\n## W01\nProvisional: history helps. Source: [{first}]({first}).'
    consolidate_memory(if_due=True, context=context, runner=lambda p, r, s, t: response(t, wisdom=old))
    second = _analysis(context.notes_root, 'N002', 'research', 1)
    revised = f'# Wisdom\n## W01\nScope narrowed: stable states only; drift is a counterexample. Sources: [{first}]({first}), [{second}]({second}).'
    consolidate_memory(if_due=True, context=context, runner=lambda p, r, s, t: response(t, wisdom=revised))
    text = (context.notes_root / 'wisdom.md').read_text()
    assert 'Provisional: history helps.' not in text
    assert 'drift is a counterexample' in text and first in text and second in text
    assert consolidate_memory(if_due=True, context=context,
        runner=lambda *_: pytest.fail('completed knowledge repeated')).status == 'not_due'


def test_partial_wisdom_publication_recovers_cached_output_without_model_rerun(tmp_path):
    import autoresearch.memory as memory
    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    calls = []
    def run(prompt, roots, schema, scratch):
        calls.append(prompt)
        return response(scratch)
    def publish(kind, content):
        if kind == 'wisdom':
            raise OSError('wisdom publish interrupted')
        return memory._host_publish(context, kind, content)
    with pytest.raises(OSError, match='interrupted'):
        consolidate_memory(if_due=True, context=context, runner=run, publisher=publish)
    assert memory_due(context.notes_root).due
    consolidate_memory(if_due=True, context=context, runner=run)
    assert len(calls) == 1
    assert read_processed_analyses(context.notes_root / 'wisdom.md') == {ref}
    assert not memory_due(context.notes_root).due


def test_legacy_summaries_require_wisdom_bootstrap_once(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    refs = [_analysis(context.notes_root, f'N{i:03}', 'research', 1) for i in (1, 2)]
    for name in ('research_summary.md', 'engineering_summary.md'):
        (context.notes_root / name).write_text(_summary(refs))
    assert memory_due(context.notes_root).due
    def run(prompt, roots, schema, scratch):
        metadata = json.loads(prompt.rsplit('\n\n', 1)[1])
        assert set(metadata['required_refs']) == set(refs)
        return response(scratch)
    consolidate_memory(if_due=True, context=context, runner=run)
    assert not memory_due(context.notes_root).due


def test_wisdom_cannot_cite_unavailable_analysis(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    _analysis(context.notes_root, 'N001', 'research', 1)
    with pytest.raises(ValueError, match='wisdom|Wisdom|reference'):
        consolidate_memory(if_due=True, context=context,
            runner=lambda p, r, s, t: response(t, wisdom='# Wisdom\nSource: nodes/N999/research_analysis_v1.md'))
    assert not (context.notes_root / 'wisdom.md').exists()


def test_staging_omits_unreferenced_logs_and_includes_referenced_research_evidence(tmp_path):
    from autoresearch.memory import _stage_public_inputs
    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    useful = context.island_root / 'eval_logs/sha/run/metric.json'
    useful.parent.mkdir(parents=True)
    useful.write_text('{"loss":3.0}')
    unrelated = useful.parent / 'unrelated.json'
    unrelated.write_text('{"irrelevant":true}')
    diagnostic = context.island_root / 'research_diagnostics/N001/probe/metric.csv'
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text('gap\n0.1\n')
    (context.notes_root / ref).write_text(json.dumps({'evidence_manifest': {'evidence_records': [
        {'path': 'eval_logs/sha/run/metric.json', 'status': 'present',
         'sha256': hashlib.sha256(useful.read_bytes()).hexdigest()},
        {'path': 'research_diagnostics/N001/probe/metric.csv', 'status': 'present',
         'sha256': hashlib.sha256(diagnostic.read_bytes()).hexdigest()}]}}))
    scratch = tmp_path / 'staged'
    _stage_public_inputs(context, scratch, evidence_refs={ref})
    assert (scratch / 'public/eval_logs/sha/run/metric.json').read_text() == useful.read_text()
    assert not (scratch / 'public/eval_logs/sha/run/unrelated.json').exists()
    assert (scratch / 'public/research_diagnostics/N001/probe/metric.csv').is_file()


def test_existing_legacy_queue_and_two_document_cache_are_bootstrapped(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    memory_due(context.notes_root)
    journal = context.coral_dir / 'private/memory_state.json'
    state = json.loads(journal.read_text())
    state['nodes']['N001'].update(status='completed')
    state['nodes']['N001'].pop('knowledge_version', None)
    journal.write_text(json.dumps(state))
    legacy = {key: _summary([ref]) for key in ('research_summary', 'engineering_summary')}
    for key, text in legacy.items():
        (context.notes_root / f'{key}.md').write_text(text)
    cache = context.coral_dir / 'private/memory_outputs/N001.json'
    cache.parent.mkdir()
    cache.write_text(json.dumps({'trigger_refs': [ref], 'document': legacy}))
    calls = []
    def run(p, r, s, t):
        calls.append(p)
        return response(t)
    consolidate_memory(if_due=True, context=context, runner=run)
    assert len(calls) == 1 and (context.notes_root / 'wisdom.md').is_file()
    assert not memory_due(context.notes_root).due


def test_model_frontmatter_cannot_invent_coverage(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    def run(p, r, s, t):
        path = response(t)
        authored = json.loads(path.read_text())
        authored['research_summary'] = _summary(['nodes/N999/research_analysis_v1.md'])
        path.write_text(json.dumps(authored))
        return path
    consolidate_memory(if_due=True, context=context, runner=run)
    assert read_processed_analyses(context.notes_root / 'research_summary.md') == {ref}


def test_pending_nodes_are_not_supplied_as_history_before_their_turn(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    first = _analysis(context.notes_root, 'N001', 'research', 1)
    second = _analysis(context.notes_root, 'N002', 'research', 1)
    calls = []
    def run(prompt, roots, schema, scratch):
        calls.append(prompt)
        assert (roots[0] / first).is_file()
        assert (roots[0] / second).exists() == (len(calls) == 2)
        return response(scratch)
    consolidate_memory(if_due=True, context=context, runner=run)
    assert len(calls) == 2


def test_historical_research_records_do_not_enter_incremental_inputs(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    for i in (1, 2):
        node = f'N{i:03}'
        ref = _analysis(context.notes_root, node, 'research', 1)
        for name in ('spec.md', 'evaluation.md'):
            (context.notes_root / f'nodes/{node}/{name}').write_text('Current input')
        (context.notes_root / f'nodes/{node}/research.json').write_text('{"large_history":true}')
        (context.notes_root / ref).write_text(json.dumps({'evidence_manifest': {'evidence_records': [
            {'path': f'notes/nodes/{node}/research.json', 'status': 'present',
             'sha256': hashlib.sha256((context.notes_root / f'nodes/{node}/research.json').read_bytes()).hexdigest()}]}}))
    calls = []
    def run(prompt, roots, schema, scratch):
        calls.append(prompt)
        current = 'N001' if len(calls) == 1 else 'N002'
        assert (roots[0] / f'nodes/{current}/research.json').is_file()
        if len(calls) == 2:
            for name in ('research.json', 'spec.md', 'evaluation.md'):
                assert not (roots[0] / f'nodes/N001/{name}').exists()
        return response(scratch)
    consolidate_memory(if_due=True, context=context, runner=run)
    assert len(calls) == 2


@pytest.mark.parametrize('digest', [None, '0' * 64])
def test_raw_evidence_without_matching_frozen_hash_is_unavailable(tmp_path, digest):
    from autoresearch.memory import _stage_public_inputs
    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    source = context.island_root / 'eval_logs/changed.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"loss":0}')
    record = {'path': 'eval_logs/changed.json', 'status': 'present'}
    if digest is not None:
        record['sha256'] = digest
    (context.notes_root / ref).write_text(json.dumps({
        'evidence_manifest': {'evidence_records': [record]}}))
    scratch = tmp_path / 'staged'
    _stage_public_inputs(context, scratch, evidence_refs={ref})
    assert not (scratch / 'public/eval_logs/changed.json').exists()
