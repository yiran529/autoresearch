"""Cross-Node context is public, frozen per Analysis version, and recoverable."""
# ruff: noqa: S101

import hashlib
import json
from pathlib import Path

import pytest
from test_analysis_recovery import _analysis_payload, _analyze, _context, _publisher

PREFIX = 'diagnostics/analysis-context-N001-research-v1'


def public_inputs(context):
    for name in ('tree.md', 'research_summary.md', 'engineering_summary.md', 'wisdom.md'):
        (context.notes_root / name).write_text('original ' + name)
    node = context.notes_root / 'nodes/N002'
    node.mkdir()
    for name in ('README.md', 'spec.md', 'evaluation.md', 'research.json',
                 'research_analysis_v1.md', 'research_analysis_v1.json', 'engineering_analysis_v1.md'):
        (node / name).write_text('published ' + name)
    (node / 'unrelated-secret').write_text('never supply this secret')


def response(scratch, reference=None):
    value = _analysis_payload(None)
    if reference:
        value['handover'] = dict(mechanism_verdict='inconclusive', practical_verdict='inconclusive',
                                evidence_refs=[reference], new_information=[], follow_up_questions=[])
    output = Path(scratch) / 'output.json'
    output.write_text(json.dumps(value))
    return output


def test_other_nodes_and_global_knowledge_are_frozen_without_live_mounts(tmp_path):
    context = _context(tmp_path)
    public_inputs(context)
    observed = {}

    def runner(prompt, roots, schema, scratch):
        evidence = next(Path(p) for p in roots if (Path(p) / 'spec.md').is_file())
        frozen = evidence / PREFIX
        for name in ('tree.md', 'research_summary.md', 'engineering_summary.md', 'wisdom.md'):
            assert (frozen / 'notes' / name).read_text() == 'original ' + name
        for name in ('README.md', 'spec.md', 'evaluation.md', 'research.json',
                     'research_analysis_v1.md', 'research_analysis_v1.json', 'engineering_analysis_v1.md'):
            assert (frozen / 'notes/nodes/N002' / name).read_text() == 'published ' + name
        assert not list(frozen.rglob('unrelated-secret'))
        assert context.notes_root not in roots and context.checkout_root not in roots
        assert not (frozen / 'notes/nodes/N001').exists()
        observed['prompt'] = prompt
        return response(scratch, PREFIX + '/notes/wisdom.md')

    result = _analyze('N001', 'research', context=context, runner=runner,
                      publisher=_publisher(context.notes_root))
    refs = [r.path for r in result.evidence.evidence_records]
    assert PREFIX + '/notes/wisdom.md' in refs
    assert 'analysis-context-N001-research-v1' in observed['prompt']


def test_live_updates_do_not_change_input_or_invalidate_published_analysis(tmp_path):
    from autoresearch.analysis import _artifact_evidence, resolve_evidence

    context = _context(tmp_path)
    public_inputs(context)

    def runner(prompt, roots, schema, scratch):
        (context.notes_root / 'wisdom.md').write_text('updated during analysis')
        (context.notes_root / 'nodes/N002/research.json').write_text('later progress')
        return response(scratch, PREFIX + '/notes/wisdom.md')

    first = _analyze('N001', 'research', context=context, runner=runner,
                     publisher=_publisher(context.notes_root))
    document = json.loads(first.analysis_path.read_text())
    recovered = _artifact_evidence(document, resolve_evidence('N001', 'research', context=context), context)
    assert recovered.manifest_hash == first.evidence.manifest_hash
    (context.notes_root / 'wisdom.md').unlink()
    second = _analyze('N001', 'research', context=context,
                      runner=lambda *_: pytest.fail('published Analysis must be reused'),
                      publisher=_publisher(context.notes_root))
    assert second.evidence.manifest_hash == first.evidence.manifest_hash
    snapshot = context.coral_dir / 'public' / PREFIX / 'notes/wisdom.md'
    assert snapshot.read_text() == 'original wisdom.md'
    snapshot.write_text('tampered snapshot')
    with pytest.raises(ValueError, match='evidence changed'):
        _analyze('N001', 'research', context=context,
                 publisher=_publisher(context.notes_root), runner=lambda *_: pytest.fail('must reject tampering'))


def test_interrupted_analysis_reuses_context_but_revision_gets_new_snapshot(tmp_path):
    context = _context(tmp_path)
    public_inputs(context)
    calls = []

    def runner(prompt, roots, schema, scratch):
        frozen = next(Path(p) for p in roots if (Path(p) / 'spec.md').is_file())
        copies = list(frozen.glob('diagnostics/analysis-context-*/notes/wisdom.md'))
        assert len(copies) == 1
        calls.append(copies[0].read_text())
        if len(calls) == 1:
            raise RuntimeError('interrupted before publication')
        return response(scratch)

    with pytest.raises(RuntimeError, match='interrupted'):
        _analyze('N001', 'research', context=context, runner=runner,
                 publisher=_publisher(context.notes_root))
    (context.notes_root / 'wisdom.md').write_text('new knowledge')
    first = _analyze('N001', 'research', context=context, runner=runner,
                     publisher=_publisher(context.notes_root))
    _analyze('N001', 'research', revise=True, context=context, runner=runner,
             publisher=_publisher(context.notes_root))
    assert calls == ['original wisdom.md', 'original wisdom.md', 'new knowledge']
    assert first.analysis_version == 1


def test_context_never_follows_symlink_sources(tmp_path):
    context = _context(tmp_path)
    public_inputs(context)
    private = tmp_path / 'private'
    private.mkdir()
    (private / 'spec.md').write_text('private secret payload')
    (context.notes_root / 'wisdom.md').unlink()
    (context.notes_root / 'wisdom.md').symlink_to(private / 'spec.md')
    (context.notes_root / 'nodes/N003').symlink_to(private, target_is_directory=True)

    def runner(prompt, roots, schema, scratch):
        evidence = next(Path(p) for p in roots if (Path(p) / 'spec.md').is_file())
        frozen = evidence / PREFIX
        assert frozen.is_dir()
        assert not (frozen / 'notes/wisdom.md').exists()
        assert not (frozen / 'notes/nodes/N003').exists()
        assert all(b'private secret payload' not in p.read_bytes() for p in frozen.rglob('*') if p.is_file())
        return response(scratch)

    _analyze('N001', 'research', context=context, runner=runner,
             publisher=_publisher(context.notes_root))


def test_live_notes_alias_does_not_bypass_frozen_inventory(tmp_path):
    context = _context(tmp_path)
    public_inputs(context)
    with pytest.raises(ValueError, match='not in the frozen evidence'):
        _analyze('N001', 'research', context=context,
                 runner=lambda p, r, s, t: response(t, 'notes/wisdom.md'),
                 publisher=_publisher(context.notes_root))


def test_memory_stages_only_hash_matched_context_from_analysis_manifest(tmp_path):
    from autoresearch.memory import MemoryContext, _stage_public_inputs

    context = MemoryContext(tmp_path / '.coral')
    report = context.notes_root / 'nodes/N001/research_analysis_v1.md'
    report.parent.mkdir(parents=True)
    source = context.island_root / PREFIX / 'notes/wisdom.md'
    source.parent.mkdir(parents=True)
    source.write_text('frozen historical knowledge')
    (source.parent / 'unrelated.md').write_text('not referenced')
    report.write_text(json.dumps({'evidence_manifest': {'evidence_records': [{
        'path': PREFIX + '/notes/wisdom.md', 'status': 'present',
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    }]}}))
    stage = tmp_path / 'stage'
    _stage_public_inputs(context, stage, evidence_refs={'nodes/N001/research_analysis_v1.md'})
    assert (stage / 'public' / PREFIX / 'notes/wisdom.md').read_text() == source.read_text()
    assert not (stage / 'public' / PREFIX / 'notes/unrelated.md').exists()


def test_existing_unfinished_task_preserves_its_original_evidence_identity(tmp_path):
    from dataclasses import replace

    from autoresearch.analysis import resolve_evidence
    from autoresearch.analysis_context import with_public_context
    from autoresearch.diagnostics import create_task

    context = _context(tmp_path)
    context = replace(context, attempts=(replace(context.attempts[0], commit_hash='a' * 40),))
    public_inputs(context)
    original = resolve_evidence('N001', 'research', context=context)
    task = create_task(context.coral_dir, 'N001', 'research', 1, original.research_sha,
                       evidence_hash=original.manifest_hash)
    assert with_public_context(original, context, 1) == original
    assert not (context.coral_dir / 'public' / PREFIX).exists()
    assert task.read()['evidence_hash'] == original.manifest_hash
    revised = with_public_context(original, context, 2)
    assert revised.manifest_hash != original.manifest_hash
    assert resolve_evidence('N001', 'research', context=context) == original


def test_bounded_context_inventory_reports_omissions(tmp_path, monkeypatch):
    from autoresearch import analysis_context as module
    from autoresearch.analysis import resolve_evidence

    context = _context(tmp_path)
    public_inputs(context)
    (context.notes_root / 'tree.md').write_text('t' * 50)
    (context.notes_root / 'research_summary.md').write_text('r' * 24)
    (context.notes_root / 'engineering_summary.md').write_text('e' * 24)
    (context.notes_root / 'wisdom.md').write_text('w' * 8)
    monkeypatch.setattr(module, 'MAX_FILE_BYTES', 32)
    monkeypatch.setattr(module, 'MAX_TOTAL_BYTES', 40)
    monkeypatch.setattr(module, 'MAX_FILES', 2)
    result = module.with_public_context(resolve_evidence('N001', 'research', context=context), context, 1)
    # context.json has its own metadata bound, independent of source-file limits.
    root = context.coral_dir / 'public' / PREFIX
    inventory = json.loads((root / 'context.json').read_text())['inventory']
    statuses = {item['source']: item['status'] for item in inventory}
    assert statuses['tree.md'] == 'file_byte_limit'
    assert statuses['engineering_summary.md'] == 'total_byte_limit'
    assert 'file_limit' in statuses.values()
    assert sum(item.get('bytes', 0) for item in inventory) <= 40
    assert len([r for r in result.evidence_records if r.path.startswith(PREFIX + '/notes/')]) == 2


def test_concurrent_collection_publishes_once_and_reuses_exact_bytes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from autoresearch import analysis_context as module
    from autoresearch.analysis import resolve_evidence

    context = _context(tmp_path)
    public_inputs(context)
    evidence = resolve_evidence('N001', 'research', context=context)
    collect = module._collect
    calls = []

    def counted(*args):
        calls.append(1)
        return collect(*args)

    monkeypatch.setattr(module, '_collect', counted)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(module.with_public_context, evidence, context, 1) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert calls == [1]
    assert results[0] == results[1]
    assert not list((context.coral_dir / 'public/diagnostics').glob('.analysis-context-*'))


@pytest.mark.parametrize('corruption', ['traversal', 'identity', 'symlink', 'missing'])
def test_corrupt_snapshot_is_rejected_without_recollection(tmp_path, corruption):
    from autoresearch.analysis import resolve_evidence
    from autoresearch.analysis_context import with_public_context

    context = _context(tmp_path)
    public_inputs(context)
    evidence = resolve_evidence('N001', 'research', context=context)
    with_public_context(evidence, context, 1)
    root = context.coral_dir / 'public' / PREFIX
    metadata = root / 'context.json'
    document = json.loads(metadata.read_text())
    if corruption == 'traversal':
        document['inventory'][0]['source'] = '../../private/secret'
        metadata.write_text(json.dumps(document))
    elif corruption == 'identity':
        document['identity']['node_id'] = 'N002'
        metadata.write_text(json.dumps(document))
    elif corruption == 'symlink':
        frozen = root / 'notes/wisdom.md'
        frozen.unlink()
        frozen.symlink_to(context.notes_root / 'wisdom.md')
    else:
        metadata.unlink()
    (context.notes_root / 'wisdom.md').write_text('must not recapture')
    with pytest.raises((ValueError, OSError)):
        with_public_context(evidence, context, 1)


def test_failed_collection_never_publishes_partial_snapshot(tmp_path, monkeypatch):
    from autoresearch import analysis_context as module
    from autoresearch.analysis import resolve_evidence

    context = _context(tmp_path)
    public_inputs(context)
    evidence = resolve_evidence('N001', 'research', context=context)
    collect = module._collect

    def interrupted(*args):
        collect(*args)
        raise RuntimeError('host interrupted before publication')

    monkeypatch.setattr(module, '_collect', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.with_public_context(evidence, context, 1)
    assert not (context.coral_dir / 'public' / PREFIX).exists()
    assert not list((context.coral_dir / 'public/diagnostics').glob('.analysis-context-*'))
    monkeypatch.setattr(module, '_collect', collect)
    assert module.with_public_context(evidence, context, 1).manifest_hash != evidence.manifest_hash


def test_memory_preserves_raw_results_before_history_across_required_reports(tmp_path):
    from autoresearch.memory import MemoryContext, _stage_public_inputs

    context = MemoryContext(tmp_path / '.coral')
    node = context.notes_root / 'nodes/N001'
    node.mkdir(parents=True)
    history = []
    for index in range(256):
        ref = PREFIX + f'/notes/nodes/N002/research_analysis_v{index + 1}.md'
        source = context.island_root / ref
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('published history')
        history.append({'path': ref, 'status': 'present',
                        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    raw_ref = 'eval_logs/current/results.json'
    raw = context.island_root / raw_ref
    raw.parent.mkdir(parents=True)
    raw.write_text('{"quality":3.8}')
    # History occurs in the first selected report, raw evidence in a later one.
    for name, records in [('engineering_analysis_v1.md', history), ('research_analysis_v1.md', [{
        'path': raw_ref, 'status': 'present', 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    }])]:
        (node / name).write_text(json.dumps({'evidence_manifest': {'evidence_records': records}}))
    stage = tmp_path / 'stage'
    _stage_public_inputs(context, stage, evidence_refs={
        'nodes/N001/engineering_analysis_v1.md', 'nodes/N001/research_analysis_v1.md',
    })
    assert (stage / 'public' / raw_ref).read_bytes() == raw.read_bytes()
    assert len(list((stage / 'public' / PREFIX).rglob('*.md'))) == 255
