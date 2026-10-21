"""Completed-Node Memory queue and Summary format checks."""

# ruff: noqa: S101

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest
import yaml
from autoresearch.artifacts import validate_artifact
from autoresearch.memory import (
    MemoryContext,
    consolidate_memory,
    list_analysis_refs,
    memory_due,
    read_processed_analyses,
    validate_summary,
)


def _complete(notes, node, *, engineering='skipped', research='completed', analysis='completed'):
    path = notes.parent.parent / 'private/autoresearch_state.json'
    state = json.loads(path.read_text()) if path.exists() else {'nodes': {}}
    state['nodes'][node] = {
        'research': {'status': research},
        'analysis': {'status': analysis, 'analysis_ref': f'nodes/{node}/research_analysis_v1.md'},
        'engineering_status': engineering,
        'engineering_skip_reason': 'quality_failed' if engineering == 'skipped' else None,
        'engineering_claims': [{'status': 'completed'}] if engineering == 'completed' else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _analysis(notes: Path, node: str, mode: str, version: int) -> str:
    relative = f"nodes/{node}/{mode}_analysis_v{version}.md"
    path = notes / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("published analysis\n")
    _complete(notes, node)
    return relative


def _summary(refs: list[str]) -> str:
    entries = "\n".join(f"  - {ref}" for ref in refs)
    return f"---\nprocessed_analyses:\n{entries or '  []'}\n---\n# Summary\n\nA finding.\n"


def test_memory_stage_retains_json_for_analysis_lookup(tmp_path):
    from autoresearch.memory import _stage_public_inputs

    from coral.grader.analysis_artifacts import load_analysis, publish_analysis

    context = MemoryContext(tmp_path / '.coral')
    path = context.notes_root / 'nodes/N001/research_analysis_v1.md'
    document = {'summary': 'Measured result.', 'findings': ['A finding.'],
                'limitations': ['One batch.'], 'node_id': 'N001', 'mode': 'research',
                'research_sha': 'a' * 40, 'evaluation_snapshot': 'frozen evaluation'}
    publish_analysis(path, document)
    staged, *_ = _stage_public_inputs(context, tmp_path / 'staged')
    assert load_analysis(staged / 'nodes/N001/research_analysis_v1.md') == document


@pytest.mark.parametrize('node', ['N000', 'N001'])
@pytest.mark.parametrize('last', ['analysis', 'engineering'])
def test_gate_requires_complete_node_in_either_branch_order(tmp_path, node, last):
    notes = tmp_path / '.coral/public/notes'
    research = _analysis(notes, node, 'research', 1)
    engineering = _analysis(notes, node, 'engineering', 1)
    _complete(notes, node, engineering='active' if last == 'engineering' else 'completed',
              analysis='pending' if last == 'analysis' else 'completed')
    assert not memory_due(notes).due
    _complete(notes, node, engineering='completed')
    gate = memory_due(notes)
    assert gate.due
    assert gate.trigger_refs == frozenset({research, engineering})
    assert gate.pending_nodes == (node,)


def test_research_file_alone_does_not_trigger_memory(tmp_path):
    notes = tmp_path / '.coral/public/notes'
    _analysis(notes, 'N001', 'research', 1)
    _complete(notes, 'N001', research='active')
    assert not memory_due(notes).due


def test_partial_summary_publication_keeps_gate_due(tmp_path):
    notes = tmp_path / ".coral" / "public" / "notes"
    first = _analysis(notes, "N001", "research", 1)
    (notes / "research_summary.md").write_text(_summary([first]))
    (notes / "engineering_summary.md").write_text(_summary([]))
    assert memory_due(notes).due


def test_summary_validates_only_canonical_existing_refs_and_required_trigger(tmp_path):
    notes = tmp_path / ".coral" / "public" / "notes"
    first = _analysis(notes, "N001", "research", 1)
    second = _analysis(notes, "N002", "research", 1)
    assert validate_summary(_summary([first]), notes) == frozenset({first})
    with pytest.raises(ValueError, match="trigger references"):
        validate_summary(_summary([first]), notes, required_refs=frozenset({first, second}))
    with pytest.raises(ValueError, match="canonical"):
        validate_summary(_summary(["nodes/../private/file.md"]), notes)
    with pytest.raises(ValueError, match="does not exist"):
        validate_summary(_summary(["nodes/N999/research_analysis_v1.md"]), notes)
    with pytest.raises(ValueError, match="frontmatter"):
        validate_summary("# No frontmatter\n", notes)
    with pytest.raises(ValueError, match="body is empty"):
        validate_summary("---\nprocessed_analyses: []\n---\n", notes)


def test_analysis_listing_ignores_noncanonical_and_engineering_by_mode(tmp_path):
    notes = tmp_path / ".coral" / "public" / "notes"
    first = _analysis(notes, "N001", "research", 1)
    _analysis(notes, "N001", "engineering", 1)
    _analysis(notes, "N0X1", "research", 1)
    assert list_analysis_refs(notes, mode="research") == (first,)
    assert read_processed_analyses(notes / "missing_summary.md") == frozenset()


def test_memory_worker_publishes_two_valid_summaries_per_node(tmp_path):
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    output = tmp_path / "output.json"
    output.write_text(json.dumps({
        "research_summary": _summary(refs),
        "engineering_summary": _summary(refs),
    }))
    calls = []

    def runner(prompt, read_roots, schema, scratch):
        calls.append((prompt, read_roots))
        return output

    published = []
    context = MemoryContext(coral_dir)
    result = consolidate_memory(
        if_due=True, context=context, runner=runner,
        publisher=lambda kind, content: published.append((kind, content)),
    )
    assert result.status == "published"
    assert [kind for kind, _ in published] == ["research_summary", "engineering_summary"] * 3
    assert len(calls) == 3


def test_memory_runner_receives_only_staged_public_inputs(tmp_path):
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    (notes / "tree.md").write_text("# Tree\n")
    private = coral_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "secret.txt").write_text("private marker")
    output = tmp_path / "output.json"
    output.write_text(json.dumps({
        "research_summary": _summary(refs),
        "engineering_summary": _summary(refs),
    }))

    def runner(prompt, read_roots, schema, scratch):
        assert all(root.is_relative_to(scratch) for root in read_roots)
        assert (read_roots[0] / "tree.md").read_text() == "# Tree\n"
        assert not list(scratch.rglob("secret.txt"))
        return output

    assert consolidate_memory(
        if_due=True, context=MemoryContext(coral_dir), runner=runner,
        publisher=lambda *_: None,
    ).status == "published"


def test_host_publisher_writes_fixed_summaries_and_partial_failure_remains_due(tmp_path):
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    output = tmp_path / "output.json"
    output.write_text(json.dumps({
        "research_summary": _summary(refs),
        "engineering_summary": _summary(refs),
    }))
    context = MemoryContext(coral_dir)

    def partial(kind, content):
        if kind == "engineering_summary":
            raise RuntimeError("publication interrupted")
        (notes / "research_summary.md").write_text(content)

    with pytest.raises(RuntimeError, match="interrupted"):
        consolidate_memory(
            if_due=True, context=context, runner=lambda *_: output,
            publisher=partial,
        )
    assert memory_due(notes).due
    result = consolidate_memory(
        if_due=True, context=context, runner=lambda *_: output,
    )
    assert result.status == "published"
    assert read_processed_analyses(notes / "research_summary.md") == frozenset(refs)
    assert read_processed_analyses(notes / "engineering_summary.md") == frozenset(refs)
    assert not memory_due(notes).due


def test_fake_consolidation_does_not_modify_static_reference_toolkit(tmp_path):
    task_root = Path(__file__).resolve().parents[1]
    reference_paths = sorted((task_root / "references").glob("*.md"))
    assert reference_paths
    before = {path: path.read_bytes() for path in reference_paths}
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    output = tmp_path / "output.json"
    output.write_text(json.dumps({
        "research_summary": _summary(refs),
        "engineering_summary": _summary(refs),
    }))
    consolidate_memory(if_due=True, context=MemoryContext(coral_dir), runner=lambda *_: output)
    assert {path: path.read_bytes() for path in reference_paths} == before


def test_memory_worker_skips_occupied_lock(tmp_path):
    coral_dir = tmp_path / ".coral"
    root = coral_dir / "public"
    root.mkdir(parents=True)
    with (root / "memory-consolidation.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = consolidate_memory(
            if_due=True, context=MemoryContext(coral_dir),
            runner=lambda *_: pytest.fail("locked worker must not start Codex"),
        )
    assert result.status == "locked"


def test_processed_summaries_skip_memory_runner(tmp_path):
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    (notes / "research_summary.md").write_text(_summary(refs))
    (notes / "engineering_summary.md").write_text(_summary(refs))
    result = consolidate_memory(
        if_due=True, context=MemoryContext(coral_dir),
        runner=lambda *_: pytest.fail("not due"),
        publisher=lambda *_: pytest.fail("not due"),
    )
    assert result.status == "not_due"


def test_async_cli_skips_socket_below_gate_and_sends_empty_payload_when_due(tmp_path, monkeypatch, capsys):
    from autoresearch import cli

    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    notes.mkdir(parents=True)
    monkeypatch.setattr(cli, "_coral_dir", lambda: coral_dir)
    calls = []
    monkeypatch.setattr(cli, "request_trusted", lambda *args: (
        calls.append(args) or {"status": "started"}
    ))
    cli.main(["consolidate-memory", "--if-due", "--async"])
    assert json.loads(capsys.readouterr().out) == {"status": "not_due"}
    assert calls == []

    for index in (1, 2, 3):
        _analysis(notes, f"N{index:03d}", "research", 1)
    cli.main(["consolidate-memory", "--if-due", "--async"])
    assert json.loads(capsys.readouterr().out) == {"status": "started"}
    assert calls == [(coral_dir, "start_memory_if_due", {})]


def test_task_uses_scheduler_finalization_instead_of_memory_startup_hook():
    task_root = Path(__file__).resolve().parents[1]
    task = yaml.safe_load((task_root / "task.yaml").read_text())
    templates = {item["role"]: item for item in task["agents"]["assignments"]}
    assert task["grader"]["args"]["trusted_workflow_worker"] == "autoresearch.lifecycle:run_worker"
    assert all("startup_hook" not in item["runtime_options"] for item in templates.values())


@pytest.mark.parametrize("kind", ["research_summary", "engineering_summary"])
def test_agent_artifact_router_cannot_publish_host_summary(kind):
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        validate_artifact(kind, _summary([]), {
            "node_id": None, "mode": None, "version": None,
        })


def test_memory_stages_only_matching_public_diagnostic_records_and_bounded_results(tmp_path):
    from autoresearch.memory import _stage_public_inputs

    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N001', 'research', 1)
    (context.notes_root / ref).write_text(json.dumps({'research_sha': 'a' * 40}))
    record = context.island_root / 'diagnostics/N001-research-v1/records/probe'
    record.mkdir(parents=True)
    metadata = {'node_id': 'N001', 'mode': 'research', 'analysis_version': 1,
                'research_sha': 'a' * 40, 'diagnostic_id': 'N001-research-v1/probe', 'status': 'completed'}
    (record / 'record.json').write_text(json.dumps(metadata))
    (record / 'results').mkdir()
    (record / 'results/metric.json').write_text('{"error":0.25}')
    (record / 'results/large.txt').write_bytes(b'x' * 1_000_001)
    (record / 'results/model.pt').write_bytes(b'not a text result')
    private = context.coral_dir / 'private'
    private.mkdir(exist_ok=True)
    (private / 'secret.txt').write_text('not staged')
    (record / 'results/link.txt').symlink_to(private / 'secret.txt')
    (record / 'scripts').mkdir()
    (record / 'scripts/probe.py').write_text('not staged')
    foreign = record.parent / 'foreign'
    foreign.mkdir()
    (foreign / 'record.json').write_text(json.dumps({**metadata, 'research_sha': 'b' * 40}))
    scratch = tmp_path / 'scratch'
    roots = _stage_public_inputs(context, scratch)
    staged = scratch / 'public/diagnostics/N001-research-v1/records/probe'
    assert (staged / 'record.json').is_file()
    assert (staged / 'results/metric.json').read_text() == '{"error":0.25}'
    for excluded in ('results/large.txt', 'results/model.pt', 'results/link.txt', 'scripts/probe.py'):
        assert not (staged / excluded).exists()
    assert not (staged.parent / 'foreign').exists()
    assert all(root.is_relative_to(scratch) for root in roots)
    assert not list(scratch.rglob('secret.txt'))


def test_completed_node_queue_drains_and_does_not_repeat_after_restart(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    _analysis(context.notes_root, 'N000', 'research', 1)
    calls = []

    def runner(prompt, roots, schema, scratch):
        trigger = json.loads(prompt[prompt.rfind('\n\n') + 2:])
        calls.append(trigger['trigger_node'])
        if len(calls) == 1:
            _analysis(context.notes_root, 'N001', 'research', 1)
            assert memory_due(context.notes_root).due
        refs = list_analysis_refs(roots[0])
        output = scratch / 'output.json'
        output.write_text(json.dumps({k: _summary(list(refs)) for k in
                                     ('research_summary', 'engineering_summary')}))
        return output

    consolidate_memory(if_due=True, context=context, runner=runner)
    assert calls == ['N000', 'N001']
    journal = json.loads((context.coral_dir / 'private/memory_state.json').read_text())
    assert all(row['status'] == 'completed' for row in journal['nodes'].values())
    assert consolidate_memory(if_due=True, context=MemoryContext(context.coral_dir),
                              runner=lambda *_: pytest.fail('duplicate')).status == 'not_due'


def test_engineering_ref_is_required_in_both_summaries(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    research = _analysis(context.notes_root, 'N001', 'research', 1)
    _analysis(context.notes_root, 'N001', 'engineering', 1)
    output = tmp_path / 'output.json'
    output.write_text(json.dumps({k: _summary([research]) for k in
                                 ('research_summary', 'engineering_summary')}))
    with pytest.raises(ValueError, match='trigger references'):
        consolidate_memory(if_due=True, context=context, runner=lambda *_: output)
    assert memory_due(context.notes_root).due


def test_published_output_is_reused_after_crash_before_queue_ack(tmp_path, monkeypatch):
    import autoresearch.memory as memory

    context = MemoryContext(tmp_path / '.coral')
    ref = _analysis(context.notes_root, 'N000', 'research', 1)
    calls = []
    ack = memory._record_memory_result

    def runner(prompt, roots, schema, scratch):
        calls.append(prompt)
        output = scratch / 'output.json'
        output.write_text(json.dumps({k: _summary([ref]) for k in
                                     ('research_summary', 'engineering_summary')}))
        return output

    monkeypatch.setattr(memory, '_record_memory_result', lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError('crash before ack')))
    with pytest.raises(RuntimeError, match='crash before ack'):
        consolidate_memory(if_due=True, context=context, runner=runner)
    monkeypatch.setattr(memory, '_record_memory_result', ack)
    consolidate_memory(if_due=True, context=context, runner=runner)
    assert len(calls) == 1
    assert not memory_due(context.notes_root).due


def test_new_summary_cannot_drop_previous_node_coverage(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    first = _analysis(context.notes_root, 'N000', 'research', 1)
    for name in ('research_summary.md', 'engineering_summary.md'):
        (context.notes_root / name).write_text(_summary([first]))
    assert not memory_due(context.notes_root).due
    second = _analysis(context.notes_root, 'N001', 'research', 1)
    output = tmp_path / 'output.json'
    output.write_text(json.dumps({k: _summary([second]) for k in
                                 ('research_summary', 'engineering_summary')}))
    with pytest.raises(ValueError, match='trigger references'):
        consolidate_memory(if_due=True, context=context, runner=lambda *_: output)


def test_prior_summary_refs_remain_readable_before_their_node_finishes(tmp_path):
    context = MemoryContext(tmp_path / '.coral')
    old = _analysis(context.notes_root, 'N001', 'research', 1)
    _complete(context.notes_root, 'N001', engineering='active')
    for name in ('research_summary.md', 'engineering_summary.md'):
        (context.notes_root / name).write_text(_summary([old]))
    root = _analysis(context.notes_root, 'N000', 'research', 1)

    def runner(prompt, roots, schema, scratch):
        assert (roots[0] / old).is_file()
        output = scratch / 'output.json'
        output.write_text(json.dumps({k: _summary([root, old]) for k in
                                     ('research_summary', 'engineering_summary')}))
        return output

    consolidate_memory(if_due=True, context=context, runner=runner)
    assert not memory_due(context.notes_root).due
    _complete(context.notes_root, 'N001')
    assert memory_due(context.notes_root).pending_nodes == ('N001',)
