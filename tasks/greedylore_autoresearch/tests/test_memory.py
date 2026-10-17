"""Pure filename gate and Summary format checks for Global Memory."""

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


def _analysis(notes: Path, node: str, mode: str, version: int) -> str:
    relative = f"nodes/{node}/{mode}_analysis_v{version}.md"
    path = notes / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("published analysis\n")
    return relative


def _summary(refs: list[str]) -> str:
    entries = "\n".join(f"  - {ref}" for ref in refs)
    return f"---\nprocessed_analyses:\n{entries or '  []'}\n---\n# Summary\n\nA finding.\n"


def test_gate_counts_unprocessed_research_nodes_not_versions(tmp_path):
    notes = tmp_path / "notes"
    first = _analysis(notes, "N001", "research", 1)
    revised = _analysis(notes, "N001", "research", 2)
    second = _analysis(notes, "N002", "research", 1)
    _analysis(notes, "N003", "engineering", 1)
    assert not memory_due(notes).due
    third = _analysis(notes, "N003", "research", 1)
    gate = memory_due(notes)
    assert gate.due
    assert gate.trigger_research_refs == frozenset({first, revised, second, third})


def test_partial_summary_publication_keeps_gate_due(tmp_path):
    notes = tmp_path / "notes"
    first = _analysis(notes, "N001", "research", 1)
    (notes / "research_summary.md").write_text(_summary([first]))
    (notes / "engineering_summary.md").write_text(_summary([]))
    assert memory_due(notes, threshold=1).due


def test_summary_validates_only_canonical_existing_refs_and_required_trigger(tmp_path):
    notes = tmp_path / "notes"
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
    notes = tmp_path / "notes"
    first = _analysis(notes, "N001", "research", 1)
    _analysis(notes, "N001", "engineering", 1)
    _analysis(notes, "N0X1", "research", 1)
    assert list_analysis_refs(notes, mode="research") == (first,)
    assert read_processed_analyses(notes / "missing_summary.md") == frozenset()


def test_memory_worker_publishes_two_valid_summaries_once(tmp_path):
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
    assert [kind for kind, _ in published] == ["research_summary", "engineering_summary"]
    assert len(calls) == 1


def test_memory_runner_receives_only_staged_public_inputs(tmp_path):
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    refs = [_analysis(notes, f"N{index:03d}", "research", 1) for index in (1, 2, 3)]
    (notes / "tree.md").write_text("# Tree\n")
    private = coral_dir / "private"
    private.mkdir(parents=True)
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
    reference_paths = [task_root / "references" / name for name in (
        "algorithm_ref.md", "system_ref.md", "semantic_criteria.md",
    )]
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


def test_only_researcher_template_has_memory_startup_hook():
    task_root = Path(__file__).resolve().parents[1]
    task = yaml.safe_load((task_root / "task.yaml").read_text())
    templates = {item["role"]: item for item in task["agents"]["assignments"]}
    assert templates["research"]["runtime_options"]["startup_hook"] == [
        ".venv/bin/autoresearch", "consolidate-memory", "--if-due", "--async",
    ]
    assert "startup_hook" not in templates["engineering"]["runtime_options"]


@pytest.mark.parametrize("kind", ["research_summary", "engineering_summary"])
def test_agent_artifact_router_cannot_publish_host_summary(kind):
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        validate_artifact(kind, _summary([]), {
            "node_id": None, "mode": None, "version": None,
        })
