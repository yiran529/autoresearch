from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from autoresearch.nodes import NodeError, create_node


def _parent(notes: Path) -> dict[str, object]:
    node = notes / "nodes" / "N001"
    node.mkdir(parents=True)
    analysis = node / "research_analysis_v2.md"
    analysis.write_text("analysis\n")
    return {
        "notes_root": notes,
        "parent_node": "N001",
        "parent_commit": "a" * 40,
        "base_commit": "b" * 40,
        "analysis_ref": "nodes/N001/research_analysis_v2.md",
    }


def test_child_freezes_selected_parent_versions(tmp_path):
    evidence = _parent(tmp_path / "notes")
    path = create_node(
        {
            "hypothesis": "Compress only stable gradients.",
            "selection_rationale": "The parent Analysis identified communication as the bottleneck.",
            "delta": "Gate compression by stability.",
            "risk": "The gate may reduce compression.",
            "falsification_criteria": "Reject if either quality workload exceeds the 10% gate.",
        },
        evidence,
    )

    spec = yaml.safe_load(path.read_text())
    assert spec["node_id"] == "N002"
    assert spec["parent_commit"] == "a" * 40
    assert spec["analysis_refs"]["parent_research"].endswith("research_analysis_v2.md")
    assert "review_refs" not in spec
    assert spec["selection_rationale"]


def test_child_requires_existing_parent_research_analysis(tmp_path):
    evidence = _parent(tmp_path / "notes")
    (tmp_path / "notes/nodes/N001/research_analysis_v2.md").unlink()
    with pytest.raises(NodeError, match="does not exist"):
        create_node(
            {
                "hypothesis": "h",
                "selection_rationale": "s",
                "delta": "d",
                "risk": "r",
                "falsification_criteria": "f",
            },
            evidence,
        )


def test_child_rejects_analysis_from_different_parent(tmp_path):
    evidence = _parent(tmp_path / "notes")
    evidence["analysis_ref"] = "nodes/N002/research_analysis_v2.md"
    with pytest.raises(NodeError, match="selected Node"):
        create_node(
            {
                "hypothesis": "h",
                "selection_rationale": "s",
                "delta": "d",
                "risk": "r",
                "falsification_criteria": "f",
            },
            evidence,
        )


def test_node_cli_publishes_spec_through_trusted_entry(tmp_path, monkeypatch):
    from autoresearch import cli

    from coral.types import Attempt

    notes = tmp_path / "notes"
    _parent(notes)
    research = Attempt(
        commit_hash="a" * 40,
        agent_id="researcher",
        title="research",
        score=1.0,
        status="improved",
        parent_hash=None,
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"mode": "research", "node_id": "N001"},
    )
    calls = []
    monkeypatch.setattr(cli, "_workflow_context", lambda: (tmp_path / ".coral", notes, None))
    monkeypatch.setattr(cli, "read_attempts", lambda *_args, **_kwargs: [research])
    monkeypatch.setattr(
        cli,
        "request_trusted",
        lambda _coral, operation, payload: (
            calls.append((operation, payload)) or {"path": "nodes/N002/spec.md"}
        ),
    )

    cli.main(
        [
            "node",
            "create",
            "--parent",
            "N001",
            "--analysis-version",
            "2",
            "--hypothesis",
            "h",
            "--selection-rationale",
            "s",
            "--delta",
            "d",
            "--risk",
            "r",
            "--falsification-criteria",
            "f",
        ]
    )

    assert calls[0][0] == "publish_artifact"
    assert calls[0][1]["kind"] == "spec"
    assert calls[0][1]["node_id"] == "N002"
