"""Minimal, artifact-driven Research Node creation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .atomic import atomic_write_text

_NODE = re.compile(r"^N([0-9]{3})$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ANALYSIS = re.compile(r"^nodes/(N[0-9]{3})/research_analysis_v([1-9][0-9]*)\.md$")


class NodeError(ValueError):
    pass


def _required_text(values: Mapping[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise NodeError(f"{field} must be non-empty text")
    return value.strip()


def _existing_reference(
    notes_root: Path,
    value: Any,
    pattern: re.Pattern[str],
    parent_node: str,
) -> tuple[str, int]:
    if not isinstance(value, str):
        raise NodeError("invalid parent artifact reference")
    match = pattern.fullmatch(value)
    if match is None or match.group(1) != parent_node:
        raise NodeError("parent artifact reference does not match the selected Node")
    path = notes_root / value
    if not path.is_file():
        raise NodeError(f"referenced artifact does not exist: {value}")
    return value, int(match.group(2))


def _next_node_id(notes_root: Path) -> str:
    nodes = notes_root / "nodes"
    numbers: list[int] = []
    if nodes.is_dir():
        for path in nodes.iterdir():
            match = _NODE.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    if number > 999:
        raise NodeError("Node ID space is exhausted")
    return f"N{number:03d}"


def create_node(request: Mapping[str, Any], evidence: Mapping[str, Any]) -> Path:
    """Create one child spec with frozen parent evidence references."""
    notes_root = Path(evidence["notes_root"])
    parent_node = evidence.get("parent_node")
    if not isinstance(parent_node, str) or not _NODE.fullmatch(parent_node):
        raise NodeError("invalid parent Node")
    parent_commit = evidence.get("parent_commit")
    base_commit = evidence.get("base_commit")
    if not isinstance(parent_commit, str) or not _SHA.fullmatch(parent_commit):
        raise NodeError("invalid parent commit")
    if not isinstance(base_commit, str) or not _SHA.fullmatch(base_commit):
        raise NodeError("invalid base commit")
    analysis_ref, _ = _existing_reference(
        notes_root, evidence.get("analysis_ref"), _ANALYSIS, parent_node
    )

    requested_id = request.get("node_id")
    node_id = _next_node_id(notes_root) if requested_id is None else str(requested_id)
    if not _NODE.fullmatch(node_id) or node_id == "N000":
        raise NodeError("invalid child Node ID")
    spec = {
        "node_id": node_id,
        "parent_node": parent_node,
        "parent_commit": parent_commit,
        "base_commit": base_commit,
        "hypothesis": _required_text(request, "hypothesis"),
        "selection_rationale": _required_text(request, "selection_rationale"),
        "delta": _required_text(request, "delta"),
        "risk": _required_text(request, "risk"),
        "falsification_criteria": _required_text(request, "falsification_criteria"),
        "analysis_refs": {
            "parent_research": analysis_ref,
            "parent_engineering": evidence.get("engineering_analysis_ref"),
        },
    }
    content = yaml.safe_dump(spec, sort_keys=False)
    publisher = evidence.get("publish")
    if publisher is not None:
        if not callable(publisher):
            raise NodeError("publish must be callable")
        result = publisher("spec", content, {"node_id": node_id, "mode": None, "version": None})
        relative = result.get("path") if isinstance(result, dict) else None
        if relative != f"nodes/{node_id}/spec.md":
            raise NodeError("trusted publisher returned an unexpected target")
        return notes_root / relative
    target = notes_root / "nodes" / node_id / "spec.md"
    if target.exists():
        raise NodeError(f"Node already exists: {node_id}")
    return atomic_write_text(target, content)
