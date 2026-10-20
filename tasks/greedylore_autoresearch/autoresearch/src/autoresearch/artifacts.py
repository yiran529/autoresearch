"""Task-specific canonical Artifact target validation."""

from __future__ import annotations

import re
from typing import Any

_NODE = re.compile(r"^N[0-9]{3}$")
_MODES = {"research", "engineering"}


def validate_artifact(kind: str, content: str, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("artifact content must be non-empty text")
    if kind in {"research_summary", "engineering_summary"}:
        raise ValueError(f"unsupported artifact kind for Agent publication: {kind}")
    node_id = context.get("node_id")
    mode = context.get("mode")
    version = context.get("version")
    if kind == "tree":
        if node_id is not None or mode is not None or version is not None:
            raise ValueError("tree does not accept node, mode, or version")
        relative = "tree.md"
    else:
        if not isinstance(node_id, str) or not _NODE.fullmatch(node_id):
            raise ValueError("invalid node_id")
        if kind in {"spec", "evaluation"}:
            if mode is not None or version is not None:
                raise ValueError(f"{kind} does not accept mode or version")
            relative = f"nodes/{node_id}/{kind}.md"
        elif kind == "analysis":
            if mode not in _MODES or not isinstance(version, int) or version < 1:
                raise ValueError("analysis requires a valid mode and positive version")
            relative = f"nodes/{node_id}/{mode}_{kind}_v{version}.md"
        else:
            raise ValueError(f"unsupported artifact kind: {kind}")
    result = {"relative_path": relative}
    if kind == "analysis":
        result["format"] = "analysis-json-markdown"
    return result
