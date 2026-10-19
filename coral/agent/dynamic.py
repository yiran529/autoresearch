"""Decode trusted task assignments into fixed CORAL role templates."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coral.agent.assignments import AgentSpec
from coral.agent.registry import default_model_for_runtime
from coral.config import CoralConfig
from coral.grader.trusted_writes import request_trusted

_RESEARCH = re.compile(r"^research-(N[0-9]{3})$")
_ENGINEER = re.compile(r"^engineer-E[0-9]{4}$")


@dataclass(frozen=True)
class DynamicAssignment:
    agent_id: str
    role: str
    node_id: str | None
    status: str
    research_mode: str | None = None


class DynamicAssignmentClient:
    def __init__(
        self,
        coral_dir: Path,
        *,
        request: Callable[[Path, str, dict[str, Any]], dict[str, Any]] = request_trusted,
    ) -> None:
        self.coral_dir = Path(coral_dir)
        self.request = request

    def reconcile(self) -> tuple[DynamicAssignment, ...]:
        response = self.request(self.coral_dir, "reconcile_workflow", {})
        raw = response.get("assignments") if isinstance(response, dict) else None
        if not isinstance(raw, list) or len(raw) > 100:
            raise ValueError("invalid dynamic assignment roster")
        assignments = []
        seen = set()
        for item in raw:
            fields = {"agent_id", "role", "node_id", "status"}
            if not isinstance(item, dict) or set(item) not in (fields, fields | {"research_mode"}):
                raise ValueError("invalid dynamic assignment entry")
            agent_id, role = item["agent_id"], item["role"]
            node_id, status = item["node_id"], item["status"]
            research_mode = item.get("research_mode", "REFINE") if role == "research" else item.get("research_mode")
            if role == "research":
                matched = _RESEARCH.fullmatch(agent_id) if isinstance(agent_id, str) else None
                valid = matched is not None and node_id == matched.group(1)
            elif role == "engineering":
                valid = (
                    isinstance(agent_id, str)
                    and _ENGINEER.fullmatch(agent_id) is not None
                    and (
                        node_id is None
                        or isinstance(node_id, str)
                        and _RESEARCH.fullmatch(f"research-{node_id}") is not None
                    )
                )
            else:
                valid = False
            if (
                not valid
                or status not in {"reserved", "active", "finalizing"}
                or agent_id in seen
                or role == "research" and (not isinstance(research_mode, str) or research_mode not in {"REFINE", "EXPLORE"})
                or role == "engineering" and research_mode is not None
                or role == "engineering" and (status in {"active", "finalizing"}) != (node_id is not None)
            ):
                raise ValueError("invalid dynamic assignment entry")
            assignments.append(DynamicAssignment(agent_id, role, node_id, status, research_mode))
            seen.add(agent_id)
        return tuple(assignments)


def spec_for_assignment(config: CoralConfig, assignment: DynamicAssignment) -> AgentSpec:
    """Use trusted identity only; runtime, role file and mounts come from config."""
    for index, template in enumerate(config.agents.assignments):
        if template.role != assignment.role:
            continue
        runtime = template.runtime or config.agents.runtime
        model = template.model
        if not model:
            if template.runtime and template.runtime != config.agents.runtime:
                model = default_model_for_runtime(template.runtime) or config.agents.model
            else:
                model = config.agents.model
        options = dict(config.agents.runtime_options)
        options.update(template.runtime_options)
        return AgentSpec(
            agent_id=assignment.agent_id,
            runtime=runtime,
            model=model,
            runtime_options=options,
            assignment_index=index,
            workflow_role=assignment.role,
            node_id=assignment.node_id,
            research_mode=assignment.research_mode,
        )
    raise ValueError("dynamic assignment has no configured role template")
