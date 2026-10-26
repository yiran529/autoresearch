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


def staged_research_scope(node_id: str) -> str:
    """One staged protocol for startup messages and generated instruction files."""
    return (
        f"- You own Research Node `{node_id}`. Read its `research.json` and resume the saved iteration/stage.\n"
        "- Select under the planning lock; use `autoresearch research register` with one origin and first design. "
        "Keep Parent/base_commit, main question and core hypothesis fixed; append designs rather than rewriting history.\n"
        "- Normally use two and at most 3 candidate training trajectories on 130M, including failed/stopped/restarted ones. "
        "Exact same-SHA continuation stays in its trajectory; resumes and retries do not reset budgets.\n"
        "- Use managed training, timing and diagnostics; verify mechanism predictions and practical effects yourself. "
        "Use `autoresearch research timing` when efficiency evidence can change a decision; reuse matching results. "
        "Do not wait for an Analyzer between iterations. Diagnostic caps are 5 successful/15 total executions, 600 seconds each.\n"
        "- Finish when no justified refinement remains or the budget is exhausted. Use `autoresearch research finish` "
        "after all development work ends, nominating one completed 130M iteration or none. The host ensures eligible timing within its budget, freezes it and "
        "runs one conditional 60M transfer; do not change candidates or tune from its feedback.\n"
        "- After transfer completes or is explicitly skipped, the host retires you and starts independent Research Analysis "
        "over all Attempts and quality-gated Engineering. Their completion makes this Node an eligible Parent."
    )


def prompt_for_assignment(spec: AgentSpec, *, instruction_file: str, shared_dir: str,
                          resuming: bool, feedback: str | None = None, staged_research: bool = False) -> str:
    """Give finite workflow instructions in the actual initial/resumed user message."""
    if spec.workflow_role not in {"research", "engineering"} or not spec.node_id:
        raise ValueError("workflow prompt requires a role and a bound Node")
    action = "Resume" if resuming else "Start"
    scope = (
        "- Keep one method hypothesis, Parent/base_commit and Spec fixed; refine that method or meaningful "
        "parameters within at most 3 successful scored formal Research runs and 10 total formal Research runs, "
        "including failures, same-SHA infrastructure retries, archived Attempts and all resumes. "
        "A valid scored regression counts as success regardless of quality passing.\n"
        "- After the first terminal result, continue justified refinement within budget; do not stop merely "
        "because one result exists. Later candidates may descend from earlier candidates; core mechanism "
        "or hypothesis changes require a child Node.\n"
        "- Stop evaluating at either limit. To finish earlier when no justified refinement remains, use "
        "`autoresearch node complete --mode research` after all runs end. After explicit early completion "
        "or either cap, the host freezes Research once all runs end, selects its fixed result, and retires you; only then do independent Research "
        "Analysis over all Attempts and quality-gated Engineering start."
        if spec.workflow_role == "research" else
        "- Use at most 3 successful formal Engineering runs and 10 total formal Engineering runs, including "
        "failures, same-SHA retries and all resumes. A valid scored regression counts as success.\n"
        "- Stop evaluating at either limit. Otherwise, continue while evidence supports a clear, substantial end-to-end improvement opportunity; "
        "do not stop merely because one candidate improved.\n"
        "- Preserve algorithm semantics and author the final Engineering Analysis "
        "yourself.\n- Use the trusted Engineering finish command, then wait for publication and "
        "retirement; do not wait for Research Analysis before starting Engineering."
    )
    if spec.workflow_role == "research" and staged_research:
        scope = staged_research_scope(spec.node_id)
    if spec.workflow_role == 'engineering' and staged_research:
        scope += ('\n- Read the claim\'s frozen design source and preserve adaptive policy semantics. '
                  'Commit candidates and use `autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>` '
                  'for formal measurement; infrastructure retries add `--retry` with a new ID.')
    progress = "research.json, Spec, Evaluation/Attempts" if staged_research and spec.workflow_role == "research" else "Spec, Evaluation/Attempts"
    prompt = (
        f"{action} your host-assigned {spec.workflow_role} assignment for Node `{spec.node_id}`.\n\n"
        "## Read first\n\n"
        f"- Read `{instruction_file}` and your role at `{shared_dir}/roles/{spec.agent_id}.md` "
        "before acting, then follow that role's numbered steps and concrete work guidance.\n\n"
        "## Recover progress\n\n"
        f"- On start or resume, inspect the existing {progress} and Worktree first; "
        "continue from the recorded stage and preserve existing changes and frozen submissions.\n"
        "- Do not restart completed steps, choose another Node or reset the candidate count.\n\n"
        f"## Workflow limits and completion\n\n{scope}\n\n"
        "Work autonomously within these steps and limits."
    )
    if feedback:
        prompt += f"\n\nHost feedback for this same assignment:\n{feedback}"
    return prompt


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
                or status not in {"reserved", "active", "finalizing", "waiting"}
                or agent_id in seen
                or role == "research" and (not isinstance(research_mode, str) or research_mode not in {"REFINE", "EXPLORE"})
                or role == "engineering" and research_mode is not None
                or role == "engineering" and (status in {"active", "finalizing", "waiting"}) != (node_id is not None)
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
        staged_role = options.pop("staged_role_file", None)
        if assignment.role == "research" and config.grader.args.get("staged_research") is True and staged_role:
            options["role_file"] = staged_role
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
