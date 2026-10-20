"""Deterministically rebuild Node evaluation summaries from formal records."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from coral.types import Attempt


class EvaluationConsistencyError(ValueError):
    pass


def _display(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _submission_identity(attempt: Attempt) -> str:
    # Canonical submission metadata survives retirement of the producer Worktree.
    if attempt.metadata.get("submission_hash"):
        mode, node_id = attempt.metadata.get("mode"), attempt.metadata.get("node_id")
        if isinstance(mode, str) and isinstance(node_id, str):
            return f"{mode}:{node_id}"
    worktree = attempt.metadata.get("source_worktree")
    if not isinstance(worktree, str):
        return "—"
    try:
        result = subprocess.run(
            ["git", "show", f"{attempt.commit_hash}:submission.yaml"],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "—"
    if result.returncode != 0:
        return "—"
    try:
        declaration = yaml.safe_load(result.stdout)
    except yaml.YAMLError:
        return "—"
    if not isinstance(declaration, dict):
        return "—"
    mode = declaration.get("mode")
    node_id = declaration.get("node_id")
    if not isinstance(mode, str) or not isinstance(node_id, str):
        return "—"
    return f"{mode}:{node_id}"


def _result_links(log_root: Path, attempt: Attempt) -> str:
    """Small reading view; the Attempt retains the complete evidence inventory."""
    public = log_root.parent
    references = [f"attempts/{attempt.commit_hash}.json"]
    for run in attempt.metadata.get("grader_runs", []):
        run_id = run.get("run_id")
        if isinstance(run_id, str):
            candidate = log_root / attempt.commit_hash / run_id
            if candidate.is_dir() and candidate.resolve().is_relative_to(log_root.resolve()):
                references.append(candidate.relative_to(public).as_posix())
    return "; ".join(
        f"[{reference}](../../../{quote(reference, safe='/')})"
        for reference in dict.fromkeys(references) if (public / reference).exists()
    ) or "—"


def refresh_node_evaluation(
    node_id: str,
    attempts: Iterable[Attempt],
    log_root: str | Path,
    *,
    coral_dir: str | Path | None = None,
) -> str:
    """Return canonical Markdown; the caller publishes it as evaluation.md."""
    selected = [attempt for attempt in attempts if attempt.metadata.get("node_id") == node_id]
    research = [attempt for attempt in selected
                if attempt.metadata.get("mode") == "research"
                and attempt.budget_class in {"real", "grader_error"}]
    anchor = None
    root = Path(log_root)
    if coral_dir is None:
        coral_dir = next((parent for parent in root.parents
                          if (parent / "private/autoresearch_state.json").is_file()), None)
    frozen_sha = None
    if coral_dir is not None:
        state_path = Path(coral_dir) / "private/autoresearch_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            frozen_sha = state.get("nodes", {}).get(node_id, {}).get("research", {}).get("selected_sha")
    if frozen_sha and research:
        from .research import frozen_research_attempt

        try:
            anchor = frozen_research_attempt(Path(coral_dir), node_id, research).commit_hash
        except ValueError as exc:
            raise EvaluationConsistencyError(str(exc)) from exc
    ordered = sorted(selected, key=lambda item: (item.timestamp, item.commit_hash))
    lines = [
        f"# Evaluation {node_id}",
        "",
        "| mode | commit | submission | status | score | contract | comparable | run IDs | evidence |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    if len(research) > 1 or frozen_sha:
        selection = f"Selected Research SHA: `{anchor}`" if anchor else "Selected Research SHA: pending Research finish."
        lines[2:2] = [selection, ""]
    rendered = 0
    for attempt in ordered:
        mode = attempt.metadata.get("mode")
        if mode not in {"research", "engineering"}:
            continue
        runs = attempt.metadata.get("grader_runs", [])
        run_summaries = []
        for run in runs:
            run_id = run.get("run_id")
            if not run_id:
                continue
            status = _display(run.get("status"))
            failure = run.get("failure_class")
            suffix = f"/{failure}" if failure else ""
            run_summaries.append(f"{run_id}:{status}{suffix}")
        run_ids = ", ".join(run_summaries) or "—"
        evidence = _result_links(root, attempt)
        comparable = attempt.metadata.get("comparability", attempt.metadata.get("comparable"))
        lines.append(
            "| "
            + " | ".join(
                (
                    mode,
                    attempt.commit_hash,
                    _submission_identity(attempt),
                    attempt.status,
                    _display(attempt.score),
                    _display(attempt.metadata.get("contract_hash")),
                    _display(comparable),
                    run_ids,
                    evidence,
                )
            )
            + " |"
        )
        rendered += 1
    if rendered == 0:
        lines.extend(("", "No formal evaluations recorded."))
    lines.extend(("", "Evidence labels are relative to the public state directory; links resolve from this Node. "
                  "Open the Attempt JSON for the complete evidence inventory and the run directory for raw results. "
                  "Analysis JSON records preserve their frozen evidence manifests."))
    return "\n".join(lines) + "\n"
