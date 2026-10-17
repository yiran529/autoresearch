"""Deterministically rebuild Node evaluation summaries from formal records."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def _raw_result_paths(log_root: Path, attempt: Attempt) -> list[str]:
    paths: set[str] = set()
    for run in attempt.metadata.get("grader_runs", []):
        run_id = run.get("run_id")
        if isinstance(run_id, str):
            candidate = log_root / attempt.commit_hash / run_id
            if candidate.is_dir():
                for path in candidate.rglob("*.json"):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        paths.add(str(path.relative_to(log_root)))
    for value in attempt.metadata.get("evidence_paths", []):
        if isinstance(value, str):
            paths.add(value)
    return sorted(paths)


def refresh_node_evaluation(
    node_id: str,
    attempts: Iterable[Attempt],
    log_root: str | Path,
) -> str:
    """Return canonical Markdown; the caller publishes it as evaluation.md."""
    selected = [attempt for attempt in attempts if attempt.metadata.get("node_id") == node_id]
    research_shas = {
        attempt.commit_hash
        for attempt in selected
        if attempt.metadata.get("mode") == "research" and attempt.budget_class == "real"
    }
    if len(research_shas) > 1:
        raise EvaluationConsistencyError(f"Node {node_id} has multiple formal Research SHAs")
    ordered = sorted(selected, key=lambda item: (item.timestamp, item.commit_hash))
    lines = [
        f"# Evaluation {node_id}",
        "",
        "| mode | commit | submission | status | score | contract | comparable | run IDs | evidence |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    root = Path(log_root)
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
        evidence = ", ".join(_raw_result_paths(root, attempt)) or "—"
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
    return "\n".join(lines) + "\n"
