"""Validate committed GreedyLoRE AutoResearch submissions."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

_SHA = re.compile(r"^[0-9a-f]{40}$")
_NODE = re.compile(r"^N[0-9]{3}$")
_EVIDENCE_REF = re.compile(r"^nodes/N[0-9]{3}/(?:research|engineering)_analysis_v[1-9][0-9]*\.md$")
_FIELDS = {
    "mode",
    "node_id",
    "node_type",
    "base_commit",
    "quality_anchor",
    "analysis_refs",
    "change_summary",
    "selection_rationale",
    "semantic_self_check",
}


class SubmissionError(ValueError):
    """The committed submission does not satisfy the task contract."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise SubmissionError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise SubmissionError(message or f"git {' '.join(args)} failed")
    return result.stdout


def _full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise SubmissionError(f"{field} must be a full lowercase Git SHA")
    return value


def _attempt_payloads(coral_dir: Path, island_id: str | int | None) -> list[dict[str, Any]]:
    if island_id is None:
        roots = [coral_dir / "public" / "attempts"]
    else:
        roots = [coral_dir / "islands" / str(island_id) / "attempts"]
    payloads: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and not value.get("archived"):
                payloads.append(value)
    return payloads


def _notes_root(context: dict[str, Any]) -> Path:
    coral_dir = Path(context["coral_dir"]).resolve()
    island_id = context.get("island_id")
    if island_id is None:
        return coral_dir / "public" / "notes"
    return coral_dir / "islands" / str(island_id) / "notes"


def _safe_artifact(notes_root: Path, relative: str) -> Path:
    if not _EVIDENCE_REF.fullmatch(relative):
        raise SubmissionError("Node spec contains an invalid Artifact reference")
    candidate = notes_root / relative
    try:
        resolved_root = notes_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SubmissionError("referenced Artifact does not exist") from exc
    if not resolved.is_relative_to(resolved_root) or candidate.is_symlink():
        raise SubmissionError("referenced Artifact escapes the canonical notes tree")
    current = candidate.parent
    while current != notes_root:
        if current.is_symlink():
            raise SubmissionError("referenced Artifact traverses a symlink")
        current = current.parent
    if not candidate.is_file():
        raise SubmissionError("referenced Artifact does not exist")
    return candidate


def _validate_node_provenance(declaration: dict[str, Any], context: dict[str, Any]) -> None:
    notes_root = _notes_root(context)
    spec_path = notes_root / "nodes" / declaration["node_id"] / "spec.md"
    try:
        resolved_root = notes_root.resolve(strict=True)
        resolved_spec = spec_path.resolve(strict=True)
        if not resolved_spec.is_relative_to(resolved_root) or spec_path.is_symlink():
            raise OSError("spec escapes the canonical notes tree")
        current = spec_path.parent
        while current != notes_root:
            if current.is_symlink():
                raise OSError("spec traverses a symlink")
            current = current.parent
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SubmissionError("canonical Node spec is unavailable or invalid") from exc
    required = {"node_id", "base_commit", "parent_commit", "analysis_refs"}
    if not isinstance(spec, dict) or not required.issubset(spec):
        raise SubmissionError("canonical Node spec is unavailable or invalid")
    if spec["node_id"] != declaration["node_id"]:
        raise SubmissionError("Node spec node_id does not match submission")
    if spec["base_commit"] != declaration["base_commit"]:
        raise SubmissionError("Node spec base_commit does not match submission")
    if spec["parent_commit"] != context.get("parent_hash"):
        raise SubmissionError("Node spec parent_commit does not match candidate parent")
    if spec["analysis_refs"] != declaration["analysis_refs"]:
        raise SubmissionError("Node spec analysis_refs do not match submission")
    for group in (spec["analysis_refs"],):
        if not isinstance(group, dict) or set(group) != {
            "parent_research",
            "parent_engineering",
        }:
            raise SubmissionError("Node spec evidence references have an invalid schema")
        for reference in group.values():
            if reference is not None:
                if not isinstance(reference, str):
                    raise SubmissionError("Node spec evidence reference must be a string")
                _safe_artifact(notes_root, reference)


def validate_submission(commit_hash: str, context: dict[str, Any]) -> dict[str, Any]:
    """Read submission.yaml from *commit_hash* and return canonical metadata."""
    commit_hash = _full_sha(commit_hash, "commit_hash")
    repo = Path(context["worktree"]).resolve()
    raw_bytes = _git_bytes(repo, "show", f"{commit_hash}:submission.yaml")
    try:
        raw = raw_bytes.decode("utf-8")
        declaration = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SubmissionError("submission.yaml is not valid YAML") from exc
    if not isinstance(declaration, dict) or set(declaration) != _FIELDS:
        raise SubmissionError("submission.yaml has an invalid schema")

    mode = declaration["mode"]
    if mode not in {"research", "engineering"}:
        raise SubmissionError("mode must be research or engineering")
    node_id = declaration["node_id"]
    if not isinstance(node_id, str) or not _NODE.fullmatch(node_id):
        raise SubmissionError("invalid node_id")
    node_type = declaration["node_type"]
    if node_type not in {"research", "seed"}:
        raise SubmissionError("invalid node_type")
    if (node_id == "N000") != (node_type == "seed"):
        raise SubmissionError("only N000 may use node_type=seed")

    base = _full_sha(declaration["base_commit"], "base_commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, commit_hash],
        cwd=repo,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise SubmissionError("base_commit is not an ancestor of the candidate")

    refs = declaration["analysis_refs"]
    if not isinstance(refs, dict) or set(refs) != {"parent_research", "parent_engineering"}:
        raise SubmissionError("analysis_refs has an invalid schema")
    if any(value is not None and not isinstance(value, str) for value in refs.values()):
        raise SubmissionError("analysis_refs values must be strings or null")
    for field in ("change_summary", "selection_rationale"):
        value = declaration[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise SubmissionError(f"invalid {field}")
    semantic = declaration["semantic_self_check"]
    if not isinstance(semantic, dict) or set(semantic) != {"status", "commands", "limitations"}:
        raise SubmissionError("semantic_self_check has an invalid schema")
    if semantic["status"] not in {"not_run", "pass", "warning", "fail"}:
        raise SubmissionError("invalid semantic_self_check status")
    if not isinstance(semantic["commands"], list) or not all(
        isinstance(command, str) for command in semantic["commands"]
    ):
        raise SubmissionError("semantic_self_check commands must be strings")
    if not isinstance(semantic["limitations"], str):
        raise SubmissionError("semantic_self_check limitations must be a string")

    anchor = declaration["quality_anchor"]
    if mode == "research":
        if anchor is not None:
            raise SubmissionError("research quality_anchor must be null")
    else:
        anchor = _full_sha(anchor, "quality_anchor")
        coral_dir = Path(context["coral_dir"])
        matches = [
            attempt
            for attempt in _attempt_payloads(coral_dir, context.get("island_id"))
            if attempt.get("commit_hash") == anchor
            and attempt.get("status") not in {"pending", "crashed", "timeout"}
            and attempt.get("metadata", {}).get("mode") == "research"
            and attempt.get("metadata", {}).get("node_id") == node_id
        ]
        if len(matches) != 1:
            raise SubmissionError("quality_anchor is not the unique terminal Research SHA")

    if node_type == "seed":
        changed = _git(repo, "diff", "--name-only", base, commit_hash).splitlines()
        if changed != ["submission.yaml"]:
            raise SubmissionError("Root diff may only add submission.yaml")
    else:
        _validate_node_provenance(declaration, context)

    contract_path = Path(context["contract_path"])
    try:
        contract_bytes = contract_path.read_bytes()
        contract = yaml.safe_load(contract_bytes)
        max_retries = contract["recovery"]["max_infra_retries"]
        from greedylore_autoresearch_grader.contract import canonical_contract_hash

        contract_hash = canonical_contract_hash(contract_path)
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise SubmissionError("protected evaluation contract is unavailable or invalid") from exc
    if not isinstance(max_retries, int) or max_retries < 0:
        raise SubmissionError("max_infra_retries must be a non-negative integer")

    return {
        "mode": mode,
        "node_id": node_id,
        "node_type": node_type,
        "base_commit": base,
        "quality_anchor": anchor,
        "contract_hash": contract_hash,
        "max_infra_retries": max_retries,
        "submission_hash": hashlib.sha256(raw_bytes).hexdigest(),
    }
