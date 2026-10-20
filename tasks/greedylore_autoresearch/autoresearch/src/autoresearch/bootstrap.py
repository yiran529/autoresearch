"""One-time Root N000 bootstrap from the pinned seed commit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .atomic import atomic_write_text

_SHA = re.compile(r"^[0-9a-f]{40}$")


class BootstrapError(ValueError):
    pass


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise BootstrapError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _load_manifest(source_manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source_manifest, Mapping):
        return dict(source_manifest)
    try:
        value = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError("source manifest is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise BootstrapError("source manifest is unavailable or invalid")
    return value


def _validate_manifest(repo: Path, manifest: dict[str, Any]) -> None:
    source = manifest.get("source_commit")
    if not isinstance(source, str) or not _SHA.fullmatch(source):
        raise BootstrapError("source manifest has no pinned source commit")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BootstrapError("source manifest has no files")
    additions = manifest.get('task_files', {})
    if not isinstance(additions, dict) or set(files) & set(additions):
        raise BootstrapError('source manifest task files are invalid')
    files = {**files, **additions}
    adaptations = manifest.get("adaptations", {})
    if not isinstance(adaptations, dict):
        raise BootstrapError("source manifest adaptations are invalid")
    for source_relative, record in files.items():
        if not isinstance(record, dict):
            raise BootstrapError(f"source manifest record is invalid: {source_relative}")
        relative = record.get("seed_path", source_relative)
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or '..' in Path(relative).parts):
            raise BootstrapError(f'source manifest file path is invalid: {source_relative}')
        adaptation = adaptations.get(source_relative)
        expected = (
            adaptation.get("seed_sha256") if isinstance(adaptation, dict) else record.get("sha256")
        )
        try:
            actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise BootstrapError(f"source manifest file is missing: {relative}") from exc
        if actual != expected:
            raise BootstrapError(f"source manifest mismatch: {relative}")


def _root_spec(seed_commit: str) -> dict[str, Any]:
    return {
        "node_id": "N000",
        "node_type": "seed",
        "parent_node": None,
        "parent_commit": None,
        "base_commit": seed_commit,
        "hypothesis": "Provided GreedyLoRE r32/EF14 baseline.",
        "selection_rationale": "Use the provided implementation as the fixed Root.",
        "delta": "None; Root records the provided implementation.",
        "risk": "The baseline may not satisfy every quality or efficiency target.",
        "falsification_criteria": "Formal Root evaluation defines the baseline evidence.",
        "analysis_refs": {"parent_research": None, "parent_engineering": None},
    }


def _publish_root(manifest: dict[str, Any], seed_commit: str) -> None:
    publisher = manifest.get("publish")
    if publisher is None:
        return
    if not callable(publisher):
        raise BootstrapError("publish must be callable")
    result = publisher(
        "spec",
        yaml.safe_dump(_root_spec(seed_commit), sort_keys=False),
        {"node_id": "N000", "mode": None, "version": None},
    )
    if not isinstance(result, dict) or result.get("path") != "nodes/N000/spec.md":
        raise BootstrapError("trusted publisher returned an unexpected Root target")


def bootstrap_root(
    repo: str | Path,
    source_manifest: str | Path | Mapping[str, Any],
) -> Path:
    """Create canonical A0 and publish N000; interrupted publication is retryable."""
    root = Path(repo).resolve()
    manifest = _load_manifest(source_manifest)
    _validate_manifest(root, manifest)
    submission_path = root / "submission.yaml"
    head = _git(root, "rev-parse", "HEAD")

    if submission_path.exists():
        try:
            declaration = yaml.safe_load(submission_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise BootstrapError("existing Root submission is invalid") from exc
        seed_commit = declaration.get("base_commit") if isinstance(declaration, dict) else None
        if (
            not isinstance(seed_commit, str)
            or not _SHA.fullmatch(seed_commit)
            or declaration.get("node_id") != "N000"
            or _git(root, "diff", "--name-only", seed_commit, head).splitlines()
            != ["submission.yaml"]
        ):
            raise BootstrapError("existing Root bootstrap conflicts with the run seed")
        _publish_root(manifest, seed_commit)
        return submission_path

    seed_commit = head
    if _git(root, "status", "--porcelain"):
        raise BootstrapError("Root bootstrap requires a clean checkout at S0")
    declaration = {
        "mode": "research",
        "node_id": "N000",
        "node_type": "seed",
        "base_commit": seed_commit,
        "quality_anchor": None,
        "analysis_refs": {"parent_research": None, "parent_engineering": None},
        "change_summary": "Bootstrap the provided GreedyLoRE implementation as Root.",
        "selection_rationale": "Use the provided implementation as the fixed research-tree root.",
        "semantic_self_check": {
            "status": "not_run",
            "commands": [],
            "limitations": "Root evaluation is performed by the formal grader.",
        },
    }
    atomic_write_text(submission_path, yaml.safe_dump(declaration, sort_keys=False))
    _git(root, "add", "submission.yaml")
    _git(root, "commit", "-m", "chore: bootstrap autoresearch root")
    if _git(root, "diff", "--name-only", seed_commit, "HEAD").splitlines() != ["submission.yaml"]:
        raise BootstrapError("Root bootstrap changed files other than submission.yaml")
    _publish_root(manifest, seed_commit)
    return submission_path


def bootstrap_root_analysis(
    coral_dir: Path,
    root_worktree: Path,
    *,
    runner: Any = None,
    code_snapshotter: Any = None,
) -> Path:
    """Complete fixed N000 public evidence on the host after its formal Attempt."""
    from coral.hub.attempts import read_attempts

    from .lifecycle import run_worker
    from .state import WorkflowStore

    coral_dir = Path(coral_dir).resolve()
    root_worktree = Path(root_worktree).resolve()
    policy_path = coral_dir / "private" / "root_bootstrap.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError("fixed Root bootstrap policy is unavailable") from exc
    if (
        not isinstance(policy, dict)
        or policy.get("version") != 1
        or policy.get("node_id") != "N000"
        or policy.get("mode") != "research"
        or not isinstance(policy.get("source_commit"), str)
        or not _SHA.fullmatch(policy["source_commit"])
        or not isinstance(policy.get("submission_hash"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", policy["submission_hash"])
    ):
        raise BootstrapError("fixed Root bootstrap policy is invalid")
    attempts = tuple(
        attempt for attempt in read_attempts(coral_dir)
        if attempt.metadata.get("node_id") == "N000"
        and attempt.metadata.get("mode") == "research"
        and attempt.budget_class == "real"
    )
    if (
        len(attempts) != 1
        or attempts[0].parent_hash != policy["source_commit"]
    ):
        raise BootstrapError("Root Analysis bootstrap requires one terminal N000 Research Attempt")
    if attempts[0].metadata.get("submission_hash") != policy["submission_hash"]:
        raise BootstrapError("Root Attempt conflicts with the fixed Root policy")

    node_root = coral_dir / "public" / "notes" / "nodes" / "N000"
    spec_path = node_root / "spec.md"
    expected_spec = _root_spec(policy["source_commit"])
    if spec_path.is_file():
        try:
            current_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise BootstrapError("existing Root Spec is invalid") from exc
        if (
            not isinstance(current_spec, dict)
            or current_spec.get("node_id") != "N000"
            or current_spec.get("base_commit") != policy["source_commit"]
            or current_spec.get("parent_node") is not None
        ):
            raise BootstrapError("existing Root Spec conflicts with fixed policy")
    else:
        atomic_write_text(spec_path, yaml.safe_dump(expected_spec, sort_keys=False), mode=0o644)

    store = WorkflowStore(coral_dir, repo_path=root_worktree)
    try:
        store._terminal_research_unlocked("N000")
    except ValueError as exc:
        raise BootstrapError(str(exc)) from exc
    options = {"checkout_root": root_worktree, "only_node": "N000"}
    if runner is not None:
        options["runner"] = runner
    if code_snapshotter is not None:
        options["code_snapshotter"] = code_snapshotter
    result = run_worker(str(coral_dir), **options)
    if result["status"] == "locked":
        raise BootstrapError("Analysis worker is already running; await the existing Root Analysis job")
    from coral.grader.trusted_writes import trusted_writes_lock
    with trusted_writes_lock(coral_dir):
        state = store._load_unlocked()
        job = state["nodes"]["N000"]["analysis"]
        if not job or job["status"] != "completed":
            raise BootstrapError(f"Root Analysis is not complete: {result['errors'] or (job or {}).get('last_error')}")
        return store.notes_root / job["analysis_ref"]
