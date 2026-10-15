"""Artifact-driven Analyzer and Reviewer workflow."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coral.hub._island import island_root
from coral.hub.attempts import read_attempts
from coral.workspace.breadcrumbs import find_coral_breadcrumb, read_island_breadcrumb

from .client import request_trusted
from .evaluation import refresh_node_evaluation
from .validation import (
    ArtifactValidationError,
    load_document,
    seal_document,
    validate_analysis,
    validate_review,
)

_NODE = re.compile(r"^N[0-9]{3}$")


@dataclass(frozen=True)
class EvidenceAttempt:
    commit_sha: str
    timestamp: str
    status: str
    run_ids: tuple[str, ...]
    eval_refs: tuple[str, ...]
    comparability: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp,
            "status": self.status,
            "run_ids": list(self.run_ids),
            "eval_refs": list(self.eval_refs),
            "comparability": self.comparability,
        }


@dataclass(frozen=True)
class EvidenceRecord:
    path: str
    status: str
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "status": self.status, "sha256": self.sha256}


@dataclass(frozen=True)
class EvidenceBundle:
    node_id: str
    mode: str
    research_sha: str
    contract_hash: str
    evaluation_version: str
    evaluation_snapshot: str
    spec_sha256: str
    spec_path: str
    evaluation_path: str
    engineering_attempts: tuple[EvidenceAttempt, ...]
    evidence_paths: tuple[str, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    manifest_hash: str

    def manifest(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mode": self.mode,
            "research_sha": self.research_sha,
            "contract_hash": self.contract_hash,
            "evaluation_version": self.evaluation_version,
            "spec_sha256": self.spec_sha256,
            "engineering_attempts": [item.to_dict() for item in self.engineering_attempts],
            "evidence_paths": list(self.evidence_paths),
            "evidence_records": [item.to_dict() for item in self.evidence_records],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest(),
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class AnalysisContext:
    coral_dir: Path
    notes_root: Path
    island_id: str | None
    attempts: tuple[Any, ...]
    log_root: Path
    checkout_root: Path


@dataclass(frozen=True)
class AnalysisResult:
    evidence: EvidenceBundle
    analysis_version: int
    review_version: int
    analysis_path: Path
    review_path: Path
    decision: str


Runner = Callable[[str, Sequence[Path], Path, Path], Path]
Publisher = Callable[[str, str, dict[str, Any]], dict[str, Any]]
CodeSnapshotter = Callable[[EvidenceBundle, AnalysisContext, Path], Path]


def _default_context() -> AnalysisContext:
    found = find_coral_breadcrumb(Path.cwd().resolve())
    if found is None:
        raise ValueError("No CORAL run found from the current directory")
    coral_dir, checkout_root = found
    island_id = read_island_breadcrumb(coral_dir, checkout_root)
    root = island_root(coral_dir, island_id)
    return AnalysisContext(
        coral_dir=coral_dir,
        notes_root=root / "notes",
        island_id=island_id,
        attempts=tuple(read_attempts(coral_dir, island_id=island_id)),
        log_root=root / "eval_logs",
        checkout_root=checkout_root,
    )


def _attempt_evidence(attempt: Any) -> EvidenceAttempt:
    runs = tuple(
        str(run["run_id"]) for run in attempt.metadata.get("grader_runs", []) if run.get("run_id")
    )
    references = {
        str(run["log_dir"]) for run in attempt.metadata.get("grader_runs", []) if run.get("log_dir")
    }
    references.update(
        value for value in attempt.metadata.get("evidence_paths", []) if isinstance(value, str)
    )
    return EvidenceAttempt(
        commit_sha=attempt.commit_hash,
        timestamp=attempt.timestamp,
        status=attempt.status,
        run_ids=runs,
        eval_refs=tuple(sorted(references)),
        comparability=str(
            attempt.metadata.get("comparability", attempt.metadata.get("comparable", "—"))
        ),
    )


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return _hash_bytes(encoded)


def _reference_path(reference: str, context: AnalysisContext) -> Path:
    prefix = "eval_logs/"
    relative = reference[len(prefix) :] if reference.startswith(prefix) else reference
    return context.log_root / relative


def _evidence_records(
    references: Sequence[str], context: AnalysisContext
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    for reference in sorted(set(references)):
        path = _reference_path(reference, context)
        if path.is_file():
            records.append(EvidenceRecord(reference, "present", _hash_bytes(path.read_bytes())))
            continue
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            if files:
                for item in files:
                    child = f"{reference.rstrip('/')}/{item.relative_to(path).as_posix()}"
                    records.append(EvidenceRecord(child, "present", _hash_bytes(item.read_bytes())))
                continue
        records.append(EvidenceRecord(reference, "missing", None))
    return tuple(records)


def resolve_evidence(
    node_id: str,
    mode: str,
    *,
    context: AnalysisContext | None = None,
) -> EvidenceBundle:
    if not _NODE.fullmatch(node_id):
        raise ValueError("invalid Node ID")
    if mode not in {"research", "engineering"}:
        raise ValueError("mode must be research or engineering")
    ctx = context or _default_context()
    node_root = ctx.notes_root / "nodes" / node_id
    spec_path = node_root / "spec.md"
    evaluation_path = node_root / "evaluation.md"
    try:
        spec_bytes = spec_path.read_bytes()
        evaluation_bytes = evaluation_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Node {node_id} is missing spec.md or evaluation.md") from exc

    try:
        spec = yaml.safe_load(spec_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"Node {node_id} spec is invalid") from exc
    if not isinstance(spec, dict) or spec.get("node_id") != node_id:
        raise ValueError(f"Node {node_id} spec identity does not match")

    node_attempts = [
        attempt
        for attempt in ctx.attempts
        if attempt.metadata.get("node_id") == node_id and attempt.budget_class == "real"
    ]
    research = [
        attempt
        for attempt in node_attempts
        if attempt.metadata.get("mode") == "research" and attempt.status != "pending"
    ]
    research_shas = {attempt.commit_hash for attempt in research}
    if len(research_shas) != 1:
        raise ValueError(f"Node {node_id} requires one terminal Research SHA")
    research_attempt = sorted(research, key=lambda item: item.timestamp)[-1]
    if mode == "research":
        evaluation_snapshot = refresh_node_evaluation(node_id, tuple(research), ctx.log_root)
        evaluation_bytes = evaluation_snapshot.encode()
    else:
        evaluation_snapshot = evaluation_bytes.decode("utf-8")
    engineering = (
        sorted(
            (
                attempt
                for attempt in node_attempts
                if attempt.metadata.get("mode") == "engineering" and attempt.status != "pending"
            ),
            key=lambda item: (item.timestamp, item.commit_hash),
        )
        if mode == "engineering"
        else []
    )
    engineering_evidence = tuple(_attempt_evidence(item) for item in engineering)
    research_evidence = _attempt_evidence(research_attempt)
    raw_references = set(research_evidence.eval_refs)
    for item in engineering_evidence:
        raw_references.update(item.eval_refs)
    evidence_paths = {
        "evidence/spec.md",
        "evidence/evaluation.md",
        *(f"evidence/{reference}" for reference in raw_references),
    }
    records = _evidence_records(tuple(raw_references), ctx)
    values = {
        "node_id": node_id,
        "mode": mode,
        "research_sha": research_attempt.commit_hash,
        "contract_hash": str(research_attempt.metadata.get("contract_hash", "—")),
        "evaluation_version": _hash_bytes(evaluation_bytes),
        "spec_sha256": _hash_bytes(spec_bytes),
        "engineering_attempts": [item.to_dict() for item in engineering_evidence],
        "evidence_paths": sorted(evidence_paths),
        "evidence_records": [item.to_dict() for item in records],
    }
    return EvidenceBundle(
        node_id=node_id,
        mode=mode,
        research_sha=research_attempt.commit_hash,
        contract_hash=values["contract_hash"],
        evaluation_version=values["evaluation_version"],
        evaluation_snapshot=evaluation_snapshot,
        spec_sha256=values["spec_sha256"],
        spec_path=str(spec_path),
        evaluation_path=str(evaluation_path),
        engineering_attempts=engineering_evidence,
        evidence_paths=tuple(values["evidence_paths"]),
        evidence_records=records,
        manifest_hash=_manifest_hash(values),
    )


def run_fresh_codex(
    prompt: str,
    read_roots: Sequence[Path],
    output_schema: Path,
    scratch: Path,
) -> Path:
    """Run one non-persistent Codex session and return its final JSON document."""
    scratch.mkdir(parents=True, exist_ok=True)
    output = scratch / "output.json"
    roots = "\n".join(f"- {Path(root).resolve()}" for root in read_roots)
    fixed_prompt = (
        f"{prompt}\n\nRead only the following fixed evidence roots:\n{roots}\n"
        "Do not modify source code or run formal training."
    )
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
        str(scratch),
        "--output-schema",
        str(output_schema),
        "--output-last-message",
        str(output),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=fixed_prompt,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Codex session failed")
    if not output.is_file():
        raise RuntimeError("Codex session did not produce an output document")
    return output


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema(name: str) -> Path:
    return _package_root() / "schemas" / f"{name}.schema.json"


def _versions(node_root: Path, mode: str, kind: str) -> set[int]:
    pattern = re.compile(rf"^{mode}_{kind}_v([1-9][0-9]*)\.md$")
    return {
        int(match.group(1))
        for path in node_root.glob(f"{mode}_{kind}_v*.md")
        if (match := pattern.fullmatch(path.name))
    }


def _default_publisher(context: AnalysisContext) -> Publisher:
    def publish(kind: str, content: str, target: dict[str, Any]) -> dict[str, Any]:
        return request_trusted(
            context.coral_dir,
            "publish_artifact",
            {
                "kind": kind,
                "node_id": target["node_id"],
                "mode": target["mode"],
                "version": target["version"],
                "content": content,
            },
        )

    return publish


def _publish(
    publisher: Publisher,
    context: AnalysisContext,
    kind: str,
    mode: str,
    version: int,
    content: str,
) -> Path:
    result = publisher(
        kind,
        content,
        {"node_id": context_node(content), "mode": mode, "version": version},
    )
    relative = result.get("path") if isinstance(result, dict) else None
    expected = f"nodes/{context_node(content)}/{mode}_{kind}_v{version}.md"
    if relative != expected:
        raise ValueError("trusted publisher returned an unexpected Artifact target")
    return context.notes_root / relative


def context_node(content: str) -> str:
    value = json.loads(content)
    node_id = value.get("node_id")
    if not isinstance(node_id, str):
        raise ValueError("Artifact has no node_id")
    return node_id


def _artifact_evidence(
    document: dict[str, Any],
    current: EvidenceBundle,
    context: AnalysisContext,
) -> EvidenceBundle:
    """Recover and verify the immutable evidence snapshot stored in an Analysis."""
    manifest = document.get("evidence_manifest")
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("published Analysis has no evidence manifest")
    if document.get("evidence_manifest_hash") != _manifest_hash(manifest):
        raise ArtifactValidationError("published Analysis manifest hash does not match")
    for field in ("node_id", "mode", "research_sha", "contract_hash"):
        if manifest.get(field) != getattr(current, field):
            raise ArtifactValidationError(
                f"published Analysis {field} conflicts with current Node provenance"
            )
    if manifest.get("spec_sha256") != current.spec_sha256:
        raise ArtifactValidationError("published Analysis spec hash does not match")
    snapshot = document.get("evaluation_snapshot")
    if not isinstance(snapshot, str) or _hash_bytes(snapshot.encode()) != manifest.get(
        "evaluation_version"
    ):
        raise ArtifactValidationError("published Analysis evaluation snapshot does not match")
    try:
        engineering = tuple(
            EvidenceAttempt(
                commit_sha=item["commit_sha"],
                timestamp=item["timestamp"],
                status=item["status"],
                run_ids=tuple(item["run_ids"]),
                eval_refs=tuple(item["eval_refs"]),
                comparability=item["comparability"],
            )
            for item in manifest["engineering_attempts"]
        )
        records = tuple(
            EvidenceRecord(path=item["path"], status=item["status"], sha256=item["sha256"])
            for item in manifest["evidence_records"]
        )
        evidence_paths = tuple(manifest["evidence_paths"])
    except (KeyError, TypeError) as exc:
        raise ArtifactValidationError("published Analysis has invalid fixed evidence") from exc
    for record in records:
        if record.status not in {"present", "missing"}:
            raise ArtifactValidationError("published Analysis has invalid evidence status")
        if record.status == "present":
            path = _reference_path(record.path, context)
            if not path.is_file() or _hash_bytes(path.read_bytes()) != record.sha256:
                raise ArtifactValidationError(f"published Analysis evidence changed: {record.path}")
        elif record.sha256 is not None:
            raise ArtifactValidationError("missing evidence must not have a hash")
    recovered = EvidenceBundle(
        node_id=current.node_id,
        mode=current.mode,
        research_sha=current.research_sha,
        contract_hash=current.contract_hash,
        evaluation_version=manifest["evaluation_version"],
        evaluation_snapshot=snapshot,
        spec_sha256=current.spec_sha256,
        spec_path=current.spec_path,
        evaluation_path=current.evaluation_path,
        engineering_attempts=engineering,
        evidence_paths=evidence_paths,
        evidence_records=records,
        manifest_hash=document["evidence_manifest_hash"],
    )
    if recovered.manifest() != manifest:
        raise ArtifactValidationError("published Analysis evidence manifest is invalid")
    return recovered


def _prepare_evidence_snapshot(
    evidence: EvidenceBundle, context: AnalysisContext, scratch: Path
) -> Path:
    root = scratch / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    spec = Path(evidence.spec_path).read_bytes()
    if _hash_bytes(spec) != evidence.spec_sha256:
        raise ArtifactValidationError("Node spec changed before Analysis session")
    (root / "spec.md").write_bytes(spec)
    (root / "evaluation.md").write_text(evidence.evaluation_snapshot, encoding="utf-8")
    for record in evidence.evidence_records:
        if record.status != "present":
            continue
        source = _reference_path(record.path, context)
        if not source.is_file() or _hash_bytes(source.read_bytes()) != record.sha256:
            raise ArtifactValidationError(f"fixed evidence changed: {record.path}")
        target = root / record.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def _prepare_code_snapshot(
    evidence: EvidenceBundle, context: AnalysisContext, scratch: Path
) -> Path:
    root = scratch / "code"
    commits = [
        evidence.research_sha,
        *(item.commit_sha for item in evidence.engineering_attempts),
    ]
    for commit in dict.fromkeys(commits):
        target = root / commit
        target.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(context.checkout_root),
                "archive",
                "--format=tar",
                commit,
            ],
            capture_output=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode(errors="replace").strip()
            raise RuntimeError(message or f"cannot materialize code snapshot {commit}")
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            archive.extractall(target, filter="data")
    return root


def _prompt(
    role: str,
    evidence: EvidenceBundle,
    analysis: dict[str, Any] | None = None,
) -> str:
    prompt_name = "reviewer.md" if role == "reviewer" else f"{evidence.mode}_analysis.md"
    prompt_path = _task_root() / "prompts" / prompt_name
    role_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.is_file()
        else f"You are the independent {role}."
    )
    payload = {
        "role": role,
        "fixed_evidence": evidence.to_dict(),
        "analysis": analysis,
    }
    snapshot_rule = (
        "For review, use the published Analysis evaluation_snapshot and the supplied "
        "evidence/evaluation.md; do not substitute the mutable node evaluation.md. "
        if role == "reviewer"
        else ""
    )
    return (
        role_prompt + "\n\nProduce only the JSON document required by the supplied schema. "
        "Inspect code only in the supplied code snapshots named by fixed commit SHA. "
        "Every claim must cite the fixed evidence paths. "
        + snapshot_rule
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def analyze(
    node_id: str,
    mode: str,
    revise: bool = False,
    *,
    context: AnalysisContext | None = None,
    runner: Runner = run_fresh_codex,
    publisher: Publisher | None = None,
    code_snapshotter: CodeSnapshotter = _prepare_code_snapshot,
) -> AnalysisResult:
    ctx = context or _default_context()
    publish = publisher or _default_publisher(ctx)
    references = _task_root() / "references"
    evaluation_content = refresh_node_evaluation(node_id, ctx.attempts, ctx.log_root)
    evaluation_result = publish(
        "evaluation",
        evaluation_content,
        {"node_id": node_id, "mode": None, "version": None},
    )
    if (
        not isinstance(evaluation_result, dict)
        or evaluation_result.get("path") != f"nodes/{node_id}/evaluation.md"
    ):
        raise ValueError("trusted publisher returned an unexpected evaluation target")
    evidence = resolve_evidence(node_id, mode, context=ctx)
    node_root = ctx.notes_root / "nodes" / node_id
    analysis_versions = _versions(node_root, mode, "analysis")
    review_versions = _versions(node_root, mode, "review")
    if revise:
        version = max(analysis_versions | review_versions, default=0) + 1
    else:
        incomplete = sorted(analysis_versions - review_versions)
        if incomplete:
            version = incomplete[0]
        elif analysis_versions & review_versions:
            version = max(analysis_versions & review_versions)
        else:
            version = 1

    analysis_path = node_root / f"{mode}_analysis_v{version}.md"
    review_path = node_root / f"{mode}_review_v{version}.md"
    if analysis_path.exists() and not revise:
        analysis_document = load_document(analysis_path)
        evidence = _artifact_evidence(analysis_document, evidence, ctx)
        analysis_document = validate_analysis(analysis_document, evidence)
    else:
        analyzer_scratch = Path(tempfile.mkdtemp(prefix="autoresearch-analyzer-"))
        analyzer_snapshot = _prepare_evidence_snapshot(evidence, ctx, analyzer_scratch)
        analyzer_code = code_snapshotter(evidence, ctx, analyzer_scratch)
        analyzer_roots = [analyzer_code, analyzer_snapshot]
        if references.is_dir():
            analyzer_roots.append(references)
        output = runner(
            _prompt("analyzer", evidence),
            analyzer_roots,
            _schema("analysis"),
            analyzer_scratch,
        )
        analysis_document = validate_analysis(
            seal_document(load_document(output), evidence), evidence
        )
        content = json.dumps(analysis_document, indent=2, sort_keys=True) + "\n"
        analysis_path = _publish(publish, ctx, "analysis", mode, version, content)

    if review_path.exists() and not revise:
        review_document = validate_review(load_document(review_path), evidence, version)
    else:
        reviewer_scratch = Path(tempfile.mkdtemp(prefix="autoresearch-reviewer-"))
        reviewer_snapshot = _prepare_evidence_snapshot(evidence, ctx, reviewer_scratch)
        reviewer_code = code_snapshotter(evidence, ctx, reviewer_scratch)
        reviewer_roots = [reviewer_code, reviewer_snapshot]
        if references.is_dir():
            reviewer_roots.append(references)
        output = runner(
            _prompt("reviewer", evidence, analysis_document),
            reviewer_roots,
            _schema("review"),
            reviewer_scratch,
        )
        review_document = validate_review(
            seal_document(load_document(output), evidence), evidence, version
        )
        content = json.dumps(review_document, indent=2, sort_keys=True) + "\n"
        review_path = _publish(publish, ctx, "review", mode, version, content)

    return AnalysisResult(
        evidence=evidence,
        analysis_version=version,
        review_version=version,
        analysis_path=analysis_path,
        review_path=review_path,
        decision=str(review_document["decision"]),
    )
