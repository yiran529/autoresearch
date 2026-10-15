"""Validation for versioned Analysis and Review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    pass


_COMMON_FIELDS = {
    "node_id",
    "mode",
    "research_sha",
    "engineering_attempts",
    "evaluation_version",
    "evidence_manifest",
    "evidence_manifest_hash",
    "evidence_paths",
    "evaluation_snapshot",
}
_ANALYSIS_FIELDS = _COMMON_FIELDS | {"summary", "findings", "limitations"}
_REVIEW_FIELDS = _COMMON_FIELDS | {
    "analysis_version",
    "workflow_checks",
    "evidence_checks",
    "issues",
    "actions",
    "decision",
}


def load_document(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid Artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError("Artifact must be a JSON object")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ArtifactValidationError(f"{field} must be a list of non-empty strings")
    return value


def evidence_envelope(evidence: Any) -> dict[str, Any]:
    return {
        "node_id": evidence.node_id,
        "mode": evidence.mode,
        "research_sha": evidence.research_sha,
        "engineering_attempts": [item.to_dict() for item in evidence.engineering_attempts],
        "evaluation_version": evidence.evaluation_version,
        "evidence_manifest": evidence.manifest(),
        "evidence_manifest_hash": evidence.manifest_hash,
        "evidence_paths": list(evidence.evidence_paths),
        "evaluation_snapshot": evidence.evaluation_snapshot,
    }


def seal_document(document: dict[str, Any], evidence: Any) -> dict[str, Any]:
    sealed = dict(document)
    sealed.update(evidence_envelope(evidence))
    return sealed


def _common(document: dict[str, Any], evidence: Any) -> None:
    for field, value in evidence_envelope(evidence).items():
        if document.get(field) != value:
            raise ArtifactValidationError(f"{field} does not match fixed evidence")


def validate_analysis(document: dict[str, Any], evidence: Any) -> dict[str, Any]:
    if set(document) != _ANALYSIS_FIELDS:
        raise ArtifactValidationError("Analysis has an invalid schema")
    _common(document, evidence)
    if not isinstance(document["summary"], str) or not document["summary"].strip():
        raise ArtifactValidationError("summary must be non-empty text")
    _string_list(document["findings"], "findings")
    _string_list(document["limitations"], "limitations")
    return document


def validate_review(
    document: dict[str, Any],
    evidence: Any,
    analysis_version: int,
) -> dict[str, Any]:
    if set(document) != _REVIEW_FIELDS:
        raise ArtifactValidationError("Review has an invalid schema")
    _common(document, evidence)
    if document.get("analysis_version") != analysis_version:
        raise ArtifactValidationError("Review analysis_version does not match")
    for field in ("workflow_checks", "evidence_checks"):
        _string_list(document[field], field)
    for field in ("issues", "actions"):
        if not isinstance(document[field], list) or not all(
            isinstance(item, str) for item in document[field]
        ):
            raise ArtifactValidationError(f"{field} must be a list of strings")
    if document.get("decision") not in {"pass", "warning", "fail"}:
        raise ArtifactValidationError("Review decision must be pass, warning, or fail")
    return document
