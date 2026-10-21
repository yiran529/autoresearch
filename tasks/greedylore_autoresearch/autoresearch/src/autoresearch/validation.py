"""Validation for versioned Analysis artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    pass


def validate_authored_references(document: dict[str, Any]) -> None:
    """Apply to new author submissions, never retroactively to frozen records."""
    for field in ('summary', 'findings', 'limitations'):
        values = document.get(field, [])
        for value in ([values] if isinstance(values, str) else values if isinstance(values, list) else []):
            if isinstance(value, str) and re.search(r'(?<![\w/])/(?:tmp|var/tmp)/\S+', value):
                raise ArtifactValidationError(
                    'Analysis cites a temporary file: archive supporting files with '
                    '`autoresearch evidence archive`, or cite managed diagnostic records. '
                    'Describe unavailable checks as limitations without temporary file references.'
                )


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

def load_document(path: str | Path) -> dict[str, Any]:
    from coral.grader.analysis_artifacts import load_analysis

    try:
        value = load_analysis(Path(path))
    except (OSError, ValueError) as exc:
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
    if set(document) not in (_ANALYSIS_FIELDS, _ANALYSIS_FIELDS | {"handover"}):
        raise ArtifactValidationError("Analysis has an invalid schema")
    _common(document, evidence)
    if not isinstance(document["summary"], str) or not document["summary"].strip():
        raise ArtifactValidationError("summary must be non-empty text")
    _string_list(document["findings"], "findings")
    _string_list(document["limitations"], "limitations")
    if "handover" in document:
        from .handover import validate_body

        if document["mode"] != "research":
            raise ArtifactValidationError("structured handover belongs to Research Analysis")
        try:
            validate_body(document["handover"])
        except ValueError as exc:
            raise ArtifactValidationError(str(exc)) from exc
        allowed = {"spec.md", "evaluation.md", *evidence.evidence_paths,
                   *(record.path for record in evidence.evidence_records if record.status == "present")}
        for ref in document["handover"]["evidence_refs"]:
            if ref not in allowed:
                raise ArtifactValidationError("handover evidence_ref is not in the frozen evidence")
    if document["mode"] == "research":
        import yaml

        spec = yaml.safe_load(Path(evidence.spec_path).read_text())
        if isinstance(spec, dict) and spec.get("research_intent") and "handover" not in document:
            raise ArtifactValidationError("Research Analysis must answer the frozen research_intent in handover")
    return document
