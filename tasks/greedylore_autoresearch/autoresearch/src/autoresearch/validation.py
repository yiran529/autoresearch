"""Validation for versioned Analysis artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote


class ArtifactValidationError(ValueError):
    pass


def validate_authored_references(document: dict[str, Any], *, engineering: bool = False) -> None:
    """Reject explicit local work-file citations in new author submissions."""
    def local_reference(ref: str) -> bool:
        ref = unquote(ref.strip().strip('`<>').rstrip('.,')).split('#', 1)[0].split('?', 1)[0]
        while ref.startswith('./'):
            ref = ref[2:]
        ref = ref.removeprefix('/workspace/')
        return (ref in {'prompt.txt', 'output.json', 'execution.log'}
                or bool(re.match(r'^(?:results|experiments)/', ref)))

    for field in ('summary', 'findings', 'limitations'):
        values = document.get(field, [])
        for value in ([values] if isinstance(values, str) else values if isinstance(values, list) else []):
            if not isinstance(value, str):
                continue
            temporary = re.search(r'(?<![\w/])/(?:tmp|var/tmp)/[^\s)]+', value)
            references = re.findall(r'\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)', value)
            for clause in re.findall(r'(?i)\b(?:evidence|sources):([^\n]*)', value):
                for part in re.split(r'[,;]|\band\b', clause, flags=re.IGNORECASE):
                    words = part.strip().split()
                    if not words:
                        continue
                    # A direct availability statement describes an absence, not a citation.
                    if re.match(r'(?i)^(?:is|was|were) (?:not archived|missing|unavailable)\b',
                                ' '.join(words[1:])):
                        continue
                    references.append(words[0])
            bad = temporary.group() if temporary else next(
                (ref for ref in references if local_reference(ref)), None)
            if bad:
                remedy = (' or use `autoresearch evidence archive`' if engineering else '')
                raise ArtifactValidationError(
                    f'Analysis cites a temporary or unarchived work file: {unquote(bad)}. '
                    'Generate supporting files inside managed diagnostic results/ and cite '
                    f'the returned persistent record_ref{remedy}; describe unavailable evidence '
                    'without citing a local file.'
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
                   *(ref.removeprefix('evidence/') for ref in evidence.evidence_paths),
                   *(record.path for record in evidence.evidence_records if record.status == "present")}
        for ref in document["handover"]["evidence_refs"]:
            if ref not in allowed:
                raise ArtifactValidationError(f"handover evidence_ref is not in the frozen evidence: {ref}")
    if document["mode"] == "research":
        import yaml

        spec = yaml.safe_load(Path(evidence.spec_path).read_text())
        if isinstance(spec, dict) and spec.get("research_intent") and "handover" not in document:
            raise ArtifactValidationError("Research Analysis must answer the frozen research_intent in handover")
    return document
