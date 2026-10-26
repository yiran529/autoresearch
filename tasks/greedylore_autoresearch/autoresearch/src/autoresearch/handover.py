"""Small, host-derived problem handovers; source Analyses remain authoritative."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .atomic import atomic_write_text

VERDICTS = {"solved", "partially_solved", "not_solved", "inconclusive"}
_INTENT_FIELDS = {"problem_ref", "question", "mechanism_test", "practical_test",
                  "handover_response", "engineering_review"}
_BODY_FIELDS = {"mechanism_verdict", "practical_verdict", "evidence_refs",
                "new_information", "follow_up_questions"}


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(f"handover {field} must be nonempty text of at most 4000 characters")
    return value


def _texts(value, field):
    if not isinstance(value, list):
        raise ValueError(f"handover {field} must be a list")
    for item in value:
        _text(item, field)


def validate_body(body):
    if not isinstance(body, dict) or set(body) != _BODY_FIELDS:
        raise ValueError("handover has an invalid schema")
    for field in ("mechanism_verdict", "practical_verdict"):
        if not isinstance(body[field], str) or body[field] not in VERDICTS:
            raise ValueError(f"handover {field} is invalid")
    _texts(body["evidence_refs"], "evidence_refs")
    for ref in body["evidence_refs"]:
        if Path(ref).is_absolute() or ".." in Path(ref).parts:
            raise ValueError("handover evidence must use persistent evidence-relative paths")
    _texts(body["new_information"], "new_information")
    questions = body["follow_up_questions"]
    if not isinstance(questions, list) or len(questions) > 2:
        raise ValueError("handover follow_up_questions must contain at most two questions")
    seen = set()
    for question in questions:
        if not isinstance(question, dict) or set(question) != {"id", "question", "importance", "discriminator"}:
            raise ValueError("handover follow_up_questions has an invalid schema")
        if not isinstance(question["id"], str) or question["id"] not in {"Q1", "Q2"} or question["id"] in seen:
            raise ValueError("handover follow_up_questions requires unique Q1/Q2 IDs")
        seen.add(question["id"])
        for key in ("question", "importance", "discriminator"):
            _text(question[key], key)
    if not body["evidence_refs"] and any(body[k] != "inconclusive" for k in
                                         ("mechanism_verdict", "practical_verdict")):
        raise ValueError("handover verdicts require evidence_refs")
    return body


def validate_intent(value, parent, mode, *, allow_property_start=False):
    if not isinstance(value, dict) or set(value) != _INTENT_FIELDS:
        raise ValueError("research_intent has an invalid schema")
    for field in _INTENT_FIELDS - {"problem_ref"}:
        _text(value[field], field)
    reference = value["problem_ref"]
    available = {f"{parent['node_id']}:current"}
    available.update(f"{parent['node_id']}:{q['id']}" for q in parent.get("follow_up_questions", []))
    if reference is not None and (not isinstance(reference, str) or reference not in available):
        raise ValueError("research_intent problem_ref does not identify a parent question")
    if mode == "REFINE" and reference is None and not parent.get("legacy", True) and not allow_property_start:
        raise ValueError("REFINE research_intent requires a parent problem_ref")
    return value


def _safe_file(notes, relative):
    path = notes / relative
    if path.is_symlink() or not path.resolve().is_relative_to(notes.resolve()):
        raise ValueError("handover path escapes canonical notes")
    current = path.parent
    while current != notes:
        if current.is_symlink():
            raise ValueError("handover path traverses a symlink")
        current = current.parent
    return path


def freeze_parent(notes, parent_node, intent, mode, *, allow_property_start=False):
    relative = f"nodes/{parent_node}/handover.json"
    path = _safe_file(notes, relative)
    parent = {"node_id": parent_node, "legacy": True, "follow_up_questions": []}
    snapshot = None
    if path.is_file():
        content = path.read_bytes()
        parent = json.loads(content)
        if not isinstance(parent, dict) or parent.get("node_id") != parent_node:
            raise ValueError("parent handover identity does not match")
        snapshot = {"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "snapshot": parent}
    if intent is None and not parent.get("legacy", True):
        raise ValueError("research_intent is required to respond to this parent handover")
    if intent is not None:
        validate_intent(intent, parent, mode, allow_property_start=allow_property_start)
    return snapshot


def _source(notes, relative):
    path = _safe_file(notes, relative)
    source = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if re.fullmatch(r"(?:research|engineering)_analysis_v[1-9][0-9]*\.md", path.name):
        from .validation import load_document

        document = load_document(path)
        source["document_sha256"] = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        if path.with_suffix(".json").is_file():
            source["record_path"] = path.with_suffix(".json").relative_to(notes).as_posix()
    return source


def publish_handovers(store, state):
    """Called under the trusted lock; deterministic and recoverable, without an LLM."""
    from .validation import load_document

    notes = store.notes_root
    for node_id, record in state["nodes"].items():
        analysis = record.get("analysis") or {}
        if (analysis.get("status") != "completed" or not analysis.get("analysis_ref")
                or record.get("engineering_status") not in {"completed", "skipped"}):
            continue
        relative = analysis["analysis_ref"]
        if not re.fullmatch(rf"nodes/{node_id}/research_analysis_v[1-9][0-9]*\.md", relative):
            raise ValueError("handover Analysis reference does not match Node")
        report_path = _safe_file(notes, relative)
        spec_ref = f"nodes/{node_id}/spec.md"
        spec_path = _safe_file(notes, spec_ref)
        try:
            report = load_document(report_path)
            spec = yaml.safe_load(spec_path.read_text())
        except (OSError, ValueError, yaml.YAMLError):
            # A derived index cannot manufacture evidence or block recovery of
            # another Node. Parent eligibility still validates the source reports.
            continue
        authored = report.get("handover")
        if authored is not None:
            validate_body(authored)
        intent = spec.get("research_intent") or {}
        sources = {"research_analysis": _source(notes, relative), "spec": _source(notes, spec_ref),
                   "engineering_analysis": None}
        versions = sorted((_safe_file(notes, p.relative_to(notes).as_posix())
                           for p in (notes / "nodes" / node_id).glob("engineering_analysis_v*.md")
                           if p.stem.rsplit("_v", 1)[-1].isdigit()),
                          key=lambda p: int(p.stem.rsplit("_v", 1)[-1]))
        if versions and record["engineering_status"] == "completed":
            sources["engineering_analysis"] = _source(notes, versions[-1].relative_to(notes).as_posix())
        value = {
            "version": 1, "node_id": node_id, "legacy": authored is None,
            "research_sha": report.get("research_sha"),
            "contract_hash": report.get("evidence_manifest", {}).get("contract_hash"),
            "evaluation_version": report.get("evaluation_version"), "sources": sources,
            "problem_ref": intent.get("problem_ref"),
            "question": intent.get("question", spec.get("hypothesis", "Legacy question not recorded.")),
            **(authored or {"mechanism_verdict": "inconclusive", "practical_verdict": "inconclusive",
                           "evidence_refs": [], "new_information": [], "follow_up_questions": []}),
        }
        path = _safe_file(notes, f"nodes/{node_id}/handover.json")
        content = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if not path.exists() or path.read_text() != content:
            atomic_write_text(path, content, mode=0o644)
