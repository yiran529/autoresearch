"""Problem chains carry frozen expectations without constraining exploration."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml
from autoresearch.handover import publish_handovers, validate_body, validate_intent
from autoresearch.nodes import create_node
from test_workflow_state import root_run as root_run  # noqa: F401


def body():
    return {
        "mechanism_verdict": "solved", "practical_verdict": "not_solved",
        "evidence_refs": ["evaluation.md"],
        "new_information": ["Selection communication fell; quality failed."],
        "follow_up_questions": [{"id": "Q1", "question": "Why did quality regress?",
            "importance": "Blocks the useful mechanism.",
            "discriminator": "Same-state short continuation with and without index reuse."}],
    }


def intent(problem_ref="N001:Q1"):
    return {"problem_ref": problem_ref, "question": "Can direction reuse preserve quality?",
            "mechanism_test": "Fewer score reductions at fixed coefficient rank.",
            "practical_test": "Both quality gates pass and complete-cycle time falls.",
            "handover_response": "Investigate the reported quality failure.",
            "engineering_review": "Read overlap findings; retain producer-event safety."}


def parent_handover():
    return {"node_id": "N001", "legacy": False, **body()}


def test_refine_requires_an_existing_problem_and_explicit_engineering_review():
    assert validate_intent(intent(), parent_handover(), "REFINE")["problem_ref"] == "N001:Q1"
    with pytest.raises(ValueError, match="problem_ref"):
        validate_intent(intent("N001:Q9"), parent_handover(), "REFINE")
    missing = intent()
    missing["engineering_review"] = ""
    with pytest.raises(ValueError, match="engineering_review"):
        validate_intent(missing, parent_handover(), "REFINE")


def test_explore_can_start_an_independent_question_without_fabricating_a_parent_gap():
    assert validate_intent(intent(None), parent_handover(), "EXPLORE")["problem_ref"] is None
    with pytest.raises(ValueError, match="problem_ref"):
        validate_intent(intent(None), parent_handover(), "REFINE")


def test_handover_separates_mechanism_and_practical_verdicts_and_bounds_questions():
    assert validate_body(body())["practical_verdict"] == "not_solved"
    value = body()
    value["new_information"] = []
    assert validate_body(value)["new_information"] == []
    value["follow_up_questions"] *= 3
    with pytest.raises(ValueError, match="follow_up_questions"):
        validate_body(value)
    value = body()
    value["evidence_refs"] = ["../../private/secret"]
    with pytest.raises(ValueError, match="evidence"):
        validate_body(value)


def test_child_freezes_handover_snapshot_even_after_parent_index_changes(tmp_path):
    notes = tmp_path / "notes"
    parent = notes / "nodes/N001"
    parent.mkdir(parents=True)
    (parent / "research_analysis_v1.md").write_text("fixed report")
    (parent / "handover.json").write_text(json.dumps(parent_handover()))
    evidence = {"notes_root": notes, "parent_node": "N001", "parent_commit": "a" * 40,
                "base_commit": "a" * 40, "analysis_ref": "nodes/N001/research_analysis_v1.md",
                "research_mode": "REFINE"}
    request = {"node_id": "N002", "hypothesis": "Reuse directions",
               "selection_rationale": "Discriminate quality costs", "delta": "Reuse indices",
               "risk": "Stale indices", "falsification_criteria": "Reject on quality failure",
               "research_intent": intent()}
    spec = create_node(request, evidence)
    first = yaml.safe_load(spec.read_text())
    assert first["parent_handover"]["sha256"] == hashlib.sha256((parent / "handover.json").read_bytes()).hexdigest()
    (parent / "handover.json").write_text(json.dumps({**parent_handover(), "follow_up_questions": []}))
    assert yaml.safe_load(spec.read_text()) == first
    assert first["parent_handover"]["snapshot"]["follow_up_questions"][0]["id"] == "Q1"


def test_modern_parent_requires_handover_response_but_legacy_parent_still_works(tmp_path):
    notes = tmp_path / "notes"
    parent = notes / "nodes/N001"
    parent.mkdir(parents=True)
    (parent / "research_analysis_v1.md").write_text("legacy")
    evidence = {"notes_root": notes, "parent_node": "N001", "parent_commit": "a" * 40,
                "base_commit": "a" * 40, "analysis_ref": "nodes/N001/research_analysis_v1.md"}
    request = {"node_id": "N002", "hypothesis": "h", "selection_rationale": "s",
               "delta": "d", "risk": "r", "falsification_criteria": "f"}
    (parent / "handover.json").write_text(json.dumps(parent_handover()))
    with pytest.raises(ValueError, match="research_intent"):
        create_node(request, evidence)
    (parent / "handover.json").unlink()
    assert create_node(request, evidence).exists()


def test_host_publishes_version_bound_handover_and_recovers_missing_index(tmp_path):
    notes = tmp_path / "notes"
    node = notes / "nodes/N001"
    node.mkdir(parents=True)
    (node / "spec.md").write_text(yaml.safe_dump({"hypothesis": "Reduce score communication",
                                                "research_intent": intent("N000:Q1")}))
    report = {"node_id": "N001", "mode": "research", "research_sha": "a" * 40,
              "summary": "measured", "handover": body(),
              "evidence_manifest": {"contract_hash": "contract"}, "evaluation_version": "evalhash"}
    (node / "research_analysis_v1.md").write_text(json.dumps(report))
    store = SimpleNamespace(notes_root=notes)
    state = {"nodes": {"N001": {"analysis": {"status": "completed",
                        "analysis_ref": "nodes/N001/research_analysis_v1.md"},
                        "engineering_status": "skipped"}}}
    publish_handovers(store, state)
    path = node / "handover.json"
    first = path.read_bytes()
    value = json.loads(first)
    assert value["question"] == intent()["question"]
    assert value["problem_ref"] == "N000:Q1"
    assert value["sources"]["research_analysis"]["path"].endswith("research_analysis_v1.md")
    assert value["mechanism_verdict"] == "solved"
    assert value["practical_verdict"] == "not_solved"
    path.unlink()
    publish_handovers(store, state)
    assert path.read_bytes() == first
    expected = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
    assert value["sources"]["research_analysis"]["document_sha256"] == expected


def test_analyzer_must_answer_modern_intent_with_frozen_evidence(tmp_path):
    from autoresearch.analysis import resolve_evidence
    from autoresearch.validation import seal_document, validate_analysis
    from test_analysis_recovery import _context

    context = _context(tmp_path)
    path = context.notes_root / "nodes/N001/spec.md"
    spec = yaml.safe_load(path.read_text())
    spec["research_intent"] = intent("N000:Q1")
    path.write_text(yaml.safe_dump(spec))
    evidence = resolve_evidence("N001", "research", context=context)
    authored = {"summary": "result", "findings": ["observed"], "limitations": ["bounded"]}
    with pytest.raises(ValueError, match="research_intent"):
        validate_analysis(seal_document(authored, evidence), evidence)
    authored["handover"] = body()
    assert validate_analysis(seal_document(authored, evidence), evidence)["handover"] == body()
    authored["handover"]["evidence_refs"] = ["invented/results.json"]
    with pytest.raises(ValueError, match="frozen evidence"):
        validate_analysis(seal_document(authored, evidence), evidence)


def test_parent_handover_cannot_escape_notes_through_a_symlink(tmp_path):
    from autoresearch.handover import freeze_parent

    notes = tmp_path / "notes"
    node = notes / "nodes/N001"
    node.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(parent_handover()))
    (node / "handover.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        freeze_parent(notes, "N001", intent(), "REFINE")


def test_real_trusted_entry_accepts_intent_and_publishes_it_in_canonical_spec(root_run):
    from autoresearch.state import WorkflowStore, handle_workflow

    from coral.grader.trusted_writes import AgentBinding, TrustedWriteService

    store = WorkflowStore(root_run, p_explore=0)
    assignment = store.reconcile(1, 0)[0]
    service = TrustedWriteService(root_run, SimpleNamespace(grader=SimpleNamespace(args={})),
                                  workflow_handler=handle_workflow)
    binding = AgentBinding(assignment.agent_id, 1, 1, str(root_run.parent), None)
    question = intent("N000:current")
    result = service.handle_authenticated("initialize_research", {
        "node_id": assignment.node_id, "parent_node": "N000", "hypothesis": "mechanism",
        "selection_rationale": "test a mechanism", "delta": "change state transition",
        "risk": "quality", "falsification_criteria": "matched comparison",
        "research_intent": question,
    }, binding)
    spec = yaml.safe_load((root_run / "public/notes" / result["spec_ref"]).read_text())
    assert spec["research_intent"] == question
    assert spec["parent_handover"]["snapshot"]["node_id"] == "N000"


def test_initialize_retry_keeps_a_preupgrade_spec_byte_for_byte(root_run):
    from autoresearch.state import WorkflowStore
    from test_workflow_state import _research_request

    store = WorkflowStore(root_run, p_explore=0)
    assignment = store.reconcile(1, 0)[0]
    request = _research_request()
    store.initialize_research(assignment.agent_id, assignment.node_id, request)
    path = store.notes_root / "nodes" / assignment.node_id / "spec.md"
    old = yaml.safe_load(path.read_text())
    old.pop("parent_handover")
    path.write_text(yaml.safe_dump(old))
    before = path.read_bytes()
    store.initialize_research(assignment.agent_id, assignment.node_id, request)
    assert path.read_bytes() == before
