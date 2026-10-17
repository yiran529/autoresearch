"""Concurrent, persistent workflow-state contracts for dynamic Node agents."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from autoresearch.analysis import AnalysisContext, resolve_evidence
from autoresearch.evaluation import refresh_node_evaluation
from autoresearch.validation import seal_document

from coral.hub.attempts import read_attempts, write_attempt
from coral.types import Attempt

ROOT_SHA = "a" * 40
SOURCE_SHA = "b" * 40


def _root_attempt() -> Attempt:
    return Attempt(
        commit_hash=ROOT_SHA,
        agent_id="root-bootstrap",
        title="bootstrap Root N000",
        score=1.0,
        status="improved",
        parent_hash=SOURCE_SHA,
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={
            "mode": "research",
            "node_id": "N000",
            "contract_hash": "fixed-contract",
            "submission_hash": "c" * 64,
            "scores": {"quality_gate": {"value": 1.0, "name": "quality_gate"}},
            "grader_runs": [],
        },
    )


@pytest.fixture
def root_run(tmp_path: Path) -> Path:
    coral_dir = tmp_path / ".coral"
    notes = coral_dir / "public" / "notes"
    node = notes / "nodes" / "N000"
    node.mkdir(parents=True)
    spec = {
        "node_id": "N000",
        "node_type": "seed",
        "parent_node": None,
        "parent_commit": None,
        "base_commit": SOURCE_SHA,
        "analysis_refs": {"parent_research": None, "parent_engineering": None},
    }
    (node / "spec.md").write_text(yaml.safe_dump(spec, sort_keys=False))
    attempt = _root_attempt()
    write_attempt(coral_dir, attempt)
    (node / "evaluation.md").write_text(
        refresh_node_evaluation("N000", (attempt,), coral_dir / "public" / "eval_logs")
    )
    context = AnalysisContext(
        coral_dir=coral_dir,
        notes_root=notes,
        island_id=None,
        attempts=(attempt,),
        log_root=coral_dir / "public" / "eval_logs",
        checkout_root=tmp_path,
    )
    evidence = resolve_evidence("N000", "research", context=context)
    analysis = seal_document(
        {
            "summary": "Fixed Root observation.",
            "findings": ["The Root run was recorded."],
            "limitations": ["No further claim."],
        },
        evidence,
    )
    (node / "research_analysis_v1.md").write_text(json.dumps(analysis))
    policy = coral_dir / "private" / "root_bootstrap.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps({
            "version": 1,
            "source_commit": SOURCE_SHA,
            "node_id": "N000",
            "mode": "research",
            "submission_hash": "c" * 64,
        })
    )
    return coral_dir


def _store(coral_dir: Path, repo_path: Path | None = None):
    return import_module("autoresearch.state").WorkflowStore(coral_dir, repo_path=repo_path)


def _state(coral_dir: Path) -> dict:
    return json.loads((coral_dir / "private" / "autoresearch_state.json").read_text())


def test_memory_start_accepts_live_research_reservation_before_initialize(root_run):
    from autoresearch.state import WorkflowError, authorize_memory_start

    store = _store(root_run)
    assignment = store.reconcile(1, 0)[0]
    context = {"coral_dir": str(root_run), "island_id": None}
    assert authorize_memory_start(assignment.agent_id, context) == {}
    store.cancel_assignment(assignment.agent_id)
    with pytest.raises(WorkflowError, match="Researcher"):
        authorize_memory_start(assignment.agent_id, context)


def test_concurrent_reconcile_does_not_duplicate_node_ids(root_run):
    def reconcile(_):
        return _store(root_run).reconcile(2, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(reconcile, range(2)))

    state = _state(root_run)
    assert set(state["nodes"]) == {"N000", "N001", "N002"}
    assert state["next_node_seq"] == 3
    assert [{item.node_id for item in batch} for batch in batches] == [
        {"N001", "N002"},
        {"N001", "N002"},
    ]


def test_cancelled_node_id_is_never_reused(root_run):
    store = _store(root_run)
    first = store.reconcile(1, 0)[0]
    store.cancel_assignment(first.agent_id)
    second = _store(root_run).reconcile(1, 0)[0]

    assert (first.node_id, second.node_id) == ("N001", "N002")
    assert _state(root_run)["next_node_seq"] == 3


def test_lost_reconcile_response_returns_existing_reservation(root_run):
    first = _store(root_run).reconcile(1, 0)
    second = _store(root_run).reconcile(1, 0)

    assert first == second
    assert _state(root_run)["revision"] >= 1


def test_missing_root_analysis_is_actionable_and_does_not_allocate(root_run):
    (root_run / "public/notes/nodes/N000/research_analysis_v1.md").unlink()
    with pytest.raises(ValueError, match="Root Analysis bootstrap"):
        _store(root_run).reconcile(2, 1)

    assert not (root_run / "private/autoresearch_state.json").exists()


def test_host_root_analysis_bootstrap_unblocks_first_child(root_run, tmp_path):
    from autoresearch.bootstrap import bootstrap_root_analysis

    node = root_run / "public/notes/nodes/N000"
    (node / "spec.md").unlink()
    (node / "evaluation.md").unlink()
    (node / "research_analysis_v1.md").unlink()
    calls = []

    def runner(prompt, roots, schema, scratch):
        calls.append(prompt)
        output = scratch / "result.json"
        output.write_text(json.dumps({
            "summary": "Fixed Root observation.",
            "findings": ["The Root run was recorded."],
            "limitations": ["No further claim."],
        }))
        return output

    def snapshotter(evidence, context, scratch):
        return tmp_path

    first = bootstrap_root_analysis(
        root_run, tmp_path, runner=runner, code_snapshotter=snapshotter
    )
    assert first == node / "research_analysis_v1.md"
    assert (node / "spec.md").is_file()
    assert (node / "evaluation.md").is_file()
    assert _store(root_run).reconcile(1, 0)[0].node_id == "N001"
    second = bootstrap_root_analysis(
        root_run, tmp_path, runner=runner, code_snapshotter=snapshotter
    )
    assert second == first
    assert len(calls) == 1


def test_host_root_analysis_rejects_attempt_outside_fixed_policy(root_run, tmp_path):
    from autoresearch.bootstrap import BootstrapError, bootstrap_root_analysis

    attempt_path = root_run / "public/attempts" / f"{ROOT_SHA}.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["metadata"]["submission_hash"] = "f" * 64
    attempt_path.write_text(json.dumps(attempt))
    with pytest.raises(BootstrapError, match="fixed Root policy"):
        bootstrap_root_analysis(root_run, tmp_path)


def test_missing_state_in_existing_run_refuses_allocation(root_run):
    store = _store(root_run)
    store.reconcile(1, 0)
    existing_node = root_run / "public/notes/nodes/N001"
    existing_node.mkdir(parents=True)
    (existing_node / "spec.md").write_text("node_id: N001\n", encoding="utf-8")
    store.state_path.unlink()

    with pytest.raises(ValueError, match="state is missing"):
        _store(root_run).reconcile(2, 1)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def engineering_ready(root_run: Path, tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "model.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "model.txt")
    _git(repo, "commit", "-m", "research candidate")
    commit = _git(repo, "rev-parse", "HEAD")
    _store(root_run).reconcile(1, 0)
    state = _state(root_run)
    state["nodes"]["N001"]["research"]["status"] = "completed"
    (root_run / "private/autoresearch_state.json").write_text(json.dumps(state))
    write_attempt(root_run, Attempt(
        commit_hash=commit,
        agent_id="research-N001",
        title="N001 research",
        score=1.0,
        status="improved",
        parent_hash=ROOT_SHA,
        timestamp="2026-01-02T00:00:00+00:00",
        metadata={
            "mode": "research",
            "node_id": "N001",
            "scores": {"quality_gate": {"value": 1.0, "name": "quality_gate"}},
        },
    ))
    return root_run, repo, commit


def test_two_engineers_cannot_claim_the_same_node(engineering_ready):
    coral_dir, repo, _ = engineering_ready

    def claim(agent_id):
        try:
            return _store(coral_dir, repo).claim_engineering(agent_id, "N001")
        except ValueError:
            return None

    store = _store(coral_dir, repo)
    store.reconcile(0, 2)
    state = _state(coral_dir)
    state["engineering_workers"]["engineer-E0002"] = {"status": "reserved", "node_id": None}
    state["next_engineer_seq"] = 3
    store.state_path.write_text(json.dumps(state))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, ("engineer-E0001", "engineer-E0002")))
    assert sum(outcome is not None for outcome in outcomes) == 1
    state = _state(coral_dir)
    assert len(state["nodes"]["N001"]["engineering_claims"]) == 1


def test_unbound_engineer_cancels_only_when_no_eligible_target(engineering_ready):
    coral_dir, repo, _ = engineering_ready
    store = _store(coral_dir, repo)
    first = store.reconcile(0, 1)[0]
    assert first.agent_id == "engineer-E0001"
    with pytest.raises(ValueError, match="eligible"):
        store.cancel_assignment(first.agent_id)
    state = _state(coral_dir)
    state["nodes"]["N001"]["engineering_claims"].append({
        "agent_id": "engineer-E0000", "status": "active", "research_sha": "a" * 40,
    })
    store.state_path.write_text(json.dumps(state))
    store.cancel_assignment(first.agent_id)
    assert store.reconcile(0, 1) == ()
    assert _state(coral_dir)["next_engineer_seq"] == 2


def test_claimed_engineer_does_not_block_new_eligible_target(engineering_ready):
    coral_dir, repo, _ = engineering_ready
    store = _store(coral_dir, repo)
    first = store.reconcile(0, 2)[0]
    store.claim_engineering(first.agent_id, "N001")
    (repo / "model.txt").write_text("second candidate\n", encoding="utf-8")
    _git(repo, "add", "model.txt")
    _git(repo, "commit", "-m", "second research candidate")
    second_sha = _git(repo, "rev-parse", "HEAD")
    state = _state(coral_dir)
    state["nodes"]["N002"] = {
        "research": {"agent_id": "research-N002", "status": "completed", "spec_ref": "nodes/N002/spec.md"},
        "engineering_claims": [],
    }
    state["next_node_seq"] = 3
    store.state_path.write_text(json.dumps(state))
    write_attempt(coral_dir, Attempt(
        commit_hash=second_sha,
        agent_id="research-N002",
        title="N002 research",
        score=1.0,
        status="improved",
        parent_hash=ROOT_SHA,
        timestamp="2026-01-03T00:00:00+00:00",
        metadata={"mode": "research", "node_id": "N002", "scores": {"quality_gate": {"value": 1.0}}},
    ))
    assignments = store.reconcile(0, 2)
    assert {item.agent_id for item in assignments} == {"engineer-E0001", "engineer-E0002"}


@pytest.mark.parametrize("missing", [
    "research_complete", "quality_pass", "code_snapshot", "unclaimed",
])
def test_engineering_claim_rechecks_eligibility(engineering_ready, missing):
    coral_dir, repo, commit = engineering_ready
    store = _store(coral_dir, repo)
    store.reconcile(0, 1)
    claimant = "engineer-E0001"
    if missing == "research_complete":
        state = _state(coral_dir)
        state["nodes"]["N001"]["research"]["status"] = "active"
        store.state_path.write_text(json.dumps(state))
    elif missing == "quality_pass":
        path = coral_dir / "public/attempts" / f"{commit}.json"
        attempt = json.loads(path.read_text())
        attempt["metadata"]["scores"]["quality_gate"]["value"] = 0.0
        path.write_text(json.dumps(attempt))
    elif missing == "code_snapshot":
        store = _store(coral_dir, coral_dir / "missing-repo")
    else:
        store.claim_engineering("engineer-E0001", "N001")
        state = _state(coral_dir)
        state["engineering_workers"]["engineer-E0002"] = {"status": "reserved", "node_id": None}
        store.state_path.write_text(json.dumps(state))
        claimant = "engineer-E0002"

    with pytest.raises(ValueError, match="not eligible"):
        store.claim_engineering(claimant, "N001")



def _research_request() -> dict[str, str]:
    return {
        "parent_node": "N000",
        "hypothesis": "Stable gradient path.",
        "selection_rationale": "Root Analysis suggests it.",
        "delta": "Add a gate.",
        "risk": "Slower convergence.",
        "falsification_criteria": "Quality gate fails.",
    }


def test_initialize_requires_reserved_owner_and_is_idempotent(root_run):
    store = _store(root_run)
    reservation = store.reconcile(1, 0)[0]
    with pytest.raises(ValueError, match="owner"):
        store.initialize_research("research-N999", reservation.node_id, _research_request())
    result = store.initialize_research(
        reservation.agent_id, reservation.node_id, _research_request()
    )
    assert result["status"] == "active"
    assert result["spec_ref"] == "nodes/N001/spec.md"
    first = (root_run / "public/notes/nodes/N001/spec.md").read_bytes()
    repeat = store.initialize_research(
        reservation.agent_id, reservation.node_id, _research_request()
    )
    assert repeat == result
    assert (root_run / "public/notes/nodes/N001/spec.md").read_bytes() == first


def test_research_completion_requires_terminal_attempt_and_analysis(root_run):
    store = _store(root_run)
    reservation = store.reconcile(1, 0)[0]
    store.initialize_research(reservation.agent_id, reservation.node_id, _research_request())
    with pytest.raises(ValueError, match="Research Attempt"):
        store.complete_assignment(reservation.agent_id, "research")
    assert _state(root_run)["nodes"]["N001"]["research"]["status"] == "active"



def _publish_research_evidence(root_run: Path, node_id: str, quality: float) -> Attempt:
    attempt = Attempt(
        commit_hash="d" * 40,
        agent_id=f"research-{node_id}",
        title=f"{node_id} research",
        score=quality,
        status="improved" if quality == 1.0 else "regressed",
        parent_hash=ROOT_SHA,
        timestamp="2026-01-02T00:00:00+00:00",
        metadata={
            "mode": "research", "node_id": node_id,
            "contract_hash": "fixed-contract", "grader_runs": [],
            "scores": {"quality_gate": {"value": quality, "name": "quality_gate"}},
        },
    )
    write_attempt(root_run, attempt)
    node = root_run / "public/notes/nodes" / node_id
    (node / "evaluation.md").write_text(
        refresh_node_evaluation(node_id, (attempt,), root_run / "public/eval_logs"),
        encoding="utf-8",
    )
    context = AnalysisContext(
        coral_dir=root_run,
        notes_root=root_run / "public/notes",
        island_id=None,
        attempts=(attempt,),
        log_root=root_run / "public/eval_logs",
        checkout_root=root_run.parent,
    )
    evidence = resolve_evidence(node_id, "research", context=context)
    document = seal_document({
        "summary": "The result is recorded.",
        "findings": ["One formal Research Attempt completed."],
        "limitations": ["No broader claim."],
    }, evidence)
    (node / "research_analysis_v1.md").write_text(json.dumps(document), encoding="utf-8")
    return attempt


def _active_research(root_run):
    store = _store(root_run)
    item = store.reconcile(1, 0)[0]
    store.initialize_research(item.agent_id, item.node_id, _research_request())
    return store, item


def test_scheduler_finishes_research_without_agent_analyze_or_complete(root_run):
    from autoresearch.lifecycle import run_worker

    store, item = _active_research(root_run)
    _publish_research_evidence(root_run, item.node_id, quality=0.0)
    # Reusing a valid immutable Analysis must not start Codex again.
    result = run_worker(str(root_run), runner=lambda *_: pytest.fail("unexpected Codex"))
    assert result["completed"] == [item.agent_id]
    assert _state(root_run)["nodes"][item.node_id]["research"]["status"] == "completed"
    assert store.reconcile(1, 0)[0].node_id == "N002"


@pytest.mark.parametrize("status,failure,retries", [
    ("pending", None, 0), ("timeout", "infra", 0),
])
def test_scheduler_refreshes_evaluation_but_waits_for_research_retry(root_run, status, failure, retries):
    from autoresearch.lifecycle import run_worker

    _, item = _active_research(root_run)
    attempt = _publish_research_evidence(root_run, item.node_id, quality=0.0)
    attempt.status = status
    attempt.metadata.update(failure_class=failure, infra_retry_count=retries, max_infra_retries=2)
    write_attempt(root_run, attempt)
    result = run_worker(str(root_run), runner=lambda *_: pytest.fail("premature Analyzer"))
    assert result["completed"] == []
    research = _state(root_run)["nodes"][item.node_id]["research"]
    assert research["status"] == "active"
    assert not research.get("finish_requested")
    assert status in (root_run / f"public/notes/nodes/{item.node_id}/evaluation.md").read_text()


def test_scheduler_retries_failed_analyzer_and_blocks_agent_mutations(root_run):
    from autoresearch.lifecycle import run_worker
    from autoresearch.state import authorize_artifact, authorize_attempt

    _, item = _active_research(root_run)
    _publish_research_evidence(root_run, item.node_id, quality=0.0)
    path = root_run / f"public/notes/nodes/{item.node_id}/research_analysis_v1.md"
    valid = path.read_text()
    path.unlink()

    def fail(*_):
        raise RuntimeError("Analyzer unavailable")

    def snapshot(evidence, context, scratch):
        target = scratch / "code"
        target.mkdir()
        return target

    result = run_worker(str(root_run), runner=fail, code_snapshotter=snapshot)
    assert item.agent_id in result["errors"]
    research = _state(root_run)["nodes"][item.node_id]["research"]
    assert research["status"] == "active" and research["finish_requested"]
    context = {"coral_dir": str(root_run), "agent_id": item.agent_id,
               "node_id": item.node_id, "mode": "research"}
    with pytest.raises(ValueError, match="finaliz"):
        authorize_attempt({"node_id": item.node_id, "mode": "research"}, context)
    with pytest.raises(ValueError, match="finaliz"):
        authorize_artifact("analysis", context)
    path.write_text(valid)
    assert run_worker(str(root_run))["completed"] == [item.agent_id]


def test_finalization_worker_lock_is_nonblocking(root_run):
    import fcntl

    from autoresearch.lifecycle import run_worker

    lock = root_run / "private/workflow-finalization.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_worker(str(root_run))["status"] == "locked"





def test_engineering_completion_requires_attempt_and_analysis(engineering_ready):
    coral_dir, repo, research_sha = engineering_ready
    store = _store(coral_dir, repo)
    store.reconcile(0, 1)
    store.claim_engineering("engineer-E0001", "N001")
    with pytest.raises(ValueError, match="Engineering Attempt"):
        store.complete_assignment("engineer-E0001", "engineering")

    node = coral_dir / "public/notes/nodes/N001"
    node.mkdir(parents=True)
    (node / "spec.md").write_text(yaml.safe_dump({
        "node_id": "N001", "parent_node": "N000",
        "parent_commit": ROOT_SHA, "base_commit": ROOT_SHA,
        "analysis_refs": {"parent_research": "nodes/N000/research_analysis_v1.md",
                          "parent_engineering": None},
    }), encoding="utf-8")
    engineering = Attempt(
        commit_hash="e" * 40, agent_id="engineer-E0001",
        title="N001 efficiency", score=-0.1, status="regressed",
        parent_hash=research_sha, timestamp="2026-01-03T00:00:00+00:00",
        metadata={
            "mode": "engineering", "node_id": "N001",
            "contract_hash": "fixed-contract", "grader_runs": [],
        },
    )
    write_attempt(coral_dir, engineering)
    attempts = tuple(
        attempt for attempt in read_attempts(coral_dir)
        if attempt.metadata.get("node_id") == "N001"
    )
    (node / "evaluation.md").write_text(
        refresh_node_evaluation("N001", attempts, coral_dir / "public/eval_logs"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Engineering Analysis"):
        store.complete_assignment("engineer-E0001", "engineering")

    context = AnalysisContext(
        coral_dir=coral_dir,
        notes_root=coral_dir / "public/notes",
        island_id=None,
        attempts=attempts,
        log_root=coral_dir / "public/eval_logs",
        checkout_root=repo,
    )
    evidence = resolve_evidence("N001", "engineering", context=context)
    document = seal_document({
        "summary": "The efficiency result is recorded.",
        "findings": ["Measured speedup was negative."],
        "limitations": ["Single candidate only."],
    }, evidence)
    (node / "engineering_analysis_v1.md").write_text(
        json.dumps(document), encoding="utf-8"
    )
    assert store.complete_assignment("engineer-E0001", "engineering")["status"] == "completed"


def test_engineer_finishes_only_on_request_without_a_round_limit(engineering_ready, monkeypatch):
    from autoresearch import cli
    from autoresearch.lifecycle import run_worker
    from autoresearch.state import authorize_attempt, handle_workflow

    from coral.grader.trusted_writes import AgentBinding, TrustedWriteService

    coral_dir, repo, research_sha = engineering_ready
    store = _store(coral_dir, repo)
    store.reconcile(0, 1)
    store.claim_engineering("engineer-E0001", "N001")
    node = coral_dir / "public/notes/nodes/N001"
    node.mkdir(parents=True)
    (node / "spec.md").write_text(yaml.safe_dump({
        "node_id": "N001", "parent_node": "N000", "base_commit": ROOT_SHA,
        "parent_commit": ROOT_SHA, "analysis_refs": {"parent_research":
        "nodes/N000/research_analysis_v1.md", "parent_engineering": None},
    }))
    for index, sha in enumerate(("e" * 40, "f" * 40, "1" * 40)):
        write_attempt(coral_dir, Attempt(
            commit_hash=sha, agent_id="engineer-E0001", title="Optimization",
            score=-0.1, status="regressed", parent_hash=research_sha,
            timestamp=f"2026-01-0{index + 3}T00:00:00+00:00",
            metadata={"node_id": "N001", "mode": "engineering",
                      "contract_hash": "fixed-contract", "grader_runs": []},
        ))
    assert run_worker(str(coral_dir))["completed"] == []
    # An earlier manually published Analysis must not hide later candidates.
    old_attempts = tuple(a for a in read_attempts(coral_dir)
                         if a.metadata.get("mode") != "engineering" or a.commit_hash == "e" * 40)
    (node / "evaluation.md").write_text(refresh_node_evaluation(
        "N001", old_attempts, coral_dir / "public/eval_logs"))
    old_context = AnalysisContext(coral_dir, coral_dir / "public/notes", None,
                                  old_attempts, coral_dir / "public/eval_logs", repo)
    old_document = seal_document({"summary": "Only the first candidate.",
                                 "findings": ["No speedup."], "limitations": ["Incomplete history."]},
                                resolve_evidence("N001", "engineering", context=old_context))
    (node / "engineering_analysis_v1.md").write_text(json.dumps(old_document))
    context = {"coral_dir": str(coral_dir), "agent_id": "engineer-E0001"}
    assert authorize_attempt({"node_id": "N001", "mode": "engineering"}, context) == {}
    service = TrustedWriteService(coral_dir, SimpleNamespace(grader=SimpleNamespace(args={})),
                                  workflow_handler=handle_workflow)
    binding = AgentBinding("engineer-E0001", 1, 1, str(repo), None)
    monkeypatch.setattr(cli, "_coral_dir", lambda: coral_dir)
    monkeypatch.setattr(cli, "request_trusted", lambda _, op, payload:
                        service.handle_authenticated(op, payload, binding))
    cli.main(["engineering", "finish"])
    assert _state(coral_dir)["engineering_workers"][binding.agent_id]["finish_requested"]
    assert store.request_finish("engineer-E0001", "engineering")["status"] == "finalizing"
    with pytest.raises(ValueError, match="finaliz"):
        authorize_attempt({"node_id": "N001", "mode": "engineering"}, context)

    def snapshot(evidence, context, scratch):
        target = scratch / "code"
        target.mkdir()
        return target

    def runner(prompt, roots, schema, scratch):
        output = scratch / "output.json"
        output.write_text(json.dumps({"summary": "Recorded all three candidates.",
                                     "findings": ["No measured speedup."],
                                     "limitations": ["Limited optimization scope."]}))
        return output

    result = run_worker(str(coral_dir), runner=runner, code_snapshotter=snapshot)
    assert result["errors"] == {}
    assert result["completed"] == ["engineer-E0001"]
    assert _state(coral_dir)["engineering_workers"]["engineer-E0001"]["status"] == "completed"
    final_document = json.loads((node / "engineering_analysis_v2.md").read_text())
    assert len(final_document["evidence_manifest"]["engineering_attempts"]) == 3


def test_engineer_cannot_finish_pending_or_other_owners_work(engineering_ready):
    coral_dir, repo, _ = engineering_ready
    store = _store(coral_dir, repo)
    store.reconcile(0, 1)
    store.claim_engineering("engineer-E0001", "N001")
    with pytest.raises(ValueError, match="owner"):
        store.request_finish("engineer-E9999", "engineering")
    with pytest.raises(ValueError, match="Engineering Attempt"):
        store.request_finish("engineer-E0001", "engineering")
    write_attempt(coral_dir, Attempt(
        commit_hash="e" * 40, agent_id="engineer-E0001", title="Pending",
        score=None, parent_hash=None,
        status="pending", timestamp="2026-01-03T00:00:00+00:00",
        metadata={"node_id": "N001", "mode": "engineering"},
    ))
    with pytest.raises(ValueError, match="not terminal"):
        store.request_finish("engineer-E0001", "engineering")


def test_scheduler_checks_memory_without_another_researcher_start(root_run):
    from autoresearch.lifecycle import run_worker

    _store(root_run).reconcile(0, 0)
    calls = []

    def launch(*args):
        calls.append(args)
    assert run_worker(str(root_run), memory_launcher=launch)["errors"] == {}
    assert calls == []
    for node_id in ("N001", "N002"):
        path = root_run / f"public/notes/nodes/{node_id}/research_analysis_v1.md"
        path.parent.mkdir(parents=True)
        path.write_text("Published Analysis")
    run_worker(str(root_run), memory_launcher=launch)
    assert calls == [(root_run, "autoresearch.memory:run_worker")]


def test_research_completion_accepts_quality_failure_but_not_engineering(root_run):
    store = _store(root_run)
    reservation = store.reconcile(1, 0)[0]
    store.initialize_research(reservation.agent_id, reservation.node_id, _research_request())
    _publish_research_evidence(root_run, reservation.node_id, quality=0.0)
    assert store.complete_assignment(reservation.agent_id, "research")["status"] == "completed"
    assert not store.is_engineering_eligible(reservation.node_id)



def test_attempt_and_artifact_authorizers_reject_other_nodes(root_run):
    from autoresearch.state import authorize_artifact, authorize_attempt

    store = _store(root_run)
    first, second = store.reconcile(2, 0)
    store.initialize_research(first.agent_id, first.node_id, _research_request())
    context = {
        "coral_dir": str(root_run), "island_id": None, "agent_id": first.agent_id,
    }
    assert authorize_attempt({"node_id": first.node_id, "mode": "research"}, context) == {}
    with pytest.raises(ValueError, match="owner"):
        authorize_attempt({"node_id": second.node_id, "mode": "research"}, context)
    for kind in ("tree", "spec"):
        with pytest.raises(ValueError, match="authoritative"):
            authorize_artifact(kind, {**context, "node_id": first.node_id})
    assert authorize_artifact("analysis", {
        **context, "node_id": first.node_id, "mode": "research",
    }) == {}
    with pytest.raises(ValueError, match="owner"):
        authorize_artifact("analysis", {
            **context, "node_id": second.node_id, "mode": "research",
        })


def test_completed_research_owner_loses_attempt_and_analysis_rights(root_run):
    from autoresearch.state import authorize_artifact, authorize_attempt

    store = _store(root_run)
    reservation = store.reconcile(1, 0)[0]
    store.initialize_research(reservation.agent_id, reservation.node_id, _research_request())
    _publish_research_evidence(root_run, reservation.node_id, quality=0.0)
    store.complete_assignment(reservation.agent_id, "research")
    context = {
        "coral_dir": str(root_run), "island_id": None, "agent_id": reservation.agent_id,
        "node_id": reservation.node_id, "mode": "research",
    }
    with pytest.raises(ValueError, match="owner"):
        authorize_attempt({"node_id": reservation.node_id, "mode": "research"}, context)
    with pytest.raises(ValueError, match="owner"):
        authorize_artifact("analysis", context)
    result = store.complete_assignment(reservation.agent_id, "research")
    assert result["status"] == "completed"
    assert store.complete_assignment(reservation.agent_id, "research") == result
    assert not store.is_engineering_eligible(reservation.node_id)
