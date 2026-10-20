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
            "scores": {"quality_gate": {"value": 0.0, "name": "quality_gate"}},
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


def test_missing_root_analysis_waits_without_allocating_children(root_run):
    (root_run / "public/notes/nodes/N000/research_analysis_v1.md").unlink()
    assert _store(root_run).reconcile(2, 1) == ()
    assert set(_state(root_run)["nodes"]) == {"N000"}


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
    existing_node.mkdir(parents=True, exist_ok=True)
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
    node = root_run / "public/notes/nodes/N001"
    node.mkdir(parents=True, exist_ok=True)
    (node / "spec.md").write_text(yaml.safe_dump({
        "node_id": "N001", "parent_node": "N000", "parent_commit": ROOT_SHA,
        "base_commit": ROOT_SHA,
        "analysis_refs": {"parent_research": "nodes/N000/research_analysis_v1.md", "parent_engineering": None},
    }))
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


def test_bound_engineer_cannot_cancel_required_work(engineering_ready):
    coral_dir, repo, _ = engineering_ready
    store = _store(coral_dir, repo)
    first = store.reconcile(0, 1)[0]
    with pytest.raises(ValueError, match="cannot be cancelled"):
        store.cancel_assignment(first.agent_id)
    assert store.reconcile(0, 1) == (first,)


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
    node = coral_dir / "public/notes/nodes/N002"
    node.mkdir(parents=True)
    spec = yaml.safe_load((coral_dir / "public/notes/nodes/N001/spec.md").read_text())
    spec["node_id"] = "N002"
    (node / "spec.md").write_text(yaml.safe_dump(spec))
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


def _publish_analysis(coral_dir, node_id):
    context = AnalysisContext(
        coral_dir=coral_dir, notes_root=coral_dir / "public/notes", island_id=None,
        attempts=tuple(read_attempts(coral_dir)), log_root=coral_dir / "public/eval_logs",
        checkout_root=coral_dir.parent,
    )
    document = seal_document({"summary": "Recorded.", "findings": ["Evidence."],
                              "limitations": ["Simulation."]},
                             resolve_evidence(node_id, "research", context=context))
    (coral_dir / f"public/notes/nodes/{node_id}/research_analysis_v1.md").write_text(json.dumps(document))


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
    run_worker(str(root_run), runner=lambda *_: pytest.fail("unexpected Codex"))
    assert store.is_node_completed(item.node_id)
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
    assert research["status"] == "completed" and research["finish_requested"]
    context = {"coral_dir": str(root_run), "agent_id": item.agent_id,
               "node_id": item.node_id, "mode": "research"}
    with pytest.raises(ValueError, match="owner"):
        authorize_attempt({"node_id": item.node_id, "mode": "research"}, context)
    with pytest.raises(ValueError, match="owner"):
        authorize_artifact("analysis", context)
    path.write_text(valid)
    run_worker(str(root_run))
    assert _store(root_run).is_node_completed(item.node_id)


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
    node.mkdir(parents=True, exist_ok=True)
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
    with pytest.raises(ValueError, match="summary"):
        store.complete_assignment("engineer-E0001", "engineering")
    store.submit_engineering_analysis("engineer-E0001", {
        "summary": document["summary"], "findings": document["findings"],
        "limitations": document["limitations"], "attempt_shas": [engineering.commit_hash],
    })
    from autoresearch.lifecycle import run_worker
    _publish_analysis(coral_dir, "N001")
    assert run_worker(str(coral_dir), runner=lambda *_: pytest.fail("Engineering Analyzer"))["completed"] == ["engineer-E0001"]
    assert store.complete_assignment("engineer-E0001", "engineering")["status"] == "completed"


def test_engineer_finish_cli_versions_stale_analysis_without_analyzer(engineering_ready, monkeypatch, tmp_path):
    from autoresearch import cli
    from autoresearch.lifecycle import run_worker
    from autoresearch.state import handle_workflow

    from coral.grader.trusted_writes import AgentBinding, TrustedWriteService

    store, node, body = _engineer_with_attempts(engineering_ready)
    old_attempts = tuple(a for a in read_attempts(store.coral_dir)
                         if a.metadata.get("mode") != "engineering" or a.commit_hash == body["attempt_shas"][0])
    (node / "evaluation.md").write_text(refresh_node_evaluation(
        "N001", old_attempts, store.coral_dir / "public/eval_logs"))
    context = AnalysisContext(store.coral_dir, store.notes_root, None, old_attempts,
                              store.coral_dir / "public/eval_logs", store.repo_path)
    old_document = seal_document({"summary": "Only the first candidate.",
                                 "findings": ["No speedup."], "limitations": ["Incomplete history."]},
                                resolve_evidence("N001", "engineering", context=context))
    old_path = node / "engineering_analysis_v1.md"
    old_path.write_text(json.dumps(old_document))
    old_bytes = old_path.read_bytes()
    service = TrustedWriteService(store.coral_dir, SimpleNamespace(grader=SimpleNamespace(args={})),
                                  workflow_handler=handle_workflow)
    binding = AgentBinding("engineer-E0001", 1, 1, str(store.repo_path), None)
    monkeypatch.setattr(cli, "_coral_dir", lambda: store.coral_dir)
    monkeypatch.setattr(cli, "request_trusted", lambda _, op, payload:
                        service.handle_authenticated(op, payload, binding))
    summary = tmp_path / "final.json"
    summary.write_text(json.dumps(body))
    cli.main(["engineering", "finish", "--analysis-file", str(summary)])
    assert run_worker(str(store.coral_dir), runner=lambda *_: pytest.fail("Engineering Analyzer"))["completed"] == [binding.agent_id]
    assert old_path.read_bytes() == old_bytes
    document = json.loads((node / "engineering_analysis_v2.json").read_text())
    assert len(document["engineering_attempts"]) == 2


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


def test_reconcile_provides_public_task_toolkit(root_run):
    _store(root_run).reconcile(1, 0)
    toolkit = root_run / 'public/notes/toolkit'
    assert (toolkit / 'workflow_rules.md').is_file()
    assert (toolkit / 'references/initial_diagnostics.md').is_file()
    assert (toolkit / 'references/low_rank.md').is_file()
    assert not (toolkit / 'taskdata').exists()


def _engineer_with_attempts(engineering_ready, count=2, *, analysis_pending=False):
    coral_dir, repo, research_sha = engineering_ready
    store = _store(coral_dir, repo)
    store.reconcile(0, 1)
    store.claim_engineering('engineer-E0001', 'N001')
    node = coral_dir / 'public/notes/nodes/N001'
    node.mkdir(parents=True, exist_ok=True)
    shas = ['e' * 40, 'f' * 40][:count]
    for index, sha in enumerate(shas):
        write_attempt(coral_dir, Attempt(
            commit_hash=sha, agent_id='engineer-E0001', title='Optimization',
            score=-0.1, status='regressed', parent_hash=research_sha,
            timestamp=f'2026-01-0{index + 3}T00:00:00+00:00',
            metadata={'node_id': 'N001', 'mode': 'engineering',
                      'contract_hash': 'fixed-contract', 'grader_runs': []},
        ))
    body = {'summary': 'No measured speedup; implementation headroom is unresolved.',
            'findings': ['Both outcomes were recorded; no algorithmic lower bound established.'],
            'limitations': ['No profiling available.'], 'attempt_shas': shas}
    if not analysis_pending:
        _publish_analysis(coral_dir, 'N001')
    return store, node, body


def test_engineer_authored_finish_covers_all_attempts_and_never_launches_analyzer(engineering_ready):
    from autoresearch.lifecycle import run_worker

    store, node, body = _engineer_with_attempts(engineering_ready)
    with pytest.raises(ValueError, match='cover'):
        store.submit_engineering_analysis('engineer-E0001', {**body, 'attempt_shas': body['attempt_shas'][:1]})
    result = store.submit_engineering_analysis('engineer-E0001', body)
    assert result['status'] == 'finalizing'
    # Persisted author submission survives reopen, duplicate requests and a worker restart.
    assert _store(store.coral_dir, store.repo_path).submit_engineering_analysis('engineer-E0001', body) == result
    with pytest.raises(ValueError, match='conflict'):
        store.submit_engineering_analysis('engineer-E0001', {**body, 'summary': 'Changed after freeze'})
    completed = run_worker(str(store.coral_dir), runner=lambda *_: pytest.fail('Engineering Analyzer launched'))
    assert completed['errors'] == {}
    assert completed['completed'] == ['engineer-E0001']
    document = json.loads((node / 'engineering_analysis_v1.json').read_text())
    assert document['summary'] == body['summary']
    assert {a['commit_sha'] for a in document['engineering_attempts']} == set(body['attempt_shas'])
    assert any('inherited (provisional)' in line for line in document['limitations'])
    assert store.submit_engineering_analysis('engineer-E0001', body)['status'] == 'completed'


@pytest.mark.parametrize("successes,failures,allow", [
    (2, 0, True), (3, 0, False), (0, 9, True), (0, 10, False),
    (2, 7, True), (2, 8, False),
])
def test_engineering_success_and_total_limits(engineering_ready, successes, failures, allow):
    from autoresearch.state import authorize_attempt

    store, _, _ = _engineer_with_attempts(engineering_ready, count=0)
    for index in range(successes + failures):
        success = index < successes
        write_attempt(store.coral_dir, Attempt(
            commit_hash=f'{index:040x}', agent_id='engineer-E0001', title='Candidate',
            score=-0.1 if success else None, status='regressed' if success else 'crashed',
            parent_hash=ROOT_SHA, timestamp='2026-01-03T00:00:00+00:00',
            metadata={'node_id': 'N001', 'mode': 'engineering', 'grader_runs': [],
                      **({} if success else {'failure_class': 'candidate', 'budget_class': 'grader_error'})},
        ))
    context = {'coral_dir': str(store.coral_dir), 'agent_id': 'engineer-E0001',
               'commit_hash': 'd' * 40}
    canonical = {'node_id': 'N001', 'mode': 'engineering'}
    if allow:
        assert authorize_attempt(canonical, context) == {}
    else:
        with pytest.raises(ValueError, match='Engineering.*limit'):
            authorize_attempt(canonical, context)


@pytest.mark.parametrize("runs,allow", [(9, True), (10, False)])
@pytest.mark.parametrize("counter", ["retries", "history", "invocations"])
def test_engineering_same_sha_retries_consume_total_budget(engineering_ready, runs, allow, counter):
    from autoresearch.state import authorize_attempt

    store, _, body = _engineer_with_attempts(engineering_ready, count=1)
    attempt = next(a for a in read_attempts(store.coral_dir) if a.commit_hash == body['attempt_shas'][0])
    attempt.status, attempt.score = 'crashed', None
    attempt.metadata.update(failure_class='infra', budget_class='grader_error',
                            infra_retry_count=runs - 1 if counter == 'retries' else 0,
                            max_infra_retries=20,
                            grader_runs=[{'status': 'crashed'}] * runs if counter == 'history' else [],
                            grader_invocation_count=runs if counter == 'invocations' else 0)
    write_attempt(store.coral_dir, attempt)
    context = {'coral_dir': str(store.coral_dir), 'agent_id': 'engineer-E0001',
               'commit_hash': attempt.commit_hash, 'retry': True}
    canonical = {'node_id': 'N001', 'mode': 'engineering'}
    if allow:
        assert authorize_attempt(canonical, context) == {}
        with pytest.raises(ValueError, match='not terminal'):
            store._terminal_engineering_unlocked('engineer-E0001', 'N001')
    else:
        with pytest.raises(ValueError, match='Engineering.*limit'):
            authorize_attempt(canonical, context)
        # Exhausting the overall budget must permit final analysis despite infra retries left.
        store._terminal_engineering_unlocked('engineer-E0001', 'N001')
        assert store.submit_engineering_analysis('engineer-E0001', body)['status'] == 'finalizing'


@pytest.mark.parametrize("archived", [False, True])
def test_engineering_pending_runs_reserve_capacity_and_allow_idempotent_submission(engineering_ready, archived):
    from autoresearch.state import authorize_attempt

    store, _, body = _engineer_with_attempts(engineering_ready, count=2)
    sha = '1' * 40
    write_attempt(store.coral_dir, Attempt(
        commit_hash=sha, agent_id='engineer-E0001', title='Pending third run',
        score=None, status='pending', parent_hash=ROOT_SHA,
        timestamp='2026-01-05T00:00:00+00:00',
        metadata={'node_id': 'N001', 'mode': 'engineering', 'archived': archived,
                  'grader_runs': [{'status': 'running'}]},
    ))
    context = {'coral_dir': str(store.coral_dir), 'agent_id': 'engineer-E0001',
               'commit_hash': sha}
    canonical = {'node_id': 'N001', 'mode': 'engineering'}
    assert authorize_attempt(canonical, context) == {}
    assert authorize_attempt(canonical, {**context, 'commit_hash': body['attempt_shas'][0]}) == {}
    with pytest.raises(ValueError, match='Engineering.*limit'):
        authorize_attempt(canonical, {**context, 'commit_hash': '2' * 40})
    with pytest.raises(ValueError, match='not terminal'):
        store._terminal_engineering_unlocked('engineer-E0001', 'N001')


def test_engineering_tune_cannot_bypass_formal_run_limits(engineering_ready):
    from autoresearch.state import authorize_attempt

    store, _, _ = _engineer_with_attempts(engineering_ready, count=0)
    with pytest.raises(ValueError, match='Engineering.*tune'):
        authorize_attempt({'node_id': 'N001', 'mode': 'engineering'}, {
            'coral_dir': str(store.coral_dir), 'agent_id': 'engineer-E0001',
            'commit_hash': '1' * 40, 'budget_class': 'tune',
        })


def test_engineering_archived_successes_still_count_and_can_be_reported(engineering_ready):
    from autoresearch.state import authorize_attempt

    from coral.hub.attempts import archive_attempts

    store, _, body = _engineer_with_attempts(engineering_ready, count=2)
    # Use an existing Engineering record as the shape, with a third distinct SHA.
    third = next(a for a in read_attempts(store.coral_dir) if a.commit_hash == body['attempt_shas'][0])
    third.commit_hash = '1' * 40
    write_attempt(store.coral_dir, third)
    body['attempt_shas'].append(third.commit_hash)
    archive_attempts(store.coral_dir, set(body['attempt_shas']), reason='discarded by resume --from')
    assert not any(a.agent_id == 'engineer-E0001' for a in read_attempts(store.coral_dir))
    with pytest.raises(ValueError, match='Engineering.*limit'):
        authorize_attempt({'node_id': 'N001', 'mode': 'engineering'}, {
            'coral_dir': str(store.coral_dir), 'agent_id': 'engineer-E0001', 'commit_hash': '2' * 40,
        })
    assert store.submit_engineering_analysis('engineer-E0001', body)['status'] == 'finalizing'
    store.reconcile(0, 1)
    assert _state(store.coral_dir)['engineering_workers']['engineer-E0001']['status'] == 'completed'


def test_engineering_archived_terminal_infra_failure_can_finish_without_retry(engineering_ready):
    from coral.hub.attempts import archive_attempts

    store, _, body = _engineer_with_attempts(engineering_ready, count=1)
    failed = next(a for a in read_attempts(store.coral_dir) if a.commit_hash == body['attempt_shas'][0])
    failed.score, failed.status = None, 'crashed'
    failed.metadata.update(failure_class='infra', budget_class='grader_error',
                           infra_retry_count=0, max_infra_retries=2)
    write_attempt(store.coral_dir, failed)
    archive_attempts(store.coral_dir, {failed.commit_hash}, reason='discarded by resume --from')
    assert store.submit_engineering_analysis('engineer-E0001', body)['status'] == 'finalizing'
    store.reconcile(0, 1)
    assert _state(store.coral_dir)['engineering_workers']['engineer-E0001']['status'] == 'completed'


def test_bare_engineering_finish_cannot_start_independent_analyzer(engineering_ready):
    store, _, _ = _engineer_with_attempts(engineering_ready, count=1)
    with pytest.raises(ValueError, match='summary'):
        store.request_finish('engineer-E0001', 'engineering')


def test_engineer_publication_failure_is_recoverable_without_reauthoring(engineering_ready, monkeypatch):
    store, node, body = _engineer_with_attempts(engineering_ready, count=1)
    store.submit_engineering_analysis('engineer-E0001', body)
    from coral.grader import analysis_artifacts
    writer = analysis_artifacts._atomic_text
    def unavailable(path, content, **kwargs):
        if path.name == 'engineering_analysis_v1.md':
            raise OSError('publication temporarily unavailable')
        return writer(path, content, **kwargs)
    monkeypatch.setattr(analysis_artifacts, '_atomic_text', unavailable)
    store.reconcile(0, 1)
    assert _state(store.coral_dir)['engineering_workers']['engineer-E0001']['status'] == 'active'
    assert not (node / 'engineering_analysis_v1.md').exists()
    monkeypatch.setattr(analysis_artifacts, '_atomic_text', writer)
    store.reconcile(0, 1)
    assert _state(store.coral_dir)['engineering_workers']['engineer-E0001']['status'] == 'completed'


def test_engineer_owner_and_pending_attempts_are_verified(engineering_ready):
    store, _, body = _engineer_with_attempts(engineering_ready, count=1)
    with pytest.raises(ValueError, match='owner'):
        store.submit_engineering_analysis('engineer-E9999', body)
    attempt = next(a for a in read_attempts(store.coral_dir) if a.commit_hash == body['attempt_shas'][0])
    attempt.status = 'pending'
    write_attempt(store.coral_dir, attempt)
    with pytest.raises(ValueError, match='terminal'):
        store.submit_engineering_analysis('engineer-E0001', body)
    assert not _state(store.coral_dir)['engineering_workers']['engineer-E0001'].get('finish_requested')


def test_host_samples_research_modes_once_and_persists_through_resume(root_run, monkeypatch):
    from autoresearch.state import WorkflowStore

    store = WorkflowStore(root_run, p_explore=0.2)
    draws = iter((0.1, 0.8))
    monkeypatch.setattr('autoresearch.state.random.random', lambda: next(draws))
    first = store.reconcile(2, 0)
    assert [item.research_mode for item in first] == ['EXPLORE', 'REFINE']
    monkeypatch.setattr('autoresearch.state.random.random', lambda: pytest.fail('mode resampled'))
    restored = WorkflowStore(root_run, p_explore=1.0).reconcile(2, 0)
    assert restored == first
    store = WorkflowStore(root_run)
    store.initialize_research(first[0].agent_id, first[0].node_id, _research_request())
    spec = yaml.safe_load((root_run / 'public/notes/nodes/N001/spec.md').read_text())
    assert spec['research_mode'] == 'EXPLORE'
    assert spec['parent_node'] == 'N000'
    with pytest.raises(ValueError, match='invalid fields'):
        store.initialize_research(first[0].agent_id, first[0].node_id,
                                  {**_research_request(), 'research_mode': 'REFINE'})


def test_legacy_research_assignment_defaults_to_refine_without_drawing_again(root_run, monkeypatch):
    store = _store(root_run)
    first = store.reconcile(1, 0)[0]
    store.initialize_research(first.agent_id, first.node_id, _research_request())
    spec_path = root_run / 'public/notes/nodes/N001/spec.md'
    spec = yaml.safe_load(spec_path.read_text())
    spec.pop('research_mode', None)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    state = _state(root_run)
    state['nodes']['N001']['research'].pop('research_mode', None)
    store.state_path.write_text(json.dumps(state))
    monkeypatch.setattr('autoresearch.state.random.random', lambda: pytest.fail('legacy mode resampled'))
    assert _store(root_run).reconcile(1, 0)[0].research_mode == 'REFINE'
    # Reinitialization must not rewrite an immutable legacy Spec.
    before = spec_path.read_bytes()
    _store(root_run).initialize_research(first.agent_id, first.node_id, _research_request())
    assert spec_path.read_bytes() == before


@pytest.mark.parametrize('probability', [True, -0.1, 1.1, float('nan'), float('inf'), '0.2'])
def test_invalid_explore_probability_is_rejected(root_run, probability):
    from autoresearch.state import WorkflowStore
    with pytest.raises(ValueError, match='probability'):
        WorkflowStore(root_run, p_explore=probability)


def test_mode_probability_comes_from_trusted_task_config(root_run, monkeypatch):
    from autoresearch.state import WorkflowStore
    (root_run / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {'p_explore': 0.9}}}))
    monkeypatch.setattr('autoresearch.state.random.random', lambda: 0.5)
    assert WorkflowStore(root_run).reconcile(1, 0)[0].research_mode == 'EXPLORE'
