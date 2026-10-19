"""Node-bound Engineering and independent Analysis completion contracts."""

# ruff: noqa: S101, F811

from __future__ import annotations

import json
import threading
from copy import deepcopy

import pytest
import yaml
from autoresearch.lifecycle import run_worker
from test_workflow_state import (
    _active_research,
    _engineer_with_attempts,
    _git,
    _publish_research_evidence,
    _research_request,
    _state,
    _store,
)
from test_workflow_state import (
    engineering_ready as engineering_ready,
)
from test_workflow_state import (
    root_run as root_run,
)

from coral.hub.attempts import read_attempts, write_attempt


def test_engineer_is_bound_before_research_analysis(engineering_ready):
    coral, repo, sha = engineering_ready
    roster = _store(coral, repo).reconcile(0, 1)
    assert len(roster) == 1
    assert (roster[0].node_id, roster[0].status) == ("N001", "active")
    assert not (coral / "public/notes/nodes/N001/research_analysis_v1.md").exists()
    assert _state(coral)["engineering_workers"][roster[0].agent_id]["research_sha"] == sha
    assert _store(coral, repo).reconcile(0, 1) == roster


def test_parent_waits_for_engineering_even_after_analysis(engineering_ready):
    coral, repo, _ = engineering_ready
    store = _store(coral, repo)
    # N000 is a completed quality-failing fixture; reserve an unrelated child.
    child = next(a for a in store.reconcile(1, 1) if a.role == "research")
    from test_workflow_state import _publish_analysis

    _publish_analysis(coral, "N001")
    with pytest.raises(ValueError, match="Parent.*completed"):
        store.initialize_research(child.agent_id, child.node_id,
                                  {**_research_request(), "parent_node": "N001"})


def test_root_without_analysis_enters_pipeline_and_blocks_researchers(root_run):
    path = root_run / "public/notes/nodes/N000/research_analysis_v1.md"
    path.unlink()
    assert _store(root_run).reconcile(2, 1) == ()
    assert set(_state(root_run)["nodes"]) == {"N000"}
    assert _state(root_run)["nodes"]["N000"]["analysis"]["status"] == "pending"


def test_quality_failure_requires_analysis_before_node_completes(root_run):
    store, item = _active_research(root_run)
    _publish_research_evidence(root_run, item.node_id, 0.0)
    analysis = root_run / f"public/notes/nodes/{item.node_id}/research_analysis_v1.md"
    saved = analysis.read_bytes()
    analysis.unlink()
    store.reconcile(0, 1)
    assert not store.is_node_completed(item.node_id)
    assert _state(root_run)["nodes"][item.node_id]["engineering_status"] == "skipped"
    analysis.write_bytes(saved)
    run_worker(str(root_run), runner=lambda *_: pytest.fail("published analysis reran"))
    assert store.is_node_completed(item.node_id)


def test_engineering_can_publish_while_analyzer_is_blocked(engineering_ready):
    store, node, body = _engineer_with_attempts(engineering_ready, count=1, analysis_pending=True)
    entered, release = threading.Event(), threading.Event()
    results = []

    def snapshot(_evidence, _context, scratch):
        target = scratch / "code"
        target.mkdir(exist_ok=True)
        return target

    def analyzer(_prompt, _roots, _schema, scratch):
        entered.set()
        assert release.wait(10), "test did not release Analyzer"
        output = scratch / "result.json"
        output.write_text(json.dumps({"summary": "Recorded.", "findings": ["Evidence."],
                                      "limitations": ["Simulation."]}))
        return output

    worker = threading.Thread(target=lambda: results.append(run_worker(
        str(store.coral_dir), runner=analyzer, code_snapshotter=snapshot)))
    worker.start()
    try:
        assert entered.wait(5), "Analyzer did not start"
        store.submit_engineering_analysis("engineer-E0001", body)
        store.reconcile(0, 1)
        assert (node / "engineering_analysis_v1.md").is_file()
        assert _state(store.coral_dir)["engineering_workers"]["engineer-E0001"]["status"] == "completed"
        assert not store.is_node_completed("N001")
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert results[0]["errors"] == {}
    assert store.is_node_completed("N001")


def test_infrastructure_retry_does_not_start_either_branch(root_run):
    store, item = _active_research(root_run)
    attempt = _publish_research_evidence(root_run, item.node_id, 1.0)
    attempt.status = "timeout"
    attempt.metadata.update(failure_class="infra", infra_retry_count=0, max_infra_retries=1)
    write_attempt(root_run, attempt)
    store.reconcile(0, 1)
    record = _state(root_run)["nodes"][item.node_id]
    assert record["research"]["status"] == "active"
    assert record["analysis"] is None
    assert not record["engineering_claims"]


def test_v1_state_is_rejected_without_rewriting(root_run):
    store = _store(root_run)
    store.reconcile(0, 1)
    state = _state(root_run)
    state["version"] = 1
    store.state_path.write_text(json.dumps(state))
    before = store.state_path.read_bytes()
    with pytest.raises(ValueError, match="version 1"):
        store.reconcile(0, 1)
    assert store.state_path.read_bytes() == before


def test_root_engineering_is_required_before_first_researcher(root_run):
    repo = root_run.parent / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "model.py").write_text("baseline = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "S0")
    source = _git(repo, "rev-parse", "HEAD")
    (repo / "submission.yaml").write_text("node_id: N000\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "A0")
    root_sha = _git(repo, "rev-parse", "HEAD")
    attempt = read_attempts(root_run)[0]
    (root_run / f"public/attempts/{attempt.commit_hash}.json").unlink()
    attempt.commit_hash, attempt.parent_hash = root_sha, source
    attempt.metadata["scores"]["quality_gate"]["value"] = 1.0
    write_attempt(root_run, attempt)
    node = root_run / "public/notes/nodes/N000"
    spec = yaml.safe_load((node / "spec.md").read_text())
    spec["base_commit"] = source
    (node / "spec.md").write_text(yaml.safe_dump(spec))
    (node / "research_analysis_v1.md").unlink()
    policy = root_run / "private/root_bootstrap.json"
    body = json.loads(policy.read_text())
    body["source_commit"] = source
    policy.write_text(json.dumps(body))
    store = _store(root_run, repo)
    roster = store.reconcile(2, 1)
    assert [(a.role, a.node_id) for a in roster] == [("engineering", "N000")]
    assert store.claim_engineering(roster[0].agent_id, "N000")["research_sha"] == root_sha
    from coral.types import Attempt
    write_attempt(root_run, Attempt(
        commit_hash="e" * 40, agent_id=roster[0].agent_id, title="Root Engineering",
        score=0.5, status="regressed", parent_hash=root_sha,
        timestamp="2026-01-02T00:00:00+00:00",
        metadata={"mode": "engineering", "node_id": "N000", "grader_runs": []},
    ))
    store.submit_engineering_analysis(roster[0].agent_id, {
        "summary": "Recorded.", "findings": ["No speedup."], "limitations": ["Simulation."],
        "attempt_shas": ["e" * 40],
    })
    assert store.reconcile(2, 1) == ()  # Analysis still pending.
    from test_workflow_state import _publish_analysis
    _publish_analysis(root_run, "N000")
    assert len(store.reconcile(2, 1)) == 2
    assert store.is_node_completed("N000")
    assert _git(repo, "rev-parse", "HEAD") == root_sha


def test_baseline_wrapper_cannot_race_the_dynamic_analyzer(root_run):
    from autoresearch.bootstrap import BootstrapError, bootstrap_root_analysis

    (root_run / "public/notes/nodes/N000/research_analysis_v1.md").unlink()
    entered, release = threading.Event(), threading.Event()
    results = []

    def analyzer(_prompt, _roots, _schema, scratch):
        entered.set()
        assert release.wait(5)
        output = scratch / "output.json"
        output.write_text(json.dumps({"summary": "Recorded.", "findings": ["Evidence."],
                                      "limitations": ["Simulation."]}))
        return output

    def snapshot(_e, _ctx, scratch):
        target = scratch / "code"
        target.mkdir()
        return target

    worker = threading.Thread(target=lambda: results.append(run_worker(
        str(root_run), runner=analyzer, code_snapshotter=snapshot)))
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(BootstrapError, match="already running"):
            bootstrap_root_analysis(root_run, root_run.parent,
                                    runner=lambda *_: pytest.fail("duplicate Analyzer"))
    finally:
        release.set()
        worker.join(5)
    path = bootstrap_root_analysis(root_run, root_run.parent,
                                   runner=lambda *_: pytest.fail("published Analysis reran"))
    assert path.is_file() and results[0]["errors"] == {}
    assert _state(root_run)["nodes"]["N000"]["analysis"]["generation"] == 1


def test_root_baseline_analyzes_exhausted_infrastructure_failure(root_run):
    from autoresearch.bootstrap import bootstrap_root_analysis

    attempt = read_attempts(root_run)[0]
    attempt.status = "timeout"
    attempt.metadata.update(failure_class="infra", infra_retry_count=2, max_infra_retries=2)
    attempt.metadata["scores"]["quality_gate"]["value"] = None
    write_attempt(root_run, attempt)
    path = root_run / "public/notes/nodes/N000/research_analysis_v1.md"
    path.unlink()

    def analyzer(_p, _roots, _schema, scratch):
        output = scratch / "output.json"
        output.write_text(json.dumps({"summary": "Infrastructure failed.", "findings": ["No quality result."],
                                      "limitations": ["Unavailable quality."]}))
        return output

    result = bootstrap_root_analysis(root_run, root_run.parent, runner=analyzer,
                                     code_snapshotter=lambda *_: root_run.parent)
    assert result == path
    assert _state(root_run)["nodes"]["N000"]["engineering_skip_reason"] == "quality_unavailable"


def test_engineering_capacity_queues_every_node_in_readiness_order(engineering_ready):
    from coral.types import Attempt

    coral, repo, first_sha = engineering_ready
    store = _store(coral, repo)
    first = next(a for a in read_attempts(coral) if a.commit_hash == first_sha)
    first.status = "pending"
    write_attempt(coral, first)
    state = _state(coral)
    state["nodes"]["N001"]["research"]["status"] = "active"
    store.state_path.write_text(json.dumps(state))
    child = next(a for a in store.reconcile(2, 0) if a.node_id == "N002")
    store.initialize_research(child.agent_id, child.node_id, _research_request())
    (repo / "model.txt").write_text("second Research\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second Research")
    second = deepcopy(first)
    second.commit_hash = _git(repo, "rev-parse", "HEAD")
    second.agent_id, second.status = child.agent_id, "regressed"
    second.metadata["node_id"] = "N002"
    write_attempt(coral, second)
    store.reconcile(0, 0)  # N002 becomes ready first, despite its larger ID.
    first.status = "improved"
    write_attempt(coral, first)
    roster = store.reconcile(0, 1)
    assert [(a.role, a.node_id) for a in roster] == [("engineering", "N002")]
    assert _state(coral)["nodes"]["N001"]["engineering_status"] == "pending"
    owner = roster[0].agent_id
    write_attempt(coral, Attempt(
        commit_hash="e" * 40, agent_id=owner, title="Engineering", score=0.5,
        status="regressed", parent_hash=second.commit_hash, timestamp="2026-01-03T00:00:00+00:00",
        metadata={"node_id": "N002", "mode": "engineering", "grader_runs": []},
    ))
    store.submit_engineering_analysis(owner, {
        "summary": "No speedup.", "findings": ["Recorded."], "limitations": ["Simulation."],
        "attempt_shas": ["e" * 40],
    })
    next_roster = store.reconcile(0, 1)
    assert [(a.role, a.node_id) for a in next_roster] == [("engineering", "N001")]
    assert len(_state(coral)["nodes"]["N002"]["engineering_claims"]) == 1


@pytest.mark.parametrize("value", [None, True, 0.5, "1"])
def test_malformed_quality_is_an_error_not_an_engineering_skip(root_run, value):
    store, item = _active_research(root_run)
    attempt = _publish_research_evidence(root_run, item.node_id, 0.0)
    attempt.metadata["scores"]["quality_gate"]["value"] = value
    write_attempt(root_run, attempt)
    store.reconcile(0, 1)
    record = _state(root_run)["nodes"][item.node_id]
    assert "quality evidence" in record["error"]
    assert record["engineering_status"] == "blocked"
    assert not store.is_node_completed(item.node_id)
