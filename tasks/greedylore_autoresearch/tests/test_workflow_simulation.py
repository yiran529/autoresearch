"""Exercise the real workflow with scripted processes and synthetic scientific results."""

# ruff: noqa: S101

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["normal", "faults"])
def test_complete_workflow_without_models_or_training(tmp_path, scenario):
    import simulation_support

    root = tmp_path / scenario
    root.mkdir()
    env = os.environ.copy()
    task = Path(__file__).resolve().parents[1]
    repo = task.parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent),
            str(task / "autoresearch/src"),
            str(task / "grader/src"),
            str(repo),
            env.get("PYTHONPATH", ""),
        ]
    )
    log = root / "driver.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(simulation_support.__file__)), "driver", str(root), scenario],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            try:
                code = process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                pytest.fail(f"simulation timed out; inspect {log}\n{log.read_text()[-6000:]}")
        finally:
            simulation_support.cleanup_processes(root)
    assert code == 0, f"simulation failed; inspect {log}\n{log.read_text()[-6000:]}"
    report = json.loads((root / "report.json").read_text())
    assert report["simulated"] is True
    assert report["completed_research"] == ["N001", "N002", "N003"]
    assert report["quality_failed"] == ["N002"]
    assert report["engineering_attempts"] == 2
    assert report["engineering_completed"] == 1
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "analyzer" and e.get("mode") == "research" for e in timeline)
    assert not any(e.get("event") == "analyzer" and e.get("mode") == "engineering" for e in timeline)
    assert report["parallel_research_verified"] is True
    assert report["retired_worktrees"] is True
    assert report["memory_idempotent"] is True
    assert report["authority_rejections"] == ["other_node", "reconcile", "spec", "tree"]
    assert report["blocked_command_invocations"] == []
    assert report["remaining_simulation_processes"] == []
    if scenario == "faults":
        assert report["same_session_resumed"] is True
        assert report["analyzer_failure_recovered"] is True
        assert report["claim_conflict_verified"] is True
        assert report["infra_retry_same_sha_verified"] is True
    print(f"\nSimulation report: {root / 'report.json'}")


def test_cleanup_tracks_spawned_worker_before_its_first_event(tmp_path):
    import simulation_support
    from coral.grader.trusted_writes import process_start_time

    # The parent has launched a child, but the child has not imported its task yet.
    # A recorded, matching PID must remain visible even without worker_start.
    (tmp_path / 'timeline.jsonl').write_text(json.dumps({
        'event': 'worker_spawn', 'child_pid': os.getpid(),
        'child_start': process_start_time(os.getpid()),
    }) + '\n')
    assert simulation_support.live_processes(tmp_path) == [os.getpid()]
