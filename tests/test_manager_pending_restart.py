"""Pending evaluations must not cause false failure feedback or model restart loops."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from coral.agent.manager import AgentManager
from coral.config import CoralConfig
from coral.hub.attempts import write_attempt
from coral.types import Attempt
from coral.workspace import ProjectPaths


def _manager(tmp_path: Path, monkeypatch, *, exit_code=0, grader_alive=True, age=0, island_id=None):
    config = CoralConfig.from_dict(
        {
            "task": {"name": "pending", "description": "test"},
            "agents": {"runtime": "codex", "heartbeat_enabled": False, "timeout": 0},
        }
    )
    manager = AgentManager(config)
    manager.paths = ProjectPaths(
        results_dir=tmp_path,
        task_dir=tmp_path,
        run_dir=tmp_path,
        coral_dir=tmp_path / ".coral",
        agents_dir=tmp_path / "agents",
        repo_dir=tmp_path / "repo",
    )
    attempt = Attempt(
        commit_hash="a" * 40,
        agent_id="agent-1",
        title="queued candidate",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp=(datetime.now(UTC) - timedelta(seconds=age)).isoformat(),
    )
    write_attempt(manager.paths.coral_dir, attempt, island_id=island_id)
    if island_id is not None:
        manager._agent_island["agent-1"] = island_id
    log = tmp_path / "agent.log"
    log.write_text('{"type":"turn.completed"}\n')
    manager.handles = [
        SimpleNamespace(
            agent_id="agent-1",
            alive=False,
            process=SimpleNamespace(returncode=exit_code),
            log_path=log,
        )
    ]
    manager._running = True
    manager._started_at["agent-1"] = datetime.now(UTC).timestamp() - 120
    monkeypatch.setattr("coral.agent.manager.signal.signal", lambda *_: None)
    monkeypatch.setattr(manager, "_grader_alive", lambda: grader_alive)
    monkeypatch.setattr(manager, "_write_agent_pids", lambda: None)
    launches = []

    def restart(idx, prompt=None, prompt_source=None):
        launches.append((manager.handles[idx].agent_id, prompt))
        return SimpleNamespace(agent_id="agent-1", alive=True, log_path=log)

    # Launching a real agent would call an external model; keep the monitor real.
    monkeypatch.setattr(manager, "_restart_agent", restart)
    return manager, attempt, launches


def _ticks(manager, monkeypatch, count, after_tick=None):
    ticks = 0

    def wait(timeout):
        nonlocal ticks
        ticks += 1
        if after_tick:
            after_tick(ticks)
        return ticks >= count

    monkeypatch.setattr(manager._stop_event, "wait", wait)
    manager.monitor_loop(check_interval=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_pending_feedback_does_not_report_failure(dynamic):
    config = CoralConfig.from_dict(
        {
            "task": {"name": "pending", "description": "test"},
            "agents": {
                "dynamic": {"enabled": dynamic},
                "assignments": [{"role": "research"}, {"role": "engineering"}],
            },
        }
    )
    prompt = AgentManager(config)._build_score_prompt(
        {
            "commit_hash": "a" * 40,
            "title": "queued candidate",
            "status": "pending",
            "score": None,
        },
        1,
    )

    assert "pending" in prompt.lower()
    assert "FAILED" not in prompt
    assert "aaaaaaaaaaaa" in prompt
    assert "wait" in prompt.lower()
    assert "same-SHA infrastructure retries" not in prompt


@pytest.mark.parametrize(
    "status,score,expected",
    [
        ("improved", 0.5, "score=0.5000000000"),
        ("regressed", 0.0, "score=0.0000000000"),
        ("failed", None, "score=FAILED"),
        ("crashed", None, "score=FAILED"),
    ],
)
def test_terminal_feedback_keeps_real_result(status, score, expected):
    config = CoralConfig.from_dict({"task": {"name": "t", "description": "test"}})
    prompt = AgentManager(config)._build_score_prompt(
        {
            "commit_hash": "a" * 40,
            "title": "candidate",
            "status": status,
            "score": score,
            "feedback": "terminal details",
        },
        1,
    )
    assert expected in prompt
    assert "terminal details" in prompt


@pytest.mark.parametrize("uptime", [5, 120])
@pytest.mark.parametrize("island_id", [None, "0"])
def test_pending_clean_exit_does_not_restart_each_tick(tmp_path, monkeypatch, uptime, island_id):
    manager, _, launches = _manager(tmp_path, monkeypatch, island_id=island_id)
    manager._started_at["agent-1"] = datetime.now(UTC).timestamp() - uptime

    _ticks(manager, monkeypatch, 4)

    assert launches == []
    assert manager._crash_history == {}


@pytest.mark.parametrize(
    "status,score,expected",
    [
        ("improved", 0.8, "score=0.8000000000"),
        ("failed", None, "score=FAILED"),
    ],
)
def test_pending_completion_resumes_once_with_terminal_result(
    tmp_path,
    monkeypatch,
    status,
    score,
    expected,
):
    manager, attempt, launches = _manager(tmp_path, monkeypatch)

    def finish(tick):
        if tick == 2:
            attempt.status = status
            attempt.score = score
            write_attempt(manager.paths.coral_dir, attempt)

    _ticks(manager, monkeypatch, 4, finish)

    assert len(launches) == 1
    assert launches[0][0] == "agent-1"
    assert expected in launches[0][1]
    assert manager._agent_eval_counts["agent-1"] == 1


@pytest.mark.parametrize("case", ["crash", "grader_dead", "expired", "post_pause"])
def test_pending_does_not_disable_fault_recovery(tmp_path, monkeypatch, case):
    manager, _, launches = _manager(
        tmp_path,
        monkeypatch,
        exit_code=1 if case == "crash" else 0,
        grader_alive=case != "grader_dead",
        age=100000 if case == "expired" else 0,
    )
    if case == "post_pause":
        manager._pending_restart_after_pause.add("agent-1")

    _ticks(manager, monkeypatch, 1)

    assert len(launches) == 1
    assert "pending" in launches[0][1].lower()
    assert "FAILED" not in launches[0][1]


def test_other_agent_pending_does_not_delay_restart(tmp_path, monkeypatch):
    manager, attempt, launches = _manager(tmp_path, monkeypatch)
    attempt.agent_id = "agent-2"
    write_attempt(manager.paths.coral_dir, attempt)

    _ticks(manager, monkeypatch, 1)

    assert launches == [("agent-1", None)]
