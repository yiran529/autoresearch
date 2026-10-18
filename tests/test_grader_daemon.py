"""Tests for the grader daemon and the agent↔grader file-queue protocol."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from coral.config import CoralConfig
from coral.grader.daemon import (
    _find_pending,
    _is_git_repo,
    _repo_dir,
    process_pending_once,
    run_daemon,
)
from coral.hooks.post_commit import submit_eval
from coral.hub.attempts import read_attempt, read_eval_count, write_attempt
from coral.types import Attempt

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
#
# Instead of standing up a real grader venv (uv venv + installs), the fixture
# fakes one: `.coral/private/grader_venv/bin/python` is a shell wrapper that
# execs the test interpreter with PYTHONPATH pointed at a plain-directory
# grader package under `.coral/private/grader_pkg/`. SubprocessGrader spawns
# a fresh worker per grade, so rewriting `testgrader.py` between submissions
# changes grader behavior with no reinstall.


def _write_grader(repo: Path, source: str) -> None:
    """(Re)write the test grader module resolved by entrypoint testgrader:Grader."""
    pkg_dir = repo / ".coral" / "private" / "grader_pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "testgrader.py").write_text(source)


def _install_fake_grader_venv(coral_dir: Path) -> None:
    """Create a wrapper `grader_venv/bin/python` that runs the test interpreter."""
    pkg_dir = coral_dir / "private" / "grader_pkg"
    bin_dir = coral_dir / "private" / "grader_venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "python"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'export PYTHONPATH="{pkg_dir}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    wrapper.chmod(0o755)


def _init_repo_and_coral(base_dir: Path, score: float = 0.5) -> Path:
    """Create a git repo with .coral/ wired up to a minimal entrypoint grader."""
    repo = base_dir / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )

    (repo / "main.py").write_text("print('hello')\n")
    (repo / ".gitignore").write_text(".coral/\n.coral_dir\n.claude/\n.coral_agent_id\nCLAUDE.md\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "main.py", ".gitignore"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Initial"],
        capture_output=True,
        check=True,
    )

    coral_dir = repo / ".coral"
    (coral_dir / "public" / "attempts").mkdir(parents=True)
    (coral_dir / "private").mkdir(parents=True)

    (repo / ".coral_dir").write_text(str(coral_dir.resolve()))

    _install_fake_grader_venv(coral_dir)
    _write_grader(
        repo,
        "from coral.grader.task_grader import TaskGrader\n"
        "class Grader(TaskGrader):\n"
        "    def evaluate(self):\n"
        f"        return {score!r}\n",
    )

    config = {
        "task": {"name": "daemon_test", "description": "Daemon test"},
        "grader": {
            "entrypoint": "testgrader:Grader",
            "timeout": 60,
        },
        "agents": {"count": 1},
        "sharing": {"attempts": True, "notes": True, "skills": True},
        "workspace": {"base_dir": str(repo), "repo_path": str(repo)},
    }
    with open(coral_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return repo


# --------------------------------------------------------------------------- #
# _repo_dir — handles both production and test layouts                        #
# --------------------------------------------------------------------------- #


def test_repo_dir_detects_test_layout():
    """When .coral/ lives inside the repo, daemon falls back to coral_dir.parent."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        coral_dir = repo / ".coral"

        assert _repo_dir(coral_dir) == repo
        assert _is_git_repo(repo)
        assert not _is_git_repo(coral_dir)


def test_repo_dir_prefers_run_dir_repo():
    """Production layout places repo/ alongside .coral/. Daemon picks it."""
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        repo = run_dir / "repo"
        _init_repo_and_coral(run_dir)  # creates run_dir/repo and run_dir/repo/.coral

        # Copy .coral up to the run_dir level so we get run_dir/.coral + run_dir/repo
        production_coral = run_dir / ".coral"
        (repo / ".coral").rename(production_coral)
        assert _repo_dir(production_coral) == repo


# --------------------------------------------------------------------------- #
# process_pending_once — drains the queue without spawning a daemon           #
# --------------------------------------------------------------------------- #


def test_process_pending_once_grades_pending():
    """A submitted pending attempt gets scored after one drain."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d), score=0.42)
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="Change",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            assert pending.status == "pending"

            finalized = process_pending_once(repo / ".coral")
            assert len(finalized) == 1
            assert finalized[0].score == 0.42
            assert finalized[0].status == "improved"
            assert finalized[0].commit_hash == pending.commit_hash
            assert finalized[0].metadata["scores"] == {
                "eval": {"value": 0.42, "name": "eval", "explanation": None, "metadata": {}}
            }

            # No more pending after the drain.
            assert _find_pending(repo / ".coral") == []
        finally:
            sys.path.pop(0)


def test_process_pending_once_is_idempotent():
    """Running the drain a second time is a no-op when nothing is pending."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            submit_eval(message="c", agent_id="agent-1", workdir=str(repo), wait=False)

            first = process_pending_once(repo / ".coral")
            second = process_pending_once(repo / ".coral")
            assert len(first) == 1
            assert second == []
        finally:
            sys.path.pop(0)


def test_process_pending_once_preserves_submission_fields():
    """Grader finalization must not clobber commit_hash, title, timestamp, parent_hash."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="Preserve me",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            original_ts = pending.timestamp
            process_pending_once(repo / ".coral")

            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.commit_hash == pending.commit_hash
            assert final.title == "Preserve me"
            assert final.agent_id == "agent-1"
            assert final.timestamp == original_ts  # daemon doesn't restamp
            assert final.parent_hash == pending.parent_hash
        finally:
            sys.path.pop(0)


def test_grader_persists_score_breakdown_in_attempt_metadata():
    """Finalized attempts retain daemon-owned serialized ScoreBundle scores."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        _write_grader(
            repo,
            "from coral.grader.task_grader import TaskGrader\n"
            "from coral.types import Score, ScoreBundle\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        return ScoreBundle(\n"
            "            scores={\n"
            "                'quality': Score(value=0.9, name='quality', explanation='good', metadata={'source': 'test'}),\n"
            "                'cost': Score(value=0.2, name='cost'),\n"
            "            },\n"
            "            aggregated=0.8,\n"
            "            metadata={'grader_key': 'grader_value', 'scores': {'incorrect': 'value'}},\n"
            "        )\n",
        )
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="Persist breakdown", agent_id="agent-1", workdir=str(repo), wait=False
            )

            process_pending_once(repo / ".coral")

            finalized = read_attempt(repo / ".coral", pending.commit_hash)
            assert finalized is not None
            assert finalized.score == 0.8
            assert finalized.metadata["grader_key"] == "grader_value"
            assert finalized.metadata["scores"] == {
                "quality": {
                    "value": 0.9,
                    "name": "quality",
                    "explanation": "good",
                    "metadata": {"source": "test"},
                },
                "cost": {"value": 0.2, "name": "cost", "explanation": None, "metadata": {}},
            }
        finally:
            sys.path.pop(0)


def test_private_grader_omits_score_breakdown_from_attempt_metadata():
    """Private bundles suppress daemon-, grader-, and pending-provided scores."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        _write_grader(
            repo,
            "from coral.grader.task_grader import TaskGrader\n"
            "from coral.types import Score, ScoreBundle\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        return ScoreBundle(\n"
            "            scores={\n"
            "                'hidden': Score(value=0.4, name='hidden', explanation='secret', metadata={'case': 'private'}),\n"
            "            },\n"
            "            aggregated=0.4,\n"
            "            is_public=False,\n"
            "            metadata={'grader_key': 'grader_value', 'scores': {'grader': 'leak'}},\n"
            "        )\n",
        )
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="Keep breakdown private",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            pending.metadata["scores"] = {"pending": "leak"}
            write_attempt(repo / ".coral", pending)

            process_pending_once(repo / ".coral")

            finalized = read_attempt(repo / ".coral", pending.commit_hash)
            assert finalized is not None
            assert finalized.score == 0.4
            assert finalized.metadata["grader_key"] == "grader_value"
            assert "scores" not in finalized.metadata
        finally:
            sys.path.pop(0)


def test_process_pending_multiple_in_submission_order():
    """Pending attempts are graded in submission (timestamp) order."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        # This test exercises consumer-side ordering, so disable the
        # producer-side per-agent pending cap (default 1) to allow stacking.
        cfg_path = repo / ".coral" / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["grader"]["max_pending_per_agent"] = 0
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('a')\n")
            a = submit_eval(message="a", agent_id="agent-1", workdir=str(repo), wait=False)
            (repo / "main.py").write_text("print('b')\n")
            b = submit_eval(message="b", agent_id="agent-1", workdir=str(repo), wait=False)

            finalized = process_pending_once(repo / ".coral")
            assert [f.commit_hash for f in finalized] == [a.commit_hash, b.commit_hash]
        finally:
            sys.path.pop(0)


# --------------------------------------------------------------------------- #
# Atomic write — writer and concurrent reader never collide                   #
# --------------------------------------------------------------------------- #


def test_write_attempt_is_atomic():
    """Rapid writes interleaved with reads never yield a partial JSON.

    Cheap proxy: write_attempt should use tmp+rename so any read either sees
    the previous complete version or the new complete version.
    """
    with tempfile.TemporaryDirectory() as d:
        coral_dir = Path(d) / ".coral"
        (coral_dir / "public" / "attempts").mkdir(parents=True)

        commit_hash = "a" * 40
        attempt = Attempt(
            commit_hash=commit_hash,
            agent_id="a1",
            title="t",
            score=None,
            status="pending",
            parent_hash=None,
            timestamp=datetime.now(UTC).isoformat(),
        )
        write_attempt(str(coral_dir), attempt)

        target = coral_dir / "public" / "attempts" / f"{commit_hash}.json"
        # Hammer the writer while reading; every read must parse as JSON.
        for i in range(50):
            attempt.score = float(i)
            write_attempt(str(coral_dir), attempt)
            data = json.loads(target.read_text())
            assert data["score"] == float(i)


# --------------------------------------------------------------------------- #
# Isolated worktree — grader doesn't see agent's post-submit edits            #
# --------------------------------------------------------------------------- #


def test_grader_sees_committed_code_not_working_tree():
    """If the agent mutates files after submit, grader must grade the commit snapshot."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        # Grader reports sentinel = content of main.py at checkout time.
        _write_grader(
            repo,
            "import os\n"
            "from coral.grader.task_grader import TaskGrader\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        with open(os.path.join(self.codebase_path, 'main.py')) as f:\n"
            "            content = f.read()\n"
            "        return 1.0 if 'COMMITTED' in content else 0.0\n",
        )
        try:
            (repo / "main.py").write_text("# COMMITTED\nprint('x')\n")
            pending = submit_eval(
                message="stable snapshot",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            # Agent now mutates the working tree post-submission — should NOT affect grading.
            (repo / "main.py").write_text("# POST-SUBMIT\nprint('y')\n")

            process_pending_once(repo / ".coral")
            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.score == 1.0, (
                "Grader must use the isolated checkout at commit_hash, "
                "not the agent's live working tree."
            )
        finally:
            pass


# --------------------------------------------------------------------------- #
# Budget class accounting (issue #73)                                         #
# --------------------------------------------------------------------------- #


def test_submit_eval_tune_flag_marks_pending():
    """`submit_eval(tune=True)` writes budget_class=tune onto the pending record."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="sweep lr",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
                tune=True,
            )
            assert pending.metadata.get("budget_class") == "tune"
            assert pending.budget_class == "tune"
        finally:
            sys.path.pop(0)


def test_grader_receives_trusted_submission_metadata():
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        _write_grader(
            repo,
            "from coral.grader.task_grader import TaskGrader\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        metadata = self.tasks[0].metadata\n"
            "        return 1.0 if metadata.get('mode') == 'research' and metadata.get('node_id') == 'N001' else 0.0\n",
        )
        (repo / "main.py").write_text("print('formal')\n")
        pending = submit_eval(
            message="formal", agent_id="agent-1", workdir=str(repo), wait=False
        )
        pending.metadata.update(
            mode="research", node_id="N001", node_type="research",
            base_commit="a" * 40, quality_anchor=None, contract_hash="contract",
            submission_hash="submission",
        )
        write_attempt(repo / ".coral", pending)

        process_pending_once(repo / ".coral")

        final = read_attempt(repo / ".coral", pending.commit_hash)
        assert final is not None
        assert final.score == 1.0


def test_grader_preserves_tune_class_through_finalization():
    """Successfully-graded tune attempt keeps budget_class=tune (not overwritten to 'real')."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d), score=0.42)
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="sweep",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
                tune=True,
            )
            process_pending_once(repo / ".coral")
            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.score == 0.42
            assert final.status == "improved"
            assert final.budget_class == "tune", (
                f"Expected budget_class=tune to flow through, got "
                f"{final.metadata.get('budget_class')!r}"
            )
        finally:
            sys.path.pop(0)


def test_grader_sees_tune_flag_via_self_tune():
    """`coral eval --tune` exposes self.tune=True to the user's grader."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        # Grader returns 1.0 in tune mode, 0.0 otherwise — the score
        # is how we observe what self.tune saw.
        _write_grader(
            repo,
            "from coral.grader.task_grader import TaskGrader\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        return 1.0 if self.tune else 0.0\n",
        )
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            tune_pending = submit_eval(
                message="tune sweep",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
                tune=True,
            )
            process_pending_once(repo / ".coral")
            tune_final = read_attempt(repo / ".coral", tune_pending.commit_hash)
            assert tune_final is not None
            assert tune_final.score == 1.0, (
                "Grader should have seen self.tune=True (got score=0.0, "
                "meaning self.tune was False)."
            )

            # And a non-tune submission must NOT see self.tune=True.
            (repo / "main.py").write_text("print('v3')\n")
            real_pending = submit_eval(
                message="real attempt",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            process_pending_once(repo / ".coral")
            real_final = read_attempt(repo / ".coral", real_pending.commit_hash)
            assert real_final is not None
            assert real_final.score == 0.0
        finally:
            sys.path.pop(0)


def test_grader_marks_real_class_on_normal_success():
    """Default eval (no --tune) ends up classified as 'real' after grading."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d), score=0.5)
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="real attempt",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            process_pending_once(repo / ".coral")
            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.budget_class == "real"
        finally:
            sys.path.pop(0)


def test_grader_marks_grader_error_on_exception():
    """A grader that raises is classified as 'grader_error' (not a real fail)."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        # Overwrite the grader to raise.
        _write_grader(
            repo,
            "from coral.grader.task_grader import TaskGrader\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        raise RuntimeError('grader-side failure')\n",
        )
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="x",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            process_pending_once(repo / ".coral")
            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.status == "crashed"
            assert final.budget_class == "grader_error", (
                "Grader exceptions should be classified as grader_error, not real attempts."
            )
            assert "scores" not in final.metadata
        finally:
            sys.path.pop(0)


def test_grader_marks_grader_error_on_timeout():
    """A grader that hangs past `grader.timeout` is killed and classified
    as 'grader_error'.

    SubprocessGrader runs its worker under subprocess.run(timeout=...) and
    raises TimeoutError when it expires; the daemon turns that into
    status="timeout" / budget_class="grader_error". The sleep is well past
    the timeout so the kill is the only thing that can end the eval.
    """
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        # Tighten the timeout so the test runs quickly.
        config_path = repo / ".coral" / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["grader"]["timeout"] = 2
        config_path.write_text(yaml.dump(config))
        _write_grader(
            repo,
            "import time\n"
            "from coral.grader.task_grader import TaskGrader\n"
            "class Grader(TaskGrader):\n"
            "    def evaluate(self):\n"
            "        time.sleep(120)\n"
            "        return 0.5\n",
        )
        sys.path.insert(0, str(repo))
        try:
            (repo / "main.py").write_text("print('v2')\n")
            pending = submit_eval(
                message="hang",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )
            process_pending_once(repo / ".coral")
            final = read_attempt(repo / ".coral", pending.commit_hash)
            assert final is not None
            assert final.status == "timeout"
            assert final.budget_class == "grader_error"
            assert final.metadata["failure_class"] == "candidate"
        finally:
            sys.path.pop(0)


# --------------------------------------------------------------------------- #
# run_daemon subprocess — submit from main process, daemon in child           #
# --------------------------------------------------------------------------- #


def test_run_daemon_subprocess_grades_pending():
    """End-to-end: spawn the daemon in a subprocess and verify it picks up pending."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d), score=0.9)

        try:
            (repo / "main.py").write_text("print('real daemon')\n")
            pending = submit_eval(
                message="daemon run",
                agent_id="agent-1",
                workdir=str(repo),
                wait=False,
            )

            stop_event = multiprocessing.Event()
            proc = multiprocessing.Process(
                target=run_daemon,
                args=(str(repo / ".coral"), stop_event),
            )
            proc.start()
            try:
                deadline = time.monotonic() + 30.0
                final = None
                while time.monotonic() < deadline:
                    final = read_attempt(repo / ".coral", pending.commit_hash)
                    if final and final.status != "pending":
                        break
                    time.sleep(0.2)
                assert final is not None and final.status != "pending"
                assert final.score == 0.9
            finally:
                stop_event.set()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                proc.close()
        finally:
            pass


# --------------------------------------------------------------------------- #
# Parallel drain (issue #81)                                                  #
# --------------------------------------------------------------------------- #


def test_default_max_workers_is_1():
    """Configs without `grader.parallel` get max_workers=1 (legacy behavior)."""
    cfg = CoralConfig.from_dict({"task": {"name": "x", "description": "y"}, "agents": {"count": 1}})
    assert cfg.grader.parallel.max_workers == 1


def _install_concurrency_probe_grader(repo: Path, sleep_seconds: float) -> Path:
    """Overwrite the test grader with one that reports peak concurrent executions.

    Each grade increments a shared counter on entry, sleeps, decrements on
    exit. The counter is in a JSON file under private_dir, mutated under
    fcntl.flock so concurrent grades can't lose updates. Returns the path to
    that file so the test can read `max` after the drain.
    """
    coral_dir = repo / ".coral"
    log_path = coral_dir / "private" / "concurrency.json"
    grader_src = (
        "import fcntl, json, time\n"
        "from pathlib import Path\n"
        "from coral.grader.task_grader import TaskGrader\n"
        "\n"
        f"LOG_PATH = {str(log_path)!r}\n"
        f"SLEEP = {sleep_seconds!r}\n"
        "\n"
        "def _bump(delta):\n"
        "    p = Path(LOG_PATH)\n"
        "    p.touch(exist_ok=True)\n"
        "    with open(p, 'r+') as f:\n"
        "        fcntl.flock(f, fcntl.LOCK_EX)\n"
        "        text = f.read() or '{}'\n"
        "        try: data = json.loads(text)\n"
        "        except Exception: data = {}\n"
        "        data['current'] = data.get('current', 0) + delta\n"
        "        if delta > 0:\n"
        "            data['max'] = max(data.get('max', 0), data['current'])\n"
        "        f.seek(0); f.truncate()\n"
        "        f.write(json.dumps(data))\n"
        "        f.flush()\n"
        "        fcntl.flock(f, fcntl.LOCK_UN)\n"
        "\n"
        "class Grader(TaskGrader):\n"
        "    def evaluate(self):\n"
        "        _bump(1)\n"
        "        try:\n"
        "            time.sleep(SLEEP)\n"
        "        finally:\n"
        "            _bump(-1)\n"
        "        return 1.0\n"
    )
    _write_grader(repo, grader_src)
    return log_path


def _set_config(repo: Path, **grader_overrides) -> None:
    """Patch grader fields in .coral/config.yaml in-place."""
    cfg_path = repo / ".coral" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("grader", {}).update(grader_overrides)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)


def _submit_n(repo: Path, n: int) -> list[str]:
    """Make N distinct commits and submit each as a pending attempt."""
    hashes = []
    for i in range(n):
        (repo / "main.py").write_text(f"print('v{i}')\n")
        attempt = submit_eval(
            message=f"v{i}",
            agent_id="agent-1",
            workdir=str(repo),
            wait=False,
        )
        hashes.append(attempt.commit_hash)
    return hashes


def test_drain_runs_in_parallel_when_max_workers_gt_1():
    """With max_workers=4, four 0.4s grades overlap (peak concurrency > 1)."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        log_path = _install_concurrency_probe_grader(repo, sleep_seconds=0.4)
        _set_config(
            repo,
            max_pending_per_agent=0,  # allow stacking 4 pending from one agent
            parallel={"max_workers": 4},
        )

        sys.path.insert(0, str(repo))
        try:
            _submit_n(repo, 4)
            finalized = process_pending_once(repo / ".coral")
            assert len(finalized) == 4
            assert all(a.score == 1.0 for a in finalized)

            data = json.loads(log_path.read_text())
            assert data["max"] >= 2, (
                f"Expected overlapping grades with max_workers=4, got max={data['max']}"
            )
            assert data["current"] == 0  # all grades finished
        finally:
            sys.path.pop(0)


def test_drain_serializes_when_max_workers_is_1():
    """max_workers=1 keeps the legacy serial behavior — peak concurrency stays 1."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        log_path = _install_concurrency_probe_grader(repo, sleep_seconds=0.2)
        _set_config(
            repo,
            max_pending_per_agent=0,
            parallel={"max_workers": 1},
        )

        sys.path.insert(0, str(repo))
        try:
            _submit_n(repo, 3)
            finalized = process_pending_once(repo / ".coral")
            assert len(finalized) == 3

            data = json.loads(log_path.read_text())
            assert data["max"] == 1, (
                f"Expected serial grading with max_workers=1, got max={data['max']}"
            )
        finally:
            sys.path.pop(0)


def test_eval_count_correct_under_parallel_grading():
    """Race-prone increment_eval_count stays correct when grades run in parallel."""
    with tempfile.TemporaryDirectory() as d:
        repo = _init_repo_and_coral(Path(d))
        _install_concurrency_probe_grader(repo, sleep_seconds=0.1)
        _set_config(
            repo,
            max_pending_per_agent=0,
            parallel={"max_workers": 4},
        )

        sys.path.insert(0, str(repo))
        try:
            _submit_n(repo, 5)
            process_pending_once(repo / ".coral")
            assert read_eval_count(repo / ".coral") == 5
        finally:
            sys.path.pop(0)


def test_invalid_max_workers_rejected():
    """grader.parallel.max_workers must be >= 1."""
    with pytest.raises(ValueError, match="max_workers"):
        CoralConfig.from_dict(
            {
                "task": {"name": "x", "description": "y"},
                "grader": {"parallel": {"max_workers": 0}},
            }
        )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_waiting_worker_capacity_rejected(value):
    from coral.config import GraderConfig, ParallelGraderConfig

    with pytest.raises(ValueError):
        GraderConfig(parallel=ParallelGraderConfig(max_waiting_workers=value))


@pytest.mark.parametrize("research_count", [1, 3])
def test_priority_daemon_admits_late_engineering_without_restarting_research(
    tmp_path, monkeypatch, research_count,
):
    import threading

    from coral.grader.daemon import _serve_resource_pending

    repo = _init_repo_and_coral(tmp_path)
    _set_config(repo, parallel={"max_workers": 1, "max_waiting_workers": 2})
    config = CoralConfig.from_yaml(repo / ".coral/config.yaml")
    _submit_n(repo, 1)
    research = _find_pending(repo / ".coral")[0]
    research.metadata["mode"] = "research"
    write_attempt(str(repo / ".coral"), research)
    researchers = [research]
    for index in range(1, research_count):
        (repo / "main.py").write_text(f"print('research {index}')\n")
        queued = submit_eval(message="Research", agent_id=f"research-{index}",
                             workdir=str(repo), wait=False)
        queued.metadata["mode"] = "research"
        write_attempt(str(repo / ".coral"), queued)
        researchers.append(queued)
    research_waiting, engineer_finished = threading.Event(), threading.Event()
    stop = threading.Event()
    starts, finished = [], []

    def grade(attempt, *_args):
        starts.append(attempt.agent_id)
        if attempt.metadata["mode"] == "research":
            if len(starts) >= min(research_count, 2):
                research_waiting.set()
            assert engineer_finished.wait(5), "late Engineer never reached a worker"
        attempt.status = "improved"
        attempt.score = 1
        write_attempt(str(repo / ".coral"), attempt)
        finished.append(attempt.agent_id)
        if attempt.metadata["mode"] == "engineering":
            engineer_finished.set()
        if len(finished) == research_count + 1:
            stop.set()
        return attempt

    monkeypatch.setattr("coral.grader.daemon._safe_grade_one", grade)
    errors = []
    def serve():
        try:
            _serve_resource_pending(repo / ".coral/config.yaml", repo / ".coral", config,
                                    should_stop=stop.is_set)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=serve)
    worker.start()
    try:
        assert research_waiting.wait(3)
        (repo / "main.py").write_text("print('engineering')\n")
        engineering = submit_eval(message="Engineer", agent_id="engineer", workdir=str(repo), wait=False)
        engineering.metadata["mode"] = "engineering"
        write_attempt(str(repo / ".coral"), engineering)
        worker.join(8)
    finally:
        stop.set()
        engineer_finished.set()
        worker.join(5)
    assert not errors
    assert not worker.is_alive()
    assert starts[:min(research_count, 2)] == [a.agent_id for a in researchers[:2]]
    assert starts[min(research_count, 2)] == "engineer"
    assert finished[0] == "engineer"
    assert sorted(finished) == sorted(["engineer", *[a.agent_id for a in researchers]])


def test_late_engineer_gets_gpu_first_with_real_grader_workers(tmp_path, monkeypatch):
    """Actual subprocess workers share the allocator; inventory alone is synthetic."""
    import threading

    source = Path(__file__).resolve().parents[1] / "tasks/greedylore_autoresearch/grader/src"
    monkeypatch.syspath_prepend(str(source))
    from greedylore_autoresearch_grader.gpu import (
        GpuDevice,
        GpuInventory,
        IdleGpuPolicy,
        reserve_idle_gpus,
    )
    repo = _init_repo_and_coral(tmp_path)
    coral_dir = repo / ".coral"
    locks = tmp_path / "gpu-locks"
    log = coral_dir / "private/order.jsonl"
    _set_config(repo, parallel={"max_workers": 1, "max_waiting_workers": 2})
    _write_grader(repo, f'''
import json, sys, time
from pathlib import Path
sys.path.insert(0, {str(source)!r})
from coral.grader import TaskGrader
from greedylore_autoresearch_grader.gpu import GpuDevice, GpuInventory, IdleGpuPolicy, reserve_idle_gpus

class Grader(TaskGrader):
    def evaluate(self):
        mode = self.tasks[0].metadata['mode']
        def inventory():
            return GpuInventory(tuple(GpuDevice(i, f'GPU-{{i}}', 'A100', 0, 0, (), 0) for i in range(4)), 'topology')
        def event(state, **details):
            if state == 'waiting':
                (Path(self.private_dir) / (mode + '.waiting')).touch()
        with reserve_idle_gpus(4, {str(locks)!r}, IdleGpuPolicy(0,512,5),
                source_role='engineer' if mode == 'engineering' else 'researcher',
                execution_group={str(coral_dir)!r}, execution_limit=1,
                inventory_provider=inventory, wait_timeout=5, poll_interval=.01, on_event=event):
            with Path({str(log)!r}).open('a') as stream:
                stream.write(json.dumps(mode) + '\\n')
            time.sleep(.03)
        return 1.0
''')
    _submit_n(repo, 1)
    research = _find_pending(coral_dir)[0]
    research.metadata["mode"] = "research"
    write_attempt(str(coral_dir), research)
    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4))
    holder = reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=lambda: GpuInventory(devices, "topology"))
    stop = threading.Event()
    daemon = threading.Thread(target=run_daemon, args=(coral_dir, stop))
    def until(predicate):
        deadline = time.monotonic() + 8
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.01)
        assert predicate()
    try:
        daemon.start()
        until(lambda: (coral_dir / "private/research.waiting").exists())
        (repo / "main.py").write_text("print('later engineer')\n")
        engineer = submit_eval(message="later Engineer", agent_id="engineer", workdir=str(repo), wait=False)
        engineer.metadata["mode"] = "engineering"
        write_attempt(str(coral_dir), engineer)
        until(lambda: (coral_dir / "private/engineering.waiting").exists())
        holder.release()
        until(lambda: read_attempt(str(coral_dir), research.commit_hash).status != "pending")
        until(lambda: read_attempt(str(coral_dir), engineer.commit_hash).status != "pending")
    finally:
        holder.release()
        stop.set()
        daemon.join(8)
    assert not daemon.is_alive()
    assert [json.loads(row) for row in log.read_text().splitlines()] == ["engineering", "research"]
    for attempt in (research, engineer):
        final = read_attempt(str(coral_dir), attempt.commit_hash)
        assert final.score == 1
        assert final.metadata["grader_invocation_count"] == 1
        assert len(final.metadata["grader_runs"]) == 1


def test_find_pending_multi_island_scans_every_island(tmp_path):
    """_find_pending picks up attempts from every islands/<id>/attempts/ dir."""
    from coral.grader.daemon import _find_pending
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    coral_dir = tmp_path / ".coral"
    for i in range(3):
        (coral_dir / "islands" / str(i) / "attempts").mkdir(parents=True)

    a0 = Attempt(
        commit_hash="aaa000",
        agent_id="0-agent-1",
        title="x",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp="2026-05-31T10:00:00Z",
        metadata={"island_id": "0"},
    )
    a1 = Attempt(
        commit_hash="bbb111",
        agent_id="1-agent-1",
        title="y",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp="2026-05-31T10:01:00Z",
        metadata={"island_id": "1"},
    )
    write_attempt(coral_dir, a0, island_id="0")
    write_attempt(coral_dir, a1, island_id="1")
    pending = _find_pending(coral_dir)
    hashes = {a.commit_hash for a in pending}
    assert hashes == {"aaa000", "bbb111"}


def test_find_pending_skips_archived(tmp_path):
    """Archived (soft-deleted) pending attempts must never be graded."""
    from coral.grader.daemon import _find_pending
    from coral.hub.attempts import archive_attempts, write_attempt
    from coral.types import Attempt

    coral_dir = tmp_path / ".coral"

    def _pending(commit: str) -> Attempt:
        return Attempt(
            commit_hash=commit,
            agent_id="agent-1",
            title="x",
            score=None,
            status="pending",
            parent_hash=None,
            timestamp="2026-05-31T10:00:00Z",
        )

    write_attempt(coral_dir, _pending("aaa000"))
    write_attempt(coral_dir, _pending("bbb111"))
    archive_attempts(coral_dir, {"bbb111"}, reason="discarded by resume --from aaa000")

    pending = _find_pending(coral_dir)
    assert {a.commit_hash for a in pending} == {"aaa000"}


def test_grade_one_finalizes_migrated_pending_attempt_in_current_island(tmp_path, monkeypatch):
    """If migration moves a pending attempt mid-grade, finalize into its current island."""
    import coral.grader.daemon as daemon

    coral_dir = tmp_path / ".coral"
    for island in ("0", "1"):
        (coral_dir / "islands" / island / "attempts").mkdir(parents=True)
        (coral_dir / "islands" / island / "eval_logs").mkdir(parents=True)
    (coral_dir / "public").mkdir(parents=True)

    queued = Attempt(
        commit_hash="abc123",
        agent_id="0-agent-1",
        title="queued before migration",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp="2026-05-31T10:00:00Z",
        metadata={"island_id": "0"},
    )
    write_attempt(coral_dir, queued, island_id="0")

    moved = Attempt(
        commit_hash="abc123",
        agent_id="0-agent-1",
        title="queued before migration",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp="2026-05-31T10:00:00Z",
        metadata={"island_id": "1"},
    )
    write_attempt(coral_dir, moved, island_id="1")
    (coral_dir / "islands" / "0" / "attempts" / "abc123.json").unlink()
    old_log = coral_dir / "islands" / "0" / "eval_logs" / "abc123" / "metrics.json"
    old_log.parent.mkdir(parents=True)
    old_log.write_text("{}")

    config = CoralConfig.from_dict(
        {
            "task": {"name": "daemon_test", "description": "Daemon test"},
            "grader": {"timeout": 60},
            "agents": {"count": 1},
        }
    )
    config_path = coral_dir / "config.yaml"
    config_path.write_text("task:\n  name: daemon_test\n  description: Daemon test\n")
    repo = tmp_path / "repo"
    repo.mkdir()

    class Bundle:
        aggregated = 0.9
        feedback = "ok"
        metadata = {"grader_key": "grader_value"}
        scores = {}

    monkeypatch.setattr(daemon, "_repo_dir", lambda _coral_dir: repo)
    monkeypatch.setattr(
        daemon,
        "_add_isolated_worktree",
        lambda _repo_dir, _commit_hash, dest: dest.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(daemon, "_remove_worktree", lambda _repo_dir, _dest: None)
    monkeypatch.setattr(daemon, "_run_grader", lambda *args, **kwargs: Bundle())

    finalized = daemon._grade_one(queued, config_path, coral_dir, config)

    assert finalized.status == "improved"
    assert read_attempt(coral_dir, "abc123", island_id="0") is None
    current = read_attempt(coral_dir, "abc123", island_id="1")
    assert current is not None
    assert current.score == 0.9
    assert current.metadata["island_id"] == "1"
    assert current.metadata["grader_key"] == "grader_value"
    assert current.metadata["scores"] == {}
    assert read_eval_count(coral_dir, island_id="0") == 0
    assert read_eval_count(coral_dir, island_id="1") == 1
    assert read_eval_count(coral_dir) == 1
    assert not (coral_dir / "islands" / "0" / "eval_logs" / "abc123").exists()
    assert (coral_dir / "islands" / "1" / "eval_logs" / "abc123" / "metrics.json").exists()


def test_find_pending_single_island_unchanged(tmp_path):
    """Single-island: _find_pending scans only public/attempts/."""
    from coral.grader.daemon import _find_pending
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    coral_dir = tmp_path / ".coral"
    (coral_dir / "public" / "attempts").mkdir(parents=True)

    a = Attempt(
        commit_hash="ccc",
        agent_id="agent-1",
        title="x",
        score=None,
        status="pending",
        parent_hash=None,
        timestamp="2026-05-31T10:00:00Z",
    )
    write_attempt(coral_dir, a)
    pending = _find_pending(coral_dir)
    assert {p.commit_hash for p in pending} == {"ccc"}


# --- _append_eval_logs_hint: universal trace-log footer --------------------
#
# Regression: agent attempt e90ca6c3 timed out, feedback was just
# "Evaluation timed out after 1200s" with no eval_logs path. The agent had
# no way to find their trajectory/recording. The daemon now appends a footer
# on every feedback path (success, timeout, crashed) so the agent always
# knows where to look.


def test_append_eval_logs_hint_includes_worktree_relative_path():
    """Footer must contain the worktree-relative form (runtime-agnostic)."""
    from coral.grader.daemon import _append_eval_logs_hint

    out = _append_eval_logs_hint("Eval timed out.", "deadbeef1234", "claude_code")

    assert "deadbeef1234" in out
    assert "eval_logs/deadbeef1234/" in out


def test_append_eval_logs_hint_includes_concrete_shared_dir_path():
    """Footer must also show the concrete `.claude/` form so the agent
    doesn't have to guess what "shared state dir" means."""
    from coral.grader.daemon import _append_eval_logs_hint

    out = _append_eval_logs_hint("", "abc1234", "claude_code")

    assert ".claude/eval_logs/abc1234/" in out


def test_append_eval_logs_hint_preserves_original_feedback():
    """The original feedback (e.g. "Eval timed out after 1200s") must come
    first, footer second. No silent replacement."""
    from coral.grader.daemon import _append_eval_logs_hint

    original = "Evaluation timed out after 1200s"
    out = _append_eval_logs_hint(original, "h1", "claude_code")

    assert out.startswith(original)
    assert "### Trace logs" in out


def test_append_eval_logs_hint_handles_empty_feedback():
    """A blank feedback (rare but possible from a crashed grader that raised
    an empty exception) must still get the footer."""
    from coral.grader.daemon import _append_eval_logs_hint

    out = _append_eval_logs_hint("", "h1", "claude_code")

    assert "eval_logs/h1/" in out


def test_append_eval_logs_hint_uses_correct_shared_dir_per_runtime():
    """For other runtimes, the concrete path uses the matching shared dir
    (.codex, .opencode, .kiro). Verified via the registry path; the fallback
    map mirrors shared_dir_name in coral/agent/builtin/*."""
    from coral.grader.daemon import _append_eval_logs_hint

    for runtime, expected in [
        ("claude_code", ".claude"),
        ("codex", ".codex"),
        ("opencode", ".opencode"),
        ("kiro", ".kiro"),
    ]:
        out = _append_eval_logs_hint("", "h1", runtime)
        assert f"{expected}/eval_logs/h1/" in out, f"{runtime} should produce {expected}/ path"


def test_append_eval_logs_hint_unknown_runtime_falls_back_to_claude():
    """A typo'd or custom runtime name shouldn't crash — fall back to .claude."""
    from coral.grader.daemon import _append_eval_logs_hint

    out = _append_eval_logs_hint("feedback", "h1", "totally-made-up-runtime")

    assert ".claude/eval_logs/h1/" in out
    assert "feedback" in out  # original preserved


def test_grader_run_history_records_each_actual_invocation(tmp_path):
    repo = _init_repo_and_coral(tmp_path, score=0.42)
    (repo / "main.py").write_text("print('history')\n")
    pending = submit_eval(
        message="history", agent_id="agent-1", workdir=str(repo), wait=False
    )

    process_pending_once(repo / ".coral")
    finalized = read_attempt(repo / ".coral", pending.commit_hash)

    assert finalized is not None
    assert finalized.metadata["grader_invocation_count"] == 1
    assert len(finalized.metadata["grader_runs"]) == 1
    run = finalized.metadata["grader_runs"][0]
    assert run["status"] == "improved"
    assert run["failure_class"] is None
    assert run["finished_at"] is not None
    assert run["log_dir"].startswith(f"eval_logs/{pending.commit_hash}/")


def test_checkout_failure_is_retryable_infrastructure(tmp_path, monkeypatch):
    repo = _init_repo_and_coral(tmp_path)
    (repo / "main.py").write_text("print('checkout')\n")
    pending = submit_eval(
        message="checkout", agent_id="agent-1", workdir=str(repo), wait=False
    )
    monkeypatch.setattr(
        "coral.grader.daemon._add_isolated_worktree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("git checkout failed")),
    )

    process_pending_once(repo / ".coral")
    finalized = read_attempt(repo / ".coral", pending.commit_hash)

    assert finalized is not None
    assert finalized.status == "crashed"
    assert finalized.metadata["failure_class"] == "infra"
    assert finalized.metadata["grader_runs"][0]["failure_class"] == "infra"


def test_outer_grader_exception_is_not_retryable_without_infra_evidence(tmp_path):
    repo = _init_repo_and_coral(tmp_path)
    _write_grader(
        repo,
        "from coral.grader.task_grader import TaskGrader\n"
        "class Grader(TaskGrader):\n"
        "    def evaluate(self):\n"
        "        raise RuntimeError('worker exploded')\n",
    )
    (repo / "main.py").write_text("print('crash')\n")
    pending = submit_eval(
        message="crash", agent_id="agent-1", workdir=str(repo), wait=False
    )

    process_pending_once(repo / ".coral")
    finalized = read_attempt(repo / ".coral", pending.commit_hash)

    assert finalized is not None
    assert finalized.status == "crashed"
    assert finalized.metadata["failure_class"] == "candidate"
    assert finalized.metadata["grader_runs"][0]["failure_class"] == "candidate"


def _enable_trusted_server(repo: Path) -> None:
    config_path = repo / ".coral" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["grader"].setdefault("args", {})["trusted_submission_validator"] = (
        "unused:validator"
    )
    config_path.write_text(yaml.safe_dump(config))


def _raw_trusted_request(socket_path: Path, operation: str) -> dict:
    import socket

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect(str(socket_path))
        client.sendall(
            json.dumps(
                {"version": 1, "operation": operation, "payload": {}}
            ).encode()
            + b"\n"
        )
        return json.loads(client.recv(4096).split(b"\n", 1)[0])
    finally:
        client.close()


def test_daemon_trusted_socket_stops_and_restarts(tmp_path):
    import threading

    repo = _init_repo_and_coral(tmp_path)
    _enable_trusted_server(repo)
    coral_dir = repo / ".coral"
    socket_path = coral_dir / "public" / "trusted.sock"

    for _ in range(2):
        stop = threading.Event()
        thread = threading.Thread(target=run_daemon, args=(coral_dir, stop))
        thread.start()
        deadline = time.monotonic() + 3
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert socket_path.exists()
        response = _raw_trusted_request(socket_path, "unknown")
        assert response["ok"] is False
        stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not socket_path.exists()


def test_trusted_socket_responds_while_grader_is_running(tmp_path):
    import threading

    repo = _init_repo_and_coral(tmp_path)
    _write_grader(
        repo,
        "import time\n"
        "from coral.grader.task_grader import TaskGrader\n"
        "class Grader(TaskGrader):\n"
        "    def evaluate(self):\n"
        "        time.sleep(1.0)\n"
        "        return 0.5\n",
    )
    (repo / "main.py").write_text("print('slow')\n")
    submit_eval(message="slow", agent_id="agent-1", workdir=str(repo), wait=False)
    _enable_trusted_server(repo)
    coral_dir = repo / ".coral"
    socket_path = coral_dir / "public" / "trusted.sock"
    stop = threading.Event()
    thread = threading.Thread(target=run_daemon, args=(coral_dir, stop))
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        started = time.monotonic()
        response = _raw_trusted_request(socket_path, "unknown")
        elapsed = time.monotonic() - started
        assert response["ok"] is False
        assert elapsed < 0.5
    finally:
        stop.set()
        thread.join(timeout=4)
    assert not thread.is_alive()
