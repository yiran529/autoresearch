"""Lifecycle ownership of detached workflow workers and their model children."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from coral.grader.trusted_writes import process_start_time


def test_recovery_stops_orphan_model_when_worker_is_already_dead(tmp_path):
    from coral.grader.trusted_worker import stop_workflow_worker

    coral = tmp_path / ".coral"
    private = coral / "private"
    private.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (private / "workflow_model.json").write_text(json.dumps({
            "pid": child.pid, "start_time": process_start_time(child.pid), "pgid": child.pid,
        }))
        stop_workflow_worker(coral)
        assert child.wait(timeout=5) < 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_cleanup_does_not_signal_reused_pid(tmp_path):
    from coral.grader.trusted_worker import stop_workflow_worker

    private = tmp_path / ".coral/private"
    private.mkdir(parents=True)
    (private / "workflow_model.json").write_text(json.dumps({
        "pid": os.getpid(), "start_time": process_start_time(os.getpid()) + 1,
        "pgid": os.getpgrp(),
    }))
    stop_workflow_worker(private.parent)
    assert not (private / "workflow_model.json").exists()


def test_model_timeout_reaps_process_group(tmp_path):
    from coral.grader.trusted_worker import run_workflow_model

    with pytest.raises(subprocess.TimeoutExpired):
        run_workflow_model([sys.executable, "-c", "import time; time.sleep(60)"],
                           input="", timeout=0.05)


def test_cleanup_of_exited_worker_is_idempotent(tmp_path):
    from coral.grader.trusted_worker import stop_workflow_worker

    private = tmp_path / ".coral/private"
    private.mkdir(parents=True)
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    record = {"pid": process.pid, "start_time": process_start_time(process.pid), "pgid": process.pid}
    process.wait(timeout=5)
    (private / "workflow_process.json").write_text(json.dumps(record))
    stop_workflow_worker(private.parent)
    stop_workflow_worker(private.parent)
    assert not (private / "workflow_process.json").exists()


def test_normal_model_exit_reaps_term_ignoring_child(tmp_path):
    from coral.grader.trusted_worker import run_workflow_model

    marker = tmp_path / "child.json"
    code = (
        "import subprocess,sys,json,os; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(1,flush=True); time.sleep(60)'],"
        "stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); p.stdout.readline(); "
        f"open({str(marker)!r},'w').write(str(p.pid))"
    )
    result = run_workflow_model([sys.executable, "-c", code], input="", timeout=10)
    assert result.returncode == 0
    pid = int(marker.read_text())
    stat = Path(f"/proc/{pid}/stat")
    try:
        assert not stat.exists() or stat.read_text().split(") ")[1].startswith("Z ")
    finally:
        if stat.exists() and not stat.read_text().split(") ")[1].startswith("Z "):
            os.kill(pid, signal.SIGKILL)


def test_parallel_reconcile_and_stop_cannot_spawn_after_stopping(tmp_path, monkeypatch):
    from coral.grader.trusted_writes import TrustedWriteService

    barrier = threading.Barrier(2)
    entered, release = threading.Event(), threading.Event()
    launches = []

    def handler(*_):
        barrier.wait(timeout=5)
        return {"assignments": []}

    def launch(*_, **__):
        launches.append(1)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(poll=lambda: None, wait=lambda **_: 0)

    config = SimpleNamespace(grader=SimpleNamespace(args={"trusted_workflow_worker": "task:worker"}))
    service = TrustedWriteService(tmp_path / ".coral", config, workflow_handler=handler)
    monkeypatch.setattr("coral.grader.trusted_worker.start_task_worker", launch)
    with ThreadPoolExecutor(max_workers=3) as pool:
        one = pool.submit(service.handle, "reconcile_workflow", {}, os.getppid())
        two = pool.submit(service.handle, "reconcile_workflow", {}, os.getppid())
        assert entered.wait(5)
        stopping = pool.submit(service.stop_workflow)
        release.set()
        assert one.result(timeout=5) == two.result(timeout=5) == {"assignments": []}
        stopping.result(timeout=5)
    service.workflow_handler = lambda *_: {"assignments": []}
    service.handle("reconcile_workflow", {}, os.getppid())
    assert len(launches) == 1


def test_diagnostic_drain_is_cancelled_without_waiting_for_its_lock(tmp_path):
    import fcntl

    from coral.grader.trusted_worker import stop_workflow_worker

    private = tmp_path / ".coral/private"
    task_lock = private / "diagnostics/task/task.lock"
    task_lock.parent.mkdir(parents=True)
    entered = threading.Event()
    released = threading.Event()

    def executor():
        with task_lock.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            entered.set()
            deadline = time.monotonic() + 5
            while not (private / "workflow_stopping").exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
        released.set()  # Represents executor finally/releasing its reservation.

    thread = threading.Thread(target=executor)
    thread.start()
    assert entered.wait(5)
    stop_workflow_worker(private.parent)
    thread.join(5)
    assert released.is_set() and not thread.is_alive()


def test_incomplete_drain_retains_recovery_registration(tmp_path):
    import fcntl

    from coral.grader.trusted_worker import stop_workflow_worker

    private = tmp_path / ".coral/private"
    task_lock = private / "diagnostics/task/task.lock"
    task_lock.parent.mkdir(parents=True)
    registry = private / "workflow_process.json"
    registry.write_text(json.dumps({"pid": 0, "start_time": None, "pgid": 0}))
    with task_lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="draining"):
            stop_workflow_worker(private.parent, drain_timeout=0.02)
    assert registry.exists()
    stop_workflow_worker(private.parent)
    assert not registry.exists()


def test_failed_model_registration_still_reaps_model(tmp_path, monkeypatch):
    from coral.grader import trusted_worker

    observed = []

    def fail(process, **_):
        observed.append(process)
        raise OSError("registration failed")

    monkeypatch.setattr(trusted_worker, "register_workflow_model", fail)
    try:
        with pytest.raises(OSError, match="registration failed"):
            trusted_worker.run_workflow_model(
                [sys.executable, "-c", "import time; time.sleep(60)"], input="")
        assert observed[0].poll() is not None
    finally:
        if observed and observed[0].poll() is None:
            os.killpg(observed[0].pid, signal.SIGKILL)
            observed[0].wait()


def test_worker_death_between_spawn_and_registration_is_recoverable(tmp_path):
    from coral.grader.trusted_worker import stop_workflow_worker

    coral = tmp_path / ".coral"
    coral.mkdir()
    code = (
        "import os,subprocess,sys,time; "
        "from coral.grader.trusted_worker import prepare_workflow_model,workflow_model_environment; "
        "env=workflow_model_environment(); prepare_workflow_model(env); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "print(p.pid,flush=True); time.sleep(60)"
    )
    environment = dict(os.environ, CORAL_WORKFLOW_CORAL_DIR=str(coral))
    worker = subprocess.Popen([sys.executable, "-c", code], env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True)
    child = None
    try:
        child = int(worker.stdout.readline())  # Spawned, deliberately not registered.
        registry = coral / "private/workflow_model.json"
        assert json.loads(registry.read_text())["phase"] == "spawning"
        worker.kill()
        worker.wait(timeout=5)
        stop_workflow_worker(coral)
        stat = Path(f"/proc/{child}/stat")
        assert not stat.exists() or stat.read_text().split(") ")[1].startswith("Z ")
        assert not registry.exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
