import os
import sys
from pathlib import Path

from greedylore_autoresearch_grader.process import ProcessSpec, run_process_group


def test_process_records_logs_and_allowlisted_env(tmp_path):
    result = run_process_group(
        ProcessSpec(
            (sys.executable, "-c", "print('ok')"),
            tmp_path,
            {"TOKEN": "secret", "SAFE": "yes"},
            ("SAFE",),
            5,
        )
    )
    assert result.returncode == 0
    assert result.env == {"SAFE": "yes"}
    assert Path(result.stdout_log).read_text().strip() == "ok"
    assert "TOKEN" not in result.env


def test_reported_ipc_failure_stops_a_live_process_and_preserves_cause(tmp_path):
    code = "import sys,time; print('OSError: AF_UNIX path too long',file=sys.stderr,flush=True); time.sleep(30)"
    result = run_process_group(ProcessSpec((sys.executable, '-c', code), tmp_path,
        {}, (), 5, terminate_grace=.1))
    assert result.infra_failure == 'OSError: AF_UNIX path too long'
    assert result.failure_phase == 'runtime_ipc'
    assert not result.timed_out
    assert 'AF_UNIX' in Path(result.stderr_log).read_text()


def test_startup_timeout_is_bounded_and_does_not_classify_unknown_failure_as_infra(tmp_path):
    result = run_process_group(ProcessSpec((sys.executable, '-c', 'import time; time.sleep(30)'),
        tmp_path, {}, (), 5, terminate_grace=.1, startup_timeout=.15, startup_marker='WARMUP step=1 '))
    assert result.timed_out and result.failure_phase == 'startup'
    assert result.infra_failure is None


def test_startup_marker_allows_measurement_and_generic_candidate_errors_stay_candidate(tmp_path):
    code = "import time,sys; print('WARMUP step=1 loss=1',flush=True); time.sleep(.3); print('RuntimeError: invalid algorithm',file=sys.stderr); sys.exit(1)"
    result = run_process_group(ProcessSpec((sys.executable, '-c', code), tmp_path,
        {}, (), 5, terminate_grace=.1, startup_timeout=.2, startup_marker='WARMUP step=1 '))
    assert result.returncode == 1 and not result.timed_out
    assert result.infra_failure is None


def test_dataloader_failure_stops_waiting_without_converting_candidate_error_to_infra(tmp_path):
    code = "import sys,time; print('RuntimeError: DataLoader worker (pid 123) exited unexpectedly',file=sys.stderr,flush=True); time.sleep(30)"
    result = run_process_group(ProcessSpec((sys.executable, '-c', code), tmp_path,
        {}, (), 5, terminate_grace=.1))
    assert result.failure_phase == 'data_loading' and not result.timed_out
    assert result.infra_failure is None


def test_timeout_cleans_entire_process_group(tmp_path):
    code = "import subprocess,time,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid, flush=True); time.sleep(60)"
    result = run_process_group(
        ProcessSpec((sys.executable, "-c", code), tmp_path, {}, (), 0.2, terminate_grace=0.1)
    )
    assert result.timed_out
    assert "SIGTERM" in result.cleanup_signals
    child = int(Path(result.stdout_log).read_text().splitlines()[0])
    try:
        os.kill(child, 0)
    except ProcessLookupError:
        pass
    else:
        # A killed child may remain briefly as a zombie, but must not be running.
        assert Path(f"/proc/{child}/stat").read_text().split()[2] == "Z"


def test_process_drops_ambient_nccl_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    code = "import os; print(os.environ.get('NCCL_ALGO', 'unset')); print(os.environ['NCCL_SHM_DISABLE'])"
    result = run_process_group(ProcessSpec(
        (sys.executable, "-c", code), tmp_path,
        {"NCCL_SHM_DISABLE": "0"}, ("NCCL_SHM_DISABLE",), 5,
    ))
    assert Path(result.stdout_log).read_text().splitlines() == ["unset", "0"]


def test_monitor_interference_cleans_own_group_and_returns_infra(tmp_path):
    from greedylore_autoresearch_grader.runtime import InfraFailure

    def monitor(pid):
        assert pid != os.getpid()
        raise InfraFailure("timing interference: GPU-0 external PID 123")

    result = run_process_group(
        ProcessSpec((sys.executable, "-c", "import time; time.sleep(60)"),
                    tmp_path, {}, (), 5, terminate_grace=0.1),
        monitor=monitor, monitor_interval=0.01,
    )
    assert "interference" in result.infra_failure
    assert not result.timed_out
    assert "SIGTERM" in result.cleanup_signals


def test_cancellation_during_execution_cleans_descendants(tmp_path):
    import time

    from greedylore_autoresearch_grader.runtime import InfraFailure

    deadline = time.monotonic() + 0.15

    def cancel():
        if time.monotonic() >= deadline:
            raise InfraFailure("cancelled")

    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    result = run_process_group(
        ProcessSpec((sys.executable, "-c", code), tmp_path, {}, (), 5, terminate_grace=0.1),
        check_cancelled=cancel,
    )
    assert result.infra_failure == "cancelled"
    child = int(Path(result.stdout_log).read_text())
    proc = Path(f"/proc/{child}/stat")
    assert not proc.exists() or proc.read_text().split()[2] == "Z"


def test_monitor_failure_after_success_cannot_publish_clean_execution(tmp_path):
    from greedylore_autoresearch_grader.runtime import InfraFailure

    def monitor(pid):
        if not Path(f"/proc/{pid}").exists():
            raise InfraFailure("final inventory unavailable")

    result = run_process_group(
        ProcessSpec((sys.executable, "-c", "pass"), tmp_path, {}, (), 5),
        monitor=monitor,
    )
    assert result.infra_failure == "final inventory unavailable"


def test_lifecycle_observes_parent_identity_and_restores_signals():
    import signal

    import pytest
    from greedylore_autoresearch_grader.runtime import InfraFailure, Lifecycle, ProcessIdentity

    original = signal.getsignal(signal.SIGTERM)
    with Lifecycle() as lifecycle:
        lifecycle.check()
        lifecycle.parent = ProcessIdentity(os.getppid(), os.getuid(), -1, 0, 0)
        with pytest.raises(InfraFailure, match="parent|daemon"):
            lifecycle.check()
        signal.raise_signal(signal.SIGTERM)
        with pytest.raises(InfraFailure, match="cancelled"):
            lifecycle.check()
    assert signal.getsignal(signal.SIGTERM) == original


def test_lifecycle_cancels_worker_after_real_parent_exits(tmp_path):
    import subprocess
    import time

    import greedylore_autoresearch_grader

    source = str(Path(greedylore_autoresearch_grader.__file__).parents[1])
    worker = """
import pathlib,sys,time
from greedylore_autoresearch_grader.runtime import Lifecycle, InfraFailure
root = pathlib.Path(sys.argv[1])
with Lifecycle() as guard:
    (root/'ready').write_text('yes')
    try:
        while True:
            guard.check()
            time.sleep(.01)
    except InfraFailure as exc:
        (root/'done').write_text(str(exc))
"""
    parent_code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); print(p.pid,flush=True); sys.stdin.read()"
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, worker, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": source},
    )
    child = None
    try:
        child = int(parent.stdout.readline())
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "ready").exists()
        parent.terminate()
        parent.wait(timeout=5)
        while not (tmp_path / "done").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert "parent daemon" in (tmp_path / "done").read_text()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child is not None and not (tmp_path / "done").exists():
            import signal
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_lifecycle_rejects_reparenting_before_worker_initializes():
    import pytest
    from greedylore_autoresearch_grader.runtime import InfraFailure, Lifecycle

    # A trusted launch identity differs from the process that adopted the worker.
    with pytest.raises(InfraFailure, match="parent|daemon"):
        with Lifecycle(expected_parent={"pid": os.getppid(), "start_time": -1}):
            pass
