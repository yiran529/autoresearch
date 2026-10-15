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
