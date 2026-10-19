"""Invoke task-owned trusted validators inside the grader-private Python."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from coral.grader.timeouts import TRUSTED_WORKER_TIMEOUT
from coral.workspace.grader_env import grader_python_path


def workflow_model_environment(environment=None):
    result = dict(os.environ if environment is None else environment)
    result["CORAL_WORKFLOW_PROCESS_TOKEN"] = uuid.uuid4().hex
    return result


def _process_record(process: subprocess.Popen, token=None) -> dict:
    from coral.grader.trusted_writes import process_start_time

    if token is None:
        for item in Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0"):
            if item.startswith(b"CORAL_WORKFLOW_PROCESS_TOKEN="):
                token = item.split(b"=", 1)[1].decode()
    return {"pid": process.pid, "start_time": process_start_time(process.pid),
            "pgid": os.getpgid(process.pid), "token": token}


def _group_members(pgid):
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid and int(fields[3]) == pgid and fields[0] != "Z":
                members.append(directory)
        except (OSError, ValueError, IndexError):
            continue
    return members


def _stop_record(record: dict, *, timeout: float = 5) -> None:
    from coral.grader.trusted_writes import TrustedWriteError, process_start_time

    pid, pgid = record.get("pid"), record.get("pgid")
    if not isinstance(pid, int) or record.get("start_time") is None:
        return
    try:
        current = process_start_time(pid)
    except TrustedWriteError:
        if Path(f"/proc/{pid}").exists():
            raise
        current = None
    if current is not None and current != record["start_time"]:
        return  # PID has been reused; it is no longer ours.
    if not isinstance(pgid, int) or pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError("refusing to stop an unverified workflow process group")
    members = _group_members(pgid)
    if not members:
        return
    try:
        verified = current == record["start_time"] and os.getpgid(pid) == pgid
    except ProcessLookupError:
        verified = False
    if not verified and record.get("token"):
        marker = f"CORAL_WORKFLOW_PROCESS_TOKEN={record['token']}".encode()
        for member in members:
            try:
                if marker in (member / "environ").read_bytes().split(b"\0"):
                    verified = True
                    break
            except FileNotFoundError:
                continue
    if not verified:
        raise RuntimeError("workflow descendants remain but their identity cannot be verified")
    try:
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _group_members(pgid):
            time.sleep(0.02)
        if _group_members(pgid):
            os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and _group_members(pgid):
                time.sleep(0.01)
        if _group_members(pgid):
            raise RuntimeError("workflow process group did not stop; recovery registration retained")
    except ProcessLookupError:
        pass


def _stop_prepared_models(token: str) -> None:
    """Find a model spawned after preparation but before its PID was registered."""
    marker = f"CORAL_WORKFLOW_PROCESS_TOKEN={token}".encode()
    groups = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            pgid = int(fields[2])
            if pgid == int(fields[3]) and fields[0] != "Z":
                groups[pgid] = {"pid": int(directory.name), "start_time": int(fields[19]),
                                "pgid": pgid, "token": token}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    for record in groups.values():
        _stop_record(record)


def prepare_workflow_model(environment: dict) -> Path | None:
    from coral.grader.trusted_writes import _atomic_json

    registry = os.environ.get("CORAL_WORKFLOW_CORAL_DIR")
    if not registry:
        return None
    path = Path(registry) / "private/workflow_model.json"
    _atomic_json(path, {"phase": "spawning", "token": environment["CORAL_WORKFLOW_PROCESS_TOKEN"]})
    return path


def stop_workflow_worker(coral_dir: str | Path, *, drain_timeout: float = 5) -> None:
    """Recover a previous service's worker and orphan model before starting anew."""
    private = Path(coral_dir) / "private"
    if not private.exists():
        return
    (private / "workflow_stopping").touch()
    # Models have their own session so worker death cannot hide live descendants.
    for name in ("workflow_process.json", "workflow_model.json"):
        path = private / name
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if record.get("phase") == "spawning":
            _stop_prepared_models(record["token"])
        else:
            _stop_record(record)
    # Diagnostic executors own separate process groups and hold this lock while
    # queued/running. Revocation is visible without acquiring the held lock.
    deadline = time.monotonic() + drain_timeout
    for path in private.glob("diagnostics/*/task.lock"):
        with path.open("a+b") as stream:
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("diagnostic execution is still draining; workflow registration retained") from None
                    time.sleep(0.02)
    for name in ("workflow_process.json", "workflow_model.json"):
        (private / name).unlink(missing_ok=True)


def register_workflow_model(process: subprocess.Popen, *, token=None) -> Path | None:
    from coral.grader.trusted_writes import _atomic_json

    if os.getpgid(process.pid) != process.pid:
        raise RuntimeError("workflow model must own an isolated process group")
    process._workflow_record = _process_record(process, token)
    registry = os.environ.get("CORAL_WORKFLOW_CORAL_DIR")
    if not registry:
        return None
    path = Path(registry) / "private/workflow_model.json"
    _atomic_json(path, process._workflow_record)
    return path


def stop_workflow_model(process: subprocess.Popen, *, token=None) -> None:
    record = getattr(process, "_workflow_record", None)
    if record is None:
        if token is not None:
            _stop_prepared_models(token)
            process.wait()
        return
    _stop_record(record)
    process.wait()


def run_workflow_model(command, *, input: str, timeout: float = 18000):
    """Bound model execution and retain enough identity to recover its process group."""
    environment = workflow_model_environment()
    record = prepare_workflow_model(environment)
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True,
                                   env=environment)
        register_workflow_model(process, token=environment["CORAL_WORKFLOW_PROCESS_TOKEN"])
        stdout, stderr = process.communicate(input, timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process is not None:
            stop_workflow_model(process, token=environment["CORAL_WORKFLOW_PROCESS_TOKEN"])
        if record is not None:
            record.unlink(missing_ok=True)

_WORKER = r"""
import importlib, json, sys, traceback

try:
    request = json.loads(sys.stdin.read())
    module_name, function_name = request["entrypoint"].rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    result = function(*request.get("args", []), **request.get("kwargs", {}))
    if not isinstance(result, dict):
        raise TypeError("trusted validator must return an object")
    sys.stdout.write(json.dumps({"ok": True, "result": result}))
except Exception as exc:
    sys.stderr.write(traceback.format_exc())
    sys.stdout.write(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }))
"""


def invoke_task_validator(
    coral_dir: str | Path,
    entrypoint: str,
    *args: Any,
    timeout: float = TRUSTED_WORKER_TIMEOUT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call one configured function in the already-created grader venv."""
    if not entrypoint or ":" not in entrypoint:
        raise ValueError("trusted validator entrypoint must be 'module:function'")
    python = grader_python_path(Path(coral_dir))
    if not python.exists():
        raise RuntimeError(f"grader-private Python is unavailable at {python}")
    completed = subprocess.run(
        [str(python), "-c", _WORKER],
        input=json.dumps({"entrypoint": entrypoint, "args": args, "kwargs": kwargs}),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"trusted validator worker exited {completed.returncode}: "
            f"{completed.stderr.strip()[-1000:]}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("trusted validator returned invalid JSON") from exc
    if not response.get("ok"):
        raise RuntimeError(
            f"trusted validator rejected request: {response.get('error', 'unknown error')}"
        )
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("trusted validator returned a non-object result")
    return result


def start_task_worker(
    coral_dir: str | Path, entrypoint: str, *, log_name: str = "memory_worker.err",
) -> subprocess.Popen:
    """Launch one grader-owned task function without waiting for completion."""
    if not entrypoint or ":" not in entrypoint:
        raise ValueError("trusted worker entrypoint must be 'module:function'")
    coral_dir = Path(coral_dir).resolve()
    python = grader_python_path(coral_dir)
    if not python.exists():
        raise RuntimeError(f"grader-private Python is unavailable at {python}")
    log_root = coral_dir / "public" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    if Path(log_name).name != log_name:
        raise ValueError("trusted worker log must be a filename")
    environment = workflow_model_environment()
    workflow = log_name == "workflow_worker.err"
    if workflow:
        stop_workflow_worker(coral_dir)
        (coral_dir / "private/workflow_stopping").unlink(missing_ok=True)
        environment["CORAL_WORKFLOW_CORAL_DIR"] = str(coral_dir)
    else:
        environment.pop("CORAL_WORKFLOW_CORAL_DIR", None)
    log_file = (log_root / log_name).open("ab")
    try:
        process = subprocess.Popen(
            [str(python), "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            start_new_session=True,
            env=environment,
        )
    finally:
        log_file.close()
    if process.stdin is None:
        raise RuntimeError("trusted worker stdin is unavailable")
    if workflow:
        from coral.grader.trusted_writes import _atomic_json

        _atomic_json(coral_dir / "private/workflow_process.json",
                     _process_record(process, environment["CORAL_WORKFLOW_PROCESS_TOKEN"]))
    process.stdin.write(json.dumps({
        "entrypoint": entrypoint,
        "args": [str(coral_dir)],
        "kwargs": {},
    }).encode())
    process.stdin.close()
    return process


def main() -> None:
    """Small module CLI, useful for diagnostics without importing task code in host."""
    request = json.loads(sys.stdin.read())
    module_name, function_name = request["entrypoint"].rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    result = function(*request.get("args", []), **request.get("kwargs", {}))
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
