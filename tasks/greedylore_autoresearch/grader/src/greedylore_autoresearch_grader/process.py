"""Process-group execution with deterministic evidence and cleanup."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .runtime import InfraFailure


@dataclass(frozen=True)
class ProcessSpec:
    command: tuple[str, ...]
    log_dir: Path
    env: Mapping[str, str]
    env_allowlist: tuple[str, ...]
    timeout: float | None
    terminate_grace: float = 15.0
    cwd: Path | None = None
    startup_timeout: float | None = None
    startup_marker: str | None = None

    @property
    def argv(self) -> tuple[str, ...]:
        return self.command


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    env: Mapping[str, str]
    returncode: int | None
    started_at: str
    finished_at: str
    timed_out: bool
    cleanup_signals: tuple[str, ...]
    stdout_log: str
    stderr_log: str
    stderr_tail: str
    infra_failure: str | None = None
    failure_phase: str | None = None


def run_process_group(spec: ProcessSpec, *, check_cancelled: Callable | None = None,
                      monitor: Callable | None = None,
                      monitor_interval: float = 1, on_start=None, on_finish=None) -> ProcessResult:
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stdout_path = spec.log_dir / f"{token}.stdout.log"
    stderr_path = spec.log_dir / f"{token}.stderr.log"
    started = datetime.now(UTC).isoformat()
    timed_out = False
    infra_failure = None
    cleanup: list[str] = []
    process: subprocess.Popen[bytes] | None = None
    failure_phase = None
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
    environment.update(spec.env)
    if check_cancelled:
        check_cancelled()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            list(spec.command),
            stdout=stdout,
            stderr=stderr,
            env=environment,
            cwd=spec.cwd,
            start_new_session=True,
        )
        try:
            if on_start:
                on_start(process.pid)
            deadline = None if spec.timeout is None else time.monotonic() + spec.timeout
            startup_deadline = (None if spec.startup_timeout is None or not spec.startup_marker
                                else time.monotonic() + spec.startup_timeout)
            startup_complete = startup_deadline is None
            next_monitor = 0.0
            next_log_check = 0.0

            def tail(path):
                with path.open('rb') as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 65536))
                    return stream.read().decode(errors='replace')

            def check_logs():
                nonlocal startup_complete, failure_phase
                lines = [re.sub(r'^\[rank[0-9]+\]:\s*', '', line)
                         for line in tail(stderr_path).splitlines()]
                if 'OSError: AF_UNIX path too long' in lines:
                    failure_phase = 'runtime_ipc'
                    raise InfraFailure('OSError: AF_UNIX path too long')
                if any(line.startswith(('RuntimeError: DataLoader worker',
                                        'RuntimeError: DataLoader timed out after')) for line in lines):
                    failure_phase = 'data_loading'
                    return True
                if not startup_complete and spec.startup_marker in tail(stdout_path):
                    startup_complete = True
                return False

            while process.poll() is None:
                if check_cancelled:
                    check_cancelled()
                now = time.monotonic()
                if now >= next_log_check:
                    if check_logs():
                        break
                    next_log_check = now + .1
                if not startup_complete and now >= startup_deadline:
                    timed_out, failure_phase = True, 'startup'
                    break
                if monitor is not None and now >= next_monitor:
                    monitor(process.pid)
                    next_monitor = time.monotonic() + monitor_interval
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
            check_logs()
        except InfraFailure as exc:
            infra_failure = str(exc)
        except Exception as exc:
            infra_failure = f"process monitoring failed: {type(exc).__name__}: {exc}"
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                cleanup.append("SIGTERM")
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + spec.terminate_grace
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    cleanup.append("SIGKILL")
                except ProcessLookupError:
                    pass
            process.wait()
            if on_finish:
                on_finish(process.pid)
        if infra_failure is None:
            try:
                if check_cancelled:
                    check_cancelled()
                if monitor is not None:
                    monitor(process.pid)
            except Exception as exc:
                infra_failure = str(exc)
    with stderr_path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 16_384))
        stderr_tail = stream.read().decode(errors="replace")[-4000:]
    return ProcessResult(
        spec.command,
        {key: spec.env[key] for key in spec.env_allowlist if key in spec.env},
        process.returncode if process else None,
        started,
        datetime.now(UTC).isoformat(),
        timed_out,
        tuple(cleanup),
        str(stdout_path),
        str(stderr_path),
        stderr_tail,
        infra_failure,
        failure_phase,
    )
