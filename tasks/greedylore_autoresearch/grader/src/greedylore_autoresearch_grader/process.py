"""Process-group execution with deterministic evidence and cleanup."""

from __future__ import annotations

import os
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
            next_monitor = 0.0
            while process.poll() is None:
                if check_cancelled:
                    check_cancelled()
                now = time.monotonic()
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
    )
