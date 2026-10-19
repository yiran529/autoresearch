"""Invoke task-owned trusted validators inside the grader-private Python."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from coral.grader.timeouts import TRUSTED_WORKER_TIMEOUT
from coral.workspace.grader_env import grader_python_path

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
    log_file = (log_root / log_name).open("ab")
    try:
        process = subprocess.Popen(
            [str(python), "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            start_new_session=True,
        )
    finally:
        log_file.close()
    if process.stdin is None:
        raise RuntimeError("trusted worker stdin is unavailable")
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
