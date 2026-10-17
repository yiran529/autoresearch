"""Codex startup hooks are argv-only and never block runtime startup."""

from __future__ import annotations

import json

from coral.agent.builtin.codex import _wrap_startup_hook
from coral.agent.startup_hook import run_startup_hook


def test_hook_failure_still_executes_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr("coral.agent.startup_hook.subprocess.run", lambda hook, check: (
        calls.append(("hook", tuple(hook)))
    ))
    monkeypatch.setattr("coral.agent.startup_hook.os.execvpe", lambda binary, argv, env: (
        calls.append(("exec", binary, tuple(argv)))
    ))
    run_startup_hook(["hook", "--fail"], ["codex", "exec"])
    assert calls == [
        ("hook", ("hook", "--fail")),
        ("exec", "codex", ("codex", "exec")),
    ]


def test_codex_wrapper_keeps_hook_out_of_runtime_options():
    hook = [".venv/bin/autoresearch", "consolidate-memory", "--if-due", "--async"]
    wrapped = _wrap_startup_hook(["codex", "exec", "prompt"], hook)
    assert wrapped[-3:] == ["codex", "exec", "prompt"]
    assert json.loads(wrapped[wrapped.index("--hook-json") + 1]) == hook


def test_hook_oserror_still_executes_runtime(monkeypatch):
    calls = []

    def fail_hook(*args, **kwargs):
        raise OSError("hook executable missing")

    monkeypatch.setattr("coral.agent.startup_hook.subprocess.run", fail_hook)
    monkeypatch.setattr(
        "coral.agent.startup_hook.os.execvpe",
        lambda binary, argv, env: calls.append((binary, tuple(argv))),
    )
    run_startup_hook(["missing-hook"], ["codex", "exec"])
    assert calls == [("codex", ("codex", "exec"))]
