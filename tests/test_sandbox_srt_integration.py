"""Opt-in Linux SRT checks: CORAL_SRT_SMOKE=1 with srt/bwrap/socat on PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from coral.config import CoralConfig
from coral.grader.trusted_writes import TrustedWriteServer, TrustedWriteService
from coral.sandbox.protocol import AgentSandboxContext
from coral.sandbox.srt import SrtSandbox

pytestmark = pytest.mark.skipif(
    os.environ.get("CORAL_SRT_SMOKE") != "1" or sys.platform != "linux",
    reason="requires explicit opt-in and Linux SRT dependencies",
)


def test_formal_srt_writes_git_ipc_and_private_boundaries():
    task = Path(__file__).resolve().parents[1] / "tasks/greedylore_autoresearch/task.yaml"
    config = CoralConfig.from_yaml(task)
    config.agents.sandbox.srt_command = [shutil.which("srt") or "srt"]
    # Under HOME, so the real denyRead/allowRead remounts are exercised.
    with tempfile.TemporaryDirectory(prefix="coral-srt-smoke-", dir=Path.home()) as directory:
        run = Path(directory)
        coral = run / ".coral"
        repo = run / "repo"
        worktree = run / "agents/engineer"
        sibling = run / "agents/researcher"
        for path in [coral / "private", coral / "public/attempts", sibling]:
            path.mkdir(parents=True)
        (coral / "private/secret").write_text("hidden")
        (sibling / "shared.txt").write_text("readable")
        (coral / "public/attempts/record.json").write_text("{}")
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        (repo / "seed.py").write_text("seed\n")
        subprocess.run(["git", "-C", str(repo), "add", "seed.py"], check=True)
        git_identity = ["-c", "user.name=Sandbox Smoke", "-c", "user.email=smoke@example.test"]
        subprocess.run(["git", "-C", str(repo), *git_identity, "commit", "-qm", "seed"], check=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "--quiet",
                        "-b", "engineer", str(worktree)], check=True)
        service = TrustedWriteService(coral, config)
        server = TrustedWriteServer(coral, service)
        provider = SrtSandbox(config.agents.sandbox)
        server.start()
        provider.start()
        try:
            spec = provider.prepare_agent(AgentSandboxContext(
                "engineer", worktree, coral, repo, ".codex", [worktree, sibling]))
            probe = r'''
import json, socket, subprocess, sys
from pathlib import Path
run = Path(sys.argv[1]); w = Path.cwd()
status = subprocess.check_output(["git", "status", "--porcelain"])
assert status == b"", status.decode()
(w / "seed.py").write_text("changed\n")
subprocess.run(["git", "add", "seed.py"], check=True)
subprocess.run(["git", "-c", "user.name=Sandbox Smoke", "-c", "user.email=smoke@example.test",
                "commit", "-qm", "sandbox write"], check=True)
assert (run / "agents/researcher/shared.txt").read_text() == "readable"
for path in [run / ".coral/public/attempts/record.json",
             run / "agents/researcher/shared.txt"]:
    try:
        path.write_text("forbidden")
    except OSError:
        pass
    else:
        raise AssertionError(f"unexpected write: {path}")
try:
    (run / ".coral/private/secret").read_text()
except OSError:
    pass
else:
    raise AssertionError("private readable")
# SRT may expose a writable, empty tmpfs at a hidden directory. Such
# writes must never reach the host's private evidence (checked host-side).
try:
    (run / ".coral/private/secret").write_text("sandbox-only")
except OSError:
    pass
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(5)
    client.connect(str(run / ".coral/public/trusted.sock"))
    client.sendall(json.dumps({"version":1,"operation":"claim_engineering",
                              "payload":{"node_id":"N000"}}).encode() + b"\n")
    response = json.loads(client.makefile("rb").readline())
    assert response["ok"] is False
    assert "registry" in response["error"], response
print("PASS worktree write, Git commit, sibling read, private hide, shared write denial, trusted IPC/auth")
'''
            result = subprocess.run(
                [*spec.command_prefix, sys.executable, "-c", probe, str(run)],
                cwd=worktree, env={**os.environ, **spec.env},
                capture_output=True, text=True, timeout=45,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "PASS worktree write" in result.stdout
            assert (coral / "private/secret").read_text() == "hidden"
            print(result.stdout.strip())
        finally:
            provider.stop()
            server.stop()
