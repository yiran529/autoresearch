"""Opt-in Linux SRT checks: CORAL_SRT_SMOKE=1 with srt/bwrap/socat on PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
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


@pytest.mark.parametrize('legacy_mount', [False, True])
def test_formal_srt_writes_git_ipc_and_private_boundaries(legacy_mount):
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
        if legacy_mount:
            (coral / "public/research_diagnostics").touch(mode=0o444)
        publication_root = coral / ("public/diagnostics/research_probes" if legacy_mount
                                    else "public/research_diagnostics")
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
             run / ".coral/public/research_diagnostics/agent-write.json",
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
import time
(w / ".publication-ready").touch()
deadline = time.monotonic() + 30
while not (w / ".publication-published").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("host did not publish diagnostic")
    time.sleep(0.02)
diagnostic = Path(sys.argv[2]) / "N001/probe/result.json"
assert json.loads(diagnostic.read_text()) == {"status": "completed"}
try:
    diagnostic.write_text("forbidden")
except OSError:
    pass
else:
    raise AssertionError("published diagnostic became agent-writable")
print("PASS worktree write, Git commit, sibling read, private hide, shared write denial, trusted IPC/auth, live publication")
'''
            process = subprocess.Popen(
                [*spec.command_prefix, sys.executable, "-c", probe, str(run), str(publication_root)],
                cwd=worktree, env={**os.environ, **spec.env},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 30
                while not (worktree / ".publication-ready").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        stdout, stderr = process.communicate(timeout=5)
                        pytest.fail(stdout + stderr)
                    time.sleep(0.02)
                # Publish while the sandbox still has its protected mounts.
                diagnostic = publication_root / "N001/probe/result.json"
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                diagnostic.write_text('{"status":"completed"}\n')
                (worktree / ".publication-published").touch()
                stdout, stderr = process.communicate(timeout=45)
                assert process.returncode == 0, stdout + stderr
                assert "PASS worktree write" in stdout
                assert (coral / "private/secret").read_text() == "hidden"
                assert diagnostic.read_text() == '{"status":"completed"}\n'
                print(stdout.strip())
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
        finally:
            provider.stop()
            server.stop()
