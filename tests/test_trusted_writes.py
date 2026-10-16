from __future__ import annotations

import os
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from coral.grader.trusted_writes import (
    AgentBinding,
    TrustedWriteError,
    TrustedWriteServer,
    TrustedWriteService,
    process_start_time,
    request_trusted,
    resolve_peer_binding,
    write_trusted_agent_registry,
)
from coral.hub.attempts import read_attempt, read_attempts, write_attempt


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "model.py").write_text("baseline = True\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "base")
    return path, _git(path, "rev-parse", "HEAD")


def _commit(repo: Path, text: str) -> tuple[str, str]:
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "submission.yaml").write_text(text)
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", text.strip())
    return _git(repo, "rev-parse", "HEAD"), parent


def _canonical(node_id: str = "N001") -> dict[str, object]:
    return {
        "mode": "research",
        "node_id": node_id,
        "node_type": "research",
        "base_commit": "b" * 40,
        "quality_anchor": None,
        "contract_hash": "contract-v1",
        "max_infra_retries": 2,
        "submission_hash": "submission-v1",
    }


def _service(coral_dir: Path, binding: AgentBinding, *, node_id: str = "N001"):
    return TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda _coral_dir, _pid: binding,
        submission_validator=lambda _sha, _ctx: _canonical(node_id),
        artifact_validator=lambda _kind, _content, _ctx: {
            "relative_path": "nodes/N001/evaluation.md"
        },
        checkpoint_fn=lambda *_args, **_kwargs: "shared-state",
    )


def _payload(binding: AgentBinding, commit: str, parent: str) -> dict[str, object]:
    return {
        "commit_hash": commit,
        "agent_id": binding.agent_id,
        "title": "research N001",
        "parent_hash": parent,
        "budget_class": "real",
        "island_id": binding.island_id,
    }


def test_registry_resolves_peer_by_process_ancestry(tmp_path):
    coral_dir = tmp_path / ".coral"
    binding = AgentBinding(
        agent_id="researcher-0",
        root_pid=os.getpid(),
        start_time=process_start_time(os.getpid()),
        worktree=str(tmp_path),
        island_id=None,
    )
    write_trusted_agent_registry(coral_dir, [binding])

    assert resolve_peer_binding(coral_dir, os.getpid()) == binding


def test_registry_rejects_stale_process_start_time(tmp_path):
    coral_dir = tmp_path / ".coral"
    binding = AgentBinding(
        agent_id="researcher-0",
        root_pid=os.getpid(),
        start_time=process_start_time(os.getpid()) + 1,
        worktree=str(tmp_path),
        island_id=None,
    )
    write_trusted_agent_registry(coral_dir, [binding])

    with pytest.raises(TrustedWriteError, match="registered Agent"):
        resolve_peer_binding(coral_dir, os.getpid())


def test_submit_derives_identity_and_rejects_same_uid_spoof(tmp_path):
    coral_dir = tmp_path / ".coral"
    engineer_repo, _ = _init_repo(tmp_path / "engineer")
    engineer_sha, engineer_parent = _commit(engineer_repo, "engineer")
    researcher_repo, _ = _init_repo(tmp_path / "researcher")
    researcher_sha, _ = _commit(researcher_repo, "researcher")
    binding = AgentBinding(
        "engineer-0", os.getpid(), process_start_time(os.getpid()), str(engineer_repo), None
    )
    service = _service(coral_dir, binding)

    forged = _payload(binding, engineer_sha, engineer_parent)
    forged["agent_id"] = "researcher-0"
    with pytest.raises(TrustedWriteError, match="agent_id"):
        service.handle("submit_attempt", forged, os.getpid())

    cross_worktree = _payload(binding, researcher_sha, engineer_parent)
    with pytest.raises(TrustedWriteError, match="current HEAD"):
        service.handle("submit_attempt", cross_worktree, os.getpid())


def test_submit_reads_canonical_metadata_and_writes_attempt(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _ = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )

    result = _service(coral_dir, binding).handle(
        "submit_attempt", _payload(binding, commit, parent), os.getpid()
    )

    assert result["commit_hash"] == commit
    assert result["agent_id"] == "researcher-0"
    assert result["metadata"]["node_id"] == "N001"
    assert result["metadata"]["mode"] == "research"
    assert len(read_attempts(coral_dir)) == 1


def test_research_uniqueness_check_and_write_share_one_lock(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, base = _init_repo(tmp_path / "repo")
    wt1 = tmp_path / "agent-1"
    wt2 = tmp_path / "agent-2"
    _git(repo, "worktree", "add", "-b", "agent-1", str(wt1), base)
    _git(repo, "worktree", "add", "-b", "agent-2", str(wt2), base)
    sha1, parent1 = _commit(wt1, "one")
    sha2, parent2 = _commit(wt2, "two")
    bindings = [
        AgentBinding("agent-1", 101, 1, str(wt1), None),
        AgentBinding("agent-2", 202, 1, str(wt2), None),
    ]

    def submit(index: int) -> bool:
        binding = bindings[index]
        service = _service(coral_dir, binding)
        try:
            service.handle(
                "submit_attempt",
                _payload(binding, (sha1, sha2)[index], (parent1, parent2)[index]),
                binding.root_pid,
            )
            return True
        except TrustedWriteError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (0, 1)))

    assert sorted(results) == [False, True]
    assert len(read_attempts(coral_dir)) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"path": "../../private"},
        {"command": "sh -c id"},
        {"operation": "write_file"},
    ],
)
def test_fixed_operations_reject_untrusted_fields(tmp_path, payload):
    repo, _ = _init_repo(tmp_path / "agent")
    binding = AgentBinding("agent-0", os.getpid(), process_start_time(os.getpid()), str(repo), None)
    with pytest.raises(TrustedWriteError):
        _service(tmp_path / ".coral", binding).handle(
            "submit_attempt", payload, os.getpid()
        )


def test_publish_artifact_rejects_validator_path_escape(tmp_path):
    repo, _ = _init_repo(tmp_path / "agent")
    binding = AgentBinding("agent-0", os.getpid(), process_start_time(os.getpid()), str(repo), None)
    service = TrustedWriteService(
        tmp_path / ".coral",
        SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda *_args: binding,
        artifact_validator=lambda *_args: {"relative_path": "../../private/secret"},
    )
    with pytest.raises(TrustedWriteError, match="artifact target"):
        service.handle(
            "publish_artifact",
            {"kind": "evaluation", "node_id": "N001", "mode": None, "version": None, "content": "x"},
            os.getpid(),
        )


def test_trusted_server_round_trip_permissions_and_restart(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _ = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    write_trusted_agent_registry(coral_dir, [binding])
    server = TrustedWriteServer(coral_dir, _service(coral_dir, binding))

    for _ in range(2):
        server.start()
        try:
            result = request_trusted(coral_dir, "submit_attempt", _payload(binding, commit, parent))
            assert result["commit_hash"] == commit
            assert stat.S_IMODE((coral_dir / "public" / "trusted.sock").stat().st_mode) == 0o600
        finally:
            server.stop()
        assert not (coral_dir / "public" / "trusted.sock").exists()


def test_manager_publishes_and_cleans_private_agent_registry(tmp_path):
    from types import SimpleNamespace

    from coral.agent.manager import AgentManager
    from coral.config import CoralConfig

    config = CoralConfig.from_dict(
        {
            "task": {"name": "t", "description": "d"},
            "workspace": {"repo_path": str(tmp_path / "repo")},
        }
    )
    manager = AgentManager(config)
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    worktree = tmp_path / "agent"
    worktree.mkdir()
    manager.paths = SimpleNamespace(coral_dir=coral_dir)
    manager.handles = [
        SimpleNamespace(
            agent_id="agent-0",
            process=SimpleNamespace(pid=os.getpid()),
            worktree_path=worktree,
        )
    ]

    manager._write_agent_pids()
    binding = resolve_peer_binding(coral_dir, os.getpid())
    assert binding.agent_id == "agent-0"
    assert binding.worktree == str(worktree.resolve())

    manager._cleanup_pid_file()
    assert not (coral_dir / "private" / "trusted_agents.json").exists()


def test_agent_cannot_use_root_bootstrap_exception(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _ = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "root")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    canonical = _canonical("N000")
    canonical["node_type"] = "seed"
    service = TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda *_args: binding,
        submission_validator=lambda *_args: canonical,
        checkpoint_fn=lambda *_args, **_kwargs: "shared-state",
    )

    with pytest.raises(TrustedWriteError, match="Root bootstrap"):
        service.handle(
            "submit_attempt", _payload(binding, commit, parent), os.getpid()
        )


def test_trusted_host_can_bootstrap_only_fixed_n000(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, source = _init_repo(tmp_path / "root")
    commit, parent = _commit(repo, "root submission")
    assert parent == source
    canonical = _canonical("N000")
    canonical.update(
        node_type="seed", base_commit=source, submission_hash="f" * 64
    )
    (coral_dir / "private").mkdir(parents=True)
    (coral_dir / "private" / "root_bootstrap.json").write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "source_commit": source,
                "node_id": "N000",
                "mode": "research",
                "submission_hash": "f" * 64,
            }
        )
    )
    service = TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        submission_validator=lambda *_args: canonical,
        checkpoint_fn=lambda *_args, **_kwargs: "root-checkpoint",
    )

    attempt = service.submit_root_attempt(worktree=repo, commit_hash=commit)

    assert attempt.agent_id == "root-bootstrap"
    assert attempt.metadata["node_id"] == "N000"
    assert attempt.parent_hash == source


def test_registry_resolves_descendant_process(tmp_path):
    import subprocess

    coral_dir = tmp_path / ".coral"
    binding = AgentBinding(
        "agent-0", os.getpid(), process_start_time(os.getpid()), str(tmp_path), None
    )
    write_trusted_agent_registry(coral_dir, [binding])
    child = subprocess.Popen(["sleep", "5"])
    try:
        assert resolve_peer_binding(coral_dir, child.pid) == binding
    finally:
        child.terminate()
        child.wait(timeout=2)


def test_validator_invocation_uses_grader_private_python(tmp_path):
    import sys

    from coral.grader.trusted_worker import invoke_task_validator

    coral_dir = tmp_path / ".coral"
    package = coral_dir / "private" / "validator_pkg"
    package.mkdir(parents=True)
    (package / "taskvalidator.py").write_text(
        "def validate(value, context):\n"
        "    return {\"value\": value, \"source\": context[\"source\"]}\n"
    )
    bin_dir = coral_dir / "private" / "grader_venv" / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "python"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"export PYTHONPATH=\"{package}${{PYTHONPATH:+:$PYTHONPATH}}\"\n"
        f"exec \"{sys.executable}\" \"$@\"\n"
    )
    wrapper.chmod(0o755)

    result = invoke_task_validator(
        coral_dir, "taskvalidator:validate", "committed", {"source": "private"}
    )

    assert result == {"value": "committed", "source": "private"}


def test_client_rejects_oversized_request_before_connect(tmp_path):
    with pytest.raises(TrustedWriteError, match="64 KiB"):
        request_trusted(
            tmp_path / ".coral",
            "publish_artifact",
            {"content": "x" * (70 * 1024)},
        )


def test_retry_rechecks_bound_worktree_and_head_inside_transaction(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _base = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    service = _service(coral_dir, binding)
    service.handle("submit_attempt", _payload(binding, commit, parent), os.getpid())
    attempt = read_attempt(coral_dir, commit)
    attempt.status = "crashed"
    attempt.metadata["failure_class"] = "infra"
    write_attempt(coral_dir, attempt)
    (repo / "model.py").write_text("moved = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "move head")

    with pytest.raises(TrustedWriteError, match="current HEAD"):
        service.handle("retry_attempt", {"commit_hash": commit}, os.getpid())

    _git(repo, "reset", "--hard", commit)
    attempt = read_attempt(coral_dir, commit)
    attempt.metadata["source_worktree"] = str(tmp_path / "other-agent")
    write_attempt(coral_dir, attempt)
    with pytest.raises(TrustedWriteError, match="source worktree"):
        service.handle("retry_attempt", {"commit_hash": commit}, os.getpid())


def test_retry_rechecks_active_assignment_before_reset(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _base = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    service = _service(coral_dir, binding)
    service.handle("submit_attempt", _payload(binding, commit, parent), os.getpid())
    attempt = read_attempt(coral_dir, commit)
    attempt.status = "crashed"
    attempt.metadata["failure_class"] = "infra"
    write_attempt(coral_dir, attempt)
    service.attempt_authorizer = lambda *_: (_ for _ in ()).throw(
        TrustedWriteError("assignment completed")
    )

    with pytest.raises(TrustedWriteError, match="assignment completed"):
        service.handle("retry_attempt", {"commit_hash": commit}, os.getpid())
    assert read_attempt(coral_dir, commit).status == "crashed"


def test_incomplete_trusted_connection_times_out_without_unbounded_thread(tmp_path):
    import socket

    coral_dir = tmp_path / ".coral"
    repo, _ = _init_repo(tmp_path / "agent")
    binding = AgentBinding(
        "agent-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    write_trusted_agent_registry(coral_dir, [binding])
    server = TrustedWriteServer(
        coral_dir, _service(coral_dir, binding), read_timeout=0.05, max_connections=1
    )
    server.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect(str(server.socket_path))
        started = time.monotonic()
        response = client.recv(4096)
        assert time.monotonic() - started < 0.5
        assert b'"ok":false' in response
    finally:
        client.close()
        server.stop()


def test_root_host_policy_rejects_engineering_seed(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, source = _init_repo(tmp_path / "root")
    commit, _parent = _commit(repo, "root submission")
    (coral_dir / "private").mkdir(parents=True)
    (coral_dir / "private" / "root_bootstrap.json").write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "source_commit": source,
                "node_id": "N000",
                "mode": "research",
                "submission_hash": "f" * 64,
            }
        )
    )
    canonical = _canonical("N000")
    canonical.update(
        node_type="seed",
        mode="engineering",
        base_commit=source,
        quality_anchor=source,
        submission_hash="f" * 64,
    )
    service = TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        submission_validator=lambda *_args: canonical,
        checkpoint_fn=lambda *_args, **_kwargs: "root-checkpoint",
    )
    with pytest.raises(TrustedWriteError, match="fixed Root policy"):
        service.submit_root_attempt(worktree=repo, commit_hash=commit)


def test_root_policy_requires_sha256_submission_hash(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, source = _init_repo(tmp_path / "root-hash")
    commit, _parent = _commit(repo, "root submission")
    (coral_dir / "private").mkdir(parents=True)
    (coral_dir / "private" / "root_bootstrap.json").write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "source_commit": source,
                "node_id": "N000",
                "mode": "research",
                "submission_hash": "not-a-sha256",
            }
        )
    )
    service = TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        submission_validator=lambda *_args: _canonical("N000"),
    )

    with pytest.raises(TrustedWriteError, match="fixed Root policy"):
        service.submit_root_attempt(worktree=repo, commit_hash=commit)


def test_client_bridges_short_manager_registry_publication_race(tmp_path):
    coral_dir = tmp_path / ".coral"
    repo, _ = _init_repo(tmp_path / "race-agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding(
        "researcher-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    service = TrustedWriteService(
        coral_dir,
        SimpleNamespace(grader=SimpleNamespace(args={})),
        submission_validator=lambda *_args: _canonical("N001"),
        checkpoint_fn=lambda *_args, **_kwargs: "shared-state",
    )
    server = TrustedWriteServer(coral_dir, service)
    server.start()
    publisher = threading.Timer(
        0.05, lambda: write_trusted_agent_registry(coral_dir, [binding])
    )
    publisher.start()
    try:
        result = request_trusted(
            coral_dir,
            "submit_attempt",
            _payload(binding, commit, parent),
            startup_retry_window=0.5,
        )
    finally:
        publisher.join()
        server.stop()

    assert result["commit_hash"] == commit


def test_publish_immutable_artifact_is_idempotent_for_same_content(tmp_path):
    repo, _ = _init_repo(tmp_path / "agent")
    binding = AgentBinding(
        "agent-0", os.getpid(), process_start_time(os.getpid()), str(repo), None
    )
    service = TrustedWriteService(
        tmp_path / ".coral",
        SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda *_args: binding,
        artifact_validator=lambda *_args: {"relative_path": "nodes/N001/spec.md"},
    )
    payload = {
        "kind": "spec", "node_id": "N001", "mode": None, "version": None,
        "content": "node_id: N001\n",
    }

    first = service.handle("publish_artifact", payload, os.getpid())
    second = service.handle("publish_artifact", payload, os.getpid())

    assert first == second == {"path": "nodes/N001/spec.md"}



def test_workflow_reconcile_accepts_only_manager_peer(tmp_path, monkeypatch):
    coral_dir = tmp_path / ".coral"
    calls = []
    config = SimpleNamespace(grader=SimpleNamespace(args={}))
    service = TrustedWriteService(
        coral_dir, config,
        identity_resolver=lambda *_: (_ for _ in ()).throw(AssertionError("not an Agent")),
        workflow_handler=lambda op, payload, context: (
            calls.append((op, payload, context)) or {"assignments": []}
        ),
    )
    monkeypatch.setattr("coral.grader.trusted_writes.os.getppid", lambda: 4242)
    assert service.handle("reconcile_workflow", {}, peer_pid=4242) == {"assignments": []}
    assert calls[0][0:2] == ("reconcile_workflow", {})
    with pytest.raises(TrustedWriteError, match="manager"):
        service.handle("reconcile_workflow", {}, peer_pid=4243)
    with pytest.raises(TrustedWriteError, match="manager"):
        service.handle_authenticated(
            "reconcile_workflow", {},
            AgentBinding("research-N001", 1, 1, str(tmp_path), None),
        )


def test_agent_workflow_operation_cannot_supply_identity_or_state(tmp_path):
    binding = AgentBinding("research-N001", 1, 1, str(tmp_path), None)
    service = TrustedWriteService(
        tmp_path / ".coral",
        SimpleNamespace(grader=SimpleNamespace(args={})),
        workflow_handler=lambda op, payload, context: {
            "operation": op, "owner": context["agent_id"], "payload": payload,
        },
    )
    result = service.handle_authenticated(
        "claim_engineering", {"node_id": "N001"}, binding
    )
    assert result["owner"] == "research-N001"
    with pytest.raises(TrustedWriteError, match="unknown"):
        service.handle_authenticated(
            "claim_engineering", {"agent_id": "research-N999", "node_id": "N001"},
            binding,
        )



def test_submit_authorizer_blocks_foreign_assignment_before_attempt(tmp_path):
    repo, _ = _init_repo(tmp_path / "agent")
    commit, parent = _commit(repo, "research")
    binding = AgentBinding("research-N001", 1, 1, str(repo), None)
    service = TrustedWriteService(
        tmp_path / ".coral",
        SimpleNamespace(grader=SimpleNamespace(args={})),
        submission_validator=lambda *_: _canonical(),
        attempt_authorizer=lambda *_: (_ for _ in ()).throw(
            TrustedWriteError("Attempt owner is not active")
        ),
        checkpoint_fn=lambda *_args, **_kwargs: "shared-state",
    )
    with pytest.raises(TrustedWriteError, match="owner"):
        service.handle_authenticated("submit_attempt", _payload(binding, commit, parent), binding)
    assert read_attempt(tmp_path / ".coral", commit) is None


def test_artifact_authorizer_rebuilds_stale_evaluation_before_replace(tmp_path):
    binding = AgentBinding("research-N001", 1, 1, str(tmp_path), None)
    service = TrustedWriteService(
        tmp_path / ".coral",
        SimpleNamespace(grader=SimpleNamespace(args={})),
        artifact_validator=lambda *_: {"relative_path": "nodes/N001/evaluation.md"},
        artifact_authorizer=lambda *_: {"content": "fresh from latest Attempts\n"},
    )
    result = service.handle_authenticated("publish_artifact", {
        "kind": "evaluation", "node_id": "N001", "mode": None,
        "version": None, "content": "stale snapshot\n",
    }, binding)
    assert result == {"path": "nodes/N001/evaluation.md"}
    assert (tmp_path / ".coral/public/notes/nodes/N001/evaluation.md").read_text() == (
        "fresh from latest Attempts\n"
    )
