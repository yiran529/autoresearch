# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path

from coral.config import CoralConfig, SandboxConfig
from coral.sandbox.srt import build_srt_settings


def _paths(tmp_path: Path) -> dict:
    worktree = tmp_path / "agents" / "researcher"
    coral_dir = tmp_path / ".coral"
    repo_dir = tmp_path / "repo"
    for path in (worktree, coral_dir / "private", coral_dir / "public", repo_dir / ".git"):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "worktree_path": worktree,
        "coral_dir": coral_dir,
        "repo_dir": repo_dir,
        "shared_dir_name": ".codex",
    }


def test_settings_protect_formal_records_and_keep_normal_agent_writes(tmp_path):
    paths = _paths(tmp_path)
    sibling = tmp_path / "agents" / "engineer"
    sibling.mkdir()
    settings = build_srt_settings(
        SandboxConfig(enabled=True),
        proxy_port=1,
        sibling_worktrees=[sibling],
        **paths,
    )
    filesystem = settings["filesystem"]
    denied = set(filesystem["denyWrite"])
    expected = {
        str((tmp_path / ".coral/private").resolve()),
        str((tmp_path / ".coral/.git").resolve()),
        str((tmp_path / ".coral/public/attempts").resolve()),
        str((tmp_path / ".coral/public/eval_logs").resolve()),
        str((tmp_path / ".coral/public/notes/tree.md").resolve()),
        str((tmp_path / ".coral/public/notes/research_summary.md").resolve()),
        str((tmp_path / ".coral/public/notes/engineering_summary.md").resolve()),
        str((tmp_path / ".coral/public/notes/nodes").resolve()),
    }
    assert expected <= denied
    assert str(paths["worktree_path"].resolve()) in filesystem["allowWrite"]
    assert str((paths["repo_dir"] / ".git").resolve()) in filesystem["allowWrite"]
    assert str(sibling.resolve()) not in filesystem["allowWrite"]
    assert str((tmp_path / ".coral/public").resolve()) in filesystem["allowRead"]
    assert str((tmp_path / ".coral/public/memory-consolidation.lock").resolve()) not in denied


def test_multi_island_protects_existing_formal_trees(tmp_path):
    paths = _paths(tmp_path)
    (tmp_path / ".coral/islands/alpha").mkdir(parents=True)
    (tmp_path / ".coral/islands/beta").mkdir(parents=True)

    (paths["worktree_path"] / ".coral_island").write_text("alpha\n")
    settings = build_srt_settings(SandboxConfig(enabled=True), proxy_port=1, **paths)
    protected = set(settings["filesystem"]["denyWrite"])

    assert str((tmp_path / ".coral/islands/alpha/attempts").resolve()) in protected
    assert str((tmp_path / ".coral/islands/alpha/eval_logs").resolve()) in protected
    assert str((tmp_path / ".coral/islands/alpha/notes/nodes").resolve()) in protected
    assert str((tmp_path / ".coral/islands/beta/attempts").resolve()) in protected


def test_formal_task_enables_trusted_ipc_without_changing_network_allowlist(tmp_path):
    task = Path(__file__).resolve().parents[1] / "task.yaml"
    config = CoralConfig.from_yaml(task)
    settings = build_srt_settings(config.agents.sandbox, proxy_port=None, **_paths(tmp_path))
    assert settings["network"]["allowAllUnixSockets"] is True
    assert settings["network"]["strictAllowlist"] is True
    assert settings["network"]["allowedDomains"] == config.agents.sandbox.allowed_domains
