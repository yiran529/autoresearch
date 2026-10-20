"""Preflight safety for the real Engineer-only experiment driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_workflow_state import root_run as root_run

from coral.hub.attempts import read_attempts, write_attempt

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_root_engineer.py"
spec = importlib.util.spec_from_file_location("root_engineer_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_preflight_requires_actual_quality_gate_not_improved_label(root_run):
    with pytest.raises(ValueError, match="passing quality"):
        runner.validate_root(root_run)
    root = read_attempts(root_run)[0]
    root.metadata["scores"]["quality_gate"]["value"] = 1.0
    write_attempt(root_run, root)
    assert runner.validate_root(root_run).commit_hash == root.commit_hash


def test_driver_rejects_output_inside_original_run(tmp_path):
    source = tmp_path / "original"
    output = source / "experiment"
    result = subprocess.run([sys.executable, str(SCRIPT), "--source-run", str(source),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "independent" in result.stderr
    assert not source.exists()


def test_sandbox_anchors_preserve_existing_files_and_keep_worktree_clean(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    existing = tmp_path / ".gitconfig"
    existing.write_text("[user]\nname = Existing\n")
    runner.prepare_srt_worktree(tmp_path)
    first_exclude = (tmp_path / ".git/info/exclude").read_text()
    runner.prepare_srt_worktree(tmp_path)
    assert existing.read_text() == "[user]\nname = Existing\n"
    assert (tmp_path / ".git/info/exclude").read_text() == first_exclude
    assert subprocess.check_output(["git", "-C", str(tmp_path), "status", "--porcelain"]) == b""


def test_clean_codex_home_preserves_refreshed_auth_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "original-home"
    source.mkdir()
    (source / "auth.json").write_text('{"test":"initial"}')
    monkeypatch.setenv("CODEX_HOME", str(source))
    config = SimpleNamespace(agents=SimpleNamespace(
        sandbox=SimpleNamespace(allow_read=[], allow_write=[])))
    output = tmp_path / "experiment"
    runner.prepare_codex_home(output, config)
    copied = output / "run/.sandbox/codex-cli-home/auth.json"
    assert copied.read_text() == '{"test":"initial"}'
    assert copied.stat().st_mode & 0o777 == 0o600
    copied.write_text('{"test":"refreshed"}')
    runner.prepare_codex_home(output, config)
    assert copied.read_text() == '{"test":"refreshed"}'
    assert (source / "auth.json").read_text() == '{"test":"initial"}'


def test_clean_codex_home_accepts_newer_external_refresh(tmp_path, monkeypatch):
    source = tmp_path / "original-home"
    source.mkdir()
    auth = source / "auth.json"
    auth.write_text(json.dumps({"last_refresh": "2026-10-20T03:00:00Z"}))
    config = SimpleNamespace(agents=SimpleNamespace(
        sandbox=SimpleNamespace(allow_read=[], allow_write=[])))
    monkeypatch.setenv("CODEX_HOME", str(source))
    runner.prepare_codex_home(tmp_path / "experiment", config)
    target = tmp_path / "experiment/run/.sandbox/codex-cli-home/auth.json"
    auth.write_text(json.dumps({"last_refresh": "2026-10-20T05:00:00Z"}))
    monkeypatch.setenv("CODEX_HOME", str(source))
    runner.prepare_codex_home(tmp_path / "experiment", config)
    assert target.read_text() == auth.read_text()
