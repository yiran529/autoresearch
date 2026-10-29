from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from autoresearch.analysis import _prepare_code_snapshot, run_fresh_codex


def test_run_fresh_codex_uses_ephemeral_read_only_session(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text("{}")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("coral.grader.trusted_worker.run_workflow_model", fake_run)

    output = run_fresh_codex(
        "analyze fixed evidence",
        [tmp_path / "checkout", tmp_path / "notes"],
        schema,
        scratch,
    )

    command, kwargs = calls[0]
    assert command[:2] == ["codex", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "background_terminal_max_timeout=3600000" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "resume" not in command
    assert kwargs["input"].startswith("analyze fixed evidence")
    assert str((tmp_path / "checkout").resolve()) in kwargs["input"]
    assert str((tmp_path / "notes").resolve()) in kwargs["input"]
    assert output.read_text() == "{}"


def test_code_snapshot_materializes_fixed_commit_not_live_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    source = repo / "model.py"
    source.write_text("fixed = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "snapshot"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("live = 2\n")
    evidence = SimpleNamespace(research_sha=commit, engineering_attempts=())
    context = SimpleNamespace(checkout_root=repo)

    root = _prepare_code_snapshot(evidence, context, tmp_path / "scratch")

    assert (root / commit / "model.py").read_text() == "fixed = 1\n"


def test_cli_exposes_analyze_command(monkeypatch, capsys):
    from autoresearch import analysis, cli

    result = SimpleNamespace(
        analysis_version=2,
        analysis_path=Path("/notes/nodes/N001/research_analysis_v2.md"),
    )
    monkeypatch.setattr(analysis, "analyze", lambda node, mode, revise=False: result)

    cli.main(["analyze", "--node", "N001", "--mode", "research", "--revise"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "analysis_version": 2,
        "analysis_path": "/notes/nodes/N001/research_analysis_v2.md",
    }


def test_engineer_timing_cli_sends_committed_sha_without_autocommitting(monkeypatch, capsys):
    from autoresearch import cli

    calls = []
    monkeypatch.setattr(cli, '_coral_dir', lambda: Path('/run/.coral'))
    monkeypatch.setattr(cli, 'request_trusted', lambda root, operation, payload: calls.append((operation, payload)) or {'status': 'regressed', 'commit_hash': 'a' * 40})
    cli.main(['engineering', 'timing', '--sha', 'a' * 40, '--request-id', 'timing-1', '--retry'])
    assert calls == [('engineering_timing', {'sha': 'a' * 40, 'request_id': 'timing-1', 'retry': True})]
    assert json.loads(capsys.readouterr().out)['status'] == 'regressed'


def test_engineering_timing_receipt_precedes_poll(monkeypatch, capsys):
    from autoresearch import cli

    from coral.hooks import post_commit
    sha = 'a' * 40
    monkeypatch.setattr(cli, '_coral_dir', lambda: Path('/run/.coral'))
    monkeypatch.setattr(cli, '_workflow_context', lambda: (Path('/run/.coral'), Path('/notes'), None))
    monkeypatch.setattr(cli, 'request_trusted', lambda *args: {'status': 'pending', 'commit_hash': sha})
    def poll(*args, **kwargs):
        captured = capsys.readouterr()
        assert sha in captured.err and 'timing-1' in captured.err
        assert captured.out == ''
        return SimpleNamespace(to_dict=lambda: {'status': 'improved'})
    monkeypatch.setattr(post_commit, '_poll_until_graded', poll)
    cli.main(['engineering', 'timing', '--sha', sha, '--request-id', 'timing-1'])
    assert json.loads(capsys.readouterr().out)['status'] == 'improved'


def test_engineering_finish_wait_replays_identical_submission(monkeypatch, tmp_path, capsys):
    from autoresearch import cli
    path = tmp_path / 'analysis.json'
    path.write_text('{"summary": "done"}')
    calls = []
    monkeypatch.setattr(cli, '_coral_dir', lambda: Path('/run/.coral'))
    def request(root, operation, payload):
        calls.append((operation, payload))
        return {'status': 'completed' if len(calls) == 2 else 'finalizing'}
    monkeypatch.setattr(cli, 'request_trusted', request)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    cli.main(['engineering', 'finish', '--analysis-file', str(path), '--wait'])
    assert len(calls) == 2 and calls[0] == calls[1]
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'
