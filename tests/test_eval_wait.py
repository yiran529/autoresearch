"""Evaluation waits return on terminal results, with an optional client deadline."""
# ruff: noqa: S101

import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coral.types import Attempt


@pytest.mark.parametrize('terminal', ['improved', 'crashed'])
@pytest.mark.parametrize('timeout', [None, 10.0])
def test_wait_command_observes_terminal_result(monkeypatch, tmp_path, capsys, terminal, timeout):
    from coral.cli import eval as cli

    clock = [0.0]
    pending = Attempt(commit_hash='a' * 40, agent_id='r', title='run', score=None,
                      status='pending', parent_hash=None, timestamp='2026-01-01T00:00:00Z')
    ready_after = 250000 if timeout is None else 2
    monkeypatch.setattr(cli, 'find_coral_dir_and_island', lambda *a: (tmp_path, None))
    monkeypatch.setattr(cli, '_resolve_attempt_file', lambda *a: tmp_path / (pending.commit_hash + '.json'))
    monkeypatch.setattr(cli, '_read_attempt_file', lambda *a:
                        replace(pending, status=terminal) if clock[0] >= ready_after else pending)
    monkeypatch.setattr(cli, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda _: clock.__setitem__(0, clock[0] + 1)))
    cli.cmd_wait(argparse.Namespace(hash=pending.commit_hash, timeout=timeout))
    assert terminal in capsys.readouterr().out


def test_wait_command_explicit_timeout_still_returns_pending(monkeypatch, tmp_path, capsys):
    from coral.cli import eval as cli

    clock = [0.0]
    pending = Attempt(commit_hash='a' * 40, agent_id='r', title='run', score=None,
                      status='pending', parent_hash=None, timestamp='2026-01-01T00:00:00Z')
    monkeypatch.setattr(cli, 'find_coral_dir_and_island', lambda *a: (tmp_path, None))
    monkeypatch.setattr(cli, '_resolve_attempt_file', lambda *a: tmp_path / (pending.commit_hash + '.json'))
    monkeypatch.setattr(cli, '_read_attempt_file', lambda *a: pending)
    monkeypatch.setattr(cli, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda _: clock.__setitem__(0, clock[0] + 1)))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_wait(argparse.Namespace(hash=pending.commit_hash, timeout=2))
    assert exc.value.code == 2
    assert 'STILL PENDING' in capsys.readouterr().err


def test_submission_poll_honors_explicit_timeout(monkeypatch, tmp_path):
    from coral.hooks import post_commit

    clock = [0.0]
    monkeypatch.setattr(post_commit, 'read_attempt', lambda *a, **k: None)
    monkeypatch.setattr(post_commit, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda _: clock.__setitem__(0, clock[0] + 1)))
    with pytest.raises(TimeoutError):
        post_commit._poll_until_graded(tmp_path, 'a' * 40, 2)
