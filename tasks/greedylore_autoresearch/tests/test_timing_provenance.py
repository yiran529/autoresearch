"""Timing metadata must work in exported directories without a Git checkout."""
# ruff: noqa: S101
import ast
import os
import subprocess
from pathlib import Path

import pytest


def metadata_function():
    source = Path(__file__).resolve().parents[1] / 'seed/c4/table_v_timing.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optional_git_head')
    namespace = {'os': os, 'subprocess': subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['optional_git_head']


def test_no_git_archive_metadata_is_optional(tmp_path, monkeypatch):
    function = metadata_function()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('CORAL_CODE_SHA', raising=False)
    assert function() is None
    monkeypatch.setenv('CORAL_CODE_SHA', 'a' * 40)
    assert function() == 'a' * 40
    monkeypatch.setenv('CORAL_CODE_SHA', 'invalid')
    with pytest.raises(ValueError, match='SHA'):
        function()


def test_timing_command_records_host_sha(contract, tmp_path):
    from greedylore_autoresearch_grader.efficiency import build_timing_command
    cell = {**next(iter(contract.efficiency.cells.values())), 'code_sha': 'b' * 40}
    spec = build_timing_command(cell, 0, tmp_path, tmp_path / 'out', (0, 1, 2, 3))
    assert spec.env['CORAL_CODE_SHA'] == 'b' * 40
    assert 'CORAL_CODE_SHA' in spec.env_allowlist
