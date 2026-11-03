"""Run preparation must supply the same usable Python to every seed's diagnostics."""
# ruff: noqa: S101
from pathlib import Path

import pytest


def test_runtime_configuration_mounts_venv_and_its_interpreter(tmp_path):
    from autoresearch.diagnostic_runtime import configure_diagnostic_runtime

    runtime = tmp_path / 'python-runtime'
    (runtime / 'bin').mkdir(parents=True)
    (runtime / 'bin/python3').write_text('interpreter')
    venv = tmp_path / 'run/repo/.venv'
    (venv / 'bin').mkdir(parents=True)
    python = venv / 'bin/python'
    python.symlink_to(runtime / 'bin/python3')
    config = {'grader': {'args': {'diagnostic_runtime_roots': [],
                                 'diagnostic_path': '/usr/bin:/bin'}},
              'agents': {'sandbox': {'allow_read': ['/existing']}}}
    configure_diagnostic_runtime(config, python)
    configure_diagnostic_runtime(config, python)
    options = config['grader']['args']
    assert options['analyzer_allow_unix_sockets'] is True
    assert options['diagnostic_runtime_roots'] == [str(venv), str(runtime)]
    assert options['diagnostic_path'] == f'{venv}/bin:/usr/bin:/bin'
    assert config['agents']['sandbox']['allow_read'] == ['/existing', str(venv), str(runtime)]


def test_runtime_configuration_rejects_private_grader_python(tmp_path):
    from autoresearch.diagnostic_runtime import configure_diagnostic_runtime

    python = tmp_path / '.coral/private/grader_venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('interpreter')
    with pytest.raises(ValueError, match='private'):
        configure_diagnostic_runtime({}, python)


def test_runtime_probe_fails_for_the_unconfigured_system_python(tmp_path, monkeypatch):
    from autoresearch.diagnostic_runtime import probe_diagnostic_runtime

    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-cross')
    with pytest.raises(ValueError, match='diagnostic Python runtime'):
        probe_diagnostic_runtime(tmp_path, {'diagnostic_path': '/usr/bin:/bin'})


def test_runtime_probe_uses_declared_python_inside_real_namespace(tmp_path):
    from autoresearch.diagnostic_runtime import (
        configure_diagnostic_runtime,
        probe_diagnostic_runtime,
    )

    python = Path('/home/wyr/greedy_lore/.venv/bin/python')
    if not python.exists():
        pytest.skip('local public PyTorch runtime is unavailable')
    config = {}
    configure_diagnostic_runtime(config, python)
    result = probe_diagnostic_runtime(tmp_path, config['grader']['args'])
    assert result['python'] == str(python)
    assert result['torch'] and result['distributed'] is True
    assert result['cuda_devices_exposed'] is False
