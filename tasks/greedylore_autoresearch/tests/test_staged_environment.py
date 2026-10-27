"""Provision fresh candidate snapshots before asking for retained GPUs."""
# ruff: noqa: S101
import os
from pathlib import Path

import pytest
import yaml
from autoresearch.staged_execution import Execution
from test_staged_research import host


def execution(tmp_path):
    h, _, _ = host(tmp_path)
    external=tmp_path/'external'
    (h.store.coral_dir/'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'dynamic_artifact_root': str(external), 'research_uv_cache_dir': str(tmp_path/'shared-cache')}}}))
    return Execution(h)


def test_dependency_cache_is_shared_but_generated_caches_are_node_owned(tmp_path):
    backend=execution(tmp_path)
    work=backend.artifacts.path('N001','runs/test')
    work.mkdir(parents=True)
    env=backend._environment(work)
    assert env['UV_CACHE_DIR']==str(tmp_path/'shared-cache')
    assert Path(env['TRITON_CACHE_DIR']).is_relative_to(backend.artifacts.node_root('N001'))


def test_real_offline_provisioning_failure_is_infra_and_blocks_admission(tmp_path, monkeypatch):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    backend=execution(tmp_path)
    work=backend.artifacts.path('N001','runs/test')
    work.mkdir(parents=True)
    code=work/'code'
    code.mkdir()
    uv=work/'uv'
    uv.write_text('#!/bin/sh\necho "missing offline wheel" >&2\nexit 2\n')
    uv.chmod(0o755)
    monkeypatch.setenv('PATH',str(work)+os.pathsep+os.environ['PATH'])
    with pytest.raises(InfraFailure,match='environment'):
        backend._prepare_environment(code,work)
    assert (backend.coral/'private/research_infrastructure_block.json').is_file()
    assert 'missing offline wheel' in (work/'environment.stderr.log').read_text()


def test_setup_only_retry_does_not_require_nonexistent_checkpoint(tmp_path):
    backend=execution(tmp_path)
    row={'runs':[{'request_id':'one.begin'},{'request_id':'two.begin'}],
         'observations':[{'status':'infra_failed','phase':'environment_setup'}]}
    assert backend._resume(row,'llama130m_muon') is None


def test_runtime_preflight_failure_is_infra_before_gpu_launch(tmp_path):
    from greedylore_autoresearch_grader.runtime import InfraFailure
    backend = execution(tmp_path)
    work = backend.artifacts.path('N001', 'runs/preflight')
    work.mkdir(parents=True)
    code = work / 'code'
    interpreter = code / '.venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/bin/sh\necho "OSError: AF_UNIX path too long" >&2\nexit 1\n')
    interpreter.chmod(0o755)
    with pytest.raises(InfraFailure, match='preflight'):
        backend._prepare_runtime(code, work)
    assert 'AF_UNIX' in (work / 'runtime-preflight.stderr.log').read_text()
    assert (backend.coral / 'private/research_infrastructure_block.json').exists()
