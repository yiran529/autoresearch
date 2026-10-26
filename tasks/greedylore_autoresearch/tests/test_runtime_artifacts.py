"""Managed external artifacts must never spill into the run or another node."""

import json
from pathlib import Path

import pytest
import yaml


def configured(tmp_path):
    coral = tmp_path / 'run/.coral'
    coral.mkdir(parents=True)
    (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'dynamic_artifact_root': str(tmp_path / 'external'),
        'dynamic_artifact_min_free_bytes': 0,
    }}}))
    return coral


def test_external_outputs_and_checkpoint_resolution_survive_public_snapshot(tmp_path):
    from autoresearch.runtime_artifacts import RuntimeArtifacts

    coral = configured(tmp_path)
    store = RuntimeArtifacts(coral)
    checkpoint = store.path('N001', 'iterations/I001/checkpoints/step-00001000')
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.pt').write_bytes(b'tensor')
    manifest = {'status': 'complete', 'update_step': 1000}
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    public = coral / 'public/eval_logs/sha/run/checkpoints/llama130m_muon/manifest.json'
    store.publish_checkpoint('N001', checkpoint, public)
    assert public.is_file()
    assert not (public.parent / 'model.pt').exists()
    assert store.resolve_checkpoint(public) == checkpoint
    (checkpoint / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        store.resolve_checkpoint(public)


def test_cleanup_requires_terminal_node_and_only_removes_owned_namespace(tmp_path):
    from autoresearch.runtime_artifacts import RuntimeArtifacts

    coral = configured(tmp_path)
    store = RuntimeArtifacts(coral)
    a = store.path('N001', 'scratch/a.pt')
    b = store.path('N002', 'scratch/b.pt')
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b'a')
    b.write_bytes(b'b')
    assert store.cleanup_node('N001', terminal=False) is False
    assert a.exists()
    assert store.cleanup_node('N001', terminal=True)
    assert not a.exists() and b.exists()
    assert store.cleanup_node('N001', terminal=True)  # crash/reconcile idempotency
    ledger = json.loads((coral / 'private/dynamic_artifacts.json').read_text())
    assert ledger['nodes']['N001']['cleaned'] is True


def test_rejects_escape_symlink_and_capacity_without_root_fallback(tmp_path, monkeypatch):
    from autoresearch.runtime_artifacts import RuntimeArtifacts

    coral = configured(tmp_path)
    store = RuntimeArtifacts(coral)
    with pytest.raises(ValueError):
        store.path('N001', '../N002/x')
    with pytest.raises(ValueError):
        store.path('other', 'x')
    root = store.node_root('N001')
    (root / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        store.path('N001', 'escape/x')
    from types import SimpleNamespace

    monkeypatch.setattr('autoresearch.runtime_artifacts.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='space'):
        store.require_space('N002', 10)
    assert not list((coral / 'public').rglob('*.pt'))
