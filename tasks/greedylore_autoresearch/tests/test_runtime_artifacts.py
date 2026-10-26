"""Managed external artifacts must never spill into the run or another node."""

import json

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


def test_parent_traversal_is_rejected_even_when_lexically_contained(tmp_path):
    from autoresearch.runtime_artifacts import safe_path
    with pytest.raises(ValueError, match='traversal'):
        safe_path(tmp_path / 'owned' / '..' / 'foreign')


def test_staged_analyzer_dynamic_results_are_external_and_only_small_evidence_persists(tmp_path):
    import os

    from autoresearch.diagnostics import create_task
    from autoresearch.runtime_artifacts import RuntimeArtifacts

    coral = configured(tmp_path)
    (coral / 'private').mkdir()
    state = {'version': 2, 'revision': 1, 'next_node_seq': 2, 'next_engineer_seq': 1,
             'next_ready_seq': 1, 'engineering_workers': {}, 'nodes': {
                 'N000': {'research': {'status': 'completed'}},
                 'N001': {'research': {'status': 'active', 'protocol': 'staged-v2'}}}}
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    task = create_task(coral, 'N001', 'research', 1, 'a'*40)
    task.bind(os.getpid())
    assert task.workspace.is_relative_to(RuntimeArtifacts(coral).node_root('N001'))
    def run(item, payload, cancelled):
        cancelled()
        target = item.records / payload['request_id']
        (target / 'probe.json').write_text('{"simulated":true}')
        (target / 'tensor.pt').write_bytes(b'large dynamic tensor stand-in')
        return {'status': 'completed', 'started': True}
    request = {'request_id': 'probe', 'argv': ['true'], 'gpus': 4, 'timeout': 1}
    assert task.run(request, os.getpid(), run)['status'] == 'completed'
    durable = task.root / 'records/probe'
    assert (durable / 'probe.json').is_file()
    assert not (durable / 'tensor.pt').exists()
    assert (task.records / 'probe/tensor.pt').is_file()


def test_staged_analyzer_citations_seal_and_read_after_tensor_cleanup(tmp_path):
    import os
    from types import SimpleNamespace

    from autoresearch.analysis import EvidenceBundle
    from autoresearch.analysis_sources import read_source, seal_sources
    from autoresearch.diagnostics import create_task
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    coral = configured(tmp_path)
    (coral / 'private').mkdir()
    state = {'version': 2, 'revision': 1, 'next_node_seq': 2, 'next_engineer_seq': 1,
             'next_ready_seq': 1, 'engineering_workers': {}, 'nodes': {
                 'N000': {'research': {'status': 'completed'}},
                 'N001': {'research': {'status': 'active', 'protocol': 'staged-v2'}}}}
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    evidence = EvidenceBundle('N001', 'research', 'a'*40, 'contract', 'version', '',
        'spec-hash', '', '', (), (), (), 'fixed-input-hash')
    context = SimpleNamespace(coral_dir=coral, checkout_root=tmp_path)
    task = create_task(coral, 'N001', 'research', 1, 'a'*40, evidence_hash=evidence.manifest_hash)
    task.bind(os.getpid())
    def run(item, payload, cancelled):
        results = item.records / payload['request_id'] / 'results'
        results.mkdir()
        (results / 'probe.csv').write_text('metric,value\ngap,0.25\n')
        (results / 'tensor.pt').write_bytes(b'dynamic')
        return {'status': 'completed', 'started': True}
    task.run({'request_id': 'probe', 'argv': ['true'], 'gpus': 4, 'timeout': 1}, os.getpid(), run)
    _, sealed = seal_sources({'summary': 'simulated'}, evidence, context, 1)
    reference = 'diagnostics/N001-research-v1/records/probe/results/probe.csv'
    assert reference in {record.path for record in sealed.evidence_records}
    expected = read_source(reference, evidence, context, version=1)
    RuntimeArtifacts(coral).cleanup_node('N001', terminal=True)
    assert read_source(reference, evidence, context, version=1) == expected


def test_cancelled_staged_node_archives_small_evidence_then_cleans_only_own_files(tmp_path):
    from types import SimpleNamespace

    from autoresearch.node_cleanup import cleanup_completed_nodes
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    coral = configured(tmp_path)
    artifacts = RuntimeArtifacts(coral)
    own = artifacts.path('N001', 'runs/run')
    own.mkdir(parents=True)
    (own / 'failure.log').write_text('cancelled run evidence')
    (own / 'model.pt').write_bytes(b'tensor')
    other = artifacts.path('N002', 'model.pt')
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b'other')
    state = {'nodes': {'N001': {'research': {'protocol': 'staged-v2', 'status': 'cancelled'}}}}
    store = SimpleNamespace(coral_dir=coral, island_id=None, _node_completed=lambda _: False)
    assert cleanup_completed_nodes(store, state)
    assert state['nodes']['N001']['cleanup']['status'] == 'done'
    assert not own.exists() and other.exists()
    assert (coral / 'public/cancelled/N001/runs/run/failure.log').is_file()
