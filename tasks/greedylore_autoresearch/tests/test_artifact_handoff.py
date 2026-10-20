"""Persistent evidence and navigation after the producing agent retires."""
# Imported pytest fixtures intentionally share names with injected parameters.
# ruff: noqa: F811
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote

import pytest
from autoresearch.analysis import resolve_evidence
from autoresearch.state import handle_workflow
from test_workflow_state import (  # noqa: F401
    _engineer_with_attempts,
    _research_request,
    _store,
    engineering_ready,
    root_run,
)


def _archive(store, *, owner='engineer-E0001', **changes):
    payload = {
        'node_id': 'N001', 'request_id': 'self-check',
        'commit_sha': 'a' * 40 if store.repo_path is None else subprocess.check_output(
            ['git', '-C', str(store.repo_path), 'rev-parse', 'HEAD'], text=True).strip(),
        'files': [{'name': 'check.py', 'content': base64.b64encode(b'print(42)\n').decode()}],
        **changes,
    }
    return handle_workflow('archive_evidence', payload, {
        'coral_dir': str(store.coral_dir), 'repo_path': store.repo_path,
        'agent_id': owner, 'island_id': None, 'is_manager': False,
    })


def test_attachment_is_immutable_and_sealed_into_engineering_analysis(engineering_ready):
    from autoresearch.lifecycle import run_worker
    from autoresearch.validation import load_document

    store, node, body = _engineer_with_attempts(engineering_ready)
    result = _archive(store)
    reference = result['path']
    assert reference.startswith('diagnostics/N001-engineer-E0001/attachments/')
    directory = store.island_root / reference
    assert (directory / 'check.py').read_bytes() == b'print(42)\n'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert (directory / 'manifest.json').stat().st_mode & 0o444 == 0o444
    assert manifest['files'][0]['sha256'] == hashlib.sha256(b'print(42)\n').hexdigest()
    assert manifest['execution_verified'] is False
    assert _archive(store) == result
    with pytest.raises(ValueError, match='immutable|conflict'):
        _archive(store, files=[{'name': 'check.py', 'content': base64.b64encode(b'changed').decode()}])
    body['findings'] = [f'Self-check script: {reference}/check.py. Agent-reported result.']
    store.submit_engineering_analysis('engineer-E0001', body)
    assert run_worker(str(store.coral_dir), runner=lambda *_: pytest.fail('unexpected Analyzer'))['errors'] == {}
    document = load_document(node / 'engineering_analysis_v1.md')
    records = document['evidence_manifest']['evidence_records']
    assert any(r['path'] == reference + '/check.py' and r['status'] == 'present' for r in records)
    with pytest.raises(ValueError, match='active|finaliz'):
        _archive(store, request_id='late')
    assert (directory / 'check.py').read_bytes() == b'print(42)\n'


@pytest.mark.parametrize('changes', [
    {'node_id': 'N999'}, {'request_id': '../escape'},
    {'files': [{'name': '../check.py', 'content': 'eA=='}]},
    {'files': [{'name': 'check.py', 'content': 'not base64!'}]},
    {'files': [{'name': 'check.py', 'content': base64.b64encode(b'x' * 32769).decode()}]},
    {'files': [{'name': 'manifest.json', 'content': 'eA=='}]},
    {'commit_sha': 'not-a-commit'},
])
def test_attachment_rejects_wrong_owner_or_unsafe_payload(engineering_ready, changes):
    store, _, _ = _engineer_with_attempts(engineering_ready)
    _archive(store, request_id='valid')
    with pytest.raises(ValueError):
        _archive(store, **changes)
    assert len(list((store.island_root / 'diagnostics').glob('*/attachments/*/manifest.json'))) == 1


def test_research_attachments_are_visible_to_later_analysis(root_run):
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    store = _store(root_run)
    reservation = store.reconcile(1, 0)[0]
    store.initialize_research(reservation.agent_id, 'N001', _research_request())
    result = _archive(store, owner=reservation.agent_id)
    write_attempt(root_run, Attempt(
        commit_hash='c' * 40, agent_id=reservation.agent_id, title='Child research',
        score=1.0, status='improved', parent_hash='a' * 40,
        timestamp='2026-01-02T00:00:00+00:00',
        metadata={'node_id': 'N001', 'mode': 'research'},
    ))
    node = root_run / 'public/notes/nodes/N001'
    from autoresearch.evaluation import refresh_node_evaluation

    from coral.hub.attempts import read_attempts

    (node / 'evaluation.md').write_text(refresh_node_evaluation(
        'N001', read_attempts(root_run), root_run / 'public/eval_logs'))
    evidence = resolve_evidence('N001', 'research', context=store._context())
    assert any(r.path == result['path'] + '/check.py' and r.status == 'present'
               for r in evidence.evidence_records)
    # Root evidence must not change when a child archives evidence.
    assert all(result['path'] not in r.path for r in resolve_evidence(
        'N000', 'research', context=store._context()).evidence_records)
    assert (store.island_root / result['path'] / 'check.py').is_file()


def test_new_engineering_summary_must_not_cite_temporary_files(engineering_ready):
    store, _, body = _engineer_with_attempts(engineering_ready)
    body['findings'] = ['See /tmp/engineer-E0001/check.py for the semantic check.']
    with pytest.raises(ValueError, match='archive|temporary'):
        store.submit_engineering_analysis('engineer-E0001', body)


def test_generated_node_index_links_to_durable_files_and_code(engineering_ready):
    from autoresearch.lifecycle import run_worker

    store, node, body = _engineer_with_attempts(engineering_ready)
    archive = _archive(store)
    store.submit_engineering_analysis('engineer-E0001', body)
    assert run_worker(str(store.coral_dir), runner=lambda *_: pytest.fail('unexpected Analyzer'))['errors'] == {}
    index = node / 'README.md'
    assert index.exists()
    assert index.stat().st_mode & 0o444 == 0o444
    assert store.tree_path.stat().st_mode & 0o444 == 0o444
    text = index.read_text()
    targets = [(node / unquote(link)).resolve() for link in re.findall(r'\]\(([^)]+)\)', text)]
    assert (node / 'spec.md').resolve() in targets
    assert (node / 'engineering_analysis_v1.md').resolve() in targets
    assert (store.island_root / archive['path'] / 'manifest.json').resolve() in targets
    assert all(p.exists() and p.is_relative_to(store.island_root) for p in targets)
    assert 'git show' in text and 'run/repo' in text
    assert body['attempt_shas'][0] in text
    assert '(nodes/N001/README.md)' in store.tree_path.read_text()


def test_node_index_distinguishes_tune_and_archived_attempts(engineering_ready):
    from coral.hub.attempts import read_attempts, write_attempt

    store, node, body = _engineer_with_attempts(engineering_ready)
    attempt = next(a for a in read_attempts(store.coral_dir) if a.commit_hash == body['attempt_shas'][0])
    attempt.metadata.update(budget_class='tune', archived=True)
    write_attempt(store.coral_dir, attempt)
    store.reconcile(0, 1)
    row = next(line for line in (node / 'README.md').read_text().splitlines()
               if attempt.commit_hash in line)
    assert '| tune | archived |' in row


@pytest.mark.parametrize('draw,expected_mode', [(0.0, 'EXPLORE'), (0.99, 'REFINE')])
def test_tree_and_node_index_show_parent_without_inventing_reserved_parent(root_run, monkeypatch, draw, expected_mode):
    monkeypatch.setattr('autoresearch.state.random.random', lambda: draw)
    store = _store(root_run)
    assigned, reserved = store.reconcile(2, 0)
    store.initialize_research(assigned.agent_id, assigned.node_id, _research_request())
    node = store.notes_root / 'nodes' / assigned.node_id
    original_spec = (node / 'spec.md').read_bytes()
    store.reconcile(2, 0)

    tree = store.tree_path.read_text()
    assert '| Node | Parent | Research mode |' in tree
    row = next(line for line in tree.splitlines() if f'[{assigned.node_id}]' in line)
    assert f'| [N000](nodes/N000/README.md) | {expected_mode} |' in row
    pending = next(line for line in tree.splitlines() if f'[{reserved.node_id}]' in line)
    assert f'| Not selected | {expected_mode} |' in pending
    root = next(line for line in tree.splitlines() if line.startswith('| [N000]'))
    assert '| Root | — |' in root
    index = (node / 'README.md').read_text()
    assert '[N000](../N000/README.md)' in index
    assert f'Research mode: {expected_mode}' in index
    assert (node / '../N000/README.md').resolve().is_file()
    assert (node / 'spec.md').read_bytes() == original_spec


def test_lineage_uses_actual_parent_and_does_not_guess_legacy_mode(engineering_ready):
    from autoresearch.lifecycle import run_worker

    store, node, body = _engineer_with_attempts(engineering_ready)
    store.submit_engineering_analysis('engineer-E0001', body)
    assert run_worker(str(store.coral_dir), runner=lambda *_: pytest.fail('unexpected Analyzer'))['errors'] == {}
    assigned = next(a for a in store.reconcile(1, 1) if a.role == 'research')
    store.initialize_research(assigned.agent_id, assigned.node_id,
                              {**_research_request(), 'parent_node': 'N001'})
    tree = store.tree_path.read_text()
    row = next(line for line in tree.splitlines() if f'[{assigned.node_id}]' in line)
    assert '| [N001](nodes/N001/README.md) |' in row
    assert '[N001](../N001/README.md)' in (
        store.notes_root / 'nodes' / assigned.node_id / 'README.md').read_text()
    # N001's legacy Spec/state deliberately have no research_mode field.
    state = store._load_unlocked()
    state['nodes']['N001']['research'].pop('research_mode', None)
    store._save_unlocked(state)
    legacy = next(line for line in store.tree_path.read_text().splitlines() if line.startswith('| [N001]'))
    assert '| Unknown |' in legacy
