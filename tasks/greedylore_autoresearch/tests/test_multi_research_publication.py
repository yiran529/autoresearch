"""Archived Research evidence remains available in derived publication paths."""
# ruff: noqa: F811
from __future__ import annotations

from dataclasses import replace

import pytest
from autoresearch.lifecycle import run_worker
from test_workflow_state import (
    _active_research,
    _engineer_with_attempts,
    _root_attempt,
    engineering_ready,  # noqa: F401
    root_run,  # noqa: F401
)

from coral.hub.attempts import read_attempts, write_attempt


def _archived_candidate(coral, node_id, agent_id):
    attempt = replace(
        _root_attempt(), commit_hash='d' * 40, agent_id=agent_id,
        status='crashed', score=None,
        metadata={'mode': 'research', 'node_id': node_id, 'archived': True,
                  'budget_class': 'grader_error'},
    )
    write_attempt(coral, attempt)
    return attempt


def test_worker_evaluation_keeps_archived_research_before_finish(root_run):
    store, assignment = _active_research(root_run)
    candidate = _archived_candidate(root_run, assignment.node_id, assignment.agent_id)
    result = run_worker(str(root_run), runner=lambda *_: pytest.fail('premature Analyzer'))
    assert result['errors'] == {}
    text = (store.notes_root / 'nodes' / assignment.node_id / 'evaluation.md').read_text()
    assert candidate.commit_hash in text


def test_engineering_publication_keeps_archived_research(engineering_ready):
    store, node, body = _engineer_with_attempts(engineering_ready)
    research = next(a for a in read_attempts(store.coral_dir)
                    if a.metadata.get('node_id') == 'N001'
                    and a.metadata.get('mode') == 'research')
    candidate = _archived_candidate(store.coral_dir, 'N001', research.agent_id)
    store.submit_engineering_analysis('engineer-E0001', body)
    state = store._load_unlocked()
    document = state['engineering_workers']['engineer-E0001']['analysis_submission']['document']
    candidates = document['evidence_manifest']['research_attempts']
    assert candidate.commit_hash in {item['commit_sha'] for item in candidates}
    assert candidate.commit_hash in (node / 'evaluation.md').read_text()
