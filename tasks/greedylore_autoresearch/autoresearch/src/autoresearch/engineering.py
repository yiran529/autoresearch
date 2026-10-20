"""Engineer-authored final summaries, frozen and sealed by the trusted host."""

from __future__ import annotations

import json
from pathlib import Path

from coral.grader.trusted_writes import trusted_writes_lock
from coral.hub.attempts import read_attempts

from .analysis import AnalysisContext, resolve_evidence
from .atomic import atomic_write_text
from .evaluation import refresh_node_evaluation
from .validation import seal_document, validate_analysis, validate_authored_references


def submit(store, agent_id: str, body: dict) -> dict:
    from .engineer_diagnostics import publication_guard

    with publication_guard(store, agent_id):
        return _submit(store, agent_id, body)


def _submit(store, agent_id: str, body: dict) -> dict:
    # Import here: WorkflowStore uses this helper without a module cycle.
    from .state import WorkflowError

    if not isinstance(body, dict) or set(body) != {'summary', 'findings', 'limitations', 'attempt_shas'}:
        raise WorkflowError('Engineering summary requires summary, findings, limitations and attempt_shas')
    shas = body['attempt_shas']
    if (not isinstance(shas, list) or not shas or
            not all(isinstance(sha, str) for sha in shas) or len(set(shas)) != len(shas)):
        raise WorkflowError('Engineering summary must cover each claim Attempt exactly once')
    # Detach the body from any caller-owned mutable object before persistence.
    body = json.loads(json.dumps(body))
    with trusted_writes_lock(store.coral_dir):
        state = store._load_unlocked()
        worker = state['engineering_workers'].get(agent_id)
        if not isinstance(worker, dict) or not worker.get('node_id'):
            raise WorkflowError('Engineering summary owner was not found')
        node_id = worker['node_id']
        previous = worker.get('analysis_submission')
        if previous:
            if previous['body'] != body:
                raise WorkflowError('Engineering summary conflicts with frozen submission')
            return {'node_id': node_id, 'status': 'completed' if worker['status'] == 'completed' else 'finalizing',
                    'analysis_ref': f"nodes/{node_id}/engineering_analysis_v{previous['version']}.md"}
        if worker['status'] != 'active':
            raise WorkflowError('Engineering summary owner is not active')
        validate_authored_references(body)
        store._terminal_engineering_unlocked(agent_id, node_id)
        attempts = tuple(
            a for a in read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True)
        )
        owned = {a.commit_hash for a in attempts if a.agent_id == agent_id
                 and a.metadata.get('node_id') == node_id
                 and a.metadata.get('mode') == 'engineering' and a.budget_class != 'tune'}
        if set(shas) != owned:
            raise WorkflowError('Engineering summary must cover all claim Attempts')
        node = store.notes_root / 'nodes' / node_id
        evaluation = refresh_node_evaluation(node_id, attempts, store.island_root / 'eval_logs')
        atomic_write_text(node / 'evaluation.md', evaluation, mode=0o644)
        context = AnalysisContext(store.coral_dir, store.notes_root, store.island_id,
                                  attempts, store.island_root / 'eval_logs', store.repo_path or Path.cwd())
        evidence = resolve_evidence(node_id, 'engineering', context=context)
        authored = {key: body[key] for key in ('summary', 'findings', 'limitations')}
        validate_analysis(seal_document(authored, evidence), evidence)
        # Known host provenance, not a newly measured scientific result.
        if isinstance(authored['limitations'], list):
            authored['limitations'] = [*authored['limitations'],
                                      f'Quality is inherited (provisional) from Research SHA {evidence.research_sha}; it was not remeasured.']
        document = validate_analysis(seal_document(authored, evidence), evidence)
        versions = [int(p.stem.rsplit('_v', 1)[1]) for p in node.glob('engineering_analysis_v*.md')
                    if p.stem.rsplit('_v', 1)[1].isdigit()]
        version = max(versions, default=0) + 1
        worker['analysis_submission'] = {'body': body, 'document': document, 'version': version,
                                         'author_agent_id': agent_id}
        worker['finish_requested'] = True
        store._save_unlocked(state)
        return {'node_id': node_id, 'status': 'finalizing',
                'analysis_ref': f'nodes/{node_id}/engineering_analysis_v{version}.md'}
