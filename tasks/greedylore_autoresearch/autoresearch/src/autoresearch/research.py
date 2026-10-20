"""Research run accounting and deterministic, host-frozen candidate selection."""

from __future__ import annotations

import json
import math
from pathlib import Path


def run_budget(attempts) -> tuple[int, int, int]:
    successful = total = pending = 0
    for attempt in attempts:
        total += max(1 + int(attempt.metadata.get('infra_retry_count', 0)),
                     len(attempt.metadata.get('grader_runs') or []),
                     int(attempt.metadata.get('grader_invocation_count', 0)))
        pending += attempt.status == 'pending'
        successful += scored_success(attempt)
    return successful, total, pending


def scored_success(attempt) -> bool:
    return (attempt.status in {'improved', 'baseline', 'regressed', 'reverted'}
            and type(attempt.score) in (float, int) and math.isfinite(attempt.score))


def research_attempts(attempts, node_id, agent_id=None):
    return [a for a in attempts if a.metadata.get('node_id') == node_id
            and a.metadata.get('mode') == 'research' and a.budget_class != 'tune'
            and (agent_id is None or a.agent_id == agent_id)]


def quality_eligible(attempt) -> bool:
    quality = attempt.metadata.get('scores', {}).get('quality_gate', {}).get('value')
    comparable = attempt.metadata.get('comparability', attempt.metadata.get('comparable'))
    return (not attempt.archived and scored_success(attempt)
            and not attempt.metadata.get('failure_class')
            and type(quality) in (float, int) and quality == 1
            and comparable in (None, True, 'comparable'))


def select_research_attempt(attempts):
    """Best valid passing score; stable earliest tie; latest failure fallback.

    Archived records are excluded from success selection. If every record is
    archived, retain one for failure Analysis only; it cannot enable Engineering.
    """
    terminal = [a for a in attempts if a.status != 'pending']
    if not terminal:
        raise ValueError('Research requires a terminal Attempt')
    eligible = [a for a in terminal if quality_eligible(a)]
    if eligible:
        return min(eligible, key=lambda a: (-a.score, a.timestamp, a.commit_hash))
    live = [a for a in terminal if not a.archived]
    return max(live or terminal, key=lambda a: (a.timestamp, a.commit_hash))


def frozen_research_attempt(coral_dir, node_id, attempts):
    """Old singleton records remain readable; multiple candidates need a pin."""
    candidates = research_attempts(attempts, node_id)
    path = Path(coral_dir) / 'private/autoresearch_state.json'
    research = {}
    if path.exists():
        research = json.loads(path.read_text()).get('nodes', {}).get(node_id, {}).get('research', {})
    selected = research.get('selected_sha')
    if selected is not None:
        matches = [a for a in candidates if a.commit_hash == selected and a.status != 'pending']
        if len(matches) != 1:
            raise ValueError('frozen Research SHA is missing or not terminal')
        return matches[0]
    candidates = [a for a in candidates if a.status != 'pending']
    if len(candidates) != 1:
        raise ValueError(f'Node {node_id} requires a frozen Research SHA')
    return candidates[0]
