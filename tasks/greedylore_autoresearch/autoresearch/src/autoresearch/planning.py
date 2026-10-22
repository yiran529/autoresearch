"""Run-wide exclusive method selection, held across separate Agent tool calls."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import yaml

from coral.grader.trusted_writes import trusted_writes_lock


class PlanningError(ValueError):
    pass


def _owner_alive(lease):
    try:
        fields = (Path('/proc') / str(lease['root_pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == lease['root_start_time']
    except (OSError, KeyError, ValueError, IndexError):
        return False


def current_lease(state):
    lease = state.get('research_planning')
    if not lease:
        return None
    owner = state['nodes'].get(lease['node_id'], {}).get('research', {})
    if (lease['expires_at'] <= time.time() or owner.get('status') != 'reserved'
            or owner.get('agent_id') != lease['agent_id'] or not _owner_alive(lease)):
        return None
    return lease


def require_lease(state, agent_id, token, identity):
    lease = current_lease(state)
    if (not lease or lease['agent_id'] != agent_id or not token or lease['token'] != token
            or not identity or lease['root_pid'] != identity.get('agent_root_pid')
            or lease['root_start_time'] != identity.get('agent_start_time')):
        raise PlanningError('current exclusive research planning lock required; acquire again and reread registered methods')
    return lease


def release_owned(state, agent_id, token):
    lease = state.get('research_planning') or {}
    if lease.get('agent_id') == agent_id and lease.get('token') == token:
        state.pop('research_planning', None)


def recover_registration(store, state):
    """A trusted Spec is the durable commit point if the following state save crashed."""
    from .handover import _safe_file

    lease = state.get('research_planning')
    if not lease:
        return False
    research = state['nodes'].get(lease['node_id'], {}).get('research', {})
    if research.get('status') != 'reserved' or research.get('agent_id') != lease['agent_id']:
        return False
    relative = f"nodes/{lease['node_id']}/spec.md"
    path = _safe_file(store.notes_root, relative)
    if not path.is_file():
        return False
    spec = yaml.safe_load(path.read_text())
    if (not isinstance(spec, dict) or spec.get('node_id') != lease['node_id']
            or not isinstance(spec.get('method_registration'), dict)
            or not spec['method_registration'].get('overlap_review')):
        raise PlanningError('cannot recover planning registration from invalid Spec')
    research.update(status='active', spec_ref=relative)
    state.pop('research_planning', None)
    return True


def planning_operation(store, operation, agent_id, payload, identity):
    """Only short state transitions hold the global write lock; selection owns a lease."""
    with trusted_writes_lock(store.coral_dir):
        state = store._load_unlocked()
        if recover_registration(store, state):
            store._save_unlocked(state)
        node_id = next((node for node, record in state['nodes'].items()
                        if node != 'N000' and record['research']['agent_id'] == agent_id
                        and record['research']['status'] == 'reserved'), None)
        if node_id is None:
            raise PlanningError('planning requires an uninitialized reserved Research Node')
        if store.research_planning_lease_seconds == 0:
            return {'status': 'disabled', 'registry_ref': 'tree.md'}
        lease = current_lease(state)
        if operation == 'acquire_research_planning':
            if lease and lease['agent_id'] == agent_id and (
                    lease['root_pid'] != identity['agent_root_pid']
                    or lease['root_start_time'] != identity['agent_start_time']):
                # The current authenticated binding supersedes the prior process,
                # even if an orphan with the old identity has not exited yet.
                lease = None
            if lease and lease['agent_id'] != agent_id:
                return {'status': 'busy', 'owner_node': lease['node_id'], 'expires_at': lease['expires_at']}
            if not lease:
                lease = {'agent_id': agent_id, 'node_id': node_id, 'token': uuid.uuid4().hex,
                         'expires_at': time.time() + store.research_planning_lease_seconds,
                         'root_pid': identity['agent_root_pid'], 'root_start_time': identity['agent_start_time']}
                if not _owner_alive(lease):
                    raise PlanningError('planning owner process identity is not alive')
                state['research_planning'] = lease
                store._save_unlocked(state)
        else:
            lease = require_lease(state, agent_id, payload['token'], identity)
            if operation == 'release_research_planning':
                state.pop('research_planning', None)
                store._save_unlocked(state)
                return {'status': 'released'}
            lease['expires_at'] = time.time() + store.research_planning_lease_seconds
            store._save_unlocked(state)
        return {'status': 'acquired', 'token': lease['token'], 'expires_at': lease['expires_at'],
                'registry_ref': 'tree.md', 'specs_glob': 'nodes/N*/spec.md'}
