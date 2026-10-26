"""Trusted, resumable 130M development followed by one frozen 60M transfer.

Requests only schedule work. Expensive jobs run outside the trusted-write service
and its global lock. The private journal, rather than client process lifetime,
accounts for requests and completed phases.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import re
from pathlib import Path

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock

from .research_record import ResearchRecord
from .runtime_artifacts import safe_path, settings


class StagedResearch:
    def __init__(self, store, agent_id, node_id, *, backend=None):
        self.store, self.agent_id, self.node_id = store, agent_id, node_id
        self.record = ResearchRecord(store, agent_id, node_id)
        self.path = safe_path(store.coral_dir / 'private/research_staged' / f'{node_id}.json')
        if backend is None:
            from .staged_execution import Execution
            backend = Execution(self)
        self.backend = backend

    def _read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {
            'version': 1, 'node_id': self.node_id, 'agent_id': self.agent_id,
            'jobs': {}, 'transfer': None, 'transfer_terminal': False}

    def status(self):
        self.record._owner()
        return copy.deepcopy(self._read())

    def _save(self, value):
        _atomic_json(self.path, value)

    def validate_candidate(self, sha):
        from .submission import _git, validate_submission
        repo = self.store.repo_path
        parent = _git(repo, 'rev-parse', f'{sha}^')
        canonical = validate_submission(sha, {'worktree': str(repo),
            'coral_dir': str(self.store.coral_dir), 'parent_hash': parent,
            'contract_path': str(self.store.coral_dir / 'private/eval_contract.yaml')})
        if canonical['node_id'] != self.node_id or canonical['mode'] != 'research':
            raise ValueError('candidate submission belongs to another Node or mode')
        from coral.hub.attempts import read_attempts
        for attempt in read_attempts(self.store.coral_dir, include_archived=True):
            if attempt.commit_hash == sha and attempt.agent_id != self.agent_id:
                raise PermissionError('candidate SHA belongs to another assignment')
        return canonical

    def enqueue(self, kind, data, request_id, context):
        fields = {'train': ({'iteration', 'sha'}, {'pause_after_steps'}),
                  'finish': ({'selected_iteration', 'reason'}, {'sha'}),
                  'diagnostic': ({'iteration', 'argv', 'timeout', 'scripts_dir'}, set())}
        if kind not in fields or not isinstance(data, dict):
            raise ValueError('unknown research execution action')
        required, optional = fields[kind]
        if not required <= data.keys() or data.keys() - required - optional:
            raise ValueError('invalid research execution fields')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request_id):
            raise ValueError('invalid research execution request ID')
        digest = hashlib.sha256(json.dumps([kind, data], sort_keys=True).encode()).hexdigest()
        # Candidate validation precedes reservation/budget mutation.
        canonical = self.validate_candidate(data['sha']) if kind in {'train', 'finish'} and data.get('sha') else None
        if kind == 'train':
            pause = data.get('pause_after_steps')
            if pause is not None and (type(pause) is not int or pause <= 0):
                raise ValueError('pause_after_steps must be a positive absolute update')
        with trusted_writes_lock(self.store.coral_dir):
            owner = self.record._owner()
            state = self._read()
            if request_id in state['jobs']:
                if state['jobs'][request_id]['digest'] != digest:
                    raise ValueError('research request ID conflicts')
                return copy.deepcopy(state['jobs'][request_id])
            document = self.record.read()
            if owner.get('status') != 'active' or state['transfer_terminal']:
                raise ValueError('research is terminal')
            if document['frozen'] and kind != 'finish':
                raise ValueError('research record is frozen')
            if document['frozen'] and data['selected_iteration'] != document['selected_iteration']:
                raise ValueError('frozen candidate cannot change')
            if any(j['status'] in {'queued', 'running'} for j in state['jobs'].values()):
                raise ValueError('another research job is active; recover or wait')
            if kind == 'finish' and not any(row['code_sha'] for row in document['iterations']) and not data.get('sha'):
                raise ValueError('finish without training requires a committed representative sha')
            if kind == 'train':
                row = ResearchRecord._row(document, data['iteration'])
                if row['status'] not in {'planned', 'paused', 'infra_failed'}:
                    raise ValueError('append a new design to restart')
                if row['code_sha'] and row['code_sha'] != data['sha']:
                    raise ValueError('resume must keep candidate SHA')
                if pause is not None and row['observations'] and pause <= row['observations'][-1].get('update_step', 0):
                    raise ValueError('pause must advance the existing trajectory')
            if kind == 'diagnostic':
                row = ResearchRecord._row(document, data['iteration'])
                if row['status'] not in {'paused', 'completed'}:
                    raise ValueError('diagnostics require a saved trajectory')
                from coral.grader.timeouts import DIAGNOSTIC_EXECUTION_TIMEOUT
                timeout = data['timeout']
                if (type(timeout) not in (int, float) or not 0 < timeout <= DIAGNOSTIC_EXECUTION_TIMEOUT
                        or not isinstance(data['argv'], list) or not data['argv']
                        or any(not isinstance(v, str) or '\x00' in v for v in data['argv'])):
                    raise ValueError('invalid diagnostic command or timeout')
                diagnostics = [j for j in state['jobs'].values() if j['kind'] == 'diagnostic']
                if len(diagnostics) >= 15 or sum(j['status'] == 'completed' for j in diagnostics) >= 5:
                    raise ValueError('Researcher diagnostic budget exhausted (5 completed / 15 total)')
            if kind == 'finish' and document['frozen']:
                failures = sum(j['kind'] == 'finish' and j.get('result', {}).get('status') == 'infra_failed'
                               for j in state['jobs'].values())
                if failures > settings(self.store.coral_dir).get('research_infra_retry_limit', 2):
                    raise ValueError('transfer infrastructure retry budget exhausted')
            job = {'kind': kind, **copy.deepcopy(data), 'request_id': request_id,
                   'digest': digest, 'status': 'queued', 'context': copy.deepcopy(context),
                   'canonical': canonical}
            state['jobs'][request_id] = job
            self._save(state)
            return copy.deepcopy(job)

    def drain(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for request_id in list(self._read()['jobs']):
                state = self._read()
                job = state['jobs'][request_id]
                publication = (job['kind'] == 'finish' and state['transfer_terminal']
                               and not state.get('publication_complete'))
                if job['status'] not in {'queued', 'running'} and not publication:
                    continue
                owner = self.record._owner()
                if owner.get('status') != 'active' and not publication:
                    job.update(status='cancelled', result={'status': 'cancelled'})
                    self._save(state)
                    self.backend.release()
                    continue
                # A complete result is written before publishing observations. On
                # restart this branch never reruns a completed expensive segment.
                if job['status'] == 'running' and 'result' not in job and not (job['kind'] == 'finish' and state['transfer'] is not None):
                    result = {'status': 'infra_failed', 'reason': 'host job interrupted'}
                    if job['kind'] == 'train':
                        self.record.begin(job['iteration'], job['sha'], request_id + '.begin')
                        self.record.observe(job['iteration'], result, request_id + '.observation')
                    job.update(status='failed', result=result)
                    self._save(state)
                    continue
                job['status'] = 'running'
                self._save(state)
                try:
                    if job['kind'] == 'finish':
                        self._finish(job)
                        continue
                    if job['kind'] == 'train':
                        self.record.begin(job['iteration'], job['sha'], request_id + '.begin')
                    if 'result' not in job:
                        try:
                            job['result'] = self.backend.run(self, job)
                        except Exception as exc:
                            from greedylore_autoresearch_grader.gpu import InfraFailure
                            job['result'] = {'status': 'infra_failed' if isinstance(exc, (InfraFailure, OSError)) else 'failed',
                                             'reason': f'{type(exc).__name__}: {exc}',
                                             **self._failure_evidence(job)}
                        self._save(state)
                    result = job['result']
                    if job['kind'] == 'train':
                        self.record.observe(job['iteration'], result, request_id + '.observation')
                    job['status'] = 'completed' if result['status'] in {'completed', 'paused'} else 'failed'
                    self._save(state)
                except Exception as exc:
                    state = self._read()
                    state['jobs'][request_id].update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    self._save(state)

    def _finish(self, job):
        state = self._read()
        document = self.record.read()
        selected = job['selected_iteration']
        if not document['frozen'] and not any(row['code_sha'] for row in document['iterations']):
            document = self.record.bind_unrun(job['sha'], job['request_id'] + '.unrun')
        if not document['frozen']:
            document = self.record.freeze(selected, job['reason'], job['request_id'] + '.freeze')
        transfer = state['transfer']
        if transfer is None or transfer.get('status') == 'infra_failed':
            if selected is None:
                transfer = {'status': 'skipped', 'reason': 'no eligible 130M candidate'}
            else:
                row = document['iterations'][selected - 1]
                transfer_job = {**job, 'kind': 'transfer', 'sha': row['code_sha'], 'iteration': selected}
                try:
                    transfer = self.backend.run(self, transfer_job)
                except Exception as exc:
                    from greedylore_autoresearch_grader.gpu import InfraFailure
                    transfer = {'status': 'infra_failed' if isinstance(exc, (InfraFailure, OSError)) else 'failed',
                                'reason': f'{type(exc).__name__}: {exc}', **self._failure_evidence(transfer_job)}
            state['transfer'] = transfer
            self._save(state)
        if transfer['status'] == 'infra_failed':
            used = 1 + sum(item['kind'] == 'finish' and item.get('result', {}).get('status') == 'infra_failed'
                           for key, item in state['jobs'].items() if key != job['request_id'])
            if used <= settings(self.store.coral_dir).get('research_infra_retry_limit', 2):
                state['jobs'][job['request_id']].update(status='failed', result=transfer)
                self._save(state)
                return
            transfer['retry_budget_exhausted'] = True
            state['transfer'] = transfer
        # Repeatable publication, independent of whether the costly transfer ran
        # in this process. Keep a durable terminal marker for the lease keeper.
        state['transfer_terminal'] = True
        self._save(state)
        self.backend.release()
        self.backend.finish(self, selected)
        state['publication_complete'] = True
        state['jobs'][job['request_id']].update(status='completed', result=transfer)
        self._save(state)

    def _failure_evidence(self, job):
        suffix = '-transfer' if job['kind'] == 'transfer' else ''
        sha = job.get('sha') or self.record.read()['iterations'][job['iteration'] - 1]['code_sha']
        relative = f"eval_logs/{sha}/{job['request_id']}{suffix}"
        return {'evidence_path': relative} if (self.store.coral_dir / 'public' / relative).exists() else {}


def handle_action(store, owner, payload, context):
    if set(payload) != {'node_id', 'action', 'data', 'request_id'}:
        raise ValueError('invalid staged research action')
    node, action, data, request_id = (payload[k] for k in ('node_id', 'action', 'data', 'request_id'))
    host = StagedResearch(store, owner, node)
    if action == 'register':
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', request_id):
            raise ValueError('invalid research registration request ID')
        from .research_record import validate_design
        if isinstance(data, dict) and set(data) == {'registration', 'design'}:
            # Only replay an already-published old registration. Fresh Nodes
            # cannot bypass the unified origin by authoring a compatibility intent.
            spec_path = store.notes_root / 'nodes' / node / 'spec.md'
            if (not spec_path.is_file()
                    or yaml.safe_load(spec_path.read_text()).get('research_origin_version') == 3
                    or host.record.private.exists() and host.record.read().get('version', 2) >= 3):
                raise ValueError('new registration requires origin and design')
            registration, design, origin = data['registration'], validate_design(data['design']), None
        else:
            from .research_origin import registration_from_origin
            registration, design, origin = registration_from_origin(node, data)
        store.initialize_research(owner, node, registration, owner_identity=context,
                                  origin_version=3 if origin is not None else None)
        spec = yaml.safe_load((store.notes_root / 'nodes' / node / 'spec.md').read_text())
        result = host.record.initialize(spec, design, request_id, origin=origin)
        with trusted_writes_lock(store.coral_dir):
            state = store._load_unlocked()
            state['nodes'][node]['research']['protocol'] = 'staged-v2'
            store._save_unlocked(state)
        return result
    if action == 'status':
        return host.status()
    if action == 'design':
        return host.record.append_design(data, request_id)
    if action == 'decision':
        if set(data) != {'iteration', 'action', 'reason'}:
            raise ValueError('decision requires iteration, action, reason')
        return host.record.decide(data['iteration'], data['action'], data['reason'], request_id)
    if action not in {'train', 'diagnostic', 'finish', 'recover'}:
        raise ValueError('unknown research action')
    if action == 'recover':
        result = host.status()
    else:
        result = host.enqueue(action, data, request_id, context)
    from coral.grader.trusted_worker import start_task_worker
    start_task_worker(store.coral_dir, 'autoresearch.staged_research:run_worker', log_name='staged_worker.err')
    return result


def run_worker(coral_dir):
    from .state import WorkflowStore
    store = WorkflowStore(coral_dir)
    for path in sorted((store.coral_dir / 'private/research_staged').glob('N*.json')):
        state = json.loads(path.read_text())
        StagedResearch(store, state['agent_id'], state['node_id']).drain()
    return {'status': 'drained'}


def node_idle(coral_dir, node_id):
    """Nonblocking cleanup gate; a terminal assignment cannot admit new jobs."""
    path = Path(coral_dir) / 'private/research_staged' / f'{node_id}.lock'
    if not path.exists():
        return True
    with path.open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
