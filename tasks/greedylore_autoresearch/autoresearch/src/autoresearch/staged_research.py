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
from concurrent.futures import ThreadPoolExecutor
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
        from .diagnostics import DIAGNOSTIC_SUCCESS_LIMIT, DIAGNOSTIC_TOTAL_LIMIT

        state = copy.deepcopy(self._read())
        state['diagnostic_limits'] = {'successful': DIAGNOSTIC_SUCCESS_LIMIT, 'total': DIAGNOSTIC_TOTAL_LIMIT}
        return state

    def wait(self, request_id):
        """One authenticated read; waiting and retry delays stay in the CLI."""
        state = self.status()
        if request_id not in state['jobs']:
            raise ValueError('unknown research request ID')
        job = state['jobs'][request_id]
        job['diagnostic_limits'] = state['diagnostic_limits']
        # Older journals may contain an outer controller error without a result.
        if job['status'] in {'failed', 'cancelled'} and 'result' not in job:
            job['result'] = {'status': job['status'], 'reason': job.get('error', job['status'])}
        return job

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

    def enqueue(self, kind, data, request_id, context, *, _parent_finish=None):
        fields = {'train': ({'iteration', 'sha'}, {'pause_after_steps'}),
                  'timing': ({'iteration', 'sha'}, {'repeat_reason'}),
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
        canonical = self.validate_candidate(data['sha']) if kind in {'train', 'timing', 'finish'} and data.get('sha') else None
        if kind == 'timing' and 'repeat_reason' in data:
            if not isinstance(data['repeat_reason'], str) or not data['repeat_reason'].strip() or len(data['repeat_reason']) > 4000:
                raise ValueError('repeat_reason must be bounded nonempty text')
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
            if any((j['status'] in {'queued', 'running'} or (
                    j['kind'] == 'timing' and 'result' in j and not j.get('reused_from') and not j.get('record_published'))) and not (
                    j['request_id'] == _parent_finish and j['kind'] == 'finish') for j in state['jobs'].values()):
                raise ValueError('another research job is active; recover or wait')
            if kind == 'finish' and not any(row['code_sha'] for row in document['iterations']) and not data.get('sha'):
                raise ValueError('finish without training requires a committed representative sha')
            if kind in {'train', 'timing'}:
                row = ResearchRecord._row(document, data['iteration'])
                if kind == 'train' and row['status'] not in {'planned', 'paused', 'infra_failed'}:
                    raise ValueError('append a new design to restart')
                if row['status'] == 'running':
                    raise ValueError('another research job is active; recover or wait')
                if kind == 'timing' and row['status'] not in {'planned', 'paused', 'completed'}:
                    raise ValueError('timing requires a planned, paused or completed trajectory')
                if row['code_sha'] and row['code_sha'] != data['sha']:
                    raise ValueError('resume must keep candidate SHA')
                if kind == 'train' and pause is not None and row['observations'] and pause <= row['observations'][-1].get('update_step', 0):
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
                from .diagnostics import DIAGNOSTIC_SUCCESS_LIMIT, DIAGNOSTIC_TOTAL_LIMIT
                if (len(diagnostics) >= DIAGNOSTIC_TOTAL_LIMIT
                        or sum(j['status'] == 'completed' for j in diagnostics) >= DIAGNOSTIC_SUCCESS_LIMIT):
                    raise ValueError('Researcher diagnostic budget exhausted '
                                     f'({DIAGNOSTIC_SUCCESS_LIMIT} completed / {DIAGNOSTIC_TOTAL_LIMIT} total)')
            if kind == 'finish' and document['frozen']:
                failures = sum(j['kind'] == 'finish' and j.get('result', {}).get('status') == 'infra_failed'
                               for j in state['jobs'].values())
                if failures > settings(self.store.coral_dir).get('research_infra_retry_limit', 2):
                    raise ValueError('transfer infrastructure retry budget exhausted')
            job = {'kind': kind, **copy.deepcopy(data), 'request_id': request_id,
                   'digest': digest, 'status': 'queued', 'context': copy.deepcopy(context),
                   'canonical': canonical}
            if kind == 'timing':
                from .staged_timing import check_budget, latest_timing, timing_context
                job['timing_context'] = timing_context(self, canonical)
                cached = latest_timing(row, job['timing_context'])
                if cached and 'repeat_reason' not in data:
                    job.update(status='completed', result=cached['result'], reused_from=cached['request_id'])
                else:
                    check_budget(self, state, data['sha'])
            if kind in {'train', 'timing'}:
                self.record.bind_candidate(data['iteration'], data['sha'], request_id + '.bind', _already_locked=True)
            state['jobs'][request_id] = job
            self._save(state)
            return copy.deepcopy(job)

    def drain(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            processed = set()
            while True:
                with trusted_writes_lock(self.store.coral_dir):
                    state = self._read()
                    request_id = next((key for key in state['jobs'] if key not in processed), None)
                    if request_id is None:
                        # Enqueue holds this same lock. Release the Node lock
                        # before allowing another enqueue, so its worker either
                        # owns the Node or leaves the new job for this loop.
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        return True
                processed.add(request_id)
                job = state['jobs'][request_id]
                if job['kind'] == 'diagnostic':
                    result = job.get('result', {})
                    legacy_mirror_failure = ('NotADirectoryError' in result.get('reason', '')
                                             and 'research_diagnostics' in result.get('reason', ''))
                    if (result.get('publication', {}).get('status') == 'failed'
                            or legacy_mirror_failure
                            or job['status'] == 'running' and 'result' not in job):
                        recover = getattr(self.backend, 'recover_diagnostic', None)
                        try:
                            recovered = recover(job) if recover else None
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            if job['status'] == 'running' and 'result' not in job:
                                job.update(status='failed', result={
                                    'status': 'infra_failed',
                                    'reason': 'saved diagnostic result could not be verified',
                                    **self._failure_evidence(job)})
                            job.setdefault('result', {}).update(publication={
                                'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
                            self._save(state)
                            continue
                        if recovered is not None:
                            job.setdefault('publication_recovery', {
                                'previous_status': job['status'], 'previous_result': copy.deepcopy(result)})
                            job.update(result=recovered, status=(
                                'completed' if recovered['status'] == 'completed' else 'failed'))
                            self._save(state)
                            continue
                publication = (job['kind'] == 'finish' and state['transfer_terminal']
                               and not state.get('publication_complete')) or bool(job.get('retry_publication'))
                if job['kind'] == 'timing' and 'result' in job and not job.get('reused_from'):
                    publication = not job.get('record_published')
                if job['status'] not in {'queued', 'running'} and not publication:
                    continue
                owner = self.record._owner()
                if owner.get('status') != 'active' and not publication:
                    job.update(status='cancelled', result={'status': 'cancelled'})
                    self._save(state)
                    self.backend.release()
                    continue
                if job['kind'] == 'timing':
                    try:
                        self._execute_timing(job)
                    except Exception as exc:
                        state = self._read()
                        state['jobs'][request_id].update(status='failed', error=f'{type(exc).__name__}: {exc}')
                        self._save(state)
                    continue
                # A complete result is written before publishing observations. On
                # restart this branch never reruns a completed expensive segment.
                if job['status'] == 'running' and 'result' not in job and job['kind'] != 'finish':
                    result = {'status': 'infra_failed', 'reason': 'host job interrupted', **self._failure_evidence(job)}
                    if job['kind'] == 'train':
                        self.record.begin(job['iteration'], job['sha'], request_id + '.begin')
                        self.record.observe(job['iteration'], result, request_id + '.observation')
                    job.update(status='failed', result=result)
                    self._save(state)
                    continue
                job['status'] = 'running'
                job.pop('retry_publication', None)
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
                                             **({'phase': exc.phase} if getattr(exc, 'phase', None) else {}),
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
                    if job['kind'] == 'finish' and isinstance(exc, OSError):
                        state['jobs'][request_id]['retry_publication'] = True
                    self._save(state)

    def _record_timing(self, job, result):
        self.record.add_timing(job['iteration'], {
            'request_id': job['request_id'], 'sha': job['sha'], 'context': job['timing_context'],
            'repeat_reason': job.get('repeat_reason'), 'result': result}, job['request_id'] + '.timing')

    def _execute_timing(self, job):
        state = self._read()
        job = state['jobs'][job['request_id']]
        if 'result' not in job:
            if job['status'] == 'running':
                recover = getattr(self.backend, 'recover_timing', None)
                try:
                    recovered = recover(job) if recover else None
                    reason = 'host timing interrupted'
                except Exception as exc:
                    recovered, reason = None, f'{type(exc).__name__}: {exc}'
                job['result'] = recovered or {'status': 'infra_failed', 'reason': reason, **self._failure_evidence(job)}
            else:
                job['status'] = 'running'
                self._save(state)
                try:
                    job['result'] = self.backend.run(self, job)
                except Exception as exc:
                    from greedylore_autoresearch_grader.gpu import InfraFailure
                    failure = exc
                    recover = getattr(self.backend, 'recover_timing', None)
                    try:
                        recovered = recover(job) if recover else None
                    except Exception as recovery_error:
                        recovered = None
                        failure = recovery_error
                    job['result'] = recovered or {
                        'status': 'infra_failed' if isinstance(failure, (InfraFailure, OSError)) else 'failed',
                        'reason': f'{type(failure).__name__}: {failure}',
                        **({'phase': failure.phase} if getattr(failure, 'phase', None) else {}), **self._failure_evidence(job)}
            self._save(state)
        publisher = getattr(self.backend, 'publish_timing', None)
        if publisher:
            publisher(job, job['result'])
        self._record_timing(job, job['result'])
        job['record_published'] = True
        job['status'] = 'completed' if job['result']['status'] == 'completed' else 'failed'
        self._save(state)
        return job['result']

    def _ensure_timing(self, finish, row):
        from .staged_timing import latest_timing, timing_context, validate_result
        canonical = self.validate_candidate(row['code_sha'])
        context = timing_context(self, canonical)
        cached = latest_timing(row, context)
        if cached is None:
            request_id = 'finish-timing-' + hashlib.sha256(finish['request_id'].encode()).hexdigest()[:24]
            job = self.enqueue('timing', {'iteration': row['id'], 'sha': row['code_sha']},
                               request_id, finish['context'], _parent_finish=finish['request_id'])
            self._execute_timing(job)
            cached = latest_timing(self.record.read()['iterations'][row['id'] - 1], context)
        if cached:
            validate_result(cached['result'], context)
        if not cached or not cached['result'].get('performance_eligible'):
            raise ValueError('selected candidate requires eligible timing; inspect timing evidence before finishing')

    def _finish(self, job):
        if self.record.read().get('evaluation_stages'):
            from .staged_final import finalize
            return finalize(self, job)
        state = self._read()
        document = self.record.read()
        selected = job['selected_iteration']
        if not document['frozen'] and not any(row['code_sha'] for row in document['iterations']):
            document = self.record.bind_unrun(job['sha'], job['request_id'] + '.unrun')
        if not document['frozen']:
            if any(row['status'] in {'running', 'paused', 'infra_failed', 'planned'} for row in document['iterations']):
                raise ValueError('all trajectories must end before freeze')
            if selected is not None:
                row = ResearchRecord._row(document, selected)
                if row['status'] != 'completed' or not row['observations'][-1].get('quality_pass', row['observations'][-1].get('eligible')):
                    raise ValueError('selected iteration must have completed passing 130M quality evidence')
                self._ensure_timing(job, row)
                state = self._read()
            document = self.record.freeze(selected, job['reason'], job['request_id'] + '.freeze')
        else:
            document = self.record.publish()
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
    if action == 'wait':
        if data:
            raise ValueError('wait accepts no action data')
        return host.wait(request_id)
    if action == 'design':
        return host.record.append_design(data, request_id)
    if action == 'decision':
        if set(data) != {'iteration', 'action', 'reason'}:
            raise ValueError('decision requires iteration, action, reason')
        return host.record.decide(data['iteration'], data['action'], data['reason'], request_id)
    if action not in {'train', 'timing', 'diagnostic', 'finish', 'recover'}:
        raise ValueError('unknown research action')
    if action == 'recover':
        result = host.status()
    else:
        result = host.enqueue(action, data, request_id, context)
    from coral.grader.trusted_worker import start_task_worker
    start_task_worker(store.coral_dir, 'autoresearch.staged_research:run_worker',
                      log_name='staged_worker.err', kwargs={'only_node': node})
    return result


def run_worker(coral_dir, *, only_node=None):
    """Dispatch one Node, or recover journals independently up to Node capacity."""
    from .state import WorkflowStore

    if only_node is not None and (not isinstance(only_node, str) or not re.fullmatch(r'N[0-9]{3}', only_node)):
        raise ValueError('invalid staged research Node ID')
    store = WorkflowStore(coral_dir)
    root = store.coral_dir / 'private/research_staged'
    paths = ([root / f'{only_node}.json'] if only_node is not None
             else sorted(root.glob('N*.json')))
    if store.max_active_nodes:
        with trusted_writes_lock(store.coral_dir):
            state = store._load_unlocked()
            admitted = store._admitted_node_ids_unlocked(state)
            # Terminal journals may still need publication or cancellation;
            # active Nodes outside admission must keep waiting for their slot.
            paths = [path for path in paths if path.stem in admitted
                     or state['nodes'].get(path.stem, {}).get('research', {}).get('status')
                     not in {'active', 'reserved'}]

    def drain_node(path):
        if not path.exists():
            return
        state = json.loads(path.read_text())
        StagedResearch(store, state['agent_id'], state['node_id']).drain()

    if only_node is not None:
        for path in paths:
            drain_node(path)
    else:
        with ThreadPoolExecutor(max_workers=store.max_active_nodes or max(1, len(paths))) as pool:
            list(pool.map(drain_node, paths))
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
