"""Append-only research designs and decisions, published by the trusted host."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from contextlib import nullcontext

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock

from .runtime_artifacts import safe_path, settings


def validate_design(design):
    required = {'method', 'mechanism', 'predictions', 'tests'}
    if not isinstance(design, dict) or set(design) != required:
        raise ValueError('design requires method, mechanism, predictions and tests')
    if any(not isinstance(value, str) or not value.strip() or len(value) > 8000
           for value in design.values()):
        raise ValueError('design fields must be bounded nonempty text')
    return copy.deepcopy(design)


class ResearchRecord:
    def __init__(self, store, agent_id, node_id):
        if not isinstance(node_id, str) or not re.fullmatch(r'N[0-9]{3}', node_id):
            raise ValueError('invalid research record node')
        self.store, self.agent_id, self.node_id = store, agent_id, node_id
        self.private = safe_path(store.coral_dir / 'private/research_records' / f'{node_id}.json')
        self.public = safe_path(store.notes_root / 'nodes' / node_id / 'research.json')

    def _owner(self):
        if settings(self.store.coral_dir).get('staged_research') is not True:
            raise ValueError('staged_research is not enabled by the host')
        state = json.loads((self.store.coral_dir / 'private/autoresearch_state.json').read_text())
        research = state['nodes'].get(self.node_id, {}).get('research', {})
        if research.get('agent_id') != self.agent_id:
            raise PermissionError('research record requires the assigned Researcher')
        return research

    def read(self):
        self._owner()
        return copy.deepcopy(json.loads(self.private.read_text())['document'])

    def _change(self, request_id, operation, payload, update, *, initial=None, _already_locked=False, _allow_frozen=False):
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', request_id):
            raise ValueError('invalid research request ID')
        digest = hashlib.sha256(json.dumps([operation, payload], sort_keys=True).encode()).hexdigest()
        with (nullcontext() if _already_locked else trusted_writes_lock(self.store.coral_dir)):
            owner = self._owner()
            state = json.loads(self.private.read_text()) if self.private.exists() else {
                'document': initial, 'requests': {}}
            document = state['document']
            if request_id in state['requests']:
                if state['requests'][request_id] != digest:
                    raise ValueError('research request ID conflicts with previous payload')
                _atomic_json(self.public, document, mode=0o644)
                return copy.deepcopy(document)
            if owner.get('status') != 'active':
                raise PermissionError('research assignment is not active')
            if document is None:
                raise ValueError('research record is not initialized')
            if document.get('frozen') and not _allow_frozen:
                raise ValueError('research record is frozen')
            update(document)
            state['requests'][request_id] = digest
            _atomic_json(self.private, state)
            _atomic_json(self.public, document, mode=0o644)
            return copy.deepcopy(document)

    def initialize(self, spec, design, request_id, *, origin=None):
        design = validate_design(design)
        unified = origin is not None
        origin = copy.deepcopy(origin if unified else spec)
        if unified:
            origin['initial_design_sha256'] = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
            origin.update({key: copy.deepcopy(spec[key]) for key in (
                'node_id', 'parent_commit', 'base_commit', 'research_mode',
                'analysis_refs', 'parent_handover') if key in spec})
        document = {'version': 3 if unified else 2, 'node_id': self.node_id, 'agent_id': self.agent_id,
                    'question': origin['question'] if unified else (spec.get('research_intent') or {}).get('question', spec.get('hypothesis')),
                    'origin': origin, 'iterations': [], 'selected_iteration': None,
                    'selection_rationale': None, 'frozen': False}
        contract_path = self.store.coral_dir / 'private/eval_contract.yaml'
        contract = yaml.safe_load(contract_path.read_text()) if contract_path.exists() else {}
        if contract.get('schema_version', 1) >= 2:
            document['evaluation_stages'] = copy.deepcopy(contract['evaluation'])
            document['final_evaluation'] = {}

        def update(value):
            if value['iterations']:
                raise ValueError('research record is already initialized')
            value['iterations'].append(self._iteration(1, design))

        return self._change(request_id, 'initialize', [origin, design], update, initial=document)

    @staticmethod
    def _iteration(index, design):
        return {'id': index, 'design': design, 'status': 'planned', 'code_sha': None,
                'observations': [], 'decisions': [], 'runs': [], 'timings': []}

    def _require_idle(self):
        path = self.store.coral_dir / 'private/research_staged' / f'{self.node_id}.json'
        jobs = json.loads(path.read_text())['jobs'] if path.exists() else {}
        if any(job['status'] in {'queued', 'running'} or (
                job['kind'] == 'timing' and 'result' in job and not job.get('reused_from') and not job.get('record_published'))
                for job in jobs.values()):
            raise ValueError('another research job is active; recover or wait')

    def publish(self):
        """Replay the public projection, including a privately committed freeze."""
        with trusted_writes_lock(self.store.coral_dir):
            document = self.read()
            _atomic_json(self.public, document, mode=0o644)
            return document

    def bind_candidate(self, iteration, sha, request_id, *, _already_locked=False):
        """The first admitted measurement or training pins this design's code."""
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('candidate SHA must be a full commit hash')
        def update(value):
            row = self._row(value, iteration)
            if row['code_sha'] is not None and row['code_sha'] != sha:
                raise ValueError('candidate SHA is fixed; append a new design to change it')
            row['code_sha'] = sha
        return self._change(request_id, 'bind_candidate', [iteration, sha], update,
                            _already_locked=_already_locked)

    def add_timing(self, iteration, entry, request_id):
        def update(value):
            row = self._row(value, iteration)
            if row['code_sha'] != entry['sha']:
                raise ValueError('timing SHA does not match the fixed candidate')
            row.setdefault('timings', []).append(copy.deepcopy(entry))
        return self._change(request_id, 'timing', [iteration, entry], update)

    def final_observe(self, component, entry, request_id):
        """Only the host appends final evidence; the candidate remains frozen."""
        def update(value):
            if not value['frozen'] or value['selected_iteration'] is None:
                raise ValueError('final evidence requires a frozen selected candidate')
            row = self._row(value, value['selected_iteration'])
            if entry['sha'] != row['code_sha']:
                raise ValueError('final evidence SHA mismatch')
            value['final_evaluation'].setdefault(component, []).append(copy.deepcopy(entry))
        return self._change(request_id, 'final_observe', [component, entry], update, _allow_frozen=True)

    @staticmethod
    def _row(value, iteration):
        if type(iteration) is not int or not 1 <= iteration <= len(value['iterations']):
            raise ValueError('unknown research iteration')
        return value['iterations'][iteration - 1]

    def append_design(self, design, request_id):
        design = validate_design(design)

        def update(value):
            self._require_idle()
            if len(value['iterations']) >= 3:
                raise ValueError('research budget is at most 3 trajectories')
            if value['iterations'][-1]['status'] not in {'completed', 'failed', 'stopped'}:
                raise ValueError('previous trajectory is not terminal')
            value['iterations'].append(self._iteration(len(value['iterations']) + 1, design))

        return self._change(request_id, 'append_design', design, update)

    def begin(self, iteration, sha, request_id):
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('candidate SHA must be a full commit hash')

        def update(value):
            row = self._row(value, iteration)
            if any(r['status'] == 'running' for r in value['iterations']):
                raise ValueError('a trajectory is already running')
            if row['status'] not in {'planned', 'paused', 'infra_failed'}:
                raise ValueError('trajectory cannot continue; append a new design to restart')
            if row['code_sha'] is not None and row['code_sha'] != sha:
                raise ValueError('resume must keep the original candidate SHA')
            if row['status'] == 'infra_failed':
                cap = settings(self.store.coral_dir).get('research_infra_retry_limit', 2)
                if sum(r['status'] == 'infra_failed' for r in row['runs']) > cap:
                    raise ValueError('infrastructure retry budget exhausted')
            row['code_sha'], row['status'] = sha, 'running'
            row['runs'].append({'request_id': request_id, 'status': 'running'})

        return self._change(request_id, 'begin', [iteration, sha], update)

    def observe(self, iteration, observation, request_id):
        observation = copy.deepcopy(observation)
        status = observation.get('status')
        if status not in {'paused', 'completed', 'failed', 'stopped', 'infra_failed'}:
            raise ValueError('invalid trajectory observation status')

        def update(value):
            row = self._row(value, iteration)
            if row['status'] != 'running':
                raise ValueError('observation requires a running trajectory')
            row['observations'].append(observation)
            row['status'] = status
            row['runs'][-1]['status'] = status

        return self._change(request_id, 'observe', [iteration, observation], update)

    def decide(self, iteration, action, reason, request_id):
        if action not in {'continue', 'revise', 'stop'} or not isinstance(reason, str) or not reason.strip():
            raise ValueError('decision requires continue/revise/stop and a reason')

        def update(value):
            self._require_idle()
            row = self._row(value, iteration)
            if row['status'] == 'running':
                raise ValueError('cannot decide while training is running')
            row['decisions'].append({'action': action, 'reason': reason})
            if action != 'continue' and row['status'] in {'paused', 'infra_failed', 'planned'}:
                row['status'] = 'stopped'

        return self._change(request_id, 'decide', [iteration, action, reason], update)

    def freeze(self, selected, reason, request_id):
        if selected is not None and type(selected) is not int:
            raise ValueError('selected_iteration must be an integer or null')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('selection requires a reason')

        def update(value):
            if any(row['status'] in {'running', 'paused', 'infra_failed', 'planned'}
                   for row in value['iterations']):
                raise ValueError('all trajectories must end before freeze')
            if selected is not None:
                row = self._row(value, selected)
                from .staged_timing import eligible
                development_ok = (row['status'] == 'completed' and row['observations']
                    and row['observations'][-1].get('quality_pass'))
                if not (development_ok if value.get('evaluation_stages') else eligible(row)):
                    raise ValueError('selected iteration must have completed eligible 130M evidence')
            value.update(selected_iteration=selected, selection_rationale=reason, frozen=True)

        return self._change(request_id, 'freeze', [selected, reason], update)

    def bind_unrun(self, sha, request_id):
        """Host-only: give an unexecuted, stopped design a committed code witness."""
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('representative SHA must be a full commit')
        def update(value):
            if any(row['code_sha'] for row in value['iterations']):
                raise ValueError('an executed candidate already represents this Node')
            row = value['iterations'][-1]
            if row['status'] != 'stopped':
                raise ValueError('unexecuted design must be explicitly stopped')
            row['code_sha'] = sha
            row['observations'].append({'status': 'stopped', 'executed': False,
                'reason': 'development ended before training; committed code witness only'})
        return self._change(request_id, 'bind_unrun', sha, update)
