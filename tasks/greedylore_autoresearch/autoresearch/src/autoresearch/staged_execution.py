"""Expensive execution boundary for the trusted staged controller."""
from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock
from coral.hub.attempts import read_attempt, write_attempt
from coral.types import Attempt, ScoreBundle, Task

from .runtime_artifacts import RuntimeArtifacts, safe_path, settings


class Execution:
    def __init__(self, host):
        self.host = host
        self.coral = host.store.coral_dir
        self.artifacts = RuntimeArtifacts(self.coral)
        self.options = settings(self.coral)
        self.interval = self.options.get('research_checkpoint_interval_steps', 1000)
        if type(self.interval) is not int or self.interval <= 0:
            raise ValueError('research_checkpoint_interval_steps must be a positive integer')
        from .lease import ResearchLease
        self.lease = ResearchLease(self.coral, host.node_id)

    def _check(self):
        from greedylore_autoresearch_grader.gpu import InfraFailure
        owner = self.host.record._owner()
        if owner.get('status') != 'active' or not self.lease.check():
            raise InfraFailure('Researcher assignment or GPU lease ended')
        if (self.coral / 'private/workflow_stopping').exists():
            raise InfraFailure('workflow is stopping')

    def _ready(self, context):
        from greedylore_autoresearch_grader.gpu import InfraFailure
        deadline = time.monotonic() + self.options.get('gpu_wait_timeout_seconds', 86400)
        while not self.lease.ensure(context):
            if time.monotonic() >= deadline:
                raise InfraFailure('Researcher GPU reservation queue timed out')
            time.sleep(0.1)
        self._check()

    def run(self, host, job):
        try:
            return self._execute(host, job)
        finally:
            suffix = '-transfer' if job['kind'] == 'transfer' else ''
            row = host.record.read()['iterations'][job['iteration'] - 1]
            sha = job.get('sha') or row['code_sha']
            run_id = job['request_id'] + suffix
            work = safe_path(self.artifacts.namespace / 'nodes' / host.node_id / 'runs' / run_id)
            if work.exists():
                self._archive(work, self.coral / 'public/eval_logs' / sha / run_id)

    def _execute(self, host, job):
        from greedylore_autoresearch_grader.contract import EvalContract
        from greedylore_autoresearch_grader.gpu import InfraFailure
        from greedylore_autoresearch_grader.process import run_process_group
        from greedylore_autoresearch_grader.quality import build_quality_command
        from greedylore_autoresearch_grader.resources import ResourceOptions
        from greedylore_autoresearch_grader.results import quality_gate

        from .engineer_diagnostics import _snapshot_commit

        root = self.artifacts.require_space(host.node_id)
        self._ready(job['context'])
        document = host.record.read()
        row = document['iterations'][job['iteration'] - 1]
        sha = job.get('sha') or row['code_sha']
        code = self.artifacts.path(host.node_id, f'code/{sha}')
        _snapshot_commit(host.store.repo_path, sha, code, root)
        run_id = job['request_id'] + ('-transfer' if job['kind'] == 'transfer' else '')
        work = self.artifacts.path(host.node_id, f'runs/{run_id}')
        public = self.coral / 'public/eval_logs' / sha / run_id
        work.mkdir(parents=True, exist_ok=True)
        public.mkdir(parents=True, exist_ok=True)
        contract = EvalContract.load(self.coral / 'private/eval_contract.yaml')
        if job['kind'] == 'diagnostic':
            return self._diagnostic(job, code, work, public, row)
        name = 'llama60m_adamw' if job['kind'] == 'transfer' else 'llama130m_muon'
        profile = contract.quality[name]
        target = job.get('pause_after_steps') or profile['stop_after_steps']
        if target > profile['stop_after_steps']:
            raise ValueError('pause endpoint exceeds fixed training budget')
        output = work / 'quality' / name
        # A segment result cache prevents a crash during timing/publication from
        # rerunning already-completed quality training.
        cache = work / 'quality-result.json'
        if not cache.exists() and job['kind'] == 'train':
            for previous in reversed(row['runs'][:-1]):
                old = self.artifacts.path(host.node_id,
                    f"runs/{previous['request_id'].removesuffix('.begin')}/quality-result.json")
                if old.exists():
                    saved = json.loads(old.read_text())
                    if saved.get('status') == 'completed':
                        _atomic_json(cache, saved)
                        break
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            resume = self._resume(row, name) if job['kind'] == 'train' else None
            spec = build_quality_command(profile, code, output, self.lease.borrow().indices,
                checkpoint_root=work / 'checkpoints' / name, checkpoint_interval_steps=self.interval,
                resume_checkpoint=resume, pause_after_steps=job.get('pause_after_steps'),
                training_metrics=work / 'training_metrics.jsonl')
            from dataclasses import replace
            spec = replace(spec, env={**spec.env, **self._environment(work)})
            reservation = self.lease.borrow()
            result = run_process_group(spec, check_cancelled=self._check,
                monitor=reservation.check_exclusive,
                monitor_interval=ResourceOptions.from_args(self.options).monitor_interval,
                on_start=self.lease.register_job, on_finish=self.lease.clear_job)
            _atomic_json(work / 'process.json', {
                'command': list(spec.command), 'returncode': result.returncode,
                'timed_out': result.timed_out, 'infra_failure': result.infra_failure,
                'stdout_log': result.stdout_log, 'stderr_log': result.stderr_log}, mode=0o644)
            self._archive(work, public)
            if result.infra_failure:
                raise InfraFailure(result.infra_failure)
            if result.returncode != 0 or result.timed_out:
                raise ValueError(f'candidate training failed: {result.stderr_tail}')
            try:
                payload = json.loads((output / 'all_results.json').read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(f'candidate quality result missing or malformed: {exc}') from exc
            if (payload.get('status') not in {'paused', 'completed'}
                    or payload.get('contract_hash') != contract.contract_hash
                    or payload.get('world_size') != contract.hardware.gpu_count
                    or payload.get('update_step') != target
                    or (target < profile['stop_after_steps']) != (payload['status'] == 'paused')):
                raise ValueError('invalid staged quality result')
            if payload['status'] == 'completed' and not math.isfinite(float(payload.get('final_eval_loss', math.nan))):
                raise ValueError('invalid final evaluation loss')
            checkpoint = safe_path(Path(payload['checkpoint_path']))
            if not checkpoint.is_relative_to(work / 'checkpoints' / name):
                raise ValueError('candidate checkpoint path escapes this segment')
            self.artifacts.publish_checkpoint(host.node_id, checkpoint,
                public / 'checkpoints' / name / 'manifest.json')
            _atomic_json(cache, payload)
        observation = {**payload, 'run_id': run_id, 'evidence_path': public.relative_to(self.coral / 'public').as_posix(),
                       'eligible': False}
        if payload['status'] == 'completed' and job['kind'] == 'train':
            timing_cache = work / 'timing-bundle.json'
            if timing_cache.exists():
                bundle = ScoreBundle.from_dict(json.loads(timing_cache.read_text()))
            else:
                grader = self._grader(code, work / 'timing', job['canonical'])
                reference = grader._load_references(contract.contract_hash)
                bundle = grader._formal(contract, 'engineering', reference)
                _atomic_json(timing_cache, bundle.to_dict())
            reference = self._grader(code, work, {})._load_references(contract.contract_hash)
            passed = quality_gate(payload['final_eval_loss'],
                reference['quality']['workloads'][name]['final_eval_loss'],
                max_relative=contract.raw['quality']['max_relative_eval_loss'])
            observation.update(timing=bundle.to_dict(), eligible=bool(passed and bundle.metadata.get('performance_eligible')),
                               quality_pass=passed)
        self._archive(work, public)
        return observation

    def _environment(self, work):
        root = self.artifacts.node_root(self.host.node_id) / 'caches'
        return {'TMPDIR': str(work), 'UV_CACHE_DIR': str(root / 'uv'),
                'TORCHINDUCTOR_CACHE_DIR': str(root / 'torchinductor'),
                'TRITON_CACHE_DIR': str(root / 'triton'), 'CUDA_CACHE_PATH': str(root / 'cuda'),
                'XDG_CACHE_HOME': str(root / 'xdg'), 'HF_HOME': str(root / 'huggingface'),
                'HF_DATASETS_CACHE': str(root / 'datasets')}

    def _archive(self, work, public):
        records = self.artifacts.preserve_small(self.host.node_id, work, public)
        _atomic_json(public / 'artifact_inventory.json', {'files': records}, mode=0o644)

    def _resume(self, row, name):
        # Only private host-produced checkpoint roots are considered. A partial
        # save cannot replace an older complete optimizer boundary.
        from greedylore_autoresearch_grader.gpu import InfraFailure
        roots = []
        for run in row['runs'][:-1]:
            root = self.artifacts.path(self.host.node_id,
                f"runs/{run['request_id'].removesuffix('.begin')}/checkpoints/{name}")
            for path in root.glob('step-*/manifest.json'):
                value = json.loads(safe_path(path).read_text())
                if value.get('status') == 'complete':
                    roots.append((value['update_step'], path.parent))
        if roots:
            return max(roots, key=lambda item: item[0])[1]
        if len(row['runs']) > 1:
            raise InfraFailure('no complete checkpoint remains for exact continuation; stop and redesign')
        return None

    def _grader(self, code, output, canonical):
        from greedylore_autoresearch_grader.grader import Grader
        from greedylore_autoresearch_grader.process import run_process_group
        from greedylore_autoresearch_grader.resources import ResourceOptions

        from coral.config import GraderConfig, ParallelGraderConfig
        execution = self
        class BorrowedGrader(Grader):
            @property
            def eval_logs_dir(self):
                output.mkdir(parents=True, exist_ok=True)
                return output
            def _run(self, spec):
                from dataclasses import replace
                spec = replace(spec, env={**spec.env, **execution._environment(output)})
                reservation = execution.lease.borrow()
                return run_process_group(spec, check_cancelled=execution._check,
                    monitor=reservation.check_exclusive,
                    monitor_interval=ResourceOptions.from_args(execution.options).monitor_interval,
                    on_start=execution.lease.register_job, on_finish=execution.lease.clear_job)
        config = yaml.safe_load((self.coral / 'config.yaml').read_text())['grader']
        grader = BorrowedGrader(GraderConfig(args=config.get('args', {}),
            parallel=ParallelGraderConfig(**config.get('parallel', {}))))
        grader.codebase_path, grader.private_dir, grader.island_id = str(code), str(self.coral / 'private'), None
        grader.tasks = [Task(id='staged', name='staged', description='host-controlled stage', metadata=canonical)]
        grader._staged_reservation = self.lease.borrow
        grader._journal = None
        return grader

    def release(self):
        self.lease.release()

    def finish(self, host, selected):
        from greedylore_autoresearch_grader.contract import EvalContract
        from greedylore_autoresearch_grader.grader import score_results
        document, state = host.record.read(), host._read()
        rows = [row for row in document['iterations'] if row['code_sha']]
        if not rows:
            raise ValueError('finish requires at least one committed candidate trajectory')
        selected_row = document['iterations'][selected - 1] if selected else rows[-1]
        rows.sort(key=lambda row: row['id'] == selected)
        contract = EvalContract.load(self.coral / 'private/eval_contract.yaml')
        for row in rows:
            sha = row['code_sha']
            jobs = [job for job in state['jobs'].values() if job.get('sha') == sha and job.get('canonical')]
            canonical = jobs[0]['canonical']
            bundle = None
            if row['id'] == selected and state['transfer']['status'] == 'completed':
                observation = row['observations'][-1]
                timing_bundle = ScoreBundle.from_dict(observation['timing'])
                timing = {cell: {'mean': timing_bundle.scores[f'{label}_timing_mean'].value,
                                'relative_spread': timing_bundle.scores[f'{label}_timing_spread'].value}
                          for cell, label in [('130m_adamw_bf16','bf16'), ('130m_adamw_fp32','fp32')]}
                reference = self._grader(self.artifacts.path(host.node_id, f'code/{sha}'),
                    self.artifacts.path(host.node_id, 'publication'), canonical)._load_references(contract.contract_hash)
                metadata = timing_bundle.metadata
                bundle = score_results('research', {'llama130m_muon': observation,
                    'llama60m_adamw': state['transfer']}, timing, reference,
                    contract_hash=contract.contract_hash,
                    comparable=metadata['comparability'] != 'incomparable_hardware', evidence_paths=[],
                    max_relative_eval_loss=contract.raw['quality']['max_relative_eval_loss'],
                    hardware_comparison=metadata.get('hardware_comparison'),
                    timing_protocol=metadata.get('timing_protocol'),
                    performance_eligible=metadata.get('performance_eligible', False))
            previous = read_attempt(self.coral, sha)
            if previous and previous.agent_id != host.agent_id:
                raise PermissionError('cannot overwrite another assignment Attempt')
            paths = [ob['evidence_path'] for candidate in rows for ob in candidate['observations'] if ob.get('evidence_path')]
            # Include failed/partial attempts and probes even when execution
            # raised before producing a normal trajectory observation.
            for job in state['jobs'].values():
                candidate_sha = job.get('sha') or (document['iterations'][job['iteration'] - 1]['code_sha'] if job.get('iteration') else None)
                if candidate_sha:
                    relative = f"eval_logs/{candidate_sha}/{job['request_id']}"
                    if (self.coral / 'public' / relative).exists():
                        paths.append(relative)
            if state['transfer'].get('evidence_path'):

                paths.append(state['transfer']['evidence_path'])
            # Diagnostic evidence and research.json are added independently to
            # Analyzer's immutable evidence envelope.
            metadata = {**canonical, **(bundle.metadata if bundle else {}),
                'staged_research': True, 'budget_class': 'real', 'evidence_paths': sorted(set(paths)),
                'performance_eligible': bool(bundle and bundle.metadata.get('performance_eligible')),
                'scores': bundle.to_dict()['scores'] if bundle else {},
                'failure_class': bundle.metadata.get('failure_class') if bundle else (
                    'infra' if state['transfer'].get('status') == 'infra_failed' else 'candidate'),
                'max_infra_retries': 0}
            attempt = Attempt(sha, host.agent_id, 'Staged Research', bundle.aggregated if bundle else None,
                'regressed' if bundle else 'crashed', document['origin'].get('parent_commit'),
                previous.timestamp if previous else datetime.now(UTC).isoformat(),
                feedback=bundle.feedback or '' if bundle else 'Development/transfer incomplete; see research.json',
                metadata=metadata)
            write_attempt(self.coral, attempt)
        with trusted_writes_lock(self.coral):
            workflow = host.store._load_unlocked()
            workflow['nodes'][host.node_id]['research'].update(
                protocol='staged-v2', transfer_terminal=True, finish_requested=True,
                selected_sha=selected_row['code_sha'])
            host.store._save_unlocked(workflow)
        host.store.advance()

    def _diagnostic(self, job, code, work, public, row):
        from .staged_diagnostics import execute
        result = execute(self, job, code, work, row)
        self._archive(work, public)
        return {**result, 'evidence_path': public.relative_to(self.coral / 'public').as_posix()}
