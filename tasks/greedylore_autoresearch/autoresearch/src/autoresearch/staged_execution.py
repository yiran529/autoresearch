"""Expensive execution boundary for the trusted staged controller."""
from __future__ import annotations

import hashlib
import json
import math
import os
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
        if job['kind'] != 'diagnostic':
            self._prepare_environment(code, work)
        try:
            self._ready(job['context'])
        except InfraFailure as exc:
            exc.phase = 'before_launch'
            raise
        contract = EvalContract.load(self.coral / 'private/eval_contract.yaml')
        if job['kind'] == 'diagnostic':
            return self._diagnostic(job, code, work, public, row)
        if job['kind'] in {'timing', 'final_timing'}:
            return self._timing(job, code, work, public, contract)
        if job['kind'] == 'final_quality':
            recovered = self.recover_final(job)
            if recovered is not None:
                return recovered
        name = job['workload'] if job['kind'] == 'final_quality' else (
            contract.final_workloads[0] if job['kind'] == 'transfer' else contract.development_workload)
        profile = contract.quality[name]
        target = job.get('pause_after_steps') or profile['stop_after_steps']
        if target > profile['stop_after_steps']:
            raise ValueError('pause endpoint exceeds fixed training budget')
        output = work / 'quality' / name
        # A segment result cache prevents a crash during publication from
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
            if profile['model'] == 'resnet18':
                from greedylore_autoresearch_grader.vision import validate_vision_result
                validate_vision_result(payload, profile)
            elif (payload.get('status') not in {'paused', 'completed'}
                    or payload.get('contract_hash') != contract.contract_hash
                    or payload.get('world_size') != contract.hardware.gpu_count
                    or payload.get('update_step') != target
                    or (target < profile['stop_after_steps']) != (payload['status'] == 'paused')):
                raise ValueError('invalid staged quality result')
            if profile['model'] != 'resnet18' and payload['status'] == 'completed' and not math.isfinite(float(payload.get('final_eval_loss', math.nan))):
                raise ValueError('invalid final evaluation loss')
            checkpoint_value = payload.get('checkpoint_path') or (payload.get('diagnostic_checkpoint') or {}).get('path')
            if profile['model'] == 'resnet18' and (payload.get('diagnostic_checkpoint') or {}).get('status') == 'complete':
                checkpoint_value = str(work / 'checkpoints' / name / f"step-{payload['update_step']:08d}")
            if checkpoint_value:
                checkpoint = safe_path(Path(checkpoint_value))
                if not checkpoint.is_relative_to(work / 'checkpoints' / name):
                    raise ValueError('candidate checkpoint path escapes this segment')
                self.artifacts.publish_checkpoint(host.node_id, checkpoint,
                    public / 'checkpoints' / name / 'manifest.json')
            _atomic_json(cache, payload)
        observation = {**payload, 'run_id': run_id, 'evidence_path': public.relative_to(self.coral / 'public').as_posix(),
                       'eligible': False}
        if payload['status'] == 'completed' and job['kind'] == 'train':
            reference = self._grader(code, work, {})._load_references(contract.contract_hash)
            passed = quality_gate(payload['final_eval_loss'],
                reference['quality']['workloads'][name]['final_eval_loss'],
                max_relative=contract.raw['quality']['max_relative_eval_loss'])
            observation.update(quality_pass=passed)
        if job['kind'] == 'final_quality':
            self._save_final_quality(job, observation)
        self._archive(work, public)
        return observation

    def _final_cache(self, job):
        return safe_path(self.coral / 'private/research_final_results' / self.host.node_id / f"{job['request_id']}.json")

    def _save_final_quality(self, job, result):
        _atomic_json(self._final_cache(job), {'sha': job['sha'], 'workload': job['workload'],
            'contract_hash': job['canonical']['contract_hash'], 'result': result,
            'sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})

    def publish_final(self, job, result):
        if job['kind'] == 'final_timing':
            return self.publish_timing(job, result)
        work = self.artifacts.path(self.host.node_id, f"runs/{job['request_id']}")
        work.mkdir(parents=True, exist_ok=True)
        public = self.coral / 'public/eval_logs' / job['sha'] / job['request_id']
        result.setdefault('evidence_path', public.relative_to(self.coral / 'public').as_posix())
        _atomic_json(work / 'final-result.json', result, mode=0o644)
        self._archive(work, public)

    def _validate_final_quality(self, result, job, contract):
        if job['canonical']['contract_hash'] != contract.contract_hash:
            raise ValueError('final request contract mismatch')
        if not isinstance(result, dict):
            raise ValueError('final quality result must be an object')
        profile = contract.quality[job['workload']]
        if profile['model'] == 'resnet18':
            from greedylore_autoresearch_grader.vision import validate_vision_result
            validate_vision_result(result, profile)
        else:
            loss = result.get('final_eval_loss')
            if (result.get('status') != 'completed' or result.get('contract_hash') != contract.contract_hash
                    or type(result.get('world_size')) is not int
                    or result['world_size'] != contract.hardware.gpu_count
                    or type(result.get('update_step')) is not int
                    or result['update_step'] != profile['stop_after_steps']
                    or type(loss) not in (int, float) or not math.isfinite(loss) or loss <= 0):
                raise ValueError('invalid cached final language result')

    def _recover_external_final(self, job, contract):
        from greedylore_autoresearch_grader.quality import build_quality_command

        work = self.artifacts.path(self.host.node_id, f"runs/{job['request_id']}")
        output = work / 'quality' / job['workload']
        result_path = safe_path(work / 'quality-result.json')
        if not result_path.exists():
            result_path = safe_path(output / 'all_results.json')
        process_path = safe_path(work / 'process.json')
        if not result_path.exists() or not process_path.exists():
            return None
        process = json.loads(process_path.read_text())
        profile = contract.quality[job['workload']]
        code = self.artifacts.path(self.host.node_id, f"code/{job['sha']}")
        expected = build_quality_command(profile, code, output,
            list(range(contract.hardware.gpu_count)),
            checkpoint_root=work / 'checkpoints' / job['workload'],
            checkpoint_interval_steps=self.interval,
            training_metrics=work / 'training_metrics.jsonl')
        # process.json is host-produced after the child exits. Match its complete
        # command, including candidate snapshot, contract and request-local output;
        # self-declared payload identity cannot certify a different run.
        if (not isinstance(process, dict) or process.get('command') != list(expected.command)
                or type(process.get('returncode')) is not int or process['returncode'] != 0
                or process.get('timed_out') is not False or process.get('infra_failure') is not None):
            raise ValueError('external final process is incomplete or belongs to another run')
        payload = json.loads(result_path.read_text())
        self._validate_final_quality(payload, job, contract)
        public = self.coral / 'public/eval_logs' / job['sha'] / job['request_id']
        result = {**payload, 'run_id': job['request_id'],
                  'evidence_path': public.relative_to(self.coral / 'public').as_posix(), 'eligible': False}
        # Persist completed work before public projection, including when the
        # previous host stopped between all_results and quality-result writes.
        self._save_final_quality(job, result)
        self._archive(work, public)
        return result

    def recover_final(self, job):
        if job['kind'] == 'final_timing':
            return self.recover_timing(job)
        from greedylore_autoresearch_grader.contract import EvalContract
        from greedylore_autoresearch_grader.gpu import InfraFailure
        cache = self._final_cache(job)
        try:
            contract = EvalContract.load(self.coral / 'private/eval_contract.yaml')
            if not cache.exists():
                return self._recover_external_final(job, contract)
            saved = json.loads(cache.read_text())
            result = saved['result']
            if (saved['sha'] != job['sha'] or saved['workload'] != job['workload']
                    or saved['contract_hash'] != job['canonical']['contract_hash']
                    or saved['sha256'] != hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()):
                raise ValueError('final cache identity mismatch')
            self._validate_final_quality(result, job, contract)
        except (KeyError, TypeError, ValueError) as exc:
            raise InfraFailure(f'invalid final component cache: {exc}') from exc
        return result

    def recover_timing(self, job):
        from greedylore_autoresearch_grader.gpu import InfraFailure

        from .staged_timing import timing_context, validate_result
        cache = self._timing_cache(job)
        if not cache.exists():
            return None
        try:
            saved = json.loads(cache.read_text())
            result = saved['result']
            digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
            if (saved.get('version') != 1 or saved.get('sha') != job['sha']
                    or saved.get('context') != job['timing_context'] or saved.get('result_sha256') != digest
                    or job['timing_context'] != timing_context(self.host, job['canonical'])):
                raise ValueError('timing cache identity mismatch')
            validate_result(result, job['timing_context'])
        except (KeyError, TypeError, ValueError) as exc:
            raise InfraFailure(f'invalid durable timing cache: {exc}') from exc
        return result

    def _timing_cache(self, job):
        return safe_path(self.coral / 'private/research_timing_results' / self.host.node_id / f"{job['request_id']}.json")

    def publish_timing(self, job, result):
        work = self.artifacts.path(self.host.node_id, f"runs/{job['request_id']}")
        work.mkdir(parents=True, exist_ok=True)
        public = self.coral / 'public/eval_logs' / job['sha'] / job['request_id']
        result.setdefault('evidence_path', public.relative_to(self.coral / 'public').as_posix())
        _atomic_json(work / 'timing-result.json', result, mode=0o644)
        self._archive(work, public)

    def _timing(self, job, code, work, public, contract):
        recovered = self.recover_timing(job)
        if recovered is not None:
            return recovered
        grader = self._grader(code, work / 'timing', job['canonical'])
        reference = grader._load_references(contract.contract_hash)
        bundle = grader._formal(contract, 'timing', reference)
        # This is an efficiency measurement, not an Engineering Attempt or a
        # provisional inherited-quality result. Final Research scoring is later.
        bundle.scores = {key: value for key, value in bundle.scores.items()
                         if key.endswith(('_timing_mean', '_timing_spread', '_speedup')) or key == 'efficiency_score'}
        bundle.metadata.pop('utility', None)
        bundle.metadata.update(measurement_kind='research_timing', quality_source='not_measured')
        result = {'status': 'completed', 'bundle': bundle.to_dict(),
                  'performance_eligible': bool(bundle.metadata.get('performance_eligible')),
                  'evidence_path': public.relative_to(self.coral / 'public').as_posix()}
        from .staged_timing import validate_result
        validate_result(result, job['timing_context'])
        _atomic_json(self._timing_cache(job), {'version': 1, 'sha': job['sha'],
            'context': job['timing_context'], 'result': result,
            'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
        return result

    def _prepare_environment(self, code, work):
        """Warm a snapshot's locked environment before reserving training GPUs."""
        import subprocess

        from greedylore_autoresearch_grader.gpu import InfraFailure

        environment = {**os.environ, **self._environment(work), 'UV_OFFLINE': '1'}
        command = ['uv', 'sync', '--frozen', '--offline', '--project', str(code)]
        command_path = work / 'environment-command.json'
        _atomic_json(command_path, {'command': command, 'uv_cache_dir': environment['UV_CACHE_DIR']})
        try:
            with (work / 'environment.stdout.log').open('w') as stdout, (work / 'environment.stderr.log').open('w') as stderr:
                result = subprocess.run(command, cwd=code, env=environment, stdout=stdout, stderr=stderr, timeout=600)
            if result.returncode:
                raise RuntimeError((work / 'environment.stderr.log').read_text()[-4000:])
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            detail = {'phase': 'environment_setup', 'node_id': self.host.node_id,
                      'reason': str(error), 'evidence': str(command_path)}
            _atomic_json(self.coral / 'private/research_infrastructure_block.json', detail)
            failure = InfraFailure(f'candidate environment preparation failed: {error}')
            failure.phase = 'environment_setup'
            raise failure from error

    def _environment(self, work):
        root = self.artifacts.node_root(self.host.node_id) / 'caches'
        return {'TMPDIR': str(work), 'UV_CACHE_DIR': str(Path(self.options.get('research_uv_cache_dir') or os.environ.get('UV_CACHE_DIR') or Path.home() / '.cache/uv').expanduser()),
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
        setup_only = row.get('observations') and all(
            observation.get('status') == 'infra_failed' and observation.get('phase') in {'environment_setup', 'before_launch'}
            for observation in row['observations'])
        if len(row['runs']) > 1 and not setup_only:
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
        from greedylore_autoresearch_grader.efficiency import timing_cell_labels
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
                from .staged_timing import latest_timing, timing_context
                observation = row['observations'][-1]
                canonical = host.validate_candidate(sha)
                final_components = state.get('final_evaluation', {}).get('components')
                timing_entry = ({'result': final_components['timing'][-1]['result']} if final_components else
                                latest_timing(row, timing_context(host, canonical)))
                if timing_entry is None and observation.get('timing'):
                    # A Node frozen under the previous staged protocol retains
                    # its original certified measurement; never relabel its context.
                    timing_entry = {'result': {'bundle': observation['timing']}}
                timing_bundle = ScoreBundle.from_dict(timing_entry['result']['bundle'])
                timing = {cell: {'mean': timing_bundle.scores[f'{label}_timing_mean'].value,
                                'relative_spread': timing_bundle.scores[f'{label}_timing_spread'].value}
                          for cell, label in timing_cell_labels(contract.efficiency.cells).items()}
                reference = self._grader(self.artifacts.path(host.node_id, f'code/{sha}'),
                    self.artifacts.path(host.node_id, 'publication'), canonical)._load_references(contract.contract_hash)
                metadata = timing_bundle.metadata
                from greedylore_autoresearch_grader.timing_policy import reference_for_window
                reference, _ = reference_for_window(reference,
                    (metadata.get('timing_protocol') or {}).get('measured_iterations', contract.raw['efficiency']['measured_iterations']),
                    contract.raw['efficiency']['measured_iterations'])
                quality = {contract.development_workload: observation}
                if final_components:
                    quality.update({name: final_components[name][-1]['result'] for name in contract.final_workloads})
                else:
                    quality[contract.final_workloads[0]] = state['transfer']
                bundle = score_results('research', quality, timing, reference,
                    contract_hash=contract.contract_hash,
                    comparable=metadata['comparability'] != 'incomparable_hardware', evidence_paths=[],
                    max_relative_eval_loss=contract.raw['quality']['max_relative_eval_loss'],
                    hardware_comparison=metadata.get('hardware_comparison'),
                    timing_protocol=metadata.get('timing_protocol'),
                    performance_eligible=metadata.get('performance_eligible', False), quality_profiles=contract.quality,
                    utility_version=contract.utility_version)
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
            for component in state.get('final_evaluation', {}).get('components', {}).values():
                paths.extend(item['result']['evidence_path'] for item in component
                             if item.get('result', {}).get('evidence_path'))
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
