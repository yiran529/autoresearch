"""Offline expensive boundaries for the real staged workflow integration driver."""
# ruff: noqa: S101

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import simulation_support as sim
import yaml


def reference():
    return {'quality': {'workloads': {name: {'final_eval_loss': loss} for name, loss in
            [('llama130m_muon', 4.0), ('llama60m_adamw', 3.0)]}},
            'efficiency': {'cells': {name: {'mean_iteration_seconds': .1} for name in
            ['130m_adamw_bf16', '130m_adamw_fp32']}}}


class InventoryLease:
    """Simulated GPU ownership; lease flocks have a separate multiprocess test."""
    def __init__(self, *_):
        self.reservation = SimpleNamespace(indices=(0, 1, 2, 3), check_exclusive=lambda *_: None)
    def ensure(self, context):
        sim.event('gpu_inventory_simulated', owner=context['agent_id'])
        return True
    def check(self): return True
    def borrow(self): return self.reservation
    def register_job(self, *_): pass
    def clear_job(self, *_): pass
    def release(self): sim.event('lease_released')


def quality_process(spec, **kwargs):
    from greedylore_autoresearch_grader.process import ProcessResult
    args = list(spec.command)
    def value(flag): return args[args.index(flag) + 1]
    if '--resume_checkpoint' in args:
        assert (Path(value('--resume_checkpoint')) / 'manifest.json').is_file()
    output = Path(value('--output_dir'))
    name = output.name
    transfer = name == 'llama60m_adamw'
    sim.event('quality_segment', workload=name, command=args,
              resume=value('--resume_checkpoint') if '--resume_checkpoint' in args else None)
    if transfer and os.environ['STAGED_SCENARIO'] == 'transfer_failed':
        raise ValueError('SIMULATED transfer algorithm failure')
    target = int(value('--pause_after_steps') if '--pause_after_steps' in args else value('--stop_after_steps'))
    paused = target < int(value('--stop_after_steps'))
    checkpoint = Path(value('--checkpoint_root')) / f'step-{target}'
    checkpoint.mkdir(parents=True)
    # Real artifact ledger sees a manifest plus a tensor that must stay external.
    files = {}
    for name_part in ['model.pt', *(f'rank-{i}.pt' for i in range(4))]:
        payload_bytes = b'SIMULATED optimizer/RNG/compressor state'
        (checkpoint / name_part).write_bytes(payload_bytes)
        files[name_part] = {'sha256': hashlib.sha256(payload_bytes).hexdigest(), 'size_bytes': len(payload_bytes)}
    manifest = {'version': 1, 'status': 'complete', 'update_step': target, 'world_size': 4,
                'contract_hash': value('--contract_hash'), 'files': files}
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    output.mkdir(parents=True)
    loss = (3.01 if os.environ['STAGED_SCENARIO'] == 'success' else 10.0) if transfer else (10.0 if os.environ['STAGED_SCENARIO'] == 'no_eligible' else 4.01)
    payload = {'status': 'paused' if paused else 'completed', 'contract_hash': value('--contract_hash'),
               'world_size': 4, 'update_step': target, 'checkpoint_path': str(checkpoint),
               'final_eval_loss': loss, 'simulated': True}
    (output / 'all_results.json').write_text(json.dumps(payload))
    Path(value('--training_metrics')).write_text(json.dumps({'update_step': target, 'simulated': True}) + '\n')
    spec.log_dir.mkdir(parents=True)
    stdout, stderr = spec.log_dir / 'stdout.log', spec.log_dir / 'stderr.log'
    stdout.write_text('SIMULATED quality process\n')
    stderr.write_text('')
    return ProcessResult(spec.command, spec.env, 0, 'simulated', 'simulated', False, (),
                         str(stdout), str(stderr), '')


def timing_formal(grader, contract, mode, refs):
    from greedylore_autoresearch_grader.grader import score_results
    from greedylore_autoresearch_grader.timing_policy import plan_timing
    sim.event('timing_execution', mode=mode, code=str(grader.codebase_path))
    plan = plan_timing(grader.tasks[0].metadata.get('timing_declaration', {'period_steps': contract.raw['compressor']['update_gap']}),
        minimum=contract.raw['efficiency']['measured_iterations'], maximum=800)
    return score_results('engineering', {}, {name: {'mean': .1, 'relative_spread': .01}
        for name in refs['efficiency']['cells']}, refs, contract_hash=contract.contract_hash,
        comparable=True, evidence_paths=[], timing_protocol={**plan, 'simulated': True})


def driver(output, scenario):
    from autoresearch.lifecycle import run_worker
    from autoresearch.staged_research import StagedResearch
    from autoresearch.state import WorkflowStore

    from coral.grader.trusted_writes import (
        AgentBinding,
        TrustedWriteServer,
        TrustedWriteService,
        process_start_time,
        write_trusted_agent_registry,
    )

    os.environ.update(CORAL_SIM_ROOT=str(output), CORAL_SIM_SCENARIO='normal', STAGED_SCENARIO=scenario)
    binary = output / 'bin'
    binary.mkdir()
    codex = binary / 'codex'
    codex.write_text(f'#!{sys.executable}\nfrom prompt_audit_support import main\nmain()\n')
    codex.chmod(0o755)
    os.environ['PATH'] = str(binary) + os.pathsep + os.environ['PATH']
    config, paths = sim.prepare()
    saved = yaml.safe_load((paths.coral_dir / 'config.yaml').read_text())
    saved['grader']['args'].update(staged_research=True, dynamic_artifact_root=str(output / 'tensors'),
                                 dynamic_artifact_min_free_bytes=0, research_planning_lease_seconds=0)
    (paths.coral_dir / 'config.yaml').write_text(yaml.safe_dump(saved))
    config.grader.args.update(saved['grader']['args'])
    store = WorkflowStore(paths.coral_dir, repo_path=paths.repo_dir)
    assignment = store.reconcile(1, 0)[0]
    aid, node = assignment.agent_id, assignment.node_id
    repo = paths.repo_dir
    with (repo / '.git/info/exclude').open('a') as stream:
        stream.write('\n.coral_dir\n.coral_agent_id\nprobes/\n')
    (repo / '.coral_dir').write_text(str(paths.coral_dir))
    (repo / '.coral_agent_id').write_text(aid)
    write_trusted_agent_registry(paths.coral_dir, [AgentBinding(aid, os.getpid(), process_start_time(os.getpid()), str(repo), None)])
    os.chdir(repo)
    service = TrustedWriteService(paths.coral_dir, config)
    server = TrustedWriteServer(paths.coral_dir, service)
    def action(kind, data=None, request_id=None, check=True, target=None):
        file = output / 'action.json'
        file.write_text(json.dumps(data or {}))
        return sim.cli('autoresearch.cli', 'research', kind, '--node', target or node,
                       '--file', str(file), '--request-id', request_id or kind, check=check)
    with ExitStack() as stack:
        # Scheduling is controlled by the driver, while queueing/draining remain real.
        stack.enter_context(patch('coral.grader.trusted_worker.start_task_worker', lambda *_a, **_k: None))
        stack.enter_context(patch('autoresearch.lease.ResearchLease', InventoryLease))
        stack.enter_context(patch('greedylore_autoresearch_grader.process.run_process_group', quality_process))
        stack.enter_context(patch('greedylore_autoresearch_grader.grader.Grader._load_references', lambda *_: reference()))
        stack.enter_context(patch('greedylore_autoresearch_grader.grader.Grader._formal', timing_formal))
        server.start()
        try:
            # Root Engineering is part of the real parent completion prerequisite.
            def bind(identity):
                (repo / '.coral_agent_id').write_text(identity)
                write_trusted_agent_registry(paths.coral_dir, [AgentBinding(identity, os.getpid(),
                    process_start_time(os.getpid()), str(repo), None)])
            def engineering(identity, target):
                from coral.grader.daemon import process_pending_once
                bind(identity)
                spec_path = paths.coral_dir / f'public/notes/nodes/{target}/spec.md'
                original_spec = spec_path.read_text()
                def tamper_root():
                    altered = yaml.safe_load(original_spec)
                    altered['hypothesis'] = 'SIMULATED tampered baseline policy'
                    spec_path.write_text(yaml.safe_dump(altered))
                if target == 'N000':
                    tamper_root()
                    rejected = sim.cli('autoresearch.cli', 'engineering', 'claim', target, check=False)
                    assert rejected.returncode != 0 and 'baseline' in rejected.stderr
                    spec_path.write_text(original_spec)
                claim = json.loads(sim.cli('autoresearch.cli', 'engineering', 'claim', target).stdout)
                base = sim.git(repo, 'rev-parse', 'HEAD')
                if target == 'N000':
                    assert claim['design_source'] == {'path': 'notes/nodes/N000/spec.md', 'iteration': None}
                else:
                    record = json.loads((paths.coral_dir / 'public' / claim['design_source']['path']).read_text())
                    selected = record['iterations'][claim['design_source']['iteration'] - 1]
                    assert selected['code_sha'] == base == claim['research_sha']
                    assert record['selected_iteration'] < len(record['iterations'])
                stop = threading.Event()
                def grade():
                    while not stop.is_set():
                        process_pending_once(paths.coral_dir)
                        time.sleep(.02)
                thread = threading.Thread(target=grade, daemon=True)
                thread.start()
                try:
                    for index in range(1, 4):
                        sim.declaration(paths.coral_dir, target, 'engineering', base, index)
                        sim.git(repo, 'add', 'submission.yaml', 'simulation_marker.txt')
                        sim.git(repo, 'commit', '-m', f'SIMULATED Engineering {index}')
                        sha = sim.git(repo, 'rev-parse', 'HEAD')
                        if target == 'N000' and index == 1:
                            tamper_root()
                            rejected = sim.cli('autoresearch.cli', 'engineering', 'timing', '--sha', sha,
                                               '--request-id', 'tampered-root', check=False)
                            assert rejected.returncode != 0 and 'baseline' in rejected.stderr
                            spec_path.write_text(original_spec)
                        args = ('engineering', 'timing', '--sha', sha, '--request-id', f'timing-{index}')
                        result = json.loads(sim.cli('autoresearch.cli', *args).stdout)
                        assert result['status'] != 'pending'
                        assert result['metadata']['quality_anchor'] == base
                        repeated = json.loads(sim.cli('autoresearch.cli', *args).stdout)
                        assert repeated['metadata']['grader_invocation_count'] == result['metadata']['grader_invocation_count']
                    sim.declaration(paths.coral_dir, target, 'engineering', base, 4)
                    sim.git(repo, 'add', 'submission.yaml', 'simulation_marker.txt')
                    sim.git(repo, 'commit', '-m', 'SIMULATED over-budget Engineering')
                    fourth = sim.cli('autoresearch.cli', 'engineering', 'timing', '--sha', sim.git(repo, 'rev-parse', 'HEAD'),
                                     '--request-id', 'timing-4', check=False)
                    assert fourth.returncode != 0 and 'run limit' in fourth.stderr
                    sim.event('engineering_timing_verified', node=target, anchor=base,
                              selected_iteration=claim['design_source']['iteration'],
                              formal_runs=3, duplicate_no_execution=True, fourth_rejected=True)
                finally:
                    stop.set()
                    thread.join(timeout=5)
                from coral.hub.attempts import read_attempts
                shas = [a.commit_hash for a in read_attempts(paths.coral_dir) if a.agent_id == identity]
                summary = output / 'engineering.json'
                summary.write_text(json.dumps({'summary': 'SIMULATED Engineering terminal',
                    'findings': ['Synthetic fixture only'], 'limitations': ['No real GPU result'], 'attempt_shas': shas}))
                sim.cli('autoresearch.cli', 'engineering', 'finish', '--analysis-file', str(summary))
                store.advance()
            engineering(aid, node)
            assignment = store.reconcile(1, 0)[0]
            aid, node = assignment.agent_id, assignment.node_id
            bind(aid)
            # Actual AgentManager -> CodexRuntime -> test executable captures
            # initial and resumed delivery. It performs no model/GPU work.
            from coral.agent.dynamic import spec_for_assignment
            from coral.agent.manager import AgentManager
            manager = AgentManager(config, config_dir=sim.TASK)
            manager.paths = paths
            spec = spec_for_assignment(config, assignment)
            manager.specs = [spec]
            manager.specs_by_id[aid] = spec
            manager.runtimes[aid] = manager.runtime
            with patch.dict(os.environ, {'CORAL_SIM_CAPTURE_ONLY': '1'}):
                for session in (None, f'simulation-{aid}'):
                    handle = manager._setup_and_start_agent(aid, resume_session_id=session)
                    assert handle.process.wait(timeout=20) == 0
            captures = [json.loads(p.read_text()) for p in (output / 'prompt_audit').glob('*.json')]
            delivered = [p for p in captures if p['role'] == 'researcher']
            assert len(delivered) == 2 and {p['resumed'] for p in delivered} == {False, True}
            for capture in delivered:
                assert capture['role_text'] == (sim.TASK / 'roles/researcher.md').read_text()
                assert capture['resources']['workflow_rules.md'] == (sim.TASK / 'roles/workflow_rules.md').read_text()
                assert capture['resources']['references/research_commands.md'] == (sim.TASK / 'references/research_commands.md').read_text()
                assert 'templates/research_origin.json' in capture['resources']
                assert 'templates/research_intent.json' not in capture['resources']
                for content in (capture['instructions'], capture['startup_prompt']):
                    assert 'autoresearch research register' in content
                    assert 'at most 3 candidate training trajectories' in content
                    assert 'autoresearch node complete --mode research' not in content
                    assert '3 successful scored formal Research runs' not in content
            sim.git(repo, 'reset', '--hard', store._terminal_research_unlocked('N000').commit_hash)
            mode = store._load_unlocked()['nodes'][node]['research']['research_mode']
            origin = dict(parent_node='N000', hypothesis='SIMULATED history helps',
                selection_rationale='Synthetic Root', delta='Synthetic marker', risk='Synthetic only',
                falsification_criteria='Workflow fails',
                question='SIMULATED history mechanism question',
                problem_ref=None if mode == 'EXPLORE' else 'N000:Q1',
                parent_response='Investigate synthetic Root mechanism.',
                engineering_review='Read synthetic parent Engineering handover.')
            design = dict(method='Synthetic history', mechanism='retain history',
                          predictions='less distortion', tests='matched-state probe')
            registered = json.loads(action('register', {'origin': origin, 'design': design}).stdout)
            assert registered['version'] == 3 and 'research_intent' not in registered['origin']
            sim.event('origin_registered', question=registered['question'],
                      initial_design_sha256=registered['origin']['initial_design_sha256'])
            sim.declaration(paths.coral_dir, node, 'research', sim.git(repo, 'rev-parse', 'HEAD'), 1)
            sim.git(repo, 'add', 'submission.yaml', 'simulation_marker.txt')
            sim.git(repo, 'commit', '-m', 'SIMULATED staged candidate')
            sha = sim.git(repo, 'rev-parse', 'HEAD')
            bind('engineer-E0001')
            rejected = action('status', check=False).returncode != 0
            bind(aid)
            def drain():
                # Reconstruct host and backend at every segment (durable restart boundary).
                host = StagedResearch(store, aid, node)
                host.drain()
                return host
            def timing_count():
                return sum(e['event'] == 'timing_execution' and e['mode'] == 'timing' for e in sim.events())
            timing_phase = 'before_training' if scenario in {'success', 'no_eligible'} else (
                'paused' if scenario == 'transfer_quality_failed' else 'finish_fallback')
            if timing_phase == 'before_training':
                action('timing', {'iteration': 1, 'sha': sha}, 'pre-timing')
                host = drain()
                row = host.record.read()['iterations'][0]
                assert row['status'] == 'planned' and not row['runs'] and not row['observations']
                assert timing_count() == 1
                result = host.status()['jobs']['pre-timing']['result']
                assert result['bundle']['metadata']['quality_source'] == 'not_measured'
                assert 'quality_gate' not in result['bundle']['scores']
                action('timing', {'iteration': 1, 'sha': sha}, 'pre-timing')
                drain()
                action('timing', {'iteration': 1, 'sha': sha}, 'reuse-timing')
                drain()
                assert timing_count() == 1
                rejected_sha = action('train', {'iteration': 1, 'sha': 'f'*40}, 'changed-sha', check=False)
                assert rejected_sha.returncode != 0
            action('train', {'iteration': 1, 'sha': sha, 'pause_after_steps': 1000}, 'pause')
            host = drain()
            assert host.record.read()['iterations'][0]['status'] == 'paused', host.status()
            before = len([e for e in sim.events() if e['event'] == 'quality_segment'])
            action('train', {'iteration': 1, 'sha': sha, 'pause_after_steps': 1000}, 'pause')
            drain()
            repeated = len([e for e in sim.events() if e['event'] == 'quality_segment']) == before
            assert Path(host.record.read()['iterations'][0]['observations'][0]['checkpoint_path']).exists()
            scripts = repo / 'probes'
            scripts.mkdir()
            (scripts / 'probe.py').write_text("print('SIMULATED probe')\n")
            action('diagnostic', {'iteration': 1, 'argv': ['python', '/workspace/experiments/probe.py'],
                    'timeout': 5, 'scripts_dir': str(scripts)}, 'probe')
            drain()
            if timing_phase == 'paused':
                action('timing', {'iteration': 1, 'sha': sha}, 'paused-timing')
                host = drain()
                assert host.record.read()['iterations'][0]['status'] == 'paused'
            timing_before_resume = timing_count()
            action('train', {'iteration': 1, 'sha': sha}, 'resume')
            host = drain()
            assert host.record.read()['iterations'][0]['status'] == 'completed', host.status()
            assert timing_count() == timing_before_resume  # Quality completion never starts timing.
            budget = False
            if scenario == 'no_eligible':
                for i in (2, 3):
                    action('design', {**design, 'method': f'candidate {i}'}, f'design-{i}')
                    action('decision', {'iteration': i, 'action': 'stop', 'reason': 'refuted'}, f'stop-{i}')
                budget = action('design', {**design, 'method': 'forbidden fourth'}, 'design-4', check=False).returncode != 0
            elif scenario == 'success':
                # An abandoned later design must not replace the selected code or design.
                action('design', {**design, 'method': 'unexecuted later design'}, 'design-2')
                action('decision', {'iteration': 2, 'action': 'stop',
                       'reason': 'No justified refinement; retain eligible iteration 1'}, 'stop-2')
            finish = {'selected_iteration': None if scenario == 'no_eligible' else 1, 'reason': 'frozen synthetic result'}
            action('finish', finish, 'finish')
            host = drain()
            assert host.status()['transfer_terminal'], host.status()
            assert timing_count() == 1
            action('finish', finish, 'finish')
            drain()
            frozen_rejected = action('design', design, 'late', check=False).returncode != 0
            # Freeze evidence, launch exactly one independent Research Analysis, publish and reclaim.
            outcome = run_worker(str(paths.coral_dir), runner=sim.analysis_runner, checkout_root=repo, only_node=node)
            assert not outcome['errors'], outcome
            store.advance()
            if scenario == 'success':
                worker = next(a for a in store.reconcile(0, 1) if a.node_id == node)
                engineering(worker.agent_id, node)
            final = store._load_unlocked()['nodes'][node]
            ledger = json.loads((paths.coral_dir / 'private/dynamic_artifacts.json').read_text())
            assert ledger['nodes'][node]['cleaned'], final
            assert not Path(ledger['nodes'][node]['path']).exists()
            segments = [e for e in sim.events() if e['event'] == 'quality_segment']
            analysis = list((store.notes_root / 'nodes' / node).glob('research_analysis_v*.json'))
            sealed = json.loads(analysis[0].read_text())
            records = sealed['evidence_manifest']['evidence_records']
            assert any(item['path'] == f'notes/nodes/{node}/research.json' for item in records)
            assert any(item['path'].startswith(f'research_diagnostics/{node}/probe/') for item in records)
            timings = host.record.read()['iterations'][0]['timings']
            assert len(timings) == 1
            assert any(item['path'].startswith(timings[0]['result']['evidence_path'] + '/') for item in records)
            assert not list((paths.coral_dir / 'public').rglob('*.pt'))
            assert sum(e['workload'] == 'llama130m_muon' for e in segments) == 2
            report = dict(simulated=True, scenario=scenario, authenticated_cli=True, staged_prompt_delivery=True,
                other_owner_rejected=rejected, analysis_count=len(analysis),
                engineering_status=final['engineering_status'], cleanup_status=final['cleanup']['status'],
                public_evidence_survives=all((paths.coral_dir / 'public' / ob['evidence_path']).exists()
                    for ob in host.record.read()['iterations'][0]['observations'] if ob.get('evidence_path')),
                same_sha_resume=bool(segments[1]['resume']), repeat_request_idempotent=repeated,
                post_freeze_rejected=frozen_rejected, transfer_runs=sum(e['workload']=='llama60m_adamw' for e in segments),
                budget_enforced=budget, limitations=['GPU ownership, quality and timing are simulated.',
                    'Diagnostic may be unavailable when bubblewrap is absent; no sandbox execution claim.'])
            report.update(timing_phase=timing_phase, timing_executions=timing_count(),
                independent_timing=True, no_timing_on_resume=True,
                timing_evidence_survives=(paths.coral_dir / 'public' / timings[0]['result']['evidence_path']).exists())
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        finally:
            server.stop()


if __name__ == '__main__':
    driver(Path(sys.argv[1]).resolve(), sys.argv[2])
