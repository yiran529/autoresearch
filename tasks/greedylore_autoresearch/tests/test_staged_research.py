"""Exercise the real staged controller with only expensive execution replaced."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_research_record import design, initialize, record


def timing_result(contract_hash='fixed'):
    from coral.types import Score, ScoreBundle
    bundle = ScoreBundle(scores={
        'bf16_timing_mean': Score(.1, 'bf16_timing_mean'),
        'bf16_timing_spread': Score(.01, 'bf16_timing_spread'),
        'fp32_timing_mean': Score(.1, 'fp32_timing_mean'),
        'fp32_timing_spread': Score(.01, 'fp32_timing_spread'),
    }, aggregated=1.0, metadata={'performance_eligible': True, 'comparability': 'comparable',
        'contract_hash': contract_hash, 'failure_class': None})
    return {'status': 'completed', 'performance_eligible': True, 'bundle': bundle.to_dict()}


class Backend:
    def __init__(self):
        self.calls = []
        self.fail_transfer = False
    def run(self, host, job):
        self.calls.append((job['kind'], job.get('sha'), job.get('pause_after_steps')))
        if job['kind'] == 'timing':
            result = timing_result(job['canonical'].get('contract_hash'))
            result['bundle']['metadata']['timing_protocol'] = job['timing_context']['plan']
            return result
        if job['kind'] == 'transfer' and self.fail_transfer:
            raise ValueError('simulated candidate failure')
        return {'status': 'paused' if job.get('pause_after_steps') else 'completed',
                'update_step': job.get('pause_after_steps') or 2500, 'eligible': True,
                'checkpoint_path': None, 'simulated': True}
    def finish(self, host, selected):
        self.calls.append(('publish', selected))
    def release(self):
        self.calls.append(('release',))


def host(tmp_path):
    from autoresearch.staged_research import StagedResearch
    r = record(tmp_path)
    initialize(r)
    source = Path(__file__).resolve().parents[1] / 'taskdata/eval_contract_v1.yaml'
    (r.store.coral_dir / 'private/eval_contract.yaml').write_text(source.read_text())
    backend = Backend()
    result = StagedResearch(r.store, r.agent_id, r.node_id, backend=backend)
    # validate_candidate is a git/submission boundary; controller ownership and budgets stay real.
    result.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': 'fixed'}
    return result, r, backend


def test_pause_resume_freeze_transfer_is_idempotent_and_no_retuning(tmp_path):
    h, r, b = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40, 'pause_after_steps': 1000}, 'run1', {})
    h.drain()
    assert r.read()['iterations'][0]['status'] == 'paused'
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'run2', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'mechanism works'}, 'end', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'mechanism works'}, 'end', {})
    h.drain()
    assert sum(c[0] == 'transfer' for c in b.calls) == 1
    assert h.status()['transfer_terminal'] is True
    with pytest.raises(ValueError, match='frozen|terminal'):
        h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'late', {})


def test_failures_count_and_no_candidate_skips_transfer(tmp_path):
    h, r, b = host(tmp_path)
    for i in range(1, 4):
        if i > 1:
            r.append_design(design(str(i)), f'design{i}')
        r.decide(i, 'stop', 'mechanism refuted', f'stop{i}')
    with pytest.raises(ValueError, match='3|budget'):
        r.append_design(design('4'), 'design4')
    h.enqueue('finish', {'selected_iteration': None, 'reason': 'none worked', 'sha': 'b'*40}, 'end', {})
    h.drain()
    assert not any(c[0] == 'transfer' for c in b.calls)
    assert h.status()['transfer_terminal']
    assert ('publish', None) in b.calls


def test_failed_transfer_keeps_design_frozen_and_cannot_select_another(tmp_path):
    h, r, b = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'run', {})
    h.drain()
    b.fail_transfer = True
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'select'}, 'end', {})
    h.drain()
    assert r.read()['frozen']
    assert h.status()['transfer_terminal']
    assert h.status()['transfer']['status'] == 'failed'
    with pytest.raises(ValueError, match='frozen|terminal'):
        h.enqueue('finish', {'selected_iteration': None, 'reason': 'change'}, 'other', {})


def test_recover_crash_before_begin_does_not_wedge_queue(tmp_path):
    h, r, b = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'run', {})
    state = h._read()
    state['jobs']['run']['status'] = 'running'
    h._save(state)
    h.drain()
    assert h.status()['jobs']['run']['status'] != 'running'
    assert r.read()['iterations'][0]['status'] != 'running'


def test_publication_retry_reuses_completed_transfer(tmp_path):
    h, r, b = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'run', {})
    h.drain()
    finish = b.finish
    b.finish = lambda *args: (_ for _ in ()).throw(OSError('publication interrupted'))
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'select'}, 'end', {})
    h.drain()
    b.finish = finish
    h.drain()
    assert sum(c[0] == 'transfer' for c in b.calls) == 1
    assert h.status()['jobs']['end']['status'] == 'completed'


def test_transfer_infrastructure_retries_end_at_host_budget(tmp_path):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    h, r, b = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'run', {})
    h.drain()
    original = b.run
    def fail(host, job):
        if job['kind'] == 'transfer':
            raise InfraFailure('simulated infrastructure outage')
        return original(host, job)
    b.run = fail
    for i in range(3):
        h.enqueue('finish', {'selected_iteration': 1, 'reason': 'select'}, f'end{i}', {})
        h.drain()
    assert h.status()['transfer_terminal']
    assert h.status()['transfer']['retry_budget_exhausted']
    assert ('publish', 1) in b.calls and ('release',) in b.calls


def test_timing_failure_archives_logs_before_cleanup(tmp_path):
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_execution import Execution
    h, r, _ = host(tmp_path)
    config = {'grader': {'args': {'staged_research': True,
        'dynamic_artifact_root': str(tmp_path / 'dynamic'), 'dynamic_artifact_min_free_bytes': 0}}}
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    r.begin(1, 'b'*40, 'run.begin')
    backend = Execution.__new__(Execution)
    backend.host, backend.coral = h, h.store.coral_dir
    backend.artifacts = RuntimeArtifacts(h.store.coral_dir)
    def failed(host, job):
        work = backend.artifacts.path('N001', 'runs/run/timing')
        work.mkdir(parents=True)
        (work / 'failure.log').write_text('SIMULATED timing failure')
        raise ValueError('timing failed')
    backend._execute = failed
    with pytest.raises(ValueError, match='timing failed'):
        backend.run(h, {'kind': 'train', 'iteration': 1, 'sha': 'b'*40, 'request_id': 'run'})
    durable = h.store.coral_dir / 'public/eval_logs' / ('b'*40) / 'run/timing/failure.log'
    assert durable.read_text() == 'SIMULATED timing failure'
    backend.artifacts.cleanup_node('N001', terminal=True)
    assert durable.is_file()


def test_research_diagnostic_borrows_lease_without_reentering_queue(tmp_path, monkeypatch):
    import sys

    from autoresearch import diagnostic_sandbox as sandbox
    workspace = tmp_path / 'workspace'
    for name in ('results', 'experiments'):
        (workspace / name).mkdir(parents=True)
    (workspace / 'experiments/probe.py').write_text('# simulated probe\n')
    identities = []
    class Borrowed:
        uuids = ('GPU-0', 'GPU-1', 'GPU-2', 'GPU-3')
        indices = (0, 1, 2, 3)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def check_exclusive(self, pid):
            pass
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/usr/bin/true')
    monkeypatch.setattr(sandbox, 'device_minors_for_uuids', lambda _: [])
    monkeypatch.setattr(sandbox, 'build_sandbox', lambda *a, **kw: [sys.executable, '-c', 'print("SIMULATED probe")'])
    monkeypatch.setattr(sandbox, 'reserve_idle_gpus', lambda *a, **kw: pytest.fail('borrower reacquired GPUs'))
    task = SimpleNamespace(coral_dir=tmp_path / '.coral', workspace=workspace,
        records=tmp_path / 'records', read=lambda: {'source_role': 'researcher', 'code_path': str(tmp_path)},
        register_job=lambda pid: identities.append(('start', pid)),
        clear_job=lambda pid: identities.append(('finish', pid)))
    result = sandbox.execute_diagnostic(task, {'request_id': 'probe', 'argv': ['unused'], 'gpus': 4, 'timeout': 5},
        lambda: None, borrowed=Borrowed())
    assert result['status'] == 'completed'
    assert identities[0][0] == 'start' and identities[1] == ('finish', identities[0][1])


def test_agent_cannot_override_host_checkpoint_period(tmp_path):
    h, r, _ = host(tmp_path)
    with pytest.raises(ValueError, match='fields'):
        h.enqueue('train', {'iteration': 1, 'sha': 'b'*40,
            'checkpoint_interval_steps': 1}, 'override', {})
    assert r.read()['iterations'][0]['status'] == 'planned'


def test_staged_registration_checks_clean_tree_and_checks_out_fixed_parent(tmp_path, monkeypatch):
    from autoresearch import cli
    data = tmp_path / 'register.json'
    data.write_text('{}')
    calls = []
    monkeypatch.setattr(cli, '_coral_dir', lambda: tmp_path)
    monkeypatch.setattr(cli, '_require_clean_worktree', lambda: calls.append('clean'))
    monkeypatch.setattr(cli, '_checkout_base', lambda sha: calls.append(('checkout', sha)))
    def trusted(*_):
        calls.append('host')
        return {'origin': {'base_commit': 'b'*40}}
    monkeypatch.setattr(cli, 'request_trusted', trusted)
    cli.main(['research', 'register', '--node', 'N001', '--file', str(data)])
    assert calls == ['clean', 'host', ('checkout', 'b'*40)]
    def dirty():
        raise SystemExit('uncommitted work')
    monkeypatch.setattr(cli, '_require_clean_worktree', dirty)
    calls.clear()
    with pytest.raises(SystemExit, match='uncommitted'):
        cli.main(['research', 'register', '--node', 'N001', '--file', str(data)])
    assert calls == []
