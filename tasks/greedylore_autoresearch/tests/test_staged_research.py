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


@pytest.mark.parametrize('result_status,limit', [('completed', 10), ('failed', 20)])
def test_research_diagnostic_budget_survives_reopening(tmp_path, result_status, limit):
    from autoresearch.staged_research import StagedResearch

    h, r, backend = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    backend.run = lambda *_: {'status': result_status}
    payload = {'iteration': 1, 'argv': ['python', 'experiments/probe.py'],
               'timeout': 600, 'scripts_dir': '/tmp/scripts'}
    for number in range(limit):
        h.enqueue('diagnostic', payload, f'probe-{number}', {})
        h.drain()
    reopened = StagedResearch(r.store, r.agent_id, r.node_id, backend=backend)
    # Retrieving an existing request remains allowed at the limit.
    assert reopened.enqueue('diagnostic', payload, 'probe-0', {})['status'] == result_status
    with pytest.raises(ValueError, match='diagnostic budget exhausted'):
        reopened.enqueue('diagnostic', payload, 'over-budget', {})
    assert len(reopened.status()['jobs']) == limit + 1  # Includes training.


def test_status_and_wait_report_current_diagnostic_limits_without_changing_history(tmp_path):
    h, _, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    saved = h.path.read_bytes()
    assert h.status()['diagnostic_limits'] == {'successful': 10, 'total': 20}
    assert h.wait('train')['diagnostic_limits'] == {'successful': 10, 'total': 20}
    assert h.path.read_bytes() == saved


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


@pytest.mark.parametrize('body_status,exit_code', [('completed', 0), ('failed', 1)])
@pytest.mark.parametrize('recovery_mode', ['publication', 'legacy', 'interrupted'])
@pytest.mark.parametrize('blocker_kind', ['file', 'symlink'])
def test_diagnostic_publication_failure_keeps_execution_result_and_retries_without_gpu(
        tmp_path, monkeypatch, body_status, exit_code, recovery_mode, blocker_kind):
    import json

    from autoresearch import staged_diagnostics
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_execution import Execution

    h, r, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'dynamic_artifact_root': str(tmp_path / 'dynamic'),
        'dynamic_artifact_min_free_bytes': 0}}}))
    backend = Execution.__new__(Execution)
    backend.host, backend.coral = h, h.store.coral_dir
    backend.artifacts = RuntimeArtifacts(backend.coral)
    backend.options = {}
    backend._check = lambda: None
    backend.lease = SimpleNamespace(borrow=lambda: object(), register_job=lambda *a: None,
                                    clear_job=lambda *a: None)
    worktree = tmp_path / 'worktree'
    scripts = worktree / 'probe'
    scripts.mkdir(parents=True)
    (scripts / 'probe.py').write_text('print("measured")\n')
    executions = []

    def compute(task, payload, check, *, borrowed):
        executions.append(payload['request_id'])
        (task.workspace / 'results/measurement.json').write_text('{"measured": 42}\n')
        return {'status': body_status, 'exit_code': exit_code, 'runtime_seconds': 1.0,
                'results_path': str(task.workspace / 'results')}

    monkeypatch.setattr(staged_diagnostics, 'execute_diagnostic', compute)
    public = backend.coral / 'public/eval_logs' / ('b'*40) / 'probe'
    work = backend.artifacts.path('N001', 'runs/probe')
    work.mkdir(parents=True)
    backend._execute = lambda host, job: backend._diagnostic(
        job, worktree, work, public, r.read()['iterations'][0])
    h.backend = backend
    blocker = backend.coral / 'public/research_diagnostics'
    if blocker_kind == 'file':
        blocker.write_text('unexpected nonempty file')
    else:
        outside = tmp_path / 'outside'
        outside.mkdir()
        blocker.symlink_to(outside, target_is_directory=True)
    h.enqueue('diagnostic', {'iteration': 1, 'argv': ['python', 'experiments/probe.py'],
                            'timeout': 600, 'scripts_dir': str(scripts)},
              'probe', {'worktree': str(worktree)})
    h.drain()
    job = h.wait('probe')
    assert job['status'] == body_status
    assert job['result']['status'] == body_status
    assert job['result']['exit_code'] == exit_code
    assert job['result']['publication']['status'] == 'failed'
    assert json.loads((public / 'diagnostic-result.json').read_text())['exit_code'] == exit_code

    if recovery_mode != 'publication':
        state = h._read()
        if recovery_mode == 'legacy':
            state['jobs']['probe'].update(status='failed', result={
                'status': 'infra_failed', 'evidence_path': job['result']['evidence_path'],
                'reason': 'NotADirectoryError: public/research_diagnostics/N001/probe'})
        else:
            state['jobs']['probe']['status'] = 'running'
            state['jobs']['probe'].pop('result')
        h._save(state)

    # A new controller can publish from the durable archive after tensor cleanup.
    backend.artifacts.cleanup_node('N001', terminal=True)
    blocker.unlink()
    blocker.mkdir()
    saved = public / 'diagnostic-result.json'
    contents = saved.read_bytes()
    saved.write_text('{}\n')
    h.drain()
    assert executions == ['probe']
    assert h.wait('probe')['status'] != 'running'
    assert h.wait('probe')['result']['publication']['status'] == 'failed'
    saved.write_bytes(contents)
    h.drain()
    assert executions == ['probe']
    job = h.wait('probe')
    assert job['status'] == body_status
    assert job['result']['publication']['status'] == 'completed'
    assert (blocker / 'N001/probe/workspace/results/measurement.json').read_text() == '{"measured": 42}\n'
    h.drain()
    assert executions == ['probe']


def test_diagnostic_publishes_under_existing_protected_root_without_replacing_live_mount(tmp_path):
    import hashlib
    import json

    from autoresearch.staged_execution import Execution

    coral = tmp_path / '.coral'
    public = coral / 'public/eval_logs' / ('b'*40) / 'probe'
    public.mkdir(parents=True)
    original = coral / 'public/research_diagnostics'
    original.touch(mode=0o444)
    inode = original.stat().st_ino
    data = b'{"status":"completed","exit_code":0}\n'
    (public / 'diagnostic-result.json').write_bytes(data)
    (public / 'artifact_inventory.json').write_text(json.dumps({'files': [{
        'path': 'diagnostic-result.json', 'preserved': True,
        'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}]}))
    backend = Execution.__new__(Execution)
    backend.coral, backend.host = coral, SimpleNamespace(node_id='N001')
    result = backend.publish_diagnostic({'request_id': 'probe'}, {
        'status': 'completed', 'exit_code': 0,
        'evidence_path': 'eval_logs/' + 'b'*40 + '/probe'})
    assert result['publication']['status'] == 'completed'
    mirror = coral / 'public/diagnostics/research_probes/N001/probe/diagnostic-result.json'
    assert mirror.read_bytes() == data
    assert original.is_file() and original.stat().st_ino == inode
    from autoresearch.analysis import _evidence_records

    records = _evidence_records([result['publication']['evidence_path']], SimpleNamespace(coral_dir=coral))
    assert any(r.path == 'diagnostics/research_probes/N001/probe/diagnostic-result.json'
               and r.status == 'present' and r.sha256 == hashlib.sha256(data).hexdigest()
               for r in records)


@pytest.mark.parametrize('blocker', ['file', 'symlink', 'broken_symlink'])
def test_invalid_legacy_fallback_does_not_switch_to_normal_publication_root(tmp_path, blocker):
    from autoresearch.staged_execution import Execution

    coral = tmp_path / '.coral'
    archive = coral / 'public/eval_logs' / ('b'*40) / 'probe'
    archive.mkdir(parents=True)
    (archive / 'artifact_inventory.json').write_text('{"files":[]}')
    fallback = coral / 'public/diagnostics/research_probes'
    fallback.parent.mkdir(parents=True)
    foreign = tmp_path / 'foreign'
    if blocker == 'file':
        fallback.write_text('existing data')
    else:
        if blocker == 'symlink':
            foreign.mkdir()
        fallback.symlink_to(foreign)
    backend = Execution.__new__(Execution)
    backend.coral, backend.host = coral, SimpleNamespace(node_id='N001')
    result = backend.publish_diagnostic({'request_id': 'probe'}, {
        'status': 'completed', 'exit_code': 0,
        'evidence_path': 'eval_logs/' + 'b'*40 + '/probe'})
    assert result['status'] == 'completed'
    assert result['publication']['status'] == 'failed'
    assert not (coral / 'public/research_diagnostics').exists()
    assert not foreign.exists() or not list(foreign.iterdir())


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


def test_status_summary_is_readonly(tmp_path):
    h, r, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
    before = h.path.read_bytes()
    record_before = r.private.read_bytes()
    value = h.status(summary=True)
    assert value['jobs']['train']['status'] == 'queued'
    assert value['jobs']['train']['sha'] == 'b' * 40
    assert value['progress']['formal_eligibility'] == 'pending'
    assert h.path.read_bytes() == before
    assert r.private.read_bytes() == record_before


def test_progress_tracks_stages_and_published_attempt(tmp_path):
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    h, r, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
    h.drain()
    doc = r.read()
    doc['selected_iteration'] = 1
    doc['evaluation_stages'] = {'final_quality': ['small', 'large']}
    private = __import__('json').loads(r.private.read_text())
    private['document'] = doc
    r.private.write_text(__import__('json').dumps(private))
    state = h._read()
    state['final_evaluation'] = {'status': 'running', 'components': {
        'timing': [{'status': 'completed', 'result': {'status': 'completed', 'samples': ['x' * 100000]}}],
        'small': [{'status': 'failed', 'result': {'status': 'failed'}}]}}
    h._save(state)
    value = h.status(summary=True)
    assert value['progress']['development'][0]['quality_pass'] is True
    assert value['progress']['final']['components']['large']['status'] == 'pending'
    assert value['progress']['final']['components']['small']['status'] == 'failed'
    assert value['progress']['formal_eligibility'] == 'pending'
    assert len(__import__('json').dumps(value)) < 10000
    state['transfer_terminal'] = True
    h._save(state)
    assert h.status()['progress']['publication'] == 'pending'
    state['publication_complete'] = True
    h._save(state)
    assert h.status()['progress']['formal_eligibility'] == 'unknown'
    write_attempt(h.store.coral_dir, Attempt('b' * 40, h.agent_id, 'Research', 1.0,
        'regressed', None, '2026-01-01', metadata={'staged_research': True, 'performance_eligible': False}))
    assert h.status()['progress']['formal_eligibility'] == 'ineligible'


def test_progress_legacy_transfer_and_skipped_components(tmp_path):
    h, r, _ = host(tmp_path)
    state = h._read()
    state.update(transfer={'status': 'completed', 'evidence_path': 'legacy/final'},
                 transfer_terminal=True, publication_complete=True)
    h._save(state)
    progress = h.status()['progress']
    assert progress['final']['status'] == 'completed'
    assert progress['final']['components']['transfer']['result_status'] == 'completed'
    assert progress['final']['components']['timing']['status'] == 'unknown'
    assert progress['formal_eligibility'] == 'unknown'
    document = r.read()
    document['selected_iteration'] = 1
    document['iterations'][0]['observations'] = [{'timing': {'metadata': {}}}]
    private = __import__('json').loads(r.private.read_text())
    private['document'] = document
    r.private.write_text(__import__('json').dumps(private))
    assert h.status()['progress']['final']['components']['timing']['status'] == 'completed'
    document['selected_iteration'] = None
    document['evaluation_stages'] = {'final_quality': ['small', 'large']}
    private = __import__('json').loads(r.private.read_text())
    private['document'] = document
    r.private.write_text(__import__('json').dumps(private))
    state['final_evaluation'] = {'status': 'skipped', 'components': {}}
    state['transfer']['status'] = 'skipped'
    h._save(state)
    progress = h.status()['progress']
    assert progress['final']['status'] == 'skipped'
    assert all(item['status'] == 'skipped' for item in progress['final']['components'].values())


def test_status_preserves_sealed_analysis_and_evidence(tmp_path):
    import hashlib
    import json

    from autoresearch.analysis import resolve_evidence
    from autoresearch.validation import seal_document
    from test_analysis_recovery import _analysis_payload, _context

    h, _, _ = host(tmp_path)
    from dataclasses import replace
    context = replace(_context(tmp_path / 'analysis-source'),
                      log_root=h.store.coral_dir / 'public/eval_logs')
    source = context.log_root / ('r' * 40) / 'raw.json'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"measurement": 42}\n')
    evidence = resolve_evidence('N001', 'research', context=context)
    assert any(item.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
               for item in evidence.evidence_records)
    sealed = h.store.notes_root / 'nodes/N001/research_analysis_v1.md'
    sealed.write_text(json.dumps(seal_document(_analysis_payload(evidence), evidence)))
    before = {p: (p.read_bytes(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in (sealed, source)}
    h.status()
    h.status(summary=True)
    assert {p: (p.read_bytes(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in before} == before


def test_progress_reports_published_eligible_attempt(tmp_path):
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    h, r, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
    h.drain()
    private = __import__('json').loads(r.private.read_text())
    private['document']['selected_iteration'] = 1
    r.private.write_text(__import__('json').dumps(private))
    state = h._read()
    state.update(transfer_terminal=True, publication_complete=True)
    h._save(state)
    write_attempt(h.store.coral_dir, Attempt('b' * 40, h.agent_id, 'Research', 1.0,
        'regressed', None, '2026-01-01', metadata={'staged_research': True, 'performance_eligible': True}))
    assert h.status()['progress']['formal_eligibility'] == 'eligible'


def test_status_wait_and_replayed_action_project_checkpoint_without_mutating_evidence(tmp_path, monkeypatch):
    import json

    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_research import handle_action
    h, r, backend = host(tmp_path)
    config = r.store.coral_dir / 'config.yaml'
    value = yaml.safe_load(config.read_text())
    value['grader']['args'].update(dynamic_artifact_root=str(tmp_path / 'dynamic'), dynamic_artifact_min_free_bytes=0)
    config.write_text(yaml.safe_dump(value))
    artifacts = RuntimeArtifacts(r.store.coral_dir)
    checkpoint = artifacts.path(h.node_id, 'checkpoints/step-1')
    checkpoint.mkdir(parents=True)
    (checkpoint / 'manifest.json').write_text(json.dumps({'status': 'complete', 'update_step': 1}))
    public = r.store.coral_dir / 'public/eval_logs/sha/run/checkpoints/model/manifest.json'
    artifacts.publish_checkpoint(h.node_id, checkpoint, public)
    backend.run = lambda *_: {'status': 'completed', 'update_step': 1, 'checkpoint_path': str(checkpoint)}
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'original', {})
    h.drain()
    saved = {path: path.read_bytes() for path in (h.path, r.private, r.public, public, artifacts.ledger_path)}
    for _ in range(2):
        job = h.wait('original')
        assert job['result']['checkpoint_path'] == str(public.parent)
        assert job['result']['checkpoint']['retention'] == 'available'
        assert h.status()['jobs']['original']['result'] == job['result']
    # Exercise the public mutation/replay boundary without launching a worker.
    monkeypatch.setattr('autoresearch.staged_research._handle_action', lambda *_: h._read()['jobs']['original'])
    replay = handle_action(r.store, h.agent_id, {}, {})
    assert replay['request_id'] == 'original'
    assert replay['result'] == job['result']
    assert all(path.read_bytes() == content for path, content in saved.items())


def test_historical_status_does_not_require_current_diagnostic_resources(tmp_path):
    h, r, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'saved', {})
    h.drain()
    config = r.store.coral_dir / 'config.yaml'
    options = yaml.safe_load(config.read_text())
    options['grader']['args']['diagnostic_resources'] = [
        {'id': 'missing', 'path': str(tmp_path / 'absent-data'), 'target_roots': []}]
    config.write_text(yaml.safe_dump(options))
    saved = h.path.read_bytes()
    assert h.status()['jobs']['saved']['status'] == 'completed'
    assert h.wait('saved')['request_id'] == 'saved'
    assert h.path.read_bytes() == saved
