"""Independent timing uses real admission, records, budgets and recovery."""
# ruff: noqa: S101
import pytest
import yaml
from test_research_record import design
from test_staged_research import host, timing_result


def timing_host(tmp_path):
    h, r, b = host(tmp_path)
    original = b.run
    def run(host, job):
        if job['kind'] == 'timing':
            b.calls.append(('timing', job['sha']))
            result = timing_result(job['canonical']['contract_hash'])
            result['bundle']['metadata']['timing_protocol'] = job['timing_context']['plan']
            return {**result,
                    'evidence_path': f"eval_logs/{job['sha']}/{job['request_id']}"}
        result = original(host, job)
        if job['kind'] == 'train':
            result.update(eligible=False, quality_pass=True)
        return result
    b.run = run
    h.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': 'fixed',
                                      'timing_declaration': {'period_steps': 200}}
    return h, r, b


def measure(h, request='timing', iteration=1, sha='b'*40, **extra):
    result = h.enqueue('timing', {'iteration': iteration, 'sha': sha, **extra}, request, {})
    h.drain()
    return result


def test_pretrain_timing_pins_design_without_training_or_checkpoint(tmp_path):
    h, r, b = timing_host(tmp_path)
    measure(h)
    row = r.read()['iterations'][0]
    assert row['status'] == 'planned' and row['runs'] == [] and row['observations'] == []
    assert row['code_sha'] == 'b'*40
    assert row['timings'][0]['result']['status'] == 'completed'
    assert not any(c[0] == 'train' for c in b.calls)
    with pytest.raises(ValueError, match='SHA|sha'):
        h.enqueue('train', {'iteration': 1, 'sha': 'c'*40}, 'changed', {})


def test_duplicate_and_new_request_reuse_but_reason_allows_repeat(tmp_path):
    h, r, b = timing_host(tmp_path)
    measure(h)
    measure(h)
    measure(h, 'reuse')
    assert sum(c[0] == 'timing' for c in b.calls) == 1
    measure(h, 'repeat', repeat_reason='Resolve unusually wide variability')
    assert sum(c[0] == 'timing' for c in b.calls) == 2
    assert len(r.read()['iterations'][0]['timings']) == 2


def test_timing_does_not_change_paused_trajectory_or_repeat_on_resume(tmp_path):
    h, r, b = timing_host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40, 'pause_after_steps': 1000}, 'pause', {})
    h.drain()
    before = r.read()['iterations'][0]['observations']
    measure(h)
    assert r.read()['iterations'][0]['status'] == 'paused'
    assert r.read()['iterations'][0]['observations'] == before
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'resume', {})
    h.drain()
    assert r.read()['iterations'][0]['status'] == 'completed'
    assert sum(c[0] == 'timing' for c in b.calls) == 1
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'quality and timing'}, 'finish', {})
    h.drain()
    assert r.read()['frozen'] and h.status()['transfer_terminal']
    assert sum(c[0] == 'timing' for c in b.calls) == 1


def test_finish_supplies_missing_timing_before_freeze_and_transfer(tmp_path):
    h, r, b = timing_host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    assert not r.read()['iterations'][0]['timings']
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'finished'}, 'finish', {})
    h.drain()
    assert h.status()['transfer_terminal'] and r.read()['frozen']
    assert [c[0] for c in b.calls][:3] == ['train', 'timing', 'transfer']


def test_timing_rejected_while_training_queued_and_no_budget_consumed(tmp_path):
    h, r, _ = timing_host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    with pytest.raises(ValueError, match='active'):
        h.enqueue('timing', {'iteration': 1, 'sha': 'b'*40}, 'timing', {})
    with pytest.raises(ValueError, match='active'):
        r.decide(1, 'stop', 'cannot race queued training', 'stop')
    assert len(h.status()['jobs']) == 1


def test_timing_only_revisions_consume_existing_design_budget(tmp_path):
    h, r, _ = timing_host(tmp_path)
    for i in range(1, 4):
        if i > 1:
            r.append_design(design(str(i)), f'design{i}')
        measure(h, f'timing{i}', iteration=i, sha=str(i)*40)
        r.decide(i, 'stop', 'not worth full training', f'stop{i}')
    with pytest.raises(ValueError, match='3|budget'):
        r.append_design(design('fourth'), 'design4')
    h.enqueue('finish', {'selected_iteration': None, 'reason': 'timing rejected all'}, 'finish', {})
    h.drain()
    assert h.status()['transfer_terminal']


def test_completed_timing_budget_is_separate_from_diagnostics_and_survives_reopen(tmp_path):
    h, r, b = timing_host(tmp_path)
    config = {'grader': {'args': {'staged_research': True, 'research_timing_success_limit': 2}}}
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    measure(h)
    measure(h, 'second', repeat_reason='repeat spread')
    reopened = type(h)(h.store, h.agent_id, h.node_id, backend=b)
    reopened.validate_candidate = h.validate_candidate
    h = reopened
    with pytest.raises(ValueError, match='timing.*budget'):
        measure(h, 'third', repeat_reason='more precision')
    measure(h, 'cached-after-cap')
    assert sum(c[0] == 'timing' for c in b.calls) == 2
    assert r.read()['iterations'][0]['status'] == 'planned'
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40, 'pause_after_steps': 1000}, 'pause', {})
    h.drain()
    h.enqueue('diagnostic', {'iteration': 1, 'argv': ['python', 'experiments/probe.py'],
        'timeout': 5, 'scripts_dir': '/tmp/probe'}, 'probe', {})
    h.drain()
    assert h.status()['jobs']['probe']['status'] == 'completed'


def test_interrupted_timing_keeps_training_state_and_archives_failed_record(tmp_path):
    h, r, b = timing_host(tmp_path)
    h.enqueue('timing', {'iteration': 1, 'sha': 'b'*40}, 'interrupted', {})
    state = h._read()
    state['jobs']['interrupted']['status'] = 'running'
    h._save(state)
    h.drain()
    row = r.read()['iterations'][0]
    assert row['status'] == 'planned' and not row['runs']
    assert row['timings'][0]['result']['status'] == 'infra_failed'
    measure(h, 'recovered')
    assert len(r.read()['iterations'][0]['timings']) == 2


def test_finish_cannot_promote_ineligible_timing_and_keeps_record_open(tmp_path):
    h, r, b = timing_host(tmp_path)
    original = b.run
    def invalid(host, job):
        result = original(host, job)
        if job['kind'] == 'timing':
            result['performance_eligible'] = False
        return result
    b.run = invalid
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'try'}, 'finish', {})
    h.drain()
    assert not r.read()['frozen']
    assert not any(c[0] == 'transfer' for c in b.calls)
    assert r.read()['iterations'][0]['timings'][0]['result']['performance_eligible'] is False


def test_other_protocol_result_is_not_reused(tmp_path):
    h, r, b = timing_host(tmp_path)
    measure(h)
    h.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': 'changed',
                                      'timing_declaration': {'period_steps': 200}}
    measure(h, 'new-protocol')
    assert sum(c[0] == 'timing' for c in b.calls) == 2


def test_completed_timing_republishes_after_record_write_failure_without_rerun(tmp_path, monkeypatch):
    h, r, b = timing_host(tmp_path)
    add = h.record.add_timing
    monkeypatch.setattr(h.record, 'add_timing', lambda *a, **k: (_ for _ in ()).throw(OSError('publication interrupted')))
    measure(h)
    assert h.status()['jobs']['timing']['result']['status'] == 'completed'
    monkeypatch.setattr(h.record, 'add_timing', add)
    h.drain()
    assert len(r.read()['iterations'][0]['timings']) == 1
    assert sum(c[0] == 'timing' for c in b.calls) == 1


def test_total_budget_counts_failed_timings_and_reserves_no_diagnostic_slots(tmp_path):
    h, r, b = timing_host(tmp_path)
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'research_timing_total_limit': 2}}}))
    from greedylore_autoresearch_grader.gpu import InfraFailure
    b.run = lambda *_: (_ for _ in ()).throw(InfraFailure('GPU disappeared'))
    measure(h)
    measure(h, 'retry')
    with pytest.raises(ValueError, match='timing.*budget'):
        measure(h, 'over-budget')
    assert len(r.read()['iterations'][0]['timings']) == 2
    assert all(j['kind'] == 'timing' for j in h.status()['jobs'].values())


def test_final_missing_timing_cannot_bypass_timing_budget(tmp_path):
    h, r, b = timing_host(tmp_path)
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'research_timing_success_limit': 1}}}))
    measure(h)
    r.decide(1, 'stop', 'different within-hypothesis design', 'stop')
    r.append_design(design('second'), 'design2')
    h.enqueue('train', {'iteration': 2, 'sha': 'c'*40}, 'train2', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 2, 'reason': 'try'}, 'finish', {})
    h.drain()
    assert not r.read()['frozen'] and not any(c[0] == 'transfer' for c in b.calls)
    assert 'budget' in h.status()['jobs']['finish']['error']


def test_newest_measurement_is_used_even_if_earlier_timing_was_faster(tmp_path):
    h, r, b = timing_host(tmp_path)
    measure(h)
    original = b.run
    def invalid(host, job):
        result = original(host, job)
        if job['kind'] == 'timing':
            result['performance_eligible'] = False
        return result
    b.run = invalid
    measure(h, 'repeat', repeat_reason='check initial measurement')
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'try'}, 'finish', {})
    h.drain()
    assert not r.read()['frozen'] and not any(c[0] == 'transfer' for c in b.calls)


def test_interrupted_finish_restores_completed_timing_without_second_execution(tmp_path):
    h, r, b = timing_host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
    h.drain()
    h.enqueue('finish', {'selected_iteration': 1, 'reason': 'select'}, 'finish', {})
    original = h.record.freeze
    h.record.freeze = lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        h.drain()
    assert not r.read()['frozen'] and len(r.read()['iterations'][0]['timings']) == 1
    h.record.freeze = original
    h.drain()
    assert r.read()['frozen'] and h.status()['transfer_terminal']
    assert sum(c[0] == 'timing' for c in b.calls) == 1


def test_candidate_timing_failure_requires_new_design_instead_of_same_sha_retry(tmp_path):
    h, r, b = timing_host(tmp_path)
    b.run = lambda *_: (_ for _ in ()).throw(ValueError('invalid candidate telemetry'))
    measure(h)
    with pytest.raises(ValueError, match='candidate.*failed|new design'):
        measure(h, 'retry')


def test_identified_ipc_infra_retains_phase_and_allows_bounded_same_sha_recovery(tmp_path):
    from greedylore_autoresearch_grader.runtime import InfraFailure
    h, r, b = timing_host(tmp_path)
    original = b.run
    error = InfraFailure('OSError: AF_UNIX path too long')
    error.phase = 'runtime_ipc'
    b.run = lambda *_: (_ for _ in ()).throw(error)
    measure(h, 'ipc-failure')
    result = h.status()['jobs']['ipc-failure']['result']
    assert result['status'] == 'infra_failed' and result['phase'] == 'runtime_ipc'
    assert 'AF_UNIX' in result['reason']
    b.run = original
    measure(h, 'ipc-recovered')
    assert h.status()['jobs']['ipc-recovered']['status'] == 'completed'
    assert len(r.read()['iterations']) == 1 and not r.read()['iterations'][0]['runs']


@pytest.mark.parametrize('corruption', [None, 'sha', 'context', 'digest', 'bundle', 'contract', 'protocol', 'nonfinite', 'eligibility'])
@pytest.mark.parametrize('phase', ['queued', 'running'])
def test_executor_durable_timing_cache_recovers_without_gpu_reexecution(tmp_path, contract, corruption, phase):
    import json

    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_execution import Execution

    h, r, _ = timing_host(tmp_path)
    (h.store.coral_dir / 'private/eval_contract.yaml').write_text(contract.path.read_text())
    h.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': contract.contract_hash,
                                      'timing_declaration': {'period_steps': 200}}
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'dynamic_artifact_root': str(tmp_path / 'dynamic'),
        'dynamic_artifact_min_free_bytes': 0}}}))
    h.enqueue('timing', {'iteration': 1, 'sha': 'b'*40}, 'cached', {})
    backend = Execution.__new__(Execution)
    backend.host, backend.coral = h, h.store.coral_dir
    backend.artifacts = RuntimeArtifacts(h.store.coral_dir)
    work = backend.artifacts.path('N001', 'runs/cached')
    work.mkdir(parents=True)
    result = timing_result(contract.contract_hash)
    result['bundle']['metadata']['timing_protocol'] = {
        'version': 'bounded-cycles-v2', 'minimum_iterations': 400, 'maximum_iterations': 800,
        'period_steps': 200, 'requested_steps': None, 'host_override': None,
        'measured_iterations': 400, 'window_covers_period': True}
    job = h._read()['jobs']['cached']
    import hashlib
    cache = h.store.coral_dir / 'private/research_timing_results/N001/cached.json'
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({'version': 1, 'sha': 'b'*40, 'context': job['timing_context'], 'result': result,
        'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}))
    if corruption:
        saved = json.loads(cache.read_text())
        if corruption == 'sha':
            saved['sha'] = 'c'*40
        elif corruption == 'context':
            saved['context']['monitor_interval_seconds'] = 10
        elif corruption == 'bundle':
            saved['result'].pop('bundle')
        elif corruption == 'contract':
            saved['result']['bundle']['metadata']['contract_hash'] = 'wrong-contract'
        elif corruption == 'protocol':
            saved['result']['bundle']['metadata']['timing_protocol']['measured_iterations'] = 800
        elif corruption == 'nonfinite':
            saved['result']['bundle']['scores']['bf16_timing_mean']['value'] = float('nan')
        elif corruption == 'eligibility':
            saved['result']['performance_eligible'] = False
        saved['result_sha256'] = 'wrong-digest' if corruption == 'digest' else hashlib.sha256(
            json.dumps(saved['result'], sort_keys=True).encode()).hexdigest()
        cache.write_text(json.dumps(saved))
    state = h._read()
    state['jobs']['cached']['status'] = phase
    h._save(state)
    h.backend = backend  # No GPU lease exists: replay must use the durable result.
    h.drain()
    assert h.status()['jobs']['cached']['status'] == ('failed' if corruption else 'completed')
    assert len(r.read()['iterations'][0]['timings']) == 1
    if corruption:
        assert 'invalid durable timing cache' in r.read()['iterations'][0]['timings'][0]['result']['reason']
    else:
        assert (h.store.coral_dir / 'public/eval_logs' / ('b'*40) / 'cached/timing-result.json').is_file()


def test_work_directory_cannot_supply_authoritative_timing_result(tmp_path):
    import json

    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_execution import Execution
    h, r, _ = timing_host(tmp_path)
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'dynamic_artifact_root': str(tmp_path / 'dynamic'),
        'dynamic_artifact_min_free_bytes': 0}}}))
    h.enqueue('timing', {'iteration': 1, 'sha': 'b'*40}, 'untrusted', {})
    backend = Execution.__new__(Execution)
    backend.host, backend.coral = h, h.store.coral_dir
    backend.artifacts = RuntimeArtifacts(h.store.coral_dir)
    work = backend.artifacts.path('N001', 'runs/untrusted')
    work.mkdir(parents=True)
    (work / 'timing-result.json').write_text(json.dumps({'status': 'completed', 'performance_eligible': True}))
    (work / 'failure.log').write_text('Interrupted timing process; partial evidence')
    state = h._read()
    state['jobs']['untrusted']['status'] = 'running'
    h._save(state)
    h.backend = backend
    h.drain()
    assert r.read()['iterations'][0]['timings'][0]['result']['status'] == 'infra_failed'
    assert (h.store.coral_dir / 'public/eval_logs' / ('b'*40) / 'untrusted/failure.log').is_file()


def test_archive_failure_recovers_completed_measurement_instead_of_marking_failed(tmp_path):
    h, r, b = timing_host(tmp_path)
    b.run = lambda *_: (_ for _ in ()).throw(OSError('archive failed after measurement'))
    b.recover_timing = lambda job: timing_result('fixed')
    measure(h)
    assert h.status()['jobs']['timing']['status'] == 'completed'
    assert r.read()['iterations'][0]['timings'][0]['result']['status'] == 'completed'


def test_explicit_default_monitor_setting_reuses_effectively_same_protocol(tmp_path):
    h, r, b = timing_host(tmp_path)
    measure(h)
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'timing_monitor_interval_seconds': 1}}}))
    measure(h, 'same-effective-context')
    assert sum(c[0] == 'timing' for c in b.calls) == 1


@pytest.mark.parametrize('boundary', ['timing', 'freeze'])
def test_private_write_then_public_failure_replays_projection_before_transfer(tmp_path, monkeypatch, boundary):
    import json

    from autoresearch import research_record
    h, r, b = timing_host(tmp_path)
    if boundary == 'freeze':
        h.enqueue('train', {'iteration': 1, 'sha': 'b'*40}, 'train', {})
        h.drain()
    write = research_record._atomic_json
    failures = []
    def flaky(path, value, **kwargs):
        target = value.get('frozen') if boundary == 'freeze' else bool(value.get('iterations', [{}])[0].get('timings'))
        if path == r.public and target and not failures:
            failures.append('failed-public-projection')
            raise OSError('public disk interrupted after private commit')
        return write(path, value, **kwargs)
    monkeypatch.setattr(research_record, '_atomic_json', flaky)
    if boundary == 'timing':
        measure(h)
        assert not json.loads(r.public.read_text())['iterations'][0]['timings']
    else:
        h.enqueue('finish', {'selected_iteration': 1, 'reason': 'select'}, 'finish', {})
        h.drain()
        assert r.read()['frozen'] and not json.loads(r.public.read_text())['frozen']
        assert not any(c[0] == 'transfer' for c in b.calls)
    h.drain()
    assert json.loads(r.public.read_text()) == r.read()
    assert sum(c[0] == 'timing' for c in b.calls) == 1
    if boundary == 'freeze':
        assert h.status()['transfer_terminal']


@pytest.mark.parametrize('limit', [1, 2])
def test_legacy_timing_counts_budget_without_relabeling_unknown_context(tmp_path, limit):
    h, r, b = timing_host(tmp_path)
    r.begin(1, 'b'*40, 'legacy.begin')
    r.observe(1, {'status': 'completed', 'quality_pass': True, 'eligible': True,
        'timing': timing_result()['bundle']}, 'legacy.complete')
    (h.store.coral_dir / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'research_timing_success_limit': limit}}}))
    if limit == 1:
        with pytest.raises(ValueError, match='timing.*budget'):
            measure(h)
    else:
        measure(h)
        assert sum(c[0] == 'timing' for c in b.calls) == 1
