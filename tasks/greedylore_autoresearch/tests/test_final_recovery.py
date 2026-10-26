"""Recover completed final work across the external-to-private cache window."""
# ruff: noqa: S101
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from test_final_contract import final_contract


def final_run(tmp_path, workload='llama60m_adamw', *, validated_cache=True):
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from autoresearch.staged_execution import Execution
    from greedylore_autoresearch_grader.quality import build_quality_command

    contract = final_contract(tmp_path)
    coral = tmp_path / '.coral'
    (coral / 'private').mkdir(parents=True)
    (coral / 'private/eval_contract.yaml').write_text(contract.path.read_text())
    (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'dynamic_artifact_root': str(tmp_path / 'runtime'), 'dynamic_artifact_min_free_bytes': 0}}}))
    execution = Execution.__new__(Execution)
    execution.coral = coral
    execution.host = SimpleNamespace(node_id='N001')
    execution.interval = 1000
    execution.artifacts = RuntimeArtifacts(coral)
    job = dict(kind='final_quality', sha='b' * 40, request_id='final-quality-one',
               workload=workload, canonical={'contract_hash': contract.contract_hash})
    work = execution.artifacts.path('N001', 'runs/final-quality-one')
    output = work / 'quality' / workload
    output.mkdir(parents=True)
    code = execution.artifacts.path('N001', f"code/{job['sha']}")
    code.mkdir(parents=True)
    profile = contract.quality[workload]
    spec = build_quality_command(profile, code, output, [0, 1, 2, 3],
        checkpoint_root=work / 'checkpoints' / workload, checkpoint_interval_steps=1000,
        training_metrics=work / 'training_metrics.jsonl')
    process = dict(command=list(spec.command), returncode=0, timed_out=False,
                   infra_failure=None, stdout_log='train.stdout.log', stderr_log='train.stderr.log')
    (work / 'process.json').write_text(json.dumps(process))
    result = dict(status='completed', contract_hash=contract.contract_hash,
                  world_size=4, update_step=profile['stop_after_steps'])
    if profile['model'] == 'resnet18':
        from test_validation_vision import validation_payload
        result.update(validation_payload(profile))
    else:
        result['final_eval_loss'] = 3.
    result_path = work / 'quality-result.json' if validated_cache else output / 'all_results.json'
    result_path.write_text(json.dumps(result))
    return execution, job, work, result_path, process


@pytest.mark.parametrize('workload', ['llama60m_adamw', 'resnet18_cifar100_muon'])
@pytest.mark.parametrize('validated_cache', [True, False])
def test_completed_external_result_is_promoted_without_training(tmp_path, workload, validated_cache):
    execution, job, work, result_path, _ = final_run(tmp_path, workload, validated_cache=validated_cache)
    original = result_path.read_bytes()
    result = execution.recover_final(job)
    assert result is not None
    assert result['status'] == 'completed'
    assert result['run_id'] == 'final-quality-one'
    assert result['evidence_path'] == f"eval_logs/{job['sha']}/final-quality-one"
    assert result['eligible'] is False
    saved = json.loads(execution._final_cache(job).read_text())
    assert saved['result'] == result
    assert result_path.read_bytes() == original
    public = execution.coral / 'public' / result['evidence_path']
    assert json.loads((public / 'process.json').read_text())['returncode'] == 0
    assert (public / result_path.relative_to(work)).read_bytes() == original
    # The host-owned cache survives loss of the external runtime result.
    result_path.unlink()
    (work / 'process.json').unlink()
    assert execution.recover_final(job) == result


@pytest.mark.parametrize('change', ['candidate', 'request', 'contract', 'failed', 'timed_out', 'infra'])
def test_foreign_or_incomplete_process_cannot_certify_external_result(tmp_path, change):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    execution, job, work, _, process = final_run(tmp_path)
    if change in {'candidate', 'request', 'contract'}:
        flag = {'candidate': '--project', 'request': '--output_dir', 'contract': '--contract_hash'}[change]
        process['command'][process['command'].index(flag) + 1] = 'foreign'
    elif change == 'failed':
        process['returncode'] = 1
    elif change == 'timed_out':
        process['timed_out'] = True
    else:
        process['infra_failure'] = 'process ownership lost'
    (work / 'process.json').write_text(json.dumps(process))
    with pytest.raises(InfraFailure):
        execution.recover_final(job)
    assert not execution._final_cache(job).exists()


@pytest.mark.parametrize('workload,field,value', [
    ('llama60m_adamw', 'status', 'paused'),
    ('llama60m_adamw', 'update_step', 1),
    ('llama60m_adamw', 'world_size', 1),
    ('llama60m_adamw', 'contract_hash', 'foreign'),
    ('llama60m_adamw', 'final_eval_loss', float('nan')),
    ('llama60m_adamw', 'final_eval_loss', True),
    ('resnet18_cifar100_muon', 'status', 'smoke_completed'),
    ('resnet18_cifar100_muon', 'epoch', 39),
    ('resnet18_cifar100_muon', 'validation_samples', 4999),
    ('resnet18_cifar100_muon', 'final_validation_accuracy', 101),
])
def test_external_payload_is_revalidated_before_promotion(tmp_path, workload, field, value):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    execution, job, _, result_path, _ = final_run(tmp_path, workload)
    result = json.loads(result_path.read_text())
    result[field] = value
    result_path.write_text(json.dumps(result))
    with pytest.raises(InfraFailure):
        execution.recover_final(job)
    assert not execution._final_cache(job).exists()


def test_external_result_cannot_supply_host_observation_identity(tmp_path):
    execution, job, _, result_path, _ = final_run(tmp_path)
    payload = json.loads(result_path.read_text())
    payload.update(run_id='foreign', evidence_path='eval_logs/foreign', eligible=True)
    result_path.write_text(json.dumps(payload))
    result = execution.recover_final(job)
    assert result['run_id'] == job['request_id']
    assert result['evidence_path'] == f"eval_logs/{job['sha']}/{job['request_id']}"
    assert result['eligible'] is False


def test_missing_process_record_is_not_proof_of_completion(tmp_path):
    execution, job, work, _, _ = final_run(tmp_path)
    (work / 'process.json').unlink()
    assert execution.recover_final(job) is None
    assert not execution._final_cache(job).exists()


def test_validated_result_remains_durable_when_public_archive_fails(tmp_path, monkeypatch):
    execution, job, _, _, _ = final_run(tmp_path)
    def fail_archive(*args):
        raise OSError('interrupted publication')
    with monkeypatch.context() as scoped:
        scoped.setattr(execution, '_archive', fail_archive)
        with pytest.raises(OSError, match='interrupted publication'):
            execution.recover_final(job)
    assert execution._final_cache(job).exists()
    assert execution.recover_final(job)['status'] == 'completed'
