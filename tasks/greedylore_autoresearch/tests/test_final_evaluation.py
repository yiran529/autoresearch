"""Frozen final tasks never reuse development timing or reopen search."""
# ruff: noqa: S101

import pytest
import yaml
from test_final_contract import final_contract
from test_research_record import initialize, record
from test_staged_research import timing_result


def final_host(tmp_path):
    from autoresearch.staged_research import StagedResearch
    r = record(tmp_path)
    contract = final_contract(tmp_path)
    (r.store.coral_dir / 'private/eval_contract.yaml').write_text(contract.path.read_text())
    initialize(r)
    class Backend:
        calls = []
        fail_workload = None
        def run(self, host, job):
            self.calls.append((job['kind'], job.get('workload')))
            if job['kind'].startswith('final_'):
                assert r.read()['frozen']
            if self.fail_workload and job.get('workload') == self.fail_workload:
                from greedylore_autoresearch_grader.gpu import InfraFailure
                raise InfraFailure('SIMULATED final component infrastructure failure')
            if job['kind'] in {'timing', 'final_timing'}:
                result = timing_result(contract.contract_hash)
                result['bundle']['metadata']['timing_protocol'] = job['timing_context']['plan']
                return result
            return {'status': 'completed', 'quality_pass': True, 'eligible': False,
                    'final_eval_loss': 4.0 if job['kind']=='train' else 3.0,
                    'final_test_accuracy': 75.0, 'final_test_loss': 1.3}
        def release(self): self.calls.append(('release', None))
        def finish(self, host, selected): self.calls.append(('publish', selected))
    b = Backend()
    h = StagedResearch(r.store, r.agent_id, r.node_id, backend=b)
    h.validate_candidate = lambda sha: {'commit_hash': sha, 'contract_hash': contract.contract_hash,
                                     'timing_declaration': {'period_steps': 200}}
    return h,r,b


def train(h):
    h.enqueue('train', {'iteration':1, 'sha':'b'*40}, 'train', {})
    h.drain()


def finish(h, request='finish'):
    h.enqueue('finish', {'selected_iteration':1, 'reason':'development works'}, request, {})
    h.drain()


def test_final_freezes_before_fresh_timing_and_both_quality_tasks(tmp_path):
    h,r,b = final_host(tmp_path)
    h.enqueue('timing', {'iteration':1, 'sha':'b'*40}, 'dev-timing', {})
    h.drain()
    train(h)
    finish(h)
    assert h.status()['transfer_terminal']
    assert [kind for kind,_ in b.calls] == ['timing','train','final_timing','final_quality','final_quality','release','publish']
    assert set(r.read()['final_evaluation']) == {'timing','llama60m_adamw','resnet18_cifar100_muon'}
    with pytest.raises(ValueError, match='frozen|terminal'):
        h.enqueue('timing', {'iteration':1, 'sha':'b'*40}, 'late', {})


def test_development_timing_exhaustion_cannot_starve_mandatory_final(tmp_path):
    h,r,b = final_host(tmp_path)
    (h.store.coral_dir/'config.yaml').write_text(yaml.safe_dump({'grader':{'args':{
        'staged_research':True,'research_timing_success_limit':1}}}))
    h.enqueue('timing', {'iteration':1, 'sha':'b'*40}, 'dev', {})
    h.drain()
    train(h)
    finish(h)
    assert h.status()['transfer_terminal'] and r.read()['frozen']
    assert sum(kind=='final_timing' for kind,_ in b.calls)==1


def test_component_infra_retry_preserves_other_results_and_frozen_choice(tmp_path):
    h,r,b = final_host(tmp_path)
    train(h)
    b.fail_workload='resnet18_cifar100_muon'
    finish(h)
    assert r.read()['frozen'] and not h.status()['transfer_terminal']
    assert sum(kind=='final_timing' for kind,_ in b.calls)==1
    b.fail_workload=None
    finish(h,'retry-final')
    assert h.status()['transfer_terminal']
    assert sum(kind=='final_timing' for kind,_ in b.calls)==1
    assert b.calls.count(('final_quality','llama60m_adamw'))==1
    assert b.calls.count(('final_quality','resnet18_cifar100_muon'))==2
    assert len(r.read()['final_evaluation']['resnet18_cifar100_muon'])==2


def test_infra_retry_exhaustion_terminates_final_without_reopening(tmp_path):
    h,r,b = final_host(tmp_path)
    train(h)
    b.fail_workload='resnet18_cifar100_muon'
    for index in range(3):
        finish(h,f'finish{index}')
    assert h.status()['transfer_terminal'] and r.read()['frozen']
    assert b.calls.count(('final_quality','resnet18_cifar100_muon'))==3
    assert sum(kind=='final_timing' for kind,_ in b.calls)==1
    assert h.status()['transfer']['status'] == 'infra_failed'


def test_invalid_final_timing_is_terminal_evidence_without_reopening(tmp_path):
    h,r,b = final_host(tmp_path)
    train(h)
    original = b.run
    def invalid(host, job):
        result = original(host, job)
        if job['kind'] == 'final_timing':
            result['performance_eligible'] = not result['performance_eligible']
        return result
    b.run = invalid
    finish(h)
    assert r.read()['frozen'] and h.status()['transfer_terminal']
    result = r.read()['final_evaluation']['timing'][-1]['result']
    assert result['status'] == 'failed' and result['invalid_result']['status'] == 'completed'
    assert sum(kind == 'final_quality' for kind, _ in b.calls) == 2


def test_interrupted_final_publication_reuses_completed_components(tmp_path):
    h,r,b = final_host(tmp_path)
    train(h)
    published = []
    def publish(job, result):
        published.append(job['component'])
        if len(published) == 1:
            raise OSError('SIMULATED public projection interrupted')
    b.publish_final = publish
    finish(h)
    assert r.read()['frozen'] and not h.status()['transfer_terminal']
    finish(h, 'recover-publication')
    assert h.status()['transfer_terminal']
    assert sum(kind == 'final_timing' for kind, _ in b.calls) == 1
    assert len(r.read()['final_evaluation']['timing']) == 1
