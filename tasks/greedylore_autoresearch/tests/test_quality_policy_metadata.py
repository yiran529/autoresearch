"""Published quality recipes must distinguish loss from final classification accuracy."""
# ruff: noqa: S101
import json

import pytest
from greedylore_autoresearch_grader.evaluation_context import publish_context
from greedylore_autoresearch_grader.grader import score_results
from test_final_contract import final_contract

VISION = 'resnet18_cifar100_muon'


def _results():
    quality = {
        'llama60m_adamw': {'final_eval_loss': 3.0},
        'llama130m_muon': {'final_eval_loss': 4.0},
        VISION: {'final_test_accuracy': 74.8},
    }
    reference = {'quality': {'workloads': {
        'llama60m_adamw': {'final_eval_loss': 3.0},
        'llama130m_muon': {'final_eval_loss': 4.0},
        VISION: {'final_test_accuracy': 75.91},
    }}, 'efficiency': {'cells': {
        name: {'mean_iteration_seconds': .2}
        for name in ('130m_adamw_bf16', '130m_adamw_fp32')
    }}}
    timing = {name: {'mean': .1, 'relative_spread': .01}
              for name in reference['efficiency']['cells']}
    return quality, reference, timing


def _assert_vision_policy(policy):
    assert policy == {
        'metric': 'final_test_accuracy',
        'comparison': 'dense final_test_accuracy - candidate final_test_accuracy <= max_accuracy_drop_pp',
        'max_accuracy_drop_pp': .75,
        'selection': 'final_epoch',
        'test_examples': 10000,
    }


def test_final_context_publishes_workload_specific_quality_policies(tmp_path):
    contract = final_contract(tmp_path)
    _, reference, _ = _results()
    # Operator-configurable thresholds must survive publication.
    from dataclasses import replace
    contract = replace(contract, raw={**contract.raw, 'vision': {
        **contract.raw['vision'], 'max_accuracy_drop_pp': .75}})
    value = json.loads(publish_context(contract, reference, tmp_path / 'logs', 'research').read_text())
    assert value['quality']['metric'] == 'workload_specific'
    policies = value['quality']['workload_policies']
    assert set(policies) == set(contract.quality)
    _assert_vision_policy(policies[VISION])
    assert policies['llama60m_adamw'] == {
        'metric': 'final_eval_loss',
        'comparison': 'candidate final_eval_loss / dense final_eval_loss <= max_relative_eval_loss',
        'max_relative_eval_loss': 1.05,
    }
    assert value['quality']['workloads'][VISION]['epochs'] == 40


@pytest.mark.parametrize('mode', ['research', 'engineering'])
def test_score_metadata_identifies_accuracy_policy_with_configured_drop(mode):
    quality, reference, timing = _results()
    profiles = {'llama60m_adamw': {}, 'llama130m_muon': {},
                VISION: {'max_accuracy_drop_pp': .75}}
    result = score_results(mode, quality, timing, reference, contract_hash='fixed',
                           comparable=True, evidence_paths=[], quality_profiles=profiles)
    policies = result.metadata['evaluation_policy']['workload_policies']
    _assert_vision_policy(policies[VISION])
    assert policies['llama130m_muon']['metric'] == 'final_eval_loss'
    if mode == 'research':
        assert result.scores['resnet18_quality_pass'].value == 0


def test_legacy_quality_context_and_score_metadata_are_unchanged(contract, tmp_path):
    quality, reference, timing = _results()
    quality.pop(VISION)
    reference['quality']['workloads'].pop(VISION)
    value = json.loads(publish_context(contract, reference, tmp_path / 'logs', 'research').read_text())
    assert set(value['quality']) == {'metric', 'comparison', 'max_relative_eval_loss', 'workloads'}
    assert value['quality']['metric'] == 'final_eval_loss'
    assert value['quality']['comparison'] == (
        'candidate final_eval_loss / dense final_eval_loss <= max_relative_eval_loss')
    result = score_results('research', quality, timing, reference, contract_hash=contract.contract_hash,
                           comparable=True, evidence_paths=[], quality_profiles=contract.quality)
    assert result.metadata['evaluation_policy'] == {
        'version': 'quality-gated-efficiency-v2', 'max_relative_eval_loss': 1.05,
    }
