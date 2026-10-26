"""The search contract evaluates held-out training images, never official test."""
# ruff: noqa: S101
import pytest
from greedylore_autoresearch_grader.vision import (
    build_vision_command,
    expected_vision_recipe,
    validate_vision_result,
)
from test_final_contract import final_contract


def test_current_contract_uses_fixed_stratified_validation(tmp_path):
    contract = final_contract(tmp_path)
    profile = contract.quality['resnet18_cifar100_muon']
    assert contract.raw['schema_version'] == 3
    assert profile['evaluation_split'] == 'validation'
    assert profile['validation_per_class'] == 50 and profile['split_seed'] == 1243
    assert profile['stop_after_steps'] == 14080
    command = list(build_vision_command(profile, tmp_path, tmp_path/'out', [0,1,2,3]).command)
    assert command[command.index('--evaluation-split')+1] == 'validation'
    assert command[command.index('--validation-per-class')+1] == '50'
    assert command[command.index('--split-seed')+1] == '1243'


def validation_payload(profile):
    return dict(status='completed', contract_hash=profile['contract_hash'], world_size=4,
        epoch=40, update_step=14080, evaluation_split='validation', train_samples=45000,
        validation_samples=5000, final_validation_accuracy=75., final_validation_loss=1.3,
        split_sha256='a'*64, training_recipe=expected_vision_recipe(profile))


def test_validation_result_cannot_be_replaced_by_official_test_metrics(tmp_path):
    profile = final_contract(tmp_path).quality['resnet18_cifar100_muon']
    result = validation_payload(profile)
    validate_vision_result(result, profile)
    for change in ({'evaluation_split':'test'}, {'validation_samples':10000}, {'train_samples':50000},
                   {'split_sha256':''}, {'final_validation_accuracy':float('nan')}):
        with pytest.raises(ValueError):
            validate_vision_result(result | change, profile)


def score_validation(tmp_path, *, accuracy=75., split='a'*64):
    from greedylore_autoresearch_grader.grader import score_results
    from test_utility import reference
    contract = final_contract(tmp_path)
    vision = validation_payload(contract.quality['resnet18_cifar100_muon'])
    dense = {'quality': {'workloads': {'llama60m_adamw': {'final_eval_loss': 3.},
        'llama130m_muon': {'final_eval_loss': 4.}, 'resnet18_cifar100_muon': vision}},
        'efficiency': {'cells': {name: {'mean_iteration_seconds': t} for name,t in reference()['timing'].items()}}}
    quality = {'llama60m_adamw': {'final_eval_loss': 3.}, 'llama130m_muon': {'final_eval_loss': 4.},
        'resnet18_cifar100_muon': vision | {'final_validation_accuracy': accuracy, 'split_sha256': split}}
    return score_results('research', quality, {name: {'mean': t/1.1, 'relative_spread': .01}
        for name,t in reference()['timing'].items()}, dense, contract_hash=contract.contract_hash,
        comparable=True, evidence_paths=[], quality_profiles=contract.quality, utility_reference=reference())


def test_validation_accuracy_gates_quality_but_does_not_enter_utility(tmp_path):
    result = score_validation(tmp_path, accuracy=70.)
    assert result.scores['resnet18_validation_accuracy'].value == 70.
    assert result.scores['quality_gate'].value == 0 and result.aggregated == 0
    assert result.metadata['utility']['value'] == pytest.approx(1.1)
    assert result.metadata['utility']['mean_loss_delta'] == 0


def test_different_validation_indices_fail_quality_without_wedging_publication(tmp_path):
    result = score_validation(tmp_path, split='b'*64)
    assert result.scores['quality_gate'].value == 0
    assert result.metadata['failure_class'] == 'candidate'
    assert 'validation_split_mismatch' in result.metadata['ineligibility_reasons']


def test_saved_v2_contract_keeps_official_test_and_legacy_utility(tmp_path):
    import yaml
    from greedylore_autoresearch_grader.contract import EvalContract
    from test_final_contract import TASK
    current = final_contract(tmp_path)
    raw = yaml.safe_load((TASK/'taskdata/eval_contract_v2.yaml').read_text())
    raw['data'] = dict(current.raw['data'])
    raw['vision']['data_path'] = current.raw['vision']['data_path']
    path = tmp_path/'old.yaml'
    path.write_text(yaml.safe_dump(raw))
    old = EvalContract.load(path)
    assert old.utility_version == 'worst-ppl-discount-v1'
    vision = old.quality['resnet18_cifar100_muon']
    assert 'evaluation_split' not in vision and vision['stop_after_steps'] == 15640
    result = dict(status='completed',contract_hash=old.contract_hash,world_size=4,epoch=40,
        update_step=15640,test_samples=10000,final_test_accuracy=75.,final_test_loss=1.3,
        training_recipe=expected_vision_recipe(vision))
    validate_vision_result(result, vision)
