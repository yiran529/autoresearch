"""Phase-based contract and classification gate retain legacy readability."""
# ruff: noqa: S101
from pathlib import Path

import pytest
import yaml

TASK = Path(__file__).resolve().parents[1]


def final_contract(tmp_path):
    from greedylore_autoresearch_grader.contract import EvalContract
    raw = yaml.safe_load((TASK / 'taskdata/eval_contract.yaml').read_text())
    for key in ('c4_path', 'tokenizer_path'):
        path = tmp_path / key
        path.mkdir(exist_ok=True)
        raw['data'][key] = str(path)
    data = tmp_path / 'cifar'
    data.mkdir(exist_ok=True)
    if 'vision' in raw:
        raw['vision']['data_path'] = str(data)
    path = tmp_path / 'contract.yaml'
    path.write_text(yaml.safe_dump(raw))
    return EvalContract.load(path)


def test_final_contract_separates_development_and_final_quality(tmp_path):
    contract = final_contract(tmp_path)
    assert contract.raw['schema_version'] == 3
    assert contract.development_workload == 'llama130m_muon'
    assert contract.final_workloads == ('llama60m_adamw', 'resnet18_cifar100_muon')
    assert set(contract.quality) == {'llama130m_muon', 'llama60m_adamw', 'resnet18_cifar100_muon'}
    vision = contract.quality['resnet18_cifar100_muon']
    assert vision['epochs'] == 40 and vision['stop_after_steps'] == 14080
    assert vision['per_device_batch_size'] == 32 and vision['precision'] == 'fp32'
    assert vision['matrix_learning_rate'] == .02 and vision['scalar_learning_rate'] == .0005
    assert 'rank' not in vision and 'quality_start_step' not in vision
    assert 'c4_path' not in vision and 'tokenizer_path' not in vision


def test_classification_gate_uses_final_accuracy_and_all_required_workloads():
    from greedylore_autoresearch_grader.grader import score_results
    quality = {'llama60m_adamw': {'final_eval_loss': 3.0},
               'llama130m_muon': {'final_eval_loss': 4.0},
               'resnet18_cifar100_muon': {'final_test_accuracy': 74.8, 'best_test_accuracy': 99.0, 'final_test_loss': 1.3}}
    reference = {'quality': {'workloads': {'llama60m_adamw': {'final_eval_loss': 3.0},
        'llama130m_muon': {'final_eval_loss': 4.0},
        'resnet18_cifar100_muon': {'final_test_accuracy': 75.91, 'final_test_loss': 1.27233}}},
        'efficiency': {'cells': {name: {'mean_iteration_seconds': .2} for name in ('130m_adamw_bf16','130m_adamw_fp32')}}}
    timing = {name: {'mean': .1, 'relative_spread': .01} for name in reference['efficiency']['cells']}
    profiles = {'resnet18_cifar100_muon': {'max_accuracy_drop_pp': 1.0}}
    result = score_results('research', quality, timing, reference, contract_hash='fixed', comparable=True,
                           evidence_paths=[], quality_profiles=profiles)
    assert result.scores['resnet18_accuracy_drop_pp'].value == pytest.approx(1.11)
    assert result.scores['resnet18_quality_pass'].value == 0
    assert result.scores['quality_gate'].value == 0 and result.aggregated == 0
    quality['resnet18_cifar100_muon']['final_test_accuracy'] = 74.94
    result = score_results('research', quality, timing, reference, contract_hash='fixed', comparable=True,
                           evidence_paths=[], quality_profiles=profiles)
    assert result.scores['quality_gate'].value == 1 and result.aggregated == pytest.approx(2)
