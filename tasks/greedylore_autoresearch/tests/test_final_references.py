"""New-contract dense vision references must carry complete fixed-task evidence."""
import json

import pytest
from test_final_contract import final_contract


def references(tmp_path):
    from greedylore_autoresearch_grader.grader import Grader
    from test_validation_vision import validation_payload

    from coral.config import GraderConfig
    contract = final_contract(tmp_path)
    private = tmp_path / 'private'
    root = private / 'dense_references'
    root.mkdir(parents=True)
    (private / 'eval_contract.yaml').write_text(contract.path.read_text())
    vision = validation_payload(contract.quality['resnet18_cifar100_muon']) | {'compressor': 'none'}
    quality = {'contract_hash': contract.contract_hash, 'workloads': {
        'llama60m_adamw': {'final_eval_loss': 3.0}, 'llama130m_muon': {'final_eval_loss': 4.0},
        'resnet18_cifar100_muon': vision}}
    (root / 'quality.json').write_text(json.dumps(quality))
    efficiency = {'contract_hash': contract.contract_hash, 'cells': {
        name: {'mean_iteration_seconds': .1} for name in ['130m_adamw_bf16', '130m_adamw_fp32']},
        'environment': {'gpu_uuids': ['a','b','c','d'], 'gpu_names': ['test'] * 4,
                        'numa_nodes': [0]*4, 'topology_fingerprint': 'test', 'nccl_env': {}}}
    (root / 'efficiency.json').write_text(json.dumps(efficiency))
    grader = Grader(GraderConfig())
    grader.private_dir = str(private)
    return grader, contract, quality, root


def test_dense_vision_complete_matching_result_accepted(tmp_path):
    grader, contract, _, _ = references(tmp_path)
    assert grader._load_references(contract.contract_hash)['quality']['workloads']['resnet18_cifar100_muon']['compressor'] == 'none'


@pytest.mark.parametrize('mutation', ['historical_metrics', 'incomplete', 'other_recipe', 'compressed'])
def test_dense_vision_rejects_relabelled_metrics_or_wrong_recipe(tmp_path, mutation):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    grader, contract, quality, root = references(tmp_path)
    vision = quality['workloads']['resnet18_cifar100_muon']
    if mutation == 'historical_metrics':
        quality['workloads']['resnet18_cifar100_muon'] = {'final_test_loss': 1.27233, 'final_test_accuracy': 75.91}
    elif mutation == 'incomplete':
        vision['epoch'] = 39
    elif mutation == 'other_recipe':
        vision['training_recipe']['matrix_learning_rate'] = .1
    else:
        vision['compressor'] = 'top_subspace'
    (root / 'quality.json').write_text(json.dumps(quality))
    with pytest.raises(InfraFailure, match='dense'):
        grader._load_references(contract.contract_hash)
