"""Publish the actual evaluation recipe and comparators, never grader internals."""

from collections.abc import Mapping
from pathlib import Path

from coral.grader.trusted_writes import _atomic_json


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def quality_policies(workloads, max_relative_eval_loss):
    """Describe each workload's actual quality gate for public evidence."""
    policies = {}
    for name, profile in workloads.items():
        if name == 'resnet18_cifar100_muon':
            from .vision import vision_metric_fields
            metric = vision_metric_fields(profile)[0]
            policies[name] = {
                'metric': metric,
                'comparison': f'dense {metric} - candidate {metric} <= max_accuracy_drop_pp',
                'max_accuracy_drop_pp': profile['max_accuracy_drop_pp'],
                'selection': 'final_epoch',
                **({'validation_examples': 100 * profile['validation_per_class'],
                    'split_seed': profile['split_seed']} if profile.get('evaluation_split') == 'validation'
                   else {'test_examples': 10000}),
            }
        else:
            policies[name] = {
                'metric': 'final_eval_loss',
                'comparison': 'candidate final_eval_loss / dense final_eval_loss <= max_relative_eval_loss',
                'max_relative_eval_loss': max_relative_eval_loss,
            }
    return policies


def publish_context(contract, reference, root: Path, mode: str, timing_protocol=None) -> Path:
    """Called once before workloads, so failed runs also retain their real recipe."""
    path = root / 'evaluation_context.json'
    value = {
        'version': 1, 'contract_hash': contract.contract_hash, 'mode': mode,
        'evaluation_policy': {'version': 'quality-gated-efficiency-v2',
                              'utility_is_advisory': True},
        'quality': {
            'metric': 'final_eval_loss',
            'comparison': 'candidate final_eval_loss / dense final_eval_loss <= max_relative_eval_loss',
            'max_relative_eval_loss': contract.raw['quality']['max_relative_eval_loss'],
            'workloads': contract.quality,
        },
        'dense_quality': reference['quality']['workloads'],
        'timing': {'cells': contract.efficiency.cells,
                   'run_order': contract.efficiency.run_order,
                   'comparison': 'dense mean_iteration_seconds / candidate mean_iteration_seconds'},
        'dense_timing': reference['efficiency']['cells'],
        'timing_protocol': timing_protocol,
    }
    if contract.raw['schema_version'] >= 2:
        value['quality'].update(
            metric='workload_specific',
            comparison='apply each workload policy against its same-contract dense reference',
            workload_policies=quality_policies(
                contract.quality, contract.raw['quality']['max_relative_eval_loss']),
        )
    _atomic_json(path, _plain(value))
    return path


def publish_hardware_comparison(root: Path, comparison: dict) -> None:
    """Retain the actual timing allocation and differences before timing starts."""
    import json

    path = root / 'evaluation_context.json'
    value = json.loads(path.read_text())
    value['hardware_comparison'] = comparison
    _atomic_json(path, _plain(value))


def publish_timing_protocol(root: Path, protocol: dict) -> None:
    import json

    path = root / 'evaluation_context.json'
    value = json.loads(path.read_text())
    value['timing_protocol'] = protocol
    _atomic_json(path, _plain(value))
