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


def publish_context(contract, reference, root: Path, mode: str) -> Path:
    """Called once before workloads, so failed runs also retain their real recipe."""
    path = root / 'evaluation_context.json'
    value = {
        'version': 1, 'contract_hash': contract.contract_hash, 'mode': mode,
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
    }
    _atomic_json(path, _plain(value))
    return path


def publish_hardware_comparison(root: Path, comparison: dict) -> None:
    """Retain the actual timing allocation and differences before timing starts."""
    import json

    path = root / 'evaluation_context.json'
    value = json.loads(path.read_text())
    value['hardware_comparison'] = comparison
    _atomic_json(path, _plain(value))
