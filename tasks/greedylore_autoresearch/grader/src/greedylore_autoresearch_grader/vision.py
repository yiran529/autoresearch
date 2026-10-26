"""Build and validate the bounded final CIFAR100 evaluation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .process import ProcessSpec


def build_vision_command(
    profile: Mapping[str, Any], candidate: str | Path, output: str | Path,
    gpus: Sequence[int], *, checkpoint_root: str | Path | None = None,
    checkpoint_interval_steps: int = 1000, training_metrics: str | Path | None = None,
) -> ProcessSpec:
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    count = int(profile['gpu_count'])
    if len(gpus) != count or len(set(gpus)) != count:
        raise ValueError('vision allocation must contain the requested number of distinct GPUs')
    compressor = profile.get('compressor', {'name': 'top_subspace'})
    name = compressor['name'] if isinstance(compressor, Mapping) else compressor
    command = [
        'uv', 'run', '--frozen', '--project', str(candidate),
        'python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(count),
        'pytorch-cifar/train_ddp.py', '--dataset', str(profile['dataset']),
        '--data-dir', str(profile['data_path']), '--output-dir', str(output),
        '--epochs', str(profile['epochs']), '--stop-after-steps', str(profile['stop_after_steps']),
        '--batch-size', str(profile['per_device_batch_size']), '--workers', str(profile['workers']),
        '--seed', str(profile['seed']), '--lr', str(profile['matrix_learning_rate']),
        '--weight-decay', str(profile['weight_decay']), '--muon_mu', '0.95',
        '--muon_epsilon', '1e-8', '--muon_scalar_lr', str(profile['scalar_learning_rate']),
        '--muon_scalar_beta1', '0.9', '--muon_scalar_beta2', '0.999',
        '--muon_scalar_eps', '1e-8', '--muon_scalar_weight_decay', str(profile['weight_decay']),
        '--muon_adjust_lr', 'spectral_norm', '--compressor', str(name),
        '--contract-hash', str(profile['contract_hash']), '--wandb-mode', 'disabled',
        '--no-download', '--no-communication-bit-estimates',
    ]
    if checkpoint_root is not None:
        if type(checkpoint_interval_steps) is not int or checkpoint_interval_steps <= 0:
            raise ValueError('checkpoint_interval_steps must be a positive integer')
        command += ['--checkpoint-root', str(Path(checkpoint_root).resolve()),
                    '--checkpoint-interval-steps', str(checkpoint_interval_steps)]
    if training_metrics is not None:
        command += ['--training-metrics', str(Path(training_metrics).resolve())]
    env = {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)), 'WANDB_MODE': 'disabled',
           'OMP_NUM_THREADS': '2', 'PYTHONPATH': str(candidate),
           'CORAL_WORKLOAD': str(profile['name'])}
    return ProcessSpec(tuple(command), output.parent / f'{output.name}.process', env,
                       tuple(env), 24 * 60 * 60, cwd=candidate)


def expected_vision_recipe(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {key: profile[key] for key in ('model', 'dataset', 'epochs', 'per_device_batch_size',
            'workers', 'seed', 'precision', 'optimizer', 'matrix_learning_rate',
            'scalar_learning_rate', 'weight_decay')} | {
        'matrix_momentum': .95, 'nesterov': True, 'adjust_lr': 'spectral_norm',
        'scalar_betas': [.9, .999], 'scalar_eps': 1e-8,
        'schedule': 'epoch_cosine', 'lr_warmup_steps': 0,
        'augmentation': 'cifar_random_crop4_flip', 'bias_bn_weight_decay': 0.,
    }


def validate_vision_result(payload: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    """Reject incomplete, synthetic, mismatched, or nonfinite final results."""
    if payload.get('status') != 'completed' or payload.get('contract_hash') != profile['contract_hash']:
        raise ValueError('vision result status or contract mismatch')
    if payload.get('training_recipe') != expected_vision_recipe(profile):
        raise ValueError('vision result training recipe mismatch')
    for key, expected in [('world_size', profile['gpu_count']), ('epoch', profile['epochs']),
                          ('update_step', profile['stop_after_steps']), ('test_samples', 10000)]:
        if type(payload.get(key)) is not int or payload[key] != expected:
            raise ValueError(f'vision result {key} mismatch')
    for key in ('final_test_loss', 'final_test_accuracy'):
        value = payload.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'vision result {key} must be finite')
    if payload['final_test_loss'] <= 0 or not 0 <= payload['final_test_accuracy'] <= 100:
        raise ValueError('vision final metrics are outside valid bounds')
