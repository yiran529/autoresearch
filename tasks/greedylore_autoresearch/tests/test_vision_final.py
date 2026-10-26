"""Bounded vision evaluation commands and CPU execution contracts."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return dict(name='resnet18_cifar100_muon', model='resnet18', dataset='cifar100',
                data_path='/data/cifar', epochs=40, stop_after_steps=15640,
                per_device_batch_size=32, workers=2, seed=1243, precision='fp32',
                optimizer='muon', matrix_learning_rate=.02, scalar_learning_rate=.0005,
                weight_decay=.0005, contract_hash='vision-contract', gpu_count=4)


def test_vision_command_uses_training_contract_and_candidate_compression_policy(tmp_path):
    from greedylore_autoresearch_grader.vision import build_vision_command
    spec = build_vision_command(profile(), tmp_path / 'candidate', tmp_path / 'output', [4, 5, 6, 7],
                                checkpoint_root=tmp_path / 'checkpoints',
                                checkpoint_interval_steps=750, training_metrics=tmp_path / 'train.jsonl')
    command = list(spec.command)
    for flag, value in [('--nproc_per_node', '4'), ('--dataset', 'cifar100'),
                        ('--data-dir', '/data/cifar'), ('--epochs', '40'),
                        ('--stop-after-steps', '15640'), ('--batch-size', '32'),
                        ('--workers', '2'), ('--seed', '1243'), ('--lr', '0.02'),
                        ('--muon_scalar_lr', '0.0005'), ('--muon_mu', '0.95'),
                        ('--muon_adjust_lr', 'spectral_norm'), ('--contract-hash', 'vision-contract'),
                        ('--checkpoint-interval-steps', '750'), ('--wandb-mode', 'disabled')]:
        assert command[command.index(flag) + 1] == value
    assert '--no-download' in command and '--no-communication-bit-estimates' in command
    assert not {'--compress-rank', '--start-compress-iter', '--update-proj-gap', '--smoke-steps', '--dry-run'} & set(command)
    assert spec.env['CORAL_WORKLOAD'] == 'resnet18_cifar100_muon'
    assert spec.env['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'
    assert spec.cwd == (tmp_path / 'candidate').resolve()
    dense = dict(profile(), compressor={'name': 'none'})
    command = list(build_vision_command(dense, tmp_path, tmp_path / 'dense', [0, 1, 2, 3]).command)
    assert command[command.index('--compressor') + 1] == 'none'


def test_vision_command_rejects_invalid_allocation_and_checkpoint_period(tmp_path):
    from greedylore_autoresearch_grader.vision import build_vision_command
    for gpus in ([0], [0, 0, 1, 2]):
        with pytest.raises(ValueError):
            build_vision_command(profile(), tmp_path, tmp_path / 'output', gpus)
    for period in (0, -1, True):
        with pytest.raises(ValueError):
            build_vision_command(profile(), tmp_path, tmp_path / 'output', [0, 1, 2, 3],
                                 checkpoint_root=tmp_path / 'checkpoints', checkpoint_interval_steps=period)


def test_vision_result_requires_complete_final_epoch_and_full_test_set():
    from greedylore_autoresearch_grader.vision import expected_vision_recipe, validate_vision_result
    result = dict(status='completed', contract_hash='vision-contract', world_size=4,
                  epoch=40, update_step=15640, test_samples=10000,
                  final_test_loss=1.5, final_test_accuracy=65.2, best_test_accuracy=99.,
                  training_recipe=expected_vision_recipe(profile()), compressor='top_subspace')
    assert validate_vision_result(result, profile()) is None
    for key, value in [('status', 'smoke_completed'), ('contract_hash', 'other'), ('world_size', 1),
                       ('epoch', 39), ('update_step', 15639), ('test_samples', 9999),
                       ('final_test_loss', float('nan')), ('final_test_loss', 0),
                       ('final_test_accuracy', 101), ('final_test_accuracy', float('inf')),
                       ('final_test_loss', True), ('epoch', 40.0), ('training_recipe', {}),
                       ('training_recipe', dict(result['training_recipe'], seed=999))]:
        with pytest.raises(ValueError):
            validate_vision_result(dict(result, **{key: value}), profile())


def test_real_vision_runner_cpu_contract(tmp_path):
    result = subprocess.run(
        ['/home/wyr/greedy_lore/.venv/bin/python', str(Path(__file__).with_name('vision_cpu_cases.py')),
         str(ROOT / 'seed'), str(tmp_path)],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'GLOO_SOCKET_IFNAME': 'lo'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
