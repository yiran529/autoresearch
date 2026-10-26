"""Run staged checkpoint contracts in the existing real PyTorch environment."""
import os
from pathlib import Path
import subprocess


SEED = Path(__file__).resolve().parents[1] / 'seed'


def test_staged_replay_and_periodic_checkpoints(tmp_path):
    result = subprocess.run(
        ['/home/wyr/greedy_lore/.venv/bin/python',
         str(Path(__file__).with_name('training_stream_cpu_cases.py')), str(SEED), str(tmp_path)],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_c4_runner_staged_segments_on_cpu(tmp_path):
    result = subprocess.run(
        ['/home/wyr/greedy_lore/.venv/bin/python',
         str(Path(__file__).with_name('training_runner_cpu_cases.py')), str(SEED), str(tmp_path)],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_staged_two_rank_fresh_ddp_restarts(tmp_path):
    result = subprocess.run(
        ['/home/wyr/greedy_lore/.venv/bin/python',
         str(Path(__file__).with_name('staged_checkpoint_distributed_cases.py')), str(SEED), str(tmp_path)],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'GLOO_SOCKET_IFNAME': 'lo'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
