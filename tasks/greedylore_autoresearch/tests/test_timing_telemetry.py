# ruff: noqa: S101
import json
import os
import subprocess
from pathlib import Path

import pytest
from greedylore_autoresearch_grader.timing_policy import plan_timing, validate_cycle_evidence


def test_real_cpu_hook_refresh_events_survive_timing_result_serialization(tmp_path):
    python = Path('/home/wyr/greedy_lore/.venv/bin/python')
    if not python.is_file():
        pytest.skip('local CPU PyTorch environment unavailable')
    seed = Path(__file__).resolve().parents[1] / 'seed'
    result = subprocess.run(
        [str(python), str(Path(__file__).with_name('timing_cpu_cases.py')), str(seed), str(tmp_path)],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'GLOO_SOCKET_IFNAME': 'lo'},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((tmp_path / 'telemetry.json').read_text())
    assert payload['timing_cycle'] == {'period_steps': 4, 'rank_event_steps': [[7, 11]]}
    plan = plan_timing({'period_steps': 4}, minimum=8, maximum=8)
    assert validate_cycle_evidence(plan, payload)['confirmed'] is True
