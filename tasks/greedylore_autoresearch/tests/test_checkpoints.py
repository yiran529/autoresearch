import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

SEED = Path(__file__).resolve().parents[1] / 'seed'


def test_checkpoint_roundtrip_with_real_cpu_torch(tmp_path):
    python = Path('/home/wyr/greedy_lore/.venv/bin/python')
    if not python.is_file():
        pytest.skip('existing local PyTorch environment is unavailable')
    result = subprocess.run([str(python), str(Path(__file__).with_name('checkpoint_cpu_cases.py')),
                             str(SEED), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_rank_checkpoint_continuation_and_partial_failure(tmp_path):
    python = Path('/home/wyr/greedy_lore/.venv/bin/python')
    if not python.is_file():
        pytest.skip('existing local PyTorch environment is unavailable')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'GLOO_SOCKET_IFNAME': 'lo'}
    result = subprocess.run([str(python), str(Path(__file__).with_name('checkpoint_distributed_cases.py')),
                             str(SEED), str(tmp_path)], capture_output=True, text=True,
                            env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads((tmp_path / 'passed.json').read_text())
    assert value['ranks'] == 2
    assert value['optimizers'] == ['adamw', 'muon']


def test_quality_enables_one_final_checkpoint_in_persistent_output(contract, tmp_path):
    from greedylore_autoresearch_grader.quality import build_quality_command

    candidate = tmp_path / 'temporary-checkout'
    candidate.mkdir()
    (candidate / 'diagnostic_checkpoint.py').write_text('# checkpoint-capable seed\n')
    output = tmp_path / 'persistent/quality/llama60m_adamw'
    command = build_quality_command(contract.quality['llama60m_adamw'], candidate, output,
                                    (0, 1, 2, 3)).argv
    assert command[command.index('--diagnostic_checkpoint') + 1] == str(
        tmp_path / 'persistent/checkpoints/llama60m_adamw')


def test_legacy_candidate_without_checkpoint_support_keeps_quality_command(contract, tmp_path):
    from greedylore_autoresearch_grader.quality import build_quality_command

    command = build_quality_command(contract.quality['llama60m_adamw'], tmp_path,
                                    tmp_path / 'output', (0, 1, 2, 3)).argv
    assert '--diagnostic_checkpoint' not in command


def _checkpoint(context):
    root = context.log_root / ('r' * 40) / ('run-' + 'r' * 40) / 'checkpoints/llama60m_adamw'
    root.mkdir(parents=True)
    files = {}
    for name in ('model.pt', 'rank-0.pt'):
        data = b'opaque tensor bytes'
        (root / name).write_bytes(data)
        files[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    (root / 'manifest.json').write_text(json.dumps({
        'version': 1, 'status': 'complete', 'world_size': 1, 'update_step': 4000,
        'contract_hash': 'contract-v1', 'files': files, 'config': {'compressor': 'top_subspace'},
    }))
    return root


def test_normal_analyzer_mounts_checkpoint_without_copying_tensors(tmp_path):
    from test_analysis_recovery import FakeCodex, _analyze, _context, _publisher

    context = _context(tmp_path)
    checkpoint = _checkpoint(context)
    runner = FakeCodex(context)
    runner.mode = 'research'
    result = _analyze('N001', 'research', context=context, runner=runner,
                      publisher=_publisher(context.notes_root))
    assert checkpoint in runner.read_roots[0]
    assert str(checkpoint) in runner.prompts[0]
    assert 'load_checkpoint' in runner.prompts[0]
    snapshot = runner.scratch[0] / 'evidence'
    assert list(snapshot.rglob('manifest.json'))
    assert not list(snapshot.rglob('*.pt'))
    assert not any(record.path.endswith('.pt') for record in result.evidence.evidence_records)
    assert (checkpoint / 'model.pt').is_file()  # Deletion is explicitly not implemented.


def test_changed_checkpoint_is_rejected_before_analyzer_launch(tmp_path):
    from autoresearch.validation import ArtifactValidationError
    from test_analysis_recovery import FakeCodex, _analyze, _context, _publisher

    context = _context(tmp_path)
    checkpoint = _checkpoint(context)
    (checkpoint / 'rank-0.pt').write_bytes(b'changed')
    runner = FakeCodex(context)
    runner.mode = 'research'
    with pytest.raises(ArtifactValidationError, match='checkpoint.*integrity'):
        _analyze('N001', 'research', context=context, runner=runner,
                 publisher=_publisher(context.notes_root))
    assert runner.analysis_calls == 0


def test_incomplete_checkpoint_is_reported_but_not_mounted(tmp_path):
    from test_analysis_recovery import FakeCodex, _analyze, _context, _publisher

    context = _context(tmp_path)
    checkpoint = _checkpoint(context)
    (checkpoint / 'manifest.json').write_text(json.dumps({'version': 1, 'status': 'unavailable',
                                                        'errors': ['disk full']}))
    runner = FakeCodex(context)
    runner.mode = 'research'
    _analyze('N001', 'research', context=context, runner=runner,
             publisher=_publisher(context.notes_root))
    assert checkpoint not in runner.read_roots[0]
    assert 'disk full' in runner.prompts[0]
