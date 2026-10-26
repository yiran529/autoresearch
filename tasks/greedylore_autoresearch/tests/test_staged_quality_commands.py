from pathlib import Path

import pytest


def test_authoritative_checkpoint_period_defaults_and_override(contract, tmp_path):
    from greedylore_autoresearch_grader.quality import build_quality_command

    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    root = tmp_path / 'external/checkpoints'
    profile = contract.quality['llama130m_muon']
    default = build_quality_command(profile, candidate, tmp_path / 'out', (0, 1, 2, 3),
                                    checkpoint_root=root).argv
    assert default[default.index('--checkpoint_interval_steps') + 1] == '1000'
    changed = build_quality_command(profile, candidate, tmp_path / 'out-2', (0, 1, 2, 3),
        checkpoint_root=root, checkpoint_interval_steps=750, pause_after_steps=1500,
        resume_checkpoint=root / 'step-00000750', training_metrics=tmp_path / 'metrics.jsonl').argv
    assert changed[changed.index('--checkpoint_interval_steps') + 1] == '750'
    assert changed[changed.index('--pause_after_steps') + 1] == '1500'
    assert changed[changed.index('--resume_checkpoint') + 1] == str(root / 'step-00000750')
    assert changed[changed.index('--stop_after_steps') + 1] == '2500'


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5])
def test_invalid_period_rejected_before_training(contract, tmp_path, invalid):
    from greedylore_autoresearch_grader.quality import build_quality_command

    with pytest.raises(ValueError):
        build_quality_command(contract.quality['llama130m_muon'], tmp_path, tmp_path / 'out', (0,1,2,3),
                              checkpoint_root=tmp_path / 'checkpoints', checkpoint_interval_steps=invalid)
