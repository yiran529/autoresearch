import json

import pytest
from test_analysis_recovery import FakeCodex, _analyze, _context, _publisher


def test_grader_records_actual_contract_and_dense_without_private_paths(contract, tmp_path):
    from greedylore_autoresearch_grader.evaluation_context import publish_context

    references = {
        'quality': {'contract_hash': contract.contract_hash, 'workloads': {
            'llama60m_adamw': {'final_eval_loss': 3.5},
            'llama130m_muon': {'final_eval_loss': 3.6}}},
        'efficiency': {'contract_hash': contract.contract_hash, 'cells': {
            '130m_adamw_bf16': {'mean_iteration_seconds': 0.1}}},
    }
    path = publish_context(contract, references, tmp_path / 'logs', 'research')
    value = json.loads(path.read_text())
    assert value['contract_hash'] == contract.contract_hash
    assert value['quality']['max_relative_eval_loss'] == 1.05
    assert value['quality']['workloads']['llama60m_adamw']['stop_after_steps'] == 4000
    assert value['dense_quality']['llama60m_adamw']['final_eval_loss'] == 3.5
    assert 'environment' not in value


def test_normal_analyzer_receives_recorded_context_not_current_config(tmp_path):
    context = _context(tmp_path)
    directory = context.log_root / ('r' * 40) / ('run-' + 'r' * 40)
    directory.mkdir(parents=True)
    (directory / 'evaluation_context.json').write_text(json.dumps({
        'version': 1, 'contract_hash': 'contract-v1', 'mode': 'research',
        'quality': {'metric': 'final_eval_loss', 'max_relative_eval_loss': 1.10,
                    'workloads': {'llama60m_adamw': {'stop_after_steps': 4000}}},
        'dense_quality': {'llama60m_adamw': {'final_eval_loss': 3.5}},
    }))
    context.coral_dir.mkdir()
    (context.coral_dir / 'config.yaml').write_text('grader:\n  args:\n    fake_gate: 999\n')
    runner = FakeCodex(context)
    runner.mode = 'research'
    result = _analyze('N001', 'research', context=context, runner=runner,
                      publisher=_publisher(context.notes_root))
    assert 'Evaluation context (recorded for this Attempt)' in runner.prompts[0]
    assert '1.1' in runner.prompts[0] and '3.5' in runner.prompts[0]
    assert '999' not in runner.prompts[0]
    assert 'evaluation_context.json' in result.analysis_path.read_text()


def test_legacy_analyzer_reports_unavailable_context(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context)
    runner.mode = 'research'
    _analyze('N001', 'research', context=context, runner=runner,
             publisher=_publisher(context.notes_root))
    assert 'Historical evaluation context: unavailable' in runner.prompts[0]


def test_analyzer_rejects_context_for_another_contract(tmp_path):
    from autoresearch.validation import ArtifactValidationError

    context = _context(tmp_path)
    directory = context.log_root / ('r' * 40) / ('run-' + 'r' * 40)
    directory.mkdir(parents=True)
    (directory / 'evaluation_context.json').write_text(json.dumps({
        'version': 1, 'contract_hash': 'another-contract', 'mode': 'research'}))
    runner = FakeCodex(context)
    runner.mode = 'research'
    with pytest.raises(ArtifactValidationError, match='context.*contract'):
        _analyze('N001', 'research', context=context, runner=runner,
                 publisher=_publisher(context.notes_root))
