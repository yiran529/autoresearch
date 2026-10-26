"""Timing model changes must reach commands, references and utility together."""
# ruff: noqa: S101
import pytest
import yaml
from greedylore_autoresearch_grader.contract import ContractError, EvalContract
from greedylore_autoresearch_grader.efficiency import iter_timing_runs
from greedylore_autoresearch_grader.grader import score_results
from greedylore_autoresearch_grader.utility import advisory_utility
from test_final_contract import TASK, final_contract


def test_default_timing_runs_use_350m_and_preserve_protocol(tmp_path):
    contract = final_contract(tmp_path)
    assert contract.raw['schema_version'] == 4
    runs = list(iter_timing_runs(contract.efficiency, tmp_path, tmp_path / 'out', (0, 1, 2, 3)))
    assert [r.cell_name for r in runs] == ['350m_adamw_bf16', '350m_adamw_fp32', '350m_adamw_fp32', '350m_adamw_bf16']
    for run in runs:
        argv = run.spec.argv
        assert argv[argv.index('--model_config') + 1] == 'c4/configs/llama_350m.json'
        assert argv[argv.index('--batch_size') + 1] == '1'
        assert argv[argv.index('--measured_iterations') + 1] == '400'
    assert contract.quality['llama130m_muon']['stop_after_steps'] == 4000
    assert contract.quality['llama60m_adamw']['stop_after_steps'] == 5000


def test_350m_scores_and_root_utility_use_same_model():
    timing = {f'350m_adamw_{p}': {'mean': .1, 'relative_spread': .01} for p in ('bf16', 'fp32')}
    quality = {'llama60m_adamw': {'final_eval_loss': 3.0}, 'llama130m_muon': {'final_eval_loss': 4.0}}
    ref = {'quality': {'workloads': quality}, 'efficiency': {'cells': {k: {'mean_iteration_seconds': .2} for k in timing}}}
    root = {'quality': {k: v['final_eval_loss'] for k, v in quality.items()}, 'timing': {k: .2 for k in timing}}
    bundle = score_results('research', quality, timing, ref, contract_hash='350m', comparable=True, evidence_paths=[], utility_reference=root)
    assert bundle.aggregated == pytest.approx(2)
    assert bundle.metadata['utility']['value'] == pytest.approx(2)
    old = {**root, 'timing': {k.replace('350m', '130m'): v for k, v in root['timing'].items()}}
    assert advisory_utility(root['quality'], {k: .1 for k in timing}, old)['value'] is None


def test_v3_snapshot_keeps_130m_timing(tmp_path):
    raw = yaml.safe_load((TASK / 'taskdata/eval_contract_v3.yaml').read_text())
    current = final_contract(tmp_path)
    raw['data']['c4_path'] = str(current.data.c4_path)
    raw['data']['tokenizer_path'] = str(current.data.tokenizer_path)
    raw['vision']['data_path'] = str(tmp_path / 'cifar')
    path = tmp_path / 'v3.yaml'
    path.write_text(yaml.safe_dump(raw))
    contract = EvalContract.load(path)
    run = next(iter_timing_runs(contract.efficiency, tmp_path, tmp_path / 'out', (0, 1, 2, 3)))
    assert run.spec.argv[run.spec.argv.index('--model_config') + 1] == 'c4/configs/llama_130m.json'
    assert contract.utility_version == 'mean-log-ppl-ratio-v2'


@pytest.mark.parametrize('field,value', [('model', 'llama130m'), ('cells', {'130m_adamw_bf16': {}, '130m_adamw_fp32': {}})])
def test_contract_rejects_model_cell_mismatch(tmp_path, field, value):
    contract = final_contract(tmp_path)
    raw = yaml.safe_load(contract.path.read_text())
    raw['efficiency'][field] = value
    contract.path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError):
        EvalContract.load(contract.path)


def test_350m_dense_reference_rejects_relabelled_130m_cells(tmp_path):
    import json

    from greedylore_autoresearch_grader.gpu import InfraFailure
    from test_final_references import references

    grader, contract, _, directory = references(tmp_path)
    path = directory / 'efficiency.json'
    payload = json.loads(path.read_text())
    payload['cells'] = {name.replace('350m', '130m'): value for name, value in payload['cells'].items()}
    path.write_text(json.dumps(payload))
    with pytest.raises(InfraFailure, match='schema'):
        grader._load_references(contract.contract_hash)
