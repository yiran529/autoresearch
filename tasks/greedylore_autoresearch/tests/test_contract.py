from pathlib import Path

import pytest
import yaml
from greedylore_autoresearch_grader.contract import (
    ContractError,
    EvalContract,
    canonical_contract_hash,
)

SOURCE = Path(__file__).parents[1] / "taskdata" / "eval_contract.yaml"


def make_contract(tmp_path):
    raw = yaml.safe_load(SOURCE.read_text())
    data = tmp_path / "c4"
    data.mkdir(parents=True)
    tok = tmp_path / "tokenizer"
    tok.mkdir()
    raw["data"]["c4_path"] = str(data)
    raw["data"]["tokenizer_path"] = str(tok)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_contract_is_immutable_and_hash_stable(tmp_path):
    path = make_contract(tmp_path)
    first = EvalContract.load(path)
    second = EvalContract.load(path)
    assert first.contract_hash == second.contract_hash
    assert first.contract_hash == canonical_contract_hash(path)
    assert first.hardware.gpu_count == 4
    assert first.budget_class == "real"
    with pytest.raises(TypeError):
        first.raw["training"] = {}


def test_legacy_contract_remains_readable_without_rewriting_hash(tmp_path):
    path = make_contract(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['efficiency'].pop('max_measured_iterations')
    raw['quality']['max_relative_eval_loss'] = 1.06
    path.write_text(yaml.safe_dump(raw))
    old_hash = canonical_contract_hash(path)
    assert EvalContract.load(path).contract_hash == old_hash


def test_contract_rejects_unknown_keys_and_missing_paths(tmp_path):
    path = make_contract(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["surprise"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match="unknown"):
        EvalContract.load(path)
    path = make_contract(tmp_path / "again")
    raw = yaml.safe_load(path.read_text())
    raw["data"]["c4_path"] = str(tmp_path / "missing")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match="does not exist"):
        EvalContract.load(path)


def test_validation_mode_has_distinct_budget_and_hash(tmp_path):
    formal = EvalContract.load(make_contract(tmp_path))
    smoke = EvalContract.load(formal.path, validation=True)
    assert smoke.budget_class == "validation"
    assert smoke.contract_hash != formal.contract_hash
    assert (
        smoke.raw["efficiency"]["measured_iterations"]
        < formal.raw["efficiency"]["measured_iterations"]
    )


def test_contract_fails_closed_without_fixed_schema(tmp_path, monkeypatch):
    import greedylore_autoresearch_grader.contract as module

    path = make_contract(tmp_path)
    monkeypatch.setattr(module, "_SCHEMA", None)
    with pytest.raises(ContractError, match="fixed contract schema"):
        EvalContract.load(path)


def test_contract_rejects_invalid_scalar_invariants(tmp_path):
    path = make_contract(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["hardware"]["gpu_count"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match="gpu_count"):
        EvalContract.load(path)
