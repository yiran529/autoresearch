"""Expose task-local packages without installing into CORAL's environment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
for source in (TASK_ROOT / "grader" / "src", TASK_ROOT / "autoresearch" / "src"):
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)


@pytest.fixture
def contract(tmp_path):
    from greedylore_autoresearch_grader.contract import EvalContract

    source = TASK_ROOT / "taskdata" / "eval_contract.yaml"
    raw = yaml.safe_load(source.read_text())
    data = tmp_path / "c4"
    data.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    raw["data"]["c4_path"] = str(data)
    raw["data"]["tokenizer_path"] = str(tokenizer)
    path = tmp_path / "eval_contract.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return EvalContract.load(path)
