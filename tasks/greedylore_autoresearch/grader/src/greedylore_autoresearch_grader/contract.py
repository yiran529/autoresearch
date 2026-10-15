"""Strict immutable evaluation contract parsing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class ContractError(ValueError):
    pass


def _schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _schema(child) for key, child in value.items()}
    return None


def _check_keys(value: Any, schema: Any, path: str = "contract") -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ContractError(f"{path} must be a mapping")
        unknown = set(value) - set(schema)
        missing = set(schema) - set(value)
        if unknown or missing:
            raise ContractError(f"{path} has unknown/missing keys: {sorted(unknown | missing)}")
        for key in schema:
            _check_keys(value[key], schema[key], f"{path}.{key}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def canonical_contract_hash(path: str | Path) -> str:
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read contract: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("contract must be a mapping")
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ContractError(f"{field} must be finite and positive")
    return number


def _validate_values(raw: dict[str, Any]) -> None:
    _positive_int(raw["hardware"]["gpu_count"], "hardware.gpu_count")
    _positive_int(raw["data"]["dataloader_workers"], "data.dataloader_workers")
    _positive_int(raw["training"]["seed"], "training.seed")
    _positive_int(raw["training"]["sequence_length"], "training.sequence_length")
    _positive_int(raw["efficiency"]["repetitions"], "efficiency.repetitions")
    if raw["efficiency"]["repetitions"] != 2:
        raise ContractError("efficiency.repetitions must equal 2")
    for field in ("warmup_iterations", "measured_iterations", "bucket_cap_mb"):
        _positive_int(raw["efficiency"][field], f"efficiency.{field}")
    _finite_positive(raw["quality"]["max_relative_eval_loss"], "quality.max_relative_eval_loss")
    cells = raw["efficiency"]["cells"]
    if not isinstance(cells, dict) or set(cells) != {"130m_adamw_bf16", "130m_adamw_fp32"}:
        raise ContractError("efficiency.cells must contain the two fixed cells")
    order = raw["efficiency"]["run_order"]
    if (
        not isinstance(order, list)
        or len(order) != 2
        or any(not isinstance(row, list) or set(row) != set(cells) for row in order)
    ):
        raise ContractError("efficiency.run_order must contain both cells twice")
    for name, workload in raw["quality"]["workloads"].items():
        for field in ("per_device_batch_size", "scheduler_steps", "stop_after_steps"):
            _positive_int(workload[field], f"quality.workloads.{name}.{field}")


_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "taskdata" / "eval_contract.yaml"
try:
    _SCHEMA = _schema(yaml.safe_load(_TEMPLATE_PATH.read_text()))
except OSError:
    _SCHEMA = None


@dataclass(frozen=True)
class DataPaths:
    c4_path: Path
    tokenizer_path: Path


@dataclass(frozen=True)
class HardwareContract:
    gpu_count: int
    selection: str


@dataclass(frozen=True)
class EvalContract:
    path: Path
    contract_hash: str
    budget_class: str
    data: DataPaths
    hardware: HardwareContract
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, validation: bool = False) -> EvalContract:
        path = Path(path).resolve()
        try:
            loaded = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ContractError(f"cannot read contract: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ContractError("contract must be a mapping")
        if _SCHEMA is None:
            raise ContractError("fixed contract schema is unavailable")
        _check_keys(loaded, _SCHEMA)
        raw = copy.deepcopy(loaded)
        _validate_values(raw)
        for key in ("c4_path", "tokenizer_path"):
            resolved = Path(raw["data"][key]).expanduser().resolve()
            if not resolved.exists():
                raise ContractError(f"data.{key} does not exist: {resolved}")
            raw["data"][key] = str(resolved)
        budget_class = "validation" if validation else "real"
        if validation:
            raw["quality"]["workloads"]["llama60m_adamw"]["stop_after_steps"] = 2
            raw["quality"]["workloads"]["llama130m_muon"]["stop_after_steps"] = 2
            raw["efficiency"]["warmup_iterations"] = 1
            raw["efficiency"]["measured_iterations"] = 2
            raw["_budget_class"] = budget_class
        digest = hashlib.sha256(_canonical(raw)).hexdigest()
        return cls(
            path,
            digest,
            budget_class,
            DataPaths(Path(raw["data"]["c4_path"]), Path(raw["data"]["tokenizer_path"])),
            HardwareContract(int(raw["hardware"]["gpu_count"]), str(raw["hardware"]["selection"])),
            _freeze(raw),
        )
