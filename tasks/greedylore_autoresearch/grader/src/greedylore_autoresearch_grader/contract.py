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
        # Existing private contracts remain readable without changing their hashes.
        if path == "contract.efficiency":
            missing.discard("max_measured_iterations")
        if unknown or missing:
            raise ContractError(f"{path} has unknown/missing keys: {sorted(unknown | missing)}")
        for key in schema:
            if key not in value:
                continue
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
    if type(raw['schema_version']) is not int or raw['schema_version'] not in (1, 2, 3, 4):
        raise ContractError('unsupported contract schema_version')
    _positive_int(raw["hardware"]["gpu_count"], "hardware.gpu_count")
    _positive_int(raw["data"]["dataloader_workers"], "data.dataloader_workers")
    _positive_int(raw["training"]["seed"], "training.seed")
    _positive_int(raw["training"]["sequence_length"], "training.sequence_length")
    _positive_int(raw["efficiency"]["repetitions"], "efficiency.repetitions")
    if raw["efficiency"]["repetitions"] != 2:
        raise ContractError("efficiency.repetitions must equal 2")
    for field in ("warmup_iterations", "measured_iterations", "bucket_cap_mb"):
        _positive_int(raw["efficiency"][field], f"efficiency.{field}")
    maximum = raw["efficiency"].get("max_measured_iterations", 800)
    _positive_int(maximum, "efficiency.max_measured_iterations")
    if raw["efficiency"]["measured_iterations"] > maximum:
        raise ContractError("efficiency.measured_iterations exceeds maximum")
    _finite_positive(raw["quality"]["max_relative_eval_loss"], "quality.max_relative_eval_loss")
    cells = raw["efficiency"]["cells"]
    size = '350m' if raw['schema_version'] >= 4 else '130m'
    if raw['efficiency']['model'] != f'llama{size}':
        raise ContractError('efficiency.model does not match contract version')
    if not isinstance(cells, dict) or set(cells) != {f"{size}_adamw_bf16", f"{size}_adamw_fp32"}:
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
    if raw['schema_version'] >= 2:
        vision = raw['vision']
        train_samples = 50000
        if raw['schema_version'] >= 3:
            _positive_int(vision['validation_per_class'], 'vision.validation_per_class')
            _positive_int(vision['split_seed'], 'vision.split_seed')
            if vision['evaluation_split'] != 'validation' or vision['validation_per_class'] >= 500:
                raise ContractError('vision requires a nonempty fixed training/validation split')
            train_samples -= 100 * vision['validation_per_class']
        for key in ('epochs', 'stop_after_steps', 'per_device_batch_size', 'workers'):
            _positive_int(vision[key], f'vision.{key}')
        for key in ('matrix_learning_rate', 'scalar_learning_rate', 'weight_decay'):
            _finite_positive(vision[key], f'vision.{key}')
        drop = vision['max_accuracy_drop_pp']
        if type(drop) not in (int, float) or not math.isfinite(drop) or not 0 <= drop < 100:
            raise ContractError('vision.max_accuracy_drop_pp must be finite in [0,100)')
        if (vision['name'] != 'resnet18_cifar100_muon' or vision['model'] != 'resnet18'
                or vision['dataset'] != 'cifar100' or vision['precision'] != 'fp32' or vision['optimizer'] != 'muon'
                or vision['stop_after_steps'] != vision['epochs'] * math.ceil(train_samples / (raw['hardware']['gpu_count'] * vision['per_device_batch_size']))):
            raise ContractError('invalid vision workload or complete epoch budget')
        if raw['evaluation'] != {'development': 'llama130m_muon',
                'final_quality': ['llama60m_adamw', vision['name']]}:
            raise ContractError('invalid development/final evaluation stages')


_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "taskdata" / "eval_contract.yaml"
try:
    _SCHEMA = _schema(yaml.safe_load(_TEMPLATE_PATH.read_text()))
    _LEGACY_SCHEMA = _schema(yaml.safe_load(_TEMPLATE_PATH.with_name('eval_contract_v1.yaml').read_text()))
    _V3_SCHEMA = _schema(yaml.safe_load(_TEMPLATE_PATH.with_name('eval_contract_v3.yaml').read_text()))
    _V2_SCHEMA = _schema(yaml.safe_load(_TEMPLATE_PATH.with_name('eval_contract_v2.yaml').read_text()))
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
class EfficiencyPlan:
    cells: Mapping[str, Mapping[str, Any]]
    run_order: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class EvalContract:
    path: Path
    contract_hash: str
    budget_class: str
    data: DataPaths
    hardware: HardwareContract
    raw: Mapping[str, Any]

    @property
    def development_workload(self):
        return self.raw.get('evaluation', {}).get('development', 'llama130m_muon')

    @property
    def final_workloads(self):
        return tuple(self.raw.get('evaluation', {}).get('final_quality', ('llama60m_adamw',)))

    @property
    def utility_version(self):
        return 'mean-log-ppl-ratio-v2' if self.raw['schema_version'] >= 3 else 'worst-ppl-discount-v1'

    @property
    def quality(self) -> Mapping[str, Mapping[str, Any]]:
        common = {
            "contract_hash": self.contract_hash,
            "c4_path": str(self.data.c4_path),
            "tokenizer_path": str(self.data.tokenizer_path),
            "workers": self.raw["data"]["dataloader_workers"],
            "seed": self.raw["training"]["seed"],
            "max_length": self.raw["training"]["sequence_length"],
            "gpu_count": self.hardware.gpu_count,
            "compressor": self.raw["compressor"],
        }
        workloads = {
            name: {"name": name, **dict(workload), **common}
            for name, workload in self.raw["quality"]["workloads"].items()
        }
        if self.raw['schema_version'] >= 2:
            vision = dict(self.raw['vision'])
            workloads[vision['name']] = {**vision, 'contract_hash': self.contract_hash,
                'seed': self.raw['training']['seed'], 'gpu_count': self.hardware.gpu_count,
                'compressor': {'name': self.raw['compressor']['name']}}
        return _freeze(workloads)

    @property
    def efficiency(self) -> EfficiencyPlan:
        raw = self.raw["efficiency"]
        common = {
            key: value for key, value in raw.items()
            if key not in {"cells", "run_order"}
        }
        shared = {
            **common,
            "contract_hash": self.contract_hash,
            "c4_path": str(self.data.c4_path),
            "tokenizer_path": str(self.data.tokenizer_path),
            "workers": self.raw["data"]["dataloader_workers"],
            "seed": self.raw["training"]["seed"],
            "max_length": self.raw["training"]["sequence_length"],
            "gpu_count": self.hardware.gpu_count,
            "compressor": self.raw["compressor"],
        }
        cells = _freeze({
            name: {"name": name, **dict(cell), **shared}
            for name, cell in raw["cells"].items()
        })
        return EfficiencyPlan(cells, tuple(tuple(row) for row in raw["run_order"]))

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
        _check_keys(loaded, {1: _LEGACY_SCHEMA, 2: _V2_SCHEMA, 3: _V3_SCHEMA}.get(loaded.get('schema_version'), _SCHEMA))
        raw = copy.deepcopy(loaded)
        _validate_values(raw)
        for key in ("c4_path", "tokenizer_path"):
            resolved = Path(raw["data"][key]).expanduser().resolve()
            if not resolved.exists():
                raise ContractError(f"data.{key} does not exist: {resolved}")
            raw["data"][key] = str(resolved)
        if raw['schema_version'] >= 2:
            path_vision = Path(raw['vision']['data_path']).expanduser().resolve()
            if not path_vision.is_dir():
                raise ContractError(f'vision.data_path does not exist: {path_vision}')
            raw['vision']['data_path'] = str(path_vision)
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
