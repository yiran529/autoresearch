from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO_ROOT / "tasks" / "greedylore_autoresearch"
CONTRACT_PATH = TASK_ROOT / "taskdata" / "eval_contract.yaml"


def _contract() -> dict[str, object]:
    return yaml.safe_load(CONTRACT_PATH.read_text())


def test_contract_has_reviewed_paths_and_two_efficiency_cells() -> None:
    contract = _contract()

    assert contract["data"]["c4_path"] == "/home/wyr/greedy_lore/c4/c4_en/en"
    assert contract["data"]["tokenizer_path"].endswith(
        "models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    )
    assert list(contract["efficiency"]["cells"]) == [
        "130m_adamw_bf16",
        "130m_adamw_fp32",
    ]
    assert contract["efficiency"]["repetitions"] == 2


def test_contract_freezes_quality_workloads() -> None:
    contract = _contract()
    quality = contract["quality"]

    assert contract["hardware"]["gpu_count"] == 4
    assert contract["training"]["seed"] == 1243
    assert contract["training"]["sequence_length"] == 256
    assert quality["max_relative_eval_loss"] == 1.06
    assert quality["workloads"]["llama60m_adamw"] == {
        "model": "llama60m",
        "optimizer": "adamw",
        "precision": "bf16",
        "per_device_batch_size": 128,
        "scheduler_steps": 8393,
        "stop_after_steps": 4000,
        "learning_rate": 0.0025,
        "activation_checkpointing": True,
    }
    assert quality["workloads"]["llama130m_muon"] == {
        "model": "llama130m",
        "optimizer": "muon",
        "precision": "bf16",
        "per_device_batch_size": 128,
        "scheduler_steps": 20000,
        "stop_after_steps": 2500,
        "matrix_learning_rate": 0.01,
        "scalar_learning_rate": 0.001,
        "activation_checkpointing": True,
    }


def test_contract_freezes_efficiency_protocol() -> None:
    contract = _contract()
    efficiency = contract["efficiency"]

    assert efficiency["per_device_batch_size"] == 1
    assert efficiency["warmup_iterations"] == 101
    assert efficiency["measured_iterations"] == 400
    assert efficiency["run_order"] == [
        ["130m_adamw_bf16", "130m_adamw_fp32"],
        ["130m_adamw_fp32", "130m_adamw_bf16"],
    ]
    assert efficiency["strict_blocking_communication"] is False
    assert efficiency["bucket_cap_mb"] == 1024
    assert efficiency["compression_start_step"] == 100
    assert efficiency["aggregation"] == "geometric_mean_dense_over_candidate"


def test_contract_freezes_compressor_and_idle_gpu_policy() -> None:
    contract = _contract()

    assert contract["compressor"] == {
        "name": "top_subspace",
        "rank": 32,
        "ef_rank": 14,
        "update_gap": 200,
        "min_compression_rate": 1.0,
        "beta_ef": 0.0,
        "error_inherit": 0.0,
        "quality_start_step": 1000,
    }
    assert contract["hardware"]["idle_gpu"] == {
        "compute_processes": 0,
        "max_memory_used_mib": 512,
        "max_utilization_percent": 5,
    }


def test_task_wires_fixed_trusted_validators() -> None:
    task = yaml.safe_load((TASK_ROOT / "task.yaml").read_text())
    assert task["grader"]["args"].items() >= {
        "trusted_submission_validator": "autoresearch.submission:validate_submission",
        "trusted_artifact_validator": "autoresearch.artifacts:validate_artifact",
    }.items()
    assert task["grader"]["args"]["trusted_memory_worker"] == (
        "autoresearch.memory:run_worker"
    )
