from pathlib import Path

import pytest
from greedylore_autoresearch_grader.grader import Grader, score_results

from coral.config import GraderConfig
from coral.types import Task


def test_validation_mode_checks_contract_and_commands_without_gpu(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "eval_contract.yaml").write_bytes(contract.path.read_bytes())
    grader = Grader(GraderConfig())
    grader.codebase_path = str(candidate)
    grader.private_dir = str(private)
    grader.tasks = [Task("validation", "validation", "validation", {"coral_validation": True})]

    bundle = grader.evaluate()

    assert bundle.aggregated == 1.0
    assert bundle.scores["validation"].value == 1.0
    assert bundle.metadata["contract_hash"]
    assert bundle.metadata["failure_class"] is None


def test_formal_mode_plan_keeps_research_and_engineering_distinct():
    assert Grader.workloads_for("research") == ("quality", "efficiency")
    assert Grader.workloads_for("engineering") == ("efficiency",)


def test_exploration_rejects_old_full_budget_dense_reference(contract, tmp_path):
    import json

    import yaml
    from greedylore_autoresearch_grader.contract import EvalContract
    from greedylore_autoresearch_grader.runtime import InfraFailure

    legacy = yaml.safe_load(contract.path.read_text())
    legacy["quality"]["workloads"]["llama60m_adamw"]["stop_after_steps"] = 8393
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    legacy_hash = EvalContract.load(path).contract_hash
    private = tmp_path / "private"
    references = private / "dense_references"
    references.mkdir(parents=True)
    for name in ("quality", "efficiency"):
        (references / f"{name}.json").write_text(json.dumps({"contract_hash": legacy_hash}))
    grader = Grader(GraderConfig())
    grader.private_dir = str(private)
    with pytest.raises(InfraFailure, match="contract hash mismatch"):
        grader._load_references(contract.contract_hash)


@pytest.mark.parametrize("max_relative,passes", [(1.06, True), (1.04, False)])
def test_research_scores_quality_gate_and_comparable_speedup(max_relative, passes):
    reference = {
        "quality": {
            "contract_hash": "contract",
            "workloads": {
                "llama60m_adamw": {"final_eval_loss": 1.0},
                "llama130m_muon": {"final_eval_loss": 2.0},
            },
        },
        "efficiency": {
            "contract_hash": "contract",
            "cells": {
                "130m_adamw_bf16": {"mean_iteration_seconds": 0.4},
                "130m_adamw_fp32": {"mean_iteration_seconds": 0.4},
            },
        },
    }
    quality = {
        "llama60m_adamw": {"final_eval_loss": 1.05, "best_eval_loss": 0.5},
        "llama130m_muon": {"final_eval_loss": 2.1, "best_eval_loss": 0.5},
    }
    timing = {
        "130m_adamw_bf16": {"mean": 0.2, "relative_spread": 0.1},
        "130m_adamw_fp32": {"mean": 0.4, "relative_spread": 0.05},
    }

    bundle = score_results(
        "research", quality, timing, reference,
        contract_hash="contract", comparable=True, evidence_paths=["eval_logs/a"],
        max_relative_eval_loss=max_relative,
    )

    assert bundle.scores["llama60m_loss_ratio"].value == 1.05
    assert bundle.scores["llama60m_eval_loss"].value == 1.05
    assert bundle.scores["llama130m_eval_loss"].value == 2.1
    assert bundle.scores["llama130m_loss_ratio"].value == 1.05
    assert bundle.scores["llama130m_quality_pass"].value == float(passes)
    assert bundle.scores["bf16_speedup"].value == 2.0
    assert bundle.scores["fp32_speedup"].value == 1.0
    assert bundle.scores["quality_gate"].value == float(passes)
    assert bundle.scores["efficiency_score"].value == pytest.approx(2 ** 0.5)
    assert bundle.aggregated == pytest.approx(2 ** 0.5 if passes else 0.0)
    assert bundle.metadata["evidence_paths"] == ["eval_logs/a"]


def test_engineering_score_is_efficiency_only_and_quality_is_inherited():
    reference = {
        "quality": {"contract_hash": "contract", "workloads": {}},
        "efficiency": {
            "contract_hash": "contract",
            "cells": {
                "130m_adamw_bf16": {"mean_iteration_seconds": 0.3},
                "130m_adamw_fp32": {"mean_iteration_seconds": 0.6},
            },
        },
    }
    timing = {
        "130m_adamw_bf16": {"mean": 0.2, "relative_spread": 0.01},
        "130m_adamw_fp32": {"mean": 0.3, "relative_spread": 0.02},
    }

    bundle = score_results(
        "engineering", {}, timing, reference, contract_hash="contract",
        comparable=True, evidence_paths=[],
    )

    assert bundle.scores["quality_gate"].value is None
    assert bundle.scores["quality_gate"].explanation == "inherited (provisional)"
    assert bundle.aggregated == bundle.scores["efficiency_score"].value


def test_incomparable_timing_never_produces_efficiency_score():
    reference = {
        "quality": {
            "contract_hash": "contract",
            "workloads": {
                "llama60m_adamw": {"final_eval_loss": 1.0},
                "llama130m_muon": {"final_eval_loss": 2.0},
            },
        },
        "efficiency": {
            "contract_hash": "contract",
            "cells": {
                "130m_adamw_bf16": {"mean_iteration_seconds": 0.4},
                "130m_adamw_fp32": {"mean_iteration_seconds": 0.4},
            },
        },
    }
    quality = {
        "llama60m_adamw": {"final_eval_loss": 1.0},
        "llama130m_muon": {"final_eval_loss": 2.0},
    }
    timing = {
        name: {"mean": 0.2, "relative_spread": 0.01}
        for name in ("130m_adamw_bf16", "130m_adamw_fp32")
    }

    bundle = score_results(
        "research", quality, timing, reference, contract_hash="contract",
        comparable=False, evidence_paths=[],
    )

    assert bundle.aggregated is None
    assert bundle.scores["efficiency_score"].value is None
    assert bundle.metadata["failure_class"] == "infra"


def test_formal_evaluation_fails_as_infrastructure_before_gpu_when_references_missing(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "eval_contract.yaml").write_bytes(contract.path.read_bytes())
    grader = Grader(GraderConfig())
    grader.codebase_path = str(candidate)
    grader.private_dir = str(private)
    grader.tasks = [Task(
        "formal", "formal", "formal",
        {"mode": "research", "contract_hash": contract.contract_hash},
    )]

    bundle = grader.evaluate()

    assert bundle.aggregated is None
    assert bundle.metadata["failure_class"] == "infra"
    assert "dense reference" in bundle.feedback


@pytest.mark.parametrize("mode", ["research", "engineering"])
@pytest.mark.parametrize("fault", [None, "missing_monitoring", "incomparable", "interference"])
def test_formal_stages_reserve_separately_and_validate_before_scoring(
    contract, tmp_path, monkeypatch, mode, fault,
):
    import json
    from types import SimpleNamespace

    from greedylore_autoresearch_grader.gpu import (
        GpuDevice,
        gpu_topology_fingerprint,
    )
    from greedylore_autoresearch_grader.resources import ALLOCATION_POLICY_VERSION

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "eval_contract.yaml").write_bytes(contract.path.read_bytes())
    references = private / "dense_references"
    references.mkdir()
    devices = tuple(
        GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4)
    )
    nccl = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0", "NCCL_DEBUG": "INFO"}
    topology = gpu_topology_fingerprint(
        tuple(device.uuid for device in devices), "topology", nccl, (0, 0, 0, 0)
    )
    (references / "quality.json").write_text(json.dumps({
        "contract_hash": contract.contract_hash,
        "workloads": {
            "llama60m_adamw": {"final_eval_loss": 1.0},
            "llama130m_muon": {"final_eval_loss": 2.0},
        },
    }))
    monitor_policy = {
        "allocation_policy_version": ALLOCATION_POLICY_VERSION,
        "interval_seconds": 1, "evidence_path": "monitoring.json",
    }
    environment = {
        "gpu_uuids": [device.uuid for device in devices],
        "gpu_names": [device.name for device in devices],
        "topology_fingerprint": topology, "numa_nodes": [0, 0, 0, 0],
        "nccl_env": nccl,
    }
    (references / "monitoring.json").write_text(json.dumps({
        "kind": "monitored_reference", "contract_hash": contract.contract_hash,
        "environment": environment, "allocation_policy_version": ALLOCATION_POLICY_VERSION,
        "interval_seconds": 1,
    }))
    (references / "efficiency.json").write_text(json.dumps({
        "contract_hash": contract.contract_hash,
        "environment": environment,
        "monitoring": {} if fault == "missing_monitoring" else monitor_policy,
        "cells": {
            "130m_adamw_bf16": {"mean_iteration_seconds": 0.4},
            "130m_adamw_fp32": {"mean_iteration_seconds": 0.4},
        },
    }))

    reservations = []
    held = []

    class Reservation:
        def __init__(self, stage, on_event):
            self.stage, self.on_event = stage, on_event
            self.indices = (4, 5, 6, 7) if stage == "quality" else (0, 1, 2, 3)
            self.devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0)
                                 for i in self.indices)
            self.uuids = tuple(d.uuid for d in self.devices)
            self.topology = "wrong-topology" if fault == "incomparable" else "topology"
            assert not held
            held.append(self)
            reservations.append(self)
            on_event("reserved", gpu_indices=self.indices, gpu_uuids=self.uuids)

        def __enter__(self):
            return self
        def __exit__(self, *_args):
            held.remove(self)
            self.on_event("released", gpu_uuids=self.uuids)

        def check_exclusive(self, _pid=None):
            assert self.stage == "timing"

    def reserve(_count, lock_dir, _policy, **kwargs):
        assert Path(lock_dir) == tmp_path / "shared-locks"
        assert kwargs["wait_timeout"] == 86400
        stage = "quality" if kwargs["allow_other_users"] else "timing"
        return Reservation(stage, kwargs["on_event"])

    monkeypatch.setattr(
        "greedylore_autoresearch_grader.gpu.reserve_idle_gpus",
        reserve,
    )

    class ArtifactGrader(Grader):
        commands = []
        def _run(self, spec):
            assert held
            assert spec.env["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, held[0].indices))
            self.commands.append(spec.argv)
            output = Path(spec.argv[spec.argv.index("--output_dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            if "c4/run_llama_pretraining.py" in spec.argv:
                is_60m = "c4/configs/llama_60m_table_iv.json" in spec.argv
                payload = {
                    "status": "completed", "contract_hash": contract.contract_hash,
                    "world_size": 4, "update_step": 4000 if is_60m else 2500,
                    "final_eval_loss": 1.0 if is_60m else 2.0,
                }
            else:
                payload = {
                    "status": "completed", "contract_hash": contract.contract_hash,
                    "measured_steps": 400, "strict_blocking_communication": False,
                    "bucket_layout_recorded": True, "observed_bucket_bytes": [[100]],
                    "topology_fingerprint": topology, "mean_iteration_seconds": 0.2,
                }
            (output / "all_results.json").write_text(json.dumps(payload))
            return SimpleNamespace(
                timed_out=False, returncode=0, stderr_tail="",
                infra_failure="timing interference" if fault == "interference"
                and "c4/table_v_timing.py" in spec.argv else None,
            )

    grader = ArtifactGrader(GraderConfig(args={
        "pretraining_allow_other_users": True, "gpu_lock_dir": str(tmp_path / "shared-locks"),
    }))
    grader.codebase_path = str(candidate)
    grader.private_dir = str(private)
    grader.tasks = [Task(
        "formal", "formal", "formal",
        {"mode": mode, "contract_hash": contract.contract_hash, "quality_anchor": "research-sha"},
    )]

    bundle = grader.evaluate()

    assert not held
    if fault is not None:
        assert bundle.aggregated is None
        assert bundle.metadata["failure_class"] == "infra"
        if fault in {"missing_monitoring", "incomparable"}:
            assert not any("c4/table_v_timing.py" in command for command in grader.commands)
        if fault == "interference":
            assert "interference" in bundle.feedback
        return
    assert bundle.aggregated == pytest.approx(2.0)
    assert [r.stage for r in reservations] == (["quality", "timing"] if mode == "research" else ["timing"])
    assert len(grader.commands) == (6 if mode == "research" else 4)
    assert sum("c4/run_llama_pretraining.py" in command for command in grader.commands) == (2 if mode == "research" else 0)
    assert sum("c4/table_v_timing.py" in command for command in grader.commands) == 4
    assert bundle.metadata["failure_class"] is None
    assert len(bundle.metadata["evidence_paths"]) == (14 if mode == "research" else 10)
    assert any(path.endswith("gpu_resources.jsonl") for path in bundle.metadata["evidence_paths"])


def test_invalid_dense_reference_is_infrastructure_failure_before_gpu(contract, tmp_path, monkeypatch):
    import json

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "eval_contract.yaml").write_bytes(contract.path.read_bytes())
    references = private / "dense_references"
    references.mkdir()
    (references / "quality.json").write_text(json.dumps({
        "contract_hash": contract.contract_hash, "workloads": {},
    }))
    (references / "efficiency.json").write_text(json.dumps({
        "contract_hash": contract.contract_hash, "cells": {}, "environment": {},
    }))
    monkeypatch.setattr(
        "greedylore_autoresearch_grader.gpu.reserve_idle_gpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU touched")),
    )
    grader = Grader(GraderConfig())
    grader.codebase_path = str(candidate)
    grader.private_dir = str(private)
    grader.tasks = [Task(
        "formal", "formal", "formal",
        {"mode": "research", "contract_hash": contract.contract_hash},
    )]

    bundle = grader.evaluate()

    assert bundle.aggregated is None
    assert bundle.metadata["failure_class"] == "infra"
    assert "dense reference" in bundle.feedback
