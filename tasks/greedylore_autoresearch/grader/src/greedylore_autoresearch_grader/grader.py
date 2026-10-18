"""Packaged CORAL grader for the fixed GreedyLoRE protocol."""

from __future__ import annotations

from pathlib import Path

from coral.grader import TaskGrader
from coral.types import Score, ScoreBundle

from .contract import EvalContract
from .efficiency import build_timing_command
from .quality import build_quality_command
from .results import geometric_mean_speedup, quality_gate


def score_results(
    mode, quality, timing, reference, *, contract_hash, comparable, evidence_paths,
    max_relative_eval_loss=1.06,
):
    scores = {}
    quality_ok = True
    workload_labels = {
        "llama60m_adamw": "llama60m",
        "llama130m_muon": "llama130m",
    }
    if mode == "research":
        for workload, label in workload_labels.items():
            candidate_loss = float(quality[workload]["final_eval_loss"])
            dense_loss = float(reference["quality"]["workloads"][workload]["final_eval_loss"])
            ratio = candidate_loss / dense_loss
            passed = quality_gate(candidate_loss, dense_loss, max_relative=max_relative_eval_loss)
            quality_ok = quality_ok and passed
            scores[f"{label}_eval_loss"] = Score(candidate_loss, f"{label}_eval_loss")
            scores[f"{label}_loss_ratio"] = Score(ratio, f"{label}_loss_ratio")
            scores[f"{label}_quality_pass"] = Score(float(passed), f"{label}_quality_pass")
        scores["quality_gate"] = Score(float(quality_ok), "quality_gate")
    else:
        for label in workload_labels.values():
            scores[f"{label}_eval_loss"] = Score(None, f"{label}_eval_loss", "inherited (provisional)")
            scores[f"{label}_loss_ratio"] = Score(None, f"{label}_loss_ratio", "inherited (provisional)")
            scores[f"{label}_quality_pass"] = Score(None, f"{label}_quality_pass", "inherited (provisional)")
        scores["quality_gate"] = Score(None, "quality_gate", "inherited (provisional)")

    speedups = {}
    for cell_name, label in (("130m_adamw_bf16", "bf16"), ("130m_adamw_fp32", "fp32")):
        cell = timing[cell_name]
        scores[f"{label}_timing_mean"] = Score(cell["mean"], f"{label}_timing_mean")
        scores[f"{label}_timing_spread"] = Score(cell["relative_spread"], f"{label}_timing_spread")
        value = None
        if comparable:
            dense = float(reference["efficiency"]["cells"][cell_name]["mean_iteration_seconds"])
            value = dense / float(cell["mean"])
            speedups[label] = value
        scores[f"{label}_speedup"] = Score(value, f"{label}_speedup")
    efficiency = geometric_mean_speedup(speedups) if comparable else None
    scores["efficiency_score"] = Score(efficiency, "efficiency_score")
    if not comparable:
        aggregated = None
        failure_class = "infra"
    elif mode == "research" and not quality_ok:
        aggregated = 0.0
        failure_class = "candidate"
    else:
        aggregated = efficiency
        failure_class = None
    return ScoreBundle(
        scores=scores, aggregated=aggregated,
        metadata={
            "failure_class": failure_class, "contract_hash": contract_hash,
            "comparability": "comparable" if comparable else "incomparable_hardware",
            "evidence_paths": list(evidence_paths),
        },
    )


class CandidateError(RuntimeError):
    pass


class Grader(TaskGrader):
    async def grade(self, codebase_path, tasks, **kwargs):
        from .runtime import InfraFailure, Lifecycle

        self._journal = None
        try:
            expected = getattr(self, "host_parent_identity", None)
            if expected is None:
                raise InfraFailure("trusted grader parent daemon identity is unavailable")
            with Lifecycle(expected_parent=expected) as lifecycle:
                self._lifecycle = lifecycle
                try:
                    return await super().grade(codebase_path, tasks, **kwargs)
                finally:
                    self._lifecycle = None
        except InfraFailure as exc:
            return self._failure(str(exc), "infra")

    def _check_cancelled(self):
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is not None:
            lifecycle.check()

    def _resource_paths(self):
        journal = getattr(self, "_journal", None)
        if journal is None:
            return []
        return [str(self.eval_logs_worktree_path(journal.root / name))
                for name in ("gpu_resources.jsonl", "gpu_resources.json")
                if (journal.root / name).exists()]

    @staticmethod
    def workloads_for(mode: str) -> tuple[str, ...]:
        if mode == "research":
            return ("quality", "efficiency")
        if mode == "engineering":
            return ("efficiency",)
        raise ValueError(f"unsupported submission mode: {mode}")

    def _failure(self, message: str, failure_class: str) -> ScoreBundle:
        journal = getattr(self, "_journal", None)
        if journal is not None:
            journal.event(getattr(self, "_stage", "setup"), "failed", reason=message)
        return ScoreBundle(
            scores={"eval": Score(None, "eval", message)}, aggregated=None,
            feedback=message, metadata={"failure_class": failure_class,
                                        "evidence_paths": self._resource_paths()},
        )

    def _load_references(self, contract_hash: str):
        import json

        root = Path(self.private_dir) / "dense_references"
        try:
            quality = json.loads((root / "quality.json").read_text())
            efficiency = json.loads((root / "efficiency.json").read_text())
        except (OSError, ValueError) as exc:
            from .gpu import InfraFailure
            raise InfraFailure(f"dense reference is unavailable: {exc}") from exc
        from .gpu import InfraFailure
        if quality.get("contract_hash") != contract_hash or efficiency.get("contract_hash") != contract_hash:
            raise InfraFailure("dense reference contract hash mismatch")
        import math
        quality_names = {"llama60m_adamw", "llama130m_muon"}
        efficiency_names = {"130m_adamw_bf16", "130m_adamw_fp32"}
        try:
            quality_workloads = quality["workloads"]
            efficiency_cells = efficiency["cells"]
            environment = efficiency["environment"]
            valid = (
                set(quality_workloads) == quality_names
                and set(efficiency_cells) == efficiency_names
                and all(
                    math.isfinite(float(quality_workloads[name]["final_eval_loss"]))
                    and float(quality_workloads[name]["final_eval_loss"]) > 0
                    for name in quality_names
                )
                and all(
                    math.isfinite(float(efficiency_cells[name]["mean_iteration_seconds"]))
                    and float(efficiency_cells[name]["mean_iteration_seconds"]) > 0
                    for name in efficiency_names
                )
                and len(environment["gpu_uuids"]) == 4
                and len(environment["gpu_names"]) == 4
                and len(environment["numa_nodes"]) == 4
                and bool(environment["topology_fingerprint"])
                and isinstance(environment["nccl_env"], dict)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise InfraFailure("dense reference schema or values are invalid")
        return {"quality": quality, "efficiency": efficiency}

    def _run(self, spec):
        from .process import run_process_group
        reservation = getattr(self, "_timing_reservation", None)
        return run_process_group(
            spec, check_cancelled=self._check_cancelled,
            monitor=reservation.check_exclusive if reservation is not None else None,
            monitor_interval=self._resource_options.monitor_interval,
        )

    def _completed_json(self, spec, output):
        import json

        result = self._run(spec)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        (spec.log_dir / "process.json").write_text(json.dumps({
            "command": list(spec.command),
            "environment": dict(getattr(result, "env", {})),
            "returncode": result.returncode,
            "started_at": getattr(result, "started_at", None),
            "finished_at": getattr(result, "finished_at", None),
            "timed_out": result.timed_out,
            "cleanup_signals": list(getattr(result, "cleanup_signals", ())),
            "stdout_log": getattr(result, "stdout_log", None),
            "stderr_log": getattr(result, "stderr_log", None),
            "infra_failure": getattr(result, "infra_failure", None),
        }, indent=2) + "\n")
        if getattr(result, "infra_failure", None):
            from .gpu import InfraFailure
            raise InfraFailure(result.infra_failure)
        if result.timed_out or result.returncode != 0:
            raise CandidateError(
                f"candidate process failed with return code {result.returncode}; "
                f"stderr: {result.stderr_tail}"
            )
        try:
            payload = json.loads((output / "all_results.json").read_text())
        except (OSError, ValueError) as exc:
            raise CandidateError(f"candidate result is unavailable: {exc}") from exc
        if payload.get("status") != "completed":
            raise CandidateError("candidate result is not completed")
        return payload

    def _formal(self, contract, mode, reference):
        import hashlib
        import json
        import math

        from .gpu import (
            EfficiencyEnvironment,
            GpuInventory,
            IdleGpuPolicy,
            InfraFailure,
            gpu_topology_fingerprint,
            reserve_idle_gpus,
            validate_efficiency_comparability,
        )
        from .resources import ResourceJournal, ResourceOptions, validate_monitoring_reference
        from .results import TimingRepetition, aggregate_timing_cell

        idle = contract.raw["hardware"]["idle_gpu"]
        policy = IdleGpuPolicy(
            int(idle["compute_processes"]), int(idle["max_memory_used_mib"]),
            int(idle["max_utilization_percent"]),
        )
        options = self._resource_options = ResourceOptions.from_args(self.args)
        candidate = Path(self.codebase_path)
        evidence_root = self.eval_logs_dir
        journal = self._journal = ResourceJournal(evidence_root, options)
        self._stage = "setup"
        journal.event("setup", "configured")
        validate_monitoring_reference(reference["efficiency"], options,
                                      Path(self.private_dir) / "dense_references")
        quality_results = {}
        timing_repetitions = {name: [] for name in contract.efficiency.cells}
        evidence_paths = []
        def acquire(stage):
            self._stage = stage
            return reserve_idle_gpus(
                contract.hardware.gpu_count, options.lock_dir, policy,
                allow_other_users=stage == "quality" and options.pretraining_allow_other_users,
                wait_timeout=options.wait_timeout, poll_interval=options.poll_interval,
                check_cancelled=self._check_cancelled,
                on_event=lambda state, **details: journal.event(stage, state, **details),
                inventory_provider=GpuInventory.from_nvidia_smi,
                source_role="engineer" if mode == "engineering" else "researcher",
                execution_group=str(Path(self.private_dir).parent.resolve()),
                execution_limit=self.config.parallel.max_workers,
            )

        if mode == "research":
            with acquire("quality") as reservation:
                journal.event("quality", "running", gpu_uuids=reservation.uuids)
                for name, profile in contract.quality.items():
                    self._check_cancelled()
                    output = evidence_root / "quality" / name
                    output.mkdir(parents=True, exist_ok=False)
                    spec = build_quality_command(profile, candidate, output, reservation.indices)
                    payload = self._completed_json(spec, output)
                    if (payload.get("contract_hash") != contract.contract_hash
                            or payload.get("world_size") != contract.hardware.gpu_count
                            or payload.get("update_step") != profile["stop_after_steps"]
                            or not math.isfinite(float(payload.get("final_eval_loss", math.nan)))):
                        raise CandidateError(f"invalid quality result for {name}")
                    quality_results[name] = payload
                    evidence_paths.extend((
                        str(self.eval_logs_worktree_path(output)),
                        str(self.eval_logs_worktree_path(spec.log_dir)),
                    ))

        with acquire("timing") as reservation:
            nccl_env = {
                "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0",
                "NCCL_DEBUG": "INFO",
            }
            numa = tuple(device.numa_node for device in reservation.devices)
            topology = gpu_topology_fingerprint(
                reservation.uuids, reservation.topology, nccl_env, numa
            )
            hardware = {
                "gpu_uuids": reservation.uuids,
                "topology_fingerprint": topology, "numa_nodes": numa,
            }
            current_env = EfficiencyEnvironment(
                reservation.uuids, tuple(device.name for device in reservation.devices),
                topology, numa, nccl_env,
            )
            env = reference["efficiency"]["environment"]
            reference_env = EfficiencyEnvironment(
                tuple(env["gpu_uuids"]), tuple(env["gpu_names"]), env["topology_fingerprint"],
                tuple(env["numa_nodes"]), dict(env["nccl_env"]),
            )
            if not validate_efficiency_comparability(reference_env, current_env).valid:
                raise InfraFailure("incomparable_hardware before timing")
            self._timing_reservation = reservation
            try:
                for repetition, row in enumerate(contract.efficiency.run_order):
                    for name in row:
                        self._check_cancelled()
                        reservation.check_exclusive()
                        journal.event("timing", "running", gpu_uuids=reservation.uuids,
                                      repetition=repetition, cell=name)
                        output = evidence_root / "efficiency" / f"r{repetition + 1}-{name}"
                        cell = contract.efficiency.cells[name]
                        spec = build_timing_command(
                            cell, repetition, candidate, output, reservation.indices, hardware
                        )
                        payload = self._completed_json(spec, output)
                        reservation.check_exclusive()
                        if (payload.get("contract_hash") != contract.contract_hash
                                or payload.get("measured_steps") != cell["measured_iterations"]
                                or payload.get("strict_blocking_communication") is not False
                                or payload.get("bucket_layout_recorded") is not True):
                            raise CandidateError(f"invalid timing result for {name}")
                        bucket = hashlib.sha256(json.dumps(
                            payload.get("observed_bucket_bytes"), sort_keys=True
                        ).encode()).hexdigest()
                        timing_repetitions[name].append(TimingRepetition(
                            float(payload["mean_iteration_seconds"]),
                            payload["contract_hash"], bucket, payload["topology_fingerprint"],
                        ))
                        evidence_paths.extend((
                            str(self.eval_logs_worktree_path(output)),
                            str(self.eval_logs_worktree_path(spec.log_dir)),
                        ))
            finally:
                self._timing_reservation = None

        timing = {}
        for name, repetitions in timing_repetitions.items():
            summary = aggregate_timing_cell(repetitions)
            timing[name] = {"mean": summary.mean, "relative_spread": summary.relative_spread}
        comparable = validate_efficiency_comparability(reference_env, current_env).valid
        return score_results(
            mode, quality_results, timing, reference,
            contract_hash=contract.contract_hash, comparable=comparable,
            evidence_paths=evidence_paths + self._resource_paths(),
            max_relative_eval_loss=contract.raw["quality"]["max_relative_eval_loss"],
        )

    def evaluate(self) -> ScoreBundle:
        from .gpu import InfraFailure

        validation = bool(self.tasks and self.tasks[0].metadata.get("coral_validation"))
        try:
            contract = EvalContract.load(
                Path(self.private_dir) / "eval_contract.yaml", validation=validation
            )
            if validation:
                candidate = Path(self.codebase_path)
                output = self.eval_logs_dir / "validation"
                gpus = tuple(range(contract.hardware.gpu_count))
                for profile in contract.quality.values():
                    build_quality_command(profile, candidate, output / profile["name"], gpus)
                for name, cell in contract.efficiency.cells.items():
                    build_timing_command(cell, 0, candidate, output / name, gpus)
                return ScoreBundle(
                    scores={"validation": Score(1.0, "validation")}, aggregated=1.0,
                    metadata={"failure_class": None, "contract_hash": contract.contract_hash},
                )
            metadata = self.tasks[0].metadata if self.tasks else {}
            mode = str(metadata.get("mode", ""))
            self.workloads_for(mode)
            if metadata.get("contract_hash") != contract.contract_hash:
                raise CandidateError("submission contract hash mismatch")
            if mode == "engineering" and not metadata.get("quality_anchor"):
                raise CandidateError("engineering submission requires quality_anchor")
            reference = self._load_references(contract.contract_hash)
            return self._formal(contract, mode, reference)
        except InfraFailure as exc:
            return self._failure(str(exc), "infra")
        except CandidateError as exc:
            return self._failure(str(exc), "candidate")
