"""Packaged CORAL grader for the fixed GreedyLoRE protocol."""

from __future__ import annotations

from pathlib import Path

from coral.grader import TaskGrader
from coral.types import Score, ScoreBundle

from .contract import EvalContract
from .efficiency import build_timing_command, timing_cell_labels
from .evaluation_context import quality_policies
from .quality import build_quality_command
from .results import geometric_mean_speedup, quality_gate


def score_results(
    mode, quality, timing, reference, *, contract_hash, comparable, evidence_paths,
    max_relative_eval_loss=1.05, hardware_comparison=None,
    utility_reference=None, inherited_quality=None,
    timing_protocol=None, performance_eligible=True, ineligibility_reasons=(),
    quality_profiles=None,
    utility_version='mean-log-ppl-ratio-v2',
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
        if 'resnet18_cifar100_muon' in (quality_profiles or {}):
            import math
            name = 'resnet18_cifar100_muon'
            from .vision import vision_metric_fields
            accuracy_key = vision_metric_fields(quality_profiles[name])[0]
            accuracy = quality[name][accuracy_key]
            dense_result = reference['quality']['workloads'][name]
            dense_accuracy = dense_result[accuracy_key]
            split_matches = not (quality_profiles[name].get('evaluation_split') == 'validation' and (
                not quality[name].get('split_sha256') or quality[name]['split_sha256'] != dense_result.get('split_sha256')))
            if not split_matches:
                ineligibility_reasons = [*ineligibility_reasons, 'validation_split_mismatch']
            if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
                   for value in (accuracy, dense_accuracy)):
                raise ValueError('invalid final classification accuracy')
            drop = dense_accuracy - accuracy
            passed = split_matches and drop <= quality_profiles[name]['max_accuracy_drop_pp'] + 1e-12
            quality_ok = quality_ok and passed
            score_name = 'resnet18_validation_accuracy' if 'validation' in accuracy_key else 'resnet18_test_accuracy'
            scores[score_name] = Score(accuracy, score_name)
            scores['resnet18_accuracy_drop_pp'] = Score(drop, 'resnet18_accuracy_drop_pp')
            scores['resnet18_quality_pass'] = Score(float(passed), 'resnet18_quality_pass')
        scores["quality_gate"] = Score(float(quality_ok), "quality_gate")
    else:
        for label in workload_labels.values():
            scores[f"{label}_eval_loss"] = Score(None, f"{label}_eval_loss", "inherited (provisional)")
            scores[f"{label}_loss_ratio"] = Score(None, f"{label}_loss_ratio", "inherited (provisional)")
            scores[f"{label}_quality_pass"] = Score(None, f"{label}_quality_pass", "inherited (provisional)")
        scores["quality_gate"] = Score(None, "quality_gate", "inherited (provisional)")

    speedups = {}
    for cell_name, label in timing_cell_labels(timing).items():
        cell = timing[cell_name]
        scores[f"{label}_timing_mean"] = Score(cell["mean"], f"{label}_timing_mean")
        scores[f"{label}_timing_spread"] = Score(cell["relative_spread"], f"{label}_timing_spread")
        value = None
        if comparable and (timing_protocol or {}).get("reference_window_matches", True):
            dense = float(reference["efficiency"]["cells"][cell_name]["mean_iteration_seconds"])
            value = dense / float(cell["mean"])
            speedups[label] = value
        scores[f"{label}_speedup"] = Score(value, f"{label}_speedup")
    efficiency = geometric_mean_speedup(speedups) if speedups else None
    scores["efficiency_score"] = Score(efficiency, "efficiency_score")
    if not comparable:
        aggregated = None
        failure_class = "infra"
    elif not performance_eligible:
        aggregated = 0.0
        failure_class = "candidate"
    elif mode == "research" and not quality_ok:
        aggregated = 0.0
        failure_class = "candidate"
    else:
        aggregated = efficiency
        failure_class = None
    from .utility import advisory_utility

    utility_reason = None if comparable else "incomparable_hardware"
    if not performance_eligible:
        utility_reason = "timing_evidence_unconfirmed"
    if utility_reference and timing_protocol:
        baseline_steps = (utility_reference.get("timing_protocol") or {}).get("measured_iterations", 400)
        if baseline_steps != timing_protocol["measured_iterations"]:
            utility_reason = "reference_window_mismatch"
    utility = advisory_utility(
        {name: payload["final_eval_loss"] for name, payload in quality.items() if 'final_eval_loss' in payload}
        if mode == "research" else (inherited_quality or {}),
        {name: cell["mean"] for name, cell in timing.items()}, utility_reference,
        unavailable_reason=utility_reason, version=utility_version,
    )
    utility["quality_source"] = "measured" if mode == "research" else "inherited (provisional)"
    scores["advisory_utility"] = Score(utility["value"], "advisory_utility", "optional reference; never a gate")
    return ScoreBundle(
        scores=scores, aggregated=aggregated,
        feedback=("Warning [cross_topology]: " + " ".join(hardware_comparison["warnings"])
                  if hardware_comparison and hardware_comparison["warnings"] else ""),
        metadata={
            "evaluation_policy": {"version": "quality-gated-efficiency-v2",
                                  "max_relative_eval_loss": max_relative_eval_loss,
                                  **({"workload_policies": quality_policies(
                                      quality_profiles, max_relative_eval_loss)}
                                     if 'resnet18_cifar100_muon' in (quality_profiles or {}) else {})},
            "utility": utility,
            "timing_protocol": timing_protocol,
            "performance_eligible": bool(comparable and performance_eligible and quality_ok),
            "ineligibility_reasons": [*ineligibility_reasons,
                                      *([] if comparable else ["incomparable_hardware"]),
                                      *([] if quality_ok else ["quality_gate_failed"])],
            "failure_class": failure_class, "contract_hash": contract_hash,
            "comparability": (hardware_comparison["status"] if hardware_comparison
                              else "comparable" if comparable else "incomparable_hardware"),
            **({"hardware_comparison": hardware_comparison} if hardware_comparison else {}),
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
                for name in ("gpu_resources.jsonl", "gpu_resources.json", "evaluation_context.json")
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
        comparison = getattr(self, "_hardware_comparison", None)
        if comparison and comparison["warnings"]:
            message += "\nWarning [" + comparison["status"] + "]: " + " ".join(comparison["warnings"])
        return ScoreBundle(
            scores={"eval": Score(None, "eval", message)}, aggregated=None,
            feedback=message, metadata={"failure_class": failure_class,
                                        **({"hardware_comparison": comparison,
                                            "comparability": comparison["status"]}
                                           if comparison else {}),
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
        contract_path = Path(self.private_dir) / 'eval_contract.yaml'
        loss_fields = {name: 'final_eval_loss' for name in quality_names}
        efficiency_names = {"130m_adamw_bf16", "130m_adamw_fp32"}
        if contract_path.exists():
            contract = EvalContract.load(contract_path)
            quality_names = set(contract.quality)
            efficiency_names = set(contract.efficiency.cells)
            if 'resnet18_cifar100_muon' in quality_names:
                from .vision import vision_metric_fields
                loss_fields['resnet18_cifar100_muon'] = vision_metric_fields(contract.quality['resnet18_cifar100_muon'])[1]
        try:
            quality_workloads = quality["workloads"]
            efficiency_cells = efficiency["cells"]
            environment = efficiency["environment"]
            valid = (
                set(quality_workloads) == quality_names
                and set(efficiency_cells) == efficiency_names
                and all(
                    math.isfinite(float(quality_workloads[name][loss_fields[name]]))
                    and float(quality_workloads[name][loss_fields[name]]) > 0
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
        if 'resnet18_cifar100_muon' in quality_names:
            from .vision import validate_vision_result
            dense_vision = quality_workloads['resnet18_cifar100_muon']
            try:
                validate_vision_result(dense_vision, EvalContract.load(contract_path).quality['resnet18_cifar100_muon'])
                if dense_vision.get('compressor') != 'none':
                    raise ValueError('reference must have compression disabled')
            except (KeyError, TypeError, ValueError) as exc:
                raise InfraFailure(f'invalid dense vision reference: {exc}') from exc
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
            "failure_phase": getattr(result, "failure_phase", None),
        }, indent=2) + "\n")
        if getattr(result, "infra_failure", None):
            from .gpu import InfraFailure
            failure = InfraFailure(result.infra_failure)
            failure.phase = getattr(result, 'failure_phase', None)
            raise failure
        if result.timed_out or result.returncode != 0:
            failure = CandidateError(
                f"candidate process failed with return code {result.returncode}; "
                f"phase={getattr(result, 'failure_phase', None)}; timed_out={result.timed_out}; "
                f"stderr: {result.stderr_tail}"
            )
            failure.phase = getattr(result, 'failure_phase', None)
            raise failure
        try:
            payload = json.loads((output / "all_results.json").read_text())
        except (OSError, ValueError) as exc:
            raise CandidateError(f"candidate result is unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise CandidateError('candidate result must be a JSON object')
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
        from .evaluation_context import publish_context
        from .timing_policy import plan_timing, reference_for_window, validate_cycle_evidence

        declaration = self.tasks[0].metadata.get("timing_declaration", {
            "period_steps": contract.raw["compressor"]["update_gap"],
        })
        timing_protocol = plan_timing(
            declaration, minimum=contract.raw["efficiency"]["measured_iterations"],
            maximum=contract.raw["efficiency"].get("max_measured_iterations", 800),
            override=self.args.get("timing_measured_iterations"),
        )
        reference, window_matches = reference_for_window(
            reference, timing_protocol["measured_iterations"],
            contract.raw["efficiency"]["measured_iterations"],
        )
        timing_protocol["reference_window_matches"] = window_matches
        cycle_evidence = []

        publish_context(contract, reference, evidence_root, mode, timing_protocol)
        evidence_paths = []
        def acquire(stage):
            self._stage = stage
            borrowed = getattr(self, '_staged_reservation', None)
            if borrowed is not None:
                return borrowed()
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
                    checkpoint_options = {}
                    artifacts = None
                    if self.args.get('staged_research') and self.tasks[0].metadata.get('node_id') == 'N000':
                        from autoresearch.runtime_artifacts import RuntimeArtifacts
                        artifacts = RuntimeArtifacts(Path(self.private_dir).parent)
                        artifacts.require_space('N000')
                        checkpoint_root = artifacts.path('N000', f'root/{evidence_root.name}/checkpoints/{name}')
                        checkpoint_options = dict(checkpoint_root=checkpoint_root,
                            checkpoint_interval_steps=self.args.get('research_checkpoint_interval_steps', 1000),
                            training_metrics=output / 'training_metrics.jsonl')
                    spec = build_quality_command(profile, candidate, output, reservation.indices, **checkpoint_options)
                    payload = self._completed_json(spec, output)
                    if profile['model'] == 'resnet18':
                        from .vision import validate_vision_result
                        try:
                            validate_vision_result(payload, profile)
                        except (TypeError, ValueError) as exc:
                            raise CandidateError(f'invalid quality result for {name}: {exc}') from exc
                    elif (payload.get("contract_hash") != contract.contract_hash
                            or payload.get("world_size") != contract.hardware.gpu_count
                            or payload.get("update_step") != profile["stop_after_steps"]
                            or not math.isfinite(float(payload.get("final_eval_loss", math.nan)))):
                        raise CandidateError(f"invalid quality result for {name}")
                    if artifacts is not None:
                        checkpoint_value = payload.get('checkpoint_path')
                        if profile['model'] == 'resnet18' and (payload.get('diagnostic_checkpoint') or {}).get('status') == 'complete':
                            checkpoint_value = str(checkpoint_root / f"step-{payload['update_step']:08d}")
                        if checkpoint_value:
                            checkpoint = Path(checkpoint_value).resolve()
                            if not checkpoint.is_relative_to(checkpoint_root):
                                raise CandidateError('Root checkpoint escapes owned artifact directory')
                            artifacts.publish_checkpoint('N000', checkpoint,
                                evidence_root / 'checkpoints' / name / 'manifest.json')
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
            from dataclasses import asdict

            from .evaluation_context import _plain, publish_hardware_comparison

            comparison = validate_efficiency_comparability(reference_env, current_env)
            hardware_comparison = _plain({
                "status": comparison.status, "allowed": comparison.valid,
                "reason": comparison.reason,
                "baseline": asdict(reference_env), "candidate": asdict(current_env),
                "differences": {
                    field: {"baseline": getattr(reference_env, field),
                            "candidate": getattr(current_env, field)}
                    for field in comparison.differences
                },
                "warnings": comparison.warnings,
            })
            self._hardware_comparison = hardware_comparison
            publish_hardware_comparison(evidence_root, hardware_comparison)
            if not comparison.valid:
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
                        cell = {**contract.efficiency.cells[name],
                                "measured_iterations": timing_protocol["measured_iterations"],
                                "startup_timeout_seconds": self.args.get('timing_startup_timeout_seconds', 600),
                                "code_sha": self.tasks[0].metadata.get('commit_hash')}
                        (evidence_root / 'source_identity.json').write_text(json.dumps({
                            'commit_sha': cell['code_sha'], 'contract_hash': contract.contract_hash,
                            'source': 'host_assignment'}, indent=2) + '\n')
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
                        cycle = validate_cycle_evidence(timing_protocol, payload)
                        cycle_payload = payload.get("timing_cycle")
                        rank_events = cycle_payload.get("rank_event_steps") if isinstance(cycle_payload, dict) else None
                        if not isinstance(rank_events, list) or len(rank_events) != contract.hardware.gpu_count:
                            cycle = {"confirmed": False, "reason": "cycle_rank_count_mismatch"}
                        cycle_evidence.append({"repetition": repetition, "cell": name, **cycle})
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
        comparable = comparison.valid
        timing_protocol["cycle_evidence"] = cycle_evidence
        reasons = sorted({item["reason"] for item in cycle_evidence if not item["confirmed"]})
        if not window_matches:
            reasons.append("reference_window_mismatch")
        from .evaluation_context import publish_timing_protocol

        publish_timing_protocol(evidence_root, timing_protocol)
        from .utility import losses_from_attempt, root_reference

        utility_reference = root_reference(Path(self.private_dir).parent, mode, contract.contract_hash, timing_cells=contract.efficiency.cells)
        inherited_quality = None
        if mode == "engineering":
            from coral.hub.attempts import read_attempts

            anchor = self.tasks[0].metadata.get("quality_anchor")
            matches = [a for a in read_attempts(Path(self.private_dir).parent, include_archived=True)
                       if a.commit_hash == anchor and a.metadata.get("mode") == "research"
                       and a.metadata.get("contract_hash") == contract.contract_hash]
            if len(matches) == 1:
                inherited_quality = losses_from_attempt(matches[0])
        return score_results(
            mode, quality_results, timing, reference,
            contract_hash=contract.contract_hash, comparable=comparable,
            evidence_paths=evidence_paths + self._resource_paths(),
            hardware_comparison=hardware_comparison,
            max_relative_eval_loss=contract.raw["quality"]["max_relative_eval_loss"],
            utility_reference=utility_reference, inherited_quality=inherited_quality,
            timing_protocol=timing_protocol, performance_eligible=not reasons,
            ineligibility_reasons=reasons,
            quality_profiles=contract.quality,
            utility_version=contract.utility_version,
        )

    def evaluate(self) -> ScoreBundle:
        from .gpu import InfraFailure
        from .timing_policy import TimingPolicyError

        self._hardware_comparison = None
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
        except (CandidateError, TimingPolicyError) as exc:
            return self._failure(str(exc), "candidate")
