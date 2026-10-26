"""Test-only subprocess replacements. No model, training, or GPU code is executed."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import yaml

from coral.agent.builtin.codex import CodexRuntime
from coral.grader.task_grader import TaskGrader
from coral.grader.trusted_worker import start_task_worker as _real_start_task_worker
from coral.grader.trusted_writes import process_start_time
from coral.types import Score, ScoreBundle

TASK = Path(__file__).resolve().parents[1]


def root() -> Path:
    return Path(os.environ["CORAL_SIM_ROOT"])


def event(kind: str, **fields) -> None:
    path = root() / "timeline.jsonl"
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(
            json.dumps(
                {
                    "event": kind,
                    "pid": os.getpid(),
                    "pid_start": process_start_time(os.getpid()),
                    "at_ns": time.monotonic_ns(),
                    **fields,
                }
            )
            + "\n"
        )
        stream.flush()


def events() -> list[dict]:
    return read_events(root())


def read_events(output: Path) -> list[dict]:
    path = output / "timeline.jsonl"
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return [json.loads(line) for line in stream if line.strip()]


def wait_for(predicate, label: str, timeout=100):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        failures = [e for e in events() if e["event"] == "agent_error"]
        if failures:
            raise RuntimeError(failures[0]["traceback"])
        time.sleep(0.025)
    raise TimeoutError(f"waiting for {label}; recent events: {events()[-12:]}")


def gate(name: str) -> Path:
    return root() / f"gate-{name}"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def cli(module: str, *args: str, check=True):
    result = subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, timeout=40
    )
    event(
        "command",
        module=module,
        argv=list(args),
        code=result.returncode,
        stdout=result.stdout[-1200:],
        stderr=result.stderr[-2000:],
    )
    if check and result.returncode:
        raise RuntimeError(f"{module} {args}: {result.stderr}\n{result.stdout}")
    return result


class ScriptRuntime(CodexRuntime):
    """Use the real Codex launcher with an offline executable at its boundary."""

    def start(self, worktree_path, coral_md_path, log_dir=None, resume_session_id=None, **kwargs):
        aid = (worktree_path / ".coral_agent_id").read_text().strip()
        handle = super().start(worktree_path, coral_md_path, log_dir=log_dir,
                               resume_session_id=resume_session_id, **kwargs)
        event("runtime_start", agent=aid, session=resume_session_id or f"simulation-{aid}",
              resume=resume_session_id, worktree=str(worktree_path),
              child_pid=handle.process.pid, child_start=process_start_time(handle.process.pid))
        return handle


class SyntheticGrader(TaskGrader):
    def evaluate(self):
        metadata = self.tasks[0].metadata
        coral_dir = Path(self.private_dir).parent
        node, mode = metadata["node_id"], metadata["mode"]
        quality = 0.0 if node == "N002" else 1.0
        marker = Path(self.codebase_path) / "simulation_marker.txt"
        if node != "N000" and not marker.is_file():
            raise RuntimeError("Grader did not receive committed simulation code")
        evidence = {
            "simulated": True,
            "node_id": node,
            "mode": mode,
            "committed_marker": marker.read_text() if marker.exists() else None,
        }
        if (
            os.environ["CORAL_SIM_SCENARIO"] == "faults"
            and node == "N003"
            and not gate("infra-failed").exists()
        ):
            gate("infra-failed").touch()
            event("infra_failure", node=node, commit=metadata["commit_hash"])
            return ScoreBundle(
                scores={"quality_gate": Score(None, "quality_gate")},
                aggregated=None,
                feedback="SIMULATED infrastructure failure.",
                metadata={"simulated": True, "failure_class": "infra"},
            )
        (self.eval_logs_dir / "simulation_result.json").write_text(json.dumps(evidence))
        event(
            "graded",
            node=node,
            mode=mode,
            commit=metadata["commit_hash"],
            quality=quality,
            checkout=str(self.codebase_path),
            run_id=metadata["grader_run_id"],
        )
        from greedylore_autoresearch_grader.grader import score_results
        from greedylore_autoresearch_grader.timing_policy import plan_timing
        from greedylore_autoresearch_grader.utility import losses_from_attempt, root_reference

        from coral.hub.attempts import read_attempts

        if node != "N000" and "timing_declaration" not in metadata:
            raise RuntimeError("trusted submission/daemon dropped the timing declaration")
        protocol = plan_timing(metadata.get("timing_declaration", {"period_steps": 200}),
                               minimum=400, maximum=800)
        protocol.update(reference_window_matches=True, simulated=True)
        seconds = {"N000": 0.10, "N001": 0.12, "N002": 0.08, "N003": 0.09}[node]
        loss_delta = {"N000": 0.0, "N001": 0.04, "N002": 0.3, "N003": 0.03}[node]
        baseline_losses = {"llama60m_adamw": 3.0, "llama130m_muon": 4.0}
        reference = {"quality": {"workloads": {
            name: {"final_eval_loss": loss} for name, loss in baseline_losses.items()
        }}, "efficiency": {"cells": {
            name: {"mean_iteration_seconds": 0.1}
            for name in ("130m_adamw_bf16", "130m_adamw_fp32")
        }}}
        from greedylore_autoresearch_grader.contract import EvalContract
        contract = EvalContract.load(coral_dir / "private/eval_contract.yaml")
        quality_payload = {name: {"final_eval_loss": loss + loss_delta} for name, loss in baseline_losses.items()}
        if contract.raw["schema_version"] >= 2:
            reference["quality"]["workloads"]["resnet18_cifar100_muon"] = {"final_test_accuracy":75.91,"final_test_loss":1.27233}
            quality_payload["resnet18_cifar100_muon"] = {"final_test_accuracy":74.94,"final_test_loss":1.27259}
        if contract.raw['schema_version'] >= 3:
            for row in (reference['quality']['workloads']['resnet18_cifar100_muon'], quality_payload['resnet18_cifar100_muon']):
                row['final_validation_accuracy'] = row.pop('final_test_accuracy')
                row['final_validation_loss'] = row.pop('final_test_loss')
                row['split_sha256'] = 'a'*64
        inherited = None
        if mode == "engineering":
            anchors = [a for a in read_attempts(coral_dir)
                       if a.commit_hash == metadata["quality_anchor"]]
            inherited = losses_from_attempt(anchors[0]) if anchors else None
        bundle = score_results(
            mode,
            quality_payload if mode == "research" else {},
            {name: {"mean": seconds, "relative_spread": 0.01}
             for name in reference["efficiency"]["cells"]},
            reference, contract_hash=metadata["contract_hash"], comparable=True,
            evidence_paths=[], utility_reference=root_reference(coral_dir, mode, metadata["contract_hash"]),
            inherited_quality=inherited, timing_protocol=protocol, quality_profiles=contract.quality, utility_version=contract.utility_version,
        )
        bundle.feedback = "SIMULATED: synthetic inputs to the real scoring code; no training or timing."
        bundle.metadata["simulated"] = True
        return bundle



def workflow_handler(operation, payload, context):
    from itertools import cycle

    from autoresearch.state import handle_workflow

    # Fixed host draws exercise both modes without changing production policy.
    with patch("autoresearch.state.random.random", side_effect=cycle((0.1, 0.8))):
        return handle_workflow(operation, payload, context)


def analysis_runner(prompt, read_roots, schema, scratch):
    from autoresearch.analysis import run_fresh_codex
    return run_fresh_codex(prompt, read_roots, schema, scratch)


def memory_runner(prompt, read_roots, schema, scratch):
    from autoresearch.analysis import run_fresh_codex
    return run_fresh_codex(prompt, read_roots, schema, scratch)



def launch_logged_worker(coral_dir, entrypoint, **kwargs):
    process = _real_start_task_worker(coral_dir, entrypoint, **kwargs)
    event("worker_spawn", child_pid=process.pid, child_start=process_start_time(process.pid))
    return process


def worker_shutdown(signum, _frame):
    # Manager shutdown intentionally sends SIGTERM to a late finalizer. Let the
    # test wrapper record its exit before the process ends, just as on success.
    raise SystemExit(128 + signum)


def run_memory(coral_dir: str):
    from autoresearch.memory import MemoryContext, consolidate_memory

    previous_handler = signal.signal(signal.SIGTERM, worker_shutdown)
    try:
        event("worker_start", worker="memory")
        result = consolidate_memory(
            if_due=True, context=MemoryContext(Path(coral_dir)), runner=memory_runner
        )
        event("memory_done", status=result.status)
        return {"status": result.status}
    finally:
        try:
            event("worker_end", worker="memory")
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def run_finalization(coral_dir: str):
    from autoresearch.lifecycle import run_worker

    previous_handler = signal.signal(signal.SIGTERM, worker_shutdown)
    try:
        event("worker_start", worker="finalization")
        result = run_worker(coral_dir, runner=analysis_runner, now=lambda: time.monotonic() * 10000)
        event("finalization_done", **result)
        return result
    finally:
        try:
            event("worker_end", worker="finalization")
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def declaration(coral_dir, node, mode, base, round_number):
    spec = yaml.safe_load((coral_dir / f"public/notes/nodes/{node}/spec.md").read_text())
    body = {
        "mode": mode,
        "node_id": node,
        "node_type": "seed" if node == "N000" else "research",
        "base_commit": base,
        "quality_anchor": base if mode == "engineering" else None,
        "analysis_refs": spec["analysis_refs"],
        "change_summary": f"SIMULATED {mode} {node} candidate {round_number}",
        "selection_rationale": "Fixed target for deterministic workflow simulation.",
        "semantic_self_check": {
            "status": "not_run",
            "commands": [],
            "limitations": "Simulation; no scientific self-check.",
        },
        "timing": {"period_steps": 200, "requested_steps": 400},
    }
    Path("submission.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
    Path("simulation_marker.txt").write_text(f"SIMULATED {mode} {node} {round_number}\n")


def archive_self_check(node):
    """Exercise real CLI upload, then discard the scratch source immediately."""
    with tempfile.TemporaryDirectory(prefix='coral-sim-self-check-') as directory:
        script, log = Path(directory) / 'check.py', Path(directory) / 'check.log'
        script.write_text("print('SIMULATED self-check attachment; no scientific measurement.')\n")
        log.write_bytes(subprocess.check_output([sys.executable, str(script)]))
        result = cli('autoresearch.cli', 'evidence', 'archive', '--node', node,
                     '--request-id', 'self-check-1', '--files', str(script), str(log))
        reference = json.loads(result.stdout)['path']
    event('supporting_evidence_archived', node=node, reference=reference,
          scratch_removed=not script.exists())
    return reference


def agent(coral_dir: Path, aid: str):
    event("agent_ready", agent=aid)
    if aid.startswith("research-"):
        node = aid.removeprefix("research-")
        if node not in {"N001", "N002", "N003"}:
            wait_for(lambda: gate("stop").exists(), "unused reservation stop")
            return
        spec_path = coral_dir / f"public/notes/nodes/{node}/spec.md"
        if not spec_path.exists():
            from autoresearch.client import request_trusted

            while True:
                planning = request_trusted(coral_dir, 'acquire_research_planning', {})
                if planning['status'] != 'busy':
                    break
                time.sleep(.05)
            event('selection_acquired', node=node)
            handover = json.loads((coral_dir / "public/notes/nodes/N000/handover.json").read_text())
            parent_engineering = handover['sources']['engineering_analysis']['path']
            engineering_text = (coral_dir / 'public/notes' / parent_engineering).read_text()
            if not engineering_text:
                raise RuntimeError('parent Engineering report was not delivered')
            engineering_summary = coral_dir / 'public/notes/engineering_summary.md'
            summary_available = engineering_summary.is_file()
            if summary_available and not engineering_summary.read_text():
                raise RuntimeError('available Engineering Summary is empty')
            state = read_state(coral_dir)
            mode = state['nodes'][node]['research']['research_mode']
            intent_path = root() / f'{aid}-intent.json'
            template = json.loads(Path('.codex/notes/toolkit/templates/research_intent.json').read_text())
            if not Path('.codex/notes/toolkit/README.md').read_text():
                raise RuntimeError('timing/telemetry instructions were not delivered')
            template.update({
                'problem_ref': None if mode == 'EXPLORE' else 'N000:Q1',
                'question': 'SIMULATED mechanism question; no algorithm discovery claim.',
                'mechanism_test': 'SIMULATED matched comparison of a communication state transition.',
                'practical_test': 'SIMULATED quality gate and complete-cycle runtime comparison.',
                'handover_response': 'Exercise independent EXPLORE or linked REFINE without score gating.',
                'engineering_review': f'Inspected {parent_engineering}; synthetic report, no optimization conclusion.',
            })
            intent_path.write_text(json.dumps(template))
            event('handover_read', node=node, engineering_ref=parent_engineering, mode=mode,
                  engineering_summary_read=summary_available)
            cli(
                "autoresearch.cli",
                "node",
                "initialize",
                "--parent",
                "N000",
                "--hypothesis",
                "Simulation only",
                "--selection-rationale",
                "Root fixture",
                "--delta",
                "Synthetic marker",
                "--risk",
                "No scientific test",
                "--falsification-criteria",
                "Workflow assertion fails",
                "--intent-file",
                str(intent_path),
                '--planning-token',
                planning['token'],
                '--overlap-review',
                'Read registered synthetic Specs; this fixture tests lifecycle, not scientific novelty.',
            )
        event("initialized", node=node, agent=aid, head=git(Path.cwd(), "rev-parse", "HEAD"))
        if node == "N001":
            from coral.grader.trusted_writes import TrustedWriteError, request_trusted

            for name, operation, payload in (
                ("reconcile", "reconcile_workflow", {}),
                (
                    "tree",
                    "publish_artifact",
                    {
                        "kind": "tree",
                        "content": "# Forbidden",
                        "node_id": None,
                        "mode": None,
                        "version": None,
                    },
                ),
                (
                    "spec",
                    "publish_artifact",
                    {
                        "kind": "spec",
                        "content": "node_id: N001",
                        "node_id": "N001",
                        "mode": None,
                        "version": None,
                    },
                ),
                (
                    "other_node",
                    "publish_artifact",
                    {
                        "kind": "analysis",
                        "content": "{}",
                        "node_id": "N002",
                        "mode": "research",
                        "version": 99,
                    },
                ),
            ):
                try:
                    request_trusted(coral_dir, operation, payload)
                except TrustedWriteError as exc:
                    event("authority_rejected", check=name, reason=str(exc))
                else:
                    raise RuntimeError(f"unauthorized operation succeeded: {name}")
        if (
            node == "N002"
            and os.environ["CORAL_SIM_SCENARIO"] == "faults"
            and not gate("crashed").exists()
        ):
            gate("crashed").touch()
            event("agent_crash", agent=aid)
            raise SystemExit(17)
        wait_for(lambda: gate("research").exists(), "parallel Research release")
        archive_self_check(node)
        base = yaml.safe_load(spec_path.read_text())["base_commit"]
        declaration(coral_dir, node, "research", base, 1)
        cli("coral.cli", "eval", "-m", f"SIMULATED research {node}", "--no-wait")
        event("submitted", node=node, mode="research", commit=git(Path.cwd(), "rev-parse", "HEAD"))
        if node == "N003" and os.environ["CORAL_SIM_SCENARIO"] == "faults":
            from coral.hub.attempts import read_attempt

            sha = git(Path.cwd(), "rev-parse", "HEAD")

            def failed():
                attempt = read_attempt(coral_dir, sha)
                return attempt if attempt and attempt.status != "pending" else None

            first = wait_for(failed, "injected infrastructure failure")
            if first.status != "crashed" or first.metadata.get("failure_class") != "infra":
                raise RuntimeError("expected the controlled infrastructure failure")
            if (spec_path.parent / "research_analysis_v1.md").exists():
                raise RuntimeError("Research was finalized before its infrastructure retry")
            cli("autoresearch.cli", "retry", sha)
            event(
                "same_sha_retry",
                node=node,
                commit=sha,
                head_after=git(Path.cwd(), "rev-parse", "HEAD"),
            )
        from coral.hub.attempts import read_attempt

        sha = git(Path.cwd(), "rev-parse", "HEAD")

        def terminal():
            attempt = read_attempt(coral_dir, sha)
            return attempt if attempt and attempt.status != "pending" else None

        wait_for(terminal, "terminal Research result before explicit finish")
        cli("autoresearch.cli", "node", "complete", "--mode", "research")
        event("research_finish_requested", node=node, commit=sha)
    else:
        node = re.search(r"You own Engineering for Node `(N[0-9]{3})`", Path("AGENTS.md").read_text()).group(1)
        if node != "N000":
            wait_for(lambda: gate("claims").exists(), "Engineer release")
        foreign = cli("autoresearch.cli", "engineering", "claim", "N999", check=False)
        if foreign.returncode == 0:
            raise RuntimeError("Engineer was allowed to select a foreign Node")
        event("foreign_claim_rejected", agent=aid, node=node)
        cli("autoresearch.cli", "engineering", "claim", node)
        event("claim_won", agent=aid, node=node)
        base = git(Path.cwd(), "rev-parse", "HEAD")
        for number in (1, 2, 3):
            declaration(coral_dir, node, "engineering", base, number)
            cli("coral.cli", "eval", "-m", f"SIMULATED engineering {number}", "--timeout", "30")
            event(
                "submitted",
                node=node,
                mode="engineering",
                commit=git(Path.cwd(), "rev-parse", "HEAD"),
                round=number,
            )
        from coral.hub.attempts import read_attempts
        attachment = archive_self_check(node)
        summary = root() / f"{aid}-final.json"
        shas = [a.commit_hash for a in read_attempts(coral_dir)
                if a.agent_id == aid and a.metadata.get("mode") == "engineering"]
        summary.write_text(json.dumps({
            "summary": "SIMULATED Engineer-authored result; no scientific measurement.",
            "findings": [f"All three scripted candidate results are recorded. Agent-provided self-check: {attachment}/check.log."],
            "limitations": ["No training, real profiling or speedup validation."],
            "attempt_shas": shas,
        }))
        cli("autoresearch.cli", "engineering", "finish", "--analysis-file", str(summary))
        event("finish_requested", agent=aid)
    wait_for(lambda: gate("stop").exists(), "Agent retirement")


def offline_grader_env(coral_dir, *_args, **_kwargs):
    path = coral_dir / "private/grader_venv/bin/python"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    path.chmod(0o755)
    return path


def prepare(*, contract_version=1):
    from autoresearch.bootstrap import bootstrap_root, bootstrap_root_analysis

    from coral.config import CoralConfig
    from coral.grader.daemon import process_pending_once
    from coral.grader.trusted_writes import TrustedWriteService, _atomic_json
    from coral.workspace.project import create_project

    config = yaml.safe_load((TASK / "task.yaml").read_text())
    config["task"]["name"] = "SIMULATED GreedyLoRE workflow"
    config["workspace"] = {
        "repo_path": str(TASK / "seed"),
        "results_dir": str(root()),
        "run_dir": str(root() / "run"),
        "setup": [],
    }
    config["grader"].update(
        entrypoint="simulation_support:SyntheticGrader",
        setup=[],
        timeout=30,
        private=["taskdata/eval_contract.yaml"],
    )
    # This driver explicitly covers the retained pre-staged compatibility protocol.
    config["grader"]["args"]["staged_research"] = False
    config["grader"]["args"]["trusted_workflow_handler"] = "simulation_support:workflow_handler"
    config["grader"]["args"]["trusted_workflow_worker"] = "simulation_support:run_finalization"
    config["grader"]["args"]["trusted_memory_worker"] = "simulation_support:run_memory"
    config["agents"].update(
        runtime="simulation_support:ScriptRuntime",
        model="simulation",
        timeout=0,
        heartbeat_enabled=False,
    )
    config["agents"]["sandbox"] = {"enabled": False}
    config["agents"]["dynamic"]["max_engineers"] = (
        2 if os.environ["CORAL_SIM_SCENARIO"] == "faults" else 1
    )
    for assignment in config["agents"]["assignments"]:
        for field in ("role_file", "staged_role_file"):
            if field in assignment["runtime_options"]:
                assignment["runtime_options"][field] = str(TASK / assignment["runtime_options"][field])
    cfg = CoralConfig.from_dict(config)
    cfg.task_dir = TASK
    with patch("coral.workspace.grader_env.setup_grader_env", offline_grader_env):
        paths = create_project(cfg, config_dir=TASK)
    if contract_version in (1, 2):
        (paths.coral_dir / "private/eval_contract.yaml").write_text((TASK / f"taskdata/eval_contract_v{contract_version}.yaml").read_text())
    seed = git(paths.repo_dir, "rev-parse", "HEAD")
    bootstrap_root(paths.repo_dir, paths.repo_dir / "source_manifest.json")
    a0 = git(paths.repo_dir, "rev-parse", "HEAD")
    _atomic_json(
        paths.coral_dir / "private/root_bootstrap.json",
        {
            "version": 1,
            "source_commit": seed,
            "node_id": "N000",
            "mode": "research",
            "submission_hash": hashlib.sha256(
                (paths.repo_dir / "submission.yaml").read_bytes()
            ).hexdigest(),
        },
    )
    service = TrustedWriteService(paths.coral_dir, cfg)
    service.submit_root_attempt(worktree=paths.repo_dir, commit_hash=a0)
    process_pending_once(paths.coral_dir)
    bootstrap_root_analysis(paths.coral_dir, paths.repo_dir, runner=analysis_runner)
    event("root_bootstrapped", seed=seed, attempt=a0)
    return cfg, paths


def read_state(coral_dir):
    return json.loads((coral_dir / "private/autoresearch_state.json").read_text())


def read_state_memory(coral_dir):
    return json.loads((coral_dir / 'private/memory_state.json').read_text())['nodes']


def controller(manager, paths, errors):
    try:
        wait_for(
            lambda: (
                {e.get("node") for e in events() if e["event"] == "initialized"} >= {"N001", "N002"}
            ),
            "two initialized Research worktrees",
        )
        gate("research").touch()
        event("parallel_barrier_released")

        def research_done():
            state = read_state(paths.coral_dir)
            return all(
                state["nodes"].get(n, {}).get("research", {}).get("status") == "completed"
                for n in ("N001", "N002", "N003")
            )

        wait_for(research_done, "three Research completions")
        engineers = manager.config.agents.dynamic.max_engineers
        wait_for(
            lambda: (
                len(
                    {
                        e["agent"]
                        for e in events()
                        if e["event"] == "agent_ready" and e["agent"].startswith("engineer-")
                    }
                )
                >= engineers
            ),
            "Engineer processes ready for claim barrier",
        )
        gate("claims").touch()
        event("claim_barrier_released")

        def engineering_done():
            state = read_state(paths.coral_dir)
            return all(state["nodes"].get(n, {}).get("engineering_status") == "completed"
                       and state["nodes"][n]["analysis"]["status"] == "completed"
                       for n in ("N000", "N001", "N003"))

        wait_for(engineering_done, "Engineering Analysis and completion")
        wait_for(
            lambda: (paths.coral_dir / "public/notes/engineering_summary.md").exists()
            and (paths.coral_dir / "private/memory_state.json").exists()
            and all(read_state_memory(paths.coral_dir).get(n, {}).get("status") == "completed"
                    for n in ("N000", "N001", "N002", "N003")),
            "Memory publication",
        )
        wait_for(
            lambda: all(
                not (paths.agents_dir / f"research-{n}").exists() for n in ("N001", "N002", "N003")
            ),
            "Research worktree retirement",
        )
        winner = next(e["agent"] for e in events() if e["event"] == "claim_won")
        wait_for(lambda: not (paths.agents_dir / winner).exists(), "Engineer worktree retirement")
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        gate("stop").touch()
        manager._stop_event.set()


def make_report(paths):
    from autoresearch.memory import MemoryContext, consolidate_memory

    from coral.hub.attempts import read_attempts

    state = read_state(paths.coral_dir)
    recorded = events()
    notes = paths.coral_dir / "public/notes"
    before = [
        (notes / name).read_bytes() for name in ("research_summary.md", "engineering_summary.md", "wisdom.md")
    ]
    result = consolidate_memory(
        if_due=True,
        context=MemoryContext(paths.coral_dir),
        runner=lambda *_: (_ for _ in ()).throw(RuntimeError("duplicate Memory")),
    )
    after = [
        (notes / name).read_bytes() for name in ("research_summary.md", "engineering_summary.md", "wisdom.md")
    ]
    starts = [
        e for e in recorded if e["event"] == "runtime_start" and e["agent"] == "research-N002"
    ]
    claims = [e for e in recorded if e["event"] == "claim_won"]
    attempts = read_attempts(paths.coral_dir)
    roots = {"N001", "N002", "N003"}
    report = {
        "simulated": True,
        "run_dir": str(paths.run_dir),
        "completed_research": sorted(
            n for n in roots if state["nodes"][n]["research"]["status"] == "completed"
        ),
        "quality_failed": sorted(
            a.metadata["node_id"]
            for a in attempts
            if a.metadata["mode"] == "research"
            and a.metadata["scores"]["quality_gate"]["value"] == 0
        ),
        "engineering_attempts": sum(a.metadata["mode"] == "engineering" for a in attempts),
        "engineering_completed": sum(
            w["status"] == "completed" for w in state["engineering_workers"].values()
        ),
        "parallel_research_verified": all(
            any(
                e["event"] == "initialized"
                and e.get("node") == node
                and e["at_ns"] < barrier["at_ns"]
                for e in recorded
            )
            for node in ("N001", "N002")
            for barrier in recorded
            if barrier["event"] == "parallel_barrier_released"
        )
        and any(e["event"] == "parallel_barrier_released" for e in recorded),
        "retired_worktrees": all(not (paths.agents_dir / f"research-{n}").exists() for n in roots)
        and all(not (paths.agents_dir / c["agent"]).exists() for c in claims),
        "memory_idempotent": result.status == "not_due" and before == after,
        "wisdom_published": (notes / "wisdom.md").is_file(),
        "same_session_resumed": len(starts) >= 2
        and starts[1]["resume"] == starts[0]["session"]
        and starts[1]["worktree"] == starts[0]["worktree"],
        "analyzer_failure_recovered": any(e["event"] == "analyzer_failure" for e in recorded)
        and state["nodes"]["N002"]["research"]["status"] == "completed",
        "host_binding_verified": {e["node"] for e in claims} == {"N000", "N001", "N003"}
        and sum(e["event"] == "foreign_claim_rejected" for e in recorded) == 3,
        "infra_retry_same_sha_verified": any(
            e["event"] == "same_sha_retry" and e["commit"] == e["head_after"] for e in recorded
        )
        and any(
            a.metadata.get("node_id") == "N003"
            and len(a.metadata.get("grader_runs", [])) == 2
            and a.metadata["grader_runs"][0]["failure_class"] == "infra"
            and a.metadata["grader_runs"][1]["failure_class"] is None
            for a in attempts
        ),
        "authority_rejections": sorted(
            {e["check"] for e in recorded if e["event"] == "authority_rejected"}
        ),
        "residual_reservations": [
            item
            for item in state["nodes"]
            if state["nodes"][item]["research"]["status"] == "reserved"
        ],
        "blocked_command_invocations": (root() / "blocked-commands.log").read_text().splitlines()
        if (root() / "blocked-commands.log").exists()
        else [],
        "remaining_simulation_processes": live_processes(root()),
        "residual_engineer_reservations": [
            aid for aid, w in state["engineering_workers"].items() if w["status"] == "reserved"
        ],
        "not_covered": [
            "Live Codex model execution, instruction loading and model session persistence",
            "Model compliance with the delivered prompts and scientific judgment",
            "Real quality/semantic/efficiency/GPU training",
            "SRT/OS permissions and network isolation",
            "Cold start_all and dependency installation",
            "Manager/Grader crash and host power loss",
            "Manual pause/resume, circuit breaker, exhausted infrastructure retries",
            "Memory partial publication and Memory worker crash recovery",
            "Multi-island",
            "Long-running/load/race stress",
        ],
        "srt_available": shutil.which("srt") is not None,
        "artifacts": [str(path.relative_to(paths.run_dir)) for path in sorted(notes.rglob("*.md"))],
        "attempts": [
            {
                "node": a.metadata["node_id"],
                "mode": a.metadata["mode"],
                "agent": a.agent_id,
                "commit": a.commit_hash,
                "status": a.status,
                "run_ids": [r["run_id"] for r in a.metadata.get("grader_runs", [])],
            }
            for a in attempts
        ],
    }
    (root() / "report.json").write_text(json.dumps(report, indent=2))
    covered = [
        "真实 Seed Root bootstrap、S0/A0 与固定提交快照",
        "两 Researcher 并行初始化、唯一 Node ID、真实 UDS/PID 认证与提交校验",
        "模拟 quality 通过/失败后自动 Analysis、完成、干净 Worktree 回收与槽位补位",
        "Engineering 原子认领、独立 Worktree、两个候选、Engineer 提交总结后由宿主发布收尾",
        "Memory 三不同 Research Node 门槛（含 N000）、两 Summary 发布与幂等跳过",
        "普通 Agent 无法 reconcile、写 Tree/Spec 或其他 Node Analysis",
        "所有测试进程收尾后按 PID 和启动时间核查无残留",
    ]
    if os.environ["CORAL_SIM_SCENARIO"] == "faults":
        covered += [
            "脚本崩溃恢复同 Node、Worktree 和模拟 Session ID",
            "Analyzer 首次失败后重试，完成前保持 assignment",
            "两个 Engineer 竞争同一目标仅一人成功",
            "可重试 infra failure 不提前收尾，同 SHA 重试保留两份 run 日志",
        ]
    report_md = [
        "# Workflow 模拟覆盖报告",
        "",
        "所有科研结果均为模拟；未调用模型或 GPU。",
        "",
        "## 已覆盖",
        "",
        *[f"- {item}" for item in covered],
        "",
        "## 未覆盖",
        "",
        *[f"- {item}" for item in report["not_covered"]],
        "",
        "## 运行记录",
        "",
        "| Node | mode | Agent | Commit | status |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {a['node']} | {a['mode']} | {a['agent']} | {a['commit']} | {a['status']} |"
            for a in report["attempts"]
        ],
        "",
        "未执行的补位 reservation 保留在权威状态中；其进程已停止。",
        "",
        f"Research reservations: {report['residual_reservations']}",
        f"Engineer reservations: {report['residual_engineer_reservations']}",
        "",
        "完整事件与判定依据见同目录 timeline.jsonl、report.json。",
    ]
    (root() / "coverage.md").write_text("\n".join(report_md) + "\n")


def driver(output: Path, scenario: str):
    from coral.agent.manager import AgentManager

    os.environ["CORAL_SIM_ROOT"] = str(output)
    os.environ["CORAL_SIM_SCENARIO"] = scenario
    forbidden = output / "blocked-bin"
    forbidden.mkdir()
    for name in ("torchrun", "nvidia-smi"):
        command = forbidden / name
        command.write_text(
            "#!/bin/sh\nprintf '%s\\n' "
            + shlex.quote(name)
            + " >> "
            + shlex.quote(str(output / "blocked-commands.log"))
            + "\nexit 90\n"
        )
        command.chmod(0o755)
    recorder = forbidden / "codex"
    recorder.write_text(f"#!{sys.executable}\nfrom prompt_audit_support import main\nmain()\n")
    recorder.chmod(0o755)
    os.environ["PATH"] = str(forbidden) + os.pathsep + os.environ["PATH"]
    config, paths = prepare()
    manager = AgentManager(config, config_dir=TASK)
    errors = []
    launcher_patch = patch("coral.grader.trusted_worker.start_task_worker", launch_logged_worker)
    launcher_patch.start()
    try:
        manager.resume_all(paths)
        event(
            "daemon_started",
            child_pid=manager._grader_proc.pid,
            child_start=process_start_time(manager._grader_proc.pid),
        )
        control = threading.Thread(target=controller, args=(manager, paths, errors), daemon=True)
        control.start()
        manager.monitor_loop(check_interval=0.1)
        control.join(timeout=2)
    finally:
        gate("stop").touch()
        manager.stop_all()
        cleanup_processes(output)
        launcher_patch.stop()
    if errors:
        raise RuntimeError(errors[0])
    wait_for(
        lambda: (
            len([e for e in events() if e["event"] == "worker_start"])
            == len([e for e in events() if e["event"] == "worker_end"])
        ),
        "worker exit",
        timeout=10,
    )
    # worker_end is emitted in a finally block before the interpreter exits.
    # A late-starting finalizer can finish after cleanup's first PID snapshot.
    wait_for(lambda: not live_processes(output), "recorded process exit", timeout=10)
    make_report(paths)


def live_processes(output: Path) -> list[int]:
    identities = set()
    for entry in read_events(output):
        if entry["event"] in {"runtime_start", "daemon_started", "worker_spawn"}:
            identities.add((entry["child_pid"], entry["child_start"]))
        elif entry["event"] in {"worker_start", "graded"}:
            identities.add((entry["pid"], entry["pid_start"]))
    live = []
    for pid, started in identities:
        try:
            status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            if process_start_time(pid) == started and status not in {"Z", "X"}:
                live.append(pid)
        except (OSError, ValueError):
            pass
    return sorted(live)


def cleanup_processes(output: Path) -> None:
    # Only PIDs recorded by this test, with matching /proc start identities.
    deadline = time.monotonic() + 3
    while live_processes(output) and time.monotonic() < deadline:
        time.sleep(0.025)
    for pid in live_processes(output):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    if sys.argv[1] == "driver":
        driver(Path(sys.argv[2]), sys.argv[3])
    else:
        try:
            agent(Path(sys.argv[2]), sys.argv[3])
        except Exception:
            event("agent_error", agent=sys.argv[3], traceback=traceback.format_exc())
            raise
