"""Authoritative, run-local ownership state for dynamic AutoResearch agents."""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import subprocess
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock
from coral.hub._island import island_root
from coral.hub.attempts import read_attempt, read_attempts

from .analysis import AnalysisContext, _artifact_evidence, resolve_evidence
from .atomic import atomic_write_text
from .nodes import create_node
from .research import (
    frozen_research_attempt,
    quality_eligible,
    research_attempts,
    run_budget,
    select_research_attempt,
)
from .validation import load_document, validate_analysis

ENGINEERING_SUCCESS_LIMIT = 4
FORMAL_TOTAL_LIMIT = 10


class WorkflowError(ValueError):
    """An ownership transition or persisted workflow state is invalid."""


def _engineering_budget(attempts) -> tuple[int, int, int]:
    """Count successful runs, reserved/used runs and pending success slots."""
    successful = total = pending = 0
    for attempt in attempts:
        # Submission/retry reserves its run before dispatch; persistently count that
        # reservation so queued work and restarts cannot oversubscribe either cap.
        total += max(1 + int(attempt.metadata.get("infra_retry_count", 0)),
                     len(attempt.metadata.get("grader_runs") or []),
                     int(attempt.metadata.get("grader_invocation_count", 0)))
        pending += attempt.status == "pending"
        successful += (
            attempt.status in {"improved", "baseline", "regressed", "reverted"}
            and attempt.score is not None and math.isfinite(attempt.score)
            and not attempt.metadata.get("failure_class")
        )
    return successful, total, pending


class ResearchNotReadyError(WorkflowError):
    """The formal Attempt does not exist yet or still permits infrastructure retry."""


@dataclass(frozen=True)
class WorkflowAssignment:
    agent_id: str
    role: str
    node_id: str | None
    status: str
    research_mode: str | None = None


class WorkflowStore:
    """The small authoritative state document for one CORAL run."""

    def __init__(
        self, coral_dir: str | Path, island_id: str | None = None,
        *, repo_path: str | Path | None = None,
        max_engineering_runs_per_node: int = 1,
        p_explore: float | None = None,
    ) -> None:
        self.max_engineering_runs_per_node = max_engineering_runs_per_node
        if type(max_engineering_runs_per_node) is not int or max_engineering_runs_per_node != 1:
            raise WorkflowError("node-bound Engineering requires max_engineering_runs_per_node=1")
        self.coral_dir = Path(coral_dir).resolve()
        self.island_id = island_id
        self.island_root = island_root(self.coral_dir, island_id)
        self.notes_root = self.island_root / "notes"
        self.state_path = self.coral_dir / "private" / "autoresearch_state.json"
        self.tree_path = self.notes_root / "tree.md"
        config_path = self.coral_dir / "config.yaml"
        config = (yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
        if repo_path is None:
            # Saved workspace.repo_path can name the original source, which lacks
            # Agent commits. All entrypoints must prefer the run's actual clone.
            for candidate in (self.coral_dir.parent / 'repo', self.coral_dir.parent):
                if (candidate / '.git').exists():
                    repo_path = candidate
                    break
            if repo_path is None:
                repo_path = config.get('workspace', {}).get('repo_path') or None
        self.repo_path = Path(repo_path).resolve() if repo_path is not None else None
        self.max_active_nodes = config.get('grader', {}).get('args', {}).get('max_active_nodes', 0)
        if type(self.max_active_nodes) is not int or self.max_active_nodes < 0:
            raise WorkflowError('max_active_nodes must be a non-negative integer')
        self.research_planning_lease_seconds = config.get('grader', {}).get('args', {}).get(
            'research_planning_lease_seconds', 0)
        if (type(self.research_planning_lease_seconds) is not int
                or self.research_planning_lease_seconds < 0):
            raise WorkflowError('research_planning_lease_seconds must be a non-negative integer')
        if p_explore is None:
            p_explore = config.get("grader", {}).get("args", {}).get("p_explore", 0.20)
        if type(p_explore) not in (int, float) or not math.isfinite(p_explore) or not 0 <= p_explore <= 1:
            raise WorkflowError("p_explore must be a finite probability between 0 and 1")
        self.p_explore = p_explore

    def _verify_root(self) -> None:
        node_root = self.notes_root / "nodes" / "N000"
        spec_path = node_root / "spec.md"
        try:
            policy = json.loads(
                (self.coral_dir / "private" / "root_bootstrap.json").read_text(encoding="utf-8")
            )
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise WorkflowError("Root Analysis bootstrap requires a valid fixed Root policy/spec") from exc
        if (
            not isinstance(policy, dict)
            or policy.get("version") != 1
            or policy.get("node_id") != "N000"
            or policy.get("mode") != "research"
            or not isinstance(spec, dict)
            or spec.get("node_id") != "N000"
            or spec.get("base_commit") != policy.get("source_commit")
            or not isinstance(policy.get("submission_hash"), str)
        ):
            raise WorkflowError("Root Analysis bootstrap conflicts with the fixed Root policy")
        attempts = tuple(
            attempt
            for attempt in read_attempts(self.coral_dir, island_id=self.island_id)
            if attempt.metadata.get("node_id") == "N000"
            and attempt.metadata.get("mode") == "research"
            and attempt.budget_class == "real"
        )
        if (
            len(attempts) != 1
            or attempts[0].parent_hash != policy["source_commit"]
            or attempts[0].metadata.get("submission_hash") != policy["submission_hash"]
        ):
            raise WorkflowError("Root Analysis bootstrap requires one fixed N000 Research Attempt")

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            nodes_root = self.notes_root / "nodes"
            has_child = nodes_root.is_dir() and any(
                path.name.startswith("N") and path.name != "N000"
                for path in nodes_root.iterdir()
            )
            if has_child:
                raise WorkflowError("workflow state is missing from an existing run")
            return {
                "version": 2,
                "revision": 0,
                "next_node_seq": 1,
                "next_engineer_seq": 1,
                "next_ready_seq": 1,
                "nodes": {
                    "N000": {
                        "research": {
                            "agent_id": "root-bootstrap",
                            "status": "active",
                            "spec_ref": "nodes/N000/spec.md",
                        },
                        "engineering_claims": [],
                        "engineering_status": "blocked",
                        "analysis": None,
                    }
                },
                "engineering_workers": {},
            }
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkflowError("workflow state is invalid") from exc
        if isinstance(state, dict) and state.get("version") == 1:
            raise WorkflowError("workflow state version 1 requires its original runtime; automatic migration is not supported")
        if (
            not isinstance(state, dict)
            or state.get("version") != 2
            or not isinstance(state.get("revision"), int)
            or not isinstance(state.get("next_node_seq"), int)
            or not isinstance(state.get("next_engineer_seq"), int)
            or not isinstance(state.get("nodes"), dict)
            or not isinstance(state.get("engineering_workers"), dict)
            or "N000" not in state["nodes"]
            or not isinstance(state.get("next_ready_seq"), int)
        ):
            raise WorkflowError("workflow state is invalid")
        for record in state["nodes"].values():
            if record["research"].get("research_mode", "REFINE") not in {"REFINE", "EXPLORE"}:
                raise WorkflowError("workflow state has an invalid research mode")
        return state

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        state["revision"] += 1
        _atomic_json(self.state_path, state)
        self._rebuild_tree_unlocked(state)

    def _rebuild_tree_unlocked(self, state: dict[str, Any]) -> None:
        from .handover import publish_handovers
        from .navigation import node_relationship, rebuild_node_indexes

        publish_handovers(self, state)
        rebuild_node_indexes(self, state)
        rows = ["# Research Tree", "", "Open a Node's index for its code SHAs, reports and persistent evidence.", "",
                "| Node | Parent | Research mode | Research | Analysis | Engineering | Node status |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        admitted = self._admitted_node_ids_unlocked(state)
        for node_id, record in sorted(state["nodes"].items()):
            analysis = (record.get("analysis") or {}).get("status", "—")
            claims = record.get("engineering_claims", [])
            engineering = claims[-1]["status"] if claims else record.get("engineering_status", "blocked")
            status = "completed" if self._node_completed(record) else "in_progress"
            if (node_id not in admitted and status != 'completed'
                    and record['research']['status'] != 'cancelled'):
                status = 'waiting_for_node_slot'
            if record.get("error"):
                status = "error"
            parent, research_mode = node_relationship(self.notes_root, node_id, record)
            if parent in state['nodes']:
                parent = f'[{parent}](nodes/{parent}/README.md)'
            rows.append(f"| [{node_id}](nodes/{node_id}/README.md) | {parent} | {research_mode} | {record['research']['status']} | {analysis} | {engineering} | {status} |")
        atomic_write_text(self.tree_path, "\n".join(rows) + "\n", mode=0o644)

    @staticmethod
    def _node_completed(record: dict) -> bool:
        claims = record.get("engineering_claims", [])
        engineering_done = (bool(claims) and claims[-1]["status"] == "completed") or (
            record.get("engineering_status") == "skipped" and bool(record.get("engineering_skip_reason")))
        return (record["research"]["status"] == "completed"
                and (record.get("analysis") or {}).get("status") == "completed"
                and engineering_done)

    def _node_completed_unlocked(self, state: dict, node_id: str) -> bool:
        record = state["nodes"].get(node_id)
        if not record or not self._node_completed(record):
            return False
        reference = self._validated_analysis_ref_unlocked(node_id, "research")
        if reference != record["analysis"].get("analysis_ref"):
            return False
        if record["engineering_claims"]:
            self._validated_analysis_ref_unlocked(node_id, "engineering")
        return True

    def is_node_completed(self, node_id: str) -> bool:
        with trusted_writes_lock(self.coral_dir):
            return self._node_completed_unlocked(self._load_unlocked(), node_id)

    def _unfinished_node_ids_unlocked(self, state: dict) -> list[str]:
        """Oldest unfinished Nodes keep their slots through Analysis and Engineering."""
        unfinished = []
        for node_id, record in sorted(state['nodes'].items()):
            if record['research']['status'] == 'cancelled':
                continue
            try:
                completed = self._node_completed_unlocked(state, node_id)
            except (ValueError, OSError):
                completed = False
            if not completed:
                unfinished.append(node_id)
        return unfinished

    def _admitted_node_ids_unlocked(self, state: dict) -> set[str]:
        if not self.max_active_nodes:
            return set(state['nodes'])
        return set(self._unfinished_node_ids_unlocked(state)[:self.max_active_nodes])

    def _require_node_slot_unlocked(self, state: dict, node_id: str) -> None:
        if node_id not in self._admitted_node_ids_unlocked(state):
            raise WorkflowError('Node is waiting for an active Node slot')

    def rebuild_tree(self) -> Path:
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            self._rebuild_tree_unlocked(state)
        return self.tree_path

    def _assignments(self, state: dict[str, Any]) -> tuple[WorkflowAssignment, ...]:
        admitted = self._admitted_node_ids_unlocked(state)
        assignments = [
            WorkflowAssignment(record["research"]["agent_id"], "research", node_id,
                               "waiting" if node_id not in admitted else
                               "finalizing" if record["research"].get("finish_requested")
                               else record["research"]["status"],
                               research_mode=record["research"].get("research_mode", "REFINE"))
            for node_id, record in sorted(state["nodes"].items())
            if node_id != "N000" and record["research"]["status"] in {"reserved", "active"}
        ]
        assignments.extend(
            WorkflowAssignment(agent_id, "engineering", worker.get("node_id"),
                               "waiting" if worker.get('node_id') is not None
                               and worker['node_id'] not in admitted else
                               "finalizing" if worker.get("finish_requested") and worker.get("analysis_submission") else worker["status"])
            for agent_id, worker in sorted(state["engineering_workers"].items())
            if worker["status"] in {"reserved", "active"}
        )
        return tuple(assignments)

    def _qualified_research_unlocked(
        self, state: dict[str, Any], node_id: str, *, admitted: set[str] | None = None,
    ) -> str | None:
        record = state["nodes"].get(node_id)
        if not record or record.get("engineering_status") != "pending" or record["engineering_claims"]:
            return None
        if node_id not in (admitted if admitted is not None else self._admitted_node_ids_unlocked(state)):
            return None
        attempt = self._terminal_research_unlocked(node_id)
        if self.repo_path is None or not self.repo_path.is_dir():
            raise WorkflowError(f"Node {node_id} Research code snapshot is unavailable")
        check = subprocess.run(
            ["git", "cat-file", "-e", attempt.commit_hash + "^{commit}"],
            cwd=self.repo_path, capture_output=True, check=False,
        )
        if check.returncode:
            raise WorkflowError(f"Node {node_id} Research code snapshot is unavailable")
        return attempt.commit_hash

    def _context(self, *, include_archived_engineering: bool = False) -> AnalysisContext:
        attempts = tuple(
            a for a in read_attempts(self.coral_dir, island_id=self.island_id,
                                     include_archived=True)
            if not a.archived or a.metadata.get("mode") in {"engineering", "research"}
        )
        return AnalysisContext(self.coral_dir, self.notes_root, self.island_id,
                               attempts,
                               self.island_root / "eval_logs", self.repo_path or self.coral_dir.parent)

    def _prepare_research_unlocked(self, state: dict, node_id: str, *, finishing=False) -> None:
        from .evaluation import refresh_node_evaluation

        record = state["nodes"][node_id]
        if record.get("analysis") is not None:
            return
        attempt = self._terminal_research_unlocked(node_id, finishing=finishing, state=state)
        if attempt.agent_id != record["research"]["agent_id"]:
            raise WorkflowError("Research Attempt owner does not match Node")
        quality = attempt.metadata.get("scores", {}).get("quality_gate", {}).get("value")
        if record['research'].get('iterative') and not quality_eligible(attempt):
            skip = 'quality_failed' if quality == 0 else 'quality_or_timing_unavailable'
        elif type(quality) in (int, float) and quality in (0, 1):
            skip = "quality_failed" if quality == 0 else None
        elif quality is None and attempt.metadata.get("failure_class") in {"infra", "candidate"}:
            skip = "quality_unavailable"
        else:
            raise WorkflowError(f"Node {node_id} has missing or invalid quality evidence")
        if record['research'].get('iterative'):
            record['research'].update(selected_sha=attempt.commit_hash, finish_requested=True)
            # Persist the immutable choice before constructing its evidence envelope.
            _atomic_json(self.state_path, state)
        context = self._context()
        if record['research'].get('iterative') and record['research'].get('protocol') != 'staged-v2':
            from .checkpoint_retention import prune_research_checkpoints

            prune_research_checkpoints(self.coral_dir, node_id, context.attempts,
                selected_sha=attempt.commit_hash, island_id=self.island_id)
        content = refresh_node_evaluation(node_id, context.attempts, context.log_root)
        atomic_write_text(self.notes_root / "nodes" / node_id / "evaluation.md", content, mode=0o644)
        evidence = resolve_evidence(node_id, "research", context=context)
        record["research"].update(status="completed", finish_requested=True)
        record["ready_seq"] = state["next_ready_seq"]
        state["next_ready_seq"] += 1
        record["engineering_status"] = "skipped" if skip else "pending"
        record["engineering_skip_reason"] = skip
        record["analysis"] = {"status": "pending", "research_sha": attempt.commit_hash,
                              "evidence_hash": evidence.manifest_hash, "version": 1,
                              "generation": 0, "failures": 0, "next_retry_at": 0,
                              "analysis_ref": None, "last_error": None}

    def _advance_unlocked(self, state: dict) -> list[str]:
        """Short transitions only; never wait for an Analyzer or acquire another workflow lock."""
        before = copy.deepcopy(state)
        from .planning import recover_registration

        recover_registration(self, state)
        for node_id, record in state["nodes"].items():
            if record["research"]["status"] not in {"active", "completed"}:
                continue
            try:
                if (record['research'].get('iterative') and record.get('analysis') is None
                        and record['research'].get('protocol') != 'staged-v2'):
                    from .checkpoint_retention import prune_research_checkpoints

                    attempts = research_attempts(read_attempts(self.coral_dir, island_id=self.island_id,
                                                               include_archived=True), node_id)
                    if any(a.agent_id != record['research']['agent_id'] for a in attempts):
                        raise WorkflowError('Research Attempt owner does not match Node')
                    if any(a.status != 'pending' for a in attempts):
                        prune_research_checkpoints(self.coral_dir, node_id, attempts,
                            selected_sha=record['research'].get('selected_sha'), island_id=self.island_id)
                self._terminal_research_unlocked(node_id, state=state)
            except ResearchNotReadyError:
                continue
            except (ValueError, OSError, TypeError) as exc:
                record["error"] = str(exc)
                continue
            try:
                self._prepare_research_unlocked(state, node_id)
                job = record["analysis"]
                path = self.notes_root / "nodes" / node_id / f"research_analysis_v{job['version']}.md"
                if job["status"] != "running" and path.exists():
                    self._adopt_analysis_unlocked(node_id, job)
                record.pop("error", None)
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                record["error"] = str(exc)
        completed = []
        for agent_id, worker in state["engineering_workers"].items():
            if worker["status"] != "active" or not worker.get("analysis_submission"):
                continue
            try:
                self._publish_engineering_unlocked(state, agent_id)
                worker.pop("last_error", None)
                completed.append(agent_id)
            except (ValueError, OSError) as exc:
                worker["last_error"] = str(exc)
        if state != before or not self.state_path.exists():
            self._save_unlocked(state)
        from .node_cleanup import cleanup_completed_nodes

        if cleanup_completed_nodes(self, state):
            self._save_unlocked(state)
        return completed

    def _adopt_analysis_unlocked(self, node_id, job):
        context = self._context()
        evidence = resolve_evidence(node_id, "research", context=context)
        if evidence.manifest_hash != job["evidence_hash"] or evidence.research_sha != job["research_sha"]:
            raise WorkflowError("Analysis job evidence changed")
        path = self.notes_root / "nodes" / node_id / f"research_analysis_v{job['version']}.md"
        document = load_document(path)
        validate_analysis(document, _artifact_evidence(document, evidence, context))
        job.update(status="completed", analysis_ref=f"nodes/{node_id}/{path.name}", last_error=None)

    def advance(self) -> list[str]:
        with trusted_writes_lock(self.coral_dir):
            return self._advance_unlocked(self._load_unlocked())

    def is_engineering_eligible(self, node_id: str) -> bool:
        with trusted_writes_lock(self.coral_dir):
            return self._qualified_research_unlocked(self._load_unlocked(), node_id) is not None

    def reconcile(
        self, max_researchers: int, max_engineers: int
    ) -> tuple[WorkflowAssignment, ...]:
        if (
            not isinstance(max_researchers, int)
            or not isinstance(max_engineers, int)
            or max_researchers < 0
            or max_engineers < 0
        ):
            raise WorkflowError("workflow capacities must be non-negative integers")
        if self.max_active_nodes:
            # Node admission is the only task concurrency limit in this mode.
            max_researchers = max_engineers = self.max_active_nodes
        with trusted_writes_lock(self.coral_dir):
            self._verify_root()
            toolkit = self.notes_root / "toolkit"
            toolkit.mkdir(parents=True, exist_ok=True)
            task_root = Path(__file__).resolve().parents[3]
            from .runtime_artifacts import settings
            options = settings(self.coral_dir)
            staged = options.get("staged_research") is True
            from .runtime_resources import publish_catalog
            publish_catalog(toolkit, options)
            rules = "workflow_rules.md" if staged else "workflow_rules_legacy.md"
            shutil.copyfile(task_root / "roles" / rules, toolkit / "workflow_rules.md")
            shutil.copytree(task_root / "references", toolkit / "references", dirs_exist_ok=True)
            shutil.copytree(task_root / "templates", toolkit / "templates", dirs_exist_ok=True)
            if staged:
                shutil.copyfile(task_root / "roles/workflow_research.md", toolkit / "workflow_research.md")
                (toolkit / "templates/research_intent.json").unlink(missing_ok=True)
            shutil.copyfile(task_root / "README.md", toolkit / "README.md")
            state = self._load_unlocked()
            self._advance_unlocked(state)
            admitted = self._admitted_node_ids_unlocked(state)
            live_research = sum(
                record["research"]["status"] in {"reserved", "active"}
                for node_id, record in state["nodes"].items()
                if node_id != "N000" and node_id in admitted
            )
            changed = not self.state_path.is_file()
            while (live_research < max_researchers and self._node_completed_unlocked(state, "N000")
                   and not (self.coral_dir / "private/research_infrastructure_block.json").exists()):
                if (self.max_active_nodes
                        and len(self._unfinished_node_ids_unlocked(state)) >= self.max_active_nodes):
                    break
                sequence = state["next_node_seq"]
                if sequence > 999:
                    raise WorkflowError("Node ID space is exhausted")
                node_id = f"N{sequence:03d}"
                if node_id in state["nodes"]:
                    raise WorkflowError("Node ID counter conflicts with existing state")
                state["nodes"][node_id] = {
                    "research": {"agent_id": f"research-{node_id}", "status": "reserved",
                                 "spec_ref": None,
                                 "iterative": True,
                                 "research_mode": "EXPLORE" if random.random() < self.p_explore else "REFINE"},
                    "engineering_claims": [], "engineering_status": "blocked", "analysis": None,
                }
                state["next_node_seq"] = sequence + 1
                live_research += 1
                changed = True
            admitted = self._admitted_node_ids_unlocked(state)
            live_engineers = sum(worker["status"] == "active"
                                 and (worker.get('node_id') is None
                                      or worker['node_id'] in admitted)
                                 for worker in state["engineering_workers"].values())
            for node_id in sorted(state["nodes"], key=lambda n: state["nodes"][n].get("ready_seq", float("inf"))):
                if live_engineers >= max_engineers:
                    break
                record = state["nodes"][node_id]
                try:
                    research_sha = self._qualified_research_unlocked(state, node_id, admitted=admitted)
                except WorkflowError as exc:
                    record["error"] = str(exc)
                    changed = True
                    continue
                if research_sha is None:
                    continue
                sequence = state["next_engineer_seq"]
                agent_id = f"engineer-E{sequence:04d}"
                if agent_id in state["engineering_workers"]:
                    raise WorkflowError("Engineer ID counter conflicts with existing state")
                claim = {"agent_id": agent_id, "status": "active", "research_sha": research_sha}
                record["engineering_claims"].append(claim)
                record["engineering_status"] = "active"
                record.pop("error", None)
                state["engineering_workers"][agent_id] = {
                    "status": "active", "node_id": node_id, "research_sha": research_sha}
                state["next_engineer_seq"] = sequence + 1
                live_engineers += 1
                changed = True
            if changed:
                self._save_unlocked(state)
            else:
                self._rebuild_tree_unlocked(state)
            return self._assignments(state)

    def claim_engineering(self, agent_id: str, node_id: str) -> dict[str, Any]:
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            worker = state["engineering_workers"].get(agent_id)
            if not isinstance(worker, dict):
                raise WorkflowError("Engineering assignment owner was not found")
            if worker["status"] == "active" and worker.get("node_id") == node_id:
                self._reject_changed_anchor_unlocked(state, worker, node_id)
                return {
                    "agent_id": agent_id, "node_id": node_id,
                    "status": "active", "research_sha": worker["research_sha"],
                    "run_limits": {"successful": ENGINEERING_SUCCESS_LIMIT, "total": FORMAL_TOTAL_LIMIT},
                    **({'design_source': self._engineering_design_source(node_id, worker['research_sha'])}
                       if self._staged_enabled() else {}),
                }
            raise WorkflowError("Engineering assignment is not eligible for another Node; use the host-bound target")

    def _staged_enabled(self) -> bool:
        from .runtime_artifacts import settings

        return settings(self.coral_dir).get('staged_research') is True

    def _engineering_design_source(self, node_id: str, research_sha: str) -> dict[str, Any]:
        # Root is a fixed baseline, not a developed candidate with iterations.
        if node_id == 'N000':
            from .bootstrap import _root_spec

            try:
                policy = json.loads((self.coral_dir / 'private/root_bootstrap.json').read_text())
                spec = yaml.safe_load((self.notes_root / 'nodes/N000/spec.md').read_text())
                expected = _root_spec(policy['source_commit'])
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                raise WorkflowError('fixed Root baseline handover is required') from exc
            if spec != expected:
                raise WorkflowError('Root baseline Spec must match the fixed host baseline')
            return {'path': 'notes/nodes/N000/spec.md', 'iteration': None}
        try:
            public = json.loads((self.notes_root / 'nodes' / node_id / 'research.json').read_text())
            record = json.loads((self.coral_dir / 'private/research_records' / f'{node_id}.json').read_text())['document']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WorkflowError('frozen research record is required for Engineering') from exc
        if not isinstance(record, dict) or public != record:
            raise WorkflowError('research record must match the frozen host record')
        selected = record.get('selected_iteration')
        rows = record.get('iterations', [])
        if (record.get('version') != 3 or record.get('frozen') is not True
                or type(selected) is not int or not 1 <= selected <= len(rows)
                or rows[selected - 1].get('id') != selected
                or rows[selected - 1].get('status') != 'completed'
                or rows[selected - 1].get('code_sha') != research_sha):
            raise WorkflowError('frozen selected design must match the Engineering Research SHA')
        return {'path': f'notes/nodes/{node_id}/research.json', 'iteration': selected}


    def cancel_assignment(self, agent_id: str) -> dict[str, Any]:
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            worker = state["engineering_workers"].get(agent_id)
            if isinstance(worker, dict):
                raise WorkflowError("a host-bound Engineering assignment cannot be cancelled")
            for node_id, record in state["nodes"].items():
                research = record["research"]
                if node_id == "N000" or research["agent_id"] != agent_id:
                    continue
                staged = research.get('protocol') == 'staged-v2'
                if research['status'] != 'reserved' and not (staged and research['status'] == 'active'):
                    raise WorkflowError('only an unstarted reservation or active staged research can be cancelled')
                if staged:
                    from .lease import ResearchLease
                    ResearchLease(self.coral_dir, node_id).release()
                research["status"] = "cancelled"
                self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": node_id, "status": "cancelled"}
            raise WorkflowError("assignment owner was not found")


    def _terminal_research_unlocked(self, node_id: str, *, finishing=False, state=None):
        state = state if state is not None else self._load_unlocked()
        record = state['nodes'][node_id]['research']
        attempts = research_attempts(read_attempts(self.coral_dir, island_id=self.island_id,
                                                 include_archived=True), node_id)
        if record.get('protocol') == 'staged-v2' and not record.get('transfer_terminal'):
            raise ResearchNotReadyError(f'Node {node_id} staged development has not ended')
        if not attempts:
            raise ResearchNotReadyError(f"Node {node_id} requires one formal Research Attempt")
        iterative = bool(record.get('iterative'))
        if not iterative and len(attempts) != 1:
            raise WorkflowError(f"Node {node_id} requires one formal Research Attempt")
        if any(a.agent_id != record['agent_id'] for a in attempts):
            raise WorkflowError('Research Attempt owner does not match Node')
        successful, total, _ = run_budget(attempts)
        exhausted = iterative and (successful >= 3 or total >= 10)
        if any(a.status == 'pending' or (a.status in {'crashed', 'timeout'}
                and a.metadata.get('failure_class') == 'infra'
                and int(a.metadata.get('infra_retry_count', 0)) < int(a.metadata.get('max_infra_retries', 0))
                and not iterative) for a in attempts):
            raise ResearchNotReadyError(f"Node {node_id} Research Attempt is not terminal")
        if iterative and not (finishing or record.get('finish_requested') or exhausted
                              or record.get('status') == 'completed'):
            raise ResearchNotReadyError(f'Node {node_id} Research is still active')
        if record.get('selected_sha'):
            return frozen_research_attempt(self.coral_dir, node_id, attempts)
        return select_research_attempt(attempts)

    def _validated_analysis_ref_unlocked(self, node_id: str, mode: str) -> str:
        node_root = self.notes_root / "nodes" / node_id
        versions = []
        for path in node_root.glob(f"{mode}_analysis_v*.md"):
            suffix = path.stem.removeprefix(f"{mode}_analysis_v")
            if suffix.isdigit() and int(suffix) > 0:
                versions.append((int(suffix), path))
        if not versions:
            raise WorkflowError(f"Node {node_id} requires a valid {mode.title()} Analysis")
        context = AnalysisContext(
            coral_dir=self.coral_dir,
            notes_root=self.notes_root,
            island_id=self.island_id,
            attempts=self._context().attempts,
            log_root=self.island_root / "eval_logs",
            checkout_root=self.repo_path or self.coral_dir.parent,
        )
        for _, path in sorted(versions, reverse=True):
            try:
                evidence = resolve_evidence(node_id, mode, context=context)
                document = load_document(path)
                evidence = _artifact_evidence(document, evidence, context)
                validate_analysis(document, evidence)
            except (OSError, ValueError):
                continue
            return f"nodes/{node_id}/{path.name}"
        raise WorkflowError(f"Node {node_id} requires a valid {mode.title()} Analysis")

    def initialize_research(
        self, agent_id: str, node_id: str, request: Mapping[str, Any],
        *, owner_identity: Mapping[str, Any] | None = None, origin_version: int | None = None,
    ) -> dict[str, Any]:
        if origin_version not in (None, 3):
            raise WorkflowError("unsupported research origin version")
        required = {
            "parent_node", "hypothesis", "selection_rationale",
            "delta", "risk", "falsification_criteria",
        }
        optional = {'research_intent', 'planning_token', 'overlap_review'}
        if (not isinstance(request, Mapping) or not required.issubset(request)
                or set(request) - required - optional):
            raise WorkflowError("Research initialization has invalid fields")
        request = dict(request)
        planning_token = request.pop('planning_token', None)
        overlap_review = request.pop('overlap_review', None)
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            record = state["nodes"].get(node_id)
            if not isinstance(record, dict) or record["research"]["agent_id"] != agent_id:
                raise WorkflowError("Research reservation owner does not match")
            self._require_node_slot_unlocked(state, node_id)
            research = record["research"]
            if research["status"] not in {"reserved", "active"}:
                raise WorkflowError("Research reservation is not active")
            from .handover import _safe_file

            target = _safe_file(self.notes_root, f"nodes/{node_id}/spec.md")
            from .planning import release_owned, require_lease

            if target.exists():
                fixed = yaml.safe_load(target.read_text())
                if (not isinstance(fixed, dict) or fixed.get("node_id") != node_id
                        or origin_version is not None and fixed.get("research_origin_version") != origin_version
                        or any(fixed.get(key) != value for key, value in request.items())
                        or overlap_review is not None and overlap_review != (
                            fixed.get('method_registration') or {}).get('overlap_review')):
                    raise WorkflowError("immutable Node Spec conflicts with reservation")
                before = copy.deepcopy(state)
                release_owned(state, agent_id, planning_token)
                if research["status"] == "reserved":
                    research.update(status="active", spec_ref=f"nodes/{node_id}/spec.md")
                if state != before:
                    self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": node_id, "status": "active",
                        "spec_ref": f"nodes/{node_id}/spec.md", "base_commit": fixed["base_commit"],
                        "research_mode": research.get("research_mode", "REFINE")}
            if self.research_planning_lease_seconds:
                require_lease(state, agent_id, planning_token, owner_identity)
                if not isinstance(overlap_review, str) or not overlap_review.strip() or len(overlap_review) > 4000:
                    raise WorkflowError('overlap_review must explain differences from registered methods in at most 4000 characters')
            parent_node = request["parent_node"]
            parent = state["nodes"].get(parent_node)
            if not isinstance(parent, dict) or not self._node_completed_unlocked(state, parent_node):
                raise WorkflowError("selected Parent Node is not completed")
            parent_attempt = self._terminal_research_unlocked(parent_node)
            from .handover import publish_handovers

            publish_handovers(self, state)
            analysis_ref = self._validated_analysis_ref_unlocked(parent_node, "research")
            target = self.notes_root / "nodes" / node_id / "spec.md"

            def publish(kind: str, content: str, metadata: dict[str, Any]) -> dict[str, str]:
                if kind != "spec" or metadata["node_id"] != node_id:
                    raise WorkflowError("unexpected Node Spec publication")
                if target.exists():
                    if target.read_text(encoding="utf-8") != content:
                        raise WorkflowError("immutable Node Spec conflicts with reservation")
                else:
                    atomic_write_text(target, content, mode=0o644)
                return {"path": f"nodes/{node_id}/spec.md"}

            evidence = {
                "notes_root": self.notes_root,
                "parent_node": parent_node,
                "parent_commit": parent_attempt.commit_hash,
                "base_commit": parent_attempt.commit_hash,
                "analysis_ref": analysis_ref,
                "engineering_analysis_ref": (
                    self._validated_analysis_ref_unlocked(parent_node, "engineering")
                    if parent.get("engineering_status") == "completed" else None),
                "publish": publish,
            }
            if origin_version is not None:
                evidence['research_origin_version'] = origin_version
            if self.research_planning_lease_seconds:
                evidence['method_registration'] = {
                    'overlap_review': overlap_review,
                    'reviewed_specs': [f'nodes/{other}/spec.md' for other in sorted(state['nodes'])
                                       if other != node_id and (self.notes_root / 'nodes' / other / 'spec.md').is_file()]}
            # Do not rewrite an immutable pre-upgrade Spec just to add mode metadata.
            if not target.exists() or "research_mode" in yaml.safe_load(target.read_text()):
                evidence["research_mode"] = research.get("research_mode", "REFINE")
            create_node({**request, "node_id": node_id}, evidence)
            release_owned(state, agent_id, planning_token)
            if research["status"] == "reserved":
                research.update({"status": "active", "spec_ref": f"nodes/{node_id}/spec.md"})
                self._save_unlocked(state)
            return {"agent_id": agent_id, "node_id": node_id,
                    "status": "active", "spec_ref": f"nodes/{node_id}/spec.md",
                    "base_commit": parent_attempt.commit_hash,
                    "research_mode": research.get("research_mode", "REFINE")}

    def complete_assignment(self, agent_id: str, mode: str) -> dict[str, Any]:
        if mode not in {"research", "engineering"}:
            raise WorkflowError("invalid assignment mode")
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            if mode == "research":
                for node_id, record in state["nodes"].items():
                    research = record["research"]
                    if node_id == "N000" or research["agent_id"] != agent_id:
                        continue
                    if research["status"] == "completed":
                        return {"agent_id": agent_id, "node_id": node_id, "status": "completed"}
                    if research["status"] != "active":
                        raise WorkflowError("Research assignment is not active")
                    self._prepare_research_unlocked(state, node_id, finishing=True)
                    self._save_unlocked(state)
                    return {"agent_id": agent_id, "node_id": node_id, "status": "completed"}
            if mode == "engineering":
                worker = state["engineering_workers"].get(agent_id)
                if not isinstance(worker, dict) or not worker.get("node_id"):
                    raise WorkflowError("Engineering assignment owner was not found")
                node_id = worker["node_id"]
                if worker["status"] == "completed":
                    return {"agent_id": agent_id, "node_id": node_id, "status": "completed"}
                if worker["status"] != "active":
                    raise WorkflowError("Engineering assignment is not active")
                self._complete_engineering_unlocked(state, agent_id)
                self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": node_id, "status": "completed"}
            raise WorkflowError("assignment owner was not found")

    def _reject_changed_anchor_unlocked(self, state, worker, node_id):
        record = state["nodes"][node_id]
        attempt = self._terminal_research_unlocked(node_id)
        quality = attempt.metadata.get("scores", {}).get("quality_gate", {}).get("value")
        if (record["research"]["status"] != "completed" or type(quality) not in (float, int)
                or quality != 1 or attempt.commit_hash != worker["research_sha"]):
            raise WorkflowError("Node is not eligible for Engineering")
        if self.repo_path is None or not self.repo_path.is_dir() or subprocess.run(
            ["git", "cat-file", "-e", attempt.commit_hash + "^{commit}"],
            cwd=self.repo_path, capture_output=True, check=False).returncode:
            raise WorkflowError("Node is not eligible: Research code snapshot is unavailable")

    def _complete_engineering_unlocked(self, state, agent_id):
        worker = state["engineering_workers"][agent_id]
        node_id = worker["node_id"]
        self._terminal_engineering_unlocked(agent_id, node_id)
        self._validated_analysis_ref_unlocked(node_id, "engineering")
        submission = worker.get("analysis_submission")
        if not submission:
            raise WorkflowError("Engineering completion requires an Engineer-authored summary")
        path = self.notes_root / "nodes" / node_id / f"engineering_analysis_v{submission['version']}.md"
        current = resolve_evidence(node_id, "engineering", context=self._context(include_archived_engineering=True))
        validate_analysis(submission["document"], current)
        if load_document(path) != submission["document"]:
            raise WorkflowError("published Engineering Analysis differs from author submission")
        claims = state["nodes"][node_id]["engineering_claims"]
        match = [claim for claim in claims if claim["agent_id"] == agent_id]
        if len(match) != 1 or match[0]["status"] != "active":
            raise WorkflowError("Engineering claim does not match active owner")
        match[0]["status"] = "completed"
        worker["status"] = "completed"
        state["nodes"][node_id]["engineering_status"] = "completed"

    def _publish_engineering_unlocked(self, state, agent_id):
        from coral.grader.analysis_artifacts import publish_analysis

        worker = state["engineering_workers"][agent_id]
        node_id = worker["node_id"]
        submission = worker["analysis_submission"]
        self._terminal_engineering_unlocked(agent_id, node_id)
        current = resolve_evidence(node_id, "engineering", context=self._context(include_archived_engineering=True))
        validate_analysis(submission["document"], current)
        path = self.notes_root / "nodes" / node_id / f"engineering_analysis_v{submission['version']}.md"
        if path.exists():
            if load_document(path) != submission["document"]:
                raise WorkflowError("immutable Engineering Analysis conflicts with frozen submission")
        publish_analysis(path, submission["document"])
        self._complete_engineering_unlocked(state, agent_id)

    def submit_engineering_analysis(self, agent_id: str, body: dict) -> dict[str, Any]:
        from .engineering import submit

        return submit(self, agent_id, body)

    def request_finish(self, agent_id: str, mode: str) -> dict[str, Any]:
        """Freeze an owned flow for asynchronous Analysis and completion."""
        if mode not in {"research", "engineering"}:
            raise WorkflowError("invalid assignment mode")
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            for item in self._assignments(state):
                if item.agent_id != agent_id or item.role != mode or not item.node_id:
                    continue
                if item.status not in {"active", "finalizing"}:
                    raise WorkflowError("assignment is not active")
                if mode == "research":
                    self._terminal_research_unlocked(item.node_id, finishing=True, state=state)
                    record = state["nodes"][item.node_id]["research"]
                else:
                    self._terminal_engineering_unlocked(agent_id, item.node_id)
                    record = state["engineering_workers"][agent_id]
                    if not record.get("analysis_submission"):
                        raise WorkflowError("Engineering finish requires an Engineer-authored summary")
                if not record.get("finish_requested"):
                    record["finish_requested"] = True
                    self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": item.node_id, "status": "finalizing"}
            raise WorkflowError("active assignment owner was not found")

    def _terminal_engineering_unlocked(self, agent_id: str, node_id: str) -> None:
        attempts = [
            attempt for attempt in read_attempts(self.coral_dir, island_id=self.island_id, include_archived=True)
            if attempt.agent_id == agent_id and attempt.metadata.get("node_id") == node_id
            and attempt.metadata.get("mode") == "engineering" and attempt.budget_class != "tune"
        ]
        if not attempts:
            raise WorkflowError(f"Node {node_id} requires an Engineering Attempt")
        successful, total, _ = _engineering_budget(attempts)
        budget_exhausted = successful >= ENGINEERING_SUCCESS_LIMIT or total >= FORMAL_TOTAL_LIMIT
        if any(attempt.status == "pending" or (
            attempt.status in {"crashed", "timeout"}
            and attempt.metadata.get("failure_class") == "infra"
            and not attempt.archived
            and int(attempt.metadata.get("infra_retry_count", 0))
            < int(attempt.metadata.get("max_infra_retries", 0))
            and not budget_exhausted
        ) for attempt in attempts):
            raise WorkflowError("Engineering Attempt is not terminal")



def filter_dispatch(*, candidates: list[dict], context: dict) -> dict:
    """Host-only dispatch policy; recheck canonical Attempts without consuming runs."""
    coral_dir = context['coral_dir']
    store = WorkflowStore(coral_dir)
    with nullcontext() if context.get('lock_held') else trusted_writes_lock(coral_dir):
        state = store._load_unlocked()
        admitted = store._admitted_node_ids_unlocked(state)
        hashes = []
        for candidate in candidates:
            attempt = read_attempt(coral_dir, candidate['commit_hash'],
                                   island_id=candidate.get('island_id'))
            if (attempt is not None and not attempt.archived and attempt.status == 'pending'
                    and attempt.metadata.get('node_id') in admitted):
                hashes.append(attempt.commit_hash)
        return {'eligible_hashes': hashes}


def _owned_active(
    state: dict[str, Any], agent_id: str | None, node_id: str, mode: str,
) -> bool:
    record = state["nodes"].get(node_id)
    if not isinstance(record, dict):
        return False
    if mode == "research":
        research = record["research"]
        return research["agent_id"] == agent_id and research["status"] == "active"
    if mode == "engineering":
        return any(
            claim["agent_id"] == agent_id and claim["status"] == "active"
            for claim in record["engineering_claims"]
        )
    return False


def authorize_attempt(canonical: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Read-only owner check; caller already holds trusted_writes.lock."""
    store = WorkflowStore(context["coral_dir"], context.get("island_id"))
    state = store._load_unlocked()
    if not _owned_active(
        state, context.get("agent_id"), canonical.get("node_id"), canonical.get("mode")
    ):
        raise WorkflowError("Attempt owner is not the active Node assignment")
    store._require_node_slot_unlocked(state, canonical.get('node_id'))
    _reject_finalizing(state, context.get("agent_id"), canonical.get("node_id"), canonical.get("mode"))
    mode = canonical.get("mode")
    from .runtime_artifacts import settings
    if mode == 'research' and settings(store.coral_dir).get('staged_research') is True:
        raise WorkflowError('staged Research requires autoresearch research; coral eval cannot bypass it')
    if mode == 'engineering' and store._staged_enabled():
        if context.get('entrypoint') != 'engineering_timing':
            raise WorkflowError('Engineering requires autoresearch engineering timing')
        store._engineering_design_source(canonical['node_id'],
                                         state['engineering_workers'][context['agent_id']]['research_sha'])
    iterative = mode == "research" and state["nodes"][canonical["node_id"]]["research"].get("iterative")
    if mode == "engineering" or iterative:
        label = "Research" if iterative else "Engineering"
        if context.get("budget_class", "real") != "real":
            raise WorkflowError(f"{label} requires formal runs; --tune is not supported")
        attempts = [a for a in read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True)
                    if a.agent_id == context.get("agent_id") and a.metadata.get("node_id") == canonical.get("node_id")
                    and a.metadata.get("mode") == mode and a.budget_class != "tune"]
        existing = any(a.commit_hash == context.get("commit_hash") for a in attempts)
        if not existing or context.get("retry"):
            successful, total, pending = (_engineering_budget(attempts) if mode == "engineering"
                                          else run_budget(attempts))
            success_limit = ENGINEERING_SUCCESS_LIMIT if mode == 'engineering' else 3
            if successful + pending >= success_limit or total >= FORMAL_TOTAL_LIMIT:
                raise WorkflowError(f"{label} run limit reached: at most {success_limit} successful runs "
                                    f"and {FORMAL_TOTAL_LIMIT} total runs")
    return {"allow_multiple_research": True} if iterative else {}



def _reject_finalizing(state, agent_id, node_id, mode) -> None:
    record = (state["nodes"][node_id]["research"] if mode == "research"
              else state["engineering_workers"].get(agent_id, {}))
    if record.get("finish_requested"):
        raise WorkflowError("assignment is finalizing; no new candidate or Agent Analysis is allowed")


def authorize_artifact(kind: str, context: dict[str, Any]) -> dict[str, Any]:
    """Read-only owner/freshness check; caller already holds trusted_writes.lock."""
    if kind in {"tree", "spec"}:
        raise WorkflowError("Tree and Spec require the authoritative workflow entry")
    if kind not in {"analysis", "evaluation"}:
        raise WorkflowError("Artifact kind is not Agent-writable")
    store = WorkflowStore(context["coral_dir"], context.get("island_id"))
    state = store._load_unlocked()
    node_id = context.get("node_id")
    agent_id = context.get("agent_id")
    if kind == "analysis":
        if context.get("mode") == "engineering":
            raise WorkflowError("Engineering Analysis requires the trusted Engineering finish summary workflow")
        if not _owned_active(state, agent_id, node_id, context.get("mode")):
            raise WorkflowError("Artifact owner is not the active Node assignment")
        _reject_finalizing(state, agent_id, node_id, context.get("mode"))
        return {}
    if not (
        _owned_active(state, agent_id, node_id, "research")
        or _owned_active(state, agent_id, node_id, "engineering")
    ):
        raise WorkflowError("Evaluation owner is not the active Node assignment")
    from .evaluation import refresh_node_evaluation

    content = refresh_node_evaluation(
        node_id,
        read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True),
        store.island_root / "eval_logs",
    )
    return {"content": content}


def authorize_memory_start(agent_id: str, context: dict[str, Any]) -> dict[str, Any]:
    """Only a live Researcher assignment may request the one-shot worker."""
    if context.get("island_id") is not None:
        raise WorkflowError("dynamic Memory supports only one island")
    store = WorkflowStore(context["coral_dir"])
    with trusted_writes_lock(store.coral_dir):
        state = store._load_unlocked()
        if not any(
            node_id != "N000"
            and record["research"]["agent_id"] == agent_id
            and record["research"]["status"] in {"reserved", "active"}
            for node_id, record in state["nodes"].items()
        ):
            raise WorkflowError("live Researcher required for Memory launch")
    return {}


def handle_workflow(
    operation: str, payload: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch only the fixed task-local state transitions."""
    if context.get("island_id") is not None:
        raise WorkflowError("dynamic AutoResearch supports only one island")
    store = WorkflowStore(
        context["coral_dir"],
        repo_path=context.get("repo_path"),
        max_engineering_runs_per_node=context.get("max_engineering_runs_per_node", 1),
    )
    owner = context.get("agent_id")
    if operation == "reconcile_workflow":
        if not context.get("is_manager") or payload:
            raise WorkflowError("only the manager may reconcile workflow")
        if not store.max_active_nodes and context["max_engineers"] < 1:
            raise WorkflowError("node-bound workflow requires agents.dynamic.max_engineers >= 1")
        assignments = store.reconcile(
            context["max_researchers"], context["max_engineers"]
        )
        from .memory import memory_due

        result = {"assignments": [asdict(item) for item in assignments]}
        try:
            result["memory_due"] = memory_due(store.notes_root).due
        except Exception as exc:
            result["memory_error"] = str(exc)
        return result
    if context.get("is_manager") or not isinstance(owner, str):
        raise WorkflowError("workflow mutation requires an authenticated Agent")
    if operation in {'acquire_research_planning', 'renew_research_planning', 'release_research_planning'}:
        from .planning import planning_operation

        return planning_operation(store, operation, owner, payload, context)
    if operation == 'research_action':
        from .staged_research import handle_action
        return handle_action(store, owner, payload, context)
    if operation == "initialize_research":
        from .runtime_artifacts import settings
        if settings(store.coral_dir).get("staged_research") is True:
            raise WorkflowError("staged Research requires autoresearch research register with origin and design")
        node_id = payload["node_id"]
        request = {key: value for key, value in payload.items() if key != "node_id"}
        return store.initialize_research(owner, node_id, request, owner_identity=context)
    if operation == "claim_engineering":
        return store.claim_engineering(owner, payload["node_id"])
    if operation == "archive_evidence":
        from .attachments import archive

        return archive(store, owner, payload)
    if operation == "complete_assignment":
        return store.complete_assignment(owner, payload["mode"])
    if operation == "finish_assignment":
        if payload["mode"] == "engineering" and "analysis" in payload:
            return store.submit_engineering_analysis(owner, payload["analysis"])
        return store.request_finish(owner, payload["mode"])
    if operation == "cancel_assignment":
        return store.cancel_assignment(owner)
    raise WorkflowError("unsupported workflow operation")
