"""Authoritative, run-local ownership state for dynamic AutoResearch agents."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock
from coral.hub._island import island_root
from coral.hub.attempts import read_attempts

from .analysis import AnalysisContext, _artifact_evidence, resolve_evidence
from .atomic import atomic_write_text
from .nodes import create_node
from .validation import load_document, validate_analysis


class WorkflowError(ValueError):
    """An ownership transition or persisted workflow state is invalid."""


@dataclass(frozen=True)
class WorkflowAssignment:
    agent_id: str
    role: str
    node_id: str | None
    status: str


class WorkflowStore:
    """The small authoritative state document for one CORAL run."""

    def __init__(
        self, coral_dir: str | Path, island_id: str | None = None,
        *, repo_path: str | Path | None = None,
        max_engineering_runs_per_node: int = 1,
    ) -> None:
        self.repo_path = Path(repo_path).resolve() if repo_path is not None else None
        self.max_engineering_runs_per_node = max_engineering_runs_per_node
        self.coral_dir = Path(coral_dir).resolve()
        self.island_id = island_id
        self.island_root = island_root(self.coral_dir, island_id)
        self.notes_root = self.island_root / "notes"
        self.state_path = self.coral_dir / "private" / "autoresearch_state.json"
        self.tree_path = self.notes_root / "tree.md"

    def _verify_root(self) -> None:
        node_root = self.notes_root / "nodes" / "N000"
        spec_path = node_root / "spec.md"
        analysis_path = node_root / "research_analysis_v1.md"
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
            or attempts[0].status == "pending"
            or attempts[0].parent_hash != policy["source_commit"]
            or attempts[0].metadata.get("submission_hash") != policy["submission_hash"]
        ):
            raise WorkflowError("Root Analysis bootstrap requires one terminal N000 Research Attempt")
        if not analysis_path.is_file():
            raise WorkflowError("Root Analysis bootstrap is required before dynamic agents start")
        context = AnalysisContext(
            coral_dir=self.coral_dir,
            notes_root=self.notes_root,
            island_id=self.island_id,
            attempts=attempts,
            log_root=self.island_root / "eval_logs",
            checkout_root=self.coral_dir.parent,
        )
        try:
            evidence = resolve_evidence("N000", "research", context=context)
            document = load_document(analysis_path)
            evidence = _artifact_evidence(document, evidence, context)
            validate_analysis(document, evidence)
        except ValueError as exc:
            raise WorkflowError("Root Analysis bootstrap has invalid fixed evidence") from exc

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
                "version": 1,
                "revision": 0,
                "next_node_seq": 1,
                "next_engineer_seq": 1,
                "nodes": {
                    "N000": {
                        "research": {
                            "agent_id": "root-bootstrap",
                            "status": "completed",
                            "spec_ref": "nodes/N000/spec.md",
                        },
                        "engineering_claims": [],
                    }
                },
                "engineering_workers": {},
            }
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkflowError("workflow state is invalid") from exc
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or not isinstance(state.get("revision"), int)
            or not isinstance(state.get("next_node_seq"), int)
            or not isinstance(state.get("next_engineer_seq"), int)
            or not isinstance(state.get("nodes"), dict)
            or not isinstance(state.get("engineering_workers"), dict)
            or state["nodes"].get("N000", {}).get("research", {}).get("status") != "completed"
        ):
            raise WorkflowError("workflow state is invalid")
        return state

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        state["revision"] += 1
        _atomic_json(self.state_path, state)
        self._rebuild_tree_unlocked(state)

    def _rebuild_tree_unlocked(self, state: dict[str, Any]) -> None:
        rows = ["# Research Tree", "", "| Node | Research | Engineering |", "| --- | --- | --- |"]
        for node_id, record in sorted(state["nodes"].items()):
            claims = record.get("engineering_claims", [])
            engineering = claims[-1]["status"] if claims else "—"
            rows.append(f"| {node_id} | {record['research']['status']} | {engineering} |")
        atomic_write_text(self.tree_path, "\n".join(rows) + "\n")

    def rebuild_tree(self) -> Path:
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            self._rebuild_tree_unlocked(state)
        return self.tree_path

    @staticmethod
    def _assignments(state: dict[str, Any]) -> tuple[WorkflowAssignment, ...]:
        assignments = [
            WorkflowAssignment(record["research"]["agent_id"], "research", node_id,
                               "finalizing" if record["research"].get("finish_requested")
                               else record["research"]["status"])
            for node_id, record in sorted(state["nodes"].items())
            if node_id != "N000" and record["research"]["status"] in {"reserved", "active"}
        ]
        assignments.extend(
            WorkflowAssignment(agent_id, "engineering", worker.get("node_id"),
                               "finalizing" if worker.get("finish_requested") else worker["status"])
            for agent_id, worker in sorted(state["engineering_workers"].items())
            if worker["status"] in {"reserved", "active"}
        )
        return tuple(assignments)

    def _qualified_research_unlocked(self, state: dict[str, Any], node_id: str) -> str | None:
        record = state["nodes"].get(node_id)
        if node_id == "N000" or not isinstance(record, dict):
            return None
        if record["research"]["status"] != "completed":
            return None
        if len(record["engineering_claims"]) >= self.max_engineering_runs_per_node:
            return None
        attempts = [
            attempt for attempt in read_attempts(self.coral_dir, island_id=self.island_id)
            if attempt.metadata.get("node_id") == node_id
            and attempt.metadata.get("mode") == "research"
            and attempt.budget_class == "real"
        ]
        if len(attempts) != 1:
            return None
        attempt = attempts[0]
        if attempt.status in {"pending", "crashed", "timeout"}:
            return None
        quality = attempt.metadata.get("scores", {}).get("quality_gate", {})
        if not isinstance(quality, dict) or quality.get("value") != 1.0:
            return None
        if self.repo_path is None or not self.repo_path.is_dir():
            return None
        check = subprocess.run(
            ["git", "cat-file", "-e", attempt.commit_hash + "^{commit}"],
            cwd=self.repo_path, capture_output=True, check=False,
        )
        return attempt.commit_hash if check.returncode == 0 else None

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
        with trusted_writes_lock(self.coral_dir):
            self._verify_root()
            toolkit = self.notes_root / "toolkit"
            toolkit.mkdir(parents=True, exist_ok=True)
            task_root = Path(__file__).resolve().parents[3]
            shutil.copyfile(task_root / "roles/workflow_rules.md", toolkit / "workflow_rules.md")
            shutil.copytree(task_root / "references", toolkit / "references", dirs_exist_ok=True)
            state = self._load_unlocked()
            live_research = sum(
                record["research"]["status"] in {"reserved", "active"}
                for node_id, record in state["nodes"].items()
                if node_id != "N000"
            )
            changed = not self.state_path.is_file()
            while live_research < max_researchers:
                sequence = state["next_node_seq"]
                if sequence > 999:
                    raise WorkflowError("Node ID space is exhausted")
                node_id = f"N{sequence:03d}"
                if node_id in state["nodes"]:
                    raise WorkflowError("Node ID counter conflicts with existing state")
                state["nodes"][node_id] = {
                    "research": {"agent_id": f"research-{node_id}", "status": "reserved",
                                 "spec_ref": None},
                    "engineering_claims": [],
                }
                state["next_node_seq"] = sequence + 1
                live_research += 1
                changed = True
            eligible = sum(
                self._qualified_research_unlocked(state, node_id) is not None
                for node_id in state["nodes"]
            )
            live_engineers = sum(
                worker["status"] in {"reserved", "active"}
                for worker in state["engineering_workers"].values()
            )
            unbound_engineers = sum(
                worker["status"] == "reserved" and worker.get("node_id") is None
                for worker in state["engineering_workers"].values()
            )
            while live_engineers < max_engineers and unbound_engineers < eligible:
                sequence = state["next_engineer_seq"]
                agent_id = f"engineer-E{sequence:04d}"
                if agent_id in state["engineering_workers"]:
                    raise WorkflowError("Engineer ID counter conflicts with existing state")
                state["engineering_workers"][agent_id] = {
                    "status": "reserved", "node_id": None,
                }
                state["next_engineer_seq"] = sequence + 1
                live_engineers += 1
                unbound_engineers += 1
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
                return {
                    "agent_id": agent_id, "node_id": node_id,
                    "status": "active", "research_sha": worker["research_sha"],
                }
            if worker["status"] != "reserved" or worker.get("node_id") is not None:
                raise WorkflowError("Engineering assignment is not available")
            research_sha = self._qualified_research_unlocked(state, node_id)
            if research_sha is None:
                raise WorkflowError(f"Node {node_id} is not eligible for Engineering")
            claim = {
                "agent_id": agent_id, "status": "active",
                "research_sha": research_sha,
            }
            state["nodes"][node_id]["engineering_claims"].append(claim)
            worker.update({"status": "active", "node_id": node_id, "research_sha": research_sha})
            self._save_unlocked(state)
            return {
                "agent_id": agent_id, "node_id": node_id,
                "status": "active", "research_sha": research_sha,
            }


    def cancel_assignment(self, agent_id: str) -> dict[str, Any]:
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            worker = state["engineering_workers"].get(agent_id)
            if isinstance(worker, dict):
                if worker["status"] != "reserved" or worker.get("node_id") is not None:
                    raise WorkflowError("only an unstarted reservation can be cancelled")
                if any(
                    self._qualified_research_unlocked(state, node_id) is not None
                    for node_id in state["nodes"]
                ):
                    raise WorkflowError("an eligible Engineering target still exists")
                worker["status"] = "cancelled"
                self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": None, "status": "cancelled"}
            for node_id, record in state["nodes"].items():
                research = record["research"]
                if node_id == "N000" or research["agent_id"] != agent_id:
                    continue
                if research["status"] != "reserved":
                    raise WorkflowError("only an unstarted reservation can be cancelled")
                research["status"] = "cancelled"
                self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": node_id, "status": "cancelled"}
            raise WorkflowError("assignment owner was not found")


    def _terminal_research_unlocked(self, node_id: str):
        attempts = [
            attempt for attempt in read_attempts(self.coral_dir, island_id=self.island_id)
            if attempt.metadata.get("node_id") == node_id
            and attempt.metadata.get("mode") == "research"
            and attempt.budget_class == "real"
        ]
        if len(attempts) != 1:
            raise WorkflowError(f"Node {node_id} requires one formal Research Attempt")
        attempt = attempts[0]
        if attempt.status == "pending" or (
            attempt.status in {"crashed", "timeout"}
            and attempt.metadata.get("failure_class") == "infra"
            and int(attempt.metadata.get("infra_retry_count", 0))
            < int(attempt.metadata.get("max_infra_retries", 0))
        ):
            raise WorkflowError(f"Node {node_id} Research Attempt is not terminal")
        return attempt

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
            attempts=tuple(read_attempts(self.coral_dir, island_id=self.island_id)),
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
        self, agent_id: str, node_id: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping) or set(request) != {
            "parent_node", "hypothesis", "selection_rationale",
            "delta", "risk", "falsification_criteria",
        }:
            raise WorkflowError("Research initialization has invalid fields")
        with trusted_writes_lock(self.coral_dir):
            state = self._load_unlocked()
            record = state["nodes"].get(node_id)
            if not isinstance(record, dict) or record["research"]["agent_id"] != agent_id:
                raise WorkflowError("Research reservation owner does not match")
            research = record["research"]
            if research["status"] not in {"reserved", "active"}:
                raise WorkflowError("Research reservation is not active")
            parent_node = request["parent_node"]
            parent = state["nodes"].get(parent_node)
            if not isinstance(parent, dict) or parent["research"]["status"] != "completed":
                raise WorkflowError("selected Parent Research is not completed")
            parent_attempt = self._terminal_research_unlocked(parent_node)
            analysis_ref = self._validated_analysis_ref_unlocked(parent_node, "research")
            target = self.notes_root / "nodes" / node_id / "spec.md"

            def publish(kind: str, content: str, metadata: dict[str, Any]) -> dict[str, str]:
                if kind != "spec" or metadata["node_id"] != node_id:
                    raise WorkflowError("unexpected Node Spec publication")
                if target.exists():
                    if target.read_text(encoding="utf-8") != content:
                        raise WorkflowError("immutable Node Spec conflicts with reservation")
                else:
                    atomic_write_text(target, content)
                return {"path": f"nodes/{node_id}/spec.md"}

            evidence = {
                "notes_root": self.notes_root,
                "parent_node": parent_node,
                "parent_commit": parent_attempt.commit_hash,
                "base_commit": parent_attempt.commit_hash,
                "analysis_ref": analysis_ref,
                "engineering_analysis_ref": None,
                "publish": publish,
            }
            create_node({**request, "node_id": node_id}, evidence)
            if research["status"] == "reserved":
                research.update({"status": "active", "spec_ref": f"nodes/{node_id}/spec.md"})
                self._save_unlocked(state)
            return {"agent_id": agent_id, "node_id": node_id,
                    "status": "active", "spec_ref": f"nodes/{node_id}/spec.md",
                    "base_commit": parent_attempt.commit_hash}

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
                    self._terminal_research_unlocked(node_id)
                    self._validated_analysis_ref_unlocked(node_id, "research")
                    research["status"] = "completed"
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
                self._terminal_engineering_unlocked(agent_id, node_id)
                self._validated_analysis_ref_unlocked(node_id, "engineering")
                claims = state["nodes"][node_id]["engineering_claims"]
                match = [claim for claim in claims if claim["agent_id"] == agent_id]
                if len(match) != 1 or match[0]["status"] != "active":
                    raise WorkflowError("Engineering claim does not match active owner")
                match[0]["status"] = "completed"
                worker["status"] = "completed"
                self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": node_id, "status": "completed"}
            raise WorkflowError("assignment owner was not found")

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
                    self._terminal_research_unlocked(item.node_id)
                    record = state["nodes"][item.node_id]["research"]
                else:
                    self._terminal_engineering_unlocked(agent_id, item.node_id)
                    record = state["engineering_workers"][agent_id]
                if not record.get("finish_requested"):
                    record["finish_requested"] = True
                    self._save_unlocked(state)
                return {"agent_id": agent_id, "node_id": item.node_id, "status": "finalizing"}
            raise WorkflowError("active assignment owner was not found")

    def _terminal_engineering_unlocked(self, agent_id: str, node_id: str) -> None:
        attempts = [
            attempt for attempt in read_attempts(self.coral_dir, island_id=self.island_id)
            if attempt.agent_id == agent_id and attempt.metadata.get("node_id") == node_id
            and attempt.metadata.get("mode") == "engineering" and attempt.budget_class == "real"
        ]
        if not attempts:
            raise WorkflowError(f"Node {node_id} requires an Engineering Attempt")
        if any(attempt.status == "pending" or (
            attempt.status in {"crashed", "timeout"}
            and attempt.metadata.get("failure_class") == "infra"
            and int(attempt.metadata.get("infra_retry_count", 0))
            < int(attempt.metadata.get("max_infra_retries", 0))
        ) for attempt in attempts):
            raise WorkflowError("Engineering Attempt is not terminal")



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
    _reject_finalizing(state, context.get("agent_id"), canonical.get("node_id"), canonical.get("mode"))
    return {}


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
        read_attempts(store.coral_dir, island_id=store.island_id),
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
        assignments = store.reconcile(
            context["max_researchers"], context["max_engineers"]
        )
        return {"assignments": [asdict(item) for item in assignments]}
    if context.get("is_manager") or not isinstance(owner, str):
        raise WorkflowError("workflow mutation requires an authenticated Agent")
    if operation == "initialize_research":
        node_id = payload["node_id"]
        request = {key: value for key, value in payload.items() if key != "node_id"}
        return store.initialize_research(owner, node_id, request)
    if operation == "claim_engineering":
        return store.claim_engineering(owner, payload["node_id"])
    if operation == "complete_assignment":
        return store.complete_assignment(owner, payload["mode"])
    if operation == "finish_assignment":
        return store.request_finish(owner, payload["mode"])
    if operation == "cancel_assignment":
        return store.cancel_assignment(owner)
    raise WorkflowError("unsupported workflow operation")
