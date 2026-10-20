"""Research Analysis queue, independent of short Engineering publication."""

from __future__ import annotations

import fcntl
import json
import logging
import time
from pathlib import Path

from coral.grader.analysis_artifacts import publish_analysis
from coral.grader.trusted_worker import start_task_worker
from coral.grader.trusted_writes import trusted_writes_lock
from coral.hub.attempts import read_attempts

from .analysis import (
    _artifact_evidence,
    _prepare_code_snapshot,
    analyze,
    resolve_evidence,
    run_fresh_codex,
)
from .artifacts import validate_artifact
from .atomic import atomic_write_text
from .evaluation import refresh_node_evaluation
from .memory import memory_due
from .state import WorkflowError, WorkflowStore
from .validation import load_document, validate_analysis

logger = logging.getLogger(__name__)


def _publisher(store, node_id, generation):
    def publish(kind, content, metadata):
        if metadata.get("node_id") != node_id or kind not in {"evaluation", "analysis"}:
            raise WorkflowError("Analyzer can only publish its fixed Node artifacts")
        if kind == "analysis" and metadata.get("mode") != "research":
            raise WorkflowError("Analyzer may only publish Research Analysis")
        target = validate_artifact(kind, content, metadata)["relative_path"]
        with trusted_writes_lock(store.coral_dir):
            state = store._load_unlocked()
            job = state["nodes"][node_id]["analysis"]
            if job["status"] != "running" or job["generation"] != generation:
                raise WorkflowError("Analyzer execution is no longer current")
            evidence = resolve_evidence(node_id, "research", context=store._context())
            if evidence.manifest_hash != job["evidence_hash"]:
                raise WorkflowError("Analysis job evidence changed")
            if kind == "analysis":
                document = json.loads(content)
                validate_analysis(document, _artifact_evidence(document, evidence, store._context()))
            path = store.notes_root / target
            if kind == "evaluation":
                # The public projection includes Engineering, even when this Analyzer
                # started before its Attempts. Research input remains a fixed snapshot.
                content = refresh_node_evaluation(node_id, read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True),
                                                  store.island_root / "eval_logs")
            elif metadata.get("version") != job["version"]:
                raise WorkflowError("Analysis version does not match its fixed job")
            elif path.exists():
                if load_document(path) != json.loads(content):
                    raise WorkflowError("immutable Analysis already exists")
                publish_analysis(path, json.loads(content))
                return {"path": target}
            if kind == "analysis":
                # A crash after this write can publish without another model session.
                sealed = store.coral_dir / "private/analysis_outputs" / f"{node_id}.json"
                atomic_write_text(sealed, content)
            if kind == "analysis":
                publish_analysis(path, document)
            else:
                atomic_write_text(path, content, mode=0o644)
        return {"path": target}

    return publish


def run_worker(coral_dir: str, *, runner=run_fresh_codex,
               code_snapshotter=_prepare_code_snapshot, memory_launcher=start_task_worker,
               now=time.time, checkout_root=None, only_node=None) -> dict:
    """One pass of eligible Analysis jobs. Never serialize Engineering behind Codex."""
    store = WorkflowStore(coral_dir, repo_path=checkout_root or Path(coral_dir).resolve().parent / "repo")
    lock_path = store.coral_dir / "private/workflow-finalization.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "completed": [], "errors": {}}
        completed = store.advance()
        with trusted_writes_lock(store.coral_dir):
            state = store._load_unlocked()
            nodes = sorted(state["nodes"], key=lambda n: state["nodes"][n].get("ready_seq", float("inf")))
            # This projection is also refreshed while waiting for formal retries.
            for node_id in nodes:
                if (store.notes_root / "nodes" / node_id / "spec.md").exists():
                    atomic_write_text(store.notes_root / "nodes" / node_id / "evaluation.md",
                                      refresh_node_evaluation(node_id, read_attempts(store.coral_dir, island_id=store.island_id, include_archived=True),
                                                              store.island_root / "eval_logs"), mode=0o644)
        errors = {}
        for node_id in nodes:
            if only_node is not None and node_id != only_node:
                continue
            generation = None
            try:
                with trusted_writes_lock(store.coral_dir):
                    state = store._load_unlocked()
                    record = state["nodes"][node_id]
                    job = record.get("analysis")
                    if not job or job["status"] == "completed":
                        continue
                    path = store.notes_root / "nodes" / node_id / f"research_analysis_v{job['version']}.md"
                    if path.exists():
                        store._adopt_analysis_unlocked(node_id, job)
                        store._save_unlocked(state)
                        completed.append(record["research"]["agent_id"])
                        continue
                    if now() < job["next_retry_at"]:
                        continue
                    job["generation"] += 1
                    generation = job["generation"]
                    job["status"] = "running"
                    store._save_unlocked(state)
                    context = store._context()
                publish = _publisher(store, node_id, generation)
                sealed = store.coral_dir / "private/analysis_outputs" / f"{node_id}.json"
                if sealed.exists():
                    publish("analysis", sealed.read_text(),
                            {"node_id": node_id, "mode": "research", "version": job["version"]})
                else:
                    analyze(node_id, "research", context=context, runner=runner, publisher=publish,
                            code_snapshotter=code_snapshotter)
                with trusted_writes_lock(store.coral_dir):
                    state = store._load_unlocked()
                    record = state["nodes"][node_id]
                    job = record["analysis"]
                    if job["generation"] != generation or job["status"] != "running":
                        raise WorkflowError("Analyzer execution is no longer current")
                    store._adopt_analysis_unlocked(node_id, job)
                    store._save_unlocked(state)
                    completed.append(record["research"]["agent_id"])
            except Exception as exc:
                logger.exception("Analysis failed for %s", node_id)
                with trusted_writes_lock(store.coral_dir):
                    state = store._load_unlocked()
                    record = state["nodes"][node_id]
                    job = record.get("analysis")
                    if job and job["generation"] == generation:
                        job["failures"] += 1
                        job.update(status="pending", last_error=str(exc),
                                   next_retry_at=now() + min(1800, 60 * 2 ** min(job["failures"] - 1, 5)))
                        store._save_unlocked(state)
                    errors[record["research"]["agent_id"]] = str(exc)
        try:
            if only_node is None and memory_due(store.notes_root).due:
                memory_launcher(store.coral_dir, "autoresearch.memory:run_worker")
        except Exception as exc:
            logger.exception("Memory launch failed")
            errors["memory"] = str(exc)
        return {"status": "advanced", "completed": completed, "errors": errors}
