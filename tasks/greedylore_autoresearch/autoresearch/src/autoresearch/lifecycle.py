"""One-shot task-side finalization, driven by the existing Manager reconcile."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path

from coral.grader.trusted_worker import start_task_worker
from coral.grader.trusted_writes import trusted_writes_lock
from coral.hub.attempts import read_attempts

from .analysis import (
    AnalysisContext,
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
from .validation import load_document

logger = logging.getLogger(__name__)


def _publisher(store, item):
    def publish(kind, content, metadata):
        if metadata.get("node_id") != item.node_id or kind not in {"evaluation", "analysis"}:
            raise WorkflowError("finalizer can only publish its fixed Node artifacts")
        if kind == "analysis" and metadata.get("mode") != item.role:
            raise WorkflowError("finalizer Analysis mode does not match assignment")
        target = validate_artifact(kind, content, metadata)["relative_path"]
        with trusted_writes_lock(store.coral_dir):
            state = store._load_unlocked()
            if not any(
                current.agent_id == item.agent_id
                and current.node_id == item.node_id
                and current.role == item.role
                and current.status == "finalizing"
                for current in store._assignments(state)
            ):
                raise WorkflowError("finalizer assignment is no longer active")
            record = (
                state["nodes"][item.node_id]["research"]
                if item.role == "research"
                else state["engineering_workers"][item.agent_id]
            )
            if not record.get("finish_requested"):
                raise WorkflowError("finalizer requires a finish request")
            path = store.notes_root / target
            if kind == "evaluation":
                content = refresh_node_evaluation(
                    item.node_id, read_attempts(store.coral_dir), store.island_root / "eval_logs"
                )
            elif path.exists():
                if path.read_text(encoding="utf-8") != content:
                    raise WorkflowError("immutable Analysis already exists")
                return {"path": target}
            atomic_write_text(path, content)
        return {"path": target}

    return publish


def run_worker(
    coral_dir: str,
    *,
    runner=run_fresh_codex,
    code_snapshotter=_prepare_code_snapshot,
    memory_launcher=start_task_worker,
) -> dict:
    """Refresh projections, finalize ready assignments, then check Memory once."""
    store = WorkflowStore(coral_dir, repo_path=Path(coral_dir).resolve().parent / "repo")
    lock_path = store.coral_dir / "private/workflow-finalization.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "completed": [], "errors": {}}
        return _advance(store, runner, code_snapshotter, memory_launcher)


def _advance(store, runner, code_snapshotter, memory_launcher):
    completed, errors = [], {}
    with trusted_writes_lock(store.coral_dir):
        state = store._load_unlocked()
        assignments = store._assignments(state)
        # Refresh from latest Attempts while holding the same lock as Grader finalization.
        for node_id in state["nodes"]:
            if not (store.notes_root / "nodes" / node_id / "spec.md").is_file():
                continue
            content = refresh_node_evaluation(
                node_id, read_attempts(store.coral_dir), store.island_root / "eval_logs"
            )
            atomic_write_text(store.notes_root / "nodes" / node_id / "evaluation.md", content)
    for item in assignments:
        if item.status not in {"active", "finalizing"} or not item.node_id:
            continue
        try:
            if item.role == "engineering":
                with trusted_writes_lock(store.coral_dir):
                    record = store._load_unlocked()["engineering_workers"][item.agent_id]
                    if not record.get("finish_requested"):
                        continue
                    submission = record.get("analysis_submission")
                if not submission:
                    raise WorkflowError("Engineering requires an Engineer-authored summary; resume its owner to submit it")
                _publisher(store, item)("analysis", json.dumps(submission["document"], sort_keys=True, ensure_ascii=False) + "\n",
                                        {"node_id": item.node_id, "mode": "engineering", "version": submission["version"]})
                store.complete_assignment(item.agent_id, item.role)
                completed.append(item.agent_id)
                continue
            # Research automatically freezes only after a non-retryable terminal Attempt.
            try:
                store.request_finish(item.agent_id, item.role)
            except WorkflowError:
                if item.role == "research":
                    continue
                raise
            context = AnalysisContext(
                coral_dir=store.coral_dir,
                notes_root=store.notes_root,
                island_id=None,
                attempts=tuple(read_attempts(store.coral_dir)),
                log_root=store.island_root / "eval_logs",
                checkout_root=store.repo_path,
            )
            versions = list(
                (store.notes_root / "nodes" / item.node_id).glob(f"{item.role}_analysis_v[0-9]*.md")
            )
            latest = max(versions, key=lambda p: int(p.stem.rsplit("_v", 1)[1]), default=None)
            current = resolve_evidence(item.node_id, item.role, context=context)
            revise = latest is not None and (
                load_document(latest).get("evidence_manifest_hash") != current.manifest_hash
            )
            analyze(
                item.node_id,
                item.role,
                revise=revise,
                context=context,
                runner=runner,
                publisher=_publisher(store, item),
                code_snapshotter=code_snapshotter,
            )
            store.complete_assignment(item.agent_id, item.role)
            completed.append(item.agent_id)
        except Exception as exc:
            logger.exception("Finalization failed for %s", item.agent_id)
            errors[item.agent_id] = str(exc)
    # Also retry a previous Memory failure when no further Researcher starts occur.
    try:
        if memory_due(store.notes_root).due:
            memory_launcher(store.coral_dir, "autoresearch.memory:run_worker")
    except Exception as exc:
        logger.exception("Memory launch failed")
        errors["memory"] = str(exc)
    return {"status": "advanced", "completed": completed, "errors": errors}
