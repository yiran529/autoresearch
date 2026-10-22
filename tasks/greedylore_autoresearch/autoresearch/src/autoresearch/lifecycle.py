"""Research Analysis queue, independent of short Engineering publication."""

from __future__ import annotations

import fcntl
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from coral.grader.analysis_artifacts import publish_analysis
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
from .state import WorkflowError, WorkflowStore
from .validation import ArtifactValidationError, load_document, validate_analysis

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
               code_snapshotter=_prepare_code_snapshot,
               now=time.time, checkout_root=None, only_node=None) -> dict:
    """Supervise independent Node Analyses without blocking Engineering or retries."""
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
        return _run_parallel(store, completed, runner, code_snapshotter, now, only_node)


def _run_parallel(store, completed, runner, code_snapshotter, now, only_node):
    errors, deferred = {}, set()
    with trusted_writes_lock(store.coral_dir):
        count = len(store._load_unlocked()['nodes'])
    # The singleton supervisor does not impose an Analyzer role limit. Each
    # admitted Node gets an independent session and a nonblocking Node lock.
    with ThreadPoolExecutor(max_workers=store.max_active_nodes or max(1, count)) as pool:
        running = {}
        while True:
            for node_id, future in list(running.items()):
                if not future.done():
                    continue
                result = future.result()
                del running[node_id]
                completed.extend(result['completed'])
                errors.update(result['errors'])
                for agent_id in result['completed']:
                    errors.pop(agent_id, None)
                if result['status'] in {'locked', 'deferred'}:
                    deferred.add(node_id)
            with trusted_writes_lock(store.coral_dir):
                state = store._load_unlocked()
                admitted = store._admitted_node_ids_unlocked(state)
                nodes = sorted(state['nodes'], key=lambda n: state['nodes'][n].get('ready_seq', float('inf')))
                stopping = (store.coral_dir / 'private/workflow_stopping').exists()
                eligible = []
                for node_id in nodes:
                    if (stopping or node_id not in admitted or node_id in running
                            or node_id in deferred or only_node is not None and node_id != only_node):
                        continue
                    job = state['nodes'][node_id].get('analysis')
                    if not job or job['status'] in {'completed', 'blocked'}:
                        continue
                    path = store.notes_root / 'nodes' / node_id / f"research_analysis_v{job['version']}.md"
                    if now() >= job['next_retry_at'] or path.exists():
                        eligible.append(node_id)
            for node_id in eligible:
                running[node_id] = pool.submit(_run_node, store, node_id, runner, code_snapshotter, now)
            if not running:
                break
            # Keep servicing ready jobs and retries while another model is busy.
            wait(running.values(), timeout=0.2, return_when=FIRST_COMPLETED)
    return {'status': 'advanced', 'completed': completed, 'errors': errors}


def _run_node(store, node_id, runner, code_snapshotter, now):
    folder = store.coral_dir / 'private/analysis-locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{node_id}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'locked', 'completed': [], 'errors': {}}
        completed, errors = [], {}
        generation = None
        try:
            with trusted_writes_lock(store.coral_dir):
                state = store._load_unlocked()
                if ((store.coral_dir / 'private/workflow_stopping').exists()
                        or node_id not in store._admitted_node_ids_unlocked(state)):
                    return {"status": "deferred", "completed": [], "errors": {}}
                record = state["nodes"][node_id]
                job = record.get("analysis")
                if not job or job["status"] in {"completed", "blocked"}:
                    return {"status": "deferred", "completed": [], "errors": {}}
                path = store.notes_root / "nodes" / node_id / f"research_analysis_v{job['version']}.md"
                if path.exists():
                    store._adopt_analysis_unlocked(node_id, job)
                    store._save_unlocked(state)
                    completed.append(record["research"]["agent_id"])
                    return {"status": "advanced", "completed": completed, "errors": {}}
                if now() < job["next_retry_at"]:
                    return {"status": "deferred", "completed": [], "errors": {}}
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
                        code_snapshotter=code_snapshotter, validation_feedback=job.get('last_error'))
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
                    if isinstance(exc, ArtifactValidationError):
                        job['validation_failures'] = job.get('validation_failures', 0) + 1
                    job.update(status="pending", last_error=str(exc),
                               next_retry_at=now() + min(1800, 60 * 2 ** min(job["failures"] - 1, 5)))
                    if job.get('validation_failures', 0) >= 3:
                        job['status'] = 'blocked'
                    store._save_unlocked(state)
                errors[record["research"]["agent_id"]] = str(exc)
        return {'status': 'deferred' if errors and generation is None else 'advanced',
                'completed': completed, 'errors': errors}
