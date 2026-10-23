"""Minimal trusted submit, retry, and artifact publication for task workflows."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coral.grader.timeouts import DIAGNOSTIC_WORKER_TIMEOUT, TRUSTED_CLIENT_TIMEOUT
from coral.hub._island import island_root
from coral.hub.attempts import read_attempt, read_attempts, write_attempt
from coral.hub.checkpoint import checkpoint
from coral.types import BUDGET_CLASS_REAL, BUDGET_CLASS_TUNE, Attempt

PROTOCOL_VERSION = 1
logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024
WORKFLOW_OPERATIONS = frozenset({
    "reconcile_workflow", "initialize_research", "claim_engineering",
    "complete_assignment", "cancel_assignment", "finish_assignment", "archive_evidence",
    "acquire_research_planning", "renew_research_planning", "release_research_planning",
})
OPERATIONS = frozenset({
    "submit_attempt", "retry_attempt", "publish_artifact", "start_memory_if_due", "run_diagnostic",
}) | WORKFLOW_OPERATIONS
_WORKFLOW_FIELDS = {
    "reconcile_workflow": frozenset(),
    "initialize_research": frozenset({"node_id", "parent_node", "hypothesis", "selection_rationale", "delta", "risk", "falsification_criteria"}),
    "claim_engineering": frozenset({"node_id"}),
    "complete_assignment": frozenset({"mode"}),
    "finish_assignment": frozenset({"mode"}),
    "cancel_assignment": frozenset(),
    "archive_evidence": frozenset({"node_id", "request_id", "commit_sha", "files"}),
    "acquire_research_planning": frozenset(),
    "renew_research_planning": frozenset({"token"}),
    "release_research_planning": frozenset({"token"}),
}
_SHA = re.compile(r"^[0-9a-f]{40}$")
_NODE_ARTIFACT = re.compile(
    r"^nodes/N[0-9]{3}/(?:spec|evaluation|(?:research|engineering)_(?:analysis|review)_v[1-9][0-9]*)\.md$"
)


class TrustedWriteError(RuntimeError):
    """A trusted-write request was malformed or unauthorized."""


class DispatchClaimError(TrustedWriteError):
    """A pending attempt cannot be claimed by this grader daemon."""


@dataclass(frozen=True)
class AgentBinding:
    agent_id: str
    root_pid: int
    start_time: int
    worktree: str
    island_id: str | None


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _proc_stat(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        raise TrustedWriteError(f"cannot inspect process {pid}") from exc


def process_start_time(pid: int, proc_root: Path = Path("/proc")) -> int:
    return _proc_stat(pid, proc_root)[1]


def _registry_path(coral_dir: str | Path) -> Path:
    return Path(coral_dir) / "private" / "trusted_agents.json"


def _atomic_json(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_trusted_agent_registry(
    coral_dir: str | Path, bindings: list[AgentBinding]
) -> Path:
    """Atomically publish manager-owned live Agent bindings under private/."""
    path = _registry_path(coral_dir)
    normalized = []
    for binding in bindings:
        item = asdict(binding)
        item["worktree"] = str(Path(binding.worktree).resolve())
        normalized.append(item)
    _atomic_json(path, {"version": 1, "agents": normalized})
    return path


def clear_trusted_agent_registry(coral_dir: str | Path) -> None:
    try:
        _registry_path(coral_dir).unlink()
    except FileNotFoundError:
        pass


def _read_registry(coral_dir: str | Path) -> list[AgentBinding]:
    path = _registry_path(coral_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("agents"), list):
            raise ValueError("invalid registry envelope")
        return [AgentBinding(**item) for item in payload["agents"]]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TrustedWriteError("trusted Agent registry is unavailable or invalid") from exc


def resolve_peer_binding(
    coral_dir: str | Path,
    peer_pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> AgentBinding:
    """Map a kernel-authenticated peer PID to a live registered Agent ancestor."""
    bindings = {binding.root_pid: binding for binding in _read_registry(coral_dir)}
    current = peer_pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        visited.add(current)
        try:
            parent, start_time = _proc_stat(current, proc_root)
        except TrustedWriteError:
            break
        binding = bindings.get(current)
        if binding is not None:
            if binding.start_time != start_time:
                raise TrustedWriteError("registered Agent PID start time no longer matches")
            if not Path(binding.worktree).is_absolute():
                raise TrustedWriteError("registered Agent worktree is not absolute")
            return binding
        current = parent
    raise TrustedWriteError("peer process does not belong to a registered Agent")


@contextmanager
def trusted_writes_lock(coral_dir: str | Path) -> Iterator[None]:
    path = Path(coral_dir) / "private" / "trusted_writes.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _copy_attempt(attempt: Attempt, **changes: Any) -> Attempt:
    values = {
        "commit_hash": attempt.commit_hash,
        "agent_id": attempt.agent_id,
        "title": attempt.title,
        "score": attempt.score,
        "status": attempt.status,
        "parent_hash": attempt.parent_hash,
        "timestamp": attempt.timestamp,
        "feedback": attempt.feedback,
        "shared_state_hash": attempt.shared_state_hash,
        "parent_shared_state_hash": attempt.parent_shared_state_hash,
        "metadata": dict(attempt.metadata or {}),
    }
    values.update(changes)
    return Attempt(**values)


def _read_attempt_unlocked(
    coral_dir: str | Path, commit_hash: str, island_id: str | int | None
) -> Attempt:
    attempt = read_attempt(coral_dir, commit_hash, island_id=island_id)
    if attempt is None:
        raise TrustedWriteError(f"attempt not found: {commit_hash}")
    return attempt


def retry_attempt(
    coral_dir: str | Path,
    commit_hash: str,
    island_id: str | int | None,
    *,
    expected_agent_id: str,
    expected_contract_hash: str,
    max_infra_retries: int,
    expected_source_worktree: str | Path | None = None,
    authorize: Callable[[], Any] | None = None,
) -> Attempt:
    """Atomically reset one terminal infrastructure failure to pending."""
    with trusted_writes_lock(coral_dir):
        attempt = _read_attempt_unlocked(coral_dir, commit_hash, island_id)
        if authorize is not None:
            authorize()
        if expected_source_worktree is not None:
            worktree = Path(expected_source_worktree).resolve()
            recorded = attempt.metadata.get("source_worktree")
            if not isinstance(recorded, str) or Path(recorded).resolve() != worktree:
                raise TrustedWriteError("retry source worktree does not match the submitted attempt")
            if _git(worktree, "rev-parse", "HEAD") != commit_hash:
                raise TrustedWriteError("retry commit is not the authenticated worktree current HEAD")
        if attempt.agent_id != expected_agent_id:
            raise TrustedWriteError("only the originating Agent may retry this attempt")
        if attempt.metadata.get("contract_hash") != expected_contract_hash:
            raise TrustedWriteError("evaluation contract changed since the failed run")
        if attempt.status not in {"crashed", "timeout"}:
            raise TrustedWriteError(f"attempt is not a terminal failure: {attempt.status}")
        if attempt.metadata.get("failure_class") != "infra":
            raise TrustedWriteError("only failure_class=infra attempts may be retried")
        retry_count = int(attempt.metadata.get("infra_retry_count", 0))
        if retry_count >= max_infra_retries:
            raise TrustedWriteError("infrastructure retry budget is exhausted")
        metadata = dict(attempt.metadata or {})
        metadata["infra_retry_count"] = retry_count + 1
        retried = _copy_attempt(
            attempt,
            score=None,
            status="pending",
            feedback="",
            metadata=metadata,
        )
        write_attempt(coral_dir, retried, island_id=island_id)
        return retried


def begin_grader_run(
    coral_dir: str | Path,
    commit_hash: str,
    island_id: str | int | None,
    *,
    dispatch_validator: Callable[[Attempt], bool] | None = None,
) -> tuple[Attempt, str]:
    """Append one running record before an actual grader invocation."""
    with trusted_writes_lock(coral_dir):
        attempt = _read_attempt_unlocked(coral_dir, commit_hash, island_id)
        if attempt.status != "pending" or attempt.score is not None:
            raise TrustedWriteError("only a pending attempt can be dispatched")
        if dispatch_validator is not None:
            try:
                allowed = dispatch_validator(attempt)
            except Exception as exc:
                raise DispatchClaimError('cannot verify dispatch eligibility') from exc
            if not allowed:
                raise DispatchClaimError('Node is waiting for dispatch eligibility')
        metadata = dict(attempt.metadata or {})
        runs = list(metadata.get("grader_runs") or [])
        for index, existing in enumerate(runs):
            if existing.get("status") != "running":
                continue
            pid = existing.get("dispatcher_pid")
            start_time = existing.get("dispatcher_start_time")
            live = False
            if isinstance(pid, int) and isinstance(start_time, int):
                try:
                    live = process_start_time(pid) == start_time
                except TrustedWriteError:
                    live = False
            if live:
                raise DispatchClaimError("attempt is already running")
            stale = dict(existing)
            stale.update(
                {
                    "finished_at": _utcnow(),
                    "status": "crashed",
                    "failure_class": "infra",
                }
            )
            runs[index] = stale
            metadata["grader_runs"] = runs
            metadata["failure_class"] = "infra"
            crashed = _copy_attempt(
                attempt,
                score=None,
                status="crashed",
                feedback="Stale grader dispatch detected; explicit retry is required.",
                metadata=metadata,
            )
            write_attempt(coral_dir, crashed, island_id=island_id)
            raise DispatchClaimError("stale dispatch finalized; explicit retry is required")
        run_id = uuid.uuid4().hex
        dispatcher_pid = os.getpid()
        runs.append(
            {
                "run_id": run_id,
                "started_at": _utcnow(),
                "finished_at": None,
                "status": "running",
                "failure_class": None,
                "contract_hash": metadata.get("contract_hash"),
                "log_dir": f"eval_logs/{commit_hash}/{run_id}",
                "dispatcher_pid": dispatcher_pid,
                "dispatcher_start_time": process_start_time(dispatcher_pid),
                "resource_usage": None,
                "gpu_uuids": None,
            }
        )
        metadata["grader_runs"] = runs
        metadata["grader_invocation_count"] = int(metadata.get("grader_invocation_count", 0)) + 1
        dispatched = _copy_attempt(attempt, metadata=metadata)
        write_attempt(coral_dir, dispatched, island_id=island_id)
        return dispatched, run_id


def finalize_grader_run(
    coral_dir: str | Path,
    commit_hash: str,
    island_id: str | int | None,
    run_id: str,
    *,
    status: str,
    failure_class: str | None,
    score: float | None,
    feedback: str,
    metadata_updates: dict[str, Any] | None = None,
    metadata_remove: set[str] | None = None,
) -> Attempt:
    """Finalize exactly one run record without losing concurrent metadata."""
    with trusted_writes_lock(coral_dir):
        attempt = _read_attempt_unlocked(coral_dir, commit_hash, island_id)
        metadata = dict(attempt.metadata or {})
        runs = list(metadata.get("grader_runs") or [])
        matches = [index for index, run in enumerate(runs) if run.get("run_id") == run_id]
        if len(matches) != 1 or runs[matches[0]].get("status") != "running":
            raise TrustedWriteError("grader run is missing or already finalized")
        run = dict(runs[matches[0]])
        run.update(
            {
                "finished_at": _utcnow(),
                "status": status,
                "failure_class": failure_class,
            }
        )
        updates = dict(metadata_updates or {})
        for key in ("resource_usage", "gpu_uuids"):
            if key in updates:
                run[key] = updates[key]
        runs[matches[0]] = run
        for key in metadata_remove or set():
            metadata.pop(key, None)
        metadata.update(updates)
        metadata["grader_runs"] = runs
        if failure_class is None:
            metadata.pop("failure_class", None)
        else:
            metadata["failure_class"] = failure_class
        finalized = _copy_attempt(
            attempt,
            score=score,
            status=status,
            feedback=feedback,
            metadata=metadata,
        )
        write_attempt(coral_dir, finalized, island_id=island_id)
        return finalized


def _git(worktree: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=worktree, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise TrustedWriteError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _all_attempts(coral_dir: Path, island_id: str | None) -> list[Attempt]:
    if island_id is not None or not (coral_dir / "islands").is_dir():
        return read_attempts(coral_dir, island_id=island_id)
    attempts: list[Attempt] = []
    for child in sorted((coral_dir / "islands").iterdir()):
        if child.is_dir():
            attempts.extend(read_attempts(coral_dir, island_id=child.name))
    return attempts


class TrustedWriteService:
    """Three fixed task operations; no provider registry or arbitrary routing."""

    def __init__(
        self,
        coral_dir: str | Path,
        config: Any,
        *,
        identity_resolver: Callable[[Path, int], AgentBinding] = resolve_peer_binding,
        submission_validator: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        artifact_validator: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
        checkpoint_fn: Callable[..., str | None] = checkpoint,
        workflow_handler: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        attempt_authorizer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        artifact_authorizer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        memory_authorizer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        memory_launcher: Callable[[Path, str], Any] | None = None,
        diagnostic_handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.coral_dir = Path(coral_dir).resolve()
        self.config = config
        self.identity_resolver = identity_resolver
        self.submission_validator = submission_validator
        self.artifact_validator = artifact_validator
        self.checkpoint_fn = checkpoint_fn
        self.workflow_handler = workflow_handler
        self.attempt_authorizer = attempt_authorizer
        self.artifact_authorizer = artifact_authorizer
        self.memory_authorizer = memory_authorizer
        self.memory_launcher = memory_launcher
        self.diagnostic_handler = diagnostic_handler
        self._workflow_process = None
        self._memory_process = None
        self._workflow_stopping = False
        self._workflow_recovery_error: str | None = None
        self._workflow_lifecycle_lock = threading.Lock()

    def stop_workflow(self) -> None:
        from coral.grader.trusted_worker import stop_workflow_worker

        with self._workflow_lifecycle_lock:
            self._workflow_stopping = True
            stop_workflow_worker(self.coral_dir)
            if self._workflow_process is not None:
                self._workflow_process.wait(timeout=5)

    def _validator_entrypoint(self, name: str) -> str:
        args = getattr(getattr(self.config, "grader", None), "args", {}) or {}
        value = args.get(name, "")
        if not isinstance(value, str) or not value:
            raise TrustedWriteError(f"{name} is not configured")
        return value

    def _authorize(
        self, name: str, callback: Callable[..., dict[str, Any]] | None,
        *arguments: Any,
    ) -> dict[str, Any] | None:
        configured = getattr(getattr(self.config, "grader", None), "args", {}) or {}
        entrypoint = configured.get(name)
        if callback is None and not entrypoint:
            return None
        if callback is not None:
            result = callback(*arguments)
        else:
            from coral.grader.trusted_worker import invoke_task_validator

            result = invoke_task_validator(self.coral_dir, self._validator_entrypoint(name), *arguments)
        if not isinstance(result, dict):
            raise TrustedWriteError(f"{name} returned a non-object result")
        return result


    def _validate_submission(
        self,
        commit_hash: str,
        binding: AgentBinding,
        parent_hash: str | None = None,
    ) -> dict[str, Any]:
        context = {
            "worktree": binding.worktree,
            "coral_dir": str(self.coral_dir),
            "contract_path": str(self.coral_dir / "private" / "eval_contract.yaml"),
            "agent_id": binding.agent_id,
            "island_id": binding.island_id,
            "parent_hash": parent_hash,
        }
        if self.submission_validator is not None:
            result = self.submission_validator(commit_hash, context)
        else:
            from coral.grader.trusted_worker import invoke_task_validator

            result = invoke_task_validator(
                self.coral_dir,
                self._validator_entrypoint("trusted_submission_validator"),
                commit_hash,
                context,
            )
        required = {
            "mode",
            "node_id",
            "node_type",
            "base_commit",
            "quality_anchor",
            "contract_hash",
            "max_infra_retries",
            "submission_hash",
        }
        if not isinstance(result, dict) or not required.issubset(result):
            raise TrustedWriteError("submission validator returned incomplete metadata")
        return result

    def _workflow(
        self, operation: str, payload: dict[str, Any],
        binding: AgentBinding | None,
    ) -> dict[str, Any]:
        expected = _WORKFLOW_FIELDS[operation]
        if operation == "initialize_research" and "research_intent" in payload:
            expected = expected | {"research_intent"}
        if operation == "initialize_research":
            expected = expected | ({"planning_token", "overlap_review"} & payload.keys())
        if operation == "finish_assignment" and payload.get("mode") == "engineering" and "analysis" in payload:
            expected = expected | {"analysis"}
        if set(payload) != expected:
            raise TrustedWriteError(f"{operation} contains unknown or missing fields")
        dynamic = getattr(getattr(self.config, "agents", None), "dynamic", None)
        context = {
            "coral_dir": str(self.coral_dir),
            "agent_id": binding.agent_id if binding else None,
            "agent_root_pid": binding.root_pid if binding else None,
            "agent_start_time": binding.start_time if binding else None,
            "island_id": binding.island_id if binding else None,
            "max_engineering_runs_per_node": (getattr(getattr(self.config, "grader", None), "args", {}) or {}).get("max_engineering_runs_per_node", 1),
            "is_manager": binding is None,
            "repo_path": str(self.coral_dir.parent / "repo"),
            "max_researchers": getattr(dynamic, "max_researchers", 0),
            "max_engineers": getattr(dynamic, "max_engineers", 0),
        }
        if self.workflow_handler is not None:
            result = self.workflow_handler(operation, payload, context)
        else:
            from coral.grader.trusted_worker import invoke_task_validator

            result = invoke_task_validator(
                self.coral_dir,
                self._validator_entrypoint("trusted_workflow_handler"),
                operation, payload, context,
            )
        if not isinstance(result, dict):
            raise TrustedWriteError("workflow handler returned a non-object result")
        return result

    def handle(self, operation: str, payload: dict[str, Any], peer_pid: int) -> dict[str, Any]:
        if operation == "run_diagnostic":
            if not isinstance(payload, dict):
                raise TrustedWriteError("trusted payload must be an object")
            context = {"coral_dir": str(self.coral_dir), "peer_pid": peer_pid}
            if self.diagnostic_handler is not None:
                return self.diagnostic_handler(payload, context)
            from coral.grader.trusted_worker import invoke_task_validator

            return invoke_task_validator(
                self.coral_dir, self._validator_entrypoint("trusted_diagnostic_handler"),
                payload, context, timeout=DIAGNOSTIC_WORKER_TIMEOUT,
            )
        if operation == "reconcile_workflow":
            if peer_pid != os.getppid():
                raise TrustedWriteError("only the manager peer may reconcile workflow")
            if not isinstance(payload, dict):
                raise TrustedWriteError("trusted payload must be an object")
            result = self._workflow(operation, payload, None)
            configured = getattr(getattr(self.config, "grader", None), "args", {}) or {}
            entrypoint = configured.get("trusted_workflow_worker")
            with self._workflow_lifecycle_lock:
                if entrypoint and not self._workflow_stopping and (
                    self._workflow_process is None or self._workflow_process.poll() is not None
                ):
                    from coral.grader.trusted_worker import (
                        WorkflowRecoveryPendingError,
                        start_task_worker,
                    )

                    try:
                        self._workflow_process = start_task_worker(
                            self.coral_dir, entrypoint, log_name="workflow_worker.err"
                        )
                    except WorkflowRecoveryPendingError as exc:
                        # Keep returning assignments so the manager and grader
                        # survive. Its normal reconcile cadence retries recovery.
                        message = str(exc)
                        if message != self._workflow_recovery_error:
                            logger.warning("Workflow recovery deferred: %s", message)
                        self._workflow_recovery_error = message
                        result["workflow_recovery_pending"] = True
                        result["workflow_error"] = message
                    else:
                        self._workflow_recovery_error = None
                memory_entry = configured.get("trusted_memory_worker")
                if result.get("memory_due") and memory_entry and not self._workflow_stopping and (
                    self._memory_process is None or self._memory_process.poll() is not None
                ):
                    from coral.grader.trusted_worker import start_task_worker

                    try:
                        self._memory_process = start_task_worker(self.coral_dir, memory_entry)
                    except Exception as exc:
                        result["memory_error"] = str(exc)
            return result
        binding = self.identity_resolver(self.coral_dir, peer_pid)
        return self.handle_authenticated(operation, payload, binding)

    def handle_authenticated(
        self,
        operation: str,
        payload: dict[str, Any],
        binding: AgentBinding,
    ) -> dict[str, Any]:
        if operation not in OPERATIONS:
            raise TrustedWriteError(f"unsupported trusted operation: {operation}")
        if operation == "run_diagnostic":
            raise TrustedWriteError("diagnostics require a kernel-authenticated Analyzer peer")
        if not isinstance(payload, dict):
            raise TrustedWriteError("trusted payload must be an object")
        if operation == "reconcile_workflow":
            raise TrustedWriteError("only the manager peer may reconcile workflow")
        if operation in WORKFLOW_OPERATIONS:
            return self._workflow(operation, payload, binding)
        if operation == "start_memory_if_due":
            return self._start_memory(payload, binding)
        if operation == "submit_attempt":
            return self._submit(payload, binding).to_dict()
        if operation == "retry_attempt":
            return self._retry(payload, binding).to_dict()
        return self._publish(payload, binding)

    def _start_memory(
        self, payload: dict[str, Any], binding: AgentBinding,
    ) -> dict[str, str]:
        if payload:
            raise TrustedWriteError("start_memory_if_due contains unknown fields")
        configured = getattr(getattr(self.config, "grader", None), "args", {}) or {}
        if self.memory_authorizer is None and not configured.get("trusted_memory_authorizer"):
            raise TrustedWriteError("Memory authorization is not configured")
        self._authorize(
            "trusted_memory_authorizer",
            self.memory_authorizer,
            binding.agent_id,
            {
                "coral_dir": str(self.coral_dir),
                "island_id": binding.island_id,
            },
        )
        entrypoint = self._validator_entrypoint("trusted_memory_worker")
        if self.memory_launcher is not None:
            self.memory_launcher(self.coral_dir, entrypoint)
        else:
            from coral.grader.trusted_worker import start_task_worker

            start_task_worker(self.coral_dir, entrypoint)
        return {"status": "started"}

    def submit_root_attempt(
        self,
        *,
        worktree: str | Path,
        commit_hash: str,
    ) -> Attempt:
        """Trusted-host-only Root bootstrap; never exposed as a socket operation."""
        policy_path = self.coral_dir / "private" / "root_bootstrap.json"
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrustedWriteError("fixed Root policy is unavailable or invalid") from exc
        required = {"version", "source_commit", "node_id", "mode", "submission_hash"}
        if (
            not isinstance(policy, dict)
            or set(policy) != required
            or policy.get("version") != 1
            or policy.get("node_id") != "N000"
            or policy.get("mode") != "research"
            or not isinstance(policy.get("source_commit"), str)
            or not _SHA.fullmatch(policy["source_commit"])
            or not isinstance(policy.get("submission_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", policy["submission_hash"])
        ):
            raise TrustedWriteError("fixed Root policy is unavailable or invalid")
        source_commit = policy["source_commit"]
        binding = AgentBinding(
            agent_id="root-bootstrap",
            root_pid=0,
            start_time=0,
            worktree=str(Path(worktree).resolve()),
            island_id=None,
        )
        parents = _git(Path(binding.worktree), "show", "-s", "--format=%P", commit_hash).split()
        parent_hash = parents[0] if parents else None
        canonical = self._validate_submission(commit_hash, binding, parent_hash)
        if (
            canonical["mode"] != policy["mode"]
            or canonical["node_id"] != policy["node_id"]
            or canonical["node_type"] != "seed"
            or canonical["base_commit"] != source_commit
            or canonical["quality_anchor"] is not None
            or canonical["submission_hash"] != policy["submission_hash"]
        ):
            raise TrustedWriteError("submission does not match the fixed Root policy")
        return self._submit(
            {
                "commit_hash": commit_hash,
                "agent_id": "root-bootstrap",
                "title": "bootstrap Root N000",
                "parent_hash": source_commit,
                "budget_class": BUDGET_CLASS_REAL,
                "island_id": None,
            },
            binding,
            root_source_commit=source_commit,
            canonical=canonical,
        )

    def _submit(
        self,
        payload: dict[str, Any],
        binding: AgentBinding,
        *,
        root_source_commit: str | None = None,
        canonical: dict[str, Any] | None = None,
    ) -> Attempt:
        expected_fields = {
            "commit_hash",
            "agent_id",
            "title",
            "parent_hash",
            "budget_class",
            "island_id",
        }
        if set(payload) != expected_fields:
            raise TrustedWriteError("submit_attempt contains unknown or missing fields")
        if payload["agent_id"] != binding.agent_id:
            raise TrustedWriteError("declared agent_id does not match authenticated Agent")
        if payload["island_id"] != binding.island_id:
            raise TrustedWriteError("declared island_id does not match authenticated Agent")
        commit_hash = payload["commit_hash"]
        if not isinstance(commit_hash, str) or not _SHA.fullmatch(commit_hash):
            raise TrustedWriteError("commit_hash must be a full lowercase Git SHA")
        title = payload["title"]
        if not isinstance(title, str) or not title.strip() or len(title) > 500:
            raise TrustedWriteError("invalid attempt title")
        if payload["budget_class"] not in {BUDGET_CLASS_REAL, BUDGET_CLASS_TUNE}:
            raise TrustedWriteError("invalid budget_class")
        worktree = Path(binding.worktree).resolve()
        if _git(worktree, "rev-parse", "HEAD") != commit_hash:
            raise TrustedWriteError("commit_hash is not the authenticated worktree current HEAD")
        parents = _git(worktree, "show", "-s", "--format=%P", commit_hash).split()
        parent_hash = parents[0] if parents else None
        if payload["parent_hash"] != parent_hash:
            raise TrustedWriteError("declared parent_hash does not match committed parent")
        canonical = canonical or self._validate_submission(
            commit_hash, binding, parent_hash
        )
        if canonical["node_type"] == "seed" and canonical["mode"] == "research":
            if root_source_commit is None:
                raise TrustedWriteError(
                    "Root bootstrap requires the separate trusted host entry"
                )
            if (
                canonical["mode"] != "research"
                or canonical["node_id"] != "N000"
                or canonical["base_commit"] != root_source_commit
                or canonical["quality_anchor"] is not None
                or parent_hash != root_source_commit
                or payload["budget_class"] != BUDGET_CLASS_REAL
            ):
                raise TrustedWriteError("submission does not match the fixed Root policy")
        elif root_source_commit is not None:
            raise TrustedWriteError("trusted Root entry accepts only N000 seed")

        with trusted_writes_lock(self.coral_dir):
            if _git(worktree, "rev-parse", "HEAD") != commit_hash:
                raise TrustedWriteError("authenticated worktree HEAD moved during submission")
            authorization = None
            if root_source_commit is None:
                authorization = self._authorize(
                    "trusted_workflow_authorizer",
                    self.attempt_authorizer,
                    canonical,
                    {
                        "coral_dir": str(self.coral_dir),
                        "agent_id": binding.agent_id,
                        "island_id": binding.island_id,
                        "commit_hash": commit_hash,
                        "budget_class": payload["budget_class"],
                    },
                )
            existing_same = read_attempt(
                self.coral_dir, commit_hash, island_id=binding.island_id
            )
            if existing_same is not None:
                if existing_same.agent_id != binding.agent_id:
                    raise TrustedWriteError("commit already belongs to another Agent")
                return existing_same
            if canonical["mode"] == "research" and payload["budget_class"] == BUDGET_CLASS_REAL:
                clashes = [
                    attempt
                    for attempt in _all_attempts(self.coral_dir, binding.island_id)
                    if attempt.metadata.get("mode") == "research"
                    and attempt.metadata.get("node_id") == canonical["node_id"]
                    and attempt.commit_hash != commit_hash
                    and attempt.budget_class == BUDGET_CLASS_REAL
                ]
                if clashes and (authorization or {}).get("allow_multiple_research") is not True:
                    raise TrustedWriteError(
                        f"node {canonical['node_id']} already has a different Research SHA"
                    )
            shared_state_hash = self.checkpoint_fn(
                str(self.coral_dir),
                binding.agent_id,
                title,
                island_id=binding.island_id,
            )
            parent_shared_state_hash = None
            if parent_hash:
                parent_attempt = read_attempt(
                    self.coral_dir, parent_hash, island_id=binding.island_id
                )
                if parent_attempt is not None:
                    parent_shared_state_hash = parent_attempt.shared_state_hash
            metadata = {
                "budget_class": payload["budget_class"],
                "mode": canonical["mode"],
                "node_id": canonical["node_id"],
                "node_type": canonical["node_type"],
                "submission_hash": canonical["submission_hash"],
                "base_commit": canonical["base_commit"],
                "quality_anchor": canonical["quality_anchor"],
                **({"timing_declaration": canonical["timing_declaration"]}
                   if "timing_declaration" in canonical else {}),
                "contract_hash": canonical["contract_hash"],
                "max_infra_retries": int(canonical["max_infra_retries"]),
                "infra_retry_count": 0,
                "grader_invocation_count": 0,
                "grader_runs": [],
                "source_worktree": str(worktree),
            }
            if binding.island_id is not None:
                metadata["island_id"] = binding.island_id
            attempt = Attempt(
                commit_hash=commit_hash,
                agent_id=binding.agent_id,
                title=title,
                score=None,
                status="pending",
                parent_hash=parent_hash,
                timestamp=_utcnow(),
                feedback="",
                shared_state_hash=shared_state_hash,
                parent_shared_state_hash=parent_shared_state_hash,
                metadata=metadata,
            )
            write_attempt(self.coral_dir, attempt, island_id=binding.island_id)
            return attempt

    def _retry(self, payload: dict[str, Any], binding: AgentBinding) -> Attempt:
        if set(payload) != {"commit_hash"}:
            raise TrustedWriteError("retry_attempt contains unknown or missing fields")
        commit_hash = payload["commit_hash"]
        if not isinstance(commit_hash, str) or not _SHA.fullmatch(commit_hash):
            raise TrustedWriteError("commit_hash must be a full lowercase Git SHA")
        worktree = Path(binding.worktree).resolve()
        _git(worktree, "cat-file", "-e", f"{commit_hash}^{{commit}}")
        parents = _git(worktree, "show", "-s", "--format=%P", commit_hash).split()
        parent_hash = parents[0] if parents else None
        canonical = self._validate_submission(commit_hash, binding, parent_hash)
        def authorize_retry() -> None:
            if canonical.get("node_id") == "N000" and canonical.get("mode") == "research":
                return
            self._authorize(
                "trusted_workflow_authorizer",
                self.attempt_authorizer,
                canonical,
                {
                    "coral_dir": str(self.coral_dir),
                    "agent_id": binding.agent_id,
                    "island_id": binding.island_id,
                    "commit_hash": commit_hash,
                    "budget_class": "real",
                    "retry": True,
                },
            )

        return retry_attempt(
            self.coral_dir,
            commit_hash,
            binding.island_id,
            expected_agent_id=binding.agent_id,
            expected_contract_hash=str(canonical["contract_hash"]),
            max_infra_retries=int(canonical["max_infra_retries"]),
            expected_source_worktree=worktree,
            authorize=authorize_retry,
        )

    def _publish(self, payload: dict[str, Any], binding: AgentBinding) -> dict[str, Any]:
        fields = {"kind", "node_id", "mode", "version", "content"}
        if set(payload) != fields:
            raise TrustedWriteError("publish_artifact contains unknown or missing fields")
        content = payload["content"]
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_REQUEST_BYTES // 2:
            raise TrustedWriteError("artifact content is invalid or too large")
        context = {
            "node_id": payload["node_id"],
            "mode": payload["mode"],
            "version": payload["version"],
            "agent_id": binding.agent_id,
            "island_id": binding.island_id,
        }
        if self.artifact_validator is not None:
            result = self.artifact_validator(payload["kind"], content, context)
        else:
            from coral.grader.trusted_worker import invoke_task_validator

            result = invoke_task_validator(
                self.coral_dir,
                self._validator_entrypoint("trusted_artifact_validator"),
                payload["kind"],
                content,
                context,
            )
        relative = result.get("relative_path") if isinstance(result, dict) else None
        if not isinstance(relative, str) or not (
            relative == "tree.md" or _NODE_ARTIFACT.fullmatch(relative)
        ):
            raise TrustedWriteError("artifact target is outside the canonical notes tree")
        notes_root = island_root(self.coral_dir, binding.island_id) / "notes"
        target = notes_root / relative
        with trusted_writes_lock(self.coral_dir):
            authorized = self._authorize(
                "trusted_artifact_authorizer",
                self.artifact_authorizer,
                payload["kind"],
                {
                    **context,
                    "coral_dir": str(self.coral_dir),
                    "relative_path": relative,
                },
            )
            if authorized is not None and payload["kind"] == "evaluation":
                fresh = authorized.get("content")
                if not isinstance(fresh, str) or not fresh.strip():
                    raise TrustedWriteError("Evaluation authorizer returned invalid fresh content")
                content = fresh
            notes_root.mkdir(parents=True, exist_ok=True)
            resolved_root = notes_root.resolve()
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != notes_root.parent):
                raise TrustedWriteError("artifact target traverses a symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(resolved_root):
                raise TrustedWriteError("artifact target escapes the notes tree")
            if result.get("format") == "analysis-json-markdown":
                from coral.grader.analysis_artifacts import publish_analysis

                if (payload["kind"] != "analysis" or not re.fullmatch(
                        r"(?:research|engineering)_analysis_v[1-9][0-9]*\.md", target.name)):
                    raise TrustedWriteError("invalid Analysis format target")
                try:
                    publish_analysis(target, json.loads(content))
                except (OSError, ValueError) as exc:
                    raise TrustedWriteError(str(exc)) from exc
                return {"path": relative}
            if target.exists() and ("_v" in target.stem or target.name == "spec.md"):
                try:
                    existing = target.read_text(encoding="utf-8")
                except OSError as exc:
                    raise TrustedWriteError("immutable artifact cannot be read") from exc
                if existing == content:
                    return {"path": relative}
                raise TrustedWriteError("immutable artifact already exists")
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                directory_fd = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        return {"path": relative}


class TrustedWriteServer:
    """Small newline-delimited JSON UDS server owned by the grader daemon."""

    def __init__(
        self,
        coral_dir: str | Path,
        service: TrustedWriteService,
        *,
        read_timeout: float = 2.0,
        max_connections: int = 32,
    ) -> None:
        if read_timeout <= 0 or max_connections < 1:
            raise ValueError("trusted server bounds must be positive")
        self.coral_dir = Path(coral_dir)
        self.service = service
        self.socket_path = self.coral_dir / "public" / "trusted.sock"
        self.read_timeout = read_timeout
        self.max_connections = max_connections
        self._connections = threading.BoundedSemaphore(max_connections)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        from coral.grader.trusted_worker import stop_workflow_worker

        with self.service._workflow_lifecycle_lock:
            stop_workflow_worker(self.coral_dir)
            (self.coral_dir / "private/workflow_stopping").unlink(missing_ok=True)
            self.service._workflow_stopping = False
            self.service._workflow_process = None
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        listener.listen(32)
        listener.settimeout(0.2)
        self._listener = listener
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="coral-trusted-writes", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.service.stop_workflow()
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.socket_path.unlink(missing_ok=True)

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self._connections.acquire(blocking=False):
                connection.close()
                continue
            threading.Thread(
                target=self._handle_connection,
                args=(connection,),
                daemon=True,
            ).start()

    def _handle_connection(self, connection: socket.socket) -> None:
        try:
            with connection:
                try:
                    if not hasattr(socket, "SO_PEERCRED"):
                        raise TrustedWriteError("SO_PEERCRED is required for trusted writes")
                    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                    peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
                    if peer_uid != os.getuid():
                        raise TrustedWriteError("trusted peer uid does not own this run")
                    connection.settimeout(self.read_timeout)
                    request = self._read_request(connection)
                    result = self.service.handle(
                        request["operation"], request["payload"], peer_pid
                    )
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                try:
                    connection.sendall(
                        json.dumps(response, separators=(",", ":")).encode() + b"\n"
                    )
                except OSError:
                    pass
        finally:
            self._connections.release()

    @staticmethod
    def _read_request(connection: socket.socket) -> dict[str, Any]:
        raw = bytearray()
        while b"\n" not in raw:
            chunk = connection.recv(4096)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_REQUEST_BYTES:
                raise TrustedWriteError("trusted request exceeds 64 KiB")
        try:
            request = json.loads(bytes(raw).split(b"\n", 1)[0])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TrustedWriteError("invalid trusted request JSON") from exc
        if not isinstance(request, dict) or set(request) != {"version", "operation", "payload"}:
            raise TrustedWriteError("invalid trusted request envelope")
        if request["version"] != PROTOCOL_VERSION:
            raise TrustedWriteError("unsupported trusted protocol version")
        if request["operation"] not in OPERATIONS:
            raise TrustedWriteError(f"unsupported trusted operation: {request['operation']}")
        if not isinstance(request["payload"], dict):
            raise TrustedWriteError("trusted payload must be an object")
        return request


def _request_trusted_once(
    coral_dir: str | Path,
    request: bytes,
    *,
    timeout: float,
) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(Path(coral_dir) / "public" / "trusted.sock"))
        client.sendall(request)
        raw = bytearray()
        while b"\n" not in raw:
            chunk = client.recv(4096)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_REQUEST_BYTES:
                raise TrustedWriteError("trusted response exceeds 64 KiB")
    except OSError as exc:
        raise TrustedWriteError(f"trusted write channel failed: {exc}") from exc
    finally:
        client.close()
    try:
        response = json.loads(bytes(raw).split(b"\n", 1)[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrustedWriteError("invalid trusted response") from exc
    if not response.get("ok"):
        raise TrustedWriteError(str(response.get("error", "trusted write rejected")))
    result = response.get("result")
    if not isinstance(result, dict):
        raise TrustedWriteError("trusted response result is not an object")
    return result


def request_trusted(
    coral_dir: str | Path,
    operation: str,
    payload: dict[str, Any],
    *,
    timeout: float = TRUSTED_CLIENT_TIMEOUT,
    startup_retry_window: float = 1.0,
) -> dict[str, Any]:
    """Send one bounded request to this run's fixed trusted socket.

    A newly spawned Agent can execute before the manager has published its
    private PID binding. Retry only those narrow startup/authentication races;
    all submission validation failures remain immediate and fail closed.
    """
    if operation not in OPERATIONS:
        raise TrustedWriteError(f"unsupported trusted operation: {operation}")
    request = json.dumps(
        {"version": PROTOCOL_VERSION, "operation": operation, "payload": payload},
        separators=(",", ":"),
    ).encode() + b"\n"
    if len(request) > MAX_REQUEST_BYTES:
        raise TrustedWriteError("trusted request exceeds 64 KiB")
    deadline = time.monotonic() + max(0.0, startup_retry_window)
    retryable = (
        "trusted Agent registry is unavailable or invalid",
        "peer process does not belong to a registered Agent",
        "No such file or directory",
        "Connection refused",
    )
    while True:
        try:
            return _request_trusted_once(coral_dir, request, timeout=timeout)
        except TrustedWriteError as exc:
            if time.monotonic() >= deadline or not any(
                marker in str(exc) for marker in retryable
            ):
                raise
            time.sleep(0.05)
