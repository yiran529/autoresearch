"""Validated scheduling configuration, per-attempt evidence and reference compatibility."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .gpu import default_lock_dir
from .runtime import InfraFailure

ALLOCATION_POLICY_VERSION = "greedylore-gpu-v1"


def _number(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise InfraFailure(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number")
    return float(value)


@dataclass(frozen=True)
class ResourceOptions:
    pretraining_allow_other_users: bool
    lock_dir: Path
    poll_interval: float
    wait_timeout: float
    monitor_interval: float

    @classmethod
    def from_args(cls, args):
        share = args.get("pretraining_allow_other_users", False)
        if not isinstance(share, bool):
            raise InfraFailure("pretraining_allow_other_users must be a boolean")
        lock = args.get("gpu_lock_dir")
        if lock is not None and (not isinstance(lock, str) or not Path(lock).is_absolute()):
            raise InfraFailure("gpu_lock_dir must be null or an absolute host path")
        return cls(
            share, (Path(lock) if lock is not None else default_lock_dir()).resolve(),
            _number(args.get("gpu_poll_interval_seconds", 5), "gpu_poll_interval_seconds"),
            _number(args.get("gpu_wait_timeout_seconds", 86400),
                    "gpu_wait_timeout_seconds", zero=True),
            _number(args.get("timing_monitor_interval_seconds", 1), "timing_monitor_interval_seconds"),
        )


class ResourceJournal:
    def __init__(self, root: Path, options: ResourceOptions):
        self.root = root
        self.options = options
        self._last_wait = None
        self._last_write = 0.0
        self._started = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)

    def event(self, stage, state, **details):
        now = time.monotonic()
        wait_key = (stage, details.get("reason")) if state == "waiting" else None
        if state == "waiting" and wait_key == self._last_wait and now - self._last_write < 60:
            return
        self._last_wait, self._last_write = wait_key, now
        options = asdict(self.options)
        options["lock_dir"] = str(self.options.lock_dir)
        row = {
            "time": datetime.now(UTC).isoformat(), "stage": stage, "state": state,
            "elapsed_seconds": now - self._started, "uid": os.getuid(),
            "allocation_policy_version": ALLOCATION_POLICY_VERSION,
            "options": options, **details,
        }
        data = json.dumps(row, ensure_ascii=False) + "\n"
        with (self.root / "gpu_resources.jsonl").open("a") as stream:
            stream.write(data)
        snapshot = self.root / "gpu_resources.json"
        temporary = snapshot.with_suffix(".tmp")
        temporary.write_text(data)
        temporary.replace(snapshot)


def validate_monitoring_reference(reference, options: ResourceOptions, root: Path):
    """Require operator-produced measurement evidence; never relabel old measurements."""
    try:
        metadata = reference["monitoring"]
        if (metadata["allocation_policy_version"] != ALLOCATION_POLICY_VERSION
                or _number(metadata["interval_seconds"], "reference monitor interval")
                != options.monitor_interval):
            raise InfraFailure("Dense reference monitoring policy mismatch")
        relative = Path(metadata["evidence_path"])
        evidence_path = (root / relative).resolve()
        if relative.is_absolute() or not evidence_path.is_relative_to(root.resolve()):
            raise InfraFailure("Dense reference monitoring evidence must be inside dense_references")
        evidence = json.loads(evidence_path.read_text())
        if (evidence["contract_hash"] != reference["contract_hash"]
                or evidence["environment"] != reference["environment"]
                or evidence["allocation_policy_version"] != ALLOCATION_POLICY_VERSION
                or _number(evidence["interval_seconds"], "evidence monitor interval")
                != options.monitor_interval):
            raise InfraFailure("Dense reference monitoring evidence does not match the reference")
        if evidence["kind"] == "monitored_reference":
            return evidence_path
        if evidence["kind"] != "legacy_equivalence":
            raise InfraFailure("unknown Dense reference monitoring evidence kind")
        tolerance = _number(evidence["relative_tolerance"], "relative_tolerance")
        bounds = evidence["relative_difference_interval"]
        if (evidence["predeclared"] is not True or tolerance >= 1
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) for v in bounds)
                or not -tolerance <= bounds[0] <= bounds[1] <= tolerance):
            raise InfraFailure("Dense reference monitoring equivalence is not established")
        return evidence_path
    except InfraFailure:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise InfraFailure(f"Dense reference monitoring evidence unavailable or invalid: {exc}") from exc
