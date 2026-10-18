"""Ephemeral, host-owned role/FIFO GPU requests in the common lock domain."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from .runtime import InfraFailure, ProcessIdentity

ROLE_PRIORITY = {"analyzer": 0, "engineer": 0, "researcher": 1}


class GpuRequest:
    def __init__(self, root, *, role, count, policy, share, uid, deadline, group, limit):
        if role not in ROLE_PRIORITY:
            raise InfraFailure("invalid GPU source role")
        if (group is not None and (not isinstance(group, str) or not group)
                or type(limit) is not int or limit < 1):
            raise InfraFailure("invalid GPU execution group or limit")
        owner = ProcessIdentity.read(os.getpid())
        if owner is None:
            raise InfraFailure("cannot establish GPU request owner identity")
        self.root = Path(root) / "requests"
        self.key = uuid.uuid4().hex
        self.path = self.root / f"{self.key}.json"
        self.registered = False
        self.row = {
            "id": self.key, "role": role, "count": count, "policy": policy,
            "share": share, "uid": uid, "deadline": deadline,
            "group": group, "limit": limit, "state": "waiting",
            "pid": owner.pid, "start_time": owner.start_time,
            "queued_at": time.monotonic_ns(),
        }

    def _write(self):
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.row) + "\n")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def register(self):
        """Caller holds allocator.lock; registration and ranking are one transaction."""
        if not self.registered:
            self.root.mkdir(mode=0o700, exist_ok=True)
            self._write()
            self.registered = True

    def records(self):
        rows = []
        for path in self.root.glob("*.json"):
            try:
                row = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # Owner released while this allocator was querying.
            owner = ProcessIdentity.read(row["pid"])
            expired = row["state"] == "waiting" and row["deadline"] < time.monotonic()
            if (owner is None or owner.start_time != row["start_time"]
                    or expired and row["id"] != self.key):
                path.unlink(missing_ok=True)
                continue
            rows.append(row)
        return rows

    def choose(self, devices, free_uuids, eligible):
        """Plan feasible grants; earlier eligible requests reserve virtual cards."""
        self.register()
        rows = self.records()
        active = Counter(row["group"] for row in rows
                         if row["state"] == "reserved" and row["group"] is not None)
        waiting = sorted((row for row in rows if row["state"] == "waiting"),
                         key=lambda row: (ROLE_PRIORITY[row["role"]], row["queued_at"], row["id"]))
        available = set(free_uuids)
        for row in waiting:
            group = row["group"]
            if group is not None and active[group] >= row["limit"]:
                continue
            selected = [d for d in devices if d.uuid in available and eligible(d, row)]
            if len(selected) < row["count"]:
                continue
            selected = selected[:row["count"]]
            if row["id"] == self.key:
                return selected
            available.difference_update(d.uuid for d in selected)
            if group is not None:
                active[group] += 1
        return None

    def reserved(self, uuids):
        self.row.update(state="reserved", gpu_uuids=list(uuids))
        self._write()

    def close(self):
        self.path.unlink(missing_ok=True)
