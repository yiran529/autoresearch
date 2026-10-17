"""Host process identities shared by GPU ownership and execution guards."""

from __future__ import annotations

import os
import select
import signal
import threading
from dataclasses import dataclass
from pathlib import Path


class InfraFailure(RuntimeError):  # noqa: N818
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    uid: int
    start_time: int
    pgid: int
    session_id: int

    @classmethod
    def read(cls, pid: int, proc_root: Path = Path("/proc")) -> ProcessIdentity | None:
        try:
            path = proc_root / str(pid)
            first = (path / "stat").read_text().rsplit(")", 1)[1].split()
            status = (path / "status").read_text().splitlines()
            uid = int(next(line for line in status if line.startswith("Uid:")).split()[1])
            second = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if first[19] != second[19] or second[0] == "Z":
                return None
            return cls(pid, uid, int(second[19]), int(second[2]), int(second[3]))
        except (OSError, ValueError, IndexError, StopIteration):
            return None


class Lifecycle:
    """Cancel an evaluation after a signal or loss of its original daemon."""

    def __init__(self, expected_parent: dict[str, int] | None = None):
        self.cancelled = threading.Event()
        self.parent = None
        self._pidfd = None
        self._handlers = {}
        self.expected_parent = expected_parent

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise InfraFailure("grader lifecycle must start on the main thread")
        self.parent = ProcessIdentity.read(os.getppid())
        if self.parent is None:
            raise InfraFailure("cannot establish grader daemon identity")
        if self.expected_parent is not None and (
            self.parent.pid != self.expected_parent.get("pid")
            or self.parent.start_time != self.expected_parent.get("start_time")
        ):
            raise InfraFailure("grader parent daemon exited before worker initialization")
        try:
            self._pidfd = os.pidfd_open(self.parent.pid)
        except (AttributeError, OSError):
            pass
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._handlers[signum] = signal.signal(signum, self._cancel)
        return self

    def _cancel(self, _signum, _frame):
        self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise InfraFailure("evaluation cancelled by signal")
        if self._pidfd is not None:
            poll = select.poll()
            poll.register(self._pidfd, select.POLLIN)
            if poll.poll(0):
                raise InfraFailure("grader parent daemon exited")
        current = ProcessIdentity.read(self.parent.pid)
        if current is None or current.start_time != self.parent.start_time:
            raise InfraFailure("grader parent daemon exited or changed identity")

    def __exit__(self, *_exc):
        for signum, handler in self._handlers.items():
            signal.signal(signum, handler)
        if self._pidfd is not None:
            os.close(self._pidfd)
