"""GPU discovery, reservation, and efficiency comparability."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .runtime import InfraFailure, ProcessIdentity


class ResourceBusy(InfraFailure):
    """Transient contention; distinct from a failed inventory or configuration."""

    pass


@dataclass(frozen=True)
class IdleGpuPolicy:
    compute_processes: int
    max_memory_used_mib: int
    max_utilization_percent: int


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]
    numa_node: int | None = None
    compute_processes: tuple[ProcessIdentity, ...] = ()

    def idle(self, policy: IdleGpuPolicy) -> bool:
        return (
            len(self.compute_pids) <= policy.compute_processes
            and self.memory_used_mib <= policy.max_memory_used_mib
            and self.utilization_percent <= policy.max_utilization_percent
        )


@dataclass(frozen=True)
class GpuInventory:
    devices: tuple[GpuDevice, ...]
    topology: str

    @classmethod
    def from_nvidia_smi(
        cls,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sysfs_root: Path = Path("/sys/bus/pci/devices"),
        identity_reader: Callable[[int], ProcessIdentity | None] = ProcessIdentity.read,
    ) -> GpuInventory:
        query = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        apps = runner(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if apps.returncode != 0:
            raise InfraFailure("incomplete GPU inventory evidence: compute-app query failed")
        pids: dict[str, list[int]] = {}
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or not parts[1].isdigit():
                raise InfraFailure("incomplete GPU inventory evidence: invalid compute-app row")
            pids.setdefault(parts[0], []).append(int(parts[1]))
        topo = runner(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=1)
        if topo.returncode != 0 or not topo.stdout.strip():
            raise InfraFailure("incomplete GPU inventory evidence: topology query failed")
        devices = []
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 5)]
            if len(parts) != 6:
                raise InfraFailure("invalid nvidia-smi GPU inventory")
            index, uuid, name, bus_id, memory, util = parts
            domain, address = bus_id.split(":", 1)
            normalized_bus_id = f"{domain[-4:]}:{address}".lower()
            try:
                numa_node = int((sysfs_root / normalized_bus_id / "numa_node").read_text().strip())
            except (OSError, ValueError):
                numa_node = None
            if numa_node is not None and numa_node < 0:
                numa_node = None
            devices.append(
                GpuDevice(
                    int(index),
                    uuid,
                    name,
                    int(memory),
                    int(util),
                    tuple(pids.get(uuid, ())),
                    numa_node,
                    tuple(identity for pid in pids.get(uuid, ())
                          if (identity := identity_reader(pid)) is not None),
                )
            )
        return cls(
            tuple(sorted(devices, key=lambda d: d.index)),
            topo.stdout,
        )


@dataclass(frozen=True)
class EfficiencyEnvironment:
    gpu_uuids: tuple[str, ...]
    gpu_names: tuple[str, ...]
    topology_fingerprint: str
    numa_nodes: tuple[int | None, ...]
    nccl_env: Mapping[str, str]


@dataclass(frozen=True)
class Comparability:
    valid: bool
    reason: str | None = None


def gpu_topology_fingerprint(
    gpu_uuids: tuple[str, ...],
    topology: str,
    nccl_env: Mapping[str, str],
    numa_nodes: tuple[int | None, ...],
) -> str:
    if len(gpu_uuids) != len(numa_nodes):
        raise ValueError("GPU and NUMA cardinality differs")
    ordered_numa = [numa for _uuid, numa in sorted(zip(gpu_uuids, numa_nodes, strict=True))]
    payload = {
        "logical_numa": ordered_numa,
        "selected_topology": topology.strip(),
        "nccl": sorted((str(k), str(v)) for k, v in nccl_env.items()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_efficiency_comparability(
    dense: EfficiencyEnvironment, candidate: EfficiencyEnvironment
) -> Comparability:
    same_cardinality = (
        len(dense.gpu_uuids) == len(candidate.gpu_uuids)
        and len(set(dense.gpu_uuids)) == len(dense.gpu_uuids)
        and len(set(candidate.gpu_uuids)) == len(candidate.gpu_uuids)
    )
    environment_matches = (
        sorted(dense.gpu_names) == sorted(candidate.gpu_names)
        and dense.topology_fingerprint == candidate.topology_fingerprint
        and sorted(str(x) for x in dense.numa_nodes) == sorted(str(x) for x in candidate.numa_nodes)
        and dict(dense.nccl_env) == dict(candidate.nccl_env)
    )
    same_gpus = set(dense.gpu_uuids) == set(candidate.gpu_uuids)
    equivalent = same_cardinality and environment_matches and same_gpus
    if not same_gpus:
        complete_numa = all(node is not None for node in (*dense.numa_nodes, *candidate.numa_nodes))
        equivalent = same_cardinality and environment_matches and complete_numa
    return Comparability(equivalent, None if equivalent else "incomparable_hardware")


class GpuReservation:
    def __init__(self, devices: tuple[GpuDevice, ...], streams: list[object], topology: str,
                 on_event: Callable | None = None, inventory_provider=None):
        self.devices = devices
        self._streams = streams
        self.topology = topology
        self.indices = tuple(device.index for device in devices)
        self.uuids = tuple(device.uuid for device in devices)
        self.cuda_visible_devices = ",".join(str(index) for index in self.indices)
        self._on_event = on_event
        self._inventory_provider = inventory_provider or GpuInventory.from_nvidia_smi

    def check_exclusive(self, session_pid: int | None = None) -> None:
        """Allow compute only from the process group/session created for this command."""
        for attempt in range(2):
            try:
                sample = self._inventory_provider()
            except Exception as exc:
                raise InfraFailure(f"timing inventory unavailable: {exc}") from exc
            by_uuid = {device.uuid: device for device in sample.devices}
            unstable = False
            for uuid in self.uuids:
                device = by_uuid.get(uuid)
                if device is None:
                    raise InfraFailure(f"timing GPU disappeared: {uuid}")
                owners = {p.pid: p for p in device.compute_processes}
                for pid in device.compute_pids:
                    owner = owners.get(pid)
                    current = ProcessIdentity.read(pid)
                    # An identified foreign process is interference even if it
                    # exits immediately. Only ambiguous/vanished ranks get a retry.
                    external = session_pid is None or any(
                        identity is not None and (
                            identity.uid != os.getuid() or identity.session_id != session_pid
                            or identity.pgid != session_pid
                        ) for identity in (owner, current)
                    )
                    if external or (attempt == 1 and (owner is None or current != owner)):
                        raise InfraFailure(f"timing interference: {uuid} PID {pid}, "
                                           f"UID {owner.uid if owner else 'unknown'}")
                    if owner is None or current != owner:
                        unstable = True
            if not unstable:
                return

    def __enter__(self) -> GpuReservation:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        held = bool(self._streams)
        while self._streams:
            stream = self._streams.pop()
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        if held and self._on_event:
            self._on_event("released", gpu_uuids=self.uuids)


def default_lock_dir() -> Path:
    """Stable across runs on one host, distinct on hosts sharing a home directory."""
    try:
        host = Path("/etc/machine-id").read_bytes().strip()
    except OSError as exc:
        raise InfraFailure("cannot resolve host identity; configure gpu_lock_dir") from exc
    if not host:
        raise InfraFailure("empty machine-id; configure gpu_lock_dir")
    host_id = hashlib.sha256(host).hexdigest()[:16]
    return Path.home() / ".coral" / "gpu-locks" / host_id / f"uid-{os.getuid()}"


def _eligible(device: GpuDevice, policy: IdleGpuPolicy, share: bool, uid: int) -> bool:
    if device.compute_pids:
        owners = {identity.pid: identity for identity in device.compute_processes}
        return share and all(pid in owners and owners[pid].uid != uid
                             for pid in device.compute_pids)
    return device.idle(policy)


def _lock_file(path: Path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(fd, "a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def _reserve_once(count, lock_dir, policy, inventory_provider, share, uid):
    streams: list[object] = []
    allocator = None
    try:
        try:
            allocator = _lock_file(lock_dir / "allocator.lock")
        except BlockingIOError as exc:
            raise ResourceBusy("GPU allocator busy") from exc
        first = inventory_provider()
        selected = []
        for device in sorted(first.devices, key=lambda d: (bool(d.compute_pids), d.index)):
            if not _eligible(device, policy, share, uid):
                continue
            safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", device.uuid)
            try:
                stream = _lock_file(lock_dir / f"{safe_uuid}.lock")
            except BlockingIOError:
                continue
            streams.append(stream)
            selected.append(device)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ResourceBusy("fewer than four idle GPUs (or eligible shared GPUs)")
        second = inventory_provider()
        by_uuid = {device.uuid: device for device in second.devices}
        confirmed = tuple(by_uuid.get(device.uuid) for device in selected)
        if any(device is None or not _eligible(device, policy, share, uid)
               for device in confirmed):
            raise ResourceBusy("fewer than four idle GPUs after reservation recheck")
        return GpuReservation(confirmed, streams, second.topology,
                              inventory_provider=inventory_provider)
    except BaseException:
        GpuReservation((), streams, "").release()
        raise
    finally:
        if allocator is not None:
            allocator.close()


def reserve_idle_gpus(
    count: int,
    lock_dir: str | Path,
    policy: IdleGpuPolicy,
    *,
    inventory_provider: Callable[[], GpuInventory] = GpuInventory.from_nvidia_smi,
    allow_other_users: bool = False,
    current_uid: int | None = None,
    wait_timeout: float = 0,
    poll_interval: float = 5,
    check_cancelled: Callable[[], None] | None = None,
    on_event: Callable | None = None,
) -> GpuReservation:
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or not isinstance(allow_other_users, bool)
            or not math.isfinite(wait_timeout) or wait_timeout < 0
            or not math.isfinite(poll_interval) or poll_interval <= 0):
        raise InfraFailure("invalid GPU allocation options")
    lock_dir = Path(lock_dir)
    try:
        lock_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        stat = lock_dir.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise InfraFailure("GPU lock directory must belong to this UID and have mode 0700")
    except OSError as exc:
        raise InfraFailure(f"GPU lock directory unavailable: {exc}") from exc
    uid = os.getuid() if current_uid is None else current_uid
    deadline = time.monotonic() + wait_timeout
    inventory_errors = 0
    while True:
        if check_cancelled:
            check_cancelled()
        try:
            held = _reserve_once(count, lock_dir, policy, inventory_provider,
                                 allow_other_users, uid)
        except ResourceBusy as exc:
            reason = str(exc)
            inventory_errors = 0
        except (InfraFailure, subprocess.SubprocessError, OSError, ValueError) as exc:
            inventory_errors += 1
            reason = f"GPU inventory failed: {exc}"
            if inventory_errors >= 3 or wait_timeout == 0 or isinstance(exc, OSError):
                raise InfraFailure(reason) from exc
        else:
            held._on_event = on_event
            try:
                if on_event:
                    on_event("reserved", gpu_uuids=held.uuids, gpu_indices=held.indices,
                             devices=[{"uuid": d.uuid, "memory_used_mib": d.memory_used_mib,
                                       "utilization_percent": d.utilization_percent,
                                       "compute_pids": d.compute_pids,
                                       "compute_uids": [p.uid for p in d.compute_processes]}
                                      for d in held.devices])
                return held
            except BaseException:
                held.release()
                raise
        if wait_timeout == 0:
            raise ResourceBusy(reason)
        if time.monotonic() >= deadline:
            raise InfraFailure(f"timed out waiting for GPUs: {reason}")
        if on_event:
            on_event("waiting", reason=reason)
        wake = min(deadline, time.monotonic() + poll_interval * random.uniform(0.9, 1.1))
        while time.monotonic() < wake:
            if check_cancelled:
                check_cancelled()
            time.sleep(min(0.1, max(0, wake - time.monotonic())))
