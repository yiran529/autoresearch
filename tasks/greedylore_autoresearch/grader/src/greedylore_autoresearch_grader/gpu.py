"""GPU discovery, reservation, and efficiency comparability."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


class InfraFailure(RuntimeError):  # noqa: N818 - contract term
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
        )
        apps = runner(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
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
        topo = runner(["nvidia-smi", "topo", "-m"], capture_output=True, text=True)
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
    def __init__(self, devices: tuple[GpuDevice, ...], streams: list[object], topology: str):
        self.devices = devices
        self._streams = streams
        self.topology = topology
        self.indices = tuple(device.index for device in devices)
        self.uuids = tuple(device.uuid for device in devices)
        self.cuda_visible_devices = ",".join(str(index) for index in self.indices)

    def __enter__(self) -> GpuReservation:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        while self._streams:
            stream = self._streams.pop()
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


def reserve_idle_gpus(
    count: int,
    lock_dir: str | Path,
    policy: IdleGpuPolicy,
    *,
    inventory_provider: Callable[[], GpuInventory] = GpuInventory.from_nvidia_smi,
) -> GpuReservation:
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    first = inventory_provider()
    selected: list[GpuDevice] = []
    streams: list[object] = []
    try:
        for device in sorted(first.devices, key=lambda item: item.index):
            if not device.idle(policy):
                continue
            safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", device.uuid)
            stream = (lock_dir / f"{safe_uuid}.lock").open("a+b")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            selected.append(device)
            streams.append(stream)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise InfraFailure("fewer than four idle GPUs")
        second = {device.uuid: device for device in inventory_provider().devices}
        confirmed = tuple(second.get(device.uuid) for device in selected)
        if any(device is None or not device.idle(policy) for device in confirmed):
            raise InfraFailure("fewer than four idle GPUs")
        return GpuReservation(
            tuple(device for device in confirmed if device is not None), streams, first.topology
        )
    except Exception:
        GpuReservation((), streams, first.topology).release()
        raise
