from types import SimpleNamespace

import pytest
from greedylore_autoresearch_grader.gpu import (
    GpuDevice,
    GpuInventory,
    IdleGpuPolicy,
    InfraFailure,
    reserve_idle_gpus,
)


def inventory(busy=False):
    devices = tuple(
        GpuDevice(
            i,
            f"GPU-{i}",
            "A100",
            0 if i in {0, 2, 5, 7} else 900,
            0,
            () if not busy or i != 7 else (99,),
            i % 2,
        )
        for i in range(8)
    )
    return GpuInventory(devices, "topology")


def test_allocator_uses_any_four_idle_cards(tmp_path):
    with reserve_idle_gpus(
        4, tmp_path, IdleGpuPolicy(0, 512, 5), inventory_provider=inventory
    ) as reservation:
        assert reservation.indices == (0, 2, 5, 7)
        assert reservation.cuda_visible_devices == "0,2,5,7"
        assert len(reservation.uuids) == 4


def test_allocator_rechecks_after_lock_and_releases(tmp_path):
    calls = iter([inventory(), inventory(busy=True)])
    with pytest.raises(InfraFailure, match="fewer than four idle GPUs"):
        reserve_idle_gpus(
            4, tmp_path, IdleGpuPolicy(0, 512, 5), inventory_provider=lambda: next(calls)
        )
    with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), inventory_provider=inventory):
        pass


def test_inventory_parses_processes_topology_and_numa(tmp_path):
    pci = tmp_path / "0000:65:00.0"
    pci.mkdir()
    (pci / "numa_node").write_text("1\n")

    def runner(command, **_kwargs):
        joined = " ".join(command)
        if "--query-gpu" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout="0, GPU-a, NVIDIA A100, 00000000:65:00.0, 12, 3\n",
            )
        if "--query-compute-apps" in joined:
            return SimpleNamespace(returncode=0, stdout="GPU-a, 4321\n")
        return SimpleNamespace(returncode=0, stdout="GPU0 X\n")

    found = GpuInventory.from_nvidia_smi(runner=runner, sysfs_root=tmp_path)
    assert found.devices[0].compute_pids == (4321,)
    assert found.devices[0].numa_node == 1
    assert found.topology == "GPU0 X\n"


def test_inventory_fails_closed_when_compute_query_or_topology_fails(tmp_path):
    def runner(command, **_kwargs):
        joined = " ".join(command)
        if "--query-gpu" in joined:
            return SimpleNamespace(returncode=0, stdout="0, GPU-a, A100, 0000:65:00.0, 0, 0\n")
        return SimpleNamespace(returncode=1, stdout="", stderr="query failed")

    with pytest.raises(InfraFailure, match="inventory evidence"):
        GpuInventory.from_nvidia_smi(runner=runner, sysfs_root=tmp_path)
