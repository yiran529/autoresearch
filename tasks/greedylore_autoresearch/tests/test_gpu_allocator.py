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


def test_sharing_prefers_idle_and_rejects_own_or_unknown_processes(tmp_path):
    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    devices = (
        GpuDevice(0, "own", "A100", 900, 90, (101,), 0,
                  (ProcessIdentity(101, 1000, 1, 101, 101),)),
        GpuDevice(1, "foreign", "A100", 900, 90, (102,), 0,
                  (ProcessIdentity(102, 2000, 2, 102, 102),)),
        GpuDevice(2, "unknown", "A100", 900, 90, (103,), 0),
        GpuDevice(3, "idle", "A100", 0, 0, (), 0),
        GpuDevice(4, "unexplained-memory", "A100", 900, 0, (), 0),
    )
    def provider():
        return GpuInventory(devices, "topology")
    with reserve_idle_gpus(2, tmp_path, IdleGpuPolicy(0, 512, 5),
                           allow_other_users=True, current_uid=1000,
                           inventory_provider=provider) as reservation:
        assert reservation.indices == (3, 1)
    with pytest.raises(InfraFailure):
        reserve_idle_gpus(2, tmp_path, IdleGpuPolicy(0, 512, 5),
                          inventory_provider=provider)


def test_wait_retries_without_retaining_partial_locks(tmp_path):
    calls = iter([inventory(busy=True), inventory(), inventory()])
    events = []
    with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                           wait_timeout=1, poll_interval=0.01,
                           inventory_provider=lambda: next(calls),
                           on_event=lambda state, **details: events.append(state)) as reservation:
        assert reservation.indices == (0, 2, 5, 7)
        assert "waiting" in events


def _hold_cards(lock_dir, ready, release):
    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(8))
    with reserve_idle_gpus(4, lock_dir, IdleGpuPolicy(0, 512, 5),
                           inventory_provider=lambda: GpuInventory(devices, "topology")) as held:
        ready.put(held.indices)
        release.wait(10)


def test_independent_processes_reserve_disjoint_sets_before_cuda_starts(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("fork")
    ready, release = context.Queue(), context.Event()
    holder = context.Process(target=_hold_cards, args=(tmp_path, ready, release))
    holder.start()
    try:
        assert ready.get(timeout=5) == (0, 1, 2, 3)
        devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(8))
        def provider():
            return GpuInventory(devices, "topology")
        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=provider) as second:
            assert second.indices == (4, 5, 6, 7)
            with pytest.raises(InfraFailure):
                reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                                  inventory_provider=provider)
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.kill()
            holder.join()
    assert holder.exitcode == 0
    assert len(list(tmp_path.glob("GPU-*.lock"))) == 8


def test_wait_deadline_and_cancellation_are_infrastructure_failures(tmp_path):
    with pytest.raises(InfraFailure, match="waiting|timed out"):
        reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                          wait_timeout=0.02, poll_interval=0.01,
                          inventory_provider=lambda: inventory(busy=True))

    def cancelled():
        raise InfraFailure("cancelled")

    with pytest.raises(InfraFailure, match="cancelled"):
        reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                          check_cancelled=cancelled, inventory_provider=inventory)


def test_timing_guard_accepts_only_current_session_compute_processes(tmp_path):
    import os
    import subprocess
    import sys
    from dataclasses import replace

    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    own = ProcessIdentity.read(os.getpid())
    assert own is not None
    idle = GpuDevice(0, "GPU-0", "A100", 0, 0, (), 0)
    samples = [idle]
    def provider():
        return GpuInventory(tuple(samples), "topology")
    with reserve_idle_gpus(1, tmp_path, IdleGpuPolicy(0, 512, 5),
                           inventory_provider=provider) as reservation:
        samples[:] = [replace(idle, compute_pids=(own.pid,), compute_processes=(own,))]
        # This process belongs to a different session than the proposed launcher.
        with pytest.raises(InfraFailure, match="interference"):
            reservation.check_exclusive(own.pid)
        with pytest.raises(InfraFailure, match="interference"):
            reservation.check_exclusive()

        launcher = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
        )
        try:
            rank = ProcessIdentity.read(launcher.pid)
            assert rank is not None
            samples[:] = [replace(idle, compute_pids=(rank.pid,), compute_processes=(rank,))]
            reservation.check_exclusive(launcher.pid)
        finally:
            launcher.terminate()
            launcher.wait(timeout=5)


def test_waiting_run_acquires_cards_when_other_process_releases(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("fork")
    ready, release = context.Queue(), context.Event()
    holder = context.Process(target=_hold_cards, args=(tmp_path, ready, release))
    holder.start()
    try:
        assert ready.get(timeout=5) == (0, 1, 2, 3)
        devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4))

        def event(state, **_details):
            if state == "waiting":
                release.set()

        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=lambda: GpuInventory(devices, "topology"),
                               wait_timeout=3, poll_interval=0.01, on_event=event) as reservation:
            assert reservation.indices == (0, 1, 2, 3)
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.kill()
            holder.join()
    assert holder.exitcode == 0


def test_inventory_identity_read_handles_pid_reuse_and_unknown_ownership(tmp_path):
    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    proc = tmp_path / "123"
    proc.mkdir()
    # Linux stat fields 3 through 22: state, ppid, pgrp, session, ..., starttime.
    (proc / "stat").write_text("123 (name with spaces) " + " ".join(
        ["S", "1", "123", "123"] + ["0"] * 15 + ["777"]
    ))
    (proc / "status").write_text("Uid:\t2000\t2000\t2000\t2000\n")
    assert ProcessIdentity.read(123, tmp_path) == ProcessIdentity(123, 2000, 777, 123, 123)
    (proc / "status").unlink()
    assert ProcessIdentity.read(123, tmp_path) is None


@pytest.mark.parametrize("identity_was_read", [True, False])
def test_timing_rechecks_pid_that_disappeared_during_inventory(tmp_path, identity_was_read):
    import os

    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    idle = GpuDevice(0, "GPU-0", "A100", 0, 0, (), 0)
    # This PID is deliberately absent from procfs: a rank just exited.
    identity = ProcessIdentity(999999999, os.getuid(), 1, 1234, 1234)
    stale = GpuDevice(0, "GPU-0", "A100", 100, 20, (identity.pid,), 0,
                      (identity,) if identity_was_read else ())
    samples = iter([GpuInventory((idle,), "topology"), GpuInventory((idle,), "topology"),
                    GpuInventory((stale,), "topology"), GpuInventory((idle,), "topology")])
    with reserve_idle_gpus(1, tmp_path, IdleGpuPolicy(0, 512, 5),
                           inventory_provider=lambda: next(samples)) as reservation:
        reservation.check_exclusive(1234)
