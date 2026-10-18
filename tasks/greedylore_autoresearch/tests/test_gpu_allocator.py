import subprocess
import sys
from types import SimpleNamespace

import pytest
from greedylore_autoresearch_grader.gpu import (
    GpuDevice,
    GpuInventory,
    IdleGpuPolicy,
    InfraFailure,
    reserve_idle_gpus,
)


@pytest.mark.parametrize("slow_query", ["gpu", "apps", "topology"])
def test_inventory_tolerates_driver_query_latency_under_load(tmp_path, slow_query):
    """Real subprocess delays reproduce the busy-host one-second timeout."""
    def runner(command, **kwargs):
        if "--query-gpu" in " ".join(command):
            query, output = "gpu", "0, GPU-a, RTX 4090, 0000:65:00.0, 9000, 99\n"
        elif "--query-compute-apps" in " ".join(command):
            query, output = "apps", "GPU-a, 4321\n"
        else:
            query, output = "topology", "GPU0 X\n"
        code = f"import time; time.sleep({1.2 if query == slow_query else 0}); print({output!r}, end='')"
        return subprocess.run([sys.executable, "-c", code], **kwargs)

    found = GpuInventory.from_nvidia_smi(runner=runner, sysfs_root=tmp_path)
    assert found.devices[0].compute_pids == (4321,)
    assert found.devices[0].memory_used_mib == 9000


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
        with pytest.raises(InfraFailure, match="interference"):
            reservation.check_exclusive()

        launcher = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
        )
        try:
            # The test runner may itself be a session leader. Use a distinct
            # real launcher to establish that its parent is not an owned rank.
            with pytest.raises(InfraFailure, match="interference"):
                reservation.check_exclusive(launcher.pid)
            rank = ProcessIdentity.read(launcher.pid)
            assert rank is not None
            samples[:] = [replace(idle, compute_pids=(rank.pid,), compute_processes=(rank,))]
            reservation.check_exclusive(launcher.pid)
        finally:
            launcher.terminate()
            launcher.wait(timeout=5)


def test_timing_guard_accepts_launcher_rank_in_separate_session(tmp_path):
    import subprocess
    import sys
    from dataclasses import replace

    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
        "print(p.pid,flush=True); sys.stdin.read(); p.terminate(); p.wait()"
    )
    launcher = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        rank = ProcessIdentity.read(int(launcher.stdout.readline()))
        assert rank is not None
        assert rank.session_id != launcher.pid
        idle = GpuDevice(0, 'GPU-0', 'A100', 0, 0, (), 0)
        samples = [idle]
        def provider():
            return GpuInventory(tuple(samples), 'topology')
        with reserve_idle_gpus(1, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=provider) as reservation:
            samples[:] = [replace(idle, compute_pids=(rank.pid,), compute_processes=(rank,))]
            reservation.check_exclusive(launcher.pid)
    finally:
        launcher.communicate('', timeout=5)


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


@pytest.mark.parametrize("high_role", ["engineer", "analyzer"])
def test_late_high_priority_request_passes_waiting_researcher(tmp_path, high_role):
    import threading
    import time

    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4))
    def provider():
        return GpuInventory(devices, "topology")

    holder = reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=provider, source_role="researcher")
    order, errors = [], []
    low_ready, high_ready = threading.Event(), threading.Event()

    def run(role, ready, interval):
        try:
            def event(state, **_details):
                if state == "waiting":
                    ready.set()
            with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                                   inventory_provider=provider, source_role=role,
                                   wait_timeout=3, poll_interval=interval, on_event=event):
                order.append(role)
                time.sleep(.03)
        except Exception as exc:
            errors.append(exc)
            ready.set()

    low = threading.Thread(target=run, args=("researcher", low_ready, .002))
    high = threading.Thread(target=run, args=(high_role, high_ready, .03))
    try:
        low.start()
        assert low_ready.wait(2)
        high.start()
        assert high_ready.wait(2)
    finally:
        holder.release()
        low.join(4)
        if high.ident is not None:
            high.join(4)
    assert not errors
    assert order == [high_role, "researcher"]


def test_execution_limit_blocks_same_run_but_allows_other_runs(tmp_path):
    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(8))
    def provider():
        return GpuInventory(devices, "topology")

    with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), inventory_provider=provider,
                           execution_group="run-a", execution_limit=1) as first:
        with pytest.raises(InfraFailure):
            reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                              inventory_provider=provider, source_role="engineer",
                              execution_group="run-a", execution_limit=1)
        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=provider, source_role="engineer",
                               execution_group="run-b", execution_limit=1) as second:
            assert set(first.uuids).isdisjoint(second.uuids)


def test_cancelled_waiter_does_not_block_later_requests(tmp_path):
    def cancelled():
        raise InfraFailure("cancelled")

    with pytest.raises(InfraFailure, match="cancelled"):
        reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), source_role="engineer",
                          inventory_provider=inventory, check_cancelled=cancelled)
    with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), source_role="researcher",
                           inventory_provider=inventory) as reservation:
        assert reservation.indices == (0, 2, 5, 7)


def test_same_priority_analyzer_and_engineer_follow_fifo(tmp_path):
    import threading
    import time

    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4))
    def provider():
        return GpuInventory(devices, "topology")
    holder = reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), inventory_provider=provider)
    order, errors = [], []
    ready = [threading.Event(), threading.Event()]
    def run(role, index):
        try:
            def event(state, **_details):
                if state == "waiting":
                    ready[index].set()
            with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                                   inventory_provider=provider, source_role=role,
                                   wait_timeout=3, poll_interval=.03 if index == 0 else .002,
                                   on_event=event):
                order.append(role)
                time.sleep(.02)
        except Exception as exc:
            errors.append(exc)
            ready[index].set()
    first = threading.Thread(target=run, args=("engineer", 0))
    second = threading.Thread(target=run, args=("analyzer", 1))
    try:
        first.start()
        assert ready[0].wait(2)
        second.start()
        assert ready[1].wait(2)
    finally:
        holder.release()
        first.join(4)
        if second.ident is not None:
            second.join(4)
    assert not errors
    assert order == ["engineer", "analyzer"]


def test_infeasible_high_request_does_not_block_shared_quality(tmp_path):
    import os
    import threading

    from greedylore_autoresearch_grader.runtime import ProcessIdentity

    foreign = ProcessIdentity(99999999, os.getuid() + 1, 1, 1, 1)
    devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 900, 90, (foreign.pid,), 0,
                              (foreign,)) for i in range(4))
    def provider():
        return GpuInventory(devices, "topology")
    waiting, stop = threading.Event(), threading.Event()
    errors = []
    def high():
        def cancel():
            if stop.is_set():
                raise InfraFailure("cancelled")
        try:
            reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                              source_role="engineer", inventory_provider=provider,
                              wait_timeout=3, poll_interval=.01, check_cancelled=cancel,
                              on_event=lambda state, **_: waiting.set() if state == "waiting" else None)
        except InfraFailure as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=high)
    worker.start()
    try:
        assert waiting.wait(2)
        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=provider, source_role="researcher",
                               allow_other_users=True) as reservation:
            assert reservation.indices == (0, 1, 2, 3)
    finally:
        stop.set()
        worker.join(4)
    assert errors == ["cancelled"]


def test_crashed_reservation_owner_is_removed_from_queue(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("fork")
    ready, release = context.Queue(), context.Event()
    holder = context.Process(target=_hold_cards, args=(tmp_path, ready, release))
    holder.start()
    try:
        assert ready.get(timeout=5) == (0, 1, 2, 3)
        holder.kill()
        holder.join(5)
        devices = tuple(GpuDevice(i, f"GPU-{i}", "A100", 0, 0, (), 0) for i in range(4))
        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5), source_role="engineer",
                               inventory_provider=lambda: GpuInventory(devices, "topology")) as held:
            assert held.indices == (0, 1, 2, 3)
        assert not list((tmp_path / "requests").glob("*.json"))
    finally:
        if holder.is_alive():
            holder.kill()
            holder.join()


def test_expired_waiter_cannot_claim_cards_after_delayed_wakeup(tmp_path):
    import time

    samples = iter([inventory(busy=True), inventory(), inventory()])
    def delayed_event(state, **_details):
        if state == "waiting":
            time.sleep(.03)
    with pytest.raises(InfraFailure, match="timed out"):
        with reserve_idle_gpus(4, tmp_path, IdleGpuPolicy(0, 512, 5),
                               inventory_provider=lambda: next(samples), wait_timeout=.01,
                               poll_interval=.001, on_event=delayed_event):
            pass
