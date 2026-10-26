"""Real process/flock coverage; only the GPU inventory and spawn boundary are replaced."""
import json
import os
import subprocess
import time

import pytest
import yaml
from greedylore_autoresearch_grader.gpu import (
    GpuDevice,
    GpuInventory,
    IdleGpuPolicy,
    ResourceBusy,
    reserve_idle_gpus,
)
from greedylore_autoresearch_grader.runtime import ProcessIdentity


def inventory():
    return GpuInventory(tuple(GpuDevice(i, f'GPU-{i}', 'A100', 0, 0, ())
                              for i in range(4)), 'test topology')


def wait_until(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    assert predicate()


@pytest.fixture
def lease_env(tmp_path, contract, monkeypatch):
    from autoresearch import lease as module
    coral = tmp_path / '.coral'
    (coral / 'private').mkdir(parents=True)
    (coral / 'private/eval_contract.yaml').write_text(contract.path.read_text())
    locks = tmp_path / 'locks'
    (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'gpu_lock_dir': str(locks),
        'gpu_poll_interval_seconds': .03, 'gpu_wait_timeout_seconds': 5,
        'timing_monitor_interval_seconds': .03}, 'parallel': {'max_workers': 1}}}))
    state = {'nodes': {'N001': {'research': {'agent_id': 'R001', 'status': 'active'}}}}
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(module.GpuInventory, 'from_nvidia_smi', inventory)
    children = []

    def spawn(coral_dir, node_id, token):
        pid = os.fork()
        if pid == 0:
            os.closerange(3, 4096)
            try:
                module._keeper(coral_dir, node_id, token)
            finally:
                os._exit(0)
        children.append(pid)
        return pid

    monkeypatch.setattr(module, '_spawn', spawn)
    identity = ProcessIdentity.read(os.getpid())
    context = {'agent_id': 'R001', 'agent_root_pid': identity.pid,
               'agent_start_time': identity.start_time, 'host_pid': identity.pid,
               'host_start_time': identity.start_time}
    lease = module.ResearchLease(coral, 'N001')
    yield lease, context, locks, state
    lease.release()
    for pid in children:
        wait_until(lambda: ProcessIdentity.read(pid) is None)
        os.waitpid(pid, 0)


def ready(lease, context):
    lease.ensure(context)
    wait_until(lease.check)


def test_keeper_preserves_four_gpu_locks_across_borrower_exit(lease_env):
    lease, context, locks, _ = lease_env
    ready(lease, context)
    first = lease.borrow()
    assert first.uuids == ('GPU-0', 'GPU-1', 'GPU-2', 'GPU-3')
    assert first._streams == []
    with first:
        pass
    assert lease.ensure(context)
    assert lease.borrow().uuids == first.uuids
    with pytest.raises(ResourceBusy):
        reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5), inventory_provider=inventory)
    lease.release()
    wait_until(lambda: not lease.check())
    wait_until(lambda: can_reserve(locks))


def can_reserve(locks):
    try:
        held = reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5), inventory_provider=inventory)
    except ResourceBusy:
        return False
    held.release()
    return True


@pytest.mark.parametrize('lost', ['host', 'agent'])
def test_keeper_releases_when_bound_process_dies(lease_env, lost):
    lease, context, locks, _ = lease_env
    child = subprocess.Popen(['sleep', '30'])
    identity = ProcessIdentity.read(child.pid)
    prefix = 'host' if lost == 'host' else 'agent_root'
    context[f'{prefix}_pid'] = identity.pid
    context['host_start_time' if lost == 'host' else 'agent_start_time'] = identity.start_time
    try:
        ready(lease, context)
        child.terminate()
        child.wait()
        wait_until(lambda: can_reserve(locks))
        assert not lease.check()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize('ending', ['cancel', 'transfer'])
def test_keeper_releases_and_stops_only_registered_job(lease_env, ending):
    lease, context, locks, state = lease_env
    ready(lease, context)
    job = subprocess.Popen(['sleep', '30'], start_new_session=True)
    unrelated = subprocess.Popen(['sleep', '30'], start_new_session=True)
    try:
        lease.register_job(job.pid)
        if ending == 'cancel':
            state['nodes']['N001']['research']['status'] = 'cancelled'
            (lease.coral_dir / 'private/autoresearch_state.json').write_text(json.dumps(state))
        else:
            path = lease.coral_dir / 'private/research_staged/N001.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'transfer_terminal': True}))
        wait_until(lambda: job.poll() is not None)
        assert unrelated.poll() is None
        wait_until(lambda: can_reserve(locks))
    finally:
        for process in (job, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_rejects_forged_identity_and_inactive_assignment(lease_env):
    lease, context, _, state = lease_env
    context['agent_start_time'] += 1
    with pytest.raises(PermissionError):
        lease.ensure(context)
    context['agent_start_time'] -= 1
    state['nodes']['N001']['research']['agent_id'] = 'someone_else'
    (lease.coral_dir / 'private/autoresearch_state.json').write_text(json.dumps(state))
    with pytest.raises(PermissionError):
        lease.ensure(context)


def test_keeper_survives_request_process_return(lease_env):
    lease, context, locks, _ = lease_env
    request = os.fork()
    if request == 0:
        try:
            lease.ensure(context)
        finally:
            os._exit(0)
    os.waitpid(request, 0)
    wait_until(lease.check)
    keeper = json.loads(lease.path.read_text())['keeper']
    assert ProcessIdentity.read(keeper['pid']) is not None
    assert lease.borrow().uuids == ('GPU-0', 'GPU-1', 'GPU-2', 'GPU-3')
    with pytest.raises(ResourceBusy):
        reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5), inventory_provider=inventory)
    lease.release()
    wait_until(lambda: ProcessIdentity.read(keeper['pid']) is None)
    assert can_reserve(locks)


def test_clear_job_is_idempotent(lease_env):
    lease, context, _, _ = lease_env
    ready(lease, context)
    lease.clear_job(123)
    lease.clear_job(123)


def test_rejects_non_four_gpu_contract(lease_env):
    lease, context, _, _ = lease_env
    path = lease.coral_dir / 'private/eval_contract.yaml'
    raw = yaml.safe_load(path.read_text())
    raw['hardware']['gpu_count'] = 3
    path.write_text(yaml.safe_dump(raw))
    lease.ensure(context)
    wait_until(lambda: json.loads(lease.path.read_text())['status'] == 'released')
    assert not lease.check()
    assert 'four' in json.loads(lease.path.read_text())['error']


def test_interference_during_thinking_releases_lease(lease_env, monkeypatch):
    from autoresearch import lease as module
    lease, context, locks, _ = lease_env
    inventory_file = lease.coral_dir / 'private/interference'

    def changing_inventory():
        sample = inventory()
        if inventory_file.exists():
            from dataclasses import replace
            return GpuInventory((replace(sample.devices[0], compute_pids=(12345678,)),
                                 *sample.devices[1:]), sample.topology)
        return sample

    monkeypatch.setattr(module.GpuInventory, 'from_nvidia_smi', changing_inventory)
    ready(lease, context)
    inventory_file.touch()
    wait_until(lambda: not lease.check())
    wait_until(lambda: can_reserve(locks))


def test_release_stops_registered_jobs_with_rank_sessions(lease_env):
    import sys
    lease, context, _, _ = lease_env
    ready(lease, context)
    code = ('import subprocess,time; '
            'p=subprocess.Popen(["sleep","30"],start_new_session=True); '
            'print(p.pid,flush=True); time.sleep(30)')
    job = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                           stdout=subprocess.PIPE, text=True)
    rank = int(job.stdout.readline())
    try:
        lease.register_job(job.pid)
        lease.release()
        wait_until(lambda: job.poll() is not None)
        wait_until(lambda: ProcessIdentity.read(rank) is None)
    finally:
        if job.poll() is None:
            job.kill()
        job.wait()
        identity = ProcessIdentity.read(rank)
        if identity:
            os.kill(rank, 9)
        job.stdout.close()


def test_keeper_retains_execution_admission_while_idle(lease_env, monkeypatch):
    from autoresearch import lease as module
    lease, context, locks, _ = lease_env

    def eight_cards():
        return GpuInventory(tuple(GpuDevice(i, f'GPU-{i}', 'A100', 0, 0, ())
                                  for i in range(8)), 'topology')

    monkeypatch.setattr(module.GpuInventory, 'from_nvidia_smi', eight_cards)
    ready(lease, context)
    with pytest.raises(ResourceBusy):
        reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5),
                          inventory_provider=eight_cards,
                          execution_group=str(lease.coral_dir), execution_limit=1)
    # The remaining cards are available to a different execution group.
    with reserve_idle_gpus(4, locks, IdleGpuPolicy(0, 512, 5),
                           inventory_provider=eight_cards, execution_group='other') as other:
        assert other.uuids == ('GPU-4', 'GPU-5', 'GPU-6', 'GPU-7')
