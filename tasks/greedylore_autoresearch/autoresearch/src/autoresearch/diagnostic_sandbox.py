"""Fail-closed Linux isolation for diagnostic subprocesses, not formal grading."""

from __future__ import annotations

import errno
import os
import shutil
import signal
import stat
import subprocess
import time
import uuid
from pathlib import Path

import yaml

from coral.grader.timeouts import DIAGNOSTIC_QUEUE_TIMEOUT

_NVIDIA_DRIVER_ROOT = Path('/proc/driver/nvidia/gpus')


def device_minors_for_uuids(uuids, *, driver_root=None) -> list[int]:
    """Resolve allocated identities, never assume NVML index equals device minor."""
    root = _NVIDIA_DRIVER_ROOT if driver_root is None else Path(driver_root)
    requested = tuple(uuids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('invalid NVIDIA device mapping request')
    found = {}
    for path in root.glob('*/information'):
        fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
        uuid = fields.get('GPU UUID', '').strip()
        if uuid not in requested:
            continue
        minor = fields.get('Device Minor', '').strip()
        if not minor.isdecimal() or uuid in found or int(minor) in found.values():
            raise ValueError(f'invalid NVIDIA device mapping for {uuid}')
        found[uuid] = int(minor)
    missing = set(requested) - found.keys()
    if missing:
        raise ValueError('NVIDIA device mapping unavailable for ' + ', '.join(sorted(missing)))
    return [found[uuid] for uuid in requested]


def reserve_idle_gpus(*args, **kwargs):
    # Only the host diagnostic executor has the grader package installed.
    from greedylore_autoresearch_grader.gpu import reserve_idle_gpus as reserve

    return reserve(*args, **kwargs)


def _no_symlinks(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f'diagnostic mount traverses a symlink: {path}')


def isolation_available() -> bool:
    """Probe namespace creation, not merely whether an executable is on PATH."""
    binary = shutil.which('bwrap')
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, '--unshare-all', '--die-with-parent', '--ro-bind', '/', '/',
             '--', '/usr/bin/true'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={'PATH': os.defpath}, timeout=3,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_sandbox(workspace: Path, readonly_paths, devices, argv, *, network=False,
                  write_sources=None, checkpoint_paths=()) -> list[str]:
    """Create a minimal namespace; devices are NVIDIA minors, not NVML indices."""
    workspace = Path(workspace).absolute()
    for path in (workspace, workspace / 'experiments', workspace / 'results'):
        _no_symlinks(path)
    command = [shutil.which('bwrap') or 'bwrap', '--die-with-parent', '--unshare-all', '--new-session',
               '--cap-drop', 'ALL']
    if network:
        command.append('--share-net')
    # No host /home, /tmp, /run or /proc is inherited.
    for name in ('/usr', '/bin', '/lib', '/lib64', '/etc/ld.so.cache', '/etc/alternatives'):
        path = Path(name)
        if path.exists():
            if path.is_symlink():
                command.extend(['--symlink', os.readlink(path), name])
            else:
                command.extend(['--ro-bind', name, name])
    command.extend(['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                    '--tmpfs', '/dev/shm', '--dir', '/run'])
    coral_root = next((parent for parent in workspace.parents if parent.name == '.coral'), None)
    checkpoints = {Path(value).absolute() for value in checkpoint_paths}
    for path in checkpoints:
        _no_symlinks(path)
        if (coral_root is None or not path.is_relative_to(coral_root / 'public/eval_logs')
                or path.parent.name != 'checkpoints' or not path.is_dir()):
            raise ValueError('invalid host-selected checkpoint mount')
    for value in readonly_paths:
        path = Path(value).absolute()
        _no_symlinks(path)
        if 'private' in path.parts or not path.exists() or path == Path('/'):
            raise ValueError('diagnostic runtime mount is private, absent or too broad')
        if coral_root is not None:
            own_task = workspace.parent
            if coral_root.is_relative_to(path):
                raise ValueError('diagnostic runtime mount is too broad')
            if ('.coral' in path.parts and not path.is_relative_to(own_task)
                    and path != coral_root / 'public/trusted.sock' and path not in checkpoints):
                raise ValueError('diagnostic runtime mount exposes another task or private run data')
        command.extend(['--ro-bind', str(path), str(path)])
    command.extend(['--ro-bind', str(workspace), str(workspace)])
    for name in ('experiments', 'results'):
        path = str(workspace / name)
        source = Path((write_sources or {}).get(name, path))
        _no_symlinks(source)
        if not source.is_relative_to(workspace.parent):
            raise ValueError('diagnostic writable mount escapes this task')
        command.extend(['--bind', str(source), path])
    for index in devices:
        path = f'/dev/nvidia{index}'
        command.extend(['--dev-bind', path, path])
    if devices:
        for name in ('nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools'):
            path = Path('/dev') / name
            if path.exists():
                command.extend(['--dev-bind', str(path), str(path)])
        if Path('/sys').is_dir():
            command.extend(['--ro-bind', '/sys', '/sys'])
    command.extend(['--chdir', str(workspace), '--', *argv])
    return command


def _directory_fd(path: Path, *, create=False) -> int:
    """Pin every component without ever following an intermediate symlink."""
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).absolute().parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def snapshot_files(source: Path, target: Path, limit=64 * 1024 * 1024) -> None:
    """FD-relative regular-file copies, bounded by actual bytes and entry count."""
    source_fd = _directory_fd(source)
    try:
        target_fd = _directory_fd(target, create=True)
    except BaseException:
        os.close(source_fd)
        raise
    size = entries = 0

    def copy_tree(src, dst):
        nonlocal size, entries
        with os.scandir(src) as children:
            for entry in children:
                entries += 1
                if entries > 4096:
                    raise ValueError('diagnostic archive entry budget exceeded')
                try:
                    child = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                    dir_fd=src)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENXIO}:
                        continue
                    raise
                try:
                    info = os.fstat(child)
                    if stat.S_ISDIR(info.st_mode):
                        try:
                            os.mkdir(entry.name, dir_fd=dst)
                        except FileExistsError:
                            pass
                        directory = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=dst)
                        try:
                            copy_tree(child, directory)
                        finally:
                            os.close(directory)
                    elif stat.S_ISREG(info.st_mode):
                        temporary = f'.snapshot-{uuid.uuid4().hex}'
                        output_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                            0o600, dir_fd=dst)
                        os.fchmod(output_fd, info.st_mode & 0o700)
                        try:
                            with os.fdopen(output_fd, 'wb') as output:
                                while True:
                                    block = os.read(child, min(1024 * 1024, limit - size + 1))
                                    if not block:
                                        break
                                    size += len(block)
                                    if size > limit:
                                        raise ValueError('diagnostic archive byte budget exceeded')
                                    output.write(block)
                                output.flush()
                                os.fsync(output.fileno())
                            # Replace, never truncate a model-created link/hardlink.
                            os.replace(temporary, entry.name, src_dir_fd=dst, dst_dir_fd=dst)
                        finally:
                            try:
                                os.unlink(temporary, dir_fd=dst)
                            except FileNotFoundError:
                                pass
                finally:
                    os.close(child)
    try:
        copy_tree(source_fd, target_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)


def execute_diagnostic(task, payload, cancelled) -> dict:
    """Reserve existing GPU locks, execute only inside bubblewrap, then release."""
    if not shutil.which('bwrap'):
        return {'status': 'unavailable', 'started': False,
                'reason': 'bubblewrap is required for diagnostic filesystem/device isolation'}
    from greedylore_autoresearch_grader.gpu import IdleGpuPolicy, default_lock_dir

    state = task.read()
    runtime = state.get('runtime', {})
    # Validate all mount paths before acquiring GPUs.
    record = task.records / payload['request_id']
    try:
        build_sandbox(task.workspace, runtime.get('readonly_paths', []), [], payload['argv'],
                      checkpoint_paths=state.get('checkpoint_paths', []))
    except (ValueError, OSError) as exc:
        return {'status': 'unavailable', 'started': False, 'reason': str(exc)}
    queued = time.monotonic()
    try:
        config_path = task.coral_dir / 'config.yaml'
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        execution_limit = config.get('grader', {}).get('parallel', {}).get('max_workers', 1)
        allow_other_users = config.get('grader', {}).get('args', {}).get(
            'pretraining_allow_other_users', False)
        reservation = reserve_idle_gpus(
            payload['gpus'], runtime.get('gpu_lock_dir') or default_lock_dir(),
            IdleGpuPolicy(0, 512, 5), wait_timeout=DIAGNOSTIC_QUEUE_TIMEOUT, poll_interval=5,
            allow_other_users=allow_other_users,
            check_cancelled=cancelled,
            source_role=state.get('source_role', 'analyzer'), execution_group=str(task.coral_dir.resolve()),
            execution_limit=execution_limit,
        )
    except Exception as exc:
        return {'status': 'unavailable', 'started': False, 'reason': str(exc),
                'queue_seconds': time.monotonic() - queued}
    with reservation:
        try:
            device_minors = device_minors_for_uuids(reservation.uuids)
        except (ValueError, OSError) as exc:
            return {'status': 'unavailable', 'started': False, 'reason': str(exc),
                    'queue_seconds': time.monotonic() - queued}
        # Freeze both mutable inputs after queueing, then execute isolated copies.
        # The live Analyzer cannot change the bytes loaded by the command.
        frozen = record / 'execution'
        try:
            snapshot_files(task.workspace / 'experiments', record / 'scripts')
            snapshot_files(record / 'scripts', frozen / 'experiments')
            snapshot_files(task.workspace / 'results', record / 'input-results')
            snapshot_files(record / 'input-results', frozen / 'results')
        except (ValueError, OSError) as exc:
            return {'status': 'unavailable', 'started': False, 'reason': str(exc),
                    'queue_seconds': time.monotonic() - queued}
        result = {'started': True, 'queue_seconds': time.monotonic() - queued,
                  'gpu_indices': list(reservation.indices), 'gpu_uuids': list(reservation.uuids),
                  'gpu_device_minors': device_minors,
                  'stdout_path': str(record / 'stdout.log'),
                  'stderr_path': str(record / 'stderr.log'),
                  'results_path': str(record / 'results')}
        command = build_sandbox(task.workspace, runtime.get('readonly_paths', []),
                                device_minors, payload['argv'], write_sources={
                                    'experiments': frozen / 'experiments',
                                    'results': frozen / 'results',
                                }, checkpoint_paths=state.get('checkpoint_paths', []))
        # Do not inherit grader credentials, Python paths or its virtual environment.
        # UUID order matches the reservation; device nodes were resolved by UUID,
        # not by the independently ordered nvidia-smi indices.
        environment = {'PATH': runtime.get('path', '/usr/local/bin:/usr/bin:/bin'),
                       'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
                       'CUDA_VISIBLE_DEVICES': ','.join(reservation.uuids),
                       'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
        if state.get('source_role') == 'engineer':
            environment['PYTHONPATH'] = state['code_path']
            environment['CORAL_DIAGNOSTIC_CODE'] = state['code_path']
        started = time.monotonic()
        result['started_at'] = time.time()
        process = None
        with (record / 'stdout.log').open('wb') as stdout, (record / 'stderr.log').open('wb') as stderr:
            try:
                cancelled()
                process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                next_inventory = 0
                while process.poll() is None:
                    cancelled()
                    if time.monotonic() - started >= payload['timeout']:
                        raise TimeoutError('diagnostic exceeded execution wall-clock budget')
                    if time.monotonic() >= next_inventory:
                        if not allow_other_users:
                            reservation.check_exclusive(process.pid)
                        next_inventory = time.monotonic() + 1
                    time.sleep(0.1)
                result.update(status='completed' if process.returncode == 0 else 'failed',
                              exit_code=process.returncode)
            except Exception as exc:
                result.update(status='timed_out' if isinstance(exc, TimeoutError) else 'failed',
                              reason=str(exc))
            finally:
                if process is not None:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
                result['runtime_seconds'] = time.monotonic() - started
        # Namespace init exit kills every remaining rank, including new-session ranks.
        try:
            snapshot_files(frozen / 'results', record / 'results')
            snapshot_files(frozen / 'results', task.workspace / 'results')
        except (ValueError, OSError) as exc:
            result.update(status='failed', reason=f'could not preserve result evidence: {exc}')
        return result
