"""Distributed training checkpoints for diagnostics and opt-in exact stream replay.

Load only trusted, recorded checkpoints. Use the matching code snapshot and world
size. DDP bucket order must match the saved order before reusing bucket-indexed
compressor state; SubspaceState checks this on the next hook invocation.
"""

import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _atomic(path, writer):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pack(value):
    if isinstance(value, torch.Generator):
        return {'__generator__': str(value.device), 'state': value.get_state()}
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, (str, int, float, bool)) for key in value):
            raise TypeError('unsupported checkpoint dictionary key')
        return {key: _pack(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(child) for child in value)
    raise TypeError(f'unsupported checkpoint state: {type(value).__name__}')


def _unpack(value, device):
    if isinstance(value, dict):
        if '__generator__' in value:
            target = 'cpu' if value['__generator__'] == 'cpu' else device
            return torch.Generator(device=target).set_state(value['state'].cpu())
        return {key: _unpack(child, device) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_unpack(child, device) for child in value)
    return value


def _module(model):
    return model.module if hasattr(model, 'module') else model


def _layout(model):
    return [[name, list(tensor.shape), str(tensor.dtype)]
            for name, tensor in _module(model).state_dict().items()]


def _identity(value):
    return f'{type(value).__module__}.{type(value).__qualname__}'


def register_checkpoint_hook(model, state, hook):
    """Quality-only wrapper: record/check the parameter order of bucket state.

    Timing registers its original hook directly and never enters this wrapper.
    Opaque/slot-only states are left alone; unsupported serialization is reported
    as unavailable at save time, not as a training failure.
    """
    if state is None or not hasattr(state, '__dict__'):
        model.register_comm_hook(state, hook)
        return
    state._checkpoint_bucket_layout = {}

    def recorded_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        layout = [(state.param_to_name[param], tuple(param.shape)) for param in bucket.parameters()]
        expected = getattr(state, '_checkpoint_expected_layout', None)
        if expected is not None and expected.get(bucket.index()) != layout:
            raise ValueError('checkpoint DDP bucket layout mismatch; establish matching buckets before restoration')
        state._checkpoint_bucket_layout[bucket.index()] = layout
        return hook(state, bucket)

    model.register_comm_hook(state, recorded_hook)


def capture_rng_state():
    numpy = np.random.get_state()
    return {
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        'python_rng': random.getstate(),
        'numpy_rng': (numpy[0], numpy[1].tolist(), *numpy[2:]),
    }


def restore_rng_state(state, device=None):
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        if (device is not None and torch.device(device).type != 'cuda') or not torch.cuda.is_available():
            raise ValueError('CUDA RNG restoration requires CUDA; use restore_rng=False for CPU inspection')
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device if device is not None else 'cuda')
    random.setstate(state['python_rng'])
    numpy = state['numpy_rng']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))


def save_final_checkpoint(root, model, optimizer, scheduler, hook_state, *, config,
                          update_step, batch=None, min_free_bytes=1024**3,
                          training_state=None, data_stream_state=None):
    """All ranks participate. File failures disable diagnostics, not quality scoring.

    One model file plus per-rank state; manifest is the final completion marker.
    Optional training/data-stream state enables staged continuation. Legacy final
    diagnostic callers retain the original API and unavailable-on-error behavior.
    """
    root = Path(root)
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if (root / 'manifest.json').exists():
        return {'status': 'unavailable', 'reason': 'checkpoint already exists; refusing replacement'}
    files = {}
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError('checkpoint root must not be a symlink')
        estimated = sum(t.numel() * t.element_size() for t in model.parameters()) * (8 * world + 1)
        if shutil.disk_usage(root).free < min_free_bytes + estimated:
            raise OSError('insufficient free disk space for final checkpoint')
        excluded = {'process_group', 'param_to_name', 'param_state'}
        hook = None if hook_state is None else {
            'class': _identity(hook_state),
            'state': _pack({key: value for key, value in vars(hook_state).items()
                            if key not in excluded}),
        }
        if training_state is not None and training_state.get('update_step') != update_step:
            raise ValueError('training_state update_step mismatch')
        state = {
            'optimizer_class': _identity(optimizer), 'optimizer': optimizer.state_dict(),
            'scheduler_class': _identity(scheduler), 'scheduler': scheduler.state_dict(),
            'hook': hook, 'batch': batch, **capture_rng_state(),
            'training_state': _pack(training_state),
            'data_stream_state': _pack(data_stream_state),
        }
        items = {f'rank-{rank}.pt': state}
        if rank == 0:
            items['model.pt'] = _module(model).state_dict()
        for name, value in items.items():
            path = root / name
            _atomic(path, lambda stream, value=value: torch.save(value, stream))
            files[name] = {'sha256': _hash(path), 'size_bytes': path.stat().st_size}
        local = {'files': files}
    except Exception as exc:
        local = {'error': f'{type(exc).__name__}: {exc}'}
    ranks = [None] * world
    if distributed:
        dist.all_gather_object(ranks, local)
    else:
        ranks[0] = local
    errors = [item['error'] for item in ranks if 'error' in item]
    manifest = {
        'version': 1, 'status': 'unavailable' if errors else 'complete',
        'world_size': world, 'update_step': update_step,
        'contract_hash': config.get('contract_hash', ''),
        'config': config, 'model_layout': _layout(model),
        'files': {} if errors else {name: info for item in ranks for name, info in item['files'].items()},
        'data_stream_replay': 'exact' if data_stream_state is not None else 'unavailable',
        'limitations': (['Replay requires unchanged deterministic data, tokenizer, code and ordered DataLoader settings.']
                         if data_stream_state is not None else
                         ['No original dataloader position is saved; not exact training-stream replay.'])
                        + ['Use matching code, world size and DDP bucket layout.'],
        'errors': errors,
    }
    if rank == 0:
        try:
            _atomic(root / 'manifest.json', lambda stream: stream.write(
                (json.dumps(manifest, indent=2) + '\n').encode()))
        except Exception as exc:
            manifest = {'status': 'unavailable', 'reason': str(exc)}
    result = [manifest if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(result, src=0)
    return result[0]


def load_checkpoint(root, model, optimizer=None, scheduler=None, hook_state=None, *,
                    contract_hash=None, rank=None, world_size=None, restore_rng=True,
                    data_stream=None, expected_config=None):
    """Strict restoration; optional optimizer/hook objects must match saved classes.

    Returns manifest, paired batch and optional full training/stream state. If a
    data_stream is supplied, replay its position before restoring main RNG. No
    implicit EF reset, world-size conversion or formal Attempt is performed.
    """
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    world = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)
    rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
    if manifest.get('version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('checkpoint is incomplete or unsupported')
    if manifest['world_size'] != world or not 0 <= rank < world:
        raise ValueError('checkpoint world size/rank mismatch')
    if contract_hash is not None and manifest['contract_hash'] != contract_hash:
        raise ValueError('checkpoint contract mismatch')
    if expected_config is not None and manifest['config'] != expected_config:
        raise ValueError('checkpoint code/config mismatch')
    if manifest['model_layout'] != _layout(model):
        raise ValueError('checkpoint model layout mismatch')
    expected = {'model.pt', *(f'rank-{i}.pt' for i in range(world))}
    if set(manifest['files']) != expected:
        raise ValueError('checkpoint rank files are incomplete')
    for name, record in manifest['files'].items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != record['size_bytes'] or _hash(path) != record['sha256']:
            raise ValueError('checkpoint integrity mismatch')
    device = next(model.parameters()).device
    state = torch.load(root / f'rank-{rank}.pt', map_location=device, weights_only=True)
    for item, key in ((optimizer, 'optimizer'), (scheduler, 'scheduler')):
        if item is not None and _identity(item) != state[key + '_class']:
            raise ValueError(f'checkpoint {key} class mismatch')
    hook = state['hook']
    if hook is not None and optimizer is not None and hook_state is None:
        raise ValueError('checkpoint compressor state is required for optimizer continuation')
    if hook_state is not None and (hook is None or _identity(hook_state) != hook['class']):
        raise ValueError('checkpoint compressor class mismatch')
    if (hook_state is not None and '_checkpoint_bucket_layout' in hook['state']
            and not hasattr(hook_state, '_checkpoint_bucket_layout')):
        raise ValueError('register_checkpoint_hook before restoring DDP bucket state')
    _module(model).load_state_dict(torch.load(root / 'model.pt', map_location=device, weights_only=True))
    for item, key in ((optimizer, 'optimizer'), (scheduler, 'scheduler')):
        if item is not None:
            item.load_state_dict(state[key])
    if hook_state is not None:
        restored = _unpack(hook['state'], device)
        vars(hook_state).update(restored)
        if '_checkpoint_bucket_layout' in restored:
            hook_state._checkpoint_expected_layout = restored['_checkpoint_bucket_layout'].copy()
    training_state = state.get('training_state')
    stream_state = state.get('data_stream_state')
    if training_state is not None and training_state.get('update_step') != manifest['update_step']:
        raise ValueError('checkpoint training_state update_step mismatch')
    if data_stream is not None:
        if stream_state is None:
            raise ValueError('checkpoint has no training stream state')
        data_stream.load_state_dict(stream_state)
    if restore_rng:
        restore_rng_state(state, device=device)
    return {'manifest': manifest, 'batch': state['batch'], 'hook_state': hook,
            'training_state': training_state, 'data_stream_state': stream_state}


def prepare_ddp_for_resume(root, model, *, contract_hash=None, expected_config=None):
    """Establish the fresh reducer's stable buckets before registering its hook.

    DDP rebuilds buckets after its first backward. Two zero-loss backwards use
    only the saved paired batch, perform no optimizer updates, and do not advance
    the training stream. The caller then registers the compressor hook and calls
    load_checkpoint again to restore model, optimizer, compressor, stream and RNG.
    The saved bucket-layout guard rejects incompatible layouts on the next hook.
    """
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError('prepare_ddp_for_resume requires a DDP model')
    restored = load_checkpoint(root, model, contract_hash=contract_hash,
                               expected_config=expected_config, restore_rng=False)
    batch = restored['batch']
    if batch is None or 'labels' not in batch:
        raise ValueError('DDP resume requires a paired training batch')
    stabilize_ddp_buckets(model, batch)


def stabilize_ddp_buckets(model, batch):
    """Two zero-loss backwards establish stable buckets without a trained prefix.

    Used before the first staged update as well as on restart, so even a pause at
    update 1 has the same bucket order. Preserve RNG and buffers and clear grads;
    no optimizer/scheduler or training-stream progress occurs here. The caller
    must register the compressor hook afterwards.
    """
    rng = capture_rng_state()
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    try:
        model.train()
        for _ in range(2):
            (model(**batch).loss * 0).backward()
            model.zero_grad(set_to_none=True)
    finally:
        restore_rng_state(rng)
        with torch.no_grad():
            for name, tensor in model.named_buffers():
                if name in buffers:
                    tensor.copy_(buffers[name])
        model.zero_grad(set_to_none=True)
