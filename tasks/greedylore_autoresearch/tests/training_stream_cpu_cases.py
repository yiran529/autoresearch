"""Changing deterministic data, stochastic model, and multiple process-style resumes."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def main():
    seed, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(seed))
    import diagnostic_checkpoint as checkpoint
    import training_stream as streams

    identity = dict(rank=0, world_size=1, batch_size=2, workers=0, requested_workers=4,
                    seed=19, shuffle_seed=42, repeat=True)
    seen = []

    def batches():
        # Iterator creation consumes main RNG, as torch DataLoader does.
        torch.rand(1)
        random.random()
        np.random.rand()
        for i in range(20):
            seen.append(i)
            yield {'input_ids': torch.arange(6).reshape(2, 3).float() / 10 + i,
                   'labels': torch.full((2, 2), float(i))}

    def fresh():
        torch.manual_seed(19)
        random.seed(19)
        np.random.seed(19)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        hook = SimpleNamespace(error_dict={0: torch.zeros(2)}, iteration=0)
        stream = streams.ReplayTrainingStream(batches, identity=identity)
        return model, optimizer, scheduler, hook, stream

    def step(objects):
        model, optimizer, scheduler, hook, stream = objects
        batch = next(stream)
        # Python/NumPy randomness also affects the update.
        loss = (model(batch['input_ids']) - batch['labels']).square().mean()
        loss = loss * (1 + random.random() + np.random.rand())
        loss.backward()
        hook.error_dict[0].add_(model[0].bias.grad)
        hook.iteration += 1
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return batch

    full = fresh()
    for _ in range(7):
        last = step(full)
    expected = [full[0].state_dict(), full[1].state_dict(), full[2].state_dict(),
                vars(full[3]), torch.rand(4), random.random(), np.random.rand(), next(full[4])]

    partial = fresh()
    completed = 0
    for boundary in (2, 4, 7):
        while completed < boundary:
            last = step(partial)
            completed += 1
        model, optimizer, scheduler, hook, stream = partial
        root = output / f'step-{completed:08d}'
        result = checkpoint.save_final_checkpoint(
            root, model, optimizer, scheduler, hook, config={'contract_hash': 'staged', 'code': 'abc'},
            update_step=completed, batch=last, min_free_bytes=0,
            training_state={'update_step': completed, 'global_step': completed, 'tokens_seen': completed * 6},
            data_stream_state=stream.state_dict(),
        )
        assert result['status'] == 'complete', result
        assert result['data_stream_replay'] == 'exact'
        # Fresh objects mimic a process restart and consume different initialization RNG.
        partial = fresh()
        restored = checkpoint.load_checkpoint(
            root, *partial[:4], contract_hash='staged', expected_config={'contract_hash': 'staged', 'code': 'abc'},
            data_stream=partial[4],
        )
        assert restored['training_state']['tokens_seen'] == completed * 6
        equal(restored['batch'], last)
        assert partial[4].consumed_batches == completed
    actual = [partial[0].state_dict(), partial[1].state_dict(), partial[2].state_dict(),
              vars(partial[3]), torch.rand(4), random.random(), np.random.rand(), next(partial[4])]
    equal(expected, actual)
    assert seen[-1] == 7  # Replay stops before fetching the next unconsumed batch.
    for field, bad in [('rank', 1), ('world_size', 2), ('batch_size', 1), ('seed', 20),
                       ('requested_workers', 2)]:
        wrong = streams.ReplayTrainingStream(batches, identity={**identity, field: bad})
        try:
            checkpoint.load_checkpoint(root, *partial[:4], data_stream=wrong)
        except ValueError as exc:
            assert 'stream' in str(exc)
        else:
            raise AssertionError(f'accepted changed stream identity: {field}')
    try:
        checkpoint.load_checkpoint(root, *partial[:4], expected_config={'code': 'changed'})
    except ValueError as exc:
        assert 'config' in str(exc)
    else:
        raise AssertionError('changed code/config accepted')

    # Scheduling is optimizer-step based, and nonperiodic termination is saved once.
    for interval, expected_steps in [(1000, [1000, 2000, 2001]), (3, [3, 6, 7])]:
        schedule = streams.CheckpointSchedule(output, interval_steps=interval)
        saved = []
        terminal = expected_steps[-1]
        for step_number in range(1, terminal + 1):
            if schedule.due(step_number):
                saved.append(step_number)
                schedule.mark_saved(step_number)
        if schedule.due(terminal, terminal=True):
            saved.append(terminal)
            schedule.mark_saved(terminal)
        assert saved == expected_steps
        assert not schedule.due(terminal, terminal=True)
        assert schedule.path(3).name == 'step-00000003'
    for bad in (0, -1):
        try:
            streams.CheckpointSchedule(output, interval_steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('nonpositive interval accepted')
    # CPU inspection must not silently restore a CUDA checkpoint into a different
    # current device; loading CUDA RNG requires the model's CUDA device.
    cuda_rng = {**checkpoint.capture_rng_state(), 'cuda_rng': torch.get_rng_state()}
    try:
        checkpoint.restore_rng_state(cuda_rng, device=torch.device('cpu'))
    except ValueError as exc:
        assert 'CUDA' in str(exc)
    else:
        raise AssertionError('CUDA RNG restored for a CPU model')
    # Reducer setup may evaluate dropout, but must not consume training RNG or
    # mutate parameters/leave gradients before the first staged update.
    class SetupModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(0.4)
        def forward(self, input_ids, labels):
            return SimpleNamespace(loss=(self.dropout(self.proj(input_ids)) - labels).square().mean())
    setup_model = SetupModel()
    before_weights = {key: value.clone() for key, value in setup_model.state_dict().items()}
    before_rng = checkpoint.capture_rng_state()
    checkpoint.stabilize_ddp_buckets(setup_model, last)
    equal(setup_model.state_dict(), before_weights)
    equal(checkpoint.capture_rng_state(), before_rng)
    assert all(param.grad is None for param in setup_model.parameters())
    # Staged-only args must activate the production compressor layout guard.
    from comm_hooks.utils import register_comm_hook_for_ddp_model
    guarded = torch.nn.Linear(2, 2)
    callbacks = []
    guarded.register_comm_hook = lambda state, callback: callbacks.append((state, callback))
    hook_args = SimpleNamespace(
        checkpoint_root='staged-only', diagnostic_checkpoint=None, compressor='topk_sync',
        compress_ratio=0.5, sparse_type='tensor', use_error_feedback='ef14',
        start_compress_iter=0, seed=19, optimizer='adamw')
    registered = register_comm_hook_for_ddp_model(guarded, None, hook_args)
    registered._checkpoint_expected_layout = {0: [('different-parameter', (2, 2))]}
    bucket = SimpleNamespace(parameters=lambda: [guarded.weight], index=lambda: 0)
    try:
        callbacks[0][1](registered, bucket)
    except ValueError as exc:
        assert 'bucket layout' in str(exc)
    else:
        raise AssertionError('staged hook registration omitted the bucket-layout guard')
    # Interrupted rank writes may leave files, but never a restorable complete manifest.
    original_atomic = checkpoint._atomic
    def interrupted(path, writer):
        if path.name == 'rank-0.pt':
            path.write_bytes(b'partial tensor')
            raise OSError('injected interrupted rank write')
        original_atomic(path, writer)
    checkpoint._atomic = interrupted
    failed_root = output / 'interrupted'
    failed = checkpoint.save_final_checkpoint(
        failed_root, *partial[:4], config={}, update_step=7, min_free_bytes=0,
        training_state={'update_step': 7}, data_stream_state=partial[4].state_dict())
    checkpoint._atomic = original_atomic
    assert failed['status'] == 'unavailable' and failed['files'] == {}
    try:
        checkpoint.load_checkpoint(failed_root, partial[0])
    except ValueError as exc:
        assert 'incomplete' in str(exc)
    else:
        raise AssertionError('interrupted checkpoint loaded')
    (failed_root / 'manifest.json').unlink()
    try:
        checkpoint.load_checkpoint(failed_root, partial[0])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('unpublished checkpoint loaded')
    print('staged CPU replay, complete states, paired batches, identity rejection and scheduling passed')


if __name__ == '__main__':
    main()
