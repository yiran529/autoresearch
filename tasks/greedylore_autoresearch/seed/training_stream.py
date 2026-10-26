"""Deterministic streamed-data replay and optimizer-boundary checkpoint scheduling.

The staged C4 runner uses workers=0: worker prefetch/scheduling is deliberately
unsupported. Replay reads consumed batches again and requires unchanged source
content, tokenizer, fixed shuffle/order and configuration. Tokenization must be
deterministic; random data transforms interleaved with model RNG are unsupported.
Replay cost is linear in consumed batches; no large data buffers are serialized.
"""
from pathlib import Path

from diagnostic_checkpoint import capture_rng_state, restore_rng_state


class ReplayTrainingStream:
    def __init__(self, batch_factory, *, identity):
        if identity.get('workers') != 0:
            raise ValueError('exact training stream replay requires workers=0')
        self.batch_factory = batch_factory
        self.identity = dict(identity)
        self.consumed_batches = 0
        self.initial_rng = capture_rng_state()
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.batch_factory())
        batch = next(self.iterator)
        self.consumed_batches += 1
        return batch

    def state_dict(self):
        return {'version': 1, 'replay': 'exact', 'identity': self.identity,
                'consumed_batches': self.consumed_batches, 'initial_rng': self.initial_rng}

    def load_state_dict(self, state):
        if (state.get('version') != 1 or state.get('replay') != 'exact'
                or state.get('identity') != self.identity):
            raise ValueError('checkpoint training stream identity mismatch')
        count = state.get('consumed_batches')
        if type(count) is not int or count < 0:
            raise ValueError('checkpoint training stream position is invalid')
        ambient_rng = capture_rng_state()
        try:
            restore_rng_state(state['initial_rng'])
            iterator = iter(self.batch_factory())
            for _ in range(count):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise ValueError('checkpoint training stream exhausted during replay') from exc
            self.iterator = iterator
            self.consumed_batches = count
            self.initial_rng = state['initial_rng']
        finally:
            restore_rng_state(ambient_rng)


class CheckpointSchedule:
    def __init__(self, root, interval_steps=1000):
        if type(interval_steps) is not int or interval_steps <= 0:
            raise ValueError('checkpoint_interval_steps must be positive')
        self.root = Path(root)
        self.interval_steps = interval_steps
        self.saved_steps = set()

    def path(self, update_step):
        return self.root / f'step-{update_step:08d}'

    def due(self, update_step, *, terminal=False):
        return (update_step > 0 and update_step not in self.saved_steps
                and (terminal or update_step % self.interval_steps == 0))

    def mark_saved(self, update_step):
        self.saved_steps.add(update_step)
