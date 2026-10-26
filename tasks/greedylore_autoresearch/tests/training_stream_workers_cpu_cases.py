"""Ordered HF/C4 DataLoader worker replay with real multiprocessing and dropout."""
import gc
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from training_stream_cpu_cases import equal


class Tokenizer:
    def __call__(self, text, **kwargs):
        value = int(text) + 1
        worker = torch.utils.data.get_worker_info()
        return {'input_ids': torch.tensor([[value, value + 1, 0]]),
                # Test metadata proves every configured worker actually delivered
                # batches; the stochastic model below does not use this field.
                'attention_mask': torch.tensor([[1, 1, worker.id + 1]])}


def main():
    seed, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(seed))
    from c4.pept_utils.dataloader import PreprocessedIterableDataset
    from c4.run_llama_pretraining import training_batches
    from diagnostic_checkpoint import load_checkpoint, save_final_checkpoint
    from training_stream import ReplayTrainingStream

    for workers in (2, 4):
        identity = {'rank': 0, 'world_size': 1, 'batch_size': 2, 'workers': workers,
                    'requested_workers': workers, 'seed': 27, 'shuffle_seed': 42,
                    'repeat': True, 'prefetch_factor': 2, 'persistent_workers': False,
                    'ordered_delivery': True}

        ReplayTrainingStream(lambda: iter(()), identity=identity)

        def fresh(replay=True):
            torch.manual_seed(27)
            random.seed(27)
            np.random.seed(27)
            model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.3))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
            # Match C4: HF controls fixed shuffle and assigns source shards to
            # DataLoader workers; PreprocessedIterableDataset forms each batch.
            data = Dataset.from_dict({'text': [str(i) for i in range(16)]})
            # datasets 5 otherwise interleaves all four source shards into one
            # before shuffling, leaving one active worker. Keep shard parallelism
            # here to exercise actual delivery from every configured worker.
            data = data.to_iterable_dataset(num_shards=4).shuffle(
                seed=42, buffer_size=7, max_buffer_input_shards=1)
            assert data.num_shards == 4
            dataset = PreprocessedIterableDataset(data, Tokenizer(), batch_size=2, max_length=3)
            loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers)
            assert loader.num_workers == workers and loader.prefetch_factor == 2
            factory = lambda: training_batches(loader, repeat=True, batch_size=2)
            stream = ReplayTrainingStream(factory, identity=identity) if replay else iter(factory())
            return model, optimizer, scheduler, stream

        def step(objects):
            model, optimizer, scheduler, stream = objects
            batch = next(stream)
            labels = batch['input_ids'][:, :2].float() / 10
            prediction = model(batch['input_ids'].float() / 10)
            loss = (prediction - labels).square().mean()
            loss = loss * (1 + random.random() + np.random.rand())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return {**batch, 'labels': labels}

        # The reference uses the original plain DataLoader, with the same worker
        # count and preprocessing. There is no workers=0 reference substitution.
        full = fresh(replay=False)
        order = []
        delivered_workers = set()
        for _ in range(13):  # Cross the eight-batch loader exhaustion boundary.
            batch = step(full)
            order.append(batch['input_ids'].clone())
            delivered_workers.add(batch['attention_mask'][0, 2].item())
        assert delivered_workers == set(range(1, workers + 1)), delivered_workers
        expected = [full[0].state_dict(), full[1].state_dict(), full[2].state_dict(),
                    torch.rand(4), random.random(), np.random.rand(), next(full[3])]

        partial = fresh()
        completed = 0
        for boundary in (3, 7, 11):
            while completed < boundary:
                batch = step(partial)
                equal(batch['input_ids'], order[completed])
                completed += 1
            root = output / f'workers-{workers}' / f'step-{completed:08d}'
            result = save_final_checkpoint(
                root, *partial[:3], None, config={'contract_hash': 'workers'}, update_step=completed,
                batch=batch, min_free_bytes=0, training_state={'update_step': completed},
                data_stream_state=partial[3].state_dict())
            assert result['status'] == 'complete', result
            partial = fresh()
            restored = load_checkpoint(root, *partial[:3], data_stream=partial[3], contract_hash='workers')
            assert restored['data_stream_state']['identity']['workers'] == workers
            equal(restored['batch'], batch)
        while completed < 13:
            batch = step(partial)
            equal(batch['input_ids'], order[completed])
            completed += 1
        actual = [partial[0].state_dict(), partial[1].state_dict(), partial[2].state_dict(),
                  torch.rand(4), random.random(), np.random.rand(), next(partial[3])]
        equal(expected, actual)
        for field, changed in [('workers', 1), ('prefetch_factor', 3), ('ordered_delivery', False)]:
            wrong = ReplayTrainingStream(lambda: iter(()), identity={**identity, field: changed})
            try:
                wrong.load_state_dict(restored['data_stream_state'])
            except ValueError as exc:
                assert 'stream' in str(exc)
            else:
                raise AssertionError(f'worker replay accepted changed {field}')
        del full, partial, actual, expected, restored
        gc.collect()
    assert not multiprocessing.active_children(), multiprocessing.active_children()
    print('real ordered C4/HF workers 2 and 4 match original-loader data, dropout, optimizer, scheduler, RNG and fresh replay across exhaustion')


if __name__ == '__main__':
    main()
