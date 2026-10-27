"""Fresh DDP reducer restart with changing inputs, dropout and real EF14."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training_stream_cpu_cases import equal


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=(self.dropout(self.proj(input_ids)) - labels).square().mean())


def worker(rank, seed, output):
    sys.path.insert(0, seed)
    from comm_hooks.subspace_hook import SubspaceState, subspace_hook
    from diagnostic_checkpoint import (save_final_checkpoint, load_checkpoint,
                                       register_checkpoint_hook, prepare_ddp_for_resume,
                                       stabilize_ddp_buckets, capture_rng_state, restore_rng_state)
    from optimizer import build_muon_optimizer
    from training_stream import ReplayTrainingStream

    output = Path(output)
    dist.init_process_group('gloo', init_method=f'file://{output}/staged-rendezvous', rank=rank, world_size=2)
    try:
        for optimizer_name in ('adamw', 'muon'):
            config = {'contract_hash': 'staged-ddp', 'model': {'id2label': {0: 'LABEL_0'}}}
            def batches():
                torch.rand(1)  # Iterator construction is part of replay, not training RNG.
                for i in range(12):
                    x = torch.arange(8).reshape(2, 4).float() / 10 + rank / 10 + i / 20
                    yield {'input_ids': x, 'labels': x / 3}

            def fresh(resume=None):
                torch.manual_seed(31)
                model = torch.nn.parallel.DistributedDataParallel(Model())
                optimizer = (torch.optim.AdamW(model.parameters(), lr=0.01)
                             if optimizer_name == 'adamw' else
                             build_muon_optimizer(model, lr=0.01, scalar_lr=0.001))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
                if resume is not None:
                    prepare_ddp_for_resume(resume, model, contract_hash='staged-ddp',
                                           expected_config=config)
                else:
                    setup_rng = capture_rng_state()
                    stabilize_ddp_buckets(model, next(batches()))
                    restore_rng_state(setup_rng)
                hook = SubspaceState(dist.group.WORLD, matrix_approximation_rank=1,
                                     start_compress_iter=0, update_proj_gap=20,
                                     use_error_feedback='ef14', uncompressed_names=[])
                hook.param_to_name = {param: key for key, param in model.named_parameters()}
                register_checkpoint_hook(model, hook, subspace_hook)
                stream = ReplayTrainingStream(batches, identity={
                    'workers': 0, 'rank': rank, 'world_size': 2, 'batch_size': 2, 'seed': 31})
                if resume is not None:
                    load_checkpoint(resume, model, optimizer, scheduler, hook,
                                    data_stream=stream, expected_config=config)
                return model, optimizer, scheduler, hook, stream

            def step(objects):
                model, optimizer, scheduler, hook, stream = objects
                batch = next(stream)
                model(**batch).loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                return batch

            full = fresh()
            for _ in range(7):
                step(full)
            expected = [full[0].state_dict(), full[1].state_dict(), full[2].state_dict(),
                        full[3].error_dict, full[3].iter, torch.rand(3), next(full[4])]
            partial = fresh()
            completed = 0
            for boundary in (1, 3, 5):
                while completed < boundary:
                    batch = step(partial)
                    completed += 1
                root = output / optimizer_name / f'step-{completed:08d}'
                saved = save_final_checkpoint(root, *partial[:4], config=config,
                    update_step=completed, batch=batch, min_free_bytes=0,
                    training_state={'update_step': completed}, data_stream_state=partial[4].state_dict())
                assert saved['status'] == 'complete', saved
                partial = fresh(root)
            while completed < 7:
                step(partial)
                completed += 1
            actual = [partial[0].state_dict(), partial[1].state_dict(), partial[2].state_dict(),
                      partial[3].error_dict, partial[3].iter, torch.rand(3), next(partial[4])]
            equal(expected, actual)
        if rank == 0:
            (output / 'staged-passed.json').write_text(json.dumps({'ranks': 2, 'optimizers': ['adamw', 'muon']}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(sys.argv[1], sys.argv[2]), nprocs=2, join=True)
