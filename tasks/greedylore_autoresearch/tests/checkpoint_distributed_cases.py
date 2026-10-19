"""Two-rank CPU/Gloo checkpoint continuation with the real EF14 hook and optimizers."""

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, seed, output):
    sys.path.insert(0, seed)
    from comm_hooks.subspace_hook import SubspaceState, subspace_hook
    from diagnostic_checkpoint import (
        load_checkpoint,
        register_checkpoint_hook,
        save_final_checkpoint,
    )
    from optimizer import build_muon_optimizer

    output = Path(output)
    dist.init_process_group('gloo', init_method=f'file://{output}/rendezvous', rank=rank, world_size=2)
    try:
        for name in ('adamw', 'muon'):
            torch.manual_seed(42)
            model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4, 4))
            optimizer = (torch.optim.AdamW(model.parameters(), lr=0.01) if name == 'adamw' else
                         build_muon_optimizer(model, lr=0.01, scalar_lr=0.001))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
            hook = SubspaceState(dist.group.WORLD, matrix_approximation_rank=1,
                                  start_compress_iter=0, update_proj_gap=20,
                                  use_error_feedback='ef14', uncompressed_names=[])
            hook.param_to_name = {param: key for key, param in model.named_parameters()}
            register_checkpoint_hook(model, hook, subspace_hook)
            batch = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10 + rank

            def step():
                optimizer.zero_grad()
                model(batch).square().mean().backward()
                optimizer.step()
                scheduler.step()

            for _ in range(3):
                step()  # Let DDP establish the stable bucket layout.
            root = output / name
            result = save_final_checkpoint(root, model, optimizer, scheduler, hook,
                config={'contract_hash': 'distributed-test'}, update_step=3,
                batch={'input_ids': batch}, min_free_bytes=0)
            assert result['status'] == 'complete', result
            saved_error = {key: value.clone() for key, value in hook.error_dict.items()}
            step()
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            load_checkpoint(root, model, optimizer, scheduler, hook, contract_hash='distributed-test')
            assert all(torch.equal(value, hook.error_dict[key]) for key, value in saved_error.items())
            step()
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected.items())
            if rank == 1:
                hook.unsupported = object()
            failed = save_final_checkpoint(output / f'{name}-failed', model, optimizer, scheduler,
                                            hook, config={}, update_step=4, min_free_bytes=0)
            assert failed['status'] == 'unavailable'
            assert failed['files'] == {}
        if rank == 0:
            (output / 'passed.json').write_text(json.dumps({'ranks': 2, 'optimizers': ['adamw', 'muon'],
                'checks': ['exact continuation', 'EF state', 'rank-local failure propagation']}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(sys.argv[1], sys.argv[2]), nprocs=2, join=True)
