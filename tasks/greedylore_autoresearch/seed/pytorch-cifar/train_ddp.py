"""Paper-aligned CIFAR comparison: dense Muon versus GreedyLore + Muon."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset
import torchvision
from torchvision import transforms

from models.resnet import BasicBlock, ResNet
from optimizer import add_muon_args, build_muon_optimizer
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.subspace_hook import SubspaceState, _should_compress_subspace, subspace_hook


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--data-dir', default='/home/wyr/ARC-TopK-release/data')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1243)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--compressor', choices=['none', 'top_subspace'], default='none')
    parser.add_argument('--compress-rank', dest='compress_rank', type=int, default=64)
    parser.add_argument('--start-compress-iter', dest='start_compress_iter', type=int)
    parser.add_argument('--update-proj-gap', dest='update_proj_gap', type=int)
    parser.add_argument('--min-compression-rate', dest='min_compression_rate', type=float, default=2.)
    parser.add_argument('--wandb-project', default='GreedyLore-CIFAR')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-steps', type=int, default=0, help='Synthetic data; no dataset download or W&B')
    add_muon_args(parser)
    args = parser.parse_args(argv)
    if args.start_compress_iter is None:
        args.start_compress_iter = 500 if args.dataset == 'cifar10' else 4000
    if args.update_proj_gap is None:
        args.update_proj_gap = 750 if args.dataset == 'cifar10' else 1200
    if args.muon_scalar_lr is None:
        args.muon_scalar_lr = .005 if args.dataset == 'cifar10' else .0005
    if args.muon_scalar_weight_decay is None:
        args.muon_scalar_weight_decay = args.weight_decay
    if args.start_compress_iter < 0 or args.update_proj_gap < 1 or args.epochs < 1:
        parser.error('Invalid compression schedule or epoch budget')
    return args


def build_model(dataset):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10 if dataset == 'cifar10' else 100)


def evaluation_indices(size, rank, world_size):
    # DistributedSampler pads uneven datasets with duplicate evaluation samples.
    return range(rank, size, world_size)


class CommunicationState:
    def __init__(self, subspace=None):
        self.subspace = subspace
        self.dense_bits = 0
        self.payload_bits = 0
        self.steps = 0


def counted_hook(state, bucket):
    """Count per-rank logical collective input bits, excluding wire replication."""
    dense_bits = bucket.buffer().numel() * bucket.buffer().element_size() * 8
    state.dense_bits += dense_bits
    subspace = state.subspace
    payload = dense_bits
    if subspace is not None and subspace.iter >= subspace.start_compress_iter:
        refresh = (subspace.iter - subspace.start_compress_iter) % subspace.update_proj_gap == 0
        if not refresh:
            payload = 0
            for tensor, parameter in zip(bucket.gradients(), bucket.parameters()):
                rows, columns = tensor.shape[0], tensor.numel() // tensor.shape[0]
                rank = min(rows, columns, subspace.matrix_approximation_rank)
                eligible = _should_compress_subspace(rows, columns, rank, subspace.min_compression_rate)[0]
                excluded = any(name in subspace.param_to_name[parameter] for name in subspace.uncompressed_names)
                if eligible and not excluded:
                    payload += max(rows, columns) * rank * tensor.element_size() * 8
                    payload += min(rows, columns) * 32  # FP32 globally averaged score probes
                else:
                    payload += tensor.numel() * tensor.element_size() * 8
    state.payload_bits += payload
    if bucket.is_last():
        state.steps += 1
    if subspace is None:
        return _allreduce_fut(dist.group.WORLD, bucket.buffer())
    return subspace_hook(subspace, bucket)


def attach_hook(model, args):
    subspace = None
    if args.compressor == 'top_subspace':
        subspace = SubspaceState(
            dist.group.WORLD, matrix_approximation_rank=args.compress_rank,
            start_compress_iter=args.start_compress_iter, update_proj_gap=args.update_proj_gap,
            min_compression_rate=args.min_compression_rate, use_error_feedback='ef14',
            uncompressed_names=['linear'], random_seed=args.seed,
            beta_ef=0, error_inherit=0,
        )
        subspace.param_to_name = {p: name for name, p in model.named_parameters()}
    state = CommunicationState(subspace)
    model.register_comm_hook(state, counted_hook)
    return state


def make_datasets(args, rank):
    if args.smoke_steps:
        generator = torch.Generator().manual_seed(args.seed)
        classes = 10 if args.dataset == 'cifar10' else 100
        dataset = TensorDataset(torch.randn(128, 3, 32, 32, generator=generator),
                                torch.randint(classes, (128,), generator=generator))
        return dataset, dataset
    # Reuse the repository's augmentation/normalization for both arms.
    normalize = transforms.Normalize((.4914, .4822, .4465), (.2023, .1994, .2010))
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset_type = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100
    if rank == 0:
        dataset_type(args.data_dir, train=True, download=True)
        dataset_type(args.data_dir, train=False, download=True)
    dist.barrier()
    return (dataset_type(args.data_dir, train=True, download=False, transform=train_transform),
            dataset_type(args.data_dir, train=False, download=False, transform=test_transform))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local_rank)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = build_model(args.dataset).to(device)
    optimizer = build_muon_optimizer(
        model, lr=args.lr, scalar_lr=args.muon_scalar_lr, mu=args.muon_mu,
        weight_decay=args.weight_decay, scalar_weight_decay=args.muon_scalar_weight_decay,
        scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
        scalar_epsilon=args.muon_scalar_eps, muon_epsilon=args.muon_epsilon,
        adjust_lr=None if args.muon_adjust_lr == 'none' else args.muon_adjust_lr,
        compile_orthogonalization=args.muon_compile,
        distributed_orthogonalization=not args.muon_local_orthogonalization,
    )
    model = DDP(model, device_ids=[local_rank])
    hook = attach_hook(model, args)
    # Identical model initialization, rank-specific deterministic augmentation.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    trainset, testset = make_datasets(args, rank)
    sampler = DistributedSampler(trainset, seed=args.seed)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                             num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)
    testloader = DataLoader(Subset(testset, evaluation_indices(len(testset), rank, world_size)),
                            batch_size=100, num_workers=args.workers, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    config = dict(vars(args), world_size=world_size, global_batch_size=world_size * args.batch_size,
                  steps_per_epoch=len(trainloader), model='ResNet18', optimizer='muon', dtype='float32',
                  orthogonalization_dtype='bfloat16', torch_version=torch.__version__,
                  torchvision_version=torchvision.__version__, cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(local_rank),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  compression_stage='DDP gradients before Muon momentum/orthogonalization',
                  use_error_feedback='ef14', beta_ef=0, error_inherit=0,
                  scheduler='cosine to zero, epoch steps, no LR warmup',
                  evaluation_split='test', uncompressed_names=['linear'],
                  checkpoint_policy='none; failed runs cannot resume')
    tracker = None
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        if not args.smoke_steps and args.wandb_mode != 'disabled':
            import wandb
            tracker = wandb.init(project=args.wandb_project, name=output.name,
                                 dir=str(output), config=config, mode=args.wandb_mode)
            (output / 'tracker.json').write_text(json.dumps({'id': tracker.id, 'url': tracker.url}) + '\n')
        print('CONFIG ' + json.dumps(config), flush=True)
    dist.barrier()
    best_accuracy, best_epoch, update_step = 0., 0, 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        stats = torch.zeros(3, device=device, dtype=torch.float64)
        lrs = [group['lr'] for group in optimizer.param_groups]
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch + 1}, batch {batch_idx}')
            optimizer.step()
            update_step += 1
            stats += torch.stack((loss.detach().double() * len(targets),
                                  predictions.argmax(1).eq(targets).sum().double(),
                                  torch.tensor(len(targets), device=device, dtype=torch.float64)))
            if rank == 0 and (update_step == 1 or update_step % 100 == 0 or args.smoke_steps):
                print(f'TRAIN epoch={epoch+1} step={update_step} loss={loss.item():.6f}', flush=True)
            if args.smoke_steps and update_step >= args.smoke_steps:
                break
        dist.all_reduce(stats)
        # Canonical rank-0 BatchNorm buffers, and exact, unpadded test partition.
        for buffer in model.module.buffers():
            dist.broadcast(buffer, src=0)
        model.module.eval()
        evaluation = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model.module(inputs)
                evaluation += torch.stack((criterion(predictions, targets).double() * len(targets),
                                            predictions.argmax(1).eq(targets).sum().double(),
                                            torch.tensor(len(targets), device=device, dtype=torch.float64)))
        dist.all_reduce(evaluation)
        test_loss = (evaluation[0] / evaluation[2]).item()
        accuracy = (100 * evaluation[1] / evaluation[2]).item()
        if not math.isfinite(test_loss):
            raise RuntimeError('Nonfinite evaluation loss')
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch + 1
        row = dict(epoch=epoch+1, update_step=update_step, train_loss=(stats[0]/stats[2]).item(),
                   train_accuracy=(100*stats[1]/stats[2]).item(), test_loss=test_loss,
                   test_accuracy=accuracy, test_samples=int(evaluation[2].item()),
                   best_test_accuracy=best_accuracy, best_epoch=best_epoch, learning_rates=lrs,
                   ddp_dense_reference_bits=hook.dense_bits, ddp_payload_bits=hook.payload_bits,
                   muon_collective_bits=optimizer.communication_bits_stats()['total'],
                   elapsed_seconds=time.monotonic()-started)
        if rank == 0:
            with (output / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            print('EPOCH ' + json.dumps(row), flush=True)
            if tracker is not None:
                tracker.log(row, step=update_step)
        scheduler.step()
        if args.smoke_steps:
            # Catch rank divergence in actual compressed Muon updates.
            delta = torch.zeros((), device=device)
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                delta = torch.maximum(delta, (reference - parameter).abs().max())
            dist.all_reduce(delta, op=dist.ReduceOp.MAX)
            if delta.item() > 1e-6:
                raise RuntimeError(f'Parameter divergence: {delta.item()}')
            if update_step >= args.smoke_steps:
                break
    if rank == 0:
        result = dict(row, status='smoke_completed' if args.smoke_steps else 'completed',
                      final_test_accuracy=accuracy, final_test_loss=test_loss,
                      ddp_payload_scope='estimated per-rank collective inputs including warmup, SVD refresh and score probes; excludes network replication and DDP buffers',
                      muon_payload_scope='existing optimizer AllGather estimate: world_size*(world_size-1)*input_bits; different scope from per-rank DDP inputs, do not sum directly')
        (output / 'all_results.json').write_text(json.dumps(result, indent=2) + '\n')
        if tracker is not None:
            tracker.summary.update(result)
            tracker.finish()
        print('TRAINING_COMPLETED', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
