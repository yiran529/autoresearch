"""Paper-aligned CIFAR comparison: dense Muon versus GreedyLore + Muon."""
import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.subspace_hook import SubspaceState, _should_compress_subspace, subspace_hook
from diagnostic_checkpoint import save_final_checkpoint
from compression_config import apply_compression_policy
from models.resnet import BasicBlock, ResNet
from optimizer import add_muon_args, build_muon_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset
from torchvision import transforms


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
    parser.add_argument('--contract-hash', default='')
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--evaluation-split', choices=['test', 'validation'], default='test')
    parser.add_argument('--validation-per-class', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=1243)
    parser.add_argument('--no-communication-bit-estimates', dest='communication_bit_estimates',
                        action='store_false', default=True)
    parser.add_argument('--checkpoint-root', help='External diagnostic snapshot root; retries restart from scratch')
    parser.add_argument('--checkpoint-interval-steps', type=int, default=1000)
    parser.add_argument('--training-metrics')
    parser.add_argument('--use-error-feedback', default='ef14')
    parser.add_argument('--beta-ef', type=float, default=0.)
    parser.add_argument('--error-inherit', type=float, default=0.)
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
    if args.checkpoint_interval_steps <= 0:
        parser.error('checkpoint interval must be positive')
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        parser.error('stop-after-steps must be positive')
    if args.contract_hash and (args.smoke_steps or args.dry_run):
        parser.error('formal results cannot use synthetic smoke data or dry-run mode')
    if args.start_compress_iter < 0 or args.update_proj_gap < 1 or args.epochs < 1:
        parser.error('Invalid compression schedule or epoch budget')
    if (not 0 <= args.validation_per_class < 500 or
            args.evaluation_split == 'validation' and (args.dataset != 'cifar100' or not args.validation_per_class)):
        parser.error('validation requires CIFAR100 and a nonempty per-class training split')
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
            min_compression_rate=args.min_compression_rate, use_error_feedback=args.use_error_feedback,
            uncompressed_names=['linear'], random_seed=args.seed,
            beta_ef=args.beta_ef, error_inherit=args.error_inherit,
        )
        subspace.param_to_name = {p: name for name, p in model.named_parameters()}
    state = CommunicationState(subspace)
    if args.communication_bit_estimates:
        model.register_comm_hook(state, counted_hook)
    elif subspace is not None:
        # Candidate states need not expose stock GreedyLoRE estimator fields.
        model.register_comm_hook(subspace, subspace_hook)
    else:
        model.register_comm_hook(state, counted_hook)
    return state


def stratified_split_indices(targets, per_class, seed):
    groups = {}
    for index, target in enumerate(targets):
        groups.setdefault(int(target), []).append(index)
    rng = random.Random(seed)
    train, validation = [], []
    for label in sorted(groups):
        indices = groups[label].copy()
        if not 0 < per_class < len(indices):
            raise ValueError('each class requires nonempty training and validation sets')
        rng.shuffle(indices)
        validation.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return sorted(train), sorted(validation)


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
    if rank == 0 and not args.no_download:
        dataset_type(args.data_dir, train=True, download=True)
        if args.evaluation_split == 'test':
            dataset_type(args.data_dir, train=False, download=True)
    dist.barrier()
    trainset = dataset_type(args.data_dir, train=True, download=False, transform=train_transform)
    if args.validation_per_class:
        if len(trainset) != 50000 or set(trainset.targets) != set(range(100)):
            raise ValueError('validation split requires the complete official CIFAR100 training set')
        train_indices, validation_indices = stratified_split_indices(
            trainset.targets, args.validation_per_class, args.split_seed)
        args.split_sha256 = hashlib.sha256(json.dumps(dict(train=train_indices,
            validation=validation_indices, seed=args.split_seed), sort_keys=True).encode()).hexdigest()
        args.partition_train_samples = len(train_indices)
        trainset = Subset(trainset, train_indices)
        if args.evaluation_split == 'validation':
            validation_source = dataset_type(args.data_dir, train=True, download=False, transform=test_transform)
            return trainset, Subset(validation_source, validation_indices)
    return trainset, dataset_type(args.data_dir, train=False, download=False, transform=test_transform)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def initialize_distributed():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device('cuda', local_rank)


def main(args=None):
    args = parse_args() if args is None else args
    workload = os.environ.get('CORAL_WORKLOAD', '')
    args = apply_compression_policy(args, workload)
    if args.compressor not in {'none', 'top_subspace'}:
        raise ValueError('unsupported vision compressor')
    if args.start_compress_iter < 0 or args.update_proj_gap < 1 or args.compress_rank < 1:
        raise ValueError('invalid vision compression parameters')
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    local_rank, rank, world_size, device = initialize_distributed()
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
    planned_updates = len(trainloader) * args.epochs
    if args.stop_after_steps is not None and args.stop_after_steps != planned_updates:
        raise ValueError(f'epoch/update budget mismatch: {planned_updates} != {args.stop_after_steps}')
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
                  scheduler='cosine to zero, epoch steps, no LR warmup',
                  uncompressed_names=['linear'],
                  workload=workload, resume_policy='restart_from_scratch',
                  checkpoint_policy=('diagnostic periodic and final snapshots; no exact vision resume'
                                     if args.checkpoint_root else 'none; retries restart from scratch'))
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
    last_batch = None
    last_checkpoint_step = None
    checkpoint = None
    evaluation_split = args.evaluation_split

    def save_diagnostic():
        return save_final_checkpoint(
            Path(args.checkpoint_root) / f'step-{update_step:08d}',
            model, optimizer, scheduler, hook.subspace, config=config,
            update_step=update_step, batch=last_batch,
        )
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
            last_batch = {'inputs': inputs.detach(), 'targets': targets.detach()}
            if rank == 0 and args.training_metrics:
                metrics_path = Path(args.training_metrics)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open('a') as stream:
                    stream.write(json.dumps(dict(update_step=update_step, epoch=epoch+1,
                        loss=loss.item(), learning_rates=lrs, contract_hash=args.contract_hash,
                        world_size=world_size)) + '\n')
            # Epoch-boundary saves happen after evaluation and the scheduler step.
            if (args.checkpoint_root and update_step % args.checkpoint_interval_steps == 0
                    and batch_idx + 1 < len(trainloader)):
                checkpoint = save_diagnostic()
                last_checkpoint_step = update_step
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
                   train_accuracy=(100*stats[1]/stats[2]).item(),
                   **{f'{evaluation_split}_loss': test_loss, f'{evaluation_split}_accuracy': accuracy,
                      f'{evaluation_split}_samples': int(evaluation[2].item()),
                      f'best_{evaluation_split}_accuracy': best_accuracy},
                   best_epoch=best_epoch, learning_rates=lrs,
                   ddp_dense_reference_bits=hook.dense_bits if args.communication_bit_estimates else None,
                   ddp_payload_bits=hook.payload_bits if args.communication_bit_estimates else None,
                   muon_collective_bits=optimizer.communication_bits_stats()['total'],
                   elapsed_seconds=time.monotonic()-started)
        if rank == 0:
            with (output / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            print('EPOCH ' + json.dumps(row), flush=True)
            if tracker is not None:
                tracker.log(row, step=update_step)
        scheduler.step()
        if args.checkpoint_root and update_step % args.checkpoint_interval_steps == 0:
            checkpoint = save_diagnostic()
            last_checkpoint_step = update_step
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
    if args.stop_after_steps is not None and update_step != args.stop_after_steps:
        raise RuntimeError('vision training did not complete the fixed update budget')
    if args.checkpoint_root and last_checkpoint_step != update_step:
        checkpoint = save_diagnostic()
    if rank == 0:
        result = dict(row, contract_hash=args.contract_hash, world_size=world_size,
                      compressor=args.compressor,
                      training_recipe=dict(model='resnet18', dataset=args.dataset, epochs=args.epochs,
                          per_device_batch_size=args.batch_size, workers=args.workers, seed=args.seed,
                          precision='fp32', optimizer='muon', matrix_learning_rate=args.lr,
                          scalar_learning_rate=args.muon_scalar_lr, weight_decay=args.weight_decay,
                          matrix_momentum=args.muon_mu, nesterov=True,
                          adjust_lr=args.muon_adjust_lr, scalar_betas=[args.muon_scalar_beta1, args.muon_scalar_beta2],
                          scalar_eps=args.muon_scalar_eps, schedule='epoch_cosine', lr_warmup_steps=0,
                          augmentation='cifar_random_crop4_flip', bias_bn_weight_decay=0.),
                      resume_policy='restart_from_scratch', diagnostic_checkpoint=checkpoint,
                      status='smoke_completed' if args.smoke_steps else 'completed',
                      **{f'final_{evaluation_split}_accuracy': accuracy, f'final_{evaluation_split}_loss': test_loss},
                      ddp_payload_scope='estimated per-rank collective inputs including warmup, SVD refresh and score probes; excludes network replication and DDP buffers',
                      muon_payload_scope='existing optimizer AllGather estimate: world_size*(world_size-1)*input_bits; different scope from per-rank DDP inputs, do not sum directly')
        if args.validation_per_class:
            result['training_recipe'].update(evaluation_split=evaluation_split,
                validation_per_class=args.validation_per_class, split_seed=args.split_seed,
                split_algorithm='stratified-per-class-random-v1')
            result.update(evaluation_split=evaluation_split, train_samples=args.partition_train_samples,
                          split_sha256=args.split_sha256)
        (output / 'all_results.json').write_text(json.dumps(result, indent=2) + '\n')
        if tracker is not None:
            tracker.summary.update(result)
            tracker.finish()
        print('TRAINING_COMPLETED', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
