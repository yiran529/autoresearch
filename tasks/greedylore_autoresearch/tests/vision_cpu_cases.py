"""Exercise actual vision training on CPU, replacing GPU and dataset boundaries."""
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset


def main():
    seed, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(seed))
    spec = importlib.util.spec_from_file_location('vision_runner', seed / 'pytorch-cifar/train_ddp.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import method_policy

    # The committed policy changes only stock GreedyLoRE compression settings.
    args = SimpleNamespace(compressor='top_subspace', compress_rank=None, start_compress_iter=None,
                           update_proj_gap=None, min_compression_rate=None, use_error_feedback=None,
                           beta_ef=None, error_inherit=None, lr=.07, epochs=17)
    configured = method_policy.configure_compression(args, 'resnet18_cifar100_muon')
    assert configured is args
    assert (args.compress_rank, args.start_compress_iter, args.update_proj_gap) == (64, 4000, 1200)
    assert (args.min_compression_rate, args.use_error_feedback, args.beta_ef, args.error_inherit) == (2., 'ef14', 0, 0)
    assert (args.lr, args.epochs) == (.07, 17)
    args.compress_rank, args.start_compress_iter, args.update_proj_gap = 3, 2, 7
    method_policy.configure_compression(args, 'resnet18_cifar100_muon')
    assert (args.compress_rank, args.start_compress_iter, args.update_proj_gap) == (3, 2, 7)
    dense = SimpleNamespace(compressor='none', lr=.03)
    method_policy.configure_compression(dense, 'resnet18_cifar100_muon')
    assert vars(dense) == {'compressor': 'none', 'lr': .03}

    args = runner.parse_args(['--dataset', 'cifar100', '--output-dir', str(output),
                              '--epochs', '2', '--batch-size', '2', '--workers', '0',
                              '--stop-after-steps', '6', '--no-download', '--wandb-mode', 'disabled',
                              '--contract-hash', 'cpu-vision', '--checkpoint-root', str(output / 'checkpoints'),
                              '--checkpoint-interval-steps', '2', '--training-metrics', str(output / 'train.jsonl'),
                              '--no-communication-bit-estimates', '--compressor', 'top_subspace'])
    for bad in ('0', '-2'):
        try:
            runner.parse_args(['--output-dir', str(output), '--checkpoint-interval-steps', bad])
        except SystemExit:
            pass
        else:
            raise AssertionError('nonpositive checkpoint interval accepted')
    try:
        runner.parse_args(['--output-dir', str(output), '--contract-hash', 'formal', '--smoke-steps', '1'])
    except SystemExit:
        pass
    else:
        raise AssertionError('synthetic formal scoring accepted')

    # No network or data download: call the real dataset factory with a local
    # fixture constructor and inspect its actual download argument.
    downloads = []
    def local_cifar(root, *, train, download, transform=None):
        downloads.append(download)
        return TensorDataset(torch.zeros(6, 3), torch.zeros(6, dtype=torch.long))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(3, 3)
            self.linear = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.linear(self.hidden(x))

    dist.init_process_group('gloo', init_method=f'file://{output}/gloo', rank=0, world_size=1)
    try:
        with patch.object(runner.torchvision.datasets, 'CIFAR100', local_cifar):
            runner.make_datasets(args, 0)
        assert downloads == [False, False], downloads

        # This candidate deliberately has no stock iteration/rank/schedule
        # fields. The formal hook must execute it without static estimation.
        class CandidateState:
            pass
        def custom_hook(state, bucket):
            return runner._allreduce_fut(dist.group.WORLD, bucket.buffer())
        original_policy = runner.apply_compression_policy
        def candidate_policy(args, workload):
            args = original_policy(args, workload)
            args.compress_rank, args.start_compress_iter, args.update_proj_gap = 3, 1, 7
            return args
        features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                 [1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
        dataset = TensorDataset(features, torch.tensor([0, 1, 0, 1, 0, 1]))
        with ExitStack() as stack:
            for item in [patch.object(runner, 'initialize_distributed', return_value=(0, 0, 1, torch.device('cpu'))),
                         patch.object(runner, 'build_model', side_effect=lambda _: TinyModel()),
                         patch.object(runner, 'make_datasets', return_value=(dataset, dataset)),
                         patch.object(runner, 'DDP', side_effect=lambda model, **kw: torch.nn.parallel.DistributedDataParallel(model)),
                         patch.object(runner.torch.cuda, 'get_device_name', return_value='cpu-test'),
                         patch.object(runner, 'SubspaceState', side_effect=lambda *a, **kw: CandidateState()),
                         patch.object(runner, 'subspace_hook', custom_hook),
                         patch.object(runner, 'apply_compression_policy', candidate_policy),
                         patch.object(runner.dist, 'destroy_process_group')]:
                stack.enter_context(item)
            runner.main(args)
        result = json.loads((output / 'all_results.json').read_text())
        config = json.loads((output / 'config.json').read_text())
        epochs = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert result['status'] == 'completed'
        assert (result['contract_hash'], result['world_size'], result['epoch'], result['update_step'], result['test_samples']) == ('cpu-vision', 1, 2, 6, 6)
        assert result['final_test_loss'] == epochs[-1]['test_loss']
        assert result['final_test_accuracy'] == epochs[-1]['test_accuracy']
        assert result['resume_policy'] == 'restart_from_scratch'
        assert (config['compress_rank'], config['start_compress_iter'], config['update_proj_gap']) == (3, 1, 7)
        assert config['communication_bit_estimates'] is False
        rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
        assert [row['update_step'] for row in rows] == [1, 2, 3, 4, 5, 6]
        assert all(row['contract_hash'] == 'cpu-vision' and row['world_size'] == 1 for row in rows)
        assert {path.name for path in (output / 'checkpoints').iterdir()} == {'step-00000002', 'step-00000004', 'step-00000006'}
        final = json.loads((output / 'checkpoints/step-00000006/manifest.json').read_text())
        assert (final['status'], final['update_step'], final['contract_hash']) == ('complete', 6, 'cpu-vision')
        assert final['data_stream_replay'] == 'unavailable'
        saved = torch.load(output / 'checkpoints/step-00000006/rank-0.pt', weights_only=True)
        assert saved['batch']['inputs'].shape == (2, 3)
        assert final['config']['checkpoint_policy'].startswith('diagnostic')
    finally:
        dist.destroy_process_group()
    print('vision CPU training, candidate policy, direct hook, metrics and diagnostic checkpoints passed')


if __name__ == '__main__':
    main()
