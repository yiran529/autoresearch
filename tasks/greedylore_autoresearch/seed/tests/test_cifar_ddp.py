import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist


def load_entrypoint():
    path = Path(__file__).resolve().parents[1] / 'pytorch-cifar' / 'train_ddp.py'
    spec = importlib.util.spec_from_file_location('cifar_ddp', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CifarDDPTests(unittest.TestCase):
    def test_paper_protocol_for_both_datasets(self):
        entry = load_entrypoint()
        for dataset, start, gap, scalar_lr in [('cifar10', 500, 750, .005),
                                               ('cifar100', 4000, 1200, .0005)]:
            args = entry.parse_args(['--dataset', dataset, '--output-dir', '/tmp/dry'])
            self.assertEqual((args.epochs, args.batch_size, args.compress_rank), (40, 32, 64))
            self.assertEqual((args.start_compress_iter, args.update_proj_gap), (start, gap))
            self.assertEqual(args.muon_scalar_lr, scalar_lr)
            self.assertEqual(args.lr, .02)
            model = entry.build_model(dataset)
            self.assertEqual(model.linear.out_features, 10 if dataset == 'cifar10' else 100)

    def test_evaluation_shards_cover_dataset_without_duplicates(self):
        entry = load_entrypoint()
        shards = [list(entry.evaluation_indices(10003, rank, 4)) for rank in range(4)]
        flat = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flat), list(range(10003)))

    def test_conv_hook_compresses_and_keeps_scalar_and_head_dense(self):
        entry = load_entrypoint()
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('gloo', init_method=f'file://{directory}/init', rank=0, world_size=1)
            try:
                model = torch.nn.Module()
                model.conv = torch.nn.Conv2d(8, 8, 3, bias=False)
                model.bn = torch.nn.BatchNorm2d(8)
                model.linear = torch.nn.Linear(8, 3)
                # Exercise the real hook with 4D, 1D and head gradients.
                class Network(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layers = model
                    def forward(self, x):
                        return self.layers.linear(self.layers.bn(self.layers.conv(x)).mean((2, 3)))
                ddp = torch.nn.parallel.DistributedDataParallel(Network())
                args = entry.parse_args(['--output-dir', directory, '--compressor', 'top_subspace',
                                         '--compress-rank', '1', '--start-compress-iter', '0',
                                         '--update-proj-gap', '2'])
                state = entry.attach_hook(ddp, args)
                # The existing hook synchronizes CUDA when available; this
                # test deliberately runs CPU/gloo, as in test_subspace_table_iv.
                with patch('comm_hooks.subspace_hook.torch.cuda.is_available', return_value=False):
                    for _ in range(3):
                        ddp.zero_grad()
                        ddp(torch.randn(2, 8, 8, 8)).sum().backward()
                self.assertEqual(state.subspace.iter, 3)
                self.assertLess(state.payload_bits, state.dense_bits)
                self.assertEqual(state.subspace.uncompressed_names, ['linear'])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in ddp.parameters()))
            finally:
                dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
