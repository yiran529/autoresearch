import argparse
import copy
import importlib.util
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


MUON_SPEC = importlib.util.find_spec("optimizer.muon")
MUON_UTILS_SPEC = importlib.util.find_spec("optimizer.muon_utils")


def _distributed_muon_worker(rank, init_file):
    from optimizer.muon import Muon

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        params = [nn.Parameter(torch.ones(2, 2)) for _ in range(3)]

        def rank_tagged_orthogonalization(updates, epsilon):
            if tuple(updates.shape) != (2, 2, 2):
                raise AssertionError(f"unexpected local batch shape: {tuple(updates.shape)}")
            return torch.full_like(updates, rank + 1)

        optimizer = Muon(
            params,
            lr=0.1,
            mu=0.0,
            weight_decay=0.0,
            adjust_lr=None,
            orthogonalize=rank_tagged_orthogonalization,
            distributed_orthogonalization=True,
        )
        for param in params:
            param.grad = torch.ones_like(param)
        optimizer.step()

        for param, expected in zip(params, (0.9, 0.9, 0.8)):
            if not torch.allclose(param, torch.full_like(param, expected)):
                raise AssertionError(f"rank {rank} received the wrong gathered update")
        expected_bits = {"gradient_presence": 48, "orthogonalization_results": 512}
        if optimizer.communication_bits_stats()["this_step"] != expected_bits:
            raise AssertionError("unexpected Muon collective accounting")
    finally:
        dist.destroy_process_group()


class TestMuonAvailability(unittest.TestCase):
    def test_muon_modules_are_available(self):
        self.assertIsNotNone(MUON_SPEC)
        self.assertIsNotNone(MUON_UTILS_SPEC)


@unittest.skipIf(MUON_SPEC is None or MUON_UTILS_SPEC is None, "Muon is not implemented yet")
class TestMuon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from optimizer.muon import Muon, polar_express
        from optimizer.muon_utils import add_muon_args, build_muon_optimizer

        cls.Muon = Muon
        cls.polar_express = staticmethod(polar_express)
        cls.add_muon_args = staticmethod(add_muon_args)
        cls.build_muon_optimizer = staticmethod(build_muon_optimizer)

    @staticmethod
    def identity_orthogonalization(update, epsilon):
        return update

    def test_nesterov_momentum_and_decoupled_weight_decay(self):
        weight = nn.Parameter(torch.ones(2, 2))
        optimizer = self.Muon(
            [weight],
            lr=0.1,
            mu=0.5,
            weight_decay=0.1,
            nesterov=True,
            adjust_lr=None,
            orthogonalize=self.identity_orthogonalization,
        )

        weight.grad = torch.full_like(weight, 0.5)
        optimizer.step()
        self.assertTrue(torch.allclose(weight, torch.full_like(weight, 0.915)))
        self.assertTrue(
            torch.allclose(optimizer.state[weight]["momentum"], torch.full_like(weight, 0.5))
        )

    def test_convolution_is_flattened_and_spectrally_scaled(self):
        weight = nn.Parameter(torch.ones(2, 1, 2, 2))
        seen = []

        def orthogonalize(update, epsilon):
            seen.append(tuple(update.shape))
            return torch.ones_like(update)

        optimizer = self.Muon([weight], lr=0.1, weight_decay=0, orthogonalize=orthogonalize)
        weight.grad = torch.ones_like(weight)
        optimizer.step()

        self.assertEqual(seen, [(1, 2, 4)])
        self.assertTrue(
            torch.allclose(weight, torch.full_like(weight, 1 - 0.1 * (2 / 4) ** 0.5))
        )

    def test_adamw_fallback_updates_vector_parameter(self):
        bias = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = self.Muon(
            [{"params": [bias], "algorithm": "adamw", "lr": 0.1, "weight_decay": 0.1}],
            betas=(0.9, 0.999),
            epsilon=1e-8,
        )
        bias.grad = torch.tensor([0.5, -0.25])
        optimizer.step()

        self.assertTrue(torch.allclose(bias, torch.tensor([0.89, 2.08]), atol=1e-6))
        self.assertEqual(optimizer.state[bias]["step"], 1)

    def test_state_dict_resume_matches_uninterrupted_update(self):
        weight = nn.Parameter(torch.ones(2, 2))
        optimizer = self.Muon(
            [weight], lr=0.1, mu=0.5, orthogonalize=self.identity_orthogonalization
        )
        weight.grad = torch.full_like(weight, 0.5)
        optimizer.step()

        restored_weight = nn.Parameter(weight.detach().clone())
        restored = self.Muon(
            [restored_weight], lr=0.1, mu=0.5, orthogonalize=self.identity_orthogonalization
        )
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        weight.grad = torch.full_like(weight, 0.25)
        restored_weight.grad = torch.full_like(restored_weight, 0.25)
        optimizer.step()
        restored.step()

        self.assertTrue(torch.equal(weight, restored_weight))
        self.assertTrue(
            torch.equal(
                optimizer.state[weight]["momentum"],
                restored.state[restored_weight]["momentum"],
            )
        )

    def test_builder_assigns_every_parameter_once(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(5, 3)
                self.proj = nn.Linear(3, 3)
                self.norm = nn.LayerNorm(3)
                self.lm_head = nn.Linear(3, 5, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        model = TinyModel()
        optimizer = self.build_muon_optimizer(model, lr=0.1, scalar_lr=0.01)
        matrix_ids = {
            id(param)
            for group in optimizer.param_groups
            if group["algorithm"] == "muon"
            for param in group["params"]
        }
        all_params = [param for group in optimizer.param_groups for param in group["params"]]
        trainable = [param for param in model.parameters() if param.requires_grad]

        self.assertEqual(matrix_ids, {id(model.proj.weight)})
        self.assertEqual(len(all_params), len(trainable))
        self.assertEqual({id(param) for param in all_params}, {id(param) for param in trainable})
        self.assertTrue(
            all(group["lr"] == 0.01 for group in optimizer.param_groups if group["algorithm"] == "adamw")
        )

    def test_cli_defaults_and_scheduler_support(self):
        parser = argparse.ArgumentParser()
        self.add_muon_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.muon_mu, 0.95)
        self.assertIsNone(args.muon_scalar_lr)
        self.assertEqual(args.muon_adjust_lr, "spectral_norm")

        model = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
        optimizer = self.build_muon_optimizer(model, lr=0.1, scalar_lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 0.5)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        self.assertTrue(
            all(abs(group["lr"] - 0.005) < 1e-12 for group in optimizer.param_groups[1:])
        )

    def test_polar_express_returns_finite_matrix(self):
        result = self.polar_express(torch.eye(2))
        self.assertEqual(tuple(result.shape), (2, 2))
        self.assertTrue(torch.isfinite(result).all())

    def test_distributed_orthogonalization_gathers_rank_owned_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_muon_worker,
                args=(os.path.join(directory, "init"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
