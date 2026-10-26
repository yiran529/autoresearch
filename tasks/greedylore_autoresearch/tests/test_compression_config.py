"""Workload adaptation cannot accidentally replace the fixed training recipe."""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


def policy_module():
    path = Path(__file__).resolve().parents[1] / 'seed/compression_config.py'
    spec = importlib.util.spec_from_file_location('compression_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_can_set_method_defaults_without_access_to_training_recipe(monkeypatch):
    seen = []
    def configure(args, workload):
        seen.append((vars(args).copy(), workload))
        args.compress_rank, args.start_compress_iter, args.update_proj_gap = 12, 0, 17
        return args
    monkeypatch.setitem(sys.modules, 'method_policy', SimpleNamespace(configure_compression=configure))
    args = Namespace(lr=.02, seed=1243, epochs=40, compressor='top_subspace', compress_rank=64,
                     start_compress_iter=4000, update_proj_gap=1200)
    result = policy_module().apply_compression_policy(args, 'resnet18_cifar100_muon')
    assert (result.compress_rank, result.start_compress_iter, result.update_proj_gap) == (12, 0, 17)
    assert (result.lr, result.seed, result.epochs) == (.02, 1243, 40)
    assert not {'lr', 'seed', 'epochs'} & set(seen[0][0])


def test_policy_rejects_training_field_override_before_applying_any_changes(monkeypatch):
    def configure(args, workload):
        args.compress_rank = 12
        args.lr = 999
        return args
    monkeypatch.setitem(sys.modules, 'method_policy', SimpleNamespace(configure_compression=configure))
    args = Namespace(lr=.02, compressor='top_subspace', compress_rank=64)
    with pytest.raises(ValueError, match='compression fields'):
        policy_module().apply_compression_policy(args, 'resnet18_cifar100_muon')
    assert args.lr == .02 and args.compress_rank == 64


def test_stock_policy_preserves_dense_and_explicit_method_settings(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'seed/method_policy.py'
    spec = importlib.util.spec_from_file_location('method_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'method_policy', module)
    dense = Namespace(compressor='none', compress_rank=12, start_compress_iter=0, update_proj_gap=17)
    result = policy_module().apply_compression_policy(dense, 'resnet18_cifar100_muon')
    assert vars(result) == vars(dense) and result.compressor == 'none'
    candidate = Namespace(compressor='top_subspace', compress_rank=12, start_compress_iter=0, update_proj_gap=17)
    result = policy_module().apply_compression_policy(candidate, 'resnet18_cifar100_muon')
    assert (result.compress_rank, result.start_compress_iter, result.update_proj_gap) == (12, 0, 17)
