"""Small real two-rank ARC checks; no model/data downloads or GPUs."""

# ruff: noqa: S101, E402, N817
import argparse
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SEED, ROOT = map(Path, sys.argv[1:3])
sys.path.insert(0, str(SEED))
from c4.table_v_timing import register_timing_comm_hook
from comm_hooks.utils import add_comm_hook_args, register_comm_hook_for_ddp_model
from diagnostic_checkpoint import load_checkpoint, save_final_checkpoint


def arguments(method="group_topk", ef="ef14"):
    parser = argparse.ArgumentParser()
    add_comm_hook_args(parser)
    args = parser.parse_args(
        [
            "--compressor",
            method,
            "--r",
            "4",
            "--compress_ratio",
            ".2",
            "--start_compress_iter",
            "0",
            "--use_error_feedback",
            ef,
        ]
    )
    args.seed, args.optimizer = 1243, "adamw"
    args.strict_blocking_communication, args.record_bucket_layout = False, True
    return args


def worker(rank):
    dist.init_process_group("gloo", init_method=f"file://{ROOT / 'init'}", rank=rank, world_size=2)
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Compare training and passive formal timing on the same real DDP bucket.
    outputs = []
    for timing in (False, True):
        model = DDP(torch.nn.Linear(5, 5, bias=False))
        args = arguments()
        if timing:
            wrapper = register_timing_comm_hook(model, dist.group.WORLD, args)
            state = wrapper.inner_state
            assert state.timing_period_steps is None
        else:
            args.checkpoint_root = str(ROOT / "checkpoint")
            state = register_comm_hook_for_ddp_model(model, dist.group.WORLD, args)
        prior = 0
        for step in range(2):
            model.zero_grad()
            if timing:
                wrapper.begin_step(True)
            source = torch.arange(1.0, 6.0).unsqueeze(0) * (rank + 1)
            before = torch.get_rng_state().clone()
            model(source).sum().backward()
            assert torch.equal(before, torch.get_rng_state()), "ARC must not perturb training RNG"
            gradient = model.module.weight.grad.clone()
            local = torch.arange(1.0, 6.0).repeat(5, 1) * (rank + 1)
            residual = state.error_dict[0]
            expected = local.flatten() + prior
            compensated = gradient.flatten() + residual
            dist.all_reduce(compensated)
            compensated /= 2
            dist.all_reduce(expected)
            expected /= 2
            assert torch.allclose(compensated, expected), "EF14 conservation across ranks"
            prior = residual.clone()
            copies = [torch.empty_like(gradient) for _ in range(2)]
            dist.all_gather(copies, gradient)
            assert torch.equal(copies[0], copies[1]), "all ranks must use common row support"
            if timing:
                assert wrapper.end_step()["bucket_count"] == 1
        outputs.append(gradient)
        assert state.iter == 2 and not state.timing_refresh_steps if timing else state.iter == 2
        if not timing:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            result = save_final_checkpoint(
                ROOT / "checkpoint",
                model,
                optimizer,
                scheduler,
                state,
                config={"contract_hash": "cpu"},
                update_step=2,
                min_free_bytes=0,
            )
            assert result["status"] == "complete", result
            saved_rng = state.rng.get_state().clone()
            state.error_dict.clear()
            state.iter = 99
            load_checkpoint(
                ROOT / "checkpoint",
                model,
                optimizer,
                scheduler,
                state,
                contract_hash="cpu",
                expected_config={"contract_hash": "cpu"},
            )
            assert state.iter == 2 and torch.equal(state.rng.get_state(), saved_rng)
            assert torch.equal(state.error_dict[0], prior)
            # Existing bucket guard is exercised by the next real backward.
            model.zero_grad()
            model(source).sum().backward()
            state._checkpoint_expected_layout = {0: [("wrong", (5, 5))]}
            try:
                model.zero_grad()
                model(source).sum().backward()
            except (ValueError, RuntimeError) as exc:
                assert "bucket layout mismatch" in str(exc), exc
            else:
                raise AssertionError("restored ARC state must reject a changed bucket identity")
    assert torch.equal(outputs[0], outputs[1])
    # Dense and ARC share the same constructor and both really aggregate.
    model = DDP(torch.nn.Linear(3, 2, bias=False))
    register_comm_hook_for_ddp_model(model, dist.group.WORLD, arguments("none"))
    model(torch.ones(1, 3) * (rank + 1)).sum().backward()
    assert torch.equal(model.module.weight.grad, torch.full((2, 3), 1.5))
    for ef in ("noef", "ef21"):
        model = DDP(torch.nn.Linear(5, 5, bias=True))
        args = arguments(ef=ef)
        args.start_compress_iter = 1
        args.checkpoint_root = str(ROOT / "unused-ef-layout")
        state = register_comm_hook_for_ddp_model(model, dist.group.WORLD, args)
        for step in range(3):
            model.zero_grad()
            model(torch.arange(1.0, 6.0).unsqueeze(0) * (rank + step + 1)).sum().backward()
            gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
            replicas = [torch.empty_like(gradient) for _ in range(2)]
            dist.all_gather(replicas, gradient)
            assert torch.equal(replicas[0], replicas[1]), f"{ef} aggregation differs across ranks"
            if step == 0:
                assert torch.allclose(
                    model.module.weight.grad, torch.arange(1.0, 6.0).repeat(5, 1) * 1.5
                )
            if ef == "ef21" and step >= 1:
                estimate = state.error_dict[0].clone()
                dist.all_reduce(estimate)
                estimate /= 2
                assert torch.allclose(estimate, state.global_error_dict[0])
                ordered = torch.cat(
                    [
                        dict(model.named_parameters())[name].grad.flatten()
                        for name, shape in state._checkpoint_bucket_layout[0]
                    ]
                )
                assert torch.allclose(state.global_error_dict[0], ordered)
        assert state.iter == 3
        if ef == "noef":
            assert not state.error_dict
    # Real vision registration must keep the actual compressor for diagnostics.
    spec = importlib.util.spec_from_file_location("vision_cpu", SEED / "pytorch-cifar/train_ddp.py")
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)

    class ResNetToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.linear(self.conv(x).mean((2, 3)))

    model = DDP(ResNetToy())
    args = vision.parse_args(
        [
            "--output-dir",
            str(ROOT / "vision"),
            "--compressor",
            "group_topk",
            "--r",
            "4",
            "--compress-ratio",
            ".2",
            "--start-compress-iter",
            "0",
            "--no-communication-bit-estimates",
            "--checkpoint-root",
            str(ROOT / "vision-checkpoint"),
        ]
    )
    hook = vision.attach_hook(model, args)
    assert hook.inner_state.__class__.__name__ == "GroupTopKState"
    model(torch.ones(1, 3, 5, 5) * (rank + 1)).sum().backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    result = save_final_checkpoint(
        ROOT / "vision-checkpoint",
        model,
        optimizer,
        scheduler,
        hook.inner_state,
        config={"contract_hash": "vision-cpu"},
        update_step=1,
        min_free_bytes=0,
    )
    assert result["status"] == "complete", result
    restored = load_checkpoint(
        ROOT / "vision-checkpoint",
        model,
        optimizer,
        scheduler,
        hook.inner_state,
        contract_hash="vision-cpu",
    )
    assert restored["hook_state"]["class"].endswith(".GroupTopKState")
    assert hook.inner_state.iter == 1 and hook.inner_state.error_dict

    # Exercise main(), not only parsing and hook registration: its stale
    # compressor guard previously rejected ARC before distributed setup.
    args = vision.parse_args([
        "--dataset", "cifar100", "--output-dir", str(ROOT / "vision-main"),
        "--compressor", "group_topk", "--start-compress-iter", "1",
        "--epochs", "2", "--batch-size", "2", "--workers", "0",
        "--stop-after-steps", "6", "--no-download", "--wandb-mode", "disabled",
        "--no-communication-bit-estimates", "--contract-hash", "arc-cpu-main",
        "--checkpoint-root", str(ROOT / "vision-main-checkpoints"),
        "--checkpoint-interval-steps", "2",
        "--training-metrics", str(ROOT / "vision-main-train.jsonl"),
    ])
    generator = torch.Generator().manual_seed(1243)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(12, 3, 5, 5, generator=generator), torch.arange(12) % 2,
    )
    with ExitStack() as stack:
        for boundary in [
            patch.object(vision, "initialize_distributed",
                         return_value=(rank, rank, 2, torch.device("cpu"))),
            patch.object(vision, "build_model", side_effect=lambda _: ResNetToy()),
            patch.object(vision, "make_datasets", return_value=(dataset, dataset)),
            patch.object(vision, "DDP", side_effect=lambda model, **kw: DDP(model)),
            patch.object(vision.torch.cuda, "get_device_name", return_value="cpu-test"),
            patch.object(vision.dist, "destroy_process_group"),
        ]:
            stack.enter_context(boundary)
        vision.main(args)
    dist.barrier()
    result = json.loads((ROOT / "vision-main/all_results.json").read_text())
    assert (result["status"], result["compressor"], result["update_step"]) == (
        "completed", "group_topk", 6,
    )
    assert result["diagnostic_checkpoint"]["status"] == "complete"
    assert result["diagnostic_checkpoint"]["world_size"] == 2
    saved = torch.load(ROOT / f"vision-main-checkpoints/step-00000006/rank-{rank}.pt",
                       weights_only=True)
    assert saved["hook"]["class"].endswith(".GroupTopKState")
    assert saved["hook"]["state"]["iter"] == 6
    assert saved["hook"]["state"]["error_dict"]
    config = json.loads((ROOT / "vision-main/config.json").read_text())
    assert config["uncompressed_names"] == []
    rows = [json.loads(line) for line in (ROOT / "vision-main-train.jsonl").read_text().splitlines()]
    assert [row["update_step"] for row in rows] == [1, 2, 3, 4, 5, 6]
    dist.destroy_process_group()


def cli_checks():
    from c4 import table_v_timing
    from c4.run_llama_pretraining import parse_args

    spec = importlib.util.spec_from_file_location(
        "vision_runner", SEED / "pytorch-cifar/train_ddp.py"
    )
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    commands = json.loads((ROOT / "commands.json").read_text())
    for command, entry in zip(
        commands,
        (
            "c4/run_llama_pretraining.py",
            "pytorch-cifar/train_ddp.py",
            "c4/table_v_timing.py",
            "pytorch-cifar/train_ddp.py",
        ),
    ):
        argv = command[command.index(entry) + 1 :]
        if entry.startswith("pytorch"):
            args = vision.parse_args(argv)
        elif "table_v" in entry:
            sys.argv = [entry, *argv]
            args = table_v_timing.parse_args()
        else:
            args = parse_args(argv)
        if args.compressor == "top_subspace":
            assert (args.compress_rank, args.start_compress_iter, args.update_proj_gap) == (
                64,
                4000,
                1200,
            )
        else:
            assert (args.compressor, args.r, args.compress_ratio, args.use_error_feedback) == (
                "group_topk",
                4,
                0.2,
                "ef14",
            )


if __name__ == "__main__":
    original_argv = sys.argv[:]
    cli_checks()
    sys.argv = original_argv
    mp.spawn(worker, nprocs=2, join=True)
    print("ARC CPU/Gloo checks passed")
