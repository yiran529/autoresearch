# ruff: noqa: S101
"""Exercise real hook refresh telemetry on one CPU Gloo rank, never a GPU."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, sys.argv[1])
from c4.table_v_timing import BucketRecordingHookState, bucket_recording_hook, build_timing_result
from comm_hooks.subspace_hook import SubspaceState, subspace_hook

output = Path(sys.argv[2])
dist.init_process_group("gloo", init_method=f"file://{output / 'rendezvous'}", rank=0, world_size=1)
try:
    parameter = torch.nn.Parameter(torch.zeros(8, 8))
    state = SubspaceState(dist.group.WORLD, matrix_approximation_rank=2,
                          update_proj_gap=4, start_compress_iter=2, uncompressed_names=[])
    state.param_to_name = {parameter: "weight"}
    wrapper = BucketRecordingHookState(state, subspace_hook)

    class Bucket:
        def __init__(self):
            self.tensor = torch.randn(64)

        def buffer(self):
            return self.tensor

        def gradients(self):
            return [self.tensor.view(8, 8)]

        def parameters(self):
            return [parameter]

        def index(self):
            return 0

        def is_last(self):
            return True

    for step in range(11):
        wrapper.begin_step(step >= 3)
        result = bucket_recording_hook(wrapper, Bucket()).wait()
        assert torch.isfinite(result).all()
        wrapper.end_step()
    events = [step for step in state.timing_refresh_steps if 3 < step <= 11]
    args = SimpleNamespace(measured_iterations=8, warmup_iterations=3, optimizer="adamw",
                           compressor="top_subspace", dtype="float32", contract_hash="cpu-check",
                           repetition=0, strict_blocking_communication=False,
                           gpu_uuids="", topology_fingerprint="cpu", numa_nodes="0")
    payload = build_timing_result([{
        "elapsed": 1.0, "steps": [0.125] * 8, "losses": [1.0] * 8,
        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
        "timing_period_steps": state.update_proj_gap, "timing_event_steps": events,
    }], args, 64)
    (output / "telemetry.json").write_text(json.dumps(payload))
finally:
    dist.destroy_process_group()
