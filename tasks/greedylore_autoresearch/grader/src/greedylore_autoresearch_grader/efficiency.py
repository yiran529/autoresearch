"""Build the fixed two-cell, two-repetition timing matrix."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import EfficiencyPlan
from .process import ProcessSpec

_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
    "NCCL_DEBUG", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
)

def timing_cell_labels(names):
    """Accept one complete timing model, never a mixed-size matrix."""
    for size in ('130m', '350m'):
        labels = {f'{size}_adamw_{precision}': precision for precision in ('bf16', 'fp32')}
        if set(names) == set(labels):
            return labels
    raise ValueError('timing requires two precision cells of one supported model')


@dataclass(frozen=True)
class TimingRun:
    repetition: int
    cell_name: str
    spec: ProcessSpec

def build_timing_command(
    cell: Mapping[str, Any], repetition: int, candidate: str | Path,
    output: str | Path, gpus: Sequence[int], hardware: Mapping[str, Any] | None = None,
) -> ProcessSpec:
    candidate = Path(candidate).resolve()
    output = Path(output).resolve()
    compressor = cell["compressor"]
    command = [
        "uv", "run", "--frozen", "--project", str(candidate),
        "python", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(cell["gpu_count"]), "c4/table_v_timing.py",
        "--model_config", {"llama130m": "c4/configs/llama_130m.json",
                           "llama350m": "c4/configs/llama_350m.json"}[cell["model"]],
        "--dataset_path", str(cell["c4_path"]),
        "--tokenizer_path", str(cell["tokenizer_path"]),
        "--batch_size", "1", "--max_length", str(cell["max_length"]),
        "--dtype", "bfloat16" if cell["precision"] == "bf16" else "float32",
        "--warmup_iterations", str(cell["warmup_iterations"]),
        "--measured_iterations", str(cell["measured_iterations"]),
        "--scheduler_steps", "20000", "--lr_warmup_steps", "2000",
        "--ddp_bucket_cap_mb", str(cell["bucket_cap_mb"]),
        "--record_bucket_layout", "--workers", str(cell["workers"]),
        "--seed", str(cell["seed"]), "--optimizer", "adamw",
        "--lr", str(cell["learning_rate"]), "--beta1", "0.9",
        "--beta2", "0.999", "--eps", "1e-8", "--weight_decay", "0",
        "--grad_clipping", "1", "--compressor", str(compressor["name"]),
        "--compress_rank", str(compressor["rank"]),
        "--start_compress_iter", str(cell["compression_start_step"]),
        "--update_proj_gap", str(compressor["update_gap"]),
        "--use_error_feedback", f"ef{compressor['ef_rank']}",
        "--min_compression_rate", str(compressor["min_compression_rate"]),
        "--beta_ef", str(compressor["beta_ef"]),
        "--error_inherit", format(compressor["error_inherit"], "g"),
        "--contract_hash", str(cell["contract_hash"]),
        "--repetition", str(repetition), "--output_dir", str(output),
    ]
    if hardware:
        command += [
            "--gpu_uuids", ",".join(hardware["gpu_uuids"]),
            "--topology_fingerprint", hardware["topology_fingerprint"],
            "--numa_nodes", ",".join("none" if value is None else str(value) for value in hardware["numa_nodes"]),
        ]
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0",
        "NCCL_DEBUG": "INFO", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4", "PYTHONPATH": str(candidate),
        "CORAL_WORKLOAD": str(cell['name']),
    }
    return ProcessSpec(tuple(command), output.parent / f"{output.name}.process", env,
                       _ENV_ALLOWLIST, 2 * 60 * 60, cwd=candidate)

def iter_timing_runs(
    plan: EfficiencyPlan, candidate: str | Path, output: str | Path,
    gpus: Sequence[int],
) -> Iterator[TimingRun]:
    root = Path(output)
    for repetition, row in enumerate(plan.run_order):
        for name in row:
            run_output = root / f"r{repetition + 1}-{name}"
            yield TimingRun(
                repetition, name,
                build_timing_command(plan.cells[name], repetition, candidate, run_output, gpus),
            )
