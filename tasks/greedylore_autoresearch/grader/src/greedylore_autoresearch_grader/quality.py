"""Build the two fixed formal quality commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .process import ProcessSpec

_MODEL_CONFIG = {
    "llama60m": "c4/configs/llama_60m_table_iv.json",
    "llama130m": "c4/configs/llama_130m.json",
}
_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES", "WANDB_MODE", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
)

def _value(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)

def build_quality_command(
    profile: Mapping[str, Any], candidate: str | Path, output: str | Path,
    gpus: Sequence[int],
) -> ProcessSpec:
    candidate = Path(candidate).resolve()
    output = Path(output).resolve()
    compressor = profile["compressor"]
    count = int(profile["gpu_count"])
    command = [
        "uv", "run", "--frozen", "--project", str(candidate),
        "python", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(count), "c4/run_llama_pretraining.py",
        "--model_config", _MODEL_CONFIG[str(profile["model"])],
        "--dataset_path", str(profile["c4_path"]),
        "--tokenizer_path", str(profile["tokenizer_path"]),
        "--batch_size", str(profile["per_device_batch_size"]),
        "--total_batch_size", str(int(profile["per_device_batch_size"]) * count),
        "--gradient_accumulation", "1", "--max_length", str(profile["max_length"]),
        "--dtype", "bfloat16" if profile["precision"] == "bf16" else "float32",
        "--num_training_steps", str(profile["scheduler_steps"]),
        "--stop_after_steps", str(profile["stop_after_steps"]),
        "--warmup_steps", str(1000 if profile["model"] == "llama60m" else 2000),
        "--eval_every", str(profile["scheduler_steps"]),
        "--save_every", str(int(profile["scheduler_steps"]) + 1),
        "--workers", str(profile["workers"]), "--seed", str(profile["seed"]),
        "--optimizer", str(profile["optimizer"]),
    ]
    if profile["optimizer"] == "muon":
        command += ["--lr", _value(profile["matrix_learning_rate"]),
                    "--muon_scalar_lr", _value(profile["scalar_learning_rate"]),
                    "--muon_mu", "0.95", "--muon_adjust_lr", "spectral_norm"]
    else:
        command += ["--lr", _value(profile["learning_rate"]),
                    "--beta1", "0.9", "--beta2", "0.999", "--eps", "1e-8"]
    command += [
        "--weight_decay", "0", "--grad_clipping", "1",
        "--compressor", str(compressor["name"]),
        "--compress_rank", str(compressor["rank"]),
        "--start_compress_iter", str(compressor["quality_start_step"]),
        "--update_proj_gap", str(compressor["update_gap"]),
        "--use_error_feedback", f"ef{compressor['ef_rank']}",
        "--min_compression_rate", _value(compressor["min_compression_rate"]),
        "--beta_ef", _value(compressor["beta_ef"]),
        "--error_inherit", format(compressor["error_inherit"], "g"),
        "--contract_hash", str(profile["contract_hash"]),
        "--output_dir", str(output),
    ]
    # Old fixed-SHA candidates predate the diagnostic save API. Do not make an
    # otherwise valid historical evaluation fail on an unknown CLI argument.
    if (candidate / "diagnostic_checkpoint.py").is_file():
        command += ["--diagnostic_checkpoint", str(output.parent.parent / "checkpoints" / output.name)]
    if profile["activation_checkpointing"]:
        command.append("--activation_checkpointing")
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4", "PYTHONPATH": str(candidate),
    }
    return ProcessSpec(tuple(command), output.parent / f"{output.name}.process", env,
                       _ENV_ALLOWLIST, 24 * 60 * 60, cwd=candidate)
