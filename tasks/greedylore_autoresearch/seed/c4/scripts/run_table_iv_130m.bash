#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

arm="${1:-}"
dry_run="${2:-}"
if [[ "$arm" != dense && "$arm" != r32 && "$arm" != r256 && "$arm" != adam_dense && "$arm" != r64 && "$arm" != r32_sci2000 ]]; then
    echo "usage: $0 {dense|r32|r256|adam_dense|r64|r32_sci2000} [--dry-run]" >&2
    exit 2
fi
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then
    echo "unknown option: $dry_run" >&2
    exit 2
fi

compressor=none
rank=32
start_compress_iter=1000
method=m001-dense
experiment=CM005
optimizer_name=muon
lr_tag=""
matrix_lr="${MUON_MATRIX_LR:-0.02}"
scalar_lr="${MUON_SCALAR_LR:-0.001}"
optimizer_args=(
    --optimizer muon --lr "$matrix_lr" --weight_decay 0.0
    --muon_mu 0.95 --muon_epsilon 1e-8
    --muon_scalar_lr "$scalar_lr" --muon_scalar_beta1 0.9
    --muon_scalar_beta2 0.999 --muon_scalar_eps 1e-8
    --muon_scalar_weight_decay 0.0 --muon_adjust_lr spectral_norm
)
if [[ "$arm" == r32 ]]; then
    compressor=top_subspace
    rank=32
    method=m002-greedylore
    experiment=CM006
elif [[ "$arm" == r256 ]]; then
    compressor=top_subspace
    rank=256
    method=m002-greedylore
    experiment=CM007
elif [[ "$arm" == r64 ]]; then
    compressor=top_subspace
    rank=64
    method=m002-greedylore
    experiment=CM033
elif [[ "$arm" == r32_sci2000 ]]; then
    compressor=top_subspace
    rank=32
    start_compress_iter=2000
    method=m002-greedylore
    experiment=CM034
elif [[ "$arm" == adam_dense ]]; then
    method=m003-dense
    experiment=CM008
    optimizer_name=adamw
    lr_tag="-lr2p5e3"
    optimizer_args=(
        --optimizer adamw --lr 0.0025
        --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.0
    )
fi

seed="${SEED:-1243}"
run_id="${RUN_ID:-${experiment}-${method}-${optimizer_name}-llama130m-c4-${arm}${lr_tag}-bf16-s${seed}}"
output_dir="outputs/${run_id}"
dataset_path="${C4_DATASET_PATH:-/dev/shm/wyr_tmp/c4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
export WANDB_MODE=online
export HF_HOME="${HF_HOME:-/dev/shm/wyr_tmp/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH=.

command=(
    .venv/bin/torchrun --standalone --nproc_per_node=4
    c4/run_llama_pretraining.py
    --model_config c4/configs/llama_130m.json
    --dataset_path "$dataset_path"
    --max_length 256 --dtype bfloat16
    --num_training_steps 20000 --warmup_steps 2000
    --activation_checkpointing
    --scheduler cosine --min_lr_ratio 0.1
    --eval_every 1000 --save_every 40000
    --total_batch_size 512 --batch_size 128 --gradient_accumulation 1
    --workers 4 --seed "$seed"
    "${optimizer_args[@]}"
    --compressor "$compressor" --compress_rank "$rank"
    --min_compression_rate 1.0 --start_compress_iter "$start_compress_iter"
    --update_proj_gap 200 --use_error_feedback ef14
    --beta_ef 0 --error_inherit 0
    --grad_clipping 1.0
    --wandb_project GreedyLore-TableIV
    --wandb_job_type formal-pretraining
    --wandb_run_name "$run_id"
    --output_dir "$output_dir"
)
if [[ "$dry_run" == --dry-run ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q WANDB_MODE=%q ' "$CUDA_VISIBLE_DEVICES" "$WANDB_MODE"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$output_dir"
export WANDB_DIR="$(pwd)/${output_dir}"
"${command[@]}" 2>&1 | tee "${output_dir}/train.log"
