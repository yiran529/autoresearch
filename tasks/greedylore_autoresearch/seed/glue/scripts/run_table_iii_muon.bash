#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

dry_run="${1:-}"
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

gpu_ids="${GPU_IDS:-0,1,2,7}"
dataset_root="${GLUE_CACHE_ROOT:-/home/wyr/.cache/huggingface/datasets/glue}"
model="${MODEL_NAME_OR_PATH:-FacebookAI/roberta-base}"
seed="${SEED:-1243}"
wait_interval="${WAIT_INTERVAL_SECONDS:-300}"
required_idle_polls="${REQUIRED_IDLE_POLLS:-2}"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTHONPATH=.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online

IFS=',' read -r -a gpus <<< "$gpu_ids"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "Exactly four GPU indices are required" >&2
    exit 2
fi

wait_for_gpus() {
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; return 1; }
    idle_polls=0
    while true; do
        stats="$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
        ready=1
        status=""
        for gpu in "${gpus[@]}"; do
            utilization="$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2; exit}' <<< "$stats")"
            if [[ ! "$utilization" =~ ^[0-9]+$ ]]; then
                echo "Cannot read utilization for GPU $gpu" >&2
                return 1
            fi
            status+=" GPU${gpu}=${utilization}%"
            if (( utilization >= 20 )); then ready=0; fi
        done
        if (( ready )); then
            idle_polls=$((idle_polls + 1))
        else
            idle_polls=0
        fi
        echo "$(date '+%F %T') GPU utilization:${status}; consecutive idle polls=${idle_polls}/${required_idle_polls}"
        if (( idle_polls >= required_idle_polls )); then break; fi
        sleep "$wait_interval"
    done
}

tasks=(sst2 cola mrpc stsb rte qnli qqp mnli)
arms=(dense r8 r16)

run_one() {
    local task="$1" arm="$2" experiment="$3"
    local compressor=none rank=8 ef=noef method=m001-dense
    if [[ "$arm" != dense ]]; then
        compressor=top_subspace
        rank="${arm#r}"
        ef=ef14
        method=m002-greedylore
    fi
    local run_id="${experiment}-${method}-muon-roberta-glue-${task}-${arm}-s${seed}-rerun1"
    local wandb_project="glue_no_trainer_${task}_greedylore"
    local output_dir="outputs/${run_id}"
    local -a command=(
        .venv/bin/torchrun --standalone --nproc_per_node=4
        glue/run_glue_no_trainer_HF.py
        --task_name "$task" --model_name_or_path "$model"
        --local_glue_cache_root "$dataset_root"
        --max_length 256 --per_device_train_batch_size 4 --per_device_eval_batch_size 8
        --gradient_accumulation_steps 1 --num_train_epochs 10
        --lr_scheduler_type cosine --warmup_fraction 0.1
        --optimizer muon --learning_rate 0.0002 --muon_scalar_lr 0.00005
        --muon_mu 0.95 --muon_epsilon 1e-8
        --muon_scalar_beta1 0.9 --muon_scalar_beta2 0.999 --muon_scalar_eps 1e-8
        --muon_adjust_lr spectral_norm --weight_decay 0 --muon_scalar_weight_decay 0
        --compressor "$compressor" --compress_rank "$rank"
        --use_error_feedback "$ef" --start_compress_iter 1000 --update_proj_gap 200
        --dtype float32 --seed "$seed" --output_dir "$output_dir"
        --with_tracking --report_to wandb --wandb_project "$wandb_project"
        --wandb_job_type formal-table-iii --wandb_run_name "$run_id"
    )
    if [[ "$dry_run" == --dry-run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    mkdir -p "$output_dir"
    printf '%s starting %s\n' "$(date '+%F %T')" "$run_id"
    "${command[@]}" 2>&1 | tee "$output_dir/train.log"
    test -f "$output_dir/all_results.json"
}

if [[ "$dry_run" != --dry-run ]]; then wait_for_gpus; fi
for task_index in "${!tasks[@]}"; do
    for arm_index in "${!arms[@]}"; do
        printf -v experiment 'CM%03d' "$((9 + task_index * 3 + arm_index))"
        run_one "${tasks[$task_index]}" "${arms[$arm_index]}" "$experiment"
    done
done
