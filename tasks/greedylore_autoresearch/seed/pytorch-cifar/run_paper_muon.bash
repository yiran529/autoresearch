#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

dataset="${1:-}"
arm="${2:-}"
dry_run="${3:-}"
if [[ "$dataset" != cifar10 && "$dataset" != cifar100 ]] || [[ "$arm" != dense && "$arm" != greedylore ]]; then
    echo "usage: $0 {cifar10|cifar100} {dense|greedylore} [--dry-run]" >&2
    exit 2
fi
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then
    echo "unknown option: $dry_run" >&2
    exit 2
fi

compressor=none
method=m001-dense
if [[ "$dataset" == cifar10 ]]; then
    experiment=CM044
    scalar_lr=0.005
    start=500
    gap=750
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    project=GreedyLore-CIFAR10-Muon
else
    experiment=CM046
    scalar_lr=0.0005
    start=4000
    gap=1200
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    project=GreedyLore-CIFAR100-Muon
fi
if [[ "$arm" == greedylore ]]; then
    compressor=top_subspace
    method=m002-greedylore
    if [[ "$dataset" == cifar10 ]]; then experiment=CM045; else experiment=CM047; fi
fi
run_id="${experiment}-${method}-muon-resnet18-${dataset}-r64-lr0p02-fp32-s1243"
output_dir="outputs/${run_id}"
export WANDB_MODE=online
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
command=(
    .venv/bin/torchrun --standalone --nproc_per_node=4
    pytorch-cifar/train_ddp.py
    --dataset "$dataset" --data-dir /home/wyr/ARC-TopK-release/data
    --output-dir "$output_dir" --epochs 40 --batch-size 32 --workers 2 --seed 1243
    --lr 0.02 --weight-decay 0.0005 --muon_mu 0.95 --muon_epsilon 1e-8
    --muon_scalar_lr "$scalar_lr" --muon_scalar_beta1 0.9 --muon_scalar_beta2 0.999
    --muon_scalar_eps 1e-8 --muon_scalar_weight_decay 0.0005 --muon_adjust_lr spectral_norm
    --compressor "$compressor" --compress-rank 64 --min-compression-rate 2
    --start-compress-iter "$start" --update-proj-gap "$gap"
    --wandb-project "$project" --wandb-mode online
)
if [[ "$dry_run" == --dry-run ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q WANDB_MODE=online ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ -e "$output_dir/config.json" ]]; then
    echo "Existing run: $output_dir; refusing to overwrite or implicitly resume" >&2
    exit 1
fi
mkdir -p "$output_dir"
export WANDB_DIR="$(pwd)/$output_dir"
printf '%q ' "${command[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
git rev-parse HEAD > "$output_dir/git_commit.txt"
sha256sum pytorch-cifar/train_ddp.py pytorch-cifar/models/resnet.py optimizer/muon.py \
    optimizer/muon_utils.py comm_hooks/subspace_hook.py comm_hooks/default_hooks.py \
    pytorch-cifar/run_paper_muon.bash pytorch-cifar/queue_paper_muon.bash > "$output_dir/source_sha256.txt"
"${command[@]}" 2>&1 | tee "$output_dir/train.log"
# A successful process alone is not enough to pass the budget gate.
.venv/bin/python - "$output_dir/all_results.json" <<'PY'
import json, sys
with open(sys.argv[1]) as file:
    result = json.load(file)
assert result['status'] == 'completed' and result['epoch'] == 40
assert result['update_step'] == 15640 and result['test_samples'] == 10000
PY
