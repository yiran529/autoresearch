#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-}"
if [[ "$dataset" != cifar10 && "$dataset" != cifar100 ]]; then exit 2; fi
mkdir -p outputs/CM044-CM047-cifar
status="outputs/CM044-CM047-cifar/status-${dataset}.tsv"
if [[ -e "$status" ]]; then
    echo "Existing queue status: $status; refusing to overwrite" >&2
    exit 1
fi
printf 'timestamp\tdataset\tarm\tstatus\n' > "$status"
for arm in dense greedylore; do
    printf '%s\t%s\t%s\tstarted\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$dataset" "$arm" >> "$status"
    if bash pytorch-cifar/run_paper_muon.bash "$dataset" "$arm"; then
        printf '%s\t%s\t%s\tcompleted\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$dataset" "$arm" >> "$status"
    else
        printf '%s\t%s\t%s\tfailed\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$dataset" "$arm" >> "$status"
        exit 1
    fi
done
printf '%s\t%s\tall\tqueue_finished\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$dataset" >> "$status"
