#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

start_now="${1:-}"
if [[ -n "$start_now" && "$start_now" != --start-now ]]; then
    echo "usage: $0 [--start-now]" >&2
    exit 2
fi

queue_dir="outputs/CM033-CM034-greedylore-130m-followups"
mkdir -p "$queue_dir"
status_file="$queue_dir/status.tsv"
if [[ ! -e "$status_file" ]]; then
    printf 'run\tstate\ttime\tdetail\n' > "$status_file"
fi
printf 'queue\twaiting\t%s\tprior 130M runs and GPU3-6\n' "$(date --iso-8601=seconds)" >> "$status_file"

run_complete() {
    .venv/bin/python - "$1" <<'PY'
import json
from pathlib import Path
import sys

result = Path("outputs") / sys.argv[1] / "all_results.json"
if not result.exists() or json.loads(result.read_text()).get("update_step") != 20000:
    raise SystemExit(1)
PY
}

cm007="CM007-m002-greedylore-muon-llama130m-c4-r256-bf16-s1243"
cm008="CM008-m003-dense-adamw-llama130m-c4-adam_dense-lr2p5e3-bf16-s1243"

gpus_idle() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
        awk -F, '
            {
                for (i = 1; i <= 3; i++) gsub(/[[:space:]]/, "", $i)
                if ($1 >= 3 && $1 <= 6) {
                    count++
                    if ($2 > 50 || $3 > 1500) busy = 1
                }
            }
            END { exit (count != 4 || busy) }
        '
}

wait_for_gpus() {
    local ready=0
    while (( ready < 3 )); do
        if tmux has-session -t greedylore_tableiv_130m_bf16 2>/dev/null; then
            ready=0
        elif ! gpus_idle; then
            ready=0
        else
            ready=$((ready + 1))
        fi
        if (( ready < 3 )); then sleep 30; fi
    done
}

while tmux has-session -t greedylore_tableiv_130m_bf16 2>/dev/null; do
    sleep 30
done
if ! run_complete "$cm007"; then
    printf 'queue\tblocked\t%s\tCM007 missing complete 20000-step result\n' "$(date --iso-8601=seconds)" >> "$status_file"
    exit 1
fi

if ! run_complete "$cm008"; then
    if [[ "$start_now" != --start-now ]]; then wait_for_gpus; fi
    printf 'CM008\tstarted\t%s\tDense AdamW recovery after original queue ended\n' "$(date --iso-8601=seconds)" >> "$status_file"
    if bash c4/scripts/run_table_iv_130m.bash adam_dense >/dev/null 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf 'CM008\tfinished\t%s\texit_code=%s\n' "$(date --iso-8601=seconds)" "$exit_code" >> "$status_file"
    if (( exit_code != 0 )) || ! run_complete "$cm008"; then
        printf 'queue\tblocked\t%s\tCM008 recovery incomplete\n' "$(date --iso-8601=seconds)" >> "$status_file"
        exit 1
    fi
fi

for arm in r64 r32_sci2000; do
    if [[ "$start_now" != --start-now ]]; then wait_for_gpus; fi
    printf '%s\tstarted\t%s\t%s\n' "$arm" "$(date --iso-8601=seconds)" 'GPU3-6 assigned' >> "$status_file"
    if bash c4/scripts/run_table_iv_130m.bash "$arm" >/dev/null 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\tfinished\t%s\texit_code=%s\n' "$arm" "$(date --iso-8601=seconds)" "$exit_code" >> "$status_file"
done
