#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

seed="${SEED:-1243}"
muon_session=greedylore_tableiv_60m_bf16
last_muon_result="outputs/CM003-m002-greedylore-muon-llama60m-c4-r128-bf16-s${seed}/all_results.json"

while tmux has-session -t "$muon_session" 2>/dev/null; do
    sleep 60
done

if ! .venv/bin/python - "$last_muon_result" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Muon sequence did not finish: {path} is missing")
with path.open() as file:
    result = json.load(file)
if result.get("update_step") != 8393:
    raise SystemExit(f"Muon sequence did not reach 8393 updates: {path}")
PY
then
    exit 1
fi

bash c4/scripts/run_table_iv_60m_muon.bash adam_dense
