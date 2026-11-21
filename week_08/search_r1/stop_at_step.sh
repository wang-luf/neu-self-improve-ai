#!/usr/bin/env bash
# Stop a running VeRL training run once STEP is fully checkpointed and its
# validation metrics are logged. Usage: stop_at_step.sh RUN_NAME STEP
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"
STEP="$2"
RUN_DIR="$HERE/results/runs/$RUN"
PY="${VERL_DIR:-$HOME/verl}/.venv/bin/python"

validated() {
  "$PY" - "$RUN_DIR/metrics.jsonl" "$STEP" <<'PYEOF'
import json, sys
key = "val-core/searchR1_nq_hotpotqa/acc/mean@1"
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
sys.exit(0 if any(r.get("step") == int(sys.argv[2]) and key in r.get("data", {}) for r in rows) else 1)
PYEOF
}

until [[ -f "$RUN_DIR/latest_checkpointed_iteration.txt" ]] \
  && (( $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") >= STEP )) \
  && validated; do
  sleep 20
done

pid="$(pgrep -f "verl.trainer.main_ppo.*default_local_dir=$RUN_DIR( |$)" | head -1)"
echo "$(date -u) step $STEP checkpointed and validated; stopping trainer pid=${pid:-none}"
echo "{\"stopped_at_step\": $STEP, \"reason\": \"planned budget: both algorithms stopped at the same step\", \"time_utc\": \"$(date -u +%FT%TZ)\"}" \
  > "$RUN_DIR/planned_stop.json"
if [[ -n "$pid" ]]; then
  kill -TERM "$pid"
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
fi
echo "$(date -u) trainer stopped"
