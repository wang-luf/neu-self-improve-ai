#!/usr/bin/env bash
# Evaluate each saved checkpoint of a run on Bamboogle as soon as it is complete.
# Usage: watch_and_eval.sh RUN_NAME GPU "50 100 150 ..."
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"
GPU="$2"
STEPS="$3"
RUN_DIR="$HERE/results/runs/$RUN"

for step in $STEPS; do
  # VeRL writes latest_checkpointed_iteration.txt only after the checkpoint is fully saved.
  until [[ -f "$RUN_DIR/latest_checkpointed_iteration.txt" ]] && (( $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") >= step )); do
    sleep 30
  done
  if [[ -f "$HERE/results/bamboogle/${RUN}_step${step}.summary.json" ]]; then
    continue
  fi
  echo "$(date -u) evaluating $RUN step $step on GPU $GPU"
  bash "$HERE/eval_checkpoint.sh" "$RUN" "$step" "$GPU" 5
done
echo "$(date -u) done"
