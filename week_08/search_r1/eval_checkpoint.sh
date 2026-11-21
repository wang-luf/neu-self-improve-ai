#!/usr/bin/env bash
# Export the LoRA adapter of one VeRL checkpoint and evaluate it on all 125
# Bamboogle questions in tool mode, split over N parallel shards on one GPU.
# Usage: eval_checkpoint.sh RUN_NAME STEP GPU [SHARDS]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"
STEP="$2"
GPU="$3"
SHARDS="${4:-5}"
VERL_DIR="${VERL_DIR:-$HOME/verl}"
PY="$VERL_DIR/.venv/bin/python"
RUN_DIR="$HERE/results/runs/$RUN"
EXPORT_DIR="$RUN_DIR/adapter-step$STEP"
OUT="$HERE/results/bamboogle/${RUN}_step${STEP}"

if [[ ! -f "$EXPORT_DIR/lora_adapter/adapter_model.safetensors" ]]; then
  (cd "$VERL_DIR" && HF_HUB_OFFLINE=1 "$PY" -m verl.model_merger merge --backend fsdp \
    --local_dir "$RUN_DIR/global_step_$STEP/actor" --target_dir "$EXPORT_DIR" --use_cpu_initialization) \
    > "$RUN_DIR/adapter-step$STEP-export.log" 2>&1
fi

mkdir -p "$OUT.shards"
pids=()
for ((k = 0; k < SHARDS; k++)); do
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 "$PY" "$HERE/evaluate_bamboogle.py" --mode tool \
    --adapter "$EXPORT_DIR/lora_adapter" --shard "$k/$SHARDS" \
    --output "$OUT.shards/shard$k.jsonl" > "$OUT.shards/shard$k.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$HERE/merge_eval_shards.py" --shards "$OUT".shards/shard*.jsonl --output "$OUT.jsonl"
