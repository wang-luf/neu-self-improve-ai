#!/usr/bin/env bash
set -euo pipefail

adapter_dir="${1:?Usage: serve_adapter.sh ADAPTER_DIR [ADAPTER_NAME]}"
adapter_name="${2:-week9-planner}"
base_model="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
port="${PORT:-8000}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"

test -f "$adapter_dir/adapter_config.json"
test -f "$adapter_dir/adapter_model.safetensors"

CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$base_model" \
  --host 0.0.0.0 \
  --port "$port" \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "$adapter_name=$adapter_dir"
