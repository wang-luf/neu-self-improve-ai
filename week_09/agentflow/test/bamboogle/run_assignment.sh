#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(dirname "$HERE")"
cd "$TEST_DIR"

LABEL="${LABEL:-AgentFlow-base}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
ENABLED_TOOLS="${ENABLED_TOOLS:-Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool}"
TOOL_ENGINE="${TOOL_ENGINE:-gpt-4o-mini,dashscope-qwen2.5-coder-7b-instruct,Default,Default}"
MODEL_ENGINE="${MODEL_ENGINE:-trainable,dashscope,dashscope,dashscope}"
DATA_FILE="bamboogle/data/data.json"
OUT_DIR="bamboogle/results/$LABEL"
LOG_DIR="bamboogle/logs/$LABEL"
CACHE_DIR="bamboogle/cache"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CACHE_DIR"
for index in $(seq 0 124); do
  if [[ -f "$OUT_DIR/output_$index.json" ]]; then
    continue
  fi
  python solve.py \
    --index "$index" \
    --task bamboogle \
    --data_file "$DATA_FILE" \
    --llm_engine_name "vllm-$MODEL_NAME" \
    --root_cache_dir "$CACHE_DIR" \
    --output_json_dir "$OUT_DIR" \
    --output_types direct \
    --enabled_tools "$ENABLED_TOOLS" \
    --tool_engine "$TOOL_ENGINE" \
    --model_engine "$MODEL_ENGINE" \
    --max_time 300 \
    --max_steps 10 \
    --temperature 0.0 \
    --base_url "$BASE_URL" 2>&1 | tee "$LOG_DIR/$index.log"
done

python calculate_score_unified.py \
  --task_name bamboogle \
  --data_file "$DATA_FILE" \
  --result_dir "$OUT_DIR" \
  --response_type direct_output \
  --output_file "final_scores_direct_output.json"
