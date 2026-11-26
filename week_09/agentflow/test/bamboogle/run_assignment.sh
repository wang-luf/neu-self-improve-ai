#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(dirname "$HERE")"
cd "$TEST_DIR"

LABEL="${LABEL:-AgentFlow-base}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
ENABLED_TOOLS="${ENABLED_TOOLS:-Base_Generator_Tool,Python_Coder_Tool,Local_Wiki_Search_Tool}"
TOOL_ENGINE="${TOOL_ENGINE:-vllm-Qwen/Qwen2.5-7B-Instruct,vllm-Qwen/Qwen2.5-7B-Instruct,Default}"
MODEL_ENGINE="${MODEL_ENGINE:-trainable,vllm-Qwen/Qwen2.5-7B-Instruct,vllm-Qwen/Qwen2.5-7B-Instruct,vllm-Qwen/Qwen2.5-7B-Instruct}"
# Assignment: local services instead of API keys (judge = local vLLM, search = local wiki-18 retriever).
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
export WIKI_RETRIEVER_URL="${WIKI_RETRIEVER_URL:-http://127.0.0.1:8002/retrieve}"
DATA_FILE="bamboogle/data/data.json"
OUT_DIR="bamboogle/results/$LABEL"
LOG_DIR="bamboogle/logs/$LABEL"
CACHE_DIR="bamboogle/cache/$LABEL"  # per-run cache so the two AgentFlow runs share nothing

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CACHE_DIR"
solve_one() {
  local index=$1
  if [[ -f "$OUT_DIR/output_$index.json" ]]; then
    return 0
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
    --base_url "$BASE_URL" > "$LOG_DIR/$index.log" 2>&1
  echo "done $index"
}
export -f solve_one
export DATA_FILE OUT_DIR LOG_DIR CACHE_DIR MODEL_NAME ENABLED_TOOLS TOOL_ENGINE MODEL_ENGINE BASE_URL
# PARALLEL questions share the same vLLM server (one GPU job); results do not depend on the order.
seq 0 124 | xargs -P "${PARALLEL:-1}" -I{} bash -c "solve_one {}"

python calculate_score_unified.py \
  --task_name bamboogle \
  --data_file "$DATA_FILE" \
  --result_dir "$OUT_DIR" \
  --response_type direct_output \
  --output_file "final_scores_direct_output.json"
