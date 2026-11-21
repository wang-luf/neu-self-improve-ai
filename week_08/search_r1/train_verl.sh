#!/usr/bin/env bash
# Minimal VeRL + Search-R1 tool training entrypoint. Run inside a pinned VeRL checkout.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ALGORITHM="${1:-grpo}"
if [[ "$ALGORITHM" != "grpo" && "$ALGORITHM" != "ppo" ]]; then
  echo "Usage: $0 {grpo|ppo} [VeRL overrides...]" >&2
  exit 2
fi
shift || true

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-$HERE/data/searchr1_processed}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/results/verl-$ALGORITHM}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
ACTOR_LR="${ACTOR_LR:-2e-5}"
ROLLOUT_N="${ROLLOUT_N:-4}"
NGPUS="${NGPUS:-1}"
SEED="${SEED:-42}"
RETRIEVAL_URL="${SEARCH_R1_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

ALGORITHM_ARGS=()
if [[ "$ALGORITHM" == "grpo" ]]; then
  ALGORITHM_ARGS+=(algorithm.adv_estimator=grpo)
else
  # PPO: GAE advantages from a learned value critic initialised from the same base model.
  ALGORITHM_ARGS+=(
    algorithm.adv_estimator=gae
    critic.model.path="$MODEL_PATH"
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=1e-5
    critic.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  )
fi

export SEARCH_R1_RETRIEVAL_URL="$RETRIEVAL_URL"
export SEARCH_R1_TOPK="${SEARCH_R1_TOPK:-3}"

python3 -m verl.trainer.main_ppo \
  "${ALGORITHM_ARGS[@]}" \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed="$SEED" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$HERE/search_tool.py" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward.custom_reward_function.path="$HERE/rewards.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=search-r1-course \
  trainer.experiment_name="qwen25-3b-lora-$ALGORITHM" \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.logger='[console,file]' \
  "$@"
