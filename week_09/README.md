# Week 9 Part 1 — AgentFlow on Bamboogle

Part 1 compares three systems on the 125-question Bamboogle set, all built on
Qwen-2.5-7B-Instruct:

1. Direct Qwen inference with no tools.
2. AgentFlow with the base planner (no Flow-GRPO).
3. AgentFlow with a planner trained by Flow-GRPO using LoRA.

Results, curves and the run report are in [`results/`](results/README.md).

## What is in this directory

- `agentflow/` — the official AgentFlow source pinned in
  [`UPSTREAM_COMMIT.txt`](UPSTREAM_COMMIT.txt), plus a small set of
  assignment changes listed below.
- `retriever/gpu_retrieval_server.py` — local Wikipedia retrieval service
  (wiki-18 corpus + E5 Flat index, exact inner-product search on one GPU),
  reused from Week 8.
- `patches/verl-0.5.0-async-lora-merge.patch` — patch for the pip-installed
  `verl==0.5.0` that makes LoRA work with AgentFlow's async rollout server.
- `scripts/` — GB300 environment setup (`setup_gb300.sh`, `gb300_env.sh`) and
  the analysis scripts that produced the result tables and plots
  (`plot_training.py`, `analyze_rollouts.py`, `compare_agentflow_runs.py`,
  `show_trace.py`).
- `results/` — per-question outputs, scores, training logs, reward curve,
  LoRA adapter and the run report.

## Changes from the official AgentFlow pipeline

The course allows substitutions that make the replication cheaper as long as
the main result is shown. This run uses **no paid API**:

| Paper / official code | This replication | Where |
| --- | --- | --- |
| Executor, Verifier, fixed planner, Base_Generator and Python_Coder engines via DashScope / gpt-4o-mini | the same Qwen2.5-7B-Instruct served locally with vLLM (port 8000) | `train/config.*.yaml`, `test/bamboogle/run_assignment.sh` |
| Google_Search_Tool (Gemini) and Wikipedia_Search_Tool (OpenAI) | `Local_Wiki_Search_Tool`: top-3 passages from the wiki-18 dense index (port 8002) | `agentflow/agentflow/tools/local_wiki_search/tool.py` |
| gpt-4o answer judge for the training reward and for scoring | the same judge prompt sent to the local Qwen2.5-7B server with structured output | `agentflow/agentflow/engine/local_judge.py`, enabled by `JUDGE_BASE_URL` in `train/utils.py` and `test/calculate_score_unified.py` |
| full Flow-GRPO run | 25 steps × 4 prompts × 8 rollouts on a 256-row subset, LoRA r=16/alpha=32 on the planner | `train/config.assignment.yaml` |

To return to the paper's setup, unset `JUDGE_BASE_URL` and put the original
`ENABLE_TOOLS` / `TOOL_ENGINE` / `MODEL_ENGINE` values back in the config.

Bug fixes needed for LoRA training (the official code path had only been run
with full fine-tuning):

- `train/rollout.py` returns the outcome reward from the rollout. Otherwise the
  reward span was not matched to the last traced LLM call, every rollout
  reported `final_reward=None`, and Flow-GRPO trained on all-zero rewards.
- `agentflow/verl/trainer.py` computes reference log-probs with the actor when
  LoRA is on (`ref_in_actor`), as `verl` 0.5.0's own trainer does.
- `patches/verl-0.5.0-async-lora-merge.patch`: AgentFlow serves the policy with
  vLLM's async server, which has no LoRA support, so `verl` crashed in
  `add_lora` and would otherwise have kept rolling out with the base weights.
  The patch pushes the merged base+LoRA weights to vLLM on every sync (only the
  LoRA parameters are trained; the merge uses `writeback=False`).
- `verl.model_merger` exports the adapter with `lora_alpha: 0`; the export step
  below copies the training-time `adapter_config.json` (alpha 32) over it.

## Set up on the GPU host

The run used a GB300 node (aarch64, CUDA driver 13.2). On x86 + CUDA 12.x,
`bash agentflow/setup.sh` is enough. On aarch64 the pinned stack needs extra
steps, recorded in [`scripts/setup_gb300.sh`](scripts/setup_gb300.sh) and
[`results/RUN_REPORT.md`](results/RUN_REPORT.md):
flash-attn 2.8.1 and vLLM 0.9.2 built from source against CUDA 12.8,
Triton 3.5.1 with a CUDA 13.1 `ptxas`, `TORCHDYNAMO_DISABLE=1`, and vLLM in
eager mode.

```bash
cd week_09/agentflow
bash setup.sh                     # official installer
source .venv/bin/activate
patch -d .venv/lib/python3.11/site-packages -p1 < ../patches/verl-0.5.0-async-lora-merge.patch
pip install faiss-cpu==1.12.0     # for the local retriever
```

Data (no Bamboogle question appears in the training data):

```bash
python data/get_train_data.py     # 182,190 NQ + DeepMath rows
python data/aime24_data.py        # 30 AIME24 validation rows
python train/prepare_train_subset.py --rows 8   --output data/train/combined_train_smoke.parquet
python train/prepare_train_subset.py --rows 256 --output data/train/combined_train_small.parquet
```

`test/bamboogle/data/data.json` is the official 125-question file from the
pinned upstream commit.

## Local services

```bash
# Fixed modules, tools and judge (and the planner during evaluation): GPU 1, port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --host :: --port 8000 \
  --enable-lora --max-lora-rank 16 --enforce-eager
# Wikipedia retrieval: GPU 2, port 8002 (wiki-18 corpus + E5 Flat index from PeterJinGo/wiki-18-*)
CUDA_VISIBLE_DEVICES=2 python ../retriever/gpu_retrieval_server.py --port 8002 \
  --index_path e5_Flat.index --corpus_path wiki-18.jsonl
export JUDGE_BASE_URL=http://localhost:8000/v1 WIKI_RETRIEVER_URL=http://127.0.0.1:8002/retrieve VLLM_USE_V1=1
```

## Train (two terminals, as in the official README)

```bash
AGENTFLOW_CONFIG=train/config.smoke.yaml python train/rollout.py        # terminal A
AGENTFLOW_CONFIG=train/config.smoke.yaml python train/train_agent.py    # terminal B
```

The smoke run (1 step) must log rewards and save
`checkpoints/week9-agentflow/week9-part1-smoke/global_step_1`. Then run the same
two commands with `train/config.assignment.yaml` (25 steps, GPU 0).

Export the final adapter:

```bash
CKPT=checkpoints/week9-agentflow/week9-part1-flowgrpo-lora/global_step_25/actor
python -m verl.model_merger merge --backend fsdp --local_dir $CKPT \
  --target_dir ../results/flowgrpo-step25 --use_cpu_initialization
cp $CKPT/lora_adapter/adapter_config.json ../results/flowgrpo-step25/lora_adapter/
```

## Evaluate the three systems

```bash
# 1. Qwen direct, no tools
python evaluate_direct.py --model Qwen/Qwen2.5-7B-Instruct --output test/bamboogle/results/Qwen-direct
(cd test && python calculate_score_unified.py --task_name bamboogle --data_file bamboogle/data/data.json \
  --result_dir bamboogle/results/Qwen-direct --response_type direct_output --output_file final_scores_direct_output.json)

# Serve the base model and the trained adapter from one server, so both AgentFlow runs
# use identical fixed modules, tools, decoding and judge:
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --host :: --port 8000 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules week9-planner=../results/flowgrpo-step25/lora_adapter

# 2. AgentFlow, base planner       3. AgentFlow, Flow-GRPO planner
LABEL=AgentFlow-base MODEL_NAME=Qwen/Qwen2.5-7B-Instruct PARALLEL=8 bash test/bamboogle/run_assignment.sh
LABEL=AgentFlow-flowgrpo-step25 MODEL_NAME=week9-planner PARALLEL=8 bash test/bamboogle/run_assignment.sh
```

`PARALLEL` runs several questions at once against the same server; each
question is still solved independently with temperature 0.

## Results

| System | Judge accuracy (local Qwen judge) | Rule-based EM | Paper (gpt-4o judge) |
| --- | ---: | ---: | ---: |
| Qwen-2.5-7B-Instruct, direct, no tools | 12/125 = **9.6%** | 8/125 = 6.4% | 12.0% |
| AgentFlow, base planner (no Flow-GRPO) | 36/125 = **28.8%** | 18/125 = 14.4% | 58.4% |
| AgentFlow, Flow-GRPO planner (LoRA, 25 steps) | 37/125 = **29.6%** | 16/125 = 12.8% | 69.6% |

AgentFlow clearly beats direct inference; 25 LoRA Flow-GRPO steps did **not**
measurably improve the planner (+1 question, within noise), and the training
reward showed no sustained upward trend. See the run report for the analysis.

Paper numbers are references only (they use Google search, DashScope Qwen
modules and a gpt-4o judge); see [`results/RUN_REPORT.md`](results/RUN_REPORT.md)
for the comparison, reward curve and limitations.

## Source provenance and artifact portability

This implementation and its measured artifacts were transferred from the
collaborative source snapshot recorded in [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md).
Machine-specific workspace paths were normalized in retained text logs. Full
raw console captures were omitted; compact training logs, reward traces,
scoring logs, evaluation outputs, and the adapter are included.
