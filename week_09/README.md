# Week 9 Part 1 — AgentFlow on Bamboogle

## Assignment checklist

Part 1 asks for three Bamboogle results using Qwen-2.5-7B-Instruct:

1. Direct Qwen inference with no tools.
2. AgentFlow with the base planner and no Flow-GRPO training.
3. AgentFlow with a Flow-GRPO-trained planner.

The training run uses LoRA on the Planner. Keep the Bamboogle evaluation set
out of training. The course PDF allows a smaller training run when compute is
limited, provided the result shows improvement and the reward trend is reported.

This directory contains the official AgentFlow source pinned in
[`UPSTREAM_COMMIT.txt`](UPSTREAM_COMMIT.txt), a no-tools direct baseline, an
AgentFlow Bamboogle runner, and small-run LoRA configurations. The official
source is kept intact except for the clearly marked assignment additions and
the config-path / argument handling fix in `train/train_agent.py`.

## Set up on the GPU host

```bash
cd week_09/agentflow
bash setup.sh
source .venv/bin/activate
cp agentflow/.env.template agentflow/.env
```

Fill `agentflow/.env` with the API keys needed by the configured search,
executor, verifier, and answer judge. Do not commit `.env`. The default
AgentFlow pipeline uses more than the local Qwen planner; if replacing a tool
engine, record that change in the experiment report.

Prepare the official AgentFlow training and validation files:

```bash
python data/get_train_data.py
python data/aime24_data.py
python train/prepare_train_subset.py --rows 8 \
  --output data/train/combined_train_smoke.parquet
```

## One-step training smoke

Use two terminals on the same GPU host. The rollout server and trainer are two
processes for the same experiment.

Terminal A:

```bash
cd week_09/agentflow
source .venv/bin/activate
AGENTFLOW_CONFIG=train/config.smoke.yaml python train/rollout.py
```

Terminal B:

```bash
cd week_09/agentflow
source .venv/bin/activate
AGENTFLOW_CONFIG=train/config.smoke.yaml python train/train_agent.py
```

The smoke config uses Qwen-2.5-7B-Instruct, LoRA rank 16 / alpha 32, two
rollouts per prompt, and one training step. Confirm that it logs rewards and
saves a VeRL checkpoint before starting the Bamboogle baselines.

## Part 1 training run

Prepare a larger deterministic subset and use the assignment config, which is
set to 25 steps as a first short run:

```bash
python train/prepare_train_subset.py --rows 256 \
  --output data/train/combined_train_small.parquet
```

Start the same two processes with `AGENTFLOW_CONFIG=train/config.assignment.yaml`.
The config uses group size 8 and LoRA only on the Planner. If reward continues
to rise and the run is stable, increase `trainer.total_training_steps` and
`trainer.total_epochs` together. Do not use Bamboogle for training or
checkpoint selection.

After training, the final FSDP checkpoint is under
`checkpoints/week9-agentflow/week9-part1-flowgrpo-lora/global_step_*/actor`.
Export its PEFT adapter with the VeRL version installed by AgentFlow:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/week9-agentflow/week9-part1-flowgrpo-lora/global_step_25/actor \
  --target_dir results/week9-part1/flowgrpo-step25 \
  --use_cpu_initialization
```

The exported adapter is in `results/week9-part1/flowgrpo-step25/lora_adapter/`.

## Evaluate the three systems

### Qwen direct baseline, no tools

```bash
python evaluate_direct.py --model Qwen/Qwen2.5-7B-Instruct \
  --output test/bamboogle/results/Qwen-direct
python test/calculate_score_unified.py --task_name bamboogle \
  --data_file test/bamboogle/data/data.json \
  --result_dir test/bamboogle/results/Qwen-direct \
  --response_type direct_output \
  --output_file final_scores_direct_output.json
```

### Base AgentFlow and trained AgentFlow

Start a local vLLM server with the base model for the untrained AgentFlow run:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

Then run all 125 questions sequentially:

```bash
LABEL=AgentFlow-base MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
  bash test/bamboogle/run_assignment.sh
```

Stop the base server. Serve the trained adapter:

```bash
bash train/serve_adapter.sh results/week9-part1/flowgrpo-step25/lora_adapter week9-planner
```

In another terminal, run the trained planner through the same AgentFlow
pipeline:

```bash
LABEL=AgentFlow-flowgrpo-step25 MODEL_NAME=week9-planner \
  bash test/bamboogle/run_assignment.sh
```

The evaluator resumes from completed `output_N.json` files. Keep the
environment, search tools, scoring method, decoding, and question set fixed
between the two AgentFlow runs. The no-tools baseline uses the same official
125-question `data.json` and the same answer scorer.

## Report and demo artifacts

Save the following under `week_09/results/` after the runs:

- Per-question JSON output for all three configurations.
- Final score JSON and a three-row comparison table.
- Training config, reward log/curve, and LoRA adapter export.
- A short trace showing Planner → tool → Verifier/Generator behavior.
- Actual GPU, steps, runtime, model/tool engines, and deviations from the paper.

Paper reference values for Bamboogle are 12.0% for direct Qwen,
58.4% for AgentFlow before Flow-GRPO, and 69.6% after Flow-GRPO. Report these
as paper numbers beside your measurements, not in place of them.

## Current status

The official source and run scripts are staged. No Week 9 GPU smoke, training,
or Bamboogle evaluation has run from this checkout yet. This local Mac has no
NVIDIA GPU or Modal CLI; run the commands on the course GPU/B200 host and copy
the logs, adapter, and result JSON back here.
