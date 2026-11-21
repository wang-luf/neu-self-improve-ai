# Assignment — Search-R1 with PEFT

Use [`RUNBOOK.md`](RUNBOOK.md) for the step-by-step source map and execution
sequence.

## Assignment requirements found in the Week 8 course export

1. Understand and reproduce the class starter [`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora), then use its LoRA training setup as the starting point for Search-R1.
2. Reproduce the Search-R1 inference-only Bamboogle baselines for `Qwen/Qwen2.5-3B-Instruct`: **Direct Inference** and **Search-R1-Instruct**.
3. Train and compare Search-R1 with **GRPO and PPO**.
4. Use PEFT rather than full fine-tuning. The current baseline choice is LoRA with the same family of target modules and rank/alpha used in the class starter.
5. Submit individually and list every group member. Add names below before submission.

## Scope and separation of data

The supplied Bamboogle sheet PDF is preserved at
[`data/Bamboogle_Prerelease_Sheet1.pdf`](data/Bamboogle_Prerelease_Sheet1.pdf). Its 125
question/answer rows are in [`data/bamboogle.csv`](data/bamboogle.csv) as an evaluation
set; the sheet's unrelated canary column is omitted. Every CSV question and non-empty
answer alias was cross-checked against the PDF text. Do not use these 125 questions for
policy training. The Search-R1 paper trains on Natural Questions and HotpotQA and evaluates
Bamboogle as an out-of-domain multi-hop QA set. Use the paper's released
`nq_hotpotqa_train` data for training, then score the same Bamboogle set for all methods.

The expected paper references for this 125-row sheet are in
[`results/paper_reference.csv`](results/paper_reference.csv). They are **reference targets,
not results from this repository's run**:

| Method | Bamboogle EM | Correct out of 125 |
| --- | ---: | ---: |
| Direct inference | 0.024 | 3 |
| Search-R1-Instruct, Table 2 | 0.264 | 33 |
| Search-R1-Instruct, GRPO, Table 3 | 0.232 | 29 |
| Search-R1-Instruct, PPO, Table 3 | 0.264 | 33 |

Table 2's Search-R1-Instruct row is the PPO/default result; Table 3 breaks out GRPO and
PPO. The benchmark should report both paper references and our measured values, without
substituting one for the other.

## Algorithm and implementation notes

- Search-R1 makes search part of the rollout: the model emits a `<search>...</search>`
  query, receives retrieved passages in `<information>...</information>`, reasons again,
  and ends with `<answer>...</answer>`. It may search multiple times.
- The task reward is exact match on the extracted final answer. Search-R1 does not use a
  separate format reward in the paper.
- Retrieved passage tokens are observations, not policy actions. Exclude them from policy
  and KL losses. Train on model-generated reasoning, search queries, and answers.
- GRPO uses a relative reward baseline across multiple rollouts for a prompt. PPO uses a
  learned value baseline and clipped token-level policy updates. Keep model, data, search
  corpus, prompt, rollout budget, answer normalization, and evaluation split fixed when
  comparing them.
- LoRA freezes the pretrained model and trains low-rank adapters. Log trainable parameter
  count, adapter rank/alpha/targets, reference-policy KL, train reward, Bamboogle EM, and
  search-call count.

## Group members

Group member: **Lufei Wang** (individual submission).

## Current run status

Completed on a reserved 4x NVIDIA GB300 node (2026-11-20); see
[`REPLICATION_REPORT.md`](REPLICATION_REPORT.md#results) for the full results
and [`RUNBOOK.md`](RUNBOOK.md#gb300-local-execution-what-was-actually-run) for
the exact commands.

| Bamboogle EM (125 questions) | Measured | Paper |
| --- | ---: | ---: |
| Direct inference, Qwen2.5-3B-Instruct | 0.032 | 0.024 |
| Untrained Qwen2.5-3B-Instruct + search tool | 0.184 | — |
| Published Search-R1 PPO checkpoint | 0.240 | 0.264 |
| Published Search-R1 GRPO checkpoint | 0.344 | 0.232 |
| Ours: LoRA GRPO, 150 steps | 0.312 | (0.232, full FT) |
| Ours: LoRA PPO, 150 steps | 0.208 | (0.264, full FT) |

- One-step GRPO and PPO integration smokes passed (real retrieval, logged tool
  calls and EM rewards, LoRA checkpoint, exported PEFT adapter).
- GRPO and PPO each trained for 150 steps (64 prompts x 5 rollouts) with
  identical LoRA settings; logs, metrics, adapters and Bamboogle predictions
  are in `results/runs/` and `results/bamboogle/`.
- The tiny-zero-lora starter ran for a reduced 60 steps on its own pinned
  stack (`results/tiny_zero_starter/`); it is reported separately.
- Earlier Modal work (A10 smoke, spend-limit rejection) is unchanged in
  `results/modal_gpu_smoke.json`; no further Modal compute was used.
- Individual submission by **Lufei Wang**.

## Sources

- Class starter: [`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora)
- Search-R1 code and released data/models: [`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1)
- Search-R1 paper: [arXiv:2503.09516v5](https://arxiv.org/abs/2503.09516)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- RL's Razor: [arXiv:2509.04259](https://arxiv.org/abs/2509.04259)
- LoRA empirical guidance: [Thinking Machines — LoRA](https://thinkingmachines.ai/blog/lora/)
- Reference scores transcribed from the paper's Tables 2 and 3; the Bamboogle data came from
  the course sheet PDF above (also linked in Canvas).

## Reproduce the inference baselines

Use a CUDA machine with the model and `transformers`, `torch`, `peft`, and
`requests` installed. The evaluator writes one JSONL record per question and a
summary JSON. `--limit 2` is a quick smoke run; use all 125 questions for a
reportable score.

```bash
cd week_08/search_r1
python3 evaluate_bamboogle.py \
  --model Qwen/Qwen2.5-3B-Instruct --mode direct \
  --output results/qwen25_3b_direct.jsonl
```

For Search-R1 inference, start the Search-R1 `/retrieve` service from its
[official repository](https://github.com/PeterGriffinJin/Search-R1) and point
the evaluator at it. The repo's documented local retrieval setup uses the
released Wikipedia index and corpus; the default endpoint below is the local
server's default.

```bash
python3 evaluate_bamboogle.py \
  --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo \
  --mode search --retrieval-url http://127.0.0.1:8000/retrieve \
  --output results/search_r1_ppo_reference.jsonl
python3 evaluate_bamboogle.py \
  --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --mode search --retrieval-url http://127.0.0.1:8000/retrieve \
  --output results/search_r1_grpo_reference.jsonl
```

These two Hugging Face checkpoints are published paper models, so their
measurements validate the evaluation pipeline; they are not our own trained
adapters. For every trained adapter, evaluate the same base model and set
`--adapter PATH_TO_ADAPTER`. To make the comparison CSV and chart after runs:

```bash
python3 compare_results.py \
  --run 'Qwen direct=results/qwen25_3b_direct.summary.json' \
  --run 'Search-R1 PPO checkpoint=results/search_r1_ppo_reference.summary.json' \
  --run 'Search-R1 GRPO checkpoint=results/search_r1_grpo_reference.summary.json'
```

The output is `results/comparison.csv` plus `results/comparison.png`; the chart
visually distinguishes paper references from measured runs. `--limit` is only
for smoke checks and must not be used for final benchmark numbers.

## Training implementation status

The class starter is a TRL/PEFT LoRA GRPO example for single-turn Countdown; it
is useful for the adapter setup and metric logging, but it does not implement
Search-R1's multi-turn retrieval rollout or PPO. The public Search-R1 training
scripts implement multi-turn search and both PPO/GRPO, but the inspected v0.1
scripts use full fine-tuning. Therefore those scripts do not satisfy a PEFT
requirement unchanged.

This repository now includes a **VeRL-based training prototype**:
`search_tool.py` connects tool-agent rollouts to Search-R1 `/retrieve`,
`prepare_search_dataset.py` formats the official NQ + HotpotQA train/test
parquets, `rewards.py` implements outcome-only exact match, and
`train_verl.sh grpo|ppo` selects GRPO (`grpo`) or PPO with GAE (`gae`). The
policy actor uses the tiny-zero-lora rank 16 / alpha 32 / target-module
pattern. The pinned VeRL FSDP PEFT interface defaults LoRA dropout to 0.0
(tiny-zero-lora uses 0.05), and VeRL's PPO value critic is separate from this
policy adapter; report these implementation differences. The VeRL revision
is pinned in `verl.pin`. Its tool agent marks tool observations with response
mask 0, excluding retrieved tokens from policy loss. See [`RUNBOOK.md`](RUNBOOK.md)
for setup and launch commands.

The Modal GPU sanity smoke passed: Qwen2.5-3B-Instruct loaded on an NVIDIA A10,
and one synthetic-text LoRA forward/backward/update succeeded. The record is
in [`results/modal_gpu_smoke.json`](results/modal_gpu_smoke.json). This did not
exercise VeRL, retrieval, Search-R1 rollouts, PPO/GRPO advantages, or a saved
adapter; the VeRL integration smokes and full runs were later done on GB300
(see "Current run status").

The supplied Modal screenshot showed 96% of a $1,200 workspace budget used.
After the smoke, the workspace billing summary showed $1,450.34 billed this
month. A subsequent VeRL job launch was rejected with `Workspace has exceeded
its spend limit`; no VeRL GPU job or container was created. The post-run
active-container check was empty. Billing is shared and changed while other
students were using the workspace, so the charge cannot be attributed to this
smoke alone. No further Modal compute was used; all PPO/GRPO training and
Bamboogle scores come from the GB300 node.
