# Assignment 5 execution runbook

This runbook separates the three code sources. Do not treat running the
Countdown starter as Search-R1 results, and do not invoke another student's
ephemeral Modal app without their agreement.

## Where the code lives

1. **Class starter (LoRA + GRPO pattern):** [`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora)
   - `train.py`: TRL `GRPOTrainer`, Qwen2.5-3B-Instruct, PEFT LoRA, Countdown
     data and arithmetic rewards.
   - `modal_train.py`: Modal wrapper for that Countdown trainer.
   - This is the code the Canvas prompt asks you to understand and use as a
     starting point. It has no search environment and no PPO implementation.
   - Inspected source commit: `3659acfad61dab2f90fa46294566ff6808452e87`.
     The actual entry point uses `Jiayi-Pan/Countdown-Tasks-3to4`, a Qwen
     chat prompt, format and arithmetic rewards, TRL `GRPOTrainer`, and PEFT
     LoRA (rank 16, alpha 32, dropout 0.05; q/k/v/o and gate/up/down
     projections). Its default run is 2,000 steps with 16 generations and
     batch size 16. `modal_train.py` wraps this same `train.py` on Modal.
     This source inspection is documented; an actual training run is still
     required as reproduction evidence.
2. **Search-R1 (search rollout, reward, datasets and paper scripts):**
   [`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1)
   - `scripts/nq_hotpotqa/v0.1/train_grpo.sh`
   - `scripts/nq_hotpotqa/v0.1/train_ppo.sh`
   - `scripts/nq_hotpotqa/data_process.sh`, `retrieval_launch.sh`, and
     `infer.py`
   - The v0.1 scripts are useful references for the full search training
     pipeline, but they predate the course's PEFT requirement and use
     full-model training. Do not submit a full-fine-tune run as PEFT.
3. **Current PEFT/multi-turn integration reference:**
   - [VeRL PPO LoRA guide](https://github.com/verl-project/verl/blob/main/docs/advance/ppo_lora.rst)
   - [VeRL multi-turn search-tool guide](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
   - VeRL documents LoRA for its PPO trainer and multi-turn search rollouts.
     Its old in-tree `SearchTool` example has been removed, so the project must
     supply a compatible search tool implementation and pin a compatible VeRL
     revision. Do not combine arbitrary `main` branches and expect the old
     Search-R1 shell scripts to work unchanged.

## Steps

### 1. Reproduce the starter locally

Clone the starter into a separate scratch directory and inspect `train.py`,
`train_callback.py`, `requirements.txt`, and `modal_train.py`. Record the base
model, dataset, reward functions, LoRA rank/alpha/target modules, optimizer,
rollout count, and logging. Run a tiny Countdown smoke run only; do not use its
reward or checkpoint as a Search-R1 result.

```bash
git clone https://github.com/suhabe/tiny-zero-lora.git
cd tiny-zero-lora
```

On a GPU environment with its pinned dependencies, `python train.py` runs the
starter. `modal run modal_train.py` is its Modal wrapper. Both commands are
only for the starter exercise. Avoid the full default run when checking the
setup; temporarily reduce its dataset and steps in a local copy.

### 2. Verify Search-R1 inference and retrieval

Use the course Qwen2.5-3B-Instruct model and the supplied Bamboogle set. First
run a one-question direct inference check. Then start a compatible `/retrieve`
service and run a one-question Search-R1 check. The repo's
[`evaluate_bamboogle.py`](evaluate_bamboogle.py) writes predictions and
exact-match records; use `--limit 1` only for smoke checking. The full reported
evaluation must use all 125 questions.

The Bamboogle questions are evaluation-only. The Search-R1 training source is
the paper's released NQ + HotpotQA data, not the course Bamboogle sheet.

### 3. Prepare the Search-R1 + PEFT trainer

This repo now pins VeRL in `verl.pin` and contains `setup_verl.sh`,
`prepare_search_dataset.py`, `search_tool.py`, `rewards.py`, and
`train_verl.sh`. The launcher selects `algorithm.adv_estimator=grpo` for GRPO
or `gae` for PPO, uses VeRL's tool agent for retrieval, and applies LoRA rank
16 / alpha 32 / tiny-zero-lora target modules to the policy actor. The pinned
FSDP PEFT interface defaults dropout to 0.0 (the starter uses 0.05); PPO's
value critic is a separate VeRL model, not this policy adapter.
VeRL's tool agent encodes retrieved tool responses with response mask 0,
excluding observations from policy loss.

The remote Modal function uses the course NQ/HotpotQA and wiki-18 volumes and
writes checkpoints and trainer logs to the assignment's own results volume.
The first run is the required one-step integration smoke:

```bash
cd week_08/search_r1
modal run modal_train_verl.py --algorithm grpo --steps 1 --batch-size 2 --rollouts 2
```

The smoke prepares two real NQ/HotpotQA prompts, loads the E5 wiki-18 index,
probes a real retrieval result, then requires logged tool calls, logged custom
rewards, and a saved LoRA-only FSDP checkpoint before reporting success. The
Modal function terminates its retriever in `finally` and its container stops
when the function exits. Inspect the result and container list before training.
If the smoke passes, run GRPO and wait for the invocation to finish:

```bash
modal run modal_train_verl.py --algorithm grpo --steps 500 --batch-size 128 --rollouts 5
```

Then confirm the invocation ended and its Modal container is stopped. Run PPO
as a separate invocation with the same settings:

```bash
modal run modal_train_verl.py --algorithm ppo --steps 500 --batch-size 128 --rollouts 5
```

Each checkpoint is configured to save only LoRA weights; VeRL stores the
FSDP shards and `lora_train_meta.json` under the checkpoint's `actor` folder.
Convert/export that checkpoint to a PEFT-loadable adapter before Bamboogle
inference, and keep the export alongside the original checkpoint and logs.

The prototype is not GPU-validated. Before longer runs, verify tool-call
parsing with Qwen2.5-3B-Instruct, retrieval response shape, policy adapter
checkpoint export, PPO value checkpoint export, and actual step count. Save exact configs, image
definition, dataset revision, retrieval endpoint, seed, and output volume path.
An existing dashboard function named `train` is not enough to establish that
it uses your code or which algorithm it runs.

### 4. Run local smoke checks before Modal

Follow the class reminder: exercise the data loader, reward/answer extraction,
retriever request, and one short rollout locally or on a permitted local GPU.
Use a tiny batch and a single optimizer step. Confirm the GRPO reward grouping
and PPO value/advantage path separately. These checks are not benchmark
results.

### 5. Run the two algorithms one at a time

Before submitting a Modal job, check the Workspace budget and the Containers
page. The shared workspace is near its displayed limit, and the class reminder
asks students not to leave jobs running. Use an app built from your own source
or your group's source; the dashboard app `w8-searchr1-7k3q` shown in the
provided screenshot was created by another student, so confirm with its owner
before invoking it.

Run GRPO first. Wait for completion, confirm its container is stopped, save
its logs and LoRA adapter, and record its actual step count. Then run PPO with
the same base model, training split, LoRA config, search corpus, prompt,
retrieval top-k, answer reward, and comparable rollout/compute budget. Do not
run both jobs concurrently. The shared chat does not establish an instructor
approved minimum step count; report any compute-limited budget honestly.

### 6. Evaluate and submit

Run Direct Inference and each resulting adapter across all 125 Bamboogle
questions with the same evaluator settings. Save JSONL predictions, summary
JSON, training logs/curves, comparison CSV, and chart. Keep paper reference
scores clearly separate from measured results. Add every group member's full
name; every student submits their own Canvas entry and repository URL.

## GB300 local execution (what was actually run)

The Modal workspace stayed over its spend limit, so the replication ran on a
reserved 8-hour single node (`reserved-gb300-node`: 4x NVIDIA GB300, 284 GB
each, aarch64 Grace, driver 595.58 / CUDA 13.2, 958 GB RAM). Nothing ran on
Modal. GPU layout: GPU 0 GRPO, GPU 1 PPO, GPU 2/3 evaluation, GPU 3 also hosts
the retriever.

1. **Environment.** `setup_verl.sh` checkout at `verl.pin`, then
   `uv sync --frozen --all-packages --extra sglang --extra fsdp` (torch 2.13
   cu130, sglang 0.5.20, transformers 5.16.1, peft 0.19.1, flash-attn 2.8.3
   aarch64 wheelhouse build) plus `faiss-cpu==1.12.0`, `matplotlib`. Apply
   `verl_patches/0001-critic-ignore-ref-no-lora-flag.patch` (see below).
2. **Assets** from Hugging Face: `PeterJinGo/nq_hotpotqa_train`,
   `PeterJinGo/wiki-18-corpus` (the `.jsonl.gz` is really a tar.gz: extract
   `wiki_dump.jsonl`, 21,015,324 lines), `PeterJinGo/wiki-18-e5-index`
   (`cat part_aa part_ab > e5_Flat.index`), `intfloat/e5-base-v2`,
   `Qwen/Qwen2.5-3B-Instruct`, and the two published Search-R1 3B-it checkpoints.
3. **Data.** `python prepare_search_dataset.py --source-dir <nq_hotpotqa_train> --output-dir <processed>`
   writes train (169,615), test (11,015), a fixed 512-row validation sample
   (seed 42), and the 2/1-row smoke files; it fails if any Bamboogle question
   appears in either split.
4. **Retriever.** `gpu_retrieval_server.py` serves Search-R1's `/retrieve`
   schema. It loads the official E5 Flat index, keeps the float32 vectors on one
   GPU and runs exact inner-product top-k (identical to `IndexFlatIP`), with
   Search-R1's E5 query encoding. `verify_retriever.py` compared it with
   Search-R1's own encoder + CPU FAISS: identical top-3 ids for all probe
   queries, max score difference 5e-7 (`results/retriever_check.json`); CPU
   FAISS needs ~1.7 s per query here, the GPU service ~0.03 s.
5. **Smokes.** `local_train_verl.py --algorithm {grpo,ppo} --steps 1 --batch-size 2 --rollouts 2`
   both passed: real tool calls, logged EM rewards, LoRA checkpoint with
   `lora_train_meta.json`, exported PEFT adapter (`results/runs/*-smoke/run_summary.json`).
6. **Training.** GRPO and PPO ran concurrently on separate GPUs with identical
   settings (the handover asked for sequential runs to control Modal spend; on
   a fixed-length reservation concurrency doubles each run's budget without
   changing the comparison):

   ```bash
   python local_train_verl.py --algorithm grpo --steps 300 --batch-size 64 --mini-batch-size 32 \
     --micro-batch-size 16 --rollouts 5 --save-freq 25 --test-freq 25 --run-name grpo-main \
     --data-dir <processed> --gpus 0
   # same command with --algorithm ppo --run-name ppo-main --gpus 1
   ```
7. **Evaluation.** `eval_checkpoint.sh RUN STEP GPU` exports a checkpoint with
   `verl.model_merger` and scores the adapter on all 125 questions in 5 shards
   (`merge_eval_shards.py`); `watch_and_eval.sh` does this every 50 steps.
   Baselines use `evaluate_bamboogle.py --mode direct|tool|search`.
8. **Reports.** `summarize_results.py` -> `results/RESULTS.md`;
   `plot_training.py` -> `results/training_curves.png`;
   `compare_results.py` -> `results/comparison.{csv,png}`.

### Fixes made while porting

- `actor_rollout_ref.actor.checkpoint.save_lora_only` does not exist at the
  pinned VeRL revision; PEFT checkpoints are recognised automatically.
- VeRL's `multi_turn.max_tool_response_length` counts **characters**. The old
  value 512 cut three passages (~1,800 chars) down to their first and last 256
  characters; it is now 2048 (~500 tokens, Search-R1's `max_obs_length`).
- Pinned-VeRL bug: `_compute_ref_log_prob` leaves `no_lora_adapter=True` in the
  shared batch metadata, so PPO's (non-LoRA) critic crashed calling
  `disable_adapter()`. The patch resets the flag for value inference only; the
  actor/old-log-prob path was not affected (GRPO is unchanged by it).
- `evaluate_bamboogle.py`: `--mode search` now wraps the Search-R1 prompt in the
  chat template and formats observations exactly like Search-R1's `infer.py`;
  stopping at `</search>` matches decoded text (token-id matching missed
  merged tokens and let the model hallucinate its own `<information>`).
  `--mode tool` now uses the training system prompt, VeRL's generated tool
  schema, `search_tool.search` and VeRL's truncation. `--mode direct` asks for
  an `<answer>` block so the same answer extraction applies. All modes default
  to greedy decoding.

## Current checkpoint

- Completed: source/data review, Bamboogle PDF/CSV cross-check, syntax check,
  answer-metric smoke check, and CSV integrity check.
- Completed: VeRL-based Search-R1 training prototype, pinned source revision,
  separate GRPO/PPO selector, retrieval tool, data conversion, answer-only
  reward, local syntax/logic smoke checks, and a Modal Qwen2.5-3B LoRA GPU
  update smoke. See `results/modal_gpu_smoke.json`.
- Prepared `modal_train_verl.py` to run the one-step GRPO smoke and sequential
  GRPO/PPO jobs using the existing course data/index volumes. Modal rejected
  its earlier launch before creating an app: `Workspace has exceeded its spend limit`.
- Prepared real VeRL parquet files from the course `week08-search-r1-data`
  volume under `/tmp/searchr1-processed/`: 169,615 training rows (79,168 NQ,
  90,447 HotpotQA) and 11,015 in-domain validation rows (3,610 NQ, 7,405
  HotpotQA). The source test parquet also contains 125 Bamboogle rows; these
  and all other out-of-domain sources are excluded from validation to prevent
  benchmark leakage. Conversion and Parquet round-trip checks passed.
- Completed on GB300 (2026-11-20): VeRL tool-rollout smokes for GRPO and PPO,
  retrieval inference baselines, 150-step GRPO and PPO LoRA training, adapter
  exports, Bamboogle evaluation every 50 steps, result tables and plots, and
  the tiny-zero-lora starter run. See the GB300 section above and
  `REPLICATION_REPORT.md`.
- The post-run Modal billing summary showed $1,450.34 billed this month while
  the supplied banner showed a $1,200 budget. Billing is shared; the post-run
  active-container check showed no active containers. The API's spend-limit
  rejection confirms that an Owner/Manager must reset or raise the effective
  limit before another run can start.
