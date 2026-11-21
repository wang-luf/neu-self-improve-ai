# Search-R1 Replication Report

## Objective

This project reproduces the Week 8 Search-R1 experiment on the course-provided
125-question Bamboogle set. The required comparison is direct inference and
Search-R1-style inference using Qwen2.5-3B-Instruct, with both GRPO and PPO;
the assignment also requires PEFT. The CSV contains question and accepted
answer aliases only. It is held out from policy training.

## What the readings contribute

### Policy gradients, baselines, and actor-critic

For a stochastic policy \(\pi_\theta(a\mid s)\), the objective is expected
return \(J(\theta)\). The policy-gradient theorem writes its gradient as an
expectation of score-function terms:

\[
\nabla_\theta J(\theta) =
  \mathbb{E}_{s,a\sim\pi_\theta}
  [\nabla_\theta\log\pi_\theta(a\mid s) Q^{\pi_\theta}(s,a)].
\]

REINFORCE estimates the return from sampled episodes. Subtracting a state-only
baseline leaves the expected gradient unchanged while reducing variance.
Actor-critic methods learn a critic to estimate the baseline/advantage, so the
actor can update from shorter bootstrapped samples. This is the connection
between the Week 5 material and the policy optimization used for language
model actions.

### Monotonic improvement and PPO

The Week 6 policy-improvement results motivate controlling how far an update
moves the policy: a large policy change can invalidate the old policy's value
estimates. TRPO expresses this as a constrained update (maximize a surrogate
objective subject to a divergence constraint). PPO's clipped surrogate is a
practical approximation:

\[
L^{clip}(\theta)=\mathbb{E}_t\left[
\min(r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)
\right],\quad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
\]

PPO also uses a learned value critic (and commonly an entropy term). GRPO
instead normalizes rewards among multiple sampled responses for the same
prompt and does not require a separate value model. Keep prompt groups and
rollout counts fixed when comparing the algorithms.

### Search-R1's environment and reward

Search-R1 turns retrieval into an action in a multi-turn environment. A rollout
alternates model tokens and tool observations: the model reasons in `<think>`,
emits `<search>query</search>`, receives passages in
`<information>...</information>`, and eventually emits
`<answer>...</answer>`. The paper uses an outcome exact-match reward and masks
retrieved passage tokens from policy/KL loss, because those tokens are
environment observations rather than sampled policy actions. This project
therefore evaluates the exact answer block and logs search-call counts.

The paper's Bamboogle references for the same 125-question course sheet are:

| Method | EM | Correct |
| --- | ---: | ---: |
| Direct inference | 0.024 | 3/125 |
| Search-R1-Instruct (Table 2) | 0.264 | 33/125 |
| Search-R1 GRPO (Table 3) | 0.232 | 29/125 |
| Search-R1 PPO (Table 3) | 0.264 | 33/125 |

These are paper reference scores, not local measurements. The reported values
are kept separate in [`results/paper_reference.csv`](results/paper_reference.csv).

### PEFT / LoRA and why the starter is only a starting point

LoRA freezes pretrained \(W_0\) and learns a low-rank update
\(\Delta W=BA\), so a layer computes \(W_0x+BAx\). This reduces trainable
parameters and checkpoint size. The Week 8 class starter,
[`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora), is a
useful concrete example: it adapts Qwen2.5-3B-Instruct with PEFT LoRA and runs
TRL GRPO on a single-turn Countdown task. Its reward functions, dataset, and
training loop do not implement multi-turn search or PPO. The implementation
must borrow the adapter configuration pattern from the starter and the search
environment/trajectory logic from Search-R1; treating the Countdown notebook
as a Search-R1 trainer would be incorrect.

The paper *RL's Razor* argues that on-policy RL often changes a base policy
less (in KL) than supervised fine-tuning at comparable new-task performance,
which can help preserve prior behavior. This is useful motivation for tracking
KL and avoiding claims based only on task reward; it does not replace the
Search-R1 experiments or guarantee that a particular small run will avoid
forgetting.

## Experimental protocol

1. Evaluate `Qwen/Qwen2.5-3B-Instruct` in direct mode on all 125 rows.
2. Run Search-R1 tool inference with the published Qwen2.5-3B-Instruct GRPO
   and PPO checkpoints as reference-pipeline checks.
3. Train one GRPO and one PPO policy from the same Qwen2.5-3B-Instruct base,
   using the Search-R1 NQ + HotpotQA training data and a PEFT LoRA adapter.
   Bamboogle stays out of training and tuning.
4. Hold retrieval corpus/top-k, prompt, generation settings, reward and answer
   normalization, LoRA rank/alpha/target modules, and training budget constant
   across PPO and GRPO. Record the actual config, seed, step count, trainable
   parameter count, reward, KL, checkpoint, GPU/compute allocation, and
   validation/evaluation results.
5. Evaluate both trained adapters over all Bamboogle questions, save every
   completion and correctness flag, then create `comparison.csv` and
   `comparison.png` with `compare_results.py`.

Run commands for inference and chart generation are in [`README.md`](README.md).
The published Search-R1 v0.1 PPO/GRPO scripts provide a reference for the
multi-turn training design, but the inspected scripts full-fine-tune the
actor. They cannot be used unchanged to satisfy PEFT. The course starter
supports GRPO with LoRA but is single-turn. This repository now includes a
pinned VeRL-based integration prototype for both PPO and GRPO, with a search
function tool, outcome exact-match reward, and actor LoRA. VeRL's agent loop
sets the response mask to zero for tool observations. Its FSDP PEFT config
uses the library default LoRA dropout of 0.0 rather than the starter's 0.05;
the PPO value critic remains a separate VeRL model. See the runbook for the
implementation details and smoke-run sequence.

## Results

All numbers below were measured on the GB300 node on 2026-11-20 and are
backed by files in `results/`. Bamboogle scores use all 125 questions, greedy
decoding, wiki-18 E5 Flat retrieval with top-3 passages, up to 4 turns of at
most 512 new tokens. `results/RESULTS.md` is the generated table;
`results/comparison.{csv,png}` and `results/training_curves.png` are the charts.

### Compute and setup actually used

- Hardware: one node, 4x NVIDIA GB300 (284 GB each, aarch64 Grace host),
  reserved 2026-11-20 14:45–22:45 PST. GRPO trained on GPU 0 and PPO on GPU 1
  **concurrently**, with identical settings; GPU 2/3 ran evaluation and the
  retriever. No Modal compute was used.
- Software: VeRL `453954b3` (+ the one-line critic patch in `verl_patches/`),
  sglang 0.5.20 rollouts, FSDP actor, torch 2.13 / CUDA 13.0.
- Policy: Qwen2.5-3B-Instruct + LoRA r=16, alpha=32, dropout 0.0 (VeRL
  default; the starter uses 0.05), targets q/k/v/o/gate/up/down projections,
  learning rate 2e-5 (the starter's), KL loss 0.001 (low-variance KL), seed 42.
  PPO adds a full-parameter value critic initialised from the same base model
  (lr 1e-5, GAE with gamma=lambda=1).
- Data: 169,615 NQ + HotpotQA training rows; validation every 25 steps on a
  fixed 512-row sample of the 11,015 in-domain validation rows. Bamboogle was
  never used for training, tuning or checkpoint selection.
- Rollouts: 64 prompts x 5 samples per step (320 trajectories), PPO mini batch
  32 prompts, max 4 assistant turns, 2,048 response tokens, tool responses
  truncated to 2,048 characters. Reward = exact match of the last
  `<answer>` block only. Retrieved text is masked out of the policy loss.
- Budget: both runs were stopped at **step 150** (planned, see
  `results/runs/*/planned_stop.json`): 9,600 prompts / 48,000 trajectories per
  algorithm, about 5.7% of one epoch. The paper used ~1,000 steps x 512 prompts
  with full fine-tuning, so these are short, compute-limited runs.

### Bamboogle exact match (125 questions)

| Method | Protocol | EM | Correct | Mean searches | Paper |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct inference, Qwen2.5-3B-Instruct | direct | 0.032 | 4/125 | 0 | 0.024 |
| Qwen2.5-3B-Instruct + search tool, untrained | tool | 0.184 | 23/125 | 1.55 | — |
| Published Search-R1 PPO checkpoint (full FT) | search | 0.240 | 30/125 | 1.97 | 0.264 |
| Published Search-R1 GRPO checkpoint (full FT) | search | 0.344 | 43/125 | 1.92 | 0.232 |
| **Ours: LoRA GRPO, step 150** | tool | 0.312 | 39/125 | 2.23 | (0.232) |
| **Ours: LoRA PPO, step 150** | tool | 0.208 | 26/125 | 2.10 | (0.264) |

Intermediate checkpoints (same evaluator):

| Step | 0 (untrained) | 25 | 50 | 100 | 150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA GRPO | 0.184 | 0.248 | 0.264 | 0.280 | 0.312 |
| LoRA PPO | 0.184 | 0.176 | 0.104 | 0.168 | 0.208 |

In-domain validation EM (512 NQ/HotpotQA questions, greedy):

| Step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA GRPO | 0.184 | 0.295 | 0.330 | 0.311 | 0.354 | 0.365 | 0.342 |
| LoRA PPO | 0.180 | 0.279 | 0.283 | 0.299 | 0.303 | 0.326 | 0.318 |

### Findings

1. **Search matters.** Direct answering gets 4/125; giving the same untrained
   model the search tool already gives 23/125, and every search-trained model
   is higher still. This matches the paper's direct-inference reference
   (0.024 vs our 0.032).
2. **The evaluation pipeline reproduces the published models.** The released
   Search-R1 PPO checkpoint scores 0.240 (paper 0.264, 3 questions fewer); the
   released GRPO checkpoint scores 0.344, above the paper's 0.232. We use greedy
   decoding and exact top-3 retrieval over the same corpus/index; the paper's
   exact decoding settings for this table are not stated, so the GRPO gap is
   reported as observed rather than explained away.
3. **LoRA GRPO improved steadily** both in-domain (0.18 -> 0.34–0.37) and on
   out-of-domain Bamboogle (0.184 -> 0.312), already above the paper's
   full-FT GRPO number after only 150 small steps — but under a different tool
   protocol (see limitations), so this is not a like-for-like comparison with
   the paper.
4. **LoRA PPO improved in-domain, but only reached Bamboogle late.** Its
   in-domain validation EM rose from 0.180 to 0.318 (GRPO: 0.342 at step 150),
   but its Bamboogle EM first fell below the untrained model (0.104 at step 50)
   and only ended slightly above it (0.208 at step 150, vs GRPO's 0.312). The
   trajectories explain why: around step 50 PPO collapsed to answering after a
   single search (105/125 Bamboogle questions used one search, training
   responses ~600 tokens). One search is often enough for single-hop NQ, but
   not for Bamboogle's two-hop questions. GRPO instead moved to 2–3 searches
   (118/125 questions with >= 2 searches at step 100). PPO's KL to the
   reference spiked to ~0.25 during that phase (early in training, with a
   freshly initialised value critic); it later relaxed, and by step 150 PPO
   was back to multi-search behaviour (96/125 questions with >= 2 searches).
   A plausible reason GRPO is more sample-efficient in a 150-step budget is
   that its group-relative baseline needs no critic warm-up. The paper reports
   PPO as more stable than GRPO over ~1,000 steps; our runs are far too short
   to test that claim. PPO also left more answers unfinished at step 150 (23/125
   completions without an `<answer>` block, vs 12/125 for GRPO).
5. **PEFT is enough to learn the behaviour.** A 30M-parameter LoRA adapter
   (~1% of the 3B weights; 60 MB `adapter_model.safetensors`) changed search
   behaviour and accuracy substantially; KL to the base stayed below ~0.25.

### Class starter reproduction (tiny-zero-lora, separate from Search-R1)

The starter at commit `3659acfa` was run unchanged except for
`max_steps=2000 -> 60` and the output directory
(`results/tiny_zero_starter/reduced_config.diff`), with its pinned
requirements (TRL 1.17.1, vLLM 0.32.0, PEFT 0.21.2) on one GB300: Countdown
GRPO, LoRA r=16/alpha=32/dropout 0.05, 16 generations. It finished 60 steps in
6.6 minutes and saved the LoRA adapter and `checkpoint-60`. Held-out format
rate rose from 32.5% to 96.0% and solve rate from 4.0% to 7.5%; this run only
demonstrates that the starter works and is **not** a Search-R1 result.

### Limitations

- **Short runs.** 150 steps x 64 prompts per algorithm, one seed. Differences
  of a few Bamboogle questions are within noise (1 question = 0.008 EM; a 95%
  binomial interval at EM 0.28 is roughly +/-0.08).
- **Different tool protocol from the paper.** Our policies use Qwen's native
  function-calling format (`<tool_call>` / `<tool_response>`) through VeRL's
  tool agent, with our own system prompt; the published checkpoints use the
  paper's `<search>` / `<information>` tags. Each model is evaluated in the
  protocol it was trained with, so the rows are comparable as end-to-end
  systems, not as identical prompts.
- **Concurrent rather than sequential runs.** GRPO and PPO shared the node,
  the retriever and the CPU at the same time; each had its own GPU. This was a
  deliberate change from the handover plan to fit both runs into the
  reservation.
- **LoRA dropout 0.0** (VeRL's FSDP PEFT default) instead of the starter's 0.05.
- **Full-parameter critic for PPO.** Only the policy is PEFT; the value model
  is a separate 3B critic, so PPO used more trainable parameters and memory.
- The retriever runs the official index with exact GPU inner-product search;
  it was verified against Search-R1's CPU FAISS path
  (`results/retriever_check.json`), but it is our server, not theirs.

## Team

**Lufei Wang** (individual submission).

## References

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Ch. 13.
- Week 5: Policy Gradient | Notion (course PDF export).
- Week 6: Monotonic Improvement | Notion (course PDF export).
- Week 7: Proximal Policy Optimization (course PDF export).
- Week 8: RL-Train LLMs | Notion (course PDF export).
- Jin et al., [Search-R1, arXiv:2503.09516](https://arxiv.org/abs/2503.09516).
- Hu et al., [LoRA, arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Shenfeld et al., [RL's Razor, arXiv:2509.04259](https://arxiv.org/abs/2509.04259).
- [Thinking Machines: LoRA](https://thinkingmachines.ai/blog/lora/).
- [Search-R1 official code](https://github.com/PeterGriffinJin/Search-R1).
