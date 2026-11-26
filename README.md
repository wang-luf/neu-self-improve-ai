# neu-self-improve-ai

Course experiments are organized by week:

- `week_01/` — SPADE-inspired power-grid environment design loop.
- `week_02/` — policy iteration for a minimal power-grid MDP.
- `week_05/` — [Week 5 assignment notebook](week_05/policy_gradient_comparison.ipynb): locally run Jupyter notebook with all code, saved outputs, and five comparison charts for REINFORCE, REINFORCE with baseline, and one-step Actor–Critic.
- `week_07/` — PPO implementation and experiment; see [week_07/README.md](week_07/README.md).

## Week 9 Final Project Part 1 — AgentFlow on Bamboogle (`week_09`)

The repository includes the AgentFlow implementation, 1-step training smoke,
25-step Flow-GRPO LoRA run, all 125-question Bamboogle outputs, the adapter,
logs, and a run report. See [the Week 9 report](week_09/results/RUN_REPORT.md)
and [source provenance](week_09/SOURCE_PROVENANCE.md).

| System | Local judge | Rule EM | Paper |
| --- | ---: | ---: | ---: |
| Qwen-2.5-7B-Instruct, direct | 9.6% | 6.4% | 12.0% |
| AgentFlow, base planner | 28.8% | 14.4% | 58.4% |
| AgentFlow, Flow-GRPO LoRA (25 steps) | 29.6% | 12.8% | 69.6% |

The 25-step run did not show a measurable Flow-GRPO gain over the base
planner; the report explains the reward trend and deviations from the paper.
