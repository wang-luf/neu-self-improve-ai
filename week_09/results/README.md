# Week 9 Part 1 results

Measured on one GB300 node; see [`RUN_REPORT.md`](RUN_REPORT.md) for the full
report (setup, reward trend, runtime, deviations, limitations).

| System | Judge accuracy | Rule EM | Paper |
| --- | ---: | ---: | ---: |
| Qwen-2.5-7B-Instruct, direct | 9.6% (12/125) | 6.4% | 12.0% |
| AgentFlow, base planner | 28.8% (36/125) | 14.4% | 58.4% |
| AgentFlow, Flow-GRPO planner (LoRA, 25 steps) | 29.6% (37/125) | 12.8% | 69.6% |

| Path | Contents |
| --- | --- |
| `bamboogle/Qwen-direct/` | per-question outputs and `final_scores_direct_output.json` (no tools) |
| `bamboogle/AgentFlow-base/` | per-question AgentFlow trajectories and scores, base planner |
| `bamboogle/AgentFlow-flowgrpo-step25/` | per-question AgentFlow trajectories and scores, trained planner |
| `agentflow_base_vs_flowgrpo.txt` | per-question comparison of the two AgentFlow runs |
| `flowgrpo-step25/lora_adapter/` | final planner LoRA adapter (PEFT, r=16, alpha=32, 77 MB), served with `--lora-modules week9-planner=...` |
| `training_metrics.csv`, `reward_curve.png` | per-step Flow-GRPO metrics and the reward curve |
| `training_rollouts_by_source.csv` | training reward per step split into NQ and DeepMath rollouts |
| `configs/` | the smoke and assignment configs, and the fully resolved Hydra configs of both training segments |
| `logs/` | smoke, compact training, export, scoring and evaluation logs |
| `environment.txt` | GPU, driver, key package versions and `pip freeze` |

Scripts that produced the tables and plots are in [`../scripts/`](../scripts/).
