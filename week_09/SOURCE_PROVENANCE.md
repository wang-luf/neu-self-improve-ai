# Source provenance

The Week 9 implementation, experiment outputs, logs, evaluation results, and
LoRA adapter in this directory were transferred from the source repository
snapshot below. The snapshot's Git commit records its original contributor and
co-author; this provenance is retained so the origin of the code and measured
results remains clear.

- Source repository: https://github.com/jiacao-amd/neu-self-improve-ai
- Source commit: `4f1e2f758bcffc163fb6267faf046f26e579e471`
- Source commit title: “Add Week 9 Part 1: AgentFlow on Bamboogle with a LoRA Flow-GRPO planner”
- Upstream AgentFlow baseline: https://github.com/lupantech/AgentFlow
- Upstream AgentFlow commit: `b94006436b8712ab8682846fb0d886a5f174f2d4`

The run report distinguishes measured results from paper values and documents
its local Wikipedia retriever, local Qwen judge, reduced 25-step budget, and
lack of a demonstrated Flow-GRPO gain. Machine-specific absolute workspace
paths were normalized to `<training-workspace>`. Full raw console captures were
omitted; compact training logs, reward traces, scoring logs, and evaluation logs
remain in `results/logs/`.
