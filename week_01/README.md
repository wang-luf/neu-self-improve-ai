# Week 1 — SPADE Environment Designer for Power Grid Operations

## Assignment goal

This project uses **Grid2Op**, an open-source environment library for sequential
power-grid operation, as the real-world domain. It implements and runs a small
SPADE-inspired Environment Designer loop. It generates increasingly complex
Grid2Op scenarios, executes them for validation, records feedback in memory,
and saves results. **It does not train a policy.**

## Why Grid2Op?

The course itself lists Grid2Op as an example. Its `l2rpn_case14_sandbox` case
is small enough to inspect while retaining real grid concepts: 14 substations,
20 lines, 6 generators, and 11 loads. Grid2Op exposes the `reset()` / `step()`
interaction expected by an RL environment. The supplied repository page
showed commit `a173688`, a release-fix merge with package and workflow changes
about three months before the assignment. The [commit page](https://github.com/Grid2op/grid2op/commit/a173688)
makes this activity evidence directly checkable.

Sources: [Grid2Op repository](https://github.com/Grid2op/grid2op),
[Grid2Op environment documentation](https://grid2op.readthedocs.io/en/latest/),
[SPADE paper](https://arxiv.org/abs/2608.19197).

## Grid2Op model studied

Grid2Op advances a grid episode through `reset(seed=...)` and
`step(action)`. At each step, time-series data update demand and generation,
the backend computes power flow, and the environment returns an observation,
reward, termination flag, and diagnostic information. The observation includes
line loading `rho` (flow divided by thermal limit); values above 1 indicate a
line overload. Actions come from `env.action_space()` and can include line
status changes such as `{"set_line_status": [(0, -1)]}`. An overload can trip a
line and lead to cascading game over, so an outage specification must be run
through the simulator to determine whether it is usable. These are the
mechanics the Designer varies: outage timing/count and episode horizon.

## What the program does

`designer_loop.py` wraps the built-in case14 environment in scenarios with a
growing number of scheduled line outages and a longer horizon. For each
proposal, it:

1. Builds a structured designer prompt with the domain constraints and recent
   result memory.
2. Proposes a scenario specification (JSON-compatible name, complexity level,
   horizon, and outage schedule).
3. Executes the scenario using Grid2Op and a fixed DoNothing reference agent.
4. Records validity, steps survived, peak line loading (`rho`), reward, and
   feedback into `runs_feedback/designer_memory.json` and
   `runs_feedback/experiment_results.json`.

The reference agent is only a **difficulty probe**. The generator is
deterministic so the assignment runs without API keys; the prompt is exposed
as a method, and `--llm` connects it to an OpenAI-compatible chat endpoint for
LLM-generated proposals. SPADE in the paper uses an LLM designer plus
executable validation and feedback. This implementation reproduces that
designer/execute/feedback loop at small scale; it does not claim to reproduce
SPADE's training pipeline or its learned agent.

The course text points to Appendix C.1, and the arXiv version's
environment-generation prompt template is indeed there (page 32): it targets
fully executable Python game classes. Grid2Op scenarios are parameterized
configurations of an existing simulator rather than generated code, so this
project adapts the same practical elements — role, domain, constraints,
complexity target, and a machine-readable environment specification — rather
than reusing the prompt text verbatim.

## Setup and run

Python 3.13 is used because the installed Grid2Op version supports it.

```bash
python3.13 -m venv .venv-week01
.venv-week01/bin/python -m pip install -r week_01/requirements.txt
.venv-week01/bin/python week_01/designer_loop.py --levels 4 --seed 7
```

The first run downloads the Grid2Op case14 dataset (about 294 MB). Later runs
reuse the local dataset. Outputs are written under `week_01/runs_feedback/`.
To use the LLM proposal path, set `OPENAI_API_KEY` (and optionally
`OPENAI_BASE_URL` / `OPENAI_MODEL`) and add `--llm` to the command. The default
run is deterministic and requires no API key.

## Environment library experiment

An initial direct Grid2Op experiment reset the case14 environment with seed 7
and applied a no-op action. The observation vector had 493 values, the grid had
20 lines and 14 substations, the first reward was about 64.99, and maximum line
loading was `rho = 0.9253`. A 12-step no-op baseline survived all 12 steps with
peak `rho = 0.93`. A single initial outage on line 0 also survived 12 steps
(peak `rho = 0.964`). Initial simultaneous outages on lines 0 and 1 failed
power-flow initialization, which demonstrates why the Designer needs to
validate candidate environments instead of assuming every outage combination
is usable. The loop therefore injects outages during the episode and executes
each generated candidate. In the four-level run, level 0 (no outage, 8-step
horizon) survived 8 steps; level 1 (one outage, 10-step horizon) survived 10;
level 2 (two outages, 12-step horizon) reached game over after 6 steps. The
Designer used that feedback to space level 3's outages four steps apart rather
than three; it survived 8 of 14 steps. This shows the memory changing a later
proposal, while also exposing that the fixed no-op reference agent cannot
survive the more difficult candidates.

## Deliverable files

- `designer_loop.py` — runnable Environment Designer, Grid2Op wrapper, prompt,
  scenario validation, and persistent feedback memory.
- `requirements.txt` — Grid2Op dependency.
- `runs_feedback/experiment_results.json` — generated results after running.
- `runs_feedback/designer_memory.json` — accumulated Designer feedback.
