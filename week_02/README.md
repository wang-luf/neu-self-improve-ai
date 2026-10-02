# Week 2 — Policy Iteration for a Minimal Power Grid MDP

## Domain

This project models a simplified power-grid operator deciding whether to hold
the current configuration or redispatch generation. Power-grid operation is a
real-world sequential decision problem and is also the domain used by Grid2Op.

## Minimal MDP

The state is the grid's operating condition:

| State | Meaning | Terminal? |
| --- | --- | --- |
| `normal` | Lines are within safe operating limits. | No |
| `stressed` | At least one line is close to its limit. | No |
| `failure` | The episode has ended after a grid failure. | Yes |

At either nonterminal state, the operator chooses `hold` or `redispatch`.
Redispatch reduces the chance of entering a worse state, at an operating cost.
The transition table is in `schema.sql` and is also seeded by `models.py`.

The transition reward is `-0.2` for a redispatch action, plus `-1` when the
next state is `stressed`, plus `-10` when it is `failure`. The terminal state's
value is zero. The discount factor is `gamma = 0.9`.

For a policy `pi`, policy evaluation uses

`V_pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a,s') + gamma * V_pi(s')]`.

Policy improvement computes the same expectation for each available action,
then chooses the action with the largest expected return. `policy_iteration.py`
repeats these steps until the policy no longer changes.

## Data model

Following the lecture notebook, the policy's whole iteration history is kept
in the database rather than overwritten in place:

- `states`, `actions`, `transitions` define the MDP.
- `policies` is a named, immutable policy (`pi_0`, `pi_1`, ...); its action
  probabilities live in `policy_probabilities`.
- A `run` is one policy-iteration execution; each pass over the states adds a
  `snapshot` (`target_policy_id` is the policy being evaluated,
  `greedy_policy_id` is the improved policy derived from it).
- `state_values` and `action_values` hold `v_pi` and `q_pi` for each
  snapshot, so every iteration's values and the resulting policy change are
  recoverable afterward, not just the final result.

## Run

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python week_02/policy_iteration.py
```

The script creates a local SQLite database from the SQLAlchemy schema, then
evaluates and improves the policy, printing each iteration's values, Q-values,
and chosen actions until it converges. Re-running against an existing database
starts a new `run` rather than touching a previous one. The database file is
generated locally and is not part of the source submission.

## Source

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
  Chapters 3–4 (MDPs, value functions, policy evaluation, policy improvement).
- Course material: Week 2 policy iteration notebook and assignment prompt.
