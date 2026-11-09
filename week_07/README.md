# Week 7 — Proximal Policy Optimization

This week implements PPO from scratch in PyTorch and applies it to the five
Gymnasium environments used in Week 5: CliffWalking, Taxi, CartPole,
MountainCar, and Acrobot. Each PPO learning curve is compared with the Week 5
final-100-episode results for REINFORCE, REINFORCE with a learned baseline, and
Actor-Critic. Those baseline values are recorded in `previous_methods_summary.csv`
from the saved Week 5 notebook output. PPO learns a stochastic categorical
policy and a separate value function. Its update uses generalized advantage estimation
(GAE), normalized advantages, the clipped policy objective, clipped value
loss, entropy regularization, gradient clipping, Adam, and a linearly annealed
learning rate.

## Run

```sh
python -m venv .venv-week07
.venv-week07/bin/python -m pip install -r week_07/requirements.txt
.venv-week07/bin/python week_07/ppo.py
```

The default run uses 500 episodes for each of three seeds on each task, with a
shared 200-step episode cap to match Week 5. Runtime depends on the machine. A
shorter run is:

```sh
.venv-week07/bin/python week_07/ppo.py --episodes 50 --seeds 7 --envs CartPole-v1
```

The script saves one comparison plot per environment under `week_07/results/`.
Each plot shows PPO's learning curve, ±1 standard error across seeds, and
horizontal reference lines for Week 5's final performance. Curves use episode
index. `summary.csv` combines the prior Week 5 results with PPO's final-100
episode mean and standard deviation across seeds.

### Comparison results

Final 100-episode return (mean ± SD across three seeds):

| Environment | REINFORCE | REINFORCE + baseline | Actor-Critic | PPO |
| --- | ---: | ---: | ---: | ---: |
| CliffWalking-v1 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.00 ± 0.00 | −207.26 ± 1.51 |
| Taxi-v4 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.45 ± 0.24 | −201.32 ± 0.21 |
| CartPole-v1 | 129.75 ± 61.75 | 39.06 ± 51.03 | 24.62 ± 3.84 | 164.35 ± 21.27 |
| MountainCar-v0 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.00 ± 0.00 |
| Acrobot-v1 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.00 ± 0.00 | −200.00 ± 0.00 |

PPO's CartPole result is higher than the three Week 5 baselines in this run.
The other tasks show little separation under this shared setup; MountainCar
and Acrobot receive the time-limit return for every method. These results show
the limits of this budget and setup, not that PPO cannot solve those tasks.
This assignment evaluates episodic return, not classification accuracy. The
saved Week 5 output contains return summaries but not per-episode success
counts, so a baseline success-rate comparison cannot be reconstructed from it.

## PPO update

For rollouts collected under the old policy, PPO forms the likelihood ratio
`r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`. The policy update maximizes

```text
min(r_t A_t, clip(r_t, 1 − ε, 1 + ε) A_t)
```

GAE estimates `A_t` from TD residuals with `γ` and `λ`. The critic regresses
onto `A_t + V_old(s_t)` with a clipped value update. Entropy encourages
exploration. Minibatch epochs reuse each rollout while clipping limits policy
change from the behavior policy.

The update reuses trajectories collected under the old policy. Importance
sampling corrects the conditional action probabilities with `r_t`, while the
state visitation distribution is approximated by the old policy's states. The
clipped objective limits extra reward for favorable probability changes past
`1 ± ε`, while the `min` keeps unfavorable changes penalized. Clipping is a
stabilizing heuristic, not a hard trust-region constraint or a guarantee of
monotonic return improvement. The implementation uses the clipped PPO variant;
it does not implement PPO's separate adaptive-KL-penalty variant.

Discrete observations are one-hot encoded; Box observations are scaled to
roughly unit range. Time-limit truncations bootstrap from the final observation,
while true terminal states do not. The GAE trace stops at both kinds of episode
boundary.

## Reading notes: R1 and GRPO

The two papers in `~/Desktop/neu` study language-model RL. They share policy
optimization ideas with this control experiment but use token sequences as
actions and different reward setups.

- **DeepSeek-R1 (arXiv:2501.12948, v2):** R1-Zero starts from a pretrained
  base model and applies RL without an SFT warm-up. It uses GRPO: sample a
  group of responses per prompt, score them using rule-based correctness and
  format rewards, then use within-group relative rewards instead of a learned
  value critic. The full R1 pipeline later adds rejection sampling, SFT, and
  more RL to improve helpfulness and readability. The reported emergent
  “Aha” behaviors are observations from these training runs; they do not show
  that RL creates reasoning from scratch.
- **Understanding R1-Zero-Like Training (arXiv:2503.20783v2, COLM 2025):**
  the authors find that pretraining and prompt templates affect answer
  formatting, exploration, and apparent self-reflection. They identify
  length and standard-deviation normalization in GRPO as sources of
  optimization bias. Their Dr. GRPO removes those normalization terms and
  reports better token efficiency while maintaining reasoning performance.

These findings qualify the claim that RL alone produces reasoning: the starting
policy and reward signal matter, and optimizer details can change response
length separately from correctness. This PPO control exercise does not
reproduce GRPO, train a language model, or test the papers' claims; it
demonstrates the shared clipped policy-gradient idea in small environments
with explicit rewards.

Sources: [DeepSeek-R1](https://arxiv.org/abs/2501.12948),
[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783v2),
[PPO original paper](https://arxiv.org/abs/1707.06347),
[37 PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/),
[CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/).

## Scope note

The supplied Week 7 Notion URL currently returns `object_not_found` for both
the page and its linked block in the connected workspace. This work follows
the page title (PPO), the Week 5 notebook's experiment format, and the PPO/RL
reading materials stored in the local `neu` folder. If the missing page
contains additional rubric items, those still need to be checked.
