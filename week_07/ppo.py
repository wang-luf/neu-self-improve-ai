"""From-scratch PPO for five discrete-action Gymnasium tasks."""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

torch.set_num_threads(1)

ENV_IDS = ["CliffWalking-v1", "Taxi-v4", "CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
OBS_SCALE = {
    "CartPole-v1": np.array([4.8, 5.0, 0.42, 5.0], np.float32),
    "MountainCar-v0": np.array([1.8, 0.14], np.float32),
    "Acrobot-v1": np.array([1.0, 1.0, 1.0, 1.0, 4*math.pi, 9*math.pi], np.float32),
}


def encode_observation(env_id: str, observation: Any, space: gym.Space) -> np.ndarray:
    if isinstance(space, gym.spaces.Discrete):
        vector = np.zeros(space.n, dtype=np.float32)
        vector[int(observation)] = 1.0
        return vector
    if isinstance(space, gym.spaces.Box):
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        scale = OBS_SCALE.get(env_id)
        if scale is None or scale.shape != vector.shape:
            low = np.asarray(space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(space.high, dtype=np.float32).reshape(-1)
            scale = np.maximum(np.maximum(np.abs(low), np.abs(high)), 1.0)
            scale[~np.isfinite(scale)] = 1.0
        return np.clip(vector / scale, -5.0, 5.0).astype(np.float32)
    raise TypeError(f"Unsupported observation space: {space}")


class ActorCritic(nn.Module):
    """Separate policy and value MLPs with orthogonal initialization."""
    def __init__(self, observation_dim: int, action_count: int, hidden=(64, 64)):
        super().__init__()

        def mlp(output_dim: int, output_gain: float) -> nn.Sequential:
            layers: list[nn.Module] = []
            width = observation_dim
            for hidden_width in hidden:
                linear = nn.Linear(width, hidden_width)
                nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
                nn.init.zeros_(linear.bias)
                layers.extend([linear, nn.Tanh()])
                width = hidden_width
            output = nn.Linear(width, output_dim)
            nn.init.orthogonal_(output.weight, gain=output_gain)
            nn.init.zeros_(output.bias)
            layers.append(output)
            return nn.Sequential(*layers)

        self.actor = mlp(action_count, 0.01)
        self.critic = mlp(1, 1.0)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        distribution = Categorical(logits=self.actor(states))
        return distribution.log_prob(actions), distribution.entropy(), self.critic(states).squeeze(-1)


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64



def train_ppo(env_id: str, episodes: int, seed: int, config: PPOConfig, device: torch.device) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Match Week 5 so the learning curves use the same finite horizon.
    env = gym.make(env_id, max_episode_steps=200)
    observation, _ = env.reset(seed=seed)
    state = encode_observation(env_id, observation, env.observation_space)
    model = ActorCritic(state.size, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    episode_returns: list[float] = []
    recent_returns: list[float] = []
    episode_return = 0.0
    update = 0

    try:
        while len(episode_returns) < episodes:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * (1.0 - len(episode_returns) / episodes)
            states, actions, log_probs, rewards = [], [], [], []
            values, next_values, terminated_flags, done_flags = [], [], [], []
            for _ in range(config.rollout_steps):
                if len(episode_returns) >= episodes:
                    break
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    dist = Categorical(logits=model.actor(state_tensor))
                    action = dist.sample()
                    old_log_prob = dist.log_prob(action)
                    value = model.critic(state_tensor).squeeze(-1)
                next_observation, reward, terminated, truncated, _ = env.step(int(action.item()))
                next_state = encode_observation(env_id, next_observation, env.observation_space)
                with torch.no_grad():
                    next_value = model.critic(torch.as_tensor(next_state, dtype=torch.float32, device=device)).squeeze(-1)
                states.append(state)
                actions.append(int(action.item()))
                log_probs.append(float(old_log_prob.item()))
                rewards.append(float(reward))
                values.append(float(value.item()))
                next_values.append(float(next_value.item()))
                terminated_flags.append(float(terminated))
                done_flags.append(float(terminated or truncated))
                episode_return += float(reward)
                state = next_state
                if terminated or truncated:
                    episode_returns.append(episode_return)
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    observation, _ = env.reset()
                    state = encode_observation(env_id, observation, env.observation_space)

            # Bootstrap time-limit truncations; stop the GAE trace at every episode boundary.
            advantages = np.zeros(len(rewards), dtype=np.float32)
            last_advantage = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + config.gamma * (1.0-terminated_flags[t]) * next_values[t] - values[t]
                last_advantage = delta + config.gamma*config.gae_lambda * (1.0-done_flags[t]) * last_advantage
                advantages[t] = last_advantage
            returns = advantages + np.asarray(values, dtype=np.float32)

            obs_b = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
            act_b = torch.as_tensor(actions, dtype=torch.long, device=device)
            logp_b = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
            value_b = torch.as_tensor(values, dtype=torch.float32, device=device)
            adv_b = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            ret_b = torch.as_tensor(returns, dtype=torch.float32, device=device)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

            indices = np.arange(len(states))
            for _ in range(config.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), config.minibatch_size):
                    batch = torch.as_tensor(indices[start:start+config.minibatch_size], device=device)
                    new_logp, entropy, new_value = model.evaluate(obs_b[batch], act_b[batch])
                    ratio = (new_logp-logp_b[batch]).exp()
                    adv = adv_b[batch]
                    policy_loss = torch.maximum(
                        -adv*ratio,
                        -adv*torch.clamp(ratio, 1-config.clip_coefficient, 1+config.clip_coefficient),
                    ).mean()
                    clipped_value = value_b[batch] + torch.clamp(
                        new_value-value_b[batch], -config.value_clip, config.value_clip
                    )
                    value_loss = 0.5*torch.maximum(
                        (new_value-ret_b[batch]).square(), (clipped_value-ret_b[batch]).square()
                    ).mean()
                    loss = policy_loss + config.value_coefficient*value_loss - config.entropy_coefficient*entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                    optimizer.step()

            if update % 20 == 0:
                score = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
                print(f"PPO {env_id:16s} seed={seed:>2} update={update+1:>4} episodes={len(episode_returns):>4}/{episodes} recent_return={score:8.2f}")
            update += 1
    finally:
        env.close()
    return np.asarray(episode_returns, dtype=np.float32)


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, len(values)+1)
    starts = np.maximum(0, ends-window)
    return ((sums[ends]-sums[starts])/(ends-starts)).astype(np.float32)


def run_experiment(episodes: int, config: PPOConfig, seeds: list[int], env_ids: list[str], output_dir: str):
    from pathlib import Path
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        module_directory = Path(__file__).resolve().parent
    except NameError:  # The source is running in a notebook cell.
        module_directory = Path.cwd()
        if not (module_directory / "previous_methods_summary.csv").exists():
            module_directory = module_directory / "week_07"
    previous_path = module_directory / "previous_methods_summary.csv"
    with previous_path.open(newline="") as previous_file:
        previous_summary = list(csv.DictReader(previous_file))
    results: dict[str, list[np.ndarray]] = {env: [] for env in env_ids}
    started = time.perf_counter()
    for env_id in env_ids:
        print(f"\n{env_id} — PPO: {episodes} episodes × {len(seeds)} seeds")
        for seed in seeds:
            results[env_id].append(train_ppo(env_id, episodes, seed, config, device))
    print(f"\nTraining time: {(time.perf_counter()-started)/60:.1f} minutes on {device}")

    summary = [dict(row) for row in previous_summary if row["environment"] in env_ids]
    colors = {"PPO": "#9467bd"}
    for env_id, runs in results.items():
        length = min(map(len, runs))
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(1, length+1)
        curves = np.stack([moving_average(run[:length], window=50) for run in runs])
        mean = curves.mean(axis=0)
        error = curves.std(axis=0, ddof=1)/math.sqrt(len(runs)) if len(runs)>1 else np.zeros_like(mean)
        ax.plot(x, mean, color=colors["PPO"], label="PPO", linewidth=1.8)
        ax.fill_between(x, mean-error, mean+error, color=colors["PPO"], alpha=0.15)
        for row in previous_summary:
            if row["environment"] == env_id:
                baseline = float(row["mean_final_100_return"])
                ax.axhline(baseline, linestyle="--", linewidth=1, alpha=0.7,
                           label=f'{row["algorithm"]} prior final mean ({baseline:.1f})')
        ax.set(title=f"{env_id}: PPO vs Week 5 final performance", xlabel="Completed episode", ylabel="Trailing 50-episode return")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output/f"{env_id.replace('-', '_')}_comparison.png", dpi=160)
        # Render inline when this same source cell runs in Jupyter; keep the
        # command-line experiment non-interactive.
        if "get_ipython" in globals() and get_ipython() is not None:
            display(fig)
        plt.close(fig)
        tails = [float(run[-min(100, len(run)):].mean()) for run in runs]
        summary.append({"environment": env_id, "algorithm": "PPO",
                        "mean_final_100_return": f"{np.mean(tails):.2f}",
                        "sd_across_seeds": f"{np.std(tails, ddof=1) if len(tails)>1 else 0.0:.2f}"})
    print("\nFinal 100-episode return (mean ± SD across seeds)")
    for row in summary:
        print(f'{row["environment"]:16s} {row["algorithm"]:22s} '
              f'{float(row["mean_final_100_return"]):9.2f} ± {float(row["sd_across_seeds"]):.2f}')
    with (output / "summary.csv").open("w", newline="") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["environment", "algorithm", "mean_final_100_return", "sd_across_seeds"])
        for row in summary:
            writer.writerow([row["environment"], row["algorithm"], row["mean_final_100_return"], row["sd_across_seeds"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--envs", nargs="+", default=ENV_IDS, choices=ENV_IDS)
    parser.add_argument("--output-dir", default="week_07/results")
    args = parser.parse_args()
    run_experiment(args.episodes, PPOConfig(), args.seeds, args.envs, args.output_dir)


if __name__ == "__main__":
    main()
