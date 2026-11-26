"""Extract per-step Flow-GRPO metrics from the trainer console log and plot the reward curve.

Usage: python plot_training.py OUT_DIR TRAINER_LOG [TRAINER_LOG ...]
(several logs = a run resumed from a checkpoint; later logs win for repeated steps)
Writes OUT_DIR/training_metrics.csv and OUT_DIR/reward_curve.png.
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEYS = ["critic/score/mean", "val/reward", "actor/pg_loss", "actor/kl_loss", "actor/grad_norm",
        "actor/entropy_loss", "response_length/mean", "timing_s/step"]
BLUE, GRAY, TEXT, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#e4e3df"


def parse(log_paths):
    rows = {}
    lines = [line for path in log_paths for line in Path(path).read_text(errors="replace").splitlines()]
    for line in lines:
        match = re.search(r"step:(\d+) - (.*)", line)
        if not match:
            continue
        values = dict(re.findall(r"([\w/@.\-]+):(-?[\d.]+(?:e-?\d+)?)", match.group(2)))
        rows.setdefault(int(match.group(1)), {}).update({k: float(values[k]) for k in KEYS if k in values})
    return [{"step": step, **rows[step]} for step in sorted(rows)]


def main():
    out_dir, log_paths = Path(sys.argv[1]), sys.argv[2:]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse(log_paths)
    with (out_dir / "training_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *KEYS])
        writer.writeheader()
        writer.writerows(rows)

    steps = [r["step"] for r in rows if "critic/score/mean" in r]
    reward = [r["critic/score/mean"] for r in rows if "critic/score/mean" in r]
    window = 5
    smooth = [sum(reward[max(0, i - window + 1): i + 1]) / len(reward[max(0, i - window + 1): i + 1]) for i in range(len(reward))]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, reward, color=GRAY, linewidth=1, marker="o", markersize=4, alpha=0.6, label="per step (4 prompts x 8 rollouts)")
    ax.plot(steps, smooth, color=BLUE, linewidth=2, label=f"{window}-step moving average")
    ax.set_xlabel("Flow-GRPO step", color=GRAY)
    ax.set_ylabel("training reward (judge accuracy)", color=GRAY)
    ax.set_title("Planner training reward, 256-row NQ + DeepMath subset", loc="left", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=160)
    print(f"{len(rows)} steps -> {out_dir}/training_metrics.csv, reward_curve.png")


if __name__ == "__main__":
    main()
