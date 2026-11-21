"""Plot GRPO vs PPO training curves from VeRL `metrics.jsonl` files.

Small multiples (one metric per panel, one shared y-scale per panel), GRPO and
PPO in fixed categorical slots 1 and 2, legend plus an end-of-line label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"GRPO": "#2a78d6", "PPO": "#eb6834"}
PANELS = [
    ("val-core/searchR1_nq_hotpotqa/acc/mean@1", "Validation EM (512 NQ/HotpotQA)"),
    ("critic/score/mean", "Train reward (EM, mean over rollouts)"),
    ("actor/kl_loss", "KL to reference policy"),
    ("num_turns/mean", "Turns per trajectory"),
    ("response_length/mean", "Response length (tokens)"),
    ("actor/entropy", "Policy entropy"),
]


def load(path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        step, data = record.get("step", 0), record.get("data", record)
        for key, value in data.items():
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append((int(step), float(value)))
    return series


def label_value(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"


def load_bamboogle(directory: Path, run: str) -> list[tuple[int, float]]:
    points = []
    for path in directory.glob(f"{run}_step*.summary.json"):
        step = int(path.name.split("_step")[1].split(".")[0])
        points.append((step, json.loads(path.read_text(encoding="utf-8"))["exact_match"]))
    return sorted(points)


def smooth(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=metrics.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=None, help="Truncate curves at this step")
    parser.add_argument("--bamboogle-dir", type=Path, default=None, help="Adds a held-out Bamboogle EM panel")
    parser.add_argument("--bamboogle-baseline", type=float, default=None, help="Untrained model EM, drawn as a reference line")
    parser.add_argument("--smooth", type=int, default=5, help="Trailing moving-average window for per-step metrics")
    args = parser.parse_args()

    runs = {}
    panels = list(PANELS)
    for item in args.run:
        label, path = item.split("=", 1)
        runs[label] = load(Path(path))
        if args.bamboogle_dir:
            run_name = Path(path).parent.name
            runs[label]["bamboogle"] = load_bamboogle(args.bamboogle_dir, run_name)
    if args.bamboogle_dir:
        panels.insert(1, ("bamboogle", "Bamboogle EM (125 held-out questions)"))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY})
    columns = 3 if len(panels) <= 6 else 4
    fig, axes = plt.subplots(2, columns, figsize=(4.4 * columns, 7), facecolor=SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (key, title) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ends = []
        for label, series in runs.items():
            points = [(s, v) for s, v in series.get(key, []) if args.max_step is None or s <= args.max_step]
            if key == "bamboogle" and points and args.bamboogle_baseline is not None:
                points = [(0, args.bamboogle_baseline)] + points  # step 0 = the untrained model
            if not points:
                continue
            steps, values = zip(*points)
            is_val = key.startswith("val-") or key == "bamboogle"
            ys = list(values) if is_val else smooth(list(values), args.smooth)
            color = SERIES.get(label, TEXT_SECONDARY)
            ax.plot(steps, ys, color=color, linewidth=2, label=label,
                    marker="o" if is_val else None, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends.append((ys[-1], label, steps[-1]))
        # End-of-line labels; push apart labels whose values sit closer than 6% of the y-range.
        low, high = ax.get_ylim()
        ends.sort()
        for i, (y, label, x) in enumerate(ends):
            nudge = 0
            if len(ends) == 2 and abs(ends[1][0] - ends[0][0]) < 0.06 * (high - low):
                nudge = -5 if i == 0 else 5
            ax.annotate(f"{label} {label_value(y)}", (x, y), xytext=(4, nudge), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_SECONDARY)
        if key == "bamboogle" and args.bamboogle_baseline is not None:
            ax.axhline(args.bamboogle_baseline, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.annotate("untrained", (0.01, args.bamboogle_baseline), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel("training step")
        ax.margins(x=0.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=len(labels), labelcolor=TEXT_PRIMARY)
    fig.suptitle(
        f"Search-R1 LoRA training on NQ+HotpotQA (Qwen2.5-3B-Instruct); per-step metrics: {args.smooth}-step moving average",
        x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
