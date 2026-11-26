"""Training-reward breakdown by data source from AgentFlow's saved training rollouts.

Rollout step folders are numbered by the rollout process, not by the trainer,
so each rollout is assigned to a trainer step by time: the end time of step N
is the last log timestamp before the trainer's "step:N -" metrics line.
Rollouts in (--cutoff, --resume-after), i.e. the interrupted step that was
re-run after resuming, are dropped.

Usage: python analyze_rollouts.py ROLLOUT_ROOT TRAIN_PARQUET OUT_CSV --trainer-logs LOG [LOG ...]
       [--cutoff ISO_TIME --resume-after ISO_TIME] [--experiment NAME]
"""

import argparse
import csv
import json
import re
from bisect import bisect_left
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_root", type=Path)
    parser.add_argument("train_parquet")
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--cutoff", default=None, help="drop rollouts in (cutoff, resume_after)")
    parser.add_argument("--resume-after", default=None)
    parser.add_argument("--experiment", default="week9-part1-flowgrpo-lora")
    parser.add_argument("--trainer-logs", nargs="+", required=True)
    args = parser.parse_args()

    source = dict(zip(pd.read_parquet(args.train_parquet)["id"].astype(str), pd.read_parquet(args.train_parquet)["source"]))
    rows = []
    for path in args.rollout_root.glob(f"{args.experiment}_*/*/train/step_*/idx_*/rollout_*.json"):
        r = json.loads(path.read_text())
        if args.cutoff and args.cutoff < r["timestamp"] < args.resume_after:
            continue
        if str(r["id"]) not in source:
            continue  # validation (AIME24) rollouts are saved under train/ too
        memory = (r.get("total_result") or {}).get("memory") or {}
        tools = [step.get("tool_name") for step in memory.values() if isinstance(step, dict)]
        rows.append({
            "timestamp": r["timestamp"],
            "source": source[str(r["id"])],
            "reward": float(r["reward"]),
            "searches": sum(t == "Local_Wikipedia_Search_Tool" for t in tools),
            "tool_calls": len(tools),
        })
    rows.sort(key=lambda r: r["timestamp"])

    step_end = {}
    for log in args.trainer_logs:
        last_time = None
        for line in Path(log).read_text(errors="replace").splitlines():
            t = re.search(r"INFO:(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)", line)
            if t:
                last_time = f"{t.group(1)}T{t.group(2)}"
            m = re.search(r"step:(\d+) - .*critic/score/mean", line)
            if m and last_time:
                step_end[int(m.group(1))] = last_time
    steps = sorted(step_end)
    ends = [step_end[k] for k in steps]

    blocks = []
    for k, step in enumerate(steps):
        block = [r for r in rows if bisect_left(ends, r["timestamp"]) == k]
        if not block:
            continue
        entry = {"step": step, "rollouts": len(block), "reward": round(sum(r["reward"] for r in block) / len(block), 3)}
        for src in ("nq", "mathhard"):
            sub = [r for r in block if r["source"] == src]
            entry[f"{src}_rollouts"] = len(sub)
            entry[f"{src}_reward"] = round(sum(r["reward"] for r in sub) / len(sub), 3) if sub else ""
        nq = [r for r in block if r["source"] == "nq"]
        entry["nq_searches_per_rollout"] = round(sum(r["searches"] for r in nq) / len(nq), 2) if nq else ""
        blocks.append(entry)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)

    half = len(blocks) // 2
    for label, part in (("first half", blocks[:half]), ("second half", blocks[half:])):
        for src in ("nq", "mathhard"):
            vals = [(b[f"{src}_reward"], b[f"{src}_rollouts"]) for b in part if b[f"{src}_rollouts"]]
            total = sum(n for _, n in vals)
            print(f"{label:11s} {src:8s} rollouts={total:4d} reward={sum(v * n for v, n in vals) / total:.3f}")
    print(f"{len(rows)} rollouts in {len(blocks)} steps -> {args.out_csv}")


if __name__ == "__main__":
    main()
