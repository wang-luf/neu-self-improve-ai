"""Merge evaluate_bamboogle.py shard outputs into one JSONL + summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True, help="Shard JSONL files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for path in args.shards for line in path.read_text(encoding="utf-8").splitlines()]
    records.sort(key=lambda record: record["index"])
    indices = [record["index"] for record in records]
    if indices != list(range(len(records))):
        raise ValueError(f"shards do not cover a contiguous index range: {len(records)} records")
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    shard_summaries = [json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8")) for path in args.shards]
    correct = sum(int(record["exact_match"]) for record in records)
    calls = sum(record["search_calls"] for record in records)
    summary = {key: value for key, value in shard_summaries[0].items() if key not in {"shard", "predictions"}}
    summary.update(
        {
            "examples": len(records),
            "correct": correct,
            "exact_match": correct / len(records),
            "search_calls": calls,
            "mean_search_calls": calls / len(records),
            "seconds": max(s["seconds"] for s in shard_summaries),
            "merged_from_shards": [str(path) for path in args.shards],
            "predictions": str(args.output),
        }
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
