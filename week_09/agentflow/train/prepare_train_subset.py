"""Create a deterministic, small subset of the official training parquet."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/combined_train.parquet")
    parser.add_argument("--output", default="data/train/combined_train_small.parquet")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Training parquet not found: {source}")
    if args.rows < 1:
        raise ValueError("--rows must be at least 1")
    if source.resolve() == destination.resolve():
        raise ValueError("Output must not overwrite the source training data")

    dataset = load_dataset("parquet", data_files=str(source))["train"]
    take = min(args.rows, len(dataset))
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[:take]
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.select(selected).to_parquet(str(destination))

    manifest = {
        "source": str(source),
        "output": str(destination),
        "source_rows": len(dataset),
        "selected_rows": take,
        "seed": args.seed,
        "selected_source_indices": selected,
        "bamboogle_used_for_training": False,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {take} of {len(dataset)} rows to {destination}")


if __name__ == "__main__":
    main()
