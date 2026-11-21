"""Convert the official Search-R1 NQ + HotpotQA parquet files to VeRL format.

Reads `train.parquet`/`test.parquet` from `--source-dir` (or downloads them from
the Hugging Face dataset repo) and keeps only NQ and HotpotQA rows. The released
test parquet also contains Bamboogle and other benchmarks; those stay untouched
until final evaluation. Rows are formatted with `format_searchr1_rows`, the same
prompt used by every training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from format_searchr1_rows import format_frame


IN_DOMAIN = {"nq", "hotpotqa"}


def load_split(args: argparse.Namespace, split: str) -> pd.DataFrame:
    if args.source_dir:
        path = args.source_dir / f"{split}.parquet"
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=args.hf_repo_id, filename=f"{split}.parquet", repo_type="dataset")
    frame = pd.read_parquet(path)
    return frame[frame["data_source"].astype(str).isin(IN_DOMAIN)].reset_index(drop=True)


def write(frame: pd.DataFrame, split: str, destination: Path) -> int:
    converted = format_frame(frame, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted).to_parquet(destination, index=False)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo-id", default="PeterJinGo/nq_hotpotqa_train")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/searchr1_processed"))
    parser.add_argument("--val-sample", type=int, default=512, help="Rows in the fixed validation subset used during training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_split(args, "train")
    test = load_split(args, "test")
    bamboogle_questions = set(pd.read_csv(Path(__file__).parent / "data" / "bamboogle.csv")["question"].str.strip().str.lower())
    for name, frame in (("train", train), ("test", test)):
        leaked = frame["question"].str.strip().str.lower().isin(bamboogle_questions).sum()
        if leaked:
            raise ValueError(f"{name} contains {leaked} Bamboogle questions")

    counts = {
        "train": write(train, "train", args.output_dir / "train.parquet"),
        "test": write(test, "test", args.output_dir / "test.parquet"),
        "val_sample": write(
            test.sample(n=min(args.val_sample, len(test)), random_state=args.seed), "test", args.output_dir / "val_sample.parquet"
        ),
        "smoke_train": write(train.head(2), "train", args.output_dir / "smoke_train.parquet"),
        "smoke_test": write(test.head(1), "test", args.output_dir / "smoke_test.parquet"),
    }
    summary = {
        "source": str(args.source_dir or args.hf_repo_id),
        "kept_data_sources": sorted(IN_DOMAIN),
        "train_by_source": train["data_source"].value_counts().to_dict(),
        "test_by_source": test["data_source"].value_counts().to_dict(),
        "rows_written": counts,
        "bamboogle_overlap": 0,
        "val_sample_seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
