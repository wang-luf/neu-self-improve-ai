"""Run Search-R1 GRPO/PPO training on a local GPU host (the GB300 node).

Local counterpart of `modal_train_verl.py`: it expects a running
`gpu_retrieval_server.py` (or Search-R1's own `/retrieve` service), runs
`train_verl.sh` inside the pinned VeRL environment, exports the last LoRA
checkpoint to a PEFT adapter, and writes a `run_summary.json` with the same
pass/fail evidence checks as the Modal smoke (tool call, reward, checkpoint,
exported adapter).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("grpo", "ppo"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "results" / "runs")
    parser.add_argument("--verl-dir", type=Path, default=Path.home() / "verl")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES for the trainer")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Per-GPU micro batch (sequences)")
    parser.add_argument("--mini-batch-size", type=int, default=None, help="PPO mini batch in prompts (default: batch size)")
    parser.add_argument("--actor-lr", default="2e-5")
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--test-freq", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("overrides", nargs="*", help="Extra VeRL overrides")
    args = parser.parse_args()

    smoke = args.steps == 1
    run_name = args.run_name or f"{args.algorithm}-{'smoke' if smoke else f'{args.steps}-steps'}"
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = requests.post(
        args.retrieval_url, json={"queries": ["capital of France"], "topk": 1, "return_scores": True}, timeout=30
    )
    probe.raise_for_status()
    probe_text = probe.json()["result"][0][0]["document"]["contents"][:240]

    ngpus = len(args.gpus.split(","))
    save_freq = args.save_freq or (1 if smoke else 50)
    test_freq = args.test_freq or (-1 if smoke else 50)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpus,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SEARCH_R1_RETRIEVAL_URL": args.retrieval_url,
            "SEARCH_R1_TOPK": "3",
            "SEARCH_R1_LOG_REWARD": "1",
            "MODEL_PATH": "Qwen/Qwen2.5-3B-Instruct",
            "TRAIN_FILE": str(args.data_dir / ("smoke_train.parquet" if smoke else "train.parquet")),
            "VAL_FILE": str(args.data_dir / ("smoke_test.parquet" if smoke else "val_sample.parquet")),
            "OUTPUT_DIR": str(output_dir),
            "TRAIN_BATCH_SIZE": str(args.batch_size),
            "ROLLOUT_N": str(args.rollouts),
            "MINI_BATCH_SIZE": str(args.mini_batch_size or args.batch_size),
            "ACTOR_LR": args.actor_lr,
            "MICRO_BATCH_SIZE": str(min(args.micro_batch_size, args.batch_size * args.rollouts)),
            "TOTAL_STEPS": str(args.steps),
            "TOTAL_EPOCHS": "1",
            "NGPUS": str(ngpus),
            "VERL_FILE_LOGGER_PATH": str(output_dir / "metrics.jsonl"),
        }
    )
    trainer_command = [
        "bash",
        str(HERE / "train_verl.sh"),
        args.algorithm,
        f"trainer.val_before_train={'False' if smoke else 'True'}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        *args.overrides,
    ]
    (output_dir / "command.json").write_text(
        json.dumps({"command": trainer_command, "env": {k: env[k] for k in sorted(env) if k.startswith(("SEARCH_R1", "MODEL", "TRAIN", "VAL", "OUTPUT", "ROLLOUT", "MICRO", "MINI", "ACTOR", "TOTAL", "NGPUS", "CUDA"))}}, indent=2),
        encoding="utf-8",
    )
    started = time.time()
    trainer_log_path = output_dir / "trainer.log"
    with trainer_log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            trainer_command,
            cwd=args.verl_dir,
            env={**env, "PATH": f"{args.verl_dir / '.venv' / 'bin'}:{env['PATH']}", "VIRTUAL_ENV": str(args.verl_dir / ".venv")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    train_seconds = time.time() - started
    log_text = trainer_log_path.read_text(encoding="utf-8", errors="replace")

    steps_done = sorted(int(m) for m in re.findall(r"global_step_(\d+)", " ".join(p.name for p in output_dir.glob("global_step_*"))))
    last_step = steps_done[-1] if steps_done else None
    export_dir = output_dir / "adapter-export"
    export_returncode = None
    if last_step is not None:
        export_command = [
            str(args.verl_dir / ".venv" / "bin" / "python"), "-m", "verl.model_merger", "merge",
            "--backend", "fsdp", "--local_dir", str(output_dir / f"global_step_{last_step}" / "actor"),
            "--target_dir", str(export_dir), "--use_cpu_initialization",
        ]
        with (output_dir / "adapter-export.log").open("w", encoding="utf-8") as log:
            export_returncode = subprocess.run(
                export_command, cwd=args.verl_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode

    checkpoint_files = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file() and "global_step_" in str(p))
    lora_meta = [name for name in checkpoint_files if "lora_train_meta.json" in name]
    adapter_dir = export_dir / "lora_adapter"
    exported = sorted(p.name for p in adapter_dir.glob("adapter_*")) if adapter_dir.exists() else []
    tool_calls = log_text.count("SEARCH_TOOL_CALL")
    rewards = re.findall(r"SEARCH_R1_REWARD score=([0-9.]+)", log_text)
    checkpoint_saved = last_step is not None and any(f"global_step_{last_step}/actor/model_world_size_" in n for n in checkpoint_files)
    passed = (
        completed.returncode == 0
        and export_returncode == 0
        and checkpoint_saved
        and bool(lora_meta)
        and len(exported) >= 2
        and tool_calls > 0
        and bool(rewards)
    )
    gpu_names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", args.gpus], text=True, capture_output=True
    ).stdout.split("\n")
    summary = {
        "algorithm": args.algorithm.upper(),
        "status": "passed" if passed else "failed",
        "run_name": run_name,
        "host": platform.node(),
        "gpus": [name for name in gpu_names if name],
        "cuda_visible_devices": args.gpus,
        "steps_requested": args.steps,
        "last_checkpoint_step": last_step,
        "batch_size_prompts": args.batch_size,
        "rollouts_per_prompt": args.rollouts,
        "mini_batch_size_prompts": args.mini_batch_size or args.batch_size,
        "actor_lr": args.actor_lr,
        "retriever": f"wiki-18 E5 Flat (exact inner product on GPU), top-k=3, {args.retrieval_url}",
        "retriever_probe": probe_text,
        "tool_calls_logged": tool_calls,
        "rewards_logged": len(rewards),
        "mean_logged_reward": sum(map(float, rewards)) / len(rewards) if rewards else None,
        "lora_metadata_files": lora_meta,
        "exported_adapter_files": exported,
        "adapter_dir": str(adapter_dir),
        "adapter_export_returncode": export_returncode,
        "trainer_exit_code": completed.returncode,
        "train_seconds": round(train_seconds, 1),
        "overrides": args.overrides,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(f"run did not pass all evidence checks; see {trainer_log_path}")


if __name__ == "__main__":
    main()
