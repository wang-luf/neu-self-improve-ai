"""Search-R1 GRPO/PPO training on the course Modal workspace.

Uses the shared course NQ/HotpotQA + wiki-18 volumes read-only and writes to an
isolated results volume. Two H100s are assigned: one for the E5 query encoder,
one for the Qwen policy/rollout. The FAISS flat index is searched on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
VERL_REVISION = (HERE / "verl.pin").read_text(encoding="utf-8").strip()
APP_NAME = "searchr1-course-grpo-smoke"
DATA_VOLUME = modal.Volume.from_name("week08-search-r1-data")
CACHE_VOLUME = modal.Volume.from_name("search-r1-wiki18")
RESULTS_VOLUME = modal.Volume.from_name("searchr1-assignment5-results", create_if_missing=True)

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.20")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone --filter=blob:none https://github.com/verl-project/verl.git /opt/verl",
        f"git -C /opt/verl fetch --depth 1 origin {VERL_REVISION} && git -C /opt/verl checkout --detach {VERL_REVISION}",
        "cd /opt/verl && uv sync --frozen --all-packages --extra sglang --extra fsdp",
        "uv pip install --python /opt/verl/.venv/bin/python faiss-cpu==1.12.0",
        "git clone --filter=blob:none https://github.com/PeterGriffinJin/Search-R1.git /opt/Search-R1",
        "git -C /opt/Search-R1 fetch --depth 1 origin 598e61bd1d36895726d28a8d06b3a15bed19f5d3 && git -C /opt/Search-R1 checkout --detach 598e61bd1d36895726d28a8d06b3a15bed19f5d3",
    )
    .add_local_dir(str(HERE), remote_path="/workspace/search_r1")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_CACHE": "/tmp/hf_datasets",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
)


@app.function(
    image=image,
    gpu="H100:2",
    cpu=8,
    memory=131072,
    timeout=7200,
    volumes={
        "/shared": DATA_VOLUME,
        "/cache": CACHE_VOLUME,
        "/results": RESULTS_VOLUME,
    },
)
def run_training(algorithm: str, steps: int, batch_size: int, rollouts: int) -> dict[str, object]:
    import pandas as pd
    import requests
    from format_searchr1_rows import format_frame

    data_dir = Path("/shared/nq_hotpotqa_train")
    index_path = Path("/shared/wiki-18-e5-index/e5_Flat.index")
    corpus_path = Path("/shared/wiki-18-corpus/wiki-18.jsonl")
    train_source = data_dir / "train.parquet"
    test_source = data_dir / "test.parquet"
    for path in (train_source, test_source, index_path, corpus_path):
        if not path.exists():
            raise FileNotFoundError(f"Shared course asset missing: {path}")

    if algorithm not in {"grpo", "ppo"}:
        raise ValueError("algorithm must be 'grpo' or 'ppo'")
    smoke = steps == 1

    def prepare(source: Path, destination: Path, split: str) -> int:
        frame = pd.read_parquet(source)
        frame = frame[frame["data_source"].astype(str).isin({"nq", "hotpotqa"})]
        if smoke:
            frame = frame.head(2 if split == "train" else 1)
        converted = format_frame(frame, split)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(converted).to_parquet(destination, index=False)
        return len(converted)

    smoke_data = Path("/tmp/searchr1-smoke-data")
    train_count = prepare(train_source, smoke_data / "train.parquet", "train")
    validation_count = prepare(test_source, smoke_data / "test.parquet", "test")

    retriever_log_path = Path("/tmp/searchr1-retriever.log")
    retriever_log = retriever_log_path.open("w", encoding="utf-8")
    retriever_env = os.environ.copy()
    retriever_env["CUDA_VISIBLE_DEVICES"] = "0"
    retriever_command = [
        "/opt/verl/.venv/bin/python",
        "/opt/Search-R1/search_r1/search/retrieval_server.py",
        "--index_path",
        str(index_path),
        "--corpus_path",
        str(corpus_path),
        "--topk",
        "3",
        "--retriever_name",
        "e5",
        "--retriever_model",
        "intfloat/e5-base-v2",
    ]
    retriever = subprocess.Popen(
        retriever_command,
        cwd="/opt/Search-R1",
        env=retriever_env,
        stdout=retriever_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    endpoint = "http://127.0.0.1:8000/retrieve"
    ready = False
    try:
        deadline = time.monotonic() + 3000
        last_error = "retriever is still loading"
        while time.monotonic() < deadline:
            if retriever.poll() is not None:
                retriever_log.flush()
                raise RuntimeError(f"Retriever exited early; see {retriever_log_path}: {last_error}")
            try:
                probe = requests.post(
                    endpoint,
                    json={"queries": ["capital of France"], "topk": 1, "return_scores": True},
                    timeout=10,
                )
                probe.raise_for_status()
                if probe.json().get("result"):
                    ready = True
                    break
            except Exception as exc:  # readiness polling while index and corpus load
                last_error = repr(exc)
            time.sleep(10)
        if not ready:
            retriever_log.flush()
            raise TimeoutError(f"Retriever failed readiness check: {last_error}; see {retriever_log_path}")

        run_name = f"{algorithm}-{'smoke' if smoke else f'{steps}-steps'}"
        output_dir = Path("/results/assignment5") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "1",
                "HF_HOME": "/cache/huggingface",
                "HF_HUB_CACHE": "/cache/huggingface/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_CACHE": "/tmp/hf_datasets",
                "SEARCH_R1_RETRIEVAL_URL": endpoint,
                "SEARCH_R1_TOPK": "3",
                "SEARCH_R1_LOG_REWARD": "1",
                "MODEL_PATH": "Qwen/Qwen2.5-3B-Instruct",
                "DATA_DIR": str(smoke_data),
                "OUTPUT_DIR": str(output_dir),
                "TRAIN_BATCH_SIZE": str(batch_size),
                "ROLLOUT_N": str(rollouts),
                "TOTAL_STEPS": str(steps),
                "TOTAL_EPOCHS": "1",
                "NGPUS": "1",
            }
        )
        trainer_command = [
            "uv",
            "run",
            "--directory",
            "/opt/verl",
            "--frozen",
            "--all-packages",
            "--extra",
            "sglang",
            "--extra",
            "fsdp",
            "bash",
            "/workspace/search_r1/train_verl.sh",
            algorithm,
            f"trainer.val_before_train={'False' if smoke else 'True'}",
            f"trainer.save_freq={1 if smoke else 50}",
            "trainer.test_freq=-1" if smoke else "trainer.test_freq=50",
        ]
        completed = subprocess.run(
            trainer_command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        trainer_log_path = output_dir / "trainer.log"
        trainer_log_path.write_text(completed.stdout, encoding="utf-8")
        export_dir = output_dir / "adapter-export"
        export_log_path = output_dir / "adapter-export.log"
        export_returncode = None
        if completed.returncode == 0:
            export_command = [
                "uv", "run", "--directory", "/opt/verl", "--frozen", "--all-packages",
                "--extra", "sglang", "--extra", "fsdp", "python", "-m", "verl.model_merger", "merge",
                "--backend", "fsdp", "--local_dir", str(output_dir / f"global_step_{steps}" / "actor"),
                "--target_dir", str(export_dir), "--use_cpu_initialization",
            ]
            exported = subprocess.run(
                export_command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            export_log_path.write_text(exported.stdout, encoding="utf-8")
            export_returncode = exported.returncode
        retriever_log.flush()
        checkpoint_files = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]
        adapter_files = [name for name in checkpoint_files if "lora_train_meta.json" in name]
        exported_adapter_files = [
            str(path.relative_to(output_dir))
            for path in (export_dir / "lora_adapter").glob("adapter_*")
            if path.is_file()
        ] if (export_dir / "lora_adapter").exists() else []
        tool_call_logged = "SEARCH_TOOL_CALL" in completed.stdout
        reward_logged = "SEARCH_R1_REWARD" in completed.stdout
        checkpoint_saved = any("global_step_1/actor/model_world_size_" in name for name in checkpoint_files)
        passed = (
            completed.returncode == 0
            and export_returncode == 0
            and checkpoint_saved
            and bool(adapter_files)
            and len(exported_adapter_files) >= 2
            and tool_call_logged
            and reward_logged
        )
        RESULTS_VOLUME.commit()
        result = {
            "app": APP_NAME,
            "algorithm": algorithm.upper(),
            "status": "passed" if passed else "failed",
            "train_prompts": train_count,
            "validation_prompts": validation_count,
            "rollouts_per_prompt": rollouts,
            "steps_requested": steps,
            "retriever": "Search-R1 E5 Flat wiki-18, FAISS CPU index, top-k=3",
            "retriever_probe": (probe.json().get("result", [[]])[0][0].get("document", {}).get("contents", "")[:240]),
            "tool_call_logged": tool_call_logged,
            "reward_logged": reward_logged,
            "training_gpu": "CUDA_VISIBLE_DEVICES=1 (H100)",
            "retriever_encoder_gpu": "CUDA_VISIBLE_DEVICES=0 (H100)",
            "results_dir": str(output_dir),
            "checkpoint_files": checkpoint_files,
            "lora_metadata_files": adapter_files,
            "lora_sharded_checkpoint_saved": checkpoint_saved,
            "exported_adapter_files": exported_adapter_files,
            "adapter_export_returncode": export_returncode,
            "adapter_export_log": str(export_log_path),
            "trainer_log": str(trainer_log_path),
            "trainer_exit_code": completed.returncode,
        }
        print(json.dumps(result, indent=2))
        if not passed:
            raise RuntimeError(
                "VeRL run did not prove successful training, retrieval-tool use, and adapter checkpoint output; "
                f"see {trainer_log_path} and {export_log_path} (train={completed.returncode}, "
                f"export={export_returncode}, files={len(checkpoint_files)}, tool={tool_call_logged})"
            )
        return result
    finally:
        if retriever.poll() is None:
            retriever.terminate()
            try:
                retriever.wait(timeout=30)
            except subprocess.TimeoutExpired:
                retriever.kill()
                retriever.wait(timeout=10)
        retriever_log.close()


@app.local_entrypoint()
def main(algorithm: str = "grpo", steps: int = 1, batch_size: int = 2, rollouts: int = 2):
    print(json.dumps(run_training.remote(algorithm, steps, batch_size, rollouts), indent=2))
