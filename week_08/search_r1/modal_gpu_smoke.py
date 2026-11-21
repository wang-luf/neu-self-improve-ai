"""One-batch Qwen2.5-3B + PEFT GPU smoke test on the course Modal workspace.

This checks CUDA, model loading, LoRA gradient flow, and one optimizer update.
It does not run Search-R1 retrieval, PPO, or GRPO and uses synthetic text only.
"""

from __future__ import annotations

import json

import modal


APP_NAME = "searchr1-course-lora-gpu-smoke"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==4.55.2",
        "peft==0.17.1",
        "accelerate==1.10.1",
        "safetensors",
    )
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
)


@app.function(image=image, gpu="A10G", timeout=900, cpu=4, memory=16384)
def run_one_lora_update() -> dict[str, float | int | str | bool]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Modal function did not receive a CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(7)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    adapter_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)
    model.train()

    text = "A short synthetic sentence to verify one LoRA optimizer update."
    batch = tokenizer(text, return_tensors="pt").to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    updated = any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))

    result = {
        "status": "passed" if updated and float(grad_norm) > 0 else "failed",
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda or "unknown",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "trainable_parameters": trainable_count,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(grad_norm.detach().cpu()),
        "adapter_updated": updated,
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError(f"LoRA GPU smoke test failed: {result}")
    return result


@app.local_entrypoint()
def main():
    result = run_one_lora_update.remote()
    print(json.dumps(result, indent=2))
