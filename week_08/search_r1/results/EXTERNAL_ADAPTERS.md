# Supplemental Search-R1 adapters

This project now indexes a supplemental set of 100-step LoRA checkpoints for
GRPO and PPO. They are hosted in a separate public repository and are kept
separate from the step-150 runs reported in this repository. The scores below
are the source run's reported scores; they have not been rerun in this checkout
and must not be presented as local measurements.

Source commit: [`44e02aa0328346d0ef58a9827e535983519cbde1`](https://github.com/jiacao-amd/neu-self-improve-ai/commit/44e02aa0328346d0ef58a9827e535983519cbde1)

| Algorithm | Step | Reported Bamboogle EM | Adapter weights |
|---|---:|---:|---|
| GRPO | 25 | 0.232 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/grpo-main/adapter-step25/lora_adapter/adapter_model.safetensors) |
| GRPO | 50 | 0.208 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/grpo-main/adapter-step50/lora_adapter/adapter_model.safetensors) |
| GRPO | 100 | 0.304 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/grpo-main/adapter-step100/lora_adapter/adapter_model.safetensors) |
| PPO | 25 | 0.184 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/ppo-main/adapter-step25/lora_adapter/adapter_model.safetensors) |
| PPO | 50 | 0.080 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535519cbde1/week_08/search_r1/results/runs/ppo-main/adapter-step50/lora_adapter/adapter_model.safetensors) |
| PPO | 100 | 0.160 | [adapter](https://github.com/jiacao-amd/neu-self-improve-ai/blob/44e02aa0328346d0ef58a9827e535519cbde1/week_08/search_r1/results/runs/ppo-main/adapter-step100/lora_adapter/adapter_model.safetensors) |

Each weight file is about 60 MB. Download a checkpoint only when needed; for
example:

```bash
mkdir -p results/external/grpo-step100/lora_adapter
curl -fL \
  'https://raw.githubusercontent.com/jiacao-amd/neu-self-improve-ai/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/grpo-main/adapter-step100/lora_adapter/adapter_model.safetensors' \
  -o results/external/grpo-step100/lora_adapter/adapter_model.safetensors
curl -fL \
  'https://raw.githubusercontent.com/jiacao-amd/neu-self-improve-ai/44e02aa0328346d0ef58a9827e535983519cbde1/week_08/search_r1/results/runs/grpo-main/adapter-step100/lora_adapter/adapter_config.json' \
  -o results/external/grpo-step100/lora_adapter/adapter_config.json
```

To evaluate one, pass its local directory to `evaluate_bamboogle.py --adapter`.
Keep its output labeled as the supplemental step-100 checkpoint and do not fold
its score into the locally trained step-150 results without rerunning the full
evaluation under the same protocol.
