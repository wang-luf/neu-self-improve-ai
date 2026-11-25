"""Run the no-tools Qwen baseline on AgentFlow's Bamboogle JSON data."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--data", default="test/bamboogle/data/data.json", help="AgentFlow Bamboogle JSON"
    )
    parser.add_argument("--output", default="test/bamboogle/results/Qwen-direct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows")
    args = parser.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pid = str(row.get("pid", row.get("idx")))
        if not pid.isdigit():
            raise ValueError(f"Bamboogle row has no numeric pid/idx: {row}")
        question = row.get("query") or row.get("question")
        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and briefly. Put the final answer inside <answer> and </answer> tags.",
            },
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        completion = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        result = {
            "pid": pid,
            "query": question,
            "answer": row.get("answer", []),
            "direct_output": completion,
            "model": args.model,
            "mode": "direct_no_tools",
        }
        (output_dir / f"output_{pid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"completed {pid}/{len(rows) - 1}")


if __name__ == "__main__":
    main()
