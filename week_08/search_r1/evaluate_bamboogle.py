"""Evaluate direct or Search-R1-style inference on the course Bamboogle set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from metrics import exact_match, extract_answer


DEFAULT_DATA = Path(__file__).parent / "data" / "bamboogle.csv"
SEARCH_SYSTEM = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first "
    "every time you get new information. After reasoning, if you find you lack some knowledge, "
    "you can call a search engine by <search> query </search> and it will return the top searched "
    "results between <information> and </information>. You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)
DIRECT_INSTRUCTION = (
    "Answer the given question directly, without searching. Provide only the final short answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)
# Tool mode must match training exactly: same system prompt as the VeRL
# parquet rows, the same tool schema VeRL derives from `search_tool.search`,
# and the same character-level middle truncation of tool responses.
TOOL_RESPONSE_MAX_CHARS = 2048


class StopOnMarkers(StoppingCriteria):
    """Stop a single-sequence generation once the decoded tail contains a marker.

    Matching decoded text (not token ids) also catches markers that the
    tokenizer merges with neighbouring characters, e.g. "</search>\n".
    """

    def __init__(self, tokenizer: Any, markers: tuple[str, ...], prompt_length: int, tail_tokens: int = 12):
        self.tokenizer = tokenizer
        self.markers = markers
        self.prompt_length = prompt_length
        self.tail_tokens = tail_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated = input_ids[0, self.prompt_length :]
        tail = self.tokenizer.decode(generated[-self.tail_tokens :], skip_special_tokens=True)
        return any(marker in tail for marker in self.markers)


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def make_prompt(tokenizer: Any, question: str, mode: str) -> str:
    if mode == "search":
        # Match Search-R1's infer.py: the instruction + question as one user
        # message, wrapped in the chat template for instruct models.
        normalized_question = question.strip()
        if normalized_question and not normalized_question.endswith("?"):
            normalized_question += "?"
        prompt = f"{SEARCH_SYSTEM} Question: {normalized_question}\n"
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
        return prompt

    messages = [{"role": "user", "content": f"{DIRECT_INSTRUCTION} Question: {question.strip()}"}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return messages[0]["content"] + "\nAnswer:"


def get_search_text(url: str, query: str, topk: int, timeout: float) -> str:
    response = requests.post(
        url,
        json={"queries": [query], "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    docs = response.json()["result"][0]
    formatted: list[str] = []
    for index, item in enumerate(docs, start=1):
        contents = item.get("document", {}).get("contents", "")
        if not contents:
            continue
        title, _, body = contents.partition("\n")
        formatted.append(f"Doc {index}(Title: {title}) {body}\n")
    return "".join(formatted) or "No relevant documents were found."


def generate_direct(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)


def generate_search(
    model: Any,
    tokenizer: Any,
    prompt: str,
    retrieval_url: str,
    topk: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    prefix = prompt
    chunks: list[str] = []

    for _ in range(max_turns):
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        stop = StoppingCriteriaList(
            [StopOnMarkers(tokenizer, ("</search>", "</answer>"), inputs.input_ids.shape[1])]
        )
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stop,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
        generated = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        # Cut at the first action boundary, like Search-R1's rollout postprocessing.
        boundaries = [generated.index(m) + len(m) for m in ("</search>", "</answer>") if m in generated]
        if boundaries:
            generated = generated[: min(boundaries)]
        chunks.append(generated)

        if "</answer>" in generated or output[0, -1].item() == tokenizer.eos_token_id:
            break

        match = re.search(r"<search>(.*?)</search>", generated, flags=re.DOTALL)
        if match is None:
            break
        results = get_search_text(retrieval_url, match.group(1).strip(), topk, timeout)
        # Search-R1 infer.py: prompt += "\n\n{output_text}<information>{results}</information>\n\n"
        prefix += f"\n\n{generated}<information>{results}</information>\n\n"
        chunks.append(f"<information>{results}</information>\n\n")

    return "".join(chunks)


def tool_schema() -> list[dict[str, Any]]:
    import search_tool  # noqa: F401  (registers the VeRL function tool)
    from verl.tools.function_tool import get_function_tool

    return [get_function_tool("search").tool_schema.model_dump(exclude_unset=True, exclude_none=True)]


def truncate_tool_response(text: str, max_chars: int = TOOL_RESPONSE_MAX_CHARS) -> str:
    """VeRL ToolAgentLoop's default `middle` truncation."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "...(truncated)..." + text[-half:]


def generate_tool_search(
    model: Any,
    tokenizer: Any,
    question: str,
    retrieval_url: str,
    topk: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, int]:
    """Evaluate VeRL-trained policies using the same native function-tool protocol."""
    import os

    import search_tool
    from format_searchr1_rows import SYSTEM_PROMPT

    if not tokenizer.chat_template:
        raise ValueError("Native search-tool evaluation requires a tokenizer chat template")
    os.environ["SEARCH_R1_RETRIEVAL_URL"] = retrieval_url
    os.environ["SEARCH_R1_TOPK"] = str(topk)
    schemas = tool_schema()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    chunks: list[str] = []
    search_calls = 0
    for _ in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=schemas,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_args)
        generated = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        generated = generated.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        chunks.append(generated)
        if "<tool_call>" not in generated:
            break
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", generated, flags=re.DOTALL)
        if match is None:
            break
        try:
            call = json.loads(match.group(1))
            query = str(call["arguments"]["query"]).strip()
        except (KeyError, TypeError, json.JSONDecodeError):
            break
        if not query:
            break
        messages.append({"role": "assistant", "content": generated})
        tool_result = truncate_tool_response(search_tool.search(query))
        messages.append({"role": "tool", "content": tool_result})
        chunks.append(f"\n<tool_response>\n{tool_result}\n</tool_response>\n")
        search_calls += 1
    return "".join(chunks), search_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter directory")
    parser.add_argument("--mode", choices=("direct", "search", "tool"), default="direct")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/predictions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, metavar="K/N", help="Evaluate only contiguous shard K of N (merge with merge_eval_shards.py)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding (default for all reported runs)")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval-timeout", type=float, default=30)
    args = parser.parse_args()

    rows = load_rows(args.data, args.limit)
    offset = 0
    if args.shard:
        shard_index, num_shards = (int(x) for x in args.shard.split("/"))
        size = -(-len(rows) // num_shards)
        offset = shard_index * size
        rows = rows[offset : offset + size]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total_search_calls = 0
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            prompt = make_prompt(tokenizer, row["question"], args.mode)
            if args.mode == "search":
                completion = generate_search(
                    model,
                    tokenizer,
                    prompt,
                    args.retrieval_url,
                    args.topk,
                    args.max_turns,
                    args.max_new_tokens,
                    args.temperature,
                    args.retrieval_timeout,
                )
                search_calls = len(re.findall(r"<search>.*?</search>", completion, flags=re.DOTALL))
            elif args.mode == "tool":
                completion, search_calls = generate_tool_search(
                    model,
                    tokenizer,
                    row["question"],
                    args.retrieval_url,
                    args.topk,
                    args.max_turns,
                    args.max_new_tokens,
                    args.temperature,
                    args.retrieval_timeout,
                )
            else:
                completion = generate_direct(model, tokenizer, prompt, args.max_new_tokens)
                search_calls = 0

            prediction = extract_answer(completion)
            answers = [row.get("answer", ""), row.get("answer2", "")]
            is_correct = exact_match(prediction, answers)
            correct += int(is_correct)
            total_search_calls += search_calls
            record = {
                "index": offset + i - 1,
                "question": row["question"],
                "answers": [answer for answer in answers if answer],
                "completion": completion,
                "prediction": prediction,
                "exact_match": is_correct,
                "search_calls": search_calls,
                "mode": args.mode,
                "model": args.model,
                "adapter": args.adapter,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{i}/{len(rows)} EM={correct / i:.3f} latest={int(is_correct)}")

    elapsed = time.perf_counter() - started
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "mode": args.mode,
        "examples": len(rows),
        "shard": args.shard,
        "correct": correct,
        "exact_match": correct / len(rows) if rows else 0.0,
        "search_calls": total_search_calls,
        "mean_search_calls": total_search_calls / len(rows) if rows else 0.0,
        "seconds": elapsed,
        "temperature": args.temperature,
        "max_new_tokens_per_turn": args.max_new_tokens,
        "max_turns": args.max_turns,
        "topk": args.topk,
        "predictions": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
