"""Convert the course-preprocessed NQ/HotpotQA rows to VeRL tool-agent rows."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You answer multi-hop questions with evidence from Wikipedia. Use the search tool "
    "at least once before answering. Think step by step, use concise search queries, "
    "and finish with exactly one <answer>short answer</answer> block."
)


def format_row(raw: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    extra = raw.get("extra_info") or {}
    question = raw.get("question") or extra.get("question")
    if not question:
        messages = raw.get("prompt") or []
        question = messages[-1].get("content", "") if messages else ""

    reward = raw.get("reward_model") or {}
    ground_truth = reward.get("ground_truth", {}) if isinstance(reward, dict) else {}
    if isinstance(ground_truth, dict):
        aliases = ground_truth.get("target", ground_truth.get("answers", []))
    else:
        aliases = ground_truth
    if hasattr(aliases, "tolist"):
        aliases = aliases.tolist()
    if isinstance(aliases, str):
        aliases = [aliases]
    if not question or aliases is None or len(aliases) == 0:
        raise ValueError(f"Could not extract question/answers for {split} row {index}")

    return {
        "data_source": "searchR1_nq_hotpotqa",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question)},
        ],
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": {"target": aliases}},
        "extra_info": {
            "split": split,
            "index": extra.get("index", index),
            "question": str(question),
        },
    }


def format_frame(frame, split: str) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame while preserving source row indices."""
    return [format_row(row, split, i) for i, row in enumerate(frame.to_dict("records"))]
