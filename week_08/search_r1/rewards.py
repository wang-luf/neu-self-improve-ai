"""Search-R1 exact-match reward compatible with VeRL's custom reward hook."""

from __future__ import annotations

import re
import string
import os


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return outcome-only exact match; no separate formatting reward is used."""
    if isinstance(ground_truth, dict):
        aliases = ground_truth.get("target", ground_truth.get("answers", []))
    else:
        aliases = ground_truth
    if hasattr(aliases, "tolist"):
        aliases = aliases.tolist()
    if isinstance(aliases, str):
        aliases = [aliases]
    matches = ANSWER_RE.findall(solution_str)
    answer = matches[-1].strip() if matches else ""
    prediction = normalize(answer)
    score = float(any(prediction == normalize(str(alias)) for alias in (aliases or []) if alias))
    if os.environ.get("SEARCH_R1_LOG_REWARD") == "1":
        print(f"SEARCH_R1_REWARD score={score} answer={answer!r}", flush=True)
    return score
