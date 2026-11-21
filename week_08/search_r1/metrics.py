"""Answer extraction and exact-match scoring for the Search-R1 benchmark."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    """Return the last explicit answer block, or a cleaned fallback response."""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(text: str) -> str:
    """Normalize punctuation, articles, case, and whitespace for EM scoring."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Iterable[str]) -> bool:
    """Match a prediction against any accepted answer alias."""
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(answer) for answer in answers if answer)
