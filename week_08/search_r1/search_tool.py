"""Stateless VeRL function tool backed by the Search-R1 retrieval service."""

from __future__ import annotations

import os
import sys

import requests
from verl.tools.function_tool import function_tool


@function_tool("search")
def search(query: str) -> str:
    """Search the Wikipedia retrieval index for evidence relevant to a question.

    Args:
        query: A concise search query. Include entity names and the relation to find.
    """
    endpoint = os.environ.get("SEARCH_R1_RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve")
    topk = int(os.environ.get("SEARCH_R1_TOPK", "3"))
    timeout = float(os.environ.get("SEARCH_R1_RETRIEVAL_TIMEOUT", "45"))
    print(f"SEARCH_TOOL_CALL query={query!r}", file=sys.stderr, flush=True)
    response = requests.post(
        endpoint,
        json={"queries": [query], "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json().get("result", [[]])
    documents = results[0] if results else []

    passages = []
    for index, item in enumerate(documents, start=1):
        contents = item.get("document", {}).get("contents", "")
        if not contents:
            continue
        title, _, body = contents.partition("\n")
        passages.append(f"[Document {index}] {title.strip()}\n{body.strip()}")
    if not passages:
        return "No relevant documents were found. Continue reasoning or answer if possible."
    return "\n\n".join(passages)
