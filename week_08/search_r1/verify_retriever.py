"""Check that gpu_retrieval_server.py returns the same passages as Search-R1's
own E5 Encoder + CPU FAISS IndexFlatIP search for a few queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import requests


QUERIES = [
    "capital of France",
    "who wrote Hamlet",
    "Who was president of the U.S. when superconductivity was discovered?",
    "inventor of the telephone birthplace",
    "largest moon of Saturn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.search_r1 / "search_r1" / "search"))
    from retrieval_server import Encoder

    encoder = Encoder("e5", "intfloat/e5-base-v2", "mean", 256, False)
    index = faiss.read_index(str(args.index_path))
    started = time.time()
    scores, ids = index.search(encoder.encode(QUERIES), 3)
    cpu_seconds = time.time() - started
    started = time.time()
    response = requests.post(args.url, json={"queries": QUERIES, "topk": 3, "return_scores": True}, timeout=60)
    gpu_seconds = time.time() - started
    result = response.json()["result"]
    gpu_ids = [[int(doc["document"]["id"]) for doc in query] for query in result]
    gpu_scores = np.array([[doc["score"] for doc in query] for query in result])
    report = {
        "ids_match": gpu_ids == ids.tolist(),
        "cpu_faiss_ids": ids.tolist(),
        "gpu_service_ids": gpu_ids,
        "max_abs_score_diff": float(np.abs(scores - gpu_scores).max()),
        "cpu_faiss_seconds": cpu_seconds,
        "gpu_service_seconds": gpu_seconds,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
