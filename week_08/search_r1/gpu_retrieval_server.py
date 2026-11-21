"""Search-R1 compatible `/retrieve` service with exact inner-product search on GPU.

Loads the official wiki-18 E5 Flat FAISS index, copies its float32 vectors to one
GPU and answers queries with an exact matmul + top-k, which is the same search
an `IndexFlatIP` performs on CPU. Query encoding matches Search-R1's
`retrieval_server.py` for E5: "query: " prefix, mean pooling, L2 normalization,
max length 256. Concurrent requests are micro-batched so many rollouts can
search at once. The request/response schema is Search-R1's.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mmap
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class Corpus:
    """Random access to wiki-18.jsonl through a line-offset table over an mmap."""

    def __init__(self, path: Path):
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        buffer = np.frombuffer(self.data, dtype=np.uint8)
        newlines = np.flatnonzero(buffer == ord("\n"))
        self.starts = np.concatenate([[0], newlines + 1])[: len(newlines)]
        self.ends = newlines

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict:
        return json.loads(self.data[self.starts[index] : self.ends[index]])


class Retriever:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda")
        started = time.time()
        index = faiss.read_index(str(args.index_path))
        if not isinstance(index, faiss.IndexFlat) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(f"Expected an inner-product IndexFlat, got {type(index).__name__}")
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
        self.vectors = torch.from_numpy(vectors).to(self.device)
        self.cpu_index = index if args.keep_cpu_index else None
        print(f"index: {index.ntotal} x {index.d} on GPU in {time.time() - started:.0f}s", flush=True)
        started = time.time()
        self.corpus = Corpus(args.corpus_path)
        if len(self.corpus) != index.ntotal:
            raise ValueError(f"corpus rows {len(self.corpus)} != index rows {index.ntotal}")
        print(f"corpus: {len(self.corpus)} rows in {time.time() - started:.0f}s", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
        self.encoder = AutoModel.from_pretrained(args.retriever_model).to(self.device).eval()
        self.max_length = args.max_length
        self.default_topk = args.topk

    @torch.no_grad()
    def encode(self, queries: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            [f"query: {query}" for query in queries],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.encoder(**inputs, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"][..., None].bool()
        pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    @torch.no_grad()
    def search(self, queries: list[str], topk: int) -> tuple[list[list[float]], list[list[int]]]:
        embeddings = self.encode(queries)
        scores, indices = [], []
        for start in range(0, len(queries), 64):
            block = embeddings[start : start + 64] @ self.vectors.T
            top = torch.topk(block, k=topk, dim=-1)
            scores.extend(top.values.cpu().tolist())
            indices.extend(top.indices.cpu().tolist())
        return scores, indices


def build_app(retriever: Retriever, max_batch: int) -> FastAPI:
    app = FastAPI()
    queue: asyncio.Queue = asyncio.Queue()
    stats = {"requests": 0, "queries": 0, "batches": 0}

    async def worker() -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await queue.get()]
            await asyncio.sleep(0.003)
            while not queue.empty() and sum(len(item[0]) for item in items) < max_batch:
                items.append(queue.get_nowait())
            queries = [query for item in items for query in item[0]]
            topk = max(item[1] for item in items)
            try:
                scores, indices = await loop.run_in_executor(None, retriever.search, queries, topk)
            except Exception as exc:  # propagate failures to every waiting request
                for item in items:
                    item[2].set_exception(exc)
                continue
            stats["batches"] += 1
            offset = 0
            for item_queries, item_topk, future in items:
                count = len(item_queries)
                future.set_result((scores[offset : offset + count], indices[offset : offset + count], item_topk))
                offset += count

    @app.on_event("startup")
    async def start_worker() -> None:
        asyncio.create_task(worker())

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", **stats}

    @app.post("/retrieve")
    async def retrieve(request: QueryRequest) -> dict:
        topk = request.topk or retriever.default_topk
        future = asyncio.get_running_loop().create_future()
        await queue.put((request.queries, topk, future))
        scores, indices, topk = await future
        stats["requests"] += 1
        stats["queries"] += len(request.queries)
        results = []
        for query_scores, query_indices in zip(scores, indices):
            docs = [retriever.corpus[index] for index in query_indices[:topk]]
            if request.return_scores:
                results.append([{"document": doc, "score": score} for doc, score in zip(docs, query_scores[:topk])])
            else:
                results.append(docs)
        return {"result": results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=Path, required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_batch", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--keep_cpu_index", action="store_true")
    args = parser.parse_args()
    retriever = Retriever(args)
    uvicorn.run(build_app(retriever, args.max_batch), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
