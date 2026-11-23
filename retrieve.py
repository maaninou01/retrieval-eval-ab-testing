"""The retrieval under test, standard and untuned: chunk the corpus, rank every chunk for every question."""

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import boto3
import numpy as np
import Stemmer
from botocore.config import Config

import data

CACHE = Path(__file__).parent / "cache"
STEMMER = Stemmer.Stemmer("english")
# Lucene / Elasticsearch defaults.
BM25_K1 = 1.2
BM25_B = 0.75
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_REGION = "eu-west-1"
# Titan V2's largest output size.
EMBED_DIMS = 1024
# The constant from the RRF paper (Cormack et al., 2009).
RRF_K = 60
RERANK_MODEL = "cohere.rerank-v3-5:0"
RERANK_REGION = "eu-central-1"
# The top of the merged ranking the reranker reorders; the rest keeps its order.
RERANK_CANDIDATES = 50
# HyDE (Gao et al., 2022): dense search embeds a made-up answer passage instead of the question.
HYDE_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
HYDE_REGION = "eu-west-1"
HYDE_PROMPT = (
    "You do not have this document, so invent the passage of it that would answer the question below, worded as the "
    "document itself would word it. At most 500 characters. Output only the passage.\n\nQuestion: {question}"
)
# 500 characters are about 125 tokens; the cap only stops a runaway answer.
HYDE_MAX_TOKENS = 300
# 8 threads ran the 23,357 Titan calls of the dense setup in 6 min with no throttling.
BEDROCK_WORKERS = 8
# A Titan call answers in well under a second; retries back off adaptively when Bedrock throttles.
BEDROCK_CONFIG = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 10, "mode": "adaptive"})


def _tokens(texts: list[str]) -> list[list[str]]:
    # Tokens as strings, not ids: retrieve() maps them through the index's own vocabulary.
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=False, show_progress=False)


def bm25_rank(chunk_texts: list[str], queries: list[str]) -> np.ndarray:
    """(questions, chunks) array of chunk ids, best first."""
    retriever = bm25s.BM25(method="lucene", k1=BM25_K1, b=BM25_B)
    retriever.index(_tokens(chunk_texts), show_progress=False)
    ids, _ = retriever.retrieve(_tokens(queries), k=len(chunk_texts), show_progress=False)
    return ids


def _bedrock(label: str, key: list, inputs: list, one: Callable, model: str, region: str) -> np.ndarray:
    """One Bedrock call per input, in threads, cached in cache/ under a hash of the key."""
    path = CACHE / f"{hashlib.sha256(json.dumps(key).encode()).hexdigest()}.npy"
    if path.exists():
        print(f"{label}: {len(inputs)} inputs, 0 Bedrock calls (cached)")
        return np.load(path)
    client = boto3.client("bedrock-runtime", region_name=region, config=BEDROCK_CONFIG)

    def call(item) -> list:
        return one(json.loads(client.invoke_model(modelId=model, body=json.dumps(item))["body"].read()))

    with ThreadPoolExecutor(BEDROCK_WORKERS) as pool:
        array = np.array(list(pool.map(call, inputs)))
    print(f"{label}: {len(inputs)} inputs, {len(inputs)} Bedrock calls")
    CACHE.mkdir(exist_ok=True)
    # Temp file then rename, so a killed run never leaves half a cache file.
    partial = path.with_suffix(".partial")
    with partial.open("wb") as f:
        np.save(f, array)
    os.replace(partial, path)
    return array


def embed(texts: list[str]) -> np.ndarray:
    """Normalised Titan vectors, one per text."""
    bodies = [{"inputText": t, "dimensions": EMBED_DIMS, "normalize": True} for t in texts]
    vectors = _bedrock("embed", [EMBED_MODEL, EMBED_DIMS, texts], bodies, lambda r: r["embedding"], EMBED_MODEL, EMBED_REGION)
    return vectors.astype(np.float32)


def rerank(queries: list[str], passages: list[list[str]]) -> np.ndarray:
    """(questions, passages) array: each question's passage positions, most relevant first."""
    bodies = [{"query": q, "documents": p, "top_n": len(p), "api_version": 2} for q, p in zip(queries, passages)]
    key = [RERANK_MODEL, queries, passages]
    return _bedrock("rerank", key, bodies, lambda r: [x["index"] for x in r["results"]], RERANK_MODEL, RERANK_REGION)


def hyde(queries: list[str]) -> list[str]:
    """One made-up answer passage per question, at temperature 0 so the same question gets the same passage."""
    bodies = [{"anthropic_version": "bedrock-2023-05-31", "max_tokens": HYDE_MAX_TOKENS, "temperature": 0,
               "messages": [{"role": "user", "content": HYDE_PROMPT.format(question=q)}]} for q in queries]
    key = [HYDE_MODEL, HYDE_PROMPT, HYDE_MAX_TOKENS, queries]
    return [str(p) for p in _bedrock("hyde", key, bodies, lambda r: r["content"][0]["text"], HYDE_MODEL, HYDE_REGION)]


def dense_rank(chunk_texts: list[str], queries: list[str]) -> np.ndarray:
    """(questions, chunks) array of chunk ids, best first, by dot product of normalised vectors."""
    chunk_vectors = embed(chunk_texts)
    query_vectors = embed(queries)
    return np.argsort(-(query_vectors @ chunk_vectors.T), axis=1, kind="stable")


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion, equal weights: each chunk scores the sum of 1 / (k + its rank) over the rankings."""
    scores = np.zeros(rankings[0].shape)
    rank_scores = 1 / (k + np.arange(1, rankings[0].shape[1] + 1))
    for ranking in rankings:
        np.put_along_axis(scores, ranking, np.take_along_axis(scores, ranking, axis=1) + rank_scores, axis=1)
    return np.argsort(-scores, axis=1, kind="stable")


def run(setup: dict, questions: list[dict], corpus: dict[str, str]) -> tuple[list[data.Chunk], np.ndarray]:
    """The chunks (doc, start, end) of the whole corpus and every question's ranking over all of them.

    Search with each retriever that is on (dense with HyDE passages if hyde is on), merge by RRF (one ranking comes
    back unchanged), then rerank if on. BM25 and the reranker always use the real question.
    """
    chunks = [(doc, s, e) for doc in sorted(corpus) for s, e in data.chunk(corpus[doc], setup["chunk_size"])]
    texts = [corpus[doc][s:e] for doc, s, e in chunks]
    queries = [q["query"] for q in questions]
    rankings = []
    if setup.get("bm25"):
        rankings.append(bm25_rank(texts, queries))
    if setup.get("dense"):
        rankings.append(dense_rank(texts, hyde(queries) if setup.get("hyde") else queries))
    ranking = rrf(rankings)
    if setup.get("rerank"):
        top = ranking[:, :RERANK_CANDIDATES]
        order = rerank(queries, [[texts[i] for i in row] for row in top])
        ranking = np.concatenate([top, ranking[:, RERANK_CANDIDATES:]], axis=1)
    return chunks, ranking
