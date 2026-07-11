"""Grounding check via embedding similarity.

Pure functions so they unit-test without a DB or a live model. The GroundingGuard
embeds the answer and the retrieved chunk texts with the in-process bge embedder
and refuses when the answer isn't close enough to any chunk — the enforced version
of "answer only from context".

Note: the local bge embedder returns UN-normalized vectors (embeddings.py encodes
with ``normalize_embeddings=False``), so cosine must normalize itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def cosine_normalized(a: List[float], b: List[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))


def embed_chunks(chunks: List[Dict[str, Any]], embedder, cache: dict | None = None):
    """Embed chunk texts once, memoizing on ``cache`` (the ctx.extras dict) so the
    scope and grounding guards don't each re-embed the same chunks."""
    if cache is not None and "chunk_embs" in cache:
        return cache["chunk_embs"]
    embs = [
        embedder.generate_embedding(c["content"], input_type="search_document")
        for c in chunks
        if c.get("content")
    ]
    if cache is not None:
        cache["chunk_embs"] = embs
    return embs


def max_mean_cosine(vec: List[float], chunk_embs) -> Tuple[float, float]:
    if not chunk_embs:
        return 0.0, 0.0
    sims = [cosine_normalized(vec, ce) for ce in chunk_embs]
    return max(sims), sum(sims) / len(sims)


def grounding_scores(answer: str, chunks: List[Dict[str, Any]], embedder,
                     cache: dict | None = None) -> Tuple[float, float]:
    """Return (max, mean) cosine similarity of the answer to the retrieved chunks.

    Returns (0.0, 0.0) when there are no chunks (nothing to be grounded in).
    """
    if not chunks:
        return 0.0, 0.0
    ans_emb = embedder.generate_embedding(answer, input_type="search_query")
    return max_mean_cosine(ans_emb, embed_chunks(chunks, embedder, cache))
