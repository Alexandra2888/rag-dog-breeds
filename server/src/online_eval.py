"""Online eval — production quality signals on real /query traffic.

Runs entirely in a FastAPI ``BackgroundTask`` *after* the response is sent, so it
adds no latency to the request path. Every query records cheap deterministic
signals (retrieval-score stats + deterministic refusal detection); a sampled
fraction additionally gets a reference-free LLM judge (faithfulness + relevancy).

Results land in the ``online_eval`` Postgres table (best-effort, mirroring the
answer cache) and are rolled up into MLflow windows by
``scripts/aggregate_online_eval.py`` so quality drift is a trend line over time.
"""
from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logging_config import get_logger
from src.rag_service import normalize_query
from src.refusal import is_refusal

log = get_logger("online_eval")


def compute_retrieval_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Similarity-score distribution over the retrieved chunks (drift signals)."""
    scores = [
        c.get("similarity_score")
        for c in chunks
        if c.get("similarity_score") is not None
    ]
    return {
        "num_chunks": len(chunks),
        "sim_max": max(scores) if scores else None,
        "sim_mean": statistics.fmean(scores) if scores else None,
        "sim_min": min(scores) if scores else None,
    }


def should_sample_judge() -> bool:
    """True on the fraction of traffic that gets the async LLM judge.

    Gated by ``online_judge_enabled`` (off in dev) AND a random draw against
    ``online_judge_sample_rate``, so at most that fraction of prod traffic pays
    the extra LLM cost, and never on the synchronous path.
    """
    if not settings.online_judge_enabled:
        return False
    return random.random() < settings.online_judge_sample_rate


def _run_reference_free_judge(
    query: str, answer: str, chunks: List[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    """Ragas Faithfulness + ResponseRelevancy (both need no reference answer).

    Uses the same local-Ollama judge construction proven in evals/run_eval.py.
    Imported lazily so the API doesn't pull Ragas at startup.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    contexts = [c["content"] for c in chunks] or [""]
    sample = SingleTurnSample(
        user_input=query, response=answer, retrieved_contexts=contexts
    )
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
        )
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=[sample]),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=300, max_retries=2, max_workers=1),
        show_progress=False,
    )
    df = result.to_pandas()

    def _get(col: str) -> Optional[float]:
        return float(df.loc[0, col]) if col in df.columns else None

    rel_col = "answer_relevancy" if "answer_relevancy" in df.columns else "nv_accuracy"
    return {
        "judge_faithfulness": _get("faithfulness"),
        "judge_relevancy": _get(rel_col),
    }


def record_online_eval(
    database,
    result: Dict[str, Any],
    *,
    mode: str,
    top_k: int,
    request_id: Optional[str],
) -> None:
    """Background-task body: compute signals for one query and persist them.

    Best-effort — any failure is logged and swallowed so the task runner never
    surfaces an error (the response is already sent by the time this runs).
    """
    try:
        answer = result.get("answer")  # absent on the empty-retrieval path
        chunks = result.get("chunks", [])
        cached = result.get("cached", False)
        stats = compute_retrieval_stats(chunks)
        row = {
            "request_id": request_id,
            "query_norm": normalize_query(result["query"]),
            "query_text": result["query"],
            "mode": mode,
            "top_k": top_k,
            "cached": cached,
            "answer_len": len(answer or ""),
            "refused": is_refusal(answer, chunks),
            "provider": settings.inference_provider,
            "judge_sampled": False,
            **stats,
        }
        # Only judge fresh (non-cached) answers — a cache hit was already scored
        # when it was first generated, and re-judging wastes tokens.
        if not cached and answer and should_sample_judge():
            try:
                row.update(judge_sampled=True, **_run_reference_free_judge(
                    result["query"], answer, chunks
                ))
            except Exception as e:  # judge failure must not drop the whole row
                log.warning("online_judge_failed", error=str(e))
        database.put_online_eval(row)
    except Exception as e:
        log.warning("online_eval_failed", error=str(e))
