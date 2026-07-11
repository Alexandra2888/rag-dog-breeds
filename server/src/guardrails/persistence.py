"""Best-effort persistence of guardrail decisions (runs in a BackgroundTask)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.logging_config import get_logger
from src.rag_service import normalize_query

log = get_logger("guardrails")


def _rows_from_result(pr, *, query: str, mode: str, request_id: Optional[str]) -> List[Dict[str, Any]]:
    """Flatten one PipelineResult into per-decision DB rows."""
    query_norm = normalize_query(query)
    final = pr.final_action.value
    rows = []
    for d in pr.decisions:
        rows.append({
            "request_id": request_id,
            "query_norm": query_norm,
            "query_text": query,
            "mode": mode,
            "stage": d.stage.value,
            "guard_name": d.name,
            "action": d.action.value,
            "triggered": d.triggered,
            "final_action": final,
            "enforce": settings.guardrails_enforce,
            "reason": d.reason or None,
            "score": d.score,
            "latency_ms": d.latency_ms,
            "provider": settings.inference_provider,
        })
    return rows


def record_guardrail_events(
    database,
    pipeline_results: List[Any],
    *,
    query: str,
    mode: str,
    request_id: Optional[str],
) -> None:
    """BackgroundTask body: persist all guardrail decisions for one request.

    ``pipeline_results`` is the list of PipelineResult objects (input + output)
    stashed on the query result. Best-effort — logs and swallows any failure.
    """
    try:
        rows: List[Dict[str, Any]] = []
        for pr in pipeline_results:
            if pr is not None:
                rows.extend(_rows_from_result(pr, query=query, mode=mode, request_id=request_id))
        if rows:
            database.put_guardrail_events(rows)
    except Exception as e:
        log.warning("guardrail_persist_failed", error=str(e))
