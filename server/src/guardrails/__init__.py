"""Runtime guardrail layer for the RAG server.

Two pipelines run on the synchronous request path (both text and voice, via
``RAGService.query``):

- INPUT  — screens the query (injection, PII, toxicity) before retrieval/generation.
- OUTPUT — validates the answer (scope, grounding, prompt-leak, PII redaction,
  toxicity) before it's cached/returned.

Rolled out shadow-first (``GUARDRAILS_ENFORCE=false``): guards run and decisions are
persisted, but responses are unaltered until enforcement is turned on. Guards fail
open so they never add availability risk.
"""
from __future__ import annotations

from typing import Optional

from src.config import settings
from src.guardrails.base import (
    Action,
    CANNED_REFUSAL,
    GuardrailContext,
    GuardrailDecision,
    GuardrailPipeline,
    PipelineResult,
    Stage,
    canned_refusal,
)
from src.guardrails.input_guards import PiiInputGuard, PromptInjectionGuard, ToxicityGuard
from src.guardrails.output_guards import (
    GroundingGuard,
    LeakGuard,
    PiiRedactionGuard,
    ScopeRetrievalGuard,
)

__all__ = [
    "Action",
    "CANNED_REFUSAL",
    "GuardrailContext",
    "GuardrailDecision",
    "GuardrailPipeline",
    "PipelineResult",
    "Stage",
    "canned_refusal",
    "build_input_pipeline",
    "build_output_pipeline",
]


def build_input_pipeline() -> GuardrailPipeline:
    """Assemble the INPUT pipeline from settings. Empty when disabled."""
    guards: list = []
    if settings.guardrails_enabled:
        if settings.guardrails_injection_enabled:
            guards.append(PromptInjectionGuard())
        if settings.guardrails_pii_enabled:
            guards.append(PiiInputGuard(block=settings.guardrails_pii_block_input))
        if settings.guardrails_toxicity_enabled:
            guards.append(
                ToxicityGuard(Stage.INPUT, settings.inference_provider.lower())
            )
    return GuardrailPipeline(guards, enforce=settings.guardrails_enforce)


def build_output_pipeline(embedder, openai_client: Optional[object] = None) -> GuardrailPipeline:
    """Assemble the OUTPUT pipeline. Needs the embedder for the grounding guard."""
    guards: list = []
    if settings.guardrails_enabled:
        if settings.guardrails_scope_enabled:
            guards.append(ScopeRetrievalGuard(embedder, settings.guardrails_scope_sim_threshold))
        if settings.guardrails_grounding_enabled:
            guards.append(
                GroundingGuard(
                    embedder,
                    sim_max_threshold=settings.guardrails_grounding_sim_max_threshold,
                    sim_mean_threshold=settings.guardrails_grounding_sim_mean_threshold,
                    min_answer_chars=settings.guardrails_grounding_min_answer_chars,
                )
            )
        guards.append(LeakGuard())  # cheap, always on when guardrails enabled
        if settings.guardrails_pii_enabled:
            guards.append(PiiRedactionGuard())
        if settings.guardrails_toxicity_enabled:
            guards.append(
                ToxicityGuard(Stage.OUTPUT, settings.inference_provider.lower(), openai_client)
            )
    return GuardrailPipeline(guards, enforce=settings.guardrails_enforce)
