"""Output-stage guards: validate the generated answer before it's returned."""
from __future__ import annotations

from src.config import settings
from src.guardrails.base import Action, GuardrailContext, GuardrailDecision, Stage
from src.guardrails.grounding import embed_chunks, grounding_scores, max_mean_cosine
from src.guardrails.patterns import LEAK_STRINGS
from src.guardrails.pii import detect_pii, redact
from src.refusal import is_refusal, leak_ok


class ScopeRetrievalGuard:
    """Off-topic gate via query↔chunk cosine.

    The hybrid search's ``similarity_score`` is a small RRF rank-fusion score, not
    a calibrated similarity, so it can't gate scope by an absolute threshold.
    Instead we embed the query and the retrieved chunks (chunk embeddings are
    shared with the grounding guard via ctx.extras) and refuse when the best
    query↔chunk cosine is too low — the question isn't covered by the corpus.
    """
    name = "scope_retrieval"
    stage = Stage.OUTPUT

    def __init__(self, embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        if not ctx.chunks:
            return GuardrailDecision(
                self.name, self.stage, Action.REFUSE, True, reason="no_chunks", score=0.0,
            )
        q_emb = self.embedder.generate_embedding(ctx.query, input_type="search_query")
        chunk_embs = embed_chunks(ctx.chunks, self.embedder, ctx.extras)
        sim_max, _ = max_mean_cosine(q_emb, chunk_embs)
        if sim_max < self.threshold:
            return GuardrailDecision(
                self.name, self.stage, Action.REFUSE, True,
                reason=f"off_topic query_sim={sim_max:.3f}", score=sim_max,
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False, score=sim_max)


class GroundingGuard:
    """Enforce that the answer is anchored in the retrieved context."""
    name = "grounding"
    stage = Stage.OUTPUT

    def __init__(self, embedder, *, sim_max_threshold: float, sim_mean_threshold: float,
                 min_answer_chars: int):
        self.embedder = embedder
        self.sim_max_threshold = sim_max_threshold
        self.sim_mean_threshold = sim_mean_threshold
        self.min_answer_chars = min_answer_chars

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        answer = ctx.answer or ""
        # A raw generation error should surface as a refusal, not the error text.
        if answer.startswith("Error generating answer:"):
            return GuardrailDecision(
                self.name, self.stage, Action.BLOCK, True, reason="generation_error",
            )
        # An abstention is already safe; short answers score poorly vs long chunks
        # (false positives), so skip them.
        if is_refusal(answer, ctx.chunks) or len(answer.strip()) < self.min_answer_chars:
            return GuardrailDecision(self.name, self.stage, Action.ALLOW, False, reason="skipped")
        sim_max, sim_mean = grounding_scores(answer, ctx.chunks, self.embedder, ctx.extras)
        if sim_max < self.sim_max_threshold or sim_mean < self.sim_mean_threshold:
            return GuardrailDecision(
                self.name, self.stage, Action.REFUSE, True,
                reason=f"ungrounded max={sim_max:.3f} mean={sim_mean:.3f}", score=sim_max,
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False, score=sim_max)


class LeakGuard:
    """Block answers that echo system-prompt fragments or injection markers."""
    name = "leak"
    stage = Stage.OUTPUT

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        if not leak_ok(ctx.answer, LEAK_STRINGS):
            return GuardrailDecision(
                self.name, self.stage, Action.BLOCK, True, reason="prompt_leak_or_echo",
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False)


class PiiRedactionGuard:
    """Redact any PII the answer would expose."""
    name = "pii_redaction"
    stage = Stage.OUTPUT

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        spans = detect_pii(ctx.answer or "")
        if spans:
            kinds = ",".join(sorted({s.kind for s in spans}))
            return GuardrailDecision(
                self.name, self.stage, Action.REDACT, True, reason=f"pii:{kinds}",
                redacted_text=redact(ctx.answer, spans),
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False)
