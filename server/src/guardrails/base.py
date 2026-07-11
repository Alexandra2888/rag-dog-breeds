"""Guardrail protocol, decision types, and the pipeline runner.

A guardrail is a small deterministic (or single-call) check that runs LIVE on the
request path — unlike the offline eval and the async online-eval. The pipeline
runs a list of guards, records every decision, and in *enforce* mode applies the
strongest action; in *shadow* mode it records but never alters the response.

Design rules baked in here:
- **Fail open.** A guard that raises is treated as ALLOW (logged), so a guard bug
  or a slow moderation call never breaks a request. Correct for a shadow-first
  rollout on a memory-tight VM.
- **Strongest action wins.** ALLOW < REDACT < REFUSE < BLOCK.
- **Shadow vs enforce.** In shadow the output text is returned unchanged and the
  caller ignores INPUT blocks; only the decisions are persisted.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class Action(str, Enum):
    """Guard actions, ordered weakest → strongest (see ``_ACTION_RANK``)."""
    ALLOW = "allow"
    REDACT = "redact"
    REFUSE = "refuse"
    BLOCK = "block"


_ACTION_RANK = {Action.ALLOW: 0, Action.REDACT: 1, Action.REFUSE: 2, Action.BLOCK: 3}

# What the user sees when a guard blocks/refuses in enforce mode. Voice is short
# because it's spoken aloud.
CANNED_REFUSAL: Dict[str, str] = {
    "text": (
        "I can only answer questions about dog breeds from the reference book, "
        "and I can't help with that request."
    ),
    "voice": "Sorry, I can only answer questions about dog breeds from the book.",
}


def canned_refusal(mode: str) -> str:
    return CANNED_REFUSAL.get(mode, CANNED_REFUSAL["text"])


@dataclass
class GuardrailContext:
    """Everything a guard needs to make a decision."""
    query: str
    mode: str = "text"                      # "text" | "voice"
    top_k: int = 5
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None            # set on the OUTPUT stage
    request_id: Optional[str] = None
    # Scratch space for guard-to-guard sharing within one run (e.g. the scope and
    # grounding guards reuse one set of chunk embeddings). Not persisted.
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailDecision:
    name: str
    stage: Stage
    action: Action
    triggered: bool
    reason: str = ""
    score: Optional[float] = None
    latency_ms: float = 0.0
    redacted_text: Optional[str] = None     # set when action == REDACT


@dataclass
class PipelineResult:
    decisions: List[GuardrailDecision]
    final_action: Action
    triggered: bool
    output_text: Optional[str]              # possibly redacted/refused answer (OUTPUT); None for INPUT
    total_latency_ms: float


@runtime_checkable
class Guardrail(Protocol):
    name: str
    stage: Stage

    def check(self, ctx: GuardrailContext) -> GuardrailDecision: ...


class GuardrailPipeline:
    """Runs an ordered list of guards and aggregates their decisions."""

    def __init__(self, guards: List[Guardrail], *, enforce: bool):
        self.guards = guards
        self.enforce = enforce

    def run(self, ctx: GuardrailContext) -> PipelineResult:
        decisions: List[GuardrailDecision] = []
        for guard in self.guards:
            t0 = time.perf_counter()
            try:
                decision = guard.check(ctx)
            except Exception as e:  # fail OPEN — a guard must never break a request
                logger.warning("guardrail_error name=%s error=%s", getattr(guard, "name", "?"), e)
                decision = GuardrailDecision(
                    name=getattr(guard, "name", "unknown"),
                    stage=getattr(guard, "stage", ctx.answer is not None and Stage.OUTPUT or Stage.INPUT),
                    action=Action.ALLOW,
                    triggered=False,
                    reason=f"guard_error:{e}",
                )
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            decisions.append(decision)
            # Chain redactions so a later guard sees the redacted text. We always
            # thread it (even in shadow) so the recorded decisions are coherent.
            if decision.action == Action.REDACT and decision.redacted_text is not None:
                ctx = replace(ctx, answer=decision.redacted_text)

        strongest = max(
            (d for d in decisions if d.triggered),
            key=lambda d: _ACTION_RANK[d.action],
            default=None,
        )
        final_action = strongest.action if strongest else Action.ALLOW
        output_text = self._apply(ctx, final_action) if ctx.answer is not None else None
        return PipelineResult(
            decisions=decisions,
            final_action=final_action,
            triggered=strongest is not None,
            output_text=output_text,
            total_latency_ms=sum(d.latency_ms for d in decisions),
        )

    def _apply(self, ctx: GuardrailContext, final_action: Action) -> Optional[str]:
        """Produce the OUTPUT text to return, honoring shadow vs enforce."""
        if not self.enforce:
            return ctx.answer  # shadow: never alter the response
        if final_action in (Action.BLOCK, Action.REFUSE):
            return canned_refusal(ctx.mode)
        # REDACT already applied to ctx.answer in the run loop; ALLOW passes through.
        return ctx.answer
