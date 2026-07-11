"""Input-stage guards: screen the user query before any retrieval/generation."""
from __future__ import annotations

from src.guardrails.base import Action, GuardrailContext, GuardrailDecision, Stage
from src.guardrails.patterns import match_injection
from src.guardrails.pii import detect_pii
from src.guardrails.toxicity import moderate


class PromptInjectionGuard:
    """Deterministic screen for jailbreak / prompt-injection attempts."""
    name = "prompt_injection"
    stage = Stage.INPUT

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        hit = match_injection(ctx.query)
        if hit:
            return GuardrailDecision(
                self.name, self.stage, Action.BLOCK, True,
                reason=f"injection_pattern:{hit[:40]}", score=1.0,
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False)


class PiiInputGuard:
    """Detect PII in the query. Logs by default; blocks only if configured."""
    name = "pii_input"
    stage = Stage.INPUT

    def __init__(self, block: bool = False):
        self.block = block

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        spans = detect_pii(ctx.query)
        if spans:
            kinds = ",".join(sorted({s.kind for s in spans}))
            action = Action.BLOCK if self.block else Action.ALLOW
            return GuardrailDecision(
                self.name, self.stage, action, True, reason=f"pii:{kinds}",
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False)


class ToxicityGuard:
    """Moderation check. Stage-parameterized so it can guard input or output."""

    def __init__(self, stage: Stage, provider: str, openai_client=None):
        self.name = f"toxicity_{stage.value}"
        self.stage = stage
        self.provider = provider
        self.openai_client = openai_client

    def check(self, ctx: GuardrailContext) -> GuardrailDecision:
        text = ctx.answer if self.stage == Stage.OUTPUT else ctx.query
        res = moderate(text or "", self.provider, self.openai_client)
        if res.flagged:
            return GuardrailDecision(
                self.name, self.stage, Action.BLOCK, True,
                reason=res.reason, score=res.score,
            )
        return GuardrailDecision(self.name, self.stage, Action.ALLOW, False)
