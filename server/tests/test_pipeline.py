"""Pipeline aggregation, shadow vs enforce, and fail-open behavior."""
from src.guardrails.base import (
    Action, GuardrailContext, GuardrailDecision, GuardrailPipeline, Stage, canned_refusal,
)


class FixedGuard:
    def __init__(self, name, stage, action, triggered, redacted_text=None):
        self.name, self.stage = name, stage
        self._d = GuardrailDecision(name, stage, action, triggered, redacted_text=redacted_text)

    def check(self, ctx):
        return self._d


class BoomGuard:
    name, stage = "boom", Stage.INPUT

    def check(self, ctx):
        raise ValueError("kaboom")


def test_strongest_action_wins():
    guards = [
        FixedGuard("a", Stage.OUTPUT, Action.REDACT, True, redacted_text="redacted"),
        FixedGuard("b", Stage.OUTPUT, Action.BLOCK, True),
    ]
    res = GuardrailPipeline(guards, enforce=True).run(
        GuardrailContext(query="q", mode="text", answer="original")
    )
    assert res.final_action == Action.BLOCK
    assert res.output_text == canned_refusal("text")


def test_shadow_does_not_alter_output():
    g = FixedGuard("leak", Stage.OUTPUT, Action.BLOCK, True)
    res = GuardrailPipeline([g], enforce=False).run(
        GuardrailContext(query="q", mode="text", answer="original")
    )
    assert res.triggered is True
    assert res.output_text == "original"  # unaltered in shadow


def test_enforce_refuse_returns_canned():
    g = FixedGuard("scope", Stage.OUTPUT, Action.REFUSE, True)
    res = GuardrailPipeline([g], enforce=True).run(
        GuardrailContext(query="q", mode="voice", answer="original")
    )
    assert res.output_text == canned_refusal("voice")


def test_redaction_chains():
    g = FixedGuard("pii", Stage.OUTPUT, Action.REDACT, True, redacted_text="[REDACTED_EMAIL]")
    res = GuardrailPipeline([g], enforce=True).run(
        GuardrailContext(query="q", mode="text", answer="mail me a@b.com")
    )
    assert res.output_text == "[REDACTED_EMAIL]"


def test_guard_error_fails_open():
    res = GuardrailPipeline([BoomGuard()], enforce=True).run(GuardrailContext(query="q"))
    assert res.final_action == Action.ALLOW and not res.triggered


def test_input_stage_output_text_is_none():
    g = FixedGuard("inj", Stage.INPUT, Action.BLOCK, True)
    res = GuardrailPipeline([g], enforce=True).run(GuardrailContext(query="q"))  # answer=None
    assert res.output_text is None and res.final_action == Action.BLOCK
