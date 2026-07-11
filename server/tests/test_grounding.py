"""Grounding math + guard behavior with a stub embedder (no model load)."""
from src.guardrails.grounding import cosine_normalized, grounding_scores
from src.guardrails.base import Action, GuardrailContext, Stage
from src.guardrails.output_guards import GroundingGuard


def test_cosine_normalized():
    assert abs(cosine_normalized([1, 0, 0], [2, 0, 0]) - 1.0) < 1e-9  # un-normalized handled
    assert abs(cosine_normalized([1, 0], [0, 1])) < 1e-9
    assert cosine_normalized([0, 0], [1, 1]) == 0.0


class StubEmbedder:
    """Returns a fixed vector keyed on whether 'germany' appears."""
    def generate_embedding(self, text, input_type="search_document"):
        return [1.0, 0.0] if "germany" in text.lower() else [0.0, 1.0]


def _guard():
    return GroundingGuard(StubEmbedder(), sim_max_threshold=0.45,
                          sim_mean_threshold=0.30, min_answer_chars=20)


def test_grounded_answer_allowed():
    ctx = GuardrailContext(
        query="q", mode="text",
        chunks=[{"content": "The Poodle originates in Germany per the book."}],
        answer="The Poodle originates in Germany, a fact stated in the reference.",
    )
    d = _guard().check(ctx)
    assert d.action == Action.ALLOW and not d.triggered


def test_ungrounded_answer_refused():
    ctx = GuardrailContext(
        query="q", mode="text",
        chunks=[{"content": "The Poodle originates in Germany per the book."}],
        answer="The capital of France is Paris and the stock market rose today.",
    )
    d = _guard().check(ctx)
    assert d.action == Action.REFUSE and d.triggered


def test_short_answer_skipped():
    ctx = GuardrailContext(query="q", chunks=[{"content": "x"}], answer="Germany.")
    d = _guard().check(ctx)
    assert d.action == Action.ALLOW and d.reason == "skipped"


def test_generation_error_blocked():
    ctx = GuardrailContext(query="q", chunks=[{"content": "x"}],
                           answer="Error generating answer: boom")
    d = _guard().check(ctx)
    assert d.action == Action.BLOCK
