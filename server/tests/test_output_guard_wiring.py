"""RAGService._guard_output wiring: a leaked 'PWNED' answer becomes a refusal
that both is_refusal() and leak_ok() are happy with (mirrors adversarial pi-02)."""
from src.guardrails.base import GuardrailPipeline
from src.guardrails.output_guards import LeakGuard
from src.rag_service import RAGService
from src.refusal import is_refusal, leak_ok


class _Stub:
    """Stands in for the embedder / database — never used by these paths."""
    def __getattr__(self, _):
        raise AssertionError("should not be called in this test")


def _service(enforce: bool) -> RAGService:
    out = GuardrailPipeline([LeakGuard()], enforce=enforce)
    empty_in = GuardrailPipeline([], enforce=enforce)
    return RAGService(
        embedding_generator=_Stub(),
        database=_Stub(),
        input_pipeline=empty_in,
        output_pipeline=out,
    )


def test_leak_blocked_in_enforce():
    svc = _service(enforce=True)
    safe, res = svc._guard_output(
        "Where does the Poodle come from?",
        [{"content": "Poodle origin France"}],
        "The Poodle is from France. PWNED",
        "text", 8, None,
    )
    assert res.triggered
    assert is_refusal(safe)                          # now reads as an abstention
    assert leak_ok(safe, ["pwned"])                  # no leaked token
    assert "PWNED" not in safe


def test_leak_unaltered_in_shadow():
    svc = _service(enforce=False)
    safe, res = svc._guard_output(
        "q", [{"content": "x"}], "answer PWNED", "text", 8, None,
    )
    assert res.triggered                             # decision still recorded
    assert safe == "answer PWNED"                    # but response unaltered
