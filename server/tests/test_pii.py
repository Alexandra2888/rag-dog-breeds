"""PII regex detection + redaction, including Luhn on credit cards."""
from src.guardrails.pii import detect_pii, redact, _luhn_ok


def _kinds(text):
    return sorted({s.kind for s in detect_pii(text)})


def test_email_phone():
    k = _kinds("reach me at john.doe@example.com or +1 415 555 2671")
    assert "EMAIL" in k and "PHONE" in k


def test_ssn_and_iban():
    assert "SSN" in _kinds("ssn 123-45-6789")
    assert "IBAN" in _kinds("account DE89370400440532013000")


def test_credit_card_luhn():
    # 4111111111111111 is a valid Luhn test number; grouped variants too.
    assert "CREDIT_CARD" in _kinds("card 4111 1111 1111 1111")
    assert _luhn_ok("4111111111111111")
    # A random 16-digit run that fails Luhn must NOT be flagged as a card.
    assert not _luhn_ok("1234567890123456")
    assert "CREDIT_CARD" not in _kinds("order number 1234 5678 9012 3456")


def test_benign_not_flagged():
    assert detect_pii("The Poodle weighs 20-30 kg and lives 12-15 years") == []


def test_redaction_replaces_and_labels():
    text = "mail john@x.com now"
    out = redact(text, detect_pii(text))
    assert "john@x.com" not in out and "[REDACTED_EMAIL]" in out
