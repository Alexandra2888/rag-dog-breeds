"""PII detection + redaction.

Default backend is pure regex (zero deps, always-on). An optional Presidio backend
(``GUARDRAILS_PII_BACKEND=presidio``) is lazy-imported — it is richer but pulls
spaCy (~500MB+) and is NOT baked into the prod image, so it stays off by default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.config import settings


@dataclass
class PiiSpan:
    kind: str
    start: int
    end: int
    text: str


# --- regex detectors ------------------------------------------------------- #

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone: optional +, groups of digits with space/dot/dash separators, 9–15 digits.
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Credit-card-shaped: 13–19 digits, optionally grouped by space/dash. Luhn-checked.
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — cuts false positives on random long digit runs."""
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) < 13:
        return False
    total, parity = 0, len(ds) % 2
    for i, d in enumerate(ds):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _regex_detect(text: str) -> List[PiiSpan]:
    spans: List[PiiSpan] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append(PiiSpan("EMAIL", m.start(), m.end(), m.group()))
    for m in _SSN_RE.finditer(text):
        spans.append(PiiSpan("SSN", m.start(), m.end(), m.group()))
    for m in _IBAN_RE.finditer(text):
        spans.append(PiiSpan("IBAN", m.start(), m.end(), m.group()))
    for m in _CC_RE.finditer(text):
        if _luhn_ok(m.group()):
            spans.append(PiiSpan("CREDIT_CARD", m.start(), m.end(), m.group()))
    for m in _PHONE_RE.finditer(text):
        # Avoid double-flagging digits already covered by SSN/CC/IBAN.
        if not any(s.start <= m.start() < s.end for s in spans):
            digits = re.sub(r"\D", "", m.group())
            if 9 <= len(digits) <= 15:
                spans.append(PiiSpan("PHONE", m.start(), m.end(), m.group()))
    return spans


# --- optional Presidio backend --------------------------------------------- #

_presidio_analyzer = None


def _get_presidio():
    global _presidio_analyzer
    if _presidio_analyzer is None:
        from presidio_analyzer import AnalyzerEngine  # lazy: heavy import
        _presidio_analyzer = AnalyzerEngine()
    return _presidio_analyzer


def _presidio_detect(text: str) -> List[PiiSpan]:
    results = _get_presidio().analyze(text=text, language="en")
    return [PiiSpan(r.entity_type, r.start, r.end, text[r.start:r.end]) for r in results]


# --- public API ------------------------------------------------------------ #

def detect_pii(text: str) -> List[PiiSpan]:
    """Detect PII spans using the configured backend (regex by default)."""
    if not text:
        return []
    if settings.guardrails_pii_backend == "presidio":
        try:
            return _presidio_detect(text)
        except Exception:
            # Presidio unavailable (not installed / model missing) → regex fallback.
            return _regex_detect(text)
    return _regex_detect(text)


def redact(text: str, spans: List[PiiSpan]) -> str:
    """Replace each span with ``[REDACTED_<KIND>]`` (right-to-left to keep offsets)."""
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        text = text[: span.start] + f"[REDACTED_{span.kind}]" + text[span.end:]
    return text
