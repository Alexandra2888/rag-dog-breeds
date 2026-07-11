"""Toxicity / moderation check.

Provider-aware: with an OpenAI-compatible provider we use the moderation endpoint
(fast, free, key already present). In dev/ollama there is no moderation API, so an
optional deterministic wordlist is the only signal — weak, but keeps dev honest.
The moderation call is the only guard with a network hop; it uses a short timeout
and fails OPEN so it never blocks a request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.config import settings

# Small, obvious set for the dev fallback only — NOT a real toxicity model.
_DEV_WORDLIST = [
    r"\bkill yourself\b", r"\bk\W*y\W*s\b", r"\bidiot\b", r"\bmoron\b",
    r"\bslur\b", r"\bhate you\b", r"\bgo die\b",
]
_DEV_RE = re.compile("|".join(_DEV_WORDLIST), re.IGNORECASE)


@dataclass
class ToxResult:
    flagged: bool
    reason: str = ""
    score: Optional[float] = None


def moderate(text: str, provider: str, openai_client=None) -> ToxResult:
    """Return whether ``text`` is toxic. Fails open (not flagged) on any error."""
    if not text or not text.strip():
        return ToxResult(False)

    if provider == "openai" and openai_client is not None:
        try:
            resp = openai_client.moderations.create(
                model="omni-moderation-latest", input=text, timeout=5
            )
            result = resp.results[0]
            score = None
            try:
                score = max(result.category_scores.model_dump().values())
            except Exception:
                pass
            return ToxResult(bool(result.flagged), reason="openai_moderation", score=score)
        except Exception:
            return ToxResult(False, reason="moderation_error")  # fail open

    # Dev / ollama fallback.
    if settings.guardrails_toxicity_dev_wordlist:
        m = _DEV_RE.search(text)
        if m:
            return ToxResult(True, reason=f"dev_wordlist:{m.group(0)}", score=1.0)
    return ToxResult(False)
