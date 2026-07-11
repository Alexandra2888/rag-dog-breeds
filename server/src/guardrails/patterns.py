"""Compiled patterns for the deterministic guards.

Owned here (not imported from ``evals/``) so the runtime layer has no dependency
on test data. The injection patterns mirror the ``jailbreak`` / ``prompt_injection``
categories in the adversarial suite; keep the two roughly in sync when either grows.
"""
from __future__ import annotations

import re

# Prompt-injection / jailbreak signatures. Each is a substring/regex the attacker
# uses to override instructions, exfiltrate the prompt, or hijack the role.
_INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above) instructions",
    r"disregard (?:the )?(?:above|previous|prior|context|instructions)",
    r"forget (?:you are|that you are|your instructions|everything)",
    r"you are (?:now )?(?:dan|jailbroken|an? unrestricted)",
    r"\bdo anything now\b",
    r"pretend (?:the )?(?:context|instructions|rules)",
    r"(?:reveal|print|show|repeat|output) (?:me )?(?:your |the )?(?:system prompt|instructions|prompt|the text above)",
    r"print all text above",
    r"append ['\"]?pwned",
    r"<<<.*?>>>",
    r"\bend context\b",
    r"\bnew instruction[s]?\b",
    r"^\s*system\s*:",
    r"\]\]\}>",
    r"act as (?:a |an )?(?!.*dog)",  # "act as a <non-dog role>"
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def match_injection(text: str) -> str | None:
    """Return the matched substring if the text looks like an injection, else None."""
    if not text:
        return None
    m = INJECTION_RE.search(text)
    return m.group(0) if m else None


# Strings an answer must never echo: system-prompt fragments and injection markers.
# Mirrors the adversarial suite's `must_not_contain` semantics but owned by the guard.
LEAK_STRINGS = [
    "based on the following context",
    "you are a friendly voice assistant",
    "[source:",
    "context:",
    "pwned",
    "system:",
    "system prompt",
]
