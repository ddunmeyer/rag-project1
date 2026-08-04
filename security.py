# security.py
# -----------
# Input validation and data-protection checks for the RAG app.
#
# Validates at the system boundary — before user text reaches the vector store
# or an external LLM API. Covers two concerns in one module:
#
#   1. Security  — prompt injection, abuse, malformed input (Week 12)
#   2. Compliance — PII, secrets, optional topic guardrails (Week 17)
#
# OWASP LLM guidance: treat all user input as untrusted and validate early.

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500
REDACTION_TOKEN = "[REDACTED]"


class IssueCategory(str, Enum):
    """Why input was rejected or flagged."""

    EMPTY = "empty"
    TOO_LONG = "too_long"
    INJECTION = "injection"
    PII = "pii"
    SECRET = "secret"
    RESTRICTED_TOPIC = "restricted_topic"


USER_MESSAGES: dict[IssueCategory, str] = {
    IssueCategory.EMPTY: "Please enter a question before submitting.",
    IssueCategory.TOO_LONG: (
        f"Your query is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
    ),
    IssueCategory.INJECTION: "Your query contains content that cannot be processed.",
    IssueCategory.PII: (
        "Your message appears to contain personal information (such as an email, "
        "phone number, or ID number). Remove sensitive details before submitting."
    ),
    IssueCategory.SECRET: (
        "Your message appears to contain a secret or API key. Never paste credentials "
        "into chat applications."
    ),
    IssueCategory.RESTRICTED_TOPIC: (
        "Your message matches a restricted topic configured for this application."
    ),
}


# Prompt-injection phrases (lowercase matching after normalization).
INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore your previous instructions",
    "disregard previous instructions",
    "forget your instructions",
    "override instructions",
    "you are now",
    "act as",
    "pretend to be",
    "system prompt",
    "developer mode",
    "jailbreak",
    "do anything now",
    "dan mode",
)

# Compiled regex patterns for sensitive-data detection.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "gemini_key": re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic_credential": re.compile(
        r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*\S{8,}\b"
    ),
}

RESTRICTED_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "medical_advice": re.compile(
        r"(?i)\b(diagnose|prescription|medical advice|what medicine should i take)\b"
    ),
}

# Backward-compatible alias used by get_feature_status().
BLOCKED_PATTERNS = list(INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ValidationResult(NamedTuple):
    """Structured output from validate_input()."""

    ok: bool
    message: str
    category: IssueCategory | None = None


@dataclass
class SecurityFinding:
    """One detected issue in scanned text."""

    category: IssueCategory
    label: str
    matched_text: str


@dataclass
class SecurityReport:
    """Full scan result — useful for logging and tests."""

    ok: bool
    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.ok or not self.findings:
            return ""
        return USER_MESSAGES[self.findings[0].category]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize unicode and collapse whitespace for consistent matching."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def _find_injection(text: str) -> SecurityFinding | None:
    lowered = text.lower()
    for phrase in INJECTION_PATTERNS:
        if phrase in lowered:
            return SecurityFinding(
                category=IssueCategory.INJECTION,
                label=phrase,
                matched_text=phrase,
            )
    return None


def _find_regex_matches(
    text: str,
    patterns: dict[str, re.Pattern[str]],
    category: IssueCategory,
) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for label, pattern in patterns.items():
        for match in pattern.finditer(text):
            findings.append(
                SecurityFinding(
                    category=category,
                    label=label,
                    matched_text=match.group(0),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_input(query: str) -> str:
    """Trim whitespace from user input."""
    return query.strip()


def scan_input(
    query: str,
    *,
    check_injection: bool = True,
    check_sensitive_data: bool = True,
    check_restricted_topics: bool = False,
) -> SecurityReport:
    """
    Run all enabled checks and return a structured report.

    Checks run in fail-fast order: empty → length → injection → secrets → PII → topics.
    """
    findings: list[SecurityFinding] = []
    text = normalize_text(query) if query else ""

    if not text:
        return SecurityReport(ok=False, findings=[SecurityFinding(IssueCategory.EMPTY, "empty", "")])

    if len(text) > MAX_QUERY_LENGTH:
        return SecurityReport(
            ok=False,
            findings=[SecurityFinding(IssueCategory.TOO_LONG, "length", str(len(text)))],
        )

    if check_injection:
        injection = _find_injection(text)
        if injection:
            return SecurityReport(ok=False, findings=[injection])

    if check_sensitive_data:
        findings.extend(_find_regex_matches(text, SECRET_PATTERNS, IssueCategory.SECRET))
        if findings:
            return SecurityReport(ok=False, findings=findings)

        findings.extend(_find_regex_matches(text, PII_PATTERNS, IssueCategory.PII))
        if findings:
            return SecurityReport(ok=False, findings=findings)

    if check_restricted_topics:
        findings.extend(
            _find_regex_matches(text, RESTRICTED_TOPIC_PATTERNS, IssueCategory.RESTRICTED_TOPIC)
        )
        if findings:
            return SecurityReport(ok=False, findings=findings)

    return SecurityReport(ok=True)


def validate_input(
    query: str,
    *,
    check_injection: bool = True,
    check_sensitive_data: bool = True,
    check_restricted_topics: bool = False,
) -> tuple[bool, str]:
    """
    Validate user input before RAG processing.

    Returns:
        (True, "") — safe to proceed
        (False, message) — block and show message to the user
    """
    report = scan_input(
        query,
        check_injection=check_injection,
        check_sensitive_data=check_sensitive_data,
        check_restricted_topics=check_restricted_topics,
    )
    if report.ok:
        return True, ""
    return False, report.message


def validate_input_detailed(
    query: str,
    *,
    check_injection: bool = True,
    check_sensitive_data: bool = True,
    check_restricted_topics: bool = False,
) -> ValidationResult:
    """Like validate_input(), but returns category metadata for logging/tests."""
    report = scan_input(
        query,
        check_injection=check_injection,
        check_sensitive_data=check_sensitive_data,
        check_restricted_topics=check_restricted_topics,
    )
    if report.ok:
        return ValidationResult(True, "", None)
    category = report.findings[0].category if report.findings else None
    return ValidationResult(False, report.message, category)


def redact_sensitive_text(text: str) -> str:
    """
    Best-effort redaction before logs or debug output.

    Not a substitute for full anonymization in production systems.
    """
    redacted = text
    for pattern in {**PII_PATTERNS, **SECRET_PATTERNS}.values():
        redacted = pattern.sub(REDACTION_TOKEN, redacted)
    return redacted


def get_security_notice() -> str:
    """Short disclaimer for the UI."""
    return (
        "This demo sends questions to an external AI provider. Do not submit passwords, "
        "API keys, or personal information."
    )
