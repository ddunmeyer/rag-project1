# compliance.py
# --------------
# Week 18 — metadata tagging and automated redaction for compliance-aware RAG.
#
# Labels data with sensitivity, type, and source metadata so downstream controls
# (logging, model calls, storage) can treat sensitive information differently.
# Redaction is applied before logs, debug output, and optional pre-model sends.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from config import ENABLE_COMPLIANCE_LOGGING
from security import (
    PII_PATTERNS,
    REDACTION_TOKEN,
    SECRET_PATTERNS,
    redact_sensitive_text,
)

COMPLIANCE_VERSION = "week18-v1"

logger = logging.getLogger("rag.compliance")

if ENABLE_COMPLIANCE_LOGGING and not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataType(str, Enum):
    OPERATIONAL = "operational"
    PII = "pii"
    PHI = "phi"
    FINANCIAL = "financial"
    CREDENTIAL = "credential"


class DataSource(str, Enum):
    USER_INPUT = "user_input"
    DOCUMENT = "document"
    RETRIEVED = "retrieved"
    MODEL_OUTPUT = "model_output"


# Patterns beyond the Week 12/17 security module.
PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "medical_record": re.compile(
        r"(?i)\b(patient id|diagnosis|prescription|medical record|hipaa)\b"
    ),
    "health_detail": re.compile(
        r"(?i)\b(blood pressure|medical history|symptoms include|treatment plan)\b"
    ),
}

FINANCIAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "account_number": re.compile(r"(?i)\b(account|routing)\s*(number|#)?\s*[:#]?\s*\d{6,}\b"),
    "salary": re.compile(r"(?i)\b(salary|compensation|payroll)\s*[:=]?\s*\$?\d"),
}


@dataclass
class DataMetadata:
    """Compliance labels attached to text at pipeline boundaries."""

    sensitivity: Sensitivity
    data_type: DataType
    source: DataSource
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "compliance_version": COMPLIANCE_VERSION,
            "sensitivity": self.sensitivity.value,
            "data_type": self.data_type.value,
            "source": self.source.value,
            "tags": self.tags,
        }


def _pattern_labels(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [label for label, pattern in patterns.items() if pattern.search(text)]


def classify_text(text: str, source: DataSource) -> DataMetadata:
    """
    Assign metadata tags based on pattern matches.

    Not a full data-classification engine — consistent labeling for demo SOC 2-style
    controls and audit trails.
    """
    tags: list[str] = []
    sensitivity = Sensitivity.PUBLIC
    data_type = DataType.OPERATIONAL

    secret_labels = _pattern_labels(text, SECRET_PATTERNS)
    if secret_labels:
        sensitivity = Sensitivity.RESTRICTED
        data_type = DataType.CREDENTIAL
        tags.extend(f"secret:{label}" for label in secret_labels)

    pii_labels = _pattern_labels(text, PII_PATTERNS)
    if pii_labels:
        sensitivity = Sensitivity.CONFIDENTIAL
        data_type = DataType.PII
        tags.extend(f"pii:{label}" for label in pii_labels)

    phi_labels = _pattern_labels(text, PHI_PATTERNS)
    if phi_labels:
        sensitivity = Sensitivity.RESTRICTED
        data_type = DataType.PHI
        tags.extend(f"phi:{label}" for label in phi_labels)

    financial_labels = _pattern_labels(text, FINANCIAL_PATTERNS)
    if financial_labels and sensitivity == Sensitivity.PUBLIC:
        sensitivity = Sensitivity.CONFIDENTIAL
        data_type = DataType.FINANCIAL
        tags.extend(f"financial:{label}" for label in financial_labels)

    if source == DataSource.DOCUMENT and sensitivity == Sensitivity.PUBLIC:
        tags.append("corpus:tech_docs")

    metadata = DataMetadata(
        sensitivity=sensitivity,
        data_type=data_type,
        source=source,
        tags=tags,
    )
    metadata.tags.append(f"source:{source.value}")
    return metadata


def tag_document(text: str, doc_index: int) -> dict:
    """Metadata for knowledge-base documents at ingest time."""
    metadata = classify_text(text, DataSource.DOCUMENT)
    metadata.tags.append(f"doc_index:{doc_index}")
    return metadata.to_dict()


def tag_user_input(text: str) -> DataMetadata:
    return classify_text(text, DataSource.USER_INPUT)


def tag_retrieved_chunk(text: str, rank: int | None = None) -> DataMetadata:
    metadata = classify_text(text, DataSource.RETRIEVED)
    if rank is not None:
        metadata.tags.append(f"retrieval_rank:{rank}")
    return metadata


def tag_model_output(text: str) -> DataMetadata:
    return classify_text(text, DataSource.MODEL_OUTPUT)


def redact_for_log(text: str) -> str:
    """Mask sensitive values before writing to logs or analytics."""
    redacted = redact_sensitive_text(text)
    for pattern in {**PHI_PATTERNS, **FINANCIAL_PATTERNS}.values():
        redacted = pattern.sub(REDACTION_TOKEN, redacted)
    return redacted


def redact_for_display(text: str) -> str:
    """Safe version of text for debug panels and error messages."""
    return redact_for_log(text)


def metadata_summary(items: list[DataMetadata]) -> dict:
    """Aggregate tag summary for pipeline responses and UI."""
    if not items:
        return {"count": 0, "max_sensitivity": Sensitivity.PUBLIC.value}

    order = [
        Sensitivity.PUBLIC,
        Sensitivity.INTERNAL,
        Sensitivity.CONFIDENTIAL,
        Sensitivity.RESTRICTED,
    ]
    max_sensitivity = max(items, key=lambda m: order.index(m.sensitivity)).sensitivity
    data_types = sorted({m.data_type.value for m in items})
    return {
        "count": len(items),
        "max_sensitivity": max_sensitivity.value,
        "data_types": data_types,
    }


def log_compliance_event(event: str, text: str, metadata: DataMetadata | None = None) -> None:
    """Write a redacted audit log entry. Raw sensitive text is never logged."""
    if not ENABLE_COMPLIANCE_LOGGING:
        return

    payload = metadata.to_dict() if metadata else {}
    logger.info(
        "event=%s metadata=%s text=%s",
        event,
        payload,
        redact_for_log(text[:500]),
    )


def get_compliance_notice() -> str:
    """Short UI notice about data handling."""
    return (
        "Compliance mode: user input is validated, tagged with metadata, and redacted in logs. "
        "Do not submit real personal, financial, or health information."
    )
