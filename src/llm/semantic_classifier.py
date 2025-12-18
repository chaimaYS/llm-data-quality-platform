"""
LLM-powered semantic column classification and PII detection.
Author: Chaima Yedes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.llm.client import LLMClient
from src.profiling.engine import ColumnProfile


class PIICategory(str, Enum):
    """PII sensitivity categories."""

    NONE = "none"
    LOW = "low"           # e.g., country, city
    MEDIUM = "medium"     # e.g., full name, email
    HIGH = "high"         # e.g., SSN, credit card
    CRITICAL = "critical" # e.g., password, biometric


@dataclass
class SemanticType:
    """Classification result for a single column."""

    column_name: str
    semantic_type: str
    confidence: float
    pii_detected: bool
    pii_category: PIICategory = PIICategory.NONE
    pii_types: list[str] = field(default_factory=list)
    suggested_validation: Optional[str] = None
    rationale: str = ""


_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_type": {
            "type": "string",
            "description": "Semantic type such as email, phone, ssn, currency, country_code, date_of_birth, ip_address, url, postal_code, full_name, etc.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "pii_detected": {"type": "boolean"},
        "pii_category": {
            "type": "string",
            "enum": ["none", "low", "medium", "high", "critical"],
        },
        "pii_types": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggested_validation": {
            "type": "string",
            "description": "A regex or rule suggestion for validating this column.",
        },
        "rationale": {"type": "string"},
    },
    "required": ["semantic_type", "confidence", "pii_detected", "pii_category"],
}


class SemanticColumnClassifier:
    """
    Uses an LLM to classify columns by semantic type and detect PII.

    Given a column name, its statistical profile, and a sample of values,
    the classifier produces a :class:`SemanticType` for each column.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def classify(
        self,
        column_name: str,
        profile: ColumnProfile,
        samples: list[Any],
    ) -> SemanticType:
        """Classify a single column."""
        prompt = self._build_prompt(column_name, profile, samples)
        result = self._llm.complete(prompt, schema=_CLASSIFICATION_SCHEMA)
        return self._parse_result(column_name, result)

    def classify_batch(
        self,
        columns: list[tuple[str, ColumnProfile, list[Any]]],
    ) -> list[SemanticType]:
        """Classify multiple columns in sequence."""
        return [
            self.classify(name, prof, samp) for name, prof, samp in columns
        ]

    def _build_prompt(
        self,
        column_name: str,
        profile: ColumnProfile,
        samples: list[Any],
    ) -> str:
        sample_str = json.dumps(samples[:20], default=str, indent=2)
        return f"""Analyze this database column and classify its semantic type.

Column Name: {column_name}
Data Type: {profile.dtype}
Null Count: {profile.null_count} / {profile.total_count} ({profile.null_pct}%)
Distinct Count: {profile.distinct_count} ({profile.distinct_pct}%)
Min Length: {profile.min_length}
Max Length: {profile.max_length}
Top Values: {json.dumps(profile.top_k[:5], default=str)}

Sample Values:
{sample_str}

Tasks:
1. Determine the semantic type (email, phone, ssn, currency, country_code, date_of_birth, ip_address, url, postal_code, full_name, credit_card, latitude, longitude, uuid, free_text, categorical, numeric_id, etc.)
2. Detect if this column contains PII (Personally Identifiable Information)
3. If PII, classify the sensitivity level (none/low/medium/high/critical)
4. Suggest a validation regex or rule if applicable
5. Provide brief rationale for your classification"""

    def _parse_result(self, column_name: str, data: dict[str, Any]) -> SemanticType:
        return SemanticType(
            column_name=column_name,
            semantic_type=data.get("semantic_type", "unknown"),
            confidence=float(data.get("confidence", 0.0)),
            pii_detected=bool(data.get("pii_detected", False)),
            pii_category=PIICategory(data.get("pii_category", "none")),
            pii_types=data.get("pii_types", []),
            suggested_validation=data.get("suggested_validation"),
            rationale=data.get("rationale", ""),
        )
