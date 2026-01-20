"""
LLM-powered anomaly explainer for data quality failures.
Author: Chaima Yedes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import LLMClient
from src.rules.base import RuleResult


@dataclass
class AnomalyExplanation:
    """Human-readable explanation of a data quality failure."""

    rule_name: str
    summary: str
    root_cause: str
    impact: str
    suggested_fix: str
    severity_assessment: str
    affected_rows_estimate: int = 0
    tags: list[str] = field(default_factory=list)


_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One-sentence plain-English summary of the anomaly.",
        },
        "root_cause": {
            "type": "string",
            "description": "Most likely root cause of the failure.",
        },
        "impact": {
            "type": "string",
            "description": "Downstream impact if left unresolved.",
        },
        "suggested_fix": {
            "type": "string",
            "description": "Concrete remediation step.",
        },
        "severity_assessment": {
            "type": "string",
            "description": "Contextual severity assessment.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Categorization tags (e.g., 'schema_drift', 'upstream_issue').",
        },
    },
    "required": ["summary", "root_cause", "impact", "suggested_fix"],
}


class AnomalyExplainer:
    """
    Takes failing rows and the associated rule, then asks an LLM to
    produce a plain-English explanation of the anomaly along with
    likely root causes and remediation suggestions.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def explain(
        self,
        result: RuleResult,
        failing_rows: list[dict[str, Any]] | None = None,
        dataset_context: str = "",
    ) -> AnomalyExplanation:
        """
        Generate an explanation for a rule failure.

        Parameters
        ----------
        result : RuleResult
            The failing rule result.
        failing_rows : list[dict], optional
            Sample of rows that failed the rule. Falls back to
            ``result.failing_sample`` if not provided.
        dataset_context : str
            Optional description of the dataset for richer explanations.
        """
        rows = failing_rows or result.failing_sample
        prompt = self._build_prompt(result, rows, dataset_context)
        data = self._llm.complete(prompt, schema=_EXPLANATION_SCHEMA)
        return self._parse(result.rule_name, result.fail_count, data)

    def explain_batch(
        self,
        results: list[RuleResult],
        dataset_context: str = "",
    ) -> list[AnomalyExplanation]:
        """Explain all failing rules in a result set."""
        return [
            self.explain(r, dataset_context=dataset_context)
            for r in results
            if not r.passed
        ]

    def _build_prompt(
        self,
        result: RuleResult,
        failing_rows: list[dict[str, Any]],
        context: str,
    ) -> str:
        rows_json = json.dumps(failing_rows[:10], default=str, indent=2)
        context_block = f"\nDataset Context: {context}" if context else ""

        return f"""A data quality rule has failed. Analyze the failure and explain it in plain English.

Rule: {result.rule_name}
Dimension: {result.dimension.value}
Severity: {result.severity.value}
Column: {result.column or 'N/A'}
Details: {result.details}

Pass Rate: {result.pass_rate:.4f} ({result.pass_count} passed, {result.fail_count} failed out of {result.total_count})
Threshold: {result.threshold}
{context_block}

Sample of Failing Rows:
{rows_json}

Provide:
1. A concise summary of what went wrong
2. The most likely root cause
3. The downstream impact if this is not fixed
4. A concrete suggested fix or remediation step
5. Your assessment of the actual severity in context
6. Categorization tags (e.g., data_entry_error, schema_drift, upstream_issue, etl_bug, null_propagation, encoding_error)"""

    def _parse(
        self,
        rule_name: str,
        fail_count: int,
        data: dict[str, Any],
    ) -> AnomalyExplanation:
        return AnomalyExplanation(
            rule_name=rule_name,
            summary=data.get("summary", "No summary available."),
            root_cause=data.get("root_cause", "Unknown."),
            impact=data.get("impact", "Unknown impact."),
            suggested_fix=data.get("suggested_fix", "Manual review recommended."),
            severity_assessment=data.get("severity_assessment", ""),
            affected_rows_estimate=fail_count,
            tags=data.get("tags", []),
        )
