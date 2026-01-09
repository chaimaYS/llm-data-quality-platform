"""
LLM-powered rule proposal engine.
Author: Chaima Yedes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from src.llm.client import LLMClient
from src.profiling.engine import ColumnProfile, ProfileResult


@dataclass
class ProposedRule:
    """A candidate data quality rule proposed by the LLM."""

    rule_type: str
    column: str
    params: dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    dimension: str = "validity"
    rationale: str = ""
    confidence: float = 0.0

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to the format expected by :class:`RulesEngine`."""
        entry: dict[str, Any] = {
            "type": self.rule_type,
            "column": self.column,
            "severity": self.severity,
        }
        if self.params:
            entry["params"] = self.params
        return entry


_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_type": {"type": "string"},
                    "column": {"type": "string"},
                    "params": {"type": "object"},
                    "severity": {"type": "string"},
                    "dimension": {"type": "string"},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["rule_type", "column", "rationale"],
            },
        }
    },
    "required": ["rules"],
}


class RuleProposer:
    """
    Takes a dataset schema and profile, then uses an LLM to propose
    candidate data quality rules.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def propose(
        self,
        profile: ProfileResult,
        table_name: str = "unknown",
        context: Optional[str] = None,
    ) -> list[ProposedRule]:
        """
        Generate rule proposals for the profiled dataset.

        Parameters
        ----------
        profile : ProfileResult
            Output of the profiling engine.
        table_name : str
            Name of the table or dataset.
        context : str, optional
            Additional business context for smarter proposals.
        """
        prompt = self._build_prompt(profile, table_name, context)
        result = self._llm.complete(prompt, schema=_PROPOSAL_SCHEMA)
        return self._parse_result(result)

    def propose_as_yaml(
        self,
        profile: ProfileResult,
        table_name: str = "unknown",
        context: Optional[str] = None,
    ) -> str:
        """Return proposed rules formatted as YAML."""
        proposals = self.propose(profile, table_name, context)
        rules_list = [p.to_yaml_dict() for p in proposals]
        return yaml.dump({"rules": rules_list}, default_flow_style=False, sort_keys=False)

    def _build_prompt(
        self,
        profile: ProfileResult,
        table_name: str,
        context: Optional[str],
    ) -> str:
        col_summaries: list[str] = []
        for name, cp in profile.columns.items():
            summary = (
                f"  - {name}: type={cp.dtype}, nulls={cp.null_count}/{cp.total_count} "
                f"({cp.null_pct}%), distinct={cp.distinct_count} ({cp.distinct_pct}%), "
                f"unique={cp.is_unique}"
            )
            if cp.mean is not None:
                summary += f", mean={cp.mean:.2f}, std={cp.std:.2f}"
            if cp.min_length is not None:
                summary += f", len=[{cp.min_length},{cp.max_length}]"
            if cp.top_k:
                top_vals = [str(t["value"]) for t in cp.top_k[:3]]
                summary += f", top_values={top_vals}"
            if cp.patterns:
                top_pats = [p["pattern"] for p in cp.patterns[:3]]
                summary += f", patterns={top_pats}"
            col_summaries.append(summary)

        columns_text = "\n".join(col_summaries)
        context_block = f"\nBusiness Context: {context}" if context else ""

        return f"""Analyze this dataset profile and propose data quality rules.

Table: {table_name}
Rows: {profile.row_count}
Columns: {profile.column_count}
Duplicate Rows: {profile.duplicate_row_count}
{context_block}

Column Profiles:
{columns_text}

Available rule types:
- null_check: Check column is not null. Params: none.
- unique_check: Check column values are unique. Params: none.
- range_check: Check numeric values in range. Params: min_value, max_value.
- regex_check: Check string matches pattern. Params: pattern.
- freshness_check: Check timestamp is recent. Params: max_age_hours.
- foreign_key_check: Check referential integrity. Params: reference_values.

For each proposed rule, include:
- rule_type: one of the types above
- column: column name
- params: rule-specific parameters
- severity: critical / high / medium / low
- dimension: completeness / uniqueness / validity / consistency / timeliness / accuracy / integrity / conformity
- rationale: why this rule makes sense
- confidence: 0.0 to 1.0 how confident you are"""

    def _parse_result(self, data: dict[str, Any]) -> list[ProposedRule]:
        proposals: list[ProposedRule] = []
        for entry in data.get("rules", []):
            proposals.append(
                ProposedRule(
                    rule_type=entry.get("rule_type", "null_check"),
                    column=entry.get("column", ""),
                    params=entry.get("params", {}),
                    severity=entry.get("severity", "medium"),
                    dimension=entry.get("dimension", "validity"),
                    rationale=entry.get("rationale", ""),
                    confidence=float(entry.get("confidence", 0.0)),
                )
            )
        return proposals
