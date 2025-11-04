"""
Rules engine: loads rules from YAML config and evaluates them.
Author: Chaima Yedes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.rules.base import Dimension, Rule, RuleResult, Severity
from src.rules.builtin import (
    ForeignKeyCheck,
    FreshnessCheck,
    NullCheck,
    RangeCheck,
    RegexCheck,
    UniqueCheck,
)

logger = logging.getLogger(__name__)

_RULE_REGISTRY: dict[str, type[Rule]] = {
    "null_check": NullCheck,
    "unique_check": UniqueCheck,
    "range_check": RangeCheck,
    "regex_check": RegexCheck,
    "foreign_key_check": ForeignKeyCheck,
    "freshness_check": FreshnessCheck,
}


def register_rule(type_name: str, rule_class: type[Rule]) -> None:
    """Register a custom rule type so it can be loaded from YAML."""
    _RULE_REGISTRY[type_name] = rule_class


class RulesEngine:
    """
    Loads data quality rules from YAML configuration and evaluates
    them against pandas DataFrames.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules: list[Rule] = rules or []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RulesEngine":
        """
        Build a :class:`RulesEngine` from a YAML file.

        Expected format::

            rules:
              - type: null_check
                column: email
                severity: high
                threshold: 0.99
              - type: range_check
                column: age
                params:
                  min_value: 0
                  max_value: 150
        """
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        if raw is None or "rules" not in raw:
            raise ValueError(f"YAML at {path} must contain a top-level 'rules' key")

        rules: list[Rule] = []
        for entry in raw["rules"]:
            rule = cls._build_rule(entry)
            if rule is not None:
                rules.append(rule)
        logger.info("Loaded %d rules from %s", len(rules), path)
        return cls(rules=rules)

    @staticmethod
    def _build_rule(entry: dict[str, Any]) -> Rule | None:
        type_name = entry.get("type")
        if type_name not in _RULE_REGISTRY:
            logger.warning("Unknown rule type: %s  -- skipping", type_name)
            return None

        rule_cls = _RULE_REGISTRY[type_name]
        column = entry.get("column")
        severity = Severity(entry.get("severity", "medium"))
        threshold = float(entry.get("threshold", 1.0))
        name = entry.get("name")
        params = entry.get("params", {})

        kwargs: dict[str, Any] = {
            "column": column,
            "severity": severity,
            "threshold": threshold,
        }
        if name:
            kwargs["name"] = name

        if type_name == "range_check":
            kwargs["min_value"] = params.get("min_value")
            kwargs["max_value"] = params.get("max_value")
        elif type_name == "regex_check":
            kwargs["pattern"] = params.get("pattern", ".*")
        elif type_name == "foreign_key_check":
            kwargs["reference_values"] = set(params.get("reference_values", []))
        elif type_name == "freshness_check":
            kwargs["max_age_hours"] = params.get("max_age_hours", 24.0)

        return rule_cls(**kwargs)

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine at runtime."""
        self.rules.append(rule)

    def evaluate(self, df: pd.DataFrame) -> list[RuleResult]:
        """
        Run every registered rule against *df* and return the results.
        Rules that raise are logged and skipped.
        """
        results: list[RuleResult] = []
        for rule in self.rules:
            try:
                result = rule.evaluate(df)
                results.append(result)
                logger.debug(
                    "Rule %s: %s (pass_rate=%.4f)",
                    rule.name,
                    "PASS" if result.passed else "FAIL",
                    result.pass_rate,
                )
            except KeyError as exc:
                logger.error("Rule %s failed -- missing column: %s", rule.name, exc)
            except Exception:
                logger.exception("Rule %s raised an unexpected error", rule.name)
        return results

    def evaluate_by_dimension(
        self, df: pd.DataFrame
    ) -> dict[Dimension, list[RuleResult]]:
        """Run all rules and group results by dimension."""
        results = self.evaluate(df)
        grouped: dict[Dimension, list[RuleResult]] = {}
        for r in results:
            grouped.setdefault(r.dimension, []).append(r)
        return grouped
