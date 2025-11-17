"""
Dimension-based scoring engine for data quality assessments.
Author: Chaima Yedes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.rules.base import Dimension, RuleResult


_DEFAULT_WEIGHTS: dict[str, float] = {
    "completeness": 0.20,
    "uniqueness": 0.15,
    "validity": 0.15,
    "consistency": 0.10,
    "timeliness": 0.10,
    "accuracy": 0.10,
    "integrity": 0.10,
    "conformity": 0.10,
}


@dataclass
class DimensionScore:
    """Score for a single data quality dimension."""

    dimension: Dimension
    weight: float
    rule_count: int
    pass_count: int
    fail_count: int
    pass_rate: float
    weighted_score: float
    rules: list[str] = field(default_factory=list)


@dataclass
class DatasetScore:
    """Overall quality score for a dataset."""

    overall_score: float
    grade: str
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    total_rules: int = 0
    total_passed: int = 0
    total_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "total_rules": self.total_rules,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "dimensions": {
                name: {
                    "weight": ds.weight,
                    "pass_rate": ds.pass_rate,
                    "weighted_score": ds.weighted_score,
                    "rule_count": ds.rule_count,
                    "rules": ds.rules,
                }
                for name, ds in self.dimensions.items()
            },
        }


def _score_to_grade(score: float) -> str:
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


class DimensionScorer:
    """
    Maps rule results to quality dimensions, computes weighted pass-rates
    per dimension, and produces an overall dataset quality score.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = weights or dict(_DEFAULT_WEIGHTS)
        self._normalize_weights()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DimensionScorer":
        """Load dimension weights from a YAML config file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        weights = raw.get("dimensions", raw.get("weights", {}))
        flat: dict[str, float] = {}
        for key, val in weights.items():
            if isinstance(val, dict):
                flat[key] = float(val.get("weight", 0.0))
            else:
                flat[key] = float(val)
        return cls(weights=flat)

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, results: list[RuleResult]) -> DatasetScore:
        """Compute dimension scores and an overall score."""
        by_dim: dict[str, list[RuleResult]] = {}
        for r in results:
            by_dim.setdefault(r.dimension.value, []).append(r)

        dimension_scores: dict[str, DimensionScore] = {}
        weighted_total = 0.0
        weight_sum_active = 0.0

        for dim_name, weight in self.weights.items():
            dim_results = by_dim.get(dim_name, [])
            if not dim_results:
                dimension_scores[dim_name] = DimensionScore(
                    dimension=Dimension(dim_name),
                    weight=weight,
                    rule_count=0,
                    pass_count=0,
                    fail_count=0,
                    pass_rate=1.0,
                    weighted_score=weight,
                    rules=[],
                )
                weighted_total += weight
                weight_sum_active += weight
                continue

            pass_rates = [r.pass_rate for r in dim_results]
            avg_pass_rate = sum(pass_rates) / len(pass_rates)
            passed = sum(1 for r in dim_results if r.passed)
            failed = len(dim_results) - passed

            ws = weight * avg_pass_rate
            weighted_total += ws
            weight_sum_active += weight

            dimension_scores[dim_name] = DimensionScore(
                dimension=Dimension(dim_name),
                weight=weight,
                rule_count=len(dim_results),
                pass_count=passed,
                fail_count=failed,
                pass_rate=round(avg_pass_rate, 6),
                weighted_score=round(ws, 6),
                rules=[r.rule_name for r in dim_results],
            )

        overall = weighted_total / weight_sum_active if weight_sum_active else 1.0
        overall = round(overall, 6)

        total_passed = sum(1 for r in results if r.passed)
        return DatasetScore(
            overall_score=overall,
            grade=_score_to_grade(overall),
            dimensions=dimension_scores,
            total_rules=len(results),
            total_passed=total_passed,
            total_failed=len(results) - total_passed,
        )
