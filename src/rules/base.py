"""
Abstract rule definitions and result types for data quality checks.
Author: Chaima Yedes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class Severity(str, Enum):
    """How critical a rule failure is."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Dimension(str, Enum):
    """Standard data quality dimension."""

    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    ACCURACY = "accuracy"
    INTEGRITY = "integrity"
    CONFORMITY = "conformity"


@dataclass
class RuleResult:
    """Outcome of evaluating a single rule against a dataset."""

    rule_name: str
    dimension: Dimension
    severity: Severity
    pass_count: int
    fail_count: int
    total_count: int
    pass_rate: float
    passed: bool
    threshold: float = 1.0
    column: Optional[str] = None
    details: str = ""
    failing_sample: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        return 1.0 - self.pass_rate


class Rule(ABC):
    """
    Base class for all data quality rules.

    Subclasses must implement :meth:`evaluate` which receives a
    :class:`pandas.DataFrame` and returns a :class:`RuleResult`.
    """

    def __init__(
        self,
        name: str,
        column: Optional[str] = None,
        severity: Severity = Severity.MEDIUM,
        dimension: Dimension = Dimension.VALIDITY,
        threshold: float = 1.0,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.column = column
        self.severity = severity
        self.dimension = dimension
        self.threshold = threshold
        self.params = params or {}

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        """Evaluate the rule and return a :class:`RuleResult`."""
        ...

    def _build_result(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        details: str = "",
    ) -> RuleResult:
        """
        Helper that builds a :class:`RuleResult` from a boolean pass-mask.

        Parameters
        ----------
        df : pd.DataFrame
            The evaluated dataset.
        mask : pd.Series
            Boolean series where ``True`` = row passes the check.
        details : str
            Human-readable description.
        """
        total = len(mask)
        pass_count = int(mask.sum())
        fail_count = total - pass_count
        pass_rate = pass_count / total if total else 1.0
        failing_rows = df[~mask].head(5).to_dict(orient="records") if fail_count else []

        return RuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            severity=self.severity,
            pass_count=pass_count,
            fail_count=fail_count,
            total_count=total,
            pass_rate=round(pass_rate, 6),
            passed=pass_rate >= self.threshold,
            threshold=self.threshold,
            column=self.column,
            details=details,
            failing_sample=failing_rows,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"column={self.column!r}, severity={self.severity.value})"
        )
