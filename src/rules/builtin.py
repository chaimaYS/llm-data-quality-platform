"""
Built-in data quality rules.
Author: Chaima Yedes
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from src.rules.base import Dimension, Rule, RuleResult, Severity


class NullCheck(Rule):
    """Checks that a column contains no (or few) null values."""

    def __init__(
        self,
        column: str,
        threshold: float = 1.0,
        severity: Severity = Severity.HIGH,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or f"null_check_{column}",
            column=column,
            severity=severity,
            dimension=Dimension.COMPLETENESS,
            threshold=threshold,
        )

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        mask = df[self.column].notna()
        return self._build_result(df, mask, details=f"Null check on '{self.column}'")


class UniqueCheck(Rule):
    """Checks that values in a column (or combination) are unique."""

    def __init__(
        self,
        column: str | list[str],
        threshold: float = 1.0,
        severity: Severity = Severity.HIGH,
        name: Optional[str] = None,
    ) -> None:
        cols = [column] if isinstance(column, str) else column
        col_label = "+".join(cols)
        super().__init__(
            name=name or f"unique_check_{col_label}",
            column=col_label,
            severity=severity,
            dimension=Dimension.UNIQUENESS,
            threshold=threshold,
            params={"columns": cols},
        )

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        cols = self.params["columns"]
        duplicated = df.duplicated(subset=cols, keep=False)
        mask = ~duplicated
        return self._build_result(
            df, mask, details=f"Uniqueness check on {cols}"
        )


class RangeCheck(Rule):
    """Checks that numeric values fall within [min_value, max_value]."""

    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        threshold: float = 1.0,
        severity: Severity = Severity.MEDIUM,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or f"range_check_{column}",
            column=column,
            severity=severity,
            dimension=Dimension.VALIDITY,
            threshold=threshold,
            params={"min_value": min_value, "max_value": max_value},
        )

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        series = pd.to_numeric(df[self.column], errors="coerce")
        min_v = self.params["min_value"]
        max_v = self.params["max_value"]
        mask = pd.Series(True, index=df.index)
        if min_v is not None:
            mask &= series >= min_v
        if max_v is not None:
            mask &= series <= max_v
        mask &= series.notna()
        return self._build_result(
            df,
            mask,
            details=f"Range [{min_v}, {max_v}] on '{self.column}'",
        )


class RegexCheck(Rule):
    """Checks that string values match a regular expression."""

    def __init__(
        self,
        column: str,
        pattern: str,
        threshold: float = 1.0,
        severity: Severity = Severity.MEDIUM,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or f"regex_check_{column}",
            column=column,
            severity=severity,
            dimension=Dimension.CONFORMITY,
            threshold=threshold,
            params={"pattern": pattern},
        )
        self._compiled = re.compile(pattern)

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        not_null = df[self.column].notna()
        series = df[self.column].fillna("").astype(str)
        mask = series.apply(lambda v: bool(self._compiled.fullmatch(v)))
        mask = mask & not_null
        return self._build_result(
            df,
            mask,
            details=f"Regex '{self.params['pattern']}' on '{self.column}'",
        )


class ForeignKeyCheck(Rule):
    """Checks that every value in *column* exists in a reference set."""

    def __init__(
        self,
        column: str,
        reference_values: set[Any] | pd.Series | list[Any] | None = None,
        reference_df: Optional[pd.DataFrame] = None,
        reference_column: Optional[str] = None,
        threshold: float = 1.0,
        severity: Severity = Severity.HIGH,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or f"fk_check_{column}",
            column=column,
            severity=severity,
            dimension=Dimension.INTEGRITY,
            threshold=threshold,
        )
        if reference_values is not None:
            self._ref_set = set(reference_values)
        elif reference_df is not None and reference_column is not None:
            self._ref_set = set(reference_df[reference_column].dropna())
        else:
            raise ValueError("Provide reference_values or reference_df + reference_column")

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        non_null = df[self.column].dropna()
        mask_non_null = non_null.isin(self._ref_set)
        mask = pd.Series(True, index=df.index)
        mask.loc[non_null.index] = mask_non_null
        mask.loc[df[self.column].isna()] = True  # nulls handled by NullCheck
        return self._build_result(
            df,
            mask,
            details=f"FK check: '{self.column}' against {len(self._ref_set)} reference values",
        )


class FreshnessCheck(Rule):
    """Checks that a timestamp column has recent data."""

    def __init__(
        self,
        column: str,
        max_age_hours: float = 24.0,
        threshold: float = 1.0,
        severity: Severity = Severity.HIGH,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or f"freshness_check_{column}",
            column=column,
            severity=severity,
            dimension=Dimension.TIMELINESS,
            threshold=threshold,
            params={"max_age_hours": max_age_hours},
        )

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        series = pd.to_datetime(df[self.column], errors="coerce", utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.params["max_age_hours"]
        )
        mask = series >= cutoff
        mask = mask.fillna(False)
        return self._build_result(
            df,
            mask,
            details=(
                f"Freshness: '{self.column}' within "
                f"{self.params['max_age_hours']}h"
            ),
        )
