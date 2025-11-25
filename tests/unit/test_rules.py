"""
Tests for built-in data quality rules.
Author: Chaima Yedes
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.rules.base import Dimension, RuleResult, Severity
from src.rules.builtin import (
    ForeignKeyCheck,
    FreshnessCheck,
    NullCheck,
    RangeCheck,
    RegexCheck,
    UniqueCheck,
)


@pytest.fixture
def customer_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "email": [
                "a@test.com",
                "b@test.com",
                "invalid",
                "c@test.com",
                None,
            ],
            "age": [25, 30, -5, 200, 40],
            "status": ["active", "active", "inactive", "active", "active"],
            "updated_at": [
                datetime.now(timezone.utc) - timedelta(hours=1),
                datetime.now(timezone.utc) - timedelta(hours=2),
                datetime.now(timezone.utc) - timedelta(days=5),
                datetime.now(timezone.utc) - timedelta(hours=6),
                datetime.now(timezone.utc) - timedelta(days=60),
            ],
        }
    )


class TestNullCheck:
    def test_no_nulls(self) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        rule = NullCheck(column="col")
        result = rule.evaluate(df)
        assert result.passed is True
        assert result.pass_rate == 1.0
        assert result.fail_count == 0

    def test_with_nulls(self, customer_df: pd.DataFrame) -> None:
        rule = NullCheck(column="email")
        result = rule.evaluate(customer_df)
        assert result.fail_count == 1
        assert result.pass_count == 4
        assert result.dimension == Dimension.COMPLETENESS

    def test_threshold(self, customer_df: pd.DataFrame) -> None:
        rule = NullCheck(column="email", threshold=0.7)
        result = rule.evaluate(customer_df)
        assert result.passed is True  # 80% > 70%

    def test_severity(self) -> None:
        rule = NullCheck(column="x", severity=Severity.CRITICAL)
        assert rule.severity == Severity.CRITICAL


class TestUniqueCheck:
    def test_all_unique(self) -> None:
        df = pd.DataFrame({"col": [1, 2, 3, 4]})
        rule = UniqueCheck(column="col")
        result = rule.evaluate(df)
        assert result.passed is True

    def test_duplicates(self) -> None:
        df = pd.DataFrame({"col": [1, 2, 2, 3]})
        rule = UniqueCheck(column="col")
        result = rule.evaluate(df)
        assert result.passed is False
        assert result.fail_count == 2  # both duplicate rows flagged
        assert result.dimension == Dimension.UNIQUENESS

    def test_multi_column(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        rule = UniqueCheck(column=["a", "b"])
        result = rule.evaluate(df)
        assert result.passed is True


class TestRangeCheck:
    def test_in_range(self) -> None:
        df = pd.DataFrame({"val": [10, 20, 30]})
        rule = RangeCheck(column="val", min_value=0, max_value=100)
        result = rule.evaluate(df)
        assert result.passed is True

    def test_out_of_range(self, customer_df: pd.DataFrame) -> None:
        rule = RangeCheck(column="age", min_value=0, max_value=150)
        result = rule.evaluate(customer_df)
        assert result.fail_count == 2  # -5 and 200 are out of range
        assert result.dimension == Dimension.VALIDITY

    def test_min_only(self) -> None:
        df = pd.DataFrame({"val": [-1, 0, 5]})
        rule = RangeCheck(column="val", min_value=0)
        result = rule.evaluate(df)
        assert result.fail_count == 1

    def test_max_only(self) -> None:
        df = pd.DataFrame({"val": [50, 100, 200]})
        rule = RangeCheck(column="val", max_value=150)
        result = rule.evaluate(df)
        assert result.fail_count == 1


class TestRegexCheck:
    def test_valid_emails(self) -> None:
        df = pd.DataFrame({"email": ["a@b.com", "c@d.org"]})
        rule = RegexCheck(
            column="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        result = rule.evaluate(df)
        assert result.passed is True

    def test_invalid_emails(self, customer_df: pd.DataFrame) -> None:
        rule = RegexCheck(
            column="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        result = rule.evaluate(customer_df)
        # "invalid" fails, None fails
        assert result.fail_count >= 2
        assert result.dimension == Dimension.CONFORMITY


class TestForeignKeyCheck:
    def test_all_valid(self) -> None:
        df = pd.DataFrame({"status": ["active", "inactive"]})
        rule = ForeignKeyCheck(
            column="status",
            reference_values={"active", "inactive", "pending"},
        )
        result = rule.evaluate(df)
        assert result.passed is True

    def test_invalid_references(self) -> None:
        df = pd.DataFrame({"status": ["active", "deleted", "unknown"]})
        rule = ForeignKeyCheck(
            column="status",
            reference_values={"active", "inactive"},
        )
        result = rule.evaluate(df)
        assert result.fail_count == 2
        assert result.dimension == Dimension.INTEGRITY

    def test_from_reference_df(self) -> None:
        ref = pd.DataFrame({"id": [1, 2, 3]})
        df = pd.DataFrame({"fk": [1, 2, 99]})
        rule = ForeignKeyCheck(
            column="fk",
            reference_df=ref,
            reference_column="id",
        )
        result = rule.evaluate(df)
        assert result.fail_count == 1


class TestFreshnessCheck:
    def test_fresh_data(self) -> None:
        now = datetime.now(timezone.utc)
        df = pd.DataFrame({"ts": [now - timedelta(hours=1), now - timedelta(hours=2)]})
        rule = FreshnessCheck(column="ts", max_age_hours=24)
        result = rule.evaluate(df)
        assert result.passed is True

    def test_stale_data(self, customer_df: pd.DataFrame) -> None:
        rule = FreshnessCheck(column="updated_at", max_age_hours=24)
        result = rule.evaluate(customer_df)
        assert result.fail_count >= 2  # 5-day and 60-day old entries
        assert result.dimension == Dimension.TIMELINESS


class TestRuleResult:
    def test_fail_rate(self) -> None:
        result = RuleResult(
            rule_name="test",
            dimension=Dimension.VALIDITY,
            severity=Severity.MEDIUM,
            pass_count=80,
            fail_count=20,
            total_count=100,
            pass_rate=0.8,
            passed=True,
        )
        assert result.fail_rate == pytest.approx(0.2)
