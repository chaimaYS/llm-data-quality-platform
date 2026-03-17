"""End-to-end pipeline tests: profile → rules → score."""

import pytest
import pandas as pd
import numpy as np

from src.profiling.engine import ProfilingEngine
from src.rules.builtin import NullCheck, UniqueCheck, RangeCheck, RegexCheck
from src.rules.base import Severity, Dimension
from src.scoring.scorer import DimensionScorer


@pytest.fixture
def customer_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "email": [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, n + 1)],
        "age": [np.random.randint(18, 80) if i % 25 != 0 else -5 for i in range(1, n + 1)],
        "balance": np.round(np.random.exponential(5000, n), 2),
        "country": np.random.choice(["UAE", "US", "UK", None], n, p=[0.4, 0.3, 0.2, 0.1]),
        "status": np.random.choice(["active", "inactive", None], n, p=[0.7, 0.2, 0.1]),
    })


class TestFullPipeline:
    def test_profile_then_score(self, customer_df):
        engine = ProfilingEngine()
        profile = engine.profile(customer_df)

        assert profile.row_count == 200
        assert len(profile.columns) == 6

        rules = [
            NullCheck(column="email", severity=Severity.HIGH),
            NullCheck(column="country", severity=Severity.MEDIUM),
            NullCheck(column="status", severity=Severity.LOW),
            UniqueCheck(column="customer_id", severity=Severity.HIGH),
            RangeCheck(column="age", min_value=0, max_value=120, severity=Severity.HIGH),
            RangeCheck(column="balance", min_value=0, severity=Severity.MEDIUM),
            RegexCheck(column="email", pattern=r"^[^@]+@[^@]+\.[^@]+$", severity=Severity.HIGH),
        ]

        results = [r.evaluate(customer_df) for r in rules]

        scorer = DimensionScorer()
        score = scorer.score(results)

        assert score.overall_score > 0
        assert score.overall_score <= 1.0
        assert score.grade in ("A", "B", "C", "D", "F")
        assert score.total_rules == 7
        assert score.total_passed + score.total_failed > 0

    def test_completeness_dimension(self, customer_df):
        rules = [
            NullCheck(column="email", severity=Severity.HIGH),
            NullCheck(column="country", severity=Severity.MEDIUM),
        ]
        results = [r.evaluate(customer_df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)

        completeness = score.dimensions.get("completeness")
        assert completeness is not None
        assert completeness.rule_count == 2
        assert completeness.pass_rate < 1.0  # there are nulls

    def test_uniqueness_dimension(self, customer_df):
        rules = [UniqueCheck(column="customer_id", severity=Severity.HIGH)]
        results = [r.evaluate(customer_df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)

        uniqueness = score.dimensions.get("uniqueness")
        assert uniqueness is not None
        assert uniqueness.pass_rate == 1.0  # customer_id is unique

    def test_validity_dimension(self, customer_df):
        rules = [
            RangeCheck(column="age", min_value=0, max_value=120, severity=Severity.HIGH),
        ]
        results = [r.evaluate(customer_df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)

        validity = score.dimensions.get("validity")
        assert validity is not None
        assert validity.rule_count >= 1
        assert validity.pass_rate < 1.0  # some ages are -5

    def test_all_pass_gives_grade_a(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [10, 20, 30],
        })
        rules = [
            NullCheck(column="name", severity=Severity.HIGH),
            UniqueCheck(column="id", severity=Severity.HIGH),
            RangeCheck(column="value", min_value=0, max_value=100, severity=Severity.MEDIUM),
        ]
        results = [r.evaluate(df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)
        assert score.grade == "A"
        assert score.overall_score >= 0.95

    def test_all_fail_gives_low_grade(self):
        df = pd.DataFrame({
            "id": [1, 1, 1],
            "email": [None, None, None],
            "age": [-10, -20, -30],
        })
        rules = [
            NullCheck(column="email", severity=Severity.HIGH),
            UniqueCheck(column="id", severity=Severity.HIGH),
            RangeCheck(column="age", min_value=0, severity=Severity.HIGH),
        ]
        results = [r.evaluate(df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)
        assert score.grade in ("D", "F")
        assert score.overall_score <= 0.5

    def test_score_to_dict(self, customer_df):
        rules = [
            NullCheck(column="email", severity=Severity.HIGH),
            UniqueCheck(column="customer_id", severity=Severity.HIGH),
        ]
        results = [r.evaluate(customer_df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)
        d = score.to_dict()

        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "grade" in d
        assert "dimensions" in d
        assert isinstance(d["dimensions"], dict)

    def test_profile_captures_nulls_correctly(self, customer_df):
        engine = ProfilingEngine()
        profile = engine.profile(customer_df)

        email_profile = profile.columns["email"]
        assert email_profile.null_count == 20  # every 10th row
        assert email_profile.null_pct == pytest.approx(10.0, abs=0.5)

    def test_empty_dataset(self):
        df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=str)})
        engine = ProfilingEngine()
        profile = engine.profile(df)
        assert profile.row_count == 0

        rules = [NullCheck(column="a", severity=Severity.HIGH)]
        results = [r.evaluate(df) for r in rules]
        scorer = DimensionScorer()
        score = scorer.score(results)
        assert score.grade is not None
