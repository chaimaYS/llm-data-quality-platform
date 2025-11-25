"""
Tests for the dimension scoring engine.
Author: Chaima Yedes
"""

import pytest

from src.rules.base import Dimension, RuleResult, Severity
from src.scoring.scorer import DatasetScore, DimensionScore, DimensionScorer


def _make_result(
    name: str,
    dimension: Dimension,
    pass_rate: float,
    passed: bool = True,
    severity: Severity = Severity.MEDIUM,
) -> RuleResult:
    total = 100
    pass_count = int(pass_rate * total)
    return RuleResult(
        rule_name=name,
        dimension=dimension,
        severity=severity,
        pass_count=pass_count,
        fail_count=total - pass_count,
        total_count=total,
        pass_rate=pass_rate,
        passed=passed,
    )


class TestDimensionScorer:
    def test_perfect_score(self) -> None:
        results = [
            _make_result("r1", Dimension.COMPLETENESS, 1.0),
            _make_result("r2", Dimension.UNIQUENESS, 1.0),
            _make_result("r3", Dimension.VALIDITY, 1.0),
        ]
        scorer = DimensionScorer()
        score = scorer.score(results)
        assert score.overall_score == pytest.approx(1.0, abs=0.001)
        assert score.grade == "A"

    def test_zero_score(self) -> None:
        results = [
            _make_result("r1", Dimension.COMPLETENESS, 0.0, passed=False),
            _make_result("r2", Dimension.UNIQUENESS, 0.0, passed=False),
            _make_result("r3", Dimension.VALIDITY, 0.0, passed=False),
        ]
        scorer = DimensionScorer()
        score = scorer.score(results)
        # 5 dimensions without rules default to 1.0, so overall is 0.5
        assert score.overall_score <= 0.5
        assert score.grade in ("D", "F")

    def test_mixed_scores(self) -> None:
        results = [
            _make_result("r1", Dimension.COMPLETENESS, 0.95),
            _make_result("r2", Dimension.COMPLETENESS, 0.85),
            _make_result("r3", Dimension.UNIQUENESS, 1.0),
            _make_result("r4", Dimension.VALIDITY, 0.70, passed=False),
        ]
        scorer = DimensionScorer()
        score = scorer.score(results)
        assert 0.5 < score.overall_score < 1.0
        assert score.total_rules == 4
        assert score.total_passed == 3
        assert score.total_failed == 1

    def test_dimension_pass_rate(self) -> None:
        results = [
            _make_result("r1", Dimension.COMPLETENESS, 0.90),
            _make_result("r2", Dimension.COMPLETENESS, 0.80),
        ]
        scorer = DimensionScorer()
        score = scorer.score(results)
        comp = score.dimensions["completeness"]
        assert comp.pass_rate == pytest.approx(0.85, abs=0.001)
        assert comp.rule_count == 2

    def test_dimensions_without_rules_get_full_score(self) -> None:
        results = [_make_result("r1", Dimension.COMPLETENESS, 1.0)]
        scorer = DimensionScorer()
        score = scorer.score(results)
        # Dimensions with no rules default to pass_rate=1.0
        assert score.dimensions["uniqueness"].pass_rate == 1.0

    def test_custom_weights(self) -> None:
        results = [
            _make_result("r1", Dimension.COMPLETENESS, 1.0),
            _make_result("r2", Dimension.VALIDITY, 0.0, passed=False),
        ]
        scorer = DimensionScorer(weights={"completeness": 0.9, "validity": 0.1})
        score = scorer.score(results)
        # Completeness is 90% of the weight with pass_rate 1.0
        # Validity is 10% of the weight with pass_rate 0.0
        assert score.overall_score == pytest.approx(0.9, abs=0.01)

    def test_grade_boundaries(self) -> None:
        # Use a single-dimension scorer so grades map directly to pass rates
        scorer = DimensionScorer(weights={"completeness": 1.0})

        results_a = [_make_result("r", Dimension.COMPLETENESS, 0.96)]
        assert scorer.score(results_a).grade == "A"

        results_b = [_make_result("r", Dimension.COMPLETENESS, 0.90)]
        assert scorer.score(results_b).grade == "B"

        results_c = [_make_result("r", Dimension.COMPLETENESS, 0.75)]
        assert scorer.score(results_c).grade == "C"

        results_d = [_make_result("r", Dimension.COMPLETENESS, 0.55)]
        assert scorer.score(results_d).grade == "D"

        results_f = [_make_result("r", Dimension.COMPLETENESS, 0.30, passed=False)]
        assert scorer.score(results_f).grade == "F"

    def test_to_dict(self) -> None:
        results = [_make_result("r1", Dimension.COMPLETENESS, 0.95)]
        scorer = DimensionScorer()
        score = scorer.score(results)
        d = score.to_dict()
        assert "overall_score" in d
        assert "grade" in d
        assert "dimensions" in d
        assert "completeness" in d["dimensions"]

    def test_empty_results(self) -> None:
        scorer = DimensionScorer()
        score = scorer.score([])
        assert score.overall_score == pytest.approx(1.0, abs=0.001)
        assert score.total_rules == 0

    def test_weighted_score_in_dimension(self) -> None:
        results = [_make_result("r1", Dimension.COMPLETENESS, 0.80)]
        scorer = DimensionScorer(weights={"completeness": 0.50, "validity": 0.50})
        score = scorer.score(results)
        comp = score.dimensions["completeness"]
        assert comp.weighted_score == pytest.approx(0.40, abs=0.01)

    def test_from_yaml(self, tmp_path) -> None:
        config = tmp_path / "weights.yml"
        config.write_text(
            "dimensions:\n"
            "  completeness:\n"
            "    weight: 0.5\n"
            "  validity:\n"
            "    weight: 0.5\n"
        )
        scorer = DimensionScorer.from_yaml(config)
        assert scorer.weights["completeness"] == pytest.approx(0.5, abs=0.01)
        assert scorer.weights["validity"] == pytest.approx(0.5, abs=0.01)
