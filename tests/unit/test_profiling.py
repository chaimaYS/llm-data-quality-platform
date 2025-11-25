"""
Tests for the profiling engine.
Author: Chaima Yedes
"""

import pandas as pd
import pytest

from src.profiling.engine import ColumnProfile, ProfileResult, ProfilingEngine


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"user_{i}" for i in range(1, 101)],
            "email": [f"user{i}@example.com" for i in range(1, 101)],
            "age": [25 + (i % 50) for i in range(100)],
            "score": [float(i * 1.5) for i in range(100)],
            "city": ["NYC"] * 40 + ["LA"] * 30 + ["CHI"] * 20 + [None] * 10,
        }
    )


@pytest.fixture
def engine() -> ProfilingEngine:
    return ProfilingEngine(top_k=5)


class TestProfilingEngine:
    def test_profile_returns_profile_result(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        assert isinstance(result, ProfileResult)

    def test_row_and_column_count(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        assert result.row_count == 100
        assert result.column_count == 6

    def test_all_columns_profiled(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        assert set(result.columns.keys()) == {"id", "name", "email", "age", "score", "city"}

    def test_numeric_stats(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        age_profile = result.columns["age"]
        assert age_profile.mean is not None
        assert age_profile.std is not None
        assert age_profile.min_value is not None
        assert age_profile.max_value is not None
        assert age_profile.null_count == 0

    def test_null_detection(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        city_profile = result.columns["city"]
        assert city_profile.null_count == 10
        assert city_profile.null_pct == 10.0

    def test_uniqueness_detection(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        assert result.columns["id"].is_unique is True
        assert result.columns["city"].is_unique is False

    def test_distinct_count(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        city_profile = result.columns["city"]
        assert city_profile.distinct_count == 3  # NYC, LA, CHI (nulls not counted)

    def test_top_k_values(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        city_profile = result.columns["city"]
        assert len(city_profile.top_k) > 0
        top_value = city_profile.top_k[0]
        assert "value" in top_value
        assert "count" in top_value
        assert top_value["value"] == "NYC"
        assert top_value["count"] == 40

    def test_string_length_stats(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        name_profile = result.columns["name"]
        assert name_profile.min_length is not None
        assert name_profile.max_length is not None
        assert name_profile.avg_length is not None

    def test_pattern_inference(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        email_profile = result.columns["email"]
        assert len(email_profile.patterns) > 0
        assert "pattern" in email_profile.patterns[0]

    def test_duplicate_row_count(
        self, engine: ProfilingEngine
    ) -> None:
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        result = engine.profile(df)
        assert result.duplicate_row_count == 1  # one pair of duplicates = 2 rows - 1

    def test_empty_dataframe(self, engine: ProfilingEngine) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="str")})
        result = engine.profile(df)
        assert result.row_count == 0
        assert result.column_count == 2

    def test_memory_bytes_tracked(
        self, engine: ProfilingEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.profile(sample_df)
        assert result.memory_bytes > 0
