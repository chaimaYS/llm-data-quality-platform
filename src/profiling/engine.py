"""
Column-level profiling engine powered by DuckDB.
Author: Chaima Yedes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import pandas as pd


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""

    name: str
    dtype: str
    total_count: int
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_pct: float
    min_value: Any = None
    max_value: Any = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    top_k: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    is_unique: bool = False


@dataclass
class ProfileResult:
    """Aggregated profiling result for an entire dataset."""

    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    duplicate_row_count: int = 0
    memory_bytes: int = 0


class ProfilingEngine:
    """
    Compute rich per-column statistics for a pandas DataFrame using
    DuckDB as the analytical backend for speed.
    """

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """Run a full profile on *df* and return a :class:`ProfileResult`."""
        con = duckdb.connect(database=":memory:")
        con.register("src", df)

        row_count = len(df)
        dup_count = row_count - len(df.drop_duplicates())
        columns: dict[str, ColumnProfile] = {}

        for col in df.columns:
            safe = col.replace('"', '""')
            cp = self._profile_column(con, safe, row_count)
            columns[col] = cp

        con.close()

        return ProfileResult(
            row_count=row_count,
            column_count=len(df.columns),
            columns=columns,
            duplicate_row_count=dup_count,
            memory_bytes=df.memory_usage(deep=True).sum(),
        )

    def _profile_column(
        self, con: duckdb.DuckDBPyConnection, col: str, total: int
    ) -> ColumnProfile:
        # Get the column type; try a non-null value first, fall back to schema
        type_row = con.execute(
            f'SELECT typeof("{col}") FROM src WHERE "{col}" IS NOT NULL LIMIT 1'
        ).fetchone()
        if type_row:
            dtype = type_row[0]
        else:
            # Empty or all-null column -- infer from DuckDB view schema
            desc = con.execute("DESCRIBE src").fetchall()
            dtype_map = {r[0]: r[1] for r in desc}
            dtype = dtype_map.get(col, "VARCHAR")

        q = f"""
        SELECT
            COUNT(*)                                 AS total_count,
            SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS null_count,
            COUNT(DISTINCT "{col}")                  AS distinct_count,
            MIN("{col}")                             AS min_val,
            MAX("{col}")                             AS max_val
        FROM src
        """
        row = con.execute(q).fetchone()
        assert row is not None
        total_count, null_count, distinct_count, min_val, max_val = row
        null_pct = round(null_count / total * 100, 2) if total else 0.0
        distinct_pct = round(distinct_count / total * 100, 2) if total else 0.0

        mean = std = median = p25 = p75 = None
        if dtype in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT", "TINYINT", "HUGEINT"):
            stats = con.execute(f"""
                SELECT
                    AVG("{col}")::DOUBLE,
                    STDDEV("{col}")::DOUBLE,
                    MEDIAN("{col}")::DOUBLE,
                    QUANTILE_CONT("{col}", 0.25)::DOUBLE,
                    QUANTILE_CONT("{col}", 0.75)::DOUBLE
                FROM src
            """).fetchone()
            if stats:
                mean, std, median, p25, p75 = stats

        min_len = max_len = avg_len = None
        if dtype == "VARCHAR":
            len_stats = con.execute(f"""
                SELECT
                    MIN(LENGTH("{col}")),
                    MAX(LENGTH("{col}")),
                    AVG(LENGTH("{col}"))::DOUBLE
                FROM src WHERE "{col}" IS NOT NULL
            """).fetchone()
            if len_stats:
                min_len, max_len, avg_len = len_stats
                avg_len = round(avg_len, 2) if avg_len is not None else None

        top_k = self._compute_top_k(con, col)
        patterns = self._infer_patterns(con, col, dtype)

        return ColumnProfile(
            name=col,
            dtype=dtype,
            total_count=total_count,
            null_count=null_count,
            null_pct=null_pct,
            distinct_count=distinct_count,
            distinct_pct=distinct_pct,
            min_value=min_val,
            max_value=max_val,
            mean=mean,
            std=std,
            median=median,
            p25=p25,
            p75=p75,
            min_length=min_len,
            max_length=max_len,
            avg_length=avg_len,
            top_k=top_k,
            patterns=patterns,
            is_unique=(distinct_count == total_count and null_count == 0),
        )

    def _compute_top_k(
        self, con: duckdb.DuckDBPyConnection, col: str
    ) -> list[dict[str, Any]]:
        rows = con.execute(f"""
            SELECT "{col}" AS value, COUNT(*) AS freq
            FROM src
            WHERE "{col}" IS NOT NULL
            GROUP BY "{col}"
            ORDER BY freq DESC
            LIMIT {self._top_k}
        """).fetchall()
        return [{"value": r[0], "count": r[1]} for r in rows]

    def _infer_patterns(
        self, con: duckdb.DuckDBPyConnection, col: str, dtype: str
    ) -> list[dict[str, Any]]:
        """Generalize string values into regex-like character-class patterns."""
        if dtype != "VARCHAR":
            return []
        sample = con.execute(
            f'SELECT "{col}" FROM src WHERE "{col}" IS NOT NULL LIMIT 5000'
        ).fetchall()
        counter: dict[str, int] = {}
        for (val,) in sample:
            pat = self._to_pattern(str(val))
            counter[pat] = counter.get(pat, 0) + 1
        sorted_pats = sorted(counter.items(), key=lambda x: -x[1])[:10]
        total = sum(c for _, c in sorted_pats)
        return [
            {"pattern": p, "count": c, "pct": round(c / total * 100, 1)}
            for p, c in sorted_pats
        ]

    @staticmethod
    def _to_pattern(value: str) -> str:
        """Convert a string value to a character-class pattern."""
        result: list[str] = []
        for ch in value:
            if ch.isdigit():
                result.append("D")
            elif ch.isalpha():
                result.append("A")
            else:
                result.append(ch)
        collapsed = re.sub(r"D+", "D+", "".join(result))
        collapsed = re.sub(r"A+", "A+", collapsed)
        return collapsed
