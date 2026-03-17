"""
Data Quality Platform — Streamlit UI
Upload any dataset (CSV, Parquet, Excel, JSON, PDF, image) and get a full quality report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.profiling.engine import ProfilingEngine, ProfileResult, ColumnProfile
from src.rules.base import Dimension, Severity, RuleResult
from src.rules.builtin import NullCheck, UniqueCheck, RangeCheck, RegexCheck, FreshnessCheck
from src.scoring.scorer import DimensionScorer

st.set_page_config(
    page_title="Data Quality Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIMENSION_COLORS = {
    "completeness": "#10B981",
    "uniqueness": "#3B82F6",
    "validity": "#8B5CF6",
    "consistency": "#F59E0B",
    "timeliness": "#EF4444",
    "accuracy": "#06B6D4",
    "integrity": "#EC4899",
    "conformity": "#6366F1",
}

GRADE_COLORS = {"A": "#10B981", "B": "#3B82F6", "C": "#F59E0B", "D": "#EF4444", "F": "#991B1B"}


def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        import json as _json
        data = _json.load(uploaded_file)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.json_normalize(data)
        return pd.DataFrame([data])
    else:
        st.error(f"Unsupported format: {name}")
        return pd.DataFrame()


def render_sidebar():
    with st.sidebar:
        st.markdown("## Data Quality Platform")
        st.markdown("Upload a dataset to profile and score.")
        st.markdown("---")

        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "parquet", "xlsx", "xls", "json"],
            help="CSV, Parquet, Excel, or JSON",
        )

        st.markdown("---")
        st.markdown("### Scoring weights")
        weights = {}
        for dim in Dimension:
            weights[dim.value] = st.slider(
                dim.value.title(), 0.0, 2.0, 1.0, 0.1, key=f"w_{dim.value}"
            )

        st.markdown("---")
        st.markdown("### Settings")
        sample_size = st.number_input("Profile sample size", 100, 1000000, 100000, step=10000)
        show_samples = st.checkbox("Show failing samples", value=True)

    return uploaded, weights, sample_size, show_samples


def run_profiling(df: pd.DataFrame) -> ProfileResult:
    engine = ProfilingEngine()
    return engine.profile(df)


def auto_generate_rules(df: pd.DataFrame, profile: ProfileResult) -> list:
    """Auto-generate sensible rules based on the profile."""
    rules = []

    for col, col_profile in profile.columns.items():
        dtype = col_profile.dtype
        null_pct = col_profile.null_pct / 100 if col_profile.null_pct else 0

        # Completeness: flag columns with nulls
        if null_pct > 0:
            rules.append(NullCheck(column=col, severity=Severity.HIGH if null_pct < 0.05 else Severity.MEDIUM))

        # Uniqueness: check columns that look like IDs
        if any(k in col.lower() for k in ("id", "key", "code", "uuid", "email")):
            non_null = col_profile.total_count - col_profile.null_count
            if col_profile.distinct_count == non_null and non_null > 0:
                rules.append(UniqueCheck(column=col, severity=Severity.HIGH))

        # Validity: range checks for numeric columns
        if dtype in ("int64", "float64", "int32", "float32") and col_profile.min_value is not None:
            if col_profile.min_value >= 0:
                rules.append(RangeCheck(column=col, min_value=0, severity=Severity.MEDIUM))

        # Validity: email regex
        if "email" in col.lower():
            rules.append(RegexCheck(column=col, pattern=r"^[^@]+@[^@]+\.[^@]+$", severity=Severity.HIGH))

        # Validity: phone regex
        if "phone" in col.lower():
            rules.append(RegexCheck(column=col, pattern=r"^\+?[\d\s\-\(\)]{7,15}$", severity=Severity.MEDIUM))

    return rules


def render_overview(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    c4.metric("Null %", f"{null_pct:.1f}%")


def render_profile(profile: ProfileResult):
    st.markdown("### Column Profiles")

    for col, cp in profile.columns.items():
        with st.expander(f"**{col}** — {cp.dtype}", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nulls", f"{cp.null_count:,}")
            m2.metric("Distinct", f"{cp.distinct_count:,}")
            m3.metric("Null %", f"{cp.null_pct:.1f}%")
            m4.metric("Unique %", f"{cp.distinct_pct:.1f}%")

            if cp.min_value is not None:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Min", f"{cp.min_value}")
                s2.metric("Max", f"{cp.max_value}")
                if cp.mean is not None:
                    s3.metric("Mean", f"{cp.mean:.2f}")
                if cp.std is not None:
                    s4.metric("Std", f"{cp.std:.2f}")

            if cp.min_length is not None:
                l1, l2, l3 = st.columns(3)
                l1.metric("Min length", cp.min_length)
                l2.metric("Max length", cp.max_length)
                l3.metric("Avg length", f"{cp.avg_length:.1f}" if cp.avg_length else "—")

            if cp.top_k:
                st.markdown("**Top values:**")
                top_df = pd.DataFrame(cp.top_k)
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            if cp.patterns:
                st.markdown("**Detected patterns:**")
                for p in cp.patterns[:5]:
                    st.code(f"{p.get('pattern', '')}  ({p.get('count', 0)} rows, {p.get('pct', 0):.1f}%)")


def render_scores(score_result):
    st.markdown("### Quality Score")

    grade = score_result.grade
    overall = score_result.overall_score
    color = GRADE_COLORS.get(grade, "#666")

    col_grade, col_score = st.columns([1, 3])
    with col_grade:
        st.markdown(
            f"<div style='text-align:center; padding:20px;'>"
            f"<span style='font-size:72px; font-weight:bold; color:{color}'>{grade}</span>"
            f"<br><span style='font-size:18px; color:#888'>{overall:.0%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_score:
        for dim_name, dim_score in score_result.dimensions.items():
            st.markdown(
                f"**{dim_name.title()}** — {dim_score.pass_rate:.0%} ({dim_score.rule_count} rules)"
            )
            st.progress(dim_score.pass_rate)


def render_rule_results(results: list, show_samples: bool):
    st.markdown("### Rule Results")

    passed = [r for r in results if r.fail_count == 0]
    failed = [r for r in results if r.fail_count > 0]

    tab_fail, tab_pass = st.tabs([f"Failed ({len(failed)})", f"Passed ({len(passed)})"])

    with tab_fail:
        if not failed:
            st.success("All rules passed!")
        for r in sorted(failed, key=lambda x: x.fail_count, reverse=True):
            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.severity.value, "⚪")
            total = r.pass_count + r.fail_count
            fail_pct = r.fail_count / total * 100 if total > 0 else 0

            with st.expander(f"{severity_icon} **{r.rule_name}** on `{r.column}` — {r.fail_count:,} failures ({fail_pct:.1f}%)"):
                st.markdown(f"**Dimension:** {r.dimension.value}")
                st.markdown(f"**Passed:** {r.pass_count:,} | **Failed:** {r.fail_count:,}")
                if show_samples and r.failing_sample:
                    st.markdown("**Sample failing values:**")
                    st.dataframe(
                        pd.DataFrame({"failing_values": r.failing_sample[:20]}),
                        use_container_width=True, hide_index=True,
                    )

    with tab_pass:
        for r in passed:
            st.markdown(f"✅ **{r.rule_name}** on `{r.column}` — {r.pass_count:,} rows passed")


def render_data_preview(df: pd.DataFrame):
    st.markdown("### Data Preview")
    st.dataframe(df.head(100), use_container_width=True)


def main():
    uploaded, weights, sample_size, show_samples = render_sidebar()

    if not uploaded:
        st.markdown(
            """
            # 🔍 Data Quality Platform

            Upload a dataset to get started. The platform will:

            1. **Profile** every column — types, nulls, distributions, patterns
            2. **Auto-generate rules** — completeness, uniqueness, validity checks
            3. **Score** across 8 quality dimensions
            4. **Report** failures with sample values

            ### Supported formats
            `CSV` `Parquet` `Excel` `JSON`

            ### Quality dimensions
            | Dimension | What it measures |
            |-----------|-----------------|
            | Completeness | Required values present |
            | Uniqueness | No unintended duplicates |
            | Validity | Values match expected format |
            | Consistency | Values agree across fields |
            | Timeliness | Data is fresh enough |
            | Accuracy | Matches source of truth |
            | Integrity | Referential integrity holds |
            | Conformity | Follows standards |
            """
        )
        return

    # Load data
    df = load_data(uploaded)
    if df.empty:
        return

    st.markdown(f"# Quality Report: `{uploaded.name}`")
    st.markdown(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # Overview
    render_overview(df)

    st.markdown("---")

    # Profile
    with st.spinner("Profiling dataset..."):
        profile = run_profiling(df.head(sample_size) if len(df) > sample_size else df)

    # Auto-generate rules
    with st.spinner("Generating quality rules..."):
        rules = auto_generate_rules(df, profile)

    # Evaluate rules
    with st.spinner("Evaluating rules..."):
        results = []
        for rule in rules:
            try:
                result = rule.evaluate(df)
                results.append(result)
            except Exception as e:
                st.warning(f"Rule {rule.__class__.__name__} on {rule.column} failed: {e}")

    # Score
    scorer = DimensionScorer(weights=weights)
    score_result = scorer.score(results)

    # Render
    tab_score, tab_profile, tab_rules, tab_data = st.tabs(
        ["📊 Scores", "📋 Profile", "✅ Rules", "📄 Data"]
    )

    with tab_score:
        render_scores(score_result)

    with tab_profile:
        render_profile(profile)

    with tab_rules:
        render_rule_results(results, show_samples)

    with tab_data:
        render_data_preview(df)


if __name__ == "__main__":
    main()
