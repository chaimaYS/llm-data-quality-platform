"""
FastAPI application for the Data Quality platform.
Author: Chaima Yedes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.profiling.engine import ProfilingEngine, ProfileResult
from src.rules.engine import RulesEngine
from src.scoring.scorer import DatasetScore, DimensionScorer

app = FastAPI(
    title="Data Quality Platform",
    version="1.0.0",
    description="LLM-powered data quality profiling, rules, and scoring.",
)

# ---------------------------------------------------------------------------
# In-memory stores (swap for a real DB in production)
# ---------------------------------------------------------------------------

_datasets: dict[str, dict[str, Any]] = {}
_profiles: dict[str, ProfileResult] = {}
_scores: dict[str, list[DatasetScore]] = {}
_dataframes: dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RegisterDatasetRequest(BaseModel):
    name: str
    source_type: str = "file"
    source_path: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)


class RegisterDatasetResponse(BaseModel):
    id: str
    name: str
    source_type: str
    registered_at: str


class ProfileResponse(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: dict[str, Any]


class RunRequest(BaseModel):
    rules_path: Optional[str] = None
    rules_inline: Optional[list[dict[str, Any]]] = None
    weights_path: Optional[str] = None


class RunResponse(BaseModel):
    dataset_id: str
    overall_score: float
    grade: str
    total_rules: int
    total_passed: int
    total_failed: int
    dimensions: dict[str, Any]
    run_at: str


class ScoreHistoryItem(BaseModel):
    overall_score: float
    grade: str
    total_rules: int
    run_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/datasets", response_model=RegisterDatasetResponse)
def register_dataset(req: RegisterDatasetRequest) -> RegisterDatasetResponse:
    """Register a new dataset for quality monitoring."""
    ds_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    _datasets[ds_id] = {
        "id": ds_id,
        "name": req.name,
        "source_type": req.source_type,
        "source_path": req.source_path,
        "config": req.config,
        "registered_at": now,
    }
    return RegisterDatasetResponse(
        id=ds_id,
        name=req.name,
        source_type=req.source_type,
        registered_at=now,
    )


@app.post("/datasets/{dataset_id}/upload")
def upload_data(dataset_id: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    """Upload data rows for an in-memory dataset (for demo/testing)."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.DataFrame(data)
    _dataframes[dataset_id] = df
    return {"dataset_id": dataset_id, "rows": len(df), "columns": list(df.columns)}


@app.post("/datasets/{dataset_id}/profile", response_model=ProfileResponse)
def run_profiling(dataset_id: str) -> ProfileResponse:
    """Run the profiling engine on a registered dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _dataframes.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded for this dataset")

    engine = ProfilingEngine()
    result = engine.profile(df)
    _profiles[dataset_id] = result

    columns_dict: dict[str, Any] = {}
    for name, cp in result.columns.items():
        columns_dict[name] = {
            "dtype": cp.dtype,
            "null_count": cp.null_count,
            "null_pct": cp.null_pct,
            "distinct_count": cp.distinct_count,
            "distinct_pct": cp.distinct_pct,
            "is_unique": cp.is_unique,
            "min": cp.min_value,
            "max": cp.max_value,
            "mean": cp.mean,
            "std": cp.std,
        }

    return ProfileResponse(
        dataset_id=dataset_id,
        row_count=result.row_count,
        column_count=result.column_count,
        duplicate_row_count=result.duplicate_row_count,
        columns=columns_dict,
    )


@app.post("/datasets/{dataset_id}/run", response_model=RunResponse)
def run_quality_check(dataset_id: str, req: RunRequest) -> RunResponse:
    """Execute a full data quality run: profile + rules + scoring."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _dataframes.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded for this dataset")

    if dataset_id not in _profiles:
        engine = ProfilingEngine()
        _profiles[dataset_id] = engine.profile(df)

    if req.rules_path:
        rules_engine = RulesEngine.from_yaml(req.rules_path)
    elif req.rules_inline:
        rules_engine = RulesEngine.from_yaml_data(req.rules_inline)
    else:
        raise HTTPException(status_code=400, detail="Provide rules_path or rules_inline")

    results = rules_engine.evaluate(df)

    scorer = (
        DimensionScorer.from_yaml(req.weights_path)
        if req.weights_path
        else DimensionScorer()
    )
    score = scorer.score(results)

    _scores.setdefault(dataset_id, []).append(score)
    now = datetime.now(timezone.utc).isoformat()

    return RunResponse(
        dataset_id=dataset_id,
        overall_score=score.overall_score,
        grade=score.grade,
        total_rules=score.total_rules,
        total_passed=score.total_passed,
        total_failed=score.total_failed,
        dimensions={
            name: {
                "weight": ds.weight,
                "pass_rate": ds.pass_rate,
                "weighted_score": ds.weighted_score,
            }
            for name, ds in score.dimensions.items()
        },
        run_at=now,
    )


@app.get("/datasets/{dataset_id}/scores")
def get_latest_scores(dataset_id: str) -> dict[str, Any]:
    """Retrieve the latest quality scores for a dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    history = _scores.get(dataset_id, [])
    if not history:
        raise HTTPException(status_code=404, detail="No scores available yet")
    latest = history[-1]
    return latest.to_dict()


@app.get("/datasets/{dataset_id}/history", response_model=list[ScoreHistoryItem])
def get_score_history(dataset_id: str) -> list[ScoreHistoryItem]:
    """Retrieve the score history for a dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    history = _scores.get(dataset_id, [])
    return [
        ScoreHistoryItem(
            overall_score=s.overall_score,
            grade=s.grade,
            total_rules=s.total_rules,
            run_at=datetime.now(timezone.utc).isoformat(),
        )
        for s in history
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
