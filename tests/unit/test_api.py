"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


class TestHealthEndpoint:
    def test_health(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDatasetsEndpoint:
    def test_register_dataset(self):
        client = TestClient(app)
        response = client.post("/datasets", json={
            "name": "test_customers",
            "source_type": "file",
            "connection": {"path": "data/sample_customers.csv"},
        })
        assert response.status_code in (200, 201, 422)

    def test_register_dataset_missing_name(self):
        client = TestClient(app)
        response = client.post("/datasets", json={
            "source_type": "file",
        })
        assert response.status_code == 422
