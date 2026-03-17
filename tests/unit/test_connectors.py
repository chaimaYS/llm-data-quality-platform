"""Tests for connector layer."""

import pytest
import pandas as pd
from pathlib import Path

from src.connectors.file_connector import FileConnector


class TestFileConnector:
    @pytest.fixture
    def csv_dir(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.0, 20.0, 30.0]})
        (tmp_path / "test.csv").write_text(df.to_csv(index=False))
        return str(tmp_path)

    @pytest.fixture
    def parquet_dir(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"]})
        df.to_parquet(tmp_path / "test.parquet", index=False)
        return str(tmp_path)

    def _make_conn(self, base_path):
        conn = FileConnector(name="test", config={"base_path": base_path})
        conn.connect()
        return conn

    def test_list_datasets_csv(self, csv_dir):
        conn = self._make_conn(csv_dir)
        datasets = conn.list_datasets()
        assert len(datasets) >= 1

    def test_list_datasets_parquet(self, parquet_dir):
        conn = self._make_conn(parquet_dir)
        datasets = conn.list_datasets()
        assert len(datasets) >= 1

    def test_sample_csv(self, csv_dir):
        conn = self._make_conn(csv_dir)
        datasets = conn.list_datasets()
        df = conn.sample(datasets[0].name, n=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 3

    def test_sample_parquet(self, parquet_dir):
        conn = self._make_conn(parquet_dir)
        datasets = conn.list_datasets()
        df = conn.sample(datasets[0].name, n=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 10

    def test_no_files_in_isolated_dir(self, tmp_path):
        empty_dir = tmp_path / "isolated_empty"
        empty_dir.mkdir()
        conn = FileConnector(name="test", config={"base_path": str(empty_dir), "recursive": False})
        conn.connect()
        datasets = conn.list_datasets()
        assert len(datasets) == 0

    def test_mixed_formats(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "f1.csv", index=False)
        pd.DataFrame({"b": [2]}).to_parquet(tmp_path / "f2.parquet", index=False)
        conn = self._make_conn(str(tmp_path))
        datasets = conn.list_datasets()
        assert len(datasets) == 2
