"""
PostgreSQL connector using SQLAlchemy and pandas.
Author: Chaima Yedes
"""

import time
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.connectors.base import Connector, DatasetMetadata, ScanResult


class PostgresConnector(Connector):
    """
    Connector for PostgreSQL databases.

    Config keys
    -----------
    host : str          Database host (default ``localhost``).
    port : int          Database port (default ``5432``).
    database : str      Database name.
    user : str          Username.
    password : str      Password.
    schema : str        Schema to introspect (default ``public``).
    url : str           Full SQLAlchemy URL (overrides individual keys).
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self._engine: Optional[Engine] = None
        self._schema: str = self.config.get("schema", "public")

    def _build_url(self) -> str:
        if url := self.config.get("url"):
            return url
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5432)
        database = self.config["database"]
        user = self.config["user"]
        password = self.config.get("password", "")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    def connect(self) -> None:
        url = self._build_url()
        self._engine = create_engine(
            url,
            pool_size=self.config.get("pool_size", 5),
            pool_pre_ping=True,
        )
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        self._connected = True

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._connected = False

    def list_datasets(self) -> list[DatasetMetadata]:
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")
        inspector = inspect(self._engine)
        datasets: list[DatasetMetadata] = []
        for table_name in inspector.get_table_names(schema=self._schema):
            columns = [
                col["name"]
                for col in inspector.get_columns(table_name, schema=self._schema)
            ]
            with self._engine.connect() as conn:
                row_count = conn.execute(
                    text(f'SELECT COUNT(*) FROM "{self._schema}"."{table_name}"')
                ).scalar()
            datasets.append(
                DatasetMetadata(
                    name=table_name,
                    source=f"postgres://{self.name}/{self._schema}.{table_name}",
                    row_count=row_count,
                    column_count=len(columns),
                    columns=columns,
                )
            )
        return datasets

    def sample(self, dataset: str, n: int = 1000) -> pd.DataFrame:
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")
        query = f'SELECT * FROM "{self._schema}"."{dataset}" ORDER BY RANDOM() LIMIT :n'
        return pd.read_sql(text(query), self._engine, params={"n": n})

    def scan(self, dataset: str, query: Optional[str] = None) -> ScanResult:
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")
        base = f'SELECT * FROM "{self._schema}"."{dataset}"'
        count_base = f'SELECT COUNT(*) FROM "{self._schema}"."{dataset}"'

        where = f" WHERE {query}" if query else ""
        sql = base + where
        count_sql = count_base + where

        t0 = time.perf_counter()
        with self._engine.connect() as conn:
            total = conn.execute(text(count_sql)).scalar() or 0
        df = pd.read_sql(text(sql), self._engine)
        elapsed = (time.perf_counter() - t0) * 1000

        return ScanResult(
            data=df,
            total_rows=total,
            returned_rows=len(df),
            query=query,
            elapsed_ms=round(elapsed, 2),
        )
