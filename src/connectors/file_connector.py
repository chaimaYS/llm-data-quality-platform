"""
File-based connector for CSV, Parquet, and Excel using DuckDB.
Author: Chaima Yedes
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from src.connectors.base import Connector, DatasetMetadata, ScanResult

_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq", ".xlsx", ".xls"}

_READER_MAP = {
    ".csv": "read_csv_auto('{path}')",
    ".parquet": "read_parquet('{path}')",
    ".pq": "read_parquet('{path}')",
    ".xlsx": "st_read('{path}')",
    ".xls": "st_read('{path}')",
}


class FileConnector(Connector):
    """
    Connector that reads flat files from a local or mounted directory.

    Uses DuckDB as the query engine for fast analytical reads.

    Config keys
    -----------
    root_dir : str      Root directory to scan for files.
    recursive : bool    Whether to walk subdirectories (default ``True``).
    extensions : list   Allowed extensions (default all supported).
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self._root: Path = Path(self.config.get("root_dir", ".")).resolve()
        self._recursive: bool = self.config.get("recursive", True)
        self._allowed: set[str] = set(
            self.config.get("extensions", list(_SUPPORTED_EXTENSIONS))
        )
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> None:
        if not self._root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self._root}")
        self._conn = duckdb.connect(database=":memory:")
        try:
            self._conn.install_extension("spatial")
            self._conn.load_extension("spatial")
        except Exception:
            pass  # spatial is only needed for Excel; skip if unavailable
        self._connected = True

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._connected = False

    def _discover_files(self) -> list[Path]:
        pattern = "**/*" if self._recursive else "*"
        return sorted(
            p
            for p in self._root.glob(pattern)
            if p.is_file() and p.suffix.lower() in self._allowed
        )

    def _reader_expr(self, path: Path) -> str:
        ext = path.suffix.lower()
        template = _READER_MAP.get(ext)
        if template is None:
            raise ValueError(f"Unsupported file extension: {ext}")
        return template.format(path=str(path).replace("'", "''"))

    def list_datasets(self) -> list[DatasetMetadata]:
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        datasets: list[DatasetMetadata] = []
        for fp in self._discover_files():
            reader = self._reader_expr(fp)
            try:
                info = self._conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM {reader}"
                ).fetchone()
                row_count = info[0] if info else None
                cols_df = self._conn.execute(
                    f"SELECT * FROM {reader} LIMIT 0"
                ).df()
                columns = list(cols_df.columns)
            except Exception:
                row_count = None
                columns = []

            rel_path = str(fp.relative_to(self._root))
            datasets.append(
                DatasetMetadata(
                    name=rel_path,
                    source=str(fp),
                    row_count=row_count,
                    column_count=len(columns),
                    columns=columns,
                    size_bytes=fp.stat().st_size,
                    last_modified=str(
                        pd.Timestamp.fromtimestamp(fp.stat().st_mtime)
                    ),
                )
            )
        return datasets

    def sample(self, dataset: str, n: int = 1000) -> pd.DataFrame:
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        path = self._root / dataset
        reader = self._reader_expr(path)
        return self._conn.execute(
            f"SELECT * FROM {reader} USING SAMPLE {n} ROWS"
        ).df()

    def scan(self, dataset: str, query: Optional[str] = None) -> ScanResult:
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        path = self._root / dataset
        reader = self._reader_expr(path)
        where = f" WHERE {query}" if query else ""

        t0 = time.perf_counter()
        total = (
            self._conn.execute(
                f"SELECT COUNT(*) FROM {reader}{where}"
            ).fetchone()
            or (0,)
        )[0]
        df = self._conn.execute(f"SELECT * FROM {reader}{where}").df()
        elapsed = (time.perf_counter() - t0) * 1000

        return ScanResult(
            data=df,
            total_rows=total,
            returned_rows=len(df),
            query=query,
            elapsed_ms=round(elapsed, 2),
        )
