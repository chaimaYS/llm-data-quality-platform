"""
Abstract base connector for data source integration.
Author: Chaima Yedes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata about a discovered dataset."""

    name: str
    source: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: list[str] = field(default_factory=list)
    size_bytes: Optional[int] = None
    last_modified: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result of a filtered scan against a dataset."""

    data: pd.DataFrame
    total_rows: int
    returned_rows: int
    query: Optional[str] = None
    elapsed_ms: float = 0.0


class Connector(ABC):
    """
    Abstract base class for all data source connectors.

    Connectors provide a uniform interface to list, sample, and scan
    datasets from heterogeneous backends (databases, file systems,
    object stores, APIs).
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection."""
        ...

    @abstractmethod
    def list_datasets(self) -> list[DatasetMetadata]:
        """Return metadata for every dataset visible to this connector."""
        ...

    @abstractmethod
    def sample(self, dataset: str, n: int = 1000) -> pd.DataFrame:
        """
        Return a random sample of *n* rows from *dataset*.

        Parameters
        ----------
        dataset : str
            Identifier for the dataset (table name, file path, etc.).
        n : int
            Number of rows to return.
        """
        ...

    @abstractmethod
    def scan(self, dataset: str, query: Optional[str] = None) -> ScanResult:
        """
        Execute an optional filter query against *dataset* and return
        matching rows wrapped in a :class:`ScanResult`.

        Parameters
        ----------
        dataset : str
            Identifier for the dataset.
        query : str, optional
            SQL WHERE clause or equivalent filter expression.
        """
        ...

    def __enter__(self) -> "Connector":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, connected={self._connected})"
