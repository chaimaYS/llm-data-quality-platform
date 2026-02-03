"""
PDF document processor: text extraction, OCR, table detection.
Author: Chaima Yedes
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


class PDFType(str, Enum):
    NATIVE = "native"
    SCANNED = "scanned"
    MIXED = "mixed"


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""

    page_number: int
    data: pd.DataFrame
    bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class DocumentProfile:
    """Profile of a processed PDF document."""

    path: str
    pdf_type: PDFType
    page_count: int
    total_chars: int
    text_pages: dict[int, str] = field(default_factory=dict)
    tables: list[ExtractedTable] = field(default_factory=list)
    page_images: dict[int, Image.Image] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)
    word_count: int = 0


class PDFProcessor:
    """
    Processes PDF documents: classifies type (native vs scanned),
    extracts text, runs OCR on scanned pages, extracts tables, and
    renders pages to images.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_lang: str = "eng",
        image_dpi: int = 200,
        min_chars_for_native: int = 50,
    ) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_lang = ocr_lang
        self.image_dpi = image_dpi
        self.min_chars_for_native = min_chars_for_native

    def process(
        self,
        path: str | Path,
        extract_tables: bool = True,
        render_images: bool = False,
        pages: Optional[list[int]] = None,
    ) -> DocumentProfile:
        """
        Process a PDF file and return a :class:`DocumentProfile`.

        Parameters
        ----------
        path : str or Path
            Path to the PDF file.
        extract_tables : bool
            Whether to extract tables from each page.
        render_images : bool
            Whether to render pages as PIL images.
        pages : list[int], optional
            Specific pages to process (1-indexed). None means all.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        text_pages: dict[int, str] = {}
        tables: list[ExtractedTable] = []
        page_images: dict[int, Image.Image] = {}
        ocr_pages: list[int] = []
        native_count = 0
        scanned_count = 0

        with pdfplumber.open(path) as pdf:
            metadata = pdf.metadata or {}
            page_count = len(pdf.pages)
            target_pages = pages or list(range(1, page_count + 1))

            for page_num in target_pages:
                if page_num < 1 or page_num > page_count:
                    continue
                page = pdf.pages[page_num - 1]

                raw_text = page.extract_text() or ""
                is_native = len(raw_text.strip()) >= self.min_chars_for_native

                if is_native:
                    text_pages[page_num] = raw_text
                    native_count += 1
                else:
                    ocr_text = self._ocr_page(page) if self.ocr_enabled else ""
                    text_pages[page_num] = ocr_text or raw_text
                    if self.ocr_enabled:
                        ocr_pages.append(page_num)
                    scanned_count += 1

                if extract_tables:
                    page_tables = self._extract_tables(page, page_num)
                    tables.extend(page_tables)

                if render_images:
                    img = self._render_page(page)
                    if img is not None:
                        page_images[page_num] = img

        total_chars = sum(len(t) for t in text_pages.values())
        word_count = sum(len(t.split()) for t in text_pages.values())

        if scanned_count == 0:
            pdf_type = PDFType.NATIVE
        elif native_count == 0:
            pdf_type = PDFType.SCANNED
        else:
            pdf_type = PDFType.MIXED

        return DocumentProfile(
            path=str(path),
            pdf_type=pdf_type,
            page_count=page_count,
            total_chars=total_chars,
            text_pages=text_pages,
            tables=tables,
            page_images=page_images,
            metadata=metadata,
            ocr_pages=ocr_pages,
            word_count=word_count,
        )

    def _ocr_page(self, page: pdfplumber.page.Page) -> str:
        """Render a page to an image and run OCR via pytesseract."""
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract not installed; skipping OCR")
            return ""

        try:
            img = page.to_image(resolution=self.image_dpi)
            pil_image = img.original
            text = pytesseract.image_to_string(pil_image, lang=self.ocr_lang)
            return text.strip()
        except Exception:
            logger.exception("OCR failed for page")
            return ""

    def _extract_tables(
        self, page: pdfplumber.page.Page, page_num: int
    ) -> list[ExtractedTable]:
        """Extract all tables from a single page."""
        results: list[ExtractedTable] = []
        try:
            raw_tables = page.extract_tables()
            for tbl in raw_tables:
                if not tbl or len(tbl) < 2:
                    continue
                headers = [str(h or f"col_{i}") for i, h in enumerate(tbl[0])]
                rows = tbl[1:]
                df = pd.DataFrame(rows, columns=headers)
                results.append(ExtractedTable(page_number=page_num, data=df))
        except Exception:
            logger.exception("Table extraction failed on page %d", page_num)
        return results

    def _render_page(self, page: pdfplumber.page.Page) -> Optional[Image.Image]:
        """Render a page as a PIL Image."""
        try:
            img = page.to_image(resolution=self.image_dpi)
            return img.original.copy()
        except Exception:
            logger.exception("Page rendering failed")
            return None
