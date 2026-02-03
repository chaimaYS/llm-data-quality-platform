"""
Image quality analysis and metadata extraction.
Author: Chaima Yedes
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


@dataclass
class ImageProfile:
    """Quality and metadata profile for an image file."""

    path: str
    format: str
    width: int
    height: int
    channels: int
    file_size_bytes: int
    megapixels: float
    is_corrupt: bool = False
    blur_score: float = 0.0
    is_blurry: bool = False
    exposure_mean: float = 0.0
    exposure_std: float = 0.0
    is_overexposed: bool = False
    is_underexposed: bool = False
    phash: str = ""
    exif: dict[str, Any] = field(default_factory=dict)
    color_mode: str = ""
    aspect_ratio: float = 0.0
    bit_depth: Optional[int] = None


class ImageProcessor:
    """
    Analyzes image files for integrity, quality metrics, perceptual
    hashing, and EXIF metadata extraction.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        overexposure_threshold: float = 240.0,
        underexposure_threshold: float = 15.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold

    def process(self, path: str | Path) -> ImageProfile:
        """Analyze an image and return an :class:`ImageProfile`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        file_size = path.stat().st_size
        is_corrupt = False

        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path)  # reopen after verify
        except Exception:
            logger.warning("Corrupt or unreadable image: %s", path)
            return ImageProfile(
                path=str(path),
                format="unknown",
                width=0,
                height=0,
                channels=0,
                file_size_bytes=file_size,
                megapixels=0.0,
                is_corrupt=True,
            )

        width, height = img.size
        fmt = img.format or path.suffix.lstrip(".").upper()
        mode = img.mode
        channels = len(img.getbands())
        megapixels = round(width * height / 1_000_000, 2)
        aspect_ratio = round(width / height, 4) if height else 0.0

        arr = np.array(img.convert("RGB"))
        gray = np.array(img.convert("L"))

        blur_score = self._compute_blur(gray)
        is_blurry = blur_score < self.blur_threshold

        exposure_mean = float(np.mean(gray))
        exposure_std = float(np.std(gray))
        is_over = exposure_mean > self.overexposure_threshold
        is_under = exposure_mean < self.underexposure_threshold

        phash = self._compute_phash(img)
        exif = self._extract_exif(img)

        bit_depth_map = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "I": 32, "F": 32}
        bit_depth = bit_depth_map.get(mode)

        return ImageProfile(
            path=str(path),
            format=fmt,
            width=width,
            height=height,
            channels=channels,
            file_size_bytes=file_size,
            megapixels=megapixels,
            is_corrupt=is_corrupt,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            exposure_mean=round(exposure_mean, 2),
            exposure_std=round(exposure_std, 2),
            is_overexposed=is_over,
            is_underexposed=is_under,
            phash=phash,
            exif=exif,
            color_mode=mode,
            aspect_ratio=aspect_ratio,
            bit_depth=bit_depth,
        )

    def _compute_blur(self, gray: np.ndarray) -> float:
        """
        Compute blur metric via variance of Laplacian.
        Higher values = sharper image.
        """
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0
        padded = np.pad(gray.astype(np.float64), 1, mode="edge")
        result = np.zeros_like(gray, dtype=np.float64)
        for di in range(3):
            for dj in range(3):
                result += laplacian_kernel[di, dj] * padded[di : di + h, dj : dj + w]
        return float(np.var(result))

    def _compute_phash(self, img: Image.Image, hash_size: int = 8) -> str:
        """Compute a perceptual hash (pHash) of the image."""
        resized = img.convert("L").resize(
            (hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS
        )
        pixels = np.array(resized, dtype=np.float64)

        dct_rows = self._dct2(pixels)
        low_freq = dct_rows[:hash_size, :hash_size]

        median = np.median(low_freq)
        bits = (low_freq > median).flatten()

        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)
        return f"{hash_int:0{hash_size * hash_size // 4}x}"

    @staticmethod
    def _dct2(block: np.ndarray) -> np.ndarray:
        """Simple 2D DCT via matrix multiplication."""
        from numpy import cos, pi, sqrt

        n, m = block.shape
        result = np.zeros_like(block)
        for u in range(n):
            for v in range(m):
                cu = 1 / sqrt(n) if u == 0 else sqrt(2 / n)
                cv = 1 / sqrt(m) if v == 0 else sqrt(2 / m)
                s = 0.0
                for x in range(n):
                    for y in range(m):
                        s += block[x, y] * cos(pi * (2 * x + 1) * u / (2 * n)) * cos(
                            pi * (2 * y + 1) * v / (2 * m)
                        )
                result[u, v] = cu * cv * s
        return result

    def _extract_exif(self, img: Image.Image) -> dict[str, Any]:
        """Extract EXIF data from an image if available."""
        exif_data: dict[str, Any] = {}
        try:
            raw_exif = img.getexif()
            if not raw_exif:
                return exif_data
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                try:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    exif_data[tag_name] = value
                except Exception:
                    exif_data[tag_name] = str(value)
        except Exception:
            logger.debug("No EXIF data available")
        return exif_data
