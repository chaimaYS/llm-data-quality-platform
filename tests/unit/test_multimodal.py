"""Tests for multimodal processors (images)."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.multimodal.image_processor import ImageProcessor


class TestImageProcessor:
    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    @pytest.fixture
    def valid_jpeg(self, tmp_path):
        img = Image.new("RGB", (800, 600), color=(128, 128, 128))
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")
        return str(path)

    @pytest.fixture
    def valid_png(self, tmp_path):
        img = Image.new("RGBA", (1920, 1080), color=(255, 0, 0, 255))
        path = tmp_path / "test.png"
        img.save(path, "PNG")
        return str(path)

    @pytest.fixture
    def small_image(self, tmp_path):
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        path = tmp_path / "tiny.jpg"
        img.save(path, "JPEG")
        return str(path)

    @pytest.fixture
    def corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not an image at all")
        return str(path)

    def test_process_valid_jpeg(self, processor, valid_jpeg):
        profile = processor.process(valid_jpeg)
        assert profile.width == 800
        assert profile.height == 600
        assert profile.is_corrupt is False
        assert profile.file_size_bytes > 0

    def test_process_valid_png(self, processor, valid_png):
        profile = processor.process(valid_png)
        assert profile.width == 1920
        assert profile.height == 1080
        assert profile.is_corrupt is False

    def test_resolution(self, processor, valid_jpeg, small_image):
        large = processor.process(valid_jpeg)
        small = processor.process(small_image)
        assert large.width > small.width
        assert large.height > small.height

    def test_blur_score_exists(self, processor, valid_jpeg):
        profile = processor.process(valid_jpeg)
        assert profile.blur_score is not None
        assert isinstance(profile.blur_score, float)

    def test_perceptual_hash(self, processor, valid_jpeg):
        profile = processor.process(valid_jpeg)
        assert profile.phash is not None
        assert len(profile.phash) > 0

    def test_corrupt_file(self, processor, corrupt_file):
        profile = processor.process(corrupt_file)
        assert profile.is_corrupt is True

    def test_nonexistent_file(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/path/image.jpg")

    def test_identical_images_same_hash(self, processor, tmp_path):
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        img.save(p1, "JPEG")
        img.save(p2, "JPEG")
        h1 = processor.process(str(p1)).phash
        h2 = processor.process(str(p2)).phash
        assert h1 == h2

    def test_different_images_different_hash(self, processor, tmp_path):
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 0, 255))
        p1 = tmp_path / "red.jpg"
        p2 = tmp_path / "blue.jpg"
        img1.save(p1, "JPEG")
        img2.save(p2, "JPEG")
        h1 = processor.process(str(p1)).phash
        h2 = processor.process(str(p2)).phash
        assert h1 != h2

    def test_all_black_underexposed(self, processor, tmp_path):
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        path = tmp_path / "black.jpg"
        img.save(path, "JPEG")
        profile = processor.process(str(path))
        assert profile.is_corrupt is False
        assert profile.exposure_mean < 10
        assert profile.is_underexposed is True

    def test_all_white_overexposed(self, processor, tmp_path):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        path = tmp_path / "white.jpg"
        img.save(path, "JPEG")
        profile = processor.process(str(path))
        assert profile.exposure_mean > 240
        assert profile.is_overexposed is True

    def test_megapixels(self, processor, valid_jpeg):
        profile = processor.process(valid_jpeg)
        expected = 800 * 600 / 1_000_000
        assert profile.megapixels == pytest.approx(expected, abs=0.01)

    def test_aspect_ratio(self, processor, valid_jpeg):
        profile = processor.process(valid_jpeg)
        assert profile.aspect_ratio == pytest.approx(800 / 600, abs=0.01)

    def test_color_mode(self, processor, valid_jpeg, valid_png):
        jpeg_profile = processor.process(valid_jpeg)
        png_profile = processor.process(valid_png)
        assert jpeg_profile.color_mode == "RGB"
        assert png_profile.color_mode == "RGBA"

    def test_batch_process(self, processor, tmp_path):
        paths = []
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            p = tmp_path / f"img_{i}.jpg"
            img.save(p, "JPEG")
            paths.append(str(p))
        results = [processor.process(p) for p in paths]
        assert len(results) == 5
        assert all(not r.is_corrupt for r in results)

    def test_channels(self, processor, valid_jpeg, valid_png):
        assert processor.process(valid_jpeg).channels == 3
        assert processor.process(valid_png).channels == 4
