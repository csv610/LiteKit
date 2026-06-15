"""Tests for vision validation module."""

import os
from pathlib import Path

import pytest
from PIL import Image

from litekit.vision.validation import is_valid_image, is_valid_size, is_valid_dimensions
from litekit.vision.core import MAX_IMAGE_SIZE_BYTES, MIN_IMAGE_DIMENSION


class TestIsValidImage:
    def test_valid_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".JPG", ".PNG"]:
            assert is_valid_image(Path(f"image{ext}"))

    def test_invalid_extensions(self):
        for ext in [".txt", ".pdf", ".bmp", ".tiff", ""]:
            assert not is_valid_image(Path(f"file{ext}"))

    def test_no_extension(self):
        assert not is_valid_image(Path("image"))


class TestIsValidSize:
    def test_within_limit(self, tmp_path):
        img_path = tmp_path / "small.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path, quality=50)
        assert is_valid_size(img_path)

    def test_exceeds_limit(self, tmp_path):
        img_path = tmp_path / "large.jpg"
        large_size = int((MAX_IMAGE_SIZE_BYTES * 1.1) ** 0.5)
        img = Image.new("RGB", (large_size, 1))
        img.save(img_path, quality=100)
        if img_path.stat().st_size > MAX_IMAGE_SIZE_BYTES:
            assert not is_valid_size(img_path)
        else:
            pytest.skip("Could not create image exceeding size limit")


class TestIsValidDimensions:
    def test_valid_dimensions(self, tmp_path):
        img_path = tmp_path / "valid.png"
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION))
        img.save(img_path)
        assert is_valid_dimensions(img_path)

    def test_below_min_dimensions(self, tmp_path):
        img_path = tmp_path / "small.png"
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION - 1, MIN_IMAGE_DIMENSION - 1))
        img.save(img_path)
        assert not is_valid_dimensions(img_path)

    def test_width_below_min(self, tmp_path):
        img_path = tmp_path / "thin.png"
        img = Image.new("RGB", (1, MIN_IMAGE_DIMENSION))
        img.save(img_path)
        assert not is_valid_dimensions(img_path)

    def test_height_below_min(self, tmp_path):
        img_path = tmp_path / "short.png"
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION, 1))
        img.save(img_path)
        assert not is_valid_dimensions(img_path)

    def test_non_image_file_returns_false(self, tmp_path):
        txt_path = tmp_path / "not_an_image.txt"
        txt_path.write_text("not an image")
        assert not is_valid_dimensions(txt_path)
