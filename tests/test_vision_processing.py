"""Tests for vision processing module."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from litekit.vision.processing import (
    create_blank_image,
    create_random_image,
    create_gradient_image,
    resize_images_to_fit,
    square_image,
    resize_to_dimensions,
    convert_format,
    crop,
    auto_orient,
    remove_exif,
    resize_to_max_size,
)


class TestCreateBlankImage:
    def test_white_image(self):
        img = create_blank_image(100, 50, color=(255, 0, 0), image_mode="RGB")
        assert img.size == (100, 50)
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 0, 0)

    def test_random_color(self):
        img = create_blank_image(10, 10, color="random")
        assert img.size == (10, 10)
        assert img.mode == "RGB"

    def test_grayscale(self):
        img = create_blank_image(10, 10, color=128, image_mode="L")
        assert img.mode == "L"
        assert img.getpixel((0, 0)) == 128

    def test_rgba(self):
        img = create_blank_image(10, 10, color=(255, 0, 0, 128), image_mode="RGBA")
        assert img.mode == "RGBA"


class TestCreateRandomImage:
    def test_rgb(self):
        img = create_random_image(20, 30)
        assert img.size == (20, 30)
        assert img.mode == "RGB"

    def test_grayscale(self):
        img = create_random_image(10, 10, image_mode="L")
        assert img.mode == "L"

    def test_rgba(self):
        img = create_random_image(10, 10, image_mode="RGBA")
        assert img.mode == "RGBA"

    def test_different_pixels(self):
        img1 = create_random_image(50, 50)
        img2 = create_random_image(50, 50)
        assert np.array(img1).sum() != np.array(img2).sum()


class TestCreateGradientImage:
    def test_horizontal(self):
        img = create_gradient_image(
            100, 50, (255, 0, 0), (0, 0, 255), direction="horizontal"
        )
        assert img.size == (100, 50)
        assert img.mode == "RGB"

    def test_vertical(self):
        img = create_gradient_image(
            100, 50, (255, 0, 0), (0, 0, 255), direction="vertical"
        )
        assert img.size == (100, 50)

    def test_diagonal(self):
        img = create_gradient_image(
            100, 50, (255, 0, 0), (0, 0, 255), direction="diagonal"
        )
        assert img.size == (100, 50)

    def test_default_direction(self):
        img = create_gradient_image(10, 10, (0, 0, 0), (255, 255, 255))
        assert img.size == (10, 10)


class TestSquareImage:
    def test_square_smaller_than_max(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (50, 30))
        img.save(img_path)
        result = square_image(str(img_path), max_size=100, background_color=(0, 0, 0))
        assert result.size == (100, 100)

    def test_square_larger_than_max(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 100))
        img.save(img_path)
        result = square_image(str(img_path), max_size=50, background_color=(0, 0, 0))
        assert result.size == (50, 50)

    def test_center_position(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (20, 20))
        img.save(img_path)
        result = square_image(str(img_path), max_size=50, background_color=(255, 0, 0), position="center")
        assert result.size == (50, 50)

    def test_top_left_position(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (20, 20))
        img.save(img_path)
        result = square_image(str(img_path), max_size=50, background_color=(0, 0, 0), position="top-left")
        assert result.size == (50, 50)


class TestResizeToDimensions:
    def test_exact_dimensions(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 50))
        img.save(img_path)
        result = resize_to_dimensions(str(img_path), 200, 200)
        assert result.size == (200, 200)

    def test_custom_background(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = resize_to_dimensions(str(img_path), 50, 50, background_color=(255, 0, 0))
        assert result.size == (50, 50)


class TestConvertFormat:
    def test_to_jpeg(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = convert_format(str(img_path), "JPEG")
        assert isinstance(result, bytes)

    def test_to_png(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = convert_format(str(img_path), "PNG")
        assert isinstance(result, bytes)

    def test_to_webp(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = convert_format(str(img_path), "WEBP")
        assert isinstance(result, bytes)


class TestCrop:
    def test_crop_region(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img.save(img_path)
        result = crop(str(img_path), 10, 10, 50, 50)
        assert result.size == (40, 40)

    def test_full_crop(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (50, 50))
        img.save(img_path)
        result = crop(str(img_path), 0, 0, 50, 50)
        assert result.size == (50, 50)


class TestAutoOrient:
    def test_no_exif(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = auto_orient(str(img_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


class TestRemoveExif:
    def test_remove_exif(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = remove_exif(str(img_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


class TestResizeToMaxSize:
    def test_already_small_enough(self, tmp_path):
        img_path = tmp_path / "small.jpg"
        img = Image.new("RGB", (10, 10))
        img.save(img_path, quality=100)
        result = resize_to_max_size(str(img_path), max_size=10)
        assert isinstance(result, Image.Image)

    def test_invalid_size_unit(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        result = resize_to_max_size(str(img_path), max_size=1, size_unit="MB")
        assert isinstance(result, Image.Image)


class TestResizeImagesToFit:
    def test_small_images_no_resize(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (10, 10)).save(p)
            paths.append(str(p))
        result = resize_images_to_fit(paths)
        assert result == paths

    def test_large_images_trigger_resize(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            img = Image.new("RGB", (2000, 2000))
            img.save(p, quality=100)
            paths.append(str(p))
        result = resize_images_to_fit(paths)
        assert len(result) == 3
        for r in result:
            assert Path(r).exists()
