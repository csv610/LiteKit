"""Tests for vision I/O module."""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from litekit.vision.io import (
    encode_to_base64,
    b64_to_pil,
    pil_to_b64,
    cv2_to_pil,
    pil_to_cv2,
    get_image_info,
    save_image,
    save_images_batch,
)


class TestEncodeToBase64:
    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        result = encode_to_base64(str(img_path))
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > len("data:image/jpeg;base64,")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            encode_to_base64("/nonexistent/path.jpg")

    def test_invalid_file_type(self, tmp_path):
        txt_path = tmp_path / "not_image.txt"
        txt_path.write_text("not an image")
        with pytest.raises(ValueError, match="File is not a valid image"):
            encode_to_base64(str(txt_path))


class TestB64ToPil:
    def test_roundtrip(self):
        img = Image.new("RGB", (50, 50), (255, 0, 0))
        b64 = pil_to_b64(img, include_data_uri=True)
        decoded = b64_to_pil(b64)
        assert decoded.size == (50, 50)
        assert decoded.mode == "RGB"

    def test_without_data_uri_prefix(self):
        img = Image.new("RGB", (10, 10))
        b64 = pil_to_b64(img, include_data_uri=False)
        decoded = b64_to_pil(b64)
        assert decoded.size == (10, 10)

    def test_invalid_base64(self):
        with pytest.raises(ValueError, match="Invalid base64 image data"):
            b64_to_pil("not-valid-base64!!!")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            b64_to_pil("")


class TestPilToB64:
    def test_default_format(self):
        img = Image.new("RGB", (10, 10))
        result = pil_to_b64(img)
        assert result.startswith("data:image/jpeg;base64,")

    def test_png_format(self):
        img = Image.new("RGBA", (10, 10))
        result = pil_to_b64(img, image_format="PNG")
        assert result.startswith("data:image/png;base64,")

    def test_without_data_uri(self):
        img = Image.new("RGB", (10, 10))
        result = pil_to_b64(img, include_data_uri=False)
        assert not result.startswith("data:")

    def test_rgba_image_converted_for_jpeg(self):
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = pil_to_b64(img, image_format="JPEG")
        decoded = b64_to_pil(result)
        assert decoded.mode == "RGB"


class TestCV2ToPil:
    def test_color_image(self):
        cv_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        pil_img = cv2_to_pil(cv_img)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (50, 50)
        assert pil_img.mode == "RGB"

    def test_grayscale_image(self):
        cv_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        pil_img = cv2_to_pil(cv_img)
        assert pil_img.mode == "L"

    def test_rgba_image(self):
        cv_img = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        pil_img = cv2_to_pil(cv_img)
        assert pil_img.mode == "RGBA"

    def test_invalid_shape(self):
        cv_img = np.random.randint(0, 255, (50, 50, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            cv2_to_pil(cv_img)


class TestPilToCV2:
    def test_rgb_image(self):
        pil_img = Image.new("RGB", (50, 50), (255, 0, 0))
        cv_img = pil_to_cv2(pil_img)
        assert cv_img.shape == (50, 50, 3)

    def test_grayscale_image(self):
        pil_img = Image.new("L", (50, 50), 128)
        cv_img = pil_to_cv2(pil_img)
        assert len(cv_img.shape) == 2

    def test_rgba_image(self):
        pil_img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        cv_img = pil_to_cv2(pil_img)
        assert cv_img.shape == (50, 50, 4)


class TestGetImageInfo:
    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 200))
        img.save(img_path)
        info = get_image_info(str(img_path))
        assert info["width"] == 100
        assert info["height"] == 200
        assert info["format"] == "PNG"
        assert info["color_mode"] == "RGB"
        assert isinstance(info["file_size_bytes"], int)
        assert isinstance(info["file_size_mb"], float)
        assert "has_exif" in info
        assert "created_date" in info

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_image_info("/nonexistent.png")


class TestSaveImage:
    def test_save_from_path(self, tmp_path):
        src = tmp_path / "src.png"
        img = Image.new("RGB", (50, 50))
        img.save(src)
        dst = tmp_path / "dst.jpg"
        result = save_image(str(src), str(dst), image_format="JPEG")
        assert result == str(dst)
        assert Path(dst).exists()

    def test_save_from_pil(self, tmp_path):
        img = Image.new("RGB", (50, 50))
        dst = tmp_path / "out.png"
        result = save_image(img, str(dst), image_format="PNG", input_type="pil")
        assert result == str(dst)

    def test_save_from_cv2(self, tmp_path):
        cv_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        dst = tmp_path / "out.jpg"
        result = save_image(cv_img, str(dst), image_format="JPEG", input_type="cv2")
        assert result == str(dst)

    def test_save_from_base64(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        b64 = pil_to_b64(img, include_data_uri=True)
        dst = tmp_path / "out.png"
        result = save_image(b64, str(dst), image_format="PNG", input_type="base64")
        assert result == str(dst)

    def test_auto_detect_pil(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        dst = tmp_path / "out.png"
        result = save_image(img, str(dst))
        assert result == str(dst)

    def test_auto_detect_base64(self, tmp_path):
        b64 = pil_to_b64(Image.new("RGB", (10, 10)), include_data_uri=True)
        dst = tmp_path / "out.png"
        result = save_image(b64, str(dst))
        assert result == str(dst)

    def test_invalid_input_type(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported image_data type"):
            save_image(42, str(tmp_path / "out.png"))


class TestSaveImagesBatch:
    def test_save_multiple(self, tmp_path):
        imgs = [
            Image.new("RGB", (10, 10), (255, 0, 0)),
            Image.new("RGB", (10, 10), (0, 255, 0)),
        ]
        paths = save_images_batch(
            imgs, str(tmp_path), image_format="PNG", filename_prefix="img"
        )
        assert len(paths) == 2
        assert (tmp_path / "img_0000.png").exists()
        assert (tmp_path / "img_0001.png").exists()
