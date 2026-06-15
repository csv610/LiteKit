"""Tests for vision collection module."""

from pathlib import Path

import pytest
from PIL import Image

from litekit.vision.collection import collect_images, collect_images_with_info


class TestCollectImages:
    def test_collect_all(self, tmp_path):
        (tmp_path / "img1.jpg").write_text("not really an image")
        (tmp_path / "img2.png").write_text("not really an image")
        (tmp_path / "readme.txt").write_text("not an image")
        results = collect_images(str(tmp_path), validate=False)
        assert len(results) == 2
        assert any("img1.jpg" in r for r in results)
        assert any("img2.png" in r for r in results)

    def test_with_validation_skips_invalid(self, tmp_path):
        (tmp_path / "valid.png")
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "valid.png")
        (tmp_path / "invalid.txt").write_text("not an image")
        results = collect_images(str(tmp_path), validate=True)
        assert len(results) == 1
        assert "valid.png" in results[0]

    def test_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.png")
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "root.png")
        img.save(sub / "nested.png")
        results = collect_images(str(tmp_path), recursive=True, validate=False)
        assert len(results) == 2

    def test_filter_by_format(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "pic.jpg")
        img.save(tmp_path / "pic.png")
        results = collect_images(
            str(tmp_path), formats=["PNG"], validate=True
        )
        assert len(results) == 1
        assert results[0].endswith(".png")

    def test_sort_by_name(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "b.jpg")
        img.save(tmp_path / "a.jpg")
        results = collect_images(str(tmp_path), validate=False, sort_by="name")
        assert results[0].endswith("a.jpg")
        assert results[1].endswith("b.jpg")

    def test_sort_by_size(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "small.jpg")
        img2 = Image.new("RGB", (100, 100))
        img2.save(tmp_path / "large.jpg")
        results = collect_images(str(tmp_path), validate=False, sort_by="size")
        sizes = [Path(p).stat().st_size for p in results]
        assert sizes == sorted(sizes)

    def test_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            collect_images("/nonexistent/directory")

    def test_no_images_in_directory(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        results = collect_images(str(tmp_path), validate=False)
        assert results == []


class TestCollectImagesWithInfo:
    def test_collects_metadata(self, tmp_path):
        img = Image.new("RGB", (50, 100))
        img.save(tmp_path / "test.png")
        results = collect_images_with_info(str(tmp_path))
        assert len(results) == 1
        entry = results[0]
        assert entry["width"] == 50
        assert entry["height"] == 100
        assert entry["format"] == "PNG"
        assert "path" in entry

    def test_skips_invalid_images(self, tmp_path):
        (tmp_path / "invalid.jpg").write_text("not an image")
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "valid.png")
        results = collect_images_with_info(str(tmp_path))
        assert len(results) == 1
        assert "valid.png" in results[0]["path"]
