"""Collection and scanning utilities for images in directories."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Literal
from PIL import Image

from .core import _validate_directory_exists
from .io import get_image_info

logger = logging.getLogger(__name__)

def collect_images(
    directory_path: str,
    recursive: bool = False,
    formats: Optional[List[str]] = None,
    validate: bool = True,
    sort_by: Literal["name", "size", "date"] = "name",
) -> List[str]:
    """Collect image file paths from a directory."""
    directory = _validate_directory_exists(directory_path)
    if formats:
        formats = [fmt.upper() for fmt in formats]

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
    image_paths = []

    file_iter = directory.rglob("*") if recursive else directory.glob("*")
    for file_path in file_iter:
        if file_path.is_dir() or file_path.suffix.lower() not in valid_extensions:
            continue

        if validate:
            try:
                with Image.open(file_path) as img:
                    img_format = img.format or "Unknown"
                    if formats and img_format.upper() not in formats:
                        logger.debug("Skipping %s: format %s not in %s", file_path, img_format, formats)
                        continue
                image_paths.append(str(file_path))
            except Exception:
                logger.warning("Skipping %s: unable to open as image", file_path)
                continue
        else:
            image_paths.append(str(file_path))

    if sort_by == "size":
        image_paths.sort(key=lambda p: Path(p).stat().st_size)
    elif sort_by == "date":
        image_paths.sort(key=lambda p: Path(p).stat().st_mtime)
    else:
        image_paths.sort()

    return image_paths

def collect_images_with_info(
    directory_path: str,
    recursive: bool = False,
    formats: Optional[List[str]] = None,
    sort_by: Literal["name", "size", "date"] = "name",
) -> List[Dict]:
    """Collect image file paths and metadata from a directory."""
    directory = _validate_directory_exists(directory_path)
    image_paths = collect_images(
        directory_path=directory_path,
        recursive=recursive,
        formats=formats,
        validate=True,
        sort_by=sort_by,
    )
    
    result = []
    for path in image_paths:
        try:
            info = get_image_info(path)
            info["path"] = path
            result.append(info)
        except Exception:
            continue
    return result
