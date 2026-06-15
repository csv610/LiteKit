"""Validation functions for image files and data."""

import logging
from pathlib import Path
from PIL import Image
from .core import MAX_IMAGE_SIZE_BYTES, MIN_IMAGE_DIMENSION

logger = logging.getLogger(__name__)

def is_valid_image(path: Path) -> bool:
    """
    Check if a file is a valid image based on extension.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    return path.suffix.lower() in valid_extensions

def is_valid_size(path: Path) -> bool:
    """
    Check if image file size is within the limit.
    """
    return path.stat().st_size <= MAX_IMAGE_SIZE_BYTES

def is_valid_dimensions(path: Path) -> bool:
    """
    Check if image dimensions meet minimum requirement.
    """
    try:
        with Image.open(path) as img:
            width, height = img.size
            return width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION
    except Exception as e:
        logger.error(f"Error reading image dimensions: {e}")
        return False
