"""Image I/O and base64 conversion utilities."""

import base64
import io
import logging
import os
import platform
from pathlib import Path
from typing import List, Literal, Union, Dict, Optional
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from .core import (
    IMAGE_MIME_TYPE, 
    _is_url, 
    _download_from_url, 
    _convert_to_rgb,
    _validate_file_exists
)
from .validation import is_valid_image, is_valid_dimensions, is_valid_size

logger = logging.getLogger(__name__)

def encode_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoding.
    """
    temp_path = None
    if _is_url(image_path):
        temp_path = _download_from_url(image_path)
        image_path = temp_path

    try:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not is_valid_image(path):
            raise ValueError(f"File is not a valid image: {image_path}")

        if not is_valid_dimensions(path):
            raise ValueError(f"Image dimensions too small: {image_path}")

        if not is_valid_size(path):
            raise ValueError(f"Image file too large: {image_path}")

        with open(path, "rb") as file:
            file_data = file.read()
            encoded_file = base64.b64encode(file_data).decode("utf-8")
            base64_url = f"data:{IMAGE_MIME_TYPE};base64,{encoded_file}"
        return base64_url
    finally:
        if temp_path:
            os.unlink(temp_path)

def b64_to_pil(b64_string: str) -> Image.Image:
    """
    Convert base64 encoded image to PIL Image object.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error converting base64 to PIL Image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")

def pil_to_b64(image: Image.Image, image_format: str = "JPEG", quality: int = 85, include_data_uri: bool = True) -> str:
    """
    Convert PIL Image object to base64 encoded string.
    """
    if image_format == "JPEG":
        image = _convert_to_rgb(image)
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    output = io.BytesIO()
    save_kwargs = {"format": image_format}
    if image_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        if image_format == "JPEG":
            save_kwargs["optimize"] = True

    image.save(output, **save_kwargs)
    image_bytes = output.getvalue()
    b64_string = base64.b64encode(image_bytes).decode("utf-8")

    if include_data_uri:
        mime_type = f"image/{image_format.lower()}"
        return f"data:{mime_type};base64,{b64_string}"
    return b64_string

def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV (cv2) image to PIL Image object.
    """
    if len(cv_image.shape) == 2:
        return Image.fromarray(cv_image, mode="L")

    if len(cv_image.shape) == 3:
        if cv_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        elif cv_image.shape[2] == 4:
            rgba_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba_image, mode="RGBA")
    
    raise ValueError(f"Unsupported image shape: {cv_image.shape}")

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image object to OpenCV (cv2) image.
    """
    if image.mode == "L":
        return np.array(image)
    if image.mode == "RGBA":
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    
    rgb_array = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def get_image_info(image_path: str) -> Dict:
    """Get metadata for an image file."""
    path = _validate_file_exists(image_path)
    try:
        with Image.open(path) as img:
            file_size_bytes = path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            has_exif = bool(img.getexif())
            stat = path.stat()
            creation_time = stat.st_birthtime if platform.system() == "Darwin" else stat.st_ctime
            created_date = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")

            return {
                "width": img.width,
                "height": img.height,
                "format": img.format or "Unknown",
                "color_mode": img.mode,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(file_size_mb, 2),
                "has_exif": has_exif,
                "created_date": created_date,
            }
    except Exception as e:
        logger.error(f"Error reading image info: {e}")
        raise

def save_image(
    image_data: Union[str, Image.Image, np.ndarray],
    output_path: str,
    image_format: Literal["PNG", "JPG", "JPEG"] = "PNG",
    quality: int = 85,
    input_type: Optional[Literal["path", "pil", "cv2", "base64"]] = None,
) -> str:
    """Save image from any input type to PNG or JPG format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if image_format.upper() in ("JPG", "JPEG") else image_format.upper()

    if input_type == "path": pil_image = Image.open(image_data)
    elif input_type == "pil": pil_image = image_data
    elif input_type == "cv2": pil_image = cv2_to_pil(image_data)
    elif input_type == "base64": pil_image = b64_to_pil(image_data)
    else: # auto-detect
        if isinstance(image_data, Image.Image): pil_image = image_data
        elif isinstance(image_data, np.ndarray): pil_image = cv2_to_pil(image_data)
        elif isinstance(image_data, str):
            if image_data.startswith(("data:", "iVBORw", "/9j/", "image/")): pil_image = b64_to_pil(image_data)
            else: pil_image = Image.open(image_data)
        else: raise ValueError(f"Unsupported image_data type: {type(image_data)}")

    if fmt == "JPEG": pil_image = _convert_to_rgb(pil_image)
    pil_image.save(str(output_path), format=fmt, quality=quality if fmt == "JPEG" else None)
    return str(output_path)

def save_images_batch(
    image_data_list: List[Union[str, Image.Image, np.ndarray]],
    output_directory: str,
    image_format: Literal["PNG", "JPG", "JPEG"] = "PNG",
    quality: int = 85,
    filename_prefix: str = "image",
    input_type: Optional[Literal["path", "pil", "cv2", "base64"]] = None,
) -> List[str]:
    """Save multiple images from any input type to PNG or JPG format."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for idx, image_data in enumerate(image_data_list):
        ext = ".png" if image_format.upper() == "PNG" else ".jpg"
        output_path = output_dir / f"{filename_prefix}_{idx:04d}{ext}"
        saved_paths.append(save_image(image_data, str(output_path), image_format=image_format, quality=quality, input_type=input_type))
    return saved_paths
